"""
Depth Anything V2 파인튜닝 스크립트 (DDP 멀티 GPU 지원)
현미경 이미지 → Height Map 추정

사용법:
    torchrun --nproc_per_node=2 src/train.py --config configs/train_config.yaml
"""

import argparse
import logging
import os
import pprint
import random
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset.microscopy import MicroscopyDataset
from depth_anything_v2.dpt import DepthAnythingV2
from util.dist_helper import setup_distributed
from util.loss import SiLogLoss
from util.metric import eval_depth
from util.utils import init_log


def parse_args():
    parser = argparse.ArgumentParser(description='Depth Anything V2 Fine-tuning for Microscopy')

    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                        help='Path to config file')

    # Override config options via command line
    parser.add_argument('--encoder', type=str, choices=['vits', 'vitb', 'vitl'])
    parser.add_argument('--data-dir', type=str)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--save-path', type=str)
    parser.add_argument('--pretrained-from', type=str)
    parser.add_argument('--port', type=int, default=None)

    return parser.parse_args()


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()

    # 설정 파일 로드
    config = load_config(args.config)

    # Command line 인자로 설정 덮어쓰기
    if args.encoder:
        config['model']['encoder'] = args.encoder
    if args.data_dir:
        config['data']['data_dir'] = args.data_dir
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.lr:
        config['training']['lr'] = args.lr
    if args.save_path:
        config['training']['save_path'] = args.save_path
    if args.pretrained_from:
        config['model']['pretrained_from'] = args.pretrained_from

    warnings.simplefilter('ignore', np.RankWarning)

    # 로거 초기화
    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    # DDP 설정
    rank, world_size = setup_distributed(port=args.port)

    if rank == 0:
        all_args = {**config, 'ngpus': world_size}
        logger.info('Config:\n{}'.format(pprint.pformat(all_args)))

        # TensorBoard
        save_path = Path(config['training']['save_path'])
        save_path.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(save_path)

    # 재현성
    set_seed(config.get('seed', 42))
    cudnn.enabled = True
    cudnn.benchmark = True

    # 데이터셋
    size = (config['data']['img_size'], config['data']['img_size'])
    trainset = MicroscopyDataset(
        data_dir=config['data']['data_dir'],
        mode='train',
        size=size,
        train_ratio=config['data'].get('train_ratio', 0.9)
    )
    valset = MicroscopyDataset(
        data_dir=config['data']['data_dir'],
        mode='val',
        size=size,
        train_ratio=config['data'].get('train_ratio', 0.9)
    )

    trainsampler = torch.utils.data.distributed.DistributedSampler(trainset)
    valsampler = torch.utils.data.distributed.DistributedSampler(valset, shuffle=False)

    trainloader = DataLoader(
        trainset,
        batch_size=config['training']['batch_size'],
        pin_memory=True,
        num_workers=config['training'].get('num_workers', 4),
        drop_last=True,
        sampler=trainsampler
    )
    valloader = DataLoader(
        valset,
        batch_size=1,
        pin_memory=True,
        num_workers=config['training'].get('num_workers', 4),
        drop_last=False,
        sampler=valsampler
    )

    local_rank = int(os.environ["LOCAL_RANK"])

    # 모델 설정
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }

    encoder = config['model']['encoder']
    max_depth = config['model']['max_depth']
    min_depth = config['model']['min_depth']

    model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})

    # 사전학습 가중치 로드
    pretrained_path = config['model'].get('pretrained_from')
    if pretrained_path and os.path.exists(pretrained_path):
        if rank == 0:
            logger.info(f'Loading pretrained weights from {pretrained_path}')
        state_dict = torch.load(pretrained_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)

    # DDP 설정
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        broadcast_buffers=False,
        output_device=local_rank,
        find_unused_parameters=True
    )

    # Loss
    criterion = SiLogLoss().cuda(local_rank)

    # Optimizer (encoder는 낮은 lr, decoder는 높은 lr)
    lr = config['training']['lr']
    optimizer = AdamW([
        {'params': [p for n, p in model.named_parameters() if 'pretrained' in n], 'lr': lr},
        {'params': [p for n, p in model.named_parameters() if 'pretrained' not in n], 'lr': lr * 10.0}
    ], lr=lr, betas=(0.9, 0.999), weight_decay=0.01)

    total_iters = config['training']['epochs'] * len(trainloader)

    # 메트릭 초기화
    previous_best = {
        'd1': 0, 'd2': 0, 'd3': 0,
        'abs_rel': 100, 'sq_rel': 100, 'rmse': 100,
        'rmse_log': 100, 'log10': 100, 'silog': 100
    }

    # 학습 시작
    for epoch in range(config['training']['epochs']):
        if rank == 0:
            logger.info('=' * 80)
            logger.info(f"Epoch: {epoch + 1}/{config['training']['epochs']}")
            logger.info(f"Best - d1: {previous_best['d1']:.4f}, abs_rel: {previous_best['abs_rel']:.4f}, "
                        f"rmse: {previous_best['rmse']:.4f}")

        trainsampler.set_epoch(epoch)

        # === Training ===
        model.train()
        total_loss = 0

        for i, sample in enumerate(trainloader):
            optimizer.zero_grad()

            img = sample['image'].cuda(local_rank)
            depth = sample['depth'].cuda(local_rank)
            valid_mask = sample['valid_mask'].cuda(local_rank)

            # 랜덤 수평 뒤집기 (data augmentation)
            if random.random() < 0.5:
                img = img.flip(-1)
                depth = depth.flip(-1)
                valid_mask = valid_mask.flip(-1)

            # Forward
            pred = model(img)

            # Loss 계산
            mask = valid_mask & (depth >= min_depth) & (depth <= max_depth)
            loss = criterion(pred, depth, mask)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Learning rate schedule (polynomial decay)
            iters = epoch * len(trainloader) + i
            lr_current = lr * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr_current
            optimizer.param_groups[1]["lr"] = lr_current * 10.0

            if rank == 0:
                writer.add_scalar('train/loss', loss.item(), iters)

            if rank == 0 and i % 50 == 0:
                logger.info(f'Iter: {i}/{len(trainloader)}, LR: {lr_current:.7f}, Loss: {loss.item():.4f}')

        # === Validation ===
        model.eval()

        results = {k: torch.tensor([0.0]).cuda() for k in previous_best.keys()}
        nsamples = torch.tensor([0.0]).cuda()

        for sample in valloader:
            img = sample['image'].cuda(local_rank).float()
            depth = sample['depth'].cuda(local_rank)[0]
            valid_mask = sample['valid_mask'].cuda(local_rank)[0]

            with torch.no_grad():
                pred = model(img)
                pred = F.interpolate(pred[:, None], depth.shape[-2:], mode='bilinear', align_corners=True)[0, 0]

            mask = valid_mask & (depth >= min_depth) & (depth <= max_depth)

            if mask.sum() < 10:
                continue

            cur_results = eval_depth(pred[mask], depth[mask])

            for k in results.keys():
                results[k] += cur_results[k]
            nsamples += 1

        # 모든 GPU에서 결과 수집
        torch.distributed.barrier()

        for k in results.keys():
            dist.reduce(results[k], dst=0)
        dist.reduce(nsamples, dst=0)

        if rank == 0:
            logger.info('=' * 80)
            logger.info('Validation Results:')
            logger.info('{:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}'.format(
                'd1', 'd2', 'd3', 'abs_rel', 'rmse', 'silog'))
            logger.info('{:8.4f}, {:8.4f}, {:8.4f}, {:8.4f}, {:8.4f}, {:8.4f}'.format(
                (results['d1'] / nsamples).item(),
                (results['d2'] / nsamples).item(),
                (results['d3'] / nsamples).item(),
                (results['abs_rel'] / nsamples).item(),
                (results['rmse'] / nsamples).item(),
                (results['silog'] / nsamples).item()
            ))

            # TensorBoard에 기록
            for name, metric in results.items():
                writer.add_scalar(f'eval/{name}', (metric / nsamples).item(), epoch)

        # Best 업데이트
        for k in results.keys():
            if k in ['d1', 'd2', 'd3']:
                previous_best[k] = max(previous_best[k], (results[k] / nsamples).item())
            else:
                previous_best[k] = min(previous_best[k], (results[k] / nsamples).item())

        # 체크포인트 저장
        if rank == 0:
            save_path = Path(config['training']['save_path'])

            # Latest 저장
            checkpoint = {
                'model': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'previous_best': previous_best,
            }
            torch.save(checkpoint, save_path / 'latest.pth')

            # Best 모델 저장 (abs_rel 기준)
            if previous_best['abs_rel'] == (results['abs_rel'] / nsamples).item():
                torch.save(model.module.state_dict(), save_path / 'best_model.pth')
                logger.info(f'Best model saved! abs_rel: {previous_best["abs_rel"]:.4f}')

            # 에폭별 저장 (선택적)
            if (epoch + 1) % config['training'].get('save_every', 10) == 0:
                torch.save(checkpoint, save_path / f'epoch_{epoch + 1}.pth')

    if rank == 0:
        writer.close()
        logger.info('Training completed!')
        logger.info(f'Best results: {previous_best}')


if __name__ == '__main__':
    main()
