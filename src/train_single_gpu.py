"""
Depth Anything V2 파인튜닝 스크립트 (단일 GPU)
현미경 이미지 → Height Map 추정

사용법:
    python src/train_single_gpu.py --config configs/train_config.yaml
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
import torch.nn.functional as F
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent))

from dataset.microscopy import MicroscopyDataset
from depth_anything_v2.dpt import DepthAnythingV2
from util.loss import SiLogLoss
from util.metric import eval_depth


def parse_args():
    parser = argparse.ArgumentParser(description='Depth Anything V2 Fine-tuning for Microscopy')

    parser.add_argument('--config', type=str, default='configs/train_config.yaml')
    parser.add_argument('--encoder', type=str, choices=['vits', 'vitb', 'vitl'])
    parser.add_argument('--data-dir', type=str)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--save-path', type=str)
    parser.add_argument('--pretrained-from', type=str)
    parser.add_argument('--gpu', type=int, default=0)

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

    warnings.filterwarnings('ignore')

    # 로거 설정
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(levelname)s] %(message)s')
    logger = logging.getLogger(__name__)

    logger.info('Config:\n{}'.format(pprint.pformat(config)))

    # GPU 설정
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

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

    trainloader = DataLoader(
        trainset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        pin_memory=True,
        num_workers=config['training'].get('num_workers', 4),
        drop_last=True
    )
    valloader = DataLoader(
        valset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=config['training'].get('num_workers', 4),
        drop_last=False
    )

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
        logger.info(f'Loading pretrained weights from {pretrained_path}')
        state_dict = torch.load(pretrained_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)

    model.to(device)

    # Loss
    criterion = SiLogLoss().to(device)

    # Optimizer
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
        logger.info('=' * 80)
        logger.info(f"Epoch: {epoch + 1}/{config['training']['epochs']}")
        logger.info(f"Best - d1: {previous_best['d1']:.4f}, abs_rel: {previous_best['abs_rel']:.4f}, rmse: {previous_best['rmse']:.4f}")

        # === Training ===
        model.train()
        total_loss = 0

        pbar = tqdm(trainloader, desc=f'Epoch {epoch + 1}')
        for i, sample in enumerate(pbar):
            optimizer.zero_grad()

            img = sample['image'].to(device)
            depth = sample['depth'].to(device)
            valid_mask = sample['valid_mask'].to(device)

            # 랜덤 수평 뒤집기
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

            # Learning rate schedule
            iters = epoch * len(trainloader) + i
            lr_current = lr * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr_current
            optimizer.param_groups[1]["lr"] = lr_current * 10.0

            writer.add_scalar('train/loss', loss.item(), iters)

            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{lr_current:.2e}'})

        avg_loss = total_loss / len(trainloader)
        logger.info(f'Train Loss: {avg_loss:.4f}')

        # === Validation ===
        model.eval()

        results = {k: 0.0 for k in previous_best.keys()}
        nsamples = 0

        for sample in tqdm(valloader, desc='Validation'):
            img = sample['image'].to(device).float()
            depth = sample['depth'].to(device)[0]
            valid_mask = sample['valid_mask'].to(device)[0]

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

        if nsamples > 0:
            for k in results.keys():
                results[k] /= nsamples

            logger.info('Validation Results:')
            logger.info(f"  d1: {results['d1']:.4f}, d2: {results['d2']:.4f}, d3: {results['d3']:.4f}")
            logger.info(f"  abs_rel: {results['abs_rel']:.4f}, rmse: {results['rmse']:.4f}, silog: {results['silog']:.4f}")

            for name, metric in results.items():
                writer.add_scalar(f'eval/{name}', metric, epoch)

            # Best 업데이트
            for k in results.keys():
                if k in ['d1', 'd2', 'd3']:
                    previous_best[k] = max(previous_best[k], results[k])
                else:
                    previous_best[k] = min(previous_best[k], results[k])

            # 체크포인트 저장
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'previous_best': previous_best,
            }
            torch.save(checkpoint, save_path / 'latest.pth')

            if previous_best['abs_rel'] == results['abs_rel']:
                torch.save(model.state_dict(), save_path / 'best_model.pth')
                logger.info(f'Best model saved! abs_rel: {previous_best["abs_rel"]:.4f}')

            if (epoch + 1) % config['training'].get('save_every', 10) == 0:
                torch.save(checkpoint, save_path / f'epoch_{epoch + 1}.pth')

    writer.close()
    logger.info('Training completed!')
    logger.info(f'Best results: {previous_best}')


if __name__ == '__main__':
    main()
