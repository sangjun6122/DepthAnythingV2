"""
학습된 Depth Anything V2 모델의 결과를 시각화하는 스크립트

사용법:
    python src/visualize_results.py --checkpoint runs/microscopy_v1/best_model.pth --output vis_results
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from depth_anything_v2.dpt import DepthAnythingV2


def load_model(checkpoint_path, encoder='vitb', device='cuda'):
    """모델 로드"""
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }

    model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': 100.0})

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    return model


def colorize_depth(depth, vmin=None, vmax=None, cmap='turbo'):
    """Depth map을 컬러맵으로 변환"""
    if vmin is None:
        vmin = np.nanmin(depth)
    if vmax is None:
        vmax = np.nanmax(depth)

    depth_normalized = (depth - vmin) / (vmax - vmin + 1e-8)
    depth_normalized = np.clip(depth_normalized, 0, 1)

    cmap_func = plt.get_cmap(cmap)
    colored = cmap_func(depth_normalized)[:, :, :3]
    colored = (colored * 255).astype(np.uint8)

    return colored


def visualize_sample(model, img_path, depth_path, output_dir, device='cuda'):
    """단일 샘플 시각화"""
    # 이미지 로드
    img = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # GT depth 로드
    gt_depth = np.load(str(depth_path))

    # 추론
    with torch.no_grad():
        pred_depth = model.infer_image(img_rgb)

    # 동일한 범위로 컬러화
    vmin = min(np.nanmin(gt_depth), np.nanmin(pred_depth))
    vmax = max(np.nanmax(gt_depth), np.nanmax(pred_depth))

    gt_colored = colorize_depth(gt_depth, vmin, vmax)
    pred_colored = colorize_depth(pred_depth, vmin, vmax)

    # 에러맵 계산
    error_map = np.abs(pred_depth - gt_depth)
    error_colored = colorize_depth(error_map, 0, vmax * 0.2, cmap='hot')

    # 메트릭 계산
    mask = gt_depth > 0.1
    abs_rel = np.mean(np.abs(pred_depth[mask] - gt_depth[mask]) / gt_depth[mask])
    rmse = np.sqrt(np.mean((pred_depth[mask] - gt_depth[mask]) ** 2))

    # 비율 계산 (d1, d2, d3)
    thresh = np.maximum(pred_depth[mask] / gt_depth[mask], gt_depth[mask] / pred_depth[mask])
    d1 = np.mean(thresh < 1.25)
    d2 = np.mean(thresh < 1.25 ** 2)
    d3 = np.mean(thresh < 1.25 ** 3)

    # Figure 생성
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 원본 이미지
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title('Input Image', fontsize=12)
    axes[0, 0].axis('off')

    # GT Depth
    im1 = axes[0, 1].imshow(gt_colored)
    axes[0, 1].set_title(f'GT Height Map\n(min: {vmin:.2f}, max: {vmax:.2f} μm)', fontsize=12)
    axes[0, 1].axis('off')

    # Predicted Depth
    axes[1, 0].imshow(pred_colored)
    axes[1, 0].set_title(f'Predicted Height Map\n(min: {np.min(pred_depth):.2f}, max: {np.max(pred_depth):.2f} μm)', fontsize=12)
    axes[1, 0].axis('off')

    # Error Map
    axes[1, 1].imshow(error_colored)
    axes[1, 1].set_title(f'Absolute Error\nRMSE: {rmse:.3f} μm, abs_rel: {abs_rel:.4f}', fontsize=12)
    axes[1, 1].axis('off')

    # 메트릭 텍스트 추가
    metrics_text = f'd1: {d1:.4f}  d2: {d2:.4f}  d3: {d3:.4f}'
    fig.suptitle(f'{img_path.stem}\n{metrics_text}', fontsize=14, fontweight='bold')

    plt.tight_layout()

    # 저장
    output_path = output_dir / f'{img_path.stem}_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return {
        'name': img_path.stem,
        'abs_rel': abs_rel,
        'rmse': rmse,
        'd1': d1,
        'd2': d2,
        'd3': d3
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='runs/microscopy_v1/best_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Data directory')
    parser.add_argument('--output', type=str, default='vis_results',
                        help='Output directory')
    parser.add_argument('--encoder', type=str, default='vitb',
                        choices=['vits', 'vitb', 'vitl'])
    parser.add_argument('--num-samples', type=int, default=20,
                        help='Number of samples to visualize')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    # 출력 디렉토리 생성
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 모델 로드
    print(f'Loading model from {args.checkpoint}...')
    model = load_model(args.checkpoint, args.encoder, args.device)

    # 데이터 경로
    data_dir = Path(args.data_dir)
    image_dir = data_dir / 'images'
    depth_dir = data_dir / 'depth'

    # 이미지 파일 목록
    image_files = sorted(list(image_dir.glob('*.jpg')))

    # Validation set에서 샘플링 (마지막 10%)
    val_start = int(len(image_files) * 0.9)
    val_images = image_files[val_start:]

    # 샘플 선택
    if args.num_samples < len(val_images):
        indices = np.linspace(0, len(val_images) - 1, args.num_samples, dtype=int)
        selected_images = [val_images[i] for i in indices]
    else:
        selected_images = val_images

    print(f'Visualizing {len(selected_images)} samples...')

    # 시각화
    all_metrics = []
    for img_path in tqdm(selected_images, desc='Visualizing'):
        depth_path = depth_dir / f'{img_path.stem}.npy'
        if depth_path.exists():
            metrics = visualize_sample(model, img_path, depth_path, output_dir, args.device)
            all_metrics.append(metrics)

    # 평균 메트릭 계산
    if all_metrics:
        avg_metrics = {
            'abs_rel': np.mean([m['abs_rel'] for m in all_metrics]),
            'rmse': np.mean([m['rmse'] for m in all_metrics]),
            'd1': np.mean([m['d1'] for m in all_metrics]),
            'd2': np.mean([m['d2'] for m in all_metrics]),
            'd3': np.mean([m['d3'] for m in all_metrics]),
        }

        print('\n' + '=' * 60)
        print('Average Metrics:')
        print(f"  d1: {avg_metrics['d1']:.4f}")
        print(f"  d2: {avg_metrics['d2']:.4f}")
        print(f"  d3: {avg_metrics['d3']:.4f}")
        print(f"  abs_rel: {avg_metrics['abs_rel']:.4f}")
        print(f"  rmse: {avg_metrics['rmse']:.4f} μm")
        print('=' * 60)

        # 메트릭 저장
        with open(output_dir / 'metrics.txt', 'w') as f:
            f.write('Average Metrics\n')
            f.write('=' * 40 + '\n')
            f.write(f"d1: {avg_metrics['d1']:.4f}\n")
            f.write(f"d2: {avg_metrics['d2']:.4f}\n")
            f.write(f"d3: {avg_metrics['d3']:.4f}\n")
            f.write(f"abs_rel: {avg_metrics['abs_rel']:.4f}\n")
            f.write(f"rmse: {avg_metrics['rmse']:.4f} μm\n")
            f.write('\nPer-sample Metrics\n')
            f.write('=' * 40 + '\n')
            for m in all_metrics:
                f.write(f"{m['name']}: d1={m['d1']:.4f}, rmse={m['rmse']:.4f}\n")

    print(f'\nResults saved to {output_dir}/')


if __name__ == '__main__':
    main()
