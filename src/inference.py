"""
Depth Anything V2 추론 스크립트
파인튜닝된 모델로 Height Map 추정

사용법:
    python src/inference.py --model-path runs/microscopy_v1/best_model.pth --input-path data/images/1_0.jpg --output-path output/
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent))

from depth_anything_v2.dpt import DepthAnythingV2


def parse_args():
    parser = argparse.ArgumentParser(description='Depth Anything V2 Inference')

    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to trained model weights')
    parser.add_argument('--input-path', type=str, required=True,
                        help='Path to input image or directory')
    parser.add_argument('--output-path', type=str, default='output',
                        help='Output directory')
    parser.add_argument('--encoder', type=str, default='vitb',
                        choices=['vits', 'vitb', 'vitl'])
    parser.add_argument('--max-depth', type=float, default=100.0,
                        help='Maximum depth value (μm)')
    parser.add_argument('--input-size', type=int, default=518,
                        help='Input image size')
    parser.add_argument('--save-numpy', action='store_true',
                        help='Save depth as numpy file')
    parser.add_argument('--save-colormap', action='store_true', default=True,
                        help='Save depth as colormap image')

    return parser.parse_args()


def load_model(model_path, encoder, max_depth, device):
    """모델 로드"""
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }

    model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})

    state_dict = torch.load(model_path, map_location='cpu')

    # DDP로 학습된 모델인 경우 'module.' prefix 제거
    if 'model' in state_dict:
        state_dict = state_dict['model']

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    return model


def predict_depth(model, image_path, input_size, device):
    """단일 이미지 깊이 추정"""
    # 이미지 로드
    raw_img = cv2.imread(str(image_path))
    raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)

    # 모델 추론
    with torch.no_grad():
        depth = model.infer_image(raw_img, input_size)

    return depth, raw_img


def save_depth_colormap(depth, output_path, colormap=cv2.COLORMAP_INFERNO):
    """깊이맵을 컬러맵 이미지로 저장"""
    depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    depth_uint8 = (depth_normalized * 255).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_uint8, colormap)
    cv2.imwrite(str(output_path), depth_colored)


def main():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 모델 로드
    print(f'Loading model from {args.model_path}')
    model = load_model(args.model_path, args.encoder, args.max_depth, device)

    # 출력 디렉토리 생성
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # 입력 경로 처리
    input_path = Path(args.input_path)
    if input_path.is_file():
        image_files = [input_path]
    else:
        image_files = list(input_path.glob('*.jpg')) + list(input_path.glob('*.png'))

    print(f'Processing {len(image_files)} images...')

    for img_path in image_files:
        print(f'  Processing: {img_path.name}')

        # 깊이 추정
        depth, raw_img = predict_depth(model, img_path, args.input_size, device)

        # 결과 저장
        stem = img_path.stem

        if args.save_numpy:
            np.save(output_path / f'{stem}_depth.npy', depth)

        if args.save_colormap:
            save_depth_colormap(depth, output_path / f'{stem}_depth.png')

        # 통계 출력
        print(f'    Depth range: {depth.min():.2f} ~ {depth.max():.2f} μm')

    print(f'\nResults saved to {output_path}')


if __name__ == '__main__':
    main()
