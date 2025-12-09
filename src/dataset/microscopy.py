"""
현미경 Height Map 데이터셋
- RGB 이미지 (top/center/bottom 합성)
- Height Map GT (μm 단위)
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from .transform import NormalizeImage, PrepareForNet


class MicroscopyDataset(Dataset):
    """
    현미경 이미지 Height Map 데이터셋

    Args:
        data_dir: 데이터 디렉토리 (images/, depth/ 하위 폴더 포함)
        mode: 'train' 또는 'val'
        size: 이미지 크기 (width, height)
        train_ratio: 학습 데이터 비율 (기본 0.9)
    """

    def __init__(self, data_dir, mode='train', size=(518, 518), train_ratio=0.9):
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.size = size

        # 이미지 파일 목록 가져오기
        image_dir = self.data_dir / 'images'
        all_images = sorted(list(image_dir.glob('*.jpg')))

        # Train/Val 분할
        n_train = int(len(all_images) * train_ratio)
        if mode == 'train':
            self.image_files = all_images[:n_train]
        else:
            self.image_files = all_images[n_train:]

        # 전처리 파이프라인 (이미지가 이미 518x518이므로 Resize 불필요)
        self.transform = Compose([
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])

        print(f"[{mode.upper()}] Loaded {len(self.image_files)} samples from {data_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 이미지 경로
        img_path = self.image_files[idx]
        sample_name = img_path.stem  # e.g., "1_0"

        # Depth 경로
        depth_path = self.data_dir / 'depth' / f'{sample_name}.npy'

        # 이미지 로드
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0

        # Depth 로드 (μm 단위)
        depth = np.load(str(depth_path)).astype(np.float32)

        # 전처리 적용
        sample = self.transform({'image': image, 'depth': depth})

        sample['image'] = torch.from_numpy(sample['image'])
        sample['depth'] = torch.from_numpy(sample['depth'])

        # Valid mask (유효한 depth 값)
        sample['valid_mask'] = (sample['depth'] > 0) & (~torch.isnan(sample['depth']))
        sample['depth'][~sample['valid_mask']] = 0

        sample['image_path'] = str(img_path)

        return sample


class MicroscopyDatasetFromList(Dataset):
    """
    파일 리스트 기반 현미경 데이터셋 (train.txt, val.txt 사용)
    """

    def __init__(self, filelist_path, data_dir, mode='train', size=(518, 518)):
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.size = size

        # 파일 리스트 로드
        with open(filelist_path, 'r') as f:
            self.filelist = [line.strip() for line in f.readlines() if line.strip()]

        self.transform = Compose([
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])

        print(f"[{mode.upper()}] Loaded {len(self.filelist)} samples from {filelist_path}")

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        sample_name = self.filelist[idx]

        img_path = self.data_dir / 'images' / f'{sample_name}.jpg'
        depth_path = self.data_dir / 'depth' / f'{sample_name}.npy'

        # 이미지 로드
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0

        # Depth 로드
        depth = np.load(str(depth_path)).astype(np.float32)

        sample = self.transform({'image': image, 'depth': depth})

        sample['image'] = torch.from_numpy(sample['image'])
        sample['depth'] = torch.from_numpy(sample['depth'])

        sample['valid_mask'] = (sample['depth'] > 0) & (~torch.isnan(sample['depth']))
        sample['depth'][~sample['valid_mask']] = 0

        sample['image_path'] = str(img_path)

        return sample
