"""
데이터 전처리 변환
"""

import cv2
import numpy as np


class NormalizeImage(object):
    """이미지 정규화 (ImageNet mean/std)"""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        sample["image"] = (sample["image"] - self.mean) / self.std
        return sample


class PrepareForNet(object):
    """네트워크 입력 준비 (HWC -> CHW, contiguous)"""

    def __init__(self):
        pass

    def __call__(self, sample):
        image = np.transpose(sample["image"], (2, 0, 1))
        sample["image"] = np.ascontiguousarray(image).astype(np.float32)

        if "depth" in sample:
            depth = sample["depth"].astype(np.float32)
            sample["depth"] = np.ascontiguousarray(depth)

        return sample


class RandomHorizontalFlip(object):
    """랜덤 수평 뒤집기"""

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        if np.random.random() < self.prob:
            sample["image"] = np.fliplr(sample["image"]).copy()
            if "depth" in sample:
                sample["depth"] = np.fliplr(sample["depth"]).copy()
        return sample


class RandomVerticalFlip(object):
    """랜덤 수직 뒤집기"""

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        if np.random.random() < self.prob:
            sample["image"] = np.flipud(sample["image"]).copy()
            if "depth" in sample:
                sample["depth"] = np.flipud(sample["depth"]).copy()
        return sample
