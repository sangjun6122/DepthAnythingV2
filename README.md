# Depth Anything V2 Fine-tuning for Microscopy Height Map Estimation

Depth Anything V2 Metric 모델을 현미경 이미지 기반 Height Map 추정에 파인튜닝하는 프로젝트입니다.

## 데이터셋

- **입력**: 20X 현미경 이미지 (top/center/bottom → RGB 합성)
- **출력**: Height Map (μm 단위)
- **크기**: 518x518
- **샘플 수**: 1,190개 (119개 원본 × 10 랜덤 크롭)

## 프로젝트 구조

```
DepthAnythingV2/
├── data/
│   ├── images/          # RGB 입력 이미지
│   ├── depth/           # Height Map GT (.npy)
│   └── dataset_info.json
├── src/                 # 학습 코드
├── configs/             # 설정 파일
└── checkpoints/         # 모델 체크포인트
```

## 참고

- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)
- 원본 데이터: Keyence VK-X3000 Series 레이저 공초점 현미경
