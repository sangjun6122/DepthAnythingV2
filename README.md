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
│   ├── images/              # RGB 입력 이미지 (518x518)
│   ├── depth/               # Height Map GT (.npy)
│   └── dataset_info.json
├── src/
│   ├── dataset/             # 데이터셋 클래스
│   ├── depth_anything_v2/   # 모델 코드
│   ├── util/                # Loss, Metric 등
│   ├── train.py             # 학습 스크립트
│   └── inference.py         # 추론 스크립트
├── configs/
│   └── train_config.yaml    # 학습 설정
├── checkpoints/             # 사전학습 가중치
├── runs/                    # 학습 로그 및 체크포인트
├── train.sh                 # 학습 실행 스크립트
└── requirements.txt
```

## 설치

```bash
# 가상환경 생성
python -m venv venv
source venv/bin/activate

# 의존성 설치
pip install -r requirements.txt

# PyTorch (CUDA 버전에 맞게 설치)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## 학습

### 2 GPU DDP 학습 (A6000 Ada × 2)

```bash
# 기본 설정으로 학습
./train.sh

# 또는 직접 실행
torchrun --nproc_per_node=2 src/train.py --config configs/train_config.yaml

# 설정 오버라이드
./train.sh --epochs 100 --batch-size 8 --lr 0.00001
```

### 설정 파일 수정

`configs/train_config.yaml`에서 학습 파라미터를 조정할 수 있습니다:

```yaml
model:
  encoder: vitb              # vits, vitb, vitl
  max_depth: 100.0           # μm 단위 최대 깊이
  pretrained_from: checkpoints/depth_anything_v2_metric_hypersim_vitb.pth

training:
  epochs: 50
  batch_size: 4              # GPU당 배치 크기
  lr: 0.000005
```

## 추론

```bash
# 단일 이미지
python src/inference.py \
    --model-path runs/microscopy_v1/best_model.pth \
    --input-path data/images/1_0.jpg \
    --output-path output/

# 디렉토리 전체
python src/inference.py \
    --model-path runs/microscopy_v1/best_model.pth \
    --input-path data/images/ \
    --output-path output/ \
    --save-numpy
```

## TensorBoard

```bash
tensorboard --logdir runs/
```

## 참고

- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)
- 원본 데이터: Keyence VK-X3000 Series 레이저 공초점 현미경

## License

This project is based on [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2).
