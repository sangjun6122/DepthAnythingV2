#!/bin/bash
# Depth Anything V2 Fine-tuning 실행 스크립트
# A6000 Ada GPU 2장 사용 (DDP)

# GPU 설정
export CUDA_VISIBLE_DEVICES=0,1

# 프로젝트 디렉토리로 이동
cd "$(dirname "$0")"

# 실행
torchrun \
    --nproc_per_node=2 \
    --master_port=29500 \
    src/train.py \
    --config configs/train_config.yaml \
    "$@"
