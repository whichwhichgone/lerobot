#!/bin/bash

# Pi0 Fine-tuning Script
# Based on VSCode launch.json configuration

set -e  # Exit on any error

echo "Starting Pi0 fine-tuning with accelerate..."

# 使用accelerate启动分布式训练
accelerate launch \
    lerobot/scripts/train_accelerate.py \
    --policy.path=/liujinxin/zhaowei/models/pi0 \
    --dataset.repo_id=ur5e_benchmark_v4_with_depth_1.0.0_lerobot \
    --dataset.root=/liujinxin/dataset/ur5e/ur5e_lerobot/ur5e_benchmark_v4_with_depth_1.0.0_lerobot \
    --wandb.enable=True \
    --wandb.project=pi0_finetune \
    --wandb.entity=yijiulanpishu \
    --wandb.key=83793606f810aa3d385ea5d12dbd352514ac54e1 \
    --job_name=debug_ddp

echo "Training completed!" 