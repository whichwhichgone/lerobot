#!/bin/bash

# Pi0 Fine-tuning Script
# Based on VSCode launch.json configuration

set -e  # Exit on any error

echo "Starting Pi0 fine-tuning with accelerate..."

# 使用accelerate启动分布式训练
#accelerate launch \
python lerobot/scripts/train.py \
    --policy.path=/liujinxin/zhaowei/models/pi0 \
    --dataset.repo_id=u22_debug_v5_1.0.0_lerobot \
    --dataset.root=/liujinxin/dataset/bimanual/u22_debug_v5_1.0.0_lerobot \
    --wandb.enable=True \
    --wandb.project=pi0_finetune \
    --wandb.entity=yijiulanpishu \
    --wandb.key=83793606f810aa3d385ea5d12dbd352514ac54e1 \
    --job_name=debug_u22_v5_no_state

echo "Training completed!" 