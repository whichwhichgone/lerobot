#!/bin/bash

# Pi0 Fine-tuning with Proxy Setup Script
# 1. 启动 Clash 代理服务（后台运行）
# 2. 设置网络代理
# 3. 执行 Pi0 训练

set -e  # Exit on any error

echo "==============================================================================" 
echo "Pi0 Fine-tuning with Proxy Setup"
echo "=============================================================================="

# =============================================================================
# 1. 启动 Clash 代理服务
# =============================================================================

echo "Step 1: Starting Clash proxy service..."

# 检查 Clash 启动脚本是否存在
CLASH_SCRIPT="/liujinxin/zhaowei/mvp/launch_clash.sh"
if [ ! -f "$CLASH_SCRIPT" ]; then
    echo "Error: Clash script not found: $CLASH_SCRIPT"
    exit 1
fi

# 启动 Clash 并置于后台
echo "Launching Clash proxy service in background..."
bash "$CLASH_SCRIPT" &
CLASH_PID=$!

echo "✓ Clash started with PID: $CLASH_PID"

# 等待 Clash 启动完成
echo "Waiting for Clash to start up..."
sleep 5

# =============================================================================
# 2. 设置网络代理
# =============================================================================

echo "Step 2: Setting up network proxy..."

# 设置 HTTP/HTTPS 代理
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890

echo "✓ Proxy settings configured:"
echo "  HTTP_PROXY: $HTTP_PROXY"
echo "  HTTPS_PROXY: $HTTPS_PROXY"

# =============================================================================
# 3. 执行 Pi0 训练脚本
# =============================================================================

echo "Step 3: Starting Pi0 fine-tuning..."

# 检查训练脚本是否存在
TRAINING_SCRIPT="/liujinxin/zhaowei/lerobot/custom/run_pi0_finetune.sh"

# 确保训练脚本有执行权限
chmod +x "$TRAINING_SCRIPT"

echo ""
echo "=============================================================================="
echo "Starting Pi0 training with proxy enabled..."
echo "=============================================================================="

# 执行训练脚本
echo "Executing: $TRAINING_SCRIPT"
bash "$TRAINING_SCRIPT"

echo ""
echo "=============================================================================="
echo "All tasks completed successfully!"
echo "==============================================================================" 