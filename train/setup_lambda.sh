#!/bin/bash
# Lambda Cloud Setup Script for LeRobot PI 0.5 Training
# Run this after transferring files to the Lambda instance

set -e  # Exit on error

echo "=========================================="
echo "LeRobot Lambda Setup Script"
echo "=========================================="

# Step 1: Install lerobot
echo ""
echo "[1/7] Installing lerobot..."
pip install lerobot

# Step 2: Install exact versions known to work with LeRobot + PI05
echo ""
echo "[2/7] Installing compatible huggingface_hub, transformers, and tokenizers..."
pip install "huggingface_hub==0.35.3" \
            "transformers @ git+https://github.com/huggingface/transformers.git@fix/lerobot_openpi" \
            "tokenizers==0.21.4" \
            --no-deps

# Step 3: Force LeRobot to accept these versions (skip strict version check)
echo ""
echo "[3/7] Reinstalling lerobot with --no-deps..."
pip install "lerobot==0.4.2" --no-deps

# Step 4: Fix NumPy/SciPy compatibility
echo ""
echo "[4/7] Installing compatible NumPy and SciPy versions..."
pip install "numpy==1.24.4" "scipy==1.11.4" --force-reinstall

# Step 5: Install tmux
echo ""
echo "[5/7] Installing tmux..."
sudo apt install -y tmux

# Step 6: Verify GPU
echo ""
echo "[6/7] Verifying GPU..."
nvidia-smi
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Device: {torch.cuda.get_device_name(0)}')"

# Step 7: Login to WandB and HuggingFace
echo ""
echo "[7/7] Logging into WandB and HuggingFace..."
wandb login ADD_KEY
huggingface-cli login --token ADD_TOKEN

# Dataset will be downloaded from HuggingFace: bdhillon/PI-01.07.26-v3-quantiles
echo ""
echo "Dataset: bdhillon/PI-01.07.26-v3-quantiles (will be downloaded from HuggingFace)"
echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. tmux new -s lerobot"
echo "  2. cd ~/lerobot-training"
echo "  3. python train.py"
echo ""