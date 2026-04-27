#!/bin/bash
# Duality AI Hackathon - Environment Setup (Mac/Linux)
# Run: bash setup_env.sh

set -e

ENV_NAME="EDU"

echo "======================================================"
echo "  Duality AI Segmentation - Environment Setup"
echo "  Creating conda environment: $ENV_NAME"
echo "======================================================"

conda create -y -n $ENV_NAME python=3.10
conda activate $ENV_NAME

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install \
    numpy \
    Pillow \
    matplotlib \
    tqdm \
    scikit-learn \
    opencv-python-headless

echo ""
echo "======================================================"
echo "  Setup complete!"
echo "  Activate with: conda activate $ENV_NAME"
echo "  Then run:      python train.py"
echo "======================================================"
