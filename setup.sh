#!/bin/bash

# Quick start script for Chinese Herbal Classification

echo "=========================================="
echo "Chinese Herbal Classification Quick Start"
echo "=========================================="

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda not found. Please install Miniconda first."
    exit 1
fi

# Create conda environment
echo ""
echo "Step 1: Creating conda environment..."
conda create -y -n herbal python=3.10

# Activate environment
echo ""
echo "Step 2: Activating environment..."
source activate herbal

# Install PyTorch with CUDA
echo ""
echo "Step 3: Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install other dependencies
echo ""
echo "Step 4: Installing other dependencies..."
pip install timm scikit-learn pandas pyyaml tqdm

# Install data augmentation and visualization
echo ""
echo "Step 5: Installing albumentations and visualization tools..."
pip install albumentations opencv-python matplotlib seaborn

# Install TensorBoard for logging
echo ""
echo "Step 6: Installing TensorBoard..."
pip install tensorboard

# Install additional packages
echo ""
echo "Step 7: Installing additional packages..."
pip install kagglehub

echo ""
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""
echo "To activate the environment, run:"
echo "  conda activate herbal"
echo ""
echo "To download the dataset, run:"
echo "  python download_data.py"
echo ""
echo "To start training, run:"
echo "  cd src"
echo "  python main.py --mode train"
echo ""
