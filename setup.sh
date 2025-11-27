#!/bin/bash

# Quick start script for Chinese Herbal Classification

echo "=========================================="
echo "Chinese Herbal Classification Quick Start"
echo "=========================================="

# Locate conda: prefer PATH, fallback to $HOME/miniconda3
if command -v conda &> /dev/null; then
    CONDA_CMD=$(command -v conda)
elif [ -x "$HOME/miniconda3/bin/conda" ]; then
    CONDA_CMD="$HOME/miniconda3/bin/conda"
    export PATH="$HOME/miniconda3/bin:$PATH"
else
    echo "Error: conda not found. Please install Miniconda first."
    exit 1
fi

# Create conda environment
echo ""
echo "Step 1: Creating conda environment..."
# Ensure Conda Terms of Service are accepted for default channels (non-interactive)
echo "Checking and accepting Conda Terms of Service if required..."
"$CONDA_CMD" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
"$CONDA_CMD" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true

"$CONDA_CMD" create -y -n herbal python=3.10

# Exit if environment creation failed
if [ $? -ne 0 ]; then
    echo "Failed to create conda environment 'herbal'. Please check the conda output above."
    exit 1
fi

# Use `conda run` to execute installs inside the new environment (avoids activation issues)
echo ""
echo "Step 2: Installing packages into 'herbal' environment using 'conda run'..."

echo "Installing packages from requirements.txt (this installs torch from PyPI by default)..."
"$CONDA_CMD" run -n herbal pip install -r requirements.txt

echo "Note: If you need a specific CUDA build of PyTorch, install it manually, e.g."
echo "  conda run -n herbal pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124"

echo "Installing additional/optional packages..."
"$CONDA_CMD" run -n herbal pip install kagglehub || true

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
