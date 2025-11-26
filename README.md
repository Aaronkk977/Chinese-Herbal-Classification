# Chinese Herbal Classification

Deep learning-based Chinese herbal medicine image classification using ConvNeXt-Tiny with ACMix (Attention-Convolution Mixing) modules.

## 📋 Project Overview

This project implements a state-of-the-art image classification system for Chinese herbal medicines based on research papers. The model achieves high accuracy by combining:

- **ConvNeXt-Tiny** backbone for efficient feature extraction
- **ACMix modules** that blend CNN (local features) and Self-Attention (global features)
- **Stacked FFN Head** for enhanced classification
- **Advanced data augmentation** techniques

### Key Features

- 🎯 Top-1 Accuracy: ~85% (target based on paper)
- 📊 Comprehensive metrics: Top-1, Top-5, Precision, Recall, F1-score, Confusion Matrix
- 🔄 Mixed precision training for faster computation
- 📈 TensorBoard logging for training visualization
- 🎨 Data augmentation: rotation, flipping, cropping, color jittering

## 📁 Project Structure

```
Chinese-Herbal-Classification/
├── configs/
│   └── config.yaml              # Configuration file
├── src/
│   ├── acmix.py                # ACMix module implementation
│   ├── model.py                # ConvNeXt-Tiny + ACMix model
│   ├── dataset.py              # Data loading and preprocessing
│   ├── train.py                # Training script
│   ├── evaluate.py             # Evaluation script
│   ├── utils.py                # Utility functions
│   └── main.py                 # Main entry point
├── data/                       # Dataset directory
├── checkpoints/                # Model checkpoints
├── results/                    # Evaluation results
├── logs/                       # TensorBoard logs
├── download_data.py            # Dataset download script
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended)
- Miniconda or Anaconda

### Installation

1. **Create and activate conda environment:**

```bash
conda create -n herbal python=3.10
conda activate herbal
```

2. **Install dependencies:**

Option 1 - Use automated setup script (recommended):

```bash
bash setup.sh
```

Option 2 - Manual installation:

```bash
pip install -r requirements.txt
```

Option 3 - Install specific packages:

```bash
# Install all dependencies from requirements.txt
pip install -r requirements.txt

# Optional: if your GPU uses CUDA 12.4, install matching PyTorch wheels
# (only if not already pinned in requirements.txt)
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### Dataset Preparation

**Important:** For proper academic evaluation following the paper's methodology, the dataset must be split into train/val/test sets.

#### Step 1: Download Dataset

Download the Chinese herbal medicine dataset from Kaggle:

```bash
python download_data.py
```

This will download and extract the dataset to:
- `data/train/` - Original training images
- `data/val/` - Original validation images
- 100 classes of Chinese herbs
- Approximately 100+ images per class

#### Step 2: Split Dataset (REQUIRED for proper evaluation)

To avoid data leakage and follow academic standards, re-split the dataset into proper train/val/test sets (70%/20%/10%):

```bash
python split_dataset.py --source data --output data_split
```

This will:
- ✅ Merge all images from `data/train/` and `data/val/`
- ✅ Randomly shuffle all images
- ✅ Split into 70% train / 20% val / 10% test
- ✅ Create new directory structure in `data_split/`

**Why this is important:**
- **Train set (70%)**: Used for model training
- **Val set (20%)**: Used for hyperparameter tuning and model selection during training
- **Test set (10%)**: **Completely untouched** until final evaluation - prevents data leakage

#### Step 3: Dataset Structure After Splitting

```
data_split/
├── train/          # Training set (70%)
│   ├── Anxixiang/
│   ├── Baibiandou/
│   ├── ...
│   └── Zirantong/  # 100 herb classes
├── val/            # Validation set (20%)
│   ├── Anxixiang/
│   └── ...
└── test/           # Test set (10%) - DO NOT TOUCH until final evaluation
    ├── Anxixiang/
    └── ...
```

**Custom Split Ratios:**

```bash
# Custom ratios (must sum to 1.0)
python split_dataset.py --source data --output data_split \
    --train-ratio 0.8 --val-ratio 0.15 --test-ratio 0.05

# Set random seed for reproducibility
python split_dataset.py --source data --output data_split --seed 42
```

**Note:** Both `data/` and `data_split/` directories are git-ignored.

## 🎓 Training

### Basic Training

```bash
cd src
python main.py --mode train --config ../configs/config.yaml
```

### GPU Selection

```bash
# Use default GPU (GPU 0)
python main.py --mode train --config ../configs/config.yaml

# Use specific GPU (e.g., GPU 1)
python main.py --mode train --config ../configs/config.yaml --gpu-id 1

```

**First Time Setup:**

Before training, ensure you have:
1. ✅ Installed all dependencies (`bash setup.sh` or `pip install -r requirements.txt`)
2. ✅ Downloaded the dataset (`python download_data.py`)
3. ✅ **Split the dataset** (`python split_dataset.py --source data --output data_split`)
4. ✅ Activated the conda environment (`conda activate herbal`)

The training script will automatically create necessary directories:
- `checkpoints/` - Saved model checkpoints
- `logs/` - TensorBoard training logs  
- `results/` - Evaluation results and metrics

### Advanced Training Options

```bash
# Resume from checkpoint
python main.py --mode train --resume ../checkpoints/checkpoint.pth

# Custom configuration
python main.py --mode train --config ../configs/custom_config.yaml
```

### Training Parameters (from paper)

- **Epochs**: 100
- **Learning Rate**: 0.0002
- **Optimizer**: AdamW
- **Batch Size**: 32
- **Image Size**: 224×224
- **ACMix Blocks**: 2 (K=2)
- **Network Depth**: 22 layers
- **Activation**: GELU
- **Dataset Split**: 70% train / 20% val / 10% test

## 📊 Evaluation

### Evaluate on Test Set

```bash
python main.py --mode evaluate \
    --checkpoint ../checkpoints/model_best.pth \
    --split test
```

### Evaluation Metrics

The evaluation script computes:

- **Top-1 Accuracy**: Primary metric
- **Top-5 Accuracy**: Whether true label is in top-5 predictions
- **Precision, Recall, F1-score**: Per-class and macro-averaged
- **Confusion Matrix**: Visualizes classification errors

Results are saved to `results/` directory:
- `test_metrics.txt`: Summary of metrics
- `test_per_class_metrics.csv`: Per-class performance
- `test_confusion_matrix.png`: Confusion matrix heatmap
- `test_classification_report.txt`: Detailed classification report

## 🔮 Inference

### Single Image Prediction

```bash
python main.py --mode inference \
    --checkpoint ../checkpoints/model_best.pth \
    --image path/to/image.jpg
```

## ⚙️ Configuration

Edit `configs/config.yaml` to customize:

```yaml
data:
  image_size: 224
  num_classes: 100
  batch_size: 32

model:
  num_acmix_blocks: 2
  network_width: 768
  pretrained: true

training:
  epochs: 100
  learning_rate: 0.0002
  optimizer: "adamw"
  use_amp: true
```

## 📈 Monitoring Training

View training progress with TensorBoard:

```bash
tensorboard --logdir logs/
```

Open `http://localhost:6006` in your browser.

## 🏗️ Model Architecture

### ACMix Module

The ACMix module combines:
- **Convolution**: Captures local patterns and textures
- **Multi-Head Self-Attention**: Captures global dependencies
- **Learnable Mixing**: Adaptively weights both operations

### ConvNeXt-Tiny Backbone

- **Stage 1-2**: Low-level features (edges, textures)
  - ACMix inserted at Stage 2
- **Stage 3-4**: High-level features (shapes, semantics)
  - ACMix inserted at Stage 4
- **Classification Head**: Stacked FFN for final prediction

## 📝 Implementation Details

### Data Preprocessing (as per paper)

1. **Resize** to 224×224
2. **Median Filtering** for denoising (kernel size: 5)
3. **Normalization** with ImageNet statistics
4. **Augmentation**:
   - Random rotation (±30°)
   - Horizontal/vertical flipping
   - Random cropping
   - Color jittering

### Key Techniques

- **Mixed Precision Training**: Faster training with fp16
- **Cosine Annealing**: Smooth learning rate decay
- **Label Smoothing**: Prevents overconfidence (ε=0.1)
- **Early Stopping**: Stops when validation plateaus
- **Gradient Clipping**: Stabilizes training

## 🎯 Expected Results

Based on the research paper:

| Metric | Value |
|--------|-------|
| Training Accuracy | ~91% |
| Validation Accuracy | ~85% |
| Test Accuracy | ~80.5% |

*Note: Results may vary based on dataset and hyperparameters*

## 🔬 Research Background

This implementation is based on:

1. **ConvNeXt**: "A ConvNet for the 2020s" (Facebook AI Research)
2. **ACMix**: "On the Integration of Self-Attention and Convolution" 
3. **Paper**: "Image recognition of traditional Chinese medicine based on deep learning"

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

**Happy Training! 🌿🔬**