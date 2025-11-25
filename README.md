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

```bash
pip install -r requirements.txt
```

Or install specific packages:

```bash
# PyTorch with CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Other dependencies
pip install timm albumentations scikit-learn pandas opencv-python matplotlib seaborn pyyaml tensorboard tqdm
```

### Dataset Preparation

1. **Download dataset:**

```bash
python download_data.py
```

2. **Organize dataset structure:**

```
data/herbal/
├── train/
│   ├── class1/
│   │   ├── image1.jpg
│   │   └── ...
│   ├── class2/
│   └── ...
├── val/
│   └── ...
└── test/
    └── ...
```

## 🎓 Training

### Basic Training

```bash
cd src
python main.py --mode train --config ../configs/config.yaml
```

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