import os
import sys
import shutil
from pathlib import Path
import random
from tqdm import tqdm

# Set random seed for reproducible splits
random.seed(42)

# Configure kagglehub to use /tmp2
os.environ['KAGGLE_DATA_DIR'] = '/tmp2/b12902115/kagglehub_data'
os.environ['KAGGLE_CONFIG_DIR'] = '/tmp2/b12902115/kagglehub_config'

import kagglehub

print("="*60)
print("Chinese Herbal Medicine Dataset Setup")
print("="*60)
print(f"Cache directory: {os.environ['KAGGLE_DATA_DIR']}")

# Download dataset
print("\n[1/4] Downloading dataset...")
try:
    path = kagglehub.dataset_download("mumubushimo/herbaldata")
    print(f"✓ Dataset downloaded to: {path}")
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)

# Check dataset structure
print("\n[2/4] Analyzing dataset structure...")
train_dir = Path(path) / "train"
val_dir = Path(path) / "val"

train_classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
val_classes = sorted([d.name for d in val_dir.iterdir() if d.is_dir()])

print(f"  Train classes: {len(train_classes)}")
print(f"  Val classes: {len(val_classes)}")

train_images = sum(1 for _ in train_dir.rglob("*.jpg")) + sum(1 for _ in train_dir.rglob("*.png"))
val_images = sum(1 for _ in val_dir.rglob("*.jpg")) + sum(1 for _ in val_dir.rglob("*.png"))

print(f"  Train images: {train_images}")
print(f"  Val images: {val_images}")

# Create organized dataset structure
print("\n[3/4] Creating organized dataset structure...")
data_root = Path("data/herbal")
data_root.mkdir(parents=True, exist_ok=True)

# Create train/val/test directories
for split in ['train', 'val', 'test']:
    (data_root / split).mkdir(exist_ok=True)

# Copy training data (keep all)
print("  Copying training data...")
for class_dir in tqdm(train_dir.iterdir(), desc="Train"):
    if class_dir.is_dir():
        dst = data_root / "train" / class_dir.name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(class_dir, dst)

# Split validation data into val (50%) and test (50%)
print("  Splitting validation data into val/test...")
for class_dir in tqdm(val_dir.iterdir(), desc="Val/Test"):
    if class_dir.is_dir():
        class_name = class_dir.name
        images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
        random.shuffle(images)
        
        # Split 50-50
        mid = len(images) // 2
        val_images = images[:mid]
        test_images = images[mid:]
        
        # Create class directories
        (data_root / "val" / class_name).mkdir(exist_ok=True)
        (data_root / "test" / class_name).mkdir(exist_ok=True)
        
        # Copy images
        for img in val_images:
            shutil.copy2(img, data_root / "val" / class_name / img.name)
        for img in test_images:
            shutil.copy2(img, data_root / "test" / class_name / img.name)

print("\n[4/4] Verifying dataset structure...")
for split in ['train', 'val', 'test']:
    split_dir = data_root / split
    n_classes = len([d for d in split_dir.iterdir() if d.is_dir()])
    n_images = sum(1 for _ in split_dir.rglob("*.jpg")) + sum(1 for _ in split_dir.rglob("*.png"))
    print(f"  {split:5s}: {n_classes:3d} classes, {n_images:5d} images")

print("\n" + "="*60)
print("✓ Dataset setup complete!")
print("="*60)
print(f"\nDataset location: {data_root.absolute()}")
print("\nYou can now start training with:")
print("  cd src")
print("  python main.py --mode train")
print()