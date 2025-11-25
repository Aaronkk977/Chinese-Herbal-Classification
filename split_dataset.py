"""
Split dataset into train/val/test following the paper's methodology
Paper uses 70% train, 20% val, 10% test split
"""

import os
import shutil
from pathlib import Path
import random
from collections import defaultdict

def split_dataset(source_dir, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=42):
    """
    Split dataset into train/val/test sets
    
    Args:
        source_dir: Directory containing class folders with images
        output_dir: Output directory for split dataset
        train_ratio: Ratio for training set (default 0.7)
        val_ratio: Ratio for validation set (default 0.2)
        test_ratio: Ratio for test set (default 0.1)
        seed: Random seed for reproducibility
    """
    
    # Validate ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    # Set random seed
    random.seed(seed)
    
    # Create output directories
    splits = ['train', 'val', 'test']
    for split in splits:
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)
    
    # Get all class directories
    source_path = Path(source_dir)
    class_dirs = [d for d in source_path.iterdir() if d.is_dir()]
    
    if not class_dirs:
        raise ValueError(f"No class directories found in {source_dir}")
    
    print(f"Found {len(class_dirs)} classes")
    print(f"Split ratios - Train: {train_ratio:.1%}, Val: {val_ratio:.1%}, Test: {test_ratio:.1%}")
    print("-" * 60)
    
    total_images = 0
    split_counts = defaultdict(int)
    
    # Process each class
    for class_dir in sorted(class_dirs):
        class_name = class_dir.name
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
        images = [f for f in class_dir.iterdir() 
                 if f.is_file() and f.suffix in image_extensions]
        
        if not images:
            print(f"Warning: No images found in {class_name}, skipping...")
            continue
        
        # Shuffle images
        random.shuffle(images)
        
        # Calculate split indices
        n_images = len(images)
        n_train = int(n_images * train_ratio)
        n_val = int(n_images * val_ratio)
        # n_test will be the remainder to ensure we use all images
        
        # Split images
        train_images = images[:n_train]
        val_images = images[n_train:n_train + n_val]
        test_images = images[n_train + n_val:]
        
        # Copy images to respective directories
        for split, split_images in [('train', train_images), 
                                     ('val', val_images), 
                                     ('test', test_images)]:
            if split_images:
                # Create class directory in split
                split_class_dir = os.path.join(output_dir, split, class_name)
                os.makedirs(split_class_dir, exist_ok=True)
                
                # Copy images
                for img in split_images:
                    shutil.copy2(img, os.path.join(split_class_dir, img.name))
                
                split_counts[split] += len(split_images)
        
        total_images += n_images
        print(f"{class_name:20s}: {n_images:4d} images -> "
              f"Train: {len(train_images):3d}, Val: {len(val_images):3d}, Test: {len(test_images):3d}")
    
    print("-" * 60)
    print(f"Total images: {total_images}")
    print(f"Train set: {split_counts['train']} ({split_counts['train']/total_images:.1%})")
    print(f"Val set:   {split_counts['val']} ({split_counts['val']/total_images:.1%})")
    print(f"Test set:  {split_counts['test']} ({split_counts['test']/total_images:.1%})")
    print(f"\nDataset split completed! Output directory: {output_dir}")


def merge_and_split(train_dir, val_dir, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=42):
    """
    Merge existing train/val splits and re-split into train/val/test
    
    Args:
        train_dir: Existing train directory
        val_dir: Existing val directory
        output_dir: Output directory for new split
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        seed: Random seed
    """
    
    print("Merging existing train and val directories...")
    
    # Create temporary merged directory
    temp_merged = os.path.join(output_dir, '_temp_merged')
    os.makedirs(temp_merged, exist_ok=True)
    
    # Get all classes from train directory
    train_path = Path(train_dir)
    class_dirs = [d.name for d in train_path.iterdir() if d.is_dir()]
    
    # Merge images from both train and val
    for class_name in class_dirs:
        merged_class_dir = os.path.join(temp_merged, class_name)
        os.makedirs(merged_class_dir, exist_ok=True)
        
        # Copy from train
        train_class_dir = os.path.join(train_dir, class_name)
        if os.path.exists(train_class_dir):
            for img in Path(train_class_dir).iterdir():
                if img.is_file():
                    shutil.copy2(img, os.path.join(merged_class_dir, f"train_{img.name}"))
        
        # Copy from val
        val_class_dir = os.path.join(val_dir, class_name)
        if os.path.exists(val_class_dir):
            for img in Path(val_class_dir).iterdir():
                if img.is_file():
                    shutil.copy2(img, os.path.join(merged_class_dir, f"val_{img.name}"))
    
    print("Merging completed. Now splitting into train/val/test...")
    print()
    
    # Split the merged dataset
    split_dataset(temp_merged, output_dir, train_ratio, val_ratio, test_ratio, seed)
    
    # Clean up temp directory
    shutil.rmtree(temp_merged)
    print(f"Temporary directory removed.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Split Chinese Herbal Dataset')
    parser.add_argument('--source', type=str, default='data',
                        help='Source directory containing train and val folders')
    parser.add_argument('--output', type=str, default='data_split',
                        help='Output directory for split dataset')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                        help='Training set ratio (default: 0.7)')
    parser.add_argument('--val-ratio', type=float, default=0.2,
                        help='Validation set ratio (default: 0.2)')
    parser.add_argument('--test-ratio', type=float, default=0.1,
                        help='Test set ratio (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    # Check if source has train and val subdirectories
    train_dir = os.path.join(args.source, 'train')
    val_dir = os.path.join(args.source, 'val')
    
    if os.path.exists(train_dir) and os.path.exists(val_dir):
        print("Found existing train/val split. Merging and re-splitting...")
        merge_and_split(
            train_dir=train_dir,
            val_dir=val_dir,
            output_dir=args.output,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed
        )
    elif os.path.isdir(args.source):
        print("Splitting dataset from single directory...")
        split_dataset(
            source_dir=args.source,
            output_dir=args.output,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed
        )
    else:
        print(f"Error: Source directory {args.source} not found!")
