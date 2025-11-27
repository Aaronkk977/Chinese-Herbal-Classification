"""
Split dataset: use original val as test, split original train into train + val
- Original data/val/ -> test set (hold-out, untouched until final evaluation)
- Original data/train/ -> train + val set (split by val_ratio)
"""

import os
import shutil
from pathlib import Path
import random
from collections import defaultdict


def split_train_to_train_val(source_dir, output_dir, val_ratio=0.2, seed=42):
    """
    Split a single directory (original train) into train + val.
    
    Args:
        source_dir: Directory containing class folders with images
        output_dir: Output directory for split dataset
        val_ratio: Ratio for validation set taken from train (default 0.2)
        seed: Random seed for reproducibility
    """
    
    assert 0 < val_ratio < 1, "val_ratio must be between 0 and 1"
    
    random.seed(seed)
    
    # Create output directories
    for split in ['train', 'val']:
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)
    
    source_path = Path(source_dir)
    class_dirs = [d for d in source_path.iterdir() if d.is_dir()]
    
    if not class_dirs:
        raise ValueError(f"No class directories found in {source_dir}")
    
    print(f"Found {len(class_dirs)} classes in {source_dir}")
    print(f"Split ratio - Train: {1 - val_ratio:.1%}, Val: {val_ratio:.1%}")
    print("-" * 60)
    
    total_images = 0
    split_counts = defaultdict(int)
    
    for class_dir in sorted(class_dirs):
        class_name = class_dir.name
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
        images = [f for f in class_dir.iterdir()
                  if f.is_file() and f.suffix in image_extensions]
        
        if not images:
            print(f"Warning: No images found in {class_name}, skipping...")
            continue
        
        random.shuffle(images)
        
        n_images = len(images)
        n_val = int(n_images * val_ratio)
        
        val_images = images[:n_val]
        train_images = images[n_val:]
        
        for split, split_images in [('train', train_images), ('val', val_images)]:
            if split_images:
                split_class_dir = os.path.join(output_dir, split, class_name)
                os.makedirs(split_class_dir, exist_ok=True)
                for img in split_images:
                    shutil.copy2(img, os.path.join(split_class_dir, img.name))
                split_counts[split] += len(split_images)
        
        total_images += n_images
        print(f"{class_name:20s}: {n_images:4d} images -> Train: {len(train_images):3d}, Val: {len(val_images):3d}")
    
    print("-" * 60)
    print(f"Total images from original train: {total_images}")
    print(f"New Train set: {split_counts['train']} ({split_counts['train']/total_images:.1%})")
    print(f"New Val set:   {split_counts['val']} ({split_counts['val']/total_images:.1%})")
    
    return split_counts


def copy_as_test(source_dir, output_dir):
    """
    Copy entire source_dir (original val) as test set.
    """
    source_path = Path(source_dir)
    class_dirs = [d for d in source_path.iterdir() if d.is_dir()]
    
    if not class_dirs:
        raise ValueError(f"No class directories found in {source_dir}")
    
    test_output = os.path.join(output_dir, 'test')
    os.makedirs(test_output, exist_ok=True)
    
    total = 0
    for class_dir in sorted(class_dirs):
        class_name = class_dir.name
        image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
        images = [f for f in class_dir.iterdir()
                  if f.is_file() and f.suffix in image_extensions]
        
        if images:
            test_class_dir = os.path.join(test_output, class_name)
            os.makedirs(test_class_dir, exist_ok=True)
            for img in images:
                shutil.copy2(img, os.path.join(test_class_dir, img.name))
            total += len(images)
    
    print(f"Test set (from original val): {total} images")
    return total


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Split Chinese Herbal Dataset: val->test, train->train+val')
    parser.add_argument('--source', type=str, default='data',
                        help='Source directory containing train and val folders')
    parser.add_argument('--output', type=str, default='data_split',
                        help='Output directory for split dataset')
    parser.add_argument('--val-ratio', type=float, default=0.2,
                        help='Validation ratio taken from original train (default: 0.2)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    train_dir = os.path.join(args.source, 'train')
    val_dir = os.path.join(args.source, 'val')
    
    if os.path.exists(train_dir) and os.path.exists(val_dir):
        print("=" * 60)
        print("Splitting dataset:")
        print(f"  Original val/  -> test set (hold-out)")
        print(f"  Original train/ -> train + val (val_ratio={args.val_ratio:.0%})")
        print("=" * 60)
        print()
        
        # Step 1: Copy original val as test
        print("[Step 1] Copying original val/ as test set...")
        test_count = copy_as_test(val_dir, args.output)
        print()
        
        # Step 2: Split original train into train + val
        print("[Step 2] Splitting original train/ into train + val...")
        split_train_to_train_val(train_dir, args.output, val_ratio=args.val_ratio, seed=args.seed)
        print()
        
        print("=" * 60)
        print(f"Dataset split completed! Output directory: {args.output}")
        print("=" * 60)
    else:
        print(f"Error: Expected {train_dir} and {val_dir} to exist!")
        print("Please ensure your source directory contains 'train' and 'val' subdirectories.")
