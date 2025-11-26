"""
Reconstruct proper splits (data_split/) from a dataset produced by download_data.py

Scenario handled:
- download_data.py produced `data/herbal` with directories: train/, val/, test/
  where original `val/` was previously split into two halves (val & test).

This script will:
- Merge `data_root/val/` and `data_root/test/` into a single 'original_val' collection
- Copy that merged set as `output/test/` (hold-out)
- Split `data_root/train/` into `output/train/` and `output/val/` according to --val-ratio

Usage:
    python reconstruct_split_from_download.py --source data/herbal --output data_split --val-ratio 0.2

"""

import os
import shutil
from pathlib import Path
import random
from collections import defaultdict
import argparse


def merge_val_and_test(data_root):
    """Return a dict mapping class_name -> list of image Paths that represent the original val"""
    data_root = Path(data_root)
    val_dir = data_root / 'val'
    test_dir = data_root / 'test'

    classes = set()
    if val_dir.exists():
        classes.update([d.name for d in val_dir.iterdir() if d.is_dir()])
    if test_dir.exists():
        classes.update([d.name for d in test_dir.iterdir() if d.is_dir()])

    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}

    merged = {}
    for class_name in sorted(classes):
        imgs = []
        v_path = val_dir / class_name
        t_path = test_dir / class_name
        if v_path.exists():
            imgs += [p for p in v_path.iterdir() if p.is_file() and p.suffix in image_extensions]
        if t_path.exists():
            imgs += [p for p in t_path.iterdir() if p.is_file() and p.suffix in image_extensions]
        merged[class_name] = imgs

    return merged


def split_train(source_train_dir, output_dir, val_ratio=0.2, seed=42):
    random.seed(seed)
    source_path = Path(source_train_dir)
    class_dirs = [d for d in source_path.iterdir() if d.is_dir()]

    if not class_dirs:
        raise ValueError(f"No class directories found in {source_train_dir}")

    split_counts = defaultdict(int)
    for class_dir in sorted(class_dirs):
        class_name = class_dir.name
        image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
        images = [f for f in class_dir.iterdir() if f.is_file() and f.suffix in image_extensions]
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
                split_class_dir = Path(output_dir) / split / class_name
                split_class_dir.mkdir(parents=True, exist_ok=True)
                for img in split_images:
                    shutil.copy2(img, split_class_dir / img.name)
                split_counts[split] += len(split_images)

        print(f"{class_name:20s}: {n_images:4d} images -> Train: {len(train_images):3d}, Val: {len(val_images):3d}")

    total = sum(split_counts.values())
    print("-" * 60)
    print(f"New Train set: {split_counts['train']}")
    print(f"New Val set:   {split_counts['val']}")

    return split_counts


def write_test(merged_val_dict, output_dir):
    test_out = Path(output_dir) / 'test'
    test_out.mkdir(parents=True, exist_ok=True)
    total = 0
    for class_name, imgs in merged_val_dict.items():
        if not imgs:
            continue
        class_out = test_out / class_name
        class_out.mkdir(parents=True, exist_ok=True)
        for img in imgs:
            shutil.copy2(img, class_out / img.name)
            total += 1
    print(f"Test set (from merged val/test): {total} images")
    return total


def main():
    parser = argparse.ArgumentParser(description='Reconstruct dataset split from download_data output')
    parser.add_argument('--source', type=str, default='data/herbal', help='Source directory produced by download_data.py')
    parser.add_argument('--output', type=str, default='data_split', help='Output directory for reconstructed split')
    parser.add_argument('--val-ratio', type=float, default=0.2, help='Validation ratio to take from train')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    src = Path(args.source)
    if not src.exists():
        raise SystemExit(f"Source path not found: {src}")

    # Step 1: merge existing val/ and test/ to recover original val
    print("Merging existing val/ and test/ to reconstruct original val...")
    merged = merge_val_and_test(src)

    # Step 2: write merged as output/test
    print("Writing merged set to output/test...")
    write_test(merged, args.output)

    # Step 3: split train -> train + val
    train_dir = src / 'train'
    if not train_dir.exists():
        raise SystemExit(f"Train dir not found under source: {train_dir}")

    print("Splitting original train into train + val...")
    split_train(train_dir, args.output, val_ratio=args.val_ratio, seed=args.seed)

    print("Reconstruction complete. Output directory:", args.output)


if __name__ == '__main__':
    main()
