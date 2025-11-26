"""
Data preprocessing and loading for Chinese Herbal Classification
Based on paper specifications:
- Image size: 224x224x3
- Normalization with linear transformation
- Grayscale conversion with weighted averaging
- Median filtering for denoising
- Data augmentation: rotation, mirroring, random cropping
"""

import os
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2


class HerbalDataset(Dataset):
    """Custom dataset for Chinese Herbal Medicine images"""
    
    def __init__(self, root_dir, split='train', transform=None, config=None):
        """
        Args:
            root_dir: Path to dataset directory
            split: 'train', 'val', or 'test'
            transform: Image transformations
            config: Configuration dictionary
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.config = config
        
        # Load image paths and labels
        self.samples = self._load_samples()
        
    def _load_samples(self):
        """Load image paths and corresponding labels"""
        samples = []
        split_dir = os.path.join(self.root_dir, self.split)
        
        if not os.path.exists(split_dir):
            raise ValueError(f"Split directory not found: {split_dir}")
        
        # Assume directory structure: root_dir/split/class_name/image.jpg
        classes = sorted(os.listdir(split_dir))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
        self.idx_to_class = {idx: cls_name for cls_name, idx in self.class_to_idx.items()}
        
        for class_name in classes:
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
                
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    samples.append((img_path, self.class_to_idx[class_name]))
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Read image
        image = cv2.imread(img_path)
        
        if image is None:
            print(f"[READ FAILED] Cannot find or read image: {img_path}")
            raise ValueError(f"Image not found at {img_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply median filtering for denoising (as per paper)
        if self.config and self.config.get('preprocessing', {}).get('median_filter_kernel'):
            kernel_size = self.config['preprocessing']['median_filter_kernel']
            image = cv2.medianBlur(image, kernel_size)
        
        # Apply transformations
        if self.transform:
            image = self.transform(image=image)['image']
        
        return image, label


def get_train_transforms(config):
    """
    Training transforms with data augmentation
    Based on paper: rotation, mirroring, random cropping
    """
    img_size = config['data']['image_size']
    mean = config['preprocessing']['mean']
    std = config['preprocessing']['std']
    
    return A.Compose([
        A.Resize(img_size + 32, img_size + 32),
        A.RandomCrop(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=config['augmentation']['rotation_degrees'], p=0.5),
        A.ColorJitter(
            brightness=config['augmentation']['color_jitter']['brightness'],
            contrast=config['augmentation']['color_jitter']['contrast'],
            saturation=config['augmentation']['color_jitter']['saturation'],
            p=0.3
        ),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


def get_val_transforms(config):
    """Validation/Test transforms without augmentation"""
    img_size = config['data']['image_size']
    mean = config['preprocessing']['mean']
    std = config['preprocessing']['std']
    
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


def create_dataloaders(config):
    """Create train, validation, and test dataloaders"""
    
    data_path = config['data']['dataset_path']
    
    # Convert to absolute path if relative
    if not os.path.isabs(data_path):
        # Get the project root directory (parent of src/)
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_path = os.path.join(project_root, data_path)
    
    batch_size = config['training']['batch_size']
    num_workers = config['data']['num_workers']
    
    # Create datasets
    train_dataset = HerbalDataset(
        root_dir=data_path,
        split='train',
        transform=get_train_transforms(config),
        config=config
    )
    
    val_dataset = HerbalDataset(
        root_dir=data_path,
        split='val',
        transform=get_val_transforms(config),
        config=config
    )
    
    # Check if test split exists, otherwise use val for testing
    test_split_path = os.path.join(data_path, 'test')
    if os.path.exists(test_split_path):
        test_dataset = HerbalDataset(
            root_dir=data_path,
            split='test',
            transform=get_val_transforms(config),
            config=config
        )
    else:
        # Use validation set as test set if test split doesn't exist
        print("Warning: Test split not found, using validation set for testing")
        test_dataset = val_dataset
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, train_dataset.class_to_idx


if __name__ == "__main__":
    # Test the dataset
    import yaml
    
    with open('../configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    try:
        train_loader, val_loader, test_loader, class_to_idx = create_dataloaders(config)
        print(f"Number of classes: {len(class_to_idx)}")
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        print(f"Test batches: {len(test_loader)}")
        
        # Test one batch
        images, labels = next(iter(train_loader))
        print(f"Batch shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
    except Exception as e:
        print(f"Error: {e}")
