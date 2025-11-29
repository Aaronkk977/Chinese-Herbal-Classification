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
from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import warnings

# Suppress PIL decompression bomb warning for large images
warnings.filterwarnings('ignore', category=Image.DecompressionBombWarning)

# Allow loading truncated JPEG files (some images have broken data streams but are still usable)
ImageFile.LOAD_TRUNCATED_IMAGES = True


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
        
        # Load image using PIL
        with Image.open(img_path) as pil_img:
            # Convert to RGB if needed
            if pil_img.mode == 'P' and 'transparency' in pil_img.info:
                pil_img = pil_img.convert('RGBA')
            if pil_img.mode == 'RGBA':
                background = Image.new('RGB', pil_img.size, (255, 255, 255))
                background.paste(pil_img, mask=pil_img.split()[3])
                pil_img = background
            elif pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')
            
            image = np.array(pil_img)
        
        # Apply transformations
        if self.transform:
            image = self.transform(image=image)['image']
        
        return image, label


def get_train_transforms(config):
    """
    Training transforms with data augmentation
    Matching the specified augmentation pipeline:
    - RandomResizedCrop(224, scale=(0.6, 1.0))
    - RandomHorizontalFlip
    - RandomRotation(20)
    - ColorJitter(0.2, 0.2, 0.2)
    - Normalize (ImageNet stats)
    """
    img_size = config['data']['image_size']  # 224
    mean = config['preprocessing']['mean']
    std = config['preprocessing']['std']
    use_grayscale = config['preprocessing'].get('use_grayscale', False)
    
    transform_list = [
        # RandomResizedCrop: scale=(0.6, 1.0), output size = img_size
        # Use `size=(h, w)` to satisfy albumentations' pydantic schema validation
        A.RandomResizedCrop(
            size=(img_size, img_size),
            scale=(0.6, 1.0),
            ratio=(0.75, 1.333),
            p=1.0
        ),
        # RandomHorizontalFlip
        A.HorizontalFlip(p=0.5),
        # RandomRotation(20) - limit=20 means [-20, +20] degrees
        A.Rotate(limit=20, p=0.5, border_mode=cv2.BORDER_REFLECT_101),
        # ColorJitter(0.2, 0.2, 0.2) - brightness, contrast, saturation
        A.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.0,
            p=0.5
        ),
    ]
    
    # Optional grayscale conversion (as per paper)
    if use_grayscale:
        transform_list.append(A.ToGray(p=1.0))
    
    # Normalize with ImageNet stats + convert to tensor
    transform_list.extend([
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])
    
    return A.Compose(transform_list)


def get_val_transforms(config):
    """
    Validation/Test transforms without augmentation
    Matching the specified pipeline:
    - Resize(224, 224)
    - Normalize (ImageNet stats)
    """
    img_size = config['data']['image_size']  # 224
    mean = config['preprocessing']['mean']
    std = config['preprocessing']['std']
    use_grayscale = config['preprocessing'].get('use_grayscale', False)
    
    transform_list = [
        A.Resize(img_size, img_size),
    ]
    
    # Optional grayscale conversion (must match training if used)
    if use_grayscale:
        transform_list.append(A.ToGray(p=1.0))
    
    transform_list.extend([
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])
    
    return A.Compose(transform_list)


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
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        print(f"Test batches: {len(test_loader)}")
        
        # Test one batch
        images, labels = next(iter(train_loader))
        print(f"Batch shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
    except Exception as e:
        print(f"Error: {e}")
