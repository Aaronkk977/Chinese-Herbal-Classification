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

# Global set to track already-logged corrupt files (avoid spamming logs)
_logged_corrupt_files = set()
# Log file path for corrupt images
_corrupt_log_path = None


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
    
    def _log_corrupt_file(self, img_path, reason):
        """Log corrupt file path to a log file and optionally delete it"""
        global _logged_corrupt_files, _corrupt_log_path
        
        if img_path in _logged_corrupt_files:
            return  # Already logged
        
        _logged_corrupt_files.add(img_path)
        
        # Initialize log path if not set
        if _corrupt_log_path is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            _corrupt_log_path = os.path.join(project_root, 'corrupt_images.log')
        
        # Log to file
        with open(_corrupt_log_path, 'a') as f:
            f.write(f"{img_path} | {reason}\n")
        
        print(f"[CORRUPT] {reason}: {img_path}")
        
        # Optional: uncomment below to auto-delete corrupt files
        # try:
        #     os.remove(img_path)
        #     print(f"[DELETED] {img_path}")
        # except Exception as e:
        #     print(f"[DELETE FAILED] {img_path}: {e}")
    
    def _validate_image(self, img_path):
        """
        Validate image integrity using PIL (catches truncated/corrupt JPEGs)
        Returns: (is_valid, image_array or None, error_message)
        """
        try:
            # First, try to fully load and verify with PIL
            # This catches "Corrupt JPEG data: premature end of data segment"
            with Image.open(img_path) as pil_img:
                pil_img.load()  # Force full decode - this triggers errors for corrupt images
                
                # Convert to RGB if needed
                # Handle palette images with transparency properly to avoid PIL warning
                if pil_img.mode == 'P' and 'transparency' in pil_img.info:
                    # Convert palette with transparency to RGBA first, then to RGB
                    pil_img = pil_img.convert('RGBA')
                if pil_img.mode == 'RGBA':
                    # Create white background and composite
                    background = Image.new('RGB', pil_img.size, (255, 255, 255))
                    background.paste(pil_img, mask=pil_img.split()[3])  # Use alpha as mask
                    pil_img = background
                elif pil_img.mode != 'RGB':
                    pil_img = pil_img.convert('RGB')
                
                # Convert to numpy array (RGB format)
                image = np.array(pil_img)
                
                # Sanity check: image should have 3 channels and reasonable size
                if image is None or len(image.shape) != 3 or image.shape[2] != 3:
                    return False, None, "Invalid image dimensions"
                
                if image.shape[0] < 10 or image.shape[1] < 10:
                    return False, None, "Image too small"
                
                return True, image, None
                
        except Exception as e:
            return False, None, str(e)
    
    def __getitem__(self, idx):
        # Use while loop to avoid recursion depth issues with consecutive corrupt images
        max_attempts = len(self.samples)
        attempts = 0
        original_idx = idx
        
        while attempts < max_attempts:
            img_path, label = self.samples[idx]
            
            # Validate and load image using PIL (more robust for corrupt detection)
            is_valid, image, error_msg = self._validate_image(img_path)
            
            if is_valid:
                break  # Found a valid image
            
            self._log_corrupt_file(img_path, error_msg or "Unknown error")
            idx = (idx + 1) % len(self)
            attempts += 1
        
        if attempts >= max_attempts:
            raise RuntimeError(f"All images appear to be corrupt (checked {max_attempts} images starting from idx {original_idx})")
        
        # image is already RGB from PIL, no need for cv2.cvtColor
        
        # Note: Median filtering removed to match standard augmentation pipeline
        # If needed, uncomment below:
        # if self.config and self.config.get('preprocessing', {}).get('median_filter_kernel'):
        #     kernel_size = self.config['preprocessing']['median_filter_kernel']
        #     if isinstance(kernel_size, int) and kernel_size >= 3 and kernel_size % 2 == 1:
        #         image = cv2.medianBlur(image, kernel_size)
        
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
