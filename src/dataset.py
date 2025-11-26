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
import warnings

# Suppress PIL decompression bomb warning for large images
warnings.filterwarnings('ignore', category=Image.DecompressionBombWarning)

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
                if pil_img.mode != 'RGB':
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
        img_path, label = self.samples[idx]
        
        # Validate and load image using PIL (more robust for corrupt detection)
        is_valid, image, error_msg = self._validate_image(img_path)
        
        if not is_valid:
            self._log_corrupt_file(img_path, error_msg or "Unknown error")
            # Try next image to avoid crashing the training
            new_idx = (idx + 1) % len(self)
            if new_idx == idx:  # Prevent infinite loop if only one sample
                raise RuntimeError("All images appear to be corrupt")
            return self.__getitem__(new_idx)
        
        # image is already RGB from PIL, no need for cv2.cvtColor
        
        # Apply median filtering for denoising (as per paper)
        if self.config and self.config.get('preprocessing', {}).get('median_filter_kernel'):
            kernel_size = self.config['preprocessing']['median_filter_kernel']
            # Validate kernel size: must be odd integer >= 3
            if isinstance(kernel_size, int) and kernel_size >= 3 and kernel_size % 2 == 1:
                image = cv2.medianBlur(image, kernel_size)
            else:
                raise ValueError(f"median_filter_kernel must be an odd integer >= 3, got {kernel_size}")
        
        # Apply transformations
        if self.transform:
            image = self.transform(image=image)['image']
        
        return image, label


def get_train_transforms(config):
    """
    Training transforms with data augmentation
    Based on paper: rotation, mirroring, random cropping, grayscale
    """
    img_size = config['data']['image_size']
    mean = config['preprocessing']['mean']
    std = config['preprocessing']['std']
    use_grayscale = config['preprocessing'].get('use_grayscale', False)
    
    transform_list = [
        A.Resize(img_size + 32, img_size + 32),
        A.RandomCrop(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=config['augmentation']['rotation_degrees'], p=0.5),
    ]
    
    # Grayscale conversion (as per paper: weighted average method)
    # Converts to grayscale but keeps 3 channels for pretrained model compatibility
    # Formula: 0.299*R + 0.587*G + 0.114*B (same as paper's weighted average)
    if use_grayscale:
        # Albumentations' ToGray will convert RGB -> grayscale and keep the
        # number of channels consistent for downstream transforms when input
        # is RGB, so do not pass torchvision-specific args like
        # `num_output_channels` which albumentations does not accept.
        transform_list.append(A.ToGray(p=1.0))
    else:
        # Only apply color jitter if NOT using grayscale
        transform_list.append(A.ColorJitter(
            brightness=config['augmentation']['color_jitter']['brightness'],
            contrast=config['augmentation']['color_jitter']['contrast'],
            saturation=config['augmentation']['color_jitter']['saturation'],
            p=0.3
        ))
    
    transform_list.extend([
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])
    
    return A.Compose(transform_list)


def get_val_transforms(config):
    """Validation/Test transforms without augmentation"""
    img_size = config['data']['image_size']
    mean = config['preprocessing']['mean']
    std = config['preprocessing']['std']
    use_grayscale = config['preprocessing'].get('use_grayscale', False)
    
    transform_list = [
        A.Resize(img_size, img_size),
    ]
    
    # Grayscale conversion (must match training transforms)
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
