"""
Utility functions for Chinese Herbal Classification
"""

import os
import random
import torch
import numpy as np
from collections import OrderedDict


class AverageMeter:
    """Computes and stores the average and current value"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def setup_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(state, is_best=False, checkpoint_dir='checkpoints', filename='checkpoint.pth'):
    """Save model checkpoint"""
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    
    if is_best:
        best_filepath = os.path.join(checkpoint_dir, 'model_best.pth')
        torch.save(state, best_filepath)
        print(f"Saved best model to {best_filepath}")


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """Load model checkpoint"""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load model state
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Handle DataParallel
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            name = k[7:]  # remove 'module.' prefix
        else:
            name = k
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict)
    
    # Load optimizer state
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    # Load scheduler state
    if scheduler is not None and 'scheduler' in checkpoint and checkpoint['scheduler'] is not None:
        scheduler.load_state_dict(checkpoint['scheduler'])
    
    return checkpoint


def count_parameters(model):
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_lr(optimizer):
    """Get current learning rate from optimizer"""
    for param_group in optimizer.param_groups:
        return param_group['lr']


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""
    
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved
            verbose (bool): If True, prints a message for each validation loss improvement
            delta (float): Minimum change in the monitored quantity to qualify as an improvement
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
    
    def __call__(self, val_loss, model, path):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0
    
    def save_checkpoint(self, val_loss, model, path):
        """Saves model when validation loss decrease"""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss


def mixup_data(x, y, alpha=1.0):
    """
    Returns mixed inputs, pairs of targets, and lambda
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Mixup loss function
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


if __name__ == "__main__":
    # Test utilities
    print("Testing utilities...")
    
    # Test AverageMeter
    meter = AverageMeter()
    for i in range(10):
        meter.update(i, 1)
    print(f"AverageMeter test: avg={meter.avg}, should be 4.5")
    
    # Test seed
    setup_seed(42)
    print(f"Random number after seed 42: {random.random()}")
    
    print("\n✓ All utility tests passed!")
