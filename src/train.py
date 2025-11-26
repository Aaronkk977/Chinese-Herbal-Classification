"""
Training script for Chinese Herbal Classification
Based on paper specifications:
- 100 epochs
- Learning rate: 0.0002
- AdamW optimizer
- Cosine learning rate scheduler
- Mixed precision training
"""

import os
import time
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

from model import create_model
from dataset import create_dataloaders
from utils import AverageMeter, save_checkpoint, load_checkpoint, setup_seed


class Trainer:
    """Trainer class for model training"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['hardware']['device'] if torch.cuda.is_available() else 'cpu')
        
        # Setup random seed for reproducibility
        setup_seed(config['hardware']['seed'])
        
        # Create dataloaders
        print("Creating dataloaders...")
        self.train_loader, self.val_loader, self.test_loader, self.class_to_idx = create_dataloaders(config)
        
        # Update num_classes in config
        config['data']['num_classes'] = len(self.class_to_idx)
        print(f"[DEBUG] Number of classes detected: {config['data']['num_classes']}")
        print(f"[DEBUG] Classes: {list(self.class_to_idx.keys())}")
        
        # Create model
        print(f"Creating model on {self.device}...")
        self.model = create_model(config)
        self.model = self.model.to(self.device)
        
        # Verify model output shape
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, config['data']['image_size'], config['data']['image_size']).to(self.device)
            dummy_output = self.model(dummy_input)
            print(f"[DEBUG] Model output shape: {dummy_output.shape} (should be [1, {config['data']['num_classes']}])")
        
        # Loss function with label smoothing
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=config['training'].get('label_smoothing', 0.1)
        )
        
        # Optimizer
        self.optimizer = self._create_optimizer()
        
        # Learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Mixed precision training
        self.use_amp = config['training'].get('use_amp', True)
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None
        
        # Create directories
        os.makedirs(config['paths']['checkpoint_dir'], exist_ok=True)
        os.makedirs(config['paths']['log_dir'], exist_ok=True)
        
        # Tensorboard
        self.writer = SummaryWriter(log_dir=config['paths']['log_dir'])
        
        # Training state
        self.start_epoch = 0
        self.best_acc = 0.0
        self.patience_counter = 0
    
    def _create_optimizer(self):
        """Create optimizer based on config"""
        if self.config['training']['optimizer'].lower() == 'adamw':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config['training']['learning_rate'],
                weight_decay=self.config['training']['weight_decay']
            )
        elif self.config['training']['optimizer'].lower() == 'sgd':
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config['training']['learning_rate'],
                momentum=0.9,
                weight_decay=self.config['training']['weight_decay']
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config['training']['optimizer']}")
        
        return optimizer
    
    def _create_scheduler(self):
        """Create learning rate scheduler"""
        if self.config['training']['scheduler'].lower() == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['epochs'],
                eta_min=1e-6
            )
        elif self.config['training']['scheduler'].lower() == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        else:
            scheduler = None
        
        return scheduler
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        
        losses = AverageMeter()
        accs = AverageMeter()
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.config["training"]["epochs"]}')
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass with mixed precision
            if self.use_amp:
                with autocast(device_type='cuda'):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                # Backward pass
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            # Calculate accuracy
            _, predicted = outputs.max(1)
            acc = predicted.eq(labels).sum().item() / labels.size(0)
            
            # Update meters
            losses.update(loss.item(), images.size(0))
            accs.update(acc, images.size(0))
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{losses.avg:.4f}',
                'acc': f'{accs.avg:.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        return losses.avg, accs.avg
    
    @torch.no_grad()
    def validate(self, epoch):
        """Validate the model"""
        self.model.eval()
        
        losses = AverageMeter()
        accs = AverageMeter()
        
        pbar = tqdm(self.val_loader, desc='Validating')
        
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            if self.use_amp:
                with autocast(device_type='cuda'):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
            
            # Calculate accuracy
            _, predicted = outputs.max(1)
            acc = predicted.eq(labels).sum().item() / labels.size(0)
            
            # Update meters
            losses.update(loss.item(), images.size(0))
            accs.update(acc, images.size(0))
            
            pbar.set_postfix({
                'loss': f'{losses.avg:.4f}',
                'acc': f'{accs.avg:.4f}'
            })
        
        return losses.avg, accs.avg
    
    def train(self):
        """Main training loop"""
        print(f"\nStarting training for {self.config['training']['epochs']} epochs...")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}\n")
        
        # Debug: Check labels from first batch
        images, labels = next(iter(self.train_loader))
        print(f"[DEBUG] First batch labels: {labels}")
        print(f"[DEBUG] Label range: min={labels.min().item()}, max={labels.max().item()}")
        print(f"[DEBUG] Expected range: 0 to {self.config['data']['num_classes'] - 1}\n")
        
        for epoch in range(self.start_epoch, self.config['training']['epochs']):
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc = self.validate(epoch)
            
            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Log to tensorboard
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/val', val_acc, epoch)
            self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Print epoch summary
            print(f"\nEpoch {epoch}: "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Save checkpoint
            is_best = val_acc > self.best_acc
            if is_best:
                self.best_acc = val_acc
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            if (epoch + 1) % self.config['training']['save_freq'] == 0 or is_best:
                save_checkpoint(
                    {
                        'epoch': epoch,
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'scheduler': self.scheduler.state_dict() if self.scheduler else None,
                        'best_acc': self.best_acc,
                        'config': self.config
                    },
                    is_best=is_best,
                    checkpoint_dir=self.config['paths']['checkpoint_dir']
                )
            
            # Early stopping
            if self.patience_counter >= self.config['training'].get('early_stopping_patience', 15):
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
        
        print(f"\nTraining completed! Best validation accuracy: {self.best_acc:.4f}")
        self.writer.close()


def main():
    # Load configuration
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create trainer and start training
    trainer = Trainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
