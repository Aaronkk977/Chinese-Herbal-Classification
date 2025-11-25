"""
Main entry point for Chinese Herbal Classification
Supports training, evaluation, and inference
"""

import os
import warnings
import argparse
import yaml
import torch

# Suppress specific warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='timm')

from train import Trainer
from evaluate import Evaluator
from model import create_model
from utils import load_checkpoint


def train(config):
    """Run training"""
    trainer = Trainer(config)
    trainer.train()


def evaluate(config, checkpoint_path, split='test'):
    """Run evaluation"""
    evaluator = Evaluator(config, checkpoint_path=checkpoint_path)
    
    if split == 'train':
        metrics = evaluator.evaluate(evaluator.train_loader, 'train')
    elif split == 'val':
        metrics = evaluator.evaluate(evaluator.val_loader, 'val')
    else:
        metrics = evaluator.evaluate(evaluator.test_loader, 'test')
    
    return metrics


def inference(config, checkpoint_path, image_path):
    """Run inference on a single image"""
    import cv2
    import numpy as np
    from dataset import get_val_transforms
    
    device = torch.device(config['hardware']['device'] if torch.cuda.is_available() else 'cpu')
    
    # Create model and load checkpoint
    model = create_model(config)
    model = model.to(device)
    load_checkpoint(checkpoint_path, model)
    model.eval()
    
    # Load and preprocess image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Apply transformations
    transform = get_val_transforms(config)
    image_tensor = transform(image=image)['image']
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        
        # Get top-5 predictions
        top5_prob, top5_idx = torch.topk(probabilities, 5)
        top5_prob = top5_prob[0].cpu().numpy()
        top5_idx = top5_idx[0].cpu().numpy()
    
    print(f"\nTop-5 Predictions for {image_path}:")
    print("-" * 50)
    for i, (idx, prob) in enumerate(zip(top5_idx, top5_prob)):
        print(f"{i+1}. Class {idx}: {prob*100:.2f}%")
    
    return top5_idx[0], top5_prob[0]


def main():
    parser = argparse.ArgumentParser(description='Chinese Herbal Classification')
    parser.add_argument('--mode', type=str, required=True, 
                        choices=['train', 'evaluate', 'inference'],
                        help='Mode to run: train, evaluate, or inference')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (for evaluation/inference)')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Dataset split for evaluation')
    parser.add_argument('--image', type=str, default=None,
                        help='Path to image for inference')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"\nRunning in {args.mode} mode...")
    print(f"Configuration: {args.config}")
    
    if args.mode == 'train':
        train(config)
    
    elif args.mode == 'evaluate':
        if args.checkpoint is None:
            raise ValueError("Checkpoint path is required for evaluation")
        evaluate(config, args.checkpoint, args.split)
    
    elif args.mode == 'inference':
        if args.checkpoint is None:
            raise ValueError("Checkpoint path is required for inference")
        if args.image is None:
            raise ValueError("Image path is required for inference")
        inference(config, args.checkpoint, args.image)


if __name__ == '__main__':
    main()
