"""
Evaluation script for Chinese Herbal Classification
Computes metrics as specified in the paper:
- Top-1 Accuracy
- Top-5 Accuracy
- Precision, Recall, F1-score (per class and macro average)
- Confusion Matrix
"""

import os
import yaml
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
import pandas as pd

from model import create_model
from dataset import create_dataloaders
from utils import load_checkpoint


class Evaluator:
    """Evaluator class for model evaluation"""
    
    def __init__(self, config, checkpoint_path=None):
        self.config = config
        self.device = torch.device(config['hardware']['device'] if torch.cuda.is_available() else 'cpu')
        
        # Create dataloaders
        print("Creating dataloaders...")
        self.train_loader, self.val_loader, self.test_loader, self.class_to_idx = create_dataloaders(config)
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
        # Update num_classes
        config['data']['num_classes'] = len(self.class_to_idx)
        
        # Create model
        print(f"Creating model on {self.device}...")
        self.model = create_model(config)
        self.model = self.model.to(self.device)
        
        # Load checkpoint if provided
        if checkpoint_path:
            print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = load_checkpoint(checkpoint_path, self.model)
            print(f"Loaded model from epoch {checkpoint['epoch']} with accuracy {checkpoint['best_acc']:.4f}")
        
        self.model.eval()
        
        # Create results directory
        os.makedirs(config['paths']['results_dir'], exist_ok=True)
    
    @torch.no_grad()
    def evaluate(self, data_loader, split_name='test'):
        """
        Evaluate model on given data loader
        
        Returns:
            Dictionary containing all evaluation metrics
        """
        print(f"\nEvaluating on {split_name} set...")
        
        all_labels = []
        all_predictions = []
        all_probabilities = []
        
        for images, labels in tqdm(data_loader, desc=f'Evaluating {split_name}'):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            probabilities = torch.softmax(outputs, dim=1)
            
            # Store results
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(outputs.argmax(dim=1).cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
        
        all_labels = np.array(all_labels)
        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)
        
        # Calculate metrics
        metrics = self._calculate_metrics(
            all_labels, 
            all_predictions, 
            all_probabilities,
            split_name
        )
        
        return metrics
    
    def _calculate_metrics(self, labels, predictions, probabilities, split_name):
        """Calculate all evaluation metrics"""
        
        metrics = {}
        
        # Top-1 Accuracy
        top1_acc = accuracy_score(labels, predictions)
        metrics['top1_accuracy'] = top1_acc
        
        # Top-5 Accuracy
        top5_acc = self._top_k_accuracy(labels, probabilities, k=5)
        metrics['top5_accuracy'] = top5_acc
        
        # Precision, Recall, F1-score
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, predictions, average=None, zero_division=0
        )
        
        # Macro averages
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            labels, predictions, average='macro', zero_division=0
        )
        
        metrics['macro_precision'] = macro_precision
        metrics['macro_recall'] = macro_recall
        metrics['macro_f1'] = macro_f1
        
        # Weighted averages
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted', zero_division=0
        )
        
        metrics['weighted_precision'] = weighted_precision
        metrics['weighted_recall'] = weighted_recall
        metrics['weighted_f1'] = weighted_f1
        
        # Per-class metrics
        metrics['per_class'] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support
        }
        
        # Confusion Matrix
        cm = confusion_matrix(labels, predictions)
        metrics['confusion_matrix'] = cm
        
        # Print results
        self._print_metrics(metrics, split_name)
        
        # Save results
        self._save_results(metrics, labels, predictions, split_name)
        
        return metrics
    
    def _top_k_accuracy(self, labels, probabilities, k=5):
        """Calculate Top-K accuracy"""
        top_k_preds = np.argsort(probabilities, axis=1)[:, -k:]
        correct = np.sum([label in top_k_preds[i] for i, label in enumerate(labels)])
        return correct / len(labels)
    
    def _print_metrics(self, metrics, split_name):
        """Print evaluation metrics"""
        print(f"\n{'='*60}")
        print(f"{split_name.upper()} SET EVALUATION RESULTS")
        print(f"{'='*60}")
        print(f"Top-1 Accuracy: {metrics['top1_accuracy']:.4f} ({metrics['top1_accuracy']*100:.2f}%)")
        print(f"Top-5 Accuracy: {metrics['top5_accuracy']:.4f} ({metrics['top5_accuracy']*100:.2f}%)")
        print(f"\nMacro Average Metrics:")
        print(f"  Precision: {metrics['macro_precision']:.4f}")
        print(f"  Recall:    {metrics['macro_recall']:.4f}")
        print(f"  F1-score:  {metrics['macro_f1']:.4f}")
        print(f"\nWeighted Average Metrics:")
        print(f"  Precision: {metrics['weighted_precision']:.4f}")
        print(f"  Recall:    {metrics['weighted_recall']:.4f}")
        print(f"  F1-score:  {metrics['weighted_f1']:.4f}")
        print(f"{'='*60}\n")
    
    def _save_results(self, metrics, labels, predictions, split_name):
        """Save evaluation results to files"""
        results_dir = self.config['paths']['results_dir']
        
        # Save metrics summary
        summary_path = os.path.join(results_dir, f'{split_name}_metrics.txt')
        with open(summary_path, 'w') as f:
            f.write(f"{split_name.upper()} SET EVALUATION RESULTS\n")
            f.write("="*60 + "\n")
            f.write(f"Top-1 Accuracy: {metrics['top1_accuracy']:.4f} ({metrics['top1_accuracy']*100:.2f}%)\n")
            f.write(f"Top-5 Accuracy: {metrics['top5_accuracy']:.4f} ({metrics['top5_accuracy']*100:.2f}%)\n")
            f.write(f"\nMacro Average:\n")
            f.write(f"  Precision: {metrics['macro_precision']:.4f}\n")
            f.write(f"  Recall:    {metrics['macro_recall']:.4f}\n")
            f.write(f"  F1-score:  {metrics['macro_f1']:.4f}\n")
        
        print(f"Saved metrics summary to {summary_path}")
        
        # Save per-class metrics
        per_class_df = pd.DataFrame({
            'Class': [self.idx_to_class[i] for i in range(len(self.class_to_idx))],
            'Precision': metrics['per_class']['precision'],
            'Recall': metrics['per_class']['recall'],
            'F1-score': metrics['per_class']['f1'],
            'Support': metrics['per_class']['support']
        })
        
        per_class_path = os.path.join(results_dir, f'{split_name}_per_class_metrics.csv')
        per_class_df.to_csv(per_class_path, index=False)
        print(f"Saved per-class metrics to {per_class_path}")

        # Determine top-5 hardest and top-5 easiest classes by F1-score
        try:
            sorted_by_f1 = per_class_df.sort_values('F1-score')
            top5_hardest = sorted_by_f1.head(5).reset_index(drop=True)
            top5_easiest = sorted_by_f1.tail(5).iloc[::-1].reset_index(drop=True)

            hardest_path = os.path.join(results_dir, f'{split_name}_top5_hardest.csv')
            easiest_path = os.path.join(results_dir, f'{split_name}_top5_easiest.csv')

            top5_hardest.to_csv(hardest_path, index=False)
            top5_easiest.to_csv(easiest_path, index=False)

            print(f"Saved top-5 hardest classes to {hardest_path}")
            print(f"Saved top-5 easiest classes to {easiest_path}")

            # Append top5 summary to metrics text file
            with open(summary_path, 'a') as f:
                f.write('\nTop-5 Hardest Classes (by F1-score):\n')
                f.write(top5_hardest.to_string(index=False))
                f.write('\n\nTop-5 Easiest Classes (by F1-score):\n')
                f.write(top5_easiest.to_string(index=False))
                f.write('\n')

        except Exception as e:
            print(f"Warning: failed to compute/save top-5 classes: {e}")
        
        # Save and plot confusion matrix
        self._plot_confusion_matrix(
            metrics['confusion_matrix'],
            split_name
        )
        
        # Save detailed classification report
        report_path = os.path.join(results_dir, f'{split_name}_classification_report.txt')
        with open(report_path, 'w') as f:
            report = classification_report(
                labels, 
                predictions,
                target_names=[self.idx_to_class[i] for i in range(len(self.class_to_idx))],
                digits=4
            )
            f.write(report)
        print(f"Saved classification report to {report_path}")
    
    def _plot_confusion_matrix(self, cm, split_name):
        """Plot and save confusion matrix"""
        results_dir = self.config['paths']['results_dir']
        
        # For large number of classes, create a simplified version
        num_classes = len(self.class_to_idx)
        
        plt.figure(figsize=(max(12, num_classes // 5), max(10, num_classes // 5)))
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot
        sns.heatmap(
            cm_normalized,
            annot=False,  # Don't annotate if too many classes
            fmt='.2f',
            cmap='Blues',
            square=True,
            cbar_kws={'label': 'Proportion'}
        )
        
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title(f'Confusion Matrix - {split_name.upper()} Set (Normalized)')
        plt.tight_layout()
        
        # Save figure
        cm_path = os.path.join(results_dir, f'{split_name}_confusion_matrix.png')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved confusion matrix to {cm_path}")
        
        # Also save raw confusion matrix as numpy array
        np.save(os.path.join(results_dir, f'{split_name}_confusion_matrix.npy'), cm)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Chinese Herbal Classification Model')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'], 
                        help='Dataset split to evaluate')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create evaluator
    evaluator = Evaluator(config, checkpoint_path=args.checkpoint)
    
    # Evaluate
    if args.split == 'train':
        metrics = evaluator.evaluate(evaluator.train_loader, 'train')
    elif args.split == 'val':
        metrics = evaluator.evaluate(evaluator.val_loader, 'val')
    else:
        metrics = evaluator.evaluate(evaluator.test_loader, 'test')


if __name__ == '__main__':
    main()
