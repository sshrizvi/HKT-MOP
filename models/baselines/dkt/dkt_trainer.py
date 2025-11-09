"""
Trainer for Deep Knowledge Tracing Model
Supports binary and multiclass classification with training visualization

Author: Syed Shujaat Haider
"""

import os
import json
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

from logging_config import setup_logging
from models.baselines.dkt.dkt_model import DKTModel
from preprocessing.dkt_format_adapter import DKTDataset, collate_fn

class DKTTrainer:
    """
    Trainer for Deep Knowledge Tracing model.
    
    Features:
    - Binary and multiclass classification support
    - Standard and domain shift splits
    - Training/validation monitoring
    - Real-time loss visualization
    - Model checkpointing
    - Comprehensive metrics logging
    """
    
    def __init__(
        self,
        model: DKTModel,
        train_dataset: DKTDataset,
        val_dataset: DKTDataset,
        test_dataset: DKTDataset,
        config: Dict,
        model_name: str,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize DKT Trainer.
        
        Args:
            model: DKTModel instance
            train_dataset: Training DKTDataset
            val_dataset: Validation DKTDataset
            test_dataset: Test DKTDataset
            config: Training configuration dictionary
            model_name: Name identifier for this model (e.g., "binary_standard")
            device: Device to train on ('cuda' or 'cpu')
        """
        self.logger = logging.getLogger(f"hkt-mop.models.{model_name}")
        setup_logging()
        
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.model_name = model_name
        self.device = device
        
        # Training parameters
        self.batch_size = config.get('batch_size', 32)
        self.num_epochs = config.get('num_epochs', 50)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.weight_decay = config.get('weight_decay', 1e-5)
        self.patience = config.get('early_stopping_patience', 10)
        
        # Data loaders
        from preprocessing.dkt_format_adapter import DKTDataset
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )
        
        # Loss function
        # Use CrossEntropyLoss for both binary and multiclass
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rate': []
        }
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.epochs_without_improvement = 0
        
        # Results directory
        self.results_dir = config.get('results_dir', 'experiments/results/dkt')
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.logger.info(f"Initialized DKTTrainer for {model_name}")
        self.logger.info(f"Device: {device}")
        self.logger.info(f"Training samples: {len(train_dataset)}")
        self.logger.info(f"Validation samples: {len(val_dataset)}")
        self.logger.info(f"Test samples: {len(test_dataset)}")
    
    def train_epoch(self) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_predictions = 0
        
        for batch_idx, (inputs, targets, labels, seq_lens) in enumerate(self.train_loader):
            # Move to device
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            labels = labels.to(self.device)
            seq_lens = seq_lens.to(self.device)
            
            # Forward pass
            outputs = self.model(inputs, seq_lens)  # (batch, seq_len, num_exercises, num_outcomes)
            
            # Get predictions for target exercises
            # targets: (batch, seq_len, num_exercises) - one-hot encoded
            # outputs: (batch, seq_len, num_exercises, num_outcomes)
            
            batch_size, seq_len, num_exercises, num_outcomes = outputs.shape
            
            # Expand targets for selecting correct exercise predictions
            target_mask = targets.unsqueeze(-1)  # (batch, seq_len, num_exercises, 1)
            
            # Select predictions for target exercises
            target_outputs = (outputs * target_mask).sum(dim=2)  # (batch, seq_len, num_outcomes)
            
            # Reshape for loss computation
            target_outputs_flat = target_outputs.view(-1, num_outcomes)  # (batch*seq_len, num_outcomes)
            labels_flat = labels.view(-1)  # (batch*seq_len,)
            
            # Create mask for valid positions (non-padded)
            mask = torch.arange(seq_len, device=self.device).expand(batch_size, seq_len) < seq_lens.unsqueeze(1)
            mask_flat = mask.view(-1)
            
            # Apply mask
            target_outputs_flat = target_outputs_flat[mask_flat]
            labels_flat = labels_flat[mask_flat]
            
            # Compute loss
            loss = self.criterion(target_outputs_flat, labels_flat)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            
            # Compute accuracy
            _, predicted = torch.max(target_outputs_flat, dim=1)
            total_correct += (predicted == labels_flat).sum().item()
            total_predictions += labels_flat.size(0)
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = total_correct / total_predictions if total_predictions > 0 else 0.0
        
        return avg_loss, accuracy
    
    def validate_epoch(self) -> Tuple[float, float]:
        """
        Validate for one epoch.
        
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_predictions = 0
        
        with torch.no_grad():
            for inputs, targets, labels, seq_lens in self.val_loader:
                # Move to device
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                labels = labels.to(self.device)
                seq_lens = seq_lens.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs, seq_lens)
                
                batch_size, seq_len, num_exercises, num_outcomes = outputs.shape
                
                # Select predictions for target exercises
                target_mask = targets.unsqueeze(-1)
                target_outputs = (outputs * target_mask).sum(dim=2)
                
                # Reshape
                target_outputs_flat = target_outputs.view(-1, num_outcomes)
                labels_flat = labels.view(-1)
                
                # Apply mask
                mask = torch.arange(seq_len, device=self.device).expand(batch_size, seq_len) < seq_lens.unsqueeze(1)
                mask_flat = mask.view(-1)
                
                target_outputs_flat = target_outputs_flat[mask_flat]
                labels_flat = labels_flat[mask_flat]
                
                # Compute loss
                loss = self.criterion(target_outputs_flat, labels_flat)
                total_loss += loss.item()
                
                # Compute accuracy
                _, predicted = torch.max(target_outputs_flat, dim=1)
                total_correct += (predicted == labels_flat).sum().item()
                total_predictions += labels_flat.size(0)
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = total_correct / total_predictions if total_predictions > 0 else 0.0
        
        return avg_loss, accuracy
    
    def train(self, plot_live: bool = True) -> Dict:
        """
        Full training loop with visualization.
        
        Args:
            plot_live: Whether to show live training plot
            
        Returns:
            Dictionary with training history and best metrics
        """
        self.logger.info("=" * 80)
        self.logger.info(f"Starting training for {self.model_name}")
        self.logger.info("=" * 80)
        
        # Initialize live plot if requested
        if plot_live:
            plt.ion()  # Turn on interactive mode
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            fig.suptitle(f'Training Progress: {self.model_name}')
        
        for epoch in range(self.num_epochs):
            # Train epoch
            train_loss, train_acc = self.train_epoch()
            
            # Validate epoch
            val_loss, val_acc = self.validate_epoch()
            
            # Update learning rate
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rate'].append(current_lr)
            
            # Log progress
            self.logger.info(
                f"Epoch [{epoch+1}/{self.num_epochs}] "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                f"LR: {current_lr:.6f}"
            )
            
            # Update live plot
            if plot_live:
                ax1.clear()
                ax2.clear()
                
                # Plot loss
                epochs_range = range(1, epoch + 2)
                ax1.plot(epochs_range, self.history['train_loss'], 'b-', label='Train Loss')
                ax1.plot(epochs_range, self.history['val_loss'], 'r-', label='Val Loss')
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Loss')
                ax1.set_title('Training and Validation Loss')
                ax1.legend()
                ax1.grid(True)
                
                # Plot accuracy
                ax2.plot(epochs_range, self.history['train_acc'], 'b-', label='Train Acc')
                ax2.plot(epochs_range, self.history['val_acc'], 'r-', label='Val Acc')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Accuracy')
                ax2.set_title('Training and Validation Accuracy')
                ax2.legend()
                ax2.grid(True)
                
                plt.pause(0.01)
            
            # Check for best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.epochs_without_improvement = 0
                
                # Save best model
                self.save_checkpoint(is_best=True)
                self.logger.info(f"✓ New best model saved (val_loss: {val_loss:.4f})")
            else:
                self.epochs_without_improvement += 1
            
            # Early stopping
            if self.epochs_without_improvement >= self.patience:
                self.logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        if plot_live:
            plt.ioff()
            
        # Save final training plot
        self.save_training_plot()
        
        # Evaluate on test set
        test_metrics = self.evaluate()
        
        self.logger.info("=" * 80)
        self.logger.info(f"Training completed for {self.model_name}")
        self.logger.info(f"Best epoch: {self.best_epoch + 1}")
        self.logger.info(f"Best val loss: {self.best_val_loss:.4f}")
        self.logger.info("=" * 80)
        
        return {
            'history': self.history,
            'best_epoch': self.best_epoch,
            'best_val_loss': self.best_val_loss,
            'test_metrics': test_metrics
        }
    
    def evaluate(self) -> Dict:
        """
        Evaluate model on test set.
        
        Returns:
            Dictionary with test metrics
        """
        self.logger.info("Evaluating on test set...")
        
        # Load best model
        self.load_checkpoint(is_best=True)
        
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probs = []
        total_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets, labels, seq_lens in self.test_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                labels = labels.to(self.device)
                seq_lens = seq_lens.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs, seq_lens)
                
                batch_size, seq_len, num_exercises, num_outcomes = outputs.shape
                
                # Select predictions for target exercises
                target_mask = targets.unsqueeze(-1)
                target_outputs = (outputs * target_mask).sum(dim=2)
                
                # Reshape
                target_outputs_flat = target_outputs.view(-1, num_outcomes)
                labels_flat = labels.view(-1)
                
                # Apply mask
                mask = torch.arange(seq_len, device=self.device).expand(batch_size, seq_len) < seq_lens.unsqueeze(1)
                mask_flat = mask.view(-1)
                
                target_outputs_flat = target_outputs_flat[mask_flat]
                labels_flat = labels_flat[mask_flat]
                
                # Loss
                loss = self.criterion(target_outputs_flat, labels_flat)
                total_loss += loss.item()
                
                # Predictions
                probs = torch.softmax(target_outputs_flat, dim=1)
                _, predicted = torch.max(probs, dim=1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels_flat.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Compute metrics
        avg_loss = total_loss / len(self.test_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        
        metrics = {
            'test_loss': avg_loss,
            'test_accuracy': accuracy
        }
        
        # Add AUC for binary classification
        if self.model.num_outcomes == 2:
            probs_positive = np.array(all_probs)[:, 1]
            auc = roc_auc_score(all_labels, probs_positive)
            metrics['test_auc'] = auc
            self.logger.info(f"Test AUC: {auc:.4f}")
        
        # Add F1 score
        f1 = f1_score(all_labels, all_predictions, average='macro')
        metrics['test_f1'] = f1
        
        self.logger.info(f"Test Loss: {avg_loss:.4f}")
        self.logger.info(f"Test Accuracy: {accuracy:.4f}")
        self.logger.info(f"Test F1: {f1:.4f}")
        
        # Save metrics
        metrics_path = os.path.join(self.results_dir, f'{self.model_name}_test_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return metrics
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
            'config': self.config
        }
        
        if is_best:
            path = os.path.join(self.results_dir, f'{self.model_name}_best.pth')
        else:
            path = os.path.join(self.results_dir, f'{self.model_name}_last.pth')
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, is_best: bool = True):
        """Load model checkpoint."""
        if is_best:
            path = os.path.join(self.results_dir, f'{self.model_name}_best.pth')
        else:
            path = os.path.join(self.results_dir, f'{self.model_name}_last.pth')
        
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.history = checkpoint['history']
            self.best_val_loss = checkpoint['best_val_loss']
            self.best_epoch = checkpoint['best_epoch']
            self.logger.info(f"Loaded checkpoint from {path}")
        else:
            self.logger.warning(f"No checkpoint found at {path}")
    
    def save_training_plot(self):
        """Save training history plot to file."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Training History: {self.model_name}', fontsize=16)
        
        epochs_range = range(1, len(self.history['train_loss']) + 1)
        
        # Loss plot
        ax1.plot(epochs_range, self.history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        ax1.plot(epochs_range, self.history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        ax1.axvline(x=self.best_epoch + 1, color='g', linestyle='--', label=f'Best Epoch ({self.best_epoch + 1})')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training and Validation Loss', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax2.plot(epochs_range, self.history['train_acc'], 'b-', label='Train Acc', linewidth=2)
        ax2.plot(epochs_range, self.history['val_acc'], 'r-', label='Val Acc', linewidth=2)
        ax2.axvline(x=self.best_epoch + 1, color='g', linestyle='--', label=f'Best Epoch ({self.best_epoch + 1})')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_title('Training and Validation Accuracy', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Learning rate plot
        ax3.plot(epochs_range, self.history['learning_rate'], 'g-', linewidth=2)
        ax3.set_xlabel('Epoch', fontsize=12)
        ax3.set_ylabel('Learning Rate', fontsize=12)
        ax3.set_title('Learning Rate Schedule', fontsize=14)
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
        
        # Loss difference plot
        loss_diff = [abs(t - v) for t, v in zip(self.history['train_loss'], self.history['val_loss'])]
        ax4.plot(epochs_range, loss_diff, 'purple', linewidth=2)
        ax4.set_xlabel('Epoch', fontsize=12)
        ax4.set_ylabel('|Train Loss - Val Loss|', fontsize=12)
        ax4.set_title('Training-Validation Loss Gap', fontsize=14)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.results_dir, f'{self.model_name}_training_history.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Training plot saved to {plot_path}")
