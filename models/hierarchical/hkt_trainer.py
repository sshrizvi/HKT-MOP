"""
Trainer for Hierarchical Knowledge Tracing Model
Handles two-level hierarchical predictions with specialized loss and metrics
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
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    confusion_matrix, classification_report
)

from logging_config import setup_logging
from models.hierarchical.hkt import (
    HierarchicalKTModel, HierarchicalLoss
)
from preprocessing.hierarchical_model_adapter import (
    HierarchicalDataset, hierarchical_collate_fn
)


class HierarchicalTrainer:
    """
    Trainer for Hierarchical Knowledge Tracing model.
    
    Features:
    - Dual-level hierarchical prediction (compilation → execution)
    - Hierarchical loss computation with adaptive weighting
    - Separate metrics tracking for both levels
    - Calibration-ready probability outputs
    - Support for standard and domain shift evaluation
    - Comprehensive visualization and logging
    """
    
    def __init__(
        self,
        model: HierarchicalKTModel,
        train_dataset: HierarchicalDataset,
        val_dataset: HierarchicalDataset,
        test_dataset: HierarchicalDataset,
        config: Dict,
        model_name: str = "hierarchical_kt",
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize Hierarchical Trainer.
        
        Args:
            model: HierarchicalKTModel instance
            train_dataset: Training HierarchicalDataset
            val_dataset: Validation HierarchicalDataset
            test_dataset: Test HierarchicalDataset
            config: Training configuration dictionary
            model_name: Name identifier for this model
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
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=hierarchical_collate_fn
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=hierarchical_collate_fn
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=hierarchical_collate_fn
        )
        
        # Hierarchical loss function
        self.criterion = HierarchicalLoss(
            lambda_compilation=config.get('lambda_compilation', 1.0),
            lambda_execution=config.get('lambda_execution', 1.0),
            adaptive_weighting=config.get('adaptive_weighting', True)
        )
        
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
            'train_compilation_loss': [],
            'val_compilation_loss': [],
            'train_execution_loss': [],
            'val_execution_loss': [],
            'train_compilation_acc': [],
            'val_compilation_acc': [],
            'train_execution_acc': [],
            'val_execution_acc': [],
            'learning_rate': []
        }
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.epochs_without_improvement = 0
        
        # Results directory
        self.results_dir = config.get('results_dir', 'experiments/results/hierarchical')
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.logger.info(f"Initialized HierarchicalTrainer for {model_name}")
        self.logger.info(f"Device: {device}")
        self.logger.info(f"Training samples: {len(train_dataset)}")
        self.logger.info(f"Validation samples: {len(val_dataset)}")
        self.logger.info(f"Test samples: {len(test_dataset)}")
        self.logger.info(f"Compilation classes: {model.num_compilation_classes}")
        self.logger.info(f"Execution classes: {model.num_execution_classes}")
    
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        total_loss = 0.0
        total_compilation_loss = 0.0
        total_execution_loss = 0.0
        
        compilation_correct = 0
        compilation_total = 0
        execution_correct = 0
        execution_total = 0
        
        for batch in self.train_loader:
            # Move to device
            exercise_ids = batch['exercise_ids'].to(self.device)
            target_exercises = batch['target_exercises'].to(self.device)
            compilation_labels_t = batch['compilation_labels_t'].to(self.device)
            compilation_labels_t1 = batch['compilation_labels_t1'].to(self.device)
            execution_labels_t = batch['execution_labels_t'].to(self.device)
            execution_labels_t1 = batch['execution_labels_t1'].to(self.device)
            seq_lens = batch['seq_lens'].to(self.device)
            mask = batch['mask'].to(self.device)
            
            # Forward pass
            outputs = self.model(
                exercise_ids=exercise_ids,
                compilation_labels=compilation_labels_t,
                execution_labels=execution_labels_t,
                seq_lens=seq_lens
            )
            
            compilation_logits = outputs['compilation_logits']  # (batch, seq_len, num_ex, 2)
            execution_logits = outputs['execution_logits']      # (batch, seq_len, num_ex, 7)
            
            # Select logits for target exercises
            batch_size, seq_len = target_exercises.shape
            target_one_hot = torch.nn.functional.one_hot(
                target_exercises,
                num_classes=self.model.num_exercises
            ).float().unsqueeze(-1)  # (batch, seq_len, num_ex, 1)
            
            compilation_logits_target = (compilation_logits * target_one_hot).sum(dim=2)
            execution_logits_target = (execution_logits * target_one_hot).sum(dim=2)
            
            # Compute hierarchical loss
            loss_dict = self.criterion(
                compilation_logits=compilation_logits_target,
                execution_logits=execution_logits_target,
                compilation_targets=compilation_labels_t1,
                execution_targets=execution_labels_t1,
                mask=mask
            )
            
            loss = loss_dict['total_loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            total_compilation_loss += loss_dict['compilation_loss'].item()
            total_execution_loss += loss_dict['execution_loss'].item()
            
            # Compute accuracies
            with torch.no_grad():
                # Compilation accuracy
                _, comp_pred = torch.max(compilation_logits_target, dim=-1)
                comp_correct_batch = ((comp_pred == compilation_labels_t1) & mask).sum().item()
                compilation_correct += comp_correct_batch
                compilation_total += mask.sum().item()
                
                # Execution accuracy (only for non-CE cases)
                exec_mask = (execution_labels_t1 != -1) & mask
                if exec_mask.sum() > 0:
                    _, exec_pred = torch.max(execution_logits_target, dim=-1)
                    exec_correct_batch = ((exec_pred == execution_labels_t1) & exec_mask).sum().item()
                    execution_correct += exec_correct_batch
                    execution_total += exec_mask.sum().item()
        
        # Compute epoch metrics
        metrics = {
            'loss': total_loss / len(self.train_loader),
            'compilation_loss': total_compilation_loss / len(self.train_loader),
            'execution_loss': total_execution_loss / len(self.train_loader),
            'compilation_acc': compilation_correct / compilation_total if compilation_total > 0 else 0.0,
            'execution_acc': execution_correct / execution_total if execution_total > 0 else 0.0
        }
        
        return metrics
    
    
    def validate_epoch(self) -> Dict[str, float]:
        """
        Validate for one epoch.
        
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        total_compilation_loss = 0.0
        total_execution_loss = 0.0
        
        compilation_correct = 0
        compilation_total = 0
        execution_correct = 0
        execution_total = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move to device
                exercise_ids = batch['exercise_ids'].to(self.device)
                target_exercises = batch['target_exercises'].to(self.device)
                compilation_labels_t = batch['compilation_labels_t'].to(self.device)
                compilation_labels_t1 = batch['compilation_labels_t1'].to(self.device)
                execution_labels_t = batch['execution_labels_t'].to(self.device)
                execution_labels_t1 = batch['execution_labels_t1'].to(self.device)
                seq_lens = batch['seq_lens'].to(self.device)
                mask = batch['mask'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    exercise_ids=exercise_ids,
                    compilation_labels=compilation_labels_t,
                    execution_labels=execution_labels_t,
                    seq_lens=seq_lens
                )
                
                compilation_logits = outputs['compilation_logits']
                execution_logits = outputs['execution_logits']
                
                # Select logits for target exercises
                batch_size, seq_len = target_exercises.shape
                target_one_hot = torch.nn.functional.one_hot(
                    target_exercises,
                    num_classes=self.model.num_exercises
                ).float().unsqueeze(-1)
                
                compilation_logits_target = (compilation_logits * target_one_hot).sum(dim=2)
                execution_logits_target = (execution_logits * target_one_hot).sum(dim=2)
                
                # Compute hierarchical loss
                loss_dict = self.criterion(
                    compilation_logits=compilation_logits_target,
                    execution_logits=execution_logits_target,
                    compilation_targets=compilation_labels_t1,
                    execution_targets=execution_labels_t1,
                    mask=mask
                )
                
                total_loss += loss_dict['total_loss'].item()
                total_compilation_loss += loss_dict['compilation_loss'].item()
                total_execution_loss += loss_dict['execution_loss'].item()
                
                # Compute accuracies
                _, comp_pred = torch.max(compilation_logits_target, dim=-1)
                comp_correct_batch = ((comp_pred == compilation_labels_t1) & mask).sum().item()
                compilation_correct += comp_correct_batch
                compilation_total += mask.sum().item()
                
                exec_mask = (execution_labels_t1 != -1) & mask
                if exec_mask.sum() > 0:
                    _, exec_pred = torch.max(execution_logits_target, dim=-1)
                    exec_correct_batch = ((exec_pred == execution_labels_t1) & exec_mask).sum().item()
                    execution_correct += exec_correct_batch
                    execution_total += exec_mask.sum().item()
        
        metrics = {
            'loss': total_loss / len(self.val_loader),
            'compilation_loss': total_compilation_loss / len(self.val_loader),
            'execution_loss': total_execution_loss / len(self.val_loader),
            'compilation_acc': compilation_correct / compilation_total if compilation_total > 0 else 0.0,
            'execution_acc': execution_correct / execution_total if execution_total > 0 else 0.0
        }
        
        return metrics
    
    
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
            plt.ion()
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            fig.suptitle(f'Hierarchical KT Training Progress: {self.model_name}')
        
        for epoch in range(self.num_epochs):
            # Train epoch
            train_metrics = self.train_epoch()
            
            # Validate epoch
            val_metrics = self.validate_epoch()
            
            # Update learning rate
            self.scheduler.step(val_metrics['loss'])
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Store history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_compilation_loss'].append(train_metrics['compilation_loss'])
            self.history['val_compilation_loss'].append(val_metrics['compilation_loss'])
            self.history['train_execution_loss'].append(train_metrics['execution_loss'])
            self.history['val_execution_loss'].append(val_metrics['execution_loss'])
            self.history['train_compilation_acc'].append(train_metrics['compilation_acc'])
            self.history['val_compilation_acc'].append(val_metrics['compilation_acc'])
            self.history['train_execution_acc'].append(train_metrics['execution_acc'])
            self.history['val_execution_acc'].append(val_metrics['execution_acc'])
            self.history['learning_rate'].append(current_lr)
            
            # Log progress
            self.logger.info(
                f"Epoch [{epoch+1}/{self.num_epochs}] "
                f"Total Loss: Train={train_metrics['loss']:.4f}, Val={val_metrics['loss']:.4f} | "
                f"Comp Acc: Train={train_metrics['compilation_acc']:.4f}, Val={val_metrics['compilation_acc']:.4f} | "
                f"Exec Acc: Train={train_metrics['execution_acc']:.4f}, Val={val_metrics['execution_acc']:.4f} | "
                f"LR: {current_lr:.6f}"
            )
            
            # Update live plot
            if plot_live:
                self._update_live_plot(axes, epoch)
            
            # Check for best model
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.best_epoch = epoch
                self.epochs_without_improvement = 0
                
                # Save best model
                self.save_checkpoint(is_best=True)
                self.logger.info(f"✓ New best model saved (val_loss: {val_metrics['loss']:.4f})")
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
    
    
    def _update_live_plot(self, axes, epoch):
        """Update live training plot."""
        epochs_range = range(1, epoch + 2)
        
        # Clear all axes
        for ax_row in axes:
            for ax in ax_row:
                ax.clear()
        
        # Total loss
        axes[0, 0].plot(epochs_range, self.history['train_loss'], 'b-', label='Train')
        axes[0, 0].plot(epochs_range, self.history['val_loss'], 'r-', label='Val')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Compilation loss
        axes[0, 1].plot(epochs_range, self.history['train_compilation_loss'], 'b-', label='Train')
        axes[0, 1].plot(epochs_range, self.history['val_compilation_loss'], 'r-', label='Val')
        axes[0, 1].set_title('Compilation Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Execution loss
        axes[0, 2].plot(epochs_range, self.history['train_execution_loss'], 'b-', label='Train')
        axes[0, 2].plot(epochs_range, self.history['val_execution_loss'], 'r-', label='Val')
        axes[0, 2].set_title('Execution Loss')
        axes[0, 2].legend()
        axes[0, 2].grid(True)
        
        # Compilation accuracy
        axes[1, 0].plot(epochs_range, self.history['train_compilation_acc'], 'b-', label='Train')
        axes[1, 0].plot(epochs_range, self.history['val_compilation_acc'], 'r-', label='Val')
        axes[1, 0].set_title('Compilation Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Execution accuracy
        axes[1, 1].plot(epochs_range, self.history['train_execution_acc'], 'b-', label='Train')
        axes[1, 1].plot(epochs_range, self.history['val_execution_acc'], 'r-', label='Val')
        axes[1, 1].set_title('Execution Accuracy')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        # Learning rate
        axes[1, 2].plot(epochs_range, self.history['learning_rate'], 'g-')
        axes[1, 2].set_title('Learning Rate')
        axes[1, 2].set_yscale('log')
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        plt.pause(0.01)
    
    
    def evaluate(self) -> Dict:
        """
        Evaluate model on test set with comprehensive metrics.
        
        Returns:
            Dictionary with test metrics
        """
        self.logger.info("Evaluating on test set...")
        
        # Load best model
        self.load_checkpoint(is_best=True)
        
        self.model.eval()
        
        # Collections for metrics
        all_compilation_preds = []
        all_compilation_labels = []
        all_compilation_probs = []
        
        all_execution_preds = []
        all_execution_labels = []
        all_execution_probs = []
        
        with torch.no_grad():
            for batch in self.test_loader:
                exercise_ids = batch['exercise_ids'].to(self.device)
                target_exercises = batch['target_exercises'].to(self.device)
                compilation_labels_t = batch['compilation_labels_t'].to(self.device)
                compilation_labels_t1 = batch['compilation_labels_t1'].to(self.device)
                execution_labels_t = batch['execution_labels_t'].to(self.device)
                execution_labels_t1 = batch['execution_labels_t1'].to(self.device)
                seq_lens = batch['seq_lens'].to(self.device)
                mask = batch['mask'].to(self.device)
                
                # Get predictions
                pred_dict = self.model.predict(
                    exercise_ids=exercise_ids,
                    compilation_labels=compilation_labels_t,
                    execution_labels=execution_labels_t,
                    target_exercises=target_exercises,
                    seq_lens=seq_lens
                )
                
                compilation_probs = pred_dict['compilation_probs']
                execution_probs = pred_dict['execution_probs']
                
                # Apply mask
                mask_flat = mask.view(-1).cpu().numpy()
                compilation_probs_flat = compilation_probs.view(-1, self.model.num_compilation_classes).cpu().numpy()
                execution_probs_flat = execution_probs.view(-1, self.model.num_execution_classes).cpu().numpy()
                compilation_labels_flat = compilation_labels_t1.view(-1).cpu().numpy()
                execution_labels_flat = execution_labels_t1.view(-1).cpu().numpy()
                
                # Filter by mask
                compilation_probs_valid = compilation_probs_flat[mask_flat]
                execution_probs_valid = execution_probs_flat[mask_flat]
                compilation_labels_valid = compilation_labels_flat[mask_flat]
                execution_labels_valid = execution_labels_flat[mask_flat]
                
                # Compilation predictions
                compilation_preds_valid = np.argmax(compilation_probs_valid, axis=1)
                all_compilation_preds.extend(compilation_preds_valid)
                all_compilation_labels.extend(compilation_labels_valid)
                all_compilation_probs.extend(compilation_probs_valid)
                
                # Execution predictions (only for compiled cases)
                exec_valid_mask = execution_labels_valid != -1
                if exec_valid_mask.sum() > 0:
                    execution_preds_valid = np.argmax(execution_probs_valid[exec_valid_mask], axis=1)
                    all_execution_preds.extend(execution_preds_valid)
                    all_execution_labels.extend(execution_labels_valid[exec_valid_mask])
                    all_execution_probs.extend(execution_probs_valid[exec_valid_mask])
        
        # Compute metrics
        metrics = {}
        
        # Compilation metrics
        comp_acc = accuracy_score(all_compilation_labels, all_compilation_preds)
        comp_f1 = f1_score(all_compilation_labels, all_compilation_preds, average='macro')
        comp_probs_array = np.array(all_compilation_probs)
        if comp_probs_array.shape[1] == 2:
            comp_auc = roc_auc_score(all_compilation_labels, comp_probs_array[:, 1])
            metrics['compilation_auc'] = comp_auc
        
        metrics['compilation_accuracy'] = comp_acc
        metrics['compilation_f1'] = comp_f1
        
        # Execution metrics
        if len(all_execution_preds) > 0:
            exec_acc = accuracy_score(all_execution_labels, all_execution_preds)
            exec_f1 = f1_score(all_execution_labels, all_execution_preds, average='macro')
            
            metrics['execution_accuracy'] = exec_acc
            metrics['execution_f1'] = exec_f1
        
        # Log results
        self.logger.info(f"\nTest Results:")
        self.logger.info(f"  Compilation Accuracy: {metrics['compilation_accuracy']:.4f}")
        self.logger.info(f"  Compilation F1: {metrics['compilation_f1']:.4f}")
        if 'compilation_auc' in metrics:
            self.logger.info(f"  Compilation AUC: {metrics['compilation_auc']:.4f}")
        if 'execution_accuracy' in metrics:
            self.logger.info(f"  Execution Accuracy: {metrics['execution_accuracy']:.4f}")
            self.logger.info(f"  Execution F1: {metrics['execution_f1']:.4f}")
        
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
        """Save comprehensive training history plot to file."""
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle(f'Hierarchical KT Training History: {self.model_name}', fontsize=16)
        
        epochs_range = range(1, len(self.history['train_loss']) + 1)
        
        # Row 1: Losses
        axes[0, 0].plot(epochs_range, self.history['train_loss'], 'b-', label='Train', linewidth=2)
        axes[0, 0].plot(epochs_range, self.history['val_loss'], 'r-', label='Val', linewidth=2)
        axes[0, 0].axvline(x=self.best_epoch + 1, color='g', linestyle='--', label=f'Best')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(epochs_range, self.history['train_compilation_loss'], 'b-', label='Train', linewidth=2)
        axes[0, 1].plot(epochs_range, self.history['val_compilation_loss'], 'r-', label='Val', linewidth=2)
        axes[0, 1].set_title('Compilation Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[0, 2].plot(epochs_range, self.history['train_execution_loss'], 'b-', label='Train', linewidth=2)
        axes[0, 2].plot(epochs_range, self.history['val_execution_loss'], 'r-', label='Val', linewidth=2)
        axes[0, 2].set_title('Execution Loss')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Row 2: Accuracies
        axes[1, 0].plot(epochs_range, self.history['train_compilation_acc'], 'b-', label='Train', linewidth=2)
        axes[1, 0].plot(epochs_range, self.history['val_compilation_acc'], 'r-', label='Val', linewidth=2)
        axes[1, 0].set_title('Compilation Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(epochs_range, self.history['train_execution_acc'], 'b-', label='Train', linewidth=2)
        axes[1, 1].plot(epochs_range, self.history['val_execution_acc'], 'r-', label='Val', linewidth=2)
        axes[1, 1].set_title('Execution Accuracy')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Learning rate
        axes[1, 2].plot(epochs_range, self.history['learning_rate'], 'g-', linewidth=2)
        axes[1, 2].set_title('Learning Rate')
        axes[1, 2].set_yscale('log')
        axes[1, 2].grid(True, alpha=0.3)
        
        # Row 3: Comparisons
        loss_gap = [abs(t - v) for t, v in zip(self.history['train_loss'], self.history['val_loss'])]
        axes[2, 0].plot(epochs_range, loss_gap, 'purple', linewidth=2)
        axes[2, 0].set_title('Train-Val Loss Gap')
        axes[2, 0].grid(True, alpha=0.3)
        
        comp_gap = [abs(t - v) for t, v in zip(self.history['train_compilation_acc'], self.history['val_compilation_acc'])]
        axes[2, 1].plot(epochs_range, comp_gap, 'orange', linewidth=2)
        axes[2, 1].set_title('Compilation Acc Gap')
        axes[2, 1].grid(True, alpha=0.3)
        
        exec_gap = [abs(t - v) for t, v in zip(self.history['train_execution_acc'], self.history['val_execution_acc'])]
        axes[2, 2].plot(epochs_range, exec_gap, 'brown', linewidth=2)
        axes[2, 2].set_title('Execution Acc Gap')
        axes[2, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.results_dir, f'{self.model_name}_training_history.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Training plot saved to {plot_path}")
