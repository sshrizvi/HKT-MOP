"""
Main training script for all DKT model configurations

Author: Syed Shujaat Haider
"""

import os
import sys
import logging
import pickle
import yaml
import matplotlib.pyplot as plt
from datetime import datetime

import torch
from models.baselines.dkt.dkt_model import DKTModel
from models.baselines.dkt.dkt_trainer import DKTTrainer
from preprocessing.dkt_format_adapter import DKTDataset
from logging_config import setup_logging

def load_dkt_datasets(base_dir: str, mode: str, split_type: str):
    """
    Load DKT datasets from disk.
    
    Args:
        base_dir: Base directory containing DKT format data
        mode: 'binary' or 'multiclass'
        split_type: 'standard' or 'domainshift'
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset, vocab_data)
    """
    data_dir = os.path.join(base_dir, f'dkt_format_{mode}')
    
    # Load datasets
    train_path = os.path.join(data_dir, f'train_{split_type}_dkt.pkl')
    val_path = os.path.join(data_dir, f'val_{split_type}_dkt.pkl')
    test_path = os.path.join(data_dir, f'test_{split_type}_dkt.pkl')
    vocab_path = os.path.join(data_dir, 'dkt_vocabulary.pkl')
    
    with open(train_path, 'rb') as f:
        train_dataset = pickle.load(f)
    with open(val_path, 'rb') as f:
        val_dataset = pickle.load(f)
    with open(test_path, 'rb') as f:
        test_dataset = pickle.load(f)
    with open(vocab_path, 'rb') as f:
        vocab_data = pickle.load(f)
    
    return train_dataset, val_dataset, test_dataset, vocab_data

def create_dkt_model(vocab_data: dict, config: dict) -> DKTModel:
    """
    Create DKT model from vocabulary and config.
    
    Args:
        vocab_data: Vocabulary data dictionary
        config: Model configuration
        
    Returns:
        DKTModel instance
    """
    num_exercises = vocab_data['num_exercises']
    num_outcomes = vocab_data['num_outcomes']
    use_compressed = vocab_data['use_compressed']
    
    if use_compressed:
        input_dim = min(200, num_exercises * num_outcomes)
    else:
        input_dim = num_exercises * num_outcomes
    
    model = DKTModel(
        input_dim=input_dim,
        hidden_dim=config['hidden_dim'],
        num_exercises=num_exercises,
        num_outcomes=num_outcomes,
        num_layers=config['num_layers'],
        dropout=config['dropout']
    )
    
    return model

def train_single_model(
    mode: str,
    split_type: str,
    base_config: dict,
    data_dir: str,
    results_dir: str,
    device: str
) -> dict:
    """
    Train a single DKT model configuration.
    
    Args:
        mode: 'binary' or 'multiclass'
        split_type: 'standard' or 'domainshift'
        base_config: Base configuration dictionary
        data_dir: Data directory
        results_dir: Results directory
        device: Device to train on
        
    Returns:
        Training results dictionary
    """
    logger = logging.getLogger(f"hkt-mop.models.dkt")
    setup_logging()
    
    model_name = f"dkt_{mode}_{split_type}"
    logger.info("=" * 80)
    logger.info(f"Training model: {model_name}")
    logger.info("=" * 80)
    
    # Load datasets
    train_ds, val_ds, test_ds, vocab = load_dkt_datasets(data_dir, mode, split_type)
    
    logger.info(f"Loaded datasets:")
    logger.info(f"  Train: {len(train_ds)} sequences")
    logger.info(f"  Val: {len(val_ds)} sequences")
    logger.info(f"  Test: {len(test_ds)} sequences")
    logger.info(f"  Num exercises: {vocab['num_exercises']}")
    logger.info(f"  Num outcomes: {vocab['num_outcomes']}")
    
    # Create model
    model = create_dkt_model(vocab, base_config['model'])
    
    logger.info(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create trainer
    trainer = DKTTrainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        test_dataset=test_ds,
        config=base_config['training'],
        model_name=model_name,
        device=device
    )
    
    # Train
    results = trainer.train(plot_live=False)  # Set to True for live plotting
    
    return results

def plot_combined_losses(all_results: dict, results_dir: str):
    """
    Create a combined plot showing all four models' training losses.
    
    Args:
        all_results: Dictionary containing results for all models
        results_dir: Directory to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('DKT Training Loss Comparison - All Configurations', fontsize=16, fontweight='bold')
    
    # Define model configurations
    configs = [
        ('binary', 'standard', 'Binary - Standard Split'),
        ('multiclass', 'standard', 'Multiclass - Standard Split'),
        ('binary', 'domain_shift', 'Binary - Domain Shift Split'),
        ('multiclass', 'domain_shift', 'Multiclass - Domain Shift Split')
    ]
    
    for idx, (mode, split_type, title) in enumerate(configs):
        ax = axes[idx // 2, idx % 2]
        model_name = f"dkt_{mode}_{split_type}"
        
        if model_name in all_results:
            history = all_results[model_name]['history']
            epochs = range(1, len(history['train_loss']) + 1)
            
            # Plot losses
            ax.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2, alpha=0.8)
            ax.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2, alpha=0.8)
            
            # Mark best epoch
            best_epoch = all_results[model_name]['best_epoch']
            ax.axvline(x=best_epoch + 1, color='g', linestyle='--', 
                      label=f'Best Epoch ({best_epoch + 1})', alpha=0.7)
            
            ax.set_xlabel('Epoch', fontsize=11)
            ax.set_ylabel('Loss', fontsize=11)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(results_dir, 'all_models_training_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Combined training plot saved to: {plot_path}")

def main():
    """Main execution function."""
    # Setup logging
    logger = logging.getLogger("hkt-mop.models.dkt")
    setup_logging()
    
    logger.info("=" * 80)
    logger.info("DKT MODEL TRAINING - ALL CONFIGURATIONS")
    logger.info("=" * 80)
    
    # Load configuration
    config_path = 'experiments/configs/dkt_config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup directories
    data_dir = config['data']['dkt_format_dir']
    results_dir = config['training']['results_dir']
    os.makedirs(results_dir, exist_ok=True)
    
    # Device selection
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Train all four configurations
    configurations = [
        ('binary', 'standard'),
        ('multiclass', 'standard'),
        ('binary', 'domain_shift'),
        ('multiclass', 'domain_shift')
    ]
    
    all_results = {}
    
    for mode, split_type in configurations:
        try:
            results = train_single_model(
                mode=mode,
                split_type=split_type,
                base_config=config,
                data_dir=data_dir,
                results_dir=results_dir,
                device=device
            )
            model_name = f"dkt_{mode}_{split_type}"
            all_results[model_name] = results
            
        except Exception as e:
            logger.error(f"Error training {mode}_{split_type}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Create combined visualization
    if all_results:
        plot_combined_losses(all_results, results_dir)
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING SUMMARY - ALL MODELS")
    logger.info("=" * 80)
    
    for model_name, results in all_results.items():
        logger.info(f"\n{model_name.upper()}:")
        logger.info(f"  Best Epoch: {results['best_epoch'] + 1}")
        logger.info(f"  Best Val Loss: {results['best_val_loss']:.4f}")
        if 'test_metrics' in results:
            logger.info(f"  Test Accuracy: {results['test_metrics']['test_accuracy']:.4f}")
            if 'test_auc' in results['test_metrics']:
                logger.info(f"  Test AUC: {results['test_metrics']['test_auc']:.4f}")
            logger.info(f"  Test F1: {results['test_metrics']['test_f1']:.4f}")
    
    logger.info("\n" + "=" * 80)
    logger.info("✓ All training completed successfully!")
    logger.info(f"Results saved to: {results_dir}")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()
