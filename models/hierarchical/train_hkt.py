"""
Main training script for Hierarchical Knowledge Tracing (HKT) Model
Trains HKT model with two-level hierarchical predictions on processed dataset
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

from models.hierarchical.hkt import HierarchicalKTModel, create_hierarchical_model
from models.hierarchical.hkt_trainer import HierarchicalTrainer
from preprocessing.hierarchical_model_adapter import HierarchicalDataset
from logging_config import setup_logging


def load_hierarchical_datasets(base_dir: str, split_type: str = "standard"):
    """
    Load hierarchical datasets from disk.
    
    Args:
        base_dir: Base directory containing hierarchical format data
        split_type: "standard" or "domain_shift"
    
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset, vocab_data)
    """
    data_dir = os.path.join(base_dir, "hierarchical_format")
    
    train_path = os.path.join(data_dir, f"train_{split_type}_hierarchical.pkl")
    val_path = os.path.join(data_dir, f"val_{split_type}_hierarchical.pkl")
    test_path = os.path.join(data_dir, f"test_{split_type}_hierarchical.pkl")
    vocab_path = os.path.join(data_dir, "hierarchical_vocabulary.pkl")
    
    with open(train_path, 'rb') as f:
        train_dataset = pickle.load(f)
    
    with open(val_path, 'rb') as f:
        val_dataset = pickle.load(f)
    
    with open(test_path, 'rb') as f:
        test_dataset = pickle.load(f)
    
    with open(vocab_path, 'rb') as f:
        vocab_data = pickle.load(f)
    
    return train_dataset, val_dataset, test_dataset, vocab_data


def create_hkt_model(vocab_data: dict, config: dict) -> HierarchicalKTModel:
    """
    Create HKT model from vocabulary and configuration.
    
    Args:
        vocab_data: Vocabulary data dictionary
        config: Model configuration dictionary
    
    Returns:
        HierarchicalKTModel instance
    """
    model = HierarchicalKTModel(
        num_exercises=vocab_data['num_exercises'],
        num_compilation_classes=vocab_data['num_compilation_classes'],
        num_execution_classes=vocab_data['num_execution_classes'],
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        use_attention=config.get('use_attention', False)
    )
    
    return model


def train_hkt_model(split_type: str, config: dict, data_dir: str, 
                    results_dir: str, device: str) -> dict:
    """
    Train a single HKT model configuration.
    
    Args:
        split_type: "standard" or "domain_shift"
        config: Configuration dictionary
        data_dir: Data directory
        results_dir: Results directory
        device: Device to train on
    
    Returns:
        Training results dictionary
    """
    logger = logging.getLogger(f"hkt-mop.models.hkt")
    setup_logging()
    
    model_name = f"hkt_{split_type}"
    
    logger.info("=" * 80)
    logger.info(f"Training model: {model_name}")
    logger.info("=" * 80)
    
    # Load datasets
    trainds, valds, testds, vocab = load_hierarchical_datasets(data_dir, split_type)
    
    logger.info(f"Loaded datasets:")
    logger.info(f"  Train: {len(trainds)} sequences")
    logger.info(f"  Val: {len(valds)} sequences")
    logger.info(f"  Test: {len(testds)} sequences")
    logger.info(f"  Num exercises: {vocab['num_exercises']}")
    logger.info(f"  Num compilation classes: {vocab['num_compilation_classes']}")
    logger.info(f"  Num execution classes: {vocab['num_execution_classes']}")
    
    # Create model
    model = create_hkt_model(vocab, config['model'])
    
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Created HKT model with {num_params:,} parameters")
    
    # Create trainer
    trainer = HierarchicalTrainer(
        model=model,
        train_dataset=trainds,
        val_dataset=valds,
        test_dataset=testds,
        config=config['training'],
        model_name=model_name,
        device=device
    )
    
    # Train model
    results = trainer.train(plot_live=True)  # Set to True for live plotting
    
    return results


def plot_comparison(standard_results: dict, domain_shift_results: dict, 
                   results_dir: str):
    """
    Create comparison plots for standard vs domain shift training.
    
    Args:
        standard_results: Results from standard split training
        domain_shift_results: Results from domain shift split training
        results_dir: Directory to save plots
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("HKT Model Comparison: Standard vs Domain Shift", 
                 fontsize=16, fontweight='bold')
    
    # Extract histories
    std_history = standard_results['history']
    ds_history = domain_shift_results['history']
    
    std_epochs = range(1, len(std_history['train_loss']) + 1)
    ds_epochs = range(1, len(ds_history['train_loss']) + 1)
    
    # Plot 1: Total Loss
    axes[0, 0].plot(std_epochs, std_history['train_loss'], 'b-', 
                    label='Standard Train', linewidth=2)
    axes[0, 0].plot(std_epochs, std_history['val_loss'], 'b--', 
                    label='Standard Val', linewidth=2)
    axes[0, 0].plot(ds_epochs, ds_history['train_loss'], 'r-', 
                    label='Domain Shift Train', linewidth=2)
    axes[0, 0].plot(ds_epochs, ds_history['val_loss'], 'r--', 
                    label='Domain Shift Val', linewidth=2)
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Compilation Loss
    axes[0, 1].plot(std_epochs, std_history['train_compilation_loss'], 'b-', 
                    label='Standard Train', linewidth=2)
    axes[0, 1].plot(std_epochs, std_history['val_compilation_loss'], 'b--', 
                    label='Standard Val', linewidth=2)
    axes[0, 1].plot(ds_epochs, ds_history['train_compilation_loss'], 'r-', 
                    label='Domain Shift Train', linewidth=2)
    axes[0, 1].plot(ds_epochs, ds_history['val_compilation_loss'], 'r--', 
                    label='Domain Shift Val', linewidth=2)
    axes[0, 1].set_title('Compilation Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Execution Loss
    axes[0, 2].plot(std_epochs, std_history['train_execution_loss'], 'b-', 
                    label='Standard Train', linewidth=2)
    axes[0, 2].plot(std_epochs, std_history['val_execution_loss'], 'b--', 
                    label='Standard Val', linewidth=2)
    axes[0, 2].plot(ds_epochs, ds_history['train_execution_loss'], 'r-', 
                    label='Domain Shift Train', linewidth=2)
    axes[0, 2].plot(ds_epochs, ds_history['val_execution_loss'], 'r--', 
                    label='Domain Shift Val', linewidth=2)
    axes[0, 2].set_title('Execution Loss')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Loss')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Compilation Accuracy
    axes[1, 0].plot(std_epochs, std_history['train_compilation_acc'], 'b-', 
                    label='Standard Train', linewidth=2)
    axes[1, 0].plot(std_epochs, std_history['val_compilation_acc'], 'b--', 
                    label='Standard Val', linewidth=2)
    axes[1, 0].plot(ds_epochs, ds_history['train_compilation_acc'], 'r-', 
                    label='Domain Shift Train', linewidth=2)
    axes[1, 0].plot(ds_epochs, ds_history['val_compilation_acc'], 'r--', 
                    label='Domain Shift Val', linewidth=2)
    axes[1, 0].set_title('Compilation Accuracy')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Execution Accuracy
    axes[1, 1].plot(std_epochs, std_history['train_execution_acc'], 'b-', 
                    label='Standard Train', linewidth=2)
    axes[1, 1].plot(std_epochs, std_history['val_execution_acc'], 'b--', 
                    label='Standard Val', linewidth=2)
    axes[1, 1].plot(ds_epochs, ds_history['train_execution_acc'], 'r-', 
                    label='Domain Shift Train', linewidth=2)
    axes[1, 1].plot(ds_epochs, ds_history['val_execution_acc'], 'r--', 
                    label='Domain Shift Val', linewidth=2)
    axes[1, 1].set_title('Execution Accuracy')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Learning Rate
    axes[1, 2].plot(std_epochs, std_history['learning_rate'], 'b-', 
                    label='Standard', linewidth=2)
    axes[1, 2].plot(ds_epochs, ds_history['learning_rate'], 'r-', 
                    label='Domain Shift', linewidth=2)
    axes[1, 2].set_title('Learning Rate')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Learning Rate')
    axes[1, 2].set_yscale('log')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plotpath = os.path.join(results_dir, "hkt_training_comparison.png")
    plt.savefig(plotpath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison plot saved to {plotpath}")


def main():
    """Main execution function."""
    logger = logging.getLogger("hkt-mop.models.hkt")
    setup_logging()
    
    logger.info("=" * 80)
    logger.info("HIERARCHICAL KNOWLEDGE TRACING (HKT) MODEL TRAINING")
    logger.info("=" * 80)
    
    # Load configuration
    configpath = "experiments/configs/hkt_config.yaml"
    with open(configpath, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup directories
    data_dir = config['data']['hierarchical_format_dir']
    results_dir = config['training']['results_dir']
    os.makedirs(results_dir, exist_ok=True)
    
    # Device selection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Train on both splits
    all_results = {}
    
    # 1. Train on standard split
    try:
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING: Standard Split")
        logger.info("=" * 80)
        
        standard_results = train_hkt_model(
            split_type="standard",
            config=config,
            data_dir=data_dir,
            results_dir=results_dir,
            device=device
        )
        all_results['standard'] = standard_results
        
    except Exception as e:
        logger.error(f"Error training standard split: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # 2. Train on domain shift split
    try:
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING: Domain Shift Split")
        logger.info("=" * 80)
        
        domain_shift_results = train_hkt_model(
            split_type="domain_shift",
            config=config,
            data_dir=data_dir,
            results_dir=results_dir,
            device=device
        )
        all_results['domain_shift'] = domain_shift_results
        
    except Exception as e:
        logger.error(f"Error training domain shift split: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Create comparison visualization
    if len(all_results) == 2:
        plot_comparison(
            all_results['standard'],
            all_results['domain_shift'],
            results_dir
        )
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING SUMMARY - HKT MODEL")
    logger.info("=" * 80)
    
    for split_type, results in all_results.items():
        logger.info(f"\n{split_type.upper()} SPLIT:")
        logger.info(f"  Best Epoch: {results['best_epoch'] + 1}")
        logger.info(f"  Best Val Loss: {results['best_val_loss']:.4f}")
        
        if 'test_metrics' in results:
            logger.info(f"  Test Results:")
            logger.info(f"    Compilation Accuracy: {results['test_metrics']['compilation_accuracy']:.4f}")
            logger.info(f"    Compilation F1: {results['test_metrics']['compilation_f1']:.4f}")
            
            if 'compilation_auc' in results['test_metrics']:
                logger.info(f"    Compilation AUC: {results['test_metrics']['compilation_auc']:.4f}")
            
            if 'execution_accuracy' in results['test_metrics']:
                logger.info(f"    Execution Accuracy: {results['test_metrics']['execution_accuracy']:.4f}")
                logger.info(f"    Execution F1: {results['test_metrics']['execution_f1']:.4f}")
    
    logger.info("\n" + "=" * 80)
    logger.info("All HKT training completed successfully!")
    logger.info(f"Results saved to: {results_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
