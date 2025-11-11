"""
Model-specific format adapters for Knowledge Tracing baselines.
Converts preprocessed data from BasePreprocessor into model-specific input formats.
Author: Syed Shujaat Haider
"""

import os
import pickle
import logging
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from preprocessing.base_preprocessor import BasePreprocessor


class DKTFormatAdapter:
    """
    Format adapter for Deep Knowledge Tracing (DKT) model.
    
    Supports both binary and multiclass classification modes.
    
    DKT Input Requirements (from Piech et al., 2015):
    - Binary mode: One-hot encoding of (exercise_id, correctness) tuple
      * For M unique exercises: input dimension = 2M
    - Multiclass mode: One-hot encoding of (exercise_id, outcome_class) tuple
      * For M exercises and K outcomes: input dimension = M*K
    - Output: Probability distribution over all possible outcomes for each exercise
    """
    
    def __init__(self, config_path: str = "experiments/configs/preprocessing_config.yaml",
                 multiclass: bool = True, use_outcome_groups: bool = False):
        """
        Initialize DKT format adapter.
        
        Args:
            config_path: Path to preprocessing configuration
            multiclass: If True, use multiclass classification; if False, use binary
            use_outcome_groups: If True, use coarse outcome groups; if False, use fine-grained outcomes
        """
        self.logger = logging.getLogger('hkt-mop.preprocessing.dkt')
        self.preprocessor = BasePreprocessor(config_path)
        self.config = self.preprocessor.config
        
        # Multiclass Configuration
        self.multiclass = multiclass
        self.use_outcome_groups = use_outcome_groups
        
        # DKT-specific parameters
        self.num_exercises = None
        self.exercise_to_idx = {}
        self.idx_to_exercise = {}
        self.use_compressed = False
        
        # Multiclass Parameters
        self.num_outcomes = None
        self.outcome_to_idx = {}
        self.idx_to_outcome = {}
        
        mode_str = "multiclass" if multiclass else "binary"
        group_str = "groups" if use_outcome_groups else "fine-grained"
        self.logger.info(f"DKTFormatAdapter initialized in {mode_str} mode ({group_str})")
    
    
    def prepare_dkt_splits(self, splits: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Dict[str, 'DKTDataset']]:
        """
        Convert preprocessed splits into DKT-specific format.
        
        Args:
            splits: Dictionary from BasePreprocessor.run()
        
        Returns:
            Dictionary of DKT datasets for each split
        """
        
        self.logger.info("Preparing DKT format datasets...")
        
        # Build exercise vocabulary from all data
        all_exercises = set()
        for split_type, split_dict in splits.items():
            for split_name, split_df in split_dict.items():
                all_exercises.update(split_df['problem_id'].unique())
        
        self.num_exercises = len(all_exercises)
        self.exercise_to_idx = {ex: idx for idx, ex in enumerate(sorted(all_exercises))}
        self.idx_to_exercise = {idx: ex for ex, idx in self.exercise_to_idx.items()}
        
        self.logger.info(f"Built exercise vocabulary with {self.num_exercises} unique exercises")
        
        # Build outcome vocabulary for multiclass
        if self.multiclass:
            # Get outcome mappings from preprocessor
            if self.use_outcome_groups:
                self.num_outcomes = self.preprocessor.num_outcome_groups
                self.outcome_to_idx = self.preprocessor.outcome_group_to_idx
                self.idx_to_outcome = self.preprocessor.idx_to_outcome_group
                label_col = 'outcome_group_label'
            else:
                self.num_outcomes = self.preprocessor.num_outcomes
                self.outcome_to_idx = self.preprocessor.outcome_to_idx
                self.idx_to_outcome = self.preprocessor.idx_to_outcome
                label_col = 'multiclass_label'
            
            self.logger.info(f"Using multiclass with {self.num_outcomes} outcome classes")
        else:
            self.num_outcomes = 2  # Binary: correct/incorrect
            label_col = 'correct'
            self.logger.info("Using binary classification (correct/incorrect)")
        
        # Determine if compressed representation is needed
        input_size = self.num_exercises * self.num_outcomes
        self.use_compressed = input_size > 1000
        
        if self.use_compressed:
            self.logger.info(f"Using compressed representation (input_size={input_size} > 1000)")
        else:
            self.logger.info(f"Using one-hot encoding (input_size={input_size})")
        
        # Convert each split to DKT format
        dkt_splits = {}
        for split_type, split_dict in splits.items():
            dkt_splits[split_type] = {}
            for split_name, split_df in split_dict.items():
                dataset = self._create_dkt_dataset(split_df, split_name, split_type, label_col)
                dkt_splits[split_type][split_name] = dataset
                self.logger.info(f"Created DKT dataset for {split_name}_{split_type}: {len(dataset)} sequences")
        
        return dkt_splits
    
    
    def _create_dkt_dataset(self, df: pd.DataFrame, split_name: str, 
                           split_type: str, label_col: str) -> 'DKTDataset':
        """
        Create DKT dataset from dataframe.
        
        Args:
            df: Preprocessed dataframe with student sequences
            split_name: Name of split ('train', 'val', 'test')
            split_type: Type of split ('standard', 'domain_shift')
            label_col: Column name for labels ('correct', 'multiclass_label', or 'outcome_group_label')
        
        Returns:
            DKTDataset instance
        """
        
        # Group by student to create sequences
        student_sequences = []
        
        for student_id, group in df.groupby('creator_id'):
            # Sort by submission id (temporal order)
            group = group.sort_values('id')
            
            # Extract exercise IDs and labels
            exercise_ids = group['problem_id'].values
            labels = group[label_col].values
            
            # Convert to indices
            exercise_indices = [self.exercise_to_idx[ex] for ex in exercise_ids]
            
            student_sequences.append({
                'student_id': student_id,
                'exercise_ids': exercise_indices,
                'labels': labels,
                'sequence_length': len(exercise_indices)
            })
        
        return DKTDataset(
            sequences=student_sequences,
            num_exercises=self.num_exercises,
            num_outcomes=self.num_outcomes,
            use_compressed=self.use_compressed,
            split_name=split_name,
            split_type=split_type
        )
    
    
    def save_dkt_datasets(self, dkt_splits: Dict[str, Dict[str, 'DKTDataset']], 
                         output_dir: Optional[str] = None):
        """
        Save DKT datasets to disk.
        
        Args:
            dkt_splits: Dictionary of DKT datasets
            output_dir: Output directory (defaults to config)
        """
        
        if output_dir is None:
            mode_suffix = "multiclass" if self.multiclass else "binary"
            output_dir = os.path.join(self.config["data"]["dkt_format_dir"], f"dkt_format_{mode_suffix}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.logger.info(f"Saving DKT datasets to {output_dir}...")
        
        # Save datasets
        for split_type, split_dict in dkt_splits.items():
            for split_name, dataset in split_dict.items():
                output_path = os.path.join(output_dir, f"{split_name}_{split_type}_dkt.pkl")
                with open(output_path, 'wb') as f:
                    pickle.dump(dataset, f)
                self.logger.info(f"Saved {split_name}_{split_type} DKT dataset to {output_path}")
        
        # Save vocabulary mapping
        vocab_path = os.path.join(output_dir, "dkt_vocabulary.pkl")
        vocab_data = {
            'num_exercises': self.num_exercises,
            'exercise_to_idx': self.exercise_to_idx,
            'idx_to_exercise': self.idx_to_exercise,
            'use_compressed': self.use_compressed,
            'multiclass': self.multiclass,
            'num_outcomes': self.num_outcomes,
            'outcome_to_idx': self.outcome_to_idx,
            'idx_to_outcome': self.idx_to_outcome,
            'use_outcome_groups': self.use_outcome_groups
        }
        
        with open(vocab_path, 'wb') as f:
            pickle.dump(vocab_data, f)
        self.logger.info(f"Saved DKT vocabulary to {vocab_path}")
        
    def run(self) -> Dict[str, Dict[str, 'DKTDataset']]:
        """
        Execute complete DKT format preparation pipeline.
        
        Returns:
            Dictionary of DKT datasets
        """
        self.logger.info("=" * 80)
        self.logger.info("Starting DKT Format Preparation Pipeline")
        self.logger.info("=" * 80)
        
        # Step 1: Run base preprocessing
        splits = self.preprocessor.run()
        
        # Step 2: Convert to DKT format
        dkt_splits = self.prepare_dkt_splits(splits)
        
        # Step 3: Save DKT datasets
        self.save_dkt_datasets(dkt_splits)
        
        self.logger.info("=" * 80)
        self.logger.info("DKT Format Preparation Complete")
        self.logger.info("=" * 80)
        
        return dkt_splits


class DKTDataset(Dataset):
    """
    PyTorch Dataset for Deep Knowledge Tracing.
    
    Supports both binary and multiclass classification.
    
    Input Encoding:
    - Binary: x_t ∈ {0,1}^(2M) where M is number of exercises
      * Position [0, M-1]: exercise_id with correctness=0
      * Position [M, 2M-1]: exercise_id with correctness=1
    
    - Multiclass: x_t ∈ {0,1}^(M*K) where M is exercises, K is outcome classes
      * Position [k*M + m]: exercise m with outcome class k
    
    Output:
    - Binary: Probability distribution over {correct, incorrect} for each exercise
    - Multiclass: Probability distribution over K outcome classes for each exercise
    """
    
    def __init__(self, sequences: List[Dict], num_exercises: int, 
                 num_outcomes: int = 2, use_compressed: bool = False, 
                 split_name: str = "", split_type: str = ""):
        """
        Initialize DKT dataset.
        
        Args:
            sequences: List of student sequences
            num_exercises: Total number of unique exercises
            num_outcomes: Number of outcome classes (2 for binary, K for multiclass)
            use_compressed: Whether to use compressed representation
            split_name: Name of split ('train', 'val', 'test')
            split_type: Type of split ('standard', 'domain_shift')
        """
        self.sequences = sequences
        self.num_exercises = num_exercises
        self.num_outcomes = num_outcomes
        self.use_compressed = use_compressed
        self.split_name = split_name
        self.split_type = split_type
        
        # Input dimension
        if use_compressed:
            self.input_dim = min(200, num_exercises * num_outcomes)
            self._initialize_compressed_vectors()
        else:
            self.input_dim = num_exercises * num_outcomes  # M * K
    
    
    def _initialize_compressed_vectors(self):
        """Initialize random Gaussian vectors for compressed representation."""
        
        np.random.seed(42)
        
        self.compressed_vectors = {}
        for ex_id in range(self.num_exercises):
            for outcome in range(self.num_outcomes):
                key = (ex_id, outcome)
                self.compressed_vectors[key] = np.random.randn(self.input_dim).astype(np.float32)
    
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Get a single student sequence.
        
        Returns:
            inputs: Tensor of shape (seq_len, input_dim) - encoded (exercise, outcome) tuples
            targets: Tensor of shape (seq_len, num_exercises) - target exercise one-hot
            labels: Tensor of shape (seq_len,) - target outcome labels (binary or multiclass)
            seq_len: Actual sequence length
        """
        sequence = self.sequences[idx]
        exercise_ids = sequence['exercise_ids']
        labels = sequence['labels']
        seq_len = sequence['sequence_length']
        
        # Prepare inputs (x_t) and targets (q_{t+1}, label_{t+1})
        inputs = []
        target_exercises = []
        target_labels = []
        
        for t in range(seq_len - 1):  # Predict t+1 from 0 to t
            
            # Input at time t: (exercise_id_t, outcome_t)
            ex_t = exercise_ids[t]
            label_t = labels[t]
            
            if self.use_compressed:
                input_t = self.compressed_vectors[(ex_t, label_t)]
            else:
                # One-hot encoding for multiclass
                input_t = np.zeros(self.input_dim, dtype=np.float32)
                # Position: label_t * num_exercises + ex_t
                input_t[int(label_t) * self.num_exercises + ex_t] = 1.0
            
            inputs.append(input_t)
            
            # Target at time t+1
            ex_t1 = exercise_ids[t + 1]
            label_t1 = labels[t + 1]
            
            # Target exercise (one-hot)
            target_ex = np.zeros(self.num_exercises, dtype=np.float32)
            target_ex[ex_t1] = 1.0
            
            target_exercises.append(target_ex)
            target_labels.append(label_t1)
        
        # Convert to tensors
        inputs = torch.FloatTensor(np.array(inputs))  # (seq_len-1, input_dim)
        targets = torch.FloatTensor(np.array(target_exercises))  # (seq_len-1, num_exercises)
        labels = torch.LongTensor(target_labels)  # (seq_len-1,) - supports multiclass
        
        return inputs, targets, labels, seq_len - 1
    
    
    def get_metadata(self) -> Dict:
        """Return dataset metadata."""
        
        return {
            'num_sequences': len(self.sequences),
            'num_exercises': self.num_exercises,
            'num_outcomes': self.num_outcomes,  # NEW
            'input_dim': self.input_dim,
            'use_compressed': self.use_compressed,
            'split_name': self.split_name,
            'split_type': self.split_type,
            'avg_seq_length': np.mean([s['sequence_length'] for s in self.sequences]),
            'max_seq_length': max([s['sequence_length'] for s in self.sequences]),
            'min_seq_length': min([s['sequence_length'] for s in self.sequences])
        }



def collate_fn(batch: List[Tuple]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Custom collate function for batching variable-length sequences.
    
    Args:
        batch: List of (inputs, targets, labels, seq_len) tuples
    
    Returns:
        Padded batches of inputs, targets, labels, and sequence lengths
    """
    
    # Filter out any empty sequences (seq_len == 0)
    batch = [item for item in batch if item[3] > 0]
    
    # Handle case where entire batch is filtered out
    if len(batch) == 0:
        # Return dummy tensors with proper shapes
        return (
            torch.zeros(1, 1, 1),  # inputs_padded
            torch.zeros(1, 1, 1),  # targets_padded
            torch.zeros(1, 1, dtype=torch.long),  # labels_padded
            torch.zeros(1, dtype=torch.long)  # seq_lens
        )
    
    inputs_list, targets_list, labels_list, seq_lens = zip(*batch)
    
    # Get max sequence length in batch
    max_seq_len = max(seq_lens)
    batch_size = len(batch)
    input_dim = inputs_list[0].shape[1]
    num_exercises = targets_list[0].shape[1]
    
    # Initialize padded tensors
    inputs_padded = torch.zeros(batch_size, max_seq_len, input_dim)
    targets_padded = torch.zeros(batch_size, max_seq_len, num_exercises)
    labels_padded = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
    
    # Fill in actual sequences
    for i, (inp, tgt, lbl, seq_len) in enumerate(zip(inputs_list, targets_list, labels_list, seq_lens)):
        inputs_padded[i, :seq_len] = inp
        targets_padded[i, :seq_len] = tgt
        labels_padded[i, :seq_len] = lbl
    
    seq_lens = torch.LongTensor(seq_lens)
    
    return inputs_padded, targets_padded, labels_padded, seq_lens


def main():
    """Main execution function for DKT format preparation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="DKT Format Adapter")
    parser.add_argument("--config", type=str, 
                       default="experiments/configs/preprocessing_config.yaml",
                       help="Path to preprocessing configuration file")
    args = parser.parse_args()
    
    # Initialize and run Binary DKTFormatAdapter
    adapter = DKTFormatAdapter(config_path=args.config, multiclass=False, use_outcome_groups=False)
    binary_dkt_splits = adapter.run()
    
    # Initialize and run Multiclass DKTFormatAdapter
    adapter = DKTFormatAdapter(config_path=args.config, multiclass=True, use_outcome_groups=False)
    multiclass_dkt_splits = adapter.run()
    
    # Print summary
    for dkt_splits in (binary_dkt_splits, multiclass_dkt_splits):
        print("\n" + "=" * 80)
        print("DKT FORMAT PREPARATION SUMMARY")
        print("=" * 80)
        print(f"Number of unique exercises: {adapter.num_exercises}")
        print(f"Using compressed representation: {adapter.use_compressed}")
        print(f"Input dimension: {2 * adapter.num_exercises if not adapter.use_compressed else min(200, adapter.num_exercises)}")
        print("\nDatasets created:")
        for split_type, split_dict in dkt_splits.items():
            print(f"\n  {split_type.upper()}:")
            for split_name, dataset in split_dict.items():
                metadata = dataset.get_metadata()
                print(f"    - {split_name}: {metadata['num_sequences']} sequences, "
                    f"avg_len={metadata['avg_seq_length']:.1f}")
        print("=" * 80)


if __name__ == "__main__":
    main()
