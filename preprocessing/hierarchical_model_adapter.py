"""
Hierarchical Model Format Adapter with Complete Error Handling
Filters system-level errors and handles all student-attributable outcomes
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


class HierarchicalFormatAdapter:
    """
    Format adapter for Hierarchical Knowledge Tracing model.
    Handles all ACcoding result types with proper filtering of system errors.
    
    Architecture:
    - Level 1 (Compilation): Binary classification (Compiles vs CE)
    - Level 2 (Execution): Multi-class classification (AC, WA, TLE, MLE, RE, PE, OE)
    """
    
    def __init__(self, config_path: str = "experiments/configs/preprocessing_config.yaml"):
        """
        Initialize hierarchical format adapter with complete error handling.
        
        Args:
            config_path: Path to preprocessing configuration
        """
        self.logger = logging.getLogger('hkt-mop.preprocessing.hierarchical')
        self.preprocessor = BasePreprocessor(config_path)
        self.config = self.preprocessor.config
        
        # Define student-attributable outcomes (valid for KT)
        self.student_outcomes = ['AC', 'WA', 'TLE', 'MLE', 'RE', 'PE', 'CE', 'OE']
        
        # Define system-level outcomes (should be filtered out)
        self.system_outcomes = ['WT', 'JG', 'REG', 'REP', 'IFNR', 'OFNR', 'EFNR']
        
        # Hierarchical outcome structure
        self.compilation_outcomes = ['CE', 'COMPILED']  # Binary
        
        # Execution outcomes (when compilation succeeds)
        self.execution_outcomes = ['AC', 'WA', 'TLE', 'MLE', 'RE', 'PE', 'OE']
        
        # Outcome mappings
        self.num_compilation_classes = len(self.compilation_outcomes)
        self.num_execution_classes = len(self.execution_outcomes)
        
        # Build hierarchical label mappings
        self.outcome_to_compilation_label = {
            'CE': 0,  # Compilation failed
            'AC': 1, 'WA': 1, 'TLE': 1, 'MLE': 1, 
            'RE': 1, 'PE': 1, 'OE': 1  # All compiled successfully
        }
        
        self.outcome_to_execution_label = {
            'AC': 0,    # Accepted
            'WA': 1,    # Wrong Answer
            'TLE': 2,   # Time Limit Exceeded
            'MLE': 3,   # Memory Limit Exceeded
            'RE': 4,    # Runtime Error
            'PE': 5,    # Presentation Error
            'OE': 6,    # Other Error
            'CE': -1    # Not applicable (didn't reach execution)
        }
        
        # Exercise vocabulary
        self.num_exercises = None
        self.exercise_to_idx = {}
        self.idx_to_exercise = {}
        
        # Statistics tracking
        self.filtered_stats = {
            'total_submissions': 0,
            'filtered_system_errors': 0,
            'valid_submissions': 0,
            'outcome_distribution': {}
        }
        
        self.logger.info(f"HierarchicalFormatAdapter initialized")
        self.logger.info(f"  - Student outcomes: {self.student_outcomes}")
        self.logger.info(f"  - System outcomes (will be filtered): {self.system_outcomes}")
        self.logger.info(f"  - Compilation classes: {self.num_compilation_classes}")
        self.logger.info(f"  - Execution classes: {self.num_execution_classes}")
    
    
    def filter_system_errors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter out system-level errors that are not attributable to students.
        
        Args:
            df: DataFrame with 'result' column
        
        Returns:
            Filtered DataFrame containing only student-attributable outcomes
        """
        
        self.filtered_stats['total_submissions'] = len(df)
        
        # Count system errors before filtering
        system_error_mask = df['result'].isin(self.system_outcomes)
        system_error_count = system_error_mask.sum()
        self.filtered_stats['filtered_system_errors'] = system_error_count
        
        if system_error_count > 0:
            self.logger.warning(
                f"Filtering {system_error_count} system-level errors "
                f"({100 * system_error_count / len(df):.2f}% of total submissions)"
            )
            
            # Log breakdown of system errors
            system_error_breakdown = df[system_error_mask]['result'].value_counts()
            self.logger.info("System error breakdown:")
            for error_type, count in system_error_breakdown.items():
                self.logger.info(f"  - {error_type}: {count}")
        
        # Filter to keep only student outcomes
        filtered_df = df[df['result'].isin(self.student_outcomes)].copy()
        self.filtered_stats['valid_submissions'] = len(filtered_df)
        
        # Log outcome distribution after filtering
        outcome_dist = filtered_df['result'].value_counts()
        self.filtered_stats['outcome_distribution'] = outcome_dist.to_dict()
        
        self.logger.info(f"Valid submissions after filtering: {len(filtered_df)}")
        self.logger.info("Student outcome distribution:")
        for outcome, count in outcome_dist.items():
            pct = 100 * count / len(filtered_df)
            self.logger.info(f"  - {outcome}: {count} ({pct:.2f}%)")
        
        return filtered_df
    
    
    def prepare_hierarchical_splits(
        self, 
        splits: Dict[str, Dict[str, pd.DataFrame]]
    ) -> Dict[str, Dict[str, 'HierarchicalDataset']]:
        """
        Convert preprocessed splits into hierarchical format.
        Filters system errors and creates hierarchical labels.
        
        Args:
            splits: Dictionary from BasePreprocessor.run()
        
        Returns:
            Dictionary of hierarchical datasets for each split
        """
        
        self.logger.info("=" * 80)
        self.logger.info("Preparing hierarchical format datasets with error filtering...")
        self.logger.info("=" * 80)
        
        # First pass: Filter system errors and build exercise vocabulary
        all_exercises = set()
        filtered_splits = {}
        
        for split_type, split_dict in splits.items():
            filtered_splits[split_type] = {}
            self.logger.info(f"\nProcessing {split_type} split:")
            
            for split_name, split_df in split_dict.items():
                self.logger.info(f"  Processing {split_name}...")
                
                # Filter system errors
                filtered_df = self.filter_system_errors(split_df)
                
                # Remove students with empty sequences after filtering
                student_seq_lengths = filtered_df.groupby('creator_id').size()
                valid_students = student_seq_lengths[student_seq_lengths >= 2].index
                filtered_df = filtered_df[filtered_df['creator_id'].isin(valid_students)]
                
                self.logger.info(
                    f"    Retained {len(valid_students)} students with >= 2 valid submissions"
                )
                
                filtered_splits[split_type][split_name] = filtered_df
                all_exercises.update(filtered_df['problem_id'].unique())
        
        # Build exercise vocabulary
        self.num_exercises = len(all_exercises)
        self.exercise_to_idx = {ex: idx for idx, ex in enumerate(sorted(all_exercises))}
        self.idx_to_exercise = {idx: ex for ex, idx in self.exercise_to_idx.items()}
        
        self.logger.info(f"\n{'=' * 80}")
        self.logger.info(f"Built exercise vocabulary with {self.num_exercises} unique exercises")
        self.logger.info(f"{'=' * 80}\n")
        
        # Second pass: Convert to hierarchical format
        hierarchical_splits = {}
        for split_type, split_dict in filtered_splits.items():
            hierarchical_splits[split_type] = {}
            for split_name, split_df in split_dict.items():
                dataset = self._create_hierarchical_dataset(split_df, split_name, split_type)
                hierarchical_splits[split_type][split_name] = dataset
                self.logger.info(
                    f"Created hierarchical dataset for {split_name}_{split_type}: "
                    f"{len(dataset)} sequences"
                )
        
        return hierarchical_splits
    
    
    def _create_hierarchical_dataset(
        self, 
        df: pd.DataFrame, 
        split_name: str,
        split_type: str
    ) -> 'HierarchicalDataset':
        """
        Create hierarchical dataset from filtered dataframe.
        
        Args:
            df: Filtered dataframe (system errors already removed)
            split_name: Name of split ('train', 'val', 'test')
            split_type: Type of split ('standard', 'domain_shift')
        
        Returns:
            HierarchicalDataset instance
        """
        
        # Group by student to create sequences
        student_sequences = []
        
        for student_id, group in df.groupby('creator_id'):
            # Sort by submission id (temporal order)
            group = group.sort_values('id')
            
            # Extract features
            exercise_ids = group['problem_id'].values
            results = group['result'].values
            
            # Validate all results are student-attributable
            if not all(result in self.student_outcomes for result in results):
                invalid_results = [r for r in results if r not in self.student_outcomes]
                self.logger.warning(
                    f"Student {student_id} has invalid results: {invalid_results}. Skipping."
                )
                continue
            
            # Convert exercises to indices
            exercise_indices = [self.exercise_to_idx[ex] for ex in exercise_ids]
            
            # Create hierarchical labels
            compilation_labels = [
                self.outcome_to_compilation_label[result] 
                for result in results
            ]
            
            execution_labels = [
                self.outcome_to_execution_label[result] 
                for result in results
            ]
            
            student_sequences.append({
                'student_id': student_id,
                'exercise_ids': exercise_indices,
                'results': results,
                'compilation_labels': compilation_labels,
                'execution_labels': execution_labels,
                'sequence_length': len(exercise_indices)
            })
        
        return HierarchicalDataset(
            sequences=student_sequences,
            num_exercises=self.num_exercises,
            num_compilation_classes=self.num_compilation_classes,
            num_execution_classes=self.num_execution_classes,
            split_name=split_name,
            split_type=split_type
        )
    
    
    def save_hierarchical_datasets(
        self, 
        hierarchical_splits: Dict[str, Dict[str, 'HierarchicalDataset']], 
        output_dir: Optional[str] = None
    ):
        """
        Save hierarchical datasets and filtering statistics to disk.
        
        Args:
            hierarchical_splits: Dictionary of hierarchical datasets
            output_dir: Output directory (defaults to config)
        """
        
        if output_dir is None:
            output_dir = os.path.join(
                self.config["data"]["processed_dir"], 
                "hierarchical_format"
            )
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.logger.info(f"Saving hierarchical datasets to {output_dir}...")
        
        # Save datasets
        for split_type, split_dict in hierarchical_splits.items():
            for split_name, dataset in split_dict.items():
                output_path = os.path.join(
                    output_dir, 
                    f"{split_name}_{split_type}_hierarchical.pkl"
                )
                with open(output_path, 'wb') as f:
                    pickle.dump(dataset, f)
                self.logger.info(
                    f"Saved {split_name}_{split_type} hierarchical dataset to {output_path}"
                )
        
        # Save vocabulary and metadata
        vocab_path = os.path.join(output_dir, "hierarchical_vocabulary.pkl")
        vocab_data = {
            'num_exercises': self.num_exercises,
            'exercise_to_idx': self.exercise_to_idx,
            'idx_to_exercise': self.idx_to_exercise,
            'compilation_outcomes': self.compilation_outcomes,
            'execution_outcomes': self.execution_outcomes,
            'num_compilation_classes': self.num_compilation_classes,
            'num_execution_classes': self.num_execution_classes,
            'outcome_to_compilation_label': self.outcome_to_compilation_label,
            'outcome_to_execution_label': self.outcome_to_execution_label,
            'student_outcomes': self.student_outcomes,
            'system_outcomes': self.system_outcomes,
            'filtered_stats': self.filtered_stats
        }
        
        with open(vocab_path, 'wb') as f:
            pickle.dump(vocab_data, f)
        self.logger.info(f"Saved hierarchical vocabulary to {vocab_path}")
        
        # Save filtering report
        report_path = os.path.join(output_dir, "filtering_report.txt")
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("HIERARCHICAL FORMAT ADAPTER - FILTERING REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Total submissions processed: {self.filtered_stats['total_submissions']}\n")
            f.write(f"System errors filtered: {self.filtered_stats['filtered_system_errors']}\n")
            f.write(f"Valid submissions: {self.filtered_stats['valid_submissions']}\n\n")
            
            f.write("Valid outcome distribution:\n")
            for outcome, count in self.filtered_stats['outcome_distribution'].items():
                pct = 100 * count / self.filtered_stats['valid_submissions']
                f.write(f"  {outcome}: {count} ({pct:.2f}%)\n")
        
        self.logger.info(f"Saved filtering report to {report_path}")
    
    
    def run(self) -> Dict[str, Dict[str, 'HierarchicalDataset']]:
        """
        Execute complete hierarchical format preparation pipeline.
        
        Returns:
            Dictionary of hierarchical datasets
        """
        self.logger.info("=" * 80)
        self.logger.info("Starting Hierarchical Format Preparation Pipeline")
        self.logger.info("=" * 80)
        
        # Step 1: Run base preprocessing
        splits = self.preprocessor.run()
        
        # Step 2: Filter system errors and convert to hierarchical format
        hierarchical_splits = self.prepare_hierarchical_splits(splits)
        
        # Step 3: Save hierarchical datasets
        self.save_hierarchical_datasets(hierarchical_splits)
        
        self.logger.info("=" * 80)
        self.logger.info("Hierarchical Format Preparation Complete")
        self.logger.info("=" * 80)
        
        return hierarchical_splits


class HierarchicalDataset(Dataset):
    """
    PyTorch Dataset for Hierarchical Knowledge Tracing.
    """
    
    def __init__(
        self, 
        sequences: List[Dict], 
        num_exercises: int,
        num_compilation_classes: int,
        num_execution_classes: int,
        split_name: str = "",
        split_type: str = ""
    ):
        self.sequences = sequences
        self.num_exercises = num_exercises
        self.num_compilation_classes = num_compilation_classes
        self.num_execution_classes = num_execution_classes
        self.split_name = split_name
        self.split_type = split_type
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sequence = self.sequences[idx]
        
        exercise_ids = sequence['exercise_ids']
        compilation_labels = sequence['compilation_labels']
        execution_labels = sequence['execution_labels']
        seq_len = sequence['sequence_length']
        
        # Prepare temporal sequences
        exercise_ids_t = exercise_ids[:-1]
        target_exercises = exercise_ids[1:]
        
        compilation_labels_t = compilation_labels[:-1]
        compilation_labels_t1 = compilation_labels[1:]
        
        execution_labels_t = execution_labels[:-1]
        execution_labels_t1 = execution_labels[1:]
        
        return {
            'exercise_ids': torch.LongTensor(exercise_ids_t),
            'target_exercises': torch.LongTensor(target_exercises),
            'compilation_labels_t': torch.LongTensor(compilation_labels_t),
            'compilation_labels_t1': torch.LongTensor(compilation_labels_t1),
            'execution_labels_t': torch.LongTensor(execution_labels_t),
            'execution_labels_t1': torch.LongTensor(execution_labels_t1),
            'seq_len': seq_len - 1
        }
    
    def get_metadata(self) -> Dict:
        return {
            'num_sequences': len(self.sequences),
            'num_exercises': self.num_exercises,
            'num_compilation_classes': self.num_compilation_classes,
            'num_execution_classes': self.num_execution_classes,
            'split_name': self.split_name,
            'split_type': self.split_type,
            'avg_seq_length': np.mean([s['sequence_length'] for s in self.sequences]),
            'max_seq_length': max([s['sequence_length'] for s in self.sequences]),
            'min_seq_length': min([s['sequence_length'] for s in self.sequences])
        }


def hierarchical_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function for batching hierarchical sequences."""
    
    batch = [item for item in batch if item['seq_len'] > 0]
    
    if len(batch) == 0:
        return {
            'exercise_ids': torch.zeros(1, 1, dtype=torch.long),
            'target_exercises': torch.zeros(1, 1, dtype=torch.long),
            'compilation_labels_t': torch.zeros(1, 1, dtype=torch.long),
            'compilation_labels_t1': torch.zeros(1, 1, dtype=torch.long),
            'execution_labels_t': torch.zeros(1, 1, dtype=torch.long),
            'execution_labels_t1': torch.zeros(1, 1, dtype=torch.long),
            'seq_lens': torch.zeros(1, dtype=torch.long),
            'mask': torch.zeros(1, 1, dtype=torch.bool)
        }
    
    max_seq_len = max(item['seq_len'] for item in batch)
    batch_size = len(batch)
    
    padded_batch = {
        'exercise_ids': torch.zeros(batch_size, max_seq_len, dtype=torch.long),
        'target_exercises': torch.zeros(batch_size, max_seq_len, dtype=torch.long),
        'compilation_labels_t': torch.zeros(batch_size, max_seq_len, dtype=torch.long),
        'compilation_labels_t1': torch.zeros(batch_size, max_seq_len, dtype=torch.long),
        'execution_labels_t': torch.zeros(batch_size, max_seq_len, dtype=torch.long),
        'execution_labels_t1': torch.zeros(batch_size, max_seq_len, dtype=torch.long),
        'seq_lens': torch.zeros(batch_size, dtype=torch.long),
        'mask': torch.zeros(batch_size, max_seq_len, dtype=torch.bool)
    }
    
    for i, item in enumerate(batch):
        seq_len = item['seq_len']
        padded_batch['exercise_ids'][i, :seq_len] = item['exercise_ids']
        padded_batch['target_exercises'][i, :seq_len] = item['target_exercises']
        padded_batch['compilation_labels_t'][i, :seq_len] = item['compilation_labels_t']
        padded_batch['compilation_labels_t1'][i, :seq_len] = item['compilation_labels_t1']
        padded_batch['execution_labels_t'][i, :seq_len] = item['execution_labels_t']
        padded_batch['execution_labels_t1'][i, :seq_len] = item['execution_labels_t1']
        padded_batch['seq_lens'][i] = seq_len
        padded_batch['mask'][i, :seq_len] = True
    
    return padded_batch

def main():
    """Main execution function for hierarchical format preparation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Hierarchical Format Adapter")
    parser.add_argument(
        "--config", 
        type=str,
        default="experiments/configs/preprocessing_config.yaml",
        help="Path to preprocessing configuration file"
    )
    args = parser.parse_args()
    
    # Initialize and run adapter
    adapter = HierarchicalFormatAdapter(config_path=args.config)
    hierarchical_splits = adapter.run()
    
    # Print summary
    print("\n" + "=" * 80)
    print("HIERARCHICAL FORMAT PREPARATION SUMMARY")
    print("=" * 80)
    print(f"Number of unique exercises: {adapter.num_exercises}")
    print(f"Compilation classes: {adapter.num_compilation_classes}")
    print(f"Execution classes: {adapter.num_execution_classes}")
    print("\nDatasets created:")
    for split_type, split_dict in hierarchical_splits.items():
        print(f"\n  {split_type.upper()}:")
        for split_name, dataset in split_dict.items():
            metadata = dataset.get_metadata()
            print(
                f"    - {split_name}: {metadata['num_sequences']} sequences, "
                f"avg_len={metadata['avg_seq_length']:.1f}"
            )
    print("=" * 80)


if __name__ == "__main__":
    main()
