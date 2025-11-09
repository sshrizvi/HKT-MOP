"""
Base preprocessing pipeline for Knowledge Tracing on ACCoding dataset.
Implements shared preprocessing for all baseline models (DKT, DKVMN, SAKT, AKT, GKT)
and hierarchical model.

Author : Syed Shujaat Haider
"""

import os
import json
import hashlib
import logging
from datetime import datetime
from typing import Dict, Tuple, Optional

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import yaml
from logging_config import setup_logging


class BasePreprocessor:
    """
    Base preprocessing pipeline for Knowledge Tracing experiments.

    Handles:
    - Loading raw ACcoding submission data
    - Filtering by student ID range
    - Sequence construction with temporal ordering
    - Multi-outcome and binary outcome mapping
    - Train/validation/test splitting (standard + domain shift)
    - Metadata generation for reproducibility
    """

    def __init__(self, config_path: str = "experiments/configs/preprocessing_config.yaml"):
        """
        Initialize preprocessor with configuration.

        Args:
            config_path : Configuration for Preprocessing. (defaults to experiments/configs/preprocessing_config.yaml)
        """

        # Setup Logger
        self.logger = logging.getLogger('hkt-mop.preprocessing')
        setup_logging()

        self.config = self._load_config(config_path)
        self.config_hash = self._compute_config_hash()
        self.metadata = {
            "preprocessing_version": self.config.get("version", "v1.0.0"),
            "timestamp": datetime.now().isoformat(),
            "config_hash": self.config_hash
        }

        # Random Seeds
        self._set_seeds(self.config.get("random_seed", 42))

        # Log Initialization
        self.logger.info(
            f"BasePreprocessor initialized with config hash: {self.config_hash}")

    def _load_config(self, config_path: str) -> Dict:
        """Load preprocessing configuration from YAML file."""

        if not os.path.exists(config_path):
            self.logger.warning(f"Config file not found at {config_path}.")
            raise Exception(
                "Cannot initialize BasePreprocessor without the config.")

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        self.logger.info(f"Loaded configuration from {config_path}")

        return config

    def _compute_config_hash(self) -> str:
        """Compute hash of configuration for versioning."""

        config_str = json.dumps(self.config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]

    def _set_seeds(self, seed: int):
        """Set random seeds for reproducibility."""

        np.random.seed(seed)
        import random
        random.seed(seed)
        self.logger.info(f"Random seeds set to {seed}.")

    def load_raw_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load raw CSV files from data/raw directory.

        Returns:
            Tuple of (submissions_df, problems_df, problem_tags_df, tags_df)
        """

        self.logger.info("Loading raw data files...")

        raw_dir = self.config["data"]["raw_dir"]

        # Load submissions (primary table for KT)
        submissions_path = os.path.join(
            raw_dir, self.config["data"]["submissions_file"])
        submissions_df = pd.read_csv(submissions_path)
        self.logger.info(
            f"Loaded {len(submissions_df)} submissions from {submissions_path}")

        # Load problems (for difficulty and metadata)
        problems_path = os.path.join(
            raw_dir, self.config["data"]["problems_file"])
        problems_df = pd.read_csv(problems_path) if os.path.exists(
            problems_path) else None

        # Load problem_tags (for knowledge components)
        problem_tags_path = os.path.join(
            raw_dir, self.config["data"]["problem_tags_file"])
        problem_tags_df = pd.read_csv(problem_tags_path) if os.path.exists(
            problem_tags_path) else None

        # Load tags (knowledge points)
        tags_path = os.path.join(raw_dir, self.config["data"]["tags_file"])
        tags_df = pd.read_csv(tags_path) if os.path.exists(tags_path) else None

        self.metadata["input_data"] = {
            "submissions_file": submissions_path,
            "total_submissions": len(submissions_df),
            "unique_students": int(submissions_df['creator_id'].nunique()) if 'creator_id' in submissions_df else None,
            "unique_problems": int(submissions_df['problem_id'].nunique()) if 'problem_id' in submissions_df else None
        }

        return submissions_df, problems_df, problem_tags_df, tags_df

    def filter_by_id_range(self, submissions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter submissions by student ID range specified in config.

        Args:
            submissions_df: Raw submissions dataframe.

        Returns:
            Filtered submissions dataframe.
        """

        id_min = self.config["filtering"]["submission_id_min"]
        id_max = self.config["filtering"]["submission_id_max"]

        # Filter Submissions by ID
        filtered_df = submissions_df[
            (submissions_df['id'] >= id_min) &
            (submissions_df['id'] <= id_max)
        ].copy()

        self.logger.info(f"Filtered submissions by ID range [{id_min}, {id_max}]: "
                         f"{len(filtered_df)} submissions from {filtered_df['creator_id'].nunique()} students")

        self.metadata["filtering"] = {
            "id_range": [id_min, id_max],
            "submissions_after_filter": len(filtered_df),
            "students_after_filter": int(filtered_df['creator_id'].nunique())
        }

        return filtered_df

    def construct_sequences(self, submissions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Construct chronological student-problem interaction sequences.

        Args:
            submissions_df: Filtered submissions dataframe

        Returns:
            DataFrame with sequences grouped by student.
        """
        self.logger.info("Constructing student sequences...")

        # Critical: Use submission 'id' for temporal ordering
        submissions_df = submissions_df.sort_values(['creator_id', 'id'])

        # Filter students with minimum submission threshold
        min_subs = self.config["filtering"]["min_submissions_per_student"]
        student_counts = submissions_df.groupby('creator_id').size()
        valid_students = student_counts[student_counts >= min_subs].index
        submissions_df = submissions_df[submissions_df['creator_id'].isin(
            valid_students)].copy()

        self.logger.info(
            f"Filtered to {len(valid_students)} students with >= {min_subs} submissions")

        # Apply sequence length limit
        max_seq_len = self.config["filtering"]["max_sequence_length"]
        submissions_df['seq_position'] = submissions_df.groupby(
            'creator_id').cumcount()
        submissions_df = submissions_df[submissions_df['seq_position'] < max_seq_len]

        # Add binary correctness column for baseline compatibility
        submissions_df['correct'] = (
            submissions_df['result'] == 'AC').astype(int)

        # Add hierarchical outcome groups for multi-outcome models
        submissions_df['outcome_group'] = submissions_df['result'].apply(
            self._map_outcome_group)

        # Create multiclass mappings
        self._create_outcome_mappings(submissions_df)

        # Add numeric multiclass labels
        submissions_df['multiclass_label'] = submissions_df['result'].map(
            self.outcome_to_idx)
        self.logger.info(
            f"Created multiclass labels: {self.num_outcomes} unique outcomes")

        # Add numeric outcome group labels
        submissions_df['outcome_group_label'] = submissions_df['outcome_group'].map(
            self.outcome_group_to_idx)
        self.logger.info(
            f"Created outcome group labels: {self.num_outcome_groups} groups")

        self.logger.info(
            f"Constructed sequences with max length {max_seq_len}")

        self.metadata["sequences"] = {
            "total_students": int(len(valid_students)),
            "total_sequences": len(submissions_df),
            "avg_sequence_length": float(submissions_df.groupby('creator_id').size().mean()),
            "max_sequence_length": max_seq_len,
            # NEW: Add multiclass metadata
            "num_outcomes": self.num_outcomes,
            "num_outcome_groups": self.num_outcome_groups,
            "outcome_distribution": submissions_df['result'].value_counts().to_dict(),
            "outcome_group_distribution": submissions_df['outcome_group'].value_counts().to_dict()
        }

        return submissions_df

    def _create_outcome_mappings(self, submissions_df: pd.DataFrame):
        """
        Create mappings for multiclass outcome labels.

        Args:
            submissions_df: DataFrame with 'result' and 'outcome_group' columns
        """
        # Create outcome to index mapping (all possible results)
        unique_outcomes = sorted(submissions_df['result'].unique())
        self.outcome_to_idx = {outcome: idx for idx,
                               outcome in enumerate(unique_outcomes)}
        self.idx_to_outcome = {idx: outcome for outcome,
                               idx in self.outcome_to_idx.items()}
        self.num_outcomes = len(unique_outcomes)

        # Create outcome group to index mapping
        unique_groups = sorted(submissions_df['outcome_group'].unique())
        self.outcome_group_to_idx = {
            group: idx for idx, group in enumerate(unique_groups)}
        self.idx_to_outcome_group = {
            idx: group for group, idx in self.outcome_group_to_idx.items()}
        self.num_outcome_groups = len(unique_groups)

        self.logger.info(f"Outcome mappings created:")
        self.logger.info(f"  - Outcomes: {self.outcome_to_idx}")
        self.logger.info(f"  - Outcome groups: {self.outcome_group_to_idx}")

    def save_processed_data(self, splits: Dict[str, pd.DataFrame],
                            output_dir: Optional[str] = None):
        """
        Save preprocessed data to disk.

        Args:
            splits: Dictionary of train/val/test dataframes
            output_dir: Output directory (defaults to config)
        """
        if output_dir is None:
            output_dir = self.config["data"]["processed_dir"]

        os.makedirs(output_dir, exist_ok=True)
        self.logger.info(f"Saving processed data to {output_dir}...")

        # Save each split
        for split_type, split in splits.items():
            for split_name, split_df in split.items():
                output_path = os.path.join(
                    output_dir, f"{split_name + '_' + split_type}.pkl")
                split_df.to_pickle(output_path)
                self.logger.info(
                    f"Saved {split_name + '_' + split_type} split to {output_path}")

        # Save outcome mappings for multiclass models
        mappings = {
            'outcome_to_idx': self.outcome_to_idx,
            'idx_to_outcome': self.idx_to_outcome,
            'num_outcomes': self.num_outcomes,
            'outcome_group_to_idx': self.outcome_group_to_idx,
            'idx_to_outcome_group': self.idx_to_outcome_group,
            'num_outcome_groups': self.num_outcome_groups
        }

        mappings_path = os.path.join(output_dir, "outcome_mappings.json")
        with open(mappings_path, 'w') as f:
            json.dump(mappings, f, indent=2)
        self.logger.info(f"Saved outcome mappings to {mappings_path}")

        # Save metadata as JSON
        metadata_path = os.path.join(output_dir, "preprocessing_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
        self.logger.info(f"Saved metadata to {metadata_path}")

        # Save statistics as CSV
        stats_rows = []
        for split_type, split_type_stats in self.metadata["statistics"].items():
            for split_name, split_stats in split_type_stats.items():
                row = {"split": f'{split_name + '_' + split_type}'}
                row.update(split_stats)
                stats_rows.append(row)

        stats_df = pd.DataFrame(stats_rows)
        stats_path = os.path.join(output_dir, "preprocessing_statistics.csv")
        stats_df.to_csv(stats_path, index=False)
        self.logger.info(f"Saved statistics to {stats_path}")

    def _map_outcome_group(self, result: str) -> str:
        """Map result to hierarchical outcome group."""
        
        outcome_groups = self.config["outcomes"]["multi_outcome_groups"]

        for group_name, results in outcome_groups.items():
            if result in results:
                return group_name

        return "other"

    def split_data(self, sequences_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Split data into train/validation/test sets.

        Supports:
        - Standard split: Random student-level split
        - Domain shift split: Train on daily, test on contest

        Args:
            sequences_df: Complete sequences dataframe

        Returns:
            Dictionary with 'train', 'val', 'test' dataframes
        """
        self.logger.info("Splitting data into train/val/test sets...")

        split_config = self.config["splitting"]
        train_ratio = split_config["train_ratio"]
        val_ratio = split_config["val_ratio"]
        test_ratio = split_config["test_ratio"]

        # Get unique students
        unique_students = sequences_df['creator_id'].unique()

        # First split: train vs (val + test)
        train_students, temp_students = train_test_split(
            unique_students,
            test_size=(val_ratio + test_ratio),
            random_state=self.config["random_seed"]
        )

        # Second split: val vs test
        val_students, test_students = train_test_split(
            temp_students,
            test_size=test_ratio / (val_ratio + test_ratio),
            random_state=self.config["random_seed"]
        )

        # Create splits
        train_df = sequences_df[sequences_df['creator_id'].isin(
            train_students)].copy()
        val_df = sequences_df[sequences_df['creator_id'].isin(
            val_students)].copy()
        test_df = sequences_df[sequences_df['creator_id'].isin(
            test_students)].copy()

        splits = {}
        splits['standard'] = {
            'train': train_df,
            'val': val_df,
            'test': test_df
        }

        self.logger.info(
            f"Standard split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

        # Log split statistics
        self.metadata["splits"] = {}
        self.metadata["splits"]["standard"] = {
            "train_sequences": len(train_df),
            "val_sequences": len(val_df),
            "test_sequences": len(test_df),
            "train_students": int(len(train_students)),
            "val_students": int(len(val_students)),
            "test_students": int(len(test_students))
        }

        # Domain shift split (if enabled)
        if self.config["domain_shift"]["enabled"]:
            splits['domain_shift'] = self._create_domain_shift_split(sequences_df)

        return splits

    def _create_domain_shift_split(self, sequences_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create domain shift test set: contest submissions from train students.

        Args:
            sequences_df: Complete sequences dataframe

        Returns:
            Domain shift test dataframe
        """

        self.logger.info("Creating Domain Shift Split...")

        # Load Split Config
        split_config = self.config["splitting"]
        train_ratio = split_config["train_ratio"]
        val_ratio = split_config["val_ratio"]
        test_ratio = split_config["test_ratio"]

        # Filter Daily Submissions
        daily_unique_students = sequences_df[sequences_df['contest_id'].isna()]['creator_id'].unique()
        contests_unique_students = sequences_df[sequences_df['contest_id'].notna()]['creator_id'].unique()

        # Train & Validation Splits
        train_students, val_students = train_test_split(
            daily_unique_students,
            test_size=val_ratio / (train_ratio + val_ratio),
            random_state=self.config['random_seed']
        )

        # Test Split
        test_students, residual_students = train_test_split(
            contests_unique_students,
            train_size=len(val_students) / len(contests_unique_students),
            random_state=self.config['random_seed']
        )

        # Create Splits
        train_df = sequences_df[sequences_df['creator_id'].isin(
            train_students)].copy()
        val_df = sequences_df[sequences_df['creator_id'].isin(
            val_students)].copy()
        test_df = sequences_df[sequences_df['creator_id'].isin(
            test_students)].copy()

        splits = {
            'train': train_df,
            'val': val_df,
            'test': test_df
        }

        # Log Split Statistics
        self.metadata["splits"]["domain"] = {
            "train_sequences": len(splits['train']),
            "val_sequences": len(splits['val']),
            "test_sequences": len(splits['test']),
            "train_students": int(len(train_students)),
            "val_students": int(len(val_students)),
            "test_students": int(len(test_students))
        }

        return splits

    def compute_statistics(self, splits: Dict[str, pd.DataFrame]) -> Dict:
        """
        Compute dataset statistics for all splits.

        Args:
            splits: Dictionary of train/val/test dataframes

        Returns:
            Dictionary of statistics
        """
        self.logger.info("Computing dataset statistics...")

        stats = {}

        for split_type, split in splits.items():
            stats[split_type] = {}
            for split_name, split_df in split.items():
                # Outcome distribution
                outcome_dist = split_df['result'].value_counts(
                    normalize=True).to_dict()

                # Binary outcome distribution
                binary_dist = split_df['correct'].value_counts(
                    normalize=True).to_dict()

                # Hierarchical outcome group distribution
                group_dist = split_df['outcome_group'].value_counts(
                    normalize=True).to_dict()

                # Sequence length statistics
                seq_lengths = split_df.groupby('creator_id').size()

                stats[split_type][split_name] = {
                    "total_submissions": len(split_df),
                    "unique_students": int(split_df['creator_id'].nunique()),
                    "unique_problems": int(split_df['problem_id'].nunique()),
                    "outcome_distribution": {k: float(v) for k, v in outcome_dist.items()},
                    "binary_distribution": {int(k): float(v) for k, v in binary_dist.items()},
                    "outcome_group_distribution": {k: float(v) for k, v in group_dist.items()},
                    "avg_sequence_length": float(seq_lengths.mean()),
                    "median_sequence_length": float(seq_lengths.median()),
                    "min_sequence_length": int(seq_lengths.min()),
                    "max_sequence_length": int(seq_lengths.max())
                }

        self.metadata["statistics"] = stats

        return stats

    def save_processed_data(self, splits: Dict[str, pd.DataFrame],
                            output_dir: Optional[str] = None):
        """
        Save preprocessed data to disk.

        Args:
            splits: Dictionary of train/val/test dataframes
            output_dir: Output directory (defaults to config)
        """

        if output_dir is None:
            output_dir = self.config["data"]["processed_dir"]

        os.makedirs(output_dir, exist_ok=True)

        self.logger.info(f"Saving processed data to {output_dir}...")

        # Save each split as pickle (preserves dtypes)
        for split_type, split in splits.items():
            for split_name, split_df in split.items():
                output_path = os.path.join(
                    output_dir, f"{split_name + '_' + split_type}.pkl")
                split_df.to_pickle(output_path)
                self.logger.info(
                    f"Saved {split_name + '_' + split_type} split to {output_path}")

        # Save metadata as JSON
        metadata_path = os.path.join(output_dir, "preprocessing_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
        self.logger.info(f"Saved metadata to {metadata_path}")

        # Save statistics as CSV
        stats_rows = []
        for split_type, split_type_stats in self.metadata["statistics"].items():
            for split_name, split_stats in split_type_stats.items():
                row = {"split": f'{split_name + '_' + split_type}'}
                row.update(split_stats)
                stats_rows.append(row)

        stats_df = pd.DataFrame(stats_rows)
        stats_path = os.path.join(output_dir, "preprocessing_statistics.csv")
        stats_df.to_csv(stats_path, index=False)
        self.logger.info(f"Saved statistics to {stats_path}")

    def run(self) -> Dict[str, pd.DataFrame]:
        """
        Execute complete preprocessing pipeline.

        Returns:
            Dictionary of train/val/test splits
        """
        self.logger.info("=" * 80)
        self.logger.info("Starting Knowledge Tracing Preprocessing Pipeline")
        self.logger.info("=" * 80)

        # Step 1: Load raw data
        submissions_df, problems_df, problem_tags_df, tags_df = self.load_raw_data()

        # Step 2: Filter by ID range
        filtered_df = self.filter_by_id_range(submissions_df)

        # Step 3: Construct sequences
        sequences_df = self.construct_sequences(filtered_df)

        # Step 4: Split data
        splits = self.split_data(sequences_df)

        # Step 5: Compute statistics
        stats = self.compute_statistics(splits)

        # Step 6: Save processed data
        self.save_processed_data(splits)

        self.logger.info("=" * 80)
        self.logger.info("Preprocessing Pipeline Complete")
        self.logger.info(f"Config Hash: {self.config_hash}")
        self.logger.info(
            f"Output Directory: {self.config['data']['processed_dir']}")
        self.logger.info("=" * 80)

        return splits


def main():
    """Main execution function."""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(
        description="Knowledge Tracing Base Preprocessor")
    parser.add_argument("--config", type=str, default="experiments/configs/preprocessing_config.yaml",
                        help="Path to preprocessing configuration file")
    args = parser.parse_args()

    # Initialize and run preprocessor
    preprocessor = BasePreprocessor(config_path=args.config)
    splits = preprocessor.run()

    # Print summary
    print("\n" + "=" * 80)
    print("PREPROCESSING SUMMARY")
    print("=" * 80)
    print(f"Configuration Hash: {preprocessor.config_hash}")
    print(f"\nData Splits:")
    for split_type, split in splits.items():
        for split_name, split_df in split.items():
            print(f"  - {split_name + '_' + split_type}: {len(split_df)} sequences, "
                  f"{split_df['creator_id'].nunique()} students")
    print("=" * 80)


if __name__ == "__main__":
    main()
