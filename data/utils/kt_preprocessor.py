"""
Knowledge Tracing Preprocessor for ACcoding Dataset

This module provides a generic preprocessing pipeline for knowledge tracing models
on the ACcoding dataset. It supports binary, multi-class, and hierarchical outcome
prediction with OJ2019 subset extraction.

Author: Syed Shujaat Haider
Date: October 25, 2025
Project: Hierarchical Knowledge Tracing (HKT-MOP)
"""

import logging
import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from collections import defaultdict, Counter
from datetime import datetime
import warnings

# Suppress pandas warnings
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)


class KTPreProcessor:
    """
    Generic Knowledge Tracing Preprocessor for ACcoding Dataset.

    This preprocessor creates model-agnostic sequence data that can be used
    by any knowledge tracing model (DKT, DKVMN, GKT, HKT, etc.) through
    model-specific adapters.

    Features:
    - Binary classification (AC vs non-AC)
    - Multi-class classification (8 outcome types)
    - Hierarchical classification (compilation vs execution)
    - OJ2019 subset extraction (131,612 interactions, 3,650 students)
    - Domain shift splits (daily exercises vs contests)
    - Multiple output formats (pickle, JSON, CSV)

    Attributes:
        data_dir (Path): Directory containing raw CSV files
        output_dir (Path): Directory for preprocessed outputs
        min_seq_length (int): Minimum sequence length (default: 3)
        max_seq_length (int): Maximum sequence length (default: 200)
        logger (logging.Logger): Module-based logger
    """

    # Outcome type mappings
    BINARY_MAPPING = {
        'AC': 1
    }

    MULTICLASS_MAPPING = {
        'AC': 0,   # Accepted
        'WA': 1,   # Wrong Answer
        'CE': 2,   # Compilation Error
        'TLE': 3,  # Time Limit Exceeded
        'RE': 4,   # Runtime Error
        'PE': 5,   # Presentation Error
        'MLE': 6,  # Memory Limit Exceeded
        'OE': 7    # Other Error
    }

    COMPILATION_ERRORS = {'CE', 'REG'}  # Compilation phase errors

    def __init__(
        self,
        data_dir: str = 'data/raw',
        output_dir: str = 'data/preprocessed',
        min_seq_length: int = 3,
        max_seq_length: int = 200,
        logger_name: str = 'hkt-mop.data.utils'
    ):
        """
        Initialize KTPreProcessor.

        Args:
            data_dir: Path to raw CSV files
            output_dir: Path for preprocessed output
            min_seq_length: Minimum sequence length (students with fewer dropped)
            max_seq_length: Maximum sequence length (truncate longer sequences)
            logger_name: Module-based logger name
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.min_seq_length = min_seq_length
        self.max_seq_length = max_seq_length

        # Setup logger
        self.logger = logging.getLogger(logger_name)

        # Create output directories
        self._create_output_dirs()

        # Data containers
        self.submissions: Optional[pd.DataFrame] = None
        self.problems: Optional[pd.DataFrame] = None
        self.tags: Optional[pd.DataFrame] = None
        self.problemtags: Optional[pd.DataFrame] = None

        # Processed sequences
        self.sequences: Optional[List[Dict]] = None
        self.train_data: Optional[List[Dict]] = None
        self.val_data: Optional[List[Dict]] = None
        self.test_data: Optional[List[Dict]] = None

        # Statistics tracking
        self.stats: Dict[str, Any] = defaultdict(dict)

        # Metadata
        self.metadata: Dict[str, Any] = {
            'preprocessor_version': '1.0.0',
            'created_at': datetime.now().isoformat(),
            'min_seq_length': min_seq_length,
            'max_seq_length': max_seq_length
        }

        self.logger.info(f"KTPreProcessor initialized")
        self.logger.info(f"  Data directory: {data_dir}")
        self.logger.info(f"  Output directory: {output_dir}")
        self.logger.info(
            f"  Sequence length constraints: min={min_seq_length}, max={max_seq_length}")

    def _create_output_dirs(self) -> None:
        """Create output directory structure."""
        subdirs = [
            'binary/standard',
            'binary/domain_shift',
            'multiclass/standard',
            'multiclass/domain_shift',
            'hierarchical/standard',
            'hierarchical/domain_shift',
            'statistics'
        ]

        for subdir in subdirs:
            (self.output_dir / subdir).mkdir(parents=True, exist_ok=True)

        self.logger.debug(
            f"Created output directories under {self.output_dir}")

    # ==================== DATA LOADING ====================

    def load_raw_data(
        self,
        submissions_file: str = 'submissions.csv',
        problems_file: str = 'problems.csv',
        tags_file: str = 'tags.csv',
        problemtags_file: str = 'problemtags.csv',
        extract_oj2019: bool = True,
        oj2019_method: str = 'auto',
        year_id_range: Optional[Dict[str, int]] = None
    ) -> None:
        """
        Load raw CSV files from ACcoding dataset.

        Args:
            submissions_file: Name of submissions CSV
            problems_file: Name of problems CSV
            tags_file: Name of tags CSV
            problemtags_file: Name of problemtags CSV
            extract_oj2019: If True, extract OJ2019 subset (2019 daily exercises)
            oj2019_method: Method for year extraction ('auto', 'id_range', 'all_daily')
            year_id_range: Optional dict with 'min' and 'max' submission IDs for 2019
        """
        self.logger.info("=" * 60)
        self.logger.info("LOADING RAW DATA")
        self.logger.info("=" * 60)

        try:
            # Load tables
            self.logger.info(f"Loading {submissions_file}...")
            self.submissions = pd.read_csv(self.data_dir / submissions_file)

            self.logger.info(f"Loading {problems_file}...")
            self.problems = pd.read_csv(self.data_dir / problems_file)

            self.logger.info(f"Loading {tags_file}...")
            self.tags = pd.read_csv(self.data_dir / tags_file)

            self.logger.info(f"Loading {problemtags_file}...")
            self.problemtags = pd.read_csv(self.data_dir / problemtags_file)

            self.logger.info(f"Loaded {len(self.submissions):,} submissions")
            self.logger.info(f"Loaded {len(self.problems):,} problems")
            self.logger.info(f"Loaded {len(self.tags):,} tags")
            self.logger.info(f"Loaded {len(self.problemtags):,} problem-tag mappings")

            # Store raw statistics
            self.stats['raw'] = {
                'total_submissions': len(self.submissions),
                'total_problems': len(self.problems),
                'total_tags': len(self.tags),
                'total_problemtag_mappings': len(self.problemtags)
            }

            # Extract OJ2019 subset if requested
            if extract_oj2019:
                self._extract_oj2019_subset(
                    method=oj2019_method, year_id_range=year_id_range)

            # Data cleaning
            self._clean_data()

            # Log detailed statistics
            self._log_data_statistics()

            self.logger.info("Data loading complete")

        except FileNotFoundError as e:
            self.logger.error(f"File not found: {e}")
            self.logger.error(f"Expected files in: {self.data_dir}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise

    def _extract_oj2019_subset(
        self,
        method: str = 'auto',
        year_id_range: Optional[Dict[str, int]] = None
    ) -> None:
        """
        Extract OJ2019 subset: daily exercises from year 2019.
        Target: 131,612 interactions from 3,650 students (as per ACcoding paper).

        Args:
            method: Extraction method
                - 'auto': Use all daily exercises (no year filtering)
                - 'id_range': Use provided ID range for 2019
                - 'all_daily': Explicitly use all daily (same as auto)
            year_id_range: Dict with 'min' and 'max' submission IDs for 2019
        """
        self.logger.info("-" * 60)
        self.logger.info("EXTRACTING OJ2019 SUBSET")
        self.logger.info("-" * 60)

        original_count = len(self.submissions)
        original_users = self.submissions['creatorid'].nunique()

        # Step 1: Filter daily exercises only (contestid IS NULL)
        self.logger.info("Filtering daily exercises (contestid IS NULL)...")
        self.submissions = self.submissions[self.submissions['contestid'].isna()].copy(
        )

        daily_count = len(self.submissions)
        daily_users = self.submissions['creatorid'].nunique()

        self.logger.info(
            f"  Daily filter: {original_count:,} → {daily_count:,} submissions")
        self.logger.info(
            f"  Users: {original_users:,} → {daily_users:,} students")

        # Step 2: Filter by year 2019 (if method specified)
        if method == 'id_range' and year_id_range:
            self.logger.info(f"Applying year 2019 ID range filter...")
            min_id = year_id_range['min']
            max_id = year_id_range['max']

            self.submissions = self.submissions[
                (self.submissions['id'] >= min_id) &
                (self.submissions['id'] <= max_id)
            ].copy()

            year_count = len(self.submissions)
            year_users = self.submissions['creatorid'].nunique()

            self.logger.info(f"  Year filter (id {min_id} to {max_id}):")
            self.logger.info(
                f"    {daily_count:,} → {year_count:,} submissions")
            self.logger.info(f"    {daily_users:,} → {year_users:,} students")

        elif method == 'auto' or method == 'all_daily':
            self.logger.warning(
                "Year filtering not applied (schema lacks timestamp)")
            self.logger.warning("  Using ALL daily exercises across all years")
            self.logger.info(
                "  To filter by year 2019, use method='id_range' with year_id_range parameter")

        final_count = len(self.submissions)
        final_users = self.submissions['creatorid'].nunique()

        # Store OJ2019 statistics
        self.stats['oj2019'] = {
            'original_submissions': original_count,
            'daily_submissions': final_count,
            'original_users': original_users,
            'daily_users': final_users,
            'extraction_method': method
        }

        self.logger.info(f"OJ2019 extraction complete:")
        self.logger.info(
            f"    {final_count:,} submissions from {final_users:,} students")

        # Validate against paper statistics (131,612 interactions, 3,650 students)
        expected_count = 131612
        expected_users = 3650

        if abs(final_count - expected_count) < 5000:  # Within 5k variance
            self.logger.info(
                f"  Submission count close to paper statistics (~{expected_count:,})")
        else:
            self.logger.warning(
                f"  Submission count differs from paper: expected ~{expected_count:,}, got {final_count:,}")

        if abs(final_users - expected_users) < 500:  # Within 500 user variance
            self.logger.info(
                f"  User count close to paper statistics (~{expected_users:,})")
        else:
            self.logger.warning(
                f"  User count differs from paper: expected ~{expected_users:,}, got {final_users:,}")

    def _clean_data(self) -> None:
        """Clean data: drop NULLs, handle missing values, sort by time."""
        self.logger.info("-" * 60)
        self.logger.info("CLEANING DATA")
        self.logger.info("-" * 60)

        # Drop submissions with NULL creatorid or problemid
        before_clean = len(self.submissions)
        self.submissions = self.submissions.dropna(
            subset=['creatorid', 'problemid']).copy()
        after_clean = len(self.submissions)
        dropped = before_clean - after_clean

        if dropped > 0:
            self.logger.warning(
                f"Dropped {dropped:,} submissions with NULL creatorid/problemid")
        else:
            self.logger.info("No NULL values in creatorid/problemid")

        # Convert IDs to integers
        self.submissions['creatorid'] = self.submissions['creatorid'].astype(
            int)
        self.submissions['problemid'] = self.submissions['problemid'].astype(
            int)

        # Sort by creatorid and submission id (temporal ordering)
        self.logger.info(
            "Sorting submissions by (creatorid, id) for temporal ordering...")
        self.submissions = self.submissions.sort_values(
            ['creatorid', 'id']).reset_index(drop=True)

        # Store cleaning statistics
        self.stats['cleaning'] = {
            'submissions_before': before_clean,
            'submissions_after': after_clean,
            'dropped': dropped
        }

        self.logger.info(
            f"Data cleaning complete: {after_clean:,} submissions retained")

    def _log_data_statistics(self) -> None:
        """Log detailed statistics about the loaded data."""
        self.logger.info("-" * 60)
        self.logger.info("DATA STATISTICS")
        self.logger.info("-" * 60)

        # Submission result distribution
        result_counts = self.submissions['result'].value_counts()
        self.logger.info("Submission result distribution:")
        for result, count in result_counts.items():
            pct = (count / len(self.submissions)) * 100
            self.logger.info(f"  {result}: {count:,} ({pct:.2f}%)")

        # Store in stats
        self.stats['result_distribution'] = result_counts.to_dict()

        # User activity statistics
        user_submission_counts = self.submissions.groupby('creatorid').size()
        self.logger.info(f"User activity:")
        self.logger.info(f"  Total users: {len(user_submission_counts):,}")
        self.logger.info(
            f"  Mean submissions per user: {user_submission_counts.mean():.2f}")
        self.logger.info(
            f"  Median submissions per user: {user_submission_counts.median():.0f}")
        self.logger.info(
            f"  Max submissions per user: {user_submission_counts.max():,}")

        # Store in stats
        self.stats['user_activity'] = {
            'total_users': len(user_submission_counts),
            'mean_submissions': float(user_submission_counts.mean()),
            'median_submissions': float(user_submission_counts.median()),
            'max_submissions': int(user_submission_counts.max()),
            'min_submissions': int(user_submission_counts.min())
        }

        # Problem statistics
        unique_problems = self.submissions['problemid'].nunique()
        problem_attempt_counts = self.submissions.groupby('problemid').size()
        self.logger.info(f"Problem statistics:")
        self.logger.info(f"  Unique problems attempted: {unique_problems:,}")
        self.logger.info(
            f"  Mean attempts per problem: {problem_attempt_counts.mean():.2f}")
        self.logger.info(
            f"  Median attempts per problem: {problem_attempt_counts.median():.0f}")

        # Store in stats
        self.stats['problem_stats'] = {
            'unique_problems': unique_problems,
            'mean_attempts': float(problem_attempt_counts.mean()),
            'median_attempts': float(problem_attempt_counts.median())
        }

    # ==================== SEQUENCE CONSTRUCTION ====================

    def build_student_sequences(
        self,
        include_features: bool = True,
        include_all_labels: bool = True
    ) -> None:
        """
        Build temporal sequences for each student.

        This creates model-agnostic sequences with all possible labels
        (binary, multi-class, hierarchical) so any KT model can use them.

        Args:
            include_features: Include problem features (tags, difficulty)
            include_all_labels: Include binary, multi-class, and hierarchical labels
        """
        
        self.logger.info("=" * 60)
        self.logger.info("BUILDING STUDENT SEQUENCES")
        self.logger.info("=" * 60)

        sequences = []
        student_ids = self.submissions['creatorid'].unique()

        self.logger.info(f"Processing {len(student_ids):,} students...")

        dropped_min_length = 0
        truncated_count = 0

        for idx, student_id in enumerate(student_ids):
            if (idx + 1) % 500 == 0:
                self.logger.info(
                    f"  Processed {idx + 1:,}/{len(student_ids):,} students...")

            # Get all submissions for this student
            student_subs = self.submissions[self.submissions['creatorid'] == student_id]

            # Build sequence for this student
            sequence = self._build_single_sequence(
                student_subs,
                include_features=include_features,
                include_all_labels=include_all_labels
            )

            # Apply length constraints
            if len(sequence) < self.min_seq_length:
                dropped_min_length += 1
                continue  # Skip students with too few interactions

            if len(sequence) > self.max_seq_length:
                # Truncate to max length
                sequence = sequence[:self.max_seq_length]
                truncated_count += 1

            sequences.append({
                'user_id': int(student_id),
                'sequence': sequence,
                'sequence_length': len(sequence)
            })

        self.sequences = sequences

        # Log statistics
        total_students = len(student_ids)
        kept_students = len(sequences)

        self.logger.info("-" * 60)
        self.logger.info(f"Built {kept_students:,} student sequences")
        self.logger.warning(
            f"  Dropped {dropped_min_length:,} students (< {self.min_seq_length} interactions)")
        if truncated_count > 0:
            self.logger.info(
                f"  Truncated {truncated_count:,} sequences (> {self.max_seq_length} interactions)")

        # Sequence length distribution
        seq_lengths = [s['sequence_length'] for s in sequences]
        self.logger.info(f"Sequence length distribution:")
        self.logger.info(f"  Mean: {np.mean(seq_lengths):.2f}")
        self.logger.info(f"  Median: {np.median(seq_lengths):.0f}")
        self.logger.info(f"  Min: {np.min(seq_lengths)}")
        self.logger.info(f"  Max: {np.max(seq_lengths)}")
        self.logger.info(f"  Std: {np.std(seq_lengths):.2f}")

        # Store sequence statistics
        self.stats['sequences'] = {
            'total_students_raw': total_students,
            'students_kept': kept_students,
            'students_dropped': dropped_min_length,
            'sequences_truncated': truncated_count,
            'mean_length': float(np.mean(seq_lengths)),
            'median_length': float(np.median(seq_lengths)),
            'min_length': int(np.min(seq_lengths)),
            'max_length': int(np.max(seq_lengths)),
            'std_length': float(np.std(seq_lengths))
        }

        # Label distribution in sequences
        self._log_label_distributions()

    def _build_single_sequence(
        self,
        student_subs: pd.DataFrame,
        include_features: bool,
        include_all_labels: bool
    ) -> List[Dict]:
        """
        Build sequence for a single student.

        Args:
            student_subs: DataFrame of submissions for one student
            include_features: Include problem features (tags, difficulty)
            include_all_labels: Include all label types

        Returns:
            List of interaction dictionaries
        """
        sequence = []

        for idx, row in student_subs.iterrows():
            problem_id = int(row['problemid'])
            result = row['result']

            # Build interaction dictionary
            interaction = {
                'step': len(sequence),
                'problem_id': problem_id,
                'result': result
            }

            # Add all label types if requested
            if include_all_labels:
                interaction['label_binary'] = self._convert_to_binary(result)
                interaction['label_multiclass'] = self._convert_to_multiclass(
                    result)
                interaction['label_hierarchical'] = self._create_hierarchical_label(
                    result)

            # Add problem features if requested
            if include_features:
                interaction['tags'] = self._get_problem_tags(problem_id)
                interaction['difficulty'] = self._get_problem_difficulty(
                    problem_id)
                interaction['is_contest'] = not pd.isna(row.get('contestid'))

                # Optional: Add timecost and memorycost if available
                if 'timecost' in row:
                    interaction['timecost'] = int(
                        row['timecost']) if pd.notna(row['timecost']) else None
                if 'memorycost' in row:
                    interaction['memorycost'] = int(
                        row['memorycost']) if pd.notna(row['memorycost']) else None

            sequence.append(interaction)

        return sequence

    def _log_label_distributions(self) -> None:
        """Log distribution of labels across all sequences."""
        self.logger.info("-" * 60)
        self.logger.info("LABEL DISTRIBUTIONS IN SEQUENCES")
        self.logger.info("-" * 60)

        # Flatten all interactions
        all_interactions = []
        for seq in self.sequences:
            all_interactions.extend(seq['sequence'])

        total_interactions = len(all_interactions)

        # Binary distribution
        binary_counts = Counter(i['label_binary'] for i in all_interactions)
        self.logger.info("Binary labels (AC=1, Non-AC=0):")
        for label in [0, 1]:
            count = binary_counts[label]
            pct = (count / total_interactions) * 100
            label_name = "Correct (AC)" if label == 1 else "Incorrect (Non-AC)"
            self.logger.info(f"  {label_name}: {count:,} ({pct:.2f}%)")

        self.stats['label_distribution_binary'] = {
            'correct': binary_counts[1],
            'incorrect': binary_counts[0],
            'correct_pct': (binary_counts[1] / total_interactions) * 100
        }

        # Multi-class distribution
        multiclass_counts = Counter(i['label_multiclass']
                                    for i in all_interactions)
        self.logger.info("Multi-class labels:")
        for label_idx, label_name in enumerate(['AC', 'WA', 'CE', 'TLE', 'RE', 'PE', 'MLE', 'OE']):
            count = multiclass_counts[label_idx]
            pct = (count / total_interactions) * 100
            self.logger.info(
                f"  {label_name} ({label_idx}): {count:,} ({pct:.2f}%)")

        self.stats['label_distribution_multiclass'] = {
            label_name: multiclass_counts[idx]
            for idx, label_name in enumerate(['AC', 'WA', 'CE', 'TLE', 'RE', 'PE', 'MLE', 'OE'])
        }

        # Hierarchical distribution
        hierarchical_l1_counts = Counter(
            i['label_hierarchical']['level1'] for i in all_interactions)
        self.logger.info(
            "Hierarchical labels (Level 1: Compilation vs Execution):")
        for label in [0, 1]:
            count = hierarchical_l1_counts[label]
            pct = (count / total_interactions) * 100
            label_name = "Compilation" if label == 0 else "Execution"
            self.logger.info(
                f"  {label_name} ({label}): {count:,} ({pct:.2f}%)")

        self.stats['label_distribution_hierarchical_l1'] = {
            'compilation': hierarchical_l1_counts[0],
            'execution': hierarchical_l1_counts[1]
        }

    # ==================== LABEL CONVERSION ====================

    def _convert_to_binary(self, result: str) -> int:
        """
        Binary classification: AC = 1, all others = 0.

        Args:
            result: Outcome string (AC, WA, CE, etc.)

        Returns:
            0 or 1
        """
        return 1 if result == 'AC' else 0

    def _convert_to_multiclass(self, result: str) -> int:
        """
        Multi-class classification: Map 8 outcome types to indices.

        Mapping:
            AC: 0 (Accepted)
            WA: 1 (Wrong Answer)
            CE: 2 (Compilation Error)
            TLE: 3 (Time Limit Exceeded)
            RE: 4 (Runtime Error)
            PE: 5 (Presentation Error)
            MLE: 6 (Memory Limit Exceeded)
            OE: 7 (Other Error)

        Args:
            result: Outcome string

        Returns:
            Integer class index (0-7)
        """
        if result in self.MULTICLASS_MAPPING:
            return self.MULTICLASS_MAPPING[result]
        else:
            # Unknown result types mapped to OE (7)
            self.logger.debug(
                f"Unknown result type '{result}', mapping to OE (7)")
            return 7

    def _create_hierarchical_label(self, result: str) -> Dict[str, int]:
        """
        Hierarchical classification: Two-level hierarchy.

        Level 1: Compilation vs Execution
            - Compilation errors: CE, REG → 0
            - Execution phase: AC, WA, TLE, RE, PE, MLE, OE → 1

        Level 2 (within execution phase):
            - AC → 0
            - WA → 1
            - TLE → 2
            - RE → 3
            - PE/MLE/OE → 4

        Args:
            result: Outcome string

        Returns:
            Dict with 'level1' and 'level2' integer labels
        """
        # Compilation errors
        if result in self.COMPILATION_ERRORS:
            return {
                'level1': 0,  # Compilation phase
                'level2': 0 if result == 'CE' else 1  # CE or REG
            }

        # Execution phase
        execution_mapping = {
            'AC': 0,
            'WA': 1,
            'TLE': 2,
            'RE': 3,
            'PE': 4,
            'MLE': 4,
            'OE': 4
        }

        level2 = execution_mapping.get(result, 4)  # Unknown → 4

        return {
            'level1': 1,  # Execution phase
            'level2': level2
        }

    # ==================== FEATURE EXTRACTION ====================

    def _get_problem_tags(self, problem_id: int) -> List[int]:
        """
        Get list of tag IDs for a problem.

        Args:
            problem_id: Problem identifier

        Returns:
            List of tag IDs, sorted by weight (descending)
        """
        problem_tag_rows = self.problemtags[self.problemtags['problemid'] == problem_id]

        if len(problem_tag_rows) == 0:
            return []

        # Sort by weight (descending) to prioritize primary skills
        problem_tag_rows = problem_tag_rows.sort_values(
            'weight', ascending=False, na_position='last')

        return problem_tag_rows['tagid'].tolist()

    def _get_problem_difficulty(self, problem_id: int) -> Optional[float]:
        """
        Get difficulty rating for a problem.

        Args:
            problem_id: Problem identifier

        Returns:
            Difficulty score (float) or None if not available
        """
        problem_row = self.problems[self.problems['id'] == problem_id]

        if len(problem_row) == 0:
            return None

        difficulty = problem_row['difficulty'].values[0]
        return float(difficulty) if pd.notna(difficulty) else None

    # ==================== DATA SPLITTING ====================

    def split_data(
        self,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
        split_type: str = 'standard'
    ) -> None:
        """
        Split sequences into train/val/test sets.

        Args:
            test_size: Proportion for test set (default: 0.2)
            val_size: Proportion for validation set from remaining (default: 0.1)
            random_state: Random seed for reproducibility
            split_type: 'standard' or 'domain_shift'
        """
        self.logger.info("=" * 60)
        self.logger.info(f"DATA SPLITTING ({split_type.upper()})")
        self.logger.info("=" * 60)

        if self.sequences is None:
            raise ValueError(
                "Must call build_student_sequences() before split_data()")

        if split_type == 'standard':
            self._standard_split(test_size, val_size, random_state)
        elif split_type == 'domain_shift':
            self._domain_shift_split(val_size, random_state)
        else:
            raise ValueError(
                f"Unknown split_type: '{split_type}'. Use 'standard' or 'domain_shift'.")

        # Store split info in metadata
        self.metadata['split_type'] = split_type
        self.metadata['test_size'] = test_size
        self.metadata['val_size'] = val_size
        self.metadata['random_state'] = random_state

    def _standard_split(
        self,
        test_size: float,
        val_size: float,
        random_state: int
    ) -> None:
        """
        Standard random split with student-level partitioning.

        Args:
            test_size: Proportion for test set
            val_size: Proportion for validation set
            random_state: Random seed
        """
        from sklearn.model_selection import train_test_split

        train_pct = 1 - test_size - val_size

        self.logger.info(f"Split proportions:")
        self.logger.info(f"  Train: {train_pct:.1%}")
        self.logger.info(f"  Val:   {val_size:.1%}")
        self.logger.info(f"  Test:  {test_size:.1%}")

        np.random.seed(random_state)

        # First split: train+val vs test
        train_val, test = train_test_split(
            self.sequences,
            test_size=test_size,
            random_state=random_state
        )

        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        train, val = train_test_split(
            train_val,
            test_size=val_size_adjusted,
            random_state=random_state
        )

        self.train_data = train
        self.val_data = val
        self.test_data = test

        self.logger.info(f"Split complete:")
        self.logger.info(f"    Train: {len(train):,} students")
        self.logger.info(f"    Val:   {len(val):,} students")
        self.logger.info(f"    Test:  {len(test):,} students")

        # Store split statistics
        self.stats['split_standard'] = {
            'train_count': len(train),
            'val_count': len(val),
            'test_count': len(test)
        }

    def _domain_shift_split(
        self,
        val_size: float,
        random_state: int
    ) -> None:
        """
        Domain shift split:
        - Train: Daily exercises (contestid IS NULL)
        - Test: Contest submissions (contestid IS NOT NULL)
        - Val: 10% from training data

        Args:
            val_size: Proportion of training data for validation
            random_state: Random seed
        """
        self.logger.info("Performing domain shift split (daily → contest)...")

        # Separate sequences by domain
        daily_sequences = []
        contest_sequences = []

        for seq in self.sequences:
            # Check if ANY interaction in sequence is from a contest
            has_contest = any(
                interaction.get('is_contest', False)
                for interaction in seq['sequence']
            )

            if has_contest:
                contest_sequences.append(seq)
            else:
                daily_sequences.append(seq)

        self.logger.info(f"  Daily sequences: {len(daily_sequences):,}")
        self.logger.info(f"  Contest sequences: {len(contest_sequences):,}")

        if len(contest_sequences) == 0:
            self.logger.warning(
                "No contest sequences found! Domain shift split not possible.")
            self.logger.warning("  Falling back to standard split...")
            self._standard_split(
                test_size=0.2, val_size=val_size, random_state=random_state)
            return

        # Use daily for training, contest for testing
        np.random.seed(random_state)
        np.random.shuffle(daily_sequences)

        # Split training into train/val
        val_count = int(val_size * len(daily_sequences))

        self.val_data = daily_sequences[:val_count]
        self.train_data = daily_sequences[val_count:]
        self.test_data = contest_sequences

        self.logger.info(f"Domain shift split complete:")
        self.logger.info(
            f"    Train (daily): {len(self.train_data):,} students")
        self.logger.info(f"    Val (daily):   {len(self.val_data):,} students")
        self.logger.info(
            f"    Test (contest): {len(self.test_data):,} students")

        # Store split statistics
        self.stats['split_domain_shift'] = {
            'train_daily_count': len(self.train_data),
            'val_daily_count': len(self.val_data),
            'test_contest_count': len(self.test_data)
        }

    # ==================== SAVING & OUTPUT ====================

    def save_preprocessed_data(
        self,
        split_type: str = 'standard',
        output_formats: List[str] = ['pickle', 'csv']
    ) -> None:
        """
        Save preprocessed sequences to files.

        Args:
            split_type: 'standard' or 'domain_shift'
            output_formats: List of formats to save ('pickle', 'json', 'csv')
        """
        self.logger.info("=" * 60)
        self.logger.info("SAVING PREPROCESSED DATA")
        self.logger.info("=" * 60)

        if self.train_data is None:
            raise ValueError(
                "Must call split_data() before save_preprocessed_data()")

        # Save for each mode
        modes = ['binary', 'multiclass', 'hierarchical']

        for mode in modes:
            self.logger.info(f"Saving {mode} data ({split_type})...")

            # Create subdirectory
            mode_dir = self.output_dir / mode / split_type
            mode_dir.mkdir(parents=True, exist_ok=True)

            # Save each split
            splits = {
                'train': self.train_data,
                'val': self.val_data,
                'test': self.test_data
            }

            for split_name, split_data in splits.items():
                if split_data is None:
                    continue

                for output_format in output_formats:
                    filepath = mode_dir / \
                        f"{split_name}.{self._get_file_extension(output_format)}"

                    if output_format == 'pickle':
                        self._save_pickle(split_data, filepath, mode)
                    elif output_format == 'json':
                        self._save_json(split_data, filepath, mode)
                    elif output_format == 'csv':
                        self._save_csv(split_data, filepath, mode)
                    else:
                        self.logger.warning(
                            f"Unknown format '{output_format}', skipping")
                        continue

                    self.logger.info(
                        f"  Saved {split_name}.{self._get_file_extension(output_format)}")

        # Save statistics
        stats_dir = self.output_dir / 'statistics'
        stats_file = stats_dir / f'preprocessing_stats_{split_type}.json'
        self._save_statistics(stats_file)
        self.logger.info(f"Saved statistics to {stats_file}")

        self.logger.info("=" * 60)
        self.logger.info("ALL DATA SAVED SUCCESSFULLY")
        self.logger.info("=" * 60)

    def _get_file_extension(self, output_format: str) -> str:
        """Get file extension for output format."""
        extensions = {
            'pickle': 'pkl',
            'json': 'json',
            'csv': 'csv'
        }
        return extensions.get(output_format, output_format)

    def _save_pickle(self, data: List[Dict], filepath: Path, mode: str) -> None:
        """Save data as pickle file."""
        with open(filepath, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _save_json(self, data: List[Dict], filepath: Path, mode: str) -> None:
        """Save data as JSON file."""
        # Convert numpy types to Python types for JSON serialization
        json_data = []
        for seq in data:
            json_seq = {
                'user_id': int(seq['user_id']),
                'sequence_length': int(seq['sequence_length']),
                'sequence': []
            }
            for interaction in seq['sequence']:
                json_interaction = {k: self._convert_to_json_serializable(v)
                                    for k, v in interaction.items()}
                json_seq['sequence'].append(json_interaction)
            json_data.append(json_seq)

        with open(filepath, 'w') as f:
            json.dump(json_data, f, indent=2)

    def _convert_to_json_serializable(self, obj: Any) -> Any:
        """Convert numpy types to JSON-serializable types."""
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        else:
            return obj

    def _save_csv(self, data: List[Dict], filepath: Path, mode: str) -> None:
        """
        Save data as CSV file (flattened for inspection).

        Each row = one interaction with user context.
        """
        rows = []

        for seq in data:
            user_id = seq['user_id']
            for interaction in seq['sequence']:
                row = {
                    'user_id': user_id,
                    'step': interaction['step'],
                    'problem_id': interaction['problem_id'],
                    'result': interaction['result']
                }

                # Add label based on mode
                if mode == 'binary':
                    row['label'] = interaction['label_binary']
                elif mode == 'multiclass':
                    row['label'] = interaction['label_multiclass']
                elif mode == 'hierarchical':
                    row['label_level1'] = interaction['label_hierarchical']['level1']
                    row['label_level2'] = interaction['label_hierarchical']['level2']

                # Add features if present
                if 'tags' in interaction:
                    row['tags'] = ','.join(
                        map(str, interaction['tags'])) if interaction['tags'] else ''
                if 'difficulty' in interaction:
                    row['difficulty'] = interaction['difficulty']
                if 'is_contest' in interaction:
                    row['is_contest'] = interaction['is_contest']

                rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False)

    def _save_statistics(self, filepath: Path) -> None:
        """Save preprocessing statistics as JSON."""
        # Add metadata to stats
        output = {
            'metadata': self.metadata,
            'statistics': self.stats
        }

        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2, default=str)

    # ==================== VALIDATION ====================

    def validate_sequences(self) -> bool:
        """
        Validate processed sequences for integrity.

        Returns:
            True if all validations pass
        """
        self.logger.info("=" * 60)
        self.logger.info("VALIDATING SEQUENCES")
        self.logger.info("=" * 60)

        if self.sequences is None:
            self.logger.error("✗ No sequences to validate")
            return False

        all_valid = True

        # Check 1: All sequences have required fields
        required_fields = ['user_id', 'sequence', 'sequence_length']
        for seq in self.sequences[:100]:  # Sample check
            if not all(field in seq for field in required_fields):
                self.logger.error(
                    f"✗ Sequence missing required fields: {seq.get('user_id')}")
                all_valid = False
                break

        if all_valid:
            self.logger.info("All sequences have required fields")

        # Check 2: Sequence lengths match
        for seq in self.sequences[:100]:
            if len(seq['sequence']) != seq['sequence_length']:
                self.logger.error(
                    f"✗ Length mismatch for user {seq['user_id']}")
                all_valid = False
                break

        if all_valid:
            self.logger.info("Sequence lengths consistent")

        # Check 3: Binary labels in {0, 1}
        for seq in self.sequences[:100]:
            for interaction in seq['sequence']:
                if 'label_binary' in interaction:
                    if interaction['label_binary'] not in {0, 1}:
                        self.logger.error(
                            f"✗ Invalid binary label: {interaction['label_binary']}")
                        all_valid = False
                        break

        if all_valid:
            self.logger.info("Binary labels valid")

        # Check 4: Multi-class labels in {0..7}
        for seq in self.sequences[:100]:
            for interaction in seq['sequence']:
                if 'label_multiclass' in interaction:
                    if not (0 <= interaction['label_multiclass'] <= 7):
                        self.logger.error(
                            f"✗ Invalid multiclass label: {interaction['label_multiclass']}")
                        all_valid = False
                        break

        if all_valid:
            self.logger.info("Multi-class labels valid")

        # Check 5: No duplicate submissions in same sequence
        for seq in self.sequences[:100]:
            steps = [i['step'] for i in seq['sequence']]
            if len(steps) != len(set(steps)):
                self.logger.error(
                    f"✗ Duplicate steps in sequence for user {seq['user_id']}")
                all_valid = False
                break

        if all_valid:
            self.logger.info("No duplicate steps in sequences")

        if all_valid:
            self.logger.info("=" * 60)
            self.logger.info("ALL VALIDATIONS PASSED")
            self.logger.info("=" * 60)
        else:
            self.logger.error("=" * 60)
            self.logger.error("✗ VALIDATION FAILED")
            self.logger.error("=" * 60)

        return all_valid

    # ==================== UTILITY ====================

    def get_vocab_sizes(self) -> Dict[str, int]:
        """
        Get vocabulary sizes for model initialization.

        Returns:
            Dictionary with vocab sizes for problems, tags, etc.
        """
        if self.sequences is None:
            raise ValueError("Must build sequences first")

        # Get unique problem IDs
        all_problem_ids = set()
        all_tag_ids = set()

        for seq in self.sequences:
            for interaction in seq['sequence']:
                all_problem_ids.add(interaction['problem_id'])
                if 'tags' in interaction:
                    all_tag_ids.update(interaction['tags'])

        vocab_sizes = {
            'num_problems': len(all_problem_ids),
            'num_tags': len(all_tag_ids),
            'num_users': len(self.sequences),
            'num_classes_binary': 2,
            'num_classes_multiclass': 8,
            'num_classes_hierarchical_l1': 2,
            'num_classes_hierarchical_l2': 5
        }

        self.logger.info("Vocabulary sizes:")
        for key, value in vocab_sizes.items():
            self.logger.info(f"  {key}: {value}")

        return vocab_sizes

    def summary(self) -> str:
        """
        Generate a summary report of preprocessing.

        Returns:
            Summary string
        """
        lines = []
        lines.append("=" * 60)
        lines.append("PREPROCESSING SUMMARY")
        lines.append("=" * 60)

        if self.stats:
            # Raw data stats
            if 'raw' in self.stats:
                lines.append("Raw Data:")
                lines.append(
                    f"  Submissions: {self.stats['raw']['total_submissions']:,}")
                lines.append(
                    f"  Problems: {self.stats['raw']['total_problems']:,}")
                lines.append(f"  Tags: {self.stats['raw']['total_tags']:,}")

            # OJ2019 extraction
            if 'oj2019' in self.stats:
                lines.append("\nOJ2019 Subset:")
                lines.append(
                    f"  Daily submissions: {self.stats['oj2019']['daily_submissions']:,}")
                lines.append(
                    f"  Students: {self.stats['oj2019']['daily_users']:,}")

            # Sequences
            if 'sequences' in self.stats:
                lines.append("\nSequences:")
                lines.append(
                    f"  Total students: {self.stats['sequences']['students_kept']:,}")
                lines.append(
                    f"  Avg length: {self.stats['sequences']['mean_length']:.2f}")
                lines.append(
                    f"  Median length: {self.stats['sequences']['median_length']:.0f}")

            # Split
            if 'split_standard' in self.stats:
                lines.append("\nStandard Split:")
                lines.append(
                    f"  Train: {self.stats['split_standard']['train_count']:,}")
                lines.append(
                    f"  Val: {self.stats['split_standard']['val_count']:,}")
                lines.append(
                    f"  Test: {self.stats['split_standard']['test_count']:,}")

            if 'split_domain_shift' in self.stats:
                lines.append("\nDomain Shift Split:")
                lines.append(
                    f"  Train (daily): {self.stats['split_domain_shift']['train_daily_count']:,}")
                lines.append(
                    f"  Val (daily): {self.stats['split_domain_shift']['val_daily_count']:,}")
                lines.append(
                    f"  Test (contest): {self.stats['split_domain_shift']['test_contest_count']:,}")

        lines.append("=" * 60)

        summary_str = '\n'.join(lines)
        print(summary_str)
        return summary_str


# ==================== MAIN EXECUTION (for testing) ====================

if __name__ == "__main__":
    # Example usage
    import logging

    # Setup logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Initialize preprocessor
    preprocessor = KTPreProcessor(
        data_dir='data/raw',
        output_dir='data/preprocessed',
        min_seq_length=3,
        max_seq_length=200,
        logger_name='hkt-mop.data.utils'
    )

    # Load data (with OJ2019 extraction)
    preprocessor.load_raw_data(
        extract_oj2019=True,
        oj2019_method='all_daily'  # Use 'id_range' with year_id_range for year filtering
    )

    # Build sequences
    preprocessor.build_student_sequences(
        include_features=True,
        include_all_labels=True
    )

    # Validate
    preprocessor.validate_sequences()

    # Split data (standard)
    preprocessor.split_data(
        test_size=0.2,
        val_size=0.1,
        random_state=42,
        split_type='standard'
    )

    # Save
    preprocessor.save_preprocessed_data(
        split_type='standard',
        output_formats=['pickle', 'csv']
    )

    # Split data (domain shift)
    preprocessor.split_data(
        val_size=0.1,
        random_state=42,
        split_type='domain_shift'
    )

    # Save
    preprocessor.save_preprocessed_data(
        split_type='domain_shift',
        output_formats=['pickle', 'csv']
    )

    # Get vocab sizes
    vocab_sizes = preprocessor.get_vocab_sizes()

    # Print summary
    preprocessor.summary()
