import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.stats import entropy
import logging
from logging_config import setup_logging

class SubsetSelector:
    """
    Evaluates and selects optimal subsets from ACcoding dataset for Knowledge Tracing research.
    
    The selector assesses subsets based on:
    1. Hierarchical multi-outcome feedback distribution
    2. Student engagement quality and sequence depth
    3. Knowledge point (tag) diversity and coverage
    4. Domain shift readiness (daily vs contest practice)
    
    Note: Since submissions table has no timestamp, ordering is based on submission.id (auto-increment).
    """
    
    def __init__(self, 
                 submissions_df: pd.DataFrame,
                 problems_df: pd.DataFrame,
                 problem_tags_df: pd.DataFrame,
                 tags_df: pd.DataFrame,
                 contests_df: Optional[pd.DataFrame] = None,
                 users_df: Optional[pd.DataFrame] = None,
                 logger: Optional[logging.Logger] = None):
        """Initialize SubsetSelector with ACcoding dataset tables."""
        
        # Setup logger
        self.logger = logging.getLogger('hkt-mop.data.utils')
        setup_logging()
        
        self.logger.info("Initializing SubsetSelector...")
        
        self.submissions = submissions_df.copy()
        self.problems = problems_df.copy()
        self.problem_tags = problem_tags_df.copy()
        self.tags = tags_df.copy()
        self.contests = contests_df.copy() if contests_df is not None else None
        self.users = users_df.copy() if users_df is not None else None
        
        self.logger.debug(f"Loaded {len(self.submissions)} submissions")
        self.logger.debug(f"Loaded {len(self.problems)} problems")
        self.logger.debug(f"Loaded {len(self.problem_tags)} problem-tag associations")
        self.logger.debug(f"Loaded {len(self.tags)} tags")
        
        # Validate required columns
        self._validate_schema()
        
        # Prepare merged data for analysis
        self._prepare_enriched_submissions()
        
        self.logger.info("SubsetSelector initialized successfully")
        
    def _validate_schema(self):
        """Validate that DataFrames have required columns."""
        
        self.logger.debug("Validating schema...")
        
        required_cols = {
            'submissions': ['id', 'result', 'creator_id', 'problem_id', 'contest_id'],
            'problems': ['id', 'difficulty'],
            'problem_tags': ['tag_id', 'problem_id'],
            'tags': ['id', 'content']
        }
        
        validation_passed = True
        for df_name, cols in required_cols.items():
            df = getattr(self, df_name.replace('_df', ''))
            missing = set(cols) - set(df.columns)
            if missing:
                self.logger.error(f"{df_name} missing required columns: {missing}")
                validation_passed = False
            else:
                self.logger.debug(f"{df_name} schema validation passed")
        
        if not validation_passed:
            raise ValueError("Schema validation failed. Check logs for details.")
        
        self.logger.info("Schema validation completed successfully")
    
    def _prepare_enriched_submissions(self):
        """Merge submissions with problems and tags for efficient analysis."""
        
        self.logger.info("Preparing enriched submissions dataset...")
        
        initial_count = len(self.submissions)
        
        # Merge with problems to get difficulty
        self.enriched = self.submissions.merge(
            self.problems[['id', 'difficulty']], 
            left_on='problem_id', 
            right_on='id', 
            how='left',
            suffixes=('', '_problem')
        ).drop('id_problem', axis=1, errors='ignore')
        
        self.logger.debug(f"Merged with problems: {len(self.enriched)} records")
        
        # Merge with problem_tags and tags to get knowledge points
        self.enriched = self.enriched.merge(
            self.problem_tags[['problem_id', 'tag_id', 'weight']], 
            on='problem_id', 
            how='left'
        )
        
        self.logger.debug(f"Merged with problem_tags: {len(self.enriched)} records")
        
        self.enriched = self.enriched.merge(
            self.tags[['id', 'content']], 
            left_on='tag_id', 
            right_on='id', 
            how='left',
            suffixes=('', '_tag')
        ).drop('id_tag', axis=1, errors='ignore')
        
        # Flag contest vs daily submissions
        self.enriched['is_contest'] = self.enriched['contest_id'].notna()
        
        contest_count = self.enriched['is_contest'].sum()
        daily_count = (~self.enriched['is_contest']).sum()
        
        self.logger.info(f"Enriched dataset prepared: {len(self.enriched)} records")
        self.logger.debug(f"Contest submissions: {contest_count} ({contest_count/len(self.enriched)*100:.2f}%)")
        self.logger.debug(f"Daily submissions: {daily_count} ({daily_count/len(self.enriched)*100:.2f}%)")
        
    def select_by_id_range(self, start_id: int, end_id: int) -> 'SubsetSelector':
        """Creates a new SubsetSelector for a specific submission ID range."""
        
        self.logger.info(f"Selecting subset by ID range: [{start_id}, {end_id}]")
        
        subset_submissions = self.submissions[
            (self.submissions['id'] >= start_id) & 
            (self.submissions['id'] <= end_id)
        ]
        
        self.logger.info(f"Found {len(subset_submissions)} submissions in range")
        
        # Filter related tables to only include relevant entities
        relevant_problems = subset_submissions['problem_id'].dropna().unique()
        relevant_users = subset_submissions['creator_id'].dropna().unique()
        relevant_contests = subset_submissions['contest_id'].dropna().unique()
        
        self.logger.debug(f"Relevant entities - Problems: {len(relevant_problems)}, "
                          f"Users: {len(relevant_users)}, Contests: {len(relevant_contests)}")
        
        subset_problems = self.problems[self.problems['id'].isin(relevant_problems)]
        subset_problem_tags = self.problem_tags[self.problem_tags['problem_id'].isin(relevant_problems)]
        relevant_tags = subset_problem_tags['tag_id'].unique()
        subset_tags = self.tags[self.tags['id'].isin(relevant_tags)]
        
        subset_contests = None
        if self.contests is not None:
            subset_contests = self.contests[self.contests['id'].isin(relevant_contests)]
        
        subset_users = None
        if self.users is not None:
            subset_users = self.users[self.users['id'].isin(relevant_users)]
        
        self.logger.info("Creating new SubsetSelector for range")
        
        return SubsetSelector(
            submissions_df=subset_submissions,
            problems_df=subset_problems,
            problem_tags_df=subset_problem_tags,
            tags_df=subset_tags,
            contests_df=subset_contests,
            users_df=subset_users,
            logger=self.logger
        )
    
    def assess_feedback_distribution(self) -> Dict[str, any]:
        """
        Analyze hierarchical multi-outcome feedback distribution.
        
        Returns:
        --------
        Dictionary containing:
        - result_counts: Count of each result type
        - result_proportions: Proportion of each result type
        - hierarchical_balance: Compilation/Execution/Accept proportions
        - balance_score: Quality score (0-1, higher is better)
        """
        
        self.logger.info("Assessing feedback distribution...")
        
        result_counts = self.submissions['result'].value_counts()
        result_props = self.submissions['result'].value_counts(normalize=True)
        
        self.logger.debug(f"Result type distribution:\n{result_counts}")
        
        # Hierarchical categorization
        compilation_errors = self.submissions['result'].isin(['CE', 'REG']).sum()
        execution_errors = self.submissions['result'].isin(['WA', 'TLE', 'RE', 'MLE', 'PE', 'OE']).sum()
        accepts = self.submissions['result'].isin(['AC']).sum()
        other = len(self.submissions) - (compilation_errors + execution_errors + accepts)
        
        total = len(self.submissions)
        hierarchical = {
            'compilation': compilation_errors / total,
            'execution': execution_errors / total,
            'accept': accepts / total,
            'other': other / total
        }
        
        self.logger.info(f"Hierarchical balance - CE: {hierarchical['compilation']:.2%}, "
                        f"Execution: {hierarchical['execution']:.2%}, "
                        f"AC: {hierarchical['accept']:.2%}")
        
        # Balance score: penalize extreme imbalances
        # Ideal ranges: CE: 8-15%, Execution: 50-65%, AC: 25-35%
        ce_score = 1.0 - min(abs(hierarchical['compilation'] - 0.115) / 0.115, 1.0)
        exec_score = 1.0 - min(abs(hierarchical['execution'] - 0.575) / 0.575, 1.0)
        ac_score = 1.0 - min(abs(hierarchical['accept'] - 0.30) / 0.30, 1.0)
        
        # Penalize if AC is too low (<20%) or too high (>50%)
        if hierarchical['accept'] < 0.20 or hierarchical['accept'] > 0.50:
            ac_score *= 0.5
            self.logger.warning(f"AC rate {hierarchical['accept']:.2%} outside optimal range [20%, 50%]")
        
        balance_score = (ce_score * 0.25 + exec_score * 0.40 + ac_score * 0.35)
        
        self.logger.info(f"Feedback balance score: {balance_score:.3f}")
        
        return {
            'result_counts': result_counts.to_dict(),
            'result_proportions': result_props.to_dict(),
            'hierarchical_balance': hierarchical,
            'balance_score': balance_score
        }
    
    def assess_student_engagement(self, min_submissions: int = 20, 
                                   min_problems: int = 10) -> Dict[str, any]:
        """
        Analyze student engagement quality and sequence depth.
        
        Parameters:
        -----------
        min_submissions : Minimum submissions per student for "quality" students
        min_problems : Minimum unique problems attempted for "quality" students
        
        Returns:
        --------
        Dictionary containing student engagement metrics and quality score
        """
        
        self.logger.info("Assessing student engagement...")
        self.logger.debug(f"Quality criteria - Min submissions: {min_submissions}, Min problems: {min_problems}")
        
        student_stats = self.submissions.groupby('creator_id').agg({
            'id': 'count',  # total submissions
            'problem_id': 'nunique',  # unique problems
            'contest_id': lambda x: x.notna().sum(),  # contest submissions
            'result': lambda x: (x == 'AC').sum()  # correct submissions
        }).rename(columns={
            'id': 'total_submissions',
            'problem_id': 'unique_problems',
            'contest_id': 'contest_submissions',
            'result': 'correct_submissions'
        })
        
        student_stats['ac_ratio'] = student_stats['correct_submissions'] / student_stats['total_submissions']
        student_stats['repeat_rate'] = student_stats['total_submissions'] / student_stats['unique_problems']
        student_stats['contest_participation'] = student_stats['contest_submissions'] > 0
        
        # Quality students filter
        quality_students = student_stats[
            (student_stats['total_submissions'] >= min_submissions) &
            (student_stats['unique_problems'] >= min_problems) &
            (student_stats['contest_participation'])
        ]
        
        quality_ratio = len(quality_students) / len(student_stats) if len(student_stats) > 0 else 0
        contest_participation_rate = student_stats['contest_participation'].mean()
        
        self.logger.info(f"Total students: {len(student_stats)}, Quality students: {len(quality_students)} ({quality_ratio:.2%})")
        self.logger.info(f"Contest participation rate: {contest_participation_rate:.2%}")
        
        # Engagement score
        # Ideal: 70%+ quality students, 50%+ contest participation, avg 2-5 repeat rate
        quality_score = min(quality_ratio / 0.70, 1.0) * 0.40
        contest_score = min(contest_participation_rate / 0.50, 1.0) * 0.35
        
        avg_repeat = student_stats['repeat_rate'].mean()
        repeat_score = 1.0 - min(abs(avg_repeat - 3.5) / 3.5, 1.0) if not np.isnan(avg_repeat) else 0
        repeat_score *= 0.25
        
        engagement_score = quality_score + contest_score + repeat_score
        
        self.logger.info(f"Engagement score: {engagement_score:.3f} "
                        f"(quality: {quality_score:.3f}, contest: {contest_score:.3f}, repeat: {repeat_score:.3f})")
        
        if quality_ratio < 0.50:
            self.logger.warning(f"Low quality student ratio: {quality_ratio:.2%} (target: ≥70%)")
        if contest_participation_rate < 0.30:
            self.logger.warning(f"Low contest participation: {contest_participation_rate:.2%} (target: ≥50%)")
        
        return {
            'total_students': len(student_stats),
            'quality_students': len(quality_students),
            'quality_ratio': quality_ratio,
            'contest_participation_rate': contest_participation_rate,
            'avg_submissions_per_student': student_stats['total_submissions'].mean(),
            'median_submissions_per_student': student_stats['total_submissions'].median(),
            'avg_problems_per_student': student_stats['unique_problems'].mean(),
            'avg_repeat_rate': avg_repeat,
            'engagement_score': engagement_score,
            'student_stats': student_stats  # Full stats for further analysis
        }
    
    def assess_tag_diversity(self) -> Dict[str, any]:
        """
        Analyze knowledge point (tag) coverage and diversity.
        
        Returns:
        --------
        Dictionary containing tag diversity metrics and quality score
        """
        self.logger.info("Assessing tag diversity...")
        
        # Get submissions with tags
        submissions_with_tags = self.enriched.dropna(subset=['tag_id'])
        
        if len(submissions_with_tags) == 0:
            self.logger.warning("No submissions with tags found!")
            return {
                'total_tags': 0,
                'tag_entropy': 0,
                'avg_tags_per_problem': 0,
                'tag_coverage_score': 0,
                'tag_distribution': {}
            }
        
        self.logger.debug(f"Found {len(submissions_with_tags)} submissions with tags")
        
        # Tag frequency distribution
        tag_distribution = submissions_with_tags.groupby('content').agg({
            'id': 'count',  # submission count
            'problem_id': 'nunique',  # unique problems
            'creator_id': 'nunique'  # unique students
        }).rename(columns={
            'id': 'submission_count',
            'problem_id': 'problem_count',
            'creator_id': 'student_count'
        })
        
        # Calculate entropy (higher = more diverse)
        tag_entropy = entropy(tag_distribution['submission_count'])
        
        # Tags per problem
        tags_per_problem = self.problem_tags.groupby('problem_id')['tag_id'].count()
        avg_tags_per_problem = tags_per_problem.mean()
        
        total_tags = len(tag_distribution)
        
        self.logger.info(f"Total tags: {total_tags}, Tag entropy: {tag_entropy:.3f}, "
                        f"Avg tags per problem: {avg_tags_per_problem:.2f}")
        
        # Coverage score
        # Ideal: 30+ tags, entropy > 3.5, 5+ problems per major tag
        tag_count_score = min(total_tags / 30, 1.0) * 0.35
        entropy_score = min(tag_entropy / 3.5, 1.0) * 0.40 if not np.isnan(tag_entropy) else 0
        
        # Check if major tags have sufficient coverage
        min_problems_per_tag = tag_distribution['problem_count'].min()
        coverage_score = min(min_problems_per_tag / 5, 1.0) * 0.25
        
        diversity_score = tag_count_score + entropy_score + coverage_score
        
        self.logger.info(f"Tag diversity score: {diversity_score:.3f}")
        
        if total_tags < 30:
            self.logger.warning(f"Low tag count: {total_tags} (target: ≥30)")
        if tag_entropy < 3.5:
            self.logger.warning(f"Low tag entropy: {tag_entropy:.3f} (target: ≥3.5)")
        
        return {
            'total_tags': total_tags,
            'tag_entropy': tag_entropy,
            'avg_tags_per_problem': avg_tags_per_problem,
            'tag_distribution': tag_distribution.to_dict(),
            'diversity_score': diversity_score
        }
    
    def assess_domain_shift_readiness(self) -> Dict[str, any]:
        """
        Evaluate readiness for domain shift experiments (daily practice → contest).
        
        Returns:
        --------
        Dictionary containing domain shift metrics and readiness score
        """
        self.logger.info("Assessing domain shift readiness...")
        
        if self.enriched['contest_id'].isna().all():
            self.logger.warning("No contest submissions found in dataset!")
            return {
                'total_contests': 0,
                'contest_submissions': 0,
                'daily_submissions': 0,
                'domain_shift_score': 0.0,
                'overlap_ratio': 0.0
            }
        
        contest_submissions = self.enriched[self.enriched['is_contest']]
        daily_submissions = self.enriched[~self.enriched['is_contest']]
        
        self.logger.debug(f"Contest submissions: {len(contest_submissions)}, Daily submissions: {len(daily_submissions)}")
        
        # Get students who participated in contests
        contest_students = set(contest_submissions['creator_id'].dropna().unique())
        
        self.logger.debug(f"Analyzing prior practice for {len(contest_students)} contest students...")
        
        # For each contest student, check if they have prior daily practice
        students_with_prior_practice = set()
        
        for i, student_id in enumerate(contest_students):
            if (i + 1) % 500 == 0:
                self.logger.debug(f"Processed {i + 1}/{len(contest_students)} students")
            
            # Get their contest submission IDs
            student_contest_ids = contest_submissions[
                contest_submissions['creator_id'] == student_id
            ]['id'].min()
            
            # Check if they have daily submissions with ID < min contest ID
            prior_daily = daily_submissions[
                (daily_submissions['creator_id'] == student_id) &
                (daily_submissions['id'] < student_contest_ids)
            ]
            
            if len(prior_daily) >= 20:  # At least 20 prior submissions
                students_with_prior_practice.add(student_id)
        
        overlap_ratio = len(students_with_prior_practice) / len(contest_students) if len(contest_students) > 0 else 0
        
        self.logger.info(f"Students with prior practice: {len(students_with_prior_practice)}/{len(contest_students)} ({overlap_ratio:.2%})")
        
        # Calculate daily/contest balance
        daily_ratio = len(daily_submissions) / len(self.enriched)
        contest_ratio = len(contest_submissions) / len(self.enriched)
        
        self.logger.debug(f"Daily ratio: {daily_ratio:.2%}, Contest ratio: {contest_ratio:.2%}")
        
        # Problem difficulty overlap
        contest_problems = set(contest_submissions['problem_id'].dropna().unique())
        daily_problems = set(daily_submissions['problem_id'].dropna().unique())
        
        if len(contest_problems) > 0 and len(daily_problems) > 0:
            # Check difficulty distribution overlap
            contest_difficulties = contest_submissions.groupby('problem_id')['difficulty'].first()
            daily_difficulties = daily_submissions.groupby('problem_id')['difficulty'].first()
            
            difficulty_overlap = 0.5  # Simplified - could compute KL divergence
        else:
            difficulty_overlap = 0
        
        # Domain shift readiness score
        # Ideal: 80%+ overlap, 40-60% daily/contest split
        overlap_score = min(overlap_ratio / 0.80, 1.0) * 0.50
        
        balance_deviation = abs(daily_ratio - 0.50)
        balance_score = (1.0 - min(balance_deviation / 0.50, 1.0)) * 0.30
        
        difficulty_score = difficulty_overlap * 0.20
        
        domain_shift_score = overlap_score + balance_score + difficulty_score
        
        self.logger.info(f"Domain shift score: {domain_shift_score:.3f} "
                        f"(overlap: {overlap_score:.3f}, balance: {balance_score:.3f})")
        
        if overlap_ratio < 0.60:
            self.logger.warning(f"Low domain shift overlap: {overlap_ratio:.2%} (target: ≥80%)")
        
        return {
            'total_contests': self.enriched['contest_id'].nunique() - 1 if self.enriched['contest_id'].notna().any() else 0,
            'contest_submissions': len(contest_submissions),
            'daily_submissions': len(daily_submissions),
            'contest_ratio': contest_ratio,
            'daily_ratio': daily_ratio,
            'contest_students': len(contest_students),
            'students_with_prior_practice': len(students_with_prior_practice),
            'overlap_ratio': overlap_ratio,
            'domain_shift_score': domain_shift_score
        }
    
    def compute_composite_score(self, weights: Optional[Dict[str, float]] = None) -> Dict[str, any]:
        """
        Compute overall subset quality score.
        
        Parameters:
        -----------
        weights : Dictionary with keys ['feedback', 'engagement', 'diversity', 'domain_shift']
                  Default: {'feedback': 0.25, 'engagement': 0.20, 'diversity': 0.15, 'domain_shift': 0.25, 'scale': 0.15}
        
        Returns:
        --------
        Dictionary containing all assessment results and composite score
        """
        self.logger.info("="*60)
        self.logger.info("Computing composite quality score...")
        self.logger.info("="*60)
        
        if weights is None:
            weights = {
                'feedback': 0.25,
                'engagement': 0.20,
                'diversity': 0.15,
                'domain_shift': 0.25,
                'scale': 0.15
            }
        
        self.logger.info(f"Using weights: {weights}")
        
        # Run all assessments
        feedback_results = self.assess_feedback_distribution()
        engagement_results = self.assess_student_engagement()
        diversity_results = self.assess_tag_diversity()
        domain_shift_results = self.assess_domain_shift_readiness()
        
        # Scale score (based on minimum requirements)
        min_students = 2500
        min_submissions = 100000
        
        student_count = engagement_results['total_students']
        submission_count = len(self.submissions)
        
        scale_score = min(
            min(student_count / min_students, 1.0) * 0.50 +
            min(submission_count / min_submissions, 1.0) * 0.50,
            1.0
        )
        
        self.logger.info(f"Scale score: {scale_score:.3f} (Students: {student_count}, Submissions: {submission_count})")
        
        # Compute weighted composite score
        composite_score = (
            feedback_results['balance_score'] * weights['feedback'] +
            engagement_results['engagement_score'] * weights['engagement'] +
            diversity_results['diversity_score'] * weights['diversity'] +
            domain_shift_results['domain_shift_score'] * weights['domain_shift'] +
            scale_score * weights['scale']
        )
        
        self.logger.info("="*60)
        self.logger.info(f"COMPOSITE SCORE: {composite_score:.3f}")
        self.logger.info("="*60)
        self.logger.info(f"  Feedback Balance:  {feedback_results['balance_score']:.3f} (weight: {weights['feedback']})")
        self.logger.info(f"  Engagement:        {engagement_results['engagement_score']:.3f} (weight: {weights['engagement']})")
        self.logger.info(f"  Tag Diversity:     {diversity_results['diversity_score']:.3f} (weight: {weights['diversity']})")
        self.logger.info(f"  Domain Shift:      {domain_shift_results['domain_shift_score']:.3f} (weight: {weights['domain_shift']})")
        self.logger.info(f"  Scale:             {scale_score:.3f} (weight: {weights['scale']})")
        self.logger.info("="*60)
        
        return {
            'composite_score': composite_score,
            'scale_score': scale_score,
            'feedback_assessment': feedback_results,
            'engagement_assessment': engagement_results,
            'diversity_assessment': diversity_results,
            'domain_shift_assessment': domain_shift_results,
            'weights': weights,
            'metadata': {
                'total_submissions': submission_count,
                'total_students': student_count,
                'total_problems': self.submissions['problem_id'].nunique(),
                'id_range': (self.submissions['id'].min(), self.submissions['id'].max())
            }
        }
    
    def generate_comparison_report(self, other_selectors: Dict[str, 'SubsetSelector']) -> pd.DataFrame:
        """
        Generate comparison table for multiple subset candidates.
        
        Parameters:
        -----------
        other_selectors : Dictionary mapping subset names to SubsetSelector instances
        
        Returns:
        --------
        DataFrame with comparison metrics
        """
        self.logger.info(f"Generating comparison report for {len(other_selectors) + 1} subsets...")
        
        results = {}
        
        # Include self in comparison
        all_selectors = {'current': self}
        all_selectors.update(other_selectors)
        
        for name, selector in all_selectors.items():
            self.logger.info(f"Evaluating subset: {name}")
            assessment = selector.compute_composite_score()
            
            results[name] = {
                'Students': assessment['metadata']['total_students'],
                'Submissions': assessment['metadata']['total_submissions'],
                'Problems': assessment['metadata']['total_problems'],
                'ID Range': f"{assessment['metadata']['id_range'][0]}-{assessment['metadata']['id_range'][1]}",
                'Feedback Balance': round(assessment['feedback_assessment']['balance_score'], 3),
                'Engagement': round(assessment['engagement_assessment']['engagement_score'], 3),
                'Tag Diversity': round(assessment['diversity_assessment']['diversity_score'], 3),
                'Domain Shift': round(assessment['domain_shift_assessment']['domain_shift_score'], 3),
                'Scale': round(assessment['scale_score'], 3),
                'Composite Score': round(assessment['composite_score'], 3)
            }
        
        comparison_df = pd.DataFrame(results).T
        comparison_df = comparison_df.sort_values('Composite Score', ascending=False)
        
        self.logger.info("Comparison report generated successfully")
        self.logger.info(f"\nTop subset: {comparison_df.index[0]} (score: {comparison_df.iloc[0]['Composite Score']:.3f})")
        
        return comparison_df
    
    def get_summary_statistics(self) -> Dict[str, any]:
        """
        Get basic summary statistics for the subset.
        
        Returns:
        --------
        Dictionary with summary statistics
        """
        self.logger.info("Generating summary statistics...")
        
        assessment = self.compute_composite_score()
        
        summary = {
            'id_range': (self.submissions['id'].min(), self.submissions['id'].max()),
            'total_submissions': len(self.submissions),
            'total_students': self.submissions['creator_id'].nunique(),
            'total_problems': self.submissions['problem_id'].nunique(),
            'total_tags': self.enriched['tag_id'].nunique(),
            'composite_score': assessment['composite_score'],
            'hierarchical_balance': assessment['feedback_assessment']['hierarchical_balance'],
            'contest_participation_rate': assessment['engagement_assessment']['contest_participation_rate'],
            'tag_entropy': assessment['diversity_assessment']['tag_entropy'],
            'domain_shift_overlap': assessment['domain_shift_assessment']['overlap_ratio']
        }
        
        self.logger.info("Summary statistics generated")
        
        return summary
