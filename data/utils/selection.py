"""
Main Script for experimenting and analyzing the best subset
for research.

Author : Syed Shujaat Haider
"""


import logging
from pathlib import Path
from logging_config import setup_logging
import pandas as pd
from data.utils.subset_selector import SubsetSelector
from data.utils.utils import load_config, random_range
from tqdm.auto import tqdm


# Setup Logging
logger = logging.getLogger('hkt-mop.data.utils')
setup_logging()

# Constants
EXPERIMENT_COUNT = 10
PER_EXPERIMENT_RANGE_COUNT = 5
WINDOW_SIZE = 500000

# PATHS
CONFIG_PATH = "experiments/configs/data_config.yaml"
REPORT_DIR = Path('data/reports')
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def main():

    # Loading PATHS
    PATHS = load_config(CONFIG_PATH)['data_paths']

    # Reading CSV
    try:
        users_df = pd.read_csv(PATHS['users'])
        logger.info(f'Loaded {PATHS['users']}')

        problems_df = pd.read_csv(PATHS['problems'])
        logger.info(f'Loaded {PATHS['problems']}')

        contests_df = pd.read_csv(PATHS['contests'])
        logger.info(f'Loaded {PATHS['contests']}')

        tags_df = pd.read_csv(PATHS['tags'])
        logger.info(f'Loaded {PATHS['tags']}')

        problem_tags_df = pd.read_csv(PATHS['problem_tags'])
        logger.info(f'Loaded {PATHS['problem_tags']}')

        submissions_df = pd.read_csv(PATHS['submissions'])
        logger.info(f'Loaded {PATHS['submissions']}')

    except Exception as e:
        logger.exception(f'Error Loading DataFrames : {e}')
        raise

    # Running Comparison Experiments
    try:
        logger.info(f'Starting Comparison Experiments...')

        for exp in tqdm(range(EXPERIMENT_COUNT)):

            logger.info(f'Experiment No. {exp + 1} Started')

            # Generating Ranges & SubsetSelectors
            subset_selectors = {}
            experiment_ranges = {}
            try:
                range_count = PER_EXPERIMENT_RANGE_COUNT
                for i in tqdm(range(range_count)):

                    experiment_ranges[i] = random_range(
                        1, submissions_df['id'].max(), WINDOW_SIZE)

                    lb, ub = experiment_ranges[i]
                    subset_selectors[i] = SubsetSelector(submissions_df=submissions_df,
                                                         problems_df=problems_df,
                                                         problem_tags_df=problem_tags_df,
                                                         tags_df=tags_df,
                                                         contests_df=contests_df,
                                                         users_df=users_df).select_by_id_range(lb, ub)

            except Exception as e:
                logger.exception(f'Error Creating SubsetSelector : {e}')
                raise

            # Generating Comparison Report
            comparison_df = subset_selectors[0].generate_comparison_report(
                dict(list(subset_selectors.items())[1:]))

            # Saving Comparison Experiment
            output_file = REPORT_DIR / f'comparison_experiment_{exp + 1}.csv'
            comparison_df.to_csv(output_file, header=True, index=True)
            logger.info(f'Comparison Report No. {exp + 1} Generated')

            logger.info(f'Experiment No. {exp + 1} Finished')

    except Exception as e:
        logger.error(f'Error in Performing Comparison Experiments : {e}')
        raise


if __name__ == '__main__':
    main()
