"""
Utility Methods used in Data Module

Author : Syed Shujaat Haider
"""


from re import S
import yaml
from typing import Dict, Optional, Tuple
import random


# Utilities
def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def random_range(min_bound: int, max_bound: int, window: int) -> Optional[Tuple[int]]:
    """Generates Lower and Upper Bound of a Range with Fixed Window."""
    
    # Validating Window
    if window > (max_bound - min_bound + 1):
        raise Exception('Window is too big for the Bounds.')
    if window <= 0:
        raise Exception('Window is non-positive.')
    
    # Generating Range
    lower_bound = random.randint(min_bound, max_bound - window)
    upper_bound = lower_bound + window
    
    # Validate Range
    if window != (upper_bound - lower_bound):
        raise Exception('Error generating range.')
    
    return (lower_bound, upper_bound)
    