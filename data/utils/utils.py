"""
Utility Methods used in Data Module

Author : Syed Shujaat Haider
"""


import yaml
from typing import Dict


# Utilities
def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config