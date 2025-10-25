"""
Main script for processing ACCoding dataset from MySQL database.
"""


import logging
from ast import Dict
import yaml
from data.utils.database_connection import DatabaseConnector
from logging_config import setup_logging


# Setting Up Logger
logger = logging.getLogger('hkt-mop.data.utils')
setup_logging()


# PATHS
CONFIG_PATH = "experiments/configs/data_config.yaml"


# Utilities
def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


# Main Method
def main():
    """Main data processing pipeline."""
    
    # Loading Config
    config = load_config(CONFIG_PATH)
    
    
    # Initializing DatabaseConnector
    db_connector = DatabaseConnector(config['database'])
    
    try:
        # Connecting
        if not db_connector.connect():
            logger.error("Failed to Connect to Database")
            return
        logger.info("Connected to ACCoding Successfully")
    
    except Exception as e:
        logger.exception(f"Error during Data Processing : {e}")
        raise
    
    finally:
        # Disconnecting
        db_connector.disconnect()
    

# Executing Main Method
if __name__ == "__main__":
    main()
