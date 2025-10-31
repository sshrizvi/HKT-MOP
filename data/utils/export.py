"""
Main Script for Exporting all Tables from ACCoding
MySQL Database using the DatabaseConnector Class.

Author : Syed Shujaat Haider
"""


import logging
from data.utils.database_connection import DatabaseConnector
from logging_config import setup_logging
from data.utils.utils import load_config


# Setting Up Logger
logger = logging.getLogger('hkt-mop.data.utils')
setup_logging()


# PATHS
CONFIG_PATH = "experiments/configs/data_config.yaml"


def main():

    # Load Database Configuration
    config = load_config(CONFIG_PATH)

    # Initializing DatabaseConnector
    db_connector = DatabaseConnector(config['database'])

    try:
        # Connecting
        if not db_connector.connect():
            logger.error("Failed to Connect to Database")
            return
        logger.info("Connected to ACCoding Successfully")

        # Connection Health Check
        db_connector.validate_connection()

        # Loading TABLES
        TABLES = config['database']['tables']

        # Exporting Tables
        for table in TABLES:
            db_connector.export_to_csv(
                table_name=table, output_dir=config['exports']['raw_data_dir'])

    except Exception as e:
        logger.exception(f"Error during Data Exporting : {e}")
        raise

    finally:
        # Disconnecting
        db_connector.disconnect()


# Executing Main Method
if __name__ == '__main__':
    main()
