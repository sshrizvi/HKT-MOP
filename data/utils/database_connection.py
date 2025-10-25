"""
Database connection utilities for ACCoding dataset.
Handles MySQL connection and data extraction.
"""

from ast import Dict
import logging
from logging_config import setup_logging
from sqlalchemy import create_engine
import MySQLdb
import pandas as pd


# Setting Up Logger
logger = logging.getLogger('hkt-mop.data.utils')
setup_logging()


# DatabaseConnector Class
class DatabaseConnector():
    """Handles database connections and data extraction for ACCoding dataset."""

    def __init__(self, config: Dict):
        """
        Initialize database connector.

        Args:
            config: Database configuration dictionary containing:
                - host: Database host
                - port: Database port
                - user: Database username
                - password: Database password
                - database: Database name
        """
        self.config = config
        self.connection = None

    def connect(self):
        """Establish database connection."""

        try:
            self.connection = MySQLdb.connect(
                host=self.config['host'],
                port=self.config['port'],
                user=self.config['user'],
                password=self.config['password'],
                database=self.config['database']
            )
            logger.info("Database connection established successfully")
            return True

        except MySQLdb.Error as e:
            logger.error(f'Error Connecting to Database : {e}')
            return False

    def disconnect(self):
        """Close Database Connection."""

        if self.connection:
            self.connection.close()
        logger.info("Database Connection Closed Succesfully")

    def get_table_schema(self, tableName: str) -> pd.DataFrame:
        """Provides Table Schema."""

        query = f"DESCRIBE {tableName};"
        schema = pd.read_sql(query, self.connection)
        return schema
