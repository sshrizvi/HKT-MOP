"""
Database connection utilities for ACCoding dataset.
Handles MySQL connection and data extraction.
"""

from ast import Dict, List, Tuple
import logging
from typing import ParamSpecArgs
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

    def execute_query(self, sql):
        """Executes Only SELECT SQL Query."""

        try:
            # Check for SELECT Query
            if sql.strip().upper().startswith('SELECT'):
                cursor = self.connection.cursor()
                cursor.execute(sql)
                results = cursor.fetchall()
                cursor.close()
                return results
            else:
                logger.error('Only SELECT statements can be executed.')
                return

        except Exception as e:
            logger.exception(f"Query execution failed: {e}")
            raise

    def get_table_schema(self, tableName: str) -> pd.DataFrame:
        """Provides Table Schema."""

        query = f"DESCRIBE {tableName};"
        schema = pd.read_sql(query, self.connection)
        return schema

    def get_table_stats(self) -> pd.Series:
        """Provides Row Count for Tables in ACCoding."""

        try:
            stats = {}

            # Tables in ACCoding
            tables = ['users', 'problems', 'submissions',
                      'tags', 'contests', 'problem_tags']

            for table in tables:
                sql = f"SELECT COUNT(*) as count FROM {table}"
                result = self.execute_query(sql)
                stats[table] = result[0][0]

            return pd.Series(stats)

        except Exception as e:
            logger.exception(f"Failed to get table statistics: {e}")
            raise

    def validate_connection(self):
        """Test connection health with a simple query."""

        try:
            sql = "SELECT 1"
            result = self.execute_query(sql)

            if result and result[0][0] == 1:
                logger.info("Database connection is healthy")
                return True
            else:
                logger.error("Unexpected result from health check")
                return False

        except Exception as e:
            logger.exception(f"Connection validation failed: {e}")
            return False

    def fetch_dataframe(self, sql: str) -> pd.DataFrame:
        """Runs SQL Query and returns result as a DataFrame."""

        try:
            result = pd.read_sql(sql, self.connection)
            return result
        except Exception as e:
            logger.exception(f"Failed to Fetch DataFrame : {e}")
            raise

    def get_user_sequences(self, user_id=None, limit=None) -> pd.DataFrame:
        """
        Fetch submission sequences for a user (or all users) with problems and tags joined.

        Args:
            user_id (int, optional): Specific user ID. If None, fetches all users.
            limit (int, optional): Limit number of submissions returned.

        Returns:
            pandas.DataFrame: Submission sequences with problem and tag information
        """

        # Query
        sql = """
        SELECT 
            s.id AS submission_id,
            s.creator_id AS user_id,
            s.problem_id,
            s.result,
            s.lang,
            s.score,
            s.time_cost,
            s.memory_cost,
            s.code_length,
            p.difficulty,
            t.id AS tag_id,
            t.content AS tag_name,
            pt.weight AS tag_weight
        FROM submissions s
        LEFT JOIN problems p ON s.problem_id = p.id
        LEFT JOIN problem_tags pt ON pt.problem_id = p.id
        LEFT JOIN tags t ON t.id = pt.tag_id
        """

        # Add WHERE Clause if user_id Specified
        if user_id is not None:
            sql += f" WHERE s.creator_id = {user_id}"

        # Order by user then submission ID
        sql += " ORDER BY s.creator_id, s.id"

        # Add LIMIT
        if limit is not None:
            sql += f" LIMIT {limit}"

        return self.fetch_dataframe(sql)

    def get_submissions_with_labels(self, user_id=None, binary=True) -> pd.DataFrame:
        """
        Return submissions with binary or multi-class labels for KT.

        Args:
            user_id (int, optional): Specific user ID. If None, fetches all users.
            binary (bool): If True, returns binary labels (AC=1, else 0).
                        If False, returns multi-class outcomes (compile_error, runtime_error, accepted).

        Returns:
            pandas.DataFrame: Submissions with labeled outcomes
        """

        if binary:
            # Binary classification: AC=1, else 0
            sql = """
            SELECT 
                s.id AS submission_id,
                s.creator_id AS user_id,
                s.problem_id,
                s.result,
                CASE 
                    WHEN s.result = 'AC' THEN 1 
                    ELSE 0 
                END AS correct
            FROM submissions s
            """
        else:
            # Multi-class outcomes for hierarchical models
            sql = """
            SELECT 
                s.id AS submission_id,
                s.creator_id AS user_id,
                s.problem_id,
                s.result,
                CASE 
                    WHEN s.result IN ('CE', 'REG') THEN 'compile_error'
                    WHEN s.result IN ('WA', 'TLE', 'MLE', 'PE', 'OE') THEN 'runtime_error'
                    WHEN s.result = 'AC' THEN 'accepted'
                    ELSE 'other'
                END AS outcome_group
            FROM submissions s
            """

        # Add WHERE Clause if user_id Specified
        if user_id is not None:
            sql += f" WHERE s.creator_id = {user_id}"

        # Order by user then submission ID
        sql += " ORDER BY s.creator_id, s.id"

        return self.fetch_dataframe(sql)

    def get_problem_tags(self, problem_id) -> pd.DataFrame:
        """
        Retrieve all tags and optional weights for a specific problem.

        Args:
            problem_id (int): Problem ID to fetch tags for

        Returns:
            pandas.DataFrame: Tags associated with the problem including weights
        """

        sql = f"""
        SELECT 
            t.id AS tag_id,
            t.content AS tag_name,
            pt.weight AS tag_weight
        FROM problem_tags pt
        JOIN tags t ON pt.tag_id = t.id
        WHERE pt.problem_id = {problem_id}
        ORDER BY pt.weight DESC
        """

        return self.fetch_dataframe(sql)

    def batch_extract_sequences(self, batch_size=10000, offset=0) -> pd.DataFrame:
        """
        Stream large datasets in batches to avoid memory overflow.

        Args:
            batch_size (int): Number of submissions per batch
            offset (int): Starting offset for pagination

        Returns:
            pandas.DataFrame: Batch of submission sequences
        """

        sql = f"""
        SELECT 
            s.id AS submission_id,
            s.creator_id AS user_id,
            s.problem_id,
            s.result,
            s.lang,
            s.score,
            s.time_cost,
            s.memory_cost,
            p.difficulty,
            t.id AS tag_id,
            t.content AS tag_name,
            pt.weight AS tag_weight
        FROM submissions s
        LEFT JOIN problems p ON s.problem_id = p.id
        LEFT JOIN problem_tags pt ON pt.problem_id = p.id
        LEFT JOIN tags t ON t.id = pt.tag_id
        ORDER BY s.creator_id, s.id
        LIMIT {batch_size} OFFSET {offset}
        """

        return self.fetch_dataframe(sql)
