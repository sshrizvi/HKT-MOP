"""
Database connection utilities for ACCoding dataset.
Handles MySQL connection and data extraction.

Author: Syed Shujaat Haider
Project: Hierarchical Knowledge Tracing (HKT-MOP)
"""


from typing import Dict
import logging
from logging_config import setup_logging
import MySQLdb
import pandas as pd
from pathlib import Path
from datetime import datetime
import gc


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
        self.cursor = None
        self.TABLES = self.config['tables']

        # Setup Logger
        self.logger = logging.getLogger('hkt-mop.data.utils')
        setup_logging()

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
            self.cursor = self.connection.cursor()
            if self.cursor:
                self.logger.info("Cursor Initialized Succesfully")
            self.logger.info("Database connection established Successfully")
            return True

        except MySQLdb.Error as e:
            self.logger.error(f'Error Connecting to Database : {e}')
            return False

    def disconnect(self):
        """Close the Cursor and then Database Connection."""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
        self.logger.info("Database Connection Closed Succesfully")

    def get_table_stats(self) -> pd.Series:
        """Provides Row Count for Tables in ACCoding."""

        try:
            stats = {}

            for table in self.TABLES:
                sql = f"SELECT COUNT(*) as count FROM {table}"
                self.cursor.execute(sql)
                result = self.cursor.fetchall()
                stats[table] = result[0][0]

            return pd.Series(stats)

        except Exception as e:
            self.logger.exception(f"Failed to get table statistics: {e}")
            raise

    def validate_connection(self):
        """Test connection health with a simple query."""

        try:
            sql = "SELECT 1"
            result = self._execute_query(sql)

            if result[0][0] == 1:
                self.logger.info("Database connection is Healthy")
                return True
            else:
                self.logger.error("Unexpected result from Health Check")
                return False

        except Exception as e:
            self.logger.exception(f"Connection validation failed: {e}")
            return False

    def export_to_csv(self, table_name: str, output_dir: str = None) -> None:
        """Export Table as CSV and save it to the Output Directory."""

        row_count = self._get_row_count(table_name)
        batch_size = self._determine_batch_size(row_count)
        use_batching = False if batch_size is None else True

        # Ensuring Output Directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if not use_batching:
            self._export_full(table_name, output_path, row_count)
        else:
            self._export_in_batches(
                table_name, output_path, row_count, batch_size)

    def _execute_query(self, sql: str):
        """Private Helper Method for Executing Queries."""

        try:
            self.cursor.execute(sql)
            result = self.cursor.fetchall()
            return result
        except Exception as e:
            self.logger.exception(f"Error Executing SQL Query : {e}")
            raise

    def _get_row_count(self, table_name: str) -> int:
        """Returns Row Count of a Table."""

        query = f"SELECT COUNT(*) FROM {table_name};"
        result = self._execute_query(query)
        return result[0][0]

    def _determine_batch_size(self, row_count: int) -> int:
        """Determines Batch Size for Exporting CSV."""

        SMALL_TABLE_THRESHOLD = 10**4
        MEDIUM_TABLE_THRESHOLD = 10**5
        LARGE_TABLE_THRESHOLD = 10**6

        if row_count < SMALL_TABLE_THRESHOLD:
            return None
        elif row_count < MEDIUM_TABLE_THRESHOLD:
            return 10**4
        elif row_count < LARGE_TABLE_THRESHOLD:
            return 50 * 10**4
        else:
            return 10**5

    def _export_full(self, table_name: str, output_dir: Path, row_count: int):
        """Export Table to CSV without Batching."""

        self.logger.info("=" * 60)
        self.logger.info(f"(Without Batching) Exporting {table_name} ...")
        self.logger.info("=" * 60)

        try:
            query = f"""
                SELECT *
                FROM {table_name}
            """

            # Reading
            df = pd.read_sql(query, self.connection)

            # Saving
            output_file = output_dir / f'{table_name}.csv'
            df.to_csv(output_file, index=False)

            # Logging Completion
            file_size_mb = output_file.stat().st_size / (1024 * 1024)
            self.logger.info(f"Saved to {output_file} ({file_size_mb} MB)")

        except Exception as e:
            self.logger.exception(f"Error Exporting {table_name} : {e}")
            raise

    def _export_in_batches(self, table_name: str, output_dir: Path, row_count: int, batch_size: int):
        """Export Table to CSV with Batching."""

        self.logger.info("=" * 60)
        self.logger.info(f"(With Batching) Exporting {table_name} ...")
        self.logger.info("=" * 60)

        # Determine Number of Batches
        num_batches = (row_count + batch_size - 1) // batch_size
        self.logger.info(f"Exporting {row_count} rows in {num_batches}")

        # Record Start Time
        start_time = datetime.now()

        try:
            first_batch = True
            rows_exported = 0
            
            # Exporting Batches
            for batch in range(num_batches):

                # Per Batch Query
                offset = batch * batch_size
                batch_limit = min(batch_size, row_count - offset)
                query = f"""
                SELECT *
                FROM {table_name}
                LIMIT {batch_limit} OFFSET {offset}
                """

                # Reading Batch
                batch_df = pd.read_sql(query, self.connection)
                rows_exported += batch_df.shape[0]

                # Saving Batch
                output_file = output_dir / f'{table_name}.csv'
                mode = 'w' if first_batch else 'a'
                header = True if first_batch else False
                batch_df.to_csv(output_file, mode=mode,
                                header=header, index=False)

                # Logging Batch Completion
                progress_pct = (rows_exported / row_count) * 100
                self.logger.info(
                    f"  Batch {batch + 1}/{num_batches}: "
                    f"{len(batch_df):,} rows | "
                    f"Total: {rows_exported:,}/{row_count:,} ({progress_pct:.1f}%)"
                )

                # Changing Batch
                first_batch = False

                # Cleaning Memory
                del batch_df
                gc.collect()

            elapsed = (datetime.now() - start_time).total_seconds()

            # Logging Export Completion
            file_size_mb = output_file.stat().st_size / (1024 * 1024)
            self.logger.info(
                f"  Exported {rows_exported:,} rows in {elapsed:.2f}s")
            self.logger.info(
                f"  Saved to {output_file} ({file_size_mb:.2f} MB)")
            self.logger.info(
                f"  Average speed: {rows_exported / elapsed:.0f} rows/sec")

        except Exception as e:
            self.logger.exception(f"Error Exporting {table_name} : {e}")
            raise
