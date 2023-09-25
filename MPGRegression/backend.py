import os
import sqlite3
from datetime import datetime

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig


class DatabaseManager:
    def __init__(self, db_name: str, db_aggregation):
        if db_aggregation:
            self.conn = sqlite3.connect(os.path.join(get_original_cwd(), db_name))
        else:
            self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()

    def execute(self, query: str, parameters=None):
        """Execute a query.

        Args:
            query: The query to run against the database
            parameters: The parameters to induce in to the query
        """
        if parameters:
            self.cursor.execute(query, parameters)
        else:
            self.cursor.execute(query)
        self.conn.commit()

    def insert(self, table: str, columns: tuple, values: tuple):
        """Inserts a row into a specified table.

        Args:
            table: The name of the table.
            columns: A tuple of column names.
            values: A tuple of values.

        Return:
            The ID of the inserted row.
        """
        # Construct the SQL query
        cols = ", ".join(
            columns
        )  # Converts the tuple of columns to a comma-separated string
        placeholders = ", ".join(
            "?" for _ in values
        )  # Create placeholders for each value
        query = f"INSERT INTO {table} ({cols}) VALUES ({placeholders})"

        # Execute the query
        self.execute(query, values)

        # Return the ID of the inserted row
        return self.cursor.lastrowid

    def close(self):
        """Close the db connection"""
        self.conn.close()


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Main function for testing the db setup.

    Args:
        cfg: Hydra config object
    """
    db_manager = DatabaseManager(cfg.database.name, cfg.database.aggregate)

    for table in cfg.tables:
        db_manager.execute(table.query)

    # insert data in to the Datasets table
    dataset_columns = [
        "DatasetURL",
        "SourceColumns",
        "TargetColumn",
        "DropNA",
        "OneHotEncoding",
        "SplitFraction",
        "SplitRandomState",
    ]

    dataset_values = [
        cfg.data.csv_url,
        str(cfg.data.csv_columns),
        cfg.data.label_column,
        True,
        True,
        cfg.data.split_frac,
        cfg.data.split_random_state,
    ]

    db_manager.close()


if __name__ == "__main__":
    main()
