"""
ORTHON DuckDB Connection Manager
================================

Zero calculations. Just connect and query parquet.
"""

import duckdb
from pathlib import Path
from typing import Optional, Union
import polars as pl


class OrthonDB:
    """
    DuckDB connection manager for ORTHON.

    Reads PRISM parquet files. Never computes metrics.
    """

    # Expected PRISM outputs
    PARQUET_FILES = [
        'data.parquet',
        'vector.parquet',
        'geometry.parquet',
        'dynamics.parquet',
        'physics.parquet',
    ]

    def __init__(self, data_dir: Union[str, Path], memory: bool = True):
        """
        Initialize connection to PRISM parquet files.

        Args:
            data_dir: Directory containing PRISM parquet outputs
            memory: If True, use in-memory DuckDB. If False, create file.
        """
        self.data_dir = Path(data_dir)
        self._validate_data_dir()

        # Connect
        if memory:
            self.conn = duckdb.connect(':memory:')
        else:
            db_path = self.data_dir / 'orthon.duckdb'
            self.conn = duckdb.connect(str(db_path))

        # Register parquet files as views for easy querying
        self._register_views()

    def _validate_data_dir(self):
        """Check that PRISM outputs exist."""
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        missing = []
        for f in self.PARQUET_FILES:
            if not (self.data_dir / f).exists():
                missing.append(f)

        if missing:
            print(f"Warning: Missing parquet files: {missing}")

    def _register_views(self):
        """Register parquet files as DuckDB views."""
        for f in self.PARQUET_FILES:
            path = self.data_dir / f
            if path.exists():
                view_name = f.replace('.parquet', '')
                self.conn.execute(f"""
                    CREATE OR REPLACE VIEW {view_name} AS
                    SELECT * FROM '{path}'
                """)

    def query(self, sql: str) -> pl.DataFrame:
        """
        Execute SQL and return Polars DataFrame.

        Args:
            sql: SQL query string

        Returns:
            Polars DataFrame with results
        """
        return self.conn.execute(sql).pl()

    def query_df(self, sql: str):
        """Execute SQL and return pandas DataFrame."""
        return self.conn.execute(sql).fetchdf()

    def query_arrow(self, sql: str):
        """Execute SQL and return Arrow table."""
        return self.conn.execute(sql).fetch_arrow_table()

    def schema(self, table: str) -> pl.DataFrame:
        """
        Get schema for a parquet file.

        Args:
            table: Table name (e.g., 'vector', 'dynamics')

        Returns:
            DataFrame with column names and types
        """
        return self.query(f"DESCRIBE {table}")

    def tables(self) -> list:
        """List all available tables/views."""
        result = self.conn.execute("SHOW TABLES").fetchall()
        return [r[0] for r in result]

    def columns(self, table: str) -> list:
        """Get column names for a table."""
        schema = self.schema(table)
        return schema['column_name'].to_list()

    def sample(self, table: str, n: int = 5) -> pl.DataFrame:
        """Quick sample from a table."""
        return self.query(f"SELECT * FROM {table} LIMIT {n}")

    def count(self, table: str) -> int:
        """Row count for a table."""
        result = self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        return result[0]

    def entities(self, table: str = 'vector') -> list:
        """Get unique entity IDs."""
        result = self.query(f"SELECT DISTINCT entity_id FROM {table} ORDER BY entity_id")
        return result['entity_id'].to_list()

    def close(self):
        """Close the connection."""
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Convenience function
def connect(data_dir: Union[str, Path], memory: bool = True) -> OrthonDB:
    """
    Quick connect to PRISM parquet files.

    Usage:
        db = connect('data/')
        result = db.query("SELECT * FROM vector WHERE entity_id = 'engine_1'")
    """
    return OrthonDB(data_dir, memory)
