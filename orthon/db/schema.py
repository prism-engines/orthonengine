"""
ORTHON Schema Detection
=======================

Auto-detect what PRISM computed. Build queries from what exists.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import duckdb


@dataclass
class ColumnInfo:
    """Information about a parquet column."""
    name: str
    dtype: str
    nullable: bool = True

    @property
    def is_numeric(self) -> bool:
        numeric_types = ['INTEGER', 'BIGINT', 'FLOAT', 'DOUBLE', 'DECIMAL', 'REAL']
        return any(t in self.dtype.upper() for t in numeric_types)

    @property
    def is_timestamp(self) -> bool:
        return 'TIMESTAMP' in self.dtype.upper() or 'DATE' in self.dtype.upper()

    @property
    def is_string(self) -> bool:
        return 'VARCHAR' in self.dtype.upper() or 'STRING' in self.dtype.upper()


@dataclass
class TableSchema:
    """Schema for a parquet file."""
    name: str
    path: Path
    columns: List[ColumnInfo]
    row_count: int

    @property
    def column_names(self) -> List[str]:
        return [c.name for c in self.columns]

    @property
    def numeric_columns(self) -> List[str]:
        return [c.name for c in self.columns if c.is_numeric]

    @property
    def id_columns(self) -> List[str]:
        """Likely ID columns."""
        return [c.name for c in self.columns if 'id' in c.name.lower()]

    def has_column(self, name: str) -> bool:
        return name in self.column_names


class SchemaDetector:
    """
    Detect schemas from PRISM parquet files.

    No assumptions about what columns exist - discover them.
    """

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.schemas: Dict[str, TableSchema] = {}
        self._detect_all()

    def _detect_all(self):
        """Detect schemas for all parquet files."""
        for pq_file in self.data_dir.glob('*.parquet'):
            schema = self._detect_one(pq_file)
            if schema:
                self.schemas[schema.name] = schema

    def _detect_one(self, path: Path) -> Optional[TableSchema]:
        """Detect schema for one parquet file."""
        try:
            conn = duckdb.connect(':memory:')

            # Get schema
            result = conn.execute(f"DESCRIBE SELECT * FROM '{path}'").fetchall()
            columns = [
                ColumnInfo(name=row[0], dtype=row[1], nullable=row[2] == 'YES')
                for row in result
            ]

            # Get row count
            count = conn.execute(f"SELECT COUNT(*) FROM '{path}'").fetchone()[0]

            conn.close()

            return TableSchema(
                name=path.stem,
                path=path,
                columns=columns,
                row_count=count
            )
        except Exception as e:
            print(f"Warning: Could not read {path}: {e}")
            return None

    def get(self, table: str) -> Optional[TableSchema]:
        """Get schema for a table."""
        return self.schemas.get(table)

    def tables(self) -> List[str]:
        """List all detected tables."""
        return list(self.schemas.keys())

    def summary(self) -> str:
        """Human-readable summary of all schemas."""
        lines = ["PRISM Parquet Schema Summary", "=" * 40]

        for name, schema in self.schemas.items():
            lines.append(f"\n{name}.parquet ({schema.row_count:,} rows)")
            lines.append("-" * 30)
            for col in schema.columns[:20]:  # First 20 columns
                lines.append(f"  {col.name}: {col.dtype}")
            if len(schema.columns) > 20:
                lines.append(f"  ... and {len(schema.columns) - 20} more columns")

        return "\n".join(lines)

    def find_column(self, column_name: str) -> List[str]:
        """Find which tables contain a column."""
        tables = []
        for name, schema in self.schemas.items():
            if schema.has_column(column_name):
                tables.append(name)
        return tables

    def common_columns(self, table1: str, table2: str) -> List[str]:
        """Find columns that exist in both tables (for joins)."""
        s1 = self.schemas.get(table1)
        s2 = self.schemas.get(table2)
        if not s1 or not s2:
            return []
        return list(set(s1.column_names) & set(s2.column_names))


def detect(data_dir: str) -> SchemaDetector:
    """
    Quick schema detection.

    Usage:
        schema = detect('data/')
        print(schema.summary())
        print(schema.get('vector').numeric_columns)
    """
    return SchemaDetector(Path(data_dir))
