"""
File Inspector
==============

Parse uploaded files and extract structure:
- Entity columns
- Sequence columns
- Signals with units
- Constants from header comments
"""

import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from pathlib import Path

import pandas as pd


@dataclass
class ColumnInfo:
    name: str
    dtype: str
    unit: Optional[str]
    is_constant: bool
    is_entity_id: bool
    is_sequence: bool
    sample_values: List[Any]

    def to_dict(self):
        return {
            "name": str(self.name),
            "dtype": str(self.dtype),
            "unit": self.unit,
            "is_constant": bool(self.is_constant),
            "is_entity_id": bool(self.is_entity_id),
            "is_sequence": bool(self.is_sequence),
            "sample_values": [float(v) if isinstance(v, (int, float)) else str(v) for v in self.sample_values],
        }


@dataclass
class FileInspection:
    """Result of inspecting an uploaded file"""

    # File info
    filename: str
    row_count: int
    column_count: int

    # Detected structure
    entity_column: Optional[str]
    sequence_column: Optional[str]
    entities: List[str]

    # Signals and constants
    signals: List[ColumnInfo]
    constants: Dict[str, float]

    # Validation
    errors: List[str]
    warnings: List[str]

    def to_dict(self):
        return {
            "filename": self.filename,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "entity_column": self.entity_column,
            "sequence_column": self.sequence_column,
            "entities": self.entities,
            "signals": [s.to_dict() for s in self.signals],
            "constants": self.constants,
            "errors": self.errors,
            "warnings": self.warnings,
        }


# Unit patterns - suffix -> unit name
UNIT_PATTERNS = {
    # Pressure
    r'_psi$': 'psi',
    r'_psia$': 'psia',
    r'_psig$': 'psig',
    r'_bar$': 'bar',
    r'_kpa$': 'kPa',
    r'_pa$': 'Pa',

    # Temperature
    r'_F$': '°F',
    r'_C$': '°C',
    r'_K$': 'K',
    r'_degF$': '°F',
    r'_degR$': '°R',
    r'_degC$': '°C',

    # Flow
    r'_gpm$': 'gpm',
    r'_lpm$': 'lpm',
    r'_m3_s$': 'm³/s',
    r'_lbm_s$': 'lbm/s',

    # Length
    r'_in$': 'in',
    r'_ft$': 'ft',
    r'_m$': 'm',
    r'_mm$': 'mm',

    # Speed
    r'_rpm$': 'rpm',
    r'_hz$': 'Hz',
    r'_rad_s$': 'rad/s',

    # Electrical
    r'_V$': 'V',
    r'_A$': 'A',
    r'_W$': 'W',
    r'_kW$': 'kW',

    # Mass/Force
    r'_kg$': 'kg',
    r'_lbm$': 'lbm',
    r'_N$': 'N',

    # Concentration
    r'_mol_L$': 'mol/L',
    r'_mol_m3$': 'mol/m³',

    # Angle
    r'_deg$': '°',
    r'_rad$': 'rad',

    # Other
    r'_pct$': '%',
    r'_percent$': '%',
}

# Entity column patterns
ENTITY_PATTERNS = [
    r'^entity_id$',
    r'^entity$',
    r'^unit_id$',
    r'^unit$',
    r'^equipment_id$',
    r'^asset_id$',
    r'^machine_id$',
    r'^pump_id$',
    r'^engine_id$',
    r'^run_id$',
    r'^batch_id$',
    r'^id$',
]

# Sequence column patterns
SEQUENCE_PATTERNS = [
    r'^timestamp',
    r'^time$',
    r'^datetime',
    r'^date$',
    r'^cycle',
    r'^index$',
    r'^step',
    r'^t$',
    r'_min$',
    r'_sec$',
    r'_hr$',
]


def detect_unit(column_name: str) -> Optional[str]:
    """Detect unit from column name suffix."""
    for pattern, unit in UNIT_PATTERNS.items():
        if re.search(pattern, column_name, re.IGNORECASE):
            return unit
    return None


def detect_entity_column(columns: List[str]) -> Optional[str]:
    """Find entity ID column."""
    for col in columns:
        for pattern in ENTITY_PATTERNS:
            if re.match(pattern, col, re.IGNORECASE):
                return col
    return None


def detect_sequence_column(columns: List[str]) -> Optional[str]:
    """Find timestamp/sequence column."""
    for col in columns:
        for pattern in SEQUENCE_PATTERNS:
            if re.search(pattern, col, re.IGNORECASE):
                return col
    return None


def parse_header_constants(filepath: Path) -> Dict[str, float]:
    """Parse constants from CSV header comments."""
    constants = {}

    if filepath.suffix != '.csv':
        return constants

    try:
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    # Parse: # name = value or # name_unit = value
                    match = re.match(r'#\s*(\w+)\s*=\s*([0-9.eE+-]+)', line)
                    if match:
                        name, value = match.groups()
                        try:
                            constants[name.lower()] = float(value)
                        except ValueError:
                            pass
                else:
                    break
    except Exception:
        pass

    return constants


def inspect_file(filepath: str) -> FileInspection:
    """
    Inspect an uploaded file and extract structure.

    Args:
        filepath: Path to CSV, XLSX, or Parquet file

    Returns:
        FileInspection with all detected info
    """
    filepath = Path(filepath)
    errors = []
    warnings = []

    # Parse header constants (CSV only)
    constants = parse_header_constants(filepath)

    # Read data
    df = None
    try:
        if filepath.suffix == '.csv':
            df = pd.read_csv(filepath, comment='#')
        elif filepath.suffix in ['.xlsx', '.xls']:
            df = pd.read_excel(filepath)
        elif filepath.suffix == '.parquet':
            df = pd.read_parquet(filepath)
        else:
            errors.append(f"Unsupported file type: {filepath.suffix}")
    except Exception as e:
        errors.append(f"Failed to read file: {str(e)}")

    if df is None or df.empty:
        return FileInspection(
            filename=filepath.name,
            row_count=0,
            column_count=0,
            entity_column=None,
            sequence_column=None,
            entities=[],
            signals=[],
            constants=constants,
            errors=errors or ["File is empty"],
            warnings=warnings,
        )

    # Detect entity column
    entity_column = detect_entity_column(df.columns.tolist())
    if not entity_column:
        warnings.append("No entity column detected. Treating as single entity.")

    # Detect sequence column
    sequence_column = detect_sequence_column(df.columns.tolist())
    if not sequence_column:
        warnings.append("No timestamp/sequence column detected.")

    # Get entities
    if entity_column and entity_column in df.columns:
        entities = df[entity_column].unique().tolist()
        entities = [str(e) for e in entities]
    else:
        entities = ['default']

    # Analyze each column
    signals = []
    for col in df.columns:
        # Detect unit
        unit = detect_unit(col)

        # Check if constant (same value throughout)
        is_constant = False
        if df[col].dtype in ['float64', 'int64', 'float32', 'int32']:
            if entity_column and entity_column in df.columns:
                # Constant within each entity
                grouped = df.groupby(entity_column)[col].nunique()
                is_constant = (grouped == 1).all()
            else:
                is_constant = df[col].nunique() == 1

        # Get sample values
        try:
            sample_values = df[col].dropna().head(3).tolist()
            # Convert to JSON-serializable types
            sample_values = [
                float(v) if isinstance(v, (int, float)) else str(v)
                for v in sample_values
            ]
        except Exception:
            sample_values = []

        col_info = ColumnInfo(
            name=col,
            dtype=str(df[col].dtype),
            unit=unit,
            is_constant=is_constant,
            is_entity_id=(col == entity_column),
            is_sequence=(col == sequence_column),
            sample_values=sample_values,
        )
        signals.append(col_info)

    return FileInspection(
        filename=filepath.name,
        row_count=len(df),
        column_count=len(df.columns),
        entity_column=entity_column,
        sequence_column=sequence_column,
        entities=entities,
        signals=signals,
        constants=constants,
        errors=errors,
        warnings=warnings,
    )
