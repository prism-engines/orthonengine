"""
ORTHON Compute Interface
========================

Bridge between ORTHON (UI/config) and PRISM (compute).

ORTHON owns:
- Streamlit UI
- Config building (data_reader.py)
- Domain configs
- Visualization
- Report generation

PRISM owns:
- Computation only
- 5 parquet outputs
- Zero UI, zero config opinions
- Fails without config
"""

import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, Union
import json

import polars as pl

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


class PRISMError(Exception):
    """Error from PRISM computation."""
    pass


class ConfigError(Exception):
    """Configuration validation error."""
    pass


# Required config keys
REQUIRED_CONFIG = [
    'window_size',
    'window_stride',
]

# Optional config keys with no defaults (PRISM must handle)
OPTIONAL_CONFIG = [
    'n_clusters',
    'n_regimes',
    'clustering_method',
]


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration before sending to PRISM.

    ZERO DEFAULTS POLICY:
        ORTHON does not apply defaults.
        All required config must be explicitly set.
        PRISM fails without complete config.

    Raises:
        ConfigError: If config is invalid or incomplete
    """
    missing = [k for k in REQUIRED_CONFIG if k not in config]
    if missing:
        raise ConfigError(
            f"Missing required config keys: {missing}\n"
            f"Use orthon-config to analyze your data and get recommendations."
        )

    # Validate types
    if not isinstance(config.get('window_size'), int) or config['window_size'] < 1:
        raise ConfigError("window_size must be a positive integer")

    if not isinstance(config.get('window_stride'), int) or config['window_stride'] < 1:
        raise ConfigError("window_stride must be a positive integer")

    if config['window_stride'] > config['window_size']:
        raise ConfigError("window_stride cannot be larger than window_size")


def write_temp_parquet(data: pl.DataFrame) -> Path:
    """Write DataFrame to temporary parquet file."""
    temp_dir = Path(tempfile.mkdtemp(prefix='orthon_'))
    data_path = temp_dir / 'input.parquet'
    data.write_parquet(data_path)
    return data_path


def write_temp_config(config: Dict[str, Any]) -> Path:
    """Write config to temporary file."""
    temp_dir = Path(tempfile.mkdtemp(prefix='orthon_'))

    if HAS_YAML:
        config_path = temp_dir / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    else:
        config_path = temp_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

    return config_path


def run_prism(
    data: pl.DataFrame,
    config: Dict[str, Any],
    output_dir: Optional[Path] = None,
) -> Dict[str, pl.DataFrame]:
    """
    Send data to PRISM, get parquets back.

    Args:
        data: Input DataFrame with entity_id, timestamp, and signal columns
        config: Configuration dict (must include window_size, window_stride)
        output_dir: Optional directory for output files (uses temp if not specified)

    Returns:
        Dict with keys: observations, vector, geometry, dynamics, physics
        Each value is a Polars DataFrame

    Raises:
        ConfigError: If config is invalid
        PRISMError: If PRISM computation fails
    """
    # Validate config (ORTHON's job to ensure it's complete)
    validate_config(config)

    # Write temp files
    data_path = write_temp_parquet(data)
    config_path = write_temp_config(config)

    # Determine output directory
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix='orthon_output_'))
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Call PRISM
    try:
        import prism
        result_dir = prism.run(
            data=str(data_path),
            config=str(config_path),
            output_dir=str(output_dir),
        )
    except ImportError:
        raise PRISMError(
            "prism-engine not installed.\n"
            "Install with: pip install prism-engine\n"
            "Or: pip install git+https://github.com/prism-engines/prism.git"
        )
    except Exception as e:
        raise PRISMError(f"PRISM computation failed: {e}")

    # Load results
    result_path = Path(result_dir) if result_dir else output_dir

    outputs = {}
    for name in ['observations', 'vector', 'geometry', 'dynamics', 'physics']:
        parquet_path = result_path / f'{name}.parquet'
        if parquet_path.exists():
            outputs[name] = pl.read_parquet(parquet_path)
        else:
            outputs[name] = pl.DataFrame()  # Empty if not generated

    return outputs


def run_prism_from_files(
    data_path: Union[str, Path],
    config_path: Union[str, Path],
    output_dir: Optional[Path] = None,
) -> Dict[str, pl.DataFrame]:
    """
    Run PRISM from file paths.

    Args:
        data_path: Path to input parquet/csv file
        config_path: Path to config yaml/json file
        output_dir: Optional directory for output files

    Returns:
        Dict with PRISM output DataFrames
    """
    # Load data
    data_path = Path(data_path)
    if data_path.suffix == '.csv':
        data = pl.read_csv(data_path)
    else:
        data = pl.read_parquet(data_path)

    # Load config
    config_path = Path(config_path)
    with open(config_path) as f:
        if config_path.suffix in ['.yaml', '.yml']:
            if not HAS_YAML:
                raise ConfigError("pyyaml required for YAML config files")
            config = yaml.safe_load(f)
        else:
            config = json.load(f)

    return run_prism(data, config, output_dir)


class PRISMRunner:
    """
    Stateful PRISM runner for Streamlit integration.

    Holds data, config, and results for interactive exploration.
    """

    def __init__(self):
        self.data: Optional[pl.DataFrame] = None
        self.config: Optional[Dict[str, Any]] = None
        self.results: Optional[Dict[str, pl.DataFrame]] = None
        self.output_dir: Optional[Path] = None

    def load_data(self, path: Union[str, Path]) -> pl.DataFrame:
        """Load data from file."""
        path = Path(path)
        if path.suffix == '.csv':
            self.data = pl.read_csv(path)
        elif path.suffix == '.tsv':
            self.data = pl.read_csv(path, separator='\t')
        else:
            self.data = pl.read_parquet(path)
        return self.data

    def set_config(self, config: Dict[str, Any]) -> None:
        """Set and validate config."""
        validate_config(config)
        self.config = config

    def run(self, output_dir: Optional[Path] = None) -> Dict[str, pl.DataFrame]:
        """Run PRISM computation."""
        if self.data is None:
            raise PRISMError("No data loaded. Call load_data() first.")
        if self.config is None:
            raise ConfigError("No config set. Call set_config() first.")

        self.output_dir = output_dir
        self.results = run_prism(self.data, self.config, output_dir)
        return self.results

    @property
    def observations(self) -> Optional[pl.DataFrame]:
        return self.results.get('observations') if self.results else None

    @property
    def vector(self) -> Optional[pl.DataFrame]:
        return self.results.get('vector') if self.results else None

    @property
    def geometry(self) -> Optional[pl.DataFrame]:
        return self.results.get('geometry') if self.results else None

    @property
    def dynamics(self) -> Optional[pl.DataFrame]:
        return self.results.get('dynamics') if self.results else None

    @property
    def physics(self) -> Optional[pl.DataFrame]:
        return self.results.get('physics') if self.results else None
