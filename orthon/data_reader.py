#!/usr/bin/env python3
"""
PRISM Data Reader & Config Recommender

Analyzes uploaded data and recommends window/stride configuration.
User reviews and confirms — values are then saved to config.

ZERO DEFAULTS POLICY:
    This tool RECOMMENDS values. User MUST confirm.
    No automatic application of recommendations.

Usage:
    # CLI: Analyze and get recommendations
    python data_reader.py ./data.csv
    python data_reader.py ./data.parquet

    # Streamlit: Interactive config builder
    streamlit run data_reader.py
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import numpy as np

try:
    import polars as pl
except ImportError:
    print("ERROR: polars required. pip install polars")
    sys.exit(1)

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class DataProfile:
    """Profile of the uploaded data."""
    n_rows: int
    n_entities: int
    n_signals: int
    n_timestamps: int

    # Per-entity stats
    min_lifecycle: int
    max_lifecycle: int
    mean_lifecycle: float
    median_lifecycle: float

    # Temporal characteristics
    sampling_interval: Optional[float]
    is_regular_sampling: bool

    # Signal characteristics
    signal_names: List[str]
    has_nulls: bool
    null_fraction: float


@dataclass
class WindowRecommendation:
    """Recommended window/stride configuration."""
    window_size: int
    window_stride: int
    n_windows_per_entity: int
    overlap_fraction: float

    # Reasoning
    rationale: str
    confidence: str  # 'high', 'medium', 'low'

    # Alternatives
    conservative: Dict[str, int]  # Larger window, fewer windows
    aggressive: Dict[str, int]    # Smaller window, more windows


@dataclass
class ConfigRecommendation:
    """Complete configuration recommendation."""
    data_profile: DataProfile
    window: WindowRecommendation

    # Additional recommendations
    n_clusters: int
    n_regimes: int
    clustering_method: str

    def to_dict(self) -> dict:
        return {
            'window_size': self.window.window_size,
            'window_stride': self.window.window_stride,
            'n_clusters': self.n_clusters,
            'n_regimes': self.n_regimes,
            'clustering_method': self.clustering_method,
        }

    def to_yaml(self) -> str:
        if not HAS_YAML:
            return json.dumps(self.to_dict(), indent=2)
        return yaml.dump(self.to_dict(), default_flow_style=False)


# =============================================================================
# DATA READER
# =============================================================================

class DataReader:
    """Reads and profiles data for PRISM configuration."""

    SUPPORTED_FORMATS = ['.csv', '.parquet', '.tsv']

    def __init__(self):
        self.df: Optional[pl.DataFrame] = None
        self.profile: Optional[DataProfile] = None

    def read(self, path: Path) -> pl.DataFrame:
        """Read data from file."""
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        suffix = path.suffix.lower()

        if suffix == '.csv':
            self.df = pl.read_csv(path)
        elif suffix == '.tsv':
            self.df = pl.read_csv(path, separator='\t')
        elif suffix == '.parquet':
            self.df = pl.read_parquet(path)
        else:
            raise ValueError(f"Unsupported format: {suffix}. Use: {self.SUPPORTED_FORMATS}")

        return self.df

    def detect_columns(self) -> Dict[str, str]:
        """Auto-detect entity_id, timestamp, signal columns."""
        if self.df is None:
            raise ValueError("No data loaded. Call read() first.")

        columns = self.df.columns
        detected = {}

        # Detect entity_id
        entity_candidates = ['entity_id', 'unit_id', 'id', 'machine_id', 'asset_id',
                            'battery_id', 'engine_id', 'device_id', 'sensor_id']
        for col in columns:
            if col.lower() in [c.lower() for c in entity_candidates]:
                detected['entity_id'] = col
                break

        # Detect timestamp
        time_candidates = ['timestamp', 'time', 'cycle', 'cycles', 't', 'datetime',
                          'date', 'step', 'sample', 'index']
        for col in columns:
            if col.lower() in [c.lower() for c in time_candidates]:
                detected['timestamp'] = col
                break

        # Remaining columns are signals
        known_cols = set(detected.values())
        detected['signals'] = [c for c in columns if c not in known_cols]

        return detected

    def profile_data(self, entity_col: str = None, timestamp_col: str = None) -> DataProfile:
        """Generate data profile."""
        if self.df is None:
            raise ValueError("No data loaded. Call read() first.")

        # Auto-detect columns if not specified
        detected = self.detect_columns()
        entity_col = entity_col or detected.get('entity_id')
        timestamp_col = timestamp_col or detected.get('timestamp')
        signal_cols = detected.get('signals', [])

        if not entity_col:
            # Assume single entity
            self.df = self.df.with_columns(pl.lit('entity_1').alias('entity_id'))
            entity_col = 'entity_id'

        if not timestamp_col:
            # Assume row index is timestamp
            self.df = self.df.with_row_index('timestamp')
            timestamp_col = 'timestamp'

        # Basic counts
        n_rows = len(self.df)
        n_entities = self.df[entity_col].n_unique()
        n_signals = len(signal_cols)
        n_timestamps = self.df[timestamp_col].n_unique()

        # Lifecycle per entity
        lifecycle = self.df.group_by(entity_col).agg(
            pl.col(timestamp_col).count().alias('lifecycle')
        )['lifecycle'].to_numpy()

        min_lifecycle = int(np.min(lifecycle))
        max_lifecycle = int(np.max(lifecycle))
        mean_lifecycle = float(np.mean(lifecycle))
        median_lifecycle = float(np.median(lifecycle))

        # Sampling interval (check if regular)
        timestamps = self.df.filter(
            pl.col(entity_col) == self.df[entity_col][0]
        )[timestamp_col].to_numpy()

        if len(timestamps) > 1:
            diffs = np.diff(timestamps)
            sampling_interval = float(np.median(diffs))
            is_regular = np.std(diffs) < 0.1 * sampling_interval if sampling_interval > 0 else False
        else:
            sampling_interval = None
            is_regular = True

        # Null check
        null_counts = self.df.null_count().to_numpy().flatten()
        has_nulls = np.any(null_counts > 0)
        null_fraction = float(np.sum(null_counts) / (n_rows * len(self.df.columns)))

        self.profile = DataProfile(
            n_rows=n_rows,
            n_entities=n_entities,
            n_signals=n_signals,
            n_timestamps=n_timestamps,
            min_lifecycle=min_lifecycle,
            max_lifecycle=max_lifecycle,
            mean_lifecycle=mean_lifecycle,
            median_lifecycle=median_lifecycle,
            sampling_interval=sampling_interval,
            is_regular_sampling=is_regular,
            signal_names=signal_cols,
            has_nulls=has_nulls,
            null_fraction=null_fraction,
        )

        return self.profile


# =============================================================================
# CONFIG RECOMMENDER
# =============================================================================

class ConfigRecommender:
    """Recommends PRISM configuration based on data profile."""

    # Target number of windows per entity
    TARGET_WINDOWS_MIN = 10
    TARGET_WINDOWS_MAX = 50
    TARGET_WINDOWS_OPTIMAL = 20

    # Overlap recommendations
    OVERLAP_DEFAULT = 0.5  # 50% overlap
    OVERLAP_HIGH = 0.75    # For short lifecycles
    OVERLAP_LOW = 0.25     # For very long lifecycles

    def __init__(self, profile: DataProfile):
        self.profile = profile

    def recommend_window(self) -> WindowRecommendation:
        """Recommend window size and stride."""

        # Use median lifecycle as reference (robust to outliers)
        lifecycle = self.profile.median_lifecycle

        # Determine overlap based on lifecycle length
        if lifecycle < 100:
            overlap = self.OVERLAP_HIGH  # Need more overlap for short data
            confidence = 'low'
        elif lifecycle < 500:
            overlap = self.OVERLAP_DEFAULT
            confidence = 'medium'
        else:
            overlap = self.OVERLAP_DEFAULT
            confidence = 'high'

        # Calculate window size to get ~TARGET_WINDOWS_OPTIMAL windows
        # n_windows ≈ (lifecycle - window) / stride + 1
        # With overlap: stride = window * (1 - overlap)
        # n_windows ≈ (lifecycle - window) / (window * (1 - overlap)) + 1
        # Solving for window:
        # window ≈ lifecycle / (n_windows * (1 - overlap) + overlap)

        window_size = int(lifecycle / (self.TARGET_WINDOWS_OPTIMAL * (1 - overlap) + overlap))

        # Ensure window is reasonable
        window_size = max(10, window_size)  # At least 10 samples
        window_size = min(int(lifecycle * 0.5), window_size)  # At most 50% of lifecycle

        # Calculate stride
        window_stride = max(1, int(window_size * (1 - overlap)))

        # Calculate actual windows
        n_windows = max(1, int((lifecycle - window_size) / window_stride) + 1)
        actual_overlap = 1 - (window_stride / window_size) if window_size > 0 else 0

        # Build rationale
        rationale = (
            f"Based on median lifecycle of {lifecycle:.0f} samples:\n"
            f"  - Window size {window_size} captures meaningful patterns\n"
            f"  - Stride {window_stride} ({actual_overlap*100:.0f}% overlap) gives ~{n_windows} windows\n"
            f"  - Shortest entity ({self.profile.min_lifecycle} samples) gets "
            f"~{max(1, (self.profile.min_lifecycle - window_size) // window_stride + 1)} windows"
        )

        # Check if shortest entity is too short
        min_windows = max(1, (self.profile.min_lifecycle - window_size) // window_stride + 1)
        if min_windows < 5:
            confidence = 'low'
            rationale += f"\n  WARNING: Shortest entity only gets {min_windows} windows"

        # Conservative alternative (larger window, fewer windows)
        conservative_window = int(window_size * 1.5)
        conservative_stride = int(conservative_window * (1 - overlap))

        # Aggressive alternative (smaller window, more windows)
        aggressive_window = int(window_size * 0.67)
        aggressive_stride = int(aggressive_window * (1 - overlap))

        return WindowRecommendation(
            window_size=window_size,
            window_stride=window_stride,
            n_windows_per_entity=n_windows,
            overlap_fraction=actual_overlap,
            rationale=rationale,
            confidence=confidence,
            conservative={'window_size': conservative_window, 'window_stride': conservative_stride},
            aggressive={'window_size': aggressive_window, 'window_stride': aggressive_stride},
        )

    def recommend_clustering(self) -> Tuple[int, str]:
        """Recommend n_clusters and method."""

        n_entities = self.profile.n_entities

        if n_entities == 1:
            # Single entity - clustering on windows
            n_clusters = 3  # Healthy / Degraded / Critical
            method = 'kmeans'
        elif n_entities < 10:
            n_clusters = min(3, n_entities)
            method = 'kmeans'
        elif n_entities < 50:
            n_clusters = 3
            method = 'kmeans'
        else:
            n_clusters = 5
            method = 'kmeans'

        return n_clusters, method

    def recommend_regimes(self) -> int:
        """Recommend n_regimes for dynamics."""
        # Typically: Healthy, Degraded, Critical
        return 3

    def recommend(self) -> ConfigRecommendation:
        """Generate complete configuration recommendation."""

        window = self.recommend_window()
        n_clusters, clustering_method = self.recommend_clustering()
        n_regimes = self.recommend_regimes()

        return ConfigRecommendation(
            data_profile=self.profile,
            window=window,
            n_clusters=n_clusters,
            n_regimes=n_regimes,
            clustering_method=clustering_method,
        )


# =============================================================================
# CLI OUTPUT
# =============================================================================

def print_profile(profile: DataProfile):
    """Print data profile."""
    print("\n" + "=" * 60)
    print("DATA PROFILE")
    print("=" * 60)
    print(f"  Rows:        {profile.n_rows:,}")
    print(f"  Entities:    {profile.n_entities}")
    print(f"  Signals:     {profile.n_signals}")
    print(f"  Timestamps:  {profile.n_timestamps:,}")
    print()
    print(f"  Lifecycle (samples per entity):")
    print(f"    Min:    {profile.min_lifecycle}")
    print(f"    Max:    {profile.max_lifecycle}")
    print(f"    Mean:   {profile.mean_lifecycle:.1f}")
    print(f"    Median: {profile.median_lifecycle:.1f}")
    print()
    print(f"  Sampling: {'Regular' if profile.is_regular_sampling else 'Irregular'}")
    if profile.sampling_interval:
        print(f"    Interval: {profile.sampling_interval}")
    print()
    print(f"  Nulls: {'Yes' if profile.has_nulls else 'No'} ({profile.null_fraction*100:.2f}%)")
    print()
    print(f"  Signals: {profile.signal_names[:5]}{'...' if len(profile.signal_names) > 5 else ''}")


def print_recommendation(rec: ConfigRecommendation):
    """Print configuration recommendation."""
    print("\n" + "=" * 60)
    print("CONFIGURATION RECOMMENDATION")
    print("=" * 60)

    w = rec.window
    print(f"\n  WINDOWING ({w.confidence.upper()} confidence)")
    print(f"  ---------------------------------")
    print(f"    window_size:  {w.window_size}")
    print(f"    window_stride: {w.window_stride}")
    print(f"    overlap:      {w.overlap_fraction*100:.0f}%")
    print(f"    windows/entity: ~{w.n_windows_per_entity}")
    print()
    print(f"  Rationale:")
    for line in w.rationale.split('\n'):
        print(f"    {line}")

    print(f"\n  ALTERNATIVES")
    print(f"  ---------------------------------")
    print(f"    Conservative (fewer windows, smoother):")
    print(f"      window_size: {w.conservative['window_size']}, stride: {w.conservative['window_stride']}")
    print(f"    Aggressive (more windows, finer detail):")
    print(f"      window_size: {w.aggressive['window_size']}, stride: {w.aggressive['window_stride']}")

    print(f"\n  OTHER PARAMETERS")
    print(f"  ---------------------------------")
    print(f"    n_clusters: {rec.n_clusters}")
    print(f"    n_regimes:  {rec.n_regimes}")
    print(f"    clustering: {rec.clustering_method}")

    print("\n" + "=" * 60)
    print("RECOMMENDED CONFIG (copy to prism.yaml)")
    print("=" * 60)
    print()
    print(rec.to_yaml())


def save_config(rec: ConfigRecommendation, path: Path):
    """Save configuration to file."""
    config = rec.to_dict()

    path = Path(path)

    if path.suffix in ['.yaml', '.yml']:
        if not HAS_YAML:
            print("WARNING: pyyaml not installed, saving as JSON")
            path = path.with_suffix('.json')
            with open(path, 'w') as f:
                json.dump(config, f, indent=2)
        else:
            with open(path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
    else:
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)

    print(f"\nConfig saved to: {path}")


# =============================================================================
# STREAMLIT UI
# =============================================================================

def run_streamlit():
    """Run Streamlit UI for interactive config building."""
    try:
        import streamlit as st
    except ImportError:
        print("ERROR: streamlit required. pip install streamlit")
        print("       Or use CLI: python data_reader.py ./data.csv")
        sys.exit(1)

    st.set_page_config(page_title="PRISM Config Builder", layout="wide")
    st.title("PRISM Configuration Builder")
    st.markdown("Upload your data to get recommended window/stride configuration.")

    # File upload
    uploaded_file = st.file_uploader(
        "Upload data file",
        type=['csv', 'parquet', 'tsv'],
        help="CSV, Parquet, or TSV file with your time series data"
    )

    if uploaded_file is None:
        st.info("Upload a data file to get started")
        st.markdown("""
        **Expected format:**
        - `entity_id` column (optional if single entity)
        - `timestamp` or `cycle` column
        - Signal/measurement columns

        **Example:**
        ```
        entity_id,timestamp,temperature,pressure,vibration
        engine_1,1,85.2,1013.5,0.023
        engine_1,2,85.4,1013.2,0.025
        ...
        ```
        """)
        return

    # Read data
    reader = DataReader()

    try:
        # Save uploaded file temporarily
        suffix = Path(uploaded_file.name).suffix
        temp_path = Path(f"/tmp/upload{suffix}")
        with open(temp_path, 'wb') as f:
            f.write(uploaded_file.getvalue())

        df = reader.read(temp_path)
        st.success(f"Loaded {len(df):,} rows")
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return

    # Show data preview
    with st.expander("Data Preview", expanded=False):
        st.dataframe(df.head(100).to_pandas())

    # Column detection
    detected = reader.detect_columns()

    col1, col2 = st.columns(2)
    with col1:
        entity_col = st.selectbox(
            "Entity ID column",
            options=['(auto-detect)'] + df.columns,
            index=0 if detected.get('entity_id') else 0,
            help="Column identifying each entity/unit"
        )
        if entity_col == '(auto-detect)':
            entity_col = detected.get('entity_id')

    with col2:
        timestamp_col = st.selectbox(
            "Timestamp column",
            options=['(auto-detect)'] + df.columns,
            index=0,
            help="Column with time/cycle information"
        )
        if timestamp_col == '(auto-detect)':
            timestamp_col = detected.get('timestamp')

    # Profile data
    try:
        profile = reader.profile_data(entity_col, timestamp_col)
    except Exception as e:
        st.error(f"Error profiling data: {e}")
        return

    # Show profile
    st.subheader("Data Profile")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Entities", profile.n_entities)
    col2.metric("Signals", profile.n_signals)
    col3.metric("Median Lifecycle", f"{profile.median_lifecycle:.0f}")
    col4.metric("Total Rows", f"{profile.n_rows:,}")

    # Get recommendation
    recommender = ConfigRecommender(profile)
    rec = recommender.recommend()

    # Editable configuration
    st.subheader("Configuration")
    st.markdown(f"**Confidence:** {rec.window.confidence.upper()}")
    st.markdown(rec.window.rationale)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Windowing**")
        window_size = st.number_input(
            "Window Size",
            min_value=5,
            max_value=int(profile.max_lifecycle * 0.9),
            value=rec.window.window_size,
            help="Number of samples per window"
        )

        window_stride = st.number_input(
            "Window Stride",
            min_value=1,
            max_value=window_size,
            value=rec.window.window_stride,
            help="Samples between window starts"
        )

        overlap = 1 - (window_stride / window_size) if window_size > 0 else 0
        st.info(f"Overlap: {overlap*100:.0f}%")

    with col2:
        st.markdown("**Clustering & Regimes**")
        n_clusters = st.number_input(
            "Number of Clusters",
            min_value=2,
            max_value=10,
            value=rec.n_clusters,
            help="For geometry layer"
        )

        n_regimes = st.number_input(
            "Number of Regimes",
            min_value=2,
            max_value=10,
            value=rec.n_regimes,
            help="For dynamics layer"
        )

    # Preview resulting windows
    n_windows = max(1, int((profile.median_lifecycle - window_size) / window_stride) + 1)
    min_entity_windows = max(1, int((profile.min_lifecycle - window_size) / window_stride) + 1)

    st.markdown(f"""
    **Preview:**
    - Median entity: ~{n_windows} windows
    - Shortest entity: ~{min_entity_windows} windows
    """)

    if min_entity_windows < 5:
        st.warning(f"Shortest entity only gets {min_entity_windows} windows. Consider smaller window size.")

    # Generate config
    st.subheader("Generated Configuration")

    config = {
        'window_size': window_size,
        'window_stride': window_stride,
        'n_clusters': n_clusters,
        'n_regimes': n_regimes,
        'clustering_method': 'kmeans',
    }

    config_yaml = yaml.dump(config, default_flow_style=False) if HAS_YAML else json.dumps(config, indent=2)

    st.code(config_yaml, language='yaml')

    # Download button
    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            "Download prism.yaml",
            data=config_yaml,
            file_name="prism.yaml",
            mime="text/yaml"
        )

    with col2:
        st.download_button(
            "Download prism.json",
            data=json.dumps(config, indent=2),
            file_name="prism.json",
            mime="application/json"
        )

    st.success("Configure these values, then run PRISM with `--config prism.yaml`")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='PRISM Data Reader & Config Recommender',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
    # Analyze data and get recommendations
    python data_reader.py ./data.csv
    python data_reader.py ./data.parquet

    # Save config to file
    python data_reader.py ./data.csv --output prism.yaml

    # Interactive Streamlit UI
    streamlit run data_reader.py

ZERO DEFAULTS POLICY:
    This tool RECOMMENDS configuration values based on your data.
    You MUST review and confirm before using with PRISM.
    PRISM will not run without explicit configuration.
        """
    )

    parser.add_argument('data_file', nargs='?', help='Path to data file (CSV, Parquet, TSV)')
    parser.add_argument('--output', '-o', help='Save config to file (yaml or json)')
    parser.add_argument('--entity-col', help='Column name for entity ID')
    parser.add_argument('--timestamp-col', help='Column name for timestamp')
    parser.add_argument('--streamlit', action='store_true', help='Run Streamlit UI')

    args = parser.parse_args()

    # Check if running via streamlit
    if args.streamlit or (not args.data_file and 'streamlit' in sys.modules):
        run_streamlit()
        return

    if not args.data_file:
        parser.print_help()
        print("\nTip: Run `streamlit run data_reader.py` for interactive UI")
        sys.exit(1)

    # CLI mode
    reader = DataReader()

    try:
        reader.read(Path(args.data_file))
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    profile = reader.profile_data(args.entity_col, args.timestamp_col)
    print_profile(profile)

    recommender = ConfigRecommender(profile)
    rec = recommender.recommend()
    print_recommendation(rec)

    if args.output:
        save_config(rec, Path(args.output))
    else:
        print("\nTip: Use --output prism.yaml to save configuration")


if __name__ == '__main__':
    main()
