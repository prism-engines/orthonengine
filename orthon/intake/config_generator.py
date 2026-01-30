"""
ORTHON Config Generator
=======================

Generates default PRISM configuration based on detected units and signal types.
Uses the engine gating system from prism_config.yaml.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field, asdict
import json


# =============================================================================
# UNIT → CATEGORY MAPPING (from prism_config.yaml)
# =============================================================================

UNIT_TO_CATEGORY: Dict[str, str] = {
    # Vibration
    'g': 'vibration', 'm/s²': 'vibration', 'mm/s': 'vibration',
    'in/s': 'vibration', 'ips': 'vibration', 'mil': 'vibration',
    'μm': 'vibration', 'um': 'vibration',

    # Rotation
    'RPM': 'rotation', 'rpm': 'rotation', 'rad/s': 'rotation',

    # Force
    'N': 'force', 'kN': 'force', 'Nm': 'force', 'lbf': 'force',
    'MPa': 'force', 'GPa': 'force',

    # Electrical
    'A': 'electrical_current', 'mA': 'electrical_current',
    'μA': 'electrical_current', 'uA': 'electrical_current',
    'V': 'electrical_voltage', 'mV': 'electrical_voltage', 'kV': 'electrical_voltage',
    'W': 'electrical_power', 'kW': 'electrical_power', 'MW': 'electrical_power',
    'VA': 'electrical_power', 'VAR': 'electrical_power', 'PF': 'electrical_power',
    'Ω': 'electrical_impedance', 'ohm': 'electrical_impedance',

    # Flow
    'm³/s': 'flow_volume', 'L/s': 'flow_volume', 'L/min': 'flow_volume',
    'GPM': 'flow_volume', 'gpm': 'flow_volume', 'CFM': 'flow_volume',
    'kg/s': 'flow_mass', 'kg/hr': 'flow_mass', 'lb/hr': 'flow_mass', 'g/s': 'flow_mass',

    # Velocity
    'm/s': 'velocity', 'ft/s': 'velocity', 'km/h': 'velocity', 'mph': 'velocity',

    # Pressure
    'Pa': 'pressure', 'kPa': 'pressure', 'MPa': 'pressure', 'bar': 'pressure',
    'psi': 'pressure', 'PSI': 'pressure', 'atm': 'pressure',
    'mmHg': 'pressure', 'inH2O': 'pressure',

    # Temperature
    '°C': 'temperature', 'C': 'temperature', '°F': 'temperature',
    'F': 'temperature', 'K': 'temperature', 'degC': 'temperature', 'degF': 'temperature',

    # Heat Transfer
    'W/m²': 'heat_transfer', 'W/(m·K)': 'heat_transfer', 'BTU/hr': 'heat_transfer',

    # Chemical
    'mol/L': 'concentration', 'M': 'concentration', 'mmol/L': 'concentration',
    'ppm': 'concentration', 'ppb': 'concentration', 'mg/L': 'concentration',
    'wt%': 'concentration', 'mol%': 'concentration',
    'pH': 'ph',

    # Thermodynamic
    'J/mol': 'molar_properties', 'kJ/mol': 'molar_properties',
    'J/kg': 'specific_properties', 'kJ/kg': 'specific_properties',

    # Control
    '%': 'control', 'percent': 'control',

    # Dimensionless
    'dimensionless': 'dimensionless', 'ratio': 'dimensionless',
    'unitless': 'dimensionless', 'count': 'dimensionless',
}


# =============================================================================
# ENGINE GATING (from prism_config.yaml)
# =============================================================================

# Engines that always run (no unit requirements)
# NOTE: lyapunov removed from signal engines - it's computed in dynamics_runner
# as a phase-space metric, not a simple signal statistic
CORE_ENGINES = [
    'hurst', 'entropy', 'garch', 'lof', 'clustering', 'pca',
    'granger', 'transfer_entropy', 'cointegration', 'dmd', 'fft', 'wavelet',
    'hilbert', 'rqa', 'mst', 'mutual_info', 'copula', 'dtw', 'embedding',
    'attractor', 'basin', 'acf_decay', 'spectral_slope', 'entropy_rate',
    'convex_hull', 'divergence', 'umap', 'modes',
]

# Domain engines with their required categories
DOMAIN_ENGINES: Dict[str, List[str]] = {
    # Mechanical
    'bearing_fault': ['vibration'],
    'gear_mesh': ['vibration', 'rotation'],
    'modal_analysis': ['vibration'],
    'rotor_dynamics': ['vibration', 'rotation'],
    'fatigue': ['force'],

    # Electrical
    'motor_signature': ['electrical_current'],
    'power_quality': ['electrical_voltage'],
    'impedance': ['electrical_voltage', 'electrical_current'],

    # Fluids
    'navier_stokes': ['velocity'],
    'turbulence_spectrum': ['velocity'],
    'reynolds_stress': ['velocity'],
    'vorticity': ['velocity'],
    'two_phase_flow': ['flow_volume'],

    # Thermal
    'heat_equation': ['temperature'],
    'convection': ['temperature', 'velocity'],
    'radiation': ['temperature'],
    'stefan_problem': ['temperature'],
    'heat_exchanger': ['temperature', 'flow_mass'],

    # Thermo
    'phase_equilibria': ['temperature', 'pressure'],
    'equation_of_state': ['temperature', 'pressure'],
    'fugacity': ['temperature', 'pressure'],
    'exergy': ['temperature', 'pressure'],
    'activity_models': ['concentration', 'temperature'],

    # Chemical
    'reaction_kinetics': ['concentration'],
    'electrochemistry': ['electrical_voltage', 'concentration'],
    'separations': ['concentration'],

    # Control
    'transfer_function': ['control'],
    'kalman': ['control'],
    'stability': ['control'],

    # Process
    'reactor_ode': ['concentration', 'temperature'],
    'distillation': ['concentration', 'temperature', 'pressure'],
    'crystallization': ['concentration', 'temperature'],
}


@dataclass
class SignalConfig:
    """Configuration for a single signal."""
    name: str
    unit: str
    category: str
    description: Optional[str] = None


@dataclass
class PrismJobConfig:
    """Complete PRISM job configuration."""
    # Dataset info
    dataset_name: str
    entity_count: int
    signal_count: int
    observation_count: int

    # Signals
    signals: List[SignalConfig] = field(default_factory=list)

    # Detected categories
    categories: Set[str] = field(default_factory=set)

    # Engines to run
    core_engines: List[str] = field(default_factory=list)
    domain_engines: List[str] = field(default_factory=list)

    # Index info
    index_type: str = 'integer_sequence'
    index_unit: Optional[str] = None
    sampling_rate: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        d['categories'] = list(d['categories'])
        return d

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


def detect_category(unit: str) -> str:
    """Map a unit to its category."""
    if unit is None:
        return 'unknown'
    return UNIT_TO_CATEGORY.get(unit, 'unknown')


def get_enabled_engines(categories: Set[str]) -> Dict[str, List[str]]:
    """
    Determine which engines to run based on detected categories.

    Returns:
        Dict with 'core' and 'domain' engine lists
    """
    # Core engines always run
    core = CORE_ENGINES.copy()

    # Check domain engines
    domain = []
    for engine, required_cats in DOMAIN_ENGINES.items():
        if all(cat in categories for cat in required_cats):
            domain.append(engine)

    return {'core': core, 'domain': domain}


def generate_config(
    observations_path: Path,
    dataset_name: Optional[str] = None,
) -> PrismJobConfig:
    """
    Generate PRISM configuration from observations.parquet.

    Args:
        observations_path: Path to observations.parquet
        dataset_name: Name for the dataset (defaults to parent directory name)

    Returns:
        PrismJobConfig with all settings
    """
    import pandas as pd

    observations_path = Path(observations_path)
    df = pd.read_parquet(observations_path)

    # Validate canonical schema
    required = {'entity_id', 'signal_id', 'I', 'y', 'unit'}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"Not canonical schema. Has: {df.columns.tolist()}, needs: {required}")

    # Dataset info
    if dataset_name is None:
        dataset_name = observations_path.parent.name

    entity_count = df['entity_id'].nunique()
    signal_count = df['signal_id'].nunique()
    observation_count = len(df)

    # Build signal configs
    signals = []
    categories = set()

    for signal_id in df['signal_id'].unique():
        signal_df = df[df['signal_id'] == signal_id]
        unit = signal_df['unit'].iloc[0]
        category = detect_category(unit)

        signals.append(SignalConfig(
            name=signal_id,
            unit=unit if unit else 'unknown',
            category=category,
        ))

        if category != 'unknown':
            categories.add(category)

    # Determine engines
    engines = get_enabled_engines(categories)

    return PrismJobConfig(
        dataset_name=dataset_name,
        entity_count=entity_count,
        signal_count=signal_count,
        observation_count=observation_count,
        signals=signals,
        categories=categories,
        core_engines=engines['core'],
        domain_engines=engines['domain'],
    )


def save_config(config: PrismJobConfig, output_path: Path) -> Path:
    """Save configuration to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write(config.to_json())

    return output_path


def generate_and_save_config(
    observations_path: Path,
    output_dir: Optional[Path] = None,
    dataset_name: Optional[str] = None,
) -> Path:
    """
    Generate config and save to config.json in output directory.

    Args:
        observations_path: Path to observations.parquet
        output_dir: Output directory (defaults to same as observations)
        dataset_name: Dataset name

    Returns:
        Path to config.json
    """
    observations_path = Path(observations_path)
    output_dir = Path(output_dir) if output_dir else observations_path.parent

    config = generate_config(observations_path, dataset_name)
    config_path = output_dir / 'config.json'

    return save_config(config, config_path)


# =============================================================================
# MANIFEST GENERATION (Orthon as Brain)
# =============================================================================

from ..shared.engine_registry import (
    Granularity,
    Pillar,
    EngineSpec,
    ENGINE_SPECS,
    UNIT_TO_CATEGORY as REGISTRY_UNIT_TO_CATEGORY,
    get_engines_for_categories,
    get_engines_by_pillar,
    get_category_for_unit,
)
from .manifest_schema import (
    PrismManifest,
    EngineManifestEntry,
    ManifestMetadata,
    WindowManifest,
)


@dataclass
class DataAnalysis:
    """Results from analyzing observations.parquet."""

    # Counts
    entity_count: int = 0
    signal_count: int = 0
    observation_count: int = 0

    # Lists
    entities: List[str] = field(default_factory=list)
    signals: List[str] = field(default_factory=list)
    units: List[str] = field(default_factory=list)
    categories: Set[str] = field(default_factory=set)

    # Signal → unit mapping
    signal_units: Dict[str, str] = field(default_factory=dict)
    # Signal → category mapping
    signal_categories: Dict[str, str] = field(default_factory=dict)
    # Category → signals mapping
    category_signals: Dict[str, List[str]] = field(default_factory=dict)
    # Category → units mapping
    category_units: Dict[str, List[str]] = field(default_factory=dict)

    # I (index) statistics
    I_min: Optional[float] = None
    I_max: Optional[float] = None
    I_range: Optional[float] = None
    sampling_rate: Optional[float] = None

    # y (value) statistics
    y_min: Optional[float] = None
    y_max: Optional[float] = None

    # Computed: average observations per (entity, signal) pair
    avg_points_per_signal: Optional[float] = None


class DataAnalyzer:
    """
    Analyze observations.parquet to extract metadata for manifest generation.

    Uses Polars for efficient large-file handling.
    """

    def __init__(self, observations_path: Path):
        self.observations_path = Path(observations_path)
        self.avg_points_per_signal: Optional[float] = None

    def analyze(self) -> DataAnalysis:
        """
        Analyze the observations file and return DataAnalysis.

        Returns:
            DataAnalysis with all extracted metadata
        """
        import polars as pl

        # Read parquet with Polars
        df = pl.read_parquet(self.observations_path)

        # Validate canonical schema
        required = {'entity_id', 'signal_id', 'I', 'y', 'unit'}
        if not required.issubset(set(df.columns)):
            raise ValueError(
                f"Not canonical schema. Has: {df.columns}, needs: {required}"
            )

        analysis = DataAnalysis()

        # Basic counts
        analysis.entity_count = df['entity_id'].n_unique()
        analysis.signal_count = df['signal_id'].n_unique()
        analysis.observation_count = len(df)

        # Entity and signal lists
        analysis.entities = df['entity_id'].unique().sort().to_list()
        analysis.signals = df['signal_id'].unique().sort().to_list()

        # Unit analysis per signal
        signal_unit_df = (
            df.group_by('signal_id')
            .agg(pl.col('unit').first())
            .sort('signal_id')
        )

        for row in signal_unit_df.iter_rows(named=True):
            signal_id = row['signal_id']
            unit = row['unit']
            analysis.signal_units[signal_id] = unit

            # Get category for unit
            category = get_category_for_unit(unit)
            analysis.signal_categories[signal_id] = category

            if category != 'unknown':
                analysis.categories.add(category)

                # Track signals per category
                if category not in analysis.category_signals:
                    analysis.category_signals[category] = []
                analysis.category_signals[category].append(signal_id)

                # Track units per category
                if category not in analysis.category_units:
                    analysis.category_units[category] = []
                if unit and unit not in analysis.category_units[category]:
                    analysis.category_units[category].append(unit)

        # Unique units
        analysis.units = [u for u in df['unit'].unique().to_list() if u is not None]

        # I statistics
        I_stats = df.select([
            pl.col('I').min().alias('I_min'),
            pl.col('I').max().alias('I_max'),
        ]).row(0, named=True)

        analysis.I_min = I_stats['I_min']
        analysis.I_max = I_stats['I_max']
        if analysis.I_min is not None and analysis.I_max is not None:
            analysis.I_range = analysis.I_max - analysis.I_min

        # Estimate sampling rate from first entity/signal
        if analysis.entities and analysis.signals:
            sample_df = df.filter(
                (pl.col('entity_id') == analysis.entities[0]) &
                (pl.col('signal_id') == analysis.signals[0])
            ).sort('I')

            if len(sample_df) > 1:
                I_vals = sample_df['I'].to_list()
                diffs = [I_vals[i+1] - I_vals[i] for i in range(min(100, len(I_vals)-1))]
                if diffs:
                    median_diff = sorted(diffs)[len(diffs)//2]
                    if median_diff > 0:
                        analysis.sampling_rate = 1.0 / median_diff

        # y statistics
        y_stats = df.select([
            pl.col('y').min().alias('y_min'),
            pl.col('y').max().alias('y_max'),
        ]).row(0, named=True)

        analysis.y_min = y_stats['y_min']
        analysis.y_max = y_stats['y_max']

        # Calculate average points per signal (for stride calculation)
        if analysis.entity_count > 0 and analysis.signal_count > 0:
            self.avg_points_per_signal = (
                analysis.observation_count / (analysis.entity_count * analysis.signal_count)
            )
        else:
            self.avg_points_per_signal = float(analysis.observation_count)

        # Store on analysis object too
        analysis.avg_points_per_signal = self.avg_points_per_signal

        return analysis


# =============================================================================
# INTELLIGENT STRIDE CALCULATION
# =============================================================================

# Engine complexity tiers for stride calculation
ENGINE_COST_TIERS = {
    # Cheap (O(n)) - can afford dense output
    'cheap': [
        'rolling_mean', 'rolling_std', 'rolling_rms', 'rolling_min', 'rolling_max',
        'rolling_range', 'rolling_sum', 'derivatives',
    ],
    # Medium (O(n) but more computation per point)
    'medium': [
        'rolling_kurtosis', 'rolling_skewness', 'rolling_volatility',
        'rolling_crest', 'rolling_peak', 'rolling_pulsation',
    ],
    # Expensive (O(n log n) or iterative algorithms)
    'expensive': [
        'rolling_hurst', 'manifold', 'stability', 'granger', 'transfer_entropy',
    ],
    # Very expensive (O(n²) or complex iterative)
    'very_expensive': [
        'rolling_entropy', 'topology', 'information_flow', 'dynamics',
        'rqa', 'embedding', 'attractor',
    ],
}


def calculate_engine_stride(
    engine_name: str,
    avg_points_per_signal: float,
    window_size: int,
) -> int:
    """
    Calculate intelligent stride for an engine based on data size and engine cost.

    The goal is to produce useful output density without excessive computation.
    For expensive engines on large datasets, we use sparser windows.

    Args:
        engine_name: Name of the engine
        avg_points_per_signal: Average observations per (entity, signal) pair
        window_size: Base window size

    Returns:
        Stride value
    """
    # Determine engine cost tier
    cost_tier = 'medium'  # default
    for tier, engines in ENGINE_COST_TIERS.items():
        if engine_name in engines:
            cost_tier = tier
            break

    # Base stride multipliers by cost tier
    # Expressed as fraction of window size
    STRIDE_FRACTION = {
        'cheap': 0.10,      # 10% of window = 90% overlap
        'medium': 0.25,     # 25% of window = 75% overlap
        'expensive': 0.25,  # 25% of window = 75% overlap
        'very_expensive': 0.50,  # 50% of window = 50% overlap
    }

    base_fraction = STRIDE_FRACTION.get(cost_tier, 0.25)

    # Adjust based on data size
    if avg_points_per_signal < 500:
        # Small dataset: allow denser computation
        if cost_tier in ('cheap', 'medium'):
            base_fraction = 0.10  # Dense for cheap engines
        else:
            base_fraction = 0.25  # Still reasonable for expensive

    elif avg_points_per_signal > 10000:
        # Large dataset: be more aggressive
        base_fraction *= 2
        base_fraction = min(base_fraction, 1.0)  # Cap at non-overlapping

    elif avg_points_per_signal > 5000:
        # Medium-large dataset: moderately increase stride
        base_fraction *= 1.5
        base_fraction = min(base_fraction, 1.0)

    # Calculate stride
    stride = max(1, int(window_size * base_fraction))

    return stride


def calculate_all_engine_params(
    avg_points_per_signal: float,
    window_size: int,
) -> Dict[str, Dict[str, Any]]:
    """
    Calculate stride/window params for all rolling and windowed engines.

    Returns a dict of engine_name -> params suitable for manifest.

    Args:
        avg_points_per_signal: Average observations per (entity, signal) pair
        window_size: Base window size

    Returns:
        Dict[engine_name, params dict]
    """
    params = {}

    # All engines that need window/stride params
    windowed_engines = set()
    for tier_engines in ENGINE_COST_TIERS.values():
        windowed_engines.update(tier_engines)

    # Additional engines that use window/stride
    windowed_engines.update([
        'physics', 'dynamics', 'topology', 'information_flow',
    ])

    for engine_name in windowed_engines:
        stride = calculate_engine_stride(engine_name, avg_points_per_signal, window_size)

        if engine_name.startswith('rolling_'):
            # Rolling engines: window + stride
            params[engine_name] = {'window': window_size, 'stride': stride}

        elif engine_name in ('derivatives',):
            # Just window, no stride needed
            params[engine_name] = {'window': window_size}

        elif engine_name in ('manifold',):
            # Manifold needs n_components too
            params[engine_name] = {
                'n_components': 3,
                'window': window_size,
                'stride': stride,
            }

        elif engine_name == 'granger':
            # Granger uses max_lag, not window
            params[engine_name] = {'max_lag': min(10, window_size // 10)}

        elif engine_name == 'physics':
            # Physics engine parameters
            params[engine_name] = {
                'n_baseline': min(100, window_size),
                'coherence_window': window_size // 2,
            }

        elif engine_name in ('dynamics', 'topology', 'information_flow'):
            # Multi-signal engines use window_size/step_size
            params[engine_name] = {
                'window_size': window_size,
                'step_size': stride,
            }

        elif engine_name == 'stability':
            params[engine_name] = {'window': window_size, 'stride': stride}

    return params


class ManifestBuilder:
    """
    Build a PrismManifest from DataAnalysis.

    This is where Orthon makes decisions about what engines to run.
    The manifest specifies the complete execution plan for PRISM.

    Output parquets map to the four pillars:
    - vector.parquet: Per-signal time series features (Y1-Y9 primitives)
    - geometry.parquet: Cross-signal structure (Y10 Structure Engine)
    - physics.parquet: Thermodynamic health (Y11 Physics Engine)
    - dynamics.parquet: Stability analysis (Y12 Dynamics Engine)
    - topology.parquet: Topological health (Y13 Advanced - Betti numbers)
    - information_flow.parquet: Causal networks (Y13 Advanced - Transfer entropy)
    - pairs.parquet: Pairwise relationships
    """

    # Granularity → output parquet mapping
    GRANULARITY_TO_OUTPUT = {
        Granularity.SIGNAL: 'vector',
        Granularity.OBSERVATION: 'dynamics',
        Granularity.PAIR_DIRECTIONAL: 'pairs',
        Granularity.PAIR_SYMMETRIC: 'pairs',
        Granularity.OBSERVATION_CROSS_SIGNAL: 'geometry',
        Granularity.TOPOLOGY: 'topology',
        Granularity.INFORMATION: 'information_flow',
    }

    # Granularity → groupby columns
    GRANULARITY_TO_GROUPBY = {
        Granularity.SIGNAL: ['entity_id', 'signal_id'],
        Granularity.OBSERVATION: ['entity_id'],
        Granularity.PAIR_DIRECTIONAL: ['entity_id'],  # pairs enumerated internally
        Granularity.PAIR_SYMMETRIC: ['entity_id'],    # pairs enumerated internally
        Granularity.OBSERVATION_CROSS_SIGNAL: ['entity_id'],
        Granularity.TOPOLOGY: ['entity_id'],
        Granularity.INFORMATION: ['entity_id'],
    }

    # Pillar → output parquet (for pillar-based routing)
    PILLAR_TO_OUTPUT = {
        Pillar.GEOMETRY: 'geometry',
        Pillar.PHYSICS: 'physics',
        Pillar.DYNAMICS: 'dynamics',
        Pillar.TOPOLOGY: 'topology',
        Pillar.INFORMATION: 'information_flow',
    }

    def __init__(
        self,
        analysis: DataAnalysis,
        input_file: str,
        output_dir: str,
        avg_points_per_signal: Optional[float] = None,
    ):
        self.analysis = analysis
        self.input_file = input_file
        self.output_dir = output_dir
        # Calculate avg points if not provided
        if avg_points_per_signal is not None:
            self.avg_points_per_signal = avg_points_per_signal
        elif analysis.entity_count > 0 and analysis.signal_count > 0:
            self.avg_points_per_signal = (
                analysis.observation_count / (analysis.entity_count * analysis.signal_count)
            )
        else:
            self.avg_points_per_signal = float(analysis.observation_count)
        # Engine params will be set during build()
        self.engine_params: Dict[str, Dict[str, Any]] = {}

    def build(
        self,
        window_size: int = 100,
        window_stride: int = 50,
        min_samples: int = 50,
        constants: Optional[Dict[str, Any]] = None,
        callback_url: Optional[str] = None,
    ) -> PrismManifest:
        """
        Build the complete manifest.

        Args:
            window_size: Window size for analysis
            window_stride: Stride between windows
            min_samples: Minimum samples per window
            constants: Global constants for physics calculations
            callback_url: Optional callback URL for job completion

        Returns:
            PrismManifest ready for PRISM execution
        """
        # Build metadata
        metadata = ManifestMetadata(
            entity_count=self.analysis.entity_count,
            signal_count=self.analysis.signal_count,
            observation_count=self.analysis.observation_count,
            entities=self.analysis.entities,
            signals=self.analysis.signals,
            units_present=self.analysis.units,
            unit_categories=list(self.analysis.categories),
            sampling_rate=self.analysis.sampling_rate,
            I_min=self.analysis.I_min,
            I_max=self.analysis.I_max,
            I_range=self.analysis.I_range,
            y_min=self.analysis.y_min,
            y_max=self.analysis.y_max,
        )

        # Calculate intelligent stride for each engine based on data size
        self.engine_params = calculate_all_engine_params(
            self.avg_points_per_signal,
            window_size,
        )

        # Use the computed stride for expensive engines as the manifest-level stride
        # This is a reasonable default - specific engines may override
        computed_stride = window_stride
        if 'rolling_entropy' in self.engine_params:
            computed_stride = self.engine_params['rolling_entropy'].get('stride', window_stride)
        elif 'rolling_hurst' in self.engine_params:
            computed_stride = self.engine_params['rolling_hurst'].get('stride', window_stride)

        # Build window config
        window = WindowManifest(
            size=window_size,
            stride=computed_stride,
            min_samples=min_samples,
        )

        # Select engines based on detected categories
        selected_engines = get_engines_for_categories(self.analysis.categories)

        # Build engine manifest entries
        engine_entries = []
        for spec in selected_engines:
            entry = self._build_engine_entry(spec, window_size)
            engine_entries.append(entry)

        # Sort engines by:
        # 1. Output parquet (geometry, physics, dynamics, topology, information_flow, vector, pairs)
        # 2. Universal vs category-specific (universal first)
        # 3. Name (alphabetical)
        output_order = {
            'geometry': 0,      # Y10 Structure
            'physics': 1,       # Y11 Physics
            'dynamics': 2,      # Y12 Dynamics
            'topology': 3,      # Y13 Topology
            'information_flow': 4,  # Y13 Information
            'vector': 5,        # Y1-Y9 Signal primitives
            'pairs': 6,         # Y2/Y9 Pairwise primitives
        }
        engine_entries.sort(
            key=lambda e: (
                output_order.get(e.output, 99),
                not self._is_universal(e.name),
                e.name
            )
        )

        return PrismManifest(
            input_file=self.input_file,
            output_dir=self.output_dir,
            metadata=metadata,
            engines=engine_entries,
            window=window,
            constants=constants or {},
            callback_url=callback_url,
        )

    def _is_universal(self, engine_name: str) -> bool:
        """Check if engine is universal (no category restrictions)."""
        spec = ENGINE_SPECS.get(engine_name)
        return spec.is_universal() if spec else True

    def _build_engine_entry(
        self,
        spec: EngineSpec,
        window_size: int = 100,
    ) -> EngineManifestEntry:
        """
        Build manifest entry for a single engine.

        Routing logic:
        1. If spec has pillar AND pillar is in PILLAR_TO_OUTPUT → use pillar output
        2. Otherwise use granularity-based output

        This ensures Y10-Y13 engines route to their proper parquets:
        - Y10 Structure → geometry.parquet
        - Y11 Physics → physics.parquet
        - Y12 Dynamics → dynamics.parquet
        - Y13 Topology → topology.parquet
        - Y13 Information → information_flow.parquet

        Params are merged from:
        1. Calculated stride/window params (from calculate_all_engine_params)
        2. Engine spec defaults
        Calculated params take precedence to ensure intelligent stride.
        """

        # Determine output parquet (prefer pillar if defined)
        if spec.pillar and spec.pillar in self.PILLAR_TO_OUTPUT:
            output = self.PILLAR_TO_OUTPUT[spec.pillar]
        else:
            output = self.GRANULARITY_TO_OUTPUT.get(spec.granularity, 'dynamics')

        # Determine groupby
        groupby = self.GRANULARITY_TO_GROUPBY.get(spec.granularity, ['entity_id'])

        # Build filter for category-specific engines
        filter_expr = None
        if not spec.is_universal():
            # Get all units that match the engine's categories
            matching_units = []
            for cat in spec.categories:
                if cat in self.analysis.category_units:
                    matching_units.extend(self.analysis.category_units[cat])

            if matching_units:
                # Remove duplicates while preserving order
                seen = set()
                unique_units = []
                for u in matching_units:
                    if u not in seen:
                        seen.add(u)
                        unique_units.append(u)

                # Build Polars filter expression
                units_list = ', '.join(f'"{u}"' for u in unique_units)
                filter_expr = f'col("unit").is_in([{units_list}])'

        # Merge params: spec defaults + calculated stride params (calculated takes precedence)
        merged_params = dict(spec.params)  # Start with spec defaults
        if spec.name in self.engine_params:
            # Calculated params override defaults
            merged_params.update(self.engine_params[spec.name])

        return EngineManifestEntry(
            name=spec.name,
            output=output,
            granularity=spec.granularity.value,
            groupby=groupby,
            orderby=['I'],
            input_columns=spec.input_columns,
            output_columns=spec.output_columns,
            function=spec.function or f"prism.engines.{spec.name}.compute",
            params=merged_params,
            filter=filter_expr,
            min_rows=spec.min_rows,
            enabled=True,
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def generate_manifest(
    observations_path: Path,
    output_dir: Optional[Path] = None,
    window_size: int = 100,
    window_stride: int = 50,
    min_samples: int = 50,
    constants: Optional[Dict[str, Any]] = None,
    callback_url: Optional[str] = None,
) -> PrismManifest:
    """
    Generate a PRISM manifest from observations.parquet.

    This is the main entry point for manifest generation.
    Orthon analyzes the data and decides what engines to run.

    Args:
        observations_path: Path to observations.parquet
        output_dir: Output directory for results (defaults to same as observations)
        window_size: Window size for analysis
        window_stride: Stride between windows
        min_samples: Minimum samples per window
        constants: Global constants for physics calculations
        callback_url: Optional callback URL for job completion

    Returns:
        PrismManifest ready for PRISM execution

    Example:
        manifest = generate_manifest(
            "output/observations.parquet",
            output_dir="output/",
            window_size=100,
            constants={'density_kg_m3': 1000},
        )
        manifest.to_json("output/manifest.json")
        print(manifest.summary())
    """
    observations_path = Path(observations_path)
    output_dir = Path(output_dir) if output_dir else observations_path.parent

    # Analyze data
    analyzer = DataAnalyzer(observations_path)
    analysis = analyzer.analyze()

    # Build manifest with intelligent stride calculation
    builder = ManifestBuilder(
        analysis=analysis,
        input_file=str(observations_path),
        output_dir=str(output_dir),
        avg_points_per_signal=analyzer.avg_points_per_signal,
    )

    return builder.build(
        window_size=window_size,
        window_stride=window_stride,
        min_samples=min_samples,
        constants=constants,
        callback_url=callback_url,
    )


def generate_and_save_manifest(
    observations_path: Path,
    output_dir: Optional[Path] = None,
    manifest_name: str = "manifest.json",
    **kwargs,
) -> Path:
    """
    Generate manifest and save to file.

    Args:
        observations_path: Path to observations.parquet
        output_dir: Output directory (defaults to same as observations)
        manifest_name: Name of manifest file (default: manifest.json)
        **kwargs: Additional arguments passed to generate_manifest

    Returns:
        Path to saved manifest.json
    """
    observations_path = Path(observations_path)
    output_dir = Path(output_dir) if output_dir else observations_path.parent

    manifest = generate_manifest(
        observations_path,
        output_dir=output_dir,
        **kwargs,
    )

    manifest_path = output_dir / manifest_name
    manifest.to_json(manifest_path)

    return manifest_path


# =============================================================================
# VALIDATION
# =============================================================================

def validate_manifest_params(
    manifest: PrismManifest,
    n_signals: int,
    avg_points_per_signal: float,
) -> List[str]:
    """
    Validate manifest params for sanity.

    Checks for:
    - Stride > window (gaps in coverage)
    - Excessive computation (would cause hangs)
    - Missing required params

    Args:
        manifest: The manifest to validate
        n_signals: Number of signals in dataset
        avg_points_per_signal: Average observations per signal

    Returns:
        List of warning messages (empty if valid)
    """
    warnings = []

    for engine in manifest.engines:
        params = engine.params

        # Check stride vs window
        if 'stride' in params and 'window' in params:
            stride = params['stride']
            window = params['window']

            if stride > window:
                warnings.append(
                    f"{engine.name}: stride ({stride}) > window ({window}) - gaps in coverage"
                )

        # Check for excessive computation
        window = params.get('window') or params.get('window_size', 100)
        stride = params.get('stride') or params.get('step_size', window // 2)

        if stride > 0:
            n_computations = int((avg_points_per_signal - window) / stride * n_signals)

            if n_computations > 1_000_000:
                warnings.append(
                    f"{engine.name}: will compute {n_computations:,} times - may be slow"
                )

    return warnings


def estimate_computation_cost(
    avg_points_per_signal: float,
    n_signals: int,
    window_size: int,
) -> Dict[str, Any]:
    """
    Estimate total computation cost for a dataset.

    Returns a summary dict with estimated computations per engine.

    Args:
        avg_points_per_signal: Average observations per signal
        n_signals: Number of signals
        window_size: Window size for analysis

    Returns:
        Dict with computation estimates
    """
    computed_params = calculate_all_engine_params(avg_points_per_signal, window_size)

    estimates = {}
    total_computations = 0

    for engine_name, params in computed_params.items():
        window = params.get('window') or params.get('window_size', window_size)
        stride = params.get('stride') or params.get('step_size', window // 2)

        if stride > 0 and avg_points_per_signal > window:
            n_computations = int((avg_points_per_signal - window) / stride * n_signals)
        else:
            n_computations = n_signals

        estimates[engine_name] = {
            'computations': n_computations,
            'window': window,
            'stride': stride,
        }
        total_computations += n_computations

    return {
        'total_computations': total_computations,
        'engines': estimates,
        'summary': f"{total_computations:,} total rolling calculations across {len(estimates)} engines",
    }
