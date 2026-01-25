"""
ORTHON Interpretive Views
=========================

Four views that transform PRISM metrics into meaning.
All via SQL. Zero calculations.
"""

from typing import Optional, List, Dict, Any
from pathlib import Path
import polars as pl

from ..db.connection import OrthonDB
from ..db.schema import SchemaDetector


class BaseView:
    """Base class for interpretive views."""

    # Subclasses define these
    SOURCE_TABLE: str = ''
    VIEW_NAME: str = ''

    def __init__(self, db: OrthonDB, schema: SchemaDetector):
        self.db = db
        self.schema = schema
        self._validate()

    def _validate(self):
        """Check that required table exists."""
        if self.SOURCE_TABLE and self.SOURCE_TABLE not in self.schema.tables():
            raise ValueError(f"Required table '{self.SOURCE_TABLE}' not found in PRISM output")

    def _available_columns(self) -> List[str]:
        """Get columns available in source table."""
        table_schema = self.schema.get(self.SOURCE_TABLE)
        return table_schema.column_names if table_schema else []

    def _has_column(self, col: str) -> bool:
        """Check if column exists."""
        return col in self._available_columns()

    def query(self, sql: str) -> pl.DataFrame:
        """Execute SQL query."""
        return self.db.query(sql)


class SignalTypologyView(BaseView):
    """
    View 1: Signal Typology

    Question: "What IS this signal?"
    Source: vector.parquet
    """

    SOURCE_TABLE = 'vector'
    VIEW_NAME = 'signal_typology'

    def get_sql(
        self,
        entity_id: Optional[str] = None,
        columns: Optional[List[str]] = None,
    ) -> str:
        """Generate SQL for signal typology view."""

        # Always include IDs and timestamp
        select_cols = ['entity_id', 'signal_id', 'timestamp']

        # Add raw metric columns if they exist
        metric_cols = [
            'hurst_exponent', 'hurst_dfa',
            'sample_entropy', 'permutation_entropy',
            'determinism', 'laminarity',
            'garch_omega', 'garch_alpha', 'garch_beta',
            'spectral_centroid', 'spectral_entropy',
        ]
        for col in metric_cols:
            if self._has_column(col):
                select_cols.append(col)

        # Build CASE statements for labels
        labels_sql = []

        if self._has_column('hurst_exponent'):
            labels_sql.append("""
    CASE
        WHEN hurst_exponent > 0.6 THEN 'trending'
        WHEN hurst_exponent < 0.45 THEN 'mean_reverting'
        ELSE 'random'
    END AS persistence""")

        if self._has_column('sample_entropy'):
            labels_sql.append("""
    CASE
        WHEN sample_entropy < 1.0 THEN 'simple'
        WHEN sample_entropy > 2.0 THEN 'complex'
        ELSE 'moderate'
    END AS complexity""")

        if self._has_column('determinism'):
            labels_sql.append("""
    CASE
        WHEN determinism > 0.7 THEN 'deterministic'
        WHEN determinism < 0.4 THEN 'stochastic'
        ELSE 'mixed'
    END AS structure""")

        # Combine
        cols_sql = ",\n    ".join(select_cols)
        if labels_sql:
            cols_sql += "," + ",".join(labels_sql)

        sql = f"""
SELECT
    {cols_sql}
FROM {self.SOURCE_TABLE}
"""

        if entity_id:
            sql += f"WHERE entity_id = '{entity_id}'\n"

        sql += "ORDER BY entity_id, signal_id, timestamp"

        return sql

    def get(self, entity_id: Optional[str] = None) -> pl.DataFrame:
        """Get signal typology data."""
        return self.query(self.get_sql(entity_id))


class GeometryView(BaseView):
    """
    View 2: Geometry / Manifold

    Question: "How do signals relate to each other?"
    Source: geometry.parquet
    """

    SOURCE_TABLE = 'geometry'
    VIEW_NAME = 'geometry_manifold'

    def get_sql(
        self,
        entity_id: Optional[str] = None,
        min_correlation: Optional[float] = None,
    ) -> str:
        """Generate SQL for geometry view."""

        select_cols = ['entity_id']

        # Check what columns exist
        pair_cols = ['signal_a', 'signal_b', 'timestamp']
        metric_cols = [
            'correlation', 'mutual_information',
            'lead_lag_days', 'transfer_entropy',
            'granger_f_stat', 'granger_p_value',
            'distance', 'coupling',
        ]

        for col in pair_cols + metric_cols:
            if self._has_column(col):
                select_cols.append(col)

        # Labels
        labels_sql = []

        if self._has_column('correlation'):
            labels_sql.append("""
    CASE
        WHEN ABS(correlation) > 0.7 THEN 'strongly_coupled'
        WHEN ABS(correlation) < 0.3 THEN 'decoupled'
        ELSE 'weakly_coupled'
    END AS coupling_strength""")

        if self._has_column('lead_lag_days'):
            labels_sql.append("""
    CASE
        WHEN lead_lag_days < -1 THEN 'leading'
        WHEN lead_lag_days > 1 THEN 'lagging'
        ELSE 'synchronous'
    END AS temporal_role""")

        if self._has_column('transfer_entropy'):
            labels_sql.append("""
    CASE
        WHEN transfer_entropy > 0.1 THEN 'information_source'
        WHEN transfer_entropy < -0.1 THEN 'information_sink'
        ELSE 'neutral'
    END AS information_flow""")

        cols_sql = ",\n    ".join(select_cols)
        if labels_sql:
            cols_sql += "," + ",".join(labels_sql)

        sql = f"""
SELECT
    {cols_sql}
FROM {self.SOURCE_TABLE}
"""

        where_clauses = []
        if entity_id:
            where_clauses.append(f"entity_id = '{entity_id}'")
        if min_correlation is not None:
            where_clauses.append(f"ABS(correlation) >= {min_correlation}")

        if where_clauses:
            sql += "WHERE " + " AND ".join(where_clauses) + "\n"

        return sql

    def get(self, entity_id: Optional[str] = None) -> pl.DataFrame:
        """Get geometry data."""
        return self.query(self.get_sql(entity_id))


class DynamicsView(BaseView):
    """
    View 3: Dynamical System State

    Question: "How does the system evolve over time?"
    Source: dynamics.parquet

    THIS IS THE KEY VIEW - hd_slope lives here.
    """

    SOURCE_TABLE = 'dynamics'
    VIEW_NAME = 'dynamical_state'

    def get_sql(
        self,
        entity_id: Optional[str] = None,
        alert_only: bool = False,
    ) -> str:
        """Generate SQL for dynamics view."""

        select_cols = ['entity_id', 'timestamp']

        metric_cols = [
            'hd_slope',  # THE KEY METRIC
            'hd_position', 'hd_velocity', 'hd_acceleration',
            'regime_id', 'transition_flag',
            'lyapunov_exponent', 'dmd_eigenvalue_1',
            'break_detected', 'cointegration_stat',
        ]

        for col in metric_cols:
            if self._has_column(col):
                select_cols.append(col)

        # Labels - especially for hd_slope
        labels_sql = []

        if self._has_column('hd_slope'):
            labels_sql.append("""
    CASE
        WHEN hd_slope > -0.001 THEN 'stable'
        WHEN hd_slope > -0.01 THEN 'drifting'
        WHEN hd_slope > -0.05 THEN 'degrading'
        ELSE 'critical'
    END AS coherence_state""")

            labels_sql.append("""
    CASE
        WHEN hd_slope < -0.05 THEN TRUE
        ELSE FALSE
    END AS alert_critical""")

        if self._has_column('lyapunov_exponent'):
            labels_sql.append("""
    CASE
        WHEN lyapunov_exponent < 0 THEN 'stable'
        WHEN lyapunov_exponent > 0.1 THEN 'chaotic'
        ELSE 'edge_of_chaos'
    END AS stability_class""")

        cols_sql = ",\n    ".join(select_cols)
        if labels_sql:
            cols_sql += "," + ",".join(labels_sql)

        sql = f"""
SELECT
    {cols_sql}
FROM {self.SOURCE_TABLE}
"""

        where_clauses = []
        if entity_id:
            where_clauses.append(f"entity_id = '{entity_id}'")
        if alert_only and self._has_column('hd_slope'):
            where_clauses.append("hd_slope < -0.05")

        if where_clauses:
            sql += "WHERE " + " AND ".join(where_clauses) + "\n"

        sql += "ORDER BY entity_id, timestamp"

        return sql

    def get(self, entity_id: Optional[str] = None, alert_only: bool = False) -> pl.DataFrame:
        """Get dynamics data."""
        return self.query(self.get_sql(entity_id, alert_only))

    def critical_entities(self) -> pl.DataFrame:
        """Get entities in critical state."""
        sql = f"""
SELECT
    entity_id,
    MIN(hd_slope) as min_hd_slope,
    COUNT(*) as critical_count,
    MIN(timestamp) as first_critical,
    MAX(timestamp) as last_critical
FROM {self.SOURCE_TABLE}
WHERE hd_slope < -0.05
GROUP BY entity_id
ORDER BY min_hd_slope ASC
"""
        return self.query(sql)


class PhysicsView(BaseView):
    """
    View 4: Physics Elements

    Question: "What drives the system?"
    Source: physics.parquet
    """

    SOURCE_TABLE = 'physics'
    VIEW_NAME = 'physics_elements'

    def get_sql(
        self,
        entity_id: Optional[str] = None,
    ) -> str:
        """Generate SQL for physics view."""

        select_cols = ['entity_id']

        if self._has_column('signal_id'):
            select_cols.append('signal_id')
        select_cols.append('timestamp')

        metric_cols = [
            'kinetic_energy', 'potential_energy',
            'hamiltonian', 'lagrangian',
            'gibbs_free_energy',
            'angular_momentum', 'momentum_flux',
            'hamiltonian_variance',
        ]

        for col in metric_cols:
            if self._has_column(col):
                select_cols.append(col)

        # Derived metrics and labels
        labels_sql = []

        if self._has_column('kinetic_energy') and self._has_column('potential_energy'):
            labels_sql.append("""
    kinetic_energy / NULLIF(potential_energy, 0) AS ke_pe_ratio""")

            labels_sql.append("""
    CASE
        WHEN kinetic_energy / NULLIF(potential_energy, 0) > 2.0 THEN 'kinetic_dominant'
        WHEN kinetic_energy / NULLIF(potential_energy, 0) < 0.5 THEN 'potential_dominant'
        ELSE 'balanced'
    END AS energy_regime""")

        if self._has_column('gibbs_free_energy'):
            labels_sql.append("""
    CASE
        WHEN gibbs_free_energy < -0.01 THEN 'spontaneous'
        WHEN gibbs_free_energy > 0.01 THEN 'non_spontaneous'
        ELSE 'equilibrium'
    END AS equilibrium_state""")

        if self._has_column('hamiltonian_variance'):
            labels_sql.append("""
    CASE
        WHEN hamiltonian_variance < 0.05 THEN 'conserved'
        ELSE 'dissipative'
    END AS conservation_class""")

        cols_sql = ",\n    ".join(select_cols)
        if labels_sql:
            cols_sql += "," + ",".join(labels_sql)

        sql = f"""
SELECT
    {cols_sql}
FROM {self.SOURCE_TABLE}
"""

        if entity_id:
            sql += f"WHERE entity_id = '{entity_id}'\n"

        sql += "ORDER BY entity_id, timestamp"

        return sql

    def get(self, entity_id: Optional[str] = None) -> pl.DataFrame:
        """Get physics data."""
        return self.query(self.get_sql(entity_id))


# =============================================================================
# UNIFIED VIEW - Join all 4
# =============================================================================

class UnifiedView:
    """
    Join all 4 interpretive views.

    The full diagnostic picture.
    """

    def __init__(self, db: OrthonDB, schema: SchemaDetector):
        self.db = db
        self.schema = schema

        # Initialize sub-views
        self.typology = SignalTypologyView(db, schema) if 'vector' in schema.tables() else None
        self.geometry = GeometryView(db, schema) if 'geometry' in schema.tables() else None
        self.dynamics = DynamicsView(db, schema) if 'dynamics' in schema.tables() else None
        self.physics = PhysicsView(db, schema) if 'physics' in schema.tables() else None

    def entity_summary(self, entity_id: str) -> Dict[str, pl.DataFrame]:
        """Get all views for a single entity."""
        result = {}

        if self.typology:
            result['typology'] = self.typology.get(entity_id)
        if self.dynamics:
            result['dynamics'] = self.dynamics.get(entity_id)
        if self.physics:
            result['physics'] = self.physics.get(entity_id)
        if self.geometry:
            result['geometry'] = self.geometry.get(entity_id)

        return result

    def alert_summary(self) -> pl.DataFrame:
        """Get all entities with critical alerts."""
        if not self.dynamics:
            return pl.DataFrame()
        return self.dynamics.critical_entities()
