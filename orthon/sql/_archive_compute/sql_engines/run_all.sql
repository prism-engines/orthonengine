-- ============================================================================
-- ORTHON SQL: RUN ALL ENGINES
-- ============================================================================
-- Execute all SQL engines in dependency order.
-- Usage: duckdb -c ".read sql/sql_engines/run_all.sql"
-- ============================================================================

-- ============================================================================
-- 00: LOAD DATA
-- ============================================================================
-- Create base observation table from input parquet

-- Data path: set data_file variable before running, or use default
-- Usage from orthon/ directory:
--   duckdb -c ".read sql/sql_engines/run_all.sql"
-- Or with custom data:
--   duckdb -c "SET data_file='path/to/file.parquet'; .read sql/sql_engines/run_all.sql"
CREATE OR REPLACE TABLE observations AS
SELECT * FROM 'data/demo/demo_signals.parquet';

-- Create base view with standard columns
-- Simple schema: signal_id, I, y, unit
CREATE OR REPLACE VIEW v_base AS
SELECT
    signal_id,
    I,
    y,
    'time' AS index_dimension,
    's' AS index_unit,
    COALESCE(unit, 'unknown') AS value_unit,
    'unknown' AS signal_class
FROM observations;

-- Create empty primitives table (will be populated by PRISM)
CREATE OR REPLACE TABLE primitives (
    signal_id VARCHAR,
    hurst DOUBLE,
    lyapunov DOUBLE,
    spectrum DOUBLE[],
    garch_omega DOUBLE,
    garch_alpha DOUBLE,
    garch_beta DOUBLE,
    sample_entropy DOUBLE,
    umap_coords DOUBLE[]
);

PRAGMA enable_progress_bar;

-- ============================================================================
-- 01: CALCULUS (derivatives, curvature)
-- ============================================================================
.print '>>> Layer 1: Calculus...'
.read sql/sql_engines/01_calculus.sql

-- ============================================================================
-- 02: STATISTICS (rolling, global)
-- ============================================================================
.print '>>> Layer 2: Statistics...'
.read sql/sql_engines/02_statistics.sql

-- ============================================================================
-- 03: SIGNAL CLASSIFICATION
-- ============================================================================
.print '>>> Layer 3: Signal Classification...'
.read sql/sql_engines/03_signal_class.sql

-- ============================================================================
-- 04: TYPOLOGY (behavioral classification)
-- ============================================================================
.print '>>> Layer 4: Typology...'
.read sql/sql_engines/04_typology.sql

-- ============================================================================
-- 05: BEHAVIORAL GEOMETRY (coupling, networks)
-- ============================================================================
.print '>>> Layer 5: Behavioral Geometry...'
.read sql/sql_engines/05_geometry.sql

-- ============================================================================
-- 06: DYNAMICAL SYSTEMS (regimes, stability)
-- ============================================================================
.print '>>> Layer 6: Dynamical Systems...'
.read sql/sql_engines/06_dynamics.sql

-- ============================================================================
-- 07: CAUSAL MECHANICS
-- ============================================================================
.print '>>> Layer 7: Causal Mechanics...'
.read sql/sql_engines/07_causality.sql

-- ============================================================================
-- 08: ENTROPY & INFORMATION
-- ============================================================================
.print '>>> Layer 8: Entropy...'
.read sql/sql_engines/08_entropy.sql

-- ============================================================================
-- 09: PHYSICS & CONSERVATION
-- ============================================================================
.print '>>> Layer 9: Physics...'
.read sql/sql_engines/09_physics.sql

-- ============================================================================
-- 10: MANIFOLD ASSEMBLY & EXPORT
-- ============================================================================
.print '>>> Layer 10: Manifold Assembly...'
.read sql/sql_engines/10_manifold.sql

-- ============================================================================
-- EXPORT TO PARQUET
-- ============================================================================
.print '>>> Exporting parquets...'

COPY (SELECT * FROM v_export_signal_class) 
    TO 'output/signal_class.parquet' (FORMAT PARQUET);

COPY (SELECT * FROM v_export_signal_typology LIMIT 100000) 
    TO 'output/signal_typology.parquet' (FORMAT PARQUET);

COPY (SELECT * FROM v_export_behavioral_geometry) 
    TO 'output/behavioral_geometry.parquet' (FORMAT PARQUET);

COPY (SELECT * FROM v_export_dynamical_systems LIMIT 100000) 
    TO 'output/dynamical_systems.parquet' (FORMAT PARQUET);

COPY (SELECT * FROM v_export_causal_mechanics) 
    TO 'output/causal_mechanics.parquet' (FORMAT PARQUET);

-- ============================================================================
-- SUMMARY OUTPUT
-- ============================================================================
.print ''
.print '=== ORTHON ANALYSIS COMPLETE ==='
.print ''

.print '--- Signal Classification ---'
SELECT signal_class, COUNT(*) AS n_signals 
FROM v_signal_class 
GROUP BY signal_class;

.print ''
.print '--- Behavioral Typology ---'
SELECT behavioral_type, COUNT(*) AS n_signals 
FROM v_signal_typology 
GROUP BY behavioral_type;

.print ''
.print '--- Causal Roles ---'
SELECT causal_role, COUNT(*) AS n_signals 
FROM v_causal_roles 
GROUP BY causal_role;

.print ''
.print '--- System Summary ---'
SELECT * FROM v_system_summary;

.print ''
.print '--- Alerts (top 10) ---'
SELECT alert_type, signal_id, alert_at, description, ROUND(severity, 2) AS severity
FROM v_alerts
ORDER BY severity DESC
LIMIT 10;

.print ''
.print '>>> Output files written to output/'
.print '  - signal_class.parquet'
.print '  - signal_typology.parquet'
.print '  - behavioral_geometry.parquet'
.print '  - dynamical_systems.parquet'
.print '  - causal_mechanics.parquet'
.print ''
.print '>>> ORTHON complete.'
