-- ============================================================================
-- ORTHON SQL Engine: run_all.sql
-- ============================================================================
-- Execute all SQL engines in order
-- Run from orthon/ directory: duckdb < sql/run_all.sql
-- ============================================================================

-- Load data
.read sql/00_load.sql

-- ============================================================================
-- PASS 1: Foundation (before PRISM)
-- ============================================================================

-- 01: Calculus
.read sql/01_calculus/001_first_derivative.sql
.read sql/01_calculus/002_second_derivative.sql
.read sql/01_calculus/003_curvature.sql

-- 02: Signal Classification
.read sql/02_signal_class/001_from_units.sql
.read sql/02_signal_class/002_from_data.sql
.read sql/02_signal_class/003_classify.sql
.read sql/02_signal_class/004_prism_requests.sql

SELECT '=== PASS 1: SIGNAL CLASSIFICATION ===' AS section;
SELECT signal_id, value_unit, signal_class, interpolation_valid, class_source
FROM v_signal_class ORDER BY signal_id;

SELECT '=== PRISM WORK ORDERS ===' AS section;
SELECT signal_id, signal_class, needs_hurst, needs_fft, needs_lyapunov
FROM v_prism_requests ORDER BY priority DESC, signal_id;

SELECT '=== PRISM REQUEST SUMMARY ===' AS section;
SELECT * FROM v_prism_request_summary;

-- ============================================================================
-- PASS 2: Analysis (uses PRISM results if available)
-- ============================================================================

-- 03: Signal Typology
.read sql/03_signal_typology/001_persistence.sql
.read sql/03_signal_typology/002_periodicity.sql
.read sql/03_signal_typology/003_stationarity.sql
.read sql/03_signal_typology/004_classify.sql

SELECT '=== PASS 2: SIGNAL TYPOLOGY ===' AS section;
SELECT signal_id, signal_class, behavioral_class, stationarity_class, behavioral_profile
FROM v_signal_typology ORDER BY signal_id;

-- 04: Behavioral Geometry
.read sql/04_behavioral_geometry/001_correlation.sql
.read sql/04_behavioral_geometry/002_lagged_correlation.sql

SELECT '=== CORRELATION MATRIX (strong only) ===' AS section;
SELECT * FROM v_strong_correlations LIMIT 20;

SELECT '=== LEAD/LAG RELATIONSHIPS ===' AS section;
SELECT * FROM v_optimal_lag LIMIT 20;

-- 05: Dynamical Systems
.read sql/05_dynamical_systems/001_regime_detection.sql

SELECT '=== REGIME SUMMARY ===' AS section;
SELECT * FROM v_regime_summary WHERE regime_id > 1 OR signal_id LIKE '%regime%';

-- 06: Causal Mechanics
.read sql/06_causal_mechanics/001_lead_lag.sql
.read sql/06_causal_mechanics/002_role_assignment.sql
.read sql/06_causal_mechanics/003_causal_chain.sql

SELECT '=== CAUSAL ROLES ===' AS section;
SELECT signal_id, signal_class, causal_role, n_leads, n_led_by, influence_score
FROM v_causal_roles ORDER BY influence_score DESC;

SELECT '=== CAUSAL CHAINS (strongest) ===' AS section;
SELECT source, target, hops, path, chain_strength, chain_lag
FROM v_causal_chains LIMIT 10;

SELECT '=== CAUSAL SUMMARY ===' AS section;
SELECT * FROM v_causal_summary;

-- ============================================================================
-- VALIDATION
-- ============================================================================

SELECT '=== VALIDATION SUMMARY ===' AS section;
SELECT
    (SELECT COUNT(*) FROM v_signal_class) AS total_signals,
    (SELECT COUNT(*) FROM v_signal_class WHERE signal_class = 'analog') AS analog,
    (SELECT COUNT(*) FROM v_signal_class WHERE signal_class = 'periodic') AS periodic,
    (SELECT COUNT(*) FROM v_signal_class WHERE signal_class = 'digital') AS digital,
    (SELECT COUNT(*) FROM v_signal_class WHERE signal_class = 'event') AS event;
