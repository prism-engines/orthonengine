-- ============================================================================
-- ORTHON SQL: 00_observations.sql
-- ============================================================================
-- Load raw data and create observations.parquet
-- This is the ONLY parquet file ORTHON creates.
-- All other parquet files come from PRISM.
-- ============================================================================

-- Load from uploaded file (path injected at runtime)
CREATE OR REPLACE TABLE raw_upload AS
SELECT * FROM read_parquet('{input_path}');

-- Validate required columns exist
-- Expected: signal_id, I (index), y (value), unit (optional)

-- Create standardized observations table
CREATE OR REPLACE TABLE observations AS
SELECT
    COALESCE(signal_id, 'signal_' || ROW_NUMBER() OVER ()) AS signal_id,
    I,
    y,
    COALESCE(unit, 'unknown') AS value_unit,
    -- Row identifier for reference
    ROW_NUMBER() OVER (PARTITION BY signal_id ORDER BY I) AS row_idx
FROM raw_upload;

-- Export to parquet (ONLY file ORTHON creates)
COPY observations TO '{output_path}/observations.parquet' (FORMAT PARQUET);

-- Basic stats for UI
CREATE OR REPLACE VIEW v_observations_summary AS
SELECT
    COUNT(DISTINCT signal_id) AS n_signals,
    COUNT(*) AS n_rows,
    MIN(I) AS i_min,
    MAX(I) AS i_max,
    COUNT(DISTINCT value_unit) AS n_units
FROM observations;

-- Verify
SELECT * FROM v_observations_summary;
