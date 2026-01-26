-- ============================================================================
-- ORTHON SQL Engine: 03_signal_typology/001_persistence.sql
-- ============================================================================
-- Behavioral classification based on Hurst exponent from PRISM
--
-- Hurst exponent interpretation:
--   H < 0.4  → Mean-reverting (anti-persistent)
--   H ≈ 0.5  → Random walk (no memory)
--   H > 0.6  → Trending (persistent)
--
-- Note: This joins with PRISM primitives if available
-- ============================================================================

-- Create primitives table placeholder if not exists (PRISM populates this)
CREATE TABLE IF NOT EXISTS primitives (
    signal_id VARCHAR PRIMARY KEY,
    hurst FLOAT,
    lyapunov FLOAT,
    dominant_freq FLOAT,
    spectral_entropy FLOAT
);

-- Behavioral class from Hurst
CREATE OR REPLACE VIEW v_persistence AS
SELECT
    s.signal_id,
    s.signal_class,
    p.hurst,
    CASE
        WHEN p.hurst IS NULL THEN 'unknown'
        WHEN p.hurst > 0.6 THEN 'trending'
        WHEN p.hurst < 0.4 THEN 'mean_reverting'
        ELSE 'random'
    END AS behavioral_class,
    CASE
        WHEN p.hurst IS NULL THEN NULL
        WHEN p.hurst > 0.5 THEN p.hurst - 0.5  -- Persistence strength
        ELSE 0.5 - p.hurst                      -- Mean-reversion strength
    END AS persistence_strength
FROM v_signal_class s
LEFT JOIN primitives p USING (signal_id);
