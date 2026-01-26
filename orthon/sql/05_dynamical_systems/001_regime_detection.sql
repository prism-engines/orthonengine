-- ============================================================================
-- ORTHON SQL Engine: 05_dynamical_systems/001_regime_detection.sql
-- ============================================================================
-- Detect regime changes in signals
--
-- A regime change is a significant shift in signal behavior:
-- - Mean shift
-- - Variance shift
-- - Derivative behavior change
-- ============================================================================

-- Rolling statistics with regime markers
CREATE OR REPLACE VIEW v_regime_markers AS
SELECT
    signal_id,
    I,
    y,
    dy,
    -- Rolling mean
    AVG(y) OVER w AS rolling_mean,
    -- Rolling std
    STDDEV(y) OVER w AS rolling_std,
    -- Change from previous rolling mean
    AVG(y) OVER w - LAG(AVG(y) OVER w, 50) OVER (PARTITION BY signal_id ORDER BY I) AS mean_change,
    -- Change in rolling std
    STDDEV(y) OVER w - LAG(STDDEV(y) OVER w, 50) OVER (PARTITION BY signal_id ORDER BY I) AS std_change
FROM v_dy
WINDOW w AS (
    PARTITION BY signal_id
    ORDER BY I
    ROWS BETWEEN 25 PRECEDING AND 25 FOLLOWING
);

-- Regime change points
CREATE OR REPLACE VIEW v_regime_changes AS
WITH stats AS (
    SELECT
        signal_id,
        STDDEV(y) AS global_std,
        AVG(y) AS global_mean
    FROM v_base
    GROUP BY signal_id
)
SELECT
    r.signal_id,
    r.I,
    r.mean_change,
    r.std_change,
    -- Regime change if mean shift > 2 std OR variance doubles/halves
    CASE
        WHEN ABS(r.mean_change) > 2 * s.global_std THEN 'mean_shift'
        WHEN r.std_change > s.global_std THEN 'variance_increase'
        WHEN r.std_change < -0.5 * s.global_std THEN 'variance_decrease'
        ELSE NULL
    END AS change_type
FROM v_regime_markers r
JOIN stats s USING (signal_id)
WHERE r.mean_change IS NOT NULL;

-- Regime assignments
CREATE OR REPLACE VIEW v_regimes AS
WITH change_points AS (
    SELECT
        signal_id,
        I,
        change_type,
        ROW_NUMBER() OVER (PARTITION BY signal_id ORDER BY I) AS regime_id
    FROM v_regime_changes
    WHERE change_type IS NOT NULL
)
SELECT
    b.signal_id,
    b.I,
    b.y,
    COALESCE(
        (SELECT MAX(regime_id) FROM change_points cp
         WHERE cp.signal_id = b.signal_id AND cp.I <= b.I),
        0
    ) + 1 AS regime_id
FROM v_base b;

-- Regime summary
CREATE OR REPLACE VIEW v_regime_summary AS
SELECT
    signal_id,
    regime_id,
    COUNT(*) AS n_points,
    MIN(I) AS regime_start,
    MAX(I) AS regime_end,
    AVG(y) AS regime_mean,
    STDDEV(y) AS regime_std
FROM v_regimes
GROUP BY signal_id, regime_id
ORDER BY signal_id, regime_id;
