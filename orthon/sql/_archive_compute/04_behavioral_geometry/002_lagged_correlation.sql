-- ============================================================================
-- ORTHON SQL Engine: 04_behavioral_geometry/002_lagged_correlation.sql
-- ============================================================================
-- Lagged correlation to detect lead/lag relationships
--
-- Tests multiple lag values to find optimal alignment
-- ============================================================================

-- Lagged correlation for multiple lags
CREATE OR REPLACE VIEW v_lagged_correlation AS
WITH valid_signals AS (
    SELECT signal_id FROM v_signal_class WHERE interpolation_valid = TRUE
),
pairs AS (
    SELECT a.signal_id AS signal_a, b.signal_id AS signal_b
    FROM valid_signals a CROSS JOIN valid_signals b
    WHERE a.signal_id < b.signal_id
),
-- Test lags: -20, -10, -5, 0, 5, 10, 20
lagged AS (
    SELECT
        da.signal_id AS signal_a,
        db.signal_id AS signal_b,
        0 AS lag,
        CORR(da.y, db.y) AS correlation
    FROM v_base da
    JOIN v_base db ON db.signal_id != da.signal_id AND db.I = da.I
    WHERE da.signal_id IN (SELECT signal_id FROM valid_signals)
      AND db.signal_id IN (SELECT signal_id FROM valid_signals)
      AND da.signal_id < db.signal_id
    GROUP BY da.signal_id, db.signal_id

    UNION ALL

    SELECT
        da.signal_id AS signal_a,
        db.signal_id AS signal_b,
        10 AS lag,
        CORR(da.y, db.y) AS correlation
    FROM v_base da
    JOIN v_base db ON db.signal_id != da.signal_id AND db.I = da.I + 10
    WHERE da.signal_id IN (SELECT signal_id FROM valid_signals)
      AND db.signal_id IN (SELECT signal_id FROM valid_signals)
      AND da.signal_id < db.signal_id
    GROUP BY da.signal_id, db.signal_id

    UNION ALL

    SELECT
        da.signal_id AS signal_a,
        db.signal_id AS signal_b,
        -10 AS lag,
        CORR(da.y, db.y) AS correlation
    FROM v_base da
    JOIN v_base db ON db.signal_id != da.signal_id AND db.I = da.I - 10
    WHERE da.signal_id IN (SELECT signal_id FROM valid_signals)
      AND db.signal_id IN (SELECT signal_id FROM valid_signals)
      AND da.signal_id < db.signal_id
    GROUP BY da.signal_id, db.signal_id
)
SELECT * FROM lagged WHERE correlation IS NOT NULL;

-- Best lag for each pair
CREATE OR REPLACE VIEW v_optimal_lag AS
WITH ranked AS (
    SELECT
        signal_a,
        signal_b,
        lag,
        correlation,
        ROW_NUMBER() OVER (
            PARTITION BY signal_a, signal_b
            ORDER BY ABS(correlation) DESC
        ) AS rn
    FROM v_lagged_correlation
)
SELECT
    signal_a,
    signal_b,
    lag AS optimal_lag,
    correlation AS max_correlation,
    CASE
        WHEN lag > 0 THEN signal_a || ' leads ' || signal_b
        WHEN lag < 0 THEN signal_b || ' leads ' || signal_a
        ELSE 'synchronous'
    END AS lead_lag_relationship
FROM ranked
WHERE rn = 1 AND ABS(correlation) > 0.3;
