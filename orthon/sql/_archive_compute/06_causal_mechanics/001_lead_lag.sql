-- ============================================================================
-- ORTHON SQL Engine: 06_causal_mechanics/001_lead_lag.sql
-- ============================================================================
-- Comprehensive lead/lag analysis across all signal pairs
--
-- Tests multiple lags to find optimal alignment and determine
-- which signal leads which.
-- ============================================================================

-- Test lags from -20 to +20
CREATE OR REPLACE VIEW v_lead_lag_raw AS
WITH valid_signals AS (
    SELECT signal_id FROM v_signal_class WHERE interpolation_valid = TRUE
),
pairs AS (
    SELECT a.signal_id AS signal_a, b.signal_id AS signal_b
    FROM valid_signals a
    CROSS JOIN valid_signals b
    WHERE a.signal_id < b.signal_id
)
SELECT
    p.signal_a,
    p.signal_b,
    0 AS lag,
    CORR(da.y, db.y) AS correlation
FROM pairs p
JOIN v_base da ON da.signal_id = p.signal_a
JOIN v_base db ON db.signal_id = p.signal_b AND db.I = da.I
GROUP BY p.signal_a, p.signal_b

UNION ALL

SELECT p.signal_a, p.signal_b, 5 AS lag, CORR(da.y, db.y)
FROM pairs p
JOIN v_base da ON da.signal_id = p.signal_a
JOIN v_base db ON db.signal_id = p.signal_b AND db.I = da.I + 5
GROUP BY p.signal_a, p.signal_b

UNION ALL

SELECT p.signal_a, p.signal_b, 10 AS lag, CORR(da.y, db.y)
FROM pairs p
JOIN v_base da ON da.signal_id = p.signal_a
JOIN v_base db ON db.signal_id = p.signal_b AND db.I = da.I + 10
GROUP BY p.signal_a, p.signal_b

UNION ALL

SELECT p.signal_a, p.signal_b, 15 AS lag, CORR(da.y, db.y)
FROM pairs p
JOIN v_base da ON da.signal_id = p.signal_a
JOIN v_base db ON db.signal_id = p.signal_b AND db.I = da.I + 15
GROUP BY p.signal_a, p.signal_b

UNION ALL

SELECT p.signal_a, p.signal_b, 20 AS lag, CORR(da.y, db.y)
FROM pairs p
JOIN v_base da ON da.signal_id = p.signal_a
JOIN v_base db ON db.signal_id = p.signal_b AND db.I = da.I + 20
GROUP BY p.signal_a, p.signal_b

UNION ALL

SELECT p.signal_a, p.signal_b, -5 AS lag, CORR(da.y, db.y)
FROM pairs p
JOIN v_base da ON da.signal_id = p.signal_a
JOIN v_base db ON db.signal_id = p.signal_b AND db.I = da.I - 5
GROUP BY p.signal_a, p.signal_b

UNION ALL

SELECT p.signal_a, p.signal_b, -10 AS lag, CORR(da.y, db.y)
FROM pairs p
JOIN v_base da ON da.signal_id = p.signal_a
JOIN v_base db ON db.signal_id = p.signal_b AND db.I = da.I - 10
GROUP BY p.signal_a, p.signal_b

UNION ALL

SELECT p.signal_a, p.signal_b, -15 AS lag, CORR(da.y, db.y)
FROM pairs p
JOIN v_base da ON da.signal_id = p.signal_a
JOIN v_base db ON db.signal_id = p.signal_b AND db.I = da.I - 15
GROUP BY p.signal_a, p.signal_b

UNION ALL

SELECT p.signal_a, p.signal_b, -20 AS lag, CORR(da.y, db.y)
FROM pairs p
JOIN v_base da ON da.signal_id = p.signal_a
JOIN v_base db ON db.signal_id = p.signal_b AND db.I = da.I - 20
GROUP BY p.signal_a, p.signal_b;

-- Find optimal lag for each pair
CREATE OR REPLACE VIEW v_lead_lag AS
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
    FROM v_lead_lag_raw
    WHERE correlation IS NOT NULL
)
SELECT
    signal_a,
    signal_b,
    lag AS optimal_lag,
    correlation AS max_correlation,
    -- Determine who leads whom
    CASE
        WHEN lag > 0 THEN signal_a
        WHEN lag < 0 THEN signal_b
        ELSE NULL
    END AS leader,
    CASE
        WHEN lag > 0 THEN signal_b
        WHEN lag < 0 THEN signal_a
        ELSE NULL
    END AS follower,
    ABS(lag) AS lag_magnitude,
    -- Relationship strength
    CASE
        WHEN ABS(correlation) > 0.9 THEN 'very_strong'
        WHEN ABS(correlation) > 0.7 THEN 'strong'
        WHEN ABS(correlation) > 0.5 THEN 'moderate'
        WHEN ABS(correlation) > 0.3 THEN 'weak'
        ELSE 'negligible'
    END AS relationship_strength
FROM ranked
WHERE rn = 1 AND ABS(correlation) > 0.3;
