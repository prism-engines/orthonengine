-- ============================================================================
-- ORTHON SQL Engine: 04_behavioral_geometry/001_correlation.sql
-- ============================================================================
-- Pairwise correlation between all analog/periodic signals
--
-- Only computes for signals where correlation is meaningful
-- (interpolation_valid = TRUE)
-- ============================================================================

-- Correlation matrix (all pairs)
CREATE OR REPLACE VIEW v_correlation_matrix AS
WITH valid_signals AS (
    SELECT signal_id
    FROM v_signal_class
    WHERE interpolation_valid = TRUE
),
pairs AS (
    SELECT
        a.signal_id AS signal_a,
        b.signal_id AS signal_b
    FROM valid_signals a
    CROSS JOIN valid_signals b
    WHERE a.signal_id < b.signal_id  -- Upper triangle only
)
SELECT
    p.signal_a,
    p.signal_b,
    CORR(da.y, db.y) AS correlation
FROM pairs p
JOIN v_base da ON da.signal_id = p.signal_a
JOIN v_base db ON db.signal_id = p.signal_b AND db.I = da.I
GROUP BY p.signal_a, p.signal_b;

-- Strong correlations only
CREATE OR REPLACE VIEW v_strong_correlations AS
SELECT
    signal_a,
    signal_b,
    correlation,
    CASE
        WHEN correlation > 0.7 THEN 'strong_positive'
        WHEN correlation < -0.7 THEN 'strong_negative'
        WHEN correlation > 0.3 THEN 'moderate_positive'
        WHEN correlation < -0.3 THEN 'moderate_negative'
        ELSE 'weak'
    END AS correlation_strength
FROM v_correlation_matrix
WHERE ABS(correlation) > 0.3
ORDER BY ABS(correlation) DESC;
