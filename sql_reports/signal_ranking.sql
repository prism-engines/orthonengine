-- ============================================================================
-- PRISM SIGNAL RANKING REPORTS
-- ============================================================================
--
-- Identifies which signals are most important, problematic, or informative.
-- Helps operators focus attention on the right sensors.
--
-- Usage: Run against observations and primitives tables
-- ============================================================================


-- ============================================================================
-- REPORT 1: SIGNAL IMPORTANCE RANKING
-- Ranks signals by information content and variability
-- ============================================================================

WITH
signal_stats AS (
    SELECT
        entity_id,
        signal_id,
        COUNT(*) AS n_points,
        AVG(y) AS mean_val,
        STDDEV_POP(y) AS std_val,
        MIN(y) AS min_val,
        MAX(y) AS max_val,
        (MAX(y) - MIN(y)) / NULLIF(STDDEV_POP(y), 0) AS range_std_ratio
    FROM observations
    GROUP BY entity_id, signal_id
),

entropy_proxy AS (
    -- Approximate entropy via histogram bin diversity
    SELECT
        entity_id,
        signal_id,
        COUNT(DISTINCT FLOOR(10 * (y - MIN(y) OVER w) / NULLIF(MAX(y) OVER w - MIN(y) OVER w, 0))) AS n_bins_used
    FROM observations
    WINDOW w AS (PARTITION BY entity_id, signal_id)
    GROUP BY entity_id, signal_id, y
)

SELECT
    s.entity_id,
    s.signal_id,
    s.n_points,
    ROUND(s.std_val / NULLIF(ABS(s.mean_val), 0), 3) AS coeff_variation,
    ROUND(s.range_std_ratio, 2) AS range_std_ratio,
    ROUND(100.0 * s.std_val / (SELECT MAX(std_val) FROM signal_stats WHERE entity_id = s.entity_id), 1) AS pct_max_variability,
    CASE
        WHEN s.std_val / NULLIF(ABS(s.mean_val), 0) > 0.5 THEN 'HIGH_INFO'
        WHEN s.std_val / NULLIF(ABS(s.mean_val), 0) > 0.1 THEN 'MEDIUM_INFO'
        ELSE 'LOW_INFO'
    END AS information_class,
    RANK() OVER (PARTITION BY s.entity_id ORDER BY s.std_val DESC) AS variability_rank
FROM signal_stats s
ORDER BY s.entity_id, variability_rank;


-- ============================================================================
-- REPORT 2: PROBLEM SIGNALS (needs attention)
-- ============================================================================

WITH
time_bounds AS (
    SELECT
        entity_id,
        MIN(I) + 0.2 * (MAX(I) - MIN(I)) AS baseline_end
    FROM observations
    GROUP BY entity_id
),

baseline AS (
    SELECT
        o.entity_id,
        o.signal_id,
        AVG(o.y) AS baseline_mean,
        STDDEV_POP(o.y) AS baseline_std
    FROM observations o
    JOIN time_bounds t ON o.entity_id = t.entity_id
    WHERE o.I <= t.baseline_end
    GROUP BY o.entity_id, o.signal_id
),

current_state AS (
    SELECT
        o.entity_id,
        o.signal_id,
        AVG(o.y) AS current_mean,
        STDDEV_POP(o.y) AS current_std,
        COUNT(*) AS n_current
    FROM observations o
    JOIN time_bounds t ON o.entity_id = t.entity_id
    WHERE o.I > t.baseline_end
    GROUP BY o.entity_id, o.signal_id
),

problems AS (
    SELECT
        b.entity_id,
        b.signal_id,
        ABS((c.current_mean - b.baseline_mean) / NULLIF(b.baseline_std, 0)) AS drift_z,
        c.current_std / NULLIF(b.baseline_std, 0) AS volatility_ratio,
        CASE WHEN ABS((c.current_mean - b.baseline_mean) / NULLIF(b.baseline_std, 0)) > 2 THEN 1 ELSE 0 END AS has_drift,
        CASE WHEN c.current_std / NULLIF(b.baseline_std, 0) > 1.5 THEN 1 ELSE 0 END AS has_vol_increase
    FROM baseline b
    JOIN current_state c USING (entity_id, signal_id)
)

SELECT
    entity_id,
    signal_id,
    ROUND(drift_z, 2) AS drift_sigma,
    ROUND(volatility_ratio, 2) AS vol_ratio,
    has_drift + has_vol_increase AS problem_score,
    CASE
        WHEN has_drift = 1 AND has_vol_increase = 1 THEN 'CRITICAL'
        WHEN has_drift = 1 THEN 'DRIFTING'
        WHEN has_vol_increase = 1 THEN 'UNSTABLE'
        ELSE 'OK'
    END AS problem_type,
    RANK() OVER (PARTITION BY entity_id ORDER BY drift_z + volatility_ratio DESC) AS problem_rank
FROM problems
WHERE has_drift = 1 OR has_vol_increase = 1
ORDER BY entity_id, problem_rank;


-- ============================================================================
-- REPORT 3: LEADING INDICATOR CANDIDATES
-- Signals that change before others (potential early warnings)
-- ============================================================================

WITH
windowed AS (
    SELECT
        entity_id,
        signal_id,
        NTILE(10) OVER (PARTITION BY entity_id, signal_id ORDER BY I) AS window_id,
        y
    FROM observations
),

window_stats AS (
    SELECT
        entity_id,
        signal_id,
        window_id,
        AVG(y) AS window_mean,
        STDDEV_POP(y) AS window_std
    FROM windowed
    GROUP BY entity_id, signal_id, window_id
),

first_movement AS (
    SELECT
        entity_id,
        signal_id,
        MIN(CASE
            WHEN ABS(window_mean - LAG(window_mean) OVER w) / NULLIF(window_std, 0) > 0.5
            THEN window_id
        END) AS first_move_window
    FROM window_stats
    WINDOW w AS (PARTITION BY entity_id, signal_id ORDER BY window_id)
    GROUP BY entity_id, signal_id
)

SELECT
    entity_id,
    signal_id,
    first_move_window,
    CASE
        WHEN first_move_window <= 2 THEN 'EARLY_MOVER'
        WHEN first_move_window <= 4 THEN 'MID_MOVER'
        WHEN first_move_window <= 6 THEN 'LATE_MOVER'
        ELSE 'STABLE'
    END AS movement_class,
    RANK() OVER (PARTITION BY entity_id ORDER BY COALESCE(first_move_window, 99)) AS early_rank
FROM first_movement
WHERE first_move_window IS NOT NULL
ORDER BY entity_id, early_rank;


-- ============================================================================
-- REPORT 4: REDUNDANT SIGNALS
-- Identifies highly correlated signals (potential redundancy)
-- ============================================================================

WITH
signal_pairs AS (
    SELECT
        a.entity_id,
        a.signal_id AS signal_a,
        b.signal_id AS signal_b,
        CORR(a.y, b.y) AS correlation
    FROM observations a
    JOIN observations b
        ON a.entity_id = b.entity_id
        AND a.I = b.I
        AND a.signal_id < b.signal_id
    GROUP BY a.entity_id, a.signal_id, b.signal_id
)

SELECT
    entity_id,
    signal_a,
    signal_b,
    ROUND(correlation, 3) AS correlation,
    CASE
        WHEN ABS(correlation) > 0.95 THEN 'HIGHLY_REDUNDANT'
        WHEN ABS(correlation) > 0.85 THEN 'REDUNDANT'
        WHEN ABS(correlation) > 0.7 THEN 'CORRELATED'
        ELSE 'INDEPENDENT'
    END AS redundancy_class,
    CASE
        WHEN correlation > 0.95 THEN 'Consider removing one'
        WHEN correlation < -0.95 THEN 'Inverse relationship - keep both'
        ELSE 'OK'
    END AS recommendation
FROM signal_pairs
WHERE ABS(correlation) > 0.7
ORDER BY entity_id, ABS(correlation) DESC;


-- ============================================================================
-- REPORT 5: SIGNAL HEALTH DASHBOARD
-- Comprehensive signal-by-signal health view
-- ============================================================================

WITH
time_bounds AS (
    SELECT entity_id, MIN(I) + 0.2 * (MAX(I) - MIN(I)) AS baseline_end
    FROM observations GROUP BY entity_id
),

baseline AS (
    SELECT o.entity_id, o.signal_id,
        AVG(o.y) AS mu, STDDEV_POP(o.y) AS sigma,
        MIN(o.y) AS min_val, MAX(o.y) AS max_val
    FROM observations o JOIN time_bounds t USING (entity_id)
    WHERE o.I <= t.baseline_end
    GROUP BY o.entity_id, o.signal_id
),

current AS (
    SELECT o.entity_id, o.signal_id,
        AVG(o.y) AS current_mean, STDDEV_POP(o.y) AS current_std,
        MIN(o.y) AS current_min, MAX(o.y) AS current_max
    FROM observations o JOIN time_bounds t USING (entity_id)
    WHERE o.I > t.baseline_end
    GROUP BY o.entity_id, o.signal_id
),

health AS (
    SELECT
        b.entity_id,
        b.signal_id,
        ROUND(b.mu, 4) AS baseline_mean,
        ROUND(c.current_mean, 4) AS current_mean,
        ROUND((c.current_mean - b.mu) / NULLIF(b.sigma, 0), 2) AS mean_shift_z,
        ROUND(c.current_std / NULLIF(b.sigma, 0), 2) AS std_ratio,
        ROUND(100 * (c.current_max - b.max_val) / NULLIF(b.max_val, 0), 1) AS max_change_pct,
        ROUND(100 * (c.current_min - b.min_val) / NULLIF(ABS(b.min_val), 0), 1) AS min_change_pct
    FROM baseline b
    JOIN current c USING (entity_id, signal_id)
)

SELECT
    entity_id,
    signal_id,
    baseline_mean,
    current_mean,
    mean_shift_z,
    std_ratio,
    -- Overall health score (lower is better)
    ROUND(ABS(mean_shift_z) + ABS(std_ratio - 1) * 2, 2) AS health_score,
    -- Traffic light
    CASE
        WHEN ABS(mean_shift_z) > 2 OR std_ratio > 1.5 THEN 'RED'
        WHEN ABS(mean_shift_z) > 1 OR std_ratio > 1.2 THEN 'YELLOW'
        ELSE 'GREEN'
    END AS status,
    -- Specific issues
    CASE
        WHEN mean_shift_z > 2 THEN 'HIGH'
        WHEN mean_shift_z < -2 THEN 'LOW'
        WHEN std_ratio > 1.5 THEN 'UNSTABLE'
        ELSE 'NORMAL'
    END AS issue
FROM health
ORDER BY entity_id, ABS(mean_shift_z) + ABS(std_ratio - 1) DESC;
