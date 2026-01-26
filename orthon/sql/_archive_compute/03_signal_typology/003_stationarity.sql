-- ============================================================================
-- ORTHON SQL Engine: 03_signal_typology/003_stationarity.sql
-- ============================================================================
-- Stationarity detection using rolling statistics
--
-- A signal is stationary if its statistical properties don't change over time.
-- SQL can detect this using rolling windows.
-- ============================================================================

-- Rolling statistics for stationarity detection
CREATE OR REPLACE VIEW v_rolling_stats AS
SELECT
    signal_id,
    I,
    y,
    -- Rolling mean (window of 100)
    AVG(y) OVER (
        PARTITION BY signal_id
        ORDER BY I
        ROWS BETWEEN 50 PRECEDING AND 50 FOLLOWING
    ) AS rolling_mean,
    -- Rolling std
    STDDEV(y) OVER (
        PARTITION BY signal_id
        ORDER BY I
        ROWS BETWEEN 50 PRECEDING AND 50 FOLLOWING
    ) AS rolling_std
FROM v_base;

-- Stationarity metrics per signal
CREATE OR REPLACE VIEW v_stationarity AS
WITH stats AS (
    SELECT
        signal_id,
        -- Coefficient of variation of rolling mean
        STDDEV(rolling_mean) / NULLIF(AVG(ABS(rolling_mean)), 0) AS mean_drift,
        -- Coefficient of variation of rolling std
        STDDEV(rolling_std) / NULLIF(AVG(rolling_std), 0) AS variance_drift
    FROM v_rolling_stats
    WHERE rolling_mean IS NOT NULL
    GROUP BY signal_id
)
SELECT
    signal_id,
    mean_drift,
    variance_drift,
    CASE
        WHEN mean_drift < 0.1 AND variance_drift < 0.2 THEN 'stationary'
        WHEN mean_drift < 0.3 THEN 'weakly_nonstationary'
        ELSE 'nonstationary'
    END AS stationarity_class
FROM stats;
