-- ============================================================================
-- ORTHON SQL Engine: 03_signal_typology/004_classify.sql
-- ============================================================================
-- Final signal typology combining all behavioral metrics
--
-- Output: Complete behavioral profile for each signal
-- ============================================================================

CREATE OR REPLACE VIEW v_signal_typology AS
SELECT
    s.signal_id,
    s.value_unit,
    s.signal_class,
    s.interpolation_valid,
    s.est_period,

    -- Persistence (from PRISM hurst)
    p.hurst,
    p.behavioral_class,
    p.persistence_strength,

    -- Stationarity
    st.stationarity_class,
    st.mean_drift,
    st.variance_drift,

    -- Combined behavioral profile
    CASE
        WHEN s.signal_class = 'digital' THEN 'discrete_state'
        WHEN s.signal_class = 'event' THEN 'sparse_event'
        WHEN s.signal_class = 'periodic' THEN
            CASE
                WHEN st.stationarity_class = 'stationary' THEN 'stable_oscillation'
                ELSE 'evolving_oscillation'
            END
        WHEN p.behavioral_class = 'trending' THEN
            CASE
                WHEN st.stationarity_class = 'nonstationary' THEN 'strong_trend'
                ELSE 'weak_trend'
            END
        WHEN p.behavioral_class = 'mean_reverting' THEN 'mean_reverting'
        WHEN p.behavioral_class = 'random' THEN 'random_walk'
        ELSE 'unclassified'
    END AS behavioral_profile

FROM v_signal_class s
LEFT JOIN v_persistence p USING (signal_id)
LEFT JOIN v_stationarity st USING (signal_id);

-- Typology summary
CREATE OR REPLACE VIEW v_signal_typology_summary AS
SELECT
    behavioral_profile,
    COUNT(*) AS n_signals,
    ARRAY_AGG(signal_id ORDER BY signal_id) AS signals
FROM v_signal_typology
GROUP BY behavioral_profile
ORDER BY behavioral_profile;
