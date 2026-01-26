-- ============================================================================
-- ORTHON SQL Engine: 03_signal_typology/002_periodicity.sql
-- ============================================================================
-- Periodicity analysis using FFT results from PRISM
--
-- For signals classified as 'periodic', PRISM computes FFT.
-- SQL interprets the spectral results.
-- ============================================================================

-- Periodicity metrics from PRISM FFT results
CREATE OR REPLACE VIEW v_periodicity_analysis AS
SELECT
    s.signal_id,
    s.signal_class,
    s.est_period AS period_from_calculus,
    p.dominant_freq,
    -- Period from frequency
    CASE
        WHEN p.dominant_freq IS NOT NULL AND p.dominant_freq > 0
        THEN 1.0 / p.dominant_freq
        ELSE NULL
    END AS period_from_fft,
    p.spectral_entropy,
    -- Spectral purity (low entropy = clean periodic)
    CASE
        WHEN p.spectral_entropy IS NULL THEN 'unknown'
        WHEN p.spectral_entropy < 0.3 THEN 'pure'        -- Single dominant frequency
        WHEN p.spectral_entropy < 0.6 THEN 'harmonic'    -- Multiple related frequencies
        ELSE 'complex'                                    -- Broadband
    END AS spectral_type
FROM v_signal_class s
LEFT JOIN primitives p USING (signal_id)
WHERE s.signal_class = 'periodic';
