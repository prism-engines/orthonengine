-- ============================================================================
-- ORTHON SQL Engine: 02_signal_class/004_prism_requests.sql
-- ============================================================================
-- Generate PRISM work orders based on signal classification
--
-- SQL decides what each signal NEEDS from PRISM.
-- PRISM only computes what's requested.
-- ============================================================================

CREATE OR REPLACE VIEW v_prism_requests AS
SELECT
    s.signal_id,
    s.signal_class,
    s.interpolation_valid,

    -- Request hurst for analog signals (persistence matters)
    CASE
        WHEN s.signal_class IN ('analog', 'periodic') THEN TRUE
        ELSE FALSE
    END AS needs_hurst,

    -- Request fft for periodic signals or spectral analysis
    CASE
        WHEN s.signal_class = 'periodic' THEN TRUE
        ELSE FALSE
    END AS needs_fft,

    -- Request lyapunov if chaos suspected (high curvature variance)
    CASE
        WHEN s.signal_class IN ('analog', 'periodic')
         AND c.kappa_cv IS NOT NULL
         AND c.kappa_cv > 2.0
        THEN TRUE
        ELSE FALSE
    END AS needs_lyapunov,

    -- UMAP/PCA decided at system level, not per-signal
    FALSE AS needs_umap,
    FALSE AS needs_pca,

    -- Priority for PRISM execution (higher = more important)
    CASE
        WHEN s.signal_class = 'periodic' THEN 3  -- Periodic needs FFT
        WHEN s.signal_class = 'analog' THEN 2    -- Analog needs Hurst
        ELSE 1
    END AS priority

FROM v_signal_class s
LEFT JOIN v_curvature_stats c USING (signal_id);

-- Summary of PRISM requests
CREATE OR REPLACE VIEW v_prism_request_summary AS
SELECT
    COUNT(*) FILTER (WHERE needs_hurst) AS n_hurst,
    COUNT(*) FILTER (WHERE needs_fft) AS n_fft,
    COUNT(*) FILTER (WHERE needs_lyapunov) AS n_lyapunov,
    COUNT(*) FILTER (WHERE needs_umap) AS n_umap,
    COUNT(*) FILTER (WHERE needs_pca) AS n_pca,
    COUNT(*) AS total_signals,
    COUNT(*) FILTER (WHERE NOT needs_hurst AND NOT needs_fft AND NOT needs_lyapunov) AS skip_prism
FROM v_prism_requests;
