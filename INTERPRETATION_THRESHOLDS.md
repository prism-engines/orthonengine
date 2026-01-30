# PRISM Interpretation Thresholds & Data Requirements

> **"A finding is not significant just because the math runs."**

This document establishes **minimum data requirements**, **baseline methodology**, and **actionable thresholds** for PRISM engine outputs. Use this to avoid false positives and distinguish genuine anomalies from noise.

---

## Table of Contents

1. [The Problem](#the-problem)
2. [Baseline Methodology](#baseline-methodology)
3. [Minimum Data Requirements](#minimum-data-requirements)
4. [Window Size Guidance](#window-size-guidance)
5. [Thresholds by Engine](#thresholds-by-engine)
6. [Signal Strength Classification](#signal-strength-classification)
7. [When to Trust Results](#when-to-trust-results)
8. [SQL Validation Queries](#sql-validation-queries)

---

## The Problem

### What We've Seen

| Claim | Reality |
|-------|---------|
| "Lyapunov shows chaos!" | Only 500 data points; estimate is unreliable |
| "Transfer entropy reveals causality!" | 3 signals over 100 windows; spurious correlations dominate |
| "Betti numbers show fragmentation!" | Embedding too sparse; topological features are artifacts |
| "Health score dropped to 65!" | Baseline was never established; 65 might be normal |
| "Z-score of 2.1 - anomaly detected!" | Fleet has only 5 entities; this is expected variation |

### The Core Issues

1. **Insufficient data**: Many algorithms have hard minimums below which results are meaningless
2. **No baseline**: Without knowing "normal," you can't detect "abnormal"
3. **Weak effect sizes**: Statistically significant ≠ practically meaningful
4. **Small fleet sizes**: Z-scores assume large samples; N=5 doesn't qualify
5. **Window too short**: Some phenomena only emerge over longer timescales
6. **Sampling mismatch**: Data rate doesn't capture enough cycles of the phenomenon

---

## Baseline Methodology

### Standard Baseline: First 20% of Data

For each entity, the baseline period is defined as:

```
baseline_end = floor(0.20 × total_windows)
baseline_windows = [0, 1, 2, ..., baseline_end - 1]
```

**Requirements for valid baseline:**
- Minimum 10 windows in baseline period
- No known anomalies during baseline period
- Representative of "normal" operating conditions

### Baseline Statistics Computed

| Statistic | Formula | Use |
|-----------|---------|-----|
| μ (mean) | Σx / n | Center of normal |
| σ (std) | √(Σ(x-μ)²/n) | Spread of normal |
| IQR | P75 - P25 | Robust spread |
| P5, P95 | 5th/95th percentiles | Extreme bounds |

### Deviation Scoring

```
z_score = (current_value - baseline_mean) / baseline_std
percentile_deviation = (current_value - baseline_median) / baseline_IQR
```

---

## Minimum Data Requirements

### Sampling Rate Context

> **Critical**: All minimums assume sampling captures **≥10 characteristic cycles** of the slowest phenomenon of interest.

```
Effective observations = n_samples × (sampling_rate / characteristic_frequency)
```

**Example:**
- 1,000 samples at 1kHz capturing a 10Hz phenomenon = 100 cycles ✓ Plenty
- 1,000 samples at 1Hz capturing a 0.1Hz phenomenon = 1 cycle ✗ Insufficient

If your sampling rate doesn't capture enough cycles, multiply the minimums accordingly.

### Hard Minimums (Results Invalid Below These)

| Engine/Metric | Minimum Observations | Minimum Windows | Minimum Signals | Notes |
|---------------|---------------------|-----------------|-----------------|-------|
| **Lyapunov (max)** | 3,000 | 30 | 1 | Rosenstein/Kantz methods work with shorter series |
| **Lyapunov (spectrum)** | 20,000 | 200 | 1 | Full spectrum needs longer trajectories |
| **Correlation dimension (d ≤ 5)** | 1,000 | 10 | 1 | Low-dimensional systems |
| **Correlation dimension (d = 5-10)** | 2,000-5,000 | 20-50 | 1 | Medium embedding dimension |
| **Correlation dimension (d > 10)** | 5,000+ | 50+ | 1 | High-dimensional requires more data |
| **Recurrence (DET, LAM)** | 1,000 | 10 | 1 | Need sufficient recurrence matrix |
| **Transfer entropy** | 1,000 | 20 | 3 | Directional causality |
| **Granger causality** | 500 | 10 | 2 | VAR model fitting |
| **Betti numbers** | 500 | 10 | 1 | Persistent homology |
| **Persistence entropy** | 1,000 | 10 | 1 | Need enough topological features |
| **Coherence (eigenvalues)** | 100 | 5 | 3 | Covariance matrix stability |
| **PID (synergy/redundancy)** | 500 | 10 | 3 | Triplet information |
| **Hurst exponent** | 256 | 3 | 1 | DFA scaling regime |
| **Sample entropy** | 200 | 2 | 1 | Pattern matching |

### Soft Minimums (Results Unreliable Below These)

| Engine/Metric | Recommended Observations | Recommended Windows | Notes |
|---------------|-------------------------|---------------------|-------|
| Lyapunov (max) | 10,000 | 100 | Stable estimate with good convergence |
| Correlation dimension | 5,000-20,000 | 50-200 | Depends on embedding dimension |
| Transfer entropy | 5,000 | 100 | Robust directional inference |
| Granger causality | 2,000 | 40 | Significant lags |
| Betti numbers | 2,000 | 40 | Stable topology |
| Fleet statistics | - | - | N ≥ 30 entities |

### Data Quality Requirements

| Requirement | Threshold | Consequence if Violated |
|-------------|-----------|------------------------|
| Missing data | < 5% | Interpolation artifacts |
| Sampling uniformity | CV < 10% | Spectral leakage |
| Stationarity (baseline) | ADF p < 0.05 | Invalid baseline |
| Outlier fraction | < 1% | Biased statistics |

---

## Window Size Guidance

Window size affects which phenomena you can detect. Too small and you miss slow dynamics; too large and you blur fast transients.

### Minimum Window Sizes by Analysis Type

| Analysis Type | Minimum Window | Recommended | Notes |
|---------------|----------------|-------------|-------|
| Basic statistics | 50 samples | 100+ | Mean, std, percentiles |
| Spectral analysis | 2× longest period | 4× longest period | Capture full cycles |
| RQA | 200 samples | 500+ | Sufficient recurrence structure |
| Lyapunov | 500 samples | 1,000+ | Trajectory divergence |
| Topology (Betti) | 300 samples | 500+ | Persistent homology |
| Transfer entropy | 100 samples | 200+ | Per-window causality |

### Window vs. Phenomenon

```
minimum_window = 2 × (1 / slowest_frequency_of_interest)
recommended_window = 4 × (1 / slowest_frequency_of_interest)
```

**Example:** To detect 0.1 Hz oscillations:
- Minimum window: 2 × 10s = 20 seconds of data
- Recommended: 4 × 10s = 40 seconds of data

---

## Thresholds by Engine

### Lyapunov Exponent (λ_max)

| Value Range | Classification | Confidence Required | Action |
|-------------|----------------|--------------------:|--------|
| λ < -0.1 | Strongly stable | High | Monitor |
| -0.1 ≤ λ < 0 | Weakly stable | Medium | Watch |
| λ ≈ 0 (±0.01) | Marginal/Limit cycle | Low | Inconclusive without more data |
| 0 < λ < 0.05 | Weakly chaotic | Medium | Investigate |
| λ ≥ 0.05 | Strongly chaotic | High | **Actionable** |

**When is Lyapunov actionable?**
- λ > 0.05 with > 3,000 observations (> 10,000 for high confidence)
- Confirmed by RQA (DET < 0.5)
- Persistent over > 10 consecutive windows

### Recurrence Quantification (RQA)

| Metric | Healthy Range | Warning | Critical | Notes |
|--------|--------------|---------|----------|-------|
| RR (recurrence rate) | 0.01 - 0.10 | > 0.15 | > 0.30 | Too high = stuck state |
| DET (determinism) | 0.80 - 0.99 | < 0.70 | < 0.50 | Low = unpredictable |
| LAM (laminarity) | 0.70 - 0.95 | > 0.95 | > 0.99 | Too high = frozen |
| TT (trapping time) | 2 - 10 | > 15 | > 25 | High = stuck in states |
| ENTR (entropy) | 2.0 - 4.0 | < 1.5 | < 1.0 | Low = simple dynamics |
| **DIV (divergence)** | < 0.05 | > 0.1 | > 0.3 | DIV = 1/L_max; high = instability |

**When is RQA actionable?**
- DET drops below 0.6 over > 5 windows
- LAM exceeds 0.98 (system freezing)
- DIV exceeds 0.3 (rapid divergence/instability)
- Changes of > 0.15 from baseline

### Coherence / Coupling

| Metric | Healthy Range | Warning | Critical | Notes |
|--------|--------------|---------|----------|-------|
| Coherence ratio (λ₁/Σλ) | 0.3 - 0.7 | > 0.85 | > 0.95 | Over-coupling |
| Coherence ratio | 0.3 - 0.7 | < 0.15 | < 0.05 | Decoupling |
| Effective dimension | 2 - n/2 | < 1.5 | < 1.1 | Dimension collapse |
| Condition number | < 100 | > 500 | > 1000 | Numerical instability |

#### Coherence Rate of Change (Velocity)

| Δ Coherence per Window | Interpretation | Action |
|------------------------|----------------|--------|
| |Δ| < 0.02 | Stable | Normal |
| 0.02 ≤ |Δ| < 0.05 | Drifting | Watch |
| 0.05 ≤ |Δ| < 0.10 | Rapid change | Investigate |
| |Δ| ≥ 0.10 | **Alarm** | **Immediate attention** |

> **Key insight**: The *velocity* of coherence change is often more predictive than the absolute value. Rapid coupling or decoupling (|Δ| > 0.1/window) warrants immediate attention regardless of absolute level.

**When is coherence actionable?**
- Coherence change > 0.2 from baseline sustained > 10 windows
- Effective dimension drops below 1.5
- Coherence ratio exceeds 0.9 (rigidification)
- **Rate of change** |Δ| > 0.1 per window (rapid transition)

### Transfer Entropy / Causality

| Metric | Noise Floor | Weak | Moderate | Strong |
|--------|-------------|------|----------|--------|
| Transfer entropy | < 0.01 | 0.01-0.05 | 0.05-0.15 | > 0.15 |
| Granger F-stat | < 2.0 | 2-4 | 4-10 | > 10 |
| Net TE | < 0.02 | 0.02-0.05 | 0.05-0.10 | > 0.10 |

#### TE Noise Floor Scaling

> **Note**: The noise floor scales with histogram binning: `noise ≈ log₂(n_bins) / n_samples`

| Bins | Expected Noise Floor |
|------|---------------------|
| 8 bins | ~0.01 |
| 16 bins | ~0.02 |
| 32 bins | ~0.03 |

Increase your "weak" threshold if using more bins.

**When is causality actionable?**
- TE > 0.10 with > 1,000 observations
- Granger p < 0.01 (not just 0.05)
- Direction consistent over > 20 windows
- Confirmed by at least 2 methods

### Topology (Betti Numbers)

| Metric | Healthy | Warning | Critical | Notes |
|--------|---------|---------|----------|-------|
| β₀ (components) | 1 | 2 | > 2 | Fragmentation |
| β₁ (loops) | 0-2 | 3-5 | > 5 | Complex dynamics |
| Persistence entropy | 2-4 | < 1.5 | < 1.0 | Structural collapse |
| Topology change (Wasserstein) | < 0.2 | 0.2-0.5 | > 0.5 | Structural shift |

**When is topology actionable?**
- β₀ > 1 sustained over > 5 windows (fragmentation)
- Wasserstein distance > 0.5 from baseline
- With > 500 embedded points per window

### Health Score

| Score Range | Classification | Fleet Percentile | Action |
|-------------|----------------|------------------|--------|
| 85-100 | Healthy | Top 25% | Monitor |
| 70-84 | Good | 25-50% | Normal |
| 55-69 | Fair | 50-75% | Watch |
| 40-54 | Poor | 75-90% | Plan maintenance |
| 25-39 | At Risk | 90-95% | Schedule inspection |
| < 25 | Critical | Bottom 5% | **Immediate action** |

> **Note**: Percentiles are calculated **empirically from the fleet**, not assumed normal. For skewed health distributions, use actual distribution quantiles rather than assuming Gaussian cutoffs.

**When is health score actionable?**
- Drop of > 15 points from entity baseline
- Below fleet P10 for > 5 consecutive windows
- Multiple pillars showing degradation (agreement > 0.6)

### Z-Score Thresholds (Anomaly Detection)

| |z| Range | Classification | Fleet Size N ≥ 30 | Fleet Size N < 30 |
|-----------|----------------|------------------|-------------------|
| < 1.5 | Normal | Expected | Expected |
| 1.5 - 2.0 | Elevated | Watch | Likely noise |
| 2.0 - 2.5 | Warning | Investigate | Possibly noise |
| 2.5 - 3.0 | High | Likely real | Needs confirmation |
| > 3.0 | Critical | **Actionable** | Investigate carefully |

### Fleet Size Guidance

| Fleet Size | Statistical Approach | Recommended Methods |
|------------|---------------------|---------------------|
| N < 5 | No fleet statistics | Individual baselines only. Compare to self, not fleet. |
| N = 5-10 | Robust statistics only | Median, IQR, percentile ranks. **No z-scores.** |
| N = 10-30 | Limited parametric | Percentile ranks preferred. Z-scores with caution. |
| N ≥ 30 | Full parametric | Z-scores, clustering, full fleet analytics reliable. |

> **Rule**: Don't compute z-scores with N < 10. Don't trust them with N < 30.

---

## Signal Strength Classification

### Effect Size Requirements

Beyond statistical significance, require meaningful effect sizes:

| Metric Type | Negligible | Small | Medium | Large |
|-------------|------------|-------|--------|-------|
| Z-score | < 1.0 | 1-2 | 2-3 | > 3 |
| Percent change | < 5% | 5-15% | 15-30% | > 30% |
| Correlation change | < 0.1 | 0.1-0.2 | 0.2-0.4 | > 0.4 |
| Entropy change | < 0.2 | 0.2-0.5 | 0.5-1.0 | > 1.0 |

### Multi-Pillar Confirmation

| Pillars Agreeing | Confidence | Action |
|------------------|------------|--------|
| 1 of 4 | Low (25%) | Note, don't act |
| 2 of 4 | Moderate (50%) | Investigate |
| 3 of 4 | High (75%) | Plan action |
| 4 of 4 | Very High (95%) | **Act immediately** |

---

## When to Trust Results

### Green Flags (Trust the result)

- [ ] Data exceeds hard minimum requirements
- [ ] Sampling rate captures ≥10 cycles of phenomenon
- [ ] Baseline period is clean and representative
- [ ] Effect size is medium or large
- [ ] Finding persists over multiple windows (> 5)
- [ ] Multiple pillars agree (> 2)
- [ ] Fleet size adequate for statistical claims (N ≥ 30)
- [ ] No known data quality issues

### Red Flags (Question the result)

- [ ] Near minimum data requirements
- [ ] Single window anomaly
- [ ] Only one pillar shows the effect
- [ ] Effect size is small
- [ ] Baseline period questionable
- [ ] Recent sensor maintenance or calibration
- [ ] Operating conditions changed
- [ ] Fleet size < 10 for fleet-level claims
- [ ] Rapid coherence change without physical explanation

### Validation Checklist

Before reporting a finding as significant:

1. **Data sufficiency**: Does observation count exceed hard minimum?
2. **Sampling adequacy**: Does data capture ≥10 cycles of the phenomenon?
3. **Baseline validity**: Was baseline period representative?
4. **Effect magnitude**: Is effect size at least "medium"?
5. **Temporal persistence**: Does finding persist > 5 windows?
6. **Cross-pillar agreement**: Do ≥ 2 pillars agree?
7. **Physical plausibility**: Does the finding make sense?
8. **Actionability**: What would you actually do differently?

---

## SQL Validation Queries

### Query: Data Sufficiency Check

```sql
-- Check if data meets minimum requirements for each engine
-- Updated with revised Lyapunov minimums (3k hard, 10k soft)
CREATE VIEW v_data_sufficiency AS
WITH entity_stats AS (
    SELECT
        entity_id,
        COUNT(*) AS total_observations,
        COUNT(DISTINCT signal_id) AS n_signals,
        MAX(I) - MIN(I) AS time_span
    FROM observations
    GROUP BY entity_id
)
SELECT
    entity_id,
    total_observations,
    n_signals,

    -- Hard minimums (updated)
    CASE WHEN total_observations >= 3000 THEN 'OK' ELSE 'INSUFFICIENT' END AS lyapunov_status,
    CASE WHEN total_observations >= 1000 THEN 'OK' ELSE 'INSUFFICIENT' END AS correlation_dim_low_status,
    CASE WHEN total_observations >= 5000 THEN 'OK' ELSE 'INSUFFICIENT' END AS correlation_dim_high_status,
    CASE WHEN total_observations >= 1000 AND n_signals >= 3 THEN 'OK' ELSE 'INSUFFICIENT' END AS transfer_entropy_status,
    CASE WHEN total_observations >= 500 AND n_signals >= 2 THEN 'OK' ELSE 'INSUFFICIENT' END AS granger_status,
    CASE WHEN total_observations >= 500 THEN 'OK' ELSE 'INSUFFICIENT' END AS topology_status,
    CASE WHEN total_observations >= 1000 THEN 'OK' ELSE 'INSUFFICIENT' END AS rqa_status,
    CASE WHEN n_signals >= 3 THEN 'OK' ELSE 'INSUFFICIENT' END AS coherence_status,

    -- Soft minimums (recommended for reliability)
    CASE WHEN total_observations >= 10000 THEN 'RELIABLE'
         WHEN total_observations >= 3000 THEN 'MARGINAL'
         ELSE 'UNRELIABLE' END AS lyapunov_confidence,

    -- Overall assessment
    CASE
        WHEN total_observations >= 5000 AND n_signals >= 3 THEN 'FULL_ANALYSIS'
        WHEN total_observations >= 1000 AND n_signals >= 2 THEN 'PARTIAL_ANALYSIS'
        ELSE 'LIMITED_ANALYSIS'
    END AS analysis_capability

FROM entity_stats;
```

### Query: Baseline Validity Check

```sql
-- Validate baseline period
CREATE VIEW v_baseline_validity AS
WITH baseline_stats AS (
    SELECT
        entity_id,
        signal_id,
        COUNT(*) AS baseline_obs,
        AVG(y) AS baseline_mean,
        STDDEV(y) AS baseline_std,
        MIN(y) AS baseline_min,
        MAX(y) AS baseline_max
    FROM observations
    WHERE I <= (SELECT 0.20 * MAX(I) FROM observations o2 WHERE o2.entity_id = observations.entity_id)
    GROUP BY entity_id, signal_id
)
SELECT
    entity_id,
    signal_id,
    baseline_obs,
    baseline_mean,
    baseline_std,

    -- Validity checks
    CASE WHEN baseline_obs >= 50 THEN 'OK' ELSE 'TOO_FEW' END AS sample_size_check,
    CASE WHEN baseline_std / NULLIF(ABS(baseline_mean), 0.001) < 0.5 THEN 'STABLE' ELSE 'VOLATILE' END AS stability_check,
    CASE WHEN (baseline_max - baseline_min) / NULLIF(baseline_std, 0.001) < 6 THEN 'OK' ELSE 'OUTLIERS' END AS outlier_check,

    -- Overall
    CASE
        WHEN baseline_obs >= 50
             AND baseline_std / NULLIF(ABS(baseline_mean), 0.001) < 0.5
        THEN 'VALID'
        ELSE 'QUESTIONABLE'
    END AS baseline_validity

FROM baseline_stats;
```

### Query: Finding Significance Assessment

```sql
-- Assess whether findings meet actionability thresholds
CREATE VIEW v_finding_significance AS
SELECT
    h.entity_id,
    h.window_id,
    h.health_score,
    h.risk_level,
    h.primary_concern,

    -- Data sufficiency (from v_data_sufficiency)
    ds.analysis_capability,

    -- Effect size
    CASE
        WHEN ABS(h.health_score - b.baseline_health) > 20 THEN 'LARGE'
        WHEN ABS(h.health_score - b.baseline_health) > 10 THEN 'MEDIUM'
        WHEN ABS(h.health_score - b.baseline_health) > 5 THEN 'SMALL'
        ELSE 'NEGLIGIBLE'
    END AS effect_size,

    -- Persistence (consecutive windows below threshold)
    (SELECT COUNT(*)
     FROM health h2
     WHERE h2.entity_id = h.entity_id
       AND h2.window_id BETWEEN h.window_id - 5 AND h.window_id
       AND h2.health_score < b.baseline_health - 10
    ) AS consecutive_low_windows,

    -- Multi-pillar agreement
    CASE
        WHEN h.stability_score > 0.3
             AND h.predictability_score > 0.3
             AND h.physics_score > 0.3
             AND h.topology_score > 0.3
        THEN 4
        WHEN (CASE WHEN h.stability_score > 0.3 THEN 1 ELSE 0 END +
              CASE WHEN h.predictability_score > 0.3 THEN 1 ELSE 0 END +
              CASE WHEN h.physics_score > 0.3 THEN 1 ELSE 0 END +
              CASE WHEN h.topology_score > 0.3 THEN 1 ELSE 0 END) >= 3
        THEN 3
        WHEN (CASE WHEN h.stability_score > 0.3 THEN 1 ELSE 0 END +
              CASE WHEN h.predictability_score > 0.3 THEN 1 ELSE 0 END +
              CASE WHEN h.physics_score > 0.3 THEN 1 ELSE 0 END +
              CASE WHEN h.topology_score > 0.3 THEN 1 ELSE 0 END) >= 2
        THEN 2
        ELSE 1
    END AS pillars_agreeing,

    -- FINAL VERDICT
    CASE
        WHEN ds.analysis_capability = 'LIMITED_ANALYSIS' THEN 'INSUFFICIENT_DATA'
        WHEN ABS(h.health_score - b.baseline_health) < 5 THEN 'NOT_SIGNIFICANT'
        WHEN (SELECT COUNT(*) FROM health h2
              WHERE h2.entity_id = h.entity_id
                AND h2.window_id BETWEEN h.window_id - 5 AND h.window_id
                AND h2.health_score < b.baseline_health - 10) < 3 THEN 'NOT_PERSISTENT'
        WHEN (CASE WHEN h.stability_score > 0.3 THEN 1 ELSE 0 END +
              CASE WHEN h.predictability_score > 0.3 THEN 1 ELSE 0 END +
              CASE WHEN h.physics_score > 0.3 THEN 1 ELSE 0 END +
              CASE WHEN h.topology_score > 0.3 THEN 1 ELSE 0 END) < 2 THEN 'NOT_CONFIRMED'
        WHEN ABS(h.health_score - b.baseline_health) > 15 THEN 'ACTIONABLE'
        ELSE 'WATCH'
    END AS finding_status

FROM health h
JOIN v_data_sufficiency ds ON h.entity_id = ds.entity_id
JOIN (
    SELECT entity_id, AVG(health_score) AS baseline_health
    FROM health
    WHERE window_id <= (SELECT 0.20 * MAX(window_id) FROM health)
    GROUP BY entity_id
) b ON h.entity_id = b.entity_id
WHERE h.window_id = (SELECT MAX(window_id) FROM health);
```

### Query: Fleet Statistics Validity

```sql
-- Check if fleet is large enough for statistical claims
-- Updated with 4-tier fleet size guidance
CREATE VIEW v_fleet_validity AS
SELECT
    COUNT(DISTINCT entity_id) AS fleet_size,

    CASE
        WHEN COUNT(DISTINCT entity_id) >= 30 THEN 'RELIABLE - Full parametric statistics'
        WHEN COUNT(DISTINCT entity_id) >= 10 THEN 'MARGINAL - Use percentiles, z-scores with caution'
        WHEN COUNT(DISTINCT entity_id) >= 5 THEN 'LIMITED - Robust stats only (median, IQR), NO z-scores'
        ELSE 'MINIMAL - Individual baselines only, no fleet statistics'
    END AS z_score_validity,

    CASE
        WHEN COUNT(DISTINCT entity_id) >= 30 THEN 'Full clustering meaningful'
        WHEN COUNT(DISTINCT entity_id) >= 10 THEN 'Basic clustering OK'
        WHEN COUNT(DISTINCT entity_id) >= 5 THEN 'Limited clustering'
        ELSE 'Clustering not recommended'
    END AS clustering_validity,

    CASE
        WHEN COUNT(DISTINCT entity_id) >= 50 THEN 'Full fleet analytics'
        WHEN COUNT(DISTINCT entity_id) >= 30 THEN 'Standard fleet analytics'
        WHEN COUNT(DISTINCT entity_id) >= 10 THEN 'Basic fleet analytics'
        WHEN COUNT(DISTINCT entity_id) >= 5 THEN 'Minimal fleet comparison'
        ELSE 'Individual entity analysis only'
    END AS recommendation

FROM observations;
```

### Query: Coherence Rate of Change

```sql
-- Monitor coherence velocity (rate of change)
-- Rapid changes are often more predictive than absolute values
CREATE VIEW v_coherence_velocity AS
WITH coherence_with_lag AS (
    SELECT
        entity_id,
        window_id,
        coherence_ratio,
        LAG(coherence_ratio) OVER (PARTITION BY entity_id ORDER BY window_id) AS prev_coherence
    FROM geometry
)
SELECT
    entity_id,
    window_id,
    ROUND(coherence_ratio, 3) AS coherence,
    ROUND(coherence_ratio - prev_coherence, 4) AS delta_coherence,
    CASE
        WHEN ABS(coherence_ratio - prev_coherence) >= 0.10 THEN 'ALARM - Rapid transition'
        WHEN ABS(coherence_ratio - prev_coherence) >= 0.05 THEN 'WARNING - Fast change'
        WHEN ABS(coherence_ratio - prev_coherence) >= 0.02 THEN 'WATCH - Drifting'
        ELSE 'STABLE'
    END AS velocity_status
FROM coherence_with_lag
WHERE prev_coherence IS NOT NULL
ORDER BY ABS(coherence_ratio - prev_coherence) DESC;
```

---

## Summary: The Honest Assessment Framework

### Before Claiming a Finding is Significant:

| Check | Minimum | Recommended |
|-------|---------|-------------|
| Observations | Engine-specific hard minimum | 2-5× hard minimum |
| Sampling | ≥10 cycles of phenomenon | ≥20 cycles |
| Effect size | Small (5% change) | Medium (15% change) |
| Persistence | 3 windows | 5+ windows |
| Pillar agreement | 1 pillar | 2+ pillars |
| Fleet size (for fleet claims) | 10 entities | 30+ entities |

### The Golden Rule

**"If you wouldn't bet $1,000 of your own money on this finding being real, don't report it as significant."**

---

*Document version: 2.0*
*Updated: January 2025*
*Changes: Revised Lyapunov minimums, added sampling context, DIV metric, coherence velocity, window size guidance, expanded fleet tiers*
*Purpose: Prevent overstated findings and false positives*
