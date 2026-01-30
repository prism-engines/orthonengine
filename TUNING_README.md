# TUNING_README: How to Tune PRISM/ORTHON on Benchmark Datasets

**Don't just throw data in a bowl and bake at 350. Look at what you're cooking.**

---

## The Wrong Way

```
1. Download dataset
2. Run full pipeline
3. Compare "healthy" vs "failing" entities
4. Declare victory: "56% less coherence!"
5. Ship it

Problem: You measured entity averages, not temporal dynamics.
         You skipped the ground truth entirely.
         You have no idea if PRISM detected anything early.
```

## The Right Way

```
1. Understand the dataset structure
2. Find the ground truth (when faults occur)
3. Compute metrics WITHIN each experiment over time
4. Measure: Did metrics deviate BEFORE the fault?
5. Quantify lead time for each metric
6. Learn which metrics detect which fault types
```

---

## Phase 1: Understand the Dataset

Before computing anything, answer:

| Question | Why It Matters |
|----------|----------------|
| What is being measured? | Sensors determine which physics apply |
| What faults are injected? | Different faults have different signatures |
| When are faults injected? | Ground truth timestamps for validation |
| Is there a healthy baseline? | Need reference for "normal" behavior |
| How long are the experiments? | Determines window sizes |

### Example: SKAB Dataset

```
System: Water pump testbed
Sensors: Pressure, Temperature, Vibration (x2), Current, Voltage, Flow, Thermocouple

Fault Types:
├── valve1/     16 experiments - Outlet valve manipulation
├── valve2/      4 experiments - Inlet valve manipulation
└── other/      14 experiments - Mixed:
    ├── 1-4:    Fluid dynamics (leaks)
    ├── 5-9:    Rotor imbalance (5 behaviors)
    ├── 10-11:  Volume changes
    ├── 12-13:  Cavitation
    └── 14:     Thermal

Ground Truth: anomaly column (0/1), changepoint column (0/1)
Structure: Each experiment starts normal, fault injected mid-experiment
```

---

## Phase 2: Ingest WITH Ground Truth

**Critical:** Don't discard the labels. Keep them separate but aligned.

```
observations.parquet          # Sensor data for PRISM
├── entity_id
├── signal_id
├── I
├── y
└── unit

ground_truth.parquet          # Labels for validation
├── entity_id
├── I
├── anomaly        (0 = normal, 1 = anomaly)
├── changepoint    (0 = same regime, 1 = transition)
└── fault_type     (valve1, cavitation, etc.)
```

**Extract fault timestamps:**

```python
# For each entity, find when fault starts
fault_times = ground_truth.group_by('entity_id').agg([
    pl.col('I').filter(pl.col('changepoint') == 1).first().alias('fault_I'),
    pl.col('I').max().alias('end_I'),
])
```

---

## Phase 3: Compute Rolling Metrics

**Not entity-level averages. Time-series within each experiment.**

```python
# For each entity, compute metrics in rolling windows
rolling_coherence = []
for entity in entities:
    for window_start in range(0, max_I - window_size, stride):
        window_data = get_window(entity, window_start, window_size)
        coh = compute_coherence(window_data)
        rolling_coherence.append({
            'entity_id': entity,
            'I': window_start + window_size // 2,  # Window center
            'coherence': coh
        })
```

**Metrics to compute rolling:**

| Category | Metrics |
|----------|---------|
| Geometry | coherence, effective_dimension, mean_correlation |
| Dynamics | lyapunov, rqa_determinism, rqa_laminarity |
| Information | mean_transfer_entropy, causal_hierarchy_strength |
| Primitives | mean_entropy, mean_hurst, total_energy (rms²) |

---

## Phase 4: Align to Fault Timestamp

**Normalize time so fault = 0:**

```python
# Shift I so fault occurs at I=0
aligned = rolling_metrics.join(fault_times, on='entity_id')
aligned = aligned.with_columns([
    (pl.col('I') - pl.col('fault_I')).alias('I_relative')
])

# Now:
#   I_relative < 0  →  Pre-fault (should be stable)
#   I_relative = 0  →  Fault injection
#   I_relative > 0  →  Post-fault (should show anomaly)
```

---

## Phase 5: Establish Baseline

**Define "normal" from pre-fault window:**

```python
# Use early portion as baseline (before any degradation)
baseline_window = (-500, -200)  # Well before fault

baseline_stats = aligned.filter(
    (pl.col('I_relative') >= baseline_window[0]) &
    (pl.col('I_relative') <= baseline_window[1])
).group_by('entity_id').agg([
    pl.col('coherence').mean().alias('baseline_coherence'),
    pl.col('coherence').std().alias('baseline_coherence_std'),
    pl.col('lyapunov').mean().alias('baseline_lyapunov'),
    pl.col('lyapunov').std().alias('baseline_lyapunov_std'),
    # ... etc
])
```

---

## Phase 6: Detect Deviations

**When does each metric first deviate from baseline?**

```python
def find_deviation_point(series, baseline_mean, baseline_std, threshold=2.0):
    """Find first I where metric exceeds threshold standard deviations."""
    deviations = (series - baseline_mean).abs() / baseline_std
    first_deviation = series.filter(deviations > threshold).first()
    return first_deviation['I_relative']

# For each entity and metric
deviation_points = []
for entity in entities:
    entity_data = aligned.filter(pl.col('entity_id') == entity)
    baseline = baseline_stats.filter(pl.col('entity_id') == entity)

    coh_deviation_I = find_deviation_point(
        entity_data['coherence'],
        baseline['baseline_coherence'],
        baseline['baseline_coherence_std']
    )

    deviation_points.append({
        'entity_id': entity,
        'coherence_deviation_I': coh_deviation_I,
        'coherence_lead_time': -coh_deviation_I if coh_deviation_I < 0 else None,
        # Negative I_relative means BEFORE fault = early warning!
    })
```

---

## Phase 7: Quantify Lead Time

**The key output: How early did each metric detect the fault?**

```
┌────────────┬────────────┬───────────┬──────────┬─────────────┬───────────┐
│ entity_id  │ fault_type │ coherence │ lyapunov │ entropy     │ best_lead │
│            │            │ lead_time │ lead_time│ lead_time   │ _time     │
├────────────┼────────────┼───────────┼──────────┼─────────────┼───────────┤
│ valve1_0   │ valve      │ 47        │ 23       │ 15          │ 47        │
│ valve1_1   │ valve      │ 52        │ 31       │ 18          │ 52        │
│ other_5    │ imbalance  │ 12        │ 45       │ 38          │ 45        │
│ other_12   │ cavitation │ 8         │ 62       │ 71          │ 71        │
└────────────┴────────────┴───────────┴──────────┴─────────────┴───────────┘

Insight: Coherence is best early warning for valve faults.
         Lyapunov/entropy better for mechanical (imbalance) faults.
         Entropy best for chaotic faults (cavitation).
```

---

## Phase 8: Learn Fault Signatures

**Different faults have different signatures:**

### Valve Faults (Flow Restriction)
```
Expected signature:
- Pressure-flow correlation breaks down
- Coherence drops (decoupling)
- Entropy may stay stable (not chaotic, just restricted)
- Gradual degradation

Best detectors: Coherence, pressure-flow correlation
```

### Rotor Imbalance (Mechanical)
```
Expected signature:
- Vibration energy increases
- Accelerometer signals become coupled (both see the imbalance)
- Lyapunov may go positive (chaotic vibration)
- Can be sudden or gradual depending on type

Best detectors: Lyapunov, vibration coherence, RMS energy
```

### Cavitation (Two-Phase Flow)
```
Expected signature:
- High-frequency noise in pressure/flow
- Entropy spikes (unpredictable)
- Lyapunov strongly positive (chaotic)
- Correlations become noisy

Best detectors: Entropy, Lyapunov, high-frequency energy
```

### Thermal Anomalies
```
Expected signature:
- Temperature leads other signals (causal direction)
- Gradual drift in temperature-dependent sensors
- Transfer entropy changes (temperature becomes driver)

Best detectors: Transfer entropy, temperature-X correlations
```

---

## Phase 9: Build Detection Rules

**From the signatures, build detection logic:**

```python
def detect_anomaly(rolling_metrics, baseline, fault_type_hints=None):
    """
    Returns anomaly probability and likely fault type.
    """
    alerts = []

    # Coherence drop (valve faults)
    if coherence_z_score < -2.0:
        alerts.append(('coherence_drop', 'valve_fault', abs(coherence_z_score)))

    # Lyapunov going positive (mechanical/chaotic)
    if lyapunov > 0 and baseline_lyapunov < 0:
        alerts.append(('lyapunov_positive', 'mechanical_or_cavitation', lyapunov))

    # Entropy spike (cavitation, chaos)
    if entropy_z_score > 2.5:
        alerts.append(('entropy_spike', 'cavitation', entropy_z_score))

    # Transfer entropy inversion (thermal)
    if te_temperature_to_others > baseline_te * 1.5:
        alerts.append(('te_shift', 'thermal', te_temperature_to_others))

    return sorted(alerts, key=lambda x: x[2], reverse=True)
```

---

## Phase 10: Validate and Iterate

**Compute precision, recall, lead time:**

```python
results = []
for entity in test_entities:
    # PRISM detection
    prism_alert_I = detect_first_alert(entity)

    # Ground truth
    actual_fault_I = fault_times[entity]

    # Metrics
    if prism_alert_I is not None:
        if prism_alert_I < actual_fault_I:
            # Early detection (good!)
            results.append({
                'entity': entity,
                'outcome': 'early_detection',
                'lead_time': actual_fault_I - prism_alert_I
            })
        elif prism_alert_I < actual_fault_I + tolerance:
            # Detection (acceptable)
            results.append({
                'entity': entity,
                'outcome': 'detection',
                'lead_time': 0
            })
        else:
            # Late detection
            results.append({
                'entity': entity,
                'outcome': 'late',
                'lead_time': -(prism_alert_I - actual_fault_I)
            })
    else:
        # Missed
        results.append({
            'entity': entity,
            'outcome': 'miss',
            'lead_time': None
        })

# Summary
print(f"Early detections: {count_early} ({pct_early}%)")
print(f"Mean lead time: {mean_lead_time} samples")
print(f"Misses: {count_miss} ({pct_miss}%)")
```

---

## Summary: The Tuning Loop

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   1. UNDERSTAND          What is this data? What are the faults?│
│         ↓                                                       │
│   2. INGEST              Keep ground truth aligned with sensors │
│         ↓                                                       │
│   3. COMPUTE ROLLING     Metrics over time, not entity averages │
│         ↓                                                       │
│   4. ALIGN TO FAULT      Normalize time: fault = 0              │
│         ↓                                                       │
│   5. ESTABLISH BASELINE  What's "normal" for this system?       │
│         ↓                                                       │
│   6. DETECT DEVIATIONS   When did each metric first deviate?    │
│         ↓                                                       │
│   7. QUANTIFY LEAD TIME  How early did we catch it?             │
│         ↓                                                       │
│   8. LEARN SIGNATURES    Which metrics catch which faults?      │
│         ↓                                                       │
│   9. BUILD RULES         Codify detection logic                 │
│         ↓                                                       │
│   10. VALIDATE           Precision, recall, lead time           │
│         ↓                                                       │
│   ┌─────┴─────┐                                                 │
│   │  ITERATE  │ ←─── Adjust thresholds, windows, metrics        │
│   └───────────┘                                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Appendix: SKAB Fault Signatures (Expected)

Based on physics and SKAB documentation:

| Fault Type | Coherence | Lyapunov | Entropy | Transfer Entropy | Best Detector |
|------------|-----------|----------|---------|------------------|---------------|
| Valve (outlet) | ↓↓ | → | → | ↓ | Coherence |
| Valve (inlet) | ↓↓ | → | ↑ | ↓ | Coherence |
| Rotor imbalance | ↑ (vib) | ↑↑ | ↑ | → | Lyapunov |
| Leak | ↓ | → | ↑ | ↓ | Coherence slope |
| Cavitation | ↓↓ | ↑↑↑ | ↑↑↑ | ↓↓ | Entropy |
| Thermal | → | → | → | ↑↑ (temp) | Transfer Entropy |

Legend: ↓↓ = strong decrease, ↓ = decrease, → = stable, ↑ = increase, ↑↑ = strong increase

---

## The Point

**PRISM computes the metrics. ORTHON interprets them. But interpretation requires:**

1. Understanding what you're measuring
2. Ground truth to validate against
3. Temporal analysis, not just averages
4. Fault-specific signatures
5. Quantified lead times

**"Systems lose coherence before they fail"** is the thesis.

**Tuning proves it** - or disproves it - with numbers:
- "Coherence dropped 47 samples before labeled fault in 94% of valve failures"
- "Lyapunov went positive 62 samples before cavitation in 100% of cases"
- "Mean lead time across all fault types: 38 samples"

That's science. That's a paper. That's PRISM tuned.

---

*Don't bake at 350 and hope. Measure the ingredients. Time the oven. Check the result.*
