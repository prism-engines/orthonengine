# Ørthon

**Not monitoring. Understanding.**

---

## What is this?

Ørthon is a domain-agnostic system for understanding what the hell is going on with any system.

Give it signals over time. It tells you:
- Is energy conserved or dissipating?
- Are the signals still coupled?
- Where is energy flowing?
- Is the system stable or diverging?
- If something fails, where does it propagate?

No thresholds. No rules. No domain expertise required.

Pure math.

---

## The Problem

Traditional monitoring tools ask: **"Is this value bad?"**

```
Temperature > 100°C?  → ALARM
Vibration > 10mm/s?   → ALARM
Pressure < 5 bar?     → ALARM
```

This requires:
- Domain expertise to set thresholds
- Different rules for every system
- Constant tuning
- Missing the failures that don't fit the rules

---

## The Insight

**Systems lose coherence before they fail.**

A healthy system has structure:
- Signals are coupled
- Energy is conserved
- State is stable

A failing system loses structure:
- Signals decouple
- Energy dissipates
- State diverges

You don't need to know what "healthy" looks like for a specific pump, reactor, or engine. You just need to detect when structure is being lost.

---

## The Approach

Ørthon analyzes systems top-down, starting with thermodynamics:

```
L4: Thermodynamics  →  Is energy conserved?
        ↓               If not, something is wrong.
L3: Mechanics       →  Where is energy going?
        ↓               Which signals are sources? Sinks?
L2: Coherence       →  Are signals still coupled?
        ↓               Decoupling = pathways breaking.
L1: State           →  Where is the system now?
                        Consequence of the above.
```

Energy is the constraint. Everything else is consequence.

---

## The Math

### For Engineers

```python
# The Ørthon signal
if (energy_dissipating and coherence_dropping and state_diverging):
    print("System failing")
```

### For Scientists

The state space forms a manifold M where signals are projections from a higher-dimensional phase space.

Coherence measures the preservation of relationships between signals - mathematically, whether trajectories remain on the same symplectic leaf.

Energy (or its proxy) follows Hamiltonian dynamics when the system is healthy. Dissipation appears as non-zero Lie derivative of the symplectic form: £_X ω ≠ 0.

State distance is computed via Mahalanobis distance from a baseline distribution, equivalent to the Fisher-Rao metric on the statistical manifold.

### For Mathematicians

The coherence functional measures deviation from the symplectic leaf. When it diverges, Liouville's theorem fails locally - energy is no longer conserved, phase space volume contracts, and the system is dissipating into a lower-dimensional attractor.

---

## Key Concepts

### State Distance

How far is the system from its baseline state?

Computed as Mahalanobis distance using ALL available metrics (entropy, kurtosis, spectral features, etc.). Not just one number - the full high-dimensional fingerprint.

```
state_distance ≈ 0  →  At baseline (normal)
state_distance = 2  →  2σ from baseline (notable)
state_distance > 3  →  Significantly different (investigate)
```

### State Velocity

How fast is the system moving through phase space?

```
state_velocity ≈ 0  →  Stable
state_velocity > 0  →  Moving away from baseline
state_velocity < 0  →  Returning to baseline
```

This is the generalized version of `hd_slope` (Hurst derivative), but using ALL metrics instead of just one.

### Coherence

How coupled are the signals?

Computed as mean correlation across all signal pairs over a rolling window.

```
coherence ≈ 1.0  →  Tightly coupled
coherence ≈ 0.5  →  Moderately coupled
coherence ≈ 0.0  →  Decoupled
```

### Coherence Velocity

How fast is coupling changing?

```
coherence_velocity ≈ 0   →  Stable coupling
coherence_velocity < 0   →  Decoupling (relationships breaking)
coherence_velocity > 0   →  Coupling increasing
```

### Energy Proxy

A unit-agnostic measure of system energy:

```
E_proxy = y² + (dy/dt)²
```

This isn't "real" energy (Joules), but it BEHAVES like energy:
- Conserved in stable systems
- Dissipates in damped systems
- Flows between coupled signals

When units are provided, Ørthon computes real energy instead.

### Dissipation Rate

How fast is energy leaving the system?

```
dissipation_rate = -d(energy)/dt  when energy is decreasing
```

Positive dissipation = energy leaving = friction, damage, or intentional damping.

---

## The Ørthon Signal

The core degradation indicator:

```python
orthon_signal = (
    energy_dissipating AND
    coherence_dropping AND
    state_diverging
)
```

When all three occur together:
- Energy is leaving the system
- Signals are decoupling
- State is moving away from baseline

This combination precedes failure across domains - pumps, bearings, reactors, engines, processes.

---

## Usage

### Quick Start

```python
from orthon import analyze

# Upload any time-series data
results = analyze("sensor_data.csv")

# Get the diagnosis
print(results.diagnosis)
# {
#     'severity': 'warning',
#     'orthon_signal': True,
#     'summary': 'Energy dissipating at 0.0023/unit. Coherence dropped from 0.82 to 0.54. State distance 2.7σ from baseline.'
# }
```

### With Units (Real Physics)

```python
from orthon import analyze, PhysicsConstants

# Provide domain knowledge for real energy calculations
results = analyze(
    "pump_data.csv",
    signal_units={
        "flow": "velocity",
        "pressure": "pressure",
        "temperature": "temperature"
    },
    constants=PhysicsConstants(
        mass=10.0,        # kg
        volume=0.1,       # m³
        thermal_mass=500  # J/K
    )
)

# Now energy is in Joules, not proxy units
print(results.L4_thermodynamics['energy_mean'])  # 4523.7 J
```

### Interpreting Results

```python
# Full analysis (top-down)
analysis = results.analyze_system("pump_001")

# L4: Thermodynamics (start here)
print(analysis['L4_thermodynamics'])
# Is energy conserved? Where is it going?

# L3: Mechanics
print(analysis['L3_mechanics'])
# Which signals are sources? Which are sinks?

# L2: Coherence
print(analysis['L2_coherence'])
# Are signals still coupled? Decoupling?

# L1: State
print(analysis['L1_state'])
# Where is the system? Stable? Diverging?
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         Ørthon                              │
│                        (Brain)                              │
│                                                             │
│  • Builds manifests based on data analysis                  │
│  • Interprets physics (proxy or real)                       │
│  • Provides diagnosis and insights                          │
│  • Serves UI and API                                        │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ manifest
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                         PRISM                               │
│                       (Muscle)                              │
│                                                             │
│  • Executes exactly what manifest says                      │
│  • Computes: hurst, entropy, correlation, derivatives...    │
│  • Outputs: physics.parquet, primitives.parquet, etc.       │
│  • Zero domain knowledge. Pure math.                        │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ results
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                          ML                                 │
│                       (Models)                              │
│                                                             │
│  • Optional machine learning layer                          │
│  • Fault classification, RUL prediction                     │
│  • Uses PRISM outputs as features                           │
└─────────────────────────────────────────────────────────────┘
```

**Ørthon is the brain.** It understands.

**PRISM is the muscle.** It computes.

**ML is optional.** For when you want predictions, not just understanding.

---

## Outputs

### physics.parquet

The core output with all physics layers:

| Column | Description |
|--------|-------------|
| entity_id | Which entity (machine, process, etc.) |
| I | Index (time, cycle, sequence) |
| state_distance | Mahalanobis distance from baseline |
| state_velocity | Rate of state change |
| state_acceleration | Acceleration of state change |
| coherence | Mean coupling strength |
| coherence_velocity | Rate of coherence change |
| energy_proxy | System energy (proxy or real) |
| energy_velocity | Rate of energy change |
| dissipation_rate | Energy loss rate |
| entropy_production | Rate of disorder increase |

### primitives.parquet

Signal-level metrics (one row per signal):

| Column | Description |
|--------|-------------|
| entity_id | Which entity |
| signal_id | Which signal |
| hurst | Memory/persistence (0-1) |
| entropy | Complexity/disorder |
| lyapunov | Chaos/predictability |
| kurtosis | Tail heaviness |
| rms | Root mean square |
| ... | 30+ metrics |

### geometry.parquet

Signal pair relationships:

| Column | Description |
|--------|-------------|
| entity_id | Which entity |
| signal_a | First signal |
| signal_b | Second signal |
| correlation | Pearson correlation |
| mutual_info | Mutual information |
| cointegration | Long-term relationship |

---

## Who Is This For?

### The 2am Grad Student

"I have data. I need to understand it. I need thesis figures by morning."

Upload CSV. Get analysis. Get publication-ready insights.

### The Plant Engineer

"I have 10,000 sensors. Something is wrong somewhere. What?"

Run Ørthon across all assets. Find which ones are decoupling, dissipating, diverging.

### The Reliability Team

"We want to predict failures before they happen."

The Ørthon signal (dissipating + decoupling + diverging) precedes failure. Use it for early warning.

### The Research Scientist

"I need domain-agnostic methods that work across systems."

Same math works on pumps, reactors, turbines, processes, anything with signals over time.

---

## FAQ

**Q: How is this different from threshold monitoring?**

Threshold monitoring asks "is this value bad?" which requires domain expertise.

Ørthon asks "is this system losing structure?" which is domain-agnostic.

**Q: Do I need to provide units?**

No. Proxy physics works without units. You lose magnitude (Joules) but keep dynamics (conserved? dissipating?).

If you provide units, you get real physics.

**Q: What if I only have a few signals?**

Ørthon works with 2+ signals. More signals = richer coherence analysis, but the core metrics work regardless.

**Q: How much historical data do I need?**

Baseline is computed from the first 10% of data (or 100 points minimum). More baseline = more robust.

Different engines have different minimum data requirements for statistically meaningful results:

| Engine | Minimum | Rationale |
|--------|---------|-----------|
| Lyapunov | ~10,000 pts | Attractor reconstruction (Wolf et al., 1985) |
| Attractor/Basin | ~1,000 pts | Strange attractor reconstruction |
| Transfer Entropy | ~500 pts | Information transfer statistics |
| Recurrence | ~500 pts | Sufficient recurrence density |
| GARCH | ~250 pts | Volatility clustering detection |
| Cointegration | ~250 pts | Long-run equilibrium detection |
| Hurst | ~256 pts | R/S rescaling statistics |
| Spectral | 2× longest cycle | Nyquist + frequency resolution |
| Entropy | ~100 pts | Symbol sequence statistics |

**Results below these thresholds are flagged as `insufficient_data` in the analysis.**

Configurable in SQL: Edit `config_lyapunov` table or `00_config.sql` to adjust thresholds.

**Q: Can I use this for real-time monitoring?**

Yes. Compute metrics on rolling windows. The architecture supports streaming via API callbacks.

---

## Methodology

### Data Sufficiency

Not all metrics are meaningful with limited data. Ørthon tracks data sufficiency per entity and flags results accordingly:

- **`lyapunov_reliable`**: TRUE when n_observations >= 10,000
- **`hurst_reliable`**: TRUE when n_observations >= 256
- **`attractor_reliable`**: TRUE when n_observations >= 1,000
- **`analysis_tier`**: `full_analysis` (10k+), `standard_analysis` (1k+), `basic_analysis` (256+), `limited_analysis` (<256)

When data is insufficient:
- Lyapunov-based stability classes show `insufficient_data`
- Lyapunov statistics return `NULL`
- Fleet summaries track `n_lyapunov_reliable` vs `n_insufficient_data`

Thresholds are configurable in `orthon/sql/00_config.sql`.

### Geometric Attribution

Forces acting on a system are classified as **endogenous** or **exogenous**:

| Force Type | Centroid | Shape | Meaning |
|------------|----------|-------|---------|
| **Exogenous** | Moving | Intact | External force translating the whole system |
| **Endogenous** | Stable | Deforming | Internal tension, vectors pushing against each other |
| **Mixed** | Moving | Deforming | Both external and internal forces active |
| **Stable** | Stable | Intact | System at rest |

Key metrics:
- **Centroid drift**: Did the geometry translate?
- **Dispersion change**: Did the shape deform?
- **Attribution ratio**: exogenous_count / endogenous_count (>1 = externally driven)

See `orthon/sql/21_geometric_attribution.sql` for implementation.

### Vector Energy

Per-signal energy decomposition:

| Component | Formula | Meaning |
|-----------|---------|---------|
| **Kinetic** | Σ(return²) | How fast is the signal moving? Activity level. |
| **Potential** | (y - centroid)² | How far from equilibrium? Tension stored. |
| **Total** | kinetic + potential | Full energy contribution to system |
| **Energy Fraction** | total / Σtotal | Share of system energy this signal carries |

What you learn:
- Which signal is **driving** the system (highest energy fraction)
- Which is **absorbing** vs **releasing** energy
- Energy **migrating** between vectors across windows
- **Kinetic ratio**: high = active/volatile, low = stable
- **Potential ratio**: high = displaced from group, low = near centroid

See `orthon/sql/22_vector_energy.sql` for implementation.

### Baseline & Deviation Detection

Self-referential anomaly detection. The system defines its own "normal."

**Phase 1 — Baseline**: First 10% of data establishes normal (mean, std, percentiles)
**Phase 2 — Monitor**: Every point gets z-scores vs baseline
**Phase 3 — Flag**: Any metric exceeding threshold triggers alert

| Severity | Condition |
|----------|-----------|
| Normal | All z-scores < 2 |
| Warning | Any z-score > 2 |
| Critical | Any z-score > 3 |

The question: **"Is this still normal?"** — not "What will go wrong?"

### The Incident Summary

When incidents occur, ORTHON produces a comprehensive report:

```
INCIDENT SUMMARY: Entity-001
============================
FIRST DETECTION:    Window 4,847 / energy_proxy / z=3.2
PROPAGATION PATH:   energy → coherence → state (59 windows)
PEAK SEVERITY:      Critical at window 4,902
FORCE ATTRIBUTION:  EXOGENOUS (ratio: 2.7 — externally driven)
ENERGY BALANCE:     Injected: +0.34, Dissipated: +0.47, Gap: -0.13
                    ← UNMEASURED SINK DETECTED
ORTHON SIGNAL:      ⚠️ ACTIVE — Symplectic structure loss detected
```

See `orthon/sql/24_incident_summary.sql` for implementation.

### Eigenvalue-Based Coherence

Coherence is computed from the eigenvalue spectrum of the signal correlation matrix:

| Metric | Formula | Meaning |
|--------|---------|---------|
| `coherence` | λ₁/Σλᵢ | Fraction of variance in first mode (0-1) |
| `effective_dim` | (Σλᵢ)²/Σλᵢ² | Number of independent modes (1-N) |
| `eigenvalue_entropy` | -Σ(pᵢ log pᵢ)/log(N) | Disorder of eigenvalue distribution (0-1) |

Interpretation:

| Health | coherence | effective_dim | eigenvalue_entropy |
|--------|-----------|---------------|-------------------|
| Healthy | High (~0.7+) | Low (~1-2) | Low (~0.2) |
| Degrading | Dropping | Rising | Rising |
| Failed | Low (~1/N) | High (~N) | High (~1.0) |

---

## Theoretical Foundation

For those who want the deep math:

Ørthon is grounded in:

- **Dynamical Systems Theory**: State space, attractors, Lyapunov stability
- **Symplectic Geometry**: Energy conservation, Hamiltonian flows, phase space structure
- **Information Theory**: Entropy, mutual information, transfer entropy
- **Statistical Mechanics**: Dissipation, entropy production, equilibrium
- **Optimal Transport**: Wasserstein distance for comparing distributions

The key insight is that **symplectic structure (energy conservation + phase space preservation) is what breaks when systems fail**.

Coherence measures whether signals remain on the same symplectic leaf. When they don't, Liouville's theorem fails locally, energy dissipates, and the system collapses toward a lower-dimensional attractor.

This is universal across domains because it's geometry, not domain-specific rules.

---

## Citation

If you use Ørthon in research, please cite:

```bibtex
@software{orthon2025,
  title = {Ørthon: Domain-Agnostic System Understanding via Geometric Coherence Analysis},
  author = {Rudder, Jason and Rudder, Avery},
  year = {2025},
  url = {https://github.com/orthon/orthon}
}
```

---

## License

- **PRISM** (compute engines): MIT License - free for all uses
- **Ørthon** (brain/interpretation): Commercial license for enterprise, free for academic use with citation

---

## Acknowledgments

- Avery Rudder for the Laplace transform insight that started this
- Dr. Jeffrey Dick for early validation discussions
- The Claude instances who wrote most of this code

---

**Ørthon. Not monitoring. Understanding.**

*"The coherence functional measures deviation from the symplectic leaf. When it diverges, Liouville's theorem fails locally - energy is no longer conserved, phase space volume contracts, and the system is dissipating into a lower-dimensional attractor."*

Or in simple terms: we check if things are falling apart.
