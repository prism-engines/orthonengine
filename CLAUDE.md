# ORTHON - AI Instructions

## Architecture

```
ORTHON = Brain (orchestration, manifest, interpretation)
PRISM  = Muscle (pure computation, no decisions)

ORTHON creates observations.parquet + manifest.yaml
PRISM reads them, computes everything, writes output parquets
```

## The One Rule

```
observations.parquet and manifest.yaml ALWAYS go to:
/Users/jasonrudder/prism/data/

NO EXCEPTIONS. No subdirectories. No domain folders.
```

## PRISM Format (observations.parquet)

| Column | Type | Description |
|--------|------|-------------|
| entity_id | String | Which entity (pump, bearing, industry) |
| I | UInt32 | Observation index within entity |
| signal_id | String | Which signal (temp, pressure, return) |
| value | Float64 | The measurement |

**If data is not in this format, ORTHON transforms it first.**

---

## ORTHON Structure

```
~/orthon/
├── CLAUDE.md              ← This file
├── orthon/
│   ├── config/
│   │   ├── manifest.py    # Single source of truth (engines list)
│   │   ├── domains.py     # Physics domains (7 domains)
│   │   └── recommender.py
│   │
│   ├── ingest/
│   │   ├── paths.py           # Fixed output paths (NO EXCEPTIONS)
│   │   ├── streaming.py       # Universal streaming ingestor
│   │   └── manifest_generator.py
│   │
│   ├── intake/                # UI file handling
│   │   ├── upload.py
│   │   ├── validate.py
│   │   └── transformer.py
│   │
│   ├── analysis/
│   │   └── baseline_discovery.py  # Baseline modes
│   │
│   └── services/
│       └── manifest_builder.py
│
├── domains/               # Domain templates
├── data/                  # Benchmark data
└── fetchers/              # Data fetchers
```

---

## PRISM Structure

```
~/prism/
├── CLAUDE.md
├── data/
│   ├── observations.parquet   ← ORTHON creates
│   ├── manifest.yaml          ← ORTHON creates
│   └── *.parquet              ← PRISM writes outputs
│
└── prism/
    ├── runner.py              # Main: Geometry→Dynamics→Energy→SQL
    ├── python_runner.py       # Signal/pair/windowed engines
    ├── sql_runner.py          # SQL engines (DuckDB)
    ├── ram_manager.py         # Batch processing
    ├── cli.py
    │
    ├── engines/
    │   ├── signal/            # 27 signal-level engines
    │   ├── rolling/           # 16 rolling window engines
    │   ├── sql/               # 4 SQL engines
    │   ├── dynamics_runner.py
    │   ├── information_flow_runner.py
    │   └── topology_runner.py
    │
    └── primitives/            # 34 primitive functions
        ├── individual/        # 8 primitives
        ├── pairwise/          # 6 primitives
        ├── matrix/            # 5 primitives
        ├── information/       # 5 primitives
        ├── network/           # 4 primitives
        ├── dynamical/         # 3 primitives
        ├── topology/          # 2 primitives
        └── embedding/         # 1 primitive
```

---

## PRISM Engines (47 total)

### Signal Engines (27)
```
attractor, basin, cointegration, correlation, crest_factor,
cycle_counting, dmd, entropy, envelope, frequency_bands,
garch, granger, harmonics, hurst, kurtosis, lof, lyapunov,
mutual_info, peak, physics_stack, pulsation_index,
rate_of_change, rms, skewness, spectral, time_constant,
transfer_entropy
```

### Rolling Engines (16)
```
derivatives, manifold, stability,
rolling_crest_factor, rolling_entropy, rolling_envelope,
rolling_hurst, rolling_kurtosis, rolling_lyapunov,
rolling_mean, rolling_pulsation, rolling_range,
rolling_rms, rolling_skewness, rolling_std, rolling_volatility
```

### SQL Engines (4)
```
correlation, regime_assignment, statistics, zscore
```

---

## PRISM Outputs (12 parquet files)

### Geometry (structure)
- `primitives.parquet` - Signal-level metrics
- `primitives_pairs.parquet` - Directed pair metrics
- `geometry.parquet` - Symmetric pair metrics
- `topology.parquet` - Betti numbers, persistence
- `manifold.parquet` - Embedding metrics

### Dynamics (change)
- `dynamics.parquet` - Lyapunov, RQA, Hurst
- `information_flow.parquet` - Transfer entropy, Granger
- `observations_enriched.parquet` - Rolling window metrics

### Energy (physics)
- `physics.parquet` - Entropy, energy, free energy

### SQL Reconciliation
- `zscore.parquet` - Normalized metrics
- `statistics.parquet` - Summary statistics
- `correlation.parquet` - Correlation matrix
- `regime_assignment.parquet` - State labels

---

## Baseline Modes (orthon/analysis/baseline_discovery.py)

| Mode | Use Case |
|------|----------|
| first_n_percent | Industrial (pump, bearing) - known healthy start |
| stable_discovery | Markets, bioreactor - unknown healthy state |
| last_n_percent | Post-maintenance scenarios |
| reference_period | Known-good time window |
| rolling | Gradual drift systems |

---

## Domain Data Location

Raw domain data goes to: `/Users/jasonrudder/domains/`

```
domains/
├── bearing/           # FEMTO, IMS
├── turbomachinery/    # C-MAPSS
├── industrial/        # SKAB, MetroPT
├── finance/           # Fama-French
└── misc/              # Docs, scripts
```

---

## Commands

```bash
# ORTHON generates manifest
python -m orthon.ingest.manifest_generator /path/to/raw/data

# ORTHON ingests data
python -m orthon.ingest.streaming manifest.yaml

# PRISM computes (in prism repo)
cd ~/prism
./venv/bin/python -m prism data/manifest.yaml
```

---

## Rules

1. ALL engines run. Always. No exceptions.
2. Insufficient data → return NaN, never skip
3. No domain-specific logic in PRISM
4. No parallel/RAM management in ORTHON (PRISM handles this)
5. Output paths are FIXED - never change them
6. PRISM is HTTP only - never pip install

## Do NOT

- Skip engines based on domain
- Gate metrics by observation count
- Write to subdirectories of /Users/jasonrudder/prism/data/
- Add RAM management to ORTHON
- Make ORTHON compute anything
- pip install prism (it's HTTP only)
