# ORTHON

**Diagnostic Interpreter for PRISM Outputs**

*geometry leads — ørthon*

---

## What is ORTHON?

ORTHON reads parquet files produced by [PRISM](https://github.com/prism-engines/prism) and transforms raw metrics into actionable insights. It performs **zero calculations** — all math lives in PRISM.

```
PRISM (computes)  →  parquet files  →  ORTHON (interprets)
```

Think of it this way:
- **PRISM** is the scientific instrument (measures everything)
- **ORTHON** is the domain expert (interprets measurements)

---

## Installation

```bash
pip install orthon
```

This automatically installs PRISM as a dependency.

---

## Quick Start

```bash
# 1. Run PRISM to generate parquet files (do this once)
python -m prism.compute --input observations.parquet --output data/

# 2. Run ORTHON to interpret
python -m orthon.interpret --data data/ --domain turbofan

# 3. Launch the explorer
python -m orthon.serve --data data/
```

Open `http://localhost:8080` and explore your data.

---

## What ORTHON Does

| Task | Description |
|------|-------------|
| **Label** | Apply thresholds to metrics → human-readable labels |
| **Classify** | Map metric combinations → regime states |
| **Alert** | Detect threshold crossings → warnings |
| **Narrate** | Generate reports → thesis content, summaries |
| **Explore** | Interactive UI → SQL queries in browser |

---


## PRISM Parquet Files

ORTHON expects these 5 files from PRISM:

| File | Contents |
|------|----------|
| `data.parquet` | Raw observations + characterization |
| `vector.parquet` | Signal-level metrics (62 engines) |
| `geometry.parquet` | Pairwise relationships |
| `dynamics.parquet` | State transitions, regime detection |
| `physics.parquet` | Energy, momentum, equilibrium |

---

## Configuration

ORTHON's interpretation is driven entirely by config files:

```
orthon/config/
├── typology_rules.json      # metric thresholds → labels
├── regime_definitions.json  # state patterns → regime names
├── alert_thresholds.json    # when to warn
├── hd_slope_bands.json      # coherence velocity interpretation
│
└── domains/                 # Domain-specific overrides
    ├── turbofan.json
    ├── bearings.json
    ├── hydraulic.json
    ├── chemical.json
    └── research.json        # Discovery mode (inverted alerts)
```

### Example: Labeling Persistence

```json
// config/typology_rules.json
{
  "persistence": {
    "column": "hurst_exponent",
    "rules": {
      "trending":       { "op": ">",  "value": 0.6 },
      "mean_reverting": { "op": "<",  "value": 0.45 },
      "random":         { "op": "between", "low": 0.45, "high": 0.6 }
    }
  }
}
```

### Example: Domain Override

```json
// config/domains/turbofan.json
{
  "extends": "../typology_rules.json",
  "overrides": {
    "coherence_velocity": {
      "critical": { "op": "<", "value": -0.08 }
    }
  }
}
```

Change the JSON, change the interpretation. No code changes required.

---

## The Explorer

ORTHON includes a browser-based explorer built on DuckDB-WASM:

```bash
python -m orthon.serve --data data/ --domain turbofan
```

**Features:**
- Upload PRISM parquet files
- SQL queries run locally in your browser
- Your data never leaves your machine
- Tabs: Data Summary, Typology, Geometry, Dynamics, Physics, Advanced

**Or use the static version** (no server):
```bash
# Just open the HTML file
open orthon/explorer/index.html
```

---

## Dual Framing

ORTHON interprets the same math two ways:

### Industrial Mode (Default)
> "Systems lose coherence before they fail"

- Alerts warn of degradation
- hd_slope going negative = bad
- Goal: prevent disaster

### Research Mode
> "Signals lose coherence before breakthrough"

- Alerts flag discovery opportunities  
- hd_slope going negative = interesting
- Goal: capture discovery

```bash
# Industrial (default)
python -m orthon.interpret --data data/ --domain turbofan

# Research
python -m orthon.interpret --data data/ --domain research
```

Same parquet files. Different story.

---

## Per-Domain DuckDB

For optimized queries, ORTHON can build domain-specific DuckDB files:

```bash
python -m orthon.build_db --data data/ --domain turbofan --output turbofan.duckdb
```

This pre-joins PRISM parquet with ORTHON config for fast lookups:

```sql
-- Query the pre-built database
SELECT entity_id, hurst_exponent, persistence_label, alert_level
FROM interpreted_vector
WHERE alert_level = 'critical';
```

---

## Project Structure

```
orthon/
├── config/                  # JSON configuration files
│   ├── typology_rules.json
│   ├── regime_definitions.json
│   ├── alert_thresholds.json
│   └── domains/
│
├── interpreters/            # Apply config to parquet
│   ├── labels.py            # Metrics → labels
│   ├── regimes.py           # States → regimes
│   ├── alerts.py            # Thresholds → warnings
│   └── narrative.py         # Data → reports
│
├── explorer/                # Browser UI
│   ├── index.html
│   ├── app.js
│   └── styles.css
│
├── entry_points/
│   ├── interpret.py         # CLI interpreter
│   ├── serve.py             # Launch explorer
│   └── build_db.py          # Build domain DuckDB
│
└── output/                  # Generated files (gitignored)
    ├── interpreted.parquet
    └── reports/
```

---

## API Usage

```python
from orthon.interpreters import labels, regimes, alerts
from orthon.config import load_domain
import polars as pl

# Load PRISM output
vector = pl.read_parquet("data/vector.parquet")

# Load domain config
config = load_domain("turbofan")

# Apply interpretation
labeled = labels.apply(vector, config)
alerted = alerts.check(labeled, config)

# Generate report
from orthon.interpreters import narrative
report = narrative.generate(alerted, config, format="markdown")
```

---

## Thesis Mode

For the 2am grad student:

```bash
python -m orthon.thesis --data data/ --output thesis/
```

Generates:
- Publication-ready figures
- Statistical validation tables
- Domain-neutral methodology text
- LaTeX-compatible output

---

## License

- **Academic**: Free (citation required)
- **Enterprise**: Commercial license

---

## Links

- [PRISM](https://github.com/prism-engines/prism) — The calculation engine
- [Documentation](https://orthon.dev/docs)
- [Explorer Demo](https://orthon.dev/demo)

---

*ORTHON interprets. PRISM computes. Together: geometry leads.*
