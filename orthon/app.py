"""
Orthon — Drop Data, Get Physics

MVP Streamlit app:
1. Instructions to prepare data
2. Upload file
3. Report on what was uploaded
4. Download results back
"""

import streamlit as st
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="Orthon", page_icon="⚡", layout="wide")

# =============================================================================
# UNIT DETECTION
# =============================================================================

SUFFIX_TO_UNIT = {
    '_psi': ('psi', 'pressure'), '_bar': ('bar', 'pressure'), '_kpa': ('kPa', 'pressure'),
    '_f': ('°F', 'temperature'), '_c': ('°C', 'temperature'), '_k': ('K', 'temperature'),
    '_degf': ('°F', 'temperature'), '_degc': ('°C', 'temperature'), '_degr': ('°R', 'temperature'),
    '_gpm': ('gpm', 'flow'), '_lpm': ('L/min', 'flow'),
    '_mps': ('m/s', 'velocity'), '_fps': ('ft/s', 'velocity'),
    '_m': ('m', 'length'), '_mm': ('mm', 'length'), '_in': ('in', 'length'), '_ft': ('ft', 'length'),
    '_kg': ('kg', 'mass'), '_lb': ('lb', 'mass'),
    '_rpm': ('rpm', 'frequency'), '_hz': ('Hz', 'frequency'),
    '_v': ('V', 'voltage'), '_a': ('A', 'current'), '_w': ('W', 'power'), '_kw': ('kW', 'power'),
    '_pct': ('%', 'ratio'),
}

ENTITY_COLS = ['entity_id', 'unit_id', 'equipment_id', 'asset_id', 'machine_id', 'engine_id', 'pump_id', 'id', 'unit']
TIME_COLS = ['timestamp', 'time', 'datetime', 'date', 'cycle', 't']


def analyze(df):
    """Analyze uploaded data"""
    report = {
        'rows': len(df),
        'cols': len(df.columns),
        'columns': [],
        'signals': [],
        'constants': [],
        'entity_col': None,
        'time_col': None,
        'entities': [],
        'issues': [],
        'warnings': [],
    }

    for col in df.columns:
        name_lower = col.lower()

        # Get unit
        unit, category = None, None
        for suffix, (u, c) in SUFFIX_TO_UNIT.items():
            if name_lower.endswith(suffix):
                unit, category = u, c
                break

        info = {'name': col, 'unit': unit, 'category': category, 'dtype': str(df[col].dtype)}

        if pd.api.types.is_numeric_dtype(df[col]):
            info['min'] = float(df[col].min())
            info['max'] = float(df[col].max())
            info['mean'] = float(df[col].mean())
            info['nulls'] = int(df[col].isna().sum())
            info['unique'] = int(df[col].nunique())

        report['columns'].append(info)

        # Classify
        if name_lower in ENTITY_COLS:
            report['entity_col'] = col
            report['entities'] = df[col].unique().tolist()
        elif name_lower in TIME_COLS:
            report['time_col'] = col
        elif pd.api.types.is_numeric_dtype(df[col]):
            if df[col].nunique() == 1:
                report['constants'].append(col)
            else:
                report['signals'].append(col)

    # Sanity checks
    for info in report['columns']:
        col = info['name']
        if info.get('nulls', 0) == report['rows']:
            report['issues'].append(f"`{col}` is entirely null")
        elif info.get('nulls', 0) > 0:
            report['warnings'].append(f"`{col}` has {info['nulls']} nulls ({info['nulls']/report['rows']*100:.1f}%)")

        if info.get('category') == 'temperature':
            if info.get('min', 0) < -273:
                report['issues'].append(f"`{col}` has temp below absolute zero")

        if info.get('category') == 'pressure':
            if info.get('min', 0) < 0:
                report['warnings'].append(f"`{col}` has negative pressure")

    return report


# =============================================================================
# UI
# =============================================================================

st.title("⚡ Orthon")
st.caption("Drop data. Get physics.")

tab1, tab2, tab3 = st.tabs(["📖 Instructions", "📤 Upload", "📊 Results"])

# -----------------------------------------------------------------------------
# INSTRUCTIONS
# -----------------------------------------------------------------------------

with tab1:
    st.header("How to Prepare Your Data")

    st.markdown("""
### Quick Start

**Name your columns with units. We figure out the rest.**

```csv
timestamp, flow_gpm, pressure_psi, temp_F
2024-01-01 08:00, 50, 120, 150
2024-01-01 08:01, 51, 121, 151
```

---

### Unit Suffixes

| Measurement | Suffixes |
|-------------|----------|
| Pressure | `_psi`, `_bar`, `_kpa` |
| Temperature | `_F`, `_C`, `_K`, `_degF`, `_degR` |
| Flow | `_gpm`, `_lpm` |
| Length | `_in`, `_ft`, `_m`, `_mm` |
| Speed | `_rpm`, `_hz` |
| Electrical | `_V`, `_A`, `_W`, `_kW` |

---

### Multiple Equipment

Add an `entity_id` column:

```csv
entity_id, diameter_in, flow_gpm, pressure_psi
P-101, 4, 50, 120
P-101, 4, 51, 121
P-102, 6, 100, 115
```

We detect:
- `entity_id` → grouping column
- `diameter_in` → constant per entity
- `flow_gpm`, `pressure_psi` → signals

---

### Supported Formats

- CSV ✅
- Parquet ✅
- TSV ✅
""")

# -----------------------------------------------------------------------------
# UPLOAD
# -----------------------------------------------------------------------------

with tab2:
    st.header("Upload Your Data")

    uploaded = st.file_uploader("CSV, Parquet, or TSV", type=['csv', 'parquet', 'tsv', 'txt'])

    if uploaded:
        try:
            if uploaded.name.endswith('.parquet'):
                df = pd.read_parquet(uploaded)
            elif uploaded.name.endswith('.tsv'):
                df = pd.read_csv(uploaded, sep='\t')
            else:
                df = pd.read_csv(uploaded, comment='#')

            st.session_state['df'] = df
            st.session_state['filename'] = uploaded.name

            st.success(f"✅ `{uploaded.name}` — {len(df):,} rows × {len(df.columns)} columns")
            st.dataframe(df.head(10), use_container_width=True)

            if st.button("🔍 Analyze", type="primary"):
                st.session_state['report'] = analyze(df)
                st.success("Done! See Results tab →")

        except Exception as e:
            st.error(f"Error: {e}")

# -----------------------------------------------------------------------------
# RESULTS
# -----------------------------------------------------------------------------

with tab3:
    st.header("Results")

    if 'report' not in st.session_state:
        st.info("Upload a file and click Analyze first.")
    else:
        r = st.session_state['report']
        df = st.session_state['df']

        # Metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{r['rows']:,}")
        c2.metric("Columns", r['cols'])
        c3.metric("Signals", len(r['signals']))
        c4.metric("Entities", len(r['entities']) or 1)

        # Issues
        if r['issues']:
            st.error("**Issues**")
            for i in r['issues']:
                st.write(f"❌ {i}")

        if r['warnings']:
            st.warning("**Warnings**")
            for w in r['warnings']:
                st.write(f"⚠️ {w}")

        if not r['issues'] and not r['warnings']:
            st.success("✅ Data looks good!")

        # Structure
        st.subheader("Structure")
        st.write(f"**Time:** `{r['time_col'] or 'not detected'}`")
        st.write(f"**Entity:** `{r['entity_col'] or 'single entity'}`")
        if r['entities']:
            st.write(f"**Found:** {', '.join(str(e) for e in r['entities'][:10])}" +
                    (f" (+{len(r['entities'])-10} more)" if len(r['entities']) > 10 else ""))

        # Signals table
        st.subheader("Signals")
        signal_rows = []
        for info in r['columns']:
            if info['name'] in r['signals']:
                signal_rows.append({
                    'Column': info['name'],
                    'Unit': info['unit'] or '?',
                    'Min': f"{info.get('min', 0):.4g}",
                    'Max': f"{info.get('max', 0):.4g}",
                    'Mean': f"{info.get('mean', 0):.4g}",
                })
        if signal_rows:
            st.dataframe(pd.DataFrame(signal_rows), hide_index=True, use_container_width=True)

        # Downloads
        st.subheader("Download")

        report_json = {
            'file': st.session_state.get('filename'),
            'analyzed': datetime.now().isoformat(),
            'rows': r['rows'],
            'signals': r['signals'],
            'constants': r['constants'],
            'entities': r['entities'],
            'time_col': r['time_col'],
            'entity_col': r['entity_col'],
            'columns': r['columns'],
            'issues': r['issues'],
            'warnings': r['warnings'],
        }

        c1, c2 = st.columns(2)
        c1.download_button(
            "📄 Report (JSON)",
            data=pd.io.json.dumps(report_json, indent=2),
            file_name=f"orthon_report_{datetime.now():%Y%m%d_%H%M%S}.json",
            mime="application/json"
        )
        c2.download_button(
            "📊 Signals (CSV)",
            data=pd.DataFrame(signal_rows).to_csv(index=False) if signal_rows else "",
            file_name=f"orthon_signals_{datetime.now():%Y%m%d_%H%M%S}.csv",
            mime="text/csv"
        )

# Footer
st.divider()
st.caption("Orthon — *Systems lose coherence before they fail*")
