"""
ORTHON Data Concierge: LLM-Assisted Workflow
=============================================

Helps users at every stage:
1. Upload - Validate data, detect issues
2. Configure - Infer missing fields, suggest units
3. Analyze - Explain errors, suggest fixes
4. Interpret - Explain results in plain language
"""

import os
import json
import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


# Default model
DEFAULT_MODEL = "claude-sonnet-4-20250514"


# =============================================================================
# SYSTEM PROMPTS
# =============================================================================

SCHEMA_VALIDATION_PROMPT = """You are ORTHON's Data Concierge. Your job is to help users prepare their time series data for analysis.

CRITICAL: Units determine which physics engines run! You MUST identify units for each signal column.

## UNIT REFERENCE (use these exact strings)

### Vibration/Acceleration
g, m/s², mm/s, in/s, ips, mil, μm → category: vibration

### Temperature
°C, C, degC, °F, F, degF, K → category: temperature

### Pressure
Pa, kPa, MPa, bar, psi, PSI, psia, psig, atm → category: pressure

### Flow
m³/s, L/s, L/min, gpm, GPM, cfm, CFM → category: flow_volume
kg/s, kg/h, kg/hr, lb/hr → category: flow_mass

### Electrical
V, mV, kV → category: electrical_voltage
A, mA, μA → category: electrical_current
W, kW, MW → category: electrical_power

### Rotation
RPM, rpm, rad/s, Hz → category: rotation

### Force/Torque
N, kN, lbf → category: force
Nm, N·m → category: torque

### Other
%, percent → category: control
ppm, ppb, mg/L → category: concentration
m, mm, cm, ft, in → category: length
kg, g, lb → category: mass

## COMMON VALIDATION ERRORS TO FIX

1. "No entity column detected" → Suggest adding entity_id column or accept single entity
2. "No units detected" → MUST suggest units for each signal column
3. "High null rate (>10%)" → Suggest interpolation or removal
4. "Window too large" → Suggest reducing window_size

## YOUR TASKS

1. **Identify the index column** (time, cycles, sequence):
   - TIME: ISO 8601, Unix epoch, date strings, column names like timestamp/time/date
   - OTHER: Ask user about spatial unit, cycle duration, etc.

2. **Infer units** for EVERY numeric column:
   - From column names: "P1" → pressure (suggest PSI or bar)
   - From value ranges: 273-373 → likely Kelvin, 0-100 → likely °C or %
   - From context: industrial data often has PSI, degF, gpm

3. **Classify signals**: analog, digital, periodic, event

4. **Detect issues**: nulls, outliers, constant columns, sampling gaps

5. **Suggest fixes** for ALL problems found

## OUTPUT FORMAT

You MUST output a JSON block with suggested configuration:

```json
{
  "index_column": "timestamp",
  "index_dimension": "time",
  "index_format": "ISO 8601",
  "index_needs_user_input": false,
  "sampling": {
    "interval_seconds": 1.0,
    "unit": "seconds",
    "value": 1,
    "regularity": "regular"
  },
  "columns": {
    "P1": {"unit": "PSI", "signal_class": "analog", "quantity": "pressure"},
    "temp": {"unit": "degC", "signal_class": "analog", "quantity": "temperature"},
    "flow": {"unit": "gpm", "signal_class": "analog", "quantity": "flow"},
    "valve": {"unit": "state", "signal_class": "digital", "quantity": null}
  },
  "fixes": [
    {"column": "temp", "action": "add_unit", "suggested_unit": "degC", "reason": "Column name suggests temperature"},
    {"column": "P1", "action": "add_unit", "suggested_unit": "PSI", "reason": "Appears to be pressure signal"},
    {"column": "flow", "action": "interpolate", "method": "linear", "reason": "3% null values"}
  ]
}
```

Be helpful and specific. If unsure about a unit, make your best guess and explain why."""

ERROR_DIAGNOSIS_PROMPT = """You are ORTHON's Error Diagnostician. Your job is to explain analysis errors in plain language and suggest fixes.

When given an error, you will:
1. Explain what went wrong in simple terms (no jargon)
2. Identify the likely cause
3. Suggest specific fixes the user can apply
4. If relevant, explain what the expected data should look like

Be helpful and specific. Don't just say "check your data" - say exactly what to check.

Format your response as:
## What Went Wrong
[1-2 sentence explanation]

## Likely Cause
[Bullet points of possible causes]

## How to Fix
[Numbered steps to resolve]

## Prevention
[How to avoid this in future]"""

RESULTS_INTERPRETATION_PROMPT = """You are ORTHON's Results Interpreter. Your job is to explain analysis results to plant operators (not data scientists).

Given analysis results, you will:
1. Summarize the overall system health (good/concerning/critical)
2. Highlight key findings in plain language
3. Explain any detected patterns or anomalies
4. Identify causal relationships between signals
5. Recommend specific actions

CRITICAL RULES:
- Use plain language, no statistical jargon
- Focus on "what does this mean for operations"
- Be specific about times, values, and signals
- If something is concerning, explain WHY and WHAT TO DO
- Always end with clear action items

Format:
## System Health: [Good/Concerning/Critical]
[1-2 sentence summary]

## Key Findings
[Bullet points of important observations]

## Detected Patterns
[Any regime changes, trends, or anomalies]

## Causal Relationships
[What's driving what, with time delays]

## Recommended Actions
[Numbered list of specific things to do]"""

QA_PROMPT = """You are ORTHON's Analysis Assistant. You help users understand their analysis results by answering questions.

You have access to:
- Signal summaries (means, ranges, classifications)
- Regime information (when behavior changed)
- Causal relationships (what drives what)
- Alerts (anomalies and issues detected)

Answer questions based ONLY on the data provided. If you need more information, ask the user.
Be specific - reference actual signal names, times, and values.
If you don't know something, say so.

Keep answers concise but complete."""


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ColumnInfo:
    """Information about a data column."""
    name: str
    dtype: str
    sample_values: List[Any]
    null_pct: float = 0.0
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    mean_val: Optional[float] = None
    std_val: Optional[float] = None
    n_unique: Optional[int] = None


@dataclass
class SchemaInfo:
    """Schema information for LLM validation."""
    columns: List[ColumnInfo]
    n_rows: int
    n_cols: int
    filename: str = ""
    context: str = ""


@dataclass
class ValidationResult:
    """Result of schema validation."""
    report: str  # Markdown report
    config: Dict[str, Any]  # Suggested configuration
    issues: List[Dict[str, Any]]  # Detected issues
    confidence: float  # Overall confidence 0-1


@dataclass
class ErrorExplanation:
    """Explanation of an error."""
    summary: str
    cause: str
    fix_steps: List[str]
    prevention: str


# =============================================================================
# CONCIERGE CLASS
# =============================================================================

class DataConcierge:
    """
    LLM-powered assistant for data preparation and interpretation.

    Usage:
        concierge = DataConcierge()
        result = await concierge.validate_schema(schema_info)
        explanation = await concierge.explain_error(error, context)
        summary = await concierge.interpret_results(results)
    """

    def __init__(self, api_key: Optional[str] = None, model: str = DEFAULT_MODEL):
        """Initialize with Anthropic API key."""
        if not HAS_ANTHROPIC:
            raise ImportError(
                "anthropic package required. Install with: pip install anthropic"
            )

        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY required. Set environment variable or pass api_key."
            )

        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = model

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from markdown code blocks."""
        # Try to find ```json ... ``` block
        json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find any JSON object
        try:
            # Find first { and last }
            start = text.find('{')
            end = text.rfind('}')
            if start != -1 and end != -1:
                return json.loads(text[start:end+1])
        except json.JSONDecodeError:
            pass

        return None

    def validate_schema(self, schema: SchemaInfo) -> ValidationResult:
        """
        Validate a data schema and suggest configuration.

        Args:
            schema: SchemaInfo with column details

        Returns:
            ValidationResult with report, config, and issues
        """
        # Build schema description for LLM
        schema_desc = {
            "filename": schema.filename,
            "n_rows": schema.n_rows,
            "n_cols": schema.n_cols,
            "columns": []
        }

        for col in schema.columns:
            col_info = {
                "name": col.name,
                "dtype": col.dtype,
                "sample": col.sample_values[:5],
                "null_pct": col.null_pct,
            }
            if col.min_val is not None:
                col_info["min"] = col.min_val
                col_info["max"] = col.max_val
                col_info["mean"] = col.mean_val
            if col.n_unique is not None:
                col_info["n_unique"] = col.n_unique
            schema_desc["columns"].append(col_info)

        # Call Claude
        message = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            system=SCHEMA_VALIDATION_PROMPT,
            messages=[{
                "role": "user",
                "content": f"Please validate this data schema:\n\n```json\n{json.dumps(schema_desc, indent=2, default=str)}\n```"
            }]
        )

        report = message.content[0].text
        config = self._extract_json(report) or {}
        issues = config.get("fixes", [])

        # Estimate confidence based on response
        confidence = 0.8  # Default
        if "not sure" in report.lower() or "unclear" in report.lower():
            confidence = 0.5
        if "confident" in report.lower() or "clearly" in report.lower():
            confidence = 0.9

        return ValidationResult(
            report=report,
            config=config,
            issues=issues,
            confidence=confidence
        )

    def explain_error(
        self,
        error_message: str,
        error_context: str = "",
        column_name: str = "",
        sample_values: List[Any] = None
    ) -> ErrorExplanation:
        """
        Explain an error in plain language.

        Args:
            error_message: The error message
            error_context: Additional context (SQL query, stage, etc.)
            column_name: Column that caused the error (if known)
            sample_values: Sample data from the problematic column

        Returns:
            ErrorExplanation with summary, cause, and fix steps
        """
        context_parts = [f"Error: {error_message}"]
        if error_context:
            context_parts.append(f"Context: {error_context}")
        if column_name:
            context_parts.append(f"Column: {column_name}")
        if sample_values:
            context_parts.append(f"Sample values: {sample_values[:10]}")

        message = self.client.messages.create(
            model=self.model,
            max_tokens=1000,
            system=ERROR_DIAGNOSIS_PROMPT,
            messages=[{
                "role": "user",
                "content": "\n".join(context_parts)
            }]
        )

        response = message.content[0].text

        # Parse sections from response
        summary = ""
        cause = ""
        fix_steps = []
        prevention = ""

        current_section = None
        for line in response.split('\n'):
            if "What Went Wrong" in line:
                current_section = "summary"
            elif "Likely Cause" in line:
                current_section = "cause"
            elif "How to Fix" in line:
                current_section = "fix"
            elif "Prevention" in line:
                current_section = "prevention"
            elif line.strip():
                if current_section == "summary":
                    summary += line + " "
                elif current_section == "cause":
                    cause += line + " "
                elif current_section == "fix":
                    if line.strip().startswith(('1', '2', '3', '4', '5', '-', '*')):
                        fix_steps.append(line.strip().lstrip('0123456789.-* '))
                elif current_section == "prevention":
                    prevention += line + " "

        return ErrorExplanation(
            summary=summary.strip() or response[:200],
            cause=cause.strip() or "Unknown",
            fix_steps=fix_steps or ["Review the error details above"],
            prevention=prevention.strip() or "Validate data before analysis"
        )

    def interpret_results(
        self,
        system_summary: Dict[str, Any],
        signal_summaries: List[Dict[str, Any]] = None,
        causal_info: List[Dict[str, Any]] = None,
        alerts: List[Dict[str, Any]] = None
    ) -> str:
        """
        Interpret analysis results in plain language.

        Args:
            system_summary: Overall system statistics
            signal_summaries: Per-signal summaries
            causal_info: Causal relationships detected
            alerts: Alerts and anomalies

        Returns:
            Markdown-formatted interpretation
        """
        context = {
            "system_summary": system_summary,
        }
        if signal_summaries:
            context["signals"] = signal_summaries[:20]  # Limit for token efficiency
        if causal_info:
            context["causal_relationships"] = causal_info[:10]
        if alerts:
            context["alerts"] = alerts[:20]

        message = self.client.messages.create(
            model=self.model,
            max_tokens=1500,
            system=RESULTS_INTERPRETATION_PROMPT,
            messages=[{
                "role": "user",
                "content": f"Please interpret these analysis results:\n\n```json\n{json.dumps(context, indent=2, default=str)}\n```"
            }]
        )

        return message.content[0].text

    def answer_question(
        self,
        question: str,
        signal_summaries: List[Dict[str, Any]] = None,
        regime_info: List[Dict[str, Any]] = None,
        causal_info: List[Dict[str, Any]] = None,
        alerts: List[Dict[str, Any]] = None,
        conversation_history: List[Dict[str, str]] = None
    ) -> str:
        """
        Answer a question about the analysis results.

        Args:
            question: User's question
            signal_summaries: Available signal data
            regime_info: Regime change information
            causal_info: Causal relationships
            alerts: Detected alerts
            conversation_history: Previous Q&A turns

        Returns:
            Answer text
        """
        # Build context
        context_parts = []
        if signal_summaries:
            context_parts.append(f"Signal Summaries:\n{json.dumps(signal_summaries[:15], indent=2, default=str)}")
        if regime_info:
            context_parts.append(f"Regime Info:\n{json.dumps(regime_info[:10], indent=2, default=str)}")
        if causal_info:
            context_parts.append(f"Causal Relationships:\n{json.dumps(causal_info[:10], indent=2, default=str)}")
        if alerts:
            context_parts.append(f"Alerts:\n{json.dumps(alerts[:15], indent=2, default=str)}")

        system_prompt = QA_PROMPT + "\n\nAvailable Data:\n" + "\n\n".join(context_parts)

        # Build messages with history
        messages = []
        if conversation_history:
            for turn in conversation_history[-6:]:  # Keep last 3 exchanges
                messages.append(turn)
        messages.append({"role": "user", "content": question})

        response = self.client.messages.create(
            model=self.model,
            max_tokens=800,
            system=system_prompt,
            messages=messages
        )

        return response.content[0].text


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def schema_from_dataframe(df, filename: str = "") -> SchemaInfo:
    """
    Create SchemaInfo from a polars or pandas DataFrame.

    Args:
        df: DataFrame (polars or pandas)
        filename: Original filename

    Returns:
        SchemaInfo object
    """
    import polars as pl

    # Convert pandas to polars if needed
    if hasattr(df, 'to_pandas'):
        # It's polars
        pass
    elif hasattr(df, 'columns'):
        # It's pandas
        df = pl.from_pandas(df)

    columns = []
    for col_name in df.columns:
        col = df[col_name]
        dtype = str(col.dtype)

        # Get sample values
        samples = col.head(5).to_list()

        # Get stats for numeric columns
        col_info = ColumnInfo(
            name=col_name,
            dtype=dtype,
            sample_values=samples,
            null_pct=col.null_count() / len(col) if len(col) > 0 else 0,
            n_unique=col.n_unique()
        )

        if col.dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8):
            try:
                col_info.min_val = float(col.min())
                col_info.max_val = float(col.max())
                col_info.mean_val = float(col.mean())
                col_info.std_val = float(col.std()) if col.std() is not None else None
            except:
                pass

        columns.append(col_info)

    return SchemaInfo(
        columns=columns,
        n_rows=len(df),
        n_cols=len(df.columns),
        filename=filename
    )


def get_concierge() -> Optional[DataConcierge]:
    """
    Get a DataConcierge instance if API key is available.

    Returns:
        DataConcierge or None if not configured
    """
    try:
        return DataConcierge()
    except (ImportError, ValueError):
        return None


def concierge_available() -> bool:
    """Check if concierge is available (API key configured)."""
    return bool(os.environ.get("ANTHROPIC_API_KEY")) and HAS_ANTHROPIC
