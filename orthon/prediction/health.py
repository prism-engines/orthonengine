"""
Health Scoring Module.

Provides overall system health score (0-100) based on PRISM features.
Uses multiple indicators to compute a comprehensive health assessment.
"""

from pathlib import Path
from typing import Any, Optional

import numpy as np
import polars as pl

from .base import BasePredictor, PredictionResult


class HealthScorer(BasePredictor):
    """
    Computes overall health score (0-100) from PRISM features.

    Health is computed as inverse of degradation, with multiple
    indicators combined into a single score:

    - Statistical health: Deviation from baseline statistics
    - Dynamic health: Stability of system dynamics
    - Structural health: Topology and manifold integrity
    - Information health: Entropy and complexity metrics
    """

    required_outputs = ["primitives"]  # zscore is optional, enhances accuracy

    def __init__(
        self,
        prism_output_dir: str | Path,
        baseline_mode: str = "first_10_percent",
    ):
        """
        Initialize health scorer.

        Args:
            prism_output_dir: Directory containing PRISM outputs
            baseline_mode: How to determine healthy baseline
                - "first_10_percent": Use first 10% of data as baseline
                - "global_mean": Use dataset mean as baseline
                - "provided": Use externally provided baseline
        """
        self.baseline_mode = baseline_mode
        self._baseline_stats: dict[str, float] = {}
        super().__init__(prism_output_dir)
        self._compute_baseline()

    def _compute_baseline(self) -> None:
        """Compute baseline statistics for health comparison."""
        primitives = self.data.get("primitives", pl.DataFrame())

        if primitives.is_empty():
            return

        # Get numeric columns for baseline
        numeric_cols = [
            col for col in primitives.columns
            if primitives[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]
            and col not in ["unit_id", "entity_id", "I", "signal_id"]
        ]

        if self.baseline_mode == "first_10_percent":
            # Use first 10% of observations
            n_baseline = max(1, len(primitives) // 10)
            baseline_data = primitives.head(n_baseline)
        else:
            # Use all data
            baseline_data = primitives

        for col in numeric_cols:
            values = baseline_data[col].to_numpy()
            values = values[~np.isnan(values)]
            if len(values) > 0:
                self._baseline_stats[f"{col}_mean"] = float(np.mean(values))
                self._baseline_stats[f"{col}_std"] = float(np.std(values))

    def _compute_statistical_health(self, unit_id: Optional[str]) -> tuple[float, dict]:
        """Compute health from statistical features."""
        # Try zscore first, fall back to primitives
        zscore_path = self.output_dir / "zscore.parquet"
        if zscore_path.exists():
            zscore = pl.read_parquet(zscore_path)
        else:
            # Fall back to primitives and compute z-scores on the fly
            zscore = self.data.get("primitives", pl.DataFrame())

        if zscore.is_empty():
            return 100.0, {}

        if unit_id and "unit_id" in zscore.columns:
            zscore = zscore.filter(pl.col("unit_id") == unit_id)

        # Get all zscore columns
        zscore_cols = [col for col in zscore.columns if col.endswith("_zscore")]

        if not zscore_cols:
            # Fallback to numeric columns
            zscore_cols = [
                col for col in zscore.columns
                if zscore[col].dtype in [pl.Float64, pl.Float32]
                and col not in ["unit_id", "entity_id", "I", "signal_id"]
            ]

        contributions = {}
        deviations = []

        for col in zscore_cols:
            values = zscore[col].to_numpy()
            values = values[~np.isnan(values) & ~np.isinf(values)]
            if len(values) > 0:
                # Mean absolute zscore (deviation from baseline)
                mean_abs = float(np.mean(np.abs(values)))
                deviations.append(mean_abs)
                contributions[col] = mean_abs

        if not deviations:
            return 100.0, {}

        # Convert deviation to health (higher deviation = lower health)
        mean_deviation = np.mean(deviations)
        # Use sigmoid to map deviation to health score
        health = 100 * (1 / (1 + np.exp(mean_deviation - 2)))

        return float(health), contributions

    def _compute_dynamic_health(self, unit_id: Optional[str]) -> tuple[float, dict]:
        """Compute health from dynamic stability indicators."""
        # Try to load dynamics if available
        dynamics_path = self.output_dir / "dynamics.parquet"
        if not dynamics_path.exists():
            return 100.0, {}

        dynamics = pl.read_parquet(dynamics_path)

        if unit_id and "unit_id" in dynamics.columns:
            dynamics = dynamics.filter(pl.col("unit_id") == unit_id)

        contributions = {}
        scores = []

        # Lyapunov exponent: negative = stable, positive = chaotic
        if "lyapunov_exponent" in dynamics.columns:
            lyap = dynamics["lyapunov_exponent"].to_numpy()
            lyap = lyap[~np.isnan(lyap)]
            if len(lyap) > 0:
                mean_lyap = float(np.mean(lyap))
                # Negative = healthy (100), positive = unhealthy (0)
                lyap_health = 100 * (1 / (1 + np.exp(mean_lyap * 10)))
                scores.append(lyap_health)
                contributions["lyapunov"] = mean_lyap

        # Recurrence rate: higher = more predictable = healthier
        if "recurrence_rate" in dynamics.columns:
            rr = dynamics["recurrence_rate"].to_numpy()
            rr = rr[~np.isnan(rr)]
            if len(rr) > 0:
                mean_rr = float(np.mean(rr))
                rr_health = 100 * mean_rr  # Already 0-1
                scores.append(rr_health)
                contributions["recurrence_rate"] = mean_rr

        # Determinism: higher = more predictable = healthier
        if "determinism" in dynamics.columns:
            det = dynamics["determinism"].to_numpy()
            det = det[~np.isnan(det)]
            if len(det) > 0:
                mean_det = float(np.mean(det))
                det_health = 100 * mean_det  # Already 0-1
                scores.append(det_health)
                contributions["determinism"] = mean_det

        if not scores:
            return 100.0, {}

        return float(np.mean(scores)), contributions

    def _compute_structural_health(self, unit_id: Optional[str]) -> tuple[float, dict]:
        """Compute health from topological/structural features."""
        # Try to load topology if available
        topology_path = self.output_dir / "topology.parquet"
        if not topology_path.exists():
            return 100.0, {}

        topology = pl.read_parquet(topology_path)

        if unit_id and "unit_id" in topology.columns:
            topology = topology.filter(pl.col("unit_id") == unit_id)

        contributions = {}
        scores = []

        # Persistence entropy: higher = more complex = healthier
        for col in ["persistence_entropy_0", "persistence_entropy_1"]:
            if col in topology.columns:
                pe = topology[col].to_numpy()
                pe = pe[~np.isnan(pe)]
                if len(pe) > 0:
                    mean_pe = float(np.mean(pe))
                    # Scale to 0-100 (typical range 0-1)
                    pe_health = min(100, 100 * mean_pe)
                    scores.append(pe_health)
                    contributions[col] = mean_pe

        # Betti numbers stability
        if "betti_0" in topology.columns:
            b0 = topology["betti_0"].to_numpy()
            b0 = b0[~np.isnan(b0)]
            if len(b0) > 1:
                # Low variance = stable topology = healthy
                b0_var = float(np.var(b0))
                b0_health = 100 * (1 / (1 + b0_var))
                scores.append(b0_health)
                contributions["betti_0_stability"] = 1 / (1 + b0_var)

        if not scores:
            return 100.0, {}

        return float(np.mean(scores)), contributions

    def predict(self, unit_id: Optional[str] = None) -> PredictionResult:
        """
        Compute health score for unit(s).

        Args:
            unit_id: Specific unit to score, or None for all units.

        Returns:
            PredictionResult with health score(s) 0-100.
        """
        units = [unit_id] if unit_id else self.get_units()

        if not units:
            units = ["system"]

        predictions = {}
        all_contributions = {}
        warnings = []

        for unit in units:
            # Compute component health scores
            stat_health, stat_contrib = self._compute_statistical_health(unit)
            dyn_health, dyn_contrib = self._compute_dynamic_health(unit)
            struct_health, struct_contrib = self._compute_structural_health(unit)

            # Weighted combination
            weights = [0.4, 0.35, 0.25]  # statistical, dynamic, structural
            component_scores = [stat_health, dyn_health, struct_health]

            overall_health = sum(s * w for s, w in zip(component_scores, weights))
            predictions[unit] = overall_health

            # Combine contributions
            all_contributions[unit] = {
                "statistical": {"score": stat_health, **stat_contrib},
                "dynamic": {"score": dyn_health, **dyn_contrib},
                "structural": {"score": struct_health, **struct_contrib},
            }

            # Add warnings
            if overall_health < 30:
                warnings.append(f"CRITICAL: {unit} health < 30%")
            elif overall_health < 50:
                warnings.append(f"WARNING: {unit} health < 50%")
            elif overall_health < 70:
                warnings.append(f"CAUTION: {unit} health < 70%")

        # Confidence based on available data
        available_outputs = sum(1 for f in ["dynamics", "topology", "primitives", "zscore"]
                                if (self.output_dir / f"{f}.parquet").exists())
        confidence = min(1.0, available_outputs / 4)

        prediction = predictions if unit_id is None else predictions.get(unit_id, 50)

        return PredictionResult(
            prediction=prediction,
            confidence=confidence,
            model_name="HealthScorer",
            model_version="1.0.0",
            contributing_features=all_contributions.get(unit_id or units[0], {}),
            warnings=warnings,
            raw_scores={
                "health_scores": predictions,
                "component_scores": all_contributions,
            },
        )

    def explain(self, unit_id: str) -> dict[str, Any]:
        """
        Explain health score for a unit.

        Returns detailed breakdown of health indicators.
        """
        stat_health, stat_contrib = self._compute_statistical_health(unit_id)
        dyn_health, dyn_contrib = self._compute_dynamic_health(unit_id)
        struct_health, struct_contrib = self._compute_structural_health(unit_id)

        overall = 0.4 * stat_health + 0.35 * dyn_health + 0.25 * struct_health

        explanation = {
            "unit_id": unit_id,
            "overall_health": overall,
            "components": {
                "statistical_health": {
                    "score": stat_health,
                    "weight": 0.4,
                    "details": stat_contrib,
                    "description": "Deviation from baseline statistics",
                },
                "dynamic_health": {
                    "score": dyn_health,
                    "weight": 0.35,
                    "details": dyn_contrib,
                    "description": "Stability of system dynamics",
                },
                "structural_health": {
                    "score": struct_health,
                    "weight": 0.25,
                    "details": struct_contrib,
                    "description": "Topological and manifold integrity",
                },
            },
            "interpretation": [],
        }

        # Add interpretations
        if overall >= 80:
            explanation["interpretation"].append("System is in excellent health")
        elif overall >= 60:
            explanation["interpretation"].append("System is in good health with minor concerns")
        elif overall >= 40:
            explanation["interpretation"].append("System shows moderate degradation - monitor closely")
        elif overall >= 20:
            explanation["interpretation"].append("System shows significant degradation - action recommended")
        else:
            explanation["interpretation"].append("CRITICAL: System health severely degraded")

        # Component-specific interpretations
        if stat_health < 50:
            explanation["interpretation"].append(
                f"Statistical anomalies detected (deviation from baseline)"
            )
        if dyn_health < 50:
            explanation["interpretation"].append(
                f"Dynamic instability detected (chaotic behavior)"
            )
        if struct_health < 50:
            explanation["interpretation"].append(
                f"Structural degradation detected (topological changes)"
            )

        return explanation
