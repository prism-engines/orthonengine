"""
Remaining Useful Life (RUL) Predictor.

Predicts time-to-failure using PRISM-computed degradation features.
Uses dynamics, topology, and trend features to estimate RUL.
"""

from pathlib import Path
from typing import Any, Optional

import numpy as np
import polars as pl

from .base import BasePredictor, PredictionResult


class RULPredictor(BasePredictor):
    """
    Predicts Remaining Useful Life from PRISM features.

    Primary features used:
    - dynamics.parquet: Lyapunov exponent trends, entropy changes
    - primitives.parquet: Statistical degradation indicators
    - topology.parquet: Persistence entropy evolution

    The predictor uses feature trends and thresholds to estimate
    cycles/time remaining until failure.
    """

    required_outputs = ["primitives"]  # dynamics and topology enhance accuracy but are optional

    def __init__(
        self,
        prism_output_dir: str | Path,
        failure_threshold: float = 0.8,
        min_history: int = 10,
    ):
        """
        Initialize RUL predictor.

        Args:
            prism_output_dir: Directory containing PRISM outputs
            failure_threshold: Threshold for degradation score (0-1) indicating failure
            min_history: Minimum observations needed for trend estimation
        """
        self.failure_threshold = failure_threshold
        self.min_history = min_history
        super().__init__(prism_output_dir)

    def _compute_degradation_score(self, unit_id: str) -> tuple[float, dict[str, float]]:
        """
        Compute degradation score for a unit.

        Returns:
            Tuple of (degradation_score, feature_contributions)
        """
        contributions = {}
        scores = []

        # 1. Lyapunov exponent (chaos indicator)
        # Load dynamics if available
        dynamics_path = self.output_dir / "dynamics.parquet"
        if dynamics_path.exists():
            self.data["dynamics"] = pl.read_parquet(dynamics_path)
        if "dynamics" in self.data:
            dynamics = self.data["dynamics"]
            if "unit_id" in dynamics.columns:
                unit_dynamics = dynamics.filter(pl.col("unit_id") == unit_id)
            else:
                unit_dynamics = dynamics

            if "lyapunov_exponent" in unit_dynamics.columns:
                lyap = unit_dynamics["lyapunov_exponent"].to_numpy()
                lyap = lyap[~np.isnan(lyap)]
                if len(lyap) > 0:
                    # Positive Lyapunov = chaos = degradation
                    lyap_score = np.clip(np.mean(lyap) / 0.1, 0, 1)
                    scores.append(lyap_score * 0.3)  # 30% weight
                    contributions["lyapunov"] = lyap_score

        # 2. Entropy increase (disorder indicator)
        if "primitives" in self.data:
            primitives = self.data["primitives"]
            if "unit_id" in primitives.columns:
                unit_prim = primitives.filter(pl.col("unit_id") == unit_id)
            else:
                unit_prim = primitives

            if "entropy_sample" in unit_prim.columns:
                entropy = unit_prim["entropy_sample"].to_numpy()
                entropy = entropy[~np.isnan(entropy)]
                if len(entropy) > self.min_history:
                    # Trend in entropy
                    trend = np.polyfit(range(len(entropy)), entropy, 1)[0]
                    entropy_score = np.clip(trend * 10, 0, 1)
                    scores.append(entropy_score * 0.25)  # 25% weight
                    contributions["entropy_trend"] = entropy_score

            # 3. Kurtosis (impulsiveness indicator)
            if "kurtosis" in unit_prim.columns:
                kurtosis = unit_prim["kurtosis"].to_numpy()
                kurtosis = kurtosis[~np.isnan(kurtosis)]
                if len(kurtosis) > 0:
                    # High kurtosis = impulsive = damaged
                    kurt_score = np.clip((np.mean(kurtosis) - 3) / 10, 0, 1)
                    scores.append(kurt_score * 0.2)  # 20% weight
                    contributions["kurtosis"] = kurt_score

            # 4. RMS trend (amplitude growth)
            if "rms" in unit_prim.columns:
                rms = unit_prim["rms"].to_numpy()
                rms = rms[~np.isnan(rms)]
                if len(rms) > self.min_history:
                    # Normalize and compute trend
                    rms_norm = (rms - rms[0]) / (np.std(rms) + 1e-10)
                    trend = np.polyfit(range(len(rms_norm)), rms_norm, 1)[0]
                    rms_score = np.clip(trend, 0, 1)
                    scores.append(rms_score * 0.15)  # 15% weight
                    contributions["rms_trend"] = rms_score

        # 5. Topological changes (structural degradation)
        # Load topology if available
        topology_path = self.output_dir / "topology.parquet"
        if topology_path.exists() and "topology" not in self.data:
            self.data["topology"] = pl.read_parquet(topology_path)
        if "topology" in self.data:
            topology = self.data["topology"]
            if "unit_id" in topology.columns:
                unit_topo = topology.filter(pl.col("unit_id") == unit_id)
            else:
                unit_topo = topology

            if "persistence_entropy_0" in unit_topo.columns:
                pers_ent = unit_topo["persistence_entropy_0"].to_numpy()
                pers_ent = pers_ent[~np.isnan(pers_ent)]
                if len(pers_ent) > 0:
                    # Lower persistence entropy = simpler structure = degradation
                    topo_score = np.clip(1 - np.mean(pers_ent), 0, 1)
                    scores.append(topo_score * 0.1)  # 10% weight
                    contributions["topology"] = topo_score

        # Combine scores
        if not scores:
            return 0.0, {}

        degradation = sum(scores) / sum([0.3, 0.25, 0.2, 0.15, 0.1][:len(scores)])
        return float(np.clip(degradation, 0, 1)), contributions

    def _estimate_rul(self, degradation: float, degradation_rate: float) -> float:
        """
        Estimate remaining useful life from degradation score and rate.

        Returns:
            Estimated RUL in observation cycles
        """
        if degradation >= self.failure_threshold:
            return 0.0

        if degradation_rate <= 0:
            # No degradation trend - return large value
            return 1000.0

        remaining = (self.failure_threshold - degradation) / degradation_rate
        return float(max(0, remaining))

    def predict(self, unit_id: Optional[str] = None) -> PredictionResult:
        """
        Predict RUL for unit(s).

        Args:
            unit_id: Specific unit to predict for, or None for all units.

        Returns:
            PredictionResult with RUL prediction(s).
        """
        units = [unit_id] if unit_id else self.get_units()

        if not units:
            # No unit structure - treat as single system
            units = ["system"]

        predictions = {}
        all_contributions = {}
        warnings = []

        for unit in units:
            # Compute current degradation
            degradation, contributions = self._compute_degradation_score(unit)
            all_contributions[unit] = contributions

            # Estimate degradation rate from recent history
            # (simplified - in production would use sliding window)
            degradation_rate = degradation / 100 if degradation > 0 else 0.001

            # Estimate RUL
            rul = self._estimate_rul(degradation, degradation_rate)
            predictions[unit] = rul

            # Add warnings
            if degradation > 0.9:
                warnings.append(f"CRITICAL: {unit} degradation > 90%")
            elif degradation > 0.7:
                warnings.append(f"WARNING: {unit} degradation > 70%")

        # Compute confidence based on data quality
        confidence = min(1.0, len(self.data) / len(self.required_outputs))

        # Return single value if single unit requested
        prediction = predictions if unit_id is None else predictions.get(unit_id, 0)

        return PredictionResult(
            prediction=prediction,
            confidence=confidence,
            model_name="RULPredictor",
            model_version="1.0.0",
            contributing_features=all_contributions.get(unit_id or units[0], {}),
            warnings=warnings,
            raw_scores={
                "degradation_scores": {
                    u: self._compute_degradation_score(u)[0] for u in units
                },
                "feature_contributions": all_contributions,
            },
        )

    def explain(self, unit_id: str) -> dict[str, Any]:
        """
        Explain RUL prediction for a unit.

        Returns detailed breakdown of degradation indicators.
        """
        degradation, contributions = self._compute_degradation_score(unit_id)

        explanation = {
            "unit_id": unit_id,
            "degradation_score": degradation,
            "failure_threshold": self.failure_threshold,
            "feature_contributions": contributions,
            "interpretation": [],
        }

        # Add interpretations
        if "lyapunov" in contributions:
            if contributions["lyapunov"] > 0.5:
                explanation["interpretation"].append(
                    "High chaotic behavior detected - system dynamics unstable"
                )

        if "entropy_trend" in contributions:
            if contributions["entropy_trend"] > 0.5:
                explanation["interpretation"].append(
                    "Increasing entropy trend - system disorder growing"
                )

        if "kurtosis" in contributions:
            if contributions["kurtosis"] > 0.5:
                explanation["interpretation"].append(
                    "High kurtosis - impulsive events detected (possible damage)"
                )

        if "rms_trend" in contributions:
            if contributions["rms_trend"] > 0.5:
                explanation["interpretation"].append(
                    "Increasing RMS amplitude - vibration energy growing"
                )

        if "topology" in contributions:
            if contributions["topology"] > 0.5:
                explanation["interpretation"].append(
                    "Simplified signal topology - loss of healthy complexity"
                )

        if not explanation["interpretation"]:
            explanation["interpretation"].append("System appears healthy")

        return explanation
