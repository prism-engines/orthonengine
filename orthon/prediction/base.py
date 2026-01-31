"""
Base classes for ORTHON Prediction Module.

All predictors read PRISM outputs and generate predictions.
ORTHON does no computation - it interprets PRISM's computed features.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import polars as pl


@dataclass
class PredictionResult:
    """Standard result format for all predictors."""

    # Core prediction
    prediction: float | dict[str, float]  # Single value or per-unit values
    confidence: float  # 0-1 confidence in prediction

    # Context
    timestamp: datetime = field(default_factory=datetime.now)
    model_name: str = ""
    model_version: str = "1.0.0"

    # Explanation
    contributing_features: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    # Raw data for downstream
    raw_scores: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "prediction": self.prediction,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
            "model_name": self.model_name,
            "model_version": self.model_version,
            "contributing_features": self.contributing_features,
            "warnings": self.warnings,
            "raw_scores": self.raw_scores,
        }


class BasePredictor(ABC):
    """
    Base class for all ORTHON predictors.

    Predictors read PRISM outputs (parquet files) and generate predictions.
    They do NOT compute features - PRISM does that.
    """

    # Required PRISM outputs for this predictor
    required_outputs: list[str] = []

    def __init__(self, prism_output_dir: str | Path):
        """
        Initialize predictor with PRISM output directory.

        Args:
            prism_output_dir: Directory containing PRISM output parquets
        """
        self.output_dir = Path(prism_output_dir)
        self._validate_outputs()
        self._load_data()

    def _validate_outputs(self) -> None:
        """Verify all required PRISM outputs exist."""
        missing = []
        for output in self.required_outputs:
            path = self.output_dir / f"{output}.parquet"
            if not path.exists():
                missing.append(output)

        if missing:
            raise FileNotFoundError(
                f"Missing required PRISM outputs: {missing}. "
                f"Run PRISM first: python -m prism {self.output_dir}/manifest.yaml"
            )

    def _load_data(self) -> None:
        """Load required PRISM outputs into memory."""
        self.data: dict[str, pl.DataFrame] = {}
        for output in self.required_outputs:
            path = self.output_dir / f"{output}.parquet"
            self.data[output] = pl.read_parquet(path)

    def get_units(self) -> list[str]:
        """Get list of unique units/entities in the data."""
        # Try common column names
        for col in ["unit_id", "entity_id", "unit"]:
            for df in self.data.values():
                if col in df.columns:
                    return df[col].unique().sort().to_list()
        return []

    def get_signals(self) -> list[str]:
        """Get list of unique signals in the data."""
        for col in ["signal_id", "signal"]:
            for df in self.data.values():
                if col in df.columns:
                    return df[col].unique().sort().to_list()
        return []

    @abstractmethod
    def predict(self, unit_id: Optional[str] = None) -> PredictionResult:
        """
        Generate prediction.

        Args:
            unit_id: Optional specific unit to predict for.
                    If None, predict for all units.

        Returns:
            PredictionResult with prediction and metadata.
        """
        pass

    @abstractmethod
    def explain(self, unit_id: str) -> dict[str, Any]:
        """
        Explain prediction for a specific unit.

        Args:
            unit_id: Unit to explain prediction for.

        Returns:
            Dictionary with feature contributions and explanation.
        """
        pass


class EnsemblePredictor(BasePredictor):
    """
    Combines multiple predictors for more robust predictions.

    Uses weighted averaging or voting to combine predictions.
    """

    def __init__(
        self,
        predictors: list[BasePredictor],
        weights: Optional[list[float]] = None
    ):
        """
        Initialize ensemble with multiple predictors.

        Args:
            predictors: List of predictor instances
            weights: Optional weights for each predictor (default: equal)
        """
        self.predictors = predictors
        self.weights = weights or [1.0 / len(predictors)] * len(predictors)

        if len(self.weights) != len(self.predictors):
            raise ValueError("Number of weights must match number of predictors")

        # Normalize weights
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]

    def _validate_outputs(self) -> None:
        """Ensemble doesn't load directly - predictors handle this."""
        pass

    def _load_data(self) -> None:
        """Ensemble doesn't load directly - predictors handle this."""
        self.data = {}

    def predict(self, unit_id: Optional[str] = None) -> PredictionResult:
        """Combine predictions from all predictors."""
        results = [p.predict(unit_id) for p in self.predictors]

        # Weighted average of predictions
        if isinstance(results[0].prediction, dict):
            # Per-unit predictions
            combined = {}
            for unit in results[0].prediction.keys():
                combined[unit] = sum(
                    r.prediction.get(unit, 0) * w
                    for r, w in zip(results, self.weights)
                )
            prediction = combined
        else:
            # Single prediction
            prediction = sum(
                r.prediction * w for r, w in zip(results, self.weights)
            )

        # Weighted average confidence
        confidence = sum(
            r.confidence * w for r, w in zip(results, self.weights)
        )

        # Combine contributing features
        all_features = {}
        for r, w in zip(results, self.weights):
            for feat, val in r.contributing_features.items():
                if feat not in all_features:
                    all_features[feat] = 0
                all_features[feat] += val * w

        # Collect all warnings
        all_warnings = []
        for r in results:
            all_warnings.extend(r.warnings)

        return PredictionResult(
            prediction=prediction,
            confidence=confidence,
            model_name="Ensemble",
            contributing_features=all_features,
            warnings=list(set(all_warnings)),
            raw_scores={"individual_results": [r.to_dict() for r in results]},
        )

    def explain(self, unit_id: str) -> dict[str, Any]:
        """Combine explanations from all predictors."""
        explanations = {}
        for i, predictor in enumerate(self.predictors):
            explanations[f"predictor_{i}_{type(predictor).__name__}"] = {
                "weight": self.weights[i],
                "explanation": predictor.explain(unit_id),
            }
        return explanations
