"""
ORTHON Prediction Module.

Provides prediction capabilities built on PRISM-computed features:
- RUL (Remaining Useful Life) prediction
- Health scoring (0-100)
- Anomaly detection

All predictors read PRISM outputs and interpret them.
ORTHON does NO computation - PRISM does all the math.

Usage:
    from orthon.prediction import predict_rul, score_health, detect_anomalies

    # Simple API
    rul = predict_rul("/path/to/prism/output")
    health = score_health("/path/to/prism/output")
    anomalies = detect_anomalies("/path/to/prism/output")

    # Advanced API
    from orthon.prediction import RULPredictor, HealthScorer, AnomalyDetector

    predictor = RULPredictor("/path/to/prism/output")
    result = predictor.predict(unit_id="unit_1")
    explanation = predictor.explain("unit_1")
"""

from pathlib import Path
from typing import Optional, Union

from .base import BasePredictor, EnsemblePredictor, PredictionResult
from .rul import RULPredictor
from .health import HealthScorer
from .anomaly import AnomalyDetector, AnomalyMethod


# Simple API functions
def predict_rul(
    prism_output_dir: Union[str, Path],
    unit_id: Optional[str] = None,
    failure_threshold: float = 0.8,
) -> PredictionResult:
    """
    Predict Remaining Useful Life from PRISM outputs.

    Args:
        prism_output_dir: Directory containing PRISM output parquets
        unit_id: Specific unit to predict for (None for all units)
        failure_threshold: Degradation threshold indicating failure

    Returns:
        PredictionResult with RUL prediction(s)

    Example:
        >>> result = predict_rul("/path/to/prism/output")
        >>> print(f"RUL: {result.prediction} cycles")
        >>> print(f"Confidence: {result.confidence:.0%}")
    """
    predictor = RULPredictor(
        prism_output_dir,
        failure_threshold=failure_threshold,
    )
    return predictor.predict(unit_id)


def score_health(
    prism_output_dir: Union[str, Path],
    unit_id: Optional[str] = None,
    baseline_mode: str = "first_10_percent",
) -> PredictionResult:
    """
    Compute health score (0-100) from PRISM outputs.

    Args:
        prism_output_dir: Directory containing PRISM output parquets
        unit_id: Specific unit to score (None for all units)
        baseline_mode: How to determine healthy baseline

    Returns:
        PredictionResult with health score(s)

    Example:
        >>> result = score_health("/path/to/prism/output")
        >>> print(f"Health: {result.prediction:.0f}%")
    """
    scorer = HealthScorer(
        prism_output_dir,
        baseline_mode=baseline_mode,
    )
    return scorer.predict(unit_id)


def detect_anomalies(
    prism_output_dir: Union[str, Path],
    unit_id: Optional[str] = None,
    method: str = "zscore",
    threshold: float = 3.0,
) -> PredictionResult:
    """
    Detect anomalies in PRISM features.

    Args:
        prism_output_dir: Directory containing PRISM output parquets
        unit_id: Specific unit to analyze (None for all)
        method: Detection method ("zscore", "isolation_forest", "lof", "combined")
        threshold: Z-score threshold for zscore method

    Returns:
        PredictionResult with anomaly rate and details

    Example:
        >>> result = detect_anomalies("/path/to/prism/output")
        >>> print(f"Anomaly rate: {result.prediction:.1%}")
        >>> indices = result.raw_scores["anomaly_labels"]
    """
    detector = AnomalyDetector(
        prism_output_dir,
        method=method,
        threshold=threshold,
    )
    return detector.predict(unit_id)


# Export all public symbols
__all__ = [
    # Base classes
    "BasePredictor",
    "EnsemblePredictor",
    "PredictionResult",
    # Predictors
    "RULPredictor",
    "HealthScorer",
    "AnomalyDetector",
    "AnomalyMethod",
    # Simple API
    "predict_rul",
    "score_health",
    "detect_anomalies",
]
