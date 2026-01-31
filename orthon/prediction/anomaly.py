"""
Anomaly Detection Module.

Detects anomalies using PRISM-computed features.
Supports multiple detection methods: isolation forest, LOF, z-score.
"""

from enum import Enum
from pathlib import Path
from typing import Any, Optional

import numpy as np
import polars as pl

from .base import BasePredictor, PredictionResult


class AnomalyMethod(Enum):
    """Available anomaly detection methods."""

    ZSCORE = "zscore"
    ISOLATION_FOREST = "isolation_forest"
    LOF = "lof"  # Local Outlier Factor
    COMBINED = "combined"


class AnomalyDetector(BasePredictor):
    """
    Detects anomalies in PRISM features.

    Methods:
    - zscore: Simple statistical threshold (no sklearn dependency)
    - isolation_forest: Tree-based isolation (requires sklearn)
    - lof: Local outlier factor (requires sklearn)
    - combined: Ensemble of all available methods
    """

    required_outputs = ["primitives"]  # zscore is optional

    def __init__(
        self,
        prism_output_dir: str | Path,
        method: str | AnomalyMethod = AnomalyMethod.ZSCORE,
        threshold: float = 3.0,
        contamination: float = 0.1,
    ):
        """
        Initialize anomaly detector.

        Args:
            prism_output_dir: Directory containing PRISM outputs
            method: Detection method to use
            threshold: Z-score threshold for zscore method
            contamination: Expected proportion of outliers (for IF/LOF)
        """
        if isinstance(method, str):
            method = AnomalyMethod(method)
        self.method = method
        self.threshold = threshold
        self.contamination = contamination

        # ML models (lazy loaded)
        self._isolation_forest = None
        self._lof = None

        super().__init__(prism_output_dir)

    def _get_feature_matrix(self, unit_id: Optional[str] = None) -> tuple[np.ndarray, list[str]]:
        """
        Extract feature matrix from PRISM outputs.

        Returns:
            Tuple of (feature_matrix, feature_names)
        """
        # Prefer zscore data if available (already normalized)
        zscore_path = self.output_dir / "zscore.parquet"
        if zscore_path.exists():
            df = pl.read_parquet(zscore_path)
        else:
            df = self.data.get("primitives", pl.DataFrame())

        if df.is_empty():
            return np.array([[]]), []

        # Filter by unit if specified
        if unit_id and "unit_id" in df.columns:
            df = df.filter(pl.col("unit_id") == unit_id)

        # Get numeric feature columns
        exclude_cols = {"unit_id", "entity_id", "I", "signal_id", "timestamp"}
        feature_cols = [
            col for col in df.columns
            if col not in exclude_cols
            and df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]
        ]

        if not feature_cols:
            return np.array([[]]), []

        # Extract matrix
        matrix = df.select(feature_cols).to_numpy()

        # Handle NaN/Inf
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)

        return matrix, feature_cols

    def _detect_zscore(self, unit_id: Optional[str] = None) -> tuple[np.ndarray, dict]:
        """
        Detect anomalies using z-score threshold.

        Returns:
            Tuple of (anomaly_labels, details)
        """
        matrix, feature_cols = self._get_feature_matrix(unit_id)

        if matrix.size == 0:
            return np.array([]), {}

        # Compute z-scores
        mean = np.mean(matrix, axis=0)
        std = np.std(matrix, axis=0) + 1e-10
        zscores = np.abs((matrix - mean) / std)

        # Max z-score across features for each observation
        max_zscores = np.max(zscores, axis=1)

        # Label anomalies
        anomalies = (max_zscores > self.threshold).astype(int)

        # Find which features contributed most to anomalies
        anomaly_idx = np.where(anomalies == 1)[0]
        contributing_features = {}
        if len(anomaly_idx) > 0:
            for idx in anomaly_idx:
                max_feat_idx = np.argmax(zscores[idx])
                feat_name = feature_cols[max_feat_idx]
                if feat_name not in contributing_features:
                    contributing_features[feat_name] = 0
                contributing_features[feat_name] += 1

        details = {
            "method": "zscore",
            "threshold": self.threshold,
            "max_zscores": max_zscores.tolist(),
            "contributing_features": contributing_features,
            "n_anomalies": int(np.sum(anomalies)),
            "n_total": len(anomalies),
        }

        return anomalies, details

    def _detect_isolation_forest(self, unit_id: Optional[str] = None) -> tuple[np.ndarray, dict]:
        """
        Detect anomalies using Isolation Forest.

        Returns:
            Tuple of (anomaly_labels, details)
        """
        try:
            from sklearn.ensemble import IsolationForest
        except ImportError:
            # Fall back to zscore if sklearn not available
            return self._detect_zscore(unit_id)

        matrix, feature_cols = self._get_feature_matrix(unit_id)

        if matrix.size == 0 or matrix.shape[0] < 10:
            return np.array([]), {"error": "Insufficient data for Isolation Forest"}

        # Fit model
        if self._isolation_forest is None:
            self._isolation_forest = IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_estimators=100,
            )

        # Predict (-1 = anomaly, 1 = normal)
        predictions = self._isolation_forest.fit_predict(matrix)
        anomalies = (predictions == -1).astype(int)

        # Get anomaly scores
        scores = -self._isolation_forest.score_samples(matrix)

        details = {
            "method": "isolation_forest",
            "contamination": self.contamination,
            "anomaly_scores": scores.tolist(),
            "n_anomalies": int(np.sum(anomalies)),
            "n_total": len(anomalies),
        }

        return anomalies, details

    def _detect_lof(self, unit_id: Optional[str] = None) -> tuple[np.ndarray, dict]:
        """
        Detect anomalies using Local Outlier Factor.

        Returns:
            Tuple of (anomaly_labels, details)
        """
        try:
            from sklearn.neighbors import LocalOutlierFactor
        except ImportError:
            return self._detect_zscore(unit_id)

        matrix, feature_cols = self._get_feature_matrix(unit_id)

        if matrix.size == 0 or matrix.shape[0] < 20:
            return np.array([]), {"error": "Insufficient data for LOF"}

        # Fit and predict
        n_neighbors = min(20, matrix.shape[0] - 1)
        lof = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=self.contamination,
        )

        predictions = lof.fit_predict(matrix)
        anomalies = (predictions == -1).astype(int)

        # Get LOF scores
        scores = -lof.negative_outlier_factor_

        details = {
            "method": "lof",
            "n_neighbors": n_neighbors,
            "contamination": self.contamination,
            "lof_scores": scores.tolist(),
            "n_anomalies": int(np.sum(anomalies)),
            "n_total": len(anomalies),
        }

        return anomalies, details

    def _detect_combined(self, unit_id: Optional[str] = None) -> tuple[np.ndarray, dict]:
        """
        Detect anomalies using ensemble of methods.

        An observation is anomalous if flagged by majority of methods.
        """
        results = []
        all_details = {}

        # Z-score (always available)
        zscore_anomalies, zscore_details = self._detect_zscore(unit_id)
        if len(zscore_anomalies) > 0:
            results.append(zscore_anomalies)
            all_details["zscore"] = zscore_details

        # Isolation Forest (if sklearn available)
        if_anomalies, if_details = self._detect_isolation_forest(unit_id)
        if len(if_anomalies) > 0 and "error" not in if_details:
            results.append(if_anomalies)
            all_details["isolation_forest"] = if_details

        # LOF (if sklearn available)
        lof_anomalies, lof_details = self._detect_lof(unit_id)
        if len(lof_anomalies) > 0 and "error" not in lof_details:
            results.append(lof_anomalies)
            all_details["lof"] = lof_details

        if not results:
            return np.array([]), {"error": "No detection methods produced results"}

        # Stack and vote
        stacked = np.stack(results, axis=0)
        votes = np.sum(stacked, axis=0)

        # Majority voting
        threshold = len(results) / 2
        anomalies = (votes > threshold).astype(int)

        details = {
            "method": "combined",
            "methods_used": list(all_details.keys()),
            "votes": votes.tolist(),
            "vote_threshold": threshold,
            "method_details": all_details,
            "n_anomalies": int(np.sum(anomalies)),
            "n_total": len(anomalies),
        }

        return anomalies, details

    def predict(self, unit_id: Optional[str] = None) -> PredictionResult:
        """
        Detect anomalies for unit(s).

        Args:
            unit_id: Specific unit to analyze, or None for all units.

        Returns:
            PredictionResult with anomaly indicators.
        """
        # Select detection method
        if self.method == AnomalyMethod.ZSCORE:
            anomalies, details = self._detect_zscore(unit_id)
        elif self.method == AnomalyMethod.ISOLATION_FOREST:
            anomalies, details = self._detect_isolation_forest(unit_id)
        elif self.method == AnomalyMethod.LOF:
            anomalies, details = self._detect_lof(unit_id)
        else:  # COMBINED
            anomalies, details = self._detect_combined(unit_id)

        if len(anomalies) == 0:
            return PredictionResult(
                prediction=0.0,
                confidence=0.0,
                model_name=f"AnomalyDetector_{self.method.value}",
                warnings=["No data available for anomaly detection"],
            )

        # Compute anomaly rate
        anomaly_rate = float(np.mean(anomalies))

        # Warnings
        warnings = []
        n_anomalies = int(np.sum(anomalies))
        if anomaly_rate > 0.2:
            warnings.append(f"HIGH anomaly rate: {n_anomalies}/{len(anomalies)} ({anomaly_rate:.1%})")
        elif n_anomalies > 0:
            warnings.append(f"Detected {n_anomalies} anomalies ({anomaly_rate:.1%})")

        # Confidence based on sample size and method
        confidence = min(1.0, len(anomalies) / 100)
        if self.method == AnomalyMethod.COMBINED:
            confidence *= 1.2  # Boost for ensemble

        return PredictionResult(
            prediction=anomaly_rate,
            confidence=min(1.0, confidence),
            model_name=f"AnomalyDetector_{self.method.value}",
            model_version="1.0.0",
            contributing_features=details.get("contributing_features", {}),
            warnings=warnings,
            raw_scores={
                "anomaly_labels": anomalies.tolist(),
                "detection_details": details,
            },
        )

    def explain(self, unit_id: str) -> dict[str, Any]:
        """
        Explain anomaly detection for a unit.

        Returns details about detected anomalies and contributing features.
        """
        # Run detection
        if self.method == AnomalyMethod.ZSCORE:
            anomalies, details = self._detect_zscore(unit_id)
        elif self.method == AnomalyMethod.ISOLATION_FOREST:
            anomalies, details = self._detect_isolation_forest(unit_id)
        elif self.method == AnomalyMethod.LOF:
            anomalies, details = self._detect_lof(unit_id)
        else:
            anomalies, details = self._detect_combined(unit_id)

        # Find anomaly indices
        anomaly_indices = np.where(anomalies == 1)[0].tolist() if len(anomalies) > 0 else []

        explanation = {
            "unit_id": unit_id,
            "method": self.method.value,
            "n_anomalies": len(anomaly_indices),
            "n_observations": len(anomalies),
            "anomaly_rate": float(np.mean(anomalies)) if len(anomalies) > 0 else 0.0,
            "anomaly_indices": anomaly_indices[:100],  # Limit to first 100
            "detection_details": details,
            "interpretation": [],
        }

        # Add interpretations
        if len(anomaly_indices) == 0:
            explanation["interpretation"].append("No anomalies detected - system operating normally")
        elif len(anomaly_indices) <= 5:
            explanation["interpretation"].append(
                f"Isolated anomalies detected at indices: {anomaly_indices}"
            )
        elif len(anomaly_indices) / len(anomalies) < 0.1:
            explanation["interpretation"].append(
                "Sporadic anomalies detected - may indicate occasional disturbances"
            )
        else:
            explanation["interpretation"].append(
                "Frequent anomalies detected - systematic issue likely"
            )

        # Feature-specific insights
        if "contributing_features" in details:
            top_features = sorted(
                details["contributing_features"].items(),
                key=lambda x: x[1],
                reverse=True,
            )[:5]
            if top_features:
                explanation["interpretation"].append(
                    f"Top anomaly contributors: {[f[0] for f in top_features]}"
                )

        return explanation

    def get_anomaly_indices(self, unit_id: Optional[str] = None) -> list[int]:
        """
        Get indices of detected anomalies.

        Convenience method for downstream processing.
        """
        result = self.predict(unit_id)
        labels = result.raw_scores.get("anomaly_labels", [])
        return [i for i, label in enumerate(labels) if label == 1]
