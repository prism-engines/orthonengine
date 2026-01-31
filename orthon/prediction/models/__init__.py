"""
Pre-trained model registry for ORTHON Prediction Module.

This module provides access to pre-trained models for specific domains:
- Bearing degradation models
- Turbomachinery RUL models
- General-purpose anomaly detection

Models are loaded from the pretrained/ directory.
"""

from pathlib import Path
from typing import Optional

# Model registry
PRETRAINED_DIR = Path(__file__).parent.parent / "pretrained"

# Available pre-trained models (to be populated)
AVAILABLE_MODELS = {
    # "bearing_rul_v1": {
    #     "type": "rul",
    #     "domain": "bearing",
    #     "description": "FEMTO-trained RUL predictor for bearings",
    #     "path": "bearing_rul_v1.pkl",
    # },
}


def list_models() -> dict:
    """List all available pre-trained models."""
    return AVAILABLE_MODELS.copy()


def get_model_path(model_name: str) -> Optional[Path]:
    """
    Get path to a pre-trained model.

    Args:
        model_name: Name of the model

    Returns:
        Path to model file, or None if not found
    """
    if model_name not in AVAILABLE_MODELS:
        return None

    model_path = PRETRAINED_DIR / AVAILABLE_MODELS[model_name]["path"]
    if model_path.exists():
        return model_path
    return None


def load_model(model_name: str):
    """
    Load a pre-trained model.

    Args:
        model_name: Name of the model to load

    Returns:
        Loaded model object

    Raises:
        ValueError: If model not found
    """
    model_path = get_model_path(model_name)
    if model_path is None:
        raise ValueError(f"Model '{model_name}' not found. Available: {list(AVAILABLE_MODELS.keys())}")

    # Load based on model type
    import pickle
    with open(model_path, "rb") as f:
        return pickle.load(f)


__all__ = [
    "AVAILABLE_MODELS",
    "list_models",
    "get_model_path",
    "load_model",
]
