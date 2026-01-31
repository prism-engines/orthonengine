"""
ORTHON: The Interface Layer for PRISM

Drop data. Get physics.

ORTHON does zero calculations. All math happens in PRISM.
"""

__version__ = "0.1.0"

# Core functionality
from orthon.intake import load_file, validate, detect_columns, detect_units
from orthon.intake import prepare_for_prism, transform_for_prism, IntakeTransformer
from orthon.shared import PrismConfig, SignalInfo, DISCIPLINES
from orthon.backend import get_backend, analyze, has_prism, get_backend_info

# Prediction module - import directly from orthon.prediction when needed
# from orthon.prediction import predict_rul, score_health, detect_anomalies

# Display module deprecated - using static HTML at /static/index.html

__all__ = [
    # Version
    '__version__',
    # Intake
    'load_file',
    'validate',
    'detect_columns',
    'detect_units',
    # Transformer
    'prepare_for_prism',
    'transform_for_prism',
    'IntakeTransformer',
    # Config Schema (shared with PRISM)
    'PrismConfig',
    'SignalInfo',
    'DISCIPLINES',
    # Backend
    'get_backend',
    'analyze',
    'has_prism',
    'get_backend_info',
    # Prediction - import from orthon.prediction directly
    # 'predict_rul', 'score_health', 'detect_anomalies',
]
