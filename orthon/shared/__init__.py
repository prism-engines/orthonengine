"""ORTHON Shared — Schema definitions shared with PRISM."""

from .config_schema import (
    PrismConfig,
    SignalInfo,
    WindowConfig as WindowConfigModel,
    BaselineConfig,
    RegimeConfig,
    StateConfig,
    DISCIPLINES,
    DisciplineType,
    DOMAIN_TO_DISCIPLINE,
    DomainType,
)
from .window_config import (
    WindowConfig,
    auto_detect_window,
    validate_window,
    get_recommendation,
    format_errors_for_ui,
    format_config_summary,
    DOMAIN_DEFAULTS,
    COMPUTE_LIMITS,
)

__all__ = [
    # Config Schema
    'PrismConfig',
    'SignalInfo',
    'WindowConfigModel',
    'BaselineConfig',
    'RegimeConfig',
    'StateConfig',
    # Disciplines
    'DISCIPLINES',
    'DisciplineType',
    'DOMAIN_TO_DISCIPLINE',
    'DomainType',
    # Window Config (auto-detection)
    'WindowConfig',
    'auto_detect_window',
    'validate_window',
    'get_recommendation',
    'format_errors_for_ui',
    'format_config_summary',
    'DOMAIN_DEFAULTS',
    'COMPUTE_LIMITS',
]
