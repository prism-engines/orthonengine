"""ORTHON Analysis - Baseline discovery, cohort detection, and interpretation."""

from .baseline_discovery import (
    BaselineMode,
    discover_stable_baseline,
    get_baseline,
    BaselineResult,
)

from .cohort_detection import (
    # Thresholds
    CONSTANT_THRESHOLD,
    SYSTEM_THRESHOLD,
    COMPONENT_THRESHOLD,
    COUPLING_THRESHOLD,
    WITHIN_UNIT_THRESHOLD,
    # Enums
    CohortType,
    # Data classes
    SignalClassification,
    ConstantDetectionResult,
    CohortResult,
    # Main class (v2)
    CohortDiscovery,
    # Main entry point (v2)
    process_observations,
    # V1 compatibility functions
    should_run_cohort_discovery,
    detect_constants,
    detect_cohorts,
    classify_coupling_trajectory,
    generate_cohort_report,
)

__all__ = [
    # Baseline discovery
    "BaselineMode",
    "discover_stable_baseline",
    "get_baseline",
    "BaselineResult",
    # Cohort detection - thresholds
    "CONSTANT_THRESHOLD",
    "SYSTEM_THRESHOLD",
    "COMPONENT_THRESHOLD",
    "COUPLING_THRESHOLD",
    "WITHIN_UNIT_THRESHOLD",
    # Cohort detection - enums
    "CohortType",
    # Cohort detection - data classes
    "SignalClassification",
    "ConstantDetectionResult",
    "CohortResult",
    # Cohort detection - main class (v2)
    "CohortDiscovery",
    # Cohort detection - main entry point (v2)
    "process_observations",
    # Cohort detection - v1 compatibility
    "should_run_cohort_discovery",
    "detect_constants",
    "detect_cohorts",
    "classify_coupling_trajectory",
    "generate_cohort_report",
]
