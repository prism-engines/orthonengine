"""
Capability Detector
===================

Determine what can be computed given the inspected file.
"""

import re
from dataclasses import dataclass, asdict
from typing import List, Dict, Set, Optional
from .file_inspector import FileInspection


@dataclass
class Capabilities:
    """What can be computed given this data"""

    # Capability level
    level: int  # 0-4
    level_name: str

    # Available stages
    can_vector: bool
    can_geometry: bool
    can_dynamics: bool

    # Available engines
    available_engines: List[str]
    unavailable_engines: Dict[str, str]  # engine -> reason

    # What's missing for more capability
    missing_for_next_level: List[str]

    # Summary
    summary: str

    def to_dict(self):
        return asdict(self)


# Engine requirements
ENGINE_REQUIREMENTS = {
    # Core (always available with any signal)
    'statistical': {},
    'trend': {},
    'stationarity': {},
    'entropy': {},
    'hurst': {},
    'spectral': {},

    # Transport / Fluid
    'reynolds': {
        'constants_any': ['density', 'rho'],
        'constants_any2': ['viscosity', 'mu', 'dynamic_viscosity'],
        'constants_any3': ['diameter', 'd', 'pipe_diameter'],
        'signals_any': ['flow', 'velocity'],
    },
    'pressure_drop': {
        'constants_any': ['density', 'rho'],
        'constants_any2': ['viscosity', 'mu'],
        'constants_any3': ['diameter', 'd'],
        'constants_any4': ['length', 'pipe_length'],
        'signals_any': ['flow', 'velocity'],
    },

    # Reaction
    'conversion': {
        'signals_all': ['inlet', 'outlet'],
    },
    'arrhenius': {
        'signals_any': ['temperature', 'temp'],
        'requires_multiple_entities': True,
    },
    'residence_time': {
        'constants_any': ['reactor_volume', 'volume'],
        'constants_any2': ['flow_rate', 'feed_flow'],
    },

    # Thermodynamics
    'gibbs': {
        'signals_all': ['enthalpy', 'entropy', 'temperature'],
    },
    'heat_capacity': {
        'signals_any': ['temperature', 'temp'],
        'signals_any2': ['heat', 'q'],
    },

    # Electrochemistry
    'nernst': {
        'constants_any': ['standard_potential', 'e0'],
        'signals_any': ['concentration', 'conc'],
    },
    'faraday': {
        'constants_any': ['molecular_weight', 'mw'],
        'signals_any': ['current'],
    },
}

# Signal name patterns for matching
SIGNAL_PATTERNS = {
    'flow': [r'flow', r'_gpm', r'_lpm'],
    'velocity': [r'velocity', r'speed'],
    'temperature': [r'temp', r'_K', r'_C', r'_F', r'_degR'],
    'pressure': [r'pressure', r'_psi', r'_bar', r'_kpa'],
    'concentration': [r'conc', r'_mol'],
    'current': [r'current', r'_A'],
    'voltage': [r'voltage', r'potential', r'_V'],
    'inlet': [r'inlet'],
    'outlet': [r'outlet'],
    'enthalpy': [r'enthalpy', r'_h'],
    'entropy': [r'entropy', r'_s'],
    'heat': [r'heat', r'_q'],
}


def has_constant(available: Set[str], names: List[str]) -> bool:
    """Check if any of the constant names are available."""
    for name in names:
        if name.lower() in available:
            return True
    return False


def has_signal(available: Set[str], names: List[str]) -> bool:
    """Check if any signal matching the patterns is available."""
    for name in names:
        patterns = SIGNAL_PATTERNS.get(name, [name])
        for pattern in patterns:
            for sig in available:
                if re.search(pattern, sig, re.IGNORECASE):
                    return True
    return False


def check_engine_requirements(
    engine: str,
    reqs: dict,
    available_constants: Set[str],
    available_signals: Set[str],
    n_entities: int,
) -> tuple:
    """Check if an engine's requirements are met."""

    # No requirements = always available
    if not reqs:
        return True, ""

    # Check constant requirements (constants_any, constants_any2, etc.)
    for key, names in reqs.items():
        if key.startswith('constants_any'):
            if not has_constant(available_constants, names):
                return False, f"Missing: {' or '.join(names)}"

    # Check signal requirements
    for key, names in reqs.items():
        if key == 'signals_any' or key.startswith('signals_any'):
            if not has_signal(available_signals, names):
                return False, f"Missing signal: {' or '.join(names)}"
        elif key == 'signals_all':
            for name in names:
                if not has_signal(available_signals, [name]):
                    return False, f"Missing signal: {name}"

    # Check special requirements
    if reqs.get('requires_multiple_entities'):
        if n_entities < 2:
            return False, "Requires multiple entities"

    return True, ""


def detect_capabilities(
    inspection: FileInspection,
    discipline: Optional[str] = None,
) -> Capabilities:
    """
    Determine what can be computed given the inspected file.

    Args:
        inspection: Result from inspect_file()
        discipline: Selected discipline (optional)

    Returns:
        Capabilities object
    """
    # Gather available constants
    available_constants = set()
    for const_name in inspection.constants.keys():
        available_constants.add(const_name.lower())

    # Also check for constant columns
    for sig in inspection.signals:
        if sig.is_constant and not sig.is_entity_id and not sig.is_sequence:
            available_constants.add(sig.name.lower())

    # Gather available signals (non-constant columns)
    available_signals = set()
    for sig in inspection.signals:
        if not sig.is_constant and not sig.is_entity_id and not sig.is_sequence:
            available_signals.add(sig.name.lower())

    n_entities = len(inspection.entities)
    n_signals = len(available_signals)

    # Check each engine
    available_engines = []
    unavailable_engines = {}

    for engine, reqs in ENGINE_REQUIREMENTS.items():
        can_run, reason = check_engine_requirements(
            engine, reqs, available_constants, available_signals, n_entities
        )
        if can_run:
            available_engines.append(engine)
        else:
            unavailable_engines[engine] = reason

    # Determine capability level
    has_units = any(s.unit for s in inspection.signals)
    has_constants = len(inspection.constants) > 0 or len(available_constants) > 0

    if n_signals >= 3 and has_constants:
        level = 3
        level_name = "Full Physics"
    elif n_signals >= 2:
        level = 2
        level_name = "Geometry"
    elif has_units:
        level = 1
        level_name = "Units"
    else:
        level = 0
        level_name = "Basic"

    # What stages are available
    can_vector = n_signals >= 1
    can_geometry = n_signals >= 2
    can_dynamics = n_signals >= 2 and inspection.row_count >= 50

    # What's missing for next level
    missing = []
    if level < 1 and not has_units:
        missing.append("Add unit suffixes (e.g., flow_gpm, temp_K)")
    if level < 2 and n_signals < 2:
        missing.append("Add more signal columns")
    if level < 3 and not has_constants:
        missing.append("Add constants in header (# density = 1020)")

    # Summary
    summary = f"Level {level}: {level_name} | {n_signals} signals, {n_entities} entities, {len(available_engines)} engines"

    return Capabilities(
        level=level,
        level_name=level_name,
        can_vector=can_vector,
        can_geometry=can_geometry,
        can_dynamics=can_dynamics,
        available_engines=available_engines,
        unavailable_engines=unavailable_engines,
        missing_for_next_level=missing,
        summary=summary,
    )
