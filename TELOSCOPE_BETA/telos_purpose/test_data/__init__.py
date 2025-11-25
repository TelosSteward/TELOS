"""
Test Data Generation Module for TELOS Observatory
=================================================

Utilities for generating realistic test conversation sessions
for comprehensive platform testing.
"""

from .generate_test_sessions import (
    set_seed,
    generate_test_session,
    generate_test_suite,
    generate_session_by_type,
    generate_normal_session,
    generate_high_drift_session,
    generate_excellent_session,
    generate_long_session,
    generate_short_session,
    generate_critical_drift_session,
    generate_stable_session,
    generate_oscillating_session,
    export_sessions,
    SESSION_CONFIGS
)

from .edge_case_tests import (
    generate_all_edge_cases,
    export_edge_cases
)

__all__ = [
    # Reproducibility
    'set_seed',
    # Session generation
    'generate_test_session',
    'generate_test_suite',
    'generate_session_by_type',
    'generate_normal_session',
    'generate_high_drift_session',
    'generate_excellent_session',
    'generate_long_session',
    'generate_short_session',
    'generate_critical_drift_session',
    'generate_stable_session',
    'generate_oscillating_session',
    'export_sessions',
    'SESSION_CONFIGS',
    # Edge cases
    'generate_all_edge_cases',
    'export_edge_cases'
]
