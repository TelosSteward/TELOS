"""
TELOSCOPE v2 State Management

Centralized state management for TELOSCOPE control system.
Provides structured namespace for all TELOSCOPE-related state variables.
"""

__version__ = "2.0.0-spec"

from .teloscope_state import (
    init_teloscope_state,
    get_teloscope_state,
    update_teloscope_state,
    reset_teloscope_state,
)

__all__ = [
    'init_teloscope_state',
    'get_teloscope_state',
    'update_teloscope_state',
    'reset_teloscope_state',
]
