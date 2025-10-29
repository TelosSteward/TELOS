"""
TELOSCOPIC Tools - Visual Research Instruments (FREE)

These tools provide mathematical transparency and session navigation without API costs.
All data comes from existing backend telemetry - no new generation required.

Tools:
- Comparison Viewer: TELOS vs Baseline split view with intervention highlights
- Calculation Window: Display embedding distance, fidelity, interventions
- Turn Navigator: Timeline scrubber with playback controls for session replay

Data Sources:
- CounterfactualBranchManager: TELOS vs Baseline branches
- telemetry_utils.py: Fidelity scores, embedding distances, interventions
- WebSessionManager: Turn-level session data with metadata
"""

from .comparison_viewer import ComparisonViewer
from .calculation_window import CalculationWindow
from .turn_navigator import TurnNavigator

__all__ = [
    'ComparisonViewer',
    'CalculationWindow',
    'TurnNavigator',
]
