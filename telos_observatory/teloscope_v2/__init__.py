"""
TELOSCOPE v2 - Production-Grade Control System

This package implements the full TELOSCOPE specification with:
- Centralized state management
- Advanced navigation controls
- Tool toggle system
- Position management (drag-and-drop)
- Enhanced timeline with marker generation
- Scroll synchronization utilities

Phase: Spec Implementation (Production Build)
Coexists with: teloscope/ (Phase 1 prototype)

See: docs/TELOSCOPE_INTEGRATION_RECONCILIATION.md for migration strategy

Current Status:
- Week 1-2 Foundation: ✅ Complete (6/6 components)
- Week 3-4 Core: ⏳ Pending (teloscope_controller scheduled)
"""

__version__ = "2.0.0-spec"

# TODO: Uncomment when teloscope_controller built in Week 3-4
# from .teloscope_controller import render_teloscope_v2, get_teloscope_status
#
# __all__ = [
#     'render_teloscope_v2',
#     'get_teloscope_status',
# ]

# Foundation components only - no controller yet
__all__ = []
