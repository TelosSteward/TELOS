"""
Observation Deck - TELOS Observatory Research Panel

The Observation Deck transforms TELOS Observatory from a chat interface into a complete
research platform with collapsible analysis tools.

Components:
- Observatory Control Strip: Top-right thermometer showing turn, fidelity, calibration
- Observation Deck Control Strip: Sidebar header with telescope toggle and symbolic flow
- TELOSCOPIC Tools (FREE): Comparison Viewer, Calculation Window, Turn Navigator
- Steward Integration (PAID): Conversational Q&A about sessions (~$0.002/query)
- Calibration Logger: Mistral reasoning visualization (Turns 1-3)
- Symbolic Flow: Governance pipeline animator (👤→⚡→🔄→🤖→✓)

Architecture:
- All backend telemetry already exists (WebSessionManager, telemetry_utils.py)
- This is a UI wiring challenge, not new mathematical components
- Turn marker synchronization enables playback across all tools
"""

from .deck_manager import DeckManager
from .observatory_control_strip import ObservatoryControlStrip
from .deck_control_strip import DeckControlStrip

__all__ = [
    'DeckManager',
    'ObservatoryControlStrip',
    'DeckControlStrip',
]

__version__ = '1.0.0'
