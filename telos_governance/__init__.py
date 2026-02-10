"""
TELOS Governance -- Conversational Governance Gates

Two-tier governance for AI conversations:
- Tier 1: Fidelity Gate (purpose alignment check)
- Tier 2: Tool Selection Gate (semantic tool ranking)

Uses "Detect and Direct" pattern:
- SPC DETECTS drift via fidelity measurement
- System DIRECTS response back toward primacy attractor
"""

from telos_governance.types import (
    ActionDecision,
    DirectionLevel,
    GovernanceDecision,
    GovernanceResult,
    GovernanceTrace,
)

__version__ = "1.0.0"
