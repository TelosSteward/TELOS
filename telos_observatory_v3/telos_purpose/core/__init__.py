"""
TELOS Core Module
=================

Mathematical and runtime orchestration components.

Exports:
- UnifiedGovernanceSteward: Main runtime steward (Mitigation Bridge Layer)
- TeleologicalOperator: Alias for UnifiedGovernanceSteward
- PrimacyAttractorMath: Attractor dynamics and basin calculations
- MathematicalState: State representation in embedding space
- TelicFidelityCalculator: Fidelity metric computation
"""

from .unified_steward import UnifiedGovernanceSteward, TeleologicalOperator
from .primacy_math import (
    MathematicalState,
    PrimacyAttractorMath,
    TelicFidelityCalculator
)
from .embedding_provider import EmbeddingProvider
from .conversation_manager import ConversationManager

__all__ = [
    "UnifiedGovernanceSteward",
    "TeleologicalOperator",
    "MathematicalState",
    "PrimacyAttractorMath",
    "TelicFidelityCalculator",
    "EmbeddingProvider",
    "ConversationManager",
]
