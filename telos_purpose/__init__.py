"""
TELOS - A Mathematical Governance Framework
===========================================

A mathematical runtime framework for AI governance that maintains
alignment across multi-turn conversations. TELOS represents the
τέλος (ultimate purpose) of achieving governed equilibrium.

Version: 1.0.0
Author: Origin Industries PBC / TELOS Labs LLC
"""

__version__ = "1.0.0"
__author__ = "Origin Industries PBC / TELOS Labs LLC"

# Core exports
from .core.unified_steward import UnifiedGovernanceSteward, TeleologicalOperator
from .core.primacy_math import (
    MathematicalState,
    PrimacyAttractorMath,
    TelicFidelityCalculator
)
from .core.constants import (
    FIDELITY_MONITOR,
    FIDELITY_CORRECT,
    FIDELITY_INTERVENE,
    FIDELITY_ESCALATE
)

__all__ = [
    "UnifiedGovernanceSteward",
    "TeleologicalOperator",
    "MathematicalState",
    "PrimacyAttractorMath",
    "TelicFidelityCalculator",
    "FIDELITY_MONITOR",
    "FIDELITY_CORRECT",
    "FIDELITY_INTERVENE",
    "FIDELITY_ESCALATE",
]
