"""
TELOS Core Mathematical Framework
=================================

The battle-tested mathematical substrate for TELOS governance.
Shared by both conversational (Observatory) and agentic (LangGraph) deployments.

Core Components:
    - constants: All calibration thresholds (single source of truth)
    - fidelity: Two-layer fidelity detection (baseline + basin)
    - proportional_control: Graduated intervention (F = K·e_t)
    - semantic_specs: Fidelity → linguistic specifications
    - continuity: Semantic Continuity Index (SCI) for both turns and actions
    - adaptive_context: DMAIC-inspired context management

Mathematical Foundation:
    - Cosine similarity in embedding space
    - Basin membership geometry (center=0.50, tolerance=0.02)
    - Proportional control law from Whitepaper Section 5.3
    - SCI inheritance with decay factors

Version: 1.0.0 (factored from TELOS Observatory V3)
"""

from .constants import (
    # Two-Layer Fidelity Detection
    SIMILARITY_BASELINE,
    INTERVENTION_THRESHOLD,
    BASIN_CENTER,
    BASIN_TOLERANCE,

    # Display Zones
    FIDELITY_GREEN,
    FIDELITY_YELLOW,
    FIDELITY_ORANGE,
    FIDELITY_RED,

    # Proportional Control
    DEFAULT_K_ATTRACTOR,
    DEFAULT_K_ANTIMETA,
    DEFAULT_CONSTRAINT_TOLERANCE,

    # SCI Parameters
    SCI_STRONG_THRESHOLD,
    SCI_MODERATE_THRESHOLD,
    SCI_WEAK_THRESHOLD,
    SCI_STRONG_DECAY,
    SCI_MODERATE_DECAY,
    SCI_WEAK_DECAY,

    # Context Tiers
    TIER1_THRESHOLD,
    TIER2_THRESHOLD,
    TIER3_THRESHOLD,
)

from .fidelity import (
    calculate_raw_similarity,
    normalize_fidelity,
    calculate_fidelity,
    check_baseline_violation,
    check_basin_membership,
    get_fidelity_zone,
    FidelityZone,
    FidelityResult,
)

from .proportional_control import (
    ProportionalController,
    InterventionState,
    InterventionRecord,
    calculate_error_signal,
    calculate_intervention_strength,
)

from .semantic_specs import (
    SemanticSpec,
    StrengthBand,
    interpret_fidelity,
    get_strength_band,
    get_exemplar,
)

from .continuity import (
    calculate_semantic_continuity,
    apply_continuity_inheritance,
    ContinuityResult,
    ContinuityLevel,
)

from .adaptive_context import (
    AdaptiveContextManager,
    MessageType,
    ConversationPhase,
    ActionPhase,  # NEW: For agentic AI
    TieredBuffer,
    PhaseDetector,
    AdaptiveThresholdCalculator,
    ContextResult,
)

__version__ = "1.0.0"
__all__ = [
    # Constants
    "SIMILARITY_BASELINE",
    "INTERVENTION_THRESHOLD",
    "BASIN_CENTER",
    "BASIN_TOLERANCE",
    "FIDELITY_GREEN",
    "FIDELITY_YELLOW",
    "FIDELITY_ORANGE",
    "FIDELITY_RED",
    "DEFAULT_K_ATTRACTOR",
    "DEFAULT_K_ANTIMETA",
    "DEFAULT_CONSTRAINT_TOLERANCE",
    "SCI_STRONG_THRESHOLD",
    "SCI_MODERATE_THRESHOLD",
    "SCI_WEAK_THRESHOLD",
    "SCI_STRONG_DECAY",
    "SCI_MODERATE_DECAY",
    "SCI_WEAK_DECAY",
    "TIER1_THRESHOLD",
    "TIER2_THRESHOLD",
    "TIER3_THRESHOLD",

    # Fidelity
    "calculate_raw_similarity",
    "normalize_fidelity",
    "calculate_fidelity",
    "check_baseline_violation",
    "check_basin_membership",
    "get_fidelity_zone",
    "FidelityZone",
    "FidelityResult",

    # Proportional Control
    "ProportionalController",
    "InterventionState",
    "InterventionRecord",
    "calculate_error_signal",
    "calculate_intervention_strength",

    # Semantic Specs
    "SemanticSpec",
    "StrengthBand",
    "interpret_fidelity",
    "get_strength_band",
    "get_exemplar",

    # Continuity
    "calculate_semantic_continuity",
    "apply_continuity_inheritance",
    "ContinuityResult",
    "ContinuityLevel",

    # Adaptive Context
    "AdaptiveContextManager",
    "MessageType",
    "ConversationPhase",
    "ActionPhase",
    "TieredBuffer",
    "PhaseDetector",
    "AdaptiveThresholdCalculator",
    "ContextResult",
]
