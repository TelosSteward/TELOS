"""
TELOS Core Constants
====================

Single source of truth for all calibration thresholds.
Factored from TELOS Observatory V3 for reuse across deployments.

WARNING: Modifying these values affects both conversational and agentic governance.
All changes should be validated against the whitepaper specifications.
"""

# =============================================================================
# TWO-LAYER FIDELITY DETECTION
# =============================================================================
# Layer 1: Baseline Pre-Filter (catches extreme off-topic)
SIMILARITY_BASELINE = 0.20  # Raw cosine sim < this = HARD_BLOCK

# Layer 2: Basin Membership (detects purpose drift)
BASIN_CENTER = 0.50         # Center of primacy basin
BASIN_TOLERANCE = 0.02      # Mathematical tolerance
INTERVENTION_THRESHOLD = 0.48  # Basin boundary = center - tolerance

# =============================================================================
# DISPLAY ZONES (UI Thresholds)
# =============================================================================
FIDELITY_GREEN = 0.70       # Aligned - no intervention
FIDELITY_YELLOW = 0.60      # Minor drift - context injection
FIDELITY_ORANGE = 0.50      # Drift detected - steward redirect
FIDELITY_RED = 0.50         # Significant drift - block + review

# Zone color mappings
ZONE_COLORS = {
    "GREEN": "#27ae60",
    "YELLOW": "#f39c12",
    "ORANGE": "#e67e22",
    "RED": "#e74c3c",
}

# =============================================================================
# PROPORTIONAL CONTROL (Whitepaper Section 5.3)
# =============================================================================
# Control law: F = K * e_t where e_t = (target - current) / target
DEFAULT_K_ATTRACTOR = 1.5       # Primary proportional gain
DEFAULT_K_ANTIMETA = 2.0        # Anti-meta commentary gain
DEFAULT_CONSTRAINT_TOLERANCE = 0.40  # Tau (constraint/flexibility balance)

# Error signal boundaries (scaled by tau)
# epsilon_min = tau * 0.10, epsilon_max = tau * 0.50
DEFAULT_TAU = 0.40
EPSILON_MIN_FACTOR = 0.10
EPSILON_MAX_FACTOR = 0.50

# Intervention states
INTERVENTION_STATES = {
    "MONITOR": {"threshold": 0.04, "description": "Within tolerance"},
    "CORRECT": {"threshold": 0.08, "description": "Minor correction needed"},
    "INTERVENE": {"threshold": 0.15, "description": "Active intervention"},
    "ESCALATE": {"threshold": 0.20, "description": "Escalation required"},
}

# =============================================================================
# SEMANTIC CONTINUITY INDEX (SCI)
# =============================================================================
# Continuity thresholds and decay factors
SCI_STRONG_THRESHOLD = 0.70     # High semantic continuity
SCI_MODERATE_THRESHOLD = 0.50   # Moderate continuity
SCI_WEAK_THRESHOLD = 0.30       # Weak continuity

SCI_STRONG_DECAY = 0.99         # Almost no decay for strong continuity
SCI_MODERATE_DECAY = 0.95       # Slight decay for moderate
SCI_WEAK_DECAY = 0.90           # Notable decay for weak

# Inheritance cap (maximum inherited fidelity)
SCI_INHERITANCE_CAP = 0.95

# =============================================================================
# ADAPTIVE CONTEXT SYSTEM
# =============================================================================
# Tier thresholds (fidelity-based classification)
TIER1_THRESHOLD = 0.70      # High fidelity messages -> Tier 1
TIER2_THRESHOLD = 0.35      # Medium fidelity -> Tier 2
TIER3_THRESHOLD = 0.25      # Low fidelity -> Tier 3

# Buffer capacities
TIER1_CAPACITY = 5          # Most recent high-fidelity items
TIER2_CAPACITY = 3          # Most recent medium-fidelity
TIER3_CAPACITY = 2          # Most recent low-fidelity

# Governance safeguards
HARD_FLOOR = 0.20           # Minimum threshold (never go below)
MAX_BOOST = 0.20            # Maximum threshold increase
BASE_THRESHOLD = 0.48       # Default intervention threshold

# Context weighting
RECENCY_DECAY = 0.8         # Lambda for weighted embeddings
PHASE_WINDOW_SIZE = 5       # Rolling window for phase detection

# =============================================================================
# MESSAGE TYPE CLASSIFICATION
# =============================================================================
MESSAGE_TYPE_THRESHOLDS = {
    "DIRECT": 0.70,         # Clear, on-topic statements
    "FOLLOW_UP": 0.35,      # Continuations, elaborations
    "CLARIFICATION": 0.25,  # Questions about prior content
    "ANAPHORA": 0.25,       # Pronoun references
}

# Message type boosts for context calculation
MESSAGE_TYPE_BOOSTS = {
    "DIRECT": 0.7,          # Direct statements get less boost
    "FOLLOW_UP": 1.2,       # Follow-ups inherit context well
    "CLARIFICATION": 1.1,   # Clarifications have moderate inheritance
    "ANAPHORA": 1.5,        # Anaphora heavily relies on context
}

# =============================================================================
# MODEL-SPECIFIC CALIBRATION
# =============================================================================
# SentenceTransformer (all-MiniLM-L6-v2, 384-dim)
ST_RAW_MIN = 0.15           # Minimum expected raw similarity
ST_RAW_MAX = 0.45           # Maximum expected raw similarity
ST_DISPLAY_MIN = 0.35       # Display range minimum
ST_DISPLAY_MAX = 0.95       # Display range maximum

# Mistral (mistral-embed, 1024-dim)
MISTRAL_RAW_MIN = 0.40      # Minimum expected raw similarity
MISTRAL_RAW_MAX = 0.75      # Maximum expected raw similarity
MISTRAL_DISPLAY_MIN = 0.50  # Display range minimum
MISTRAL_DISPLAY_MAX = 1.00  # Display range maximum

# =============================================================================
# SEMANTIC INTERPRETER BANDS
# =============================================================================
STRENGTH_BANDS = {
    "MINIMAL": {"max_strength": 0.45, "fidelity_min": 0.70},
    "LIGHT": {"max_strength": 0.60, "fidelity_min": 0.60},
    "MODERATE": {"max_strength": 0.75, "fidelity_min": 0.50},
    "FIRM": {"max_strength": 0.85, "fidelity_min": 0.40},
    "STRONG": {"max_strength": 1.00, "fidelity_min": 0.00},
}

# =============================================================================
# AGENTIC AI EXTENSIONS
# =============================================================================
# Action chain governance (new for LangGraph)
ACTION_CHAIN_DECAY = 0.85       # Decay factor for action chain SCI
ACTION_CHAIN_WINDOW = 10        # Max actions to consider for continuity

# Handoff governance
HANDOFF_THRESHOLD = FIDELITY_GREEN  # Fidelity threshold for auto-approve
HANDOFF_SCI_BONUS_THRESHOLD = 0.50  # SCI threshold for handoff boost
HANDOFF_SCI_BONUS_FACTOR = 0.80     # Handoff approved if fidelity >= threshold * factor

# Delegation governance
DELEGATION_THRESHOLD = FIDELITY_GREEN  # Fidelity for auto-approve delegation
DELEGATION_REQUIRE_APPROVAL = True     # Require approval below threshold

# Tool execution governance
TOOL_CALL_THRESHOLD = FIDELITY_YELLOW  # Lower threshold for tool calls
TOOL_CALL_CONTEXT_INJECT = True        # Inject PA context for drift

# =============================================================================
# PRIMACY STATE CALCULATION
# =============================================================================
# PS = rho_PA * (2 * F_user * F_ai) / (F_user + F_ai)
# Simplified: PS = rho_PA * harmonic_mean(F_user, F_ai)
DEFAULT_RHO_PA = 1.0        # PA density (usually 1.0 for single PA)

# Attractor center calculation
# a_hat = (tau * purpose + (1-tau) * scope) / ||...||
DEFAULT_PURPOSE_WEIGHT = 0.60  # Tau for purpose vs scope


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def get_zone_from_fidelity(fidelity: float) -> str:
    """Get zone name from fidelity score."""
    if fidelity >= FIDELITY_GREEN:
        return "GREEN"
    elif fidelity >= FIDELITY_YELLOW:
        return "YELLOW"
    elif fidelity >= FIDELITY_ORANGE:
        return "ORANGE"
    else:
        return "RED"


def get_zone_color(zone: str) -> str:
    """Get color for zone."""
    return ZONE_COLORS.get(zone.upper(), ZONE_COLORS["RED"])


def get_fidelity_color(fidelity: float) -> str:
    """Get color directly from fidelity score."""
    zone = get_zone_from_fidelity(fidelity)
    return get_zone_color(zone)


def calculate_epsilon_bounds(tau: float = DEFAULT_TAU):
    """Calculate error signal bounds from constraint tolerance."""
    epsilon_min = tau * EPSILON_MIN_FACTOR
    epsilon_max = tau * EPSILON_MAX_FACTOR
    return epsilon_min, epsilon_max
