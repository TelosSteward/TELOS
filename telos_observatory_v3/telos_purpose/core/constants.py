"""
TELOS Constants
---------------
Centralized configuration constants aligned with whitepaper specifications.

Per whitepaper Section 5.3, these constants define the intervention cascade
and process control parameters for the Mitigation Bridge Layer (MBL).

Usage:
    from telos_purpose.core.constants import FIDELITY_MONITOR, DEFAULT_K_ATTRACTOR
"""

# ============================================================================
# Embedding Configuration
# ============================================================================

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
"""Default sentence transformer model for text embeddings."""

EMBEDDING_DIMENSION = 384
"""Dimensionality of all-MiniLM-L6-v2 embeddings."""


# ============================================================================
# Fidelity Thresholds - UNIFIED CONFIGURATION
# ============================================================================
# IMPORTANT: This is the SINGLE SOURCE OF TRUTH for all fidelity thresholds.
# All other files MUST import from here. Do NOT define thresholds elsewhere.
#
# ARCHITECTURE: Two-layer fidelity system
#   Layer 1: Baseline pre-filter (raw similarity) - catches extreme off-topic
#   Layer 2: Basin membership (normalized fidelity) - catches purpose drift
#
# DISPLAY ZONES: Used for UI coloring and user feedback
#   GREEN:  >= 0.70 - Aligned (no intervention)
#   YELLOW: 0.60-0.69 - Minor drift (context injection)
#   ORANGE: 0.50-0.59 - Drift detected (Steward redirect)
#   RED:    < 0.50 - Significant drift (block + review)
#
# INTERVENTION DECISION: Separate from display zones
#   Intervene when: (raw_sim < SIMILARITY_BASELINE) OR (fidelity < INTERVENTION_THRESHOLD)
# ============================================================================

# --- Display Zone Thresholds (Normalized 0-1 scale) ---
# These define UI color zones for user feedback
FIDELITY_GREEN = 0.70
"""
Display Zone: GREEN (Aligned)
Fidelity >= 0.70: No intervention needed. Conversation is on track.
"""

FIDELITY_YELLOW = 0.60
"""
Display Zone: YELLOW (Minor Drift)
Fidelity 0.60-0.69: Light context injection may occur.
"""

FIDELITY_ORANGE = 0.50
"""
Display Zone: ORANGE (Drift Detected)
Fidelity 0.50-0.59: Steward redirect likely.
"""

FIDELITY_RED = 0.50
"""
Display Zone: RED (Significant Drift)
Fidelity < 0.50: Strong intervention or block.
"""

# --- Intervention Decision Thresholds ---
# These are SEPARATE from display zones - intervention logic uses these
SIMILARITY_BASELINE = 0.20
"""
Layer 1 Threshold: Raw cosine similarity baseline for HARD_BLOCK.
If raw_similarity < this value, content is extreme off-topic.
Model-specific: calibrated for SentenceTransformer (384-dim).
"""

BASIN_CENTER = 0.50
"""
Layer 2: Basin center for fidelity normalization.
The "expected" fidelity for on-topic content.
"""

BASIN_TOLERANCE = 0.02
"""
Layer 2: Tolerance margin around basin boundary.
Small buffer to prevent oscillation at boundary.
"""

INTERVENTION_THRESHOLD = 0.48
"""
Layer 2 Threshold: Normalized fidelity below which intervention triggers.
Computed as: BASIN_CENTER - BASIN_TOLERANCE = 0.48
If fidelity < 0.48, user has drifted outside the primacy basin.
"""

# --- Legacy Goldilocks Constants (Deprecated) ---
# These were from grid search optimization but created confusion.
# Kept for backward compatibility but new code should use above constants.
FIDELITY_MONITOR = FIDELITY_GREEN  # Alias: 0.70
FIDELITY_CORRECT = 0.65  # Deprecated: use FIDELITY_YELLOW (0.60)
FIDELITY_INTERVENE = 0.55  # Deprecated: use FIDELITY_ORANGE (0.50)
FIDELITY_ESCALATE = FIDELITY_RED  # Alias: 0.50


# ============================================================================
# RAW Embedding Model Thresholds (No Rescaling Required)
# ============================================================================
# Different embedding models produce different cosine similarity ranges.
# These thresholds are calibrated for each model's native output range.
#
# LEAN SIX SIGMA: Clean lanes - use raw thresholds directly, no rescaling.
# ============================================================================

# --- SentenceTransformer (all-MiniLM-L6-v2, 384-dim) ---
# Produces cosine similarity in narrow range due to lower dimensionality.
# On-topic: typically 0.30-0.50
# Off-topic: typically 0.20-0.28 (completely unrelated content like PB&J vs AI governance)
# EMPIRICALLY TUNED: Based on observed raw scores with pure domain PA centroid
# - TELOS queries (~0.337 raw) → GREEN (0.74 display)
# - PB&J sandwich (~0.274 raw) → ORANGE (0.585 display)
ST_FIDELITY_GREEN = 0.32
"""SentenceTransformer: Clearly aligned (raw cosine >= 0.32)"""

ST_FIDELITY_YELLOW = 0.28
"""SentenceTransformer: Minor drift (raw cosine 0.28-0.32)"""

ST_FIDELITY_ORANGE = 0.24
"""SentenceTransformer: Drift detected (raw cosine 0.24-0.28)"""

ST_FIDELITY_RED = 0.24
"""SentenceTransformer: Significant drift (raw cosine < 0.24)"""


# --- Mistral Embed (mistral-embed, 1024-dim) ---
# Higher dimensionality produces higher baseline due to concentration of measure.
# On-topic: typically 0.55-0.80
# Off-topic: typically 0.35-0.55
# Note: SIMILARITY_BASELINE of 0.35 catches extreme off-topic (Layer 1).
MISTRAL_FIDELITY_GREEN = 0.60
"""Mistral: Clearly aligned (raw cosine >= 0.60)"""

MISTRAL_FIDELITY_YELLOW = 0.50
"""Mistral: Minor drift (raw cosine 0.50-0.60)"""

MISTRAL_FIDELITY_ORANGE = 0.42
"""Mistral: Drift detected (raw cosine 0.42-0.50)"""

MISTRAL_FIDELITY_RED = 0.42
"""Mistral: Significant drift (raw cosine < 0.42)"""


def get_thresholds_for_model(model_type: str) -> dict:
    """
    Get the appropriate raw thresholds for an embedding model.

    Args:
        model_type: 'sentence_transformer' or 'mistral'

    Returns:
        dict with 'green', 'yellow', 'orange', 'red' threshold values

    Example:
        >>> thresholds = get_thresholds_for_model('sentence_transformer')
        >>> if raw_similarity >= thresholds['green']:
        ...     zone = 'GREEN'
    """
    if model_type == 'sentence_transformer':
        return {
            'green': ST_FIDELITY_GREEN,
            'yellow': ST_FIDELITY_YELLOW,
            'orange': ST_FIDELITY_ORANGE,
            'red': ST_FIDELITY_RED
        }
    elif model_type == 'mistral':
        return {
            'green': MISTRAL_FIDELITY_GREEN,
            'yellow': MISTRAL_FIDELITY_YELLOW,
            'orange': MISTRAL_FIDELITY_ORANGE,
            'red': MISTRAL_FIDELITY_RED
        }
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'sentence_transformer' or 'mistral'.")


# ============================================================================
# Error Signal Thresholds
# ============================================================================

# Base thresholds (scaled by constraint_tolerance τ)
ERROR_MIN_BASE = 0.1
"""Base value for minimum error threshold (scaled by 0.3τ)."""

ERROR_MIN_SCALE = 0.3
"""Scaling factor for minimum error threshold."""

ERROR_MAX_BASE = 0.5
"""Base value for maximum error threshold (scaled by 0.4τ)."""

ERROR_MAX_SCALE = 0.4
"""Scaling factor for maximum error threshold."""

def compute_epsilon_min(constraint_tolerance: float) -> float:
    """
    Compute ε_min (correction trigger threshold).
    
    Per whitepaper Section 2.2:
        ε_min = 0.1 + 0.3τ
    
    Args:
        constraint_tolerance: τ ∈ [0, 1]
    
    Returns:
        Error threshold for triggering State 2 (CORRECT)
    
    Example:
        >>> epsilon_min = compute_epsilon_min(0.2)
        >>> print(f"ε_min = {epsilon_min:.3f}")
        ε_min = 0.160
    """
    return ERROR_MIN_BASE + (ERROR_MIN_SCALE * constraint_tolerance)


def compute_epsilon_max(constraint_tolerance: float) -> float:
    """
    Compute ε_max (regeneration trigger threshold).
    
    Per whitepaper Section 2.2:
        ε_max = 0.5 + 0.4τ
    
    Args:
        constraint_tolerance: τ ∈ [0, 1]
    
    Returns:
        Error threshold for triggering State 3 (INTERVENE)
    
    Example:
        >>> epsilon_max = compute_epsilon_max(0.2)
        >>> print(f"ε_max = {epsilon_max:.3f}")
        ε_max = 0.580
    """
    return ERROR_MAX_BASE + (ERROR_MAX_SCALE * constraint_tolerance)


# ============================================================================
# Proportional Control Gains
# ============================================================================

DEFAULT_K_ATTRACTOR = 1.5
"""
Proportional gain for attractor-based corrections.

Per whitepaper Section 2.2, correction force:
    F = K·e_t
    
where e_t is normalized error distance.
"""

DEFAULT_K_ANTIMETA = 2.0
"""
Proportional gain for meta-commentary suppression.

Higher gain ensures strong correction when model discusses
its own constraints or governance mechanisms.
"""


# ============================================================================
# Intervention Limits
# ============================================================================

DEFAULT_MAX_REGENERATIONS = 3
"""
Maximum regenerations per session.

Prevents infinite regeneration loops while allowing
correction of transient drift.
"""

INTERVENTION_BUDGET_PER_SESSION = 10
"""
Maximum total interventions per session.

Soft limit for detecting systemic issues vs transient drift.
"""


# ============================================================================
# Basin Geometry
# ============================================================================

def compute_basin_radius(constraint_tolerance: float) -> float:
    """
    Compute primacy basin radius.
    
    Per whitepaper Section 2.1:
        r = 2 / max(ρ, 0.25)
        where ρ = 1 - τ (constraint rigidity)
    
    Args:
        constraint_tolerance: τ ∈ [0, 1]
    
    Returns:
        Basin radius r
    
    Example:
        >>> radius = compute_basin_radius(0.2)
        >>> print(f"r = {radius:.3f}")
        r = 2.500
    """
    rigidity = 1.0 - constraint_tolerance
    return 2.0 / max(rigidity, 0.25)


BASIN_RADIUS_MIN = 2.0
"""Minimum basin radius (when τ = 0, maximum rigidity)."""

BASIN_RADIUS_MAX = 8.0
"""Maximum basin radius (when τ = 1, maximum tolerance)."""


# ============================================================================
# LLM Configuration
# ============================================================================

DEFAULT_MAX_TOKENS = 500
"""Default maximum tokens for LLM generation."""

DEFAULT_TEMPERATURE = 0.7
"""Default sampling temperature for LLM generation."""

MAX_CONVERSATION_HISTORY = 20
"""
Maximum conversation turns to maintain in history.

Prevents token budget overflow while preserving recent context.
"""


# ============================================================================
# Retry Configuration
# ============================================================================

MAX_API_RETRIES = 3
"""Maximum retry attempts for transient API errors."""

RETRY_BASE_DELAY = 1.0
"""Base delay (seconds) for exponential backoff."""

RETRY_MAX_DELAY = 32.0
"""Maximum delay (seconds) for exponential backoff."""

RETRY_JITTER = 0.1
"""Jitter factor to prevent thundering herd (±10%)."""


# ============================================================================
# Telemetry Configuration
# ============================================================================

TELEMETRY_VERSION = "1.0"
"""Version identifier for telemetry schema."""

TELEMETRY_CSV_COLUMNS = [
    "turn",
    "error_signal",
    "fidelity",
    "in_basin",
    "lyapunov",
    "governance_action",
    "intervention_type",
    "latency_ms"
]
"""Standard columns for CSV telemetry export."""


# ============================================================================
# Validation Configuration
# ============================================================================

# Hypothesis testing thresholds
H1_DELTA_F_THRESHOLD = 0.15
"""
Hypothesis 1: ΔF ≥ 0.15

TELOS must improve fidelity by at least 0.15 over
best baseline to pass internal validation.
"""

H2_SUPREMACY_THRESHOLD = 0.01
"""
Hypothesis 2: F_telos > F_baseline + 0.01

TELOS must achieve highest fidelity among all
conditions to pass supremacy test.
"""


# Test conversation requirements
MIN_TEST_CONVERSATION_TURNS = 5
"""Minimum turns required for valid test conversation."""

MAX_TEST_CONVERSATION_TURNS = 50
"""Maximum turns allowed in test conversation."""


# ============================================================================
# Logging Configuration
# ============================================================================

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
"""Standard log message format."""

LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
"""Standard log timestamp format."""


# ============================================================================
# File Paths
# ============================================================================

DEFAULT_CONFIG_FILE = "config.json"
"""Default configuration file name."""

DEFAULT_OUTPUT_DIR = "validation_results"
"""Default directory for validation outputs."""

DEFAULT_LOG_FILE = "telos.log"
"""Default log file name."""


# ============================================================================
# Numerical Stability
# ============================================================================

EPSILON_NUMERICAL = 1e-10
"""Small value to prevent division by zero."""

MAX_FLOAT_VALUE = 1e10
"""Maximum allowable float value before flagging instability."""


# ============================================================================
# Meta-Commentary Detection Patterns
# ============================================================================

META_COMMENTARY_PATTERNS = [
    r'\bmy purpose is\b',
    r'\bmy constraints\b',
    r'\bi am designed to\b',
    r'\bmy guardrails\b',
    r'\baccording to my instructions\b',
    r'\bas an ai language model\b',
    r'\bas a large language model\b'
]
"""
Regex patterns for detecting meta-commentary.

Used by Proportional Controller to suppress model discussion
of its own governance mechanisms.
"""


# ============================================================================
# Process Capability Thresholds
# ============================================================================

P_CAP_CAPABLE = 1.33
"""
Minimum P_cap for capable process.

P_cap ≥ 1.33 indicates process is statistically capable
of meeting specifications.
"""

P_CAP_HIGHLY_CAPABLE = 1.67
"""
P_cap threshold for highly capable process.

P_cap ≥ 1.67 indicates excellent process control.
"""


# ============================================================================
# Health Monitoring Thresholds
# ============================================================================

HEALTH_CHECK_INTERVAL_TURNS = 5
"""Turns between automatic health checks."""

LATENCY_WARNING_THRESHOLD_MS = 1000.0
"""Latency threshold for performance warnings (ms)."""

LATENCY_CRITICAL_THRESHOLD_MS = 5000.0
"""Latency threshold for critical alerts (ms)."""

FIDELITY_WARNING_THRESHOLD = 0.60
"""Fidelity threshold for governance warnings."""

FIDELITY_CRITICAL_THRESHOLD = 0.40
"""Fidelity threshold for critical governance alerts."""


# ============================================================================
# Version Information
# ============================================================================

TELOS_VERSION = "1.0.0"
"""TELOS package version."""

MBL_ARCHITECTURE_VERSION = "1.0"
"""Mitigation Bridge Layer architecture version."""

WHITEPAPER_VERSION = "2024-10-18"
"""Canonical whitepaper version identifier."""
