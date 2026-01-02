"""
TELOS Core Fidelity
===================

Two-layer fidelity detection system factored from TELOS Observatory V3.
Implements baseline pre-filter + basin membership detection.

Layer 1: Raw cosine similarity check (catches extreme off-topic)
Layer 2: Normalized fidelity within primacy basin (detects purpose drift)

This is the battle-tested mathematical foundation for TELOS governance.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional, Tuple, Union
import numpy as np
import logging

from .constants import (
    SIMILARITY_BASELINE,
    INTERVENTION_THRESHOLD,
    BASIN_CENTER,
    BASIN_TOLERANCE,
    FIDELITY_GREEN,
    FIDELITY_YELLOW,
    FIDELITY_ORANGE,
    FIDELITY_RED,
    ST_RAW_MIN,
    ST_RAW_MAX,
    ST_DISPLAY_MIN,
    ST_DISPLAY_MAX,
    MISTRAL_RAW_MIN,
    MISTRAL_RAW_MAX,
    MISTRAL_DISPLAY_MIN,
    MISTRAL_DISPLAY_MAX,
)

logger = logging.getLogger(__name__)


class FidelityZone(Enum):
    """Fidelity zones for display and intervention logic."""
    GREEN = "green"     # Aligned - no intervention
    YELLOW = "yellow"   # Minor drift - context injection
    ORANGE = "orange"   # Drift detected - steward redirect
    RED = "red"         # Significant drift - block + review


class InterventionLevel(Enum):
    """Intervention levels based on fidelity."""
    NONE = "none"           # No intervention needed
    CONTEXT = "context"     # Inject PA context
    REDIRECT = "redirect"   # Redirect to purpose
    BLOCK = "block"         # Block and require review
    HARD_BLOCK = "hard_block"  # Layer 1 violation


@dataclass
class FidelityResult:
    """Result of fidelity calculation."""
    raw_similarity: float
    normalized_fidelity: float
    zone: FidelityZone
    intervention_level: InterventionLevel
    baseline_violation: bool
    basin_member: bool
    embedding: Optional[np.ndarray] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "raw_similarity": self.raw_similarity,
            "normalized_fidelity": self.normalized_fidelity,
            "zone": self.zone.value,
            "intervention_level": self.intervention_level.value,
            "baseline_violation": self.baseline_violation,
            "basin_member": self.basin_member,
        }


# =============================================================================
# CORE FIDELITY CALCULATIONS
# =============================================================================

def calculate_raw_similarity(
    embedding: np.ndarray,
    pa_embedding: np.ndarray,
) -> float:
    """
    Calculate raw cosine similarity between embedding and PA embedding.

    This is the foundation of TELOS fidelity measurement.

    Args:
        embedding: Input embedding vector
        pa_embedding: Primacy Attractor embedding vector

    Returns:
        Raw cosine similarity in range [-1, 1]
    """
    # Ensure numpy arrays
    embedding = np.asarray(embedding, dtype=np.float64)
    pa_embedding = np.asarray(pa_embedding, dtype=np.float64)

    # Calculate norms
    norm_a = np.linalg.norm(embedding)
    norm_b = np.linalg.norm(pa_embedding)

    # Handle zero vectors
    if norm_a < 1e-10 or norm_b < 1e-10:
        logger.warning("Near-zero norm detected in similarity calculation")
        return 0.0

    # Cosine similarity
    raw_sim = float(np.dot(embedding, pa_embedding) / (norm_a * norm_b))

    # Clamp to valid range (handles numerical precision issues)
    return max(-1.0, min(1.0, raw_sim))


def normalize_fidelity(
    raw_similarity: float,
    model_type: str = "sentencetransformer",
) -> float:
    """
    Normalize raw similarity to display range.

    Different embedding models have different raw similarity ranges.
    This normalizes to a consistent 0-1 display scale.

    SentenceTransformer (all-MiniLM-L6-v2):
        Raw range: ~0.15-0.45 -> Display: 0.35-0.95

    Mistral (mistral-embed):
        Raw range: ~0.40-0.75 -> Display: 0.50-1.00

    Args:
        raw_similarity: Raw cosine similarity
        model_type: "sentencetransformer" or "mistral"

    Returns:
        Normalized fidelity in display range
    """
    model_type = model_type.lower()

    if model_type in ("mistral", "mistral-embed"):
        raw_min, raw_max = MISTRAL_RAW_MIN, MISTRAL_RAW_MAX
        display_min, display_max = MISTRAL_DISPLAY_MIN, MISTRAL_DISPLAY_MAX
    else:
        # Default to SentenceTransformer calibration
        raw_min, raw_max = ST_RAW_MIN, ST_RAW_MAX
        display_min, display_max = ST_DISPLAY_MIN, ST_DISPLAY_MAX

    # Linear interpolation with clamping
    if raw_similarity <= raw_min:
        normalized = display_min
    elif raw_similarity >= raw_max:
        normalized = display_max
    else:
        # Linear scale within expected range
        progress = (raw_similarity - raw_min) / (raw_max - raw_min)
        normalized = display_min + progress * (display_max - display_min)

    # Final clamp to valid range
    return max(0.0, min(1.0, normalized))


def check_baseline_violation(raw_similarity: float) -> bool:
    """
    Layer 1: Check for baseline violation (extreme off-topic).

    Args:
        raw_similarity: Raw cosine similarity

    Returns:
        True if baseline violated (should hard block)
    """
    return raw_similarity < SIMILARITY_BASELINE


def check_basin_membership(fidelity: float) -> bool:
    """
    Layer 2: Check if fidelity is within primacy basin.

    Basin is defined by center (0.50) with tolerance (0.02).
    Membership boundary is at INTERVENTION_THRESHOLD (0.48).

    Args:
        fidelity: Normalized fidelity score

    Returns:
        True if within basin (aligned with purpose)
    """
    return fidelity >= INTERVENTION_THRESHOLD


def get_fidelity_zone(fidelity: float) -> FidelityZone:
    """
    Get the fidelity zone for display purposes.

    Zones:
        GREEN:  >= 0.70 - Aligned, no intervention
        YELLOW: 0.60-0.69 - Minor drift, context injection
        ORANGE: 0.50-0.59 - Drift detected, steward redirect
        RED:    < 0.50 - Significant drift, block + review

    Args:
        fidelity: Normalized fidelity score

    Returns:
        FidelityZone enum value
    """
    if fidelity >= FIDELITY_GREEN:
        return FidelityZone.GREEN
    elif fidelity >= FIDELITY_YELLOW:
        return FidelityZone.YELLOW
    elif fidelity >= FIDELITY_ORANGE:
        return FidelityZone.ORANGE
    else:
        return FidelityZone.RED


def get_intervention_level(
    fidelity: float,
    raw_similarity: float,
) -> InterventionLevel:
    """
    Determine intervention level based on two-layer detection.

    Layer 1 check takes precedence - if baseline violated, hard block.
    Layer 2 uses fidelity zones for graduated intervention.

    Args:
        fidelity: Normalized fidelity score
        raw_similarity: Raw cosine similarity

    Returns:
        InterventionLevel enum value
    """
    # Layer 1: Baseline check (hard block for extreme off-topic)
    if check_baseline_violation(raw_similarity):
        return InterventionLevel.HARD_BLOCK

    # Layer 2: Zone-based intervention
    zone = get_fidelity_zone(fidelity)

    if zone == FidelityZone.GREEN:
        return InterventionLevel.NONE
    elif zone == FidelityZone.YELLOW:
        return InterventionLevel.CONTEXT
    elif zone == FidelityZone.ORANGE:
        return InterventionLevel.REDIRECT
    else:  # RED
        return InterventionLevel.BLOCK


# =============================================================================
# MAIN FIDELITY FUNCTION
# =============================================================================

def calculate_fidelity(
    content: Union[str, np.ndarray],
    pa_embedding: np.ndarray,
    embed_fn: Optional[Callable[[str], np.ndarray]] = None,
    model_type: str = "sentencetransformer",
    return_embedding: bool = False,
) -> FidelityResult:
    """
    Calculate complete fidelity result for content against PA.

    This is the main entry point for fidelity calculation, implementing
    the full two-layer detection system.

    Args:
        content: Text string or pre-computed embedding
        pa_embedding: Primacy Attractor embedding
        embed_fn: Embedding function (required if content is string)
        model_type: Embedding model type for normalization
        return_embedding: Include embedding in result

    Returns:
        FidelityResult with all fidelity metrics

    Raises:
        ValueError: If content is string but no embed_fn provided
    """
    # Get embedding
    if isinstance(content, str):
        if embed_fn is None:
            raise ValueError("embed_fn required when content is a string")
        embedding = embed_fn(content)
    else:
        embedding = np.asarray(content)

    pa_embedding = np.asarray(pa_embedding)

    # Layer 1: Raw similarity
    raw_similarity = calculate_raw_similarity(embedding, pa_embedding)
    baseline_violation = check_baseline_violation(raw_similarity)

    # Normalize to display range
    normalized_fidelity = normalize_fidelity(raw_similarity, model_type)

    # Layer 2: Basin membership
    basin_member = check_basin_membership(normalized_fidelity)

    # Determine zone and intervention
    zone = get_fidelity_zone(normalized_fidelity)
    intervention_level = get_intervention_level(normalized_fidelity, raw_similarity)

    return FidelityResult(
        raw_similarity=raw_similarity,
        normalized_fidelity=normalized_fidelity,
        zone=zone,
        intervention_level=intervention_level,
        baseline_violation=baseline_violation,
        basin_member=basin_member,
        embedding=embedding if return_embedding else None,
    )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def quick_fidelity(
    content: Union[str, np.ndarray],
    pa_embedding: np.ndarray,
    embed_fn: Optional[Callable[[str], np.ndarray]] = None,
    model_type: str = "sentencetransformer",
) -> Tuple[float, float]:
    """
    Quick fidelity calculation returning just the scores.

    For use cases where you just need (raw_similarity, normalized_fidelity).

    Args:
        content: Text or embedding
        pa_embedding: PA embedding
        embed_fn: Embedding function
        model_type: Model type for normalization

    Returns:
        Tuple of (raw_similarity, normalized_fidelity)
    """
    result = calculate_fidelity(content, pa_embedding, embed_fn, model_type)
    return result.raw_similarity, result.normalized_fidelity


def should_intervene(
    fidelity: float,
    raw_similarity: float,
    threshold: float = FIDELITY_GREEN,
) -> bool:
    """
    Simple intervention check.

    Used when you just need a boolean decision.

    Args:
        fidelity: Normalized fidelity
        raw_similarity: Raw cosine similarity
        threshold: Fidelity threshold (default: GREEN zone)

    Returns:
        True if intervention should occur
    """
    # Hard block for baseline violation
    if check_baseline_violation(raw_similarity):
        return True

    # Intervene below threshold
    return fidelity < threshold
