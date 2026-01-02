"""
TELOS Semantic Continuity Index (SCI)
=====================================

Extended SCI implementation for both conversational turns and agentic
action chains. This is the key innovation for agentic AI governance.

SCI v4.0 - Conversation Continuity:
    - Measures semantic similarity between current input and prior turns
    - Applies decay-based inheritance from high-fidelity context

SCI v5.0 - Action Chain Continuity (NEW):
    - Extends SCI to sequences of tool calls and agent handoffs
    - Enables inherited fidelity across action chains
    - Critical for multi-agent governance

Battle-tested foundation from TELOS Observatory V3,
extended for agentic AI governance.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple
import numpy as np
import logging

from .constants import (
    SCI_STRONG_THRESHOLD,
    SCI_MODERATE_THRESHOLD,
    SCI_WEAK_THRESHOLD,
    SCI_STRONG_DECAY,
    SCI_MODERATE_DECAY,
    SCI_WEAK_DECAY,
    SCI_INHERITANCE_CAP,
    ACTION_CHAIN_DECAY,
    ACTION_CHAIN_WINDOW,
    TIER1_THRESHOLD,
)

logger = logging.getLogger(__name__)


class ContinuityLevel(Enum):
    """Semantic continuity strength levels."""
    NONE = "none"           # No continuity (new topic)
    WEAK = "weak"           # Tenuous connection
    MODERATE = "moderate"   # Clear connection
    STRONG = "strong"       # Very strong continuity


@dataclass
class ContinuityResult:
    """Result of continuity calculation."""
    continuity_score: float
    level: ContinuityLevel
    best_match_index: int
    inherited_fidelity: float
    decay_applied: float
    direct_fidelity: float

    def to_dict(self) -> dict:
        return {
            "continuity_score": self.continuity_score,
            "level": self.level.value,
            "best_match_index": self.best_match_index,
            "inherited_fidelity": self.inherited_fidelity,
            "decay_applied": self.decay_applied,
            "direct_fidelity": self.direct_fidelity,
        }


def get_continuity_level(score: float) -> ContinuityLevel:
    """Get continuity level from score."""
    if score >= SCI_STRONG_THRESHOLD:
        return ContinuityLevel.STRONG
    elif score >= SCI_MODERATE_THRESHOLD:
        return ContinuityLevel.MODERATE
    elif score >= SCI_WEAK_THRESHOLD:
        return ContinuityLevel.WEAK
    else:
        return ContinuityLevel.NONE


def get_decay_factor(level: ContinuityLevel) -> float:
    """Get decay factor for continuity level."""
    decay_map = {
        ContinuityLevel.STRONG: SCI_STRONG_DECAY,
        ContinuityLevel.MODERATE: SCI_MODERATE_DECAY,
        ContinuityLevel.WEAK: SCI_WEAK_DECAY,
        ContinuityLevel.NONE: 0.0,
    }
    return decay_map[level]


# =============================================================================
# CONVERSATIONAL SCI (v4.0)
# =============================================================================

def calculate_semantic_continuity(
    current_embedding: np.ndarray,
    previous_embeddings: List[np.ndarray],
    previous_fidelities: Optional[List[float]] = None,
) -> Tuple[float, int]:
    """
    Calculate semantic continuity with previous turns.

    Uses MAX similarity to find best matching prior turn.
    This captures "tell me more about that" patterns.

    Args:
        current_embedding: Embedding of current input
        previous_embeddings: List of prior turn embeddings
        previous_fidelities: Optional fidelities for weighting

    Returns:
        Tuple of (max_similarity, best_match_index)
    """
    if not previous_embeddings:
        return 0.0, -1

    current = np.asarray(current_embedding, dtype=np.float64)
    norm_current = np.linalg.norm(current)

    if norm_current < 1e-10:
        return 0.0, -1

    max_sim = 0.0
    best_idx = -1

    for i, prev_emb in enumerate(previous_embeddings):
        prev = np.asarray(prev_emb, dtype=np.float64)
        norm_prev = np.linalg.norm(prev)

        if norm_prev < 1e-10:
            continue

        sim = float(np.dot(current, prev) / (norm_current * norm_prev))

        # Weight by fidelity if available
        if previous_fidelities and i < len(previous_fidelities):
            fidelity_weight = max(0.5, previous_fidelities[i])
            sim *= fidelity_weight

        if sim > max_sim:
            max_sim = sim
            best_idx = i

    return max_sim, best_idx


def apply_continuity_inheritance(
    direct_fidelity: float,
    continuity_score: float,
    previous_fidelity: float,
) -> Tuple[float, str]:
    """
    Apply fidelity inheritance based on semantic continuity.

    High continuity with a high-fidelity turn allows the current
    input to inherit some of that fidelity.

    Args:
        direct_fidelity: Direct fidelity of current input
        continuity_score: SCI score with prior turn
        previous_fidelity: Fidelity of best-matching prior turn

    Returns:
        Tuple of (adjusted_fidelity, explanation)
    """
    level = get_continuity_level(continuity_score)

    # No inheritance for low continuity
    if level == ContinuityLevel.NONE:
        return direct_fidelity, "No continuity - direct fidelity only"

    # Get decay factor
    decay = get_decay_factor(level)

    # Calculate inherited fidelity
    inherited = previous_fidelity * decay * continuity_score

    # Cap inheritance
    inherited = min(inherited, SCI_INHERITANCE_CAP)

    # Only inherit if it improves fidelity
    if inherited <= direct_fidelity:
        return direct_fidelity, f"Direct fidelity higher than inherited"

    # Apply inheritance
    adjusted = max(direct_fidelity, inherited)
    explanation = (
        f"Inherited fidelity via {level.value} continuity: "
        f"{direct_fidelity:.2f} -> {adjusted:.2f}"
    )

    return adjusted, explanation


def calculate_conversational_sci(
    current_embedding: np.ndarray,
    current_direct_fidelity: float,
    previous_embeddings: List[np.ndarray],
    previous_fidelities: List[float],
) -> ContinuityResult:
    """
    Full conversational SCI calculation.

    Combines similarity calculation with fidelity inheritance.

    Args:
        current_embedding: Current input embedding
        current_direct_fidelity: Direct fidelity of current input
        previous_embeddings: Prior turn embeddings
        previous_fidelities: Prior turn fidelities

    Returns:
        ContinuityResult with all SCI metrics
    """
    # Calculate continuity
    continuity_score, best_idx = calculate_semantic_continuity(
        current_embedding, previous_embeddings, previous_fidelities
    )

    level = get_continuity_level(continuity_score)
    decay = get_decay_factor(level)

    # Get previous fidelity for inheritance
    if best_idx >= 0 and best_idx < len(previous_fidelities):
        prev_fidelity = previous_fidelities[best_idx]
    else:
        prev_fidelity = 0.0

    # Apply inheritance
    inherited, _ = apply_continuity_inheritance(
        current_direct_fidelity, continuity_score, prev_fidelity
    )

    return ContinuityResult(
        continuity_score=continuity_score,
        level=level,
        best_match_index=best_idx,
        inherited_fidelity=inherited,
        decay_applied=decay,
        direct_fidelity=current_direct_fidelity,
    )


# =============================================================================
# ACTION CHAIN SCI (v5.0) - NEW FOR AGENTIC AI
# =============================================================================

@dataclass
class ActionChainEntry:
    """Entry in an action chain for SCI tracking."""
    action_text: str
    embedding: np.ndarray
    fidelity: float
    action_type: str  # "tool_call", "handoff", "delegation"
    timestamp: str
    agent_name: Optional[str] = None


def calculate_action_chain_sci(
    current_embedding: np.ndarray,
    current_direct_fidelity: float,
    action_chain: List[ActionChainEntry],
    window_size: int = ACTION_CHAIN_WINDOW,
) -> ContinuityResult:
    """
    Calculate SCI for action chains (tool calls, handoffs).

    Extends conversational SCI to sequences of agent actions.
    This is critical for multi-agent governance where actions
    should maintain purpose alignment across handoffs.

    Args:
        current_embedding: Current action embedding
        current_direct_fidelity: Direct fidelity of current action
        action_chain: Previous actions in chain
        window_size: Number of prior actions to consider

    Returns:
        ContinuityResult for action chain
    """
    if not action_chain:
        return ContinuityResult(
            continuity_score=0.0,
            level=ContinuityLevel.NONE,
            best_match_index=-1,
            inherited_fidelity=current_direct_fidelity,
            decay_applied=0.0,
            direct_fidelity=current_direct_fidelity,
        )

    # Get recent actions within window
    recent = action_chain[-window_size:]

    # Extract embeddings and fidelities
    embeddings = [e.embedding for e in recent]
    fidelities = [e.fidelity for e in recent]

    # Calculate SCI with action chain decay
    continuity_score, best_idx = calculate_semantic_continuity(
        current_embedding, embeddings, fidelities
    )

    # Apply action chain specific decay
    level = get_continuity_level(continuity_score)
    base_decay = get_decay_factor(level)

    # Additional decay for action chains (more conservative)
    action_decay = base_decay * ACTION_CHAIN_DECAY

    # Get previous fidelity
    if best_idx >= 0 and best_idx < len(fidelities):
        prev_fidelity = fidelities[best_idx]
    else:
        prev_fidelity = 0.0

    # Calculate inherited fidelity with action chain decay
    inherited = prev_fidelity * action_decay * continuity_score
    inherited = min(inherited, SCI_INHERITANCE_CAP)
    inherited = max(current_direct_fidelity, inherited)

    return ContinuityResult(
        continuity_score=continuity_score,
        level=level,
        best_match_index=best_idx,
        inherited_fidelity=inherited,
        decay_applied=action_decay,
        direct_fidelity=current_direct_fidelity,
    )


def calculate_handoff_sci(
    handoff_text: str,
    handoff_embedding: np.ndarray,
    handoff_fidelity: float,
    previous_handoff_embedding: Optional[np.ndarray],
    previous_handoff_fidelity: float,
) -> ContinuityResult:
    """
    Calculate SCI specifically for agent handoffs.

    Handoffs are special - they represent transfer of control
    between agents and need careful continuity tracking.

    Args:
        handoff_text: Description of handoff
        handoff_embedding: Embedding of handoff context
        handoff_fidelity: Direct fidelity of handoff
        previous_handoff_embedding: Previous handoff embedding
        previous_handoff_fidelity: Previous handoff fidelity

    Returns:
        ContinuityResult for handoff
    """
    if previous_handoff_embedding is None:
        return ContinuityResult(
            continuity_score=1.0,  # First handoff has full continuity
            level=ContinuityLevel.STRONG,
            best_match_index=0,
            inherited_fidelity=handoff_fidelity,
            decay_applied=1.0,
            direct_fidelity=handoff_fidelity,
        )

    # Calculate similarity with previous handoff
    continuity_score, _ = calculate_semantic_continuity(
        handoff_embedding, [previous_handoff_embedding], [previous_handoff_fidelity]
    )

    level = get_continuity_level(continuity_score)
    decay = get_decay_factor(level)

    # Apply inheritance
    inherited = previous_handoff_fidelity * decay * continuity_score
    inherited = min(inherited, SCI_INHERITANCE_CAP)
    inherited = max(handoff_fidelity, inherited)

    return ContinuityResult(
        continuity_score=continuity_score,
        level=level,
        best_match_index=0,
        inherited_fidelity=inherited,
        decay_applied=decay,
        direct_fidelity=handoff_fidelity,
    )
