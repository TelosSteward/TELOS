"""
TELOS LangGraph State Schema
============================

Defines the TypedDict state schema for TELOS-governed LangGraph agents.
Includes Primacy Attractor representation, fidelity trajectory, action chains,
and governance trace for insurance-grade audit trails.

Single Source of Truth for Thresholds:
- Constants imported from telos_purpose/core/constants.py
"""

from typing import TypedDict, Annotated, Optional, List, Dict, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import numpy as np

# Try to import LangGraph's add_messages reducer
try:
    from langgraph.graph import add_messages
except ImportError:
    # Fallback: simple message accumulator
    def add_messages(left: list, right: list) -> list:
        return left + right


# =============================================================================
# FIDELITY ZONES (from TELOS constants)
# =============================================================================

class FidelityZone(Enum):
    """Fidelity zones matching TELOS Observatory thresholds."""
    GREEN = "green"      # >= 0.70 - Aligned, no intervention
    YELLOW = "yellow"    # 0.60-0.69 - Minor drift, context injection
    ORANGE = "orange"    # 0.50-0.59 - Drift detected, steward redirect
    RED = "red"          # < 0.50 - Significant drift, block + review


class InterventionLevel(Enum):
    """Intervention levels for governance actions."""
    NONE = "none"
    CONTEXT = "context"       # Inject context, allow action
    REDIRECT = "redirect"     # Soft redirect, suggest alternative
    BLOCK = "block"           # Block action, require approval
    HARD_BLOCK = "hard_block" # Layer 1 baseline violation


# =============================================================================
# GOVERNANCE THRESHOLDS
# =============================================================================

# These match telos_purpose/core/constants.py - single source of truth
SIMILARITY_BASELINE = 0.20      # Layer 1: extreme off-topic hard block
INTERVENTION_THRESHOLD = 0.48   # Layer 2: basin membership boundary
FIDELITY_GREEN = 0.70           # Display zone: aligned
FIDELITY_YELLOW = 0.60          # Display zone: minor drift
FIDELITY_ORANGE = 0.50          # Display zone: drift detected

# SCI (Semantic Continuity Index) thresholds
SCI_CONTINUITY_THRESHOLD = 0.30  # Minimum continuity for inheritance
SCI_DECAY_FACTOR = 0.90          # Decay per step in action chain


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class PrimacyAttractor:
    """
    Embedding-space representation of user purpose.

    The PA serves as the gravitational center for fidelity measurement.
    Actions and messages are compared against the PA to detect drift.
    """
    text: str                           # Natural language purpose statement
    embedding: np.ndarray               # 1024-dim (Mistral) or 384-dim (ST) vector
    purpose: Optional[str] = None       # Extracted purpose component
    scope: Optional[str] = None         # Extracted scope/constraints
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize PA for state storage."""
        return {
            "text": self.text,
            "embedding": self.embedding.tolist() if isinstance(self.embedding, np.ndarray) else self.embedding,
            "purpose": self.purpose,
            "scope": self.scope,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PrimacyAttractor":
        """Deserialize PA from state."""
        return cls(
            text=data["text"],
            embedding=np.array(data["embedding"]),
            purpose=data.get("purpose"),
            scope=data.get("scope"),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(),
            metadata=data.get("metadata", {}),
        )


@dataclass
class GovernanceTraceEntry:
    """
    Single entry in the governance audit trail.

    Provides insurance-grade documentation of every governance decision.
    """
    timestamp: datetime
    turn_number: int
    action_type: str                    # "user_input", "tool_call", "delegation", "response"
    action_description: str
    raw_similarity: float               # Cosine similarity to PA
    fidelity_score: float               # Normalized fidelity
    zone: FidelityZone
    intervention_level: InterventionLevel
    intervention_reason: Optional[str] = None
    approved: bool = True               # False if blocked
    approval_source: Optional[str] = None  # "auto", "human", "override"
    sci_score: Optional[float] = None   # Semantic continuity from prior action
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for storage/transmission."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "turn_number": self.turn_number,
            "action_type": self.action_type,
            "action_description": self.action_description,
            "raw_similarity": self.raw_similarity,
            "fidelity_score": self.fidelity_score,
            "zone": self.zone.value,
            "intervention_level": self.intervention_level.value,
            "intervention_reason": self.intervention_reason,
            "approved": self.approved,
            "approval_source": self.approval_source,
            "sci_score": self.sci_score,
            "metadata": self.metadata,
        }


@dataclass
class ActionChainEntry:
    """
    Entry in the action chain for SCI (Semantic Continuity Index) tracking.

    Extends SCI from conversation turns to tool execution sequences.
    """
    index: int
    action_type: str                    # "tool_call", "delegation", "handoff"
    action_name: str                    # Tool name, agent name
    action_args: Dict[str, Any]
    embedding: np.ndarray
    fidelity_score: float
    timestamp: datetime = field(default_factory=datetime.now)
    parent_index: Optional[int] = None  # For branching action trees

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "action_type": self.action_type,
            "action_name": self.action_name,
            "action_args": self.action_args,
            "embedding": self.embedding.tolist() if isinstance(self.embedding, np.ndarray) else self.embedding,
            "fidelity_score": self.fidelity_score,
            "timestamp": self.timestamp.isoformat(),
            "parent_index": self.parent_index,
        }


# =============================================================================
# TELOS GOVERNED STATE
# =============================================================================

class TelosGovernedState(TypedDict):
    """
    LangGraph state schema with TELOS governance fields.

    This extends the standard LangGraph message-based state with:
    - Primacy Attractor (PA) for purpose tracking
    - Fidelity trajectory for drift detection
    - Action chain for SCI tracking
    - Governance trace for audit trails

    Usage:
        graph = StateGraph(TelosGovernedState)
        graph.add_node("telos_gate", telos_governance_node)
        graph.add_edge("agent", "telos_gate")
        graph.add_edge("telos_gate", "tools")
    """
    # Standard LangGraph message accumulation
    messages: Annotated[list, add_messages]

    # Primacy Attractor - the user's purpose
    primacy_attractor: Optional[Dict[str, Any]]  # Serialized PrimacyAttractor

    # Fidelity trajectory - history of fidelity scores
    fidelity_trajectory: List[Dict[str, Any]]

    # Action chain - for SCI tracking across tool calls
    action_chain: List[Dict[str, Any]]

    # Governance trace - full audit trail
    governance_trace: List[Dict[str, Any]]

    # Current state metrics
    current_fidelity: float
    current_zone: str  # FidelityZone value
    intervention_count: int
    turn_number: int

    # SCI state
    last_action_embedding: Optional[List[float]]
    accumulated_sci: float

    # Agent routing (for supervisor/swarm patterns)
    next_agent: Optional[str]
    delegation_approved: bool


def create_initial_state(
    primacy_attractor: Optional[PrimacyAttractor] = None,
) -> TelosGovernedState:
    """
    Create a fresh TELOS-governed state.

    Args:
        primacy_attractor: Optional PA to initialize with

    Returns:
        Initialized TelosGovernedState
    """
    return TelosGovernedState(
        messages=[],
        primacy_attractor=primacy_attractor.to_dict() if primacy_attractor else None,
        fidelity_trajectory=[],
        action_chain=[],
        governance_trace=[],
        current_fidelity=1.0,  # Start aligned
        current_zone=FidelityZone.GREEN.value,
        intervention_count=0,
        turn_number=0,
        last_action_embedding=None,
        accumulated_sci=1.0,
        next_agent=None,
        delegation_approved=True,
    )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_zone_from_fidelity(fidelity: float) -> FidelityZone:
    """Map fidelity score to zone."""
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
    Determine intervention level based on fidelity and raw similarity.

    Two-layer detection:
    - Layer 1: raw_similarity < SIMILARITY_BASELINE -> HARD_BLOCK
    - Layer 2: fidelity < thresholds -> graduated intervention
    """
    # Layer 1: Baseline check
    if raw_similarity < SIMILARITY_BASELINE:
        return InterventionLevel.HARD_BLOCK

    # Layer 2: Basin membership / zone-based
    if fidelity >= FIDELITY_GREEN:
        return InterventionLevel.NONE
    elif fidelity >= FIDELITY_YELLOW:
        return InterventionLevel.CONTEXT
    elif fidelity >= FIDELITY_ORANGE:
        return InterventionLevel.REDIRECT
    else:
        return InterventionLevel.BLOCK


def calculate_sci(
    current_embedding: np.ndarray,
    previous_embedding: Optional[np.ndarray],
    previous_fidelity: float,
) -> tuple[float, float]:
    """
    Calculate Semantic Continuity Index for action chains.

    Returns:
        (continuity_score, inherited_fidelity)
    """
    if previous_embedding is None:
        return 1.0, 1.0

    # Cosine similarity between consecutive actions
    continuity = np.dot(current_embedding, previous_embedding) / (
        np.linalg.norm(current_embedding) * np.linalg.norm(previous_embedding)
    )

    # If continuity is high enough, inherit fidelity with decay
    if continuity >= SCI_CONTINUITY_THRESHOLD:
        inherited_fidelity = previous_fidelity * SCI_DECAY_FACTOR
        return float(continuity), inherited_fidelity
    else:
        return float(continuity), 0.0  # No inheritance, use direct measurement
