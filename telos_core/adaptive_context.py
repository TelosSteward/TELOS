"""
TELOS Adaptive Context System
=============================

DMAIC-inspired context management extended for agentic AI governance.

Original (Conversational):
    - Multi-tier context buffer (fidelity-based)
    - Phase detection (EXPLORATION, FOCUS, DRIFT, RECOVERY)
    - Adaptive threshold adjustment
    - Message type classification

Extended (Agentic AI) - NEW:
    - Action phase detection for tool sequences
    - Action chain buffer (not just messages)
    - Handoff context inheritance
    - Multi-agent state tracking

This is the most sophisticated component, factored from
TELOS Observatory V3 and extended for LangGraph integration.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
import numpy as np
import logging

from .constants import (
    # Tier thresholds
    TIER1_THRESHOLD,
    TIER2_THRESHOLD,
    TIER3_THRESHOLD,
    TIER1_CAPACITY,
    TIER2_CAPACITY,
    TIER3_CAPACITY,

    # Governance safeguards
    HARD_FLOOR,
    MAX_BOOST,
    BASE_THRESHOLD,

    # Context weighting
    RECENCY_DECAY,
    PHASE_WINDOW_SIZE,

    # Message type
    MESSAGE_TYPE_THRESHOLDS,
    MESSAGE_TYPE_BOOSTS,

    # Fidelity zones
    FIDELITY_GREEN,
)

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class MessageType(Enum):
    """Message type classification for context."""
    DIRECT = "direct"           # Clear, on-topic statements
    FOLLOW_UP = "follow_up"     # Continuations, elaborations
    CLARIFICATION = "clarification"  # Questions about prior content
    ANAPHORA = "anaphora"       # Pronoun references ("it", "that")


class ConversationPhase(Enum):
    """Conversational phase for context."""
    EXPLORATION = "exploration"  # Initial topic exploration
    FOCUS = "focus"             # Focused discussion
    DRIFT = "drift"             # Drifting from purpose
    RECOVERY = "recovery"       # Returning to purpose


class ActionPhase(Enum):
    """
    Action phase for agentic AI context. (NEW)

    Extends ConversationPhase for tool execution sequences.
    """
    PLANNING = "planning"       # Agent planning actions
    EXECUTING = "executing"     # Executing tools
    VALIDATING = "validating"   # Validating results
    HANDOFF = "handoff"         # Handing off to another agent
    BLOCKED = "blocked"         # Action blocked by governance
    COMPLETE = "complete"       # Action sequence complete


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class TieredMessage:
    """Message with tier classification for buffer."""
    content: str
    embedding: np.ndarray
    fidelity: float
    tier: int
    timestamp: str
    message_type: MessageType


@dataclass
class TieredAction:
    """
    Action with tier classification for agentic buffer. (NEW)
    """
    action_text: str
    tool_name: str
    embedding: np.ndarray
    fidelity: float
    tier: int
    timestamp: str
    action_phase: ActionPhase
    agent_name: Optional[str] = None
    approved: bool = True


@dataclass
class ContextResult:
    """Result of context processing."""
    message_type: MessageType
    phase: ConversationPhase
    tier: int
    adjusted_threshold: float
    context_embedding: Optional[np.ndarray]
    boost_applied: float
    explanation: str

    def to_dict(self) -> dict:
        return {
            "message_type": self.message_type.value,
            "phase": self.phase.value,
            "tier": self.tier,
            "adjusted_threshold": self.adjusted_threshold,
            "boost_applied": self.boost_applied,
            "explanation": self.explanation,
        }


@dataclass
class ActionContextResult:
    """
    Result of action context processing. (NEW)
    """
    action_phase: ActionPhase
    tier: int
    adjusted_threshold: float
    inherited_fidelity: float
    context_embedding: Optional[np.ndarray]
    explanation: str

    def to_dict(self) -> dict:
        return {
            "action_phase": self.action_phase.value,
            "tier": self.tier,
            "adjusted_threshold": self.adjusted_threshold,
            "inherited_fidelity": self.inherited_fidelity,
            "explanation": self.explanation,
        }


# =============================================================================
# MESSAGE TYPE CLASSIFICATION
# =============================================================================

def classify_message_type(text: str, fidelity: float) -> MessageType:
    """
    Classify message type based on content and fidelity.

    Uses linguistic patterns and fidelity to determine type.
    """
    text_lower = text.lower().strip()

    # Check for anaphora (pronouns referencing prior context)
    anaphora_patterns = [
        "it ", "that ", "this ", "these ", "those ",
        "tell me more", "what about", "and also",
    ]
    if any(text_lower.startswith(p) or f" {p}" in text_lower for p in anaphora_patterns):
        if fidelity < MESSAGE_TYPE_THRESHOLDS["ANAPHORA"]:
            return MessageType.ANAPHORA

    # Check for clarification (questions about prior content)
    clarification_patterns = [
        "what do you mean", "can you explain", "?",
        "how does", "why does", "could you clarify",
    ]
    if any(p in text_lower for p in clarification_patterns):
        if fidelity < MESSAGE_TYPE_THRESHOLDS["CLARIFICATION"]:
            return MessageType.CLARIFICATION

    # Check for follow-up
    follow_up_patterns = [
        "also", "additionally", "furthermore", "another",
        "what else", "continue", "go on",
    ]
    if any(p in text_lower for p in follow_up_patterns):
        if fidelity < MESSAGE_TYPE_THRESHOLDS["FOLLOW_UP"]:
            return MessageType.FOLLOW_UP

    # Default to direct
    return MessageType.DIRECT


def get_message_type_boost(message_type: MessageType) -> float:
    """Get context boost factor for message type."""
    boost_map = {
        MessageType.DIRECT: MESSAGE_TYPE_BOOSTS["DIRECT"],
        MessageType.FOLLOW_UP: MESSAGE_TYPE_BOOSTS["FOLLOW_UP"],
        MessageType.CLARIFICATION: MESSAGE_TYPE_BOOSTS["CLARIFICATION"],
        MessageType.ANAPHORA: MESSAGE_TYPE_BOOSTS["ANAPHORA"],
    }
    return boost_map.get(message_type, 1.0)


# =============================================================================
# TIERED BUFFER
# =============================================================================

class TieredBuffer:
    """
    Multi-tier context buffer based on fidelity.

    Tier 1: High fidelity messages (>= 0.70)
    Tier 2: Medium fidelity (0.35-0.70)
    Tier 3: Low fidelity (0.25-0.35)

    Each tier has limited capacity (most recent wins).
    """

    def __init__(
        self,
        tier1_capacity: int = TIER1_CAPACITY,
        tier2_capacity: int = TIER2_CAPACITY,
        tier3_capacity: int = TIER3_CAPACITY,
    ):
        self.capacities = {1: tier1_capacity, 2: tier2_capacity, 3: tier3_capacity}
        self.buffers: Dict[int, List[TieredMessage]] = {1: [], 2: [], 3: []}

    def add(self, message: TieredMessage):
        """Add message to appropriate tier."""
        tier = message.tier
        if tier not in self.buffers:
            tier = 3  # Default to lowest tier

        self.buffers[tier].append(message)

        # Enforce capacity
        if len(self.buffers[tier]) > self.capacities[tier]:
            self.buffers[tier] = self.buffers[tier][-self.capacities[tier]:]

    def get_tier(self, tier: int) -> List[TieredMessage]:
        """Get messages from specific tier."""
        return self.buffers.get(tier, [])

    def get_all_embeddings(self) -> List[np.ndarray]:
        """Get all embeddings from all tiers."""
        embeddings = []
        for tier in [1, 2, 3]:
            for msg in self.buffers[tier]:
                embeddings.append(msg.embedding)
        return embeddings

    def get_high_fidelity_embeddings(self) -> List[np.ndarray]:
        """Get embeddings from tier 1 (high fidelity)."""
        return [msg.embedding for msg in self.buffers[1]]

    def clear(self):
        """Clear all buffers."""
        for tier in self.buffers:
            self.buffers[tier] = []

    def classify_to_tier(self, fidelity: float) -> int:
        """Classify fidelity to tier."""
        if fidelity >= TIER1_THRESHOLD:
            return 1
        elif fidelity >= TIER2_THRESHOLD:
            return 2
        elif fidelity >= TIER3_THRESHOLD:
            return 3
        else:
            return 3  # Below threshold still goes to tier 3


# =============================================================================
# ACTION BUFFER (NEW FOR AGENTIC AI)
# =============================================================================

class ActionBuffer:
    """
    Multi-tier action buffer for agentic AI context. (NEW)

    Similar to TieredBuffer but for tool calls and agent actions.
    """

    def __init__(
        self,
        tier1_capacity: int = TIER1_CAPACITY,
        tier2_capacity: int = TIER2_CAPACITY,
        tier3_capacity: int = TIER3_CAPACITY,
    ):
        self.capacities = {1: tier1_capacity, 2: tier2_capacity, 3: tier3_capacity}
        self.buffers: Dict[int, List[TieredAction]] = {1: [], 2: [], 3: []}

    def add(self, action: TieredAction):
        """Add action to appropriate tier."""
        tier = action.tier
        if tier not in self.buffers:
            tier = 3

        self.buffers[tier].append(action)

        if len(self.buffers[tier]) > self.capacities[tier]:
            self.buffers[tier] = self.buffers[tier][-self.capacities[tier]:]

    def get_recent_embeddings(self, n: int = 5) -> List[np.ndarray]:
        """Get n most recent embeddings across all tiers."""
        all_actions = []
        for tier in [1, 2, 3]:
            all_actions.extend(self.buffers[tier])

        # Sort by timestamp and get recent
        all_actions.sort(key=lambda x: x.timestamp, reverse=True)
        return [a.embedding for a in all_actions[:n]]

    def get_high_fidelity_actions(self) -> List[TieredAction]:
        """Get high-fidelity actions."""
        return self.buffers[1]

    def clear(self):
        """Clear all buffers."""
        for tier in self.buffers:
            self.buffers[tier] = []


# =============================================================================
# PHASE DETECTOR
# =============================================================================

class PhaseDetector:
    """
    Detect conversation phase based on fidelity trajectory.

    Uses rolling window to classify:
    - EXPLORATION: Initial phase, variable fidelity
    - FOCUS: Stable high fidelity
    - DRIFT: Declining fidelity trend
    - RECOVERY: Increasing fidelity after drift
    """

    def __init__(self, window_size: int = PHASE_WINDOW_SIZE):
        self.window_size = window_size
        self.history: List[float] = []
        self.current_phase = ConversationPhase.EXPLORATION

    def update(self, fidelity: float) -> ConversationPhase:
        """Update with new fidelity and return phase."""
        self.history.append(fidelity)

        # Keep only window
        if len(self.history) > self.window_size * 2:
            self.history = self.history[-self.window_size * 2:]

        if len(self.history) < self.window_size:
            self.current_phase = ConversationPhase.EXPLORATION
            return self.current_phase

        window = self.history[-self.window_size:]
        avg = sum(window) / len(window)

        # Calculate trend
        if len(self.history) >= self.window_size * 2:
            prev_window = self.history[-self.window_size * 2:-self.window_size]
            prev_avg = sum(prev_window) / len(prev_window)
            trend = avg - prev_avg
        else:
            trend = 0.0

        # Determine phase
        if avg >= FIDELITY_GREEN:
            if trend >= 0:
                self.current_phase = ConversationPhase.FOCUS
            else:
                self.current_phase = ConversationPhase.DRIFT
        else:
            if trend > 0.05:
                self.current_phase = ConversationPhase.RECOVERY
            elif trend < -0.05:
                self.current_phase = ConversationPhase.DRIFT
            else:
                self.current_phase = ConversationPhase.EXPLORATION

        return self.current_phase

    def reset(self):
        """Reset phase detector."""
        self.history.clear()
        self.current_phase = ConversationPhase.EXPLORATION


# =============================================================================
# ACTION PHASE DETECTOR (NEW FOR AGENTIC AI)
# =============================================================================

class ActionPhaseDetector:
    """
    Detect action phase for agentic AI sequences. (NEW)

    Tracks:
    - PLANNING: Agent is deciding what to do
    - EXECUTING: Tools being called
    - VALIDATING: Checking results
    - HANDOFF: Transferring to another agent
    - BLOCKED: Governance blocked action
    - COMPLETE: Sequence finished
    """

    def __init__(self):
        self.current_phase = ActionPhase.PLANNING
        self.action_count = 0
        self.blocked_count = 0

    def update(
        self,
        action_type: str,
        approved: bool,
        is_handoff: bool = False,
    ) -> ActionPhase:
        """Update with action and return phase."""
        self.action_count += 1

        if not approved:
            self.blocked_count += 1
            if self.blocked_count >= 3:
                self.current_phase = ActionPhase.BLOCKED
                return self.current_phase

        if is_handoff:
            self.current_phase = ActionPhase.HANDOFF
        elif action_type == "tool_call":
            self.current_phase = ActionPhase.EXECUTING
        elif action_type == "validation":
            self.current_phase = ActionPhase.VALIDATING
        elif action_type == "planning":
            self.current_phase = ActionPhase.PLANNING
        elif action_type == "complete":
            self.current_phase = ActionPhase.COMPLETE

        return self.current_phase

    def reset(self):
        """Reset detector."""
        self.current_phase = ActionPhase.PLANNING
        self.action_count = 0
        self.blocked_count = 0


# =============================================================================
# ADAPTIVE THRESHOLD CALCULATOR
# =============================================================================

class AdaptiveThresholdCalculator:
    """
    Calculate adaptive intervention threshold based on context.

    Adjusts base threshold with governance safeguards:
    - HARD_FLOOR: Never go below 0.20
    - MAX_BOOST: Never increase more than 0.20
    - Phase-based adjustments
    """

    def __init__(self, base_threshold: float = BASE_THRESHOLD):
        self.base_threshold = base_threshold

    def calculate(
        self,
        phase: ConversationPhase,
        message_type: MessageType,
        recent_fidelities: List[float],
    ) -> Tuple[float, str]:
        """
        Calculate adjusted threshold.

        Returns (threshold, explanation).
        """
        threshold = self.base_threshold
        adjustments = []

        # Phase adjustment
        phase_adj = self._phase_adjustment(phase)
        if phase_adj != 0:
            threshold += phase_adj
            adjustments.append(f"Phase {phase.value}: {phase_adj:+.2f}")

        # Message type adjustment
        type_adj = self._message_type_adjustment(message_type)
        if type_adj != 0:
            threshold += type_adj
            adjustments.append(f"Type {message_type.value}: {type_adj:+.2f}")

        # Recent trend adjustment
        trend_adj = self._trend_adjustment(recent_fidelities)
        if trend_adj != 0:
            threshold += trend_adj
            adjustments.append(f"Trend: {trend_adj:+.2f}")

        # Apply governance safeguards
        original = threshold
        threshold = max(HARD_FLOOR, threshold)
        threshold = min(self.base_threshold + MAX_BOOST, threshold)

        if threshold != original:
            adjustments.append(f"Safeguard clamped: {original:.2f} -> {threshold:.2f}")

        explanation = "; ".join(adjustments) if adjustments else "No adjustments"
        return threshold, explanation

    def _phase_adjustment(self, phase: ConversationPhase) -> float:
        """Get phase-based adjustment."""
        adjustments = {
            ConversationPhase.EXPLORATION: 0.0,
            ConversationPhase.FOCUS: -0.05,  # More lenient when focused
            ConversationPhase.DRIFT: 0.10,   # Stricter when drifting
            ConversationPhase.RECOVERY: 0.05,  # Slightly stricter during recovery
        }
        return adjustments.get(phase, 0.0)

    def _message_type_adjustment(self, message_type: MessageType) -> float:
        """Get message type adjustment."""
        adjustments = {
            MessageType.DIRECT: 0.0,
            MessageType.FOLLOW_UP: -0.05,  # More lenient for follow-ups
            MessageType.CLARIFICATION: -0.05,
            MessageType.ANAPHORA: -0.10,  # Most lenient for anaphora
        }
        return adjustments.get(message_type, 0.0)

    def _trend_adjustment(self, fidelities: List[float]) -> float:
        """Calculate adjustment based on recent trend."""
        if len(fidelities) < 3:
            return 0.0

        recent = fidelities[-3:]
        trend = recent[-1] - recent[0]

        if trend > 0.1:
            return -0.05  # Improving, be more lenient
        elif trend < -0.1:
            return 0.05  # Declining, be stricter

        return 0.0


# =============================================================================
# MAIN CONTEXT MANAGERS
# =============================================================================

class AdaptiveContextManager:
    """
    Unified adaptive context manager for conversational governance.

    Combines:
    - TieredBuffer for context storage
    - PhaseDetector for phase tracking
    - AdaptiveThresholdCalculator for threshold adjustment
    - Message type classification
    """

    def __init__(
        self,
        base_threshold: float = BASE_THRESHOLD,
        window_size: int = PHASE_WINDOW_SIZE,
    ):
        self.buffer = TieredBuffer()
        self.phase_detector = PhaseDetector(window_size)
        self.threshold_calculator = AdaptiveThresholdCalculator(base_threshold)
        self.fidelity_history: List[float] = []

    def process_message(
        self,
        text: str,
        embedding: np.ndarray,
        fidelity_score: float,
    ) -> ContextResult:
        """
        Process a message through the adaptive context system.

        Returns complete context analysis.
        """
        # Classify message type
        message_type = classify_message_type(text, fidelity_score)

        # Update phase
        phase = self.phase_detector.update(fidelity_score)

        # Classify tier
        tier = self.buffer.classify_to_tier(fidelity_score)

        # Calculate adjusted threshold
        self.fidelity_history.append(fidelity_score)
        threshold, explanation = self.threshold_calculator.calculate(
            phase, message_type, self.fidelity_history
        )

        # Get context embedding (weighted average of high-fidelity)
        context_embedding = self._compute_context_embedding()

        # Get boost factor
        boost = get_message_type_boost(message_type)

        # Add to buffer
        tiered_msg = TieredMessage(
            content=text,
            embedding=embedding,
            fidelity=fidelity_score,
            tier=tier,
            timestamp=datetime.now().isoformat(),
            message_type=message_type,
        )
        self.buffer.add(tiered_msg)

        return ContextResult(
            message_type=message_type,
            phase=phase,
            tier=tier,
            adjusted_threshold=threshold,
            context_embedding=context_embedding,
            boost_applied=boost,
            explanation=explanation,
        )

    def _compute_context_embedding(self) -> Optional[np.ndarray]:
        """Compute weighted context embedding from high-fidelity history."""
        embeddings = self.buffer.get_high_fidelity_embeddings()
        if not embeddings:
            return None

        # Recency-weighted average
        n = len(embeddings)
        weights = [RECENCY_DECAY ** (n - i - 1) for i in range(n)]
        weight_sum = sum(weights)

        if weight_sum < 1e-10:
            return None

        weighted = sum(w * e for w, e in zip(weights, embeddings))
        return weighted / weight_sum

    def reset(self):
        """Reset all context state."""
        self.buffer.clear()
        self.phase_detector.reset()
        self.fidelity_history.clear()


class AgenticContextManager:
    """
    Adaptive context manager for agentic AI governance. (NEW)

    Extends AdaptiveContextManager for tool calls and handoffs.
    """

    def __init__(
        self,
        base_threshold: float = BASE_THRESHOLD,
    ):
        self.action_buffer = ActionBuffer()
        self.action_phase_detector = ActionPhaseDetector()
        self.threshold_calculator = AdaptiveThresholdCalculator(base_threshold)
        self.fidelity_history: List[float] = []

    def process_action(
        self,
        action_text: str,
        tool_name: str,
        embedding: np.ndarray,
        fidelity_score: float,
        approved: bool = True,
        is_handoff: bool = False,
        agent_name: Optional[str] = None,
    ) -> ActionContextResult:
        """
        Process an action through the agentic context system.
        """
        # Update action phase
        action_type = "handoff" if is_handoff else "tool_call"
        phase = self.action_phase_detector.update(action_type, approved, is_handoff)

        # Classify tier
        tier = 1 if fidelity_score >= TIER1_THRESHOLD else (
            2 if fidelity_score >= TIER2_THRESHOLD else 3
        )

        # Update history
        self.fidelity_history.append(fidelity_score)

        # Calculate threshold (using EXPLORATION phase as default for actions)
        threshold, explanation = self.threshold_calculator.calculate(
            ConversationPhase.EXPLORATION,
            MessageType.DIRECT,
            self.fidelity_history,
        )

        # Compute inherited fidelity from action chain
        inherited = self._compute_inherited_fidelity(fidelity_score)

        # Context embedding from recent actions
        context_embedding = self._compute_action_context()

        # Add to buffer
        tiered_action = TieredAction(
            action_text=action_text,
            tool_name=tool_name,
            embedding=embedding,
            fidelity=fidelity_score,
            tier=tier,
            timestamp=datetime.now().isoformat(),
            action_phase=phase,
            agent_name=agent_name,
            approved=approved,
        )
        self.action_buffer.add(tiered_action)

        return ActionContextResult(
            action_phase=phase,
            tier=tier,
            adjusted_threshold=threshold,
            inherited_fidelity=inherited,
            context_embedding=context_embedding,
            explanation=explanation,
        )

    def _compute_inherited_fidelity(self, direct_fidelity: float) -> float:
        """Compute inherited fidelity from high-fidelity actions."""
        high_fidelity_actions = self.action_buffer.get_high_fidelity_actions()
        if not high_fidelity_actions:
            return direct_fidelity

        # Average of high-fidelity actions with decay
        fidelities = [a.fidelity for a in high_fidelity_actions]
        n = len(fidelities)
        weights = [RECENCY_DECAY ** (n - i - 1) for i in range(n)]
        weight_sum = sum(weights)

        if weight_sum < 1e-10:
            return direct_fidelity

        inherited = sum(w * f for w, f in zip(weights, fidelities)) / weight_sum

        # Return max of direct and inherited
        return max(direct_fidelity, inherited * 0.9)  # 10% decay on inheritance

    def _compute_action_context(self) -> Optional[np.ndarray]:
        """Compute context embedding from recent actions."""
        embeddings = self.action_buffer.get_recent_embeddings(5)
        if not embeddings:
            return None

        n = len(embeddings)
        weights = [RECENCY_DECAY ** (n - i - 1) for i in range(n)]
        weight_sum = sum(weights)

        if weight_sum < 1e-10:
            return None

        weighted = sum(w * e for w, e in zip(weights, embeddings))
        return weighted / weight_sum

    def reset(self):
        """Reset all context state."""
        self.action_buffer.clear()
        self.action_phase_detector.reset()
        self.fidelity_history.clear()
