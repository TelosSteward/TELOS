"""
TELOS Adaptive Context System

Multi-tier, phase-aware, pattern-classified context management for
reducing false-positive drift detection in conversational AI.

Based on ADAPTIVE_CONTEXT_PROPOSAL.md (December 18, 2025)

Key features:
- Message type classification (DIRECT, FOLLOW_UP, CLARIFICATION, ANAPHORA)
- Conversation phase detection (EXPLORATION, FOCUS, DRIFT, RECOVERY)
- Multi-tier context buffer with weighted embeddings
- Adaptive threshold calculation with governance safeguards
"""

import re
import logging
import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from collections import deque
from datetime import datetime

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS - Imported from Single Source of Truth
# =============================================================================
from telos_purpose.core.constants import (
    FIDELITY_GREEN,
    INTERVENTION_THRESHOLD,
)

# Governance Safeguards (MUST NOT BE VIOLATED)
HARD_FLOOR = 0.20  # Never boost fidelity below this threshold
MAX_BOOST = 0.20   # Threshold can never be reduced by more than this

# =============================================================================
# SEMANTIC CONTINUITY INHERITANCE (SCI) - v4.0 (2025-12-30)
# =============================================================================
# A measurement-based approach to handling follow-up messages that replaces
# arbitrary syntax boosts with actual semantic similarity measurement.
#
# PHILOSOPHY: Instead of adding +0.20 to messages matching syntactic patterns
# like "yes", "sure", "that" - we MEASURE how semantically similar the current
# message is to the previous turn (user query + AI response combined).
#
# If high continuity is detected, we INHERIT the previous fidelity with decay.
# This maps to attractor physics: orbits that stay in the basin inherit the
# parent trajectory's stability rather than being re-measured from scratch.
#
# KEY INSIGHT: "Yes, show me an example" has low direct PA similarity (~0.32)
# but HIGH continuity with the previous turn about recursion. By measuring
# this continuity, we can inherit the previous 78% fidelity with decay → ~74%.

# Continuity thresholds (cosine similarity to previous turn)
SCI_STRONG_CONTINUITY = 0.70    # Clear semantic continuation
SCI_MODERATE_CONTINUITY = 0.50  # Moderate connection
SCI_WEAK_CONTINUITY = 0.30      # Minimal connection

# Decay factors (how much of previous fidelity is inherited)
# Strong continuity: nearly full inheritance (orbit remains stable)
# Moderate continuity: standard decay (orbit slightly perturbed)
# Weak continuity: noticeable decay (orbit significantly perturbed)
SCI_STRONG_DECAY = 0.99    # Nearly full inheritance
SCI_MODERATE_DECAY = 0.95  # Standard decay
SCI_WEAK_DECAY = 0.90      # Noticeable decay

# SCI feature flag - enables/disables the new approach
SCI_ENABLED = True


# =============================================================================
# SCI HELPER FUNCTIONS
# =============================================================================

def calculate_semantic_continuity(
    current_embedding: np.ndarray,
    previous_turn_embeddings: List[np.ndarray]
) -> Tuple[float, int]:
    """
    Calculate semantic continuity between current message and previous turn.

    Uses MAX similarity to the previous turn's embeddings (user query + AI response)
    rather than arbitrary pattern matching. This is measurement-based alignment
    detection that uses the same cosine math as PA fidelity.

    Args:
        current_embedding: Embedding of current user input (normalized)
        previous_turn_embeddings: List of embeddings from previous turn
                                  (user query embedding + AI response embedding)

    Returns:
        Tuple of (max_continuity_score, best_embedding_index)
    """
    if not previous_turn_embeddings or len(previous_turn_embeddings) == 0:
        return 0.0, -1

    # Normalize current embedding
    current_norm = current_embedding / np.linalg.norm(current_embedding)

    # Compute similarity to each previous turn embedding
    similarities = []
    for emb in previous_turn_embeddings:
        emb_norm = emb / np.linalg.norm(emb)
        sim = float(np.dot(current_norm, emb_norm))
        similarities.append(sim)

    # Return MAX similarity and its index
    max_sim = max(similarities)
    best_idx = similarities.index(max_sim)

    return max_sim, best_idx


def get_decay_factor(continuity_score: float) -> float:
    """
    Get the appropriate decay factor based on continuity strength.

    Higher continuity = less decay (more of previous fidelity inherited).
    Uses variable decay model as selected by user:
    - Strong (>0.70): 0.99 decay (nearly full inheritance)
    - Moderate (0.50-0.70): 0.95 decay (standard decay)
    - Weak (<0.50): 0.90 decay (noticeable decay)

    Args:
        continuity_score: Cosine similarity to previous turn (0.0-1.0)

    Returns:
        Decay factor (0.90-0.99)
    """
    if continuity_score >= SCI_STRONG_CONTINUITY:
        return SCI_STRONG_DECAY
    elif continuity_score >= SCI_MODERATE_CONTINUITY:
        return SCI_MODERATE_DECAY
    else:
        return SCI_WEAK_DECAY


def apply_continuity_inheritance(
    direct_fidelity: float,
    continuity_score: float,
    previous_fidelity: float
) -> Tuple[float, str]:
    """
    Apply Semantic Continuity Inheritance to compute adjusted fidelity.

    Uses max(direct, inherited) approach so semantically rich follow-ups
    that also have high direct PA similarity are never punished.

    The key insight is that "Yes, show me an example" should inherit the
    previous 78% fidelity because it's clearly continuing that topic,
    even though its direct PA similarity is only ~0.32.

    Args:
        direct_fidelity: Raw fidelity computed directly against PA
        continuity_score: Semantic similarity to previous turn (0.0-1.0)
        previous_fidelity: Fidelity score from the previous turn

    Returns:
        Tuple of (adjusted_fidelity, method_used)
    """
    # Below minimum continuity threshold = no inheritance
    if continuity_score < SCI_WEAK_CONTINUITY:
        logger.info(
            f"🔗 SCI: continuity={continuity_score:.3f} < threshold={SCI_WEAK_CONTINUITY:.2f}, "
            f"no inheritance, using direct={direct_fidelity:.3f}"
        )
        print(
            f"🔗 SCI: continuity={continuity_score:.3f} < threshold={SCI_WEAK_CONTINUITY:.2f}, "
            f"no inheritance, using direct={direct_fidelity:.3f}"
        )
        return direct_fidelity, "direct_only"

    # Get decay factor based on continuity strength
    decay = get_decay_factor(continuity_score)

    # Calculate inherited fidelity
    inherited_fidelity = previous_fidelity * decay

    # Use MAX of direct and inherited (never punish semantically rich follow-ups)
    if direct_fidelity >= inherited_fidelity:
        adjusted = direct_fidelity
        method = "direct_preferred"
        logger.info(
            f"🔗 SCI: continuity={continuity_score:.3f}, decay={decay:.2f}, "
            f"direct={direct_fidelity:.3f} >= inherited={inherited_fidelity:.3f}, "
            f"using direct"
        )
        print(
            f"🔗 SCI: continuity={continuity_score:.3f}, decay={decay:.2f}, "
            f"direct={direct_fidelity:.3f} >= inherited={inherited_fidelity:.3f}, "
            f"using direct"
        )
    else:
        adjusted = inherited_fidelity
        method = "inherited"
        logger.info(
            f"🔗 SCI: continuity={continuity_score:.3f}, decay={decay:.2f}, "
            f"inherited={inherited_fidelity:.3f} > direct={direct_fidelity:.3f}, "
            f"using inherited from previous={previous_fidelity:.3f}"
        )
        print(
            f"🔗 SCI: continuity={continuity_score:.3f}, decay={decay:.2f}, "
            f"inherited={inherited_fidelity:.3f} > direct={direct_fidelity:.3f}, "
            f"using inherited from previous={previous_fidelity:.3f}"
        )

    return adjusted, method


# Base intervention threshold - imported from constants.py
BASE_THRESHOLD = INTERVENTION_THRESHOLD  # 0.48

# Tier configuration - aligned with display zone thresholds
TIER1_THRESHOLD = FIDELITY_GREEN  # 0.70 - High fidelity (GREEN zone)
TIER2_THRESHOLD = 0.35            # Medium fidelity
TIER3_THRESHOLD = 0.25            # Low fidelity (minimum)

TIER1_CAPACITY = 5
TIER2_CAPACITY = 3
TIER3_CAPACITY = 2

TIER1_WEIGHT = 0.6
TIER2_WEIGHT = 0.3
TIER3_WEIGHT = 0.1

# Recency decay factor
RECENCY_DECAY = 0.8

# Phase detection window
PHASE_WINDOW_SIZE = 5


# =============================================================================
# MESSAGE TYPE CLASSIFICATION
# =============================================================================

class MessageType(Enum):
    """Classification of user message types with associated threshold adjustments."""
    DIRECT = "direct"           # New topic or direction
    FOLLOW_UP = "follow_up"     # Continues existing topic
    CLARIFICATION = "clarification"  # Asks about previous content
    ANAPHORA = "anaphora"       # References like "it", "this", "that"


# Base thresholds per message type (lower = more lenient)
MESSAGE_TYPE_THRESHOLDS = {
    MessageType.DIRECT: 0.70,        # New topics need high alignment
    MessageType.FOLLOW_UP: 0.35,     # Following up is more lenient
    MessageType.CLARIFICATION: 0.25, # Clarifications are very lenient
    MessageType.ANAPHORA: 0.25,      # Anaphoric references are lenient
}


# Regex patterns for message classification
MESSAGE_PATTERNS = {
    MessageType.CLARIFICATION: [
        r'^(what|how|why|can you|could you)\s+(did you mean|do you mean|does that mean)',
        r'^(i don\'?t understand|explain|clarify|what was)',
        r'^(say that again|repeat|what\?)',
        r'(what do you mean|elaborate on that|tell me more about what you said)',
    ],
    MessageType.ANAPHORA: [
        # Start-of-sentence anaphora (original patterns)
        r'^(it|this|that|these|those|they|them)\s+(is|are|was|were|has|have|should|could|would|will)',
        r'^(tell me more about (it|this|that))',
        r'^(expand on (it|this|that))',
        r'^(and|also|plus|furthermore|additionally)\s+',
        r'^(what about|how about)\s+(it|this|that|them)',
        # END-OF-SENTENCE anaphora - FIX for "prepare ourselves for that" pattern
        r'\b(for|about|on|to|with|of)\s+(that|this|it|them|those)\s*[?.!]?\s*$',
        # Anaphoric pronoun at end of sentence without preposition
        r'\b(that|this|it)\s*[?.!]?\s*$',
        # "do that", "understand that", "prepare for that" etc.
        r'\b(do|understand|know|prepare|ready|handle)\s+.{0,20}\b(that|this|it)\s*[?.!]?\s*$',
    ],
    MessageType.FOLLOW_UP: [
        r'^(ok|okay|alright|sure|yes|yeah|got it|i see|thanks)',
        r'^(now|next|then|after that|moving on)',
        r'^(can you also|also|additionally|furthermore)',
        r'^(and|but|so|however)',
        r'^(what else|anything else|is there more)',
        # Short continuation phrases (v4.1 - 2025-12-30)
        r'^(tell me more|go on|continue|keep going|more|please continue)',
        r'^(explain more|elaborate|more details|more info)',
    ],
}


def classify_message_type(
    message: str,
    recent_context: Optional[List[str]] = None,
    topic_continuity_score: Optional[float] = None
) -> MessageType:
    """
    Classify a user message by its conversational function.

    Args:
        message: The user's input text
        recent_context: Previous messages for context (optional)
        topic_continuity_score: Semantic similarity to recent messages (optional)

    Returns:
        MessageType classification
    """
    message_lower = message.lower().strip()

    # Check clarification patterns first (highest priority)
    for pattern in MESSAGE_PATTERNS[MessageType.CLARIFICATION]:
        if re.search(pattern, message_lower, re.IGNORECASE):
            logger.debug(f"Message classified as CLARIFICATION: {pattern}")
            return MessageType.CLARIFICATION

    # Check anaphora patterns
    for pattern in MESSAGE_PATTERNS[MessageType.ANAPHORA]:
        if re.search(pattern, message_lower, re.IGNORECASE):
            logger.debug(f"Message classified as ANAPHORA: {pattern}")
            return MessageType.ANAPHORA

    # Check follow-up patterns
    for pattern in MESSAGE_PATTERNS[MessageType.FOLLOW_UP]:
        if re.search(pattern, message_lower, re.IGNORECASE):
            logger.debug(f"Message classified as FOLLOW_UP: {pattern}")
            return MessageType.FOLLOW_UP

    # Use topic continuity if available
    if topic_continuity_score is not None and topic_continuity_score > 0.7:
        logger.debug(f"Message classified as FOLLOW_UP via topic continuity: {topic_continuity_score:.2f}")
        return MessageType.FOLLOW_UP

    # Default to direct
    return MessageType.DIRECT


# =============================================================================
# CONVERSATION PHASE DETECTION
# =============================================================================

class ConversationPhase(Enum):
    """Phases of a conversation affecting threshold behavior."""
    EXPLORATION = "exploration"   # Early turns, discovering purpose
    FOCUS = "focus"               # Established topic, stable fidelity
    DRIFT = "drift"               # Declining fidelity trend
    RECOVERY = "recovery"         # Rising fidelity after drift


# Phase-specific threshold modifiers (added to base threshold)
PHASE_MODIFIERS = {
    ConversationPhase.EXPLORATION: -0.10,  # More lenient during exploration
    ConversationPhase.FOCUS: 0.0,          # Standard threshold
    ConversationPhase.DRIFT: 0.10,         # Stricter during drift (no boost)
    ConversationPhase.RECOVERY: -0.05,     # Slightly lenient during recovery
}


@dataclass
class PhaseDetector:
    """
    Detects conversation phase based on fidelity trajectory.

    Uses a rolling window of fidelity scores to detect trends.
    """
    window_size: int = PHASE_WINDOW_SIZE
    fidelity_history: deque = field(default_factory=lambda: deque(maxlen=PHASE_WINDOW_SIZE))
    current_phase: ConversationPhase = ConversationPhase.EXPLORATION
    turns_in_phase: int = 0

    def update(self, fidelity: float) -> ConversationPhase:
        """
        Update phase detection with new fidelity score.

        Args:
            fidelity: Current turn's fidelity score

        Returns:
            Current detected phase
        """
        self.fidelity_history.append(fidelity)
        self.turns_in_phase += 1

        # Need enough history to detect trends
        if len(self.fidelity_history) < 3:
            return self.current_phase

        # Calculate trend
        recent = list(self.fidelity_history)
        avg_recent = np.mean(recent[-3:])
        avg_older = np.mean(recent[:-3]) if len(recent) > 3 else avg_recent
        trend = avg_recent - avg_older

        # Detect phase transitions
        new_phase = self._detect_phase(avg_recent, trend)

        if new_phase != self.current_phase:
            logger.debug(f"Phase transition: {self.current_phase.value} -> {new_phase.value}")
            self.current_phase = new_phase
            self.turns_in_phase = 0

        return self.current_phase

    def _detect_phase(self, avg_fidelity: float, trend: float) -> ConversationPhase:
        """Determine phase based on fidelity level and trend."""
        # Early conversation = exploration
        if len(self.fidelity_history) <= 3:
            return ConversationPhase.EXPLORATION

        # High stable fidelity = focus
        if avg_fidelity >= 0.65 and abs(trend) < 0.05:
            return ConversationPhase.FOCUS

        # Declining trend = drift
        if trend < -0.05 and avg_fidelity < 0.60:
            return ConversationPhase.DRIFT

        # Rising trend from low point = recovery
        if trend > 0.05 and self.current_phase == ConversationPhase.DRIFT:
            return ConversationPhase.RECOVERY

        # Default to focus if stable
        if abs(trend) < 0.05:
            return ConversationPhase.FOCUS

        return self.current_phase

    def reset(self):
        """Reset phase detector for new session."""
        self.fidelity_history.clear()
        self.current_phase = ConversationPhase.EXPLORATION
        self.turns_in_phase = 0


# =============================================================================
# MULTI-TIER CONTEXT BUFFER
# =============================================================================

@dataclass
class TieredMessage:
    """A message with its embedding and fidelity stored in a tier."""
    text: str
    embedding: np.ndarray
    fidelity_score: float
    timestamp: datetime
    message_type: MessageType
    tier: int  # 1, 2, or 3

    def __post_init__(self):
        if isinstance(self.embedding, list):
            self.embedding = np.array(self.embedding)


class MultiTierContextBuffer:
    """
    Three-tier context buffer for managing conversation history.

    Tier 1: High fidelity messages (>= 0.70) - 5 messages, weight 0.6
    Tier 2: Medium fidelity (0.35-0.70) - 3 messages, weight 0.3
    Tier 3: Low fidelity (0.25-0.35) - 2 messages, weight 0.1

    Messages below 0.25 are not stored (too off-topic).

    AI Response Buffer (2025-12-29):
    Also stores recent AI responses to capture full conversational context.
    When user says "walk me through the factorial example", they're referencing
    the AI's previous response, not their own query. Without this, context
    similarity would be low, causing false positive drift detection.
    """

    def __init__(self):
        self.tier1: deque = deque(maxlen=TIER1_CAPACITY)
        self.tier2: deque = deque(maxlen=TIER2_CAPACITY)
        self.tier3: deque = deque(maxlen=TIER3_CAPACITY)
        # AI response buffer (2025-12-29) - stores recent AI responses for context matching
        self.ai_response_buffer: deque = deque(maxlen=5)  # Store last 5 AI responses

        # =======================================================================
        # SCI: Previous Turn Tracking (2025-12-30)
        # =======================================================================
        # Stores the embeddings and fidelity from the previous turn to enable
        # Semantic Continuity Inheritance. When user says "Yes, show me example",
        # we measure similarity to this previous turn to decide on inheritance.
        self.previous_turn_embeddings: List[np.ndarray] = []  # User + AI embeddings
        self.previous_turn_fidelity: Optional[float] = None   # Fidelity of previous turn

    def add_message(
        self,
        text: str,
        embedding: np.ndarray,
        fidelity_score: float,
        message_type: MessageType = MessageType.DIRECT
    ) -> Optional[int]:
        """
        Add a message to the appropriate tier based on fidelity.

        Args:
            text: Message text
            embedding: Message embedding vector
            fidelity_score: Fidelity score for this message
            message_type: Classification of message type

        Returns:
            Tier number (1, 2, 3) or None if not stored
        """
        if fidelity_score < TIER3_THRESHOLD:
            logger.debug(f"Message below minimum threshold ({fidelity_score:.2f}), not storing")
            return None

        msg = TieredMessage(
            text=text,
            embedding=embedding,
            fidelity_score=fidelity_score,
            timestamp=datetime.now(),
            message_type=message_type,
            tier=self._get_tier_for_fidelity(fidelity_score)
        )

        if fidelity_score >= TIER1_THRESHOLD:
            self.tier1.append(msg)
            return 1
        elif fidelity_score >= TIER2_THRESHOLD:
            self.tier2.append(msg)
            return 2
        else:
            self.tier3.append(msg)
            return 3

    def _get_tier_for_fidelity(self, fidelity: float) -> int:
        """Determine tier based on fidelity score."""
        if fidelity >= TIER1_THRESHOLD:
            return 1
        elif fidelity >= TIER2_THRESHOLD:
            return 2
        else:
            return 3

    def get_tier_messages(self, tier: int) -> List[TieredMessage]:
        """Get all messages from a specific tier."""
        if tier == 1:
            return list(self.tier1)
        elif tier == 2:
            return list(self.tier2)
        elif tier == 3:
            return list(self.tier3)
        return []

    def get_all_messages(self) -> List[TieredMessage]:
        """Get all messages across all tiers, sorted by timestamp."""
        all_msgs = list(self.tier1) + list(self.tier2) + list(self.tier3)
        return sorted(all_msgs, key=lambda m: m.timestamp)

    def get_weighted_context_embedding(self) -> Optional[np.ndarray]:
        """
        Compute a weighted context embedding from all tiers.

        Uses tier weights and recency decay to compute a single
        embedding representing the conversation context.

        Returns:
            Weighted embedding vector, or None if no messages
        """
        all_msgs = self.get_all_messages()
        if not all_msgs:
            return None

        embeddings = []
        weights = []

        for i, msg in enumerate(all_msgs):
            # Recency weight (more recent = higher weight)
            recency_weight = RECENCY_DECAY ** (len(all_msgs) - 1 - i)

            # Tier weight
            if msg.tier == 1:
                tier_weight = TIER1_WEIGHT
            elif msg.tier == 2:
                tier_weight = TIER2_WEIGHT
            else:
                tier_weight = TIER3_WEIGHT

            # Combined weight
            combined_weight = recency_weight * tier_weight

            embeddings.append(msg.embedding)
            weights.append(combined_weight)

        # Normalize weights
        weights = np.array(weights)
        weights = weights / np.sum(weights)

        # Compute weighted average
        embeddings = np.array(embeddings)
        weighted_embedding = np.average(embeddings, axis=0, weights=weights)

        # Normalize
        norm = np.linalg.norm(weighted_embedding)
        if norm > 0:
            weighted_embedding = weighted_embedding / norm

        return weighted_embedding

    def get_recent_texts(self, n: int = 3) -> List[str]:
        """Get the n most recent message texts."""
        all_msgs = self.get_all_messages()
        return [m.text for m in all_msgs[-n:]]

    def get_all_embeddings(self) -> List[np.ndarray]:
        """
        Get all embeddings from all tiers as a list.

        Used by Context Attractor v3 to compute MAX similarity
        instead of centroid similarity.

        Returns:
            List of normalized embedding vectors
        """
        all_msgs = self.get_all_messages()
        return [msg.embedding for msg in all_msgs]

    def add_ai_response(self, text: str, embedding: np.ndarray) -> None:
        """
        Add an AI response to the AI response buffer (2025-12-29).

        This enables context matching against AI responses, fixing false
        positive drift detection when users reference AI content like
        "walk me through the factorial example".

        Args:
            text: AI response text (for debugging/logging)
            embedding: AI response embedding vector
        """
        # Normalize the embedding before storing
        if isinstance(embedding, list):
            embedding = np.array(embedding)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        self.ai_response_buffer.append({
            'text': text[:200],  # Truncate for logging only
            'embedding': embedding,
            'timestamp': datetime.now()
        })
        logger.debug(f"Added AI response to buffer (total: {len(self.ai_response_buffer)})")

    def get_ai_response_embeddings(self) -> List[np.ndarray]:
        """
        Get all AI response embeddings from the buffer.

        Returns:
            List of normalized AI response embedding vectors
        """
        return [resp['embedding'] for resp in self.ai_response_buffer]

    def get_all_context_embeddings(self) -> List[np.ndarray]:
        """
        Get all embeddings (user messages + AI responses) for MAX similarity.

        This is the key method for Context Attractor v3.1 - includes both
        user turns AND AI responses for comprehensive context matching.

        Returns:
            List of all normalized embedding vectors
        """
        user_embeddings = self.get_all_embeddings()
        ai_embeddings = self.get_ai_response_embeddings()
        return user_embeddings + ai_embeddings

    def clear(self):
        """Clear all tiers and AI response buffer."""
        self.tier1.clear()
        self.tier2.clear()
        self.tier3.clear()
        self.ai_response_buffer.clear()
        # Also clear SCI state
        self.previous_turn_embeddings = []
        self.previous_turn_fidelity = None

    def __len__(self) -> int:
        return len(self.tier1) + len(self.tier2) + len(self.tier3)

    # =========================================================================
    # SCI: Previous Turn Management (2025-12-30)
    # =========================================================================

    def set_previous_turn(
        self,
        user_embedding: np.ndarray,
        ai_embedding: Optional[np.ndarray],
        fidelity: float
    ) -> None:
        """
        Set the previous turn data for SCI.

        Called after each turn to store the embeddings and fidelity so that
        the next message can compute continuity and potentially inherit.

        Args:
            user_embedding: Embedding of the user's query for this turn
            ai_embedding: Embedding of the AI's response (optional)
            fidelity: Adjusted fidelity score for this turn
        """
        self.previous_turn_embeddings = []

        # Normalize and store user embedding
        if isinstance(user_embedding, list):
            user_embedding = np.array(user_embedding)
        user_norm = np.linalg.norm(user_embedding)
        if user_norm > 0:
            self.previous_turn_embeddings.append(user_embedding / user_norm)

        # Normalize and store AI embedding if provided
        if ai_embedding is not None:
            if isinstance(ai_embedding, list):
                ai_embedding = np.array(ai_embedding)
            ai_norm = np.linalg.norm(ai_embedding)
            if ai_norm > 0:
                self.previous_turn_embeddings.append(ai_embedding / ai_norm)

        self.previous_turn_fidelity = fidelity

        logger.debug(
            f"SCI: Set previous turn with {len(self.previous_turn_embeddings)} embeddings, "
            f"fidelity={fidelity:.3f}"
        )

    def get_previous_turn_data(self) -> Tuple[List[np.ndarray], Optional[float]]:
        """
        Get the previous turn data for SCI computation.

        Returns:
            Tuple of (list of embeddings, previous fidelity score)
        """
        return self.previous_turn_embeddings, self.previous_turn_fidelity

    def has_previous_turn(self) -> bool:
        """Check if we have previous turn data for SCI."""
        return len(self.previous_turn_embeddings) > 0 and self.previous_turn_fidelity is not None


# =============================================================================
# ADAPTIVE THRESHOLD CALCULATOR
# =============================================================================

class AdaptiveThresholdCalculator:
    """
    Calculates adaptive intervention thresholds based on:
    - Message type (DIRECT, FOLLOW_UP, CLARIFICATION, ANAPHORA)
    - Conversation phase (EXPLORATION, FOCUS, DRIFT, RECOVERY)
    - Governance safeguards (HARD_FLOOR, MAX_BOOST)
    """

    def calculate_threshold(
        self,
        message_type: MessageType,
        phase: ConversationPhase,
        base_threshold: float = BASE_THRESHOLD
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate the adaptive threshold for intervention.

        Args:
            message_type: Classification of user message
            phase: Current conversation phase
            base_threshold: Base intervention threshold

        Returns:
            Tuple of (adjusted_threshold, metadata_dict)
        """
        # Start with message type threshold
        type_threshold = MESSAGE_TYPE_THRESHOLDS.get(message_type, 0.70)

        # Apply phase modifier
        phase_modifier = PHASE_MODIFIERS.get(phase, 0.0)

        # Calculate boost (how much we're lowering from base)
        raw_boost = base_threshold - type_threshold

        # Apply governance: cap boost at MAX_BOOST
        capped_boost = min(raw_boost, MAX_BOOST)

        # DRIFT phase: no boost allowed (stricter)
        if phase == ConversationPhase.DRIFT:
            capped_boost = 0.0
            logger.debug("Drift phase detected - no threshold boost allowed")

        # Calculate adjusted threshold
        adjusted = base_threshold - capped_boost + phase_modifier

        # Apply governance: never go below HARD_FLOOR
        final_threshold = max(adjusted, HARD_FLOOR)

        # Build metadata
        metadata = {
            "message_type": message_type.value,
            "phase": phase.value,
            "base_threshold": base_threshold,
            "type_threshold": type_threshold,
            "phase_modifier": phase_modifier,
            "raw_boost": raw_boost,
            "capped_boost": capped_boost,
            "adjusted_threshold": adjusted,
            "final_threshold": final_threshold,
            "governance_applied": final_threshold != adjusted,
        }

        logger.debug(
            f"Adaptive threshold: {base_threshold:.2f} -> {final_threshold:.2f} "
            f"(type={message_type.value}, phase={phase.value}, boost={capped_boost:.2f})"
        )

        return final_threshold, metadata


# =============================================================================
# ADAPTIVE CONTEXT MANAGER (MAIN ORCHESTRATOR)
# =============================================================================

@dataclass
class AdaptiveContextResult:
    """Result from adaptive context processing."""
    raw_fidelity: float
    adjusted_fidelity: float
    message_type: MessageType
    conversation_phase: ConversationPhase
    adaptive_threshold: float
    should_intervene: bool
    context_embedding: Optional[np.ndarray]
    metadata: Dict[str, Any]

    @property
    def phase(self) -> ConversationPhase:
        """Alias for conversation_phase for backward compatibility."""
        return self.conversation_phase

    @property
    def drift_detected(self) -> bool:
        """Check if drift was detected (fidelity below threshold)."""
        return self.should_intervene and self.adjusted_fidelity < self.adaptive_threshold


class AdaptiveContextManager:
    """
    Main orchestrator for the Adaptive Context System.

    Integrates:
    - Message type classification
    - Conversation phase detection
    - Multi-tier context buffer
    - Adaptive threshold calculation

    Usage:
        manager = AdaptiveContextManager(embedding_provider)
        result = manager.process_message(
            user_input="tell me more about that",
            input_embedding=embedding,
            pa_embedding=pa_embedding,
            raw_fidelity=0.45
        )
        if result.should_intervene:
            # Handle intervention
    """

    def __init__(self, embedding_provider=None):
        """
        Initialize the adaptive context manager.

        Args:
            embedding_provider: Optional embedding provider for context computation
        """
        self.embedding_provider = embedding_provider
        self.context_buffer = MultiTierContextBuffer()
        self.phase_detector = PhaseDetector()
        self.threshold_calculator = AdaptiveThresholdCalculator()
        self.turn_count = 0

    def process_message(
        self,
        user_input: str,
        input_embedding: np.ndarray,
        pa_embedding: np.ndarray,
        raw_fidelity: float,
        base_threshold: float = BASE_THRESHOLD
    ) -> AdaptiveContextResult:
        """
        Process a user message through the adaptive context system.

        Args:
            user_input: The user's message text
            input_embedding: Embedding of the user's message
            pa_embedding: The Primacy Attractor embedding
            raw_fidelity: Raw fidelity score (similarity to PA)
            base_threshold: Base intervention threshold

        Returns:
            AdaptiveContextResult with all computed values
        """
        self.turn_count += 1

        # Step 1: Classify message type
        recent_texts = self.context_buffer.get_recent_texts(n=3)
        topic_continuity = self._compute_topic_continuity(input_embedding)
        message_type = classify_message_type(
            user_input,
            recent_context=recent_texts,
            topic_continuity_score=topic_continuity
        )

        # Step 2: Update phase detection
        phase = self.phase_detector.update(raw_fidelity)

        # Step 3: Calculate adaptive threshold
        adaptive_threshold, threshold_metadata = self.threshold_calculator.calculate_threshold(
            message_type=message_type,
            phase=phase,
            base_threshold=base_threshold
        )

        # Step 4: Compute contextualized fidelity
        context_embedding = self.context_buffer.get_weighted_context_embedding()
        # (2025-12-29) Use get_all_context_embeddings which includes AI responses
        # This fixes false positive drift when user references AI content like
        # "walk me through the factorial example" - the AI response contains "factorial"
        context_embeddings = self.context_buffer.get_all_context_embeddings()  # Includes AI responses
        adjusted_fidelity = self._compute_adjusted_fidelity(
            input_embedding=input_embedding,
            pa_embedding=pa_embedding,
            context_embedding=context_embedding,
            raw_fidelity=raw_fidelity,
            message_type=message_type,
            context_embeddings=context_embeddings  # Context Attractor v3.1: MAX similarity with AI responses
        )

        # Step 5: Determine if intervention should occur
        should_intervene = adjusted_fidelity < adaptive_threshold

        # Step 6: Store message in buffer (if above minimum)
        tier_stored = self.context_buffer.add_message(
            text=user_input,
            embedding=input_embedding,
            fidelity_score=raw_fidelity,
            message_type=message_type
        )

        # Build comprehensive metadata
        metadata = {
            **threshold_metadata,
            "turn_number": self.turn_count,
            "raw_fidelity": raw_fidelity,
            "adjusted_fidelity": adjusted_fidelity,
            "topic_continuity": topic_continuity,
            "tier_stored": tier_stored,
            "buffer_size": len(self.context_buffer),
            "phase_turns": self.phase_detector.turns_in_phase,
        }

        logger.info(
            f"Turn {self.turn_count}: type={message_type.value}, phase={phase.value}, "
            f"raw={raw_fidelity:.2f}, adjusted={adjusted_fidelity:.2f}, "
            f"threshold={adaptive_threshold:.2f}, intervene={should_intervene}"
        )

        return AdaptiveContextResult(
            raw_fidelity=raw_fidelity,
            adjusted_fidelity=adjusted_fidelity,
            message_type=message_type,
            conversation_phase=phase,
            adaptive_threshold=adaptive_threshold,
            should_intervene=should_intervene,
            context_embedding=context_embedding,
            metadata=metadata
        )

    def _compute_topic_continuity(self, current_embedding: np.ndarray) -> Optional[float]:
        """Compute semantic similarity to recent context."""
        context_embedding = self.context_buffer.get_weighted_context_embedding()
        if context_embedding is None:
            return None

        # Normalize embeddings
        current_norm = current_embedding / np.linalg.norm(current_embedding)
        context_norm = context_embedding / np.linalg.norm(context_embedding)

        # Cosine similarity
        return float(np.dot(current_norm, context_norm))

    def _compute_adjusted_fidelity(
        self,
        input_embedding: np.ndarray,
        pa_embedding: np.ndarray,
        context_embedding: Optional[np.ndarray],
        raw_fidelity: float,
        message_type: MessageType,
        context_embeddings: Optional[List[np.ndarray]] = None
    ) -> float:
        """
        Compute adjusted fidelity using CONTEXT AS THIRD ATTRACTOR.

        ARCHITECTURE (v3 - MAX Similarity - 2025-12-26):
        =================================================
        The Context Attractor uses MAX similarity to any prior turn, not centroid.

        KEY INSIGHT: Centroid averages out the signal. When a query like
        "prepare ourselves for that" is semantically similar to ONE specific
        prior turn (0.492) but less similar to others, the centroid (0.381)
        dilutes this signal, producing negligible boosts (+0.007).

        Using MAX similarity captures the strongest contextual connection:
        - MAX similarity to any turn: 0.492 → meaningful boost
        - Centroid similarity: 0.381 → negligible boost

        DUAL ATTRACTOR MODEL:
        - Attractor 1: Primacy Attractor (PA) - user's declared purpose
        - Attractor 2: Context Attractor - MAX similarity to any high-fidelity turn

        Args:
            input_embedding: Embedding of current user input
            pa_embedding: The Primacy Attractor embedding
            context_embedding: Weighted centroid (legacy, used as fallback)
            raw_fidelity: Raw similarity between input and PA
            message_type: Classification of the message (used for logging only)
            context_embeddings: List of all context turn embeddings for MAX similarity

        Returns:
            Adjusted fidelity with context attractor applied
        """
        # Normalize input embedding once
        input_norm = input_embedding / np.linalg.norm(input_embedding)

        # =======================================================================
        # CONTEXT ATTRACTOR v3: Use MAX similarity to any stored turn
        # =======================================================================
        # This captures the strongest contextual connection rather than averaging.

        context_similarity = 0.0
        best_turn_idx = -1

        if context_embeddings and len(context_embeddings) > 0:
            # Compute MAX similarity to any stored turn
            similarities = []
            for i, turn_emb in enumerate(context_embeddings):
                turn_norm = turn_emb / np.linalg.norm(turn_emb)
                sim = float(np.dot(input_norm, turn_norm))
                similarities.append(sim)

            context_similarity = max(similarities)
            best_turn_idx = similarities.index(context_similarity)

            # DEBUG: Print to stdout for visibility (logger.info may not appear)
            print(f"🔎 CONTEXT BUFFER: {len(context_embeddings)} embeddings, sims={[f'{s:.3f}' for s in similarities]}, MAX={context_similarity:.3f} (idx={best_turn_idx})")
            logger.info(f"🔎 CONTEXT BUFFER: {len(context_embeddings)} embeddings, sims={[f'{s:.3f}' for s in similarities]}, MAX={context_similarity:.3f} (idx={best_turn_idx})")

        elif context_embedding is not None:
            # Fallback to centroid if no individual embeddings available
            context_norm = context_embedding / np.linalg.norm(context_embedding)
            context_similarity = float(np.dot(input_norm, context_norm))

        else:
            # No context available - use raw fidelity
            return raw_fidelity

        # =======================================================================
        # CONTEXT BOOST CALCULATION (v3.2 - Type-Aware Thresholds + Multipliers)
        # =======================================================================
        # Uses message-type-aware thresholds because short follow-ups like
        # "Yes, show me an example" have low cosine similarity to prior turns
        # even though they're clearly conversational continuations.

        # Maximum boost from context (governance safeguard)
        MAX_CONTEXT_BOOST = 0.30

        # Base weight for context similarity in boost calculation
        BASE_CONTEXT_WEIGHT = 0.50

        # Message type multipliers - ANAPHORA/CLARIFICATION get stronger boosts
        # because they inherently depend on prior context for meaning
        MESSAGE_TYPE_MULTIPLIERS = {
            MessageType.ANAPHORA: 1.5,       # "that", "it", "this" - strongest pull
            MessageType.CLARIFICATION: 1.4,  # Questions about prior content
            MessageType.FOLLOW_UP: 1.0,      # Baseline - continuing topic
            MessageType.DIRECT: 0.7,         # New statements - weakest pull
        }

        # TYPE-AWARE THRESHOLDS (v3.2): Lower threshold for messages that
        # syntactically indicate conversational continuity. Short follow-ups
        # like "Yes, show me an example" may have low semantic similarity
        # to prior context but are clearly part of the conversation.
        MESSAGE_TYPE_THRESHOLDS = {
            MessageType.ANAPHORA: 0.15,      # Very low - depends entirely on context
            MessageType.CLARIFICATION: 0.20, # Low - asking about prior content
            MessageType.FOLLOW_UP: 0.20,     # Low - syntactically a continuation
            MessageType.DIRECT: 0.35,        # Standard - new topic statements
        }

        type_multiplier = MESSAGE_TYPE_MULTIPLIERS.get(message_type, 1.0)
        context_threshold = MESSAGE_TYPE_THRESHOLDS.get(message_type, 0.35)

        # =======================================================================
        # SEMANTIC CONTINUITY INHERITANCE (v4.0 - 2025-12-30)
        # =======================================================================
        # Instead of arbitrary syntax-based boosts, we MEASURE semantic continuity
        # to the previous turn (user query + AI response) and INHERIT fidelity
        # with decay if high continuity is detected.
        #
        # This is measurement-based, not pattern-based. A message like
        # "Yes, show me an example" will have high continuity with the previous
        # turn about recursion (~0.65+), allowing inheritance of the 78% fidelity.
        #
        # The philosophy: orbits that stay in the basin inherit the parent
        # trajectory's stability rather than being re-measured from scratch.

        # Check if SCI is enabled and we have previous turn data
        has_prev = self.context_buffer.has_previous_turn()
        print(f"🔗 SCI GATE CHECK: SCI_ENABLED={SCI_ENABLED}, has_previous_turn={has_prev}")
        logger.info(f"🔗 SCI GATE CHECK: SCI_ENABLED={SCI_ENABLED}, has_previous_turn={has_prev}")

        if SCI_ENABLED and has_prev:
            prev_embeddings, prev_fidelity = self.context_buffer.get_previous_turn_data()

            if prev_embeddings and prev_fidelity is not None:
                # =================================================================
                # SHORT PHRASE OVERRIDE (v4.1 - 2025-12-30)
                # =================================================================
                # Short generic phrases like "Tell me more", "Yes", "Go on" have
                # inherently LOW cosine similarity to specific domain content.
                # The SCI continuity threshold (0.30) will never be reached.
                #
                # SOLUTION: If message type is FOLLOW_UP or ANAPHORA (syntactic
                # indicators of continuation), AND previous turn had high fidelity,
                # force inherit with moderate decay - bypass the continuity check.
                #
                # This preserves measurement-based philosophy for substantive
                # messages while handling the edge case of short generic phrases.
                SHORT_PHRASE_TYPES = {MessageType.FOLLOW_UP, MessageType.ANAPHORA}
                SHORT_PHRASE_MIN_PREV_FIDELITY = 0.50  # Only inherit from stable orbits
                SHORT_PHRASE_DECAY = 0.95  # Moderate decay for pattern-based inheritance

                if message_type in SHORT_PHRASE_TYPES and prev_fidelity >= SHORT_PHRASE_MIN_PREV_FIDELITY:
                    # Short generic phrase following a high-fidelity turn
                    inherited = prev_fidelity * SHORT_PHRASE_DECAY
                    print(f"🔗 SHORT PHRASE OVERRIDE: type={message_type.name}, "
                          f"prev_fidelity={prev_fidelity:.3f} -> inherited={inherited:.3f} "
                          f"(bypassing continuity check)")
                    logger.info(
                        f"🔗 SHORT PHRASE OVERRIDE: type={message_type.name}, "
                        f"prev_fidelity={prev_fidelity:.3f} -> inherited={inherited:.3f} "
                        f"(bypassing continuity check)"
                    )
                    # Return inherited fidelity if it's better than raw
                    if inherited > raw_fidelity:
                        return min(inherited, 1.0)

                # Calculate semantic continuity to previous turn
                continuity_score, best_idx = calculate_semantic_continuity(
                    input_embedding, prev_embeddings
                )

                print(f"🔗 SCI CHECK: continuity={continuity_score:.3f} to prev_turn (idx={best_idx}), "
                      f"prev_fidelity={prev_fidelity:.3f}, raw={raw_fidelity:.3f}")
                logger.info(
                    f"🔗 SCI CHECK: continuity={continuity_score:.3f} to prev_turn (idx={best_idx}), "
                    f"prev_fidelity={prev_fidelity:.3f}, raw={raw_fidelity:.3f}"
                )

                # Apply continuity inheritance
                inherited_result, method = apply_continuity_inheritance(
                    raw_fidelity, continuity_score, prev_fidelity
                )

                # If SCI produced a result, use max(SCI result, context attractor result)
                # This ensures both mechanisms work together, never against each other
                if method != "direct_only":
                    # SCI inheritance triggered - but also compute context attractor boost
                    # and use whichever is higher (belt AND suspenders)
                    print(f"🔗 SCI INHERITANCE: method={method}, result={inherited_result:.3f}")
                    logger.info(f"🔗 SCI INHERITANCE: method={method}, result={inherited_result:.3f}")

                    # Store SCI result for later comparison
                    sci_adjusted = inherited_result

        # If SCI didn't trigger or didn't produce inheritance, fall back to context threshold check
        if context_similarity < context_threshold:
            # Below semantic threshold for context attractor
            # Check if SCI produced an inheritance result we should use
            if SCI_ENABLED and 'sci_adjusted' in dir() and sci_adjusted > raw_fidelity:
                # SCI inheritance provides better fidelity than raw
                adjusted = sci_adjusted
                adjusted = min(adjusted, 1.0)  # Cap at 100%
                print(f"🔗 SCI FALLBACK: context_sim={context_similarity:.3f} < threshold={context_threshold:.2f}, "
                      f"using SCI result={adjusted:.3f} instead of raw={raw_fidelity:.3f}")
                logger.info(
                    f"🔗 SCI FALLBACK: context_sim={context_similarity:.3f} < threshold={context_threshold:.2f}, "
                    f"using SCI result={adjusted:.3f} instead of raw={raw_fidelity:.3f}"
                )
                return adjusted
            else:
                # No SCI inheritance available - no boost
                print(f"❌ CONTEXT BOOST SKIPPED: type={message_type.name}, max_sim={context_similarity:.3f} < threshold={context_threshold:.2f}, returning raw_fidelity={raw_fidelity:.3f}")
                logger.info(
                    f"❌ CONTEXT BOOST SKIPPED: type={message_type.name}, max_sim={context_similarity:.3f} < threshold={context_threshold:.2f}, returning raw_fidelity={raw_fidelity:.3f}"
                )
                return raw_fidelity

        # Calculate boost proportional to similarity above threshold
        excess_similarity = context_similarity - context_threshold
        max_excess = 1.0 - context_threshold
        normalized_excess = excess_similarity / max_excess

        # Calculate boost: proportional to normalized excess × type multiplier
        # ANAPHORA gets 1.5x boost, DIRECT gets 0.7x boost
        semantic_boost = normalized_excess * MAX_CONTEXT_BOOST * BASE_CONTEXT_WEIGHT * type_multiplier

        # Apply context attractor boost
        context_attractor_result = raw_fidelity + semantic_boost

        # Governance: Cap at MAX_CONTEXT_BOOST
        if context_attractor_result - raw_fidelity > MAX_CONTEXT_BOOST:
            context_attractor_result = raw_fidelity + MAX_CONTEXT_BOOST

        # Cap at 1.0 (fidelity can't exceed 100%)
        context_attractor_result = min(context_attractor_result, 1.0)

        # SCI Integration: Use MAX of context attractor and SCI inheritance
        # This ensures both mechanisms work together, never against each other
        if SCI_ENABLED and 'sci_adjusted' in dir() and sci_adjusted > 0:
            adjusted = max(context_attractor_result, sci_adjusted)
            if adjusted == sci_adjusted:
                print(f"🔗 SCI+CONTEXT: SCI wins! context={context_attractor_result:.3f}, sci={sci_adjusted:.3f}")
                logger.info(f"🔗 SCI+CONTEXT: SCI wins! context={context_attractor_result:.3f}, sci={sci_adjusted:.3f}")
            else:
                print(f"🧲 SCI+CONTEXT: Context attractor wins! context={context_attractor_result:.3f}, sci={sci_adjusted:.3f}")
                logger.info(f"🧲 SCI+CONTEXT: Context attractor wins! context={context_attractor_result:.3f}, sci={sci_adjusted:.3f}")
        else:
            adjusted = context_attractor_result

        logger.info(
            f"🧲 CONTEXT ATTRACTOR (v4.0-SCI): type={message_type.name}, mult={type_multiplier:.1f}, "
            f"max_sim={context_similarity:.3f} (turn {best_turn_idx}), "
            f"raw={raw_fidelity:.3f} → adjusted={adjusted:.3f}, boost={adjusted - raw_fidelity:+.3f}"
        )

        return adjusted

    def reset(self):
        """Reset the context manager for a new session."""
        self.context_buffer.clear()
        self.phase_detector.reset()
        self.turn_count = 0
        logger.info("AdaptiveContextManager reset for new session")


# =============================================================================
# CACHING UTILITY
# =============================================================================

def get_cached_adaptive_context_manager(embedding_provider=None) -> AdaptiveContextManager:
    """
    Get a cached AdaptiveContextManager instance.

    Uses Streamlit's cache_resource for singleton pattern.

    Args:
        embedding_provider: Optional embedding provider

    Returns:
        Cached AdaptiveContextManager instance
    """
    try:
        import streamlit as st

        @st.cache_resource
        def _get_manager() -> AdaptiveContextManager:
            logger.info("Creating cached AdaptiveContextManager")
            return AdaptiveContextManager(embedding_provider=embedding_provider)

        return _get_manager()
    except ImportError:
        # Fallback if Streamlit not available
        logger.warning("Streamlit not available, creating new AdaptiveContextManager")
        return AdaptiveContextManager(embedding_provider=embedding_provider)
