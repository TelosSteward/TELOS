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
# CONSTANTS - Governance Safeguards (MUST NOT BE VIOLATED)
# =============================================================================

# Hard floor - never boost fidelity below this threshold
HARD_FLOOR = 0.20

# Maximum boost - threshold can never be reduced by more than this
MAX_BOOST = 0.20

# Base intervention threshold (from constants.py)
BASE_THRESHOLD = 0.48

# Tier configuration
TIER1_THRESHOLD = 0.70  # High fidelity
TIER2_THRESHOLD = 0.35  # Medium fidelity
TIER3_THRESHOLD = 0.25  # Low fidelity (minimum)

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
        r'^(it|this|that|these|those|they|them)\s+(is|are|was|were|has|have|should|could|would|will)',
        r'^(tell me more about (it|this|that))',
        r'^(expand on (it|this|that))',
        r'^(and|also|plus|furthermore|additionally)\s+',
        r'^(what about|how about)\s+(it|this|that|them)',
    ],
    MessageType.FOLLOW_UP: [
        r'^(ok|okay|alright|sure|yes|yeah|got it|i see|thanks)',
        r'^(now|next|then|after that|moving on)',
        r'^(can you also|also|additionally|furthermore)',
        r'^(and|but|so|however)',
        r'^(what else|anything else|is there more)',
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
    """

    def __init__(self):
        self.tier1: deque = deque(maxlen=TIER1_CAPACITY)
        self.tier2: deque = deque(maxlen=TIER2_CAPACITY)
        self.tier3: deque = deque(maxlen=TIER3_CAPACITY)

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

    def clear(self):
        """Clear all tiers."""
        self.tier1.clear()
        self.tier2.clear()
        self.tier3.clear()

    def __len__(self) -> int:
        return len(self.tier1) + len(self.tier2) + len(self.tier3)


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
        adjusted_fidelity = self._compute_adjusted_fidelity(
            input_embedding=input_embedding,
            pa_embedding=pa_embedding,
            context_embedding=context_embedding,
            raw_fidelity=raw_fidelity,
            message_type=message_type
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
        message_type: MessageType
    ) -> float:
        """
        Compute adjusted fidelity using context as an ATTRACTOR.

        The Adaptive Context system functions as a mathematical attractor:
        - Context embedding represents the "basin center" from recent high-fidelity messages
        - When a query is semantically similar to this context, it gets a fidelity boost
        - This prevents false positives where contextually relevant queries (e.g., "EU AI Act"
          in a TELOS governance session) are incorrectly classified as drift

        ATTRACTOR MECHANISM:
        - If input is similar to high-fidelity context, boost raw fidelity toward context
        - The boost is proportional to how similar the input is to the context
        - Even DIRECT messages benefit from context if they're semantically related

        Args:
            input_embedding: Embedding of current user input
            pa_embedding: The Primacy Attractor embedding
            context_embedding: Weighted embedding from high-fidelity conversation context
            raw_fidelity: Raw similarity between input and PA
            message_type: Classification of the message

        Returns:
            Adjusted fidelity with context attractor applied
        """
        # No context available - use raw fidelity
        if context_embedding is None:
            return raw_fidelity

        # Compute similarity to context (how related is this query to the conversation?)
        input_norm = input_embedding / np.linalg.norm(input_embedding)
        context_norm = context_embedding / np.linalg.norm(context_embedding)
        context_similarity = float(np.dot(input_norm, context_norm))

        # =======================================================================
        # ATTRACTOR MECHANISM: Context acts as a gravitational pull on fidelity
        # =======================================================================
        # If the query is similar to high-fidelity context (context_similarity > 0.5),
        # it should receive a boost toward the context's fidelity level.
        #
        # Formula: adjusted = raw + (context_boost * context_similarity * attractor_strength)
        #
        # The attractor strength varies by message type:
        # - ANAPHORA: Strong pull (it/this/that references depend on context)
        # - CLARIFICATION: Strong pull (asking about prior content)
        # - FOLLOW_UP: Moderate pull (continuing the topic)
        # - DIRECT: Light pull (new statement, but still benefits from context)

        # Base attractor strengths by message type
        attractor_strength = {
            MessageType.ANAPHORA: 0.50,       # Strong context pull
            MessageType.CLARIFICATION: 0.45, # Strong context pull
            MessageType.FOLLOW_UP: 0.35,     # Moderate context pull
            MessageType.DIRECT: 0.25,        # Light context pull (KEY FIX: was 0.0)
        }.get(message_type, 0.20)

        # Context boost: how much above baseline is the context?
        # If context similarity is high (>0.5), this indicates the query is related
        # to the high-fidelity conversation, so we should boost toward context
        CONTEXT_RELEVANCE_THRESHOLD = 0.4  # Minimum similarity to consider context relevant

        if context_similarity < CONTEXT_RELEVANCE_THRESHOLD:
            # Query is not related to context - no boost
            logger.debug(
                f"Context not relevant: context_sim={context_similarity:.2f} < threshold={CONTEXT_RELEVANCE_THRESHOLD}"
            )
            return raw_fidelity

        # Calculate context-based fidelity boost
        # The boost is proportional to both context similarity and attractor strength
        # We use the context buffer's weighted average fidelity as a proxy for "where
        # the conversation is" in terms of alignment
        context_fidelity_proxy = context_similarity  # Simplified: high context sim = high alignment

        # Calculate adjusted fidelity using attractor formula
        # The adjusted value pulls raw_fidelity toward context_fidelity_proxy
        pull_strength = attractor_strength * context_similarity  # Stronger pull when more similar
        adjusted = raw_fidelity + pull_strength * (context_fidelity_proxy - raw_fidelity)

        # Governance: Cap the boost to prevent over-correction
        MAX_BOOST = 0.25  # Maximum fidelity increase from context
        boost_applied = adjusted - raw_fidelity
        if boost_applied > MAX_BOOST:
            adjusted = raw_fidelity + MAX_BOOST

        logger.info(
            f"🧲 CONTEXT ATTRACTOR: type={message_type.name}, "
            f"raw={raw_fidelity:.3f}, context_sim={context_similarity:.3f}, "
            f"attractor_strength={attractor_strength:.2f}, adjusted={adjusted:.3f}, "
            f"boost={adjusted - raw_fidelity:+.3f}"
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
