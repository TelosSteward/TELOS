"""
Fidelity Calculator Module
==========================

Extracted from beta_response_manager.py for single-responsibility design.

This module handles all fidelity calculation logic:
- Raw cosine similarity computation
- Two-layer fidelity architecture (baseline + basin)
- Adaptive context integration
- Model-specific threshold management
"""

import logging
import numpy as np
import streamlit as st
from typing import Tuple, Dict, Optional

# Import thresholds from single source of truth
from telos_purpose.core.constants import (
    FIDELITY_GREEN, FIDELITY_YELLOW, FIDELITY_ORANGE, FIDELITY_RED,
    SIMILARITY_BASELINE,
    INTERVENTION_THRESHOLD,
    ST_FIDELITY_GREEN, ST_FIDELITY_YELLOW, ST_FIDELITY_ORANGE, ST_FIDELITY_RED,
)

# Import display normalization
from telos_purpose.core.fidelity_display import normalize_fidelity_for_display

logger = logging.getLogger(__name__)


class FidelityCalculator:
    """
    Calculates fidelity scores for user inputs relative to their Primacy Attractor.

    Supports two modes:
    - Template mode (SentenceTransformer): Uses ST-calibrated thresholds
    - Standard mode (Mistral): Uses Goldilocks zone thresholds
    """

    def __init__(
        self,
        use_rescaled_fidelity: bool = False,
        st_embedding_provider=None,
        st_user_pa_embedding=None,
        embedding_provider=None,
        user_pa_embedding=None,
        adaptive_context_manager=None,
        adaptive_context_enabled: bool = False,
    ):
        """
        Initialize the fidelity calculator.

        Args:
            use_rescaled_fidelity: Whether using template mode (ST thresholds)
            st_embedding_provider: SentenceTransformer provider for template mode
            st_user_pa_embedding: PA embedding in ST space
            embedding_provider: Mistral embedding provider
            user_pa_embedding: PA embedding in Mistral space
            adaptive_context_manager: Optional adaptive context system
            adaptive_context_enabled: Whether adaptive context is active
        """
        self.use_rescaled_fidelity = use_rescaled_fidelity
        self.st_embedding_provider = st_embedding_provider
        self.st_user_pa_embedding = st_user_pa_embedding
        self.embedding_provider = embedding_provider
        self.user_pa_embedding = user_pa_embedding
        self.adaptive_context_manager = adaptive_context_manager
        self.adaptive_context_enabled = adaptive_context_enabled

        # Cache last adaptive context result for UI display
        self.last_adaptive_context_result = None

    def get_thresholds(self) -> Dict[str, float]:
        """
        Get model-appropriate fidelity thresholds.

        Returns thresholds based on embedding model:
        - SentenceTransformer (template mode): Raw thresholds from ST calibration
        - Mistral (custom PA mode): Goldilocks zone thresholds

        Returns:
            Dict with 'green', 'yellow', 'orange', 'red' threshold values
        """
        if self.use_rescaled_fidelity:
            return {
                'green': ST_FIDELITY_GREEN,
                'yellow': ST_FIDELITY_YELLOW,
                'orange': ST_FIDELITY_ORANGE,
                'red': ST_FIDELITY_RED
            }
        else:
            return {
                'green': FIDELITY_GREEN,
                'yellow': FIDELITY_YELLOW,
                'orange': FIDELITY_ORANGE,
                'red': FIDELITY_RED
            }

    def get_fidelity_level(self, fidelity: float) -> str:
        """
        Get human-readable fidelity level using dynamic thresholds.

        Args:
            fidelity: Raw fidelity score

        Returns:
            One of: 'green', 'yellow', 'orange', 'red'
        """
        thresholds = self.get_thresholds()

        if fidelity >= thresholds['green']:
            return "green"
        elif fidelity >= thresholds['yellow']:
            return "yellow"
        elif fidelity >= thresholds['orange']:
            return "orange"
        else:
            return "red"

    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)

    def calculate_fidelity(
        self,
        user_input: str,
        conversation_history: list = None,
        use_context: bool = True
    ) -> Tuple[float, float, bool]:
        """
        Calculate fidelity of user input relative to their PA using TWO-LAYER architecture.

        FIDELITY PIPELINE:
        1. RAW SIMILARITY: cosine(user_embedding, PA_embedding)
        2. LAYER 1 CHECK: raw_similarity < SIMILARITY_BASELINE (0.20)?
        3. ADAPTIVE CONTEXT (if enabled): Apply message-type-aware boost
        4. Return adjusted fidelity for intervention decision

        Args:
            user_input: The user's message
            conversation_history: Prior conversation turns for context
            use_context: Whether to include conversation context

        Returns:
            tuple: (fidelity, raw_similarity, baseline_hard_block)
        """
        try:
            # ================================================================
            # ADAPTIVE CONTEXT SYSTEM: Phase-aware, pattern-classified context
            # ================================================================
            if (self.adaptive_context_enabled and
                self.adaptive_context_manager and
                self.st_user_pa_embedding is not None):
                try:
                    pa_embedding = self.st_user_pa_embedding

                    # Get input embedding
                    if self.st_embedding_provider:
                        input_embedding = np.array(self.st_embedding_provider.encode(user_input))
                    elif self.embedding_provider:
                        input_embedding = np.array(self.embedding_provider.encode(user_input))
                    else:
                        raise ValueError("No embedding provider available")

                    # Calculate raw fidelity first
                    raw_fidelity = self.cosine_similarity(input_embedding, pa_embedding)

                    # Process through adaptive context system
                    adaptive_result = self.adaptive_context_manager.process_message(
                        user_input=user_input,
                        input_embedding=input_embedding,
                        pa_embedding=pa_embedding,
                        raw_fidelity=raw_fidelity,
                        base_threshold=INTERVENTION_THRESHOLD
                    )

                    # Cache result for UI display
                    self.last_adaptive_context_result = adaptive_result

                    logger.info(f"🔄 ADAPTIVE CONTEXT: type={adaptive_result.message_type.name}, "
                               f"phase={adaptive_result.phase.name}, "
                               f"raw={raw_fidelity:.3f} -> adjusted={adaptive_result.adjusted_fidelity:.3f}")

                    fidelity = adaptive_result.adjusted_fidelity
                    raw_similarity = raw_fidelity
                    baseline_hard_block = adaptive_result.should_intervene and adaptive_result.drift_detected

                    return (fidelity, raw_similarity, baseline_hard_block)

                except Exception as e:
                    logger.warning(f"Adaptive context failed, falling back to legacy: {e}")

            # ================================================================
            # LEGACY CONTEXT-AWARE FIDELITY
            # ================================================================
            contextual_input = user_input
            if use_context and conversation_history:
                recent_context = self._extract_recent_context(conversation_history)
                if recent_context:
                    contextual_input = f"[Context: {recent_context}] | {user_input}"
                    logger.info(f"📚 Legacy context-aware fidelity: added {len(recent_context)} chars")

            # ================================================================
            # TEMPLATE MODE: SentenceTransformer + Raw Thresholds
            # ================================================================
            if (self.use_rescaled_fidelity and
                self.st_embedding_provider and
                self.st_user_pa_embedding is not None):

                user_embedding = np.array(self.st_embedding_provider.encode(contextual_input))
                raw_similarity = self.cosine_similarity(user_embedding, self.st_user_pa_embedding)

                # Clean Lane: Use raw similarity directly
                fidelity = raw_similarity
                baseline_hard_block = raw_similarity < ST_FIDELITY_RED

                if baseline_hard_block:
                    logger.warning(f"TEMPLATE MODE HARD_BLOCK: raw_sim={raw_similarity:.3f}")

                return (fidelity, raw_similarity, baseline_hard_block)

            # ================================================================
            # STANDARD MODE: Mistral or fallback to ST
            # ================================================================
            if (self.st_embedding_provider and
                self.st_user_pa_embedding is not None):
                # Fast path: use SentenceTransformer
                user_embedding = np.array(self.st_embedding_provider.encode(contextual_input))
                raw_similarity = self.cosine_similarity(user_embedding, self.st_user_pa_embedding)
                logger.info(f"FAST PATH: ST embedding, raw_sim={raw_similarity:.3f}")
            elif self.embedding_provider and self.user_pa_embedding is not None:
                # Slow path: use Mistral API
                user_embedding = np.array(self.embedding_provider.encode(contextual_input))
                raw_similarity = self.cosine_similarity(user_embedding, self.user_pa_embedding)
                logger.info(f"SLOW PATH: Mistral API, raw_sim={raw_similarity:.3f}")
            else:
                logger.warning("No embedding provider available - returning default")
                return (FIDELITY_GREEN, FIDELITY_GREEN, False)

            # Layer 1: Baseline Pre-Filter
            baseline_hard_block = raw_similarity < SIMILARITY_BASELINE

            if baseline_hard_block:
                logger.warning(f"LAYER 1 HARD_BLOCK: raw_sim={raw_similarity:.3f}")

            # Layer 2: Fidelity = raw cosine similarity
            fidelity = raw_similarity

            return (fidelity, raw_similarity, baseline_hard_block)

        except Exception as e:
            logger.error(f"Error calculating fidelity: {e}")
            return (FIDELITY_GREEN, FIDELITY_GREEN, False)

    def _extract_recent_context(self, conversation_history: list) -> str:
        """
        Extract recent context from conversation history for fidelity calculation.

        Args:
            conversation_history: List of conversation messages

        Returns:
            Context string for embedding
        """
        if not conversation_history:
            return ""

        # Get last 2 user messages only
        user_messages = [msg for msg in conversation_history if msg.get('role') == 'user']
        if not user_messages:
            return ""

        recent_user = user_messages[-2:]
        context_parts = [msg.get('content', '')[:100] for msg in recent_user if msg.get('content')]

        return " | ".join(context_parts)

    def get_display_fidelity(self, raw_fidelity: float) -> float:
        """
        Convert raw fidelity to display-normalized value.

        Args:
            raw_fidelity: Raw fidelity score

        Returns:
            Display-normalized fidelity (0.70+ = GREEN, etc.)
        """
        model_type = 'sentence_transformer' if self.use_rescaled_fidelity else 'mistral'
        return normalize_fidelity_for_display(raw_fidelity, model_type)


def create_fidelity_calculator_from_manager(manager) -> FidelityCalculator:
    """
    Factory function to create FidelityCalculator from BetaResponseManager.

    Args:
        manager: BetaResponseManager instance

    Returns:
        Configured FidelityCalculator
    """
    return FidelityCalculator(
        use_rescaled_fidelity=getattr(manager, 'use_rescaled_fidelity', False),
        st_embedding_provider=getattr(manager, 'st_embedding_provider', None),
        st_user_pa_embedding=getattr(manager, 'st_user_pa_embedding', None),
        embedding_provider=getattr(manager, 'embedding_provider', None),
        user_pa_embedding=getattr(manager, 'user_pa_embedding', None),
        adaptive_context_manager=getattr(manager, 'adaptive_context_manager', None),
        adaptive_context_enabled=getattr(manager, 'adaptive_context_enabled', False),
    )
