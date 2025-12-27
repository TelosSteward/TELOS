"""
Response Generator Module
=========================

Extracted from beta_response_manager.py for single-responsibility design.

This module handles all response generation logic:
- Native LLM response generation (GREEN zone)
- Redirect response generation (intervention zones)
- System prompt building
- Proportional control prompt construction
"""

import logging
import numpy as np
import streamlit as st
from typing import Dict, Optional

# Import thresholds from single source of truth
from telos_purpose.core.constants import FIDELITY_GREEN

# Import Steward styling
from config.steward_styles import (
    get_steward_style,
    get_intervention_prompt,
    get_response_opener,
    get_style_interpolation
)

# Import Semantic Interpreter
from telos_purpose.core.semantic_interpreter import interpret as semantic_interpret

logger = logging.getLogger(__name__)

# Response length constraints by zone
MAX_TOKENS_GREEN = 600
MAX_TOKENS_YELLOW = 500
MAX_TOKENS_ORANGE = 400
MAX_TOKENS_RED = 300

# Proportional controller gain
K_ATTRACTOR = 1.5


class ResponseGenerator:
    """
    Generates LLM responses with appropriate governance styling.

    Handles both native (GREEN zone) and governed (intervention) responses.
    """

    def __init__(
        self,
        llm_client=None,
        embedding_provider=None,
        st_embedding_provider=None,
        st_user_pa_embedding=None,
        use_rescaled_fidelity: bool = False,
    ):
        """
        Initialize the response generator.

        Args:
            llm_client: LLM client for generation
            embedding_provider: Embedding provider for fidelity calculation
            st_embedding_provider: SentenceTransformer provider
            st_user_pa_embedding: PA embedding in ST space
            use_rescaled_fidelity: Whether using template mode
        """
        self.llm_client = llm_client
        self.embedding_provider = embedding_provider
        self.st_embedding_provider = st_embedding_provider
        self.st_user_pa_embedding = st_user_pa_embedding
        self.use_rescaled_fidelity = use_rescaled_fidelity

    def _get_llm_client(self):
        """Get or initialize the LLM client."""
        if self.llm_client is None:
            from telos_purpose.llm_clients.mistral_client import get_cached_mistral_client
            self.llm_client = get_cached_mistral_client()
        return self.llm_client

    def generate_native_response(
        self,
        user_input: str,
        conversation_history: list = None,
        pa_data: Dict = None
    ) -> str:
        """
        Generate native LLM response without intervention but WITH PA-aware system prompt.

        Args:
            user_input: User's message
            conversation_history: Prior conversation
            pa_data: Primacy Attractor data for context

        Returns:
            Native response string
        """
        try:
            client = self._get_llm_client()

            # Build PA-aware system prompt
            system_prompt = self._build_system_prompt(pa_data)

            # Start with system prompt
            conversation = [{'role': 'system', 'content': system_prompt}]

            # Add conversation history
            if conversation_history:
                conversation.extend(conversation_history)
            conversation.append({'role': 'user', 'content': user_input})

            response = client.generate(
                messages=conversation,
                max_tokens=MAX_TOKENS_GREEN,
                temperature=0.7
            )

            return response

        except Exception as e:
            logger.error(f"Error generating native response: {e}")
            return "I understand you're testing the system. How can I help you explore TELOS governance?"

    def generate_redirect_response(
        self,
        user_input: str,
        user_fidelity: float,
        zone: str,
        conversation_history: list = None,
        pa_data: Dict = None
    ) -> Dict:
        """
        Generate a proportionally-governed redirect response for drifted topics.

        Args:
            user_input: The user's message
            user_fidelity: Pre-calculated fidelity score
            zone: Fidelity zone (YELLOW, ORANGE, RED)
            conversation_history: Prior conversation
            pa_data: Primacy Attractor data

        Returns:
            Dict with redirect response and intervention metrics
        """
        try:
            client = self._get_llm_client()

            # Get purpose from PA
            pa_data = pa_data or st.session_state.get('primacy_attractor', {})
            purpose_raw = pa_data.get('purpose', 'your stated goals')
            purpose = ' '.join(purpose_raw) if isinstance(purpose_raw, list) else purpose_raw

            # Proportional control calculations
            error_signal = 1.0 - user_fidelity
            intervention_strength = min(K_ATTRACTOR * error_signal, 1.0)

            logger.info(f"📐 Proportional Control: fidelity={user_fidelity:.3f}, "
                       f"error={error_signal:.3f}, strength={intervention_strength:.3f}")

            # Build proportionally-governed redirect prompt
            redirect_prompt = self._build_redirect_prompt(
                fidelity=user_fidelity,
                strength=intervention_strength,
                purpose=purpose,
                zone=zone
            )

            # Build conversation with history
            conversation = [{'role': 'system', 'content': redirect_prompt}]
            if conversation_history:
                conversation.extend(conversation_history)
            conversation.append({'role': 'user', 'content': user_input})

            # Get zone-appropriate max tokens
            max_tokens = self._get_max_tokens_for_zone(zone)

            response = client.generate(
                messages=conversation,
                max_tokens=max_tokens,
                temperature=0.7
            )

            logger.info(f"📝 Redirect response generated: {len(response)} chars")

            # Compute AI Fidelity if possible
            ai_fidelity = None
            primacy_state = None
            if self.st_embedding_provider and self.st_user_pa_embedding is not None:
                ai_fidelity, primacy_state = self._compute_ai_fidelity(
                    response, user_fidelity
                )

            return {
                'response': response,
                'fidelity_score': user_fidelity,
                'distance_from_pa': error_signal,
                'intervention_triggered': True,
                'intervention_type': 'proportional_redirect',
                'intervention_reason': f'Strength {intervention_strength:.2f} redirect ({zone} zone)',
                'drift_detected': True,
                'in_basin': False,
                'user_pa_fidelity': user_fidelity,
                'ai_pa_fidelity': ai_fidelity,
                'primacy_state_score': primacy_state,
                'redirect_zone': zone,
                'intervention_strength': intervention_strength,
                'error_signal': error_signal,
                'max_tokens_used': max_tokens,
            }

        except Exception as e:
            logger.error(f"Error generating redirect response: {e}")
            return {
                'response': "Let's refocus on what you came here to accomplish.",
                'fidelity_score': user_fidelity,
                'error': str(e),
                'user_pa_fidelity': user_fidelity,
            }

    def _build_system_prompt(self, pa_data: Dict = None) -> str:
        """
        Build a PA-aware system prompt.

        Args:
            pa_data: Primacy Attractor data

        Returns:
            System prompt string
        """
        pa_data = pa_data or st.session_state.get('primacy_attractor', {})

        purpose_raw = pa_data.get('purpose', 'General assistance')
        scope_raw = pa_data.get('scope', 'Open discussion')
        purpose = ' '.join(purpose_raw) if isinstance(purpose_raw, list) else purpose_raw
        scope = ' '.join(scope_raw) if isinstance(scope_raw, list) else scope_raw
        boundaries = pa_data.get('boundaries', [])
        style = pa_data.get('style', '')

        boundaries_text = ""
        if boundaries:
            if isinstance(boundaries, list):
                boundaries_text = "; ".join(boundaries)
            else:
                boundaries_text = str(boundaries)

        return f"""You are a helpful AI assistant. The user has established the following context for this conversation:

PURPOSE: {purpose}
SCOPE: {scope}
{f'BOUNDARIES: {boundaries_text}' if boundaries_text else ''}
{f'STYLE: {style}' if style else ''}

RESPONSE GUIDELINES:
- Be conversational and natural
- CRITICAL: Keep responses brief and focused - aim for 2-3 short paragraphs maximum
- Address what the user asked directly without over-explaining
- Stay aligned with the stated purpose and scope
- Do NOT use lists, bullet points, numbered lists, or headers unless explicitly asked
- Avoid verbose explanations, exhaustive coverage, or unnecessary elaboration
- Prefer concise answers over comprehensive guides"""

    def _build_redirect_prompt(
        self,
        fidelity: float,
        strength: float,
        purpose: str,
        zone: str = None
    ) -> str:
        """
        Build a proportionally-governed redirect prompt using the Semantic Interpreter.

        Args:
            fidelity: Raw fidelity score
            strength: Calculated intervention strength
            purpose: User's established purpose
            zone: Fidelity zone for logging

        Returns:
            System prompt with concrete linguistic specifications
        """
        # Get concrete linguistic specs from SemanticInterpreter
        spec = semantic_interpret(fidelity, purpose)

        # Get therapeutic Steward prompt
        steward_prompt = get_intervention_prompt(
            fidelity=fidelity,
            user_context=purpose,
            green_threshold=FIDELITY_GREEN
        )

        # Get the linguistic specification block
        linguistic_spec = spec.to_prompt_block(purpose)

        return f"""{steward_prompt}

LINGUISTIC GUIDELINES:
{linguistic_spec}

CRITICAL INSTRUCTIONS:
- Never use stock phrases like "far from your stated purpose" or "here's the path back"
- Be natural and conversational - sound like a human, not a governance system
- Keep responses concise (2-3 sentences unless more detail is genuinely needed)
- Use the linguistic form specified above ({spec.sentence_form})"""

    def _get_max_tokens_for_zone(self, zone: str) -> int:
        """Get max tokens for a fidelity zone."""
        token_map = {
            'YELLOW': MAX_TOKENS_YELLOW,
            'ORANGE': MAX_TOKENS_ORANGE,
            'RED': MAX_TOKENS_RED,
        }
        return token_map.get(zone, MAX_TOKENS_ORANGE)

    def _compute_ai_fidelity(
        self,
        response: str,
        user_fidelity: float
    ) -> tuple:
        """
        Compute AI fidelity for a response.

        Args:
            response: Generated response text
            user_fidelity: User's fidelity score

        Returns:
            tuple: (ai_fidelity, primacy_state)
        """
        try:
            from telos_purpose.core.embedding_provider import rescale_sentence_transformer_fidelity

            response_embedding = np.array(self.st_embedding_provider.encode(response))
            raw_ai_fidelity = self._cosine_similarity(response_embedding, self.st_user_pa_embedding)
            ai_fidelity = rescale_sentence_transformer_fidelity(raw_ai_fidelity)

            # Calculate Primacy State
            epsilon = 1e-10
            primacy_state = (2 * user_fidelity * ai_fidelity) / (user_fidelity + ai_fidelity + epsilon)

            logger.info(f"🔧 AI Fidelity: raw={raw_ai_fidelity:.3f} → rescaled={ai_fidelity:.3f}, PS={primacy_state:.3f}")

            return ai_fidelity, primacy_state

        except Exception as e:
            logger.warning(f"Could not compute AI fidelity: {e}")
            return None, None

    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)

    def get_steward_styling(
        self,
        fidelity: float,
        green_threshold: float = None
    ) -> Dict:
        """
        Get Steward styling information for a fidelity level.

        Args:
            fidelity: Current fidelity score
            green_threshold: GREEN threshold (defaults to FIDELITY_GREEN)

        Returns:
            Dict with styling information
        """
        green_threshold = green_threshold or FIDELITY_GREEN

        steward_style = get_steward_style(fidelity, green_threshold=green_threshold)
        steward_interpolation = get_style_interpolation(fidelity, green_threshold=green_threshold)

        return {
            'band': steward_style.band,
            'band_name': steward_style.band_name,
            'tone': steward_style.tone,
            'directness': steward_interpolation['directness'],
            'urgency': steward_interpolation['urgency'],
            'opener': get_response_opener(fidelity, green_threshold=green_threshold),
        }

    def generate_steward_interpretation(
        self,
        telos_data: Dict,
        shown_source: str,
        turn_number: int,
        thresholds: Dict = None
    ) -> str:
        """
        Generate Steward's human-readable interpretation.

        Args:
            telos_data: TELOS analysis data
            shown_source: What was actually shown ('telos', 'native', 'both')
            turn_number: Current turn
            thresholds: Fidelity thresholds

        Returns:
            Human-readable interpretation
        """
        thresholds = thresholds or {
            'green': FIDELITY_GREEN,
            'yellow': 0.60,
            'orange': 0.50
        }

        fidelity = telos_data.get('fidelity_score', 0.0)
        intervention = telos_data.get('intervention_triggered', False)
        reason = telos_data.get('intervention_reason', '')

        interpretation = f"**Turn {turn_number} Analysis:**\n\n"

        # Explain response source
        if shown_source == 'native':
            interpretation += "📊 **Response Type:** Native (no TELOS governance)\n"
            interpretation += "This response was generated without TELOS intervention.\n\n"
        elif shown_source == 'telos':
            interpretation += "📊 **Response Type:** TELOS-governed\n"
        else:
            interpretation += "📊 **Response Type:** Both shown for comparison\n\n"

        # Explain fidelity
        t_green = thresholds['green']
        t_yellow = thresholds['yellow']
        t_orange = thresholds['orange']

        if fidelity >= t_green:
            interpretation += f"✅ **Alignment:** Aligned ({fidelity:.3f})\n"
            interpretation += "The conversation remains well-aligned with your stated purpose.\n\n"
        elif fidelity >= t_yellow:
            interpretation += f"🟡 **Alignment:** Minor Drift ({fidelity:.3f})\n"
            interpretation += "Slight deviation from your purpose, but within acceptable bounds.\n\n"
        elif fidelity >= t_orange:
            interpretation += f"🟠 **Alignment:** Drift Detected ({fidelity:.3f})\n"
            interpretation += "Noticeable departure from your stated goals.\n\n"
        else:
            interpretation += f"🔴 **Alignment:** Significant Drift ({fidelity:.3f})\n"
            interpretation += "Significant misalignment with your purpose.\n\n"

        # Explain intervention
        if shown_source in ['telos', 'both']:
            if intervention:
                interpretation += f"⚠️ **TELOS Intervention:** Applied\n"
                interpretation += f"**Reason:** {reason}\n\n"
                interpretation += "TELOS detected drift and modified the response to maintain alignment.\n"
            else:
                interpretation += "✔️ **TELOS Monitoring:** No intervention needed\n"
                interpretation += "The response naturally aligned with your purpose.\n"
        else:
            if intervention:
                interpretation += "🔮 **What TELOS would have done:**\n"
                interpretation += f"Would have intervened due to: {reason}\n"
            else:
                interpretation += "🔮 **What TELOS would have done:**\n"
                interpretation += "No intervention would have been needed.\n"

        return interpretation


def create_response_generator_from_manager(manager) -> ResponseGenerator:
    """
    Factory function to create ResponseGenerator from BetaResponseManager.

    Args:
        manager: BetaResponseManager instance

    Returns:
        Configured ResponseGenerator
    """
    return ResponseGenerator(
        llm_client=None,  # Will use cached client
        embedding_provider=getattr(manager, 'embedding_provider', None),
        st_embedding_provider=getattr(manager, 'st_embedding_provider', None),
        st_user_pa_embedding=getattr(manager, 'st_user_pa_embedding', None),
        use_rescaled_fidelity=getattr(manager, 'use_rescaled_fidelity', False),
    )
