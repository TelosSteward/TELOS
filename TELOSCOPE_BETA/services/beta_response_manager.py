"""
Beta Response Manager - Generates ALL Responses for Analysis
=============================================================

Always generates TELOS analysis even when showing native response.
Stores everything for Observatory review.
"""

import streamlit as st
from typing import Dict, Tuple, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class BetaResponseManager:
    """Manages response generation and storage for BETA testing."""

    def __init__(self, state_manager):
        """
        Initialize with reference to state manager.

        Args:
            state_manager: Reference to the main StateManager
        """
        self.state_manager = state_manager
        self.telos_engine = None

    def generate_turn_responses(self,
                               user_input: str,
                               turn_number: int,
                               sequence: Dict) -> Dict:
        """
        Generate all necessary responses for a turn.

        Args:
            user_input: The user's message
            turn_number: Current turn number
            sequence: The pre-generated test sequence

        Returns:
            Dict containing all response data
        """
        from services.beta_sequence_generator import BetaSequenceGenerator

        generator = BetaSequenceGenerator()
        turn_config = generator.get_turn_config(sequence, turn_number)

        # Initialize response data
        response_data = {
            'turn_number': turn_number,
            'test_type': turn_config['test_type'],
            'timestamp': datetime.now().isoformat(),
            'user_input': user_input
        }

        # ALWAYS generate TELOS analysis (even if not shown)
        telos_data = self._generate_telos_response(user_input, turn_number)
        response_data['telos_analysis'] = telos_data

        # Generate native response if needed
        if turn_config['test_type'] == 'head_to_head' or \
           turn_config['response_source'] in ['native', 'both']:
            native_response = self._generate_native_response(user_input)
            response_data['native_response'] = native_response

        # Determine what to show the user
        if turn_config['test_type'] == 'single_blind':
            if turn_config['response_source'] == 'telos':
                response_data['shown_response'] = telos_data['response']
                response_data['shown_source'] = 'telos'
            else:
                response_data['shown_response'] = response_data.get('native_response')
                response_data['shown_source'] = 'native'
        else:
            # Head-to-head: show both
            response_data['shown_response'] = 'both'
            response_data['shown_source'] = 'both'

        # Generate Steward's interpretation
        response_data['steward_interpretation'] = self._generate_steward_interpretation(
            telos_data,
            turn_config['response_source'],
            turn_number
        )

        # Store in session for Observatory review
        self._store_turn_data(turn_number, response_data)

        return response_data

    def _generate_telos_response(self, user_input: str, turn_number: int) -> Dict:
        """
        Generate TELOS response with full analysis.

        Args:
            user_input: User's message
            turn_number: Current turn

        Returns:
            Dict with TELOS response and metrics
        """
        try:
            # Get TELOS engine (lazy init)
            if not self.telos_engine:
                self._initialize_telos_engine()

            # Get PA from session
            pa = st.session_state.get('primacy_attractor', {})

            # Generate response with TELOS governance
            result = self.telos_engine.process_turn(
                user_input=user_input,
                primacy_attractor=pa
            )

            # Extract all metrics
            telos_data = {
                'response': result.get('response', ''),
                'fidelity_score': result.get('fidelity', 0.0),
                'distance_from_pa': result.get('distance', 0.0),
                'intervention_triggered': result.get('intervention_triggered', False),
                'intervention_type': result.get('intervention_type', None),
                'intervention_reason': result.get('intervention_reason', ''),
                'drift_detected': result.get('drift_detected', False),
                'confidence': result.get('confidence', 0.0),
                'embeddings': {
                    'user': result.get('user_embedding'),
                    'response': result.get('response_embedding'),
                    'pa': result.get('pa_embedding')
                }
            }

            # Log intervention if triggered
            if telos_data['intervention_triggered']:
                logger.info(f"Turn {turn_number}: TELOS intervention - {telos_data['intervention_reason']}")

            return telos_data

        except Exception as e:
            logger.error(f"Error generating TELOS response: {e}")
            # Return fallback
            return {
                'response': self._generate_native_response(user_input),
                'fidelity_score': 0.5,
                'error': str(e)
            }

    def _generate_native_response(self, user_input: str) -> str:
        """
        Generate native LLM response without TELOS.

        Args:
            user_input: User's message

        Returns:
            Native response string
        """
        try:
            # Use base LLM without governance
            from telos_purpose.llm_clients.mistral_client import MistralClient

            client = MistralClient()
            conversation = self._get_conversation_history()
            conversation.append({'role': 'user', 'content': user_input})

            response = client.generate(
                messages=conversation,
                max_tokens=512
            )

            return response

        except Exception as e:
            logger.error(f"Error generating native response: {e}")
            return "I understand you're testing the system. How can I help you explore TELOS governance?"

    def _generate_steward_interpretation(self,
                                        telos_data: Dict,
                                        shown_source: str,
                                        turn_number: int) -> str:
        """
        Generate Steward's human-readable interpretation.

        Args:
            telos_data: TELOS analysis data
            shown_source: What was actually shown ('telos', 'native', 'both')
            turn_number: Current turn

        Returns:
            Human-readable interpretation
        """
        fidelity = telos_data.get('fidelity_score', 0.0)
        intervention = telos_data.get('intervention_triggered', False)
        reason = telos_data.get('intervention_reason', '')
        drift = telos_data.get('drift_detected', False)

        # Build interpretation based on what happened
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
        if fidelity >= 0.85:
            interpretation += f"✅ **Alignment:** Strong ({fidelity:.3f})\n"
            interpretation += "The conversation remains well-aligned with your stated purpose.\n\n"
        elif fidelity >= 0.70:
            interpretation += f"🟡 **Alignment:** Mild drift detected ({fidelity:.3f})\n"
            interpretation += "Some deviation from your purpose, but within acceptable bounds.\n\n"
        elif fidelity >= 0.50:
            interpretation += f"🟠 **Alignment:** Moderate drift ({fidelity:.3f})\n"
            interpretation += "Noticeable departure from your stated goals.\n\n"
        else:
            interpretation += f"🔴 **Alignment:** Severe drift ({fidelity:.3f})\n"
            interpretation += "Significant misalignment with your purpose.\n\n"

        # Explain intervention (if TELOS was active)
        if shown_source in ['telos', 'both']:
            if intervention:
                interpretation += f"⚠️ **TELOS Intervention:** Applied\n"
                interpretation += f"**Reason:** {reason}\n\n"
                interpretation += "TELOS detected drift and modified the response to maintain alignment.\n"
            else:
                interpretation += "✔️ **TELOS Monitoring:** No intervention needed\n"
                interpretation += "The response naturally aligned with your purpose.\n"
        else:
            # Show what TELOS WOULD have done
            if intervention:
                interpretation += "🔮 **What TELOS would have done:**\n"
                interpretation += f"Would have intervened due to: {reason}\n"
            else:
                interpretation += "🔮 **What TELOS would have done:**\n"
                interpretation += "No intervention would have been needed.\n"

        return interpretation

    def _store_turn_data(self, turn_number: int, data: Dict):
        """Store turn data for Observatory review."""
        storage_key = f'beta_turn_{turn_number}_data'
        st.session_state[storage_key] = data

        # Also update running statistics
        if 'beta_statistics' not in st.session_state:
            st.session_state.beta_statistics = {
                'total_interventions': 0,
                'total_drifts': 0,
                'avg_fidelity': 0.0,
                'fidelity_scores': []
            }

        stats = st.session_state.beta_statistics
        telos_data = data.get('telos_analysis', {})

        if telos_data.get('intervention_triggered'):
            stats['total_interventions'] += 1
        if telos_data.get('drift_detected'):
            stats['total_drifts'] += 1

        fidelity = telos_data.get('fidelity_score', 0.0)
        stats['fidelity_scores'].append(fidelity)
        stats['avg_fidelity'] = sum(stats['fidelity_scores']) / len(stats['fidelity_scores'])

    def _get_conversation_history(self) -> list:
        """Get conversation history for context."""
        history = []
        for i in range(1, st.session_state.get('beta_turn_count', 0) + 1):
            turn_data = st.session_state.get(f'beta_turn_{i}_data', {})
            if turn_data:
                history.append({'role': 'user', 'content': turn_data.get('user_input', '')})
                history.append({'role': 'assistant', 'content': turn_data.get('shown_response', '')})
        return history

    def _initialize_telos_engine(self):
        """Initialize TELOS engine for governance."""
        try:
            from telos_purpose.core.unified_steward import UnifiedGovernanceSteward
            from telos_purpose.core.embedding_provider import SentenceTransformerProvider

            embedding_provider = SentenceTransformerProvider()
            self.telos_engine = UnifiedGovernanceSteward(embedding_provider)

            logger.info("TELOS engine initialized for BETA testing")

        except Exception as e:
            logger.error(f"Failed to initialize TELOS engine: {e}")
            self.telos_engine = None