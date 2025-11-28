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

    def __init__(self, state_manager, backend_client=None):
        """
        Initialize with reference to state manager.

        Args:
            state_manager: Reference to the main StateManager
            backend_client: Optional BackendService for delta transmission
        """
        self.state_manager = state_manager
        self.telos_engine = None
        self.backend = backend_client

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
                logger.info(f"🎲 Turn {turn_number}: A/B test selected TELOS response (governed)")
            else:
                response_data['shown_response'] = response_data.get('native_response')
                response_data['shown_source'] = 'native'
                logger.info(f"🎲 Turn {turn_number}: A/B test selected NATIVE response (ungoverned)")
        else:
            # Head-to-head: show both responses for comparison
            # Store both responses for the UI to render side-by-side
            response_data['shown_response'] = telos_data['response']  # Primary display
            response_data['shown_source'] = 'both'
            response_data['comparison_mode'] = True
            response_data['response_a'] = telos_data['response']  # TELOS response
            response_data['response_b'] = response_data.get('native_response', '')  # Native response
            logger.info(f"🎲 Turn {turn_number}: Head-to-head test - showing BOTH responses for comparison")

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
        Generate TELOS response with ACTIVE governance.

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

            # Get conversation history
            conversation_history = self._get_conversation_history()

            logger.info(f"🔍 Generating TELOS governed response for turn {turn_number}")
            logger.info(f"   User input: {user_input[:100]}")

            # Generate governed response (ACTIVE MODE - prevents drift before generation)
            result = self.telos_engine.generate_governed_response(
                user_input=user_input,
                conversation_context=conversation_history
            )

            logger.info(f"📊 TELOS Result:")
            logger.info(f"   Fidelity: {result.get('telic_fidelity', 'N/A')}")
            logger.info(f"   Intervention: {result.get('intervention_applied', False)}")
            logger.info(f"   Response preview: {result.get('governed_response', '')[:100]}")

            # Extract all metrics
            telos_data = {
                'response': result.get('governed_response', ''),  # FIX: Use 'governed_response' not 'response'
                'fidelity_score': result.get('telic_fidelity', 0.0),
                'distance_from_pa': result.get('error_signal', 0.0),
                'intervention_triggered': result.get('intervention_applied', False),
                'intervention_type': result.get('intervention_type', None),
                'intervention_reason': result.get('intervention_reason', ''),
                'drift_detected': result.get('telic_fidelity', 1.0) < 0.7,
                'in_basin': result.get('in_basin', True),
                'embeddings': {
                    'user': result.get('user_embedding'),
                    'response': result.get('response_embedding'),
                    'pa': result.get('pa_embedding')
                }
            }

            # Log intervention if triggered
            if telos_data['intervention_triggered']:
                logger.warning(f"⚠️ Turn {turn_number}: TELOS INTERVENTION APPLIED!")
                logger.warning(f"   Reason: {telos_data['intervention_reason']}")
                logger.warning(f"   Type: {telos_data['intervention_type']}")

            return telos_data

        except Exception as e:
            logger.error(f"❌ Error generating TELOS response: {e}")
            logger.error(f"   Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"   Traceback: {traceback.format_exc()}")

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
                max_tokens=16000  # BETA: Allow full responses (up to ~16k tokens)
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
        """Store turn data for Observatory review and transmit to Supabase."""
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

        # Transmit delta to Supabase (privacy-preserving - metrics only, no content)
        if self.backend and self.backend.enabled:
            try:
                session_id = st.session_state.get('session_id',
                    self.state_manager.state.session_id if hasattr(self.state_manager.state, 'session_id') else 'unknown')

                delta_data = {
                    'session_id': str(session_id),
                    'turn_number': turn_number,
                    'fidelity_score': fidelity,
                    'distance_from_pa': telos_data.get('distance_from_pa', 0.0),
                    'intervention_triggered': telos_data.get('intervention_triggered', False),
                    'intervention_type': telos_data.get('intervention_type'),
                    'intervention_reason': telos_data.get('intervention_reason', '')[:200] if telos_data.get('intervention_reason') else None,  # Truncate reason
                    'drift_detected': telos_data.get('drift_detected', False),
                    'test_type': data.get('test_type'),
                    'response_source': data.get('shown_source'),
                    'mode': 'beta'
                }

                self.backend.transmit_delta(delta_data)
                logger.info(f"✓ Transmitted BETA turn {turn_number} delta to backend")
            except Exception as e:
                logger.error(f"❌ Failed to transmit delta for turn {turn_number}: {e}")

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
            from telos_purpose.core.unified_steward import UnifiedGovernanceSteward, PrimacyAttractor
            from telos_purpose.core.embedding_provider import MistralEmbeddingProvider
            from telos_purpose.llm_clients.mistral_client import MistralClient

            # Read PA from session state (established via BETA questionnaire)
            # PAOnboarding component saves to 'primacy_attractor' and 'pa_established'
            pa_data = st.session_state.get('primacy_attractor', None)
            pa_established = st.session_state.get('pa_established', False)

            logger.warning(f"🔍 BETA TELOS Init - PA Status:")
            logger.warning(f"  - pa_data exists: {pa_data is not None}")
            logger.warning(f"  - pa_established: {pa_established}")
            logger.warning(f"  - ALL session_state keys: {list(st.session_state.keys())}")
            if pa_data:
                logger.warning(f"  - PA purpose: {pa_data.get('purpose', 'N/A')}")
                logger.warning(f"  - PA scope: {pa_data.get('scope', 'N/A')}")
                logger.warning(f"  - PA boundaries: {pa_data.get('boundaries', 'N/A')}")

            if pa_data and pa_established:
                # Use established PA from questionnaire
                purpose_str = pa_data.get('purpose', 'General assistance')
                scope_str = pa_data.get('scope', 'Open discussion')

                # Get boundaries with fallback for empty lists
                boundaries = pa_data.get('boundaries', [])
                if not boundaries:  # Handle empty boundaries from older sessions
                    boundaries = [
                        "Stay focused on stated purpose",
                        "Avoid unrelated tangents",
                        "Maintain productive dialogue"
                    ]
                    logger.warning(f"  - PA had empty boundaries, using defaults: {boundaries}")

                # Convert strings to lists as PrimacyAttractor expects List[str]
                attractor = PrimacyAttractor(
                    purpose=[purpose_str] if isinstance(purpose_str, str) else purpose_str,
                    scope=[scope_str] if isinstance(scope_str, str) else scope_str,
                    boundaries=boundaries,
                    constraint_tolerance=0.02  # STRICT governance for BETA testing (basin_radius ≈ 1.02)
                )
                logger.info(f"✅ BETA: Using established PA")
                logger.info(f"   Purpose: {purpose_str[:80]}")
                logger.info(f"   Scope: {scope_str[:80]}")
            else:
                # Fallback PA (should rarely happen - PA questionnaire runs before BETA starts)
                attractor = PrimacyAttractor(
                    purpose=["Engage in helpful conversation"],
                    scope=["General assistance"],
                    boundaries=["Maintain respectful dialogue"],
                    constraint_tolerance=0.02  # STRICT governance for BETA testing
                )
                logger.warning("⚠️ BETA: No established PA found - using fallback")

            # Initialize LLM client and embedding provider
            llm_client = MistralClient()
            embedding_provider = MistralEmbeddingProvider()  # Using Mistral embeddings (1024 dims)

            # Initialize steward with proper attractor
            self.telos_engine = UnifiedGovernanceSteward(
                attractor=attractor,
                llm_client=llm_client,
                embedding_provider=embedding_provider,
                enable_interventions=True
            )

            # CRITICAL: Start session before using the steward
            logger.info("🔧 Starting TELOS session...")
            self.telos_engine.start_session()
            logger.info("✅ TELOS session started successfully")

            # Log basin configuration
            if hasattr(self.telos_engine, 'attractor_math') and self.telos_engine.attractor_math:
                basin_radius = self.telos_engine.attractor_math.basin_radius
                tolerance = self.telos_engine.attractor_math.constraint_tolerance
                embedding_dim = embedding_provider.dimension
                logger.info(f"✅ TELOS engine initialized for BETA testing")
                logger.info(f"   Embedding model: Mistral mistral-embed ({embedding_dim} dims)")
                logger.info(f"   Constraint tolerance: {tolerance}")
                logger.info(f"   Basin radius: {basin_radius:.3f}")
                logger.info(f"   Expected fidelity for off-topic: < {(1 - 0.5/basin_radius):.3f}")
            else:
                logger.info("✅ TELOS engine initialized for BETA testing with correct PA")

        except Exception as e:
            logger.error(f"Failed to initialize TELOS engine: {e}")
            self.telos_engine = None