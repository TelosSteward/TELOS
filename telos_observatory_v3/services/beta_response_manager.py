"""
Beta Response Manager - FIDELITY-FIRST Governance Demo
======================================================

Redesigned BETA mode that demonstrates TELOS governance in action:
- Calculate user fidelity FIRST before deciding on response type
- Only show TELOS intervention when drift is detected
- Color-coded user messages based on calculated fidelity
- Pre-generated Steward interpretation for intervention cases

NO A/B testing - just pure TELOS demonstration.
"""

import streamlit as st
from typing import Dict, Tuple, Optional
from datetime import datetime
import logging
import numpy as np

logger = logging.getLogger(__name__)

# =============================================================================
# TWO-LAYER DRIFT DETECTION ARCHITECTURE
# =============================================================================
# LAYER 1: Baseline Pre-Filter (for extreme off-topic detection)
# ----------------------------------------------------------------
# Problem: In high-dimensional embedding spaces (1024-dim Mistral), completely
# unrelated content (e.g., "PB&J sandwich" vs "AI governance") produces
# cosine similarity around 0.35-0.56 due to concentration of measure.
# Raw cosine similarity alone cannot detect extreme off-topic content.
#
# Solution: SIMILARITY_BASELINE acts as a fast pre-filter. If raw_sim < baseline,
# the content is so far outside the embedding manifold of the PA that it's
# definitely off-topic - trigger HARD_BLOCK intervention immediately.
#
# This is EMBEDDING-MODEL SPECIFIC. See docs/internal/EMBEDDING_BASELINE_NORMALIZATION.md
SIMILARITY_BASELINE = 0.35  # Mistral 1024-dim empirical floor for unrelated content

# =============================================================================
# LAYER 2: TELOS Primacy State Mathematics (Basin Membership)
# =============================================================================
# Fidelity scale: 0.0 = absolute drift, 1.0 = perfect primacy state
# Basin defines the region around the PA where user input is "within purpose"
# Intervention triggers when fidelity drops BELOW (BASIN - TOLERANCE)
BASIN = 0.40       # Basin boundary - inputs with fidelity >= this are "within purpose"
TOLERANCE = 0.04   # Tolerance margin for basin boundary

# Derived intervention threshold: below this triggers governance intervention
INTERVENTION_THRESHOLD = BASIN - TOLERANCE  # 0.36

# =============================================================================
# UNIFIED INTERVENTION DECISION
# =============================================================================
# Intervene if: Layer1.HARD_BLOCK OR Layer2.outside_basin
# Both layers produce a unified Primacy State for consistent UI display

# UI color thresholds for visual feedback (GOLDILOCKS ZONE - mathematically optimized)
# Method: Grid search optimization over 60,000 threshold combinations
# Objective: Minimize total classification error across 100-question test set
# Achieves 72% accuracy (theoretical maximum given distribution overlap)
FIDELITY_GREEN = 0.76   # >= 0.76: High alignment - no intervention
FIDELITY_YELLOW = 0.73  # 0.73-0.76: Soft guidance zone
FIDELITY_ORANGE = 0.67  # 0.67-0.73: Intervention zone
# Below 0.67 = RED: Strong intervention

# Intent to Role mapping for AI PA derivation (simplified version for BETA)
INTENT_TO_ROLE_MAP = {
    'learn': 'teach',
    'understand': 'explain',
    'solve': 'help solve',
    'create': 'help create',
    'decide': 'help decide',
    'explore': 'guide exploration',
    'analyze': 'help analyze',
    'fix': 'help fix',
    'debug': 'help debug',
    'optimize': 'help optimize',
    'research': 'help research',
    'plan': 'help plan'
}


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

        # Dual PA components for Primacy State calculation
        self.user_pa_embedding = None  # User PA center embedding
        self.ai_pa_embedding = None    # AI PA center embedding
        self.ps_calculator = None      # PrimacyStateCalculator instance
        self.embedding_provider = None # Cached embedding provider

    def generate_turn_responses(self,
                               user_input: str,
                               turn_number: int,
                               sequence: Dict = None) -> Dict:
        """
        FIDELITY-FIRST response generation.

        Flow:
        1. Calculate user prompt fidelity FIRST
        2. Decide intervention level based on fidelity
        3. Generate appropriate response (governed or native)
        4. Pre-generate Steward interpretation if intervention triggered

        Args:
            user_input: The user's message
            turn_number: Current turn number
            sequence: IGNORED - kept for backward compatibility

        Returns:
            Dict containing response data with fidelity-based decision
        """
        logger.info(f"=== FIDELITY-FIRST Turn {turn_number} ===")
        logger.info(f"User input: {user_input[:100]}...")

        # Initialize response data
        response_data = {
            'turn_number': turn_number,
            'timestamp': datetime.now().isoformat(),
            'user_input': user_input,
            'governance_mode': 'fidelity_first',  # Mark as new mode
        }

        # ============================================================
        # STEP 1: Calculate User Fidelity with TWO-LAYER Architecture
        # ============================================================
        # Returns: (fidelity, raw_similarity, baseline_hard_block)
        user_fidelity, raw_similarity, baseline_hard_block = self._calculate_user_fidelity(user_input)

        # Store all metrics in response_data
        response_data['user_fidelity'] = user_fidelity
        response_data['raw_similarity'] = raw_similarity
        response_data['baseline_hard_block'] = baseline_hard_block
        response_data['fidelity_level'] = self._get_fidelity_level(user_fidelity)

        logger.info(f"User Fidelity: {user_fidelity:.3f} ({response_data['fidelity_level']})")
        logger.info(f"Baseline Hard Block: {baseline_hard_block} (raw_sim={raw_similarity:.3f}, baseline={SIMILARITY_BASELINE})")

        # ============================================================
        # STEP 2: TWO-TIER INTERVENTION DECISION (Goldilocks Zone Thresholds)
        # ============================================================
        # LAYER 1: Baseline Pre-Filter - catches extreme off-topic (raw_sim < 0.35)
        # LAYER 2: Goldilocks Zone - actual intervention for fidelity < 0.73 (yellow boundary)
        #
        # ZONE LOGIC (Goldilocks optimized thresholds):
        # - Green (>= 0.76): Aligned - native response, no intervention
        # - Yellow (0.73-0.76): Minor Drift - native response + visual warning
        # - Orange (0.67-0.73): Drift Detected - TELOS intervention required
        # - Red (<0.67): Significant Drift - strong TELOS intervention
        #
        # Yellow zone is like a yellow traffic light - cautionary awareness only
        # User is free to explore but can check with Steward if curious why it triggered
        in_basin = user_fidelity >= INTERVENTION_THRESHOLD  # Basin membership (different from UI zones)
        in_green_zone = user_fidelity >= FIDELITY_GREEN     # >= 0.76 (Aligned)
        in_yellow_zone = user_fidelity >= FIDELITY_YELLOW and user_fidelity < FIDELITY_GREEN  # 0.73-0.76 (Minor Drift)

        # Store layer-specific results for debugging/transparency
        response_data['layer1_triggered'] = baseline_hard_block
        response_data['layer2_in_basin'] = in_basin
        response_data['in_green_zone'] = in_green_zone
        response_data['in_yellow_zone'] = in_yellow_zone

        # UNIFIED DECISION: Intervene ONLY when fidelity < 0.73 (orange/red zones)
        # Yellow zone (0.73-0.76) gets native response with visual temperature warning
        should_intervene = baseline_hard_block or user_fidelity < FIDELITY_YELLOW

        if not should_intervene:
            # GREEN or YELLOW zone: No TELOS intervention - native response
            # Yellow zone still shows visual temperature warning but doesn't modify response
            zone = "GREEN" if in_green_zone else "YELLOW (temperature gauge)"
            logger.info(f"✅ {zone} zone (fidelity {user_fidelity:.3f} >= {FIDELITY_YELLOW}): No intervention")
            response_data['intervention_triggered'] = False
            response_data['intervention_reason'] = None
            response_data['shown_source'] = 'native'

            # Generate native response
            native_response = self._generate_native_response(user_input)
            response_data['shown_response'] = native_response
            response_data['native_response'] = native_response

            # OPTIMIZATION: Use lightweight metrics path instead of full TELOS response
            # This avoids a redundant LLM call (saves 3-8 seconds per message)
            telos_data = self._compute_telos_metrics_lightweight(
                user_input, native_response, user_fidelity
            )

            # Use TELOS math for intervention decisions
            telos_data['intervention_triggered'] = False  # No intervention when in basin
            telos_data['intervention_reason'] = None
            telos_data['user_pa_fidelity'] = user_fidelity  # Store for Steward to access
            telos_data['fidelity_level'] = response_data['fidelity_level']
            telos_data['in_basin'] = True

            response_data['telos_analysis'] = telos_data

        else:
            # ORANGE or RED zone: Drift detected - TELOS intervention required
            # TWO-TIER intervention based on Goldilocks fidelity zones:
            # Orange (0.67-0.73): Drift Detected - moderate intervention
            # Red (<0.67): Significant Drift - strong intervention

            if user_fidelity >= FIDELITY_ORANGE:  # Orange zone: 0.67-0.73 (Drift Detected)
                intervention_reason = "Drift from your stated purpose detected - TELOS is guiding you back"
                intervention_strength = "moderate"
            else:  # Red zone: <0.67 (Significant Drift)
                intervention_reason = "Significant drift detected - TELOS intervention activated"
                intervention_strength = "strong"

            logger.info(f"⚠️ ORANGE/RED zone (fidelity {user_fidelity:.3f} < {FIDELITY_YELLOW}): {intervention_strength}")

            response_data['intervention_triggered'] = True
            response_data['intervention_reason'] = intervention_reason
            response_data['intervention_strength'] = intervention_strength
            response_data['shown_source'] = 'telos'

            # Generate TELOS governed response
            telos_data = self._generate_telos_response(user_input, turn_number)

            # Use TELOS math for intervention decisions - outside basin means intervention
            telos_data['intervention_triggered'] = True  # Intervention when outside basin
            telos_data['intervention_reason'] = intervention_reason
            telos_data['user_pa_fidelity'] = user_fidelity  # Store for Steward to access
            telos_data['fidelity_level'] = response_data['fidelity_level']
            telos_data['in_basin'] = False  # Outside basin

            response_data['telos_analysis'] = telos_data
            response_data['shown_response'] = telos_data.get('response', '')

            # Pre-generate Steward interpretation (only for interventions)
            response_data['steward_interpretation'] = self._generate_steward_interpretation(
                telos_data,
                'telos',
                turn_number
            )
            response_data['has_steward_interpretation'] = True

        # ============================================================
        # STEP 3: Store Turn Data
        # ============================================================
        self._store_turn_data(turn_number, response_data)

        return response_data

    def _calculate_user_fidelity(self, user_input: str) -> tuple:
        """
        Calculate fidelity of user input relative to their PA using TWO-LAYER architecture.

        This is the FIRST calculation - before any response generation.

        LAYER 1: Baseline Pre-Filter
        - If raw_similarity < SIMILARITY_BASELINE (0.35), trigger HARD_BLOCK
        - This catches extreme off-topic content that cosine similarity struggles with

        LAYER 2: TELOS Primacy State (Basin Membership)
        - Uses raw cosine similarity with BASIN/TOLERANCE thresholds
        - Fidelity = cosine_similarity(user_input, user_pa)

        Args:
            user_input: The user's message

        Returns:
            tuple: (fidelity, raw_similarity, baseline_hard_block)
            - fidelity: Raw cosine similarity (0.0 to 1.0) - used for TELOS math
            - raw_similarity: Same as fidelity (kept for logging clarity)
            - baseline_hard_block: True if raw_sim < SIMILARITY_BASELINE
        """
        try:
            # Ensure TELOS engine is initialized
            if not self.telos_engine:
                self._initialize_telos_engine()

            if not self.embedding_provider or self.user_pa_embedding is None:
                logger.warning("Embedding provider or PA not initialized - returning default fidelity")
                return (FIDELITY_GREEN, FIDELITY_GREEN, False)  # Default to Aligned zone if not ready

            # Get user input embedding
            user_embedding = np.array(self.embedding_provider.encode(user_input))

            # Calculate cosine similarity to User PA (RAW - no normalization)
            # This is used by BOTH Layer 1 and Layer 2
            raw_similarity = self._cosine_similarity(user_embedding, self.user_pa_embedding)

            # ============================================================
            # LAYER 1: Baseline Pre-Filter (extreme off-topic detection)
            # ============================================================
            baseline_hard_block = raw_similarity < SIMILARITY_BASELINE

            if baseline_hard_block:
                logger.warning(f"LAYER 1 HARD_BLOCK: raw_sim={raw_similarity:.3f} < baseline={SIMILARITY_BASELINE}")
            else:
                logger.info(f"Layer 1 PASS: raw_sim={raw_similarity:.3f} >= baseline={SIMILARITY_BASELINE}")

            # ============================================================
            # LAYER 2: TELOS Primacy State (fidelity = raw cosine similarity)
            # ============================================================
            # Fidelity IS raw cosine similarity - TELOS math handles thresholds
            fidelity = raw_similarity

            logger.info(f"Layer 2 Fidelity: {fidelity:.3f} (raw cosine similarity)")

            return (fidelity, raw_similarity, baseline_hard_block)

        except Exception as e:
            logger.error(f"Error calculating user fidelity: {e}")
            return (FIDELITY_GREEN, FIDELITY_GREEN, False)  # Default to Aligned zone on error

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)

    def _get_fidelity_level(self, fidelity: float) -> str:
        """Get human-readable fidelity level."""
        if fidelity >= FIDELITY_GREEN:
            return "green"
        elif fidelity >= FIDELITY_YELLOW:
            return "yellow"
        elif fidelity >= FIDELITY_ORANGE:
            return "orange"
        else:
            return "red"

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
                'drift_detected': result.get('telic_fidelity', 1.0) < FIDELITY_YELLOW,  # < 0.73 = drift
                'in_basin': result.get('in_basin', True),
                'embeddings': {
                    'user': result.get('user_embedding'),
                    'response': result.get('response_embedding'),
                    'pa': result.get('pa_embedding')
                }
            }

            # =================================================================
            # DUAL PA: Compute Primacy State (f_user, f_ai, ps_score)
            # Formula: PS = ρ_PA · (2·F_user·F_AI)/(F_user + F_AI)
            # =================================================================
            # DEBUG: Log PS calculation prerequisites (WARNING level to ensure visibility)
            logger.warning(f"📊 PS Calculation Check:")
            logger.warning(f"   ps_calculator exists: {self.ps_calculator is not None}")
            logger.warning(f"   user_pa_embedding exists: {self.user_pa_embedding is not None}")
            logger.warning(f"   ai_pa_embedding exists: {self.ai_pa_embedding is not None}")

            if self.ps_calculator and self.user_pa_embedding is not None and self.ai_pa_embedding is not None:
                try:
                    # Get response text and compute embedding
                    response_text = result.get('governed_response', '')
                    if response_text and self.embedding_provider:
                        response_embedding = np.array(self.embedding_provider.encode(response_text))

                        # Compute Primacy State metrics
                        ps_metrics = self.ps_calculator.compute_primacy_state(
                            response_embedding=response_embedding,
                            user_pa_embedding=self.user_pa_embedding,
                            ai_pa_embedding=self.ai_pa_embedding,
                            use_cached_correlation=True
                        )

                        # Store dual PA fidelity values in telos_data
                        # NOTE: user_pa_fidelity is already set correctly using _calculate_user_fidelity()
                        # which computes USER INPUT to User PA similarity (deterministic).
                        # ps_metrics.f_user measures RESPONSE to User PA which is NOT what we want for F_user.
                        # Only update if not already set (shouldn't happen, but defensive):
                        if 'user_pa_fidelity' not in telos_data:
                            telos_data['user_pa_fidelity'] = ps_metrics.f_user
                        # AI fidelity correctly uses response embedding to AI PA
                        telos_data['ai_pa_fidelity'] = ps_metrics.f_ai
                        telos_data['pa_correlation'] = ps_metrics.rho_pa

                        # FIX: Recalculate PS using the DISPLAYED F_user (USER INPUT fidelity)
                        # not ps_metrics.ps_score which uses RESPONSE fidelity to User PA
                        # Formula: PS = harmonic_mean(F_user, F_AI) - pure harmonic mean, no rho_PA scaling
                        displayed_f_user = telos_data['user_pa_fidelity']
                        f_ai = ps_metrics.f_ai
                        rho_pa = ps_metrics.rho_pa  # Still store for reference
                        epsilon = 1e-10  # Prevent division by zero
                        harmonic_mean = (2 * displayed_f_user * f_ai) / (displayed_f_user + f_ai + epsilon)
                        # PS = pure harmonic mean (no rho_PA scaling for display consistency)
                        corrected_ps = harmonic_mean
                        telos_data['primacy_state_score'] = corrected_ps
                        telos_data['primacy_state_condition'] = ps_metrics.condition

                        # Log Primacy State metrics (WARNING level for visibility)
                        logger.warning(f"📊 Primacy State Metrics (CORRECTED):")
                        logger.warning(f"   F_user (User Input Fidelity): {displayed_f_user:.3f} [USER INPUT -> User PA]")
                        logger.warning(f"   F_AI (AI Response Fidelity): {f_ai:.3f} [RESPONSE -> AI PA]")
                        logger.warning(f"   PS (Primacy State): {corrected_ps:.3f} [pure harmonic mean]")
                        logger.warning(f"   ρ_PA (Correlation): {rho_pa:.3f} [stored for reference, not used in PS]")
                        logger.warning(f"   Condition: {ps_metrics.condition}")
                        logger.warning(f"   (Old ps_metrics.ps_score was: {ps_metrics.ps_score:.3f} - used wrong F_user)")

                except Exception as ps_error:
                    logger.warning(f"⚠️ Could not compute Primacy State: {ps_error}")
                    import traceback
                    logger.warning(f"   PS Traceback: {traceback.format_exc()}")
                    # Continue without PS metrics - don't fail the response
            else:
                # Log WHY PS calculation was skipped
                logger.warning("⚠️ PS calculation SKIPPED - missing prerequisites:")
                if not self.ps_calculator:
                    logger.warning("   - ps_calculator is None (init may have failed)")
                if self.user_pa_embedding is None:
                    logger.warning("   - user_pa_embedding is None")
                if self.ai_pa_embedding is None:
                    logger.warning("   - ai_pa_embedding is None")

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

    def _compute_telos_metrics_lightweight(
        self, user_input: str, response: str, user_fidelity: float
    ) -> Dict:
        """
        Compute TELOS metrics without API calls (ZERO OVERHEAD for GREEN/YELLOW).

        Used for GREEN/YELLOW zones where intervention isn't needed.
        This path does NO embedding API calls - just returns basic metrics
        from already-calculated values.

        GREEN/YELLOW = Just native LLM response + color indicator
        ORANGE/RED = Full TELOS with intervention (handled elsewhere)

        Args:
            user_input: The user's message
            response: The already-generated native response
            user_fidelity: Pre-calculated user fidelity score

        Returns:
            Dict with basic TELOS metrics (no API calls)
        """
        logger.info("🚀 ZERO-OVERHEAD path: GREEN/YELLOW zone - no extra API calls")

        # Return basic metrics only - NO embedding API calls
        # For GREEN/YELLOW, we only need to show the fidelity color
        # AI Fidelity and Primacy State are NOT computed - show as N/A in UI
        telos_data = {
            'response': response,
            'fidelity_score': None,  # NOT user_fidelity - that was causing AI Fidelity to show User Fidelity
            'distance_from_pa': 1.0 - user_fidelity,
            'intervention_triggered': False,
            'intervention_type': None,
            'intervention_reason': '',
            'drift_detected': user_fidelity < FIDELITY_YELLOW,  # < 0.73 = drift
            'in_basin': True,
            # NOT COMPUTED for GREEN/YELLOW zones - no extra API calls
            # UI should display "N/A" or "--" for these
            'ai_pa_fidelity': None,  # NOT COMPUTED - requires response embedding
            'primacy_state_score': None,  # NOT COMPUTED - requires PS calculation
            'primacy_state_condition': 'not_computed',
            'pa_correlation': None,  # NOT COMPUTED
            'lightweight_path': True,  # Flag to indicate this was the fast path
        }

        logger.info(f"📊 Zero-overhead metrics: F_user={user_fidelity:.3f} (no extra embedding calls)")

        return telos_data

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

        # Explain fidelity using Goldilocks zone thresholds
        if fidelity >= FIDELITY_GREEN:  # >= 0.76 (Aligned)
            interpretation += f"✅ **Alignment:** Aligned ({fidelity:.3f})\n"
            interpretation += "The conversation remains well-aligned with your stated purpose.\n\n"
        elif fidelity >= FIDELITY_YELLOW:  # 0.73-0.76 (Minor Drift)
            interpretation += f"🟡 **Alignment:** Minor Drift ({fidelity:.3f})\n"
            interpretation += "Slight deviation from your purpose, but within acceptable bounds.\n\n"
        elif fidelity >= FIDELITY_ORANGE:  # 0.67-0.73 (Drift Detected)
            interpretation += f"🟠 **Alignment:** Drift Detected ({fidelity:.3f})\n"
            interpretation += "Noticeable departure from your stated goals.\n\n"
        else:  # < 0.67 (Significant Drift)
            interpretation += f"🔴 **Alignment:** Significant Drift ({fidelity:.3f})\n"
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

        # For stats, use user_pa_fidelity as primary (always computed), fidelity_score as fallback
        fidelity = telos_data.get('user_pa_fidelity') or telos_data.get('fidelity_score') or 0.0
        if fidelity is not None:
            stats['fidelity_scores'].append(fidelity)
            # Filter out any None values when calculating average
            valid_scores = [f for f in stats['fidelity_scores'] if f is not None]
            stats['avg_fidelity'] = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

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
        # beta_current_turn is the NEXT turn to play (starts at 1), so completed turns = current_turn - 1
        completed_turns = st.session_state.get('beta_current_turn', 1) - 1
        for i in range(1, completed_turns + 1):
            turn_data = st.session_state.get(f'beta_turn_{i}_data', {})
            if turn_data:
                history.append({'role': 'user', 'content': turn_data.get('user_input', '')})
                history.append({'role': 'assistant', 'content': turn_data.get('shown_response', '')})
        return history

    def _initialize_telos_engine(self):
        """Initialize TELOS engine for governance with dual PA support."""
        try:
            from telos_purpose.core.unified_steward import UnifiedGovernanceSteward, PrimacyAttractor
            from telos_purpose.core.embedding_provider import MistralEmbeddingProvider
            from telos_purpose.llm_clients.mistral_client import MistralClient
            from telos_purpose.core.primacy_state import PrimacyStateCalculator

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
                purpose_str = "Engage in helpful conversation"
                scope_str = "General assistance"
                attractor = PrimacyAttractor(
                    purpose=[purpose_str],
                    scope=[scope_str],
                    boundaries=["Maintain respectful dialogue"],
                    constraint_tolerance=0.02  # STRICT governance for BETA testing
                )
                logger.warning("⚠️ BETA: No established PA found - using fallback")

            # Initialize LLM client and embedding provider
            llm_client = MistralClient()
            embedding_provider = MistralEmbeddingProvider()  # Using Mistral embeddings (1024 dims)
            self.embedding_provider = embedding_provider  # Cache for later use

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

            # =================================================================
            # DUAL PA SETUP: Derive AI PA and compute embeddings
            # CRITICAL: Cache PA embeddings in session state to ensure determinism
            # The Mistral embedding API may return slightly different float values
            # on each call, causing fidelity calculations to vary. By caching in
            # session state, we ensure the same PA embedding is used throughout
            # the entire BETA session.
            # =================================================================
            logger.info("🔧 Setting up Dual PA for Primacy State calculation...")

            # Check if PA embeddings are already cached in session state
            if 'cached_user_pa_embedding' in st.session_state and 'cached_ai_pa_embedding' in st.session_state:
                # Use cached embeddings for deterministic fidelity calculations
                self.user_pa_embedding = st.session_state.cached_user_pa_embedding
                self.ai_pa_embedding = st.session_state.cached_ai_pa_embedding
                logger.info("   ✅ Using CACHED PA embeddings from session state (deterministic)")
                logger.info(f"   User PA embedded: {len(self.user_pa_embedding)} dims (cached)")
                logger.info(f"   AI PA embedded: {len(self.ai_pa_embedding)} dims (cached)")
            else:
                # 1. Create User PA text for embedding
                user_pa_text = f"Purpose: {purpose_str}. Scope: {scope_str}."

                # 2. Derive AI PA from User PA using intent detection
                detected_intent = self._detect_intent_from_purpose(purpose_str)
                role_action = INTENT_TO_ROLE_MAP.get(detected_intent, 'help')
                ai_purpose = f"{role_action.capitalize()} the user as they work to: {purpose_str}"
                ai_pa_text = f"AI Role: {ai_purpose}. Supporting scope: {scope_str}."

                # PERFORMANCE: Batch both embeddings in a single API call (2x faster)
                # Check if embedding provider supports batch encoding
                if hasattr(embedding_provider, 'batch_encode'):
                    logger.info("   🚀 Using batch embedding (single API call for both PAs)")
                    embeddings = embedding_provider.batch_encode([user_pa_text, ai_pa_text])
                    self.user_pa_embedding = embeddings[0]
                    self.ai_pa_embedding = embeddings[1]
                else:
                    # Fallback to sequential encoding
                    self.user_pa_embedding = np.array(embedding_provider.encode(user_pa_text))
                    self.ai_pa_embedding = np.array(embedding_provider.encode(ai_pa_text))

                logger.info(f"   User PA embedded: {len(self.user_pa_embedding)} dims")
                logger.info(f"   AI PA derived (intent: {detected_intent} -> {role_action})")
                logger.info(f"   AI PA embedded: {len(self.ai_pa_embedding)} dims")

                # Cache in session state for future use (ensures determinism)
                st.session_state.cached_user_pa_embedding = self.user_pa_embedding
                st.session_state.cached_ai_pa_embedding = self.ai_pa_embedding
                logger.info("   📦 PA embeddings CACHED in session state for deterministic calculations")

            # 3. Initialize Primacy State Calculator
            self.ps_calculator = PrimacyStateCalculator(track_energy=True)
            logger.info("✅ PrimacyStateCalculator initialized with energy tracking")

            # 4. Compute initial PA correlation (rho_PA)
            rho_pa = self.ps_calculator.cosine_similarity(
                self.user_pa_embedding,
                self.ai_pa_embedding
            )
            logger.info(f"   PA Correlation (rho_PA): {rho_pa:.3f}")

            # Log basin configuration
            if hasattr(self.telos_engine, 'attractor_math') and self.telos_engine.attractor_math:
                basin_radius = self.telos_engine.attractor_math.basin_radius
                tolerance = self.telos_engine.attractor_math.constraint_tolerance
                embedding_dim = embedding_provider.dimension
                logger.info(f"✅ TELOS engine initialized for BETA testing (DUAL PA MODE)")
                logger.info(f"   Embedding model: Mistral mistral-embed ({embedding_dim} dims)")
                logger.info(f"   Constraint tolerance: {tolerance}")
                logger.info(f"   Basin radius: {basin_radius:.3f}")
                logger.info(f"   Expected fidelity for off-topic: < {(1 - 0.5/basin_radius):.3f}")
            else:
                logger.info("✅ TELOS engine initialized for BETA testing with DUAL PA")

        except Exception as e:
            logger.error(f"Failed to initialize TELOS engine: {e}")
            import traceback
            logger.error(f"   Traceback: {traceback.format_exc()}")
            self.telos_engine = None
            self.ps_calculator = None

    def _detect_intent_from_purpose(self, purpose: str) -> str:
        """
        Detect user intent from purpose string using keyword matching.

        Simple synchronous implementation for BETA (avoids async complexity).

        Args:
            purpose: User's stated purpose

        Returns:
            Intent verb (e.g., 'learn', 'solve', 'create')
        """
        purpose_lower = purpose.lower()

        # Check for intent keywords in purpose
        intent_keywords = {
            'learn': ['learn', 'study', 'understand better', 'education'],
            'understand': ['understand', 'comprehend', 'grasp', 'figure out'],
            'solve': ['solve', 'fix', 'resolve', 'troubleshoot', 'debug'],
            'create': ['create', 'build', 'make', 'develop', 'design', 'write'],
            'decide': ['decide', 'choose', 'select', 'pick', 'evaluate options'],
            'explore': ['explore', 'discover', 'investigate', 'look into'],
            'analyze': ['analyze', 'examine', 'review', 'assess', 'audit'],
            'fix': ['fix', 'repair', 'correct', 'patch'],
            'debug': ['debug', 'trace', 'diagnose'],
            'optimize': ['optimize', 'improve', 'enhance', 'streamline'],
            'research': ['research', 'study', 'survey', 'review literature'],
            'plan': ['plan', 'organize', 'schedule', 'strategy', 'roadmap']
        }

        for intent, keywords in intent_keywords.items():
            for keyword in keywords:
                if keyword in purpose_lower:
                    return intent

        # Default to 'understand' if no match
        return 'understand'