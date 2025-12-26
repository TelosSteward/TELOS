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

# Import TELOS command detection and PA enrichment
from services.pa_enrichment import detect_telos_command, PAEnrichmentService

# Import SentenceTransformer thresholds for template mode (Clean Lane approach)
from telos_purpose.core.constants import (
    ST_FIDELITY_GREEN, ST_FIDELITY_YELLOW, ST_FIDELITY_ORANGE, ST_FIDELITY_RED
)

# Import display normalization for SentenceTransformer scores
# This maps raw ST scores to user-expected display values (0.70+ = GREEN, etc.)
from telos_purpose.core.fidelity_display import normalize_fidelity_for_display

# Import rescaling for SentenceTransformer-based fidelity (including MPNet for AI fidelity)
from telos_purpose.core.embedding_provider import rescale_sentence_transformer_fidelity

# Import Steward styling for granular intervention responses
# Steward is the intervention personality of TELOS - appears only below GREEN threshold
from config.steward_styles import (
    get_steward_style,
    get_intervention_prompt,
    get_response_opener,
    get_style_interpolation
)

# Import Semantic Interpreter - bridges mathematical governance to linguistic output
# Two focal points: Fidelity Value + Purpose -> Concrete Linguistic Specifications
from telos_purpose.core.semantic_interpreter import (
    interpret as semantic_interpret,
    get_exemplar,
    compute_behavioral_fidelity,
    get_behavioral_fidelity_band
)

# Telemetric Keys - Cryptographic access control for governance data
try:
    from telos_privacy.cryptography.telemetric_keys import TelemetricSessionManager
    TELEMETRIC_KEYS_AVAILABLE = True
except ImportError:
    TELEMETRIC_KEYS_AVAILABLE = False

# Governance Trace Collector - Central logging for governance observability
try:
    from telos_purpose.core.governance_trace_collector import (
        get_trace_collector,
        GovernanceTraceCollector,
    )
    from telos_purpose.core.evidence_schema import (
        InterventionLevel,
        PrivacyMode,
    )
    TRACE_COLLECTOR_AVAILABLE = True
except ImportError:
    TRACE_COLLECTOR_AVAILABLE = False

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
SIMILARITY_BASELINE = 0.20  # Layer 1 hard block - only catch truly extreme off-topic (e.g., "pizza" vs "AI governance")

# =============================================================================
# LAYER 2: TELOS Primacy State Mathematics (Basin Membership)
# =============================================================================
# Fidelity scale: 0.0 = absolute drift, 1.0 = perfect primacy state
# Basin defines the region around the PA where user input is "within purpose"
# Intervention triggers when fidelity drops BELOW (BASIN - TOLERANCE)
BASIN = 0.28       # Basin boundary - aligned with ST_FIDELITY_YELLOW raw threshold
TOLERANCE = 0.04   # Tolerance margin for basin boundary

# Derived intervention threshold: below this triggers governance intervention
INTERVENTION_THRESHOLD = BASIN - TOLERANCE  # 0.24 - only intervene on clearly off-topic content

# =============================================================================
# UNIFIED INTERVENTION DECISION
# =============================================================================
# Intervene if: Layer1.HARD_BLOCK OR Layer2.outside_basin
# Both layers produce a unified Primacy State for consistent UI display

# UI color thresholds for visual feedback (GOLDILOCKS ZONE - simplified)
# Clean round numbers for intuitive understanding
# 0.7-1.0: Aligned (Green), 0.6-0.7: Minor Drift (Yellow), 0.5-0.6: Drift Detected (Orange), <0.5: Severe (Red)
FIDELITY_GREEN = 0.70   # >= 0.70: High alignment - no intervention
FIDELITY_YELLOW = 0.60  # 0.60-0.70: Soft guidance zone
FIDELITY_ORANGE = 0.50  # 0.50-0.60: Intervention zone
# Below 0.50 = RED: Strong intervention

# =============================================================================
# INTERVENTION RESPONSE LENGTH CONSTRAINTS
# =============================================================================
# Drifted responses should be CONCISE redirects, not full-length engagements.
# This implements proportional brevity: more drift = shorter response.
# Prevents the AI from giving detailed off-topic answers while "redirecting".
MAX_TOKENS_GREEN = 600    # Concise responses for fast latency (reduced from 1000)
MAX_TOKENS_YELLOW = 500   # Moderate for minor drift (gentle redirect)
MAX_TOKENS_ORANGE = 400   # Moderate-short for moderate drift (clear redirect)
MAX_TOKENS_RED = 300      # Shorter for significant drift (redirect with context)

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

        # Template mode: SentenceTransformer with rescaling for better discrimination
        self.use_rescaled_fidelity = False  # Set True when using templates
        self.st_embedding_provider = None   # SentenceTransformer provider for templates

        # MPNet provider for AI fidelity (768-dim local, replaces Mistral API for speed)
        self.mpnet_embedding_provider = None  # all-mpnet-base-v2 for AI fidelity
        self.mpnet_ai_pa_embedding = None     # AI PA embedding in mpnet space
        self.mpnet_user_pa_embedding = None   # User PA embedding in mpnet space (for GREEN zone AI fidelity)

        # PA Enrichment Service for TELOS: command handling
        self.pa_enrichment_service = None   # Lazy initialized when needed

        # Telemetric Keys - Cryptographic access control for governance data
        # Encrypts turn telemetry with session-bound keys
        self.telemetric_manager = None
        if TELEMETRIC_KEYS_AVAILABLE:
            try:
                session_id = st.session_state.get('session_id', f'beta_{id(self)}')
                self.telemetric_manager = TelemetricSessionManager(session_id)
                logger.info(f"Telemetric Keys initialized for session: {session_id}")
            except Exception as e:
                logger.warning(f"Could not initialize Telemetric Keys: {e}")
                self.telemetric_manager = None

    def _get_thresholds(self) -> dict:
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
                'red': 0.50  # FIDELITY_RED equivalent
            }

    def generate_turn_responses(self,
                               user_input: str,
                               turn_number: int,
                               sequence: Dict = None) -> Dict:
        """
        FIDELITY-FIRST response generation.

        Flow:
        1. Check for PA pending derivation (Start Fresh mode)
        2. Calculate user prompt fidelity FIRST
        3. Decide intervention level based on fidelity
        4. Generate appropriate response (governed or native)
        5. Pre-generate Steward interpretation if intervention triggered

        Args:
            user_input: The user's message
            turn_number: Current turn number
            sequence: IGNORED - kept for backward compatibility

        Returns:
            Dict containing response data with fidelity-based decision
        """
        logger.info(f"=== FIDELITY-FIRST Turn {turn_number} ===")
        logger.info(f"User input: {user_input[:100]}...")

        # ============================================================
        # START FRESH MODE: Derive PA from first message
        # ============================================================
        if st.session_state.get('pa_pending_derivation', False) and turn_number == 1:
            logger.info("🎯 START FRESH MODE: Deriving PA from first message...")
            self._derive_pa_from_first_message(user_input)
            st.session_state.pa_pending_derivation = False
            logger.info("✅ PA derived from first message - continuing with governed response")

        # Capture the "PA just shifted" flag for use in this turn
        # If true, this is the first turn after a PA shift - skip intervention openers
        pa_was_just_shifted = st.session_state.get('pa_just_shifted', False)

        # Clear the flag AFTER capturing - observation deck already showed 1.00 on prior render
        # Next turn will use actual fidelity calculations against the new PA
        if pa_was_just_shifted:
            st.session_state.pa_just_shifted = False

        # ============================================================
        # CHECK FOR TELOS COMMAND (PA Redirect)
        # ============================================================
        # Detect TELOS: or /TELOS commands for session focus pivots
        is_telos_command, new_direction = detect_telos_command(user_input)

        if is_telos_command and new_direction:
            logger.info(f"🔄 TELOS COMMAND DETECTED: {new_direction}")
            return self._handle_telos_command(new_direction, turn_number)

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
        response_data['user_fidelity'] = user_fidelity  # Raw value for calculations
        response_data['raw_similarity'] = raw_similarity
        response_data['baseline_hard_block'] = baseline_hard_block
        response_data['fidelity_level'] = self._get_fidelity_level(user_fidelity)

        # DISPLAY NORMALIZATION: Convert raw fidelity to user-expected display scale
        # For SentenceTransformer (template mode): maps 0.32→0.70, 0.25→0.60, 0.18→0.50
        # For Mistral: no change (already on display scale)
        model_type = 'sentence_transformer' if self.use_rescaled_fidelity else 'mistral'
        response_data['display_fidelity'] = normalize_fidelity_for_display(user_fidelity, model_type)

        logger.info(f"User Fidelity: {user_fidelity:.3f} ({response_data['fidelity_level']})")
        logger.info(f"Baseline Hard Block: {baseline_hard_block} (raw_sim={raw_similarity:.3f}, baseline={SIMILARITY_BASELINE})")

        # ============================================================
        # STEP 2: TWO-TIER INTERVENTION DECISION (Model-Specific Thresholds)
        # ============================================================
        # LAYER 1: Baseline Pre-Filter - catches extreme off-topic
        # LAYER 2: Zone Classification - uses model-appropriate thresholds
        #
        # CLEAN LANE APPROACH: Use raw thresholds for each embedding model.
        # SentenceTransformer (template mode): GREEN >= 0.30, YELLOW >= 0.20, etc.
        # Mistral (custom PA mode): GREEN >= 0.60, YELLOW >= 0.50, etc.
        #
        # Yellow zone is like a yellow traffic light - cautionary awareness only
        # User is free to explore but can check with Steward if curious why it triggered

        # Get model-appropriate thresholds (Clean Lane approach) - uses cached helper
        thresholds = self._get_thresholds()
        threshold_green = thresholds['green']
        threshold_yellow = thresholds['yellow']
        threshold_orange = thresholds['orange']
        threshold_red = thresholds['red']

        in_basin = user_fidelity >= threshold_orange  # Basin membership
        in_green_zone = user_fidelity >= threshold_green
        in_yellow_zone = user_fidelity >= threshold_yellow and user_fidelity < threshold_green

        # Store layer-specific results for debugging/transparency
        response_data['layer1_triggered'] = baseline_hard_block
        response_data['layer2_in_basin'] = in_basin
        response_data['in_green_zone'] = in_green_zone
        response_data['in_yellow_zone'] = in_yellow_zone

        # ============================================================
        # GOVERNANCE TRACE LOGGING - Record fidelity calculation
        # ============================================================
        if TRACE_COLLECTOR_AVAILABLE:
            try:
                session_id = st.session_state.get('session_id', f'beta_{id(self)}')
                collector = get_trace_collector(session_id=session_id)

                # Get previous fidelity for delta calculation
                previous_fidelity = None
                if hasattr(self, 'state_manager') and self.state_manager:
                    turns = getattr(self.state_manager.state, 'turns', [])
                    if turns:
                        prev_turn = turns[-1] if len(turns) > 0 else None
                        if prev_turn:
                            previous_fidelity = prev_turn.get('user_fidelity')

                # Record fidelity calculation
                collector.record_fidelity(
                    turn_number=len(self.state_manager.state.turns) + 1 if hasattr(self, 'state_manager') else 1,
                    raw_similarity=raw_similarity,
                    normalized_fidelity=response_data['display_fidelity'],
                    layer1_hard_block=baseline_hard_block,
                    layer2_outside_basin=not in_basin,
                    distance_from_pa=1.0 - raw_similarity,  # Approximation
                    in_basin=in_basin,
                    previous_fidelity=previous_fidelity,
                )
                logger.debug("Fidelity calculation recorded to governance trace")
            except Exception as e:
                logger.debug(f"Governance trace logging skipped: {e}")

        # UNIFIED DECISION: Intervene when fidelity < GREEN threshold
        # GREEN zone: Native TELOS response (no intervention)
        # YELLOW/ORANGE/RED zones: Steward intervention (therapeutic guidance)
        should_intervene = baseline_hard_block or user_fidelity < threshold_green

        if not should_intervene:
            # GREEN zone: TELOS native flow - no Steward intervention needed
            zone = "GREEN"
            logger.info(f"✅ {zone} zone (fidelity {user_fidelity:.3f} >= {threshold_green}): TELOS native flow")
            response_data['intervention_triggered'] = False
            response_data['intervention_reason'] = None
            response_data['shown_source'] = 'native'

            # Generate native response
            native_response = self._generate_native_response(user_input)
            response_data['shown_response'] = native_response
            response_data['native_response'] = native_response

            # ================================================================
            # FAST PATH: GREEN zone skips AI fidelity calculation entirely
            # ================================================================
            # In GREEN zone we ONLY need:
            # 1. User fidelity (already computed via SentenceTransformer - instant)
            # 2. LLM response (single API call - 15-20s)
            # NO AI fidelity computation needed - saves embedding call and PS math

            telos_data = {
                'response': native_response,
                'fidelity_score': None,
                'distance_from_pa': 1.0 - user_fidelity,
                'intervention_triggered': False,
                'intervention_type': None,
                'intervention_reason': None,
                'drift_detected': False,
                'in_basin': True,
                # Skip AI fidelity in GREEN zone for speed - show "---" in UI
                'ai_pa_fidelity': None,
                'primacy_state_score': None,
                'display_primacy_state': None,
                'primacy_state_condition': 'skipped_green_zone',
                'pa_correlation': None,
                'lightweight_path': True,
                'user_pa_fidelity': user_fidelity,
                'display_user_pa_fidelity': response_data['display_fidelity'],
                'fidelity_level': response_data['fidelity_level'],
            }

            response_data['telos_analysis'] = telos_data

        else:
            # YELLOW, ORANGE, or RED zone: Not fully aligned - Steward intervention required
            # Uses DYNAMIC thresholds based on embedding model (Clean Lane approach)
            # Template mode (ST): Yellow >= 0.25, Orange >= 0.18, Red < 0.18
            # Standard mode (Mistral): Yellow >= 0.50, Orange >= 0.42, Red < 0.42

            if user_fidelity >= threshold_yellow:  # Yellow zone: Minor Drift
                intervention_reason = "Minor drift from your stated purpose - Steward is gently guiding you back"
                intervention_strength = "soft"
                zone = "YELLOW"
            elif user_fidelity >= threshold_orange:  # Orange zone: Drift Detected
                intervention_reason = "Drift from your stated purpose detected - Steward is guiding you back"
                intervention_strength = "moderate"
                zone = "ORANGE"
            else:  # Red zone: Significant Drift (below threshold_orange)
                intervention_reason = "Significant drift detected - Steward intervention activated"
                intervention_strength = "strong"
                zone = "RED"
                # Offer to pivot the PA when drift is significant - maybe user wants a new direction
                response_data['offer_pivot'] = True
                response_data['suggested_pivot_direction'] = user_input  # Store what they said as potential new direction

            logger.info(f"⚠️ {zone} zone (fidelity {user_fidelity:.3f}): {intervention_strength} (thresholds: Y>={threshold_yellow}, O>={threshold_orange})")

            response_data['intervention_triggered'] = True
            response_data['intervention_reason'] = intervention_reason
            response_data['intervention_strength'] = intervention_strength
            response_data['shown_source'] = 'steward'  # STEWARD is the intervention personality

            # ============================================================
            # GOVERNANCE TRACE LOGGING - Record intervention
            # ============================================================
            if TRACE_COLLECTOR_AVAILABLE:
                try:
                    session_id = st.session_state.get('session_id', f'beta_{id(self)}')
                    collector = get_trace_collector(session_id=session_id)

                    # Map zone to InterventionLevel
                    level_map = {
                        "YELLOW": InterventionLevel.CORRECT,
                        "ORANGE": InterventionLevel.INTERVENE,
                        "RED": InterventionLevel.ESCALATE,
                    }
                    if baseline_hard_block:
                        intervention_level = InterventionLevel.HARD_BLOCK
                        trigger_reason = "hard_block"
                    else:
                        intervention_level = level_map.get(zone, InterventionLevel.INTERVENE)
                        trigger_reason = "basin_exit" if not in_basin else "drift_detected"

                    # Calculate controller strength (K_ATTRACTOR = 1.5 from constants)
                    error_signal = max(0, threshold_green - user_fidelity)
                    controller_strength = min(1.5 * error_signal, 1.0)

                    # Map strength to semantic band
                    if controller_strength < 0.45:
                        semantic_band = "minimal"
                    elif controller_strength < 0.60:
                        semantic_band = "light"
                    elif controller_strength < 0.75:
                        semantic_band = "moderate"
                    elif controller_strength < 0.85:
                        semantic_band = "firm"
                    else:
                        semantic_band = "strong"

                    collector.record_intervention(
                        turn_number=len(self.state_manager.state.turns) + 1 if hasattr(self, 'state_manager') else 1,
                        intervention_level=intervention_level,
                        trigger_reason=trigger_reason,
                        fidelity_at_trigger=response_data['display_fidelity'],
                        controller_strength=controller_strength,
                        semantic_band=semantic_band,
                        action_taken="steward_redirect",
                    )
                    logger.debug(f"Intervention recorded: {intervention_level.value} at fidelity {user_fidelity:.3f}")
                except Exception as e:
                    logger.debug(f"Governance trace intervention logging skipped: {e}")

            # ============================================================
            # STEWARD STYLING: Get granular style based on fidelity
            # ============================================================
            # Steward style varies continuously with fidelity (6 bands)
            steward_style = get_steward_style(user_fidelity, green_threshold=threshold_green)
            steward_interpolation = get_style_interpolation(user_fidelity, green_threshold=threshold_green)

            response_data['steward_style'] = {
                'band': steward_style.band,
                'band_name': steward_style.band_name,
                'tone': steward_style.tone,
                'directness': steward_interpolation['directness'],
                'urgency': steward_interpolation['urgency'],
                'opener': get_response_opener(user_fidelity, green_threshold=threshold_green),
            }
            logger.info(f"📊 Steward Style: Band {steward_style.band} ({steward_style.band_name}), Tone: {steward_style.tone}")

            # ============================================================
            # PROPORTIONAL BREVITY: Use zone-appropriate token limits
            # ============================================================
            # Drifted responses should be CONCISE redirects, not verbose engagements.
            # More drift = shorter response. This prevents giving detailed off-topic
            # answers while "redirecting".
            if zone == "YELLOW":
                max_tokens = MAX_TOKENS_YELLOW
            elif zone == "ORANGE":
                max_tokens = MAX_TOKENS_ORANGE
            else:  # RED
                max_tokens = MAX_TOKENS_RED

            logger.info(f"📏 Using max_tokens={max_tokens} for {zone} zone (proportional brevity)")

            # Generate CONCISE redirect response (pass user_fidelity for correct PS calculation)
            telos_data = self._generate_redirect_response(
                user_input=user_input,
                turn_number=turn_number,
                zone=zone,
                max_tokens=max_tokens,
                user_input_fidelity=user_fidelity
            )

            # Use TELOS math for intervention decisions - outside basin means intervention
            telos_data['intervention_triggered'] = True  # Intervention when outside basin
            telos_data['intervention_reason'] = intervention_reason
            # user_pa_fidelity is now set inside _generate_telos_response() for correct PS calculation
            telos_data['display_user_pa_fidelity'] = response_data['display_fidelity']  # Normalized for UI
            telos_data['fidelity_level'] = response_data['fidelity_level']
            telos_data['in_basin'] = False  # Outside basin

            # BEHAVIORAL FIDELITY: Compute AI Fidelity based on intervention behavior
            # Instead of comparing AI response to User PA (which is misleading during interventions),
            # we measure how well the AI's intervention matches expected intervention patterns.
            # This implements the "Behavioral PA" concept from user insight.
            ai_response_text = telos_data.get('response', '')
            if ai_response_text:
                # Compute behavioral fidelity: how well does AI response match intervention exemplars?
                behavioral_fidelity = compute_behavioral_fidelity(
                    ai_response=ai_response_text,
                    user_fidelity=response_data['display_fidelity']  # Use display fidelity for band selection
                )
                expected_band = get_behavioral_fidelity_band(response_data['display_fidelity'])
                logger.info(f"📊 Behavioral Fidelity: {behavioral_fidelity:.3f} (band={expected_band})")
            else:
                # Fallback if no response (shouldn't happen)
                behavioral_fidelity = 0.70  # Default to GREEN threshold
                logger.warning("⚠️ No AI response text for behavioral fidelity calculation")

            telos_data['ai_pa_fidelity'] = behavioral_fidelity

            # Recalculate Primacy State with computed AI Fidelity
            # IMPORTANT: Use DISPLAY-normalized user fidelity to match what user sees in Alignment Lens
            epsilon = 1e-10
            f_user_display = response_data['display_fidelity']  # Display-normalized user fidelity (e.g., 0.58)
            f_ai = behavioral_fidelity  # AI's behavioral fidelity (computed, not hardcoded)
            telos_data['primacy_state_score'] = (2 * f_user_display * f_ai) / (f_user_display + f_ai + epsilon)
            telos_data['display_primacy_state'] = telos_data['primacy_state_score']  # Already in display range
            logger.info(f"🔧 Behavioral AI Fidelity: F_user_display={f_user_display:.3f}, F_ai={f_ai:.3f}, PS={telos_data['primacy_state_score']:.3f}")

            response_data['telos_analysis'] = telos_data

            # ============================================================
            # STEWARD DIFFERENTIATION: Apply opener to make intervention feel distinct
            # ============================================================
            raw_response = telos_data.get('response', '')
            steward_opener = response_data['steward_style'].get('opener', '')

            # Prefix response with Steward opener to make intervention feel different
            # BUT skip if PA was just shifted - user explicitly chose to change focus
            if steward_opener and raw_response and not pa_was_just_shifted:
                # Add opener as first line, followed by the LLM response
                response_data['shown_response'] = f"{steward_opener}\n\n{raw_response}"
                logger.info(f"📣 Steward opener applied: '{steward_opener}'")
            else:
                response_data['shown_response'] = raw_response
                if pa_was_just_shifted:
                    logger.info("⏭️ Skipped steward opener - PA was just shifted")

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

        TEMPLATE MODE (SentenceTransformer + Rescaling):
        - Uses SentenceTransformer for better off-topic discrimination
        - Applies rescaling to map narrow score range to TELOS fidelity range
        - Formula: fidelity = 0.25 + raw_score * 1.8, clamped to [0, 1]

        STANDARD MODE (Mistral):
        - LAYER 1: Baseline Pre-Filter (raw_sim < 0.50 triggers HARD_BLOCK)
        - LAYER 2: TELOS Primacy State (fidelity = raw cosine similarity)

        Args:
            user_input: The user's message

        Returns:
            tuple: (fidelity, raw_similarity, baseline_hard_block)
            - fidelity: Processed fidelity score (0.0 to 1.0)
            - raw_similarity: Raw cosine similarity (for logging)
            - baseline_hard_block: True if extreme off-topic detected
        """
        try:
            # Ensure TELOS engine is initialized
            if not self.telos_engine:
                self._initialize_telos_engine()

            # ================================================================
            # PA SHIFT FIX: Load cached ST mode & embedding from session state
            # After a PA shift, a new BetaResponseManager instance is created.
            # We need to restore the SentenceTransformer mode and embedding
            # from session state to ensure fidelity is calculated against the
            # new PA, not the old Mistral embeddings.
            # ================================================================
            if 'use_rescaled_fidelity_mode' in st.session_state and st.session_state.use_rescaled_fidelity_mode:
                self.use_rescaled_fidelity = True
                # Initialize ST provider if needed - uses cached version to avoid expensive model reloading
                if not self.st_embedding_provider:
                    from telos_purpose.core.embedding_provider import get_cached_minilm_provider
                    self.st_embedding_provider = get_cached_minilm_provider()
                    logger.info(f"   SentenceTransformer (cached): {self.st_embedding_provider.dimension} dims")
                # Load cached PA embedding
                if not hasattr(self, 'st_user_pa_embedding') and 'cached_st_user_pa_embedding' in st.session_state:
                    self.st_user_pa_embedding = st.session_state.cached_st_user_pa_embedding
                    logger.info(f"   Loaded cached ST PA embedding from session state: {len(self.st_user_pa_embedding)} dims")

            # ================================================================
            # TEMPLATE MODE: SentenceTransformer + Raw Thresholds (Clean Lane)
            # ================================================================
            # LEAN SIX SIGMA: No rescaling. Use raw thresholds calibrated for ST.
            # ST thresholds: GREEN >= 0.32, YELLOW >= 0.25, ORANGE >= 0.18, RED < 0.18
            if self.use_rescaled_fidelity and self.st_embedding_provider and hasattr(self, 'st_user_pa_embedding'):
                from telos_purpose.core.constants import ST_FIDELITY_GREEN, ST_FIDELITY_RED

                # Embed user input with SentenceTransformer
                user_embedding = np.array(self.st_embedding_provider.encode(user_input))

                # Calculate raw cosine similarity
                raw_similarity = self._cosine_similarity(user_embedding, self.st_user_pa_embedding)

                # CLEAN LANE: Use raw similarity directly as fidelity
                # No rescaling - thresholds are calibrated for ST output range
                fidelity = raw_similarity

                # Hard block for very off-topic (below RED threshold)
                baseline_hard_block = raw_similarity < ST_FIDELITY_RED

                if baseline_hard_block:
                    logger.warning(f"TEMPLATE MODE HARD_BLOCK: raw_sim={raw_similarity:.3f} < {ST_FIDELITY_RED}")
                else:
                    logger.info(f"Template Mode PASS: raw_sim={raw_similarity:.3f}")

                logger.info(f"📊 TEMPLATE MODE Fidelity (raw): {fidelity:.3f}")

                return (fidelity, raw_similarity, baseline_hard_block)

            # ================================================================
            # STANDARD MODE: Use local SentenceTransformer for FAST embedding
            # ================================================================
            # PERFORMANCE FIX: Use local ST embedding instead of Mistral API
            # Mistral API takes 10-20s per call, ST takes <1s locally.
            # We use ST for user input embedding but keep Mistral for PA setup
            # (PA embeddings are cached and computed once per session).
            if self.st_embedding_provider and hasattr(self, 'st_user_pa_embedding') and self.st_user_pa_embedding is not None:
                # Fast path: use cached SentenceTransformer embedding
                user_embedding = np.array(self.st_embedding_provider.encode(user_input))
                raw_similarity = self._cosine_similarity(user_embedding, self.st_user_pa_embedding)
                logger.info(f"FAST PATH: ST embedding, raw_sim={raw_similarity:.3f}")
            elif not self.embedding_provider or self.user_pa_embedding is None:
                logger.warning("Embedding provider or PA not initialized - returning default fidelity")
                return (FIDELITY_GREEN, FIDELITY_GREEN, False)  # Default to Aligned zone if not ready
            else:
                # Fallback: use Mistral API (slower)
                user_embedding = np.array(self.embedding_provider.encode(user_input))
                raw_similarity = self._cosine_similarity(user_embedding, self.user_pa_embedding)
                logger.info(f"SLOW PATH: Mistral API, raw_sim={raw_similarity:.3f}")

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
        """Get human-readable fidelity level using dynamic thresholds."""
        # Use model-appropriate thresholds via cached helper
        thresholds = self._get_thresholds()
        t_green, t_yellow, t_orange = thresholds['green'], thresholds['yellow'], thresholds['orange']

        if fidelity >= t_green:
            return "green"
        elif fidelity >= t_yellow:
            return "yellow"
        elif fidelity >= t_orange:
            return "orange"
        else:
            return "red"

    def _generate_telos_response(self, user_input: str, turn_number: int, user_input_fidelity: float = None) -> Dict:
        """
        Generate TELOS response with ACTIVE governance.

        Args:
            user_input: User's message
            turn_number: Current turn
            user_input_fidelity: Pre-calculated USER INPUT to User PA similarity (F_user)
                                 This is deterministic and must be passed in from caller.

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
                'drift_detected': result.get('telic_fidelity', 1.0) < (ST_FIDELITY_YELLOW if self.use_rescaled_fidelity else FIDELITY_YELLOW),
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
            # Log PS calculation prerequisites
            logger.debug(f"📊 PS Calculation Check: ps_calculator={self.ps_calculator is not None}, user_pa={self.user_pa_embedding is not None}, ai_pa={self.ai_pa_embedding is not None}")

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
                        # CRITICAL: Use user_input_fidelity (passed from caller) for F_user, NOT ps_metrics.f_user
                        # ps_metrics.f_user = RESPONSE to User PA (not what we want for user fidelity)
                        # user_input_fidelity = USER INPUT to User PA (deterministic, correct value)
                        if user_input_fidelity is not None:
                            telos_data['user_pa_fidelity'] = user_input_fidelity
                        else:
                            # Fallback only if not passed (shouldn't happen in normal flow)
                            logger.warning("⚠️ user_input_fidelity not passed to _generate_telos_response - using ps_metrics.f_user as fallback")
                            telos_data['user_pa_fidelity'] = ps_metrics.f_user

                        # AI fidelity correctly uses response embedding to AI PA
                        telos_data['ai_pa_fidelity'] = ps_metrics.f_ai
                        telos_data['pa_correlation'] = ps_metrics.rho_pa

                        # Calculate PS using the CORRECT F_user (USER INPUT fidelity from caller)
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

                        # Calculate display-normalized Primacy State for UI consistency
                        # Raw user fidelity needs normalization for SentenceTransformer
                        model_type = 'sentence_transformer' if self.use_rescaled_fidelity else 'mistral'
                        display_user_fidelity = normalize_fidelity_for_display(displayed_f_user, model_type)
                        display_primacy_state = (2 * display_user_fidelity * f_ai) / (display_user_fidelity + f_ai + epsilon)
                        telos_data['display_primacy_state'] = display_primacy_state

                        # Log Primacy State metrics
                        logger.debug(f"📊 PS Metrics: F_user={displayed_f_user:.3f}→{display_user_fidelity:.3f}, F_AI={f_ai:.3f}, PS={corrected_ps:.3f}→{display_primacy_state:.3f}, ρ_PA={rho_pa:.3f}, condition={ps_metrics.condition}")

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
        Generate native LLM response without TELOS governance but WITH PA-aware system prompt.

        The system prompt is derived from the user's PA to provide contextual guidance
        while keeping the response natural and conversational.

        Args:
            user_input: User's message

        Returns:
            Native response string
        """
        try:
            # Use CACHED LLM client for performance (avoids HTTP connection overhead per turn)
            from telos_purpose.llm_clients.mistral_client import get_cached_mistral_client

            client = get_cached_mistral_client()

            # Build PA-aware system prompt
            system_prompt = self._build_system_prompt()

            # Start with system prompt
            conversation = [{'role': 'system', 'content': system_prompt}]

            # Add conversation history
            conversation.extend(self._get_conversation_history())
            conversation.append({'role': 'user', 'content': user_input})

            response = client.generate(
                messages=conversation,
                max_tokens=MAX_TOKENS_GREEN,  # Use constant for consistent response length
                temperature=0.7
            )

            return response

        except Exception as e:
            logger.error(f"Error generating native response: {e}")
            return "I understand you're testing the system. How can I help you explore TELOS governance?"

    def _build_system_prompt(self) -> str:
        """
        Build a PA-aware system prompt that guides response style without TELOS intervention.

        Returns:
            System prompt string incorporating user's PA
        """
        # Get PA from session state
        pa_data = st.session_state.get('primacy_attractor', {})

        purpose_raw = pa_data.get('purpose', 'General assistance')
        scope_raw = pa_data.get('scope', 'Open discussion')
        # Convert lists to strings (templates store purpose/scope as lists)
        purpose = ' '.join(purpose_raw) if isinstance(purpose_raw, list) else purpose_raw
        scope = ' '.join(scope_raw) if isinstance(scope_raw, list) else scope_raw
        boundaries = pa_data.get('boundaries', [])
        style = pa_data.get('style', '')

        # Format boundaries as text
        boundaries_text = ""
        if boundaries:
            if isinstance(boundaries, list):
                boundaries_text = "; ".join(boundaries)
            else:
                boundaries_text = str(boundaries)

        # Build the system prompt
        system_prompt = f"""You are a helpful AI assistant. The user has established the following context for this conversation:

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

        return system_prompt

    def _generate_redirect_response(
        self,
        user_input: str,
        turn_number: int,
        zone: str,
        max_tokens: int,
        user_input_fidelity: float
    ) -> Dict:
        """
        Generate a PROPORTIONALLY-GOVERNED redirect response for drifted topics.

        This method bridges mathematical governance with semantic prompts.
        The proportional controller formula determines intervention strength,
        which then shapes how the LLM redirect prompt is constructed.

        Key principle: Math governs semantics
        - Intervention strength = K_attractor * error_signal (continuous 0.0-1.0)
        - Prompt language interpolates based on exact strength, not just zone
        - A fidelity of 0.59 produces noticeably different redirect than 0.51

        Args:
            user_input: The user's message
            turn_number: Current turn number
            zone: Fidelity zone (YELLOW, ORANGE, RED) - for logging/compatibility
            max_tokens: Maximum response length
            user_input_fidelity: Pre-calculated fidelity score (continuous value)

        Returns:
            Dict with redirect response and intervention metrics
        """
        try:
            # Use CACHED LLM client for performance (avoids HTTP connection overhead per turn)
            from telos_purpose.llm_clients.mistral_client import get_cached_mistral_client
            client = get_cached_mistral_client()

            # Get PA for context
            pa_data = st.session_state.get('primacy_attractor', {})
            purpose_raw = pa_data.get('purpose', 'your stated goals')
            # Convert list to string (templates store purpose as lists)
            purpose = ' '.join(purpose_raw) if isinstance(purpose_raw, list) else purpose_raw

            # =================================================================
            # PROPORTIONAL CONTROL: Calculate intervention strength from math
            # Formula: strength = K_attractor * error_signal
            # Where: error_signal = 1.0 - fidelity, K_attractor = 1.5
            # =================================================================
            K_ATTRACTOR = 1.5  # Same gain as proportional_controller.py
            error_signal = 1.0 - user_input_fidelity
            intervention_strength = min(K_ATTRACTOR * error_signal, 1.0)

            logger.info(f"📐 Proportional Control:")
            logger.info(f"   Fidelity: {user_input_fidelity:.3f}")
            logger.info(f"   Error signal: {error_signal:.3f}")
            logger.info(f"   Intervention strength: {intervention_strength:.3f} (K={K_ATTRACTOR})")

            # Build PROPORTIONALLY-GOVERNED redirect prompt
            # Pass fidelity for continuous interpolation, not just zone
            redirect_prompt = self._build_redirect_prompt(
                fidelity=user_input_fidelity,
                strength=intervention_strength,
                purpose=purpose,
                zone=zone  # For logging/compatibility
            )

            # Build conversation WITH history (needed for context on follow-up questions)
            # FIX: Include conversation history so LLM knows what the user is referring to
            conversation = [{'role': 'system', 'content': redirect_prompt}]

            # Add conversation history for context (e.g., "what recipes?" needs prior context)
            conversation_history = self._get_conversation_history()
            conversation.extend(conversation_history)

            conversation.append({'role': 'user', 'content': user_input})

            logger.info(f"🔄 Generating redirect response (strength={intervention_strength:.2f}, zone={zone}, max_tokens={max_tokens})")

            response = client.generate(
                messages=conversation,
                max_tokens=max_tokens,
                temperature=0.7
            )

            logger.info(f"📝 Redirect response generated: {len(response)} chars")

            # Compute AI Fidelity: embed intervention response and compare to User PA
            # This measures: "How well does this intervention serve the user's stated purpose?"
            ai_fidelity = None
            primacy_state = None
            if self.st_embedding_provider and self.st_user_pa_embedding is not None:
                # Embed the intervention response using SentenceTransformer MiniLM
                response_embedding = np.array(self.st_embedding_provider.encode(response))

                # AI Fidelity = cosine similarity to User PA (same as GREEN zone calculation)
                raw_ai_fidelity = self._cosine_similarity(response_embedding, self.st_user_pa_embedding)

                # Rescale using same formula as GREEN zone
                from telos_purpose.core.embedding_provider import rescale_sentence_transformer_fidelity
                ai_fidelity = rescale_sentence_transformer_fidelity(raw_ai_fidelity)

                logger.info(f"🔧 Intervention AI Fidelity: raw={raw_ai_fidelity:.3f} → rescaled={ai_fidelity:.3f}")

                # Calculate Primacy State using harmonic mean
                epsilon = 1e-10
                primacy_state = (2 * user_input_fidelity * ai_fidelity) / (user_input_fidelity + ai_fidelity + epsilon)
                logger.info(f"📊 Intervention PS: F_user={user_input_fidelity:.3f}, F_ai={ai_fidelity:.3f}, PS={primacy_state:.3f}")

            # Build telos_data with proportional control metrics
            telos_data = {
                'response': response,
                'fidelity_score': user_input_fidelity,
                'distance_from_pa': error_signal,
                'intervention_triggered': True,
                'intervention_type': f'proportional_redirect',
                'intervention_reason': f'Strength {intervention_strength:.2f} redirect ({zone} zone)',
                'drift_detected': True,
                'in_basin': False,
                'user_pa_fidelity': user_input_fidelity,
                'ai_pa_fidelity': ai_fidelity,  # Now computed via cosine similarity to User PA
                'primacy_state_score': primacy_state,  # Now computed properly
                'redirect_zone': zone,
                'intervention_strength': intervention_strength,  # NEW: Track proportional strength
                'error_signal': error_signal,  # NEW: Track error signal
                'max_tokens_used': max_tokens,
            }

            return telos_data

        except Exception as e:
            logger.error(f"Error generating redirect response: {e}")
            # Fallback to a simple redirect
            fallback = "Let's refocus on what you came here to accomplish. How can I help you with your original goals?"
            return {
                'response': fallback,
                'fidelity_score': user_input_fidelity,
                'error': str(e),
                'user_pa_fidelity': user_input_fidelity,
            }

    def _build_redirect_prompt(
        self,
        fidelity: float,
        strength: float,
        purpose: str,
        zone: str = None  # For logging/compatibility
    ) -> str:
        """
        Build a PROPORTIONALLY-GOVERNED redirect prompt using the Semantic Interpreter.

        The Interpreter is the bridge between mathematical governance and semantic output.
        Two focal points:
          1. Fidelity Value - where we are on the alignment spectrum
          2. Purpose - what we're maintaining (the semantic anchor)

        The interpreter translates these into concrete linguistic specifications
        that the LLM can execute deterministically - no abstract tone words.

        Args:
            fidelity: Raw fidelity score (continuous 0.0-1.0)
            strength: Calculated intervention strength (for logging)
            purpose: User's established purpose
            zone: Fidelity zone for logging (YELLOW, ORANGE, RED)

        Returns:
            System prompt with concrete linguistic specifications
        """
        # Use the Semantic Interpreter to get concrete linguistic specs
        spec = semantic_interpret(fidelity, purpose)
        exemplar = get_exemplar(spec.strength, purpose)

        # Get the linguistic specification block from the interpreter
        linguistic_spec = spec.to_prompt_block(purpose)

        # Single, consolidated prompt - no redundant purpose mentions
        return f"""You are Steward. User's goal: "{purpose}"

{linguistic_spec}

MATCH THIS STYLE:
{exemplar}

Keep it to 2-3 sentences. Sound human, not robotic."""

    def _compute_telos_metrics_lightweight(
        self, user_input: str, response: str, user_fidelity: float
    ) -> Dict:
        """
        Compute TELOS metrics for GREEN/YELLOW zones with PS calculation.

        Used for GREEN/YELLOW zones where intervention isn't needed.
        This path computes AI Fidelity and Primacy State for user visibility
        even though no intervention is triggered.

        GREEN/YELLOW = Native LLM response + full fidelity metrics
        ORANGE/RED = Full TELOS with intervention (handled elsewhere)

        Args:
            user_input: The user's message
            response: The already-generated native response
            user_fidelity: Pre-calculated user fidelity score

        Returns:
            Dict with TELOS metrics including AI Fidelity and PS
        """
        logger.info("📊 Computing TELOS metrics for GREEN/YELLOW zone (no intervention)")

        # Initialize with defaults
        ai_fidelity = None
        primacy_state = None
        display_primacy_state = None  # Display-normalized PS for UI consistency
        pa_correlation = None
        ps_condition = 'not_computed'

        # Compute AI Fidelity and Primacy State if we have the necessary components
        try:
            # CONSISTENCY FIX: Use same embedding model (MiniLM 384d) for BOTH fidelities
            # to ensure comparable cosine similarity measurements in the same embedding space.
            #
            # GREEN ZONE AI FIDELITY: Measures how well AI response serves User's purpose
            # This is different from intervention zones which use behavioral fidelity
            # (similarity to intervention exemplars)
            if (self.st_embedding_provider and
                self.st_user_pa_embedding is not None and
                response):

                # Embed the response using SentenceTransformer MiniLM (same as User Fidelity)
                response_embedding = np.array(self.st_embedding_provider.encode(response))

                # AI Fidelity = cosine similarity of response to USER PA (both in MiniLM space)
                # This measures: "How well does this response serve the user's stated purpose?"
                raw_ai_fidelity = self._cosine_similarity(response_embedding, self.st_user_pa_embedding)

                # RESCALE: SentenceTransformer MiniLM produces low raw scores
                # that need rescaling to match expected display range (0.70+ = GREEN)
                ai_fidelity = rescale_sentence_transformer_fidelity(raw_ai_fidelity)
                logger.info(f"🔧 AI Fidelity: raw={raw_ai_fidelity:.3f} → rescaled={ai_fidelity:.3f}")

                # Calculate Primacy State using harmonic mean formula
                # PS = (2 * F_user * F_ai) / (F_user + F_ai)
                epsilon = 1e-10  # Prevent division by zero
                if user_fidelity is not None and ai_fidelity is not None:
                    primacy_state = (2 * user_fidelity * ai_fidelity) / (user_fidelity + ai_fidelity + epsilon)
                    ps_condition = 'computed'

                    logger.info(f"📊 PS Metrics: F_user={user_fidelity:.3f}, F_ai={ai_fidelity:.3f}, PS={primacy_state:.3f}")

                    # Calculate display-normalized Primacy State for UI consistency
                    # Both fidelities now use same model (MiniLM 384d) for consistent measurements
                    # and have already been rescaled, so use the rescaled values directly
                    model_type = 'sentence_transformer' if self.use_rescaled_fidelity else 'mistral'
                    display_user_fidelity = normalize_fidelity_for_display(user_fidelity, model_type)
                    # AI fidelity is already rescaled from MiniLM, use directly
                    display_primacy_state = (2 * display_user_fidelity * ai_fidelity) / (display_user_fidelity + ai_fidelity + epsilon)

                # NOTE: AI Fidelity now uses consistent MiniLM embeddings (384d)
                # matching User Fidelity for comparable cosine similarity measurements

        except Exception as e:
            logger.warning(f"⚠️ Could not compute PS metrics in lightweight path: {e}")
            # Continue with None values - UI will show "---"

        # Calculate display-normalized user fidelity for UI consistency
        model_type = 'sentence_transformer' if self.use_rescaled_fidelity else 'mistral'
        display_user_fidelity_value = normalize_fidelity_for_display(user_fidelity, model_type) if user_fidelity is not None else None

        telos_data = {
            'response': response,
            'fidelity_score': None,  # NOT user_fidelity - that was causing AI Fidelity to show User Fidelity
            'distance_from_pa': 1.0 - user_fidelity,
            'intervention_triggered': False,
            'intervention_type': None,
            'intervention_reason': '',
            'drift_detected': user_fidelity < (ST_FIDELITY_YELLOW if self.use_rescaled_fidelity else FIDELITY_YELLOW),
            'in_basin': True,
            # NOW COMPUTED for all zones to give user full visibility
            'ai_pa_fidelity': ai_fidelity,
            'primacy_state_score': primacy_state,
            'display_primacy_state': display_primacy_state,  # Display-normalized PS for UI
            'primacy_state_condition': ps_condition,
            'pa_correlation': pa_correlation,
            'lightweight_path': True,  # Flag to indicate this was the native response path
            'user_pa_fidelity': user_fidelity,  # Raw value for Steward
            'display_user_pa_fidelity': display_user_fidelity_value,  # Display-normalized for UI threshold checks
        }

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

        # Explain fidelity using dynamic thresholds via cached helper
        thresholds = self._get_thresholds()
        t_green, t_yellow, t_orange = thresholds['green'], thresholds['yellow'], thresholds['orange']

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

        # ================================================================
        # TELEMETRIC KEYS: Encrypt governance telemetry with session-bound key
        # ================================================================
        if self.telemetric_manager:
            try:
                telos_data = data.get('telos_analysis', {})
                import time

                # Build turn telemetry (privacy-preserving: metrics only, no content)
                turn_telemetry = {
                    'turn_number': turn_number,
                    'fidelity_score': telos_data.get('user_pa_fidelity') or telos_data.get('fidelity_score') or 0.0,
                    'distance_from_pa': telos_data.get('distance_from_pa', 0.0),
                    'intervention_triggered': telos_data.get('intervention_triggered', False),
                    'in_basin': telos_data.get('in_basin', True),
                    'ai_pa_fidelity': telos_data.get('ai_pa_fidelity'),
                    'primacy_state_score': telos_data.get('primacy_state_score'),
                    'timestamp': time.time()
                }

                # Encrypt and rotate key
                encrypted_delta = self.telemetric_manager.process_turn(turn_telemetry)
                logger.info(f"Telemetric Keys: Encrypted turn {turn_number} ({len(encrypted_delta.ciphertext)} bytes)")

                # Store encrypted delta in session state for export
                if 'encrypted_governance_deltas' not in st.session_state:
                    st.session_state.encrypted_governance_deltas = []
                st.session_state.encrypted_governance_deltas.append(encrypted_delta.to_dict())

            except Exception as e:
                logger.warning(f"Telemetric Keys encryption failed for turn {turn_number}: {e}")

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
            from telos_purpose.core.embedding_provider import (
                MistralEmbeddingProvider,
                SentenceTransformerProvider
            )
            from telos_purpose.llm_clients.mistral_client import MistralClient
            from telos_purpose.core.primacy_state import PrimacyStateCalculator

            # Check if using a template (pre-established PA)
            # Templates use SentenceTransformer with raw thresholds (Clean Lane approach)
            selected_template = st.session_state.get('selected_template', None)
            self.use_rescaled_fidelity = selected_template is not None

            if self.use_rescaled_fidelity:
                logger.debug("📊 TEMPLATE MODE: Using SentenceTransformer with raw thresholds (Clean Lane)")

            # Read PA from session state (established via BETA questionnaire)
            # PAOnboarding component saves to 'primacy_attractor' and 'pa_established'
            pa_data = st.session_state.get('primacy_attractor', None)
            pa_established = st.session_state.get('pa_established', False)

            logger.debug(f"🔍 BETA TELOS Init: pa_exists={pa_data is not None}, pa_established={pa_established}")

            if pa_data and pa_established:
                # Use established PA from questionnaire
                # Handle both string and list formats (templates use lists)
                purpose_raw = pa_data.get('purpose', 'General assistance')
                scope_raw = pa_data.get('scope', 'Open discussion')

                # Convert lists to strings for intent detection and embedding
                purpose_str = ' '.join(purpose_raw) if isinstance(purpose_raw, list) else purpose_raw
                scope_str = ' '.join(scope_raw) if isinstance(scope_raw, list) else scope_raw

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

            # For template mode: Initialize SentenceTransformer for fidelity calculation
            # SentenceTransformer has better discrimination for off-topic detection
            # Uses cached provider to avoid expensive model reloading on every rerun
            if self.use_rescaled_fidelity:
                logger.info("🔧 Getting cached SentenceTransformer for template mode fidelity...")
                from telos_purpose.core.embedding_provider import get_cached_minilm_provider
                self.st_embedding_provider = get_cached_minilm_provider()
                logger.info(f"   SentenceTransformer (cached): {self.st_embedding_provider.dimension} dims")

            # Initialize MPNet for AI Fidelity (768-dim, local, replaces Mistral API for speed)
            # all-mpnet-base-v2 provides 2x the dimensions of MiniLM with no API latency
            # Uses cached provider to avoid expensive model reloading on every rerun
            logger.info("🔧 Getting cached MPNet for AI fidelity (latency optimization)...")
            from telos_purpose.core.embedding_provider import get_cached_mpnet_provider
            self.mpnet_embedding_provider = get_cached_mpnet_provider()
            logger.info(f"   MPNet (cached): {self.mpnet_embedding_provider.dimension} dims (local)")

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
            #
            # IMPORTANT: We track PA identity to invalidate cache when PA changes.
            # This prevents stale embeddings when user switches templates.
            # =================================================================
            logger.info("🔧 Setting up Dual PA for Primacy State calculation...")

            # Create PA identity hash to detect PA changes
            import hashlib
            current_pa_identity = hashlib.md5(f"{purpose_str}|{scope_str}".encode()).hexdigest()[:16]
            cached_pa_identity = st.session_state.get('cached_pa_identity', None)

            # Check if PA has changed - if so, invalidate all cached embeddings
            if cached_pa_identity and cached_pa_identity != current_pa_identity:
                logger.warning(f"⚠️ PA CHANGED: Invalidating cached embeddings")
                logger.warning(f"   Old PA hash: {cached_pa_identity}")
                logger.warning(f"   New PA hash: {current_pa_identity}")
                # Clear all cached PA embeddings
                for key in ['cached_user_pa_embedding', 'cached_ai_pa_embedding',
                            'cached_st_user_pa_embedding', 'cached_mpnet_user_pa_embedding',
                            'cached_mpnet_ai_pa_embedding', 'cached_pa_identity']:
                    if key in st.session_state:
                        del st.session_state[key]

            # Check if PA embeddings are already cached in session state AND match current PA
            if ('cached_user_pa_embedding' in st.session_state and
                'cached_ai_pa_embedding' in st.session_state and
                st.session_state.get('cached_pa_identity') == current_pa_identity):
                # Use cached embeddings for deterministic fidelity calculations
                self.user_pa_embedding = st.session_state.cached_user_pa_embedding
                self.ai_pa_embedding = st.session_state.cached_ai_pa_embedding
                logger.info("   ✅ Using CACHED PA embeddings from session state (deterministic)")
                logger.info(f"   User PA embedded: {len(self.user_pa_embedding)} dims (cached)")
                logger.info(f"   AI PA embedded: {len(self.ai_pa_embedding)} dims (cached)")

                # For template mode: Create SentenceTransformer PA centroid from example queries
                if self.use_rescaled_fidelity and self.st_embedding_provider:
                    if 'cached_st_user_pa_embedding' in st.session_state:
                        self.st_user_pa_embedding = st.session_state.cached_st_user_pa_embedding
                        logger.info(f"   ✅ Using cached SentenceTransformer PA centroid: {len(self.st_user_pa_embedding)} dims")
                    else:
                        # Check if template has example queries for centroid-based embedding
                        example_queries = selected_template.get('example_queries', []) if selected_template else []
                        if example_queries:
                            # Import universal lane expansion weight and cached centroids
                            from config.pa_templates import UNIVERSAL_EXPANSION_WEIGHT
                            from telos_purpose.core.embedding_provider import (
                                get_cached_universal_lane_centroid,
                                get_cached_template_domain_centroid
                            )

                            # PERFORMANCE: Use cached domain centroid (saves ~8-15 embedding calls)
                            template_id = selected_template.get('id', None) if selected_template else None
                            domain_centroid = get_cached_template_domain_centroid(template_id) if template_id else None

                            if domain_centroid is None:
                                # Fallback: compute domain centroid if not cached
                                logger.info(f"   📊 Creating PA centroid from {len(example_queries)} example queries...")
                                example_embeddings = [np.array(self.st_embedding_provider.encode(ex)) for ex in example_queries]
                                domain_centroid = np.mean(example_embeddings, axis=0)
                            else:
                                logger.info(f"   📊 Using cached domain centroid for template '{template_id}'...")

                            # PERFORMANCE: Use cached universal centroid (saves ~15 embedding calls)
                            logger.info(f"   📊 Using cached universal lane centroid...")
                            universal_centroid = get_cached_universal_lane_centroid()

                            # Mix domain-specific and universal centroids (weighted average)
                            # UNIVERSAL_EXPANSION_WEIGHT = 0.3 means 30% universal, 70% domain
                            combined_centroid = (1 - UNIVERSAL_EXPANSION_WEIGHT) * domain_centroid + UNIVERSAL_EXPANSION_WEIGHT * universal_centroid
                            self.st_user_pa_embedding = combined_centroid / np.linalg.norm(combined_centroid)  # Normalize
                            logger.info(f"   📦 Created combined PA centroid: {len(self.st_user_pa_embedding)} dims (domain: {1-UNIVERSAL_EXPANSION_WEIGHT:.0%}, universal: {UNIVERSAL_EXPANSION_WEIGHT:.0%})")
                        else:
                            # Fallback to abstract PA text
                            user_pa_text = f"Purpose: {purpose_str}. Scope: {scope_str}."
                            self.st_user_pa_embedding = np.array(self.st_embedding_provider.encode(user_pa_text))
                            logger.info(f"   📦 Created text-based PA embedding: {len(self.st_user_pa_embedding)} dims")
                        st.session_state.cached_st_user_pa_embedding = self.st_user_pa_embedding

                # MPNet embeddings for fast AI fidelity (replaces Mistral API)
                # Two embeddings: User PA (for GREEN zone) and AI PA (for intervention zones)
                if self.mpnet_embedding_provider:
                    # User PA embedding in MPNet space (for GREEN zone AI Fidelity)
                    if 'cached_mpnet_user_pa_embedding' in st.session_state:
                        self.mpnet_user_pa_embedding = st.session_state.cached_mpnet_user_pa_embedding
                        logger.info(f"   ✅ Using cached MPNet User PA: {len(self.mpnet_user_pa_embedding)} dims")
                    else:
                        # Create MPNet User PA embedding
                        user_pa_text = f"Purpose: {purpose_str}. Scope: {scope_str}."
                        self.mpnet_user_pa_embedding = np.array(self.mpnet_embedding_provider.encode(user_pa_text))
                        st.session_state.cached_mpnet_user_pa_embedding = self.mpnet_user_pa_embedding
                        logger.info(f"   📦 Created MPNet User PA embedding: {len(self.mpnet_user_pa_embedding)} dims")

                    # AI PA embedding in MPNet space (for intervention zones)
                    if 'cached_mpnet_ai_pa_embedding' in st.session_state:
                        self.mpnet_ai_pa_embedding = st.session_state.cached_mpnet_ai_pa_embedding
                        logger.info(f"   ✅ Using cached MPNet AI PA: {len(self.mpnet_ai_pa_embedding)} dims")
                    else:
                        # Need to create MPNet AI PA embedding
                        detected_intent = self._detect_intent_from_purpose(purpose_str)
                        role_action = INTENT_TO_ROLE_MAP.get(detected_intent, 'help')
                        ai_purpose = f"{role_action.capitalize()} the user as they work to: {purpose_str}"
                        ai_pa_text = f"AI Role: {ai_purpose}. Supporting scope: {scope_str}."
                        self.mpnet_ai_pa_embedding = np.array(self.mpnet_embedding_provider.encode(ai_pa_text))
                        st.session_state.cached_mpnet_ai_pa_embedding = self.mpnet_ai_pa_embedding
                        logger.info(f"   📦 Created MPNet AI PA embedding: {len(self.mpnet_ai_pa_embedding)} dims")
            else:
                # FALLBACK: Compute embeddings lazily if not cached
                # This should rarely trigger now that PA establishment computes embeddings upfront
                logger.warning("   ⚠️ PA embeddings not cached - computing lazily (fallback path)")

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
                st.session_state.cached_pa_identity = current_pa_identity  # Track which PA these embeddings belong to
                logger.info("   📦 PA embeddings CACHED in session state for deterministic calculations")
                logger.info(f"   📦 PA identity cached: {current_pa_identity}")

                # For template mode: Also create SentenceTransformer PA embedding with universal expansion
                if self.use_rescaled_fidelity and self.st_embedding_provider:
                    from config.pa_templates import UNIVERSAL_EXPANSION_WEIGHT
                    from telos_purpose.core.embedding_provider import get_cached_universal_lane_centroid

                    # Check if template has example queries for centroid-based embedding
                    example_queries = selected_template.get('example_queries', []) if selected_template else []
                    if example_queries:
                        # PERFORMANCE: Use cached domain centroid (saves ~8-15 embedding calls)
                        from telos_purpose.core.embedding_provider import get_cached_template_domain_centroid
                        template_id = selected_template.get('id', None) if selected_template else None
                        domain_centroid = get_cached_template_domain_centroid(template_id) if template_id else None

                        if domain_centroid is None:
                            # Fallback: compute domain centroid if not cached
                            logger.info(f"   📊 Creating PA centroid from {len(example_queries)} example queries...")
                            example_embeddings = [np.array(self.st_embedding_provider.encode(ex)) for ex in example_queries]
                            domain_centroid = np.mean(example_embeddings, axis=0)
                        else:
                            logger.info(f"   📊 Using cached domain centroid for template '{template_id}'...")

                        # PERFORMANCE: Use cached universal centroid (saves ~15 embedding calls)
                        logger.info(f"   📊 Using cached universal lane centroid...")
                        universal_centroid = get_cached_universal_lane_centroid()

                        # Mix domain-specific and universal centroids
                        combined_centroid = (1 - UNIVERSAL_EXPANSION_WEIGHT) * domain_centroid + UNIVERSAL_EXPANSION_WEIGHT * universal_centroid
                        self.st_user_pa_embedding = combined_centroid / np.linalg.norm(combined_centroid)
                        logger.info(f"   📦 Created combined PA centroid: {len(self.st_user_pa_embedding)} dims (domain: {1-UNIVERSAL_EXPANSION_WEIGHT:.0%}, universal: {UNIVERSAL_EXPANSION_WEIGHT:.0%})")
                    else:
                        # Fallback to text-based embedding (no example queries)
                        self.st_user_pa_embedding = np.array(self.st_embedding_provider.encode(user_pa_text))
                        logger.info(f"   📦 Created text-based PA embedding: {len(self.st_user_pa_embedding)} dims")

                    st.session_state.cached_st_user_pa_embedding = self.st_user_pa_embedding

                # Create MPNet AI PA embedding for fast AI fidelity (uses ai_pa_text from above)
                if self.mpnet_embedding_provider:
                    self.mpnet_ai_pa_embedding = np.array(self.mpnet_embedding_provider.encode(ai_pa_text))
                    st.session_state.cached_mpnet_ai_pa_embedding = self.mpnet_ai_pa_embedding
                    logger.info(f"   📦 Created MPNet AI PA embedding: {len(self.mpnet_ai_pa_embedding)} dims")

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

    def _derive_pa_from_first_message(self, user_input: str):
        """
        Derive PA from user's first message (Start Fresh mode).

        Uses PA Enrichment Service for genuine semantic extraction:
        1. LLM extracts structured PA (purpose, scope, boundaries, example_queries)
        2. Derives AI PA using dual attractor lock-on from enriched structure
        3. Computes embeddings from example_queries for semantic centroid
        4. Updates session state with derived PA

        Args:
            user_input: User's first message containing their purpose
        """
        from services.beta_dual_attractor import derive_ai_pa_from_user_pa, compute_pa_embeddings
        from services.pa_enrichment import PAEnrichmentService
        from datetime import datetime

        logger.info(f"🎯 Deriving PA from: {user_input[:100]}...")

        # Detect intent from the message
        detected_intent = self._detect_intent_from_purpose(user_input)
        logger.info(f"   Detected intent: {detected_intent}")

        # Use PA Enrichment Service for genuine semantic extraction
        enriched_pa = None
        try:
            from mistralai import Mistral
            import os
            mistral_client = Mistral(api_key=os.getenv('MISTRAL_API_KEY'))
            enrichment_service = PAEnrichmentService(mistral_client)

            logger.info("   🔧 Using PA Enrichment Service for semantic extraction...")
            enriched_pa = enrichment_service.enrich_direction(
                direction=user_input,
                current_template=None,
                conversation_context=""
            )

            if enriched_pa:
                logger.info(f"   ✅ PA enriched with {len(enriched_pa.get('example_queries', []))} example queries")
        except Exception as e:
            logger.warning(f"   ⚠️ PA enrichment failed: {e}, falling back to basic extraction")

        # Build User PA from enriched structure (or fallback)
        if enriched_pa:
            user_pa = {
                "purpose": [enriched_pa.get('purpose', user_input)],
                "scope": enriched_pa.get('scope', [f"Explore: {user_input[:80]}"]),
                "boundaries": enriched_pa.get('boundaries', [
                    "Stay focused on stated purpose",
                    "Provide helpful, relevant responses"
                ]),
                "example_queries": enriched_pa.get('example_queries', []),
                "success_criteria": f"Deep understanding of: {user_input[:80]}",
                "style": "Adaptive",
                "established_turn": 1,
                "establishment_method": "fresh_start_enriched",
                "detected_intent": detected_intent
            }
        else:
            # Fallback to basic extraction (only if enrichment fails)
            user_pa = {
                "purpose": [user_input],
                "scope": [f"Explore and understand: {user_input[:80]}"],
                "boundaries": [
                    "Stay focused on stated purpose",
                    "Provide helpful, relevant responses"
                ],
                "success_criteria": f"Help with: {user_input[:80]}",
                "style": "Adaptive",
                "established_turn": 1,
                "establishment_method": "fresh_start_basic",
                "detected_intent": detected_intent
            }

        # Derive AI PA using dual attractor lock-on
        ai_pa = derive_ai_pa_from_user_pa(user_pa)
        logger.info(f"   AI PA derived: {ai_pa.get('purpose', ['N/A'])[0][:80]}...")

        # Store both PAs in session state
        st.session_state.primacy_attractor = user_pa
        st.session_state.user_pa = user_pa
        st.session_state.ai_pa = ai_pa
        st.session_state.pa_established = True
        st.session_state.pa_establishment_time = datetime.now().isoformat()

        # Update state manager
        if 'state_manager' in st.session_state:
            st.session_state.state_manager.state.primacy_attractor = user_pa
            st.session_state.state_manager.state.user_pa_established = True
            st.session_state.state_manager.state.convergence_turn = 1
            st.session_state.state_manager.state.pa_converged = True

        # Compute embeddings for fidelity calculation
        try:
            # Use CACHED provider to avoid expensive model reloading (critical for Railway cold start)
            from telos_purpose.core.embedding_provider import get_cached_minilm_provider
            embedding_provider = get_cached_minilm_provider()
            user_embedding, ai_embedding = compute_pa_embeddings(user_pa, ai_pa, embedding_provider)

            # Cache embeddings with PA identity for change detection
            st.session_state.cached_user_pa_embedding = user_embedding
            st.session_state.cached_ai_pa_embedding = ai_embedding
            # Create PA identity hash from purpose and scope
            import hashlib
            purpose_str = ' '.join(user_pa.get('purpose', [])) if isinstance(user_pa.get('purpose'), list) else user_pa.get('purpose', '')
            scope_str = ' '.join(user_pa.get('scope', [])) if isinstance(user_pa.get('scope'), list) else user_pa.get('scope', '')
            st.session_state.cached_pa_identity = hashlib.md5(f"{purpose_str}|{scope_str}".encode()).hexdigest()[:16]
            logger.info(f"   📦 PA identity cached: {st.session_state.cached_pa_identity}")

            # Also create SentenceTransformer embedding for fidelity (template mode)
            from telos_purpose.core.embedding_provider import get_cached_minilm_provider
            st_provider = get_cached_minilm_provider()

            # Create PA centroid from example_queries if available (semantic centroid)
            example_queries = user_pa.get('example_queries', [])
            if example_queries and len(example_queries) >= 3:
                # Create centroid from example query embeddings (semantically rich)
                logger.info(f"   📊 Creating PA centroid from {len(example_queries)} example queries...")
                example_embeddings = [np.array(st_provider.encode(ex)) for ex in example_queries]
                st_embedding = np.mean(example_embeddings, axis=0)
                st_embedding = st_embedding / np.linalg.norm(st_embedding)
            else:
                # Fallback to user input (less semantic but works)
                st_embedding = np.array(st_provider.encode(user_input))
                st_embedding = st_embedding / np.linalg.norm(st_embedding)

            st.session_state.cached_st_user_pa_embedding = st_embedding

            # Enable template mode for proper thresholds
            st.session_state.use_rescaled_fidelity_mode = True
            self.use_rescaled_fidelity = True
            self.st_embedding_provider = st_provider
            self.st_user_pa_embedding = st_embedding

            logger.info(f"   ✅ Embeddings computed and cached")
            logger.info(f"      User PA: {len(user_embedding)} dims (Mistral)")
            logger.info(f"      ST PA: {len(st_embedding)} dims (SentenceTransformer)")
            if example_queries:
                logger.info(f"      Centroid source: {len(example_queries)} example queries")

        except Exception as e:
            logger.warning(f"   ⚠️ Failed to compute embeddings: {e}")

        # Force TELOS engine re-initialization on next response
        self.telos_engine = None
        self.ps_calculator = None

        logger.info("   ✅ PA derivation complete")

    def _handle_telos_command(self, new_direction: str, turn_number: int) -> Dict:
        """
        Handle TELOS: command for session focus pivot.

        This method:
        1. Uses PAEnrichmentService to transform raw direction into rich PA structure
        2. Regenerates the PA centroid from new example_queries
        3. Updates session state with new PA
        4. Returns Steward acknowledgment response

        Args:
            new_direction: User's stated new focus direction
            turn_number: Current turn number

        Returns:
            Dict containing response data with Steward acknowledgment
        """
        logger.info(f"🔄 Handling TELOS command: {new_direction}")

        # Get current PA for context
        current_template = st.session_state.get('selected_template', None)
        current_pa = st.session_state.get('primacy_attractor', {})
        previous_purpose_raw = current_pa.get('purpose', 'General session')
        # Convert list to string (templates store purpose as lists)
        previous_purpose = ' '.join(previous_purpose_raw) if isinstance(previous_purpose_raw, list) else previous_purpose_raw

        # Initialize PA enrichment service if needed
        if not self.pa_enrichment_service:
            try:
                from mistralai import Mistral
                import os
                mistral_client = Mistral(api_key=os.getenv('MISTRAL_API_KEY'))
                self.pa_enrichment_service = PAEnrichmentService(mistral_client)
                logger.info("✅ PA Enrichment Service initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize PA Enrichment Service: {e}")
                # Return error response
                return self._create_telos_error_response(turn_number, new_direction, str(e))

        # Get recent conversation context
        conversation_context = self._get_recent_context()

        # Enrich the direction into a full PA structure
        logger.info("🔧 Enriching direction via LLM...")
        enriched_pa = self.pa_enrichment_service.enrich_direction(
            direction=new_direction,
            current_template=current_template,
            conversation_context=conversation_context
        )

        if not enriched_pa:
            logger.error("❌ PA enrichment failed - could not generate structure")
            return self._create_telos_error_response(turn_number, new_direction, "Could not enrich direction")

        logger.info(f"✅ PA enriched with {len(enriched_pa.get('example_queries', []))} example queries")

        # Update session state with new PA
        new_pa = {
            'purpose': enriched_pa.get('purpose', new_direction),
            'scope': ', '.join(enriched_pa.get('scope', [])) if isinstance(enriched_pa.get('scope'), list) else enriched_pa.get('scope', ''),
            'boundaries': enriched_pa.get('boundaries', []),
            'example_queries': enriched_pa.get('example_queries', []),
            'pivot_from': previous_purpose,
            'pivot_direction': new_direction,
        }
        st.session_state['primacy_attractor'] = new_pa
        st.session_state['pa_established'] = True

        # Set flag to indicate PA just shifted - subsequent turns should skip intervention openers
        st.session_state.pa_just_shifted = True

        # Clear fidelity caches to prevent stale values from polluting display after PA shift
        if 'last_telos_calibration_values' in st.session_state:
            del st.session_state.last_telos_calibration_values
            logger.info("🧹 Cleared last_telos_calibration_values cache on PA shift")

        # Regenerate PA centroid from new example queries
        self._regenerate_pa_centroid(enriched_pa.get('example_queries', []))

        # Generate Steward acknowledgment response
        steward_response = self.pa_enrichment_service.generate_pivot_response(
            enriched_pa=enriched_pa,
            previous_focus=previous_purpose
        )

        # Add Alignment Lens reminder footer
        steward_response += "\n\n*Your Alignment Lens now reflects your updated Primacy Attractors.*"

        # Build response data
        response_data = {
            'turn_number': turn_number,
            'timestamp': datetime.now().isoformat(),
            'user_input': f"TELOS: {new_direction}",
            'governance_mode': 'telos_pivot',
            'is_telos_command': True,
            'new_direction': new_direction,
            'enriched_pa': enriched_pa,
            'response': steward_response,  # Standard key for rendering
            'shown_response': steward_response,  # Legacy/display key
            'shown_source': 'steward_pivot',
            'user_fidelity': 1.0,  # Pivot commands are always aligned by definition
            'display_fidelity': 1.0,
            'fidelity_level': 'green',
            'intervention_triggered': False,  # Pivot is not an intervention
            'is_loading': False,  # CRITICAL: Ensure input form shows after pivot
            'is_streaming': False,  # CRITICAL: Ensure input form shows after pivot
            'telos_analysis': {
                'response': steward_response,
                'fidelity_score': 1.0,
                'intervention_triggered': False,
                'in_basin': True,
                'pivot_detected': True,
            }
        }

        # Store turn data
        self._store_turn_data(turn_number, response_data)

        logger.info(f"✅ TELOS pivot complete - new focus: {enriched_pa.get('purpose', new_direction)[:50]}...")
        return response_data

    def _regenerate_pa_centroid(self, example_queries: list):
        """
        Regenerate PA centroid from new example queries.

        Args:
            example_queries: List of example queries for centroid construction
        """
        if not example_queries:
            logger.warning("⚠️ No example queries provided for centroid regeneration")
            return

        try:
            # Ensure SentenceTransformer provider is initialized - uses cached version
            if not self.st_embedding_provider:
                from telos_purpose.core.embedding_provider import get_cached_minilm_provider
                self.st_embedding_provider = get_cached_minilm_provider()
                logger.info(f"   SentenceTransformer (cached): {self.st_embedding_provider.dimension} dims")

            # Create centroid from example query embeddings
            logger.info(f"📊 Regenerating PA centroid from {len(example_queries)} example queries...")
            example_embeddings = [np.array(self.st_embedding_provider.encode(ex)) for ex in example_queries]
            domain_centroid = np.mean(example_embeddings, axis=0)

            # Normalize the centroid
            self.st_user_pa_embedding = domain_centroid / np.linalg.norm(domain_centroid)

            # Cache in session state
            st.session_state.cached_st_user_pa_embedding = self.st_user_pa_embedding

            # Enable template mode (SentenceTransformer thresholds)
            self.use_rescaled_fidelity = True
            # Cache in session state so subsequent requests use same mode
            st.session_state.use_rescaled_fidelity_mode = True

            # Clear any stale Mistral PA embeddings to force recalculation
            if 'cached_user_pa_embedding' in st.session_state:
                del st.session_state['cached_user_pa_embedding']
            if 'cached_ai_pa_embedding' in st.session_state:
                del st.session_state['cached_ai_pa_embedding']

            logger.info(f"✅ PA centroid regenerated: {len(self.st_user_pa_embedding)} dims")

        except Exception as e:
            logger.error(f"❌ Failed to regenerate PA centroid: {e}")

    def _get_recent_context(self) -> str:
        """
        Get recent conversation context for PA enrichment.

        Returns:
            String containing recent conversation snippets
        """
        history = self._get_conversation_history()
        if not history:
            return ""

        # Get last 3 exchanges
        recent = history[-6:]  # 3 user + 3 assistant messages
        context_parts = []
        for msg in recent:
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')[:200]
            context_parts.append(f"{role}: {content}")

        return "\n".join(context_parts)

    def _create_telos_error_response(self, turn_number: int, direction: str, error: str) -> Dict:
        """
        Create error response when TELOS command processing fails.

        Args:
            turn_number: Current turn number
            direction: User's attempted direction
            error: Error message

        Returns:
            Dict containing error response data
        """
        error_response = (
            f"I understand you want to shift focus to: **{direction}**\n\n"
            f"However, I encountered an issue processing this request. "
            f"Let's continue with our current focus, and you can try the TELOS command again.\n\n"
            f"*Technical note: {error}*"
        )

        response_data = {
            'turn_number': turn_number,
            'timestamp': datetime.now().isoformat(),
            'user_input': f"TELOS: {direction}",
            'governance_mode': 'telos_pivot_error',
            'is_telos_command': True,
            'error': error,
            'shown_response': error_response,
            'shown_source': 'steward_error',
            'user_fidelity': 0.5,
            'display_fidelity': 0.5,
            'fidelity_level': 'orange',
            'intervention_triggered': False,
            'telos_analysis': {
                'response': error_response,
                'error': error,
            }
        }

        self._store_turn_data(turn_number, response_data)
        return response_data