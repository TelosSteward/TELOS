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

# Import ALL thresholds from single source of truth
from telos_purpose.core.constants import (
    # Display zone thresholds (normalized 0-1 scale)
    FIDELITY_GREEN, FIDELITY_YELLOW, FIDELITY_ORANGE, FIDELITY_RED,
    # Intervention decision thresholds
    SIMILARITY_BASELINE as CONSTANTS_SIMILARITY_BASELINE,
    INTERVENTION_THRESHOLD as CONSTANTS_INTERVENTION_THRESHOLD,
    BASIN_CENTER, BASIN_TOLERANCE,
    # Model-specific raw thresholds
    ST_FIDELITY_GREEN, ST_FIDELITY_YELLOW, ST_FIDELITY_ORANGE, ST_FIDELITY_RED,
)

# Import display normalization for SentenceTransformer scores
# This maps raw ST scores to user-expected display values (0.70+ = GREEN, etc.)
from telos_purpose.core.fidelity_display import (
    normalize_fidelity_for_display,
    normalize_st_fidelity,
    normalize_ai_response_fidelity,  # AI responses vs USER_PA need different calibration
)

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
# NOTE: get_exemplar is NOT imported - exemplars were leaking into LLM output
from telos_purpose.core.semantic_interpreter import (
    interpret as semantic_interpret,
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

# =============================================================================
# ADAPTIVE CONTEXT SYSTEM - Phase-aware, pattern-classified context management
# =============================================================================
# From ADAPTIVE_CONTEXT_PROPOSAL.md: Multi-tier buffer with message type
# classification, conversation phase detection, and adaptive thresholds.
# Feature flag allows gradual rollout and A/B testing.
try:
    from telos_purpose.core.adaptive_context import (
        AdaptiveContextManager,
        AdaptiveContextResult,
        MessageType,
        ConversationPhase,
    )
    ADAPTIVE_CONTEXT_AVAILABLE = True
except ImportError:
    ADAPTIVE_CONTEXT_AVAILABLE = False

# Feature flag for Semantic Continuity Inheritance (SCI) system
# SCI is the CANONICAL fidelity measurement approach as of 2025-12-30.
# It uses measurement-based semantic similarity to the previous turn for inheritance.
# Legacy context-aware fidelity is DEPRECATED and only kept as exception fallback.
ADAPTIVE_CONTEXT_ENABLED = True

logger = logging.getLogger(__name__)

# =============================================================================
# TWO-LAYER DRIFT DETECTION ARCHITECTURE
# =============================================================================
# All thresholds imported from telos_purpose/core/constants.py (single source of truth)
#
# LAYER 1: Baseline Pre-Filter (for extreme off-topic detection)
# ----------------------------------------------------------------
# SIMILARITY_BASELINE: If raw cosine similarity < this value, content is
# extreme off-topic → trigger HARD_BLOCK immediately.
# Imported as CONSTANTS_SIMILARITY_BASELINE from constants.py
#
# LAYER 2: TELOS Primacy State Mathematics (Basin Membership)
# ----------------------------------------------------------------
# INTERVENTION_THRESHOLD: If normalized fidelity < this value, user has
# drifted outside the primacy basin → trigger intervention.
# Imported as CONSTANTS_INTERVENTION_THRESHOLD from constants.py
#
# DISPLAY ZONES: For UI color feedback (separate from intervention decision)
# GREEN >= 0.70, YELLOW 0.60-0.69, ORANGE 0.50-0.59, RED < 0.50
# Imported as FIDELITY_GREEN, FIDELITY_YELLOW, FIDELITY_ORANGE, FIDELITY_RED
# =============================================================================

# Local aliases for backward compatibility with existing code in this file
SIMILARITY_BASELINE = CONSTANTS_SIMILARITY_BASELINE  # 0.20 - Layer 1 hard block
BASIN = BASIN_CENTER                                  # 0.50 - Basin boundary
TOLERANCE = BASIN_TOLERANCE                           # 0.02 - Tolerance margin
INTERVENTION_THRESHOLD = CONSTANTS_INTERVENTION_THRESHOLD  # 0.48 - Layer 2 threshold
# Note: FIDELITY_GREEN, FIDELITY_YELLOW, FIDELITY_ORANGE, FIDELITY_RED are imported directly

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
        self.st_ai_pa_embedding = None      # SentenceTransformer AI PA embedding for AI fidelity checks

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

        # Adaptive Context System - Phase-aware, pattern-classified context management
        # Replaces simple context concatenation with multi-tier buffer + phase detection
        self.adaptive_context_manager = None
        self.adaptive_context_enabled = ADAPTIVE_CONTEXT_ENABLED and ADAPTIVE_CONTEXT_AVAILABLE
        if self.adaptive_context_enabled:
            try:
                self.adaptive_context_manager = AdaptiveContextManager()
                logger.info("AdaptiveContextManager initialized for session")
            except Exception as e:
                logger.warning(f"Could not initialize AdaptiveContextManager: {e}")
                self.adaptive_context_manager = None
                self.adaptive_context_enabled = False

        # Last adaptive context result - cached for UI display
        self.last_adaptive_context_result: Optional['AdaptiveContextResult'] = None

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
        # ============================================================
        # PA ESTABLISHMENT TURN DETECTION
        # ============================================================
        # Track if this turn is the PA establishment turn - fidelity should be 100%
        # because the user's message IS the PA definition itself.
        is_pa_establishment_turn = False

        if st.session_state.get('pa_pending_derivation', False) and turn_number == 1:
            logger.info("🎯 START FRESH MODE: Deriving PA from first message...")
            self._derive_pa_from_first_message(user_input)
            st.session_state.pa_pending_derivation = False
            is_pa_establishment_turn = True  # Mark this turn as PA establishment
            logger.info("✅ PA derived from first message - all fidelity values = 100% (PA establishment turn)")

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
        # PA ESTABLISHMENT TURN: Skip fidelity calculation - user's message IS the PA
        # The user's first message defines their purpose, so fidelity = 100% by definition
        if is_pa_establishment_turn:
            logger.info("🎯 PA ESTABLISHMENT TURN: Setting fidelity to 100% (message = PA definition)")
            user_fidelity = 1.0  # Perfect alignment - message IS the PA
            raw_similarity = 1.0
            baseline_hard_block = False
        else:
            # FIDELITY PIPELINE (documented for clarity):
            #
            # 1. RAW SIMILARITY: cosine(user_embedding, PA_embedding)
            #    - This is the raw output from the embedding model
            #    - Range varies by model: ST (0.15-0.45), Mistral (0.40-0.75)
            #
            # 2. LAYER 1 CHECK: raw_similarity < SIMILARITY_BASELINE (0.20)?
            #    - If yes: extreme off-topic → baseline_hard_block = True
            #
            # 3. ADAPTIVE CONTEXT (if enabled):
            #    - Classifies message type (ANAPHORA, FOLLOW_UP, etc.)
            #    - Applies context boost based on MAX similarity to prior turns
            #    - Returns adjusted_fidelity (can be higher than raw due to context)
            #
            # 4. INTERVENTION DECISION: baseline_hard_block OR fidelity < FIDELITY_GREEN (0.70)
            #    - Uses adjusted fidelity for decision
            #    - GREEN zone (>= 0.70): No intervention
            #    - YELLOW/ORANGE/RED (< 0.70): Steward intervention
            #
            # Returns: (fidelity, raw_similarity, baseline_hard_block)
            user_fidelity, raw_similarity, baseline_hard_block = self._calculate_user_fidelity(user_input)

        # DEBUG: Print fidelity to console for visibility
        print(f"🔍 FIDELITY DEBUG Turn {turn_number}: user_fidelity={user_fidelity:.3f}, raw_sim={raw_similarity:.3f}, baseline_block={baseline_hard_block}")

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
            response_data['native_response'] = native_response

            # ================================================================
            # AI RESPONSE FIDELITY CHECK (Critical TELOS governance)
            # ================================================================
            # Even when user is aligned (GREEN), the AI response may drift.
            # TELOS must verify AI response alignment BEFORE showing to user.
            # If AI drifts, regenerate with alignment context injection.

            ai_fidelity = None
            ai_response_intervened = False
            final_response = native_response

            # ============================================================
            # CRITICAL: Ensure ST PA embedding is loaded BEFORE AI fidelity check
            # The embedding may be cached in session state from a prior path
            # ============================================================
            if not hasattr(self, 'st_user_pa_embedding') or self.st_user_pa_embedding is None:
                if 'cached_st_user_pa_embedding' in st.session_state:
                    self.st_user_pa_embedding = st.session_state.cached_st_user_pa_embedding
                    logger.info(f"📦 Loaded ST User PA embedding from session cache for AI fidelity check")
                elif self.st_embedding_provider is not None:
                    # Generate it now from the user's PA text
                    if 'user_pa' in st.session_state and hasattr(st.session_state.user_pa, 'purpose'):
                        pa_text = f"{st.session_state.user_pa.purpose} {st.session_state.user_pa.scope}"
                        self.st_user_pa_embedding = np.array(self.st_embedding_provider.encode(pa_text))
                        st.session_state.cached_st_user_pa_embedding = self.st_user_pa_embedding
                        logger.info(f"🔧 Generated ST User PA embedding for AI fidelity check: {len(self.st_user_pa_embedding)} dims")

            # CRITICAL: Also load AI PA embedding from cache (for comparing AI responses)
            # This is the CORRECT embedding for AI fidelity - designed for teaching responses
            if not hasattr(self, 'st_ai_pa_embedding') or self.st_ai_pa_embedding is None:
                if 'cached_st_ai_pa_embedding' in st.session_state:
                    self.st_ai_pa_embedding = st.session_state.cached_st_ai_pa_embedding
                    logger.info(f"📦 Loaded ST AI PA embedding from session cache for AI fidelity check")

            # DEBUG: Check conditions for AI fidelity check
            has_st_provider = self.st_embedding_provider is not None
            has_st_user_embedding = getattr(self, 'st_user_pa_embedding', None) is not None
            has_st_ai_embedding = getattr(self, 'st_ai_pa_embedding', None) is not None
            logger.warning(f"🔬 AI FIDELITY CHECK CONDITIONS: st_provider={has_st_provider}, st_user_pa={has_st_user_embedding}, st_ai_pa={has_st_ai_embedding}")

            # ================================================================
            # ZONE-BASED AI FIDELITY (2025-12-29): COSINE SIMILARITY
            # ================================================================
            # AI fidelity measurement depends on user's alignment zone:
            #
            # GREEN ZONE (user >= 0.70): cosine(AI_response, USER_PA)
            #   - Measures: Is the AI response topically aligned with user's purpose?
            #   - Reference: USER_PA embedding (st_user_pa_embedding)
            #   - Purpose: Verify AI stayed on-topic when user is on-topic
            #
            # INTERVENTION ZONE (user < 0.70): cosine(AI_response, AI_PA)
            #   - Measures: Is the AI response behaviorally aligned with steward purpose?
            #   - Reference: AI_PA embedding (st_ai_pa_embedding)
            #   - Purpose: Verify AI is correctly redirecting when user drifts
            # ================================================================

            DISPLAY_GREEN_THRESHOLD = 0.70

            if self.st_embedding_provider:
                try:
                    # Embed the AI response (used in both zones)
                    response_embedding = np.array(self.st_embedding_provider.encode(native_response))
                    response_embedding = response_embedding / (np.linalg.norm(response_embedding) + 1e-10)  # Normalize

                    # Store AI response in context buffer for future context matching (2025-12-29)
                    # This allows follow-up queries that reference AI response content to be recognized
                    if self.adaptive_context_manager and hasattr(self.adaptive_context_manager, 'context_buffer'):
                        self.adaptive_context_manager.context_buffer.add_ai_response(native_response, response_embedding)

                    # Check user's alignment zone
                    if user_fidelity >= DISPLAY_GREEN_THRESHOLD:
                        # GREEN ZONE: AI response topical alignment with USER_PA
                        # cosine(AI_response, USER_PA) - measures topical alignment
                        # Use normalize_ai_response_fidelity() because AI responses achieve
                        # lower raw similarity (~0.40) than queries (~0.57) against USER_PA
                        raw_ai_fidelity = self._cosine_similarity(response_embedding, self.st_user_pa_embedding)
                        ai_fidelity = normalize_ai_response_fidelity(raw_ai_fidelity)
                        logger.warning(f"🔍 GREEN ZONE AI Fidelity: raw={raw_ai_fidelity:.3f} → display={ai_fidelity:.3f} (measured against USER_PA for topical alignment)")
                    else:
                        # INTERVENTION ZONE: AI response behavioral alignment with AI_PA
                        # cosine(AI_response, AI_PA) - measures behavioral alignment
                        # (embedding already computed above, reuse it)
                        pa_embedding_for_ai = getattr(self, 'st_ai_pa_embedding', None)
                        if pa_embedding_for_ai is None:
                            # Fallback to user PA if no AI PA
                            pa_embedding_for_ai = getattr(self, 'st_user_pa_embedding', None)

                        if pa_embedding_for_ai is not None:
                            raw_ai_fidelity = self._cosine_similarity(response_embedding, pa_embedding_for_ai)
                            ai_fidelity = normalize_st_fidelity(raw_ai_fidelity)
                            logger.warning(f"🔍 INTERVENTION ZONE AI Fidelity: raw={raw_ai_fidelity:.3f} → display={ai_fidelity:.3f} (measured against AI_PA for behavioral alignment)")
                        else:
                            ai_fidelity = 0.50  # Default fallback if no PA available
                            logger.warning(f"🔍 INTERVENTION ZONE AI Fidelity: No PA available, using default ({ai_fidelity:.3f})")

                    logger.warning(f"🔍 AI REALIGNMENT NEEDED? ai_fidelity({ai_fidelity:.3f}) < {DISPLAY_GREEN_THRESHOLD} = {ai_fidelity < DISPLAY_GREEN_THRESHOLD}")

                    # ============================================================
                    # AI RESPONSE DRIFT INTERVENTION (Robust Safeguard v2)
                    # ============================================================
                    # If user is GREEN but AI response is below GREEN threshold,
                    # the AI has drifted from user's purpose. Regenerate with
                    # alignment context to bring AI back on track.
                    #
                    # This is the core TELOS promise: Keep AI aligned with user purpose.
                    # When user is on-topic, AI MUST also be on-topic.
                    # ============================================================
                    MAX_REALIGNMENT_ATTEMPTS = 2

                    if ai_fidelity < DISPLAY_GREEN_THRESHOLD:
                        original_ai_fidelity = ai_fidelity  # FIXED: Save BEFORE updating
                        logger.warning(f"⚠️ AI RESPONSE DRIFT DETECTED: AI fidelity {ai_fidelity:.3f} < {DISPLAY_GREEN_THRESHOLD:.2f}")
                        logger.info("🔄 Triggering AI response realignment...")

                        # Retry loop: Attempt realignment up to MAX_REALIGNMENT_ATTEMPTS times
                        current_response = native_response
                        attempt = 0

                        while ai_fidelity < DISPLAY_GREEN_THRESHOLD and attempt < MAX_REALIGNMENT_ATTEMPTS:
                            attempt += 1
                            logger.info(f"🔄 Realignment attempt {attempt}/{MAX_REALIGNMENT_ATTEMPTS}...")

                            # Generate aligned response with explicit alignment context
                            aligned_response = self._regenerate_aligned_response(
                                user_input=user_input,
                                drifted_response=current_response,
                                ai_fidelity=ai_fidelity,
                                user_fidelity=user_fidelity
                            )

                            if aligned_response:
                                # Verify the new response is better
                                new_embedding = np.array(self.st_embedding_provider.encode(aligned_response))
                                # FIX: Use pa_embedding_for_ai (consistent with initial check) and normalize_st_fidelity (centroid-calibrated)
                                new_raw_ai_fidelity = self._cosine_similarity(new_embedding, pa_embedding_for_ai)
                                new_ai_fidelity = normalize_st_fidelity(new_raw_ai_fidelity)

                                logger.info(f"🔄 Realigned AI Fidelity: {ai_fidelity:.3f} → {new_ai_fidelity:.3f} (attempt {attempt})")

                                # Use the new response if it's better
                                if new_ai_fidelity > ai_fidelity:
                                    current_response = aligned_response
                                    ai_fidelity = new_ai_fidelity
                                    final_response = aligned_response
                                    ai_response_intervened = True
                                else:
                                    logger.warning(f"⚠️ Realignment didn't improve fidelity ({new_ai_fidelity:.3f} <= {ai_fidelity:.3f})")
                                    break  # Stop if not improving
                            else:
                                logger.warning(f"⚠️ AI realignment attempt {attempt} failed to generate response")
                                break

                        # Record realignment metadata
                        if ai_response_intervened:
                            response_data['ai_response_realigned'] = True
                            response_data['original_ai_fidelity'] = original_ai_fidelity
                            response_data['realignment_attempts'] = attempt
                            logger.info(f"✅ AI response realigned: {original_ai_fidelity:.3f} → {ai_fidelity:.3f} (after {attempt} attempts)")

                        # Final warning if still below threshold after all attempts
                        if ai_fidelity < DISPLAY_GREEN_THRESHOLD:
                            logger.error(f"🚨 AI FIDELITY STILL BELOW GREEN after {attempt} attempts: {ai_fidelity:.3f} < {DISPLAY_GREEN_THRESHOLD}")
                            response_data['ai_realignment_failed'] = True

                except Exception as e:
                    logger.error(f"❌ AI fidelity check failed: {e}")
                    import traceback
                    logger.error(f"   Traceback: {traceback.format_exc()}")
                    # Continue with native response if check fails

            else:
                # AI fidelity check was SKIPPED due to missing provider/embedding
                logger.warning(f"⚠️ AI FIDELITY CHECK SKIPPED - Missing ST provider or PA embedding")
                logger.warning(f"   st_embedding_provider={self.st_embedding_provider is not None}")
                logger.warning(f"   st_user_pa_embedding={getattr(self, 'st_user_pa_embedding', None) is not None}")
                response_data['ai_fidelity_check_skipped'] = True

            response_data['shown_response'] = final_response

            # Compute Primacy State if we have both fidelities
            primacy_state = None
            if ai_fidelity is not None:
                epsilon = 1e-8
                primacy_state = (2 * user_fidelity * ai_fidelity) / (user_fidelity + ai_fidelity + epsilon)
                logger.info(f"📊 GREEN Zone PS: F_user={user_fidelity:.3f}, F_ai={ai_fidelity:.3f}, PS={primacy_state:.3f}")

            # GREEN ZONE: AI fidelity computed using USER PA (topic alignment, not behavioral)
            logger.info(f"📊 GREEN ZONE AI Fidelity: {ai_fidelity:.3f} (computed against USER PA for topic alignment)")
            telos_data = {
                'response': final_response,
                'fidelity_score': None,
                'distance_from_pa': 1.0 - user_fidelity,
                'intervention_triggered': ai_response_intervened,
                'intervention_type': 'ai_response_realignment' if ai_response_intervened else None,
                'intervention_reason': 'AI response drifted from user purpose' if ai_response_intervened else None,
                'drift_detected': ai_response_intervened,
                'in_basin': True,
                'ai_pa_fidelity': ai_fidelity,
                'primacy_state_score': primacy_state,
                'display_primacy_state': f"{primacy_state * 100:.0f}%" if primacy_state else None,
                'primacy_state_condition': 'computed',
                'pa_correlation': None,
                'lightweight_path': not ai_response_intervened,
                'user_pa_fidelity': user_fidelity,
                'display_user_pa_fidelity': response_data['display_fidelity'],
                'fidelity_level': response_data['fidelity_level'],
                'ai_response_intervened': ai_response_intervened,
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

                    # =============================================================
                    # PROPORTIONAL CONTROL: Correct error signal formula
                    # Per whitepaper Section 5.3: F = K·e_t where e_t = 1.0 - fidelity
                    # =============================================================
                    # K_ATTRACTOR = 1.5 (from proportional_controller.py)
                    # IMPORTANT: Error signal is 1.0 - fidelity, NOT threshold - fidelity
                    # The old formula (threshold_green - fidelity) produced weak interventions
                    # because it measured distance from GREEN, not from perfect alignment.
                    K_ATTRACTOR = 1.5
                    error_signal = 1.0 - user_fidelity  # FIXED: Correct formula
                    controller_strength = min(K_ATTRACTOR * error_signal, 1.0)

                    # Determine intervention state based on strength thresholds
                    # Per ProportionalController: epsilon_min ~ 0.16, epsilon_max ~ 0.58
                    # Simplified mapping for display/logging purposes:
                    # State 1 (MONITOR):   e < 0.30, strength < 0.45 → No action
                    # State 2 (CORRECT):   0.30 ≤ e < 0.50 → Context injection
                    # State 3 (INTERVENE): 0.50 ≤ e < 0.67 → Regeneration
                    # State 4 (ESCALATE):  e ≥ 0.67 → Block/escalate
                    if error_signal < 0.30:
                        intervention_state = "MONITOR"
                    elif error_signal < 0.50:
                        intervention_state = "CORRECT"
                    elif error_signal < 0.67:
                        intervention_state = "INTERVENE"
                    else:
                        intervention_state = "ESCALATE"

                    # Map strength to semantic band (for linguistic output styling)
                    # These bands control how the Semantic Interpreter generates prompts
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

                    logger.info(f"📐 Proportional Controller: e={error_signal:.3f}, "
                               f"strength={controller_strength:.3f}, state={intervention_state}, "
                               f"band={semantic_band}")

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

        # ============================================================
        # SCI INTEGRATION: Set previous turn data for Semantic Continuity Inheritance
        # ============================================================
        # This enables the next turn's fidelity calculation to inherit from this turn
        # when there's high semantic continuity (e.g., follow-up questions).
        # The "previous turn" becomes a temporal attractor that pulls continuous
        # follow-ups toward inherited fidelity with appropriate decay.
        try:
            if self.adaptive_context_manager and hasattr(self.adaptive_context_manager, 'context_buffer'):
                user_embedding = getattr(self, 'last_user_input_embedding', None)

                # Get AI response embedding from the last stored AI response in context buffer
                ai_embedding = None
                if hasattr(self.adaptive_context_manager.context_buffer, '_last_ai_embedding'):
                    ai_embedding = self.adaptive_context_manager.context_buffer._last_ai_embedding

                # Get the final fidelity for this turn
                turn_fidelity = response_data.get('user_fidelity', 0.0)

                if user_embedding is not None:
                    self.adaptive_context_manager.context_buffer.set_previous_turn(
                        user_embedding=user_embedding,
                        ai_embedding=ai_embedding,  # May be None if AI didn't respond
                        fidelity=turn_fidelity
                    )
                    logger.info(f"🔗 SCI: Set previous turn (fidelity={turn_fidelity:.3f}, "
                               f"has_ai_embed={ai_embedding is not None})")
        except Exception as e:
            logger.debug(f"SCI set_previous_turn skipped: {e}")

        return response_data

    def _calculate_user_fidelity(self, user_input: str, use_context: bool = True) -> tuple:
        """
        Calculate fidelity of user input relative to their PA using TWO-LAYER architecture.

        This is the FIRST calculation - before any response generation.

        CONTEXT-AWARE FIDELITY (NEW):
        - When use_context=True, prepends recent conversation history to user input
        - This allows queries that are contextually related to be recognized
        - Example: "EU AI Act compliance" in Turn 2 gets context from Turn 1 about TELOS
        - Prevents false positive interventions on contextually related queries

        TEMPLATE MODE (SentenceTransformer + Rescaling):
        - Uses SentenceTransformer for better off-topic discrimination
        - Applies rescaling to map narrow score range to TELOS fidelity range
        - Formula: fidelity = 0.25 + raw_score * 1.8, clamped to [0, 1]

        STANDARD MODE (Mistral):
        - LAYER 1: Baseline Pre-Filter (raw_sim < 0.50 triggers HARD_BLOCK)
        - LAYER 2: TELOS Primacy State (fidelity = raw cosine similarity)

        Args:
            user_input: The user's message
            use_context: Whether to include conversation context (default: True)

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
            # PRE-LOAD SESSION STATE EMBEDDINGS (CRITICAL FOR ADAPTIVE CONTEXT)
            # ================================================================
            # After a PA shift, a new BetaResponseManager instance is created.
            # We must restore ST mode and embedding BEFORE the adaptive context
            # check, otherwise the hasattr() condition will fail.
            if 'use_rescaled_fidelity_mode' in st.session_state and st.session_state.use_rescaled_fidelity_mode:
                self.use_rescaled_fidelity = True
                # Initialize ST provider if needed
                if not self.st_embedding_provider:
                    from telos_purpose.core.embedding_provider import get_cached_minilm_provider
                    self.st_embedding_provider = get_cached_minilm_provider()
                    logger.info(f"   SentenceTransformer (cached): {self.st_embedding_provider.dimension} dims")
                # Load cached PA embedding EARLY so adaptive context can use it
                if not hasattr(self, 'st_user_pa_embedding') and 'cached_st_user_pa_embedding' in st.session_state:
                    self.st_user_pa_embedding = st.session_state.cached_st_user_pa_embedding
                    logger.info(f"   ✅ Pre-loaded ST PA embedding for adaptive context: {len(self.st_user_pa_embedding)} dims")

            # ================================================================
            # ADAPTIVE CONTEXT SYSTEM: Phase-aware, pattern-classified context
            # ================================================================
            # If enabled, use the full adaptive context system from the proposal.
            # This replaces simple context concatenation with:
            # - Message type classification (DIRECT, FOLLOW_UP, CLARIFICATION, ANAPHORA)
            # - Multi-tier context buffer with weighted embeddings
            # - Conversation phase detection (EXPLORATION, FOCUS, DRIFT, RECOVERY)
            # - Adaptive threshold calculation with governance bounds
            # FIX (2025-12-30): Check for ANY PA embedding, not just ST
            # This ensures SCI works with both ST and Mistral embedding modes
            has_st_pa = hasattr(self, 'st_user_pa_embedding') and self.st_user_pa_embedding is not None
            has_mistral_pa = hasattr(self, 'user_pa_embedding') and self.user_pa_embedding is not None
            has_pa_embedding = has_st_pa or has_mistral_pa

            # DEBUG: Log condition states
            logger.warning(f"[SCI DEBUG] adaptive_context_enabled={self.adaptive_context_enabled}, "
                          f"adaptive_context_manager={self.adaptive_context_manager is not None}, "
                          f"has_st_pa={has_st_pa}, has_mistral_pa={has_mistral_pa}, "
                          f"has_pa_embedding={has_pa_embedding}")

            if self.adaptive_context_enabled and self.adaptive_context_manager and has_pa_embedding:
                try:
                    # Get PA embedding (use ST embedding if available)
                    pa_embedding = getattr(self, 'st_user_pa_embedding', None)
                    if pa_embedding is None:
                        pa_embedding = self.user_pa_embedding

                    # Get input embedding
                    if self.st_embedding_provider:
                        input_embedding = np.array(self.st_embedding_provider.encode(user_input))
                    elif self.embedding_provider:
                        input_embedding = np.array(self.embedding_provider.encode(user_input))
                    else:
                        raise ValueError("No embedding provider available")

                    # Calculate raw fidelity first (for adaptive system)
                    raw_fidelity = self._cosine_similarity(input_embedding, pa_embedding)

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

                    # Log adaptive context decision
                    logger.info(f"🔄 ADAPTIVE CONTEXT: type={adaptive_result.message_type.name}, "
                               f"phase={adaptive_result.phase.name}, "
                               f"raw={raw_fidelity:.3f} -> adjusted={adaptive_result.adjusted_fidelity:.3f}, "
                               f"threshold={adaptive_result.adaptive_threshold:.3f}, "
                               f"intervene={adaptive_result.should_intervene}")

                    # Use adaptive result values
                    fidelity = adaptive_result.adjusted_fidelity
                    raw_similarity = raw_fidelity  # Keep original for logging
                    baseline_hard_block = adaptive_result.should_intervene and adaptive_result.drift_detected

                    # SCI INTEGRATION: Store user input embedding for set_previous_turn()
                    self.last_user_input_embedding = input_embedding

                    return (fidelity, raw_similarity, baseline_hard_block)

                except Exception as e:
                    logger.warning(f"Adaptive context failed, falling back to legacy: {e}")
                    # Fall through to legacy context-aware fidelity

            # ================================================================
            # DEPRECATED: LEGACY CONTEXT-AWARE FIDELITY (Exception Fallback Only)
            # ================================================================
            # This code path is DEPRECATED as of 2025-12-30. SCI (Semantic Continuity
            # Inheritance) is now the canonical fidelity measurement approach.
            # This fallback only executes if the AdaptiveContextManager throws an
            # exception. It uses string-based context prepending rather than
            # measurement-based semantic similarity inheritance.
            contextual_input = user_input
            if use_context:
                recent_context = self._get_recent_context_for_fidelity()
                if recent_context:
                    # Prepend context with separator - keeps user input primary
                    # Format: "[Context: ...] | [Current query: ...]"
                    contextual_input = f"[Context: {recent_context}] | {user_input}"
                    logger.info(f"📚 Legacy context-aware fidelity: added {len(recent_context)} chars of context")

            # NOTE: ST mode loading was already done in PRE-LOAD section above.
            # This section is now redundant but kept for safety in case the
            # early loading is bypassed for some code path.

            # ================================================================
            # TEMPLATE MODE: SentenceTransformer + Raw Thresholds (Clean Lane)
            # ================================================================
            # LEAN SIX SIGMA: No rescaling. Use raw thresholds calibrated for ST.
            # ST thresholds: GREEN >= 0.32, YELLOW >= 0.25, ORANGE >= 0.18, RED < 0.18
            if self.use_rescaled_fidelity and self.st_embedding_provider and hasattr(self, 'st_user_pa_embedding'):
                from telos_purpose.core.constants import ST_FIDELITY_GREEN, ST_FIDELITY_RED

                # Embed user input with SentenceTransformer (with context if available)
                user_embedding = np.array(self.st_embedding_provider.encode(contextual_input))

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
                # Fast path: use cached SentenceTransformer embedding (with context)
                user_embedding = np.array(self.st_embedding_provider.encode(contextual_input))
                raw_similarity = self._cosine_similarity(user_embedding, self.st_user_pa_embedding)
                logger.info(f"FAST PATH: ST embedding, raw_sim={raw_similarity:.3f}")
            elif not self.embedding_provider or self.user_pa_embedding is None:
                logger.warning("Embedding provider or PA not initialized - returning default fidelity")
                return (FIDELITY_GREEN, FIDELITY_GREEN, False)  # Default to Aligned zone if not ready
            else:
                # Fallback: use Mistral API (slower) - still use contextual input
                user_embedding = np.array(self.embedding_provider.encode(contextual_input))
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

    def _regenerate_aligned_response(
        self,
        user_input: str,
        drifted_response: str,
        ai_fidelity: float,
        user_fidelity: float
    ) -> Optional[str]:
        """
        Regenerate AI response when the native response drifted from user's purpose.

        This is triggered when:
        - User is aligned (GREEN zone, fidelity >= 0.70)
        - AI response is NOT aligned (fidelity < 0.70)

        The regeneration uses explicit alignment context to guide the LLM
        back toward the user's stated purpose.

        Args:
            user_input: User's message
            drifted_response: The native response that was off-topic
            ai_fidelity: The fidelity score of the drifted response
            user_fidelity: The user's fidelity score (for context)

        Returns:
            Aligned response string, or None if regeneration fails
        """
        try:
            from telos_purpose.llm_clients.mistral_client import get_cached_mistral_client

            client = get_cached_mistral_client()

            # Get PA context
            pa_data = st.session_state.get('primacy_attractor', {})
            purpose_raw = pa_data.get('purpose', 'General assistance')
            scope_raw = pa_data.get('scope', 'Open discussion')
            purpose = ' '.join(purpose_raw) if isinstance(purpose_raw, list) else purpose_raw
            scope = ' '.join(scope_raw) if isinstance(scope_raw, list) else scope_raw

            # Build alignment-focused system prompt
            # This explicitly tells the LLM about the drift and how to correct it
            alignment_system_prompt = f"""You are a helpful AI assistant. The user has a SPECIFIC PURPOSE for this conversation that you MUST honor.

USER'S PURPOSE: {purpose}
CONVERSATION SCOPE: {scope}

CRITICAL ALIGNMENT INSTRUCTION:
Your previous response drifted from the user's stated purpose. The user is asking something directly relevant to their goal,
but your response went off-topic. You need to generate a response that:

1. DIRECTLY addresses what the user asked
2. Stays FOCUSED on their stated purpose: "{purpose}"
3. Does NOT include tangential information or generic explanations
4. Uses language and examples relevant to their specific goal

The user's fidelity to their purpose is {user_fidelity*100:.0f}% - they are staying on track.
Your response should match their focus.

RESPONSE GUIDELINES:
- Be conversational and natural
- Keep responses brief and focused (2-3 short paragraphs maximum)
- Address the user's question in the context of their stated purpose
- Do NOT use lists, bullet points, or headers unless explicitly asked
- Avoid verbose explanations or unnecessary elaboration"""

            # Build conversation with alignment context
            conversation = [{'role': 'system', 'content': alignment_system_prompt}]

            # Add conversation history
            conversation.extend(self._get_conversation_history())

            # Add user input
            conversation.append({'role': 'user', 'content': user_input})

            # Generate aligned response
            aligned_response = client.generate(
                messages=conversation,
                max_tokens=MAX_TOKENS_GREEN,
                temperature=0.5  # Lower temperature for more focused response
            )

            logger.info(f"✅ Generated aligned response ({len(aligned_response)} chars)")
            return aligned_response

        except Exception as e:
            logger.error(f"❌ Failed to regenerate aligned response: {e}")
            return None

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

                # Store AI response in context buffer for future context matching (2025-12-29)
                # This allows follow-up queries that reference AI response content to be recognized
                if self.adaptive_context_manager and hasattr(self.adaptive_context_manager, 'context_buffer'):
                    self.adaptive_context_manager.context_buffer.add_ai_response(response, response_embedding)

                # DUAL-REFERENCE AI FIDELITY (2025-12-28)
                # AI is considered aligned if response matches EITHER:
                #   1. PA centroid (topic space alignment), OR
                #   2. User query (direct response relevance)

                # Reference 1: Similarity to AI PA (behavioral role embedding - NOT centroid)
                # AI PA is derived from User PA with role transformation (e.g., "learn" → "teach")
                # This should produce HIGHER similarity than User PA centroid
                ai_pa_embedding = getattr(self, 'st_ai_pa_embedding', None)
                if ai_pa_embedding is None:
                    ai_pa_embedding = self.st_user_pa_embedding  # Fallback to User PA if AI PA not available
                raw_ai_fidelity_to_pa = self._cosine_similarity(response_embedding, ai_pa_embedding)

                # Reference 2: Similarity to user query (direct response relevance)
                user_query_embedding = np.array(self.st_embedding_provider.encode(user_input))
                raw_ai_fidelity_to_query = self._cosine_similarity(response_embedding, user_query_embedding)

                # Use MAX: AI is aligned if it matches AI PA OR user query
                # Apply appropriate normalization based on which reference wins:
                # - AI_PA: Use normalize_st_fidelity (AI_PA centroids include example_ai_responses)
                # - Query: Use normalize_ai_response_fidelity (AI response vs short query text)
                if raw_ai_fidelity_to_pa >= raw_ai_fidelity_to_query:
                    # AI PA reference won - use standard normalization
                    ai_fidelity = normalize_st_fidelity(raw_ai_fidelity_to_pa)
                    winning_ref = "AI_PA"
                else:
                    # Query reference won - use AI response calibration
                    ai_fidelity = normalize_ai_response_fidelity(raw_ai_fidelity_to_query)
                    winning_ref = "Query"

                logger.info(f"🔧 Intervention AI Fidelity (dual-ref): AI_PA={raw_ai_fidelity_to_pa:.3f}, Query={raw_ai_fidelity_to_query:.3f}, Winner={winning_ref} → {ai_fidelity:.3f}")

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
        # =======================================================================
        # HYBRID INTERVENTION STYLING (SemanticInterpreter + StewardStyles)
        # =======================================================================
        # This combines two complementary systems:
        # 1. StewardStyles: Therapeutic persona + band-based intervention intensity
        # 2. SemanticInterpreter: Concrete linguistic specifications
        #
        # NOTE: get_exemplar() is NOT used - those example phrases were being
        # copied verbatim into LLM output (e.g., "Far from your stated purpose...")
        # =======================================================================

        # Get concrete linguistic specs from SemanticInterpreter
        spec = semantic_interpret(fidelity, purpose)

        # Get therapeutic Steward prompt from steward_styles (already imported at top)
        # This provides the band-appropriate intervention tone
        steward_prompt = get_intervention_prompt(
            fidelity=fidelity,
            user_context=purpose,
            green_threshold=FIDELITY_GREEN  # Use imported constant
        )

        # Get the linguistic specification block (sentence form, hedging, etc.)
        linguistic_spec = spec.to_prompt_block(purpose)

        # Combine: Steward therapeutic persona + concrete linguistic specs
        return f"""{steward_prompt}

LINGUISTIC GUIDELINES:
{linguistic_spec}

CRITICAL INSTRUCTIONS:
- Never use stock phrases like "far from your stated purpose" or "here's the path back"
- Be natural and conversational - sound like a human, not a governance system
- Keep responses concise (2-3 sentences unless more detail is genuinely needed)
- Use the linguistic form specified above ({spec.sentence_form})"""

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

                # Store AI response in context buffer for future context matching (2025-12-29)
                # This allows follow-up queries that reference AI response content to be recognized
                if self.adaptive_context_manager and hasattr(self.adaptive_context_manager, 'context_buffer'):
                    self.adaptive_context_manager.context_buffer.add_ai_response(response, response_embedding)

                # AI FIDELITY: BEHAVIORAL ALIGNMENT (2025-12-29)
                # AI fidelity measures how well the AI response aligns with the AI's
                # behavioral role. The AI PA is derived from User purpose but represents
                # the AI's supportive role (e.g., "teach", "help solve", "guide").
                #
                # The AI PA is its OWN centroid for determining behavioral fidelity.
                # Semantic alignment is already baked into the derived AI PA.
                #
                # Formula: F_ai = MAX(similarity_to_ai_pa, similarity_to_user_query)
                # - Reference 1: Behavioral alignment via AI PA centroid
                # - Reference 2: Direct response relevance to user query

                # Reference 1: Similarity to AI PA (BEHAVIORAL alignment)
                # AI PA represents the supportive role derived from user purpose
                ai_pa_embedding = getattr(self, 'st_ai_pa_embedding', None)
                if ai_pa_embedding is None:
                    ai_pa_embedding = self.st_user_pa_embedding  # Fallback if not available
                raw_ai_fidelity_to_pa = self._cosine_similarity(response_embedding, ai_pa_embedding)

                # Reference 2: Similarity to user query (direct response relevance)
                user_input_embedding = np.array(self.st_embedding_provider.encode(user_input))
                raw_ai_fidelity_to_query = self._cosine_similarity(response_embedding, user_input_embedding)

                # Use MAX: AI is aligned if it matches topic space OR user query
                # Apply appropriate normalization based on which reference wins:
                # - AI_PA: Use normalize_st_fidelity (AI_PA centroids include example_ai_responses)
                # - Query: Use normalize_ai_response_fidelity (AI response vs short query text)
                if raw_ai_fidelity_to_pa >= raw_ai_fidelity_to_query:
                    # AI PA reference won - use standard normalization
                    ai_fidelity = normalize_st_fidelity(raw_ai_fidelity_to_pa)
                    winning_ref = "AI_PA"
                else:
                    # Query reference won - use AI response calibration
                    ai_fidelity = normalize_ai_response_fidelity(raw_ai_fidelity_to_query)
                    winning_ref = "Query"

                logger.info(f"📊 AI Fidelity (behavioral): AI_PA={raw_ai_fidelity_to_pa:.3f}, Query={raw_ai_fidelity_to_query:.3f}, Winner={winning_ref} → {ai_fidelity:.2%}")

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

        # DEBUG: Trace user fidelity normalization
        print(f"🔍 NATIVE TELOS DATA: user_fidelity={user_fidelity}, model_type={model_type}, display_user_fidelity={display_user_fidelity_value}")

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
                # Clear all cached PA embeddings EXCEPT identity
                for key in ['cached_user_pa_embedding', 'cached_ai_pa_embedding',
                            'cached_st_user_pa_embedding', 'cached_mpnet_user_pa_embedding',
                            'cached_mpnet_ai_pa_embedding']:
                    if key in st.session_state:
                        del st.session_state[key]
                # CRITICAL: Set new PA identity IMMEDIATELY so subsequent checks pass
                # Without this, the next check at line ~1754 would fail (None != current_pa_identity)
                st.session_state.cached_pa_identity = current_pa_identity
                logger.info(f"   🔑 Set new PA identity: {current_pa_identity}")

            # =================================================================
            # TEMPLATE MODE FAST PATH: Check for ST embeddings FIRST
            # After a TELOS pivot, _regenerate_pa_centroid() creates ST embedding
            # but deletes Mistral embeddings. This fast path handles that case.
            # =================================================================
            template_mode_ready = (
                self.use_rescaled_fidelity and
                'cached_st_user_pa_embedding' in st.session_state and
                'cached_mpnet_user_pa_embedding' in st.session_state and
                'cached_mpnet_ai_pa_embedding' in st.session_state and
                st.session_state.get('cached_pa_identity') == current_pa_identity
            )

            if template_mode_ready:
                # FAST PATH: Template mode with all required embeddings cached
                logger.info("   🚀 TEMPLATE MODE FAST PATH: Using cached ST/MPNet embeddings")
                self.st_user_pa_embedding = st.session_state.cached_st_user_pa_embedding
                self.mpnet_user_pa_embedding = st.session_state.cached_mpnet_user_pa_embedding
                self.mpnet_ai_pa_embedding = st.session_state.cached_mpnet_ai_pa_embedding
                logger.info(f"   ✅ ST PA centroid: {len(self.st_user_pa_embedding)} dims (cached)")
                logger.info(f"   ✅ MPNet User PA: {len(self.mpnet_user_pa_embedding)} dims (cached)")
                logger.info(f"   ✅ MPNet AI PA: {len(self.mpnet_ai_pa_embedding)} dims (cached)")

                # Create dummy Mistral embeddings (not used in template mode but required by some code paths)
                if 'cached_user_pa_embedding' in st.session_state:
                    self.user_pa_embedding = st.session_state.cached_user_pa_embedding
                    self.ai_pa_embedding = st.session_state.get('cached_ai_pa_embedding', self.user_pa_embedding)
                else:
                    # Create placeholder - won't be used in template mode fidelity calculation
                    user_pa_text = f"Purpose: {purpose_str}. Scope: {scope_str}."
                    self.user_pa_embedding = np.array(embedding_provider.encode(user_pa_text))
                    self.ai_pa_embedding = self.user_pa_embedding  # Placeholder
                    st.session_state.cached_user_pa_embedding = self.user_pa_embedding
                    st.session_state.cached_ai_pa_embedding = self.ai_pa_embedding
                    logger.info(f"   📦 Created placeholder Mistral embeddings for compatibility")

            # Check if PA embeddings are already cached in session state AND match current PA
            elif ('cached_user_pa_embedding' in st.session_state and
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
            import traceback

            # Check API key before creating client
            api_key = os.getenv('MISTRAL_API_KEY')
            if not api_key:
                logger.error("   ❌ MISTRAL_API_KEY not found in environment!")
                logger.error(f"   Environment vars: {list(os.environ.keys())[:10]}...")
            else:
                logger.info(f"   ✅ MISTRAL_API_KEY found: {api_key[:10]}...")

            mistral_client = Mistral(api_key=api_key)
            enrichment_service = PAEnrichmentService(mistral_client)

            logger.info("   🔧 Using PA Enrichment Service for semantic extraction...")
            enriched_pa = enrichment_service.enrich_direction(
                direction=user_input,
                current_template=None,
                conversation_context=""
            )

            if enriched_pa:
                logger.info(f"   ✅ PA enriched with {len(enriched_pa.get('example_queries', []))} example queries")
                logger.info(f"   ✅ PA purpose: {enriched_pa.get('purpose', 'N/A')[:100]}")
            else:
                logger.warning("   ⚠️ PA enrichment returned None (check pa_enrichment.py logs)")
        except Exception as e:
            logger.error(f"   ❌ PA enrichment exception: {type(e).__name__}: {e}")
            logger.error(f"   ❌ Traceback: {traceback.format_exc()}")

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
        Regenerate PA centroid from purpose/scope + example queries.

        Matches the centroid computation in regenerate_pa_embeddings.py:
        centroid = mean of [purpose/scope embedding] + [example_query embeddings]

        This ensures shift-focus PAs have the same semantic coverage as template PAs.

        Args:
            example_queries: List of example queries for centroid construction
        """
        try:
            # Ensure SentenceTransformer provider is initialized - uses cached version
            if not self.st_embedding_provider:
                from telos_purpose.core.embedding_provider import get_cached_minilm_provider
                self.st_embedding_provider = get_cached_minilm_provider()
                logger.info(f"   SentenceTransformer (cached): {self.st_embedding_provider.dimension} dims")

            # Get purpose/scope from current PA
            new_pa_data = st.session_state.get('primacy_attractor', {})
            purpose_str = new_pa_data.get('purpose', '')
            if isinstance(purpose_str, list):
                purpose_str = ' '.join(purpose_str)
            scope_str = new_pa_data.get('scope', '')
            if isinstance(scope_str, list):
                scope_str = ' '.join(scope_str)
            user_pa_text = f"Purpose: {purpose_str} Scope: {scope_str}"

            # Build list of all texts for centroid: purpose/scope + example_queries
            # This matches regenerate_pa_embeddings.py behavior
            if example_queries:
                all_texts = [user_pa_text] + list(example_queries)
            else:
                all_texts = [user_pa_text]

            # Embed all texts and normalize each before averaging
            logger.info(f"📊 Regenerating PA centroid from {len(all_texts)} texts (1 purpose/scope + {len(example_queries) if example_queries else 0} examples)...")
            all_embeddings = []
            for text in all_texts:
                emb = np.array(self.st_embedding_provider.encode(text))
                # Normalize each embedding before averaging
                emb = emb / (np.linalg.norm(emb) + 1e-10)
                all_embeddings.append(emb)

            # Compute centroid as mean of all normalized embeddings
            domain_centroid = np.mean(all_embeddings, axis=0)

            # Re-normalize the centroid
            self.st_user_pa_embedding = domain_centroid / (np.linalg.norm(domain_centroid) + 1e-10)

            # Cache in session state
            st.session_state.cached_st_user_pa_embedding = self.st_user_pa_embedding

            # Enable template mode (SentenceTransformer thresholds)
            self.use_rescaled_fidelity = True
            # Cache in session state so subsequent requests use same mode
            st.session_state.use_rescaled_fidelity_mode = True

            # CRITICAL: Update PA identity hash to match new PA
            # Without this, _initialize_telos_engine will detect a mismatch and
            # invalidate the cache we just updated, causing fallback to hardcoded values
            import hashlib
            new_pa_data = st.session_state.get('primacy_attractor', {})
            purpose_str = new_pa_data.get('purpose', '')
            if isinstance(purpose_str, list):
                purpose_str = ' '.join(purpose_str)
            scope_str = new_pa_data.get('scope', '')
            if isinstance(scope_str, list):
                scope_str = ' '.join(scope_str)
            new_pa_identity = hashlib.md5(f"{purpose_str}|{scope_str}".encode()).hexdigest()[:16]
            st.session_state.cached_pa_identity = new_pa_identity
            logger.info(f"   🔑 Updated PA identity hash: {new_pa_identity}")

            # Regenerate MPNet User PA embedding for AI fidelity calculation
            if self.mpnet_embedding_provider:
                user_pa_text = f"Purpose: {purpose_str}. Scope: {scope_str}."
                self.mpnet_user_pa_embedding = np.array(self.mpnet_embedding_provider.encode(user_pa_text))
                st.session_state.cached_mpnet_user_pa_embedding = self.mpnet_user_pa_embedding
                logger.info(f"   📦 Regenerated MPNet User PA: {len(self.mpnet_user_pa_embedding)} dims")

                # Regenerate MPNet AI PA embedding
                detected_intent = self._detect_intent_from_purpose(purpose_str)
                role_action = INTENT_TO_ROLE_MAP.get(detected_intent, 'help')
                ai_purpose = f"{role_action.capitalize()} the user as they work to: {purpose_str}"
                ai_pa_text = f"AI Role: {ai_purpose}. Supporting scope: {scope_str}."
                self.mpnet_ai_pa_embedding = np.array(self.mpnet_embedding_provider.encode(ai_pa_text))
                st.session_state.cached_mpnet_ai_pa_embedding = self.mpnet_ai_pa_embedding
                logger.info(f"   📦 Regenerated MPNet AI PA: {len(self.mpnet_ai_pa_embedding)} dims")

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

    def _get_recent_context_for_fidelity(self) -> str:
        """
        Get recent conversation context for context-aware fidelity calculation.

        This is optimized for fidelity calculation - it extracts topic keywords
        from recent user messages to provide semantic context when embedding
        the current query. This allows queries like "EU AI Act compliance" to
        inherit context from Turn 1 about "TELOS governance".

        Returns:
            String containing recent topic context (empty string if no history)
        """
        history = self._get_conversation_history()
        if not history:
            return ""

        # Get last 2 user messages only (not assistant) - focuses on user intent
        user_messages = [msg for msg in history if msg.get('role') == 'user']
        if not user_messages:
            return ""

        # Take last 2 user messages, truncate to 100 chars each
        recent_user = user_messages[-2:]
        context_parts = []
        for msg in recent_user:
            content = msg.get('content', '')[:100]
            if content:
                context_parts.append(content)

        return " | ".join(context_parts)

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