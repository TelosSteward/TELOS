"""
Observatory V2 - State Manager
===============================

Centralized state management for the Observatory application.
All state lives here, components read and update through this manager.

Design: Single source of truth, no scattered session_state usage.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import streamlit as st
import sys
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

# AUDIT: Module load verification (logging only, no print to avoid BrokenPipeError)
logger.debug("STATE_MANAGER.PY MODULE LOADED")

# Add parent directory for beta imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@dataclass
class ObservatoryState:
    """
    Complete state for Observatory application.

    All state variables are defined here with clear types and defaults.
    Components interact with state only through StateManager methods.
    """
    # Session data
    current_turn: int = 0
    total_turns: int = 0
    session_id: str = "unknown"
    turns: List[Dict[str, Any]] = field(default_factory=list)
    primacy_attractor: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # PA Establishment State (Calibration Phase)
    user_pa_established: bool = False
    ai_pa_established: bool = False
    calibration_phase: bool = True  # True until both PAs established
    calibration_turn_count: int = 0  # Turns spent in calibration
    convergence_turn: Optional[int] = None  # Turn when PA converged

    # UI state
    deck_expanded: bool = False
    teloscope_playing: bool = False
    teloscope_expanded: bool = False  # TELOSCOPE Controls visibility - default to collapsed
    playback_speed: float = 1.0
    scrollable_history_mode: bool = False  # Toggle between turn-by-turn and scrollable history

    # Component visibility
    show_primacy_attractor: bool = False
    show_math_breakdown: bool = False
    show_counterfactual: bool = False
    show_steward: bool = False  # Deprecated, kept for compatibility
    show_observatory_lens: bool = False  # Observatory Lens visual dashboard

    # Metrics
    avg_fidelity: float = 0.0
    total_interventions: int = 0
    drift_warnings: int = 0


class StateManager:
    """
    Manages all Observatory state.

    Responsibilities:
    - Initialize default state
    - Provide read access to state
    - Provide controlled write access to state
    - Validate state changes
    - Trigger UI updates when needed
    """

    def __init__(self):
        """Initialize state manager with default state."""
        self.state = ObservatoryState()
        self._initialized = False
        self._beta_session_manager = None  # Lazy initialization for beta testing
        self._ps_calculator = None  # Primacy State calculator

    def initialize(self, session_data: Dict[str, Any]):
        """
        Initialize state from session data.

        Args:
            session_data: Dictionary containing:
                - session_id: Unique session identifier
                - turns: List of turn dictionaries
                - statistics: Aggregate metrics (optional)
        """
        try:
            if self._initialized:
                logger.debug("State already initialized, skipping")
                return

            # Validate session data
            if not isinstance(session_data, dict):
                logger.error(f"Invalid session data type: {type(session_data)}")
                session_data = {}

            # Load session data with safe defaults
            self.state.session_id = session_data.get('session_id', 'unknown')
            self.state.turns = session_data.get('turns', [])
            self.state.total_turns = len(self.state.turns)
            self.state.primacy_attractor = session_data.get('primacy_attractor', {})
            self.state.metadata = session_data.get('metadata', {})

            # Load statistics if available
            stats = session_data.get('statistics', {})
            self.state.avg_fidelity = stats.get('avg_fidelity', 0.0)
            self.state.total_interventions = stats.get('interventions', 0)
            self.state.drift_warnings = stats.get('drift_warnings', 0)

            self._initialized = True
            logger.info(f"State initialized: session_id={self.state.session_id}, turns={self.state.total_turns}")

        except Exception as e:
            logger.error(f"Error initializing state: {type(e).__name__}: {str(e)}", exc_info=True)
            # Initialize with safe defaults
            self.state = ObservatoryState()
            self._initialized = True

    def load_from_session(self, session_data: Dict[str, Any]):
        """
        Load a saved session into current state (replaces current session).

        Args:
            session_data: Complete session data including turns, metadata, etc.
        """
        try:
            # Reset initialization flag to allow reload
            self._initialized = False

            # Re-initialize with new session data
            self.initialize(session_data)

            # Set current turn to last turn or specified position
            self.state.current_turn = session_data.get('current_turn', max(0, self.state.total_turns - 1))
            if self.state.current_turn >= self.state.total_turns:
                self.state.current_turn = max(0, self.state.total_turns - 1)

            logger.info(f"Session loaded: {self.state.session_id}, positioned at turn {self.state.current_turn}")

        except Exception as e:
            logger.error(f"Error loading session: {type(e).__name__}: {str(e)}", exc_info=True)
            # Reinitialize with blank state if load fails
            self._initialized = False
            self.initialize({})

    # =========================================================================
    # Read Access - Components use these to get current state
    # =========================================================================

    def get_current_turn_index(self) -> int:
        """Get current turn index (0-based)."""
        return self.state.current_turn

    def get_current_turn_data(self) -> Optional[Dict[str, Any]]:
        """Get data for current turn."""
        if 0 <= self.state.current_turn < self.state.total_turns:
            return self.state.turns[self.state.current_turn]
        return None

    def get_all_turns(self) -> List[Dict[str, Any]]:
        """Get all turn data."""
        return self.state.turns

    def is_deck_expanded(self) -> bool:
        """Check if Observation Deck is expanded."""
        return self.state.deck_expanded

    def is_playing(self) -> bool:
        """Check if TELOSCOPE is in play mode."""
        return self.state.teloscope_playing

    def get_session_info(self) -> Dict[str, Any]:
        """Get session metadata and metrics."""
        return {
            'session_id': self.state.session_id,
            'total_turns': self.state.total_turns,
            'current_turn': self.state.current_turn,
            'avg_fidelity': self.state.avg_fidelity,
            'total_interventions': self.state.total_interventions,
            'drift_warnings': self.state.drift_warnings
        }

    # =========================================================================
    # Write Access - Controlled mutations with validation
    # =========================================================================

    def next_turn(self) -> bool:
        """
        Advance to next turn.

        Returns:
            True if advanced, False if at end
        """
        if self.state.current_turn < self.state.total_turns - 1:
            self.state.current_turn += 1
            return True
        return False

    def previous_turn(self) -> bool:
        """
        Go back to previous turn.

        Returns:
            True if moved back, False if at beginning
        """
        if self.state.current_turn > 0:
            self.state.current_turn -= 1
            return True
        return False

    def jump_to_turn(self, turn_index: int) -> bool:
        """
        Jump to specific turn.

        Args:
            turn_index: Target turn index (0-based)

        Returns:
            True if valid jump, False if invalid index
        """
        if 0 <= turn_index < self.state.total_turns:
            self.state.current_turn = turn_index
            return True
        return False

    def toggle_deck(self):
        """Toggle Observation Deck visibility."""
        self.state.deck_expanded = not self.state.deck_expanded

    def toggle_teloscope(self):
        """Toggle TELOSCOPE Controls visibility."""
        self.state.teloscope_expanded = not self.state.teloscope_expanded

    def is_teloscope_expanded(self) -> bool:
        """Check if TELOSCOPE Controls are expanded."""
        return self.state.teloscope_expanded

    def start_playback(self):
        """Start TELOSCOPE playback."""
        self.state.teloscope_playing = True

    def stop_playback(self):
        """Stop TELOSCOPE playback."""
        self.state.teloscope_playing = False

    def set_playback_speed(self, speed: float):
        """
        Set playback speed.

        Args:
            speed: Multiplier (0.5 = half speed, 2.0 = double speed)
        """
        if 0.1 <= speed <= 5.0:  # Reasonable bounds
            self.state.playback_speed = speed

    def toggle_component(self, component: str):
        """
        Toggle visibility of a component.

        Args:
            component: One of 'primacy_attractor', 'math', 'counterfactual', 'steward', 'observatory_lens'
        """
        if component == 'primacy_attractor':
            self.state.show_primacy_attractor = not self.state.show_primacy_attractor
        elif component == 'math' or component == 'math_breakdown':
            self.state.show_math_breakdown = not self.state.show_math_breakdown
        elif component == 'counterfactual':
            self.state.show_counterfactual = not self.state.show_counterfactual
        elif component == 'steward':
            self.state.show_steward = not self.state.show_steward
        elif component == 'observatory_lens':
            self.state.show_observatory_lens = not self.state.show_observatory_lens

    def toggle_scrollable_history(self):
        """Toggle between turn-by-turn and scrollable history mode."""
        self.state.scrollable_history_mode = not self.state.scrollable_history_mode

    def clear_demo_data(self):
        """
        Clear all demo/initial data to start fresh with user conversation.

        This is CRITICAL to prevent demo data from contaminating the primacy
        attractor calibration for the user's actual conversation.
        """
        logger.info("Clearing demo data to start fresh user conversation")

        # Clear all turns
        self.state.turns = []
        self.state.total_turns = 0
        self.state.current_turn = 0

        # Reset statistics
        self.state.avg_fidelity = 0.0
        self.state.total_interventions = 0
        self.state.drift_warnings = 0

        # Reset TELOS session if active
        if hasattr(self, '_telos_steward') and self._telos_steward:
            try:
                self._telos_steward.end_session()
                self._telos_steward.start_session(session_id=f"{self.state.session_id}_user")
                logger.info("TELOS session reset for user conversation")
            except Exception as e:
                logger.warning(f"Could not reset TELOS session: {e}")

    # =========================================================================
    # Beta Testing Methods
    # =========================================================================

    def _initialize_beta_session_manager(self):
        """Initialize beta session manager (lazy loading)."""
        if self._beta_session_manager is None:
            try:
                from observatory.beta_testing.beta_session_manager import BetaSessionManager
                self._beta_session_manager = BetaSessionManager()
                logger.info("Beta session manager initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize beta session manager: {e}")
                self._beta_session_manager = None

    def _is_beta_ab_phase(self) -> bool:
        """
        Check if we're in beta mode AND at the A/B testing phase.

        Returns:
            True if beta tab active, consent given, intro complete, and PA established
        """
        # Check if BETA tab is active
        active_tab = st.session_state.get('active_tab', 'DEMO')
        if active_tab != "BETA":
            return False

        # Check if beta consent and intro completed
        beta_consent = st.session_state.get('beta_consent_given', False)
        beta_intro_complete = st.session_state.get('beta_intro_complete', False)

        if not (beta_consent and beta_intro_complete):
            return False

        # Check if PA (Primacy Attractor) is established
        # Can't do A/B testing without a PA to measure fidelity against!
        # If PA not established by turn 10, session is "not fidelity available"
        current_turn = len([t for t in self.state.turns if not t.get('is_loading', False)])

        # PA must be established (can happen early, e.g., turn 2-3)
        pa_established = self.state.ai_pa_established

        # If we're past turn 10 and PA still not established, give up on this session
        if current_turn > 10 and not pa_established:
            logger.warning(f"Turn {current_turn}: PA not established by turn 10 - not a fidelity available session")
            return False

        # A/B testing starts once PA is established
        return pa_established

    def _generate_beta_dual_response(
        self,
        message: str,
        conversation_history: List[Dict[str, str]],
        max_tokens: int
    ) -> Dict[str, Any]:
        """
        Generate both baseline and TELOS responses for A/B testing.

        Args:
            message: User's input
            conversation_history: Conversation context
            max_tokens: Max tokens for generation

        Returns:
            Dict with baseline_response, telos_response, and metrics
        """
        try:
            # Ensure beta session manager is initialized
            self._initialize_beta_session_manager()

            if self._beta_session_manager is None:
                raise Exception("Beta session manager not available")

            # Ensure TELOS is initialized
            if not hasattr(self, '_telos_steward') or self._telos_steward is None:
                raise Exception("TELOS not initialized")

            # Generate baseline response (direct LLM, no TELOS governance)
            baseline_response = self._telos_steward.llm_client.generate(
                messages=conversation_history,
                max_tokens=max_tokens
            )
            logger.info(f"✓ Generated baseline response ({len(baseline_response)} chars)")

            # Generate TELOS response (with governance)
            telos_result = self._telos_steward.process_turn(
                user_input=message,
                model_response=baseline_response
            )

            # TELOS may modify the response if intervention needed
            telos_response = telos_result.get("final_response", baseline_response)
            logger.info(f"✓ Generated TELOS response ({len(telos_response)} chars)")

            # Extract metrics
            baseline_fidelity = telos_result.get("baseline_fidelity", 0.0)
            telos_fidelity = telos_result.get("telic_fidelity", 0.0)

            return {
                "success": True,
                "baseline_response": baseline_response,
                "telos_response": telos_response,
                "baseline_fidelity": baseline_fidelity,
                "telos_fidelity": telos_fidelity,
                "drift_detected": telos_result.get("drift_detected", False),
                "intervention_applied": telos_result.get("intervention_applied", False),
                "telos_result": telos_result  # Full result for metrics extraction
            }

        except Exception as e:
            logger.error(f"Beta dual-response generation failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    # =========================================================================
    # Main Response Generation
    # =========================================================================

    def add_user_message(self, message: str):
        """
        Add a new user message and generate a TELOS-governed response.

        Args:
            message: User's input message
        """
        # =====================================================================
        # IMMEDIATE USER MESSAGE DISPLAY (Fix UI Lag)
        # =====================================================================
        # Check if we're already processing this message (rerun after placeholder)
        already_processing = False
        if self.state.turns and self.state.turns[-1].get('is_loading', False):
            # We're in the second run - already showed placeholder, now process response
            already_processing = True
            current_turn_idx = len(self.state.turns) - 1

        if not already_processing:
            # First run - add placeholder and trigger immediate UI update
            placeholder_turn = {
                'turn': self.state.total_turns,
                'timestamp': self.state.total_turns * 2.5,
                'user_input': message,
                'response': '',  # Empty response signals "loading" state
                'fidelity': None,
                'distance': None,
                'threshold': 0.76,  # Goldilocks: Aligned threshold
                'intervention_applied': False,
                'drift_detected': False,
                'status': "⏳",
                'status_text': "Processing",
                'in_basin': True,
                'phase2_comparison': None,
                'is_loading': True  # Special flag for loading state
            }

            self.state.turns.append(placeholder_turn)
            self.state.total_turns += 1
            self.state.current_turn = self.state.total_turns - 1

            # CRITICAL: Rerun immediately to show user message + loading animation
            # On next run, already_processing will be True and we'll generate response
            st.rerun()
        # =====================================================================

        # Import TELOS components here to avoid circular imports
        try:
            from telos_observatory_v3.utils.telos_demo_data import generate_telos_demo_session
            from telos_purpose.core.unified_steward import UnifiedGovernanceSteward, PrimacyAttractor
            from telos_purpose.core.embedding_provider import SentenceTransformerProvider
            from telos_purpose.llm_clients.mistral_client import MistralClient

            # Initialize TELOS if not already done
            if not hasattr(self, '_telos_steward'):
                try:
                    logger.info("Initializing TELOS engine...")
                    # Use CACHED provider to avoid expensive model reloading (critical for Railway cold start)
                    from telos_purpose.core.embedding_provider import get_cached_minilm_provider
                    embedding_provider = get_cached_minilm_provider()

                    # Check if demo mode is enabled (for TELOS framework walkthrough)
                    demo_mode = st.session_state.get('telos_demo_mode', False)

                    if demo_mode:
                        # Demo mode: Use saved TELOS framework demo configuration
                        from demo_mode.telos_framework_demo import get_demo_attractor_config
                        config = get_demo_attractor_config()
                        attractor = PrimacyAttractor(**config)

                        # LAYER 2: Initialize RAG corpus for Demo Mode
                        logger.info("Loading TELOS documentation corpus for Demo Mode...")
                        from demo_mode.telos_corpus_loader import TELOSCorpusLoader
                        self._corpus_loader = TELOSCorpusLoader(embedding_provider)
                        num_chunks = self._corpus_loader.load_corpus()
                        logger.info(f"✓ Corpus loaded: {num_chunks} chunks")
                    else:
                        # Beta/Open mode: Minimal attractor for general conversation
                        # Allow flexible conversation while still tracking alignment
                        attractor = PrimacyAttractor(
                            purpose=[
                                "Engage in helpful, informative conversation",
                                "Respond to user questions and requests",
                                "Maintain conversational coherence"
                            ],
                            scope=[
                                "General knowledge and assistance",
                                "User's topics of interest",
                                "Conversational dialogue"
                            ],
                            boundaries=[
                                "Stay relevant to user's questions",
                                "Provide accurate, helpful information",
                                "Maintain appropriate conversation tone"
                            ],
                            constraint_tolerance=0.5,  # Flexible for open conversation
                            privacy_level=0.8,
                            task_priority=0.5
                        )
                        self._corpus_loader = None  # No corpus in open mode

                    # Initialize Mistral client (Assistant Steward)
                    # API key from Streamlit secrets or environment
                    import os

                    # Try environment variable first (more reliable)
                    mistral_api_key = os.getenv("MISTRAL_API_KEY")

                    # Fall back to Streamlit secrets if env var not found
                    if not mistral_api_key:
                        try:
                            mistral_api_key = st.secrets["MISTRAL_API_KEY"]
                        except (KeyError, FileNotFoundError):
                            pass

                    if not mistral_api_key:
                        logger.error("MISTRAL_API_KEY not found in environment or secrets")
                        raise ValueError("MISTRAL_API_KEY not found in secrets or environment")

                    logger.info("MISTRAL_API_KEY found and validated")

                    # TODO: Future PIP feature - API key will become user identity/auth

                    mistral_client = MistralClient(
                        api_key=mistral_api_key,
                        model="mistral-large-latest"
                    )

                    self._telos_steward = UnifiedGovernanceSteward(
                        attractor=attractor,
                        llm_client=mistral_client,
                        embedding_provider=embedding_provider,
                        enable_interventions=True  # Enable interventions for drift detection and correction
                    )
                    self._telos_steward.start_session(session_id=self.state.session_id)
                    logger.info("TELOS engine initialized successfully")
                except Exception as init_error:
                    logger.error(f"Failed to initialize TELOS engine: {type(init_error).__name__}: {str(init_error)}")
                    # Set flag to None to trigger fallback in generation
                    self._telos_steward = None
                    self._corpus_loader = None
                    raise  # Re-raise to trigger fallback response logic

            # Check if TELOS engine is available
            if self._telos_steward is None:
                raise Exception("TELOS engine not initialized - using fallback mode")

            # Generate AI response through Mistral (Assistant Steward)
            # Build conversation history for context

            # Get system prompt based on mode
            demo_mode = st.session_state.get('telos_demo_mode', False)

            # LAYER 2 (RAG): Retrieve relevant corpus chunks if in demo mode
            retrieved_context = ""
            if demo_mode and hasattr(self, '_corpus_loader') and self._corpus_loader:
                try:
                    from demo_mode.telos_corpus_loader import format_context_for_llm

                    # Retrieve top-3 most relevant chunks for user's query
                    chunks = self._corpus_loader.retrieve(message, top_k=3)
                    retrieved_context = format_context_for_llm(chunks)
                    logger.info(f"Retrieved {len(chunks)} corpus chunks for context")
                except Exception as corpus_error:
                    logger.warning(f"Corpus retrieval failed: {corpus_error}")
                    retrieved_context = ""

            # Build system prompt with RAG context
            if demo_mode:
                from demo_mode.telos_framework_demo import get_demo_system_prompt
                base_system_prompt = get_demo_system_prompt()

                # If we have RAG context, prepend it
                if retrieved_context:
                    system_prompt = f"{retrieved_context}\n\n{base_system_prompt}\n\nIMPORTANT: Use the documentation context above to provide accurate, grounded responses. Cite specific sections when relevant."
                else:
                    system_prompt = base_system_prompt
            else:
                # Open mode: Neutral, helpful assistant
                system_prompt = """You are a helpful AI assistant. Engage naturally with the user's questions and topics.

Be informative, conversational, and adapt to what the user wants to discuss."""

            conversation_history = [
                {
                    "role": "system",
                    "content": system_prompt
                }
            ]

            # Build history from existing turns (exclude loading turn)
            for turn in self.state.turns:
                # Skip the placeholder turn (is_loading=True, empty response)
                if turn.get('is_loading', False):
                    continue

                conversation_history.append({
                    "role": "user",
                    "content": turn.get('user_input', '')
                })
                conversation_history.append({
                    "role": "assistant",
                    "content": turn.get('response', '')
                })

            # Add current message
            conversation_history.append({
                "role": "user",
                "content": message
            })

            # Determine max_tokens based on mode
            # Demo Mode: Hard limit for brevity (DEFAULT: 2 paragraphs, MAX: 3 paragraphs)
            # Open Mode: Standard limit
            if demo_mode:
                max_tokens = 350  # HARD PROTOCOL: ~2 paragraphs default, ~3 max
                logger.info("Demo Mode: Enforcing brevity protocol (max_tokens=350, target: 2 paragraphs)")
            else:
                max_tokens = 500  # Standard limit for open mode

            # =====================================================================
            # ASYNC + PARALLEL PROCESSING (Experimental)
            # =====================================================================
            # IMPORTANT: Currently only supported in Demo Mode
            # Open Mode needs separate implementation (no corpus_loader)
            # Check if experimental performance features are enabled
            enable_async = st.session_state.get('enable_async', False)
            enable_parallel = st.session_state.get('enable_parallel', False)

            # Only use async/parallel in Demo Mode (Steward has corpus)
            # Open Mode will be added after validation
            if demo_mode and (enable_async or enable_parallel):
                try:
                    import asyncio
                    from telos_observatory_v3.core.async_processor import AsyncStewardProcessor

                    # Initialize async processor if not already done
                    if not hasattr(self, '_async_processor'):
                        self._async_processor = AsyncStewardProcessor(
                            enable_async=enable_async,
                            enable_parallel=enable_parallel,
                            max_workers=4
                        )
                        logger.info(f"Async processor initialized: async={enable_async}, parallel={enable_parallel}")

                    # Update processor settings if flags changed
                    if self._async_processor.enable_async != enable_async or \
                       self._async_processor.enable_parallel != enable_parallel:
                        self._async_processor.enable_async = enable_async
                        self._async_processor.enable_parallel = enable_parallel
                        logger.info(f"Async processor settings updated: async={enable_async}, parallel={enable_parallel}")

                    # Try async processing
                    logger.info("→ Attempting async/parallel processing...")
                    async_result = asyncio.run(
                        self._async_processor.process_message(
                            message=message,
                            corpus_loader=self._corpus_loader,
                            telos_steward=self._telos_steward,
                            conversation_history=conversation_history,
                            max_tokens=max_tokens
                        )
                    )

                    # Check if async processing succeeded
                    if async_result is not None:
                        # Success! Use async results
                        response_text = async_result['response']
                        result = async_result['validation']
                        logger.info("✓ Async/parallel processing succeeded")
                        logger.info(f"  Performance: {async_result.get('processing_times', {})}")
                    else:
                        # Async processing failed, fall back to sync
                        logger.warning("Async/parallel processing returned None, using sync fallback")
                        raise Exception("Async processing fallback")

                except Exception as async_error:
                    # Async processing failed, fall back to sync
                    logger.warning(f"Async/parallel processing failed: {async_error}, using sync")

                    # FALLBACK: Synchronous processing (original code)
                    response_text = self._telos_steward.llm_client.generate(
                        messages=conversation_history,
                        max_tokens=max_tokens
                    )

                    result = self._telos_steward.process_turn(
                        user_input=message,
                        model_response=response_text
                    )
            else:
                # Performance features disabled - use sync processing
                logger.info("→ Using synchronous processing (async/parallel disabled)")

                # =====================================================================
                # BETA A/B TESTING: Dual-Response Generation
                # =====================================================================
                # Check if we're in beta A/B testing phase
                if self._is_beta_ab_phase():
                    logger.info("🧪 Beta A/B testing mode active - generating dual responses")

                    # Generate both baseline and TELOS responses
                    beta_result = self._generate_beta_dual_response(
                        message=message,
                        conversation_history=conversation_history,
                        max_tokens=max_tokens
                    )

                    if beta_result.get("success", False):
                        # Get or create beta session
                        if 'beta_session' not in st.session_state:
                            self._initialize_beta_session_manager()
                            if self._beta_session_manager:
                                st.session_state.beta_session = self._beta_session_manager.start_session()

                        beta_session = st.session_state.get('beta_session')

                        # Assign test condition if not yet assigned
                        if beta_session and not beta_session.test_condition:
                            self._beta_session_manager.assign_test_condition(beta_session)
                            logger.info(f"Assigned test condition: {beta_session.test_condition}")

                        # Determine which response to show based on test condition
                        test_condition = beta_session.test_condition if beta_session else "single_blind_baseline"

                        if test_condition == "single_blind_baseline":
                            response_text = beta_result["baseline_response"]
                            result = beta_result["telos_result"]
                            logger.info("→ Showing BASELINE response (single-blind)")
                        elif test_condition == "single_blind_telos":
                            response_text = beta_result["telos_response"]
                            result = beta_result["telos_result"]
                            logger.info("→ Showing TELOS response (single-blind)")
                        else:  # head_to_head
                            # For head-to-head, we'll show both in the UI
                            # For now, default to TELOS for single conversation history
                            response_text = beta_result["telos_response"]
                            result = beta_result["telos_result"]
                            logger.info("→ Head-to-head mode: showing TELOS in history, both in UI")

                        # Store ONLY DELTAS (NO conversation content)
                        if not already_processing:
                            current_turn_idx = len(self.state.turns) - 1
                        self.state.turns[current_turn_idx]['beta_data'] = {
                            'test_condition': test_condition,
                            'baseline_fidelity': beta_result["baseline_fidelity"],
                            'telos_fidelity': beta_result["telos_fidelity"],
                            'fidelity_delta': beta_result["telos_fidelity"] - beta_result["baseline_fidelity"],
                            'intervention_applied': beta_result.get("intervention_applied", False),
                            'drift_detected': beta_result.get("drift_detected", False),
                            'shown_response_source': 'baseline' if test_condition == 'single_blind_baseline' else 'telos',
                            'response_length_baseline': len(beta_result["baseline_response"]),
                            'response_length_telos': len(beta_result["telos_response"])
                        }

                        # Calculate PS metrics for the shown response
                        ps_metrics_for_delta = {}
                        if self._ps_calculator is None:
                            try:
                                from telos_purpose.core.primacy_state import PrimacyStateCalculator
                                self._ps_calculator = PrimacyStateCalculator(track_energy=True)
                                logger.info("PS calculator initialized for beta")
                            except ImportError as e:
                                logger.debug(f"PS module not available: {e}")

                        if self._ps_calculator and hasattr(self._telos_steward, 'embedding_provider'):
                            try:
                                # Get the response that was shown to user
                                shown_response = beta_result["baseline_response"] if test_condition == 'single_blind_baseline' else beta_result["telos_response"]

                                # Get embeddings
                                embedding_provider = self._telos_steward.embedding_provider
                                response_embedding = embedding_provider.encode(shown_response)

                                # Get PA embeddings from steward's attractor
                                attractor = self._telos_steward.attractor
                                user_pa_text = " ".join(attractor.purpose)
                                ai_pa_text = " ".join(attractor.boundaries) if attractor.boundaries else user_pa_text

                                user_pa_embedding = embedding_provider.encode(user_pa_text)
                                ai_pa_embedding = embedding_provider.encode(ai_pa_text)

                                # Compute PS
                                ps_result = self._ps_calculator.compute_primacy_state(
                                    response_embedding=response_embedding,
                                    user_pa_embedding=user_pa_embedding,
                                    ai_pa_embedding=ai_pa_embedding
                                )

                                ps_metrics_for_delta = ps_result.to_dict()
                                logger.info(f"PS for beta: {ps_metrics_for_delta.get('primacy_state_score', 0):.3f}")
                            except Exception as e:
                                logger.debug(f"PS calculation failed in beta: {e}")

                        # Transmit deltas to backend (privacy-preserving)
                        try:
                            from services.backend_client import get_backend_service
                            backend = get_backend_service()

                            if backend.enabled:
                                session_id = st.session_state.get('session_id')
                                delta_data = {
                                    'session_id': str(session_id),
                                    'turn_number': current_turn_idx + 1,
                                    'fidelity_score': beta_result["telos_fidelity"],
                                    'distance_from_pa': 1.0 - beta_result["telos_fidelity"],
                                    'baseline_fidelity': beta_result["baseline_fidelity"],
                                    'fidelity_delta': beta_result["telos_fidelity"] - beta_result["baseline_fidelity"],
                                    'intervention_triggered': beta_result.get("intervention_applied", False),
                                    'mode': 'beta',
                                    'test_condition': test_condition,
                                    'shown_response_source': 'baseline' if test_condition == 'single_blind_baseline' else 'telos',
                                    # Add PS metrics if available
                                    **ps_metrics_for_delta
                                }
                                backend.transmit_delta(delta_data)
                        except Exception as e:
                            logger.warning(f"Failed to transmit delta to backend: {e}")
                        logger.info("✓ Beta data stored in turn metadata")
                    else:
                        # Beta dual-response failed, fall back to standard flow
                        logger.warning(f"Beta dual-response failed: {beta_result.get('error')}, using standard flow")
                        response_text = self._telos_steward.llm_client.generate(
                            messages=conversation_history,
                            max_tokens=max_tokens
                        )
                        result = self._telos_steward.process_turn(
                            user_input=message,
                            model_response=response_text
                        )
                else:
                    # Standard flow (not beta A/B testing)
                    # Generate response using Mistral
                    response_text = self._telos_steward.llm_client.generate(
                        messages=conversation_history,
                        max_tokens=max_tokens
                    )

                    # Process through TELOS to get fidelity metrics
                    result = self._telos_steward.process_turn(
                        user_input=message,
                        model_response=response_text
                    )
                # =====================================================================
            # =====================================================================

            fidelity = result.get("telic_fidelity", 0.0)  # Real value from TELOS
            distance = result.get("error_signal", 0.0)
            in_basin = result.get("in_basin", True)
            intervention_applied = result.get("intervention_applied", False)

            if fidelity == 0.0:
                logger.warning(f"Turn {turn_idx}: No fidelity value from TELOS process_turn - investigation needed")

            # Calculate Primacy State if available
            ps_metrics = None
            if self._ps_calculator is None:
                try:
                    from telos_purpose.core.primacy_state import PrimacyStateCalculator
                    self._ps_calculator = PrimacyStateCalculator(track_energy=True)
                    logger.info("PS calculator initialized")
                except ImportError as e:
                    logger.debug(f"PS module not available: {e}")

            if self._ps_calculator and hasattr(self._telos_steward, 'embedding_provider'):
                try:
                    # Get embeddings
                    embedding_provider = self._telos_steward.embedding_provider
                    response_embedding = embedding_provider.encode(response_text)

                    # Get PA embeddings from steward's attractor
                    attractor = self._telos_steward.attractor
                    user_pa_text = " ".join(attractor.purpose)
                    ai_pa_text = " ".join(attractor.boundaries) if attractor.boundaries else user_pa_text

                    user_pa_embedding = embedding_provider.encode(user_pa_text)
                    ai_pa_embedding = embedding_provider.encode(ai_pa_text)

                    # Compute PS
                    from telos_purpose.core.primacy_state import PrimacyStateMetrics
                    ps_result = self._ps_calculator.compute_primacy_state(
                        response_embedding=response_embedding,
                        user_pa_embedding=user_pa_embedding,
                        ai_pa_embedding=ai_pa_embedding
                    )

                    ps_metrics = ps_result.to_dict()
                    logger.info(f"PS computed: {ps_metrics.get('ps_score', 0):.3f} "
                              f"(F_user={ps_metrics.get('f_user', 0):.2f}, "
                              f"F_AI={ps_metrics.get('f_ai', 0):.2f})")
                except Exception as e:
                    logger.debug(f"PS calculation failed: {e}")

            # Determine status using zone names (Goldilocks thresholds)
            # Import from central config to keep thresholds in sync
            try:
                from config.colors import _ZONE_ALIGNED, _ZONE_MINOR_DRIFT, _ZONE_DRIFT
            except ImportError:
                _ZONE_ALIGNED, _ZONE_MINOR_DRIFT, _ZONE_DRIFT = 0.70, 0.60, 0.50

            if fidelity >= _ZONE_ALIGNED:
                status_icon, status_text = "✓", "Aligned"
            elif fidelity >= _ZONE_MINOR_DRIFT:
                status_icon, status_text = "✓", "Minor Drift"
            elif fidelity >= _ZONE_DRIFT:
                status_icon, status_text = "⚠", "Drift Detected"
            else:
                status_icon, status_text = "⚠", "Significant Drift"

        except Exception as e:
            # Log the error for debugging
            error_type = type(e).__name__
            error_msg = str(e)
            logger.error(f"Error generating response: {error_type}: {error_msg}", exc_info=True)

            # Fallback if TELOS not available - provide helpful response
            # Simple rule-based responses for common TELOS questions
            message_lower = message.lower()

            # Check for specific error types and provide appropriate fallback
            if "API" in error_msg or "connection" in error_msg.lower():
                logger.warning("API connection error - using rule-based fallback")
                fallback_reason = "API temporarily unavailable"
            elif "system_prompt" in error_msg:
                logger.warning("API compatibility issue - using rule-based fallback")
                fallback_reason = "API compatibility mode"
            else:
                logger.warning(f"Unexpected error ({error_type}) - using rule-based fallback")
                fallback_reason = "processing error"

            # Provide context-appropriate responses
            if "telos" in message_lower or "work" in message_lower:
                response_text = "TELOS is a purpose alignment framework that keeps AI conversations focused on their intended goals. It uses mathematical geometry in embedding space to detect and correct drift."
            elif "fidelity" in message_lower or "score" in message_lower:
                response_text = "Fidelity scores measure how well responses align with your purpose. Scores 0.70 and above indicate alignment (green), 0.60-0.69 is minor drift (yellow), 0.50-0.59 is drift detected (orange), and below 0.50 triggers intervention (red)."
            elif "intervention" in message_lower or "drift" in message_lower:
                response_text = "TELOS detects drift by measuring semantic distance from the primacy attractor. When responses drift too far, the system can inject context, regenerate responses, or block and alert."
            elif "primacy" in message_lower or "attractor" in message_lower:
                response_text = "The Primacy Attractor defines your conversation's purpose, scope, and boundaries. It acts as a gravitational center that keeps responses aligned with your goals."
            else:
                response_text = f"I'm designed to help explain TELOS governance and purpose alignment. Could you rephrase your question about TELOS? (Running in fallback mode due to {fallback_reason})"

            fidelity = 0.0  # Fallback - no real governance running
            distance = 0.0
            in_basin = True
            intervention_applied = False
            logger.warning(f"Turn {turn_idx}: Using fallback response - no real fidelity available")
            status_icon, status_text = "✓", "Good"

        # Update the placeholder turn with actual response
        # current_turn_idx was set earlier (either in already_processing or when adding placeholder)
        if not already_processing:
            current_turn_idx = len(self.state.turns) - 1

        turn_update = {
            'response': response_text,
            'fidelity': fidelity,
            'distance': distance,
            'intervention_applied': intervention_applied,
            'drift_detected': fidelity < 0.70,  # Goldilocks: Aligned threshold
            'status': status_icon,
            'status_text': status_text,
            'in_basin': in_basin,
            'is_loading': False  # Response is ready
        }

        # Add PS metrics if available
        if ps_metrics:
            turn_update['ps_metrics'] = ps_metrics
            turn_update['primacy_state_score'] = ps_metrics.get('ps_score')
            turn_update['primacy_state_condition'] = ps_metrics.get('condition')
            turn_update['user_pa_fidelity'] = ps_metrics.get('f_user')
            turn_update['ai_pa_fidelity'] = ps_metrics.get('f_ai')
            turn_update['pa_correlation'] = ps_metrics.get('rho_pa')
            turn_update['v_dual_energy'] = ps_metrics.get('v_dual')
            turn_update['delta_v_dual'] = ps_metrics.get('delta_v')
            turn_update['primacy_converging'] = ps_metrics.get('converging')

        self.state.turns[current_turn_idx].update(turn_update)

        # Current turn is already set (done when we added placeholder)

        # Update statistics (filter out None fidelity values)
        fidelities = [t['fidelity'] for t in self.state.turns if t.get('fidelity') is not None]
        self.state.avg_fidelity = sum(fidelities) / len(fidelities) if fidelities else 0.0
        self.state.total_interventions = sum(1 for t in self.state.turns if t.get('intervention_applied', False))
        self.state.drift_warnings = sum(1 for t in self.state.turns if t.get('drift_detected', False))

    def add_user_message_streaming(self, message: str):
        """
        Add user message and prepare for streaming response.
        Returns turn_index for the conversation display to use.

        Args:
            message: User's input message

        Returns:
            int: Index of the new turn
        """
        # Add new turn with user message and empty response
        new_turn = {
            'turn': self.state.total_turns,
            'timestamp': self.state.total_turns * 2.5,
            'user_input': message,
            'response': '',  # Will be filled by streaming
            'fidelity': None,
            'distance': None,
            'threshold': 0.76,  # Goldilocks: Aligned threshold
            'intervention_applied': False,
            'drift_detected': False,
            'status': "⏳",
            'status_text': "Streaming",
            'in_basin': True,
            'phase2_comparison': None,
            'is_streaming': True  # Flag to indicate streaming in progress
        }

        self.state.turns.append(new_turn)
        self.state.total_turns += 1
        self.state.current_turn = self.state.total_turns - 1

        return self.state.current_turn

    def generate_response_stream(self, message: str, turn_idx: int):
        """
        Generator that yields response chunks for streaming display.
        After streaming completes, updates the turn with TELOS metrics.

        Args:
            message: User's input message
            turn_idx: Index of the turn to update

        Yields:
            str: Response text chunks
        """
        # AUDIT TRACE: Log entry to this method
        logger.info("=" * 80)
        logger.info(f"🔍 AUDIT: generate_response_stream() CALLED")
        logger.info(f"   Turn: {turn_idx}")
        logger.info(f"   Message: {message[:100]}")

        # BETA MODE ROUTER: Check if we should use BetaResponseManager instead
        active_tab = st.session_state.get('active_tab', 'DEMO')
        pa_established = st.session_state.get('pa_established', False)

        logger.info(f"🔍 AUDIT: Router check values:")
        logger.info(f"   active_tab = '{active_tab}'")
        logger.info(f"   pa_established = {pa_established}")
        logger.info(f"   Condition (active_tab == 'BETA'): {active_tab == 'BETA'}")

        if active_tab == 'BETA':
            logger.info(f"🔍 AUDIT: BETA tab detected, checking PA...")
            if pa_established:
                logger.info(f"🔀 BETA MODE DETECTED - Routing to BetaResponseManager for turn {turn_idx}")
                logger.info(f"🔍 AUDIT: Calling _generate_beta_stream()...")
                # Route to BETA-specific response generation
                yield from self._generate_beta_stream(message, turn_idx)
                logger.info(f"🔍 AUDIT: Returned from _generate_beta_stream() - EXITING")
                return
            else:
                logger.warning(f"🔍 AUDIT: BETA tab active but PA not established - falling back to standard flow")
        else:
            logger.info(f"🔍 AUDIT: Not BETA mode - using standard flow")

        # Initialize TELOS if not already done
        if not hasattr(self, '_telos_steward'):
            logger.info("🔄 STEWARD RE-INITIALIZATION TRIGGERED")
            logger.info(f"  - Session has PA: {st.session_state.get('primacy_attractor') is not None}")
            logger.info(f"  - PA Established: {st.session_state.get('pa_established', False)}")

            from telos_observatory_v3.utils.telos_demo_data import generate_telos_demo_session
            from telos_purpose.core.unified_steward import UnifiedGovernanceSteward, PrimacyAttractor
            from telos_purpose.core.embedding_provider import SentenceTransformerProvider
            from telos_purpose.llm_clients.mistral_client import MistralClient

            try:
                logger.info("Initializing TELOS engine...")
                # Use CACHED provider to avoid expensive model reloading (critical for Railway cold start)
                from telos_purpose.core.embedding_provider import get_cached_minilm_provider
                embedding_provider = get_cached_minilm_provider()

                # Check if demo mode
                demo_mode = st.session_state.get('telos_demo_mode', False)

                if demo_mode:
                    from demo_mode.telos_framework_demo import get_demo_attractor_config
                    config = get_demo_attractor_config()
                    attractor = PrimacyAttractor(**config)

                    logger.info("Loading TELOS documentation corpus for Demo Mode...")
                    from demo_mode.telos_corpus_loader import TELOSCorpusLoader
                    self._corpus_loader = TELOSCorpusLoader(embedding_provider)
                    num_chunks = self._corpus_loader.load_corpus()
                    logger.info(f"✓ Corpus loaded: {num_chunks} chunks")
                else:
                    # Beta/Open mode: Use established PA from session state
                    pa_data = st.session_state.get('primacy_attractor', None)
                    pa_established = st.session_state.get('pa_established', False)

                    logger.info(f"🔍 PA Loading Debug:")
                    logger.info(f"  - pa_data exists: {pa_data is not None}")
                    logger.info(f"  - pa_established: {pa_established}")
                    if pa_data:
                        logger.info(f"  - PA Purpose: {pa_data.get('purpose', 'N/A')}")
                        logger.info(f"  - PA Scope: {pa_data.get('scope', 'N/A')}")

                    if pa_data and pa_established:
                        # Use the PA established during onboarding
                        # Convert strings to lists as PrimacyAttractor expects List[str]
                        purpose_str = pa_data.get('purpose', 'General assistance')
                        scope_str = pa_data.get('scope', 'Open discussion')

                        attractor = PrimacyAttractor(
                            purpose=[purpose_str] if isinstance(purpose_str, str) else purpose_str,
                            scope=[scope_str] if isinstance(scope_str, str) else scope_str,
                            boundaries=pa_data.get('boundaries', [
                                "Maintain respectful dialogue",
                                "Provide accurate information",
                                "Stay within ethical guidelines"
                            ])
                        )
                        logger.info(f"✅ Using established PA - Purpose: {purpose_str[:80]}")
                        logger.info(f"✅ Using established PA - Scope: {scope_str[:80]}")
                    else:
                        # Fallback: Minimal attractor for general conversation (should rarely happen)
                        attractor = PrimacyAttractor(
                            purpose=[
                                "Engage in helpful, informative conversation",
                                "Respond to user questions and requests",
                                "Maintain conversational coherence"
                            ],
                            scope=[
                                "General knowledge and assistance",
                                "User's topics of interest",
                                "Any subject the user wishes to discuss"
                            ],
                            boundaries=[
                                "Maintain respectful dialogue",
                                "Provide accurate information",
                                "Stay within ethical guidelines"
                            ]
                        )
                        logger.warning("⚠️ No established PA found - using generic fallback")
                    self._corpus_loader = None

                # Get API key from Streamlit secrets or environment
                import os

                # Try environment variable first (more reliable)
                mistral_api_key = os.getenv("MISTRAL_API_KEY")

                # Fall back to Streamlit secrets if env var not found
                if not mistral_api_key:
                    try:
                        mistral_api_key = st.secrets["MISTRAL_API_KEY"]
                    except (KeyError, FileNotFoundError):
                        pass

                if not mistral_api_key:
                    logger.error("MISTRAL_API_KEY not found in environment or secrets")
                    raise ValueError("MISTRAL_API_KEY not found in secrets or environment")

                logger.info("MISTRAL_API_KEY found and validated")

                mistral_client = MistralClient(
                    api_key=mistral_api_key,
                    model="mistral-large-latest"
                )

                self._telos_steward = UnifiedGovernanceSteward(
                    attractor=attractor,
                    llm_client=mistral_client,
                    embedding_provider=embedding_provider,
                    enable_interventions=False
                )
                self._telos_steward.start_session(session_id=self.state.session_id)

                # Mark PA as converged if already established in BETA mode
                if not demo_mode and st.session_state.get('pa_established', False):
                    self.state.pa_converged = True
                    logger.info("PA marked as converged (established via BETA questionnaire)")

                logger.info("TELOS engine initialized successfully")
            except Exception as init_error:
                logger.error(f"Failed to initialize TELOS engine: {init_error}")
                self._telos_steward = None
                self._corpus_loader = None
                yield "I apologize, but I'm having trouble initializing. Please try again."
                return

        # Build conversation history
        demo_mode = st.session_state.get('telos_demo_mode', False)

        # Get system prompt with RAG context if available
        retrieved_context = ""
        if demo_mode and hasattr(self, '_corpus_loader') and self._corpus_loader:
            try:
                from demo_mode.telos_corpus_loader import format_context_for_llm
                chunks = self._corpus_loader.retrieve(message, top_k=3)
                retrieved_context = format_context_for_llm(chunks)
                logger.info(f"Retrieved {len(chunks)} corpus chunks for context")
            except Exception as corpus_error:
                logger.warning(f"Corpus retrieval failed: {corpus_error}")
                retrieved_context = ""

        if demo_mode:
            from demo_mode.telos_framework_demo import get_demo_system_prompt
            base_system_prompt = get_demo_system_prompt()
            if retrieved_context:
                system_prompt = f"{retrieved_context}\n\n{base_system_prompt}\n\nIMPORTANT: Use the documentation context above to provide accurate, grounded responses."
            else:
                system_prompt = base_system_prompt
        else:
            system_prompt = """You are a helpful AI assistant. Engage naturally with the user's questions and topics.

Be informative, conversational, and adapt to what the user wants to discuss."""

        conversation_history = [{"role": "system", "content": system_prompt}]

        # Build history from existing turns (exclude current streaming turn)
        for turn in self.state.turns[:turn_idx]:
            # Only include completed turns with actual responses
            if not turn.get('is_loading', False) and not turn.get('is_streaming', False) and turn.get('response'):
                conversation_history.append({"role": "user", "content": turn.get('user_input', '')})
                conversation_history.append({"role": "assistant", "content": turn.get('response', '')})

        # Add current message
        conversation_history.append({"role": "user", "content": message})

        # Determine max_tokens based on mode
        max_tokens = 350 if demo_mode else 500

        # Generate response (non-streaming for now - streaming has API issues)
        full_response = ""
        try:
            # Use regular generate for reliability
            full_response = self._telos_steward.llm_client.generate(
                messages=conversation_history,
                max_tokens=max_tokens
            )
            yield full_response  # Yield complete response
        except Exception as gen_error:
            logger.error(f"Generation error: {gen_error}", exc_info=True)
            full_response = "I apologize, but I encountered an error generating a response. Please try again."
            yield full_response

        # After streaming completes, process through TELOS for metrics
        try:
            result = self._telos_steward.process_turn(
                user_input=message,
                model_response=full_response
            )

            fidelity = result.get("telic_fidelity", 0.0)  # Real value from TELOS
            distance = result.get("error_signal", 0.0)
            in_basin = result.get("in_basin", True)
            intervention_applied = result.get("intervention_applied", False)

            if fidelity == 0.0:
                logger.warning(f"Turn {turn_idx}: No fidelity value from TELOS process_turn - investigation needed")

            # Calculate Primacy State for streaming response
            ps_metrics = None
            if self._ps_calculator and hasattr(self._telos_steward, 'embedding_provider'):
                try:
                    embedding_provider = self._telos_steward.embedding_provider
                    response_embedding = embedding_provider.encode(full_response)

                    attractor = self._telos_steward.attractor
                    user_pa_text = " ".join(attractor.purpose)
                    ai_pa_text = " ".join(attractor.boundaries) if attractor.boundaries else user_pa_text

                    user_pa_embedding = embedding_provider.encode(user_pa_text)
                    ai_pa_embedding = embedding_provider.encode(ai_pa_text)

                    from telos_purpose.core.primacy_state import PrimacyStateMetrics
                    ps_result = self._ps_calculator.compute_primacy_state(
                        response_embedding=response_embedding,
                        user_pa_embedding=user_pa_embedding,
                        ai_pa_embedding=ai_pa_embedding
                    )

                    ps_metrics = ps_result.to_dict()
                    logger.info(f"PS (streaming): {ps_metrics.get('ps_score', 0):.3f}")
                except Exception as e:
                    logger.debug(f"PS calculation failed (streaming): {e}")

            # Determine status using zone names (Goldilocks thresholds)
            try:
                from config.colors import _ZONE_ALIGNED, _ZONE_MINOR_DRIFT, _ZONE_DRIFT
            except ImportError:
                _ZONE_ALIGNED, _ZONE_MINOR_DRIFT, _ZONE_DRIFT = 0.70, 0.60, 0.50

            if fidelity >= _ZONE_ALIGNED:
                status_icon, status_text = "✓", "Aligned"
            elif fidelity >= _ZONE_MINOR_DRIFT:
                status_icon, status_text = "✓", "Minor Drift"
            elif fidelity >= _ZONE_DRIFT:
                status_icon, status_text = "⚠", "Drift Detected"
            else:
                status_icon, status_text = "⚠", "Significant Drift"
        except Exception as telos_error:
            logger.error(f"TELOS processing error: {telos_error}")
            fidelity = 0.0  # Error fallback - no real metrics available
            distance = 0.0
            in_basin = True
            intervention_applied = False
            logger.warning(f"Turn {turn_idx}: TELOS error - no real fidelity available")
            status_icon, status_text = "✓", "Good"

        # Update turn with final response and metrics
        turn_update = {
            'response': full_response,
            'fidelity': fidelity,
            'distance': distance,
            'intervention_applied': intervention_applied,
            'drift_detected': fidelity < 0.70,  # Goldilocks: Aligned threshold
            'status': status_icon,
            'status_text': status_text,
            'in_basin': in_basin,
            'is_streaming': False  # Streaming complete
        }

        # Add PS metrics if available
        if ps_metrics:
            turn_update['ps_metrics'] = ps_metrics
            turn_update['primacy_state_score'] = ps_metrics.get('ps_score')
            turn_update['primacy_state_condition'] = ps_metrics.get('condition')
            turn_update['user_pa_fidelity'] = ps_metrics.get('f_user')
            turn_update['ai_pa_fidelity'] = ps_metrics.get('f_ai')
            turn_update['pa_correlation'] = ps_metrics.get('rho_pa')
            turn_update['v_dual_energy'] = ps_metrics.get('v_dual')
            turn_update['delta_v_dual'] = ps_metrics.get('delta_v')
            turn_update['primacy_converging'] = ps_metrics.get('converging')

        self.state.turns[turn_idx].update(turn_update)

        # Update statistics
        fidelities = [t['fidelity'] for t in self.state.turns if t.get('fidelity') is not None]
        self.state.avg_fidelity = sum(fidelities) / len(fidelities) if fidelities else 0.0
        self.state.total_interventions = sum(1 for t in self.state.turns if t.get('intervention_applied', False))
        self.state.drift_warnings = sum(1 for t in self.state.turns if t.get('drift_detected', False))

    def _generate_beta_stream(self, message: str, turn_idx: int):
        """
        Generate response for BETA mode using BetaResponseManager.

        This method routes BETA mode requests to the BetaResponseManager which:
        1. Generates BOTH TELOS and Native responses
        2. Uses active governance via generate_governed_response()
        3. Stores complete data for Observatory review
        4. Returns ONE response based on A/B test sequence

        Args:
            message: User's input message
            turn_idx: Index of the turn to update

        Yields:
            str: The response to display (either TELOS or Native based on A/B sequence)
        """
        logger.info("=" * 80)
        logger.info(f"🔍 AUDIT: _generate_beta_stream() ENTERED")
        logger.info(f"   Turn index: {turn_idx}")
        logger.info(f"   Message: {message[:100]}")

        try:
            # Initialize BetaResponseManager if needed
            # Also force re-init if PA embeddings are None (stale cached instance)
            needs_reinit = 'beta_response_manager' not in st.session_state
            if not needs_reinit:
                mgr = st.session_state.beta_response_manager
                # Check if crucial PS components are None (indicates stale instance from before fix)
                if mgr.telos_engine and (mgr.user_pa_embedding is None or mgr.ai_pa_embedding is None):
                    logger.warning("⚠️ Stale BetaResponseManager detected (PA embeddings are None) - forcing re-init")
                    needs_reinit = True
                    del st.session_state['beta_response_manager']

            if needs_reinit:
                logger.info("🔍 AUDIT: BetaResponseManager not in session - initializing...")
                logger.info("📦 Initializing BetaResponseManager")
                from services.beta_response_manager import BetaResponseManager
                # Pass backend client for delta transmission
                backend = st.session_state.get('backend')
                st.session_state.beta_response_manager = BetaResponseManager(self, backend)
                logger.info("🔍 AUDIT: BetaResponseManager initialized successfully")
            else:
                logger.info("🔍 AUDIT: BetaResponseManager already exists in session")

            # Get beta sequence and current turn number
            beta_sequence = st.session_state.get('beta_sequence', {})
            turn_number = st.session_state.get('beta_current_turn', 1)

            logger.info(f"🔍 AUDIT: Beta session state:")
            logger.info(f"   beta_current_turn = {turn_number}")
            logger.info(f"   beta_sequence exists = {beta_sequence is not None}")

            logger.info(f"🎯 BETA Turn {turn_number}: Generating dual responses (TELOS + Native)")
            logger.info(f"🔍 AUDIT: Calling BetaResponseManager.generate_turn_responses()...")

            # Generate BOTH responses via BetaResponseManager
            response_data = st.session_state.beta_response_manager.generate_turn_responses(
                user_input=message,
                turn_number=turn_number,
                sequence=beta_sequence
            )

            logger.info(f"🔍 AUDIT: Returned from BetaResponseManager.generate_turn_responses()")
            logger.info(f"   response_data keys: {list(response_data.keys())}")

            # Extract which response to show and metrics
            shown_response = response_data.get('shown_response', '')
            shown_source = response_data.get('shown_source', 'unknown')
            telos_analysis = response_data.get('telos_analysis', {})

            logger.info(f"🔍 AUDIT: Response extraction:")
            logger.info(f"   shown_source = '{shown_source}'")
            logger.info(f"   shown_response length = {len(shown_response)}")
            logger.info(f"   telos_analysis keys = {list(telos_analysis.keys())}")

            logger.info(f"📤 Displaying: {shown_source} response")
            logger.info(f"📊 TELOS Fidelity: {telos_analysis.get('fidelity_score', 'N/A')}")

            # Yield the displayed response
            yield shown_response

            # Update turn with REAL metrics from TELOS analysis
            turn = self.state.turns[turn_idx]
            turn['response'] = shown_response
            turn['fidelity'] = telos_analysis.get('user_pa_fidelity') or telos_analysis.get('fidelity_score') or 0.0
            turn['distance'] = telos_analysis.get('distance_from_pa', 0.0)
            turn['intervention_applied'] = telos_analysis.get('intervention_triggered', False)
            turn['drift_detected'] = telos_analysis.get('drift_detected', False)
            turn['in_basin'] = telos_analysis.get('in_basin', True)
            turn['is_streaming'] = False
            turn['is_loading'] = False

            # Store BETA-specific metadata
            turn['beta_shown_source'] = shown_source
            turn['beta_test_type'] = response_data.get('test_type', 'unknown')

            # Store comparison data for head-to-head turns
            if response_data.get('comparison_mode', False):
                turn['comparison_mode'] = True
                turn['response_a'] = response_data.get('response_a', '')  # TELOS
                turn['response_b'] = response_data.get('response_b', '')  # Native
                turn['beta_data'] = {
                    'telos_fidelity': telos_analysis.get('fidelity_score', 0.0),
                    'comparison_mode': True
                }
            else:
                turn['beta_data'] = {
                    'telos_fidelity': telos_analysis.get('fidelity_score', 0.0),
                    'shown_response_source': shown_source
                }

            # Log metrics for verification
            if turn['fidelity'] is not None and turn['fidelity'] < 0.3:
                logger.warning(f"⚠️ Low fidelity detected: {turn['fidelity']:.3f} - Drift alert!")

            # Increment beta turn counter
            st.session_state.beta_current_turn = turn_number + 1

            logger.info(f"✅ BETA Turn {turn_number} complete - Next turn: {turn_number + 1}")

        except Exception as e:
            logger.error(f"❌ BETA stream generation failed: {e}")
            logger.error(f"   Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"   Traceback: {traceback.format_exc()}")

            # Fallback to simple response
            fallback_response = "I apologize, but I encountered an error in BETA mode. Please try again."
            yield fallback_response

            # Update turn with minimal data
            turn = self.state.turns[turn_idx]
            turn['response'] = fallback_response
            turn['fidelity'] = 0.0
            turn['is_streaming'] = False
            turn['is_loading'] = False
            turn['error'] = str(e)
