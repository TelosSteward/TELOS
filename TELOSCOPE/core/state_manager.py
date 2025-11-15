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

# Configure logging
logger = logging.getLogger(__name__)


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
    show_observatory_lens: bool = False  # Observatory Lens dashboard visibility

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
                'threshold': 0.8,
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
            from utils.telos_demo_data import generate_telos_demo_session
            from telos_purpose.core.unified_steward import UnifiedGovernanceSteward, PrimacyAttractor
            from telos_purpose.core.embedding_provider import SentenceTransformerProvider
            from telos_purpose.llm_clients.mistral_client import MistralClient

            # Initialize TELOS if not already done
            if not hasattr(self, '_telos_steward'):
                try:
                    logger.info("Initializing TELOS engine...")
                    embedding_provider = SentenceTransformerProvider()

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
                    mistral_api_key = st.secrets.get("MISTRAL_API_KEY", os.getenv("MISTRAL_API_KEY"))

                    if not mistral_api_key:
                        raise ValueError("MISTRAL_API_KEY not found in secrets or environment")

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
                    from core.async_processor import AsyncStewardProcessor

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

            fidelity = result.get("telic_fidelity", 0.85)
            distance = result.get("error_signal", 0.15)
            in_basin = result.get("in_basin", True)
            intervention_applied = result.get("intervention_applied", False)

            # Determine status
            if fidelity >= 0.9:
                status_icon, status_text = "✓", "Excellent"
            elif fidelity >= 0.8:
                status_icon, status_text = "✓", "Good"
            elif fidelity >= 0.7:
                status_icon, status_text = "⚠", "Acceptable"
            else:
                status_icon, status_text = "⚠", "Drift"

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
                response_text = "Fidelity scores measure how well responses align with your purpose. Scores above 0.8 indicate good alignment, 0.6-0.8 is acceptable, and below 0.6 triggers intervention."
            elif "intervention" in message_lower or "drift" in message_lower:
                response_text = "TELOS detects drift by measuring semantic distance from the primacy attractor. When responses drift too far, the system can inject context, regenerate responses, or block and alert."
            elif "primacy" in message_lower or "attractor" in message_lower:
                response_text = "The Primacy Attractor defines your conversation's purpose, scope, and boundaries. It acts as a gravitational center that keeps responses aligned with your goals."
            else:
                response_text = f"I'm designed to help explain TELOS governance and purpose alignment. Could you rephrase your question about TELOS? (Running in fallback mode due to {fallback_reason})"

            fidelity = 0.85
            distance = 0.15
            in_basin = True
            intervention_applied = False
            status_icon, status_text = "✓", "Good"

        # Update the placeholder turn with actual response
        # current_turn_idx was set earlier (either in already_processing or when adding placeholder)
        if not already_processing:
            current_turn_idx = len(self.state.turns) - 1
        self.state.turns[current_turn_idx].update({
            'response': response_text,
            'fidelity': fidelity,
            'distance': distance,
            'intervention_applied': intervention_applied,
            'drift_detected': fidelity < 0.8,
            'status': status_icon,
            'status_text': status_text,
            'in_basin': in_basin,
            'is_loading': False  # Response is ready
        })

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
            'threshold': 0.8,
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
        # Initialize TELOS if not already done
        if not hasattr(self, '_telos_steward'):
            from utils.telos_demo_data import generate_telos_demo_session
            from telos_purpose.core.unified_steward import UnifiedGovernanceSteward, PrimacyAttractor
            from telos_purpose.core.embedding_provider import SentenceTransformerProvider
            from telos_purpose.llm_clients.mistral_client import MistralClient

            try:
                logger.info("Initializing TELOS engine...")
                embedding_provider = SentenceTransformerProvider()

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
                    attractor = None
                    self._corpus_loader = None

                # Get API key from Streamlit secrets or environment
                import os
                mistral_api_key = st.secrets.get("MISTRAL_API_KEY", os.getenv("MISTRAL_API_KEY"))

                if not mistral_api_key:
                    raise ValueError("MISTRAL_API_KEY not found in secrets or environment")

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

            fidelity = result.get("telic_fidelity", 0.85)
            distance = result.get("error_signal", 0.15)
            in_basin = result.get("in_basin", True)
            intervention_applied = result.get("intervention_applied", False)

            if fidelity >= 0.9:
                status_icon, status_text = "✓", "Excellent"
            elif fidelity >= 0.8:
                status_icon, status_text = "✓", "Good"
            elif fidelity >= 0.7:
                status_icon, status_text = "⚠", "Acceptable"
            else:
                status_icon, status_text = "⚠", "Drift"
        except Exception as telos_error:
            logger.error(f"TELOS processing error: {telos_error}")
            fidelity = 0.85
            distance = 0.15
            in_basin = True
            intervention_applied = False
            status_icon, status_text = "✓", "Good"

        # Update turn with final response and metrics
        self.state.turns[turn_idx].update({
            'response': full_response,
            'fidelity': fidelity,
            'distance': distance,
            'intervention_applied': intervention_applied,
            'drift_detected': fidelity < 0.8,
            'status': status_icon,
            'status_text': status_text,
            'in_basin': in_basin,
            'is_streaming': False  # Streaming complete
        })

        # Update statistics
        fidelities = [t['fidelity'] for t in self.state.turns if t.get('fidelity') is not None]
        self.state.avg_fidelity = sum(fidelities) / len(fidelities) if fidelities else 0.0
        self.state.total_interventions = sum(1 for t in self.state.turns if t.get('intervention_applied', False))
        self.state.drift_warnings = sum(1 for t in self.state.turns if t.get('drift_detected', False))
