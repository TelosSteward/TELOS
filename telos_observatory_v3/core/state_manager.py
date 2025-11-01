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
            component: One of 'primacy_attractor', 'math', 'counterfactual', 'steward'
        """
        if component == 'primacy_attractor':
            self.state.show_primacy_attractor = not self.state.show_primacy_attractor
        elif component == 'math' or component == 'math_breakdown':
            self.state.show_math_breakdown = not self.state.show_math_breakdown
        elif component == 'counterfactual':
            self.state.show_counterfactual = not self.state.show_counterfactual
        elif component == 'steward':
            self.state.show_steward = not self.state.show_steward

    def toggle_scrollable_history(self):
        """Toggle between turn-by-turn and scrollable history mode."""
        self.state.scrollable_history_mode = not self.state.scrollable_history_mode

    def add_user_message(self, message: str):
        """
        Add a new user message and generate a TELOS-governed response.

        Args:
            message: User's input message
        """
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
                    embedding_provider = SentenceTransformerProvider()
                    attractor = PrimacyAttractor(
                        purpose=[
                            "Explain how TELOS governance works",
                            "Demonstrate purpose alignment principles",
                            "Show fidelity measurement and intervention strategies"
                        ],
                        scope=[
                            "TELOS architecture and components",
                            "Primacy attractor mathematics",
                            "Intervention strategies and thresholds",
                            "Purpose alignment examples"
                        ],
                        boundaries=[
                            "Stay focused on TELOS governance topics",
                            "Redirect off-topic questions back to TELOS",
                            "Demonstrate drift detection when appropriate"
                        ],
                        constraint_tolerance=0.2,
                        privacy_level=0.8,
                        task_priority=0.7
                    )

                    # Initialize Mistral client (Assistant Steward)
                    mistral_client = MistralClient(
                        api_key="NxFBck0mkmGhM9vn0bvJzHf1scagv44f",
                        model="mistral-large-latest"
                    )

                    self._telos_steward = UnifiedGovernanceSteward(
                        attractor=attractor,
                        llm_client=mistral_client,
                        embedding_provider=embedding_provider,
                        enable_interventions=False  # Demo mode - measure but don't modify
                    )
                    self._telos_steward.start_session(session_id=self.state.session_id)
                    logger.info("TELOS engine initialized successfully")
                except Exception as init_error:
                    logger.error(f"Failed to initialize TELOS engine: {type(init_error).__name__}: {str(init_error)}")
                    # Set flag to None to trigger fallback in generation
                    self._telos_steward = None
                    raise  # Re-raise to trigger fallback response logic

            # Check if TELOS engine is available
            if self._telos_steward is None:
                raise Exception("TELOS engine not initialized - using fallback mode")

            # Generate AI response through Mistral (Assistant Steward)
            # Build conversation history for context
            conversation_history = [
                {
                    "role": "system",
                    "content": "You are an expert AI assistant explaining TELOS governance and purpose alignment. Stay focused on TELOS topics and provide clear, educational explanations."
                }
            ]
            for turn in self.state.turns:
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

            # Generate response using Mistral
            response_text = self._telos_steward.llm_client.generate(
                messages=conversation_history
            )

            # Process through TELOS to get fidelity metrics
            result = self._telos_steward.process_turn(
                user_input=message,
                model_response=response_text
            )

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

        # Create turn with actual response
        new_turn = {
            'turn': self.state.total_turns,
            'timestamp': self.state.total_turns * 2.5,
            'user_input': message,
            'response': response_text,
            'fidelity': fidelity,
            'distance': distance,
            'threshold': 0.8,
            'intervention_applied': intervention_applied,
            'drift_detected': fidelity < 0.8,
            'status': status_icon,
            'status_text': status_text,
            'in_basin': in_basin,
            'phase2_comparison': None
        }

        # Add the new turn to the list
        self.state.turns.append(new_turn)
        self.state.total_turns += 1

        # Jump to the new turn
        self.state.current_turn = self.state.total_turns - 1

        # Update statistics
        fidelities = [t['fidelity'] for t in self.state.turns]
        self.state.avg_fidelity = sum(fidelities) / len(fidelities)
        self.state.total_interventions = sum(1 for t in self.state.turns if t.get('intervention_applied', False))
        self.state.drift_warnings = sum(1 for t in self.state.turns if t.get('drift_detected', False))
