"""
Observatory V2 - State Manager
===============================

Centralized state management for the Observatory application.
All state lives here, components read and update through this manager.

Design: Single source of truth, no scattered session_state usage.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


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

    # UI state
    deck_expanded: bool = False
    teloscope_playing: bool = False
    playback_speed: float = 1.0

    # Component visibility
    show_math_breakdown: bool = False
    show_counterfactual: bool = False
    show_steward: bool = False

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
        if self._initialized:
            return

        # Load session data
        self.state.session_id = session_data.get('session_id', 'unknown')
        self.state.turns = session_data.get('turns', [])
        self.state.total_turns = len(self.state.turns)

        # Load statistics if available
        stats = session_data.get('statistics', {})
        self.state.avg_fidelity = stats.get('avg_fidelity', 0.0)
        self.state.total_interventions = stats.get('interventions', 0)
        self.state.drift_warnings = stats.get('drift_warnings', 0)

        self._initialized = True

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
            component: One of 'math', 'counterfactual', 'steward'
        """
        if component == 'math':
            self.state.show_math_breakdown = not self.state.show_math_breakdown
        elif component == 'counterfactual':
            self.state.show_counterfactual = not self.state.show_counterfactual
        elif component == 'steward':
            self.state.show_steward = not self.state.show_steward
