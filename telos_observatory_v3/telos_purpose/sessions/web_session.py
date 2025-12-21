"""
Web Session Manager for TELOSCOPE
=================================

Bridges SessionStateManager with Streamlit's st.session_state for web persistence.
Handles web-specific state management and provides callbacks for UI updates.
"""

from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
import json
from pathlib import Path


class WebSessionManager:
    """
    Manages web session state and persistence for TELOSCOPE dashboard.

    This class acts as a bridge between backend session management and
    Streamlit's session state, enabling real-time UI updates and state
    persistence across page reruns.
    """

    def __init__(self, streamlit_session_state: Optional[Dict] = None):
        """
        Initialize web session manager.

        Args:
            streamlit_session_state: Reference to st.session_state (injected)
        """
        self.st_state = streamlit_session_state
        self._ui_callbacks: List[Callable] = []

    def initialize_web_session(self) -> None:
        """Initialize Streamlit session state with required keys."""
        if self.st_state is None:
            raise RuntimeError("Streamlit session state not injected")

        # Core session data
        if 'session_id' not in self.st_state:
            self.st_state['session_id'] = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        if 'current_session' not in self.st_state:
            self.st_state['current_session'] = {
                'turns': [],
                'triggers': [],
                'branches': {},
                'metadata': {
                    'started_at': datetime.now().isoformat(),
                    'total_turns': 0,
                    'total_triggers': 0
                }
            }

        # UI state
        if 'selected_trigger' not in self.st_state:
            self.st_state['selected_trigger'] = None

        if 'replay_turn' not in self.st_state:
            self.st_state['replay_turn'] = 0

        if 'replay_playing' not in self.st_state:
            self.st_state['replay_playing'] = False

        # Live metrics
        if 'live_metrics' not in self.st_state:
            self.st_state['live_metrics'] = {
                'current_fidelity': 1.0,
                'current_distance': 0.0,
                'basin_status': True,
                'last_update': None
            }

    def save_to_web_state(self, session_data: Dict[str, Any]) -> None:
        """
        Persist session data to Streamlit state.

        Args:
            session_data: Session data dict to persist
        """
        if self.st_state is None:
            return

        self.st_state['current_session'] = session_data
        self.st_state['current_session']['metadata']['last_updated'] = datetime.now().isoformat()

        # Trigger UI callbacks
        self._trigger_callbacks('session_updated', session_data)

    def load_from_web_state(self) -> Dict[str, Any]:
        """
        Retrieve session data from Streamlit state.

        Returns:
            Session data dict
        """
        if self.st_state is None:
            return {}

        return self.st_state.get('current_session', {})

    def add_turn(self, turn_data: Dict[str, Any]) -> None:
        """
        Add a turn to the current session.

        Args:
            turn_data: Turn data including user input, response, metrics
        """
        if self.st_state is None:
            return

        session = self.st_state['current_session']
        session['turns'].append(turn_data)
        session['metadata']['total_turns'] = len(session['turns'])
        session['metadata']['last_turn'] = datetime.now().isoformat()

        # Update live metrics
        if 'metrics' in turn_data:
            metrics = turn_data['metrics']
            self.st_state['live_metrics'] = {
                'current_fidelity': metrics.get('telic_fidelity', 1.0),
                'current_distance': metrics.get('drift_distance', 0.0),
                'basin_status': metrics.get('primacy_basin_membership', True),
                'last_update': datetime.now().isoformat()
            }

        # Trigger UI callbacks
        self._trigger_callbacks('turn_added', turn_data)

    def add_trigger(self, trigger_data: Dict[str, Any]) -> str:
        """
        Add a counterfactual trigger point.

        Args:
            trigger_data: Trigger information (turn, metrics, etc.)

        Returns:
            Trigger ID
        """
        if self.st_state is None:
            return ""

        trigger_id = f"trigger_{len(self.st_state['current_session']['triggers'])}"
        trigger_data['trigger_id'] = trigger_id
        trigger_data['timestamp'] = datetime.now().isoformat()

        session = self.st_state['current_session']
        session['triggers'].append(trigger_data)
        session['metadata']['total_triggers'] = len(session['triggers'])

        # Trigger UI callbacks
        self._trigger_callbacks('trigger_created', trigger_data)

        return trigger_id

    def add_branch(self, trigger_id: str, branch_data: Dict[str, Any]) -> None:
        """
        Add counterfactual branch data for a trigger.

        Args:
            trigger_id: ID of the trigger point
            branch_data: Branch comparison data
        """
        if self.st_state is None:
            return

        session = self.st_state['current_session']
        if 'branches' not in session:
            session['branches'] = {}

        session['branches'][trigger_id] = branch_data

        # Trigger UI callbacks
        self._trigger_callbacks('branch_generated', {'trigger_id': trigger_id, 'data': branch_data})

    def get_all_triggers(self) -> List[Dict[str, Any]]:
        """
        Get all trigger points in the current session.

        Returns:
            List of trigger data dicts
        """
        if self.st_state is None:
            return []

        return self.st_state['current_session'].get('triggers', [])

    def get_trigger(self, trigger_id: str) -> Optional[Dict[str, Any]]:
        """
        Get specific trigger data.

        Args:
            trigger_id: Trigger identifier

        Returns:
            Trigger data or None
        """
        triggers = self.get_all_triggers()
        for trigger in triggers:
            if trigger.get('trigger_id') == trigger_id:
                return trigger
        return None

    def get_branch(self, trigger_id: str) -> Optional[Dict[str, Any]]:
        """
        Get counterfactual branch data for a trigger.

        Args:
            trigger_id: Trigger identifier

        Returns:
            Branch data or None
        """
        if self.st_state is None:
            return None

        return self.st_state['current_session'].get('branches', {}).get(trigger_id)

    def select_trigger(self, trigger_id: Optional[str]) -> None:
        """
        Set the currently selected trigger for UI display.

        Args:
            trigger_id: Trigger to select, or None to deselect
        """
        if self.st_state is None:
            return

        self.st_state['selected_trigger'] = trigger_id

        # Trigger UI callbacks
        self._trigger_callbacks('trigger_selected', {'trigger_id': trigger_id})

    def get_selected_trigger(self) -> Optional[str]:
        """
        Get the currently selected trigger ID.

        Returns:
            Selected trigger ID or None
        """
        if self.st_state is None:
            return None

        return self.st_state.get('selected_trigger')

    def set_replay_turn(self, turn_number: int) -> None:
        """
        Set the replay position.

        Args:
            turn_number: Turn number to display (0-indexed)
        """
        if self.st_state is None:
            return

        max_turns = len(self.st_state['current_session'].get('turns', []))
        self.st_state['replay_turn'] = max(0, min(turn_number, max_turns - 1))

    def get_replay_turn(self) -> int:
        """
        Get current replay position.

        Returns:
            Current replay turn number
        """
        if self.st_state is None:
            return 0

        return self.st_state.get('replay_turn', 0)

    def clear_web_session(self) -> None:
        """Reset session for a new conversation."""
        if self.st_state is None:
            return

        self.st_state['session_id'] = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.st_state['current_session'] = {
            'turns': [],
            'triggers': [],
            'branches': {},
            'metadata': {
                'started_at': datetime.now().isoformat(),
                'total_turns': 0,
                'total_triggers': 0
            }
        }
        self.st_state['selected_trigger'] = None
        self.st_state['replay_turn'] = 0
        self.st_state['live_metrics'] = {
            'current_fidelity': 1.0,
            'current_distance': 0.0,
            'basin_status': True,
            'last_update': None
        }

        # Trigger UI callbacks
        self._trigger_callbacks('session_cleared', {})

    def export_session(self) -> str:
        """
        Export current session as JSON string.

        Returns:
            JSON string of session data
        """
        if self.st_state is None:
            return "{}"

        session_data = self.st_state['current_session'].copy()
        session_data['session_id'] = self.st_state['session_id']
        session_data['exported_at'] = datetime.now().isoformat()

        return json.dumps(session_data, indent=2, ensure_ascii=False)

    def import_session(self, json_data: str) -> bool:
        """
        Import session from JSON string.

        Args:
            json_data: JSON string of session data

        Returns:
            True if successful, False otherwise
        """
        try:
            session_data = json.loads(json_data)

            if self.st_state is None:
                return False

            self.st_state['session_id'] = session_data.get('session_id', self.st_state['session_id'])
            self.st_state['current_session'] = {
                'turns': session_data.get('turns', []),
                'triggers': session_data.get('triggers', []),
                'branches': session_data.get('branches', {}),
                'metadata': session_data.get('metadata', {})
            }

            # Trigger UI callbacks
            self._trigger_callbacks('session_imported', session_data)

            return True
        except Exception:
            return False

    def register_callback(self, callback: Callable[[str, Any], None]) -> None:
        """
        Register a callback function for UI updates.

        Callback signature: callback(event_name: str, data: Any) -> None

        Events: 'session_updated', 'turn_added', 'trigger_created',
                'branch_generated', 'trigger_selected', 'session_cleared', 'session_imported'

        Args:
            callback: Callback function
        """
        self._ui_callbacks.append(callback)

    def _trigger_callbacks(self, event_name: str, data: Any) -> None:
        """
        Trigger all registered UI callbacks.

        Args:
            event_name: Name of the event
            data: Event data to pass to callbacks
        """
        for callback in self._ui_callbacks:
            try:
                callback(event_name, data)
            except Exception as e:
                # Silently catch callback errors to prevent crashes
                print(f"Callback error for {event_name}: {e}")

    def get_session_stats(self) -> Dict[str, Any]:
        """
        Get session statistics for display.

        Returns:
            Statistics dict with counts and metrics
        """
        if self.st_state is None:
            return {}

        session = self.st_state['current_session']
        turns = session.get('turns', [])
        triggers = session.get('triggers', [])

        # Calculate aggregate metrics
        if turns:
            fidelities = [t.get('metrics', {}).get('telic_fidelity', 1.0) for t in turns]
            avg_fidelity = sum(fidelities) / len(fidelities)
            min_fidelity = min(fidelities)
            max_fidelity = max(fidelities)
        else:
            avg_fidelity = min_fidelity = max_fidelity = 1.0

        return {
            'session_id': self.st_state.get('session_id'),
            'total_turns': len(turns),
            'total_triggers': len(triggers),
            'avg_fidelity': avg_fidelity,
            'min_fidelity': min_fidelity,
            'max_fidelity': max_fidelity,
            'trigger_rate': len(triggers) / len(turns) if turns else 0.0,
            'branches_generated': len(session.get('branches', {})),
            'started_at': session.get('metadata', {}).get('started_at'),
            'last_turn': session.get('metadata', {}).get('last_turn')
        }
