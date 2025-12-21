"""
Session State Manager for TELOSCOPE
===================================

Manages pristine, immutable session state snapshots for counterfactual branching.
Each turn is captured as an immutable snapshot that can be used as a fork point.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
import copy
import numpy as np


@dataclass(frozen=True)
class TurnSnapshot:
    """
    Immutable snapshot of a single turn state.

    This represents the complete state at a specific point in the conversation,
    allowing perfect reconstruction for counterfactual branching.

    Phase 4 Update: Now stores both Native and TELOS responses for governance toggle.
    """
    turn_number: int
    timestamp: str
    user_input: str

    # Phase 4: Dual response storage for governance toggle
    native_response: str  # Original LLM response (before governance)
    telos_response: str   # Governed response (after intervention)

    # Embeddings (immutable)
    user_embedding: tuple  # np.array converted to tuple for immutability
    response_embedding: tuple
    attractor_center: tuple

    # Metrics
    telic_fidelity: float
    error_signal: float
    lyapunov_value: float
    drift_distance: float
    basin_membership: bool

    # Conversation history up to this point (immutable)
    conversation_history: tuple  # List of {role, content} dicts converted to tuple

    # Attractor state
    attractor_config: Dict[str, Any]

    # Backward compatibility: deprecated field (use telos_response instead)
    assistant_response: Optional[str] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        # Convert tuples back to lists for JSON serialization
        data['user_embedding'] = list(self.user_embedding)
        data['response_embedding'] = list(self.response_embedding)
        data['attractor_center'] = list(self.attractor_center)
        data['conversation_history'] = [dict(msg) for msg in self.conversation_history]
        return data


class SessionStateManager:
    """
    Manages pristine session state snapshots for counterfactual experiments.

    Key properties:
    - Snapshots are immutable once created
    - Live session is never modified after turn completion
    - Each snapshot is a complete fork point
    - Supports time travel and replay
    """

    def __init__(self, web_session_manager: Optional[Any] = None):
        """
        Initialize session state manager.

        Args:
            web_session_manager: WebSessionManager for UI integration (optional)
        """
        self.web_manager = web_session_manager
        self._snapshots: List[TurnSnapshot] = []
        self._session_metadata: Dict[str, Any] = {
            'session_id': f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'started_at': datetime.now().isoformat(),
            'total_turns': 0
        }

    def save_turn_snapshot(
        self,
        turn_number: int,
        user_input: str,
        native_response: str,
        telos_response: str,
        user_embedding: np.ndarray,
        response_embedding: np.ndarray,
        attractor_center: np.ndarray,
        metrics: Dict[str, float],
        conversation_history: List[Dict[str, str]],
        attractor_config: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        # Backward compatibility: deprecated parameter
        assistant_response: Optional[str] = None
    ) -> TurnSnapshot:
        """
        Save an immutable turn snapshot.

        Phase 4 Update: Now saves both Native and TELOS responses for governance toggle.

        Args:
            turn_number: Turn number (0-indexed)
            user_input: User's input text
            native_response: Original LLM response (before governance)
            telos_response: Governed response (after intervention)
            user_embedding: Embedding of user input
            response_embedding: Embedding of assistant response
            attractor_center: Current attractor center
            metrics: Dict with telic_fidelity, error_signal, lyapunov_value, etc.
            conversation_history: Full conversation up to this point
            attractor_config: Attractor configuration dict
            metadata: Additional metadata
            assistant_response: DEPRECATED - Use telos_response instead

        Returns:
            Immutable TurnSnapshot object
        """
        # Backward compatibility: if old code passes assistant_response, use it for both
        if assistant_response is not None and native_response == "":
            native_response = assistant_response
            telos_response = assistant_response
        # Create immutable snapshot
        snapshot = TurnSnapshot(
            turn_number=turn_number,
            timestamp=datetime.now().isoformat(),
            user_input=user_input,
            native_response=native_response,
            telos_response=telos_response,
            assistant_response=telos_response,  # Backward compatibility: populate deprecated field
            user_embedding=tuple(user_embedding.flatten().tolist()),
            response_embedding=tuple(response_embedding.flatten().tolist()),
            attractor_center=tuple(attractor_center.flatten().tolist()),
            telic_fidelity=metrics.get('telic_fidelity', 1.0),
            error_signal=metrics.get('error_signal', 0.0),
            lyapunov_value=metrics.get('lyapunov_value', 0.0),
            drift_distance=metrics.get('drift_distance', 0.0),
            basin_membership=metrics.get('primacy_basin_membership', True),
            conversation_history=tuple(
                tuple(msg.items()) for msg in copy.deepcopy(conversation_history)
            ),
            attractor_config=copy.deepcopy(attractor_config),
            metadata=metadata or {}
        )

        # Store snapshot
        self._snapshots.append(snapshot)
        self._session_metadata['total_turns'] = len(self._snapshots)
        self._session_metadata['last_turn'] = snapshot.timestamp

        # Update web session if connected
        if self.web_manager:
            turn_data = {
                'turn_number': turn_number,
                'user_input': user_input,
                'native_response': native_response,
                'telos_response': telos_response,
                'assistant_response': telos_response,  # Backward compatibility
                'metrics': metrics,
                'timestamp': snapshot.timestamp
            }
            self.web_manager.add_turn(turn_data)

        return snapshot

    def get_turn_snapshot(self, turn_number: int) -> Optional[TurnSnapshot]:
        """
        Retrieve a specific turn snapshot.

        Args:
            turn_number: Turn number (0-indexed)

        Returns:
            TurnSnapshot or None if not found
        """
        if 0 <= turn_number < len(self._snapshots):
            return self._snapshots[turn_number]
        return None

    def get_latest_snapshot(self) -> Optional[TurnSnapshot]:
        """
        Get the most recent turn snapshot.

        Returns:
            Latest TurnSnapshot or None if no turns yet
        """
        if self._snapshots:
            return self._snapshots[-1]
        return None

    def get_all_snapshots(self) -> List[TurnSnapshot]:
        """
        Get all turn snapshots.

        Returns:
            List of TurnSnapshot objects (immutable)
        """
        return self._snapshots.copy()

    def get_snapshot_range(self, start: int, end: int) -> List[TurnSnapshot]:
        """
        Get a range of snapshots.

        Args:
            start: Start turn number (inclusive)
            end: End turn number (exclusive)

        Returns:
            List of snapshots in range
        """
        return self._snapshots[start:end]

    def get_all_triggers(self) -> List[Dict[str, Any]]:
        """
        Get all potential trigger points (turns with drift).

        Returns:
            List of trigger data dicts
        """
        triggers = []
        for snapshot in self._snapshots:
            # Identify triggers: low fidelity or outside basin (Goldilocks: Aligned threshold)
            if snapshot.telic_fidelity < 0.76 or not snapshot.basin_membership:
                triggers.append({
                    'turn_number': snapshot.turn_number,
                    'timestamp': snapshot.timestamp,
                    'fidelity': snapshot.telic_fidelity,
                    'error_signal': snapshot.error_signal,
                    'drift_distance': snapshot.drift_distance,
                    'basin_membership': snapshot.basin_membership,
                    'reason': self._get_trigger_reason(snapshot)
                })
        return triggers

    def _get_trigger_reason(self, snapshot: TurnSnapshot) -> str:
        """Generate human-readable trigger reason (Goldilocks zones)."""
        reasons = []
        if snapshot.telic_fidelity < 0.67:  # Goldilocks: Significant Drift
            reasons.append(f"Significant drift (F={snapshot.telic_fidelity:.3f})")
        elif snapshot.telic_fidelity < 0.76:  # Goldilocks: Below Aligned
            reasons.append(f"Drift detected (F={snapshot.telic_fidelity:.3f})")

        if not snapshot.basin_membership:
            reasons.append(f"Outside basin (d={snapshot.drift_distance:.3f})")

        if snapshot.error_signal > 0.5:
            reasons.append(f"High error signal (ε={snapshot.error_signal:.3f})")

        return "; ".join(reasons) if reasons else "Potential drift"

    def export_session(self) -> Dict[str, Any]:
        """
        Export complete session data.

        Returns:
            Session data dict with all snapshots
        """
        return {
            'session_metadata': self._session_metadata.copy(),
            'snapshots': [snapshot.to_dict() for snapshot in self._snapshots],
            'triggers': self.get_all_triggers()
        }

    def get_session_metadata(self) -> Dict[str, Any]:
        """
        Get session metadata.

        Returns:
            Metadata dict
        """
        return self._session_metadata.copy()

    def get_total_turns(self) -> int:
        """
        Get total number of turns.

        Returns:
            Number of turns
        """
        return len(self._snapshots)

    def clear_session(self) -> None:
        """
        Clear all snapshots and reset session.

        WARNING: This is destructive. Only use for starting new sessions.
        """
        self._snapshots = []
        self._session_metadata = {
            'session_id': f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'started_at': datetime.now().isoformat(),
            'total_turns': 0
        }

    def reconstruct_state_at_turn(self, turn_number: int) -> Optional[Dict[str, Any]]:
        """
        Reconstruct complete state at a specific turn for branching.

        Args:
            turn_number: Turn number to reconstruct

        Returns:
            State dict ready for branching, or None if turn not found
        """
        snapshot = self.get_turn_snapshot(turn_number)
        if not snapshot:
            return None

        # Reconstruct mutable state from immutable snapshot
        state = {
            'turn_number': snapshot.turn_number,
            'user_input': snapshot.user_input,
            'native_response': snapshot.native_response,
            'telos_response': snapshot.telos_response,
            'assistant_response': snapshot.telos_response,  # Backward compatibility
            'user_embedding': np.array(snapshot.user_embedding).reshape(-1, 1),
            'response_embedding': np.array(snapshot.response_embedding).reshape(-1, 1),
            'attractor_center': np.array(snapshot.attractor_center).reshape(-1, 1),
            'conversation_history': [
                {k: v for k, v in msg} for msg in snapshot.conversation_history
            ],
            'attractor_config': copy.deepcopy(snapshot.attractor_config),
            'metrics': {
                'telic_fidelity': snapshot.telic_fidelity,
                'error_signal': snapshot.error_signal,
                'lyapunov_value': snapshot.lyapunov_value,
                'drift_distance': snapshot.drift_distance,
                'primacy_basin_membership': snapshot.basin_membership
            },
            'metadata': copy.deepcopy(snapshot.metadata)
        }

        # Backward compatibility: handle old snapshots with only assistant_response
        if not hasattr(snapshot, 'native_response') or not snapshot.native_response:
            # Old data: use assistant_response for both
            state['native_response'] = snapshot.assistant_response or ""
            state['telos_response'] = snapshot.assistant_response or ""

        return state

    def get_fidelity_history(self) -> List[float]:
        """
        Get fidelity values for all turns.

        Returns:
            List of fidelity values
        """
        return [snapshot.telic_fidelity for snapshot in self._snapshots]

    def get_distance_history(self) -> List[float]:
        """
        Get drift distance values for all turns.

        Returns:
            List of distance values
        """
        return [snapshot.drift_distance for snapshot in self._snapshots]

    def get_basin_history(self) -> List[bool]:
        """
        Get basin membership for all turns.

        Returns:
            List of basin membership booleans
        """
        return [snapshot.basin_membership for snapshot in self._snapshots]

    def get_all_turns(self) -> List[Dict[str, Any]]:
        """
        Get all turns as dictionary format for dashboard compatibility.

        Phase 4: Returns turns with both native_response and telos_response fields.

        Returns:
            List of turn dictionaries
        """
        turns = []
        for snapshot in self._snapshots:
            # Build turn dict with backward compatibility
            turn_dict = {
                'turn_number': snapshot.turn_number,
                'user_message': snapshot.user_input,
                'user_input': snapshot.user_input,  # Alias
                'native_response': snapshot.native_response if hasattr(snapshot, 'native_response') else snapshot.assistant_response,
                'telos_response': snapshot.telos_response if hasattr(snapshot, 'telos_response') else snapshot.assistant_response,
                'assistant_response': snapshot.telos_response if hasattr(snapshot, 'telos_response') else snapshot.assistant_response,  # Default to TELOS response
                'timestamp': snapshot.timestamp,
                'fidelity': snapshot.telic_fidelity,
                'governance_metadata': snapshot.metadata
            }
            turns.append(turn_dict)
        return turns
