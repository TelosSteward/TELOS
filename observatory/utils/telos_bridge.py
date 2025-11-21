"""
TELOS Bridge - Connects dev pages to Observatory StateManager.
Provides real-time access to PA status, fidelity, interventions.
"""

import streamlit as st
from observatory.core.state_manager import StateManager


class TelosBridge:
    """Bridge between dev pages and Observatory runtime state."""

    def __init__(self, state_manager=None):
        """Initialize bridge to Observatory state.

        Args:
            state_manager: Existing StateManager instance from main app
        """
        self.state_manager = state_manager

    def get_pa_status(self):
        """Get current Primacy Attractor status."""
        if not self.state_manager:
            return {
                'state': 'not_available',
                'converged': False,
                'turns_to_convergence': None,
                'drift': None
            }

        state = self.state_manager.state
        pa_converged = getattr(state, 'pa_converged', False)
        current_turn = state.current_turn

        return {
            'state': 'converged' if pa_converged else 'calibrating',
            'converged': pa_converged,
            'turns_to_convergence': max(0, 10 - current_turn) if not pa_converged else 0,
            'current_turn': current_turn,
            'total_turns': state.total_turns,
            'drift': None  # TODO: Calculate from embedding distance
        }

    def get_fidelity_metrics(self):
        """Get fidelity tracking metrics."""
        if not self.state_manager:
            return {
                'current': None,
                'average': None,
                'trend': [],
                'violations': 0
            }

        state = self.state_manager.state

        # Extract fidelity from turns
        fidelity_scores = [
            turn.get('fidelity', 0.0)
            for turn in state.turns
            if 'fidelity' in turn
        ]

        if not fidelity_scores:
            return {
                'current': None,
                'average': None,
                'trend': [],
                'violations': 0
            }

        current_fidelity = fidelity_scores[-1] if fidelity_scores else None
        avg_fidelity = sum(fidelity_scores) / len(fidelity_scores)
        violations = sum(1 for f in fidelity_scores if f < 0.7)

        return {
            'current': round(current_fidelity, 3) if current_fidelity else None,
            'average': round(avg_fidelity, 3),
            'trend': fidelity_scores[-10:],  # Last 10 turns
            'violations': violations
        }

    def get_intervention_log(self):
        """Get intervention history."""
        if not self.state_manager:
            return []

        state = self.state_manager.state

        interventions = []
        for turn in state.turns:
            if turn.get('intervention_applied', False):
                interventions.append({
                    'turn': turn.get('turn_number', 0),
                    'type': turn.get('intervention_type', 'unknown'),
                    'reason': turn.get('intervention_reason', 'N/A'),
                    'distance': turn.get('distance_to_pa', None)
                })

        return interventions

    def get_session_stats(self):
        """Get overall session statistics."""
        if not self.state_manager:
            return {
                'session_id': 'No active session',
                'total_turns': 0,
                'total_interventions': 0,
                'avg_fidelity': None
            }

        state = self.state_manager.state

        return {
            'session_id': state.session_id,
            'total_turns': state.total_turns,
            'total_interventions': state.total_interventions,
            'avg_fidelity': round(state.avg_fidelity, 3) if state.avg_fidelity else None,
            'drift_warnings': getattr(state, 'drift_warnings', 0)
        }
