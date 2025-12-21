"""
Observation Deck Control Strip - Sidebar Header

Controls and status for the Observation Deck itself.
Lives at the top of the right sidebar when Deck is present.

Components:
1. Telescope Icon: Toggle button to open/close Observation Deck
2. Symbolic Flow: Governance pipeline visualization (👤→⚡→🔄→🤖→✓)
3. Stats Summary: Intervention count, fidelity snapshot
4. Early Warning: Animated indicators for drift

Design Philosophy:
- Summoning instrument for the Deck
- Shows what's happening "behind the scenes"
- Quick glance shows governance health
- Early warning system before problems escalate
"""

import streamlit as st
from typing import Dict, Any


class DeckControlStrip:
    """
    Renders the Observation Deck Control Strip at top of sidebar.

    This is the "summoning instrument" that opens/closes the Deck and
    provides a symbolic overview of governance activity.
    """

    def __init__(self, session_manager, deck_manager):
        """
        Initialize Deck Control Strip.

        Args:
            session_manager: WebSessionManager instance
            deck_manager: DeckManager instance for controlling Deck state
        """
        self.session_manager = session_manager
        self.deck_manager = deck_manager

    def render(self):
        """
        Render the Deck Control Strip at top of sidebar.

        Shows telescope toggle, symbolic flow, and stats.
        """
        # Get current session data for stats
        session_data = self._get_session_data()
        governance_state = self._get_governance_state()

        # CSS for control strip
        st.markdown("""
            <style>
            .deck-control-strip {
                background: rgba(40, 40, 40, 0.8);
                border: 1px solid rgba(255, 255, 255, 0.2);
                border-radius: 8px;
                padding: 12px;
                margin-bottom: 15px;
            }
            .deck-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 10px;
            }
            .deck-title {
                font-size: 14px;
                font-weight: 600;
                color: rgba(255, 255, 255, 0.9);
            }
            .deck-stats {
                display: flex;
                gap: 15px;
                font-size: 11px;
                color: rgba(255, 255, 255, 0.7);
            }
            .deck-stat-item {
                display: flex;
                flex-direction: column;
                align-items: center;
            }
            .deck-stat-value {
                font-size: 16px;
                font-weight: 700;
                color: #00aaff;
            }
            .deck-stat-label {
                font-size: 9px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            </style>
        """, unsafe_allow_html=True)

        # Header with title and toggle
        col1, col2 = st.columns([3, 1])

        with col1:
            st.markdown("### 🔭 Observation Deck")

        with col2:
            is_open = self.deck_manager.session_state['observation_deck']['is_open']
            if st.button("📐" if not is_open else "✖", key="deck_toggle", help="Toggle Observation Deck"):
                self.deck_manager.toggle_deck()
                st.rerun()

        # Stats summary (compact)
        self._render_stats_summary(session_data)

        # Symbolic flow (if governance active)
        if governance_state.get('governance_enabled', False):
            st.markdown("---")
            self._render_symbolic_flow(governance_state)

    def _render_symbolic_flow(self, governance_state: Dict[str, Any]):
        """
        Render symbolic flow visualization of governance pipeline.

        Args:
            governance_state: Current governance state with telemetry
        """
        # Use SymbolicFlow component
        from .symbolic_flow import SymbolicFlow

        flow_viz = SymbolicFlow(self.session_manager)
        flow_viz.render(governance_state)

    def _render_stats_summary(self, session_data: Dict[str, Any]):
        """
        Render quick stats summary.

        Args:
            session_data: Current session data
        """
        # Extract metrics
        total_turns = session_data.get('total_turns', 0)
        interventions = session_data.get('total_interventions', 0)
        avg_fidelity = session_data.get('avg_fidelity', 0.0)
        drift_warnings = session_data.get('drift_warnings', 0)

        # Render compact stats
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Turns", total_turns, help="Total conversation turns")

        with col2:
            color = "#00ff00" if avg_fidelity >= 0.76 else "#ffaa00" if avg_fidelity >= 0.67 else "#ff0000"  # Goldilocks zones
            st.metric("Fidelity", f"{avg_fidelity:.2f}", help="Average governance fidelity")

        with col3:
            st.metric("Interventions", interventions, help="Total governance interventions")

        # Drift warning (if any)
        if drift_warnings > 0:
            st.warning(f"⚠️ {drift_warnings} drift warning(s) detected")

    def _get_session_data(self) -> Dict[str, Any]:
        """
        Get current session data with computed stats.

        Returns:
            Dictionary with session metrics
        """
        try:
            # TODO: Wire to WebSessionManager
            # For now, return stub data
            return {
                'total_turns': 0,
                'total_interventions': 0,
                'avg_fidelity': 0.0,
                'drift_warnings': 0
            }
        except Exception as e:
            return {
                'total_turns': 0,
                'total_interventions': 0,
                'avg_fidelity': 0.0,
                'drift_warnings': 0
            }

    def _get_governance_state(self) -> Dict[str, Any]:
        """
        Get current governance state.

        Returns:
            Dictionary with governance_enabled, drift_detected flags
        """
        try:
            # TODO: Wire to UnifiedGovernanceSteward
            # For now, return stub data
            return {
                'governance_enabled': False,
                'drift_detected': False
            }
        except Exception as e:
            return {
                'governance_enabled': False,
                'drift_detected': False
            }
