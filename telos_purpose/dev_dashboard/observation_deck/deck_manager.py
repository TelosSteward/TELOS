"""
Deck Manager - Main Coordinator for Observation Deck

Manages the overall Observation Deck state and coordinates between all components.
Handles dynamic layout, component rendering, and turn marker synchronization.
"""

import streamlit as st
from typing import Dict, Any, Optional
from telos_purpose.dev_dashboard.observation_deck.teloscopic_tools.turn_navigator import TurnNavigator


class DeckManager:
    """
    Manages Observation Deck state and component coordination.

    Responsibilities:
    - Track deck open/closed state
    - Calculate dynamic column widths based on sidebar/deck state
    - Coordinate turn marker synchronization across tools
    - Render Observatory Control Strip (top-right thermometer)
    - Render Observation Deck Control Strip (sidebar header)
    - Manage component visibility and layout
    """

    def __init__(self, session_state: Dict[str, Any]):
        """
        Initialize DeckManager with Streamlit session state.

        Args:
            session_state: Streamlit session state dictionary
        """
        self.session_state = session_state
        self._initialize_deck_state()

    def _initialize_deck_state(self):
        """Initialize Observation Deck state in session."""
        if 'observation_deck' not in self.session_state:
            self.session_state['observation_deck'] = {
                'is_open': False,  # Deck starts closed
                'active_tool': None,  # Current TELOSCOPIC tool
                'current_turn': 0,  # Turn marker for synchronization
            }

    def get_column_widths(self) -> list:
        """
        Calculate dynamic column widths based on sidebar and deck state.

        Returns:
            List of column width ratios [sidebar, chat, deck]

        States:
        - Sidebar open + Deck open: [15, 60, 25]
        - Sidebar open + Deck closed: [15, 85, 0]
        - Sidebar closed + Deck open: [0, 60, 40]
        - Both closed: [0, 100, 0]
        """
        sidebar_open = self.session_state.get('sidebar_state', {}).get('is_open', True)
        deck_open = self.session_state['observation_deck']['is_open']

        if sidebar_open and deck_open:
            return [15, 60, 25]
        elif sidebar_open and not deck_open:
            return [15, 85, 0]
        elif not sidebar_open and deck_open:
            return [0, 60, 40]
        else:
            return [0, 100, 0]

    def toggle_deck(self):
        """Toggle Observation Deck open/closed state."""
        current_state = self.session_state['observation_deck']['is_open']
        self.session_state['observation_deck']['is_open'] = not current_state

    def set_active_tool(self, tool_name: Optional[str]):
        """
        Set the currently active TELOSCOPIC tool.

        Args:
            tool_name: Name of tool ('comparison', 'calculation', 'navigator', 'steward', None)
        """
        self.session_state['observation_deck']['active_tool'] = tool_name

    def set_current_turn(self, turn_number: int):
        """
        Set current turn marker (for synchronization across tools).

        Args:
            turn_number: Turn number to jump to
        """
        self.session_state['observation_deck']['current_turn'] = turn_number

    def render_observatory_control_strip(self):
        """
        Render Observatory Control Strip (top-right thermometer).

        Displays:
        - Turn counter
        - Fidelity gauge (color-coded)
        - Calibration progress (Turns 1-3 only)
        - Pulse animations for drift
        """
        # TODO: Implement Observatory Control Strip rendering
        pass

    def render_observation_deck_control_strip(self):
        """
        Render Observation Deck Control Strip (sidebar header).

        Displays:
        - Telescope icon (toggle button)
        - Symbolic flow (👤→⚡→🔄→🤖→✓)
        - Stats summary
        - Early warning indicators
        """
        # TODO: Implement Observation Deck Control Strip rendering
        pass

    def render_observation_deck(self):
        """
        Render the full Observation Deck with active tool.

        Displays the currently active TELOSCOPIC tool or Steward chat.
        """
        if not self.session_state['observation_deck']['is_open']:
            return  # Deck is closed, don't render

        active_tool = self.session_state['observation_deck']['active_tool']

        # Render appropriate tool based on active selection
        if active_tool == 'comparison':
            self._render_comparison_viewer()
        elif active_tool == 'calculation':
            self._render_calculation_window()
        elif active_tool == 'navigator':
            self._render_turn_navigator()
        elif active_tool == 'steward':
            self._render_steward_chat()
        elif active_tool == 'calibration':
            self._render_calibration_logger()
        else:
            # Default: Show tool selector
            self._render_tool_selector()

    def _render_tool_selector(self):
        """Render tool selection interface."""
        st.markdown("### 🔭 TELOSCOPIC Tools")
        st.markdown("Select a research instrument:")

        # Turn Navigator
        if st.button("⏯️ Turn Navigator", use_container_width=True, key="select_navigator"):
            self.set_active_tool('navigator')
            st.rerun()

        # Comparison Viewer
        if st.button("🔀 Comparison Viewer", use_container_width=True, key="select_comparison", disabled=True):
            self.set_active_tool('comparison')
            st.rerun()

        # Calculation Window
        if st.button("🧮 Calculation Window", use_container_width=True, key="select_calculation", disabled=True):
            self.set_active_tool('calculation')
            st.rerun()

        # Calibration Logger (only for turns 1-3)
        total_turns = self.session_state.get('current_session', {}).get('metadata', {}).get('total_turns', 0)
        if total_turns <= 3:
            if st.button("📊 Calibration Logger", use_container_width=True, key="select_calibration", disabled=True):
                self.set_active_tool('calibration')
                st.rerun()

        # Steward Chat
        if st.button("💬 Steward Chat", use_container_width=True, key="select_steward", disabled=True):
            self.set_active_tool('steward')
            st.rerun()

        st.markdown("---")
        st.caption("🔨 Tools marked as disabled are coming soon")

    def _render_comparison_viewer(self):
        """Render Comparison Viewer tool."""
        # TODO: Wire to ComparisonViewer component
        pass

    def _render_calculation_window(self):
        """Render Calculation Window tool."""
        # TODO: Wire to CalculationWindow component
        pass

    def _render_turn_navigator(self):
        """Render Turn Navigator tool."""
        # Get web session manager from session state
        web_session = self.session_state.get('web_session')
        if web_session:
            navigator = TurnNavigator(
                session_manager=web_session,
                deck_manager=self
            )
            navigator.render()
        else:
            st.error("⚠️ Session manager not initialized")

    def _render_steward_chat(self):
        """Render Steward Chat interface."""
        # TODO: Wire to StewardChat component
        pass

    def _render_calibration_logger(self):
        """Render Calibration Logger (Turns 1-3)."""
        # TODO: Wire to CalibrationLogger component
        pass
