"""
Control Strip Component for TELOS Observatory V3.
Top-right clickable strip that displays current turn metrics and opens Observation Deck.
"""

import streamlit as st


class ControlStrip:
    """Clickable control strip using native Streamlit buttons."""

    def __init__(self, state_manager):
        """Initialize with state manager reference.

        Args:
            state_manager: StateManager instance for turn data and deck control
        """
        self.state_manager = state_manager

    def render(self):
        """Render the control strip as a clickable button in top-right."""
        # Get current turn data
        turn_data = self.state_manager.get_current_turn_data()
        session_info = self.state_manager.get_session_info()

        if not turn_data:
            # No data loaded yet
            st.markdown("""
            <div style="
                text-align: right;
                padding: 10px 20px;
                background-color: #2d2d2d;
                border-radius: 8px;
                margin-bottom: 10px;
            ">
                <span style="color: #888; font-size: 14px;">
                    🔭 No session loaded
                </span>
            </div>
            """, unsafe_allow_html=True)
            return

        # Build strip content
        turn_num = session_info.get('current_turn', 0) + 1
        total_turns = session_info.get('total_turns', 0)
        fidelity = turn_data.get('fidelity', 0.0)
        status = turn_data.get('status_text', 'Calibrating...')

        # Color coding for fidelity
        if fidelity >= 0.8:
            fidelity_color = "#4CAF50"  # Green
        elif fidelity >= 0.6:
            fidelity_color = "#FFA500"  # Orange
        else:
            fidelity_color = "#FF5252"  # Red

        # Create columns for strip layout (push to right)
        col1, col2 = st.columns([3, 1])

        with col2:
            # Clickable strip button
            button_label = f"🔭 Turn {turn_num}/{total_turns} | Fidelity: {fidelity:.2f} | {status}"

            if st.button(
                button_label,
                key="control_strip_toggle",
                use_container_width=True,
                help="Click to toggle Observation Deck"
            ):
                self.state_manager.toggle_deck()
                st.rerun()

            # Additional visual feedback for deck state
            if self.state_manager.is_deck_expanded():
                st.markdown("""
                <div style="
                    text-align: center;
                    color: #FFD700;
                    font-size: 11px;
                    margin-top: -10px;
                ">
                    Deck Open ▼
                </div>
                """, unsafe_allow_html=True)
