"""
Conversation Display Component for TELOS Observatory V3.
Renders ChatGPT/Claude-style conversation in center column.
"""

import streamlit as st
from typing import Dict, Any


class ConversationDisplay:
    """ChatGPT-style conversation display using native Streamlit."""

    def __init__(self, state_manager):
        """Initialize with state manager reference.

        Args:
            state_manager: StateManager instance for accessing turn data
        """
        self.state_manager = state_manager

    def render(self):
        """Render the conversation display."""
        # Get current turn data
        turn_data = self.state_manager.get_current_turn_data()

        if not turn_data:
            st.info("No conversation data loaded. Load a session to begin.")
            return

        # Container for conversation
        conv_container = st.container()

        with conv_container:
            # User message
            self._render_user_message(turn_data.get('user_input', ''))

            # Assistant response
            self._render_assistant_message(turn_data.get('response', ''))

            # Intervention indicator if applied
            if turn_data.get('intervention_applied', False):
                self._render_intervention_indicator()

    def _render_user_message(self, message: str):
        """Render user message bubble.

        Args:
            message: User's message text
        """
        st.markdown(f"""
        <div style="
            background-color: #2d2d2d;
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
            max-width: 80%;
        ">
            <div style="color: #888; font-size: 12px; margin-bottom: 5px;">
                <strong>User</strong>
            </div>
            <div style="color: #fff;">
                {message}
            </div>
        </div>
        """, unsafe_allow_html=True)

    def _render_assistant_message(self, message: str):
        """Render assistant message bubble.

        Args:
            message: Assistant's response text
        """
        st.markdown(f"""
        <div style="
            background-color: #1a1a1a;
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
            max-width: 80%;
        ">
            <div style="color: #888; font-size: 12px; margin-bottom: 5px;">
                <strong>Assistant</strong>
            </div>
            <div style="color: #fff;">
                {message}
            </div>
        </div>
        """, unsafe_allow_html=True)

    def _render_intervention_indicator(self):
        """Render indicator that TELOS intervention was applied."""
        st.markdown(f"""
        <div style="
            background-color: rgba(255, 215, 0, 0.1);
            border-left: 4px solid #FFD700;
            padding: 10px 15px;
            border-radius: 5px;
            margin: 10px 0;
            color: #FFD700;
            font-size: 13px;
        ">
            <strong>🔭 TELOS Intervention Applied</strong>
            <div style="color: #ccc; font-size: 12px; margin-top: 5px;">
                This response was modified to maintain alignment with user's deeper preferences.
            </div>
        </div>
        """, unsafe_allow_html=True)
