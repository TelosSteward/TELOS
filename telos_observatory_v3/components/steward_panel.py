"""
Steward Panel Component for TELOS Observatory V3.
Provides a helpful AI guide accessible via handshake emoji button.
"""

import streamlit as st
from datetime import datetime


class StewardPanel:
    """Side panel with Steward assistant for TELOS guidance."""

    def __init__(self, state_manager):
        """Initialize with state manager reference.

        Args:
            state_manager: StateManager instance for session operations
        """
        self.state_manager = state_manager

    def render_button(self):
        """Render button is now integrated in conversation_display.py - this method is deprecated."""
        pass

    def render_panel(self):
        """Render the Steward chat panel in a column."""
        # Initialize chat history if not exists
        if 'steward_chat_history' not in st.session_state:
            st.session_state.steward_chat_history = [
                {
                    'role': 'assistant',
                    'content': 'Hello! I am Steward, your TELOS guide. I\'m here to help you understand the TELOS framework, navigate the Observatory interface, and answer any questions about what you\'re seeing on your screen. Feel free to ask me anything!',
                    'timestamp': datetime.now().isoformat()
                }
            ]

        # Close button only (compact, top-right)
        col_spacer, col_close = st.columns([5, 1])
        with col_close:
            if st.button("✕", key="close_steward", help="Close Steward", use_container_width=True):
                st.session_state.steward_panel_open = False
                st.rerun()

        # Chat history - direct display without heavy container
        for message in st.session_state.steward_chat_history:
            if message['role'] == 'assistant':
                st.markdown(f"""
                <div class="message-container" style="background-color: rgba(255, 215, 0, 0.1); border: 1px solid #FFD700; border-radius: 8px; padding: 12px; margin-bottom: 15px;">
                    <strong style="color: #FFD700; font-size: 18px;">Steward:</strong><br>
                    <span style="color: #FFD700; font-size: 16px; line-height: 1.6;">{message['content']}</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="message-container" style="background-color: rgba(255, 255, 255, 0.05); border: 1px solid #666; border-radius: 8px; padding: 12px; margin-bottom: 15px;">
                    <strong style="color: #e0e0e0; font-size: 18px;">You:</strong><br>
                    <span style="color: #e0e0e0; font-size: 16px; line-height: 1.6;">{message['content']}</span>
                </div>
                """, unsafe_allow_html=True)

        # Small spacing before input
        st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)

        with st.form(key="steward_form", clear_on_submit=True):
            user_input = st.text_input(
                "Ask Steward anything...",
                placeholder="Type your question here...",
                label_visibility="collapsed",
                key="steward_input_form"
            )

            submit = st.form_submit_button("Send", use_container_width=True)

            if submit and user_input:
                # Add user message
                st.session_state.steward_chat_history.append({
                    'role': 'user',
                    'content': user_input,
                    'timestamp': datetime.now().isoformat()
                })

                # TODO: Call LLM API for Steward response
                # For now, placeholder response
                steward_response = f"I understand you're asking about: '{user_input}'. As Steward, I'm here to help guide you through TELOS. This is a placeholder response - the full LLM integration will provide detailed answers about TELOS concepts, Observatory features, and governance principles."

                st.session_state.steward_chat_history.append({
                    'role': 'assistant',
                    'content': steward_response,
                    'timestamp': datetime.now().isoformat()
                })

                st.rerun()

    def hide_sidebar_when_open(self):
        """Apply CSS to hide sidebar when Steward panel is open."""
        if st.session_state.get('steward_panel_open', False):
            st.markdown("""
            <style>
            /* Hide sidebar when Steward is open */
            [data-testid="stSidebar"] {
                display: none !important;
            }
            </style>
            """, unsafe_allow_html=True)
