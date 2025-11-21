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
        """Render the conversation display with main chat and analysis windows."""
        # Main chat interface
        self._render_main_chat()

        # Get current turn data for analysis windows
        turn_data = self.state_manager.get_current_turn_data()

        if not turn_data:
            return

        # Render analysis windows if toggles are enabled
        if self.state_manager.state.show_math_breakdown:
            st.markdown("<div style='margin: 15px 0;'></div>", unsafe_allow_html=True)
            self._render_math_breakdown_window(turn_data)

        if self.state_manager.state.show_counterfactual:
            st.markdown("<div style='margin: 15px 0;'></div>", unsafe_allow_html=True)
            self._render_counterfactual_window(turn_data)

        if self.state_manager.state.show_steward:
            st.markdown("<div style='margin: 15px 0;'></div>", unsafe_allow_html=True)
            self._render_steward_window(turn_data)

    def _render_main_chat(self):
        """Render main conversation - either turn-by-turn or scrollable history."""
        # Get current turn data
        current_turn_idx = self.state_manager.get_current_turn_index()
        all_turns = self.state_manager.get_all_turns()

        if len(all_turns) == 0:
            st.markdown("""
            <div style="color: #888; text-align: center; padding: 40px;">
                No turns to display. Load a session from the sidebar.
            </div>
            """, unsafe_allow_html=True)
            return

        # Render scrollable history window if enabled (at top of screen)
        if self.state_manager.state.scrollable_history_mode:
            self._render_scrollable_history_window(current_turn_idx, all_turns)

        # Render current turn in interactive mode (always show this)
        self._render_current_turn_only(current_turn_idx, all_turns)

        # Input area with scroll toggle next to send button
        self._render_input_with_scroll_toggle()

    def _render_current_turn_only(self, current_turn_idx: int, all_turns: list):
        """Render only the current turn with Turn label."""
        if current_turn_idx >= len(all_turns):
            return

        turn_data = all_turns[current_turn_idx]
        turn_number = current_turn_idx + 1

        # Render user and assistant messages for this turn with turn number
        self._render_user_message(turn_data.get('user_input', ''), turn_number)
        self._render_assistant_message(turn_data.get('response', ''), turn_number)

    def _render_scrollable_history_window(self, current_turn_idx: int, all_turns: list):
        """Render scrollable read-only history window at top of screen."""
        # Header for the scrollable window
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #2d2d2d 0%, #1a1a1a 100%);
            border: 2px solid #FFD700;
            border-radius: 8px;
            padding: 10px;
            margin-bottom: 15px;
        ">
            <div style="color: #FFD700; font-size: 18px; font-weight: bold; text-align: center;">
                📜 Conversation History (Read-Only)
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Scrollable container
        st.markdown("""
        <div style="
            max-height: 500px;
            overflow-y: auto;
            padding: 15px;
            background-color: #1a1a1a;
            border: 2px solid #FFD700;
            border-radius: 8px;
            margin-bottom: 20px;
        ">
        """, unsafe_allow_html=True)

        # Render all turns up to and including current
        for idx in range(current_turn_idx + 1):
            turn_data = all_turns[idx]
            turn_number = idx + 1

            # Render messages with turn number
            self._render_user_message(turn_data.get('user_input', ''), turn_number)
            self._render_assistant_message(turn_data.get('response', ''), turn_number)

            # Add divider between turns (except after last turn)
            if idx < current_turn_idx:
                st.markdown("""
                <div style="border-bottom: 1px solid #444; margin: 20px 0;"></div>
                """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    def _render_user_message(self, message: str, turn_number: int = None):
        """Render user message bubble with optional turn number badge."""
        # Build turn badge HTML if turn_number provided
        turn_badge = ""
        if turn_number is not None:
            turn_badge = f'<span style="background: linear-gradient(90deg, #FFD700 0%, #FFA500 100%); color: #000; padding: 4px 10px; border-radius: 5px; font-size: 20px; font-weight: bold; margin-right: 8px; display: inline-block;">Turn {turn_number}</span>'

        st.markdown(f"""
        <div style="background-color: #2d2d2d; padding: 15px; border-radius: 10px; margin: 10px 0; max-width: 80%;">
            <div style="color: #888; font-size: 12px; margin-bottom: 5px;">
                {turn_badge}<strong style="color: #888;">User</strong>
            </div>
            <div style="color: #fff; font-size: 18px;">
                {message}
            </div>
        </div>
        """, unsafe_allow_html=True)

    def _render_assistant_message(self, message: str, turn_number: int = None):
        """Render assistant message bubble with optional turn number badge."""
        # Turn badge already shown with user message, no need to repeat
        st.markdown(f"""
        <div style="background-color: #1a1a1a; padding: 15px; border-radius: 10px; margin: 10px 0; max-width: 80%;">
            <div style="color: #888; font-size: 12px; margin-bottom: 5px;">
                <strong>Assistant</strong>
            </div>
            <div style="color: #fff; font-size: 18px;">
                {message}
            </div>
        </div>
        """, unsafe_allow_html=True)

    def _render_math_breakdown_window(self, turn_data: Dict[str, Any]):
        """Render Math Breakdown analysis window with metrics and chat."""
        st.markdown("""
        <div style="
            background-color: #2d2d2d;
            border: 2px solid #FFD700;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 15px;
        ">
            <div style="color: #FFD700; font-size: 20px; font-weight: bold; text-align: center;">
                🔢 Math Breakdown
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Two-column layout for calculations
        col1, col2 = st.columns(2)

        with col1:
            # Fidelity Calculation window
            st.markdown("""
            <div style="
                background-color: #1a1a1a;
                border: 2px solid #FFD700;
                border-radius: 10px;
                padding: 15px;
                min-height: 300px;
            ">
                <p style="color: #FFD700; font-weight: bold; font-size: 16px; margin-bottom: 15px;">📊 Fidelity Calculation</p>
                <ul style="color: #e0e0e0; line-height: 1.8;">
                    <li>Base alignment score: <span style="color: #FFD700;">0.85</span></li>
                    <li>Context adjustment: <span style="color: #4CAF50;">+0.05</span></li>
                    <li>Preference weight: <span style="color: #FFD700;">0.92</span></li>
                    <li style="margin-top: 15px; padding-top: 15px; border-top: 1px solid #FFD700;">
                        <strong style="color: #FFD700;">Final Fidelity: 0.873</strong>
                    </li>
                </ul>
                <div style="margin-top: 20px; padding: 10px; background-color: #2d2d2d; border-radius: 5px;">
                    <p style="color: #888; font-size: 11px; margin: 0;">
                        Alignment Fidelity measures how well the response matches the user's deeper preferences and values.
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            # Distance Metrics window
            st.markdown("""
            <div style="
                background-color: #1a1a1a;
                border: 2px solid #FFD700;
                border-radius: 10px;
                padding: 15px;
                min-height: 300px;
            ">
                <p style="color: #FFD700; font-weight: bold; font-size: 16px; margin-bottom: 15px;">📏 Distance Metrics</p>
                <ul style="color: #e0e0e0; line-height: 1.8;">
                    <li>Semantic distance: <span style="color: #FFD700;">0.127</span></li>
                    <li>Intent deviation: <span style="color: #FFD700;">0.08</span></li>
                    <li>Preference alignment gap: <span style="color: #4CAF50;">0.05</span></li>
                    <li style="margin-top: 15px; padding-top: 15px; border-top: 1px solid #FFD700;">
                        <strong style="color: #4CAF50;">Status: Nominal</strong>
                    </li>
                </ul>
                <div style="margin-top: 20px; padding: 10px; background-color: #2d2d2d; border-radius: 5px;">
                    <p style="color: #888; font-size: 11px; margin: 0;">
                        Distance metrics quantify how far the response is from the ideal aligned output.
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Chat interface with Steward handshake
        self._render_steward_chat("math_breakdown")

    def _render_counterfactual_window(self, turn_data: Dict[str, Any]):
        """Render Counterfactual Analysis window with comparison and chat."""
        st.markdown("""
        <div style="
            background-color: #2d2d2d;
            border: 2px solid #FFD700;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 15px;
        ">
            <div style="color: #FFD700; font-size: 20px; font-weight: bold; text-align: center;">
                🔀 Counterfactual Analysis
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Two-column layout for comparison
        col1, col2 = st.columns(2)

        with col1:
            # Native LLM Response window
            st.markdown("""
            <div style="
                background-color: #1a1a1a;
                border: 2px solid #888;
                border-radius: 10px;
                padding: 15px;
                min-height: 350px;
            ">
                <p style="color: #888; font-weight: bold; font-size: 16px; margin-bottom: 15px;">🤖 Native LLM Response</p>
                <div style="background-color: #2d2d2d; padding: 12px; border-radius: 5px; margin-bottom: 15px;">
                    <p style="color: #e0e0e0; font-size: 13px; line-height: 1.6; margin: 0;">
                        "Based on your question, here's a literal interpretation of what you asked.
                        The response focuses on surface-level understanding without considering your
                        underlying preferences or values."
                    </p>
                </div>
                <div style="margin-top: 15px;">
                    <p style="color: #888; font-size: 12px; font-weight: bold; margin-bottom: 10px;">Metrics:</p>
                    <ul style="color: #e0e0e0; font-size: 12px; line-height: 1.8;">
                        <li>Alignment score: <span style="color: #FFA500;">0.65</span></li>
                        <li>Semantic distance: <span style="color: #FFA500;">0.35</span></li>
                        <li>Intent match: <span style="color: #FFA500;">72%</span></li>
                    </ul>
                </div>
                <div style="margin-top: 15px; padding: 10px; background-color: #2d2d2d; border-radius: 5px; border-left: 3px solid #888;">
                    <p style="color: #888; font-size: 11px; margin: 0;">
                        Standard response without TELOS intervention
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            # TELOS Intervention window
            st.markdown("""
            <div style="
                background-color: #1a1a1a;
                border: 2px solid #FFD700;
                border-radius: 10px;
                padding: 15px;
                min-height: 350px;
            ">
                <p style="color: #FFD700; font-weight: bold; font-size: 16px; margin-bottom: 15px;">🔭 TELOS Intervention</p>
                <div style="background-color: #2d2d2d; padding: 12px; border-radius: 5px; margin-bottom: 15px;">
                    <p style="color: #e0e0e0; font-size: 13px; line-height: 1.6; margin: 0;">
                        "Understanding your deeper preferences, here's a response that aligns with your
                        values and goals. The answer considers both your explicit question and implicit
                        intent based on your interaction history."
                    </p>
                </div>
                <div style="margin-top: 15px;">
                    <p style="color: #FFD700; font-size: 12px; font-weight: bold; margin-bottom: 10px;">Metrics:</p>
                    <ul style="color: #e0e0e0; font-size: 12px; line-height: 1.8;">
                        <li>Alignment score: <span style="color: #4CAF50;">0.873</span></li>
                        <li>Semantic distance: <span style="color: #4CAF50;">0.127</span></li>
                        <li>Intent match: <span style="color: #4CAF50;">95%</span></li>
                    </ul>
                </div>
                <div style="margin-top: 15px; padding: 10px; background-color: #2d2d2d; border-radius: 5px; border-left: 3px solid #FFD700;">
                    <p style="color: #4CAF50; font-size: 11px; margin: 0;">
                        <strong>Improvement: +34% alignment</strong>
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Chat interface with Steward handshake
        self._render_steward_chat("counterfactual")

    def _render_steward_window(self, turn_data: Dict[str, Any]):
        """Render Steward Details window."""
        st.markdown("""
        <div style="
            background-color: #2d2d2d;
            border: 2px solid #FFD700;
            border-radius: 10px;
            padding: 20px;
        ">
            <div style="color: #FFD700; font-size: 20px; font-weight: bold; margin-bottom: 15px;">
                👤 Steward Details
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.info("Steward information and consultation interface")

        # Chat interface with Steward handshake
        self._render_steward_chat("steward")

    def _render_steward_chat(self, window_type: str):
        """Render chat interface with Steward (auto-authenticated).

        Args:
            window_type: Type of window ('math_breakdown', 'counterfactual', 'steward')
        """
        # Get Mistral API key from Streamlit secrets or environment
        import os
        MISTRAL_API_KEY = st.secrets.get("MISTRAL_API_KEY", os.getenv("MISTRAL_API_KEY"))

        if not MISTRAL_API_KEY:
            st.error("⚠️ Mistral API key not configured. Please add to Streamlit secrets or .env")
            return

        st.markdown("---")

        # Chat interface (always active with hardcoded key)
        col1, col2 = st.columns([5, 1])
        with col1:
            user_input = st.text_input(
                "Message",
                key=f"chat_input_{window_type}",
                placeholder="Ask Steward about this analysis...",
                label_visibility="collapsed"
            )
        with col2:
            send_button = st.button("Send", key=f"send_{window_type}", use_container_width=True)

        # Handle sending message
        if send_button and user_input:
            # Initialize chat history for this window
            if f'chat_history_{window_type}' not in st.session_state:
                st.session_state[f'chat_history_{window_type}'] = []

            # Add user message
            st.session_state[f'chat_history_{window_type}'].append({
                'role': 'user',
                'content': user_input
            })

            # Placeholder response (would call Mistral API here with MISTRAL_API_KEY)
            ai_response = f"Steward response to: '{user_input}' (using Mistral API)"
            st.session_state[f'chat_history_{window_type}'].append({
                'role': 'assistant',
                'content': ai_response
            })

            st.rerun()

        # Display chat history
        chat_history = st.session_state.get(f'chat_history_{window_type}', [])
        if chat_history:
            st.markdown("""
            <div style="
                background-color: #1a1a1a;
                border: 1px solid #FFD700;
                border-radius: 5px;
                padding: 10px;
                margin: 10px 0;
                max-height: 200px;
                overflow-y: auto;
            ">
            """, unsafe_allow_html=True)
            for msg in chat_history:
                role_color = "#4CAF50" if msg['role'] == 'user' else "#FFD700"
                role_label = "You" if msg['role'] == 'user' else "Steward"
                st.markdown(f"""
                <div style="margin: 5px 0;">
                    <span style="color: {role_color}; font-weight: bold;">{role_label}:</span>
                    <span style="color: #e0e0e0;"> {msg['content']}</span>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    def _render_metric_card(self, title: str, value: str, icon: str, description: str, value_color: str = "#FFD700"):
        """Render a single metric card.

        Args:
            title: Metric title
            value: Metric value
            icon: Icon emoji
            description: Description text
            value_color: Color for the value
        """
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
            border: 2px solid #FFD700;
            border-radius: 8px;
            padding: 10px;
            text-align: center;
        ">
            <div style="font-size: 20px; margin-bottom: 5px;">{icon}</div>
            <div style="color: #FFD700; font-size: 10px; font-weight: bold; margin-bottom: 5px;">
                {title}
            </div>
            <div style="
                font-size: 18px;
                font-weight: bold;
                color: {value_color};
                margin: 5px 0;
            ">
                {value}
            </div>
            <div style="color: #888; font-size: 8px;">
                {description}
            </div>
        </div>
        """, unsafe_allow_html=True)

    def _render_conversation_history(self):
        """Render all messages up to the current turn."""
        current_turn_idx = self.state_manager.get_current_turn_index()
        all_turns = self.state_manager.get_all_turns()

        # Render all turns up to and including the current one
        for i in range(current_turn_idx + 1):
            if i < len(all_turns):
                turn = all_turns[i]

                # User message
                self._render_user_message(turn.get('user_input', ''))

                # Assistant response
                self._render_assistant_message(turn.get('response', ''))

                # Intervention indicator if applied
                if turn.get('intervention_applied', False):
                    self._render_intervention_indicator()

    def _render_input_with_scroll_toggle(self):
        """Render input area with scroll history toggle next to send button."""
        # Two-column layout: input box and buttons
        col1, col2, col3 = st.columns([8, 1, 1])

        with col1:
            user_input = st.text_input(
                "Message",
                key="main_chat_input",
                placeholder="Type your message...",
                label_visibility="collapsed"
            )

        with col2:
            # Send button
            send_clicked = st.button("Send", key="send_main_message", use_container_width=True)

        with col3:
            # Toggle scroll mode button
            scroll_label = "📜" if not self.state_manager.state.scrollable_history_mode else "✕"
            scroll_tooltip = "Show scrollable history" if not self.state_manager.state.scrollable_history_mode else "Hide scrollable history"
            if st.button(scroll_label, key="toggle_scroll", use_container_width=True, help=scroll_tooltip):
                self.state_manager.toggle_scrollable_history()
                st.rerun()

        # Handle sending message
        if send_clicked and user_input:
            # Add the message to the conversation
            self.state_manager.add_user_message(user_input)
            st.rerun()

    def _render_chat_input(self):
        """Render chat input box for new messages."""
        # Chat input with custom styling
        user_input = st.chat_input(
            "Type your message...",
            key="chat_input_box"
        )

        if user_input:
            # Add the message to the conversation
            self.state_manager.add_user_message(user_input)
            st.rerun()

    def _render_expanded_analysis(self, turn_data: Dict[str, Any]):
        """Render expanded analysis sections based on toggle states.

        Args:
            turn_data: Current turn data dictionary
        """
        # Check if any toggles are enabled
        if self.state_manager.state.show_math_breakdown:
            st.markdown("---")
            self._render_chat_window("Math Breakdown", "🔢")

        if self.state_manager.state.show_counterfactual:
            st.markdown("---")
            self._render_chat_window("Counterfactual Analysis", "🔀")

        if self.state_manager.state.show_steward:
            st.markdown("---")
            st.markdown("### 👤 Steward Details")
            st.info("Steward information would appear here")

    def _render_chat_window(self, title: str, icon: str):
        """Render a chat-style window for analysis.

        Args:
            title: Window title
            icon: Icon emoji
        """
        # Header with steward icon
        col1, col2 = st.columns([10, 1])
        with col1:
            st.markdown(f"### {icon} {title}")
        with col2:
            if st.button("👤", key=f"steward_{title}", help="Consult Steward"):
                st.info("Steward consultation feature coming soon")

        # Chat window container with content
        with st.container():
            st.markdown("""
            <div style="
                background-color: #2d2d2d;
                border: 2px solid #FFD700;
                border-radius: 10px;
                padding: 15px;
                min-height: 200px;
                max-height: 400px;
                overflow-y: auto;
                margin-bottom: 10px;
            ">
            """, unsafe_allow_html=True)

            # Sample analysis content
            if title == "Math Breakdown":
                st.markdown("""
                <div style="color: #e0e0e0; padding: 10px;">
                    <p style="color: #FFD700; font-weight: bold;">Fidelity Calculation:</p>
                    <ul style="color: #e0e0e0;">
                        <li>Base alignment score: 0.85</li>
                        <li>Context adjustment: +0.05</li>
                        <li>Preference weight: 0.92</li>
                        <li style="color: #FFD700;"><strong>Final Fidelity: 0.873</strong></li>
                    </ul>
                    <p style="color: #FFD700; font-weight: bold; margin-top: 15px;">Distance Metrics:</p>
                    <ul style="color: #e0e0e0;">
                        <li>Semantic distance: 0.127</li>
                        <li>Intent deviation: 0.08</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            elif title == "Counterfactual Analysis":
                st.markdown("""
                <div style="color: #e0e0e0; padding: 10px;">
                    <p style="color: #FFD700; font-weight: bold;">Without TELOS Intervention:</p>
                    <ul style="color: #e0e0e0;">
                        <li>Expected response would focus on literal interpretation</li>
                        <li>Alignment score: 0.65 (lower)</li>
                    </ul>
                    <p style="color: #FFD700; font-weight: bold; margin-top: 15px;">With TELOS Intervention:</p>
                    <ul style="color: #e0e0e0;">
                        <li>Response adapted to user's deeper preferences</li>
                        <li>Alignment score: 0.873 (higher)</li>
                        <li style="color: #4CAF50;"><strong>Improvement: +34%</strong></li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

            # Chat input integrated into the window
            analysis_input = st.chat_input(
                f"Ask about {title.lower()}...",
                key=f"chat_{title.replace(' ', '_')}"
            )

            if analysis_input:
                st.info(f"Analysis chat feature coming soon. You asked: {analysis_input}")
