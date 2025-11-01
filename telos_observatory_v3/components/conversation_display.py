"""
Conversation Display Component for TELOS Observatory V3.
Renders ChatGPT/Claude-style conversation in center column.
"""

import streamlit as st
from typing import Dict, Any
import html


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

        # Render analysis windows if toggles are enabled (in order: PA, Math, Counterfactual)
        if self.state_manager.state.show_primacy_attractor:
            st.markdown("<div style='margin: 15px 0;'></div>", unsafe_allow_html=True)
            self._render_primacy_attractor_window(turn_data)

        if self.state_manager.state.show_math_breakdown:
            st.markdown("<div style='margin: 15px 0;'></div>", unsafe_allow_html=True)
            self._render_math_breakdown_window(turn_data)

        if self.state_manager.state.show_counterfactual:
            st.markdown("<div style='margin: 15px 0;'></div>", unsafe_allow_html=True)
            self._render_counterfactual_window(turn_data)

    def _render_main_chat(self):
        """Render main conversation - either turn-by-turn or scrollable history."""
        # Get current turn data
        current_turn_idx = self.state_manager.get_current_turn_index()
        all_turns = self.state_manager.get_all_turns()

        # Initialize intro message state (respecting settings)
        if 'show_intro' not in st.session_state:
            # Check if intro examples are enabled in settings
            enable_intro = st.session_state.get('enable_intro_examples', True)
            st.session_state.show_intro = enable_intro

        if len(all_turns) == 0:
            # Blank session - check Demo Mode FIRST
            demo_mode = st.session_state.get('telos_demo_mode', False)

            if demo_mode:
                # DEMO MODE: Show demo welcome message
                if 'demo_welcome_shown' not in st.session_state:
                    self._render_demo_welcome()
                # Show input area
                self._render_input_with_scroll_toggle()
                return
            else:
                # OPEN MODE: Show intro example if enabled
                if st.session_state.show_intro and st.session_state.get('enable_intro_examples', True):
                    self._render_intro_example()
                    self._render_input_with_scroll_toggle()
                    return
                else:
                    # Just show input area
                    self._render_input_with_scroll_toggle()
                    return

        # Render scrollable history window if enabled (at top of screen)
        if self.state_manager.state.scrollable_history_mode:
            self._render_scrollable_history_window(current_turn_idx, all_turns)

        # Render current turn in interactive mode (always show this)
        self._render_current_turn_only(current_turn_idx, all_turns)

        # Input area
        self._render_input_with_scroll_toggle()

    def _render_demo_welcome(self):
        """Render Demo Mode welcome message."""
        from demo_mode.telos_framework_demo import get_demo_welcome_message

        welcome_msg = get_demo_welcome_message()

        # Render in a gold-bordered welcome box
        st.markdown(f"""
<div style="background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
            border: 3px solid #FFD700;
            border-radius: 15px;
            padding: 25px;
            margin: 20px 0;
            box-shadow: 0 0 20px rgba(255, 215, 0, 0.3);">
    {welcome_msg}
</div>
""", unsafe_allow_html=True)

        # Mark as shown
        st.session_state.demo_welcome_shown = True

    def _render_intro_example(self):
        """Render a simple intro example that dismisses when user starts typing."""
        from telos_observatory_v3.utils.intro_messages import get_random_intro_pair

        # Get random intro pair (cached for session)
        if 'intro_pair' not in st.session_state:
            st.session_state.intro_pair = get_random_intro_pair()

        user_msg, steward_msg = st.session_state.intro_pair

        # USER MESSAGE - Match exact structure of _render_user_message
        # Column structure: 0.5 (Example badge) + 9.5 (content with 8.5:1.5 split)
        col_badge, col_content = st.columns([0.5, 9.5])

        with col_badge:
            # Example badge (replaces Turn badge)
            st.markdown("""
<div style="display: flex; align-items: flex-start; height: 100%;">
    <span style="background: linear-gradient(90deg, #FFD700 0%, #FFA500 100%); color: #000; padding: 4px 10px; border-radius: 5px; font-size: 20px; font-weight: bold; display: inline-block;">Example</span>
</div>
""", unsafe_allow_html=True)

        with col_content:
            col_msg, col_dismiss = st.columns([8.5, 1.5])

            with col_msg:
                # User message with exact same styling
                st.markdown(f"""
<div style="background-color: #1a1a1a; padding: 15px; border-radius: 10px; margin: 0; border: 2px solid #FFD700;">
    <div style="color: #888; font-size: 18px; margin-bottom: 5px;">
        <strong style="color: #FFD700;">User</strong>
    </div>
    <div style="color: #fff; font-size: 18px; white-space: pre-wrap;">
        {html.escape(user_msg)}
    </div>
</div>
""", unsafe_allow_html=True)

            with col_dismiss:
                # Dismiss button in same position as scroll button
                if st.button("✕", key="dismiss_intro", use_container_width=True, help="Dismiss example"):
                    st.session_state.show_intro = False
                    st.rerun()

        # STEWARD MESSAGE - Match exact structure of _render_assistant_message
        # Column structure: 0.5 (spacer) + 9.5 (content with 8.5:1.5 split)
        col_spacer, col_content2 = st.columns([0.5, 9.5])

        with col_spacer:
            st.markdown("")

        with col_content2:
            col_msg2, col_empty = st.columns([8.5, 1.5])

            with col_msg2:
                # Steward response with exact same styling and spacing
                st.markdown(f"""
<div style="background-color: #1a1a1a; padding: 15px; border-radius: 10px; margin-top: 15px; margin-bottom: 0; border: 2px solid #FFD700;">
    <div style="color: #888; font-size: 18px; margin-bottom: 5px;">
        <strong style="color: #FFD700;">Steward</strong>
    </div>
    <div style="color: #fff; font-size: 18px; white-space: pre-wrap;">
        {html.escape(steward_msg)}
    </div>
</div>
""", unsafe_allow_html=True)

            with col_empty:
                st.markdown("")

    def _render_current_turn_only(self, current_turn_idx: int, all_turns: list):
        """Render only the current turn with Turn label."""
        if current_turn_idx >= len(all_turns):
            return

        turn_data = all_turns[current_turn_idx]
        turn_number = current_turn_idx + 1

        # Render user and assistant messages for this turn with turn number and metrics
        self._render_user_message(turn_data.get('user_input', ''), turn_number, turn_data)
        self._render_assistant_message(
            turn_data.get('response', ''),
            turn_number,
            is_loading=turn_data.get('is_loading', False)
        )

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

            # Render messages with turn number and metrics
            self._render_user_message(turn_data.get('user_input', ''), turn_number, turn_data)
            self._render_assistant_message(
                turn_data.get('response', ''),
                turn_number,
                is_loading=turn_data.get('is_loading', False)
            )

            # Add divider between turns (except after last turn)
            if idx < current_turn_idx:
                st.markdown("""
                <div style="border-bottom: 1px solid #444; margin: 20px 0;"></div>
                """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    def _render_user_message(self, message: str, turn_number: int = None, turn_data: dict = None):
        """Render user message bubble with optional turn number badge and metrics."""
        import html

        # Build turn badge HTML if turn_number provided
        turn_badge = ""
        metrics_html = ""
        scroll_button = ""

        if turn_number is not None:
            turn_badge = f'<span style="background: linear-gradient(90deg, #FFD700 0%, #FFA500 100%); color: #000; padding: 4px 10px; border-radius: 5px; font-size: 20px; font-weight: bold; display: inline-block;">Turn {turn_number}</span>'

            # Add scroll toggle button
            scroll_label = "📜 History" if not self.state_manager.state.scrollable_history_mode else "✕ Close"
            # Note: We can't add interactive button in markdown, so we'll add it via Streamlit columns before this

            # Add metrics if turn_data provided
            if turn_data:
                fidelity = turn_data.get('fidelity', 0.0)
                fidelity_color = "#4CAF50" if fidelity >= 0.8 else "#FFA500" if fidelity >= 0.6 else "#FF5252"

                # Determine PA status from session metadata
                convergence_turn = 7  # Default fallback
                if hasattr(self.state_manager.state, 'metadata'):
                    convergence_turn = self.state_manager.state.metadata.get('convergence_turn', 7)

                pa_status = "Calibrating" if turn_number <= convergence_turn else "Established"
                pa_color = "#FFA500" if pa_status == "Calibrating" else "#4CAF50"

                # Add ΔF (Delta Fidelity) if available
                delta_f_html = ""
                if 'delta_f' in turn_data:
                    delta_f = turn_data.get('delta_f', 0.0)
                    delta_f_color = "#4CAF50" if delta_f > 0 else "#FF5252" if delta_f < 0 else "#888"
                    delta_f_sign = "+" if delta_f >= 0 else ""
                    delta_f_html = f'<span style="margin-left: 15px; display: inline-block;"><span style="color: #888; font-size: 14px;">ΔF:</span> <span style="color: {delta_f_color}; font-size: 16px; font-weight: bold; margin-left: 5px;">{delta_f_sign}{delta_f:.3f}</span></span>'

                metrics_html = f'<span style="margin-left: 15px; display: inline-block;"><span style="color: #888; font-size: 14px;">Fidelity:</span> <span style="color: {fidelity_color}; font-size: 16px; font-weight: bold; margin-left: 5px;">{fidelity:.3f}</span></span>{delta_f_html}<span style="margin-left: 15px; display: inline-block;"><span style="color: #888; font-size: 14px;">Primacy Attractor Status:</span> <span style="color: {pa_color}; font-size: 14px; font-weight: bold; margin-left: 5px;">{pa_status}</span></span>'

        # Escape the message content to prevent HTML injection
        safe_message = html.escape(message)

        # Create columns: Turn badge on left, message+scroll on right (matching Steward layout)
        # Turn badge gets 0.5, rest gets 9.5 (matching Steward's 8.5 + 1.5 with offset)
        col_turn, col_content = st.columns([0.5, 9.5])

        # Turn badge on the left
        if turn_number is not None:
            with col_turn:
                st.markdown(f"""
<div style="display: flex; align-items: flex-start; height: 100%;">
    {turn_badge}
</div>
""", unsafe_allow_html=True)

        # Message and scroll button in the right section (matching Steward's 8.5:1.5 layout)
        with col_content:
            col_msg, col_scroll = st.columns([8.5, 1.5])

            with col_msg:
                st.markdown(f"""
<div style="background-color: #1a1a1a; padding: 15px; border-radius: 10px; margin: 0; border: 2px solid #FFD700;">
    <div style="color: #888; font-size: 18px; margin-bottom: 5px;">
        <strong style="color: #FFD700;">User</strong>
    </div>
    {f'<div style="margin-top: 10px; margin-bottom: 10px; display: flex; align-items: center; flex-wrap: wrap;">{metrics_html}</div>' if metrics_html else ''}
    <div style="color: #fff; font-size: 18px; white-space: pre-wrap;">
        {safe_message}
    </div>
</div>
""", unsafe_allow_html=True)

            # Only render scroll button if we're showing the current turn (not in history mode)
            if turn_number is not None and not self.state_manager.state.scrollable_history_mode:
                with col_scroll:
                    scroll_label = "📜"
                    if st.button(scroll_label, key=f"scroll_toggle_current", use_container_width=True, help="Show scrollable history"):
                        self.state_manager.toggle_scrollable_history()
                        st.rerun()
            elif turn_number is not None and self.state_manager.state.scrollable_history_mode:
                with col_scroll:
                    scroll_label = "✕"
                    if st.button(scroll_label, key=f"scroll_close_current", use_container_width=True, help="Close scrollable history"):
                        self.state_manager.toggle_scrollable_history()
                        st.rerun()

    def _render_assistant_message(self, message: str, turn_number: int = None, is_loading: bool = False):
        """Render steward message bubble - aligned with User message."""
        import html

        # Match User message structure: 0.5 (Turn badge space) + 9.5 (content area with 8.5:1.5 split)
        col_spacer, col_content = st.columns([0.5, 9.5])

        with col_spacer:
            # Empty column to align with Turn badge space from User message
            st.markdown("")

        with col_content:
            col_msg, col_empty = st.columns([8.5, 1.5])

            with col_msg:
                if is_loading:
                    # Show contemplative pulsing animation (grey ↔ yellow)
                    st.markdown(f"""
<style>
@keyframes contemplative-pulse {{
    0%, 100% {{
        border-color: #888;
        box-shadow: 0 0 10px rgba(136, 136, 136, 0.3);
    }}
    50% {{
        border-color: #FFD700;
        box-shadow: 0 0 20px rgba(255, 215, 0, 0.4);
    }}
}}
.contemplating {{
    animation: contemplative-pulse 2s ease-in-out infinite;
}}
</style>
<div class="contemplating" style="background-color: #1a1a1a; padding: 15px; border-radius: 10px; margin-top: 15px; margin-bottom: 0; border: 2px solid #888;">
    <div style="color: #888; font-size: 18px; margin-bottom: 5px;">
        <strong style="color: #FFD700;">Steward</strong>
    </div>
    <div style="color: #888; font-size: 18px; font-style: italic; opacity: 0.7;">
        Contemplating...
    </div>
</div>
""", unsafe_allow_html=True)
                else:
                    # Escape the message content and show response
                    safe_message = html.escape(message)
                    st.markdown(f"""
<div style="background-color: #1a1a1a; padding: 15px; border-radius: 10px; margin-top: 15px; margin-bottom: 0; border: 2px solid #FFD700;">
    <div style="color: #888; font-size: 18px; margin-bottom: 5px;">
        <strong style="color: #FFD700;">Steward</strong>
    </div>
    <div style="color: #fff; font-size: 18px; white-space: pre-wrap;">
        {safe_message}
    </div>
</div>
""", unsafe_allow_html=True)

            with col_empty:
                # Empty column for alignment
                st.markdown("")

    def _render_math_breakdown_window(self, turn_data: Dict[str, Any]):
        """Render Math Breakdown analysis window with metrics and chat."""
        # Header with close button
        col1, col2 = st.columns([9.5, 0.5])
        with col1:
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
        with col2:
            if st.button("✕", key="close_math", use_container_width=True, help="Close Math Breakdown"):
                self.state_manager.toggle_component('math_breakdown')
                st.rerun()

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
        # Header with close button
        col1, col2 = st.columns([9.5, 0.5])
        with col1:
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
        with col2:
            if st.button("✕", key="close_counterfactual", use_container_width=True, help="Close Counterfactual Analysis"):
                self.state_manager.toggle_component('counterfactual')
                st.rerun()

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

    def _render_primacy_attractor_window(self, turn_data: Dict[str, Any]):
        """Render Primacy Attractor window showing Purpose, Scope, Boundaries."""
        # Header with close button
        col1, col2 = st.columns([9.5, 0.5])
        with col1:
            st.markdown("""
            <div style="
                background-color: #2d2d2d;
                border: 2px solid #FFD700;
                border-radius: 10px;
                padding: 20px;
                margin-bottom: 15px;
            ">
                <div style="color: #FFD700; font-size: 20px; font-weight: bold; text-align: center;">
                    🎯 Primacy Attractor
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            if st.button("✕", key="close_pa", use_container_width=True, help="Close Primacy Attractor"):
                self.state_manager.toggle_component('primacy_attractor')
                st.rerun()

        # Get session info which should include primacy attractor
        session_info = self.state_manager.get_session_info()

        # Try to get primacy attractor from turn data or session state
        primacy_attractor = None
        all_turns = self.state_manager.get_all_turns()
        if all_turns and len(all_turns) > 0:
            # Check if first turn has primacy attractor in session data
            if hasattr(self.state_manager.state, 'turns') and self.state_manager.state.turns:
                # Look for primacy_attractor in the session's initial state
                # For Phase 2 sessions, it should be in the session metadata
                pass

        # Check if we can get it from state manager's internal session data
        primacy_attractor = getattr(self.state_manager.state, 'primacy_attractor', None)

        # Fallback: try to construct from session state
        if not primacy_attractor and 'state_manager' in st.session_state:
            # Try to access the original session data that was loaded
            primacy_attractor = st.session_state.get('primacy_attractor', None)

        # Display Primacy Attractor components
        col1, col2, col3 = st.columns(3)

        with col1:
            # Build Purpose content
            purpose_items = ""
            if primacy_attractor and 'purpose' in primacy_attractor:
                for purpose_item in primacy_attractor['purpose']:
                    purpose_items += f"<p style='margin: 8px 0; color: #e0e0e0;'>• {purpose_item}</p>"
            else:
                purpose_items = "<p style='margin: 8px 0; color: #e0e0e0;'>• Establish conversation purpose from baseline turns</p>"

            st.markdown(f"""
            <div style="
                background-color: #1a1a1a;
                border: 2px solid #FFD700;
                border-radius: 10px;
                padding: 15px;
            ">
                <p style="color: #FFD700; font-weight: bold; font-size: 16px; margin-bottom: 15px;">📋 Purpose</p>
                <div style="color: #e0e0e0; font-size: 13px; line-height: 1.6;">
                    {purpose_items}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            # Build Scope content
            scope_items = ""
            if primacy_attractor and 'scope' in primacy_attractor:
                for scope_item in primacy_attractor['scope']:
                    scope_items += f"<p style='margin: 8px 0; color: #e0e0e0;'>• {scope_item}</p>"
            else:
                scope_items = "<p style='margin: 8px 0; color: #e0e0e0;'>• Topics covered in baseline</p>"

            st.markdown(f"""
            <div style="
                background-color: #1a1a1a;
                border: 2px solid #FFD700;
                border-radius: 10px;
                padding: 15px;
            ">
                <p style="color: #FFD700; font-weight: bold; font-size: 16px; margin-bottom: 15px;">🎯 Scope</p>
                <div style="color: #e0e0e0; font-size: 13px; line-height: 1.6;">
                    {scope_items}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            # Build Boundaries content
            boundary_items = ""
            if primacy_attractor and 'boundaries' in primacy_attractor:
                for boundary_item in primacy_attractor['boundaries']:
                    boundary_items += f"<p style='margin: 8px 0; color: #e0e0e0;'>• {boundary_item}</p>"
            else:
                boundary_items = "<p style='margin: 8px 0; color: #e0e0e0;'>• Off-topic discussions</p>"

            st.markdown(f"""
            <div style="
                background-color: #1a1a1a;
                border: 2px solid #FFD700;
                border-radius: 10px;
                padding: 15px;
            ">
                <p style="color: #FFD700; font-weight: bold; font-size: 16px; margin-bottom: 15px;">🚧 Boundaries</p>
                <div style="color: #e0e0e0; font-size: 13px; line-height: 1.6;">
                    {boundary_items}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Status indicator
        st.markdown("""
        <div style="
            background-color: #1a1a1a;
            border: 2px solid #4CAF50;
            border-radius: 10px;
            padding: 15px;
            margin-top: 15px;
            text-align: center;
        ">
            <span style="color: #4CAF50; font-weight: bold; font-size: 16px;">✓ Primacy Attractor Established</span>
        </div>
        """, unsafe_allow_html=True)

        # Chat interface with Steward handshake
        self._render_steward_chat("primacy_attractor")

    def _render_steward_chat(self, window_type: str):
        """Render chat interface with Steward (auto-authenticated).

        Args:
            window_type: Type of window ('math_breakdown', 'counterfactual', 'steward')
        """
        # Hardcoded Mistral API key - can be toggled from backend settings
        MISTRAL_API_KEY = "NxFBck0mkmGhM9vn0bvJzHf1scagv44f"

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
        """Render input area with send button - this defines the vertical alignment for all windows."""
        # Two-column layout: input box and send button
        # Send button width defines the right edge that everything else aligns to
        col1, col2 = st.columns([8.5, 1.5])

        # Use a form to enable Enter key submission
        with st.form(key="message_form", clear_on_submit=True):
            form_col1, form_col2 = st.columns([8.5, 1.5])

            with form_col1:
                user_input = st.text_input(
                    "Message",
                    placeholder="Type your message and press Enter...",
                    key="main_chat_input_v4",
                    label_visibility="collapsed"
                )

            with form_col2:
                # Send button - this defines the right edge for vertical alignment
                st.markdown("""
                <style>
                div[data-testid="column"]:nth-of-type(2) button[kind="formSubmit"] {
                    font-size: 18px !important;
                    font-weight: bold !important;
                }
                </style>
                """, unsafe_allow_html=True)
                send_button = st.form_submit_button("Send", use_container_width=True, help="Send message (or press Enter)")

        # Handle sending message
        if send_button and user_input and user_input.strip():
            # Dismiss intro if showing
            if 'show_intro' in st.session_state and st.session_state.show_intro:
                st.session_state.show_intro = False

            # CRITICAL: Clear demo data on first user message to prevent contamination
            # Demo data should NOT be part of primacy attractor calibration
            if 'user_started_conversation' not in st.session_state:
                # This is the user's first message - clear ALL demo data
                print(f"[DEBUG] Before clear: {len(self.state_manager.state.turns)} turns")
                self.state_manager.clear_demo_data()
                print(f"[DEBUG] After clear: {len(self.state_manager.state.turns)} turns")
                st.session_state.user_started_conversation = True

            # Add the message to the conversation (this will generate AI response)
            print(f"[DEBUG] Before add_user_message: {len(self.state_manager.state.turns)} turns")
            self.state_manager.add_user_message(user_input.strip())
            print(f"[DEBUG] After add_user_message: {len(self.state_manager.state.turns)} turns")
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
