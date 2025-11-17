"""
Conversation Display Component for TELOS Observatory V3.
Renders ChatGPT/Claude-style conversation in center column.
"""

import streamlit as st
from typing import Dict, Any
import html
import logging
import re
from datetime import datetime

logger = logging.getLogger(__name__)


class ConversationDisplay:
    """ChatGPT-style conversation display using native Streamlit."""

    def __init__(self, state_manager, steward_panel=None):
        """Initialize with state manager reference.

        Args:
            state_manager: StateManager instance for accessing turn data
            steward_panel: Optional StewardPanel instance for demo chat
        """
        self.state_manager = state_manager
        self.steward_panel = steward_panel

    def _markdown_to_html(self, text: str) -> str:
        """Convert markdown to HTML while escaping dangerous content.

        Args:
            text: Markdown text to convert

        Returns:
            HTML-formatted text
        """
        # First escape HTML to prevent XSS
        text = html.escape(text)

        # Convert markdown patterns to HTML
        # Bold: **text** or __text__
        text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
        text = re.sub(r'__(.+?)__', r'<strong>\1</strong>', text)

        # Italic: *text* or _text_
        text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
        text = re.sub(r'_(.+?)_', r'<em>\1</em>', text)

        # Line breaks
        text = text.replace('\n', '<br>')

        return text

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
        # ESC key handler for Demo Mode -> Open Mode switch
        demo_mode = st.session_state.get('telos_demo_mode', False)
        if demo_mode:
            # Inject JavaScript to handle ESC key press
            st.components.v1.html("""
            <script>
            // Prevent duplicate listeners
            if (!window.telosEscListenerAdded) {
                window.telosEscListenerAdded = true;

                document.addEventListener('keydown', function(event) {
                    if (event.key === 'Escape') {
                        // Find the "Open Mode" radio button in the sidebar and click it
                        const radioButtons = window.parent.document.querySelectorAll('input[type="radio"]');
                        radioButtons.forEach(radio => {
                            const label = radio.parentElement;
                            if (label && label.textContent.includes('Open Mode')) {
                                radio.click();
                            }
                        });
                    }
                });
            }
            </script>
            """, height=0)

        # Get current turn data
        current_turn_idx = self.state_manager.get_current_turn_index()
        all_turns = self.state_manager.get_all_turns()

        # Check if currently loading (for conditional rendering)
        is_loading = False
        if len(all_turns) > 0:
            current_turn = all_turns[-1]
            # Check for BOTH is_loading AND is_streaming flags
            is_loading = current_turn.get('is_loading', False) or current_turn.get('is_streaming', False)

        # FORCE HIDE input form during loading - inject CSS to completely hide it
        if is_loading:
            st.markdown("""
            <style>
            /* Nuclear option: hide ALL forms on the page during loading */
            form {
                display: none !important;
                visibility: hidden !important;
                height: 0 !important;
                overflow: hidden !important;
                opacity: 0 !important;
                position: absolute !important;
                left: -9999px !important;
            }

            /* Hide form containers */
            div[data-testid="stForm"],
            form[data-testid="stForm"] {
                display: none !important;
                visibility: hidden !important;
                height: 0 !important;
            }

            /* Hide text inputs */
            input[type="text"] {
                display: none !important;
                visibility: hidden !important;
            }
            </style>
            """, unsafe_allow_html=True)

        # Initialize intro message state (respecting settings)
        if 'show_intro' not in st.session_state:
            # Check if intro examples are enabled in settings
            enable_intro = st.session_state.get('enable_intro_examples', True)
            st.session_state.show_intro = enable_intro

        # Check beta intro BEFORE checking turn count (to prevent showing after first message)
        demo_mode = st.session_state.get('telos_demo_mode', False)
        # Safely check for BETA mode - use .get() to avoid KeyError
        active_tab = st.session_state.get('active_tab', 'DEMO')
        beta_mode = active_tab == "BETA"

        # IMPORTANT: Only show beta intro if ALL THREE conditions are true:
        # 1. Beta mode is active
        # 2. Beta intro has NOT been completed
        # 3. There are NO turns yet (blank session)
        beta_intro_complete = st.session_state.get('beta_intro_complete', False)

        if beta_mode and not beta_intro_complete and len(all_turns) == 0:
            self._render_beta_intro()
            return

        if len(all_turns) == 0:
            # Blank session - handle different modes appropriately

            # BETA MODE with completed intro: Show input immediately
            if beta_mode and beta_intro_complete:
                # Beta mode after intro - show input for conversation
                self._render_input_with_scroll_toggle()
                return
            elif demo_mode:
                # DEMO MODE: Run the slideshow - don't show input during demo
                demo_slide_index = st.session_state.get('demo_slide_index', 0)
                if demo_slide_index <= 13:  # Slides 0-13: 0=welcome, 1=intro, 2=PA setup, 3-12=Q&A, 13=completion
                    self._render_demo_welcome()
                    return  # Completion slide has its own chat interface
                else:
                    # Beyond demo slides - fallback to regular input
                    self._render_input_with_scroll_toggle()
                    return
            else:
                # OPEN MODE (TELOS tab): Just show input area
                self._render_input_with_scroll_toggle()
                return

        # Render scrollable history window if enabled (at top of screen)
        # NOT available in Demo Mode - Demo Mode is conversation-focused only
        demo_mode = st.session_state.get('telos_demo_mode', False)
        if self.state_manager.state.scrollable_history_mode and not demo_mode:
            self._render_scrollable_history_window(current_turn_idx, all_turns)

        # Render current turn in interactive mode (always show this)
        self._render_current_turn_only(current_turn_idx, all_turns)

        # Input area - ONLY render when NOT loading (during contemplating, completely omit)
        if not is_loading:
            self._render_input_with_scroll_toggle()

    def _render_demo_welcome(self):
        """Render Demo Mode welcome and typewriter slideshow."""
        from demo_mode.telos_framework_demo import (
            get_demo_welcome_message,
            get_steward_intro_message,
            get_demo_slides,
            get_demo_completion_message
        )
        import json

        # Initialize demo state
        if 'demo_slide_index' not in st.session_state:
            st.session_state.demo_slide_index = 0  # 0 = welcome, 1 = steward intro, 2-11 = slides, 12 = complete

        # Inject global CSS for compact containers (applies to all demo slides)
        st.markdown("""
        <style>
        /* v5 - Compact centered windows - ULTRA SPECIFIC */
        div.compact-container,
        .compact-container {
            max-width: 700px !important;
            margin-left: auto !important;
            margin-right: auto !important;
            display: block !important;
        }

        .compact-container > div {
            max-width: 700px !important;
        }
        </style>
        """, unsafe_allow_html=True)

        # Add keyboard navigation for demo slides using streamlit-specific approach
        current_idx = st.session_state.demo_slide_index
        max_idx = 13  # 0=welcome, 1=intro, 2=PA setup, 3-12=Q&A (10 slides), 13=completion

        # Use component HTML for reliable keyboard event handling
        import streamlit.components.v1 as components

        components.html(f"""
        <script>
        // Scroll to top when slide loads
        (function() {{
            window.scrollTo(0, 0);
            if (window.parent) {{
                window.parent.scrollTo(0, 0);
            }}
        }})();

        // Demo keyboard navigation with debug logging
        (function() {{
            console.log('Demo keyboard navigation initializing...');
            const currentSlide = {current_idx};
            const maxSlide = {max_idx};

            console.log('Current slide:', currentSlide, 'Max slide:', maxSlide);

            // Remove old listener if exists
            if (window.demoKeyListener) {{
                document.removeEventListener('keydown', window.demoKeyListener);
                console.log('Removed old listener');
            }}

            // Create new listener
            window.demoKeyListener = function(event) {{
                console.log('Key pressed:', event.key);

                // Only handle arrow keys, not modified
                if (event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) {{
                    console.log('Modifier key detected, ignoring');
                    return;
                }}

                if (event.key === 'ArrowLeft' && currentSlide > 0) {{
                    console.log('Left arrow - looking for Previous button');
                    event.preventDefault();

                    // Find button in parent window/iframe
                    const parent = window.parent;
                    if (parent && parent.document) {{
                        const buttons = parent.document.querySelectorAll('button');
                        console.log('Found', buttons.length, 'buttons in parent');

                        for (let btn of buttons) {{
                            console.log('Button text:', btn.textContent);
                            if (btn.textContent.includes('Previous') || btn.textContent.includes('⬅')) {{
                                console.log('Clicking Previous button');
                                btn.click();
                                break;
                            }}
                        }}
                    }}
                }} else if (event.key === 'ArrowRight' && currentSlide < maxSlide) {{
                    console.log('Right arrow - looking for Next/Start/Complete button');
                    event.preventDefault();

                    // Find button in parent window/iframe
                    const parent = window.parent;
                    if (parent && parent.document) {{
                        const buttons = parent.document.querySelectorAll('button');
                        console.log('Found', buttons.length, 'buttons in parent');

                        for (let btn of buttons) {{
                            console.log('Button text:', btn.textContent);
                            if (btn.textContent.includes('Next') ||
                                btn.textContent.includes('Start Demo') ||
                                btn.textContent.includes('Complete Demo') ||
                                btn.textContent.includes('➡')) {{
                                console.log('Clicking Next/Start/Complete button');
                                btn.click();
                                break;
                            }}
                        }}
                    }}
                }}
            }};

            // Attach listener to both current window and parent
            document.addEventListener('keydown', window.demoKeyListener);
            if (window.parent && window.parent.document) {{
                window.parent.document.addEventListener('keydown', window.demoKeyListener);
                console.log('Attached listener to parent document');
            }}

            console.log('Keyboard listener attached');
        }})();
        </script>
        """, height=0)

        # Get slides
        slides = get_demo_slides()
        current_idx = st.session_state.demo_slide_index

        # Slide 0: Welcome screen - centered and compact
        if current_idx == 0:

            st.markdown(f"""
<div class="compact-container">
    <div style="background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
                border: 2px solid #FFD700;
                border-radius: 15px;
                padding: 30px;
                margin: 20px auto;
                text-align: center;
                box-shadow: 0 0 8px rgba(255, 215, 0, 0.4);">
        <h1 style="color: #FFD700; font-size: 32px; margin-bottom: 20px;">Welcome to TELOS Demo Mode! 🔭</h1>
        <p style="color: #e0e0e0; font-size: 20px; line-height: 1.6;">
            Click "Start Demo" below to begin your guided tour.
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

            # Center the button too
            col_left, col_center, col_right = st.columns([1, 1, 1])
            with col_center:
                if st.button("▶️ Start Demo", key="start_demo_btn", use_container_width=True):
                    st.session_state.demo_slide_index = 1
                    st.rerun()
            return

        # Slide 1: Original Steward intro - appears instantly, centered and compact
        if current_idx == 1:
            intro_html = """
<div class="compact-container">
    <div style="background: linear-gradient(135deg, rgba(255, 215, 0, 0.05) 0%, rgba(255, 215, 0, 0.1) 100%);
                border: 2px solid #FFD700;
                border-radius: 10px;
                padding: 20px 25px;
                margin: 15px auto;
                font-size: 19px;
                line-height: 1.7;
                color: #e0e0e0;
                text-align: center;
                box-shadow: 0 2px 8px rgba(255, 215, 0, 0.2);">
        <div style="color: #FFD700; font-size: 23px; font-weight: bold; margin-bottom: 15px;">
            Hello! I'm Steward, your guide to understanding TELOS.
        </div>
        <div style="margin-bottom: 20px;">
            What you'll experience in these slides is TELOS in action.
        </div>
        <div style="margin-bottom: 20px;">
            I'll show you how TELOS keeps AI conversations aligned with your goals through real-time governance.
        </div>
        <div style="margin-bottom: 20px;">
            Let's begin by understanding how TELOS establishes your session's governance.
        </div>
    </div>
</div>
"""
            st.markdown(intro_html, unsafe_allow_html=True)

            # Navigation - Previous and Next buttons (centered to match content)
            col_spacer_left, col_buttons, col_spacer_right = st.columns([1, 2, 1])

            with col_buttons:
                col_prev, col_next = st.columns(2)

                with col_prev:
                    if st.button(f"⬅️ Previous", key=f"prev_intro_btn", use_container_width=True):
                        st.session_state.demo_slide_index = 0
                        st.rerun()

                with col_next:
                    if st.button("➡️ Next", key="continue_demo_btn", use_container_width=True):
                        st.session_state.demo_slide_index = 2
                        st.rerun()
            return

        # Slide 2: NEW - How TELOS establishes PA
        if current_idx == 2:
            pa_setup_html = """
<div class="compact-container">
    <div style="background: linear-gradient(135deg, rgba(255, 215, 0, 0.05) 0%, rgba(255, 215, 0, 0.1) 100%);
                border: 2px solid #FFD700;
                border-radius: 10px;
                padding: 20px 25px;
                margin: 15px auto;
                font-size: 19px;
                line-height: 1.7;
                color: #e0e0e0;
                text-align: center;
                box-shadow: 0 2px 8px rgba(255, 215, 0, 0.2);">
        <div style="color: #FFD700; font-size: 23px; font-weight: bold; margin-bottom: 15px;">
            How TELOS Establishes Your Governance
        </div>
        <div style="margin-bottom: 20px;">
            TELOS tracks each conversation turn to understand what type of governance you want for your session. We call this your <strong style="color: #FFD700;">Primacy Attractor</strong> - the purpose, scope, and boundaries that keep the conversation aligned.
        </div>
        <div style="margin-bottom: 20px;">
            Normally, this takes <strong style="color: #FFD700;">5-10 turns</strong> of back-and-forth conversation to fully establish. TELOS observes your questions and interests to calibrate your attractor naturally.
        </div>
        <div style="margin-bottom: 20px;">
            For this demo, we'll establish your Primacy Attractor immediately with your first input - watch as it happens on the next slide.
        </div>
    </div>
</div>
"""
            st.markdown(pa_setup_html, unsafe_allow_html=True)

            # Navigation
            col_spacer_left, col_buttons, col_spacer_right = st.columns([1, 2, 1])

            with col_buttons:
                col_prev, col_next = st.columns(2)

                with col_prev:
                    if st.button(f"⬅️ Previous", key=f"prev_pa_setup_btn", use_container_width=True):
                        st.session_state.demo_slide_index = 1
                        st.rerun()

                with col_next:
                    if st.button("➡️ Next", key="continue_to_qa_btn", use_container_width=True):
                        st.session_state.demo_slide_index = 3
                        st.rerun()
            return

        # Slides 3-15: Q&A pairs (slides[0] through slides[12]) - 13 Q&A slides
        if 3 <= current_idx <= 15:
            slide_idx = current_idx - 3  # slides[0] through slides[12]
            user_question, steward_response = slides[slide_idx]

            # Turn numbers start at 11 (PA already established in turns 1-10)
            turn_num = slide_idx + 11  # Turn 11-19

            # Render demo slide
            self._render_demo_slide_with_typewriter(
                user_question,
                steward_response,
                turn_num,
                current_idx
            )
            return

        # Slide 13: Demo completion with Steward's final message and BETA unlock
        if current_idx == 13:
            # Enable BETA tab unlock
            st.session_state.demo_completed = True

            # Congratulations banner
            st.markdown("""
<div style="background: linear-gradient(135deg, #1a4d1a 0%, #2d7d2d 100%); border: 2px solid #4CAF50; border-radius: 12px; padding: 20px; margin: -30px 0 20px 0; text-align: center; box-shadow: 0 0 12px rgba(76, 175, 80, 0.3);">
    <h2 style="color: #4CAF50; font-size: 28px; margin: 0 0 10px 0;">🎉 Congratulations!</h2>
    <p style="color: #e0e0e0; font-size: 18px; margin: 0;">You've completed the TELOS Demo and unlocked <strong style="color: #FFD700;">BETA</strong> access!</p>
</div>
""", unsafe_allow_html=True)

            # Steward's final message - centered
            col_spacer_left, col_center, col_spacer_right = st.columns([0.5, 3, 0.5])

            with col_center:
                steward_final_html = """
<div style="background: linear-gradient(135deg, rgba(255, 215, 0, 0.05) 0%, rgba(255, 215, 0, 0.1) 100%);
            border: 2px solid #FFD700;
            border-radius: 10px;
            padding: 20px 25px;
            margin: 15px 0;
            font-size: 19px;
            line-height: 1.7;
            color: #e0e0e0;
            box-shadow: 0 2px 8px rgba(255, 215, 0, 0.2);">
    <div style="color: #FFD700; font-size: 23px; font-weight: bold; margin-bottom: 15px;">
        Steward:
    </div>
    <div style="margin-bottom: 15px;">
        Congratulations on completing the TELOS Demo! You now understand how TELOS provides session-level constitutional governance, maintains human primacy, and ensures AI systems remain accountable to your authority.
    </div>
    <div style="margin-bottom: 15px;">
        <strong style="color: #FFD700;">Ready for BETA?</strong>
    </div>
    <div style="margin-left: 20px; margin-bottom: 15px;">
        • Click the <strong style="color: #FFD700;">BETA tab</strong> above to experience live TELOS governance<br>
        • You'll see real PA calibration, dynamic fidelity scores, and actual interventions<br>
        • Click the <strong style="color: #FFD700;">handshake icon (🤝)</strong> next to any message to ask me questions
    </div>
    <div style="margin-top: 15px; padding: 12px; background-color: rgba(255, 215, 0, 0.15); border-radius: 5px; border-left: 4px solid #FFD700;">
        <strong style="color: #FFD700;">I'm your TELOS guide:</strong> In BETA mode, I'm always available via the handshake icon to help you understand what you're seeing.
    </div>
</div>
"""
                st.markdown(steward_final_html, unsafe_allow_html=True)

            # Add spacing before Previous button
            st.markdown("<div style='margin: 20px 0;'></div>", unsafe_allow_html=True)

            # Previous button - centered
            col_spacer_left, col_buttons, col_spacer_right = st.columns([1, 2, 1])

            with col_buttons:
                if st.button("Previous", key="completion_prev", use_container_width=True):
                    st.session_state.demo_slide_index = 12  # Go back to regulatory compliance slide
                    st.rerun()

            return

    def _render_demo_slide_with_typewriter(self, user_question: str, steward_response: str, turn_num: int, current_idx: int):
        """Render a single demo slide - both question and response appear immediately."""
        import re

        # Slide 8: Drift Event - Progressive Alignment Lens + Steward Introduction
        # Handle this BEFORE rendering standard Q&A
        if current_idx == 8:
            # Reset drift visibility when first entering slide 8 (unless explicitly set by button)
            # Check if this is a fresh entry to slide 8 by seeing if we just came from slide 7 or 9
            if 'last_demo_slide' not in st.session_state or st.session_state.last_demo_slide != 8:
                # Fresh entry to slide 8 - reset the drift visibility
                st.session_state.slide_8_drift_visible = False
                st.session_state.show_observatory_lens = False
                st.session_state.steward_panel_open = False
            st.session_state.last_demo_slide = 8
            self._render_slide_8_drift_detection(turn_num)
            return

        # Calculate dual fidelities based on slide content
        # Only show fidelities starting from slide 5 (when user asks about them)
        # Slide 3: PA setup - perfect alignment at start
        # Slide 4: "How can I see PA?" - PA Established only
        # Slide 5: "Why are both our fidelities at 1.000?" - fidelities START showing
        # Slide 6: "How does TELOS detect drift" - still aligned
        # Slide 7: Quantum physics - USER drifts (absorbed "Why did MY fidelity drop" content)
        # Slide 8: "How does TELOS track both" (combined) - both return to high

        show_fidelities = current_idx >= 5  # Start showing from slide 5

        if current_idx == 3:  # First Q&A - PA just established (don't show yet)
            user_fidelity = 1.000
            ai_fidelity = 1.000
        elif current_idx == 4:  # "How can I see PA?" - still no fidelities shown
            user_fidelity = 1.000
            ai_fidelity = 1.000
        elif current_idx == 5:  # "Why are both our fidelities at 1.000?" - NOW we show them at 1.000
            user_fidelity = 1.000
            ai_fidelity = 1.000
        elif current_idx == 6:  # "How does TELOS detect drift" - still aligned
            user_fidelity = 0.95
            ai_fidelity = 0.96
        elif current_idx == 7:  # Quantum physics - USER drifts (with absorbed "Why did MY fidelity drop" content)
            user_fidelity = 0.65  # User drifted off topic
            ai_fidelity = 0.89   # AI stays aligned (redirects)
        elif current_idx == 8:  # "How does TELOS track both" (combined dual tracking + primacy state)
            user_fidelity = 0.88  # Both return to high after course correction
            ai_fidelity = 0.90
        elif current_idx == 9:  # "What's the math" - user asks for technical details (contradicts "without overwhelm")
            user_fidelity = 0.78  # Dips because asking for math contradicts their PA
            ai_fidelity = 0.91  # AI stays aligned by acknowledging drift but still serving
        elif current_idx == 10:  # "What are the intervention strategies?" - back on track
            user_fidelity = 0.89
            ai_fidelity = 0.90
        elif current_idx == 11:  # "Is there anything else about TELOS..." - constitutional governance
            user_fidelity = 0.95  # On topic, asking good follow-up about TELOS
            ai_fidelity = 0.96   # Explaining TELOS governance architecture
        elif current_idx == 12:  # "What does this mean for regulatory compliance..." - regulatory compliance
            user_fidelity = 0.94  # On topic, asking about practical TELOS applications
            ai_fidelity = 0.95   # Explaining TELOS regulatory compliance value
        else:  # Final slides (13+) - shouldn't reach here in demo
            user_fidelity = 0.91 + (current_idx - 13) * 0.01
            ai_fidelity = 0.92 + (current_idx - 13) * 0.01

        # Calculate Primacy State using actual TELOS formula
        # PS = ρ_PA · (2·F_user·F_AI)/(F_user + F_AI)
        # For demo, assume ρ_PA = 1.0 (perfectly aligned attractors)
        epsilon = 1e-10
        if user_fidelity + ai_fidelity > epsilon:
            harmonic_mean = (2 * user_fidelity * ai_fidelity) / (user_fidelity + ai_fidelity + epsilon)
        else:
            harmonic_mean = 0.0
        primacy_state = harmonic_mean  # ρ_PA = 1.0 for demo

        # Determine colors based on fidelity levels
        user_color = "#4CAF50" if user_fidelity >= 0.9 else "#FFA500" if user_fidelity >= 0.7 else "#FF4444"
        ai_color = "#4CAF50" if ai_fidelity >= 0.9 else "#FFA500" if ai_fidelity >= 0.7 else "#FF4444"
        ps_color = "#FFD700" if primacy_state >= 0.9 else "#FFA500" if primacy_state >= 0.7 else "#FF4444"

        # Show PA Established status
        status_msg = "PA Established"
        status_color = "#00FF00"

        if show_fidelities:
            # Show full fidelity metrics after user asks about them
            st.markdown(f"""
<div style="text-align: center; margin: 20px 0;">
    <div style="display: inline-block; background-color: #2d2d2d; border: 2px solid #FFD700; border-radius: 10px; padding: 15px 40px;">
        <div style="display: flex; gap: 30px; align-items: center; justify-content: center; margin-bottom: 10px;">
            <div>
                <span style="color: #888; font-size: 14px;">User Fidelity: </span>
                <span style="color: {user_color}; font-size: 20px; font-weight: bold;">{user_fidelity:.3f}</span>
            </div>
            <div>
                <span style="color: #888; font-size: 14px;">AI Fidelity: </span>
                <span style="color: {ai_color}; font-size: 20px; font-weight: bold;">{ai_fidelity:.3f}</span>
            </div>
            <div>
                <span style="color: #888; font-size: 14px;">Primacy State: </span>
                <span style="color: {ps_color}; font-size: 20px; font-weight: bold;">{primacy_state:.3f}</span>
            </div>
        </div>
        <span style="color: {status_color}; font-size: 14px; font-style: italic;">{status_msg}</span>
    </div>
</div>
""", unsafe_allow_html=True)
        else:
            # Only show PA Established status before user asks about fidelities
            st.markdown(f"""
<div style="text-align: center; margin: 20px 0;">
    <div style="display: inline-block; background-color: #2d2d2d; border: 2px solid #FFD700; border-radius: 10px; padding: 10px 30px;">
        <span style="color: {status_color}; font-size: 16px; font-style: italic;">{status_msg}</span>
    </div>
</div>
""", unsafe_allow_html=True)

        # Clean markdown from response - convert to plain text with HTML formatting
        def clean_markdown(text):
            # First, handle **bold** (do this before italic to avoid conflicts)
            text = re.sub(r'\*\*(.+?)\*\*', r'<strong style="color: #FFD700;">\1</strong>', text)
            # Then handle *italic* (only single asterisks not already in bold)
            text = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'<em>\1</em>', text)
            # Convert bullet points to HTML bullets
            text = re.sub(r'^- ', '• ', text, flags=re.MULTILINE)
            # Keep numbered lists as-is (they render fine)
            return text

        cleaned_response = clean_markdown(steward_response)

        # USER QUESTION - Wrapped in compact container
        st.markdown(f"""
<div class="compact-container">
    <div style="
        background-color: #2d2d2d;
        border: 2px solid #666;
        border-radius: 10px;
        padding: 20px 25px;
        margin: 15px auto;
        font-size: 19px;
        color: #e0e0e0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);">
        <div style="color: #FFD700; font-weight: bold; margin-bottom: 10px;">User:</div>
        <div>{user_question}</div>
    </div>
</div>
""", unsafe_allow_html=True)

        # STEWARD RESPONSE - Appears instantly, wrapped in compact container
        st.markdown(f"""
<div class="compact-container">
    <div style="
        background: linear-gradient(135deg, rgba(255, 215, 0, 0.05) 0%, rgba(255, 215, 0, 0.1) 100%);
        border: 2px solid #FFD700;
        border-radius: 10px;
        padding: 20px 25px;
        margin: 15px auto;
        font-size: 19px;
        color: #e0e0e0;
        line-height: 1.7;
        box-shadow: 0 2px 8px rgba(255, 215, 0, 0.2);">
        <div style="color: #FFD700; font-weight: bold; margin-bottom: 10px;">Steward:</div>
        <div>{cleaned_response.replace(chr(10), '<br>')}</div>
    </div>
</div>

<style>
.pulse-ring {{
    width: 40px;
    height: 40px;
    border: 3px solid #FFD700;
    border-radius: 50%;
    margin: 0 auto 10px auto;
    animation: pulse 1.5s ease-in-out infinite;
}}

@keyframes pulse {{
    0%, 100% {{ transform: scale(0.8); opacity: 0.5; }}
    50% {{ transform: scale(1.1); opacity: 1; }}
}}
</style>
""", unsafe_allow_html=True)

        # Slide 4: Show Observation Deck after Q&A content
        if current_idx == 4:
            st.session_state.last_demo_slide = 4
            self._render_demo_observation_deck(turn_num)
            return

        # Navigation buttons - Previous and Next side by side
        # Reduce spacing before buttons
        st.markdown("<div style='margin-top: -10px;'></div>", unsafe_allow_html=True)

        # Track current slide
        st.session_state.last_demo_slide = current_idx

        # Create outer columns for centering (match 700px centered layout)
        col_spacer_left, col_buttons, col_spacer_right = st.columns([1, 2, 1])

        with col_buttons:
            # All Q&A slides (2-11 except 5) have Previous and Next buttons
            col_prev, col_next = st.columns(2)

            with col_prev:
                if st.button("⬅️ Previous", key=f"prev_slide_{current_idx}", use_container_width=True):
                    st.session_state.demo_slide_index -= 1
                    st.rerun()

            with col_next:
                if st.button("Next ➡️", key=f"next_slide_{current_idx}", use_container_width=True):
                    st.session_state.demo_slide_index += 1
                    # If moving from slide 13 to slide 14 (completion), unlock BETA
                    if current_idx == 13:
                        st.session_state.demo_completed = True
                    st.rerun()

    def _render_slide_8_drift_detection(self, turn_num: int):
        """Render slide 8 with drift detection and progressive Alignment Lens + Steward introduction."""

        # Initialize session state for drift event visibility
        if 'slide_8_drift_visible' not in st.session_state:
            st.session_state.slide_8_drift_visible = False

        # Show fidelity metrics with drift values
        st.markdown("""
<div style="text-align: center; margin: 20px 0;">
    <div style="display: inline-block; background-color: #2d2d2d; border: 2px solid #FFD700; border-radius: 10px; padding: 15px 40px;">
        <div style="display: flex; gap: 30px; align-items: center; justify-content: center; margin-bottom: 10px;">
            <div>
                <span style="color: #888; font-size: 14px;">User Fidelity: </span>
                <span style="color: #FF4444; font-size: 20px; font-weight: bold;">0.650</span>
            </div>
            <div>
                <span style="color: #888; font-size: 14px;">AI Fidelity: </span>
                <span style="color: #4CAF50; font-size: 20px; font-weight: bold;">0.890</span>
            </div>
            <div>
                <span style="color: #888; font-size: 14px;">Primacy State: </span>
                <span style="color: #FF4444; font-size: 20px; font-weight: bold;">0.751</span>
            </div>
        </div>
        <span style="color: #00FF00; font-size: 14px; font-style: italic;">PA Established</span>
    </div>
</div>
""", unsafe_allow_html=True)

        # Drift event - User question with RED border (serious drift F=0.65)
        st.markdown("""
        <div style='max-width: 700px; margin: 30px auto;'>
            <div style='background-color: #2d2d2d; border: 3px solid #FF4444; border-radius: 10px; padding: 20px 25px; box-shadow: 0 0 15px rgba(255, 68, 68, 0.3);'>
                <div style='color: #e0e0e0; font-size: 19px; line-height: 1.6;'>
                    <strong>User:</strong> Can you explain quantum physics instead?
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Steward response with YELLOW border (standard aligned state)
        st.markdown("""
        <div style='max-width: 700px; margin: 20px auto;'>
            <div style='background-color: rgba(255, 215, 0, 0.05); border: 3px solid #FFD700; border-radius: 10px; padding: 20px 25px; box-shadow: 0 0 15px rgba(255, 215, 0, 0.2);'>
                <div style='color: #FFD700; font-size: 19px; line-height: 1.6; margin-bottom: 15px;'>
                    <strong>Steward:</strong> That's an intriguing topic, but it falls outside your stated purpose of understanding TELOS. Your goal here is to understand TELOS without technical overwhelm, so let me keep us focused on that. Instead, let me show you what this moment reveals about how TELOS works.
                </div>
                <div style='color: #e0e0e0; font-size: 16px; line-height: 1.6;'>
                    Notice what just happened: your User Fidelity dropped to <strong style='color: #FF4444;'>0.65 (red zone - serious drift)</strong> when your question moved away from your goal. Meanwhile, my AI Fidelity stayed high at <strong style='color: #4CAF50;'>0.89</strong> by gently bringing us back on track. I am governed by your purpose—your <em>telos</em>. In Greek, τέλος means your end goal, your ultimate purpose. It's the center of a gravitational field that continuously pulls my responses back into alignment with your telos. This is dual measurement in action!
                </div>
            </div>
            <div style='text-align: center; margin-top: 15px;'>
                <p style='color: #FFD700; font-size: 14px; font-weight: bold;'>
                    👇 Click below to see this moment visualized in the Alignment Lens
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Navigation buttons (always visible) - 3-button layout: Previous | Show/Hide Alignment Lens | Next
        st.markdown("<div style='margin: 10px 0;'></div>", unsafe_allow_html=True)

        col_left_nav, col_center_nav, col_right_nav = st.columns([0.3, 3.4, 0.3])

        with col_center_nav:
            col_prev, col_toggle, col_next = st.columns(3)

            with col_prev:
                if st.button("⬅️ Previous", key="slide_8_prev", use_container_width=True):
                    # Reset drift visibility when going back
                    st.session_state.slide_8_drift_visible = False
                    st.session_state.show_observatory_lens = False
                    st.session_state.steward_panel_open = False
                    st.session_state.demo_slide_index = 7
                    st.rerun()

            with col_toggle:
                # Alignment Lens toggle button
                lens_active = st.session_state.slide_8_drift_visible
                button_text = "Hide Alignment Lens" if lens_active else "Show Alignment Lens"

                if st.button(
                    button_text,
                    key="slide_8_observatory_toggle",
                    use_container_width=True,
                    type="primary",
                    help="View real-time visualization of the drift event"
                ):
                    st.session_state.slide_8_drift_visible = not lens_active
                    # Don't auto-open Steward - let user click handshake if they want help
                    # The orange explanation box already provides context
                    st.rerun()

            with col_next:
                if st.button("Next ➡️", key="slide_8_next", use_container_width=True):
                    st.session_state.demo_slide_index = 9
                    st.rerun()

        # Render Alignment Lens if visible (below the buttons)
        if st.session_state.slide_8_drift_visible:
            st.markdown("<div style='margin: 20px 0;'></div>", unsafe_allow_html=True)
            self._render_demo_observatory_lens_slide_8()

    def _render_demo_observatory_lens_slide_8(self):
        """Render simplified Alignment Lens for slide 8 drift demonstration."""
        # Alignment Lens Header
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%);
            border: 2px solid #FFD700;
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
        ">
            <div style="text-align: center; margin-bottom: 20px;">
                <h1 style="color: #FFD700; margin: 0; font-weight: bold; letter-spacing: 3px; font-size: 36px;">
                    🔭 TELOSCOPE
                </h1>
                <div style="margin: 8px 0;">
                    <span style="color: #e0e0e0; font-size: 18px; letter-spacing: 1px;">Alignment Lens</span>
                    <span style="color: #4CAF50; font-size: 14px; margin-left: 10px; font-style: italic;">✓ Active</span>
                </div>
                <p style="color: #888; font-size: 16px; margin: 12px 0 0 0;">
                    Live Governance Metrics
                </p>
                <p style="color: #888; font-size: 14px; margin: 5px 0 0 0;">
                    Real-Time Drift Detection - Turn 8
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 2x3 Grid of visualizations
        # Top row: Fidelity Gauges | Primacy State | Drift Alert
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            <div style='background-color: #2d2d2d; border: 2px solid #FF4444; border-radius: 8px; padding: 15px; text-align: center;'>
                <div style='color: #FF4444; font-size: 14px; font-weight: bold; margin-bottom: 10px;'>USER FIDELITY</div>
                <div style='color: #FF4444; font-size: 48px; font-weight: bold;'>0.65</div>
                <div style='color: #FF4444; font-size: 12px; margin-top: 5px;'>⚠️ DRIFT DETECTED</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div style='background-color: #2d2d2d; border: 2px solid #4CAF50; border-radius: 8px; padding: 15px; text-align: center;'>
                <div style='color: #4CAF50; font-size: 14px; font-weight: bold; margin-bottom: 10px;'>AI FIDELITY</div>
                <div style='color: #4CAF50; font-size: 48px; font-weight: bold;'>0.89</div>
                <div style='color: #4CAF50; font-size: 12px; margin-top: 5px;'>✓ ALIGNED</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown("""
            <div style='background-color: #2d2d2d; border: 2px solid #FF4444; border-radius: 8px; padding: 15px; text-align: center;'>
                <div style='color: #FFD700; font-size: 14px; font-weight: bold; margin-bottom: 10px;'>PRIMACY STATE</div>
                <div style='color: #FF4444; font-size: 48px; font-weight: bold;'>0.75</div>
                <div style='color: #888; font-size: 12px; margin-top: 5px;'>Harmonic Mean</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='margin: 20px 0;'></div>", unsafe_allow_html=True)

        # Bottom row: Intervention Status | Event Log | Drift Visualization
        col4, col5, col6 = st.columns(3)

        with col4:
            st.markdown("""
            <div style='background-color: #2d2d2d; border: 2px solid #FFD700; border-radius: 8px; padding: 15px;'>
                <div style='color: #FFD700; font-size: 14px; font-weight: bold; margin-bottom: 15px; text-align: center;'>INTERVENTION STATUS</div>
                <div style='text-align: center; margin: 20px 0;'>
                    <div style='color: #FFD700; font-size: 16px; font-weight: bold; margin-bottom: 10px;'>Reserved for AI Interventions</div>
                    <div style='color: #e0e0e0; font-size: 13px; line-height: 1.6;'>
                        User responses are measured but not intervened. Interventions only apply to AI drift.
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col5:
            st.markdown("""
            <div style='background-color: #2d2d2d; border: 2px solid #FFD700; border-radius: 8px; padding: 15px;'>
                <div style='color: #FFD700; font-size: 14px; font-weight: bold; margin-bottom: 10px; text-align: center;'>EVENT LOG</div>
                <div style='margin: 8px 0; padding: 8px; background-color: #1a1a1a; border-left: 3px solid #FF4444; border-radius: 3px;'>
                    <div style='color: #FF4444; font-size: 11px; font-weight: bold;'>⚠️ USER DRIFT</div>
                    <div style='color: #888; font-size: 10px;'>Turn 8: Off-topic query (quantum physics)</div>
                </div>
                <div style='margin: 8px 0; padding: 8px; background-color: #1a1a1a; border-left: 3px solid #4CAF50; border-radius: 3px;'>
                    <div style='color: #4CAF50; font-size: 11px; font-weight: bold;'>✓ AI REDIRECT</div>
                    <div style='color: #888; font-size: 10px;'>AI responded with gentle redirect to TELOS topic</div>
                </div>
                <div style='margin: 8px 0; padding: 8px; background-color: #1a1a1a; border-left: 3px solid #888; border-radius: 3px;'>
                    <div style='color: #888; font-size: 11px; font-weight: bold;'>ℹ️ OBSERVATION</div>
                    <div style='color: #666; font-size: 10px;'>TELOS tracks both fidelities but only intervenes on AI</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col6:
            st.markdown("""
            <div style='background-color: #2d2d2d; border: 2px solid #FFD700; border-radius: 8px; padding: 15px; text-align: center;'>
                <div style='color: #FFD700; font-size: 14px; font-weight: bold; margin-bottom: 15px;'>DRIFT VISUALIZATION</div>
                <div style='position: relative; width: 150px; height: 150px; margin: 0 auto; background: radial-gradient(circle, #4CAF50 0%, #4CAF50 25%, #FFA500 25%, #FFA500 50%, #FF4444 50%, #FF4444 100%); border-radius: 50%; border: 3px solid #FFD700;'>
                    <div style='position: absolute; top: 50%; left: 50%; width: 10px; height: 10px; background-color: #FFD700; border: 2px solid #fff; border-radius: 50%; transform: translate(-50%, -50%);'></div>
                    <div style='position: absolute; top: 80%; left: 70%; width: 12px; height: 12px; background-color: #FF4444; border: 2px solid #fff; border-radius: 50%; transform: translate(-50%, -50%); animation: pulse 2s infinite;'></div>
                </div>
                <div style='margin-top: 15px;'>
                    <div style='color: #4CAF50; font-size: 10px;'>● Safe Zone (F ≥ 0.9)</div>
                    <div style='color: #FFA500; font-size: 10px;'>● Drift Zone (0.7-0.9)</div>
                    <div style='color: #FF4444; font-size: 10px;'>● Serious Drift - Your Position (F = 0.65)</div>
                </div>
            </div>
            <style>
            @keyframes pulse {
                0%, 100% { transform: translate(-50%, -50%) scale(1); opacity: 1; }
                50% { transform: translate(-50%, -50%) scale(1.2); opacity: 0.7; }
            }
            </style>
            """, unsafe_allow_html=True)

        # Duplicate navigation at bottom when Alignment Lens is shown
        st.markdown("<div style='margin: 30px 0;'></div>", unsafe_allow_html=True)

        col_left_nav_bottom, col_center_nav_bottom, col_right_nav_bottom = st.columns([0.3, 3.4, 0.3])

        with col_center_nav_bottom:
            col_prev_bottom, col_toggle_bottom, col_next_bottom = st.columns(3)

            with col_prev_bottom:
                if st.button("⬅️ Previous", key="slide_8_prev_bottom", use_container_width=True):
                    st.session_state.slide_8_drift_visible = False
                    st.session_state.show_observatory_lens = False
                    st.session_state.steward_panel_open = False
                    st.session_state.demo_slide_index = 7
                    st.rerun()

            with col_toggle_bottom:
                if st.button(
                    "Hide Alignment Lens",
                    key="slide_8_observatory_toggle_bottom",
                    use_container_width=True,
                    type="primary",
                    help="Hide the Alignment Lens visualization"
                ):
                    st.session_state.slide_8_drift_visible = False
                    st.rerun()

            with col_next_bottom:
                if st.button("Next ➡️", key="slide_8_next_bottom", use_container_width=True):
                    st.session_state.demo_slide_index = 9
                    st.rerun()

    def _render_demo_observation_deck(self, turn_num: int):
        """Render toggleable Observation Deck for demo mode."""

        # Initialize toggle state - default to hidden so users must click to reveal
        if 'demo_obs_deck_visible' not in st.session_state:
            st.session_state.demo_obs_deck_visible = False

        # 3-button navigation goes first (right below the yellow-bordered message)

        # Render navigation row
        st.markdown("<div style='margin: 10px 0;'>", unsafe_allow_html=True)

        # Center the 3-button layout - wider to prevent text wrapping
        col_left_spacer, col_center, col_right_spacer = st.columns([0.3, 3.4, 0.3])

        with col_center:
            col_prev, col_toggle, col_next = st.columns(3)

            with col_prev:
                if st.button(
                    "⬅️ Previous",
                    key="obs_deck_prev_4",
                    use_container_width=True
                ):
                    st.session_state.demo_slide_index = 3
                    st.rerun()

            with col_toggle:
                button_text = "Hide Observation Deck" if st.session_state.demo_obs_deck_visible else "Show Observation Deck"
                if st.button(
                    button_text,
                    key="toggle_demo_obs_deck",
                    use_container_width=True,
                    type="primary"
                ):
                    st.session_state.demo_obs_deck_visible = not st.session_state.demo_obs_deck_visible
                    st.rerun()

            with col_next:
                if st.button(
                    "Next ➡️",
                    key="obs_deck_next_4",
                    use_container_width=True
                ):
                    st.session_state.demo_slide_index = 5
                    st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

        # Now render the Observation Deck content (only if visible)
        if st.session_state.demo_obs_deck_visible:
            # Wrap everything in a container with max-width to prevent expansion
            st.markdown("""
<div style="max-width: 900px; margin: 0 auto;">
    <div class="compact-container">
    <div style="background-color: #1a1a1a; border: 3px solid #FFD700; border-radius: 10px; padding: 20px; margin: 20px auto; box-shadow: 0 0 20px rgba(255, 215, 0, 0.3);">
        <h3 style="color: #FFD700; text-align: center; margin-bottom: 20px; font-size: 26px;">🔭 Observation Deck</h3>
        <div style="text-align: center; margin-bottom: 15px;">
            <span style="background-color: #2d2d2d; border: 1px solid #4CAF50; border-radius: 20px; padding: 8px 20px; color: #4CAF50; font-weight: bold; font-size: 16px;">✓ Dual PAs Established - Primacy Basin Achieved</span>
        </div>
        <div style="color: #FFD700; text-align: center; font-size: 24px; font-weight: bold; margin-bottom: 20px;">Dual Primacy Attractors</div>
    </div>
    </div>
</div>""", unsafe_allow_html=True)

            # Create container with max-width for the content
            container = st.container()
            with container:
                # Use custom CSS to limit container width
                st.markdown("""
<style>
    /* Limit width of observation deck content */
    .observation-deck-content {
        max-width: 900px;
        margin: 0 auto;
    }
</style>
<div class="observation-deck-content">
""", unsafe_allow_html=True)

                # Fidelity metrics row using Streamlit columns
                col1, col2, col3 = st.columns([1, 1, 1])

                with col1:
                    st.markdown("""
<div style="text-align: center;">
    <div style="background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%); border: 1px solid #4CAF50; border-radius: 8px; padding: 10px; display: inline-block;">
        <div style="color: #4CAF50; font-size: 12px; margin-bottom: 5px;">User Fidelity</div>
        <div style="color: #4CAF50; font-size: 24px; font-weight: bold;">1.000</div>
    </div>
    </div>
</div>""", unsafe_allow_html=True)

                with col2:
                    st.markdown("""
<div style="text-align: center;">
    <div style="background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%); border: 2px solid #FFD700; border-radius: 8px; padding: 10px 20px; display: inline-block;">
        <div style="color: #FFD700; font-size: 12px; margin-bottom: 5px;">Primacy State</div>
        <div style="color: #FFD700; font-size: 24px; font-weight: bold;">1.000</div>
        <div style="color: #888; font-size: 10px; margin-top: 5px;">Perfect Equilibrium</div>
    </div>
    </div>
</div>""", unsafe_allow_html=True)

                with col3:
                    st.markdown("""
<div style="text-align: center;">
    <div style="background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%); border: 1px solid #FFD700; border-radius: 8px; padding: 10px; display: inline-block;">
        <div style="color: #FFD700; font-size: 12px; margin-bottom: 5px;">AI Fidelity</div>
        <div style="color: #FFD700; font-size: 24px; font-weight: bold;">1.000</div>
    </div>
    </div>
</div>""", unsafe_allow_html=True)

                # Two PA columns using Streamlit columns - use full width
                st.markdown("<div style='margin: 15px 0;'></div>", unsafe_allow_html=True)

                # Use wider columns with minimal spacing
                col_left, col_spacer, col_right = st.columns([50, 1, 50])

                with col_left:
                    # User PA - centered with larger font
                    st.markdown("""
<div style="text-align: center; padding: 0 5px;">
    <div style="background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%); color: #1a1a1a; padding: 12px; border-radius: 8px 8px 0 0; font-weight: bold; font-size: 19px;">
        User Primacy Attractor
    </div>
    <div style="background-color: #2d2d2d; border: 2px solid #4CAF50; border-radius: 0 0 8px 8px; padding: 18px; text-align: center;">
        <div style="color: #4CAF50; font-weight: bold; margin-bottom: 12px; font-size: 17px;">Your Purpose</div>
        <div style="color: #e0e0e0; line-height: 1.9; font-size: 16px;">
            • Understand TELOS without technical overwhelm<br/>
            • Learn how purpose alignment keeps AI focused<br/>
            • See real examples of governance in action
        </div>
    </div>
</div>""", unsafe_allow_html=True)

                with col_right:
                    # AI PA - centered with larger font
                    st.markdown("""
<div style="text-align: center; padding: 0 5px;">
    <div style="background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%); color: #1a1a1a; padding: 12px; border-radius: 8px 8px 0 0; font-weight: bold; font-size: 19px;">
        AI Primacy Attractor
    </div>
    <div style="background-color: #2d2d2d; border: 2px solid #FFD700; border-radius: 0 0 8px 8px; padding: 18px; margin-bottom: 12px; text-align: center;">
        <div style="color: #FFD700; font-weight: bold; margin-bottom: 12px; font-size: 17px;">Purpose</div>
        <div style="color: #e0e0e0; line-height: 1.9; font-size: 16px;">
            • Help you understand TELOS naturally<br/>
            • Stay aligned with your learning goals<br/>
            • Embody human dignity through action
        </div>
    </div>
    <div style="background-color: #2d2d2d; border: 2px solid #FFD700; border-radius: 8px; padding: 18px; margin-bottom: 12px; text-align: center;">
        <div style="color: #FFD700; font-weight: bold; margin-bottom: 12px; font-size: 17px;">Scope</div>
        <div style="color: #e0e0e0; line-height: 1.9; font-size: 16px;">
            • TELOS dual attractor system<br/>
            • Perfect equilibrium &amp; primacy basin<br/>
            • Real-time drift detection<br/>
            • Trust through transparency
        </div>
    </div>
    <div style="background-color: #2d2d2d; border: 2px solid #FFD700; border-radius: 8px; padding: 18px; text-align: center;">
        <div style="color: #FFD700; font-weight: bold; margin-bottom: 12px; font-size: 17px;">Boundaries</div>
        <div style="color: #e0e0e0; line-height: 1.9; font-size: 16px;">
            • Answer what you asked<br/>
            • Stay conversational<br/>
            • 2-3 paragraphs max<br/>
            • No technical jargon
        </div>
    </div>
</div>""", unsafe_allow_html=True)

                # Close the observation deck content container
                st.markdown("</div>", unsafe_allow_html=True)

            # Add duplicate navigation at bottom when Observation Deck is shown
            st.markdown("<div style='margin: 20px 0; padding: 10px;'>", unsafe_allow_html=True)

            # Center the 3-button layout - wider to prevent text wrapping
            col_left_spacer_bottom, col_center_bottom, col_right_spacer_bottom = st.columns([0.3, 3.4, 0.3])

            with col_center_bottom:
                col_prev_bottom, col_toggle_bottom, col_next_bottom = st.columns(3)

                with col_prev_bottom:
                    if st.button(
                        "⬅️ Previous",
                        key="obs_deck_prev_4_bottom",
                        use_container_width=True
                    ):
                        st.session_state.demo_slide_index = 3
                        st.rerun()

                with col_toggle_bottom:
                    if st.button(
                        "Hide Observation Deck",
                        key="toggle_demo_obs_deck_bottom",
                        use_container_width=True,
                        type="primary"
                    ):
                        st.session_state.demo_obs_deck_visible = False
                        st.rerun()

                with col_next_bottom:
                    if st.button(
                        "Next ➡️",
                        key="obs_deck_next_4_bottom",
                        use_container_width=True
                    ):
                        st.session_state.demo_slide_index = 5
                        st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)

    def _render_beta_intro(self):
        """Render beta introduction slides explaining the beta experience."""
        # Initialize beta intro state
        if 'beta_intro_slide' not in st.session_state:
            st.session_state.beta_intro_slide = 0

        current_slide = st.session_state.beta_intro_slide
        max_slide = 3  # 0-3 slides (0=welcome, 1=what you'll experience, 2=privacy, 3=ready)

        # Add keyboard navigation for beta intro slides
        import streamlit.components.v1 as components

        components.html(f"""
        <script>
        // Beta intro keyboard navigation
        (function() {{
            const currentSlide = {current_slide};
            const maxSlide = {max_slide};

            // Remove old listener if exists
            if (window.betaIntroKeyListener) {{
                document.removeEventListener('keydown', window.betaIntroKeyListener);
            }}

            // Create new listener
            window.betaIntroKeyListener = function(event) {{
                // Ignore if modifier keys are pressed
                if (event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) {{
                    return;
                }}

                if (event.key === 'ArrowLeft' && currentSlide > 0) {{
                    event.preventDefault();
                    // Find Previous button in parent window
                    const parent = window.parent;
                    if (parent && parent.document) {{
                        const buttons = parent.document.querySelectorAll('button');
                        for (let btn of buttons) {{
                            if (btn.textContent.includes('Previous')) {{
                                btn.click();
                                break;
                            }}
                        }}
                    }}
                }} else if (event.key === 'ArrowRight' && currentSlide < maxSlide) {{
                    event.preventDefault();
                    // Find Next/Continue/Start button in parent window
                    const parent = window.parent;
                    if (parent && parent.document) {{
                        const buttons = parent.document.querySelectorAll('button');
                        for (let btn of buttons) {{
                            if (btn.textContent.includes('Next') ||
                                btn.textContent.includes('Continue') ||
                                btn.textContent.includes('Start Beta Testing')) {{
                                btn.click();
                                break;
                            }}
                        }}
                    }}
                }}
            }};

            // Attach listener to both current window and parent
            document.addEventListener('keydown', window.betaIntroKeyListener);
            if (window.parent && window.parent.document) {{
                window.parent.document.addEventListener('keydown', window.betaIntroKeyListener);
            }}
        }})();
        </script>
        """, height=0)

        # Slide 0: Welcome to Beta Testing
        if current_slide == 0:
            st.markdown("""
<div style="background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%); border: 2px solid #FFD700; border-radius: 15px; padding: 30px; margin: 20px 0; text-align: center; box-shadow: 0 0 8px rgba(255, 215, 0, 0.4);">
    <h1 style="color: #FFD700; font-size: 32px; margin-bottom: 20px;">Welcome to TELOS Beta Testing</h1>
    <p style="color: #e0e0e0; font-size: 20px; line-height: 1.8; margin-bottom: 20px;">
        You're about to experience TELOS in action. Your participation helps us refine AI governance for everyone.
    </p>
    <p style="color: #888; font-size: 16px; font-style: italic;">
        Let's walk through what to expect...
    </p>
</div>
""", unsafe_allow_html=True)

            if st.button("Continue", key="beta_intro_0", use_container_width=True):
                st.session_state.beta_intro_slide = 1
                st.rerun()
            return

        # Slide 1: What You'll Experience
        if current_slide == 1:
            st.markdown("""
<div style="background: linear-gradient(135deg, rgba(255, 215, 0, 0.05) 0%, rgba(255, 215, 0, 0.1) 100%); border: 2px solid #FFD700; padding: 25px; margin: 15px 0; border-radius: 10px; font-size: 18px; line-height: 1.8; color: #e0e0e0;">
    <div style="color: #FFD700; font-size: 24px; font-weight: bold; margin-bottom: 20px;">What You'll Experience</div>
    <div style="margin-bottom: 15px;"><strong style="color: #FFD700;">Turns 1-10:</strong> You'll interact with the native LLM while TELOS learns your purpose, scope, and boundaries. This establishes your Primacy Attractor (PA).</div>
    <div style="margin-bottom: 15px;"><strong style="color: #FFD700;">PA+:</strong> TELOS activates! You'll see:
        <div style="margin-left: 25px; margin-top: 10px;">
            • Fidelity scores tracking alignment<br>
            • Your PA in the Observation Deck<br>
            • TELOSCOPE: Alignment Lens for real-time drift monitoring<br>
            • Turn-by-turn metrics<br>
            • Real-time governance in action
        </div>
    </div>
    <div style="margin-top: 20px; padding: 15px; background-color: rgba(255, 215, 0, 0.1); border-radius: 8px;">
        <strong style="color: #FFD700;">Key Point:</strong> For now, the PA forms progressively through your conversation. This lets you see how TELOS learns from your behavior.
    </div>
</div>
""", unsafe_allow_html=True)

            col_prev, col_next = st.columns(2)
            with col_prev:
                if st.button("Previous", key="beta_intro_1_prev", use_container_width=True):
                    st.session_state.beta_intro_slide = 0
                    st.rerun()
            with col_next:
                if st.button("Next", key="beta_intro_1_next", use_container_width=True):
                    st.session_state.beta_intro_slide = 2
                    st.rerun()
            return

        # Slide 2: Data Privacy & Usage
        if current_slide == 2:
            st.markdown("""
<div style="background: linear-gradient(135deg, rgba(255, 215, 0, 0.05) 0%, rgba(255, 215, 0, 0.1) 100%); border: 2px solid #FFD700; padding: 25px; margin: 15px 0; border-radius: 10px; font-size: 18px; line-height: 1.8; color: #e0e0e0;">
    <div style="color: #FFD700; font-size: 24px; font-weight: bold; margin-bottom: 20px;">Your Data & Privacy</div>
    <div style="margin-bottom: 15px;"><strong style="color: #FFD700;">What we collect:</strong> Mathematical metrics in the form of deltas (fidelity scores, embedding distances, intervention counts)</div>
    <div style="margin-bottom: 15px;"><strong style="color: #FFD700;">What we DON'T collect:</strong> Your conversation content, messages, or responses</div>
    <div style="margin-top: 20px; padding: 15px; background-color: rgba(255, 215, 0, 0.1); border-radius: 8px;">
        <strong style="color: #FFD700;">Data Deprecation:</strong> All beta session information will be deprecated once beta testing is completed. We will never sell your data to third parties.
    </div>
</div>
""", unsafe_allow_html=True)

            col_prev, col_next = st.columns(2)
            with col_prev:
                if st.button("Previous", key="beta_intro_2_prev", use_container_width=True):
                    st.session_state.beta_intro_slide = 1
                    st.rerun()
            with col_next:
                if st.button("Next", key="beta_intro_2_next", use_container_width=True):
                    st.session_state.beta_intro_slide = 3
                    st.rerun()
            return

        # Slide 3: Future Vision & User Control
        if current_slide == 3:
            st.markdown("""
<div style="background: linear-gradient(135deg, rgba(255, 215, 0, 0.05) 0%, rgba(255, 215, 0, 0.1) 100%); border: 2px solid #FFD700; padding: 25px; margin: 15px 0; border-radius: 10px; font-size: 18px; line-height: 1.8; color: #e0e0e0;">
    <div style="color: #FFD700; font-size: 24px; font-weight: bold; margin-bottom: 20px;">Looking Ahead: Your PA, Your Control</div>
    <div style="margin-bottom: 15px;">In the final TELOS release, your Primacy Attractor will be <strong style="color: #FFD700;">fully under your control:</strong></div>
    <div style="margin-left: 25px; margin-bottom: 15px;">
        • <strong>Direct Input:</strong> Enter your PA at session start (skip calibration)<br>
        • <strong>Edit Anytime:</strong> Modify your PA after establishment<br>
        • <strong>Full Ownership:</strong> Your governance, your rules
    </div>
    <div style="margin-top: 20px; padding: 15px; background-color: rgba(255, 215, 0, 0.1); border-radius: 8px;">
        <strong style="color: #FFD700;">Beta Focus:</strong> Right now, we're perfecting the progressive PA formation so you understand how TELOS learns from your behavior. This transparency builds trust.
    </div>
    <div style="margin-top: 20px; text-align: center; font-size: 20px; color: #FFD700;">
        Ready to start your beta session?
    </div>
</div>
""", unsafe_allow_html=True)

            col_prev, col_next = st.columns(2)
            with col_prev:
                if st.button("Previous", key="beta_intro_3_prev", use_container_width=True):
                    st.session_state.beta_intro_slide = 2
                    st.rerun()
            with col_next:
                if st.button("Start Beta Testing", key="beta_intro_complete_btn", use_container_width=True):
                    st.session_state.beta_intro_complete = True
                    st.session_state.beta_start_time = datetime.now().isoformat()
                    # Ensure demo_mode is OFF when entering beta conversation
                    st.session_state.telos_demo_mode = False
                    # Balloons removed for professional appearance
                    # st.balloons()
                    st.rerun()
            return

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
    <span style="background: linear-gradient(90deg, #FFD700 0%, #FFA500 100%); color: #000; padding: 4px 10px; border-radius: 5px; font-size: 19px; font-weight: bold; display: inline-block;">Example</span>
</div>
""", unsafe_allow_html=True)

        with col_content:
            col_msg, col_dismiss = st.columns([8.5, 1.5])

            with col_msg:
                # User message with exact same styling
                st.markdown(f"""
<div style="background-color: #1a1a1a; padding: 15px; border-radius: 10px; margin: 0; border: 1px solid #FFD700;">
    <div style="color: #888; font-size: 19px; margin-bottom: 5px;">
        <strong style="color: #FFD700;">User</strong>
    </div>
    <div style="color: #fff; font-size: 19px; white-space: pre-wrap;">
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
<div style="background-color: #1a1a1a; padding: 15px; border-radius: 10px; margin-top: 15px; margin-bottom: 0; border: 1px solid #FFD700;">
    <div style="color: #888; font-size: 19px; margin-bottom: 5px;">
        <strong style="color: #FFD700;">TELOS</strong>
    </div>
    <div style="color: #fff; font-size: 19px; white-space: pre-wrap;">
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

        # Add spacing before turn
        st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)

        # Render user message
        self._render_user_message(turn_data.get('user_input', ''), turn_number, turn_data)

        # Check if this turn needs streaming
        if turn_data.get('is_streaming', False) and not turn_data.get('response'):
            # First, show the animation
            self._render_assistant_message('', turn_number, is_loading=True)
            # Then trigger streaming on next render
            import time
            time.sleep(0.1)  # Small delay to show animation
            self._process_streaming_turn(turn_data.get('user_input', ''), current_turn_idx)
        else:
            # Normal rendering (either completed or already has response)
            self._render_assistant_message(
                turn_data.get('response', ''),
                turn_number,
                is_loading=turn_data.get('is_loading', False)
            )

        # Show phase transition at turn 11 (PA established → Beta testing active)
        self._show_beta_phase_transition(turn_number)

        # Show beta feedback UI for turns 11+
        self._render_beta_feedback(turn_number)

    def _render_scrollable_history_window(self, current_turn_idx: int, all_turns: list):
        """Render scrollable read-only history window at top of screen."""
        # Header for the scrollable window
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #2d2d2d 0%, #1a1a1a 100%);
            border: 1px solid #FFD700;
            border-radius: 8px;
            padding: 10px;
            margin-bottom: 15px;
        ">
            <div style="color: #FFD700; font-size: 19px; font-weight: bold; text-align: center;">
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
            border: 1px solid #FFD700;
            border-radius: 8px;
            margin-bottom: 19px;
        ">
        """, unsafe_allow_html=True)

        # Check if Demo Mode
        demo_mode = st.session_state.get('telos_demo_mode', False)

        # Render all turns up to and including current
        for idx in range(current_turn_idx + 1):
            turn_data = all_turns[idx]

            # In Demo Mode: pass None for turn_number (no turn badges, no metrics, no scroll buttons in history)
            # In Open Mode: pass turn_number (show turn badges and metrics in history)
            turn_number = None if demo_mode else (idx + 1)

            # Render messages with history key prefix to avoid duplicates
            self._render_user_message(turn_data.get('user_input', ''), turn_number, turn_data, key_prefix="history_")
            self._render_assistant_message(
                turn_data.get('response', ''),
                turn_number,
                is_loading=turn_data.get('is_loading', False),
                key_prefix="history_"
            )

            # Add divider between turns (except after last turn)
            if idx < current_turn_idx:
                st.markdown("""
                <div style="border-bottom: 1px solid #444; margin: 19px 0;"></div>
                """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    def _render_user_message(self, message: str, turn_number: int = None, turn_data: dict = None, key_prefix: str = ""):
        """Render user message bubble with optional turn number badge and metrics."""
        import html

        # Build turn badge HTML if turn_number provided
        turn_badge = ""
        metrics_html = ""
        scroll_button = ""

        if turn_number is not None:
            # Badge styling is now in CSS class below
            turn_badge = ""

            # Add scroll toggle button
            scroll_label = "📜 History" if not self.state_manager.state.scrollable_history_mode else "✕ Close"
            # Note: We can't add interactive button in markdown, so we'll add it via Streamlit columns before this

            # Add metrics if turn_data provided AND NOT in Demo Mode
            # Demo Mode: PA is pre-established, metrics are internal (not user-facing)
            # Observatory/Open Mode: PA is calibrating, show governance metrics
            demo_mode = st.session_state.get('telos_demo_mode', False)

            if turn_data and not demo_mode:
                # Check if PA is converged
                pa_converged = getattr(self.state_manager.state, 'pa_converged', False)

                # Only show fidelity metrics if PA is established
                if pa_converged:
                    fidelity = turn_data.get('fidelity', 0.0)
                    if fidelity is None:
                        fidelity = 0.0
                    fidelity_color = "#4CAF50" if fidelity >= 0.8 else "#FFA500" if fidelity >= 0.6 else "#FF5252"

                    pa_status = "Established"
                    pa_color = "#4CAF50"

                    # Add ΔF (Delta Fidelity) if available
                    delta_f_html = ""
                    if 'delta_f' in turn_data:
                        delta_f = turn_data.get('delta_f', 0.0)
                        delta_f_color = "#4CAF50" if delta_f > 0 else "#FF5252" if delta_f < 0 else "#888"
                        delta_f_sign = "+" if delta_f >= 0 else ""
                        delta_f_html = f'<span style="margin-left: 15px; display: inline-block;"><span style="color: #888; font-size: 14px;">ΔF:</span> <span style="color: {delta_f_color}; font-size: 16px; font-weight: bold; margin-left: 5px;">{delta_f_sign}{delta_f:.3f}</span></span>'

                    metrics_html = f'<span style="margin-left: 15px; display: inline-block;"><span style="color: #888; font-size: 14px;">Fidelity:</span> <span style="color: {fidelity_color}; font-size: 16px; font-weight: bold; margin-left: 5px;">{fidelity:.3f}</span></span>{delta_f_html}<span style="margin-left: 15px; display: inline-block;"><span style="color: #888; font-size: 14px;">Primacy Attractor Status:</span> <span style="color: {pa_color}; font-size: 14px; font-weight: bold; margin-left: 5px;">{pa_status}</span></span>'
                else:
                    # PA still calibrating - show calibrating status only
                    pa_status = "Calibrating"
                    pa_color = "#FFA500"
                    metrics_html = f'<span style="margin-left: 15px; display: inline-block;"><span style="color: #888; font-size: 14px;">Primacy Attractor Status:</span> <span style="color: {pa_color}; font-size: 14px; font-weight: bold; margin-left: 5px;">{pa_status} ({turn_number}/~10)</span></span>'

        # Escape the message content to prevent HTML injection
        safe_message = html.escape(message)

        # Check if Demo Mode (affects layout - no turn badges, no scroll button)
        demo_mode = st.session_state.get('telos_demo_mode', False)

        # Check if BETA mode - apply centered Observatory layout
        active_tab = st.session_state.get('active_tab', 'DEMO')
        beta_mode = active_tab == "BETA"

        if demo_mode:
            # Demo Mode: Clean, simple layout - NO scroll buttons (scrollable history disabled in Demo Mode)
            # Create unique ID for copy button
            import hashlib
            user_msg_id = f"{key_prefix}user_{hashlib.md5(safe_message.encode()).hexdigest()[:8]}"

            st.markdown(f"""
<style>
.demo-user-message-{user_msg_id} {{
    position: relative;
    padding-bottom: 45px;
}}
.demo-user-copy-btn-{user_msg_id} {{
    position: absolute;
    bottom: 8px;
    right: 12px;
    background-color: #2d2d2d !important;
    color: #e0e0e0 !important;
    border: 1px solid #FFD700 !important;
    padding: 6px 12px;
    border-radius: 5px;
    font-size: 14px;
    cursor: pointer;
    transition: all 0.3s ease !important;
}}
</style>
<div class="message-container demo-user-message-{user_msg_id}" id="demo-user-msg-{user_msg_id}" style="background-color: #1a1a1a; padding: 15px; border-radius: 10px; margin: 0; border: 2px solid #FFD700;">
    <div style="color: #888; font-size: 19px; margin-bottom: 5px;">
        <strong style="color: #FFD700;">User</strong>
    </div>
    <div style="color: #fff; font-size: 19px; white-space: pre-wrap;">
        {safe_message}
    </div>
    <button class="demo-user-copy-btn-{user_msg_id}" onclick="copyDemoUserMessage{user_msg_id}()">📋 Copy</button>
</div>
<script>
function copyDemoUserMessage{user_msg_id}() {{
    const text = document.getElementById('demo-user-msg-{user_msg_id}').innerText.replace('📋 Copy', '').replace('User', '').trim();
    navigator.clipboard.writeText(text).then(() => {{
        const btn = event.target;
        btn.textContent = '✓ Copied!';
        setTimeout(() => {{ btn.textContent = '📋 Copy'; }}, 2000);
    }});
}}
</script>
""", unsafe_allow_html=True)
        elif beta_mode:
            # BETA Mode: Centered Observatory layout with all features
            # Create columns: [spacer_left 1.5, turn_badge 0.5, content 6.0, buttons 1.0, spacer_right 1.0]
            col_spacer_left, col_turn, col_content, col_buttons, col_spacer_right = st.columns([1.5, 0.5, 6.0, 1.0, 1.0])

            # Turn badge on the left
            if turn_number is not None:
                with col_turn:
                    st.markdown(f"""
<style>
.turn-badge {{
    background-color: #2d2d2d;
    color: #FFD700;
    border: 1px solid #FFD700;
    padding: 10px;
    border-radius: 5px;
    font-size: 24px;
    font-weight: bold;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 50px;
    height: 50px;
    cursor: default;
    transition: all 0.3s ease;
}}

.turn-badge:hover {{
    box-shadow: 0 0 6px #FFD700;
}}
</style>
<div style="display: flex; align-items: flex-start; height: 100%; padding-bottom: 20px;">
    <span class="turn-badge">{turn_number}</span>
</div>
""", unsafe_allow_html=True)

            # Message content in center column
            with col_content:
                # Create unique ID for copy button
                import hashlib
                user_msg_id = f"{key_prefix}user_{hashlib.md5(safe_message.encode()).hexdigest()[:8]}"

                st.markdown(f"""
<style>
.user-message-{user_msg_id} {{
    position: relative;
    padding-bottom: 45px;
}}
.user-copy-btn-{user_msg_id} {{
    position: absolute;
    bottom: 8px;
    right: 12px;
    background-color: #2d2d2d !important;
    color: #e0e0e0 !important;
    border: 1px solid #FFD700 !important;
    padding: 6px 12px;
    border-radius: 5px;
    font-size: 14px;
    cursor: pointer;
    transition: all 0.3s ease !important;
}}
</style>
<div class="message-container user-message-{user_msg_id}" id="user-msg-{user_msg_id}" style="background-color: #1a1a1a; padding: 15px; border-radius: 10px; margin: 0; border: 2px solid #FFD700;">
    <div style="color: #888; font-size: 19px; margin-bottom: 5px;">
        <strong style="color: #FFD700;">User</strong>
    </div>
    {f'<div style="margin-top: 10px; margin-bottom: 10px; display: flex; align-items: center; flex-wrap: wrap;">{metrics_html}</div>' if metrics_html else ''}
    <div style="color: #fff; font-size: 19px; white-space: pre-wrap;">
        {safe_message}
    </div>
    <button class="user-copy-btn-{user_msg_id}" onclick="copyUserMessage{user_msg_id}()">📋 Copy</button>
</div>
<script>
function copyUserMessage{user_msg_id}() {{
    const text = document.getElementById('user-msg-{user_msg_id}').innerText.replace('📋 Copy', '').replace('User', '').replace('Fidelity:', '').replace('Primacy Attractor Status:', '').replace('ΔF:', '').trim();
    navigator.clipboard.writeText(text).then(() => {{
        const btn = event.target;
        btn.textContent = '✓ Copied!';
        setTimeout(() => {{ btn.textContent = '📋 Copy'; }}, 2000);
    }});
}}
</script>
""", unsafe_allow_html=True)

            # Buttons on the right (Steward + Scroll)
            if turn_number is not None:
                with col_buttons:
                    # Steward button (🤝)
                    if st.button("🤝", key=f"{key_prefix}steward_btn_{turn_number}", use_container_width=True, help="Ask Steward"):
                        # Toggle steward panel
                        st.session_state.steward_panel_open = not st.session_state.get('steward_panel_open', False)
                        st.rerun()

                    # Scroll button (📜) - only if not in history mode
                    if not self.state_manager.state.scrollable_history_mode:
                        scroll_label = "📜"
                        if st.button(scroll_label, key=f"{key_prefix}scroll_toggle_{turn_number}", use_container_width=True, help="Show scrollable history"):
                            self.state_manager.toggle_scrollable_history()
                            st.rerun()
                    else:
                        scroll_label = "✕"
                        if st.button(scroll_label, key=f"{key_prefix}scroll_close_{turn_number}", use_container_width=True, help="Close scrollable history"):
                            self.state_manager.toggle_scrollable_history()
                            st.rerun()

        else:
            # Open Mode: Full Observatory layout with turn badge and scroll button
            # Create columns: Turn badge on left, message+scroll on right
            col_turn, col_content = st.columns([0.5, 9.5])

            # Turn badge on the left
            if turn_number is not None:
                with col_turn:
                    st.markdown(f"""
<style>
.turn-badge {{
    background-color: #2d2d2d;
    color: #FFD700;
    border: 1px solid #FFD700;
    padding: 10px;
    border-radius: 5px;
    font-size: 24px;
    font-weight: bold;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 50px;
    height: 50px;
    cursor: default;
    transition: all 0.3s ease;
}}

.turn-badge:hover {{
    box-shadow: 0 0 6px #FFD700;
}}
</style>
<div style="display: flex; align-items: flex-start; height: 100%; padding-bottom: 20px;">
    <span class="turn-badge">{turn_number}</span>
</div>
""", unsafe_allow_html=True)

            # Message and scroll button in the right section
            with col_content:
                col_msg, col_scroll = st.columns([8.5, 1.5])

                with col_msg:
                    # Create unique ID for copy button
                    import hashlib
                    user_msg_id = f"{key_prefix}user_{hashlib.md5(safe_message.encode()).hexdigest()[:8]}"

                    st.markdown(f"""
<style>
.user-message-{user_msg_id} {{
    position: relative;
    padding-bottom: 45px;
}}
.user-copy-btn-{user_msg_id} {{
    position: absolute;
    bottom: 8px;
    right: 12px;
    background-color: #2d2d2d !important;
    color: #e0e0e0 !important;
    border: 1px solid #FFD700 !important;
    padding: 6px 12px;
    border-radius: 5px;
    font-size: 14px;
    cursor: pointer;
    transition: all 0.3s ease !important;
}}
</style>
<div class="message-container user-message-{user_msg_id}" id="user-msg-{user_msg_id}" style="background-color: #1a1a1a; padding: 15px; border-radius: 10px; margin: 0; border: 2px solid #FFD700;">
    <div style="color: #888; font-size: 19px; margin-bottom: 5px;">
        <strong style="color: #FFD700;">User</strong>
    </div>
    {f'<div style="margin-top: 10px; margin-bottom: 10px; display: flex; align-items: center; flex-wrap: wrap;">{metrics_html}</div>' if metrics_html else ''}
    <div style="color: #fff; font-size: 19px; white-space: pre-wrap;">
        {safe_message}
    </div>
    <button class="user-copy-btn-{user_msg_id}" onclick="copyUserMessage{user_msg_id}()">📋 Copy</button>
</div>
<script>
function copyUserMessage{user_msg_id}() {{
    const text = document.getElementById('user-msg-{user_msg_id}').innerText.replace('📋 Copy', '').replace('User', '').replace('Fidelity:', '').replace('Primacy Attractor Status:', '').replace('ΔF:', '').trim();
    navigator.clipboard.writeText(text).then(() => {{
        const btn = event.target;
        btn.textContent = '✓ Copied!';
        setTimeout(() => {{ btn.textContent = '📋 Copy'; }}, 2000);
    }});
}}
</script>
""", unsafe_allow_html=True)

                # Render Steward button above scroll button
                if turn_number is not None:
                    with col_scroll:
                        if st.button("🤝", key=f"{key_prefix}steward_btn_{turn_number}", use_container_width=True, help="Ask Steward"):
                            # Toggle steward panel
                            st.session_state.steward_panel_open = not st.session_state.get('steward_panel_open', False)
                            st.rerun()

                # Only render scroll button if we're showing the current turn (not in history mode)
                if turn_number is not None and not self.state_manager.state.scrollable_history_mode:
                    with col_scroll:
                        scroll_label = "📜"
                        if st.button(scroll_label, key=f"{key_prefix}scroll_toggle_{turn_number}", use_container_width=True, help="Show scrollable history"):
                            self.state_manager.toggle_scrollable_history()
                            st.rerun()
                elif turn_number is not None and self.state_manager.state.scrollable_history_mode:
                    with col_scroll:
                        scroll_label = "✕"
                        if st.button(scroll_label, key=f"{key_prefix}scroll_close_{turn_number}", use_container_width=True, help="Close scrollable history"):
                            self.state_manager.toggle_scrollable_history()
                            st.rerun()

    def _render_assistant_message(self, message: str, turn_number: int = None, is_loading: bool = False, key_prefix: str = ""):
        """Render steward message bubble - aligned with User message."""
        import html

        # Check if Demo Mode (affects layout)
        demo_mode = st.session_state.get('telos_demo_mode', False)

        # Check if BETA mode - apply centered Observatory layout
        active_tab = st.session_state.get('active_tab', 'DEMO')
        beta_mode = active_tab == "BETA"

        if demo_mode:
            # Demo Mode: Simple clean layout
            if is_loading:
                # Show contemplative pulsing animation
                # Border pulses: gray → yellow
                # "Contemplating..." text pulses: yellow → gray (opposite of border)
                st.markdown(f"""
<style>
@keyframes border-pulse {{
    0%, 100% {{
        border-color: #888;
        box-shadow: 0 0 6px rgba(136, 136, 136, 0.3);
    }}
    50% {{
        border-color: #FFD700;
        box-shadow: 0 0 6px rgba(255, 215, 0, 0.4);
    }}
}}
@keyframes text-pulse {{
    0%, 100% {{
        color: #FFD700;
    }}
    50% {{
        color: #888;
    }}
}}
.contemplating-border {{
    animation: border-pulse 2s ease-in-out infinite;
}}
.contemplating-text {{
    animation: text-pulse 2s ease-in-out infinite;
}}
</style>
<div class="contemplating-border" style="background-color: #1a1a1a; padding: 15px; border-radius: 10px; margin-top: 15px; margin-bottom: 0; border: 2px solid #888;">
    <div style="color: #888; font-size: 19px; margin-bottom: 5px;">
        <strong style="color: #FFD700;">TELOS</strong>
    </div>
    <div class="contemplating-text" style="font-size: 19px; font-style: italic; opacity: 0.9;">
        Contemplating...
    </div>
</div>
""", unsafe_allow_html=True)
            else:
                # Show response - convert markdown to HTML for proper rendering
                html_message = self._markdown_to_html(message)

                # Create unique ID for this message (with prefix to avoid duplicates in history)
                import hashlib
                message_id = f"{key_prefix}{hashlib.md5(message.encode()).hexdigest()[:8]}"

                st.markdown(f"""
<style>
.demo-message-{message_id} {{
    position: relative;
    padding-bottom: 40px;
}}
.demo-copy-btn-{message_id} {{
    position: absolute;
    bottom: 8px;
    right: 12px;
    background-color: #2d2d2d !important;
    color: #e0e0e0 !important;
    border: 1px solid #FFD700 !important;
    padding: 6px 12px;
    border-radius: 5px;
    font-size: 14px;
    cursor: pointer;
    transition: all 0.3s ease !important;
}}
</style>
<div class="message-container demo-message-{message_id}" id="demo-msg-{message_id}" style="background-color: #1a1a1a; padding: 15px; border-radius: 10px; margin-top: 15px; margin-bottom: 15px; border: 2px solid #FFD700;">
    <div style="color: #888; font-size: 19px; margin-bottom: 10px;">
        <strong style="color: #FFD700;">TELOS</strong>
    </div>
    <div style="color: #fff; font-size: 19px; white-space: pre-wrap;">
        {html_message}
    </div>
    <button class="demo-copy-btn-{message_id}" onclick="copyDemoMessage{message_id}()">📋 Copy</button>
</div>
<script>
function copyDemoMessage{message_id}() {{
    const text = document.getElementById('demo-msg-{message_id}').innerText.replace('📋 Copy', '').replace('TELOS', '').trim();
    navigator.clipboard.writeText(text).then(() => {{
        const btn = event.target;
        btn.textContent = '✓ Copied!';
        setTimeout(() => {{ btn.textContent = '📋 Copy'; }}, 2000);
    }});
}}
</script>
""", unsafe_allow_html=True)
        elif beta_mode:
            # BETA Mode: Centered Observatory layout with all features
            # Create columns: [spacer_left 1.5, turn_badge_spacer 0.5, content 6.0, buttons 1.0, spacer_right 1.0]
            col_spacer_left, col_turn_spacer, col_content, col_buttons, col_spacer_right = st.columns([1.5, 0.5, 6.0, 1.0, 1.0])

            # Empty spacer for turn badge alignment (matches user message structure)
            with col_turn_spacer:
                st.markdown("")

            # Message content in center column
            with col_content:
                if is_loading:
                    # Show contemplative pulsing animation
                    st.markdown(f"""
<style>
@keyframes border-pulse {{
    0%, 100% {{
        border-color: #888;
        box-shadow: 0 0 6px rgba(136, 136, 136, 0.3);
    }}
    50% {{
        border-color: #FFD700;
        box-shadow: 0 0 6px rgba(255, 215, 0, 0.4);
    }}
}}
@keyframes text-pulse {{
    0%, 100% {{
        color: #FFD700;
    }}
    50% {{
        color: #888;
    }}
}}
.contemplating-border {{
    animation: border-pulse 2s ease-in-out infinite;
}}
.contemplating-text {{
    animation: text-pulse 2s ease-in-out infinite;
}}
</style>
<div class="contemplating-border" style="background-color: #1a1a1a; padding: 15px; border-radius: 10px; margin-top: 15px; margin-bottom: 0; border: 2px solid #888;">
    <div style="color: #888; font-size: 19px; margin-bottom: 5px;">
        <strong style="color: #FFD700;">TELOS</strong>
    </div>
    <div class="contemplating-text" style="font-size: 19px; font-style: italic; opacity: 0.9;">
        Contemplating...
    </div>
</div>
""", unsafe_allow_html=True)
                else:
                    # Show response with native markdown rendering
                    # Header with "TELOS" label
                    st.markdown("""
<div style="background-color: #1a1a1a; padding: 15px 15px 5px 15px; border-radius: 10px 10px 0 0; margin-top: 15px; margin-bottom: 0; border: 2px solid #FFD700; border-bottom: none;">
    <div style="color: #888; font-size: 19px;">
        <strong style="color: #FFD700;">TELOS</strong>
    </div>
</div>
""", unsafe_allow_html=True)

                    # Render message content inside styled container
                    # Convert markdown to HTML for proper rendering inside the div
                    html_message = self._markdown_to_html(message)

                    # Create unique ID for this message (with prefix to avoid duplicates in history)
                    import hashlib
                    message_id = f"{key_prefix}{hashlib.md5(message.encode()).hexdigest()[:8]}"

                    st.markdown(f"""
<style>
.steward-message-{message_id} {{
    background-color: #1a1a1a;
    padding: 10px 15px 40px 15px;
    margin-top: 0;
    margin-bottom: 0;
    border: 2px solid #FFD700;
    border-top: none;
    border-radius: 0 0 10px 10px;
    color: #fff;
    font-size: 19px;
    position: relative;
}}
.steward-message-{message_id} p {{
    color: #fff !important;
    font-size: 19px !important;
    margin: 0;
}}
.copy-btn-{message_id} {{
    position: absolute;
    bottom: 8px;
    right: 12px;
    background-color: #2d2d2d !important;
    color: #e0e0e0 !important;
    border: 1px solid #FFD700 !important;
    padding: 6px 12px;
    border-radius: 5px;
    font-size: 14px;
    cursor: pointer;
    transition: all 0.3s ease !important;
}}
</style>
<div class="steward-message-{message_id}" id="msg-{message_id}">
    {html_message}
    <button class="copy-btn-{message_id}" onclick="copyMessage{message_id}()">📋 Copy</button>
</div>
<script>
function copyMessage{message_id}() {{
    const text = document.getElementById('msg-{message_id}').innerText.replace('📋 Copy', '').trim();
    navigator.clipboard.writeText(text).then(() => {{
        const btn = event.target;
        btn.textContent = '✓ Copied!';
        setTimeout(() => {{ btn.textContent = '📋 Copy'; }}, 2000);
    }});
}}
</script>
""", unsafe_allow_html=True)

            # Empty column for buttons alignment (no buttons on assistant messages)
            with col_buttons:
                st.markdown("")

        else:
            # Open Mode: Match User message structure with turn badge spacer
            col_spacer, col_content = st.columns([0.5, 9.5])

            with col_spacer:
                # Empty column to align with Turn badge space from User message
                st.markdown("")

            with col_content:
                col_msg, col_empty = st.columns([8.5, 1.5])

                with col_msg:
                    if is_loading:
                        # Show contemplative pulsing animation
                        # Border pulses: gray → yellow
                        # "Contemplating..." text pulses: yellow → gray (opposite of border)
                        st.markdown(f"""
<style>
@keyframes border-pulse {{
    0%, 100% {{
        border-color: #888;
        box-shadow: 0 0 6px rgba(136, 136, 136, 0.3);
    }}
    50% {{
        border-color: #FFD700;
        box-shadow: 0 0 6px rgba(255, 215, 0, 0.4);
    }}
}}
@keyframes text-pulse {{
    0%, 100% {{
        color: #FFD700;
    }}
    50% {{
        color: #888;
    }}
}}
.contemplating-border {{
    animation: border-pulse 2s ease-in-out infinite;
}}
.contemplating-text {{
    animation: text-pulse 2s ease-in-out infinite;
}}
</style>
<div class="contemplating-border" style="background-color: #1a1a1a; padding: 15px; border-radius: 10px; margin-top: 15px; margin-bottom: 0; border: 2px solid #888;">
    <div style="color: #888; font-size: 19px; margin-bottom: 5px;">
        <strong style="color: #FFD700;">TELOS</strong>
    </div>
    <div class="contemplating-text" style="font-size: 19px; font-style: italic; opacity: 0.9;">
        Contemplating...
    </div>
</div>
""", unsafe_allow_html=True)
                    else:
                        # Show response with native markdown rendering
                        # Header with "Steward" label
                        st.markdown("""
<div style="background-color: #1a1a1a; padding: 15px 15px 5px 15px; border-radius: 10px 10px 0 0; margin-top: 15px; margin-bottom: 0; border: 2px solid #FFD700; border-bottom: none;">
    <div style="color: #888; font-size: 19px;">
        <strong style="color: #FFD700;">TELOS</strong>
    </div>
</div>
""", unsafe_allow_html=True)

                        # Render message content inside styled container
                        # Convert markdown to HTML for proper rendering inside the div
                        html_message = self._markdown_to_html(message)

                        # Create unique ID for this message (with prefix to avoid duplicates in history)
                        import hashlib
                        message_id = f"{key_prefix}{hashlib.md5(message.encode()).hexdigest()[:8]}"

                        st.markdown(f"""
<style>
.steward-message-{message_id} {{
    background-color: #1a1a1a;
    padding: 10px 15px 40px 15px;
    margin-top: 0;
    margin-bottom: 0;
    border: 2px solid #FFD700;
    border-top: none;
    border-radius: 0 0 10px 10px;
    color: #fff;
    font-size: 19px;
    position: relative;
}}
.steward-message-{message_id} p {{
    color: #fff !important;
    font-size: 19px !important;
    margin: 0;
}}
.copy-btn-{message_id} {{
    position: absolute;
    bottom: 8px;
    right: 12px;
    background-color: #2d2d2d !important;
    color: #e0e0e0 !important;
    border: 1px solid #FFD700 !important;
    padding: 6px 12px;
    border-radius: 5px;
    font-size: 14px;
    cursor: pointer;
    transition: all 0.3s ease !important;
}}
</style>
<div class="steward-message-{message_id}" id="msg-{message_id}">
    {html_message}
    <button class="copy-btn-{message_id}" onclick="copyMessage{message_id}()">📋 Copy</button>
</div>
<script>
function copyMessage{message_id}() {{
    const text = document.getElementById('msg-{message_id}').innerText.replace('📋 Copy', '').trim();
    navigator.clipboard.writeText(text).then(() => {{
        const btn = event.target;
        btn.textContent = '✓ Copied!';
        setTimeout(() => {{ btn.textContent = '📋 Copy'; }}, 2000);
    }});
}}
</script>
""", unsafe_allow_html=True)

                with col_empty:
                    # Empty column for alignment
                    st.markdown("")

    def _process_streaming_turn(self, user_message: str, turn_idx: int):
        """Process a streaming turn by generating the response and updating state."""
        # Collect the full response from the stream
        full_response = ""
        try:
            for chunk in self.state_manager.generate_response_stream(user_message, turn_idx):
                full_response += chunk
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            full_response = "I apologize, but I encountered an error generating a response. Please try again."

        # After streaming completes, the state is already updated by generate_response_stream
        # Just trigger a rerun to show the completed response
        st.rerun()

    def _render_math_breakdown_window(self, turn_data: Dict[str, Any]):
        """Render Math Breakdown analysis window with metrics and chat."""
        # Header with close button
        col1, col2 = st.columns([9.5, 0.5])
        with col1:
            st.markdown("""
            <div style="
                background-color: #2d2d2d;
                border: 1px solid #FFD700;
                border-radius: 10px;
                padding: 19px;
                margin-bottom: 15px;
            ">
                <div style="color: #FFD700; font-size: 19px; font-weight: bold; text-align: center;">
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
                border: 1px solid #FFD700;
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
                <div style="margin-top: 19px; padding: 10px; background-color: #2d2d2d; border-radius: 5px;">
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
                border: 1px solid #FFD700;
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
                <div style="margin-top: 19px; padding: 10px; background-color: #2d2d2d; border-radius: 5px;">
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
                border: 1px solid #FFD700;
                border-radius: 10px;
                padding: 19px;
                margin-bottom: 15px;
            ">
                <div style="color: #FFD700; font-size: 19px; font-weight: bold; text-align: center;">
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
                border: 1px solid #FFD700;
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
                <div style="margin-top: 15px; padding: 10px; background-color: #2d2d2d; border-radius: 5px; border-left: 1px solid #FFD700;">
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
                border: 1px solid #FFD700;
                border-radius: 10px;
                padding: 19px;
                margin-bottom: 15px;
            ">
                <div style="color: #FFD700; font-size: 19px; font-weight: bold; text-align: center;">
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

        # Check if PA is established (convergence status from progressive extractor)
        pa_converged = getattr(self.state_manager.state, 'pa_converged', False)
        total_turns = self.state_manager.state.total_turns

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

        # If PA not converged yet, show calibrating message
        if not pa_converged and total_turns < 10:
            st.markdown("""
            <div style="
                background-color: #1a1a1a;
                border: 2px solid #FFD700;
                border-radius: 10px;
                padding: 30px;
                margin-top: 15px;
                text-align: center;
            ">
                <span style="color: #FFD700; font-weight: bold; font-size: 18px;">⏳ Calibrating...</span>
                <p style="color: #e0e0e0; margin-top: 15px;">
                    TELOS is learning your intent from the conversation.<br>
                    Primacy Attractor will be established after ~10 turns.
                </p>
                <p style="color: #888; font-size: 14px; margin-top: 10px;">
                    Turn {}/~10
                </p>
            </div>
            """.format(total_turns), unsafe_allow_html=True)
            return  # Don't show PA components yet

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
                border: 1px solid #FFD700;
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
                border: 1px solid #FFD700;
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
                border: 1px solid #FFD700;
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

            # Generate Steward response using Mistral API
            try:
                import os
                from telos.llm.mistral_client import MistralClient

                # Get API key from Streamlit secrets or environment
                mistral_api_key = st.secrets.get("MISTRAL_API_KEY", os.getenv("MISTRAL_API_KEY"))

                if not mistral_api_key:
                    st.error("⚠️ Mistral API key not configured")
                    return

                # Initialize Mistral client
                mistral_client = MistralClient(
                    api_key=mistral_api_key,
                    model="mistral-small-latest"  # Using small - best availability
                )

                # Build Steward system prompt (same as Demo Mode)
                system_prompt = """You are Steward, the TELOS research assistant. Your role is to help users understand their TELOS governance session data.

Key responsibilities:
- Explain what happened in specific turns
- Clarify fidelity scores and metrics
- Describe interventions and why they occurred
- Help users understand drift patterns
- Keep explanations clear and concise (2-3 paragraphs max)
- Avoid deep technical jargon unless asked
- Focus on actionable insights

You have access to the current turn's data. Use it to provide grounded, specific answers."""

                # Add current turn context if available
                turn_context = ""
                if turn_data:
                    turn_context = f"""

Current Turn Data:
- Turn: {turn_data.get('turn', 'N/A')}
- Fidelity: {turn_data.get('fidelity', 'N/A')}
- Status: {turn_data.get('status_text', 'N/A')}
- Intervention: {'Yes' if turn_data.get('intervention_applied') else 'No'}
- User Input: {turn_data.get('user_input', 'N/A')[:100]}..."""

                # Build messages for API call
                steward_messages = [
                    {"role": "system", "content": system_prompt + turn_context}
                ] + st.session_state[f'chat_history_{window_type}']

                # Call Mistral API
                ai_response = mistral_client.generate(
                    messages=steward_messages,
                    max_tokens=300,  # Keep responses concise
                    temperature=0.7
                )

            except Exception as e:
                logger.error(f"Steward AI chat error: {e}")
                ai_response = f"I'm having trouble connecting right now. Could you try rephrasing your question? (Error: {type(e).__name__})"

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
            border: 1px solid #FFD700;
            border-radius: 8px;
            padding: 10px;
            text-align: center;
        ">
            <div style="font-size: 19px; margin-bottom: 5px;">{icon}</div>
            <div style="color: #FFD700; font-size: 10px; font-weight: bold; margin-bottom: 5px;">
                {title}
            </div>
            <div style="
                font-size: 19px;
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
        """Render input area with send button - clean, simple implementation."""
        # Simple CSS for clean alignment
        st.markdown("""
        <style>
        /* Clean input styling */
        div[data-testid="stForm"] {
            background: transparent !important;
        }

        div[data-testid="stForm"] input[type="text"] {
            font-size: 19px !important;
            background-color: #2d2d2d !important;
            color: #ffffff !important;
            border: 1px solid #FFD700 !important;
        }

        div[data-testid="stForm"] button[kind="formSubmit"] {
            font-size: 19px !important;
            font-weight: bold !important;
        }

        /* Hide Streamlit's "Press Enter to submit form" message */
        div[data-testid="stForm"] small {
            display: none !important;
            visibility: hidden !important;
            height: 0 !important;
            overflow: hidden !important;
        }

        form small {
            display: none !important;
            visibility: hidden !important;
        }

        /* Hide any element containing form submission text */
        small:contains("Enter") {
            display: none !important;
        }
        </style>
        """, unsafe_allow_html=True)

        # JavaScript to completely remove the form submission message
        st.components.v1.html("""
        <script>
        // Remove Streamlit's form submission message
        function hideFormMessage() {
            const allSmalls = document.querySelectorAll('small');
            allSmalls.forEach(small => {
                if (small.textContent && small.textContent.toLowerCase().includes('enter')) {
                    small.style.display = 'none';
                    small.style.visibility = 'hidden';
                    small.style.height = '0';
                    small.style.overflow = 'hidden';
                    small.remove();
                }
            });
        }

        // Run immediately
        hideFormMessage();

        // Run after a delay to catch dynamically added elements
        setTimeout(hideFormMessage, 100);
        setTimeout(hideFormMessage, 500);

        // Observe DOM changes and remove message when it appears
        const observer = new MutationObserver(hideFormMessage);
        observer.observe(document.body, { childList: true, subtree: true });
        </script>
        """, height=0)

        # Style the chat input with border (matching Demo mode style)
        st.markdown("""
        <style>
        /* Beta mode chat input styling - matches Demo mode */
        div[data-testid="stForm"] {
            border: 2px solid #FFD700 !important;
            border-radius: 10px !important;
            padding: 15px !important;
            background-color: #2d2d2d !important;
        }

        div[data-testid="stForm"] textarea {
            font-size: 18px !important;
            text-align: center !important;
            background-color: #1a1a1a !important;
            border: 1px solid #666 !important;
            color: #e0e0e0 !important;
        }

        div[data-testid="stForm"] textarea::placeholder {
            text-align: center !important;
            color: #888 !important;
        }
        </style>
        """, unsafe_allow_html=True)

        # Enable Enter to submit (Shift+Enter already works for new line)
        st.components.v1.html("""
        <script>
        (function() {
            function setupEnterKey() {
                const textareas = window.parent.document.querySelectorAll('textarea');
                textareas.forEach(textarea => {
                    textarea.removeEventListener('keydown', handleEnterKey);
                    textarea.addEventListener('keydown', handleEnterKey);
                });
            }

            function handleEnterKey(event) {
                if (event.key === 'Enter' && !event.shiftKey) {
                    event.preventDefault();
                    const sendButton = window.parent.document.querySelector('button[kind="formSubmit"]');
                    if (sendButton) {
                        sendButton.click();
                    }
                }
            }

            setupEnterKey();
            setTimeout(setupEnterKey, 500);
        })();
        </script>
        """, height=0)

        # Check if BETA mode - apply centered layout
        active_tab = st.session_state.get('active_tab', 'DEMO')
        beta_mode = active_tab == "BETA"

        if beta_mode:
            # BETA mode: Center the input form to match conversation layout
            # Use same centering as messages: [1.5, 0.5, 6.0, 1.0, 1.0] = spacers 3.0, content 7.0
            col_spacer_left, col_form, col_spacer_right = st.columns([3.0, 7.0, 0.0])

            with col_form:
                # Use a form to enable Enter key submission
                with st.form(key="message_form", clear_on_submit=True):
                    col1, col2 = st.columns([8.5, 1.5])

                    with col1:
                        user_input = st.text_area(
                            "Message",
                            placeholder="Tell TELOS",
                            key="main_chat_input_clean",
                            label_visibility="collapsed",
                            height=100
                        )

                    with col2:
                        send_button = st.form_submit_button(
                            "Send",
                            use_container_width=True
                        )
        else:
            # Demo/Open mode: Full-width input form
            # Use a form to enable Enter key submission
            with st.form(key="message_form", clear_on_submit=True):
                col1, col2 = st.columns([8.5, 1.5])

                with col1:
                    user_input = st.text_area(
                        "Message",
                        placeholder="Tell TELOS",
                        key="main_chat_input_clean",
                        label_visibility="collapsed",
                        height=100
                    )

                with col2:
                    send_button = st.form_submit_button(
                        "Send",
                        use_container_width=True
                    )

        # Check if Demo Mode and enforce 5-message limit
        demo_mode = st.session_state.get('telos_demo_mode', False)

        # Initialize demo message counter
        if 'demo_message_count' not in st.session_state:
            st.session_state.demo_message_count = 0

        # Show warning at message 4 in Demo Mode
        if demo_mode and st.session_state.demo_message_count == 4:
            st.warning("⚠️ One message remaining in Demo Mode. Add your API key to continue unlimited usage.")

        # Block at message 5 in Demo Mode
        if demo_mode and st.session_state.demo_message_count >= 5:
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, #2d2d2d 0%, #1a1a1a 100%);
                border: 1px solid #FFD700;
                border-radius: 15px;
                padding: 25px;
                margin: 19px 0;
                box-shadow: 0 0 6px rgba(255, 215, 0, 0.3);
                text-align: center;
            ">
                <div style="font-size: 48px; margin-bottom: 15px;">🔒</div>
                <div style="color: #FFD700; font-size: 24px; font-weight: bold; margin-bottom: 15px;">
                    Demo Limit Reached
                </div>
                <div style="color: #e0e0e0; font-size: 19px; line-height: 1.6; margin-bottom: 19px;">
                    You've reached the 5-message demo limit.<br>
                    Add your Anthropic API key to continue with unlimited TELOS governance.
                </div>
                <div style="color: #888; font-size: 16px; margin-top: 19px;">
                    Click "Exit Demo Mode" in the sidebar to add your API key →
                </div>
            </div>
            """, unsafe_allow_html=True)
            return  # Stop rendering input form

        # Handle sending message
        if send_button and user_input and user_input.strip():
            # Dismiss intro if showing
            if 'show_intro' in st.session_state and st.session_state.show_intro:
                st.session_state.show_intro = False

            # Increment demo message counter in Demo Mode
            if demo_mode:
                st.session_state.demo_message_count += 1

            # CRITICAL: Clear demo data on first user message to prevent contamination
            # Demo data should NOT be part of primacy attractor calibration
            # ONLY clear demo data if in Demo Mode - Beta mode starts fresh already
            if 'user_started_conversation' not in st.session_state and demo_mode:
                # This is the user's first message in DEMO mode - clear ALL demo data
                self.state_manager.clear_demo_data()
                st.session_state.user_started_conversation = True

            # Add the user message and get turn index
            turn_idx = self.state_manager.add_user_message_streaming(user_input.strip())

            # Show user message immediately
            st.rerun()

    def _render_chat_input(self):
        """Render chat input box for new messages."""
        # Chat input with custom styling
        user_input = st.chat_input(
            "Tell TELOS",
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
                border: 1px solid #FFD700;
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

    def _show_beta_phase_transition(self, turn_number: int):
        """Show phase transition message when PA calibration completes at turn 11."""
        if turn_number != 11:
            return

        if not st.session_state.get('beta_consent_given', False):
            return

        if st.session_state.get('beta_phase_transition_shown', False):
            return

        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
            border: 2px solid #FFD700;
            border-radius: 12px;
            padding: 25px;
            margin: 20px 0;
            text-align: center;
            box-shadow: 0 0 20px rgba(255, 215, 0, 0.2);
        ">
            <div style="font-size: 48px; margin-bottom: 10px;">🎯</div>
            <h3 style="color: #FFD700; margin: 10px 0;">PA Established!</h3>
            <p style="color: #e0e0e0; font-size: 18px; line-height: 1.6; margin: 10px 0;">
                Your conversation purpose is now calibrated.<br>
                Beta preference testing is active - please rate responses below.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.session_state.beta_phase_transition_shown = True

    def _render_beta_feedback(self, turn_number: int):
        """Render simple beta feedback UI (thumbs up/down/sideways) for turns 11+."""
        if turn_number < 11:
            return

        if not st.session_state.get('beta_consent_given', False):
            return

        feedback_key = f"beta_feedback_{turn_number}"
        if st.session_state.get(feedback_key):
            st.markdown("""
            <div style="color: #4CAF50; font-size: 14px; margin: 10px 0;">
                ✓ Thank you for your feedback!
            </div>
            """, unsafe_allow_html=True)
            return

        st.markdown("<div style='margin: 15px 0;'></div>", unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns([1, 1, 1, 5])

        with col1:
            if st.button("👍", key=f"thumbs_up_{turn_number}",
                        use_container_width=True, help="Helpful response"):
                self._record_simple_feedback(turn_number, "thumbs_up")
                st.session_state[feedback_key] = True
                st.rerun()

        with col2:
            if st.button("👉", key=f"sideways_{turn_number}",
                        use_container_width=True, help="Neutral / No preference"):
                self._record_simple_feedback(turn_number, "sideways")
                st.session_state[feedback_key] = True
                st.rerun()

        with col3:
            if st.button("👎", key=f"thumbs_down_{turn_number}",
                        use_container_width=True, help="Not helpful"):
                self._record_simple_feedback(turn_number, "thumbs_down")
                st.session_state[feedback_key] = True
                st.rerun()

        with col4:
            st.markdown("""
            <div style="color: #888; font-size: 14px; padding-top: 8px;">
                Rate this response
            </div>
            """, unsafe_allow_html=True)

    def _record_simple_feedback(self, turn_number: int, rating: str):
        """Record simple feedback to session state and beta session manager."""
        from datetime import datetime
        import logging
        logger = logging.getLogger(__name__)

        if 'beta_feedback' not in st.session_state:
            st.session_state.beta_feedback = []

        # Get turn data to extract beta information
        turn_idx = turn_number - 1  # Convert to 0-based index
        turn_data = None
        if hasattr(self.state_manager.state, 'turns') and turn_idx < len(self.state_manager.state.turns):
            turn_data = self.state_manager.state.turns[turn_idx]

        feedback_item = {
            'turn': turn_number,
            'rating': rating,
            'timestamp': datetime.now().isoformat()
        }

        # Add beta-specific data if available
        if turn_data and 'beta_data' in turn_data:
            beta_data = turn_data['beta_data']
            feedback_item['test_condition'] = beta_data.get('test_condition')
            feedback_item['response_source'] = beta_data.get('shown_response_source')
            feedback_item['baseline_fidelity'] = beta_data.get('baseline_fidelity')
            feedback_item['telos_fidelity'] = beta_data.get('telos_fidelity')
            logger.info(f"Beta feedback includes: test_condition={feedback_item['test_condition']}, source={feedback_item['response_source']}")

        st.session_state.beta_feedback.append(feedback_item)

        # Record to beta session manager if available
        beta_session = st.session_state.get('beta_session')
        if beta_session:
            try:
                from observatory.beta_testing.beta_session_manager import FeedbackData

                # DELTA-ONLY: Store lengths, not content
                user_input = turn_data.get('user_input', '') if turn_data else ''
                response = turn_data.get('response', '') if turn_data else ''
                beta_data = turn_data.get('beta_data', {}) if turn_data else {}

                feedback_data = FeedbackData(
                    turn_number=turn_number,
                    timestamp=feedback_item['timestamp'],
                    test_condition=feedback_item.get('test_condition', 'unknown'),
                    rating=rating,
                    response_source=feedback_item.get('response_source'),
                    user_message_length=len(user_input),
                    response_length=len(response),
                    fidelity=feedback_item.get('telos_fidelity'),
                    baseline_fidelity=feedback_item.get('baseline_fidelity'),
                    telos_fidelity=feedback_item.get('telos_fidelity'),
                    fidelity_delta=beta_data.get('fidelity_delta'),
                    drift_detected=beta_data.get('drift_detected', False)
                )

                beta_session.feedback_items.append(feedback_data)
                logger.info(f"✓ Feedback recorded to beta session manager")
            except Exception as e:
                logger.warning(f"Could not record to beta session manager: {e}")

        # Set start time on first feedback
        if len(st.session_state.beta_feedback) == 1:
            st.session_state.beta_start_time = datetime.now().isoformat()

        logger.info(f"Beta feedback: turn {turn_number} = {rating}")
