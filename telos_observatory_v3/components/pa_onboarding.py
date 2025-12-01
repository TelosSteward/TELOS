"""
PA Onboarding Component for BETA Mode
======================================

Establishes Primacy Attractor through explicit questionnaire
within first 2 turns to ensure clear alignment from the start.
"""

import streamlit as st
from typing import Dict, Optional
from datetime import datetime

# Import color configuration
from config.colors import GOLD, GOLD_80  # GOLD = #F4D03F (refined, easier on eyes)


class PAOnboarding:
    """Handles PA establishment through user questionnaire."""

    def __init__(self):
        """Initialize PA onboarding component."""
        self.questions = [
            {
                "key": "primary_goal",
                "question": "What are you trying to accomplish in this conversation?",
                "placeholder": "Tell TELOS what you want to accomplish...",
                "help": "This helps establish your primary purpose"
            },
            {
                "key": "scope_boundaries",
                "question": "What topics should we focus on? What should we avoid?",
                "placeholder": "Focus on: technical details, practical examples... Avoid: unrelated topics, personal info...",
                "help": "This defines the scope and boundaries"
            },
            {
                "key": "success_criteria",
                "question": "How will you know if this conversation is successful?",
                "placeholder": "e.g., I got clear answers, I learned something new, I have actionable next steps...",
                "help": "This sets measurable outcomes"
            },
            {
                "key": "style_preference",
                "question": "Any communication style preferences?",
                "placeholder": "e.g., Technical/Simple, Brief/Detailed, Examples/Theory...",
                "help": "This helps tailor responses to your needs"
            }
        ]

    def render_questionnaire(self) -> Optional[Dict]:
        """
        Render the PA onboarding questionnaire.

        Returns:
            Dict with answers if submitted, None otherwise
        """
        # Check if already completed
        if st.session_state.get('pa_established', False):
            return st.session_state.get('pa_answers')

        # Scroll to top of page when PA onboarding is shown
        import streamlit.components.v1 as components
        components.html("""
        <script>
        // Scroll the parent window to the top
        window.parent.scrollTo(0, 0);
        // Also try scrolling the main container
        const mainContainer = window.parent.document.querySelector('[data-testid="stAppViewContainer"]');
        if (mainContainer) {
            mainContainer.scrollTo(0, 0);
        }
        // Scroll any scrollable ancestor
        const scrollables = window.parent.document.querySelectorAll('[data-testid="stVerticalBlock"]');
        scrollables.forEach(el => el.scrollTo(0, 0));
        </script>
        """, height=0)

        # Header with explanation
        st.markdown(f"""
        <div style="
            background: rgba(15, 15, 15, 0.4);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(244, 208, 63, 0.5);
            border-top: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 15px;
            padding: 30px;
            margin: 20px 0;
            box-shadow:
                0 8px 32px rgba(0, 0, 0, 0.3),
                inset 0 1px 0 rgba(255, 255, 255, 0.08);
        ">
            <div style="text-align: center; margin-bottom: 20px;">
                <div style="color: {GOLD}; font-size: 32px; font-weight: bold; margin-bottom: 15px;">
                    Let's Establish Your Purpose
                </div>
                <div style="color: #e0e0e0; font-size: 20px; margin-top: 10px; line-height: 1.6;">
                    TELOS works best when it understands your goals.
                    These quick questions help establish your "Primacy Attractor" -
                    the core purpose that will guide our conversation.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Quick Start button - skip questionnaire with default PA
        st.markdown(f"""
        <div style="text-align: center; margin: 10px 0 20px 0;">
            <span style="color: #888; font-size: 16px;">Already know what you want? </span>
        </div>
        """, unsafe_allow_html=True)

        col_left, col_quick, col_right = st.columns([2, 2, 2])
        with col_quick:
            if st.button("⚡ Quick Start (Use Default PA)", use_container_width=True, key="quick_start_btn"):
                self._apply_default_pa()
                st.rerun()

        st.markdown("<div style='text-align: center; color: #666; margin: 10px 0 30px 0;'>— or answer the questions below —</div>", unsafe_allow_html=True)

        # Initialize question index
        if 'pa_current_question' not in st.session_state:
            st.session_state.pa_current_question = 0

        # Progress indicator
        current_q = st.session_state.get('pa_current_question', 0)
        progress = (current_q / len(self.questions)) if current_q < len(self.questions) else 1.0
        st.progress(progress, text=f"Question {min(current_q + 1, len(self.questions))} of {len(self.questions)}")

        # Questionnaire form
        with st.form(key="pa_onboarding_form"):
            answers = {}

            # Show current question prominently
            if current_q < len(self.questions):
                q = self.questions[current_q]

                st.markdown(f"""
                <div style="
                    background: rgba(15, 15, 15, 0.35);
                    backdrop-filter: blur(10px);
                    -webkit-backdrop-filter: blur(10px);
                    border: 1px solid rgba(244, 208, 63, 0.4);
                    border-top: 1px solid rgba(255, 255, 255, 0.15);
                    border-radius: 10px;
                    padding: 20px;
                    margin: 15px 0;
                    box-shadow:
                        0 4px 20px rgba(0, 0, 0, 0.25),
                        inset 0 1px 0 rgba(255, 255, 255, 0.05);
                ">
                    <div style="color: {GOLD}; font-size: 22px; font-weight: bold; margin-bottom: 10px;">
                        {q['question']}
                    </div>
                    <div style="color: #999; font-size: 16px;">
                        {q['help']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Add CSS to increase text area font size
                st.markdown("""
                <style>
                textarea {
                    font-size: 20px !important;
                    line-height: 1.6 !important;
                }
                </style>
                """, unsafe_allow_html=True)

                # Add JavaScript to submit form on Enter key (Shift+Enter for new line)
                import streamlit.components.v1 as components
                components.html("""
                <script>
                // Wait for DOM to be ready
                setTimeout(function() {
                    // Find the textarea in the parent document
                    const textareas = window.parent.document.querySelectorAll('textarea');
                    textareas.forEach(function(textarea) {
                        // Only add listener once
                        if (!textarea.dataset.enterHandlerAdded) {
                            textarea.dataset.enterHandlerAdded = 'true';
                            textarea.addEventListener('keydown', function(e) {
                                if (e.key === 'Enter' && !e.shiftKey) {
                                    e.preventDefault();
                                    // Find the submit button (Next or Complete)
                                    const buttons = window.parent.document.querySelectorAll('button[kind="secondaryFormSubmit"], button[data-testid="stFormSubmitButton"]');
                                    // Look for Next or Complete button (rightmost button in the form)
                                    for (let btn of buttons) {
                                        const text = btn.textContent || btn.innerText;
                                        if (text.includes('Next') || text.includes('Complete')) {
                                            btn.click();
                                            break;
                                        }
                                    }
                                }
                            });
                        }
                    });
                }, 500);
                </script>
                """, height=0)

                # Input field - Load previous answer if exists
                previous_answer = st.session_state.get('pa_answers', {}).get(q['key'], '')
                answer = st.text_area(
                    "Your answer:",
                    value=previous_answer,  # Pre-populate with saved answer
                    placeholder=q['placeholder'],
                    height=100,
                    key=f"pa_{q['key']}_{current_q}",  # Unique key per question to force refresh
                    label_visibility="collapsed"
                )

                # Store answer
                if answer:
                    answers[q['key']] = answer

                # Navigation buttons
                col1, col2, col3 = st.columns([2, 1, 2])

                with col1:
                    if current_q > 0:
                        if st.form_submit_button("← Previous", use_container_width=True):
                            # Save current answer before moving back
                            if answer:
                                if 'pa_answers' not in st.session_state:
                                    st.session_state.pa_answers = {}
                                st.session_state.pa_answers[q['key']] = answer
                            st.session_state.pa_current_question = max(0, current_q - 1)
                            st.rerun()

                with col2:
                    # Skip button (optional questions)
                    if current_q >= 2:  # Only first 2 are required
                        if st.form_submit_button("Skip", use_container_width=True):
                            # Save current answer even if skipping (if provided)
                            if answer:
                                if 'pa_answers' not in st.session_state:
                                    st.session_state.pa_answers = {}
                                st.session_state.pa_answers[q['key']] = answer
                            st.session_state.pa_current_question = current_q + 1
                            st.rerun()

                with col3:
                    if current_q < len(self.questions) - 1:
                        if st.form_submit_button("Next →", use_container_width=True):
                            if answer:  # Only proceed if answered
                                # Store answer
                                if 'pa_answers' not in st.session_state:
                                    st.session_state.pa_answers = {}
                                st.session_state.pa_answers[q['key']] = answer
                                st.session_state.pa_current_question = current_q + 1
                                st.rerun()
                            else:
                                st.error("Please provide an answer or skip (if available)")
                    else:
                        # Final question - complete button
                        if st.form_submit_button("✅ Complete Setup", use_container_width=True):
                            if answer:
                                if 'pa_answers' not in st.session_state:
                                    st.session_state.pa_answers = {}
                                st.session_state.pa_answers[q['key']] = answer

                                # Mark as complete
                                st.session_state.pa_established = True
                                st.session_state.pa_establishment_time = datetime.now().isoformat()

                                # Extract PA from answers
                                self._extract_pa_from_answers()

                                # Initialize BETA sequence for 5-turn TELOS demo
                                from services.beta_sequence_generator import BetaSequenceGenerator
                                generator = BetaSequenceGenerator()
                                st.session_state.beta_sequence = generator.generate_session_sequence()
                                st.session_state.beta_current_turn = 1  # Start at turn 1

                                st.success("✅ Your Primacy Attractor has been established!")
                                st.rerun()

        # Show summary of previous answers
        if current_q > 0 and 'pa_answers' in st.session_state:
            with st.expander("Your previous answers"):
                for i in range(current_q):
                    if i < len(self.questions):
                        q_key = self.questions[i]['key']
                        if q_key in st.session_state.pa_answers:
                            st.markdown(f"**{self.questions[i]['question']}**")
                            st.markdown(st.session_state.pa_answers[q_key])
                            st.markdown("---")

        return None

    def _extract_pa_from_answers(self):
        """Extract and structure PA from questionnaire answers."""
        answers = st.session_state.get('pa_answers', {})

        # Structure the PA
        pa = {
            "purpose": answers.get('primary_goal', 'General assistance'),
            "scope": answers.get('scope_boundaries', 'Open discussion'),
            "boundaries": self._extract_boundaries(answers.get('scope_boundaries', '')),
            "success_criteria": answers.get('success_criteria', 'Helpful conversation'),
            "style": answers.get('style_preference', 'Adaptive'),
            "established_turn": 2,  # PA established by turn 2
            "establishment_method": "explicit_questionnaire"
        }

        # Store in session state
        st.session_state.primacy_attractor = pa

        # Also store for state manager
        if 'state_manager' in st.session_state:
            st.session_state.state_manager.state.primacy_attractor = pa
            st.session_state.state_manager.state.user_pa_established = True
            st.session_state.state_manager.state.convergence_turn = 2
            st.session_state.state_manager.state.pa_converged = True  # Mark as converged since PA is established

            # Force TELOS steward to re-initialize with new PA
            if hasattr(st.session_state.state_manager, '_telos_steward'):
                delattr(st.session_state.state_manager, '_telos_steward')

            # ALSO delete BETA response manager's telos_engine AND all dual PA components
            # to force full re-initialization with new PA
            if 'beta_response_manager' in st.session_state:
                brm = st.session_state.beta_response_manager
                if hasattr(brm, 'telos_engine'):
                    brm.telos_engine = None
                if hasattr(brm, 'ps_calculator'):
                    brm.ps_calculator = None
                if hasattr(brm, 'user_pa_embedding'):
                    brm.user_pa_embedding = None
                if hasattr(brm, 'ai_pa_embedding'):
                    brm.ai_pa_embedding = None
                if hasattr(brm, 'embedding_provider'):
                    brm.embedding_provider = None

        return pa

    def _extract_boundaries(self, scope_text: str) -> list:
        """Extract boundary keywords from scope text."""
        boundaries = []

        # Look for "avoid" patterns
        if "avoid" in scope_text.lower():
            parts = scope_text.lower().split("avoid")
            if len(parts) > 1:
                avoid_text = parts[1].split(".")[0]
                boundaries = [b.strip() for b in avoid_text.split(",")]

        # If no boundaries extracted, provide sensible defaults
        if not boundaries:
            boundaries = [
                "Stay focused on stated purpose",
                "Avoid unrelated tangents",
                "Maintain productive dialogue"
            ]

        return boundaries

    def _apply_default_pa(self):
        """
        Apply a default TELOS-focused PA for quick start.

        Skips the questionnaire entirely and sets up a sensible default
        PA focused on TELOS exploration and testing.
        """
        # Default answers for TELOS testing
        default_answers = {
            'primary_goal': 'Explore and test TELOS Observatory capabilities, understand AI governance through purposeful conversation',
            'scope_boundaries': 'Focus on: AI governance, TELOS features, meaningful dialogue. Avoid: completely unrelated tangents',
            'success_criteria': 'Productive conversation that demonstrates TELOS principles and stays purposefully aligned',
            'style_preference': 'Natural, conversational, thoughtful'
        }

        # Store answers
        st.session_state.pa_answers = default_answers

        # Structure the PA
        pa = {
            "purpose": default_answers['primary_goal'],
            "scope": default_answers['scope_boundaries'],
            "boundaries": [
                "Stay focused on TELOS exploration",
                "Maintain productive dialogue",
                "Allow natural conversation flow"
            ],
            "success_criteria": default_answers['success_criteria'],
            "style": default_answers['style_preference'],
            "established_turn": 0,  # Instant establishment
            "establishment_method": "quick_start_default"
        }

        # Store in session state
        st.session_state.primacy_attractor = pa
        st.session_state.pa_established = True
        st.session_state.pa_establishment_time = datetime.now().isoformat()

        # Also store for state manager
        if 'state_manager' in st.session_state:
            st.session_state.state_manager.state.primacy_attractor = pa
            st.session_state.state_manager.state.user_pa_established = True
            st.session_state.state_manager.state.convergence_turn = 0
            st.session_state.state_manager.state.pa_converged = True

            # Force TELOS steward to re-initialize with new PA
            if hasattr(st.session_state.state_manager, '_telos_steward'):
                delattr(st.session_state.state_manager, '_telos_steward')

            # ALSO delete BETA response manager's telos_engine AND all dual PA components
            # to force full re-initialization with new PA
            if 'beta_response_manager' in st.session_state:
                brm = st.session_state.beta_response_manager
                if hasattr(brm, 'telos_engine'):
                    brm.telos_engine = None
                if hasattr(brm, 'ps_calculator'):
                    brm.ps_calculator = None
                if hasattr(brm, 'user_pa_embedding'):
                    brm.user_pa_embedding = None
                if hasattr(brm, 'ai_pa_embedding'):
                    brm.ai_pa_embedding = None
                if hasattr(brm, 'embedding_provider'):
                    brm.embedding_provider = None

        # Initialize BETA sequence for 5-turn TELOS demo
        from services.beta_sequence_generator import BetaSequenceGenerator
        generator = BetaSequenceGenerator()
        st.session_state.beta_sequence = generator.generate_session_sequence()
        st.session_state.beta_current_turn = 1  # Start at turn 1

        return pa

    def render_pa_summary(self):
        """Render a summary of the established PA."""
        if not st.session_state.get('pa_established', False):
            return

        pa = st.session_state.get('primacy_attractor', {})

        st.markdown("""
        <div style="
            background-color: #1a1a1a;
            border: 1px solid #4CAF50;
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
        ">
            <div style="color: #4CAF50; font-size: 16px; font-weight: bold; margin-bottom: 10px;">
                ✅ Your Primacy Attractor is Active
            </div>
            <div style="color: #e0e0e0; font-size: 14px;">
                <strong>Purpose:</strong> {purpose}<br>
                <strong>Scope:</strong> {scope}<br>
                <strong>Success:</strong> {success}
            </div>
        </div>
        """.format(
            purpose=pa.get('purpose', 'Not set'),
            scope=pa.get('scope', 'Not set'),
            success=pa.get('success_criteria', 'Not set')
        ), unsafe_allow_html=True)