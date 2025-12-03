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

# Import PA templates for click-to-select
from config.pa_templates import get_all_templates, get_template, get_pa_config, get_onboarding_content


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
        Render the PA onboarding - template selection first, questionnaire as fallback.

        Returns:
            Dict with answers if submitted, None otherwise
        """
        # Check if already completed
        if st.session_state.get('pa_established', False):
            return st.session_state.get('pa_answers')

        # Initialize onboarding step
        if 'pa_onboarding_step' not in st.session_state:
            st.session_state.pa_onboarding_step = 'template_selection'
            st.session_state.selected_template_id = None

        # Route to appropriate step
        if st.session_state.pa_onboarding_step == 'template_selection':
            return self._render_template_selection()
        elif st.session_state.pa_onboarding_step == 'template_onboarding':
            return self._render_template_onboarding()
        elif st.session_state.pa_onboarding_step == 'questionnaire':
            return self._render_questionnaire_form()

        return None

    def _render_template_selection(self) -> Optional[Dict]:
        """Step 1: Show template buttons for quick selection."""
        # Header
        st.markdown(f"""
        <div style="text-align: center; padding: 20px 0;">
            <div style="color: {GOLD}; font-size: 32px; font-weight: bold;">
                What would you like to do?
            </div>
            <div style="color: #e0e0e0; font-size: 18px; margin-top: 10px;">
                Select a conversation type to get started
            </div>
        </div>
        """, unsafe_allow_html=True)

        templates = get_all_templates()

        # Template buttons in a row
        cols = st.columns(len(templates))
        for idx, template in enumerate(templates):
            with cols[idx]:
                if st.button(template['name'], key=f"template_{template['id']}", use_container_width=True):
                    st.session_state.selected_template_id = template['id']
                    st.session_state.pa_onboarding_step = 'template_onboarding'
                    st.rerun()
                st.markdown(f"""
                <p style="color: #888; font-size: 12px; text-align: center; margin-top: 5px;">
                    {template['short_description']}
                </p>
                """, unsafe_allow_html=True)

        # Custom purpose option
        with st.expander("Or describe your own purpose"):
            st.markdown("If none of the above fit, you can describe what you want to accomplish.")
            if st.button("Write Custom Purpose", use_container_width=True):
                st.session_state.pa_onboarding_step = 'questionnaire'
                st.rerun()

        return None

    def _render_template_onboarding(self) -> Optional[Dict]:
        """Step 2: Show brief onboarding for selected template."""
        template_id = st.session_state.selected_template_id
        template = get_template(template_id)
        onboarding = get_onboarding_content(template_id)

        # Header
        st.markdown(f"""
        <div style="text-align: center; padding: 20px 0;">
            <div style="color: {GOLD}; font-size: 32px; font-weight: bold;">
                {template['name']}
            </div>
            <div style="color: #e0e0e0; font-size: 18px; margin-top: 10px;">
                {onboarding['headline']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Onboarding content
        st.markdown(f"**{onboarding['description']}**")

        st.markdown("**Great for:**")
        for use_case in onboarding['use_cases']:
            st.markdown(f"- {use_case}")

        st.markdown("**Example prompts:**")
        for prompt in onboarding['example_prompts']:
            st.markdown(f"- *\"{prompt}\"*")

        st.info(f"**How TELOS helps:** {onboarding['governance_note']}")

        # Action buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Start Session", use_container_width=True, type="primary"):
                # Use template PA directly
                pa_config = get_pa_config(template_id)
                self._establish_from_template(template_id, pa_config)
                return pa_config
        with col2:
            if st.button("Choose Different Type", use_container_width=True):
                st.session_state.selected_template_id = None
                st.session_state.pa_onboarding_step = 'template_selection'
                st.rerun()

        return None

    def _establish_from_template(self, template_id: str, pa_config: Dict):
        """Establish PA from a template selection."""
        # Structure the PA
        pa = {
            "purpose": pa_config.get('purpose', ['General assistance'])[0],
            "scope": "; ".join(pa_config.get('scope', ['Open discussion'])),
            "boundaries": pa_config.get('boundaries', []),
            "success_criteria": f"Successful {template_id} session",
            "style": "Adaptive",
            "established_turn": 1,
            "establishment_method": f"template_{template_id}"
        }

        # Store in session state
        st.session_state.primacy_attractor = pa
        st.session_state.pa_established = True
        st.session_state.pa_establishment_time = datetime.now().isoformat()
        st.session_state.pa_answers = pa_config  # Store template config

        # Initialize BETA sequence for 5-turn testing
        from services.beta_sequence_generator import BetaSequenceGenerator
        generator = BetaSequenceGenerator()
        st.session_state.beta_sequence = generator.generate_session_sequence()
        st.session_state.beta_current_turn = 1

        # Update state manager
        if 'state_manager' in st.session_state:
            st.session_state.state_manager.state.primacy_attractor = pa
            st.session_state.state_manager.state.user_pa_established = True
            st.session_state.state_manager.state.convergence_turn = 1
            st.session_state.state_manager.state.pa_converged = True

            # Force TELOS steward to re-initialize
            if hasattr(st.session_state.state_manager, '_telos_steward'):
                delattr(st.session_state.state_manager, '_telos_steward')

            if 'beta_response_manager' in st.session_state:
                if hasattr(st.session_state.beta_response_manager, 'telos_engine'):
                    st.session_state.beta_response_manager.telos_engine = None

        st.success(f"Purpose established: {template_id.title()}")
        st.rerun()

    def _render_questionnaire_form(self) -> Optional[Dict]:
        """Render the original questionnaire form for custom purposes."""
        # Header with explanation
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #2d2d2d 0%, #1a1a1a 100%);
            border: 3px solid {GOLD};
            border-radius: 15px;
            padding: 30px;
            margin: 20px 0;
            box-shadow: 0 0 20px {GOLD_80};
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

        # Back button
        if st.button("Back to Templates"):
            st.session_state.pa_onboarding_step = 'template_selection'
            st.rerun()

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
                    background-color: #1a1a1a;
                    border: 2px solid {GOLD};
                    border-radius: 10px;
                    padding: 20px;
                    margin: 15px 0;
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

                                # Initialize BETA sequence for 15-turn A/B testing
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

            # ALSO delete BETA response manager's telos_engine
            if 'beta_response_manager' in st.session_state:
                if hasattr(st.session_state.beta_response_manager, 'telos_engine'):
                    st.session_state.beta_response_manager.telos_engine = None

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