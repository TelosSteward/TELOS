"""
Beta Steward Panel Component for TELOS Observatory V3 BETA Mode.
Specialized Steward interface for BETA testers with proper context awareness.
"""

import streamlit as st
from datetime import datetime
from services.beta_steward_llm import BetaStewardLLM
import html


class BetaStewardPanel:
    """Side panel with Beta Steward assistant for BETA mode guidance."""

    def __init__(self):
        """Initialize Beta Steward (no state_manager needed - uses session state directly)."""
        # Initialize LLM service (lazy loading)
        if 'beta_steward_llm' not in st.session_state:
            try:
                st.session_state.beta_steward_llm = BetaStewardLLM()
                st.session_state.beta_steward_llm_enabled = True
            except ValueError as e:
                st.session_state.beta_steward_llm = None
                st.session_state.beta_steward_llm_enabled = False
                st.session_state.beta_steward_error = str(e)

    def render_panel(self):
        """Render the Beta Steward chat panel."""
        # Check if opened from an intervention "Why?" button
        intervention_turn = st.session_state.get('steward_intervention_turn')

        # Initialize chat history if not exists
        if 'beta_steward_chat_history' not in st.session_state:
            # Get user's PA for personalized greeting
            pa = st.session_state.get('primacy_attractor', {})
            purpose = pa.get('purpose', 'your stated purpose')

            if intervention_turn:
                # Opened from "Why?" button - explain the specific intervention
                turn_data = st.session_state.get(f'beta_turn_{intervention_turn}_data', {})
                telos_analysis = turn_data.get('telos_analysis', {})

                user_fidelity = telos_analysis.get('user_pa_fidelity', 0.0)
                intervention_reason = telos_analysis.get('intervention_reason', 'low fidelity detected')
                user_input = turn_data.get('user_input', '')[:100]

                initial_message = f"""You asked why TELOS intervened on Turn {intervention_turn}. Let me explain:

**Your input had a fidelity score of {user_fidelity:.3f}**, which indicates your message drifted from your stated purpose.

**Your purpose:** "{purpose[:80]}{'...' if len(purpose) > 80 else ''}"
**Your input:** "{user_input}{'...' if len(user_input) >= 100 else ''}"

**Why the intervention?** {intervention_reason}

When fidelity drops below certain thresholds, TELOS governance kicks in to help steer the conversation back toward your goals. This is the system working as designed - it's protecting your stated intent.

Would you like me to explain more about fidelity scores or how interventions work?"""

                # Clear the intervention turn after using it
                del st.session_state['steward_intervention_turn']
            else:
                # Standard greeting - simple and instantaneous
                initial_message = "Welcome to your BETA test! I'm Steward, here to help you understand what you're seeing."

            st.session_state.beta_steward_chat_history = [
                {
                    'role': 'assistant',
                    'content': initial_message,
                    'timestamp': datetime.now().isoformat()
                }
            ]
        elif intervention_turn:
            # Chat history exists but opened from new "Why?" button
            # Add the intervention explanation as a new conversation
            turn_data = st.session_state.get(f'beta_turn_{intervention_turn}_data', {})
            telos_analysis = turn_data.get('telos_analysis', {})
            pa = st.session_state.get('primacy_attractor', {})
            purpose = pa.get('purpose', 'your stated purpose')

            user_fidelity = telos_analysis.get('user_pa_fidelity', 0.0)
            intervention_reason = telos_analysis.get('intervention_reason', 'low fidelity detected')
            user_input = turn_data.get('user_input', '')[:100]

            intervention_message = f"""You asked about the intervention on Turn {intervention_turn}:

**Fidelity score: {user_fidelity:.3f}** - Your message drifted from your purpose.
**Your input:** "{user_input}{'...' if len(user_input) >= 100 else ''}"
**Reason:** {intervention_reason}

The system intervened to help realign the conversation with your stated goals. Want to know more?"""

            st.session_state.beta_steward_chat_history.append({
                'role': 'assistant',
                'content': intervention_message,
                'timestamp': datetime.now().isoformat()
            })

            # Clear the intervention turn after using it
            del st.session_state['steward_intervention_turn']

        # Close button
        col_spacer, col_close = st.columns([5, 1])
        with col_close:
            if st.button("X", key="close_beta_steward", help="Close Steward", use_container_width=True):
                st.session_state.beta_steward_panel_open = False
                # Clear chat history so next open is fresh
                if 'beta_steward_chat_history' in st.session_state:
                    del st.session_state.beta_steward_chat_history
                st.rerun()

        # Chat history display
        for message in st.session_state.beta_steward_chat_history:
            if message['role'] == 'assistant':
                st.markdown(f"""
                <div style="background-color: rgba(255, 215, 0, 0.1); border: 1px solid #F4D03F; border-radius: 8px; padding: 12px; margin-bottom: 15px;">
                    <strong style="color: #F4D03F; font-size: 18px;">Steward:</strong><br>
                    <span style="color: #F4D03F; font-size: 16px; line-height: 1.6;">{message['content']}</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background-color: rgba(255, 255, 255, 0.05); border: 1px solid #666; border-radius: 8px; padding: 12px; margin-bottom: 15px;">
                    <strong style="color: #e0e0e0; font-size: 18px;">You:</strong><br>
                    <span style="color: #e0e0e0; font-size: 16px; line-height: 1.6;">{html.escape(message['content'])}</span>
                </div>
                """, unsafe_allow_html=True)

        # Input form
        st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)

        with st.form(key="beta_steward_form", clear_on_submit=True):
            user_input = st.text_input(
                "Ask Steward anything...",
                placeholder="What do these metrics mean?",
                label_visibility="collapsed",
                key="beta_steward_input_form"
            )

            submit = st.form_submit_button("Send", use_container_width=True)

            if submit and user_input:
                # Add user message
                st.session_state.beta_steward_chat_history.append({
                    'role': 'user',
                    'content': user_input,
                    'timestamp': datetime.now().isoformat()
                })

                # Get response with BETA context
                if st.session_state.beta_steward_llm_enabled:
                    context = self._gather_beta_context()

                    try:
                        steward_response = st.session_state.beta_steward_llm.get_response(
                            user_message=user_input,
                            conversation_history=st.session_state.beta_steward_chat_history[:-1],
                            context=context
                        )
                    except Exception as e:
                        steward_response = f"I apologize, but I encountered an error: {str(e)}. Please try again."
                else:
                    steward_response = f"I'm currently in setup mode. {st.session_state.get('beta_steward_error', 'Please configure the MISTRAL_API_KEY.')}"

                st.session_state.beta_steward_chat_history.append({
                    'role': 'assistant',
                    'content': steward_response,
                    'timestamp': datetime.now().isoformat()
                })

                st.rerun()

    def _gather_beta_context(self) -> dict:
        """Gather BETA-specific context for Steward.

        Returns:
            dict: Context including PA, current metrics, turn info
        """
        context = {}

        # Get the established PA
        pa = st.session_state.get('primacy_attractor', {})
        if pa:
            context['primacy_attractor'] = pa

        # Get current turn number
        current_turn = st.session_state.get('beta_current_turn', 1)
        context['current_turn'] = current_turn

        # Get latest calibration card values from the most recent turn data
        # Try multiple sources for robustness

        # Source 1: state_manager turns
        state_manager = st.session_state.get('state_manager')
        if state_manager and hasattr(state_manager, 'state') and hasattr(state_manager.state, 'turns'):
            turns = state_manager.state.turns
            if turns:
                latest_turn = turns[-1]
                telos_analysis = latest_turn.get('telos_analysis', {})

                # Extract the three metrics
                f_user = telos_analysis.get('user_pa_fidelity')
                f_ai = telos_analysis.get('ai_pa_fidelity')
                ps = telos_analysis.get('primacy_state_score')

                if f_user is not None:
                    context['f_user'] = f_user
                if f_ai is not None:
                    context['f_ai'] = f_ai
                if ps is not None:
                    context['primacy_state'] = ps

        # Source 2: Direct beta_turn_X_data storage (fallback)
        if 'f_user' not in context:
            for turn_num in range(current_turn - 1, 0, -1):
                turn_data = st.session_state.get(f'beta_turn_{turn_num}_data', {})
                if turn_data:
                    telos_analysis = turn_data.get('telos_analysis', {})
                    if telos_analysis.get('user_pa_fidelity'):
                        context['f_user'] = telos_analysis['user_pa_fidelity']
                    if telos_analysis.get('ai_pa_fidelity'):
                        context['f_ai'] = telos_analysis['ai_pa_fidelity']
                    if telos_analysis.get('primacy_state_score'):
                        context['primacy_state'] = telos_analysis['primacy_state_score']
                    if 'f_user' in context:
                        break

        return context


def render_beta_steward_button():
    """Render the Beta Steward toggle button (handshake emoji)."""
    # Initialize panel state
    if 'beta_steward_panel_open' not in st.session_state:
        st.session_state.beta_steward_panel_open = False

    button_label = "Close Steward" if st.session_state.beta_steward_panel_open else "Ask Steward"

    if st.button(button_label, key="beta_steward_toggle", use_container_width=True):
        st.session_state.beta_steward_panel_open = not st.session_state.beta_steward_panel_open
        st.rerun()
