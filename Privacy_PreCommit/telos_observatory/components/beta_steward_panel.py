"""
Beta Steward Panel Component for TELOS Observatory V3 BETA Mode.
Specialized Steward interface for BETA testers with proper context awareness.
"""

import streamlit as st
from datetime import datetime
from services.beta_steward_llm import BetaStewardLLM
import html
import re


# Zone color mapping for fidelity scores
ZONE_COLORS = {
    'green': '#2ecc71',   # Bright green
    'yellow': '#f1c40f',  # Yellow (standard Steward color)
    'orange': '#e67e22',  # Orange
    'red': '#e74c3c',     # Red
}


def _md_to_html(text: str) -> str:
    """Convert basic markdown to HTML for display in Steward panel.

    Handles: **bold**, *italic*, `code`, [[zone:value]] colored scores, and line breaks.
    """
    # Escape HTML first to prevent injection
    text = html.escape(text)

    # Zone-colored scores: [[green:0.856]] -> colored span
    def replace_zone_score(match):
        zone = match.group(1)
        value = match.group(2)
        color = ZONE_COLORS.get(zone, '#F4D03F')  # Default to yellow
        return f'<span style="color: {color}; font-weight: bold; font-size: 1.1em;">{value}</span>'

    text = re.sub(r'\[\[(\w+):([^\]]+)\]\]', replace_zone_score, text)

    # Bold: **text** -> <b>text</b>
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    # Italic: *text* -> <i>text</i>
    text = re.sub(r'\*(.+?)\*', r'<i>\1</i>', text)
    # Code: `text` -> <code>text</code>
    text = re.sub(r'`(.+?)`', r'<code style="background: rgba(0,0,0,0.3); padding: 2px 4px; border-radius: 3px;">\1</code>', text)
    # Line breaks
    text = text.replace('\n', '<br>')
    return text


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
        # Pop it immediately to prevent duplicate processing
        intervention_turn = st.session_state.pop('steward_intervention_turn', None)

        # Initialize chat history if not exists
        if 'beta_steward_chat_history' not in st.session_state:
            if intervention_turn:
                # Opened from "Why?" button - provide detailed explanation
                initial_message = self._build_why_explanation(intervention_turn)
            else:
                # Standard greeting - minimal
                initial_message = "I'm Steward. Ask me anything about what you're seeing."

            st.session_state.beta_steward_chat_history = [
                {
                    'role': 'assistant',
                    'content': initial_message,
                    'timestamp': datetime.now().isoformat()
                }
            ]
        elif intervention_turn:
            # Chat history exists but opened from new "Why?" button
            fidelity_message = self._build_why_explanation(intervention_turn)
            st.session_state.beta_steward_chat_history.append({
                'role': 'assistant',
                'content': fidelity_message,
                'timestamp': datetime.now().isoformat()
            })

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
                # Convert markdown to HTML for proper rendering
                formatted_content = _md_to_html(message['content'])
                st.markdown(f"""
                <div style="background-color: rgba(255, 215, 0, 0.1); border: 1px solid #F4D03F; border-radius: 8px; padding: 12px; margin-bottom: 15px;">
                    <strong style="color: #F4D03F; font-size: 18px;">Steward:</strong><br>
                    <span style="color: #F4D03F; font-size: 16px; line-height: 1.6;">{formatted_content}</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background-color: rgba(255, 255, 255, 0.05); border: 1px solid #666; border-radius: 8px; padding: 12px; margin-bottom: 15px;">
                    <strong style="color: #e0e0e0; font-size: 18px;">You:</strong><br>
                    <span style="color: #e0e0e0; font-size: 16px; line-height: 1.6;">{html.escape(message['content'])}</span>
                </div>
                """, unsafe_allow_html=True)

        # Input form - stacked layout matching main chat (text area above, Send button below)
        st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)

        with st.form(key="beta_steward_form", clear_on_submit=True):
            user_input = st.text_area(
                "Ask Steward anything...",
                placeholder="",
                label_visibility="collapsed",
                key="beta_steward_input_form",
                height=100  # Multi-line input
            )
            submit = st.form_submit_button("Send", use_container_width=True)

            if submit and user_input:
                # Add user message and set pending flag for two-phase processing
                st.session_state.beta_steward_chat_history.append({
                    'role': 'user',
                    'content': user_input,
                    'timestamp': datetime.now().isoformat()
                })
                # Set pending flag to trigger contemplating animation on next render
                st.session_state.steward_pending_response = True
                st.session_state.steward_pending_input = user_input
                st.rerun()

        # Add JavaScript to submit form on Enter key (Shift+Enter for new line)
        import streamlit.components.v1 as components
        components.html("""
        <script>
        setTimeout(function() {
            // Find textareas in the Steward panel specifically
            const textareas = window.parent.document.querySelectorAll('textarea');
            textareas.forEach(function(textarea) {
                // Only target the Steward form textarea (check for form key pattern)
                if (!textarea.dataset.stewardEnterHandler) {
                    textarea.dataset.stewardEnterHandler = 'true';
                    textarea.addEventListener('keydown', function(e) {
                        if (e.key === 'Enter' && !e.shiftKey) {
                            e.preventDefault();
                            // Find the Send button in parent document
                            const buttons = window.parent.document.querySelectorAll('button');
                            for (let btn of buttons) {
                                const text = btn.textContent || btn.innerText;
                                if (text.includes('Send')) {
                                    btn.click();
                                    break;
                                }
                            }
                        }
                    });
                }
            });
        }, 300);
        </script>
        """, height=0)

        # Two-phase response generation: Show contemplating, then fetch response
        if st.session_state.get('steward_pending_response', False):
            # Show contemplating animation
            st.markdown("""
            <style>
            @keyframes contemplating-pulse {
                0%, 100% { opacity: 0.5; }
                50% { opacity: 1.0; }
            }
            @keyframes dot-bounce {
                0%, 80%, 100% { transform: translateY(0); }
                40% { transform: translateY(-5px); }
            }
            </style>
            <div style="background-color: rgba(255, 215, 0, 0.1); border: 1px solid #F4D03F; border-radius: 8px; padding: 12px; margin-bottom: 15px;">
                <strong style="color: #F4D03F; font-size: 18px;">Steward:</strong><br>
                <span style="color: #F4D03F; font-size: 16px; font-style: italic; animation: contemplating-pulse 1.5s ease-in-out infinite;">
                    Contemplating<span style="display: inline-block; animation: dot-bounce 1.4s infinite ease-in-out;">.</span><span style="display: inline-block; animation: dot-bounce 1.4s infinite ease-in-out 0.2s;">.</span><span style="display: inline-block; animation: dot-bounce 1.4s infinite ease-in-out 0.4s;">.</span>
                </span>
            </div>
            """, unsafe_allow_html=True)

            # Now fetch the actual response
            user_input = st.session_state.steward_pending_input
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

            # Clear pending flags
            st.session_state.steward_pending_response = False
            st.session_state.steward_pending_input = None
            st.rerun()

    def _gather_beta_context(self) -> dict:
        """Gather BETA-specific context for Steward.

        Returns:
            dict: Context including PA, current metrics, turn info, last user input
        """
        context = {}

        # Get the established PA
        pa = st.session_state.get('primacy_attractor', {})
        if pa:
            context['primacy_attractor'] = pa

        # Get current turn number (this is the NEXT turn to play, so completed = current - 1)
        current_turn = st.session_state.get('beta_current_turn', 1)
        context['current_turn'] = current_turn
        completed_turns = current_turn - 1

        # ============================================================
        # PRIMARY SOURCE: beta_turn_X_data (where BETA mode stores data)
        # This is the CORRECT and AUTHORITATIVE source for BETA mode
        # ============================================================
        latest_turn_data = None
        for turn_num in range(completed_turns, 0, -1):
            turn_data = st.session_state.get(f'beta_turn_{turn_num}_data', {})
            if turn_data:
                latest_turn_data = turn_data
                telos_analysis = turn_data.get('telos_analysis', {})

                # Extract the three calibration metrics
                f_user = telos_analysis.get('user_pa_fidelity')
                f_ai = telos_analysis.get('ai_pa_fidelity')
                ps = telos_analysis.get('primacy_state_score')

                if f_user is not None:
                    context['f_user'] = f_user
                if f_ai is not None:
                    context['f_ai'] = f_ai
                if ps is not None:
                    context['primacy_state'] = ps

                # Get the user's input that produced these metrics
                user_input = turn_data.get('user_input', '')
                if user_input:
                    context['last_user_input'] = user_input

                # Get intervention info
                if telos_analysis.get('intervention_triggered'):
                    context['intervention_triggered'] = True
                    context['intervention_reason'] = telos_analysis.get('intervention_reason', '')
                else:
                    context['intervention_triggered'] = False

                # Found data - stop searching
                break

        # ============================================================
        # FALLBACK: state_manager.state.turns (original TELOS flow)
        # Only used if beta_turn_X_data didn't have metrics
        # ============================================================
        if 'f_user' not in context:
            state_manager = st.session_state.get('state_manager')
            if state_manager and hasattr(state_manager, 'state') and hasattr(state_manager.state, 'turns'):
                turns = state_manager.state.turns
                if turns:
                    latest_turn = turns[-1]
                    telos_analysis = latest_turn.get('telos_analysis', {})

                    f_user = telos_analysis.get('user_pa_fidelity')
                    f_ai = telos_analysis.get('ai_pa_fidelity')
                    ps = telos_analysis.get('primacy_state_score')

                    if f_user is not None:
                        context['f_user'] = f_user
                    if f_ai is not None:
                        context['f_ai'] = f_ai
                    if ps is not None:
                        context['primacy_state'] = ps

        return context

    def _build_why_explanation(self, turn_number: int) -> str:
        """Build a detailed 'lift the curtain' explanation for a turn's fidelity score.

        Args:
            turn_number: The turn to explain

        Returns:
            Detailed explanation of why the fidelity was calculated this way
        """
        turn_data = st.session_state.get(f'beta_turn_{turn_number}_data', {})
        telos_analysis = turn_data.get('telos_analysis', {})
        user_fidelity = telos_analysis.get('user_pa_fidelity', 0.0)
        raw_similarity = telos_analysis.get('raw_similarity', user_fidelity)
        user_input = turn_data.get('user_input', '')
        intervention_reason = telos_analysis.get('intervention_reason', '')
        in_basin = telos_analysis.get('in_basin', user_fidelity >= 0.70)

        # Get the PA purpose statement
        pa = st.session_state.get('primacy_attractor', {})
        purpose = pa.get('purpose', 'your stated purpose')

        # Determine zone using Goldilocks thresholds from central config
        from config.colors import _ZONE_ALIGNED, _ZONE_MINOR_DRIFT, _ZONE_DRIFT
        if user_fidelity >= _ZONE_ALIGNED:  # 0.76
            zone = "green"
            zone_meaning = "Aligned"
        elif user_fidelity >= _ZONE_MINOR_DRIFT:  # 0.73
            zone = "yellow"
            zone_meaning = "Minor Drift (no intervention)"
        elif user_fidelity >= _ZONE_DRIFT:  # 0.67
            zone = "orange"
            zone_meaning = "Drift Detected (intervention triggered)"
        else:
            zone = "red"
            zone_meaning = "Significant Drift (intervention triggered)"

        # Build the explanation
        lines = []
        lines.append(f"**Turn {turn_number} Analysis**")
        lines.append(f"*Fidelity Score:* [[{zone}:{user_fidelity:.3f}]] ({zone} zone - {zone_meaning})")
        lines.append("")

        # Show what was measured
        truncated_input = user_input[:80] + ('...' if len(user_input) > 80 else '')
        lines.append(f"*Your input:* \"{truncated_input}\"")
        lines.append("")

        # Explain the calculation
        lines.append("**How this was calculated:**")
        lines.append(f"1. Your input was embedded into a 1024-dimensional vector space")
        lines.append(f"2. Cosine similarity measured against your Primacy Attractor")
        if raw_similarity != user_fidelity:
            lines.append(f"3. Raw similarity: `{raw_similarity:.3f}` -> normalized fidelity: `{user_fidelity:.3f}`")
        lines.append("")

        # Basin membership
        lines.append("**Basin membership:**")
        if in_basin:
            lines.append(f"Your input is *inside* the purpose basin (threshold: 0.70)")
            lines.append("No intervention was required.")
        else:
            lines.append(f"Your input fell *outside* the purpose basin (threshold: 0.70)")
            if intervention_reason:
                lines.append(f"*Intervention reason:* {intervention_reason}")
        lines.append("")

        # Why this matters (uses Goldilocks zone thresholds)
        lines.append("**What this means:**")
        if user_fidelity >= _ZONE_ALIGNED:  # Aligned zone
            lines.append("Your input semantically aligns well with your stated purpose.")
        elif user_fidelity >= _ZONE_MINOR_DRIFT:  # Minor Drift zone
            lines.append("Your input shows minor drift but remains close enough to your purpose.")
        else:  # Drift Detected or Significant Drift zone
            lines.append("The semantic distance between your input and your purpose exceeded")
            lines.append("the tolerance threshold, triggering TELOS governance.")

        return "\n".join(lines)


def render_beta_steward_button():
    """Render the Beta Steward toggle button (handshake emoji)."""
    # Initialize panel state
    if 'beta_steward_panel_open' not in st.session_state:
        st.session_state.beta_steward_panel_open = False

    button_label = "Close Steward" if st.session_state.beta_steward_panel_open else "Ask Steward"

    if st.button(button_label, key="beta_steward_toggle", use_container_width=True):
        st.session_state.beta_steward_panel_open = not st.session_state.beta_steward_panel_open
        st.rerun()
