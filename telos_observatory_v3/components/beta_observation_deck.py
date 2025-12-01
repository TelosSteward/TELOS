"""
BETA Observation Deck - Simplified view for BETA testers
Shows PA + current fidelity score only (no complex metrics)
Styled to match message windows - centered and clean
"""

import streamlit as st
from config.colors import GOLD
import html


class BetaObservationDeck:
    """Simplified Observation Deck for BETA mode showing PA and fidelity."""

    def render(self):
        """Render the BETA observation deck with centered styling."""
        import streamlit.components.v1 as components

        # Only show if PA is established
        if not st.session_state.get('pa_established', False):
            return

        # Initialize deck visibility state
        if 'beta_deck_visible' not in st.session_state:
            st.session_state.beta_deck_visible = False

        # Anchor for auto-scrolling
        st.markdown('<div id="beta-observation-deck-anchor"></div>', unsafe_allow_html=True)

        # Center the toggle button using columns (matching message layout)
        col_spacer_left, col_content, col_spacer_right = st.columns([1.5, 7.0, 1.5])

        with col_content:
            # Toggle button at top
            deck_label = "🔭 Close Observation Deck" if st.session_state.beta_deck_visible else "🔭 Open Observation Deck"
            if st.button(deck_label, key="beta_deck_toggle_top", use_container_width=True):
                st.session_state.beta_deck_visible = not st.session_state.beta_deck_visible
                st.rerun()

            # Render deck content if visible
            if st.session_state.beta_deck_visible:
                st.markdown("<div style='margin: 10px 0;'></div>", unsafe_allow_html=True)
                self._render_pa_summary()

                # Only show fidelity if there's at least one turn
                # Check beta_current_turn (incremented after each turn) or state_manager turns
                current_turn = st.session_state.get('beta_current_turn', 1)
                state_manager = st.session_state.get('state_manager')
                has_turns = (current_turn > 1) or (state_manager and len(state_manager.get_all_turns()) > 0)
                if has_turns:
                    st.markdown("<div style='margin: 20px 0;'></div>", unsafe_allow_html=True)
                    self._render_fidelity()

                # Close button at bottom
                st.markdown("<div style='margin: 20px 0;'></div>", unsafe_allow_html=True)
                if st.button("🔭 Close Observation Deck", key="beta_deck_toggle_bottom", use_container_width=True):
                    st.session_state.beta_deck_visible = False
                    st.rerun()

                # Auto-scroll to observation deck when opened
                components.html("""
                    <script>
                        setTimeout(function() {
                            var anchor = window.parent.document.getElementById('beta-observation-deck-anchor');
                            if (anchor) {
                                anchor.scrollIntoView({
                                    behavior: 'smooth',
                                    block: 'start'
                                });
                            }
                        }, 100);
                    </script>
                """, height=0)

    def _render_pa_summary(self):
        """Render the user's Primacy Attractor."""
        # PAOnboarding saves to 'primacy_attractor' key
        pa = st.session_state.get('primacy_attractor', {})

        # HTML-escape all PA values to prevent HTML injection and rendering issues
        purpose = html.escape(pa.get('purpose', 'Not set'))
        scope = html.escape(pa.get('scope', 'Not set'))
        success_criteria = html.escape(pa.get('success_criteria', 'Not set'))
        style = html.escape(pa.get('style', '')) if pa.get('style') else ''

        # Build style section only if style exists
        style_html = ""
        if style:
            style_html = f'<div style="margin-bottom: 12px;"><strong style="color: {GOLD};">Style:</strong><br>{style}</div>'

        # Build complete HTML - styled to match message windows
        pa_html = f"""<div style="background: linear-gradient(135deg, #2d2d2d 0%, #1a1a1a 100%); border: 2px solid {GOLD}; border-radius: 10px; padding: 20px; margin: 10px 0;">
<div style="color: {GOLD}; font-size: 20px; font-weight: bold; margin-bottom: 15px;">Your Primacy Attractor</div>
<div style="color: #e0e0e0; font-size: 16px; line-height: 1.8;">
<div style="margin-bottom: 12px;"><strong style="color: {GOLD};">Purpose:</strong><br>{purpose}</div>
<div style="margin-bottom: 12px;"><strong style="color: {GOLD};">Scope:</strong><br>{scope}</div>
<div style="margin-bottom: 12px;"><strong style="color: {GOLD};">Success Criteria:</strong><br>{success_criteria}</div>
{style_html}
</div>
</div>"""

        st.markdown(pa_html, unsafe_allow_html=True)

    def _render_fidelity(self):
        """Render current fidelity/alignment status."""

        fidelity = None
        current_turn = st.session_state.get('beta_current_turn', 1)

        # PRIMARY SOURCE: state_manager.state.turns (same as conversation_display)
        # This ensures observation deck matches the fidelity shown on user messages
        state_manager = st.session_state.get('state_manager')
        if state_manager and hasattr(state_manager, 'state') and hasattr(state_manager.state, 'turns'):
            turns = state_manager.state.turns
            if turns and len(turns) > 0:
                latest_turn = turns[-1]

                # Match conversation_display logic exactly:
                # 1. Check beta_data first
                beta_data = latest_turn.get('beta_data', {})
                beta_fidelity = beta_data.get('telos_fidelity') or beta_data.get('fidelity_score')
                if beta_fidelity is not None and beta_fidelity > 0:
                    fidelity = beta_fidelity

                # 2. Try telos_analysis
                if fidelity is None or fidelity == 0.0 or fidelity == 0.5:
                    telos_analysis = latest_turn.get('telos_analysis', {})
                    telos_fidelity = telos_analysis.get('fidelity_score')
                    if telos_fidelity is not None and telos_fidelity > 0:
                        fidelity = telos_fidelity

                # 3. Check direct fidelity on turn
                if fidelity is None or fidelity == 0.0 or fidelity == 0.5:
                    turn_fidelity = latest_turn.get('fidelity')
                    if turn_fidelity is not None and turn_fidelity > 0 and turn_fidelity != 0.5:
                        fidelity = turn_fidelity

        # FALLBACK: Check BETA turn storage if state_manager didn't have data
        if fidelity is None or fidelity == 0.5:
            for turn_num in range(current_turn - 1, 0, -1):
                turn_data = st.session_state.get(f'beta_turn_{turn_num}_data', {})
                if turn_data:
                    telos_data = turn_data.get('telos_analysis', {})
                    fidelity = telos_data.get('fidelity_score')
                    if fidelity is not None and fidelity != 0.5:
                        break

        # NO HARDCODED DEFAULTS - If fidelity data is not available, show "---"
        # This exposes when actual TELOS mathematics isn't producing values
        # REMOVED: fidelity = 0.85 / 1.0 defaults that masked missing data

        # Determine status and color (handle None properly)
        if fidelity is None or fidelity == 0.0 or fidelity == 0.5:
            status = "Awaiting Data"
            color = "#888"  # Gray for missing data
            explanation = "TELOS is waiting for sufficient conversation data to calculate fidelity."
            fidelity_display = "---"
        elif fidelity >= 0.76:
            status = "Aligned"
            color = "#4CAF50"  # Green >= 0.76 (Goldilocks optimized)
            explanation = "Your conversation is staying well-aligned with your stated purpose."
            fidelity_display = f"{fidelity:.2f}"
        elif fidelity >= 0.73:
            status = "Minor Drift"
            color = "#F4D03F"  # Yellow 0.73-0.76 (soft guidance zone)
            explanation = "The conversation has drifted slightly - soft guidance may help."
            fidelity_display = f"{fidelity:.2f}"
        elif fidelity >= 0.67:
            status = "Drift Detected"
            color = "#FFA500"  # Orange 0.67-0.73 (intervention zone)
            explanation = "The conversation has drifted from your stated purpose. TELOS is guiding you back."
            fidelity_display = f"{fidelity:.2f}"
        else:
            status = "Significant Drift"
            color = "#FF4444"  # Red < 0.67 (strong intervention)
            explanation = "The conversation has significantly drifted. Strong TELOS intervention activated."
            fidelity_display = f"{fidelity:.2f}"

        st.markdown(f"""
<div style="
    background-color: #1a1a1a;
    border: 2px solid {color};
    border-radius: 10px;
    padding: 20px;
    margin: 10px 0;
">
    <div style="color: {color}; font-size: 20px; font-weight: bold; margin-bottom: 10px;">
        Current Alignment: {status}
    </div>
    <div style="color: #e0e0e0; font-size: 32px; font-weight: bold; margin: 15px 0;">
        {fidelity_display}
    </div>
    <div style="color: #e0e0e0; font-size: 14px; line-height: 1.6;">
        {explanation}
    </div>
    <div style="color: #888; font-size: 13px; margin-top: 15px; font-style: italic;">
        This measures how well the conversation is staying aligned with your stated purpose.
        After completing BETA, you'll see detailed metrics in the Observatory.
    </div>
</div>
""", unsafe_allow_html=True)
