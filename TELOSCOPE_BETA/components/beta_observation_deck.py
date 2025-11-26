"""
BETA Observation Deck - Simplified view for BETA testers
Shows PA + current fidelity score only (no complex metrics)
"""

import streamlit as st
from config.colors import GOLD
import html

class BetaObservationDeck:
    """Simplified Observation Deck for BETA mode showing PA and fidelity."""

    def render(self):
        """Render the BETA observation deck button and content."""

        # Only show if PA is established
        if not st.session_state.get('pa_established', False):
            return

        # Collapsible Observation Deck
        with st.expander("🔭 Observation Deck", expanded=False):
            self._render_pa_summary()

            # Only show fidelity if there's at least one turn
            current_turn = st.session_state.get('beta_current_turn', 1)
            if current_turn > 1 or len(st.session_state.get('conversation_turns', [])) > 0:
                st.markdown("<div style='margin: 20px 0;'></div>", unsafe_allow_html=True)
                self._render_fidelity()

    def _render_pa_summary(self):
        """Render the user's Primacy Attractor."""
        # PAOnboarding saves to 'primacy_attractor' key
        pa = st.session_state.get('primacy_attractor', {})

        # HTML-escape all PA values to prevent HTML injection and rendering issues
        purpose = html.escape(pa.get('purpose', 'Not set'))
        scope = html.escape(pa.get('scope', 'Not set'))
        success_criteria = html.escape(pa.get('success_criteria', 'Not set'))
        style = html.escape(pa.get('style', 'Not set')) if pa.get('style') else None

        st.markdown(f"""
<div style="
    background: linear-gradient(135deg, #2d2d2d 0%, #1a1a1a 100%);
    border: 2px solid {GOLD};
    border-radius: 10px;
    padding: 20px;
    margin: 10px 0;
">
    <div style="color: {GOLD}; font-size: 20px; font-weight: bold; margin-bottom: 15px;">
        Your Primacy Attractor
    </div>
    <div style="color: #e0e0e0; font-size: 16px; line-height: 1.8;">
        <div style="margin-bottom: 12px;">
            <strong style="color: {GOLD};">Purpose:</strong><br>
            {purpose}
        </div>
        <div style="margin-bottom: 12px;">
            <strong style="color: {GOLD};">Scope:</strong><br>
            {scope}
        </div>
        <div style="margin-bottom: 12px;">
            <strong style="color: {GOLD};">Success Criteria:</strong><br>
            {success_criteria}
        </div>
        {f'''<div style="margin-bottom: 12px;">
            <strong style="color: {GOLD};">Style:</strong><br>
            {style}
        </div>''' if style else ''}
    </div>
</div>
""", unsafe_allow_html=True)

    def _render_fidelity(self):
        """Render current fidelity/alignment status."""

        # Get actual fidelity from latest turn data
        turns = st.session_state.get('conversation_turns', [])
        if turns and len(turns) > 0:
            latest_turn = turns[-1]
            # Try to get fidelity from TELOS response data
            telos_data = latest_turn.get('telos_analysis', {})
            fidelity = telos_data.get('fidelity_score', 0.5)
        else:
            fidelity = 0.5  # Default if no turns yet

        # Determine status and color
        if fidelity >= 0.85:
            status = "Aligned"
            color = "#4CAF50"  # Green
            explanation = "Your conversation is staying well-aligned with your stated purpose."
        elif fidelity >= 0.70:
            status = "Minor Drift"
            color = "#FFA500"  # Orange
            explanation = "The conversation has drifted slightly from your original purpose."
        else:
            status = "Significant Drift"
            color = "#FF4444"  # Red
            explanation = "The conversation has drifted significantly from your stated purpose."

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
        {fidelity:.2f}
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
