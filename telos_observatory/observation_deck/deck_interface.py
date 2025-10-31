"""
Observation Deck Interface
===========================

Main interface for the Observation Deck display.
Orchestrates turn rendering with dimming effects.

Phase 1: Focus on turn-by-turn display with dimming
Phase 2: Will add comparison viewer and mathematical tools
"""

import streamlit as st
from telos_observatory.observation_deck.turn_renderer import (
    render_chat_viewport,
    render_control_strip
)


def render_observation_deck():
    """
    Render complete Observation Deck interface.

    Phase 1 Components:
    - Control strip (top-right status display)
    - Chat viewport (scrollable turn display with dimming)

    Reads from st.session_state:
    - session_data: Session with turns
    - current_turn: Active turn index
    """
    # Render control strip (fixed top-right)
    render_control_strip()

    # Main viewport title
    st.markdown("""
        <div style="margin-bottom: 1rem;">
            <h2 style="color: #FFD700; margin: 0;">📡 Observation Deck</h2>
            <p style="color: #888; font-size: 0.9rem; margin: 0.25rem 0 0 0;">
                Frame-by-frame conversation analysis with governance transparency
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Session info
    render_session_info()

    # Divider
    st.markdown("---")

    # Chat viewport with turn rendering
    render_chat_viewport()


def render_session_info():
    """
    Render session metadata summary.

    Reads from st.session_state:
    - session_data: Session with metadata
    """
    session_data = st.session_state.get('session_data', {})

    if not session_data:
        return

    session_id = session_data.get('session_id', 'Unknown')
    stats = session_data.get('statistics', {})

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Session ID",
            value=session_id[-8:]  # Last 8 chars
        )

    with col2:
        total_turns = len(session_data.get('turns', []))
        st.metric(
            label="Total Turns",
            value=total_turns
        )

    with col3:
        avg_fidelity = stats.get('avg_fidelity')
        if avg_fidelity is not None:
            st.metric(
                label="Avg Fidelity",
                value=f"{avg_fidelity:.2f}"
            )
        else:
            st.metric(
                label="Avg Fidelity",
                value="Calibrating"
            )

    with col4:
        interventions = stats.get('interventions', 0)
        st.metric(
            label="Interventions",
            value=interventions
        )
