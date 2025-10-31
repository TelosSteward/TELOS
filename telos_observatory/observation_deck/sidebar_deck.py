"""
Observation Deck - Sidebar Interface
=====================================

Right sidebar panel showing current turn metrics with collapsible sections.

Features:
- Math Breakdown: Fidelity, distance, threshold status
- Counterfactual Comparison: Phase 2/2B intervention results
- Steward Integration: AI assistant for turn-specific queries
- Deep Research: Open turn-specific Observatory view in new tab

Keyboard Shortcut: Shift+O to toggle
"""

import streamlit as st
from typing import Optional, Dict, Any
from .deep_research import render_deep_research_button
from .math_breakdown import render_math_breakdown
from .counterfactual_viewer import render_counterfactual_viewer
from .steward_interface import render_steward_interface


def render_observation_deck_sidebar():
    """
    Render Observation Deck sidebar (right-side panel).

    State Variables Used:
    - deck_expanded: Sidebar visibility
    - current_turn: Active turn index
    - session_data: Full session data
    - deck_show_math: Math breakdown expanded
    - deck_show_counterfactual: Counterfactual section expanded
    - steward_active: Steward chat active
    """

    # Only render if deck is expanded
    if not st.session_state.get('deck_expanded', False):
        return

    # Get current turn data
    current_turn = st.session_state.get('current_turn', 0)
    session_data = st.session_state.get('session_data', {})
    turns = session_data.get('turns', [])

    if not turns or current_turn >= len(turns):
        return

    turn_data = turns[current_turn]

    # Sidebar container with fixed positioning
    st.markdown("""
        <style>
        /* Observation Deck Sidebar Styling */
        .observation-deck {
            position: fixed;
            right: 0;
            top: 0;
            width: 320px;
            height: 100vh;
            background: rgba(20, 25, 35, 0.95);
            backdrop-filter: blur(10px);
            border-left: 1px solid rgba(255, 215, 0, 0.3);
            padding: 1.5rem;
            overflow-y: auto;
            z-index: 1000;
            box-shadow: -4px 0 20px rgba(0, 0, 0, 0.5);
            animation: slideInRight 0.3s ease-out;
        }

        @keyframes slideInRight {
            from {
                transform: translateX(100%);
                opacity: 0;
            }
            to {
                transform: translateX(0);
                opacity: 1;
            }
        }

        .deck-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1.5rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid rgba(255, 215, 0, 0.2);
        }

        .deck-title {
            font-size: 1.2rem;
            font-weight: bold;
            color: #FFD700;
            margin: 0;
        }

        .deck-close {
            background: rgba(255, 255, 255, 0.1);
            border: none;
            color: white;
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .deck-close:hover {
            background: rgba(255, 255, 255, 0.2);
        }

        .turn-indicator {
            background: rgba(255, 215, 0, 0.1);
            padding: 0.75rem;
            border-radius: 6px;
            margin-bottom: 1.5rem;
            border: 1px solid rgba(255, 215, 0, 0.2);
        }

        .section-divider {
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(255, 215, 0, 0.3), transparent);
            margin: 1.5rem 0;
        }
        </style>
    """, unsafe_allow_html=True)

    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("### 🔭 Observation Deck")
    with col2:
        if st.button("✕", key="deck_close", help="Close (Shift+O)"):
            st.session_state.deck_expanded = False
            st.rerun()

    # Turn indicator
    st.markdown(f"""
        <div class="turn-indicator">
            <div style="color: #888; font-size: 0.8rem; margin-bottom: 0.25rem;">OBSERVING TURN</div>
            <div style="font-size: 1.5rem; font-weight: bold; color: #FFD700;">{current_turn + 1}</div>
            <div style="color: #888; font-size: 0.8rem; margin-top: 0.25rem;">
                of {len(turns)} total turns
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Conversation ID and metadata
    conversation_id = session_data.get('session_id', 'unknown')
    st.caption(f"**Study:** {conversation_id}")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Section 1: Math Breakdown (Collapsible)
    with st.expander("📊 Mathematical Breakdown", expanded=st.session_state.get('deck_show_math', False)):
        st.session_state.deck_show_math = True
        render_math_breakdown(turn_data)

    if not st.session_state.get('deck_show_math', False):
        st.session_state.deck_show_math = False

    # Section 2: Counterfactual Comparison (Collapsible)
    with st.expander("🌿 Counterfactual Analysis", expanded=st.session_state.get('deck_show_counterfactual', False)):
        st.session_state.deck_show_counterfactual = True
        render_counterfactual_viewer(conversation_id, current_turn)

    if not st.session_state.get('deck_show_counterfactual', False):
        st.session_state.deck_show_counterfactual = False

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Section 3: Action Buttons
    st.markdown("#### 🎯 Actions")

    # Deep Research button
    render_deep_research_button(current_turn, conversation_id)

    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)

    # Steward button
    if st.button("🤝 Ask Steward", key="deck_steward_toggle", use_container_width=True):
        st.session_state.steward_active = not st.session_state.get('steward_active', False)
        st.rerun()

    # Section 4: Steward Interface (if active)
    if st.session_state.get('steward_active', False):
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        render_steward_interface(turn_data, conversation_id, current_turn)

    # Footer
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.caption("Press **Shift+O** to toggle Observation Deck")


def toggle_observation_deck():
    """
    Toggle Observation Deck visibility.
    Called by keyboard shortcut (Shift+O) or control strip click.
    """
    st.session_state.deck_expanded = not st.session_state.get('deck_expanded', False)
    st.rerun()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def is_deck_expanded() -> bool:
    """Check if Observation Deck is currently expanded."""
    return st.session_state.get('deck_expanded', False)


def expand_deck():
    """Programmatically expand Observation Deck."""
    st.session_state.deck_expanded = True


def collapse_deck():
    """Programmatically collapse Observation Deck."""
    st.session_state.deck_expanded = False
