#!/usr/bin/env python3
"""
TELOS Observatory V3 - Main Application
Built from scratch with native Streamlit layout.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

# Import V3 components
from telos_observatory_v3.core.state_manager import StateManager
from telos_observatory_v3.utils.mock_data import generate_mock_session
from telos_observatory_v3.components.sidebar_actions import SidebarActions
from telos_observatory_v3.components.control_strip import ControlStrip
from telos_observatory_v3.components.conversation_display import ConversationDisplay
from telos_observatory_v3.components.observation_deck import ObservationDeck
from telos_observatory_v3.components.teloscope_controls import TELOSCOPEControls


def initialize_session():
    """Initialize session state and load mock data."""
    if 'state_manager' not in st.session_state:
        # Create state manager
        state_manager = StateManager()

        # Generate and load mock data
        mock_data = generate_mock_session(num_turns=10)
        state_manager.initialize(mock_data)

        # Store in session state
        st.session_state.state_manager = state_manager


def main():
    """Main application entry point."""
    # Page configuration
    st.set_page_config(
        page_title="TELOS Observatory V3",
        page_icon="🔭",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for dark theme and styling
    st.markdown("""
    <style>
    /* Dark theme */
    .stApp {
        background-color: #0a0a0a;
    }

    /* Remove default padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Button styling */
    .stButton > button {
        background-color: #2d2d2d;
        color: #fff;
        border: 1px solid #FFD700;
    }

    .stButton > button:hover {
        background-color: #FFD700;
        color: #000;
        border: 1px solid #FFD700;
    }
    </style>
    """, unsafe_allow_html=True)

    # Initialize session
    initialize_session()
    state_manager = st.session_state.state_manager

    # Instantiate components
    sidebar_actions = SidebarActions(state_manager)
    control_strip = ControlStrip(state_manager)
    conversation_display = ConversationDisplay(state_manager)
    observation_deck = ObservationDeck(state_manager)
    teloscope_controls = TELOSCOPEControls(state_manager)

    # Render sidebar
    sidebar_actions.render()

    # Top bar with toggles
    top_col1, top_col2 = st.columns([3, 1])

    with top_col1:
        st.markdown("# 🔭 TELOS Observatory")

    with top_col2:
        # Dark mode toggle (placeholder - not wired up yet)
        st.checkbox("🌙 Dark Mode", value=True, key="dark_mode", disabled=True)

    st.markdown("---")

    # Control Strip (top-right area)
    control_strip.render()

    st.markdown("---")

    # Main content layout: conversation + observation deck (conditional)
    if state_manager.is_deck_expanded():
        # Deck is open: split layout
        main_col, deck_col = st.columns([7, 3])

        with main_col:
            conversation_display.render()

        with deck_col:
            observation_deck.render()

            # TELOSCOPE Controls at bottom of deck
            st.markdown("---")
            teloscope_controls.render()
    else:
        # Deck is closed: full-width conversation
        conversation_display.render()


if __name__ == "__main__":
    main()
