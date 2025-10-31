"""
TELOS Observatory - Main Entry Point
=====================================

Phase 1: Frame-by-frame conversation analysis with TELOSCOPE control system.

Architecture:
- Observation Deck: Turn display with distance-based dimming
- TELOSCOPE: Navigation controls with timeline scrubber
- Control Strip: Real-time status display

CRITICAL: Follows Streamlit patterns from docs/streamlit_patterns.md
- All state in st.session_state
- Callbacks only update state
- st.rerun() after state changes
"""

import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from telos_observatory.mock_data import generate_mock_session
from telos_observatory.observation_deck.deck_interface import render_observation_deck
from telos_observatory.teloscope.teloscope_controller import render_teloscope


# ============================================================================
# Session State Initialization
# ============================================================================

def init_session_state():
    """
    Initialize ALL session state variables.

    CRITICAL: Call this FIRST in main() before any rendering.

    State Variables:
    - initialized: Initialization flag
    - current_turn: Active turn index (0-based)
    - playing: Auto-play state
    - playback_speed: Playback speed multiplier
    - last_play_time: Timestamp for autoplay timing
    - telescope_open: TELOSCOPE visibility (Phase 1: always true)
    - session_data: Mock session data
    """
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.current_turn = 0
        st.session_state.playing = False
        st.session_state.playback_speed = 1.0
        st.session_state.last_play_time = 0
        st.session_state.telescope_open = True  # Phase 1: always visible

        # Load mock session data
        st.session_state.session_data = generate_mock_session()


# ============================================================================
# Main Application
# ============================================================================

def main():
    """
    Main application entry point.

    Flow:
    1. Configure Streamlit
    2. Initialize session state
    3. Render Observation Deck
    4. Render TELOSCOPE controls
    """
    # Page configuration
    st.set_page_config(
        layout="wide",
        page_title="TELOS Observatory",
        page_icon="🔭",
        initial_sidebar_state="collapsed"
    )

    # Initialize state FIRST
    init_session_state()

    # Custom CSS for dark theme and styling
    st.markdown("""
        <style>
        /* Dark theme base */
        .main {
            background-color: #0E1117;
            color: #E0E0E0;
        }

        /* Header styling */
        h1, h2, h3 {
            color: #FFD700;
        }

        /* Metrics */
        [data-testid="stMetricValue"] {
            font-size: 1.5rem;
            color: #FFD700;
        }

        /* Remove Streamlit branding footer */
        footer {
            visibility: hidden;
        }

        /* Adjust padding for TELOSCOPE fixed position */
        .main .block-container {
            padding-bottom: 220px !important;
            padding-top: 2rem;
        }

        /* Scrollbar styling */
        ::-webkit-scrollbar {
            width: 8px;
        }

        ::-webkit-scrollbar-track {
            background: rgba(255, 255, 255, 0.05);
        }

        ::-webkit-scrollbar-thumb {
            background: rgba(255, 215, 0, 0.3);
            border-radius: 4px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: rgba(255, 215, 0, 0.5);
        }

        /* Button styling */
        .stButton button {
            border-radius: 4px;
            transition: all 0.3s ease;
        }

        .stButton button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
        }
        </style>
    """, unsafe_allow_html=True)

    # Welcome header
    render_header()

    # Main content: Observation Deck
    render_observation_deck()

    # Bottom controls: TELOSCOPE
    render_teloscope()

    # Debug info (optional, can be removed)
    render_debug_info()


def render_header():
    """Render welcome header with Observatory branding."""
    st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h1 style="font-size: 3rem; margin: 0;">🔭 TELOS Observatory</h1>
            <p style="color: #888; font-size: 1.1rem; margin: 0.5rem 0 0 0;">
                Real-time AI Governance Analysis Platform
            </p>
            <p style="color: #666; font-size: 0.9rem; margin: 0.25rem 0 0 0;">
                Phase 1: Frame-by-Frame Conversation Observation
            </p>
        </div>
    """, unsafe_allow_html=True)


def render_debug_info():
    """
    Render debug information (optional, collapsible).

    Shows current session state for development/debugging.
    """
    with st.expander("🔧 Debug Info", expanded=False):
        st.json({
            'current_turn': st.session_state.current_turn,
            'playing': st.session_state.playing,
            'playback_speed': st.session_state.playback_speed,
            'total_turns': len(st.session_state.session_data.get('turns', [])),
            'session_id': st.session_state.session_data.get('session_id')
        })


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    main()
