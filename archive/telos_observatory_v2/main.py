"""
TELOS Observatory V2 - Main Application
========================================

Clean, modular Observatory built from scratch.

Architecture:
- StateManager: Centralized state management
- ControlStrip: Top-right turn metrics with gold theming
- ObservationDeck: Right sidebar with analysis tools
- TELOSCOPE: Bottom navigation controls
- BasicSidebar: Left sidebar with Save/Load/Reset/Export

Run:
    streamlit run main.py --server.port 8503
"""

import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from telos_observatory_v2.core.state_manager import StateManager
from telos_observatory_v2.components.control_strip import ControlStrip
from telos_observatory_v2.components.observation_deck import ObservationDeck
from telos_observatory_v2.components.teloscope import TELOSCOPEControls
from telos_observatory_v2.components.sidebar import BasicSidebar
from telos_observatory_v2.utils.mock_data import generate_mock_session


def init_state():
    """Initialize application state."""
    if 'state_manager' not in st.session_state:
        # Create state manager
        st.session_state.state_manager = StateManager()

        # Load mock data
        mock_data = generate_mock_session(num_turns=15)
        st.session_state.state_manager.initialize(mock_data)


def main():
    """Main application entry point."""
    # Page configuration
    st.set_page_config(
        layout="wide",
        page_title="TELOS Observatory V2",
        page_icon="🔭",
        initial_sidebar_state="expanded"
    )

    # Initialize state
    init_state()

    # Get state manager
    state_manager = st.session_state.state_manager

    # Apply custom CSS
    apply_custom_css()

    # Render welcome header
    render_header()

    # Render all components
    # Order matters for proper z-index layering

    # 1. Control Strip (top-right, z-index: 1000)
    control_strip = ControlStrip(state_manager)
    control_strip.render()

    # 2. Observation Deck (right sidebar, z-index: 999)
    observation_deck = ObservationDeck(state_manager)
    observation_deck.render()

    # 3. TELOSCOPE Controls (bottom, z-index: 998)
    teloscope = TELOSCOPEControls(state_manager)
    teloscope.render()

    # 4. Basic Sidebar (left sidebar, native Streamlit)
    sidebar = BasicSidebar(state_manager)
    sidebar.render()

    # 5. Main content area
    render_main_content(state_manager)


def apply_custom_css():
    """Apply global custom CSS."""
    st.markdown("""
        <style>
        /* Dark theme base */
        .main {
            background-color: #0E1117;
            color: #E0E0E0;
        }

        /* Headers */
        h1, h2, h3 {
            color: #FFD700;
        }

        /* Metrics */
        [data-testid="stMetricValue"] {
            font-size: 1.5rem;
            color: #FFD700;
        }

        /* Remove Streamlit branding */
        footer {
            visibility: hidden;
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


def render_header():
    """Render welcome header."""
    st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem; margin-top: 1rem;">
            <h1 style="font-size: 2.5rem; margin: 0;">🔭 TELOS Observatory V2</h1>
            <p style="color: #888; font-size: 1rem; margin: 0.5rem 0 0 0;">
                Frame-by-Frame AI Governance Analysis
            </p>
            <p style="color: #666; font-size: 0.85rem; margin: 0.25rem 0 0 0;">
                Modular Architecture • Clean Build • Extensible Design
            </p>
        </div>
    """, unsafe_allow_html=True)


def render_main_content(state_manager: StateManager):
    """
    Render main content area showing current turn.

    Args:
        state_manager: StateManager instance
    """
    turn_data = state_manager.get_current_turn_data()

    if not turn_data:
        st.warning("No turn data available")
        return

    # Main conversation display
    st.markdown("---")

    # User message
    with st.container():
        st.markdown("""
            <div style="
                background: rgba(0, 123, 255, 0.1);
                border-left: 3px solid #007BFF;
                padding: 1rem;
                margin: 1rem 0;
                border-radius: 4px;
            ">
                <div style="font-size: 0.85rem; color: #007BFF; margin-bottom: 0.5rem; font-weight: bold;">👤 User</div>
                <div style="color: #E0E0E0; line-height: 1.6;">
        """ + turn_data.get('user_input', '') + """
                </div>
            </div>
        """, unsafe_allow_html=True)

    # Assistant response
    with st.container():
        st.markdown("""
            <div style="
                background: rgba(40, 167, 69, 0.1);
                border-left: 3px solid #28A745;
                padding: 1rem;
                margin: 1rem 0;
                border-radius: 4px;
            ">
                <div style="font-size: 0.85rem; color: #28A745; margin-bottom: 0.5rem; font-weight: bold;">🤖 Assistant</div>
                <div style="color: #E0E0E0; line-height: 1.6;">
        """ + turn_data.get('response', '') + """
                </div>
            </div>
        """, unsafe_allow_html=True)

    # Turn metadata
    col1, col2, col3 = st.columns(3)
    with col1:
        fidelity = turn_data.get('fidelity', 0)
        st.metric("Fidelity", f"{fidelity:.3f}")
    with col2:
        distance = turn_data.get('distance', 0)
        st.metric("Distance", f"{distance:.3f}")
    with col3:
        status = turn_data.get('status_text', 'Unknown')
        st.metric("Status", status)


if __name__ == "__main__":
    main()
