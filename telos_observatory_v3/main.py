#!/usr/bin/env python3
"""
TELOS Observatory V3 - Main Application
Built from scratch with native Streamlit layout.
"""

import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

# Import V3 components
from telos_observatory_v3.core.state_manager import StateManager
from telos_observatory_v3.utils.telos_demo_data import generate_telos_demo_session
from telos_observatory_v3.components.sidebar_actions import SidebarActions
from telos_observatory_v3.components.conversation_display import ConversationDisplay
from telos_observatory_v3.components.observation_deck import ObservationDeck
from telos_observatory_v3.components.teloscope_controls import TELOSCOPEControls


def initialize_session():
    """Initialize session state - starts fresh (no pre-loaded demo data)."""
    if 'state_manager' not in st.session_state:
        # Set Demo Mode as DEFAULT (before anything else)
        if 'telos_demo_mode' not in st.session_state:
            st.session_state.telos_demo_mode = True

        # Create state manager
        state_manager = StateManager()

        # Start with EMPTY session (no pre-loaded turns)
        # Demo Mode will show welcome message and start fresh
        # Open Mode will also start fresh
        empty_data = {
            'session_id': f"session_{int(datetime.now().timestamp())}",
            'turns': [],
            'total_turns': 0,
            'current_turn': 0,
            'avg_fidelity': 0.0,
            'total_interventions': 0,
            'drift_warnings': 0
        }
        state_manager.initialize(empty_data)

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

    # Dark theme styling and hide Streamlit defaults
    st.markdown("""
    <style>
    /* Hide Streamlit's built-in sidebar collapse button */
    button[kind="header"] {{
        display: none !important;
    }}

    [data-testid="collapsedControl"] {{
        display: none !important;
    }}

    /* Hide Streamlit header, menu, and rerun elements */
    header {{visibility: hidden;}}
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}

    /* Hide "Rerun" and "Always rerun" buttons */
    .stApp [data-testid="stStatusWidget"] {{
        display: none !important;
    }}

    [data-testid="stStatusWidget"] {{
        visibility: hidden !important;
    }}

    .stApp > header {{
        display: none !important;
    }}

    /* Hide app rerun related elements */
    [data-testid="stDecoration"] {{
        display: none !important;
    }}

    /* Hide top toolbar/header area */
    section[data-testid="stHeader"] {{
        display: none !important;
    }}

    /* Main content: dark grey background */
    .stApp {{
        background-color: #1a1a1a !important;
    }}

    .main {{
        background-color: #1a1a1a !important;
    }}

    /* Sidebar: medium-dark grey */
    [data-testid="stSidebar"] {{
        background-color: #2a2a2a !important;
    }}

    /* Sidebar buttons - gold borders with hover effects */
    [data-testid="stSidebar"] .stButton > button {{
        background-color: #2d2d2d !important;
        color: #e0e0e0 !important;
        border: 2px solid #FFD700 !important;
        transition: all 0.3s ease !important;
    }}

    [data-testid="stSidebar"] .stButton > button:hover {{
        background-color: #3d3d3d !important;
        color: #e0e0e0 !important;
        border: 2px solid #FFD700 !important;
        box-shadow: 0 0 10px rgba(255, 215, 0, 0.5) !important;
    }}

    /* Increase top padding to provide proper spacing */
    .block-container {{
        padding-top: 5rem;
        padding-bottom: 1rem;
    }}

    /* Button styling - force gold borders and hover effects */
    .stButton > button {{
        background-color: #2d2d2d !important;
        color: #e0e0e0 !important;
        border: 2px solid #FFD700 !important;
        transition: all 0.3s ease !important;
    }}

    /* Hover effect - lighter background and glowing gold border */
    .stButton > button:hover,
    .stButton > button:focus,
    .stButton > button:active,
    button:hover,
    button:focus {{
        background-color: #3d3d3d !important;
        color: #e0e0e0 !important;
        border: 3px solid #FFD700 !important;
        box-shadow: 0 0 20px rgba(255, 215, 0, 0.8), 0 0 10px rgba(255, 215, 0, 0.6) !important;
        transform: scale(1.01) !important;
    }}

    /* Keep button text light on hover */
    .stButton > button:hover * {{
        color: #e0e0e0 !important;
    }}

    /* Force button text color on hover */
    .stButton > button:hover p {{
        color: #e0e0e0 !important;
    }}

    .stButton > button:hover span {{
        color: #e0e0e0 !important;
    }}

    .stButton > button:hover div {{
        color: #e0e0e0 !important;
    }}

    /* Gold border for main content */
    .main .block-container {{
        border: 2px solid #FFD700;
        border-radius: 10px;
        padding: 20px;
    }}

    /* Lighter text for better readability */
    p, span, div {{
        color: #e0e0e0 !important;
    }}

    /* Global font size increase by 2px */
    html, body, [class*="css"] {{
        font-size: 16px !important;
    }}

    /* Increase all paragraph and text elements */
    p {{
        font-size: 16px !important;
    }}

    /* Increase button text */
    button {{
        font-size: 16px !important;
    }}

    /* Increase input text */
    input {{
        font-size: 16px !important;
    }}

    /* Increase label text */
    label {{
        font-size: 16px !important;
    }}

    /* Increase heading sizes */
    h1 {{
        font-size: 36px !important;
    }}

    h2 {{
        font-size: 30px !important;
    }}

    h3 {{
        font-size: 24px !important;
    }}

    h4 {{
        font-size: 20px !important;
    }}

    /* Increase markdown text */
    .stMarkdown {{
        font-size: 16px !important;
    }}

    /* Checkbox styling - simpler gold toggle */
    .stCheckbox {{
        gap: 10px !important;
    }}

    .stCheckbox > label {{
        color: #e0e0e0 !important;
    }}

    /* Style the checkbox itself */
    .stCheckbox input[type="checkbox"] {{
        width: 44px !important;
        height: 24px !important;
        appearance: none !important;
        -webkit-appearance: none !important;
        background-color: #2d2d2d !important;
        border: 2px solid #666 !important;
        border-radius: 12px !important;
        cursor: pointer !important;
        position: relative !important;
        transition: all 0.3s ease !important;
        margin-right: 10px !important;
    }}

    /* Checkmark/slider dot */
    .stCheckbox input[type="checkbox"]::after {{
        content: "" !important;
        position: absolute !important;
        width: 16px !important;
        height: 16px !important;
        border-radius: 50% !important;
        background-color: #fff !important;
        top: 2px !important;
        left: 2px !important;
        transition: all 0.3s ease !important;
    }}

    /* Checked state */
    .stCheckbox input[type="checkbox"]:checked {{
        background-color: #FFD700 !important;
        border-color: #FFD700 !important;
    }}

    .stCheckbox input[type="checkbox"]:checked::after {{
        left: 22px !important;
        background-color: #000 !important;
    }}

    /* Hide the SVG checkmark that Streamlit adds */
    .stCheckbox svg {{
        display: none !important;
    }}

    /* Text input styling - dark background with white text */
    .stTextInput input {{
        background-color: #2d2d2d !important;
        color: #ffffff !important;
        border: 1px solid #FFD700 !important;
    }}

    .stTextInput input::placeholder {{
        color: #888 !important;
    }}

    /* Make the text input label transparent/hidden */
    .stTextInput label {{
        color: transparent !important;
    }}

    /* Force text input to have white text when typing */
    input[type="text"] {{
        color: #ffffff !important;
        background-color: #2d2d2d !important;
    }}

    /* Remove red color from toggles - use gold/grey scheme instead */
    .stCheckbox input[type="checkbox"]:not(:checked) {{
        background-color: #2d2d2d !important;
        border-color: #666 !important;
    }}

    .stCheckbox input[type="checkbox"]:checked {{
        background-color: #FFD700 !important;
        border-color: #FFD700 !important;
    }}

    /* Override any red colors in toggle switches */
    [data-baseweb="checkbox"] {{
        background-color: #2d2d2d !important;
    }}

    [data-baseweb="checkbox"][data-checked="true"] {{
        background-color: #FFD700 !important;
    }}
    </style>
    """, unsafe_allow_html=True)

    # Initialize session
    initialize_session()
    state_manager = st.session_state.state_manager

    # Instantiate components
    sidebar_actions = SidebarActions(state_manager)
    conversation_display = ConversationDisplay(state_manager)
    observation_deck = ObservationDeck(state_manager)
    teloscope_controls = TELOSCOPEControls(state_manager)

    # Render sidebar
    sidebar_actions.render()

    # Main conversation display - shows main chat and analysis windows
    conversation_display.render()

    # More spacing to push control bars down and give chat window more room
    st.markdown("<div style='margin: 40px 0;'></div>", unsafe_allow_html=True)

    # Observation Deck (collapsible) - Contains metrics and view options for analysis windows
    observation_deck.render()

    # Minimal spacing between the two bars
    st.markdown("<div style='margin: 5px 0;'></div>", unsafe_allow_html=True)

    # TELOSCOPE Controls at bottom (collapsible)
    teloscope_controls.render()


if __name__ == "__main__":
    main()
