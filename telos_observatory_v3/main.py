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
from telos_observatory_v3.components.beta_onboarding import BetaOnboarding
from telos_observatory_v3.components.steward_panel import StewardPanel


def initialize_session():
    """Initialize session state - starts fresh (no pre-loaded demo data)."""
    if 'state_manager' not in st.session_state:
        # Set Demo Mode OFF as DEFAULT (TELOS tab shows full controls)
        if 'telos_demo_mode' not in st.session_state:
            st.session_state.telos_demo_mode = False

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
    /* Cache buster: v2024-11-04-19:10 */
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
        border: 1px solid #FFD700 !important;
        transition: all 0.3s ease !important;
    }}

    [data-testid="stSidebar"] .stButton > button:hover {{
        background-color: #3d3d3d !important;
        color: #e0e0e0 !important;
        border: 1px solid #FFD700 !important;
        box-shadow: 0 0 6px rgba(255, 215, 0, 0.5) !important;
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
        border: 1px solid #FFD700 !important;
        transition: all 0.3s ease !important;
    }}

    /* ULTRA AGGRESSIVE HOVER - Target EVERYTHING */
    *:hover {{
        /* This will apply to all elements, then we'll override for buttons */
    }}

    /* Hover effect - lighter background and bright yellow border replaces gold */
    .stButton > button:hover,
    .stButton > button:focus,
    .stButton > button:active,
    div[data-testid*="stButton"] > button:hover,
    div[class*="stButton"] > button:hover,
    button:hover,
    button:focus,
    button:active {{
        background-color: #3d3d3d !important;
        color: #e0e0e0 !important;
        border-width: 4px !important;
        border-style: solid !important;
        border-color: #FFFF00 !important;
        outline: 3px solid #FFFF00 !important;
        outline-offset: 2px !important;
        box-shadow: 0 0 6px #FFFF00, 0 0 6px #FFFF00, 0 0 6px #FFFF00, inset 0 0 6px rgba(255, 255, 0, 0.5) !important;
        transform: scale(1.03) !important;
        filter: brightness(1.3) saturate(1.5) !important;
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
        border: 1px solid #FFD700;
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

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 10px;
        background-color: #1a1a1a;
    }}

    .stTabs [data-baseweb="tab"] {{
        background-color: #2d2d2d;
        border: 1px solid #FFD700;
        border-radius: 8px 8px 0 0;
        color: #e0e0e0;
        font-size: 22px;
        font-weight: bold;
        padding: 12px 24px;
    }}

    .stTabs [aria-selected="true"] {{
        background-color: #FFD700;
        color: #000;
        font-weight: bold;
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
    beta_onboarding = BetaOnboarding(state_manager)
    steward_panel = StewardPanel(state_manager)

    # Check if user has given beta consent
    has_beta_consent = st.session_state.get('beta_consent_given', False)

    # Hide sidebar if Steward panel is open
    steward_panel.hide_sidebar_when_open()

    # Render Steward button (always visible after consent)
    if has_beta_consent:
        steward_panel.render_button()

    # Only render sidebar and tabs if user has consented
    if has_beta_consent:
        # Add slide-in animation for sidebar
        st.markdown("""
        <style>
        /* Sidebar slide-in animation - smooth and elegant */
        [data-testid="stSidebar"] {
            animation: slideInFromLeft 1.2s ease-out;
        }

        @keyframes slideInFromLeft {
            0% {
                transform: translateX(-100%);
                opacity: 0;
            }
            100% {
                transform: translateX(0);
                opacity: 1;
            }
        }

        /* All buttons - quick subtle blink at the end */
        .stButton > button {
            animation: quickBlink 1.4s ease-in-out;
        }

        @keyframes quickBlink {
            0% {
                opacity: 1;
            }
            85% {
                opacity: 1;
            }
            90% {
                box-shadow: 0 0 8px #FFD700;
            }
            95% {
                box-shadow: none;
            }
            100% {
                opacity: 1;
            }
        }
        </style>
        """, unsafe_allow_html=True)

        # Render sidebar
        sidebar_actions.render()

        # Get current demo mode setting
        demo_mode = st.session_state.get('telos_demo_mode', False)

        # Simple single-content approach: show content based on selected tab via radio buttons
        st.markdown("<div style='margin: 20px 0;'></div>", unsafe_allow_html=True)

        # Active tab styling
        st.markdown("""
        <style>
        /* Active tab - just bright gold border, no fill */
        button[kind="primary"] {
            background-color: #2d2d2d !important;
            color: #e0e0e0 !important;
            border: 2px solid #FFD700 !important;
            box-shadow: 0 0 8px rgba(255, 215, 0, 0.5) !important;
        }

        /* When hovering over active tab, dim the border */
        button[kind="primary"]:hover {
            background-color: #3d3d3d !important;
            color: #e0e0e0 !important;
            border: 1px solid #FFD700 !important;
            box-shadow: 0 0 6px #FFD700 !important;
        }

        /* Inactive tabs - normal thin border */
        button[kind="secondary"] {
            background-color: #2d2d2d !important;
            color: #e0e0e0 !important;
            border: 1px solid #FFD700 !important;
        }

        /* Inactive tabs get hover effect */
        button[kind="secondary"]:hover {
            background-color: #3d3d3d !important;
            border: 1px solid #FFD700 !important;
            box-shadow: 0 0 6px #FFD700 !important;
        }
        </style>
        """, unsafe_allow_html=True)

        # Tab selection using columns for custom styling
        # Initialize active tab if not set - default to BETA for new users
        if 'active_tab' not in st.session_state:
            st.session_state.active_tab = "BETA"

        # Get current active tab
        active_tab = st.session_state.active_tab

        col_beta, col_demo, col_telos = st.columns(3)

        with col_beta:
            beta_active = active_tab == "BETA"
            if st.button("BETA", key="tab_beta", use_container_width=True, type="primary" if beta_active else "secondary"):
                st.session_state.active_tab = "BETA"
                st.rerun()

        with col_demo:
            demo_active = active_tab == "DEMO"
            if st.button("DEMO", key="tab_demo", use_container_width=True, type="primary" if demo_active else "secondary"):
                st.session_state.active_tab = "DEMO"
                st.rerun()

        with col_telos:
            telos_active = active_tab == "TELOS"
            if st.button("TELOS", key="tab_telos", use_container_width=True, type="primary" if telos_active else "secondary"):
                st.session_state.active_tab = "TELOS"
                st.rerun()

        st.markdown("<hr style='border: 1px solid #FFD700; margin: 10px 0;'>", unsafe_allow_html=True)
    else:
        # Hide sidebar completely before consent
        st.markdown("""
        <style>
        /* Hide sidebar before consent */
        [data-testid="stSidebar"] {
            display: none !important;
        }

        /* Make main content full width */
        .main .block-container {
            max-width: 100%;
            padding-left: 5rem;
            padding-right: 5rem;
        }
        </style>
        """, unsafe_allow_html=True)

    # Render content based on consent status
    if not has_beta_consent:
        # No consent yet - show full-screen consent page
        beta_onboarding.render()
    else:
        # Check if Steward panel is open
        steward_open = st.session_state.get('steward_panel_open', False)

        if steward_open:
            # Two-column layout: Main content (70%) | Steward chat (30%)
            col_main, col_steward = st.columns([7, 3])

            with col_main:
                # Render content based on active tab
                if st.session_state.active_tab == "BETA":
                    conversation_display.render()
                elif st.session_state.active_tab == "DEMO":
                    st.info("Demo Tab - Coming Soon")
                elif st.session_state.active_tab == "TELOS":
                    conversation_display.render()
                    st.markdown("<div style='margin: 40px 0;'></div>", unsafe_allow_html=True)
                    observation_deck.render()
                    st.markdown("<div style='margin: 5px 0;'></div>", unsafe_allow_html=True)
                    teloscope_controls.render()

            with col_steward:
                # Render Steward chat panel
                steward_panel.render_panel()

        else:
            # Normal full-width layout
            # Render content based on active tab
            if st.session_state.active_tab == "BETA":
                # Beta Tab - simple chat interface
                conversation_display.render()

            elif st.session_state.active_tab == "DEMO":
                # Demo Tab
                st.info("Demo Tab - Coming Soon")

            elif st.session_state.active_tab == "TELOS":
                # TELOS Tab - full Observatory
                conversation_display.render()
                st.markdown("<div style='margin: 40px 0;'></div>", unsafe_allow_html=True)
                observation_deck.render()
                st.markdown("<div style='margin: 5px 0;'></div>", unsafe_allow_html=True)
                teloscope_controls.render()

    # FINAL CSS OVERRIDE - Inject with highest specificity at runtime
    st.html("""
    <style>
    /* Runtime CSS injection - v20:35 - Further reduced glow */
    button:hover {
        border: 1px solid #FFD700 !important;
        box-shadow: 0 0 6px #FFD700 !important;
    }

    /* Message container hover glow */
    .message-container:hover {
        box-shadow: 0 0 6px #FFD700 !important;
        transition: box-shadow 0.3s ease !important;
    }
    </style>
    """)


if __name__ == "__main__":
    main()
