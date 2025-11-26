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
from core.state_manager import StateManager
# Note: generate_telos_demo_session import removed - not needed for progressive demo slideshow
from components.sidebar_actions_beta import SidebarActionsBeta
from components.conversation_display import ConversationDisplay
from components.observation_deck import ObservationDeck
from components.beta_observation_deck import BetaObservationDeck
from components.beta_completion import BetaCompletion
from components.teloscope_controls import TELOSCOPEControls
from components.beta_onboarding import BetaOnboarding
from components.pa_onboarding import PAOnboarding
from components.steward_panel import StewardPanel
from components.observatory_lens import ObservatoryLens
from services.ab_test_manager import get_ab_test_manager
from services.backend_client import get_backend_service
from config.colors import GOLD


def initialize_session():
    """Initialize session state - starts fresh (no pre-loaded demo data)."""
    if 'state_manager' not in st.session_state:
        # Set Demo Mode based on initial active_tab (DEMO is default)
        if 'telos_demo_mode' not in st.session_state:
            # Will be set to True when active_tab is "DEMO" (which is default)
            st.session_state.telos_demo_mode = True

        # Create state manager
        state_manager = StateManager()

        # Initialize with EMPTY data to enable progressive demo slideshow
        # The slideshow requires len(all_turns) == 0 to render
        # Users will see the progressive demo slides first, then can ask questions
        from datetime import datetime
        empty_data = {
            'session_id': f"session_{int(datetime.now().timestamp())}",
            'turns': [],  # Empty - enables progressive demo slideshow
            'primacy_attractor': None,
            'mode': 'demo'
        }
        state_manager.initialize(empty_data)

        # Store in session state
        st.session_state.state_manager = state_manager

        # Initialize A/B testing
        ab_manager = get_ab_test_manager()
        ab_manager.apply_experiment_configs()
        st.session_state.ab_manager = ab_manager

        # Initialize backend service for delta storage
        backend = get_backend_service()
        st.session_state.backend = backend


def check_demo_completion():
    """Check if demo mode is complete (10 turns OR reached final slide 12) and unlock BETA."""
    if st.session_state.get('demo_completed', False):
        return True

    # Check if user is in demo mode and has completed demo
    demo_mode = st.session_state.get('telos_demo_mode', False)
    if demo_mode:
        # Check completion via either method:
        # 1. Interactive Q&A: 10 conversation turns
        # 2. Progressive slides: reached slide 12 (completion slide)
        state_manager = st.session_state.get('state_manager')
        demo_slide_index = st.session_state.get('demo_slide_index', 0)

        completed_via_turns = state_manager and state_manager.state.total_turns >= 10
        completed_via_slides = demo_slide_index >= 12

        if completed_via_turns or completed_via_slides:
            st.session_state.demo_completed = True
            st.success("""
            🎉 **Demo Complete!**

            You've learned the basics of TELOS! The BETA tab is now unlocked.
            Ready to help test TELOS? Switch to the BETA tab to begin.
            """)
            return True

    return False


def check_beta_completion():
    """Check if beta testing is complete and unlock TELOS tab."""
    if not st.session_state.get('beta_consent_given', False):
        return False

    if st.session_state.get('beta_completed', False):
        return True

    from datetime import datetime, timedelta

    start_time_str = st.session_state.get('beta_start_time')
    if not start_time_str:
        return False

    start_time = datetime.fromisoformat(start_time_str)
    elapsed = datetime.now() - start_time
    two_weeks_elapsed = elapsed >= timedelta(days=14)

    feedback_items = st.session_state.get('beta_feedback', [])
    fifty_feedbacks = len(feedback_items) >= 50

    if two_weeks_elapsed or fifty_feedbacks:
        st.session_state.beta_completed = True

        # Export A/B test metrics to backend before completion
        if 'ab_manager' in st.session_state and 'backend' in st.session_state:
            try:
                ab_metrics = st.session_state.ab_manager.export_metrics_for_backend()
                # Store A/B test results
                st.session_state.backend.transmit_delta({
                    'session_id': ab_metrics['session_id'],
                    'turn_number': 999,  # Special marker for A/B test results
                    'fidelity_score': 1.0,
                    'distance_from_pa': 0.0,
                    'mode': 'beta',
                    'ab_test_data': ab_metrics
                })
            except Exception as e:
                print(f"Failed to export A/B test metrics: {e}")

        st.success("""
        🎉 **Beta Testing Complete!**

        Thank you for helping improve TELOS! The TELOS tab is now unlocked.
        You can now use the full TELOS experience without restrictions.
        """)
        return True

    return False


def show_beta_progress():
    """Show beta progress in sidebar."""
    if not st.session_state.get('beta_consent_given', False):
        return

    if st.session_state.get('beta_completed', False):
        return

    from datetime import datetime, timedelta

    start_time_str = st.session_state.get('beta_start_time')
    if not start_time_str:
        return

    start_time = datetime.fromisoformat(start_time_str)
    elapsed = datetime.now() - start_time
    days_elapsed = elapsed.days
    days_remaining = max(0, 14 - days_elapsed)

    feedback_items = st.session_state.get('beta_feedback', [])
    feedback_count = len(feedback_items)
    feedbacks_remaining = max(0, 50 - feedback_count)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Beta Progress")
    st.sidebar.markdown(f"""
    **Completion Criteria** (either one):
    - ⏰ Days: {days_elapsed}/14 ({days_remaining} remaining)
    - 📊 Feedback: {feedback_count}/50 ({feedbacks_remaining} remaining)
    """)


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
    /* Cache buster: v2025-01-08-BETA - Complete rerun button removal */
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

    /* Hide "Rerun" and "Always rerun" buttons - AGGRESSIVE */
    .stApp [data-testid="stStatusWidget"] {{
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        pointer-events: none !important;
        height: 0 !important;
        width: 0 !important;
    }}

    [data-testid="stStatusWidget"] {{
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        pointer-events: none !important;
    }}

    /* Hide the status widget container */
    div[data-testid="stStatusWidget"] {{
        display: none !important;
    }}

    /* Hide toolbar/action menu in top right */
    section[data-testid="stToolbar"] {{
        display: none !important;
        visibility: hidden !important;
    }}

    .stApp > header {{
        display: none !important;
    }}

    /* Hide app rerun related elements */
    [data-testid="stDecoration"] {{
        display: none !important;
    }}

    /* Additional rerun button selectors */
    button[data-testid="stAppRerun"] {{
        display: none !important;
    }}

    button[kind="headerNoPadding"] {{
        display: none !important;
    }}

    /* Hide any toolbar buttons */
    [data-testid="stToolbarActions"] {{
        display: none !important;
    }}

    /* Nuclear option - hide entire toolbar area */
    .stMainBlockContainer + div {{
        display: none !important;
    }}

    /* Hide top toolbar/header area */
    section[data-testid="stHeader"] {{
        display: none !important;
    }}

    /* Hide Deploy button */
    [data-testid="stDeployButton"] {{
        display: none !important;
    }}

    /* Hide the three-dot menu (manage app) */
    button[title="View app menu"] {{
        display: none !important;
    }}

    /* Hide the entire toolbar container that contains Deploy and menu */
    [data-testid="stToolbar"] {{
        display: none !important;
    }}

    /* Additional selector for the menu button */
    .stApp [data-testid="baseButton-headerNoPadding"] {{
        display: none !important;
    }}

    /* Hide the entire top right toolbar area */
    .stApp header {{
        visibility: hidden !important;
        display: none !important;
    }}

    /* Hide file change notification */
    [data-testid="stNotification"] {{
        display: none !important;
    }}

    /* Hide the "Source file changed" banner */
    .stAlert {{
        display: none !important;
    }}

    /* Hide all header buttons including rerun */
    button[kind="headerNoPadding"] {{
        display: none !important;
    }}

    /* Hide the manage app button */
    [data-testid="manage-app-button"] {{
        display: none !important;
    }}

    /* Hide the entire header toolbar */
    .stApp > header + div {{
        display: none !important;
    }}

    /* Hide any rerun-related elements */
    div[data-testid*="rerun"] {{
        display: none !important;
    }}

    /* ULTRA AGGRESSIVE - Hide absolutely everything in header area */
    .main > div:first-child {{
        display: none !important;
    }}

    /* Hide the streamlit header container */
    [data-testid="stHeader"] {{
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
        overflow: hidden !important;
    }}

    /* Hide Deploy button and menu - updated selectors */
    header[data-testid="stHeader"] {{
        display: none !important;
    }}

    /* Hide the entire header element */
    .stApp > header {{
        display: none !important;
    }}

    /* Hide the Deploy button specifically */
    button[data-testid="baseButton-header"] {{
        display: none !important;
    }}

    /* Hide the entire decoration container at top */
    [data-testid="stDecoration"] {{
        display: none !important;
    }}

    /* Hide the banner element that contains Deploy button */
    [role="banner"] {{
        display: none !important;
    }}

    /* Hide any element with baseButton in testid */
    [data-testid*="baseButton"] {{
        display: none !important;
    }}

    /* More aggressive header hiding */
    .main header {{
        display: none !important;
    }}

    /* Hide the app header */
    .appview-container > section:first-child {{
        display: none !important;
    }}

    /* Target the specific rerun message */
    [data-testid="stNotification"] {{
        display: none !important;
    }}

    /* Hide any element containing "Source file changed" text */
    *:has-text("Source file changed") {{
        display: none !important;
    }}

    /* Hide the app header completely */
    .appview-container > section:first-child {{
        display: none !important;
    }}

    /* Force hide all notification-like elements */
    [role="alert"] {{
        display: none !important;
    }}

    /* Main content: dark grey background - more aggressive selectors */
    .stApp {{
        background-color: #1a1a1a !important;
    }}

    .main {{
        background-color: #1a1a1a !important;
    }}

    /* Target the main app view container */
    [data-testid="stAppViewContainer"] {{
        background-color: #1a1a1a !important;
    }}

    /* Target block container */
    .block-container {{
        background-color: #1a1a1a !important;
    }}

    /* Force background on all main sections */
    section.main > div {{
        background-color: #1a1a1a !important;
    }}

    /* Root element */
    #root {{
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
        border: 1px solid #F4D03F !important;
        transition: all 0.3s ease !important;
    }}

    [data-testid="stSidebar"] .stButton > button:hover {{
        background-color: #3d3d3d !important;
        color: #e0e0e0 !important;
        border: 1px solid #F4D03F !important;
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
        border: 1px solid #F4D03F !important;
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
        border: 1px solid #F4D03F;
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
        background-color: #F4D03F !important;
        border-color: #F4D03F !important;
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
        border: 1px solid #F4D03F !important;
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
        background-color: #F4D03F !important;
        border-color: #F4D03F !important;
    }}

    /* Override any red colors in toggle switches */
    [data-baseweb="checkbox"] {{
        background-color: #2d2d2d !important;
    }}

    [data-baseweb="checkbox"][data-checked="true"] {{
        background-color: #F4D03F !important;
    }}

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 10px;
        background-color: #1a1a1a;
    }}

    .stTabs [data-baseweb="tab"] {{
        background-color: #2d2d2d;
        border: 1px solid #F4D03F;
        border-radius: 8px 8px 0 0;
        color: #e0e0e0;
        font-size: 22px;
        font-weight: bold;
        padding: 12px 24px;
    }}

    .stTabs [aria-selected="true"] {{
        background-color: #F4D03F;
        color: #000;
        font-weight: bold;
    }}
    </style>
    """, unsafe_allow_html=True)

    # Keyboard navigation for Demo Mode (arrow keys)
    st.markdown("""
    <script>
    // Demo Mode keyboard navigation
    document.addEventListener('keydown', function(event) {
        // Only in demo mode
        const isDemoMode = window.parent.document.querySelector('[data-testid="stApp"]');
        if (!isDemoMode) return;

        // Arrow key navigation
        switch(event.key) {
            case 'ArrowLeft':
                // Go to previous turn in history
                const scrollBtn = document.querySelector('[key*="scroll_toggle"]');
                if (scrollBtn && !event.ctrlKey && !event.metaKey) {
                    event.preventDefault();
                    scrollBtn.click();
                }
                break;

            case 'ArrowRight':
                // Close history / move forward
                const closeBtn = document.querySelector('[key*="scroll_close"]');
                if (closeBtn && !event.ctrlKey && !event.metaKey) {
                    event.preventDefault();
                    closeBtn.click();
                }
                break;

            case 'ArrowUp':
                // Scroll up through content
                if (!event.ctrlKey && !event.metaKey) {
                    event.preventDefault();
                    window.scrollBy(0, -200);
                }
                break;

            case 'ArrowDown':
                // Scroll down through content
                if (!event.ctrlKey && !event.metaKey) {
                    event.preventDefault();
                    window.scrollBy(0, 200);
                }
                break;
        }
    });
    </script>
    """, unsafe_allow_html=True)

    # Initialize session
    initialize_session()
    state_manager = st.session_state.state_manager

    # Instantiate components
    sidebar_actions = SidebarActionsBeta(state_manager)
    steward_panel = StewardPanel(state_manager)
    conversation_display = ConversationDisplay(state_manager)
    observation_deck = ObservationDeck(state_manager)
    beta_observation_deck = BetaObservationDeck()
    teloscope_controls = TELOSCOPEControls(state_manager)
    beta_onboarding = BetaOnboarding(state_manager)
    pa_onboarding = PAOnboarding()
    observatory_lens = ObservatoryLens(state_manager)

    # Check if user has given beta consent
    has_beta_consent = st.session_state.get('beta_consent_given', False)

    # Check completion status (show celebrations if just completed)
    check_demo_completion()  # Check demo completion (10 turns)
    if has_beta_consent:
        check_beta_completion()  # Check beta completion (50 feedbacks or 2 weeks)

    # Hide sidebar if Steward panel is open
    steward_panel.hide_sidebar_when_open()

    # Render Steward button (always visible after consent)
    if has_beta_consent:
        steward_panel.render_button()

    # Initialize active tab if not set - default to DEMO for public users
    if 'active_tab' not in st.session_state:
        # Check for admin mode or beta direct access
        query_params = st.query_params
        is_admin = query_params.get("admin") == "true"
        beta_direct = query_params.get("beta") == "true"

        if is_admin:
            st.session_state.active_tab = "DEVOPS"
        elif beta_direct:
            # Auto-grant consent for direct beta access (for grant demos)
            st.session_state.beta_consent_given = True
            st.session_state.active_tab = "BETA"
        else:
            st.session_state.active_tab = "DEMO"

    # DEMO tab is always accessible (no consent required)
    # BETA and TELOS tabs require consent
    # DEVOPS bypasses all restrictions for testing
    active_tab = st.session_state.active_tab

    # Check for admin mode or beta direct access
    query_params = st.query_params
    is_admin = query_params.get("admin") == "true"
    beta_direct = query_params.get("beta") == "true"

    # If user is trying to access BETA or TELOS without consent, show consent screen
    # DEVOPS mode, admin mode, and beta direct mode bypass consent requirement
    if (active_tab in ["BETA", "TELOS"]) and not has_beta_consent and not is_admin and not beta_direct:
        # Show consent screen for BETA/TELOS access
        beta_onboarding.render()
    else:
        # Render tabs and content (DEMO is always accessible, BETA/TELOS require consent, DEVOPS is unrestricted)
        render_tabs_and_content(has_beta_consent, state_manager, sidebar_actions,
                                conversation_display, observation_deck, beta_observation_deck,
                                teloscope_controls, steward_panel, beta_onboarding,
                                pa_onboarding, observatory_lens)


def render_tabs_and_content(has_beta_consent, state_manager, sidebar_actions,
                            conversation_display, observation_deck, beta_observation_deck,
                            teloscope_controls, steward_panel, beta_onboarding,
                            pa_onboarding, observatory_lens):
    """Render tabs and main content area."""
    # Check if we're in TELOS or DEVOPS mode (which have sidebar access)
    active_tab = st.session_state.get('active_tab', 'DEMO')
    sidebar_accessible = active_tab in ['TELOS', 'DEVOPS']

    # Only render sidebar and tabs styling if beta consent given OR in DEVOPS mode
    if has_beta_consent or sidebar_accessible:
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
                box-shadow: 0 0 8px #F4D03F;
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

        # Check if sidebar should be enabled (TELOS and DEVOPS modes)
        sidebar_enabled = st.session_state.get('active_tab') in ['TELOS', 'DEVOPS']

        if not sidebar_enabled:
            # Gray out sidebar in DEMO and BETA modes
            st.markdown("""
            <style>
            /* Gray out sidebar in DEMO and BETA modes */
            [data-testid="stSidebar"] {
                opacity: 0.3 !important;
                pointer-events: none !important;
            }

            /* Disable all sidebar interactions */
            [data-testid="stSidebar"] * {
                pointer-events: none !important;
                cursor: not-allowed !important;
            }
            </style>
            """, unsafe_allow_html=True)

        # Render sidebar
        sidebar_actions.render()

        # Show beta progress in sidebar
        show_beta_progress()

        # Get current demo mode setting
        demo_mode = st.session_state.get('telos_demo_mode', False)

        # Progressive unlock system
        demo_complete = st.session_state.get('demo_completed', False)
        beta_complete = st.session_state.get('beta_completed', False)

        # Simple single-content approach: show content based on selected tab via radio buttons
        st.markdown("<div style='margin: 20px 0;'></div>", unsafe_allow_html=True)

        # Active tab styling
        st.markdown(f"""
        <style>
        /* Active tab - just bright gold border, no fill */
        button[kind="primary"] {{
            background-color: #2d2d2d !important;
            color: #e0e0e0 !important;
            border: 2px solid #F4D03F !important;
            box-shadow: 0 0 8px rgba(255, 215, 0, 0.5) !important;
        }}

        /* When hovering over active tab, dim the border */
        button[kind="primary"]:hover {{
            background-color: #3d3d3d !important;
            color: #e0e0e0 !important;
            border: 1px solid #F4D03F !important;
            box-shadow: 0 0 6px #F4D03F !important;
        }}

        /* Inactive tabs - normal thin border */
        button[kind="secondary"] {{
            background-color: #2d2d2d !important;
            color: #e0e0e0 !important;
            border: 1px solid #F4D03F !important;
        }}

        /* Inactive tabs get hover effect */
        button[kind="secondary"]:hover {{
            background-color: #3d3d3d !important;
            border: 1px solid #F4D03F !important;
            box-shadow: 0 0 6px #F4D03F !important;
        }}

        /* Disabled/locked tabs - look like normal secondary buttons but with dimmed text */
        button[disabled],
        button.beta-locked {{
            background-color: #2d2d2d !important;
            color: #888 !important;
            border: 1px solid #F4D03F !important;
            cursor: not-allowed !important;
        }}

        button[disabled]:hover,
        button.beta-locked:hover {{
            background-color: #1a1a1a !important;
            color: #555 !important;
            border: 1px solid #444 !important;
            box-shadow: none !important;
            transform: none !important;
        }}
        </style>
        """, unsafe_allow_html=True)

    # Tab selection using columns for custom styling
    # Get current active tab
    active_tab = st.session_state.active_tab

    # Check demo completion BEFORE reading the state
    check_demo_completion()

    # Progressive unlock system
    demo_complete = st.session_state.get('demo_completed', False)
    beta_complete = st.session_state.get('beta_completed', False)

    # Simple single-content approach: show content based on selected tab via radio buttons
    st.markdown("<div style='margin: 20px 0;'></div>", unsafe_allow_html=True)

    # PUBLIC UI: 3 tabs only (DEMO/BETA/TELOS)
    # DEVOPS mode is admin-only, accessible via URL parameter: ?admin=true

    # Check for admin access via query params FIRST
    query_params = st.query_params
    is_admin = query_params.get("admin") == "true"

    # Check if user has entered BETA (either completed DEMO or accepted BETA consent)
    beta_entered = st.session_state.get('beta_consent_given', False) or st.session_state.get('pa_established', False)

    if beta_entered:
        # BETA mode: 2 tabs (BETA and TELOS)
        col_beta, col_telos = st.columns(2)

        with col_beta:
            beta_active = active_tab == "BETA"
            if st.button("BETA", key="tab_beta", use_container_width=True,
                        type="primary" if beta_active else "secondary"):
                st.session_state.active_tab = "BETA"
                st.rerun()

        with col_telos:
            telos_active = active_tab == "TELOS"
            # TELOS unlocks after completing BETA (15 turns) OR in admin mode
            telos_locked = not beta_complete and not is_admin
            if st.button("TELOS", key="tab_telos", use_container_width=True,
                        type="primary" if telos_active else "secondary",
                        disabled=telos_locked,
                        help="Complete BETA mode to unlock TELOS" if telos_locked else None):
                if not telos_locked:
                    st.session_state.active_tab = "TELOS"
                    st.rerun()
    else:
        # DEMO mode: 2 tabs (DEMO and BETA)
        col_demo, col_beta = st.columns(2)

        with col_demo:
            demo_active = active_tab == "DEMO"
            # DEMO is always available - starting point for everyone
            if st.button("DEMO", key="tab_demo", use_container_width=True,
                        type="primary" if demo_active else "secondary"):
                st.session_state.active_tab = "DEMO"
                st.rerun()

        with col_beta:
            beta_active = active_tab == "BETA"
            # BETA unlocks after completing demo (10 turns) OR in admin mode
            beta_locked = not demo_complete and not is_admin  # Bypass lock in admin mode
            if st.button("BETA", key="tab_beta", use_container_width=True,
                        type="primary" if beta_active else "secondary",
                        disabled=beta_locked,
                        help="Complete DEMO mode to unlock BETA" if beta_locked else None):
                if not beta_locked:
                    st.session_state.active_tab = "BETA"
                    st.rerun()

    if is_admin:
        st.markdown("<div style='margin: 10px 0;'></div>", unsafe_allow_html=True)
        if st.button("🔧 DEVOPS (Admin Mode)", key="tab_devops", use_container_width=True,
                    type="primary" if active_tab == "DEVOPS" else "secondary"):
            st.session_state.active_tab = "DEVOPS"
            st.rerun()

    # Removed unlock progression message - now shown in Steward intro

    st.markdown("<hr style='border: 1px solid #F4D03F; margin: 10px 0;'>", unsafe_allow_html=True)

    # Hide sidebar for DEMO and BETA modes - only show in TELOS and DEVOPS
    active_tab = st.session_state.get('active_tab', 'DEMO')
    if active_tab not in ['TELOS', 'DEVOPS']:
        st.markdown("""
        <style>
        /* Hide sidebar before TELOS/DEVOPS modes */
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

    # Unified rendering function - single master build with mode-based feature flags
    def render_mode_content(mode: str):
        """Unified content rendering for all modes with feature flags.

        Args:
            mode: One of DEMO, BETA, TELOS, DEVOPS
        """
        # Set demo mode flag based on current mode
        st.session_state.telos_demo_mode = (mode == "DEMO")

        # Mode-specific features
        show_devops_header = (mode == "DEVOPS")
        show_observation_deck = (mode in ["BETA", "TELOS", "DEVOPS"])
        show_teloscope = (mode in ["TELOS", "DEVOPS"])

        # DEVOPS header
        if show_devops_header:
            st.markdown("### 🔧 DEVOPS Mode - Full System Access")
            st.markdown("**All restrictions removed. Beta mode with full PA extraction and interventions enabled.**")

        # BETA mode: Show PA questionnaire first if not established
        if mode == "BETA":
            # Check if PA is already established
            if not st.session_state.get('pa_established', False):
                # Show PA questionnaire and block conversation until complete
                pa_answers = pa_onboarding.render_questionnaire()
                if pa_answers is None:
                    # PA not established yet, stop rendering (questionnaire is showing)
                    return
                # If we get here, PA was just established - continue to conversation

            # Show BETA welcome message if PA just established and no turns yet
            if st.session_state.get('pa_established', False) and st.session_state.get('beta_current_turn', 1) == 1:
                if 'conversation_turns' not in st.session_state or len(st.session_state.conversation_turns) == 0:
                    welcome_html = f"""
<div style="background: linear-gradient(135deg, #2d2d2d 0%, #1a1a1a 100%); border: 3px solid {GOLD}; border-radius: 15px; padding: 30px; margin: 20px 0;">
    <div style="text-align: center; color: {GOLD}; font-size: 28px; font-weight: bold; margin-bottom: 20px;">
        Welcome to BETA Testing
    </div>
    <div style="color: #e0e0e0; font-size: 18px; line-height: 1.8;">
        <div style="margin-bottom: 20px;">
            Your <strong style="color: {GOLD};">Primacy Attractor</strong> has been established.
            You're about to experience a 15-turn conversation where we'll test how well AI stays
            aligned with your stated purpose.
        </div>
        <div style="background-color: #1a1a1a; padding: 20px; border-radius: 10px; margin: 20px 0;">
            <div style="color: {GOLD}; font-size: 20px; font-weight: bold; margin-bottom: 15px;">
                Here's What Will Happen:
            </div>
            <div style="margin-bottom: 15px;">
                <strong style="color: {GOLD};">15 Conversation Turns</strong><br>
                You'll have a natural conversation over 15 turns. Just ask questions or make requests
                as you normally would.
            </div>
            <div style="margin-bottom: 15px;">
                <strong style="color: {GOLD};">Two Types of Responses</strong><br>
                • Some turns will show <strong>one response</strong> (you won't know the source)<br>
                • Other turns will show <strong>two responses side-by-side</strong> (you'll choose your preference)
            </div>
            <div style="margin-bottom: 15px;">
                <strong style="color: {GOLD};">Monitor Your Alignment</strong><br>
                Click the <strong>Observation Deck</strong> (below) at any time to view your Primacy Attractor
                and fidelity score. Your fidelity indicates if the system believes you may be straying
                off-topic from your stated purpose.
            </div>
            <div style="margin-bottom: 15px;">
                <strong style="color: {GOLD};">After 15 Turns</strong><br>
                You'll unlock the <strong>TELOS tab</strong> for full governed conversations with progressive
                Primacy Attractor mode (not onboarding mode like BETA). You can continue in the existing
                interface or start a fresh TELOS chat.
            </div>
        </div>
        <div style="background-color: #2d2d2d; padding: 20px; border-radius: 10px; margin: 20px 0; border: 2px solid {GOLD};">
            <div style="color: {GOLD}; font-size: 20px; font-weight: bold; margin-bottom: 15px;">
                🤝 IMPORTANT: Use Steward for TELOS Questions
            </div>
            <div style="color: #e0e0e0; margin-bottom: 15px;">
                <strong style="color: {GOLD};">Steward (🤝 handshake icon)</strong> is your TELOS guide.
                Click the handshake icon to ask questions about the system, get help navigating,
                or understand what you're seeing.
            </div>
            <div style="color: #e0e0e0;">
                <strong style="color: {GOLD};">Main chat window</strong> is a standard LLM with
                <strong>zero context about TELOS</strong> the system. Use it for your normal tasks,
                but direct all TELOS-related questions to Steward.
            </div>
        </div>
        <div style="text-align: center; margin-top: 25px; padding: 20px; background-color: #1a1a1a; border-radius: 10px;">
            <div style="color: {GOLD}; font-size: 20px; font-weight: bold; margin-bottom: 10px;">
                Ready to Begin?
            </div>
            <div style="color: #e0e0e0; font-size: 16px;">
                Start by asking a question or making a request in the chat below.<br>
                <span style="color: #888; font-size: 14px; margin-top: 10px; display: block;">
                    (Remember: Use 🤝 Steward for TELOS help, not the main chat)
                </span>
            </div>
        </div>
    </div>
</div>
"""
                    st.markdown(welcome_html, unsafe_allow_html=True)

        # Main conversation display (all modes)
        conversation_display.render()

        # Visualization Tools Toggle Buttons (DEMO and BETA modes)
        # Progressive button display based on slide for DEMO mode
        demo_slide_index = st.session_state.get('demo_slide_index', 0)

        if mode == "DEMO":
            # DEMO mode has custom navigation on each slide
            # Slide 4 has embedded observation deck
            # Slide 8 has custom Alignment Lens controls for drift detection
            # All other slides have Previous/Next buttons
            # No three-button bar needed in DEMO mode
            pass

        elif mode == "BETA":
            # In BETA mode, show simplified Observation Deck (no complex tools)
            # No toggle buttons - Observation Deck is always available
            pass

        # Observation Deck (different rendering based on mode)
        if mode == "BETA":
            # BETA mode uses simplified BetaObservationDeck (always visible if PA established)
            if st.session_state.get('pa_established', False):
                st.markdown("<div style='margin: 40px 0;'></div>", unsafe_allow_html=True)
                beta_observation_deck.render()
        elif mode == "DEMO":
            # For DEMO, don't render the main observation deck on slide 4
            # (it has its own embedded observation deck in the demo)
            if st.session_state.get('demo_slide_index', 0) == 4:
                # Skip rendering - demo slide 4 has its own observation deck
                pass
            elif st.session_state.get('show_observation_deck', False):
                # For DEMO (other slides), only show if toggled on
                st.markdown("<div style='margin: 40px 0;'></div>", unsafe_allow_html=True)
                observation_deck.render()
        elif show_observation_deck:
            # For TELOS and DEVOPS, always show full observation deck
            st.markdown("<div style='margin: 40px 0;'></div>", unsafe_allow_html=True)
            observation_deck.render()

        # TELOSCOPE Controls (TELOS, DEVOPS)
        if show_teloscope:
            st.markdown("<div style='margin: 5px 0;'></div>", unsafe_allow_html=True)
            teloscope_controls.render()

        # Alignment Lens (show if toggled or in sidebar mode)
        if st.session_state.get('show_observatory_lens', False):
            st.markdown("<div style='margin: 20px 0;'></div>", unsafe_allow_html=True)
            observatory_lens.render()

    # Check if Steward panel is open
    steward_open = st.session_state.get('steward_panel_open', False)

    # Content rendering (DEMO accessible without consent, BETA/TELOS require consent)
    if steward_open and has_beta_consent:
        # Two-column layout: Main content (70%) | Steward chat (30%)
        col_main, col_steward = st.columns([7, 3])

        with col_main:
            render_mode_content(st.session_state.active_tab)

        with col_steward:
            # Render Steward chat panel
            steward_panel.render_panel()

    else:
        # Normal full-width layout
        render_mode_content(st.session_state.active_tab)

    # FINAL CSS OVERRIDE - Inject with highest specificity at runtime
    st.html("""
    <style>
    /* Runtime CSS injection - v20:35 - Further reduced glow */

    /* PRIMARY BUTTONS - Active tab styling (just border, no bright background) */
    button[kind="primary"] {
        background-color: #2d2d2d !important;
        background: #2d2d2d !important;
        color: #e0e0e0 !important;
        border: 2px solid #F4D03F !important;
        box-shadow: 0 0 8px rgba(255, 215, 0, 0.5) !important;
    }

    /* Override Streamlit's default primary button background */
    .stButton > button[kind="primary"],
    button[data-baseweb="button"][kind="primary"] {
        background-color: #2d2d2d !important;
        background: #2d2d2d !important;
    }

    button:hover {
        border: 1px solid #F4D03F !important;
        box-shadow: 0 0 6px #F4D03F !important;
    }

    /* Message container hover glow */
    .message-container:hover {
        box-shadow: 0 0 6px #F4D03F !important;
        transition: box-shadow 0.3s ease !important;
    }
    </style>
    """)


if __name__ == "__main__":
    main()
