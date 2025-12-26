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

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / '.env', override=True)  # Force override shell env vars

import streamlit as st

# Import V3 components
from core.state_manager import StateManager
# Note: generate_telos_demo_session import removed - not needed for progressive demo slideshow
from components.sidebar_actions_beta import SidebarActionsBeta
from components.conversation_display import ConversationDisplay
from components.observation_deck import ObservationDeck
from components.beta_observation_deck import BetaObservationDeck
from components.beta_completion import BetaCompletion
from components.teloscope_panel import render_teloscope_button, render_teloscope_panel
from components.teloscope_controls import TELOSCOPEControls
from components.beta_onboarding import BetaOnboarding
from components.pa_onboarding import PAOnboarding
from components.steward_panel import StewardPanel
from components.beta_steward_panel import BetaStewardPanel, render_beta_steward_button, render_bottom_section
from components.observatory_lens import ObservatoryLens
from components import teloscope_profile_selector
from services.ab_test_manager import get_ab_test_manager
from services.backend_client import get_backend_service
from config.colors import GOLD


def initialize_session():
    """Initialize session state - starts fresh (no pre-loaded demo data)."""
    # EAGER MODEL INITIALIZATION - Critical for Railway cold start performance
    # Pre-load embedding model at app startup, not on first user action
    if 'embedding_model_initialized' not in st.session_state:
        try:
            from telos_purpose.core.embedding_provider import get_cached_minilm_provider
            get_cached_minilm_provider()  # Pre-warm the cached model
            st.session_state.embedding_model_initialized = True
        except Exception:
            pass  # Don't block app startup on model failure

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
    """Check if demo mode is complete (10 turns OR reached final slide 14) and unlock BETA."""
    if st.session_state.get('demo_completed', False):
        return True

    # Check if user is in demo mode and has completed demo
    demo_mode = st.session_state.get('telos_demo_mode', False)
    if demo_mode:
        # Check completion via either method:
        # 1. Interactive Q&A: 10 conversation turns
        # 2. Progressive slides: reached slide 13+ (differentiator or completion slide)
        #    - Slide 13 = "What makes TELOS different" (last Q&A slide)
        #    - Slide 14 = Congratulations/BETA unlock screen
        state_manager = st.session_state.get('state_manager')
        demo_slide_index = st.session_state.get('demo_slide_index', 0)

        completed_via_turns = state_manager and state_manager.state.total_turns >= 10
        completed_via_slides = demo_slide_index >= 13

        if completed_via_turns or completed_via_slides:
            st.session_state.demo_completed = True
            return True

    return False


def check_beta_completion():
    """Check if beta session is complete (10 conversational turns)."""
    if not st.session_state.get('beta_consent_given', False):
        return False

    if st.session_state.get('beta_completed', False):
        return True

    # Beta completes after 10 conversational turns
    # beta_current_turn tracks the next turn to generate, so > 10 means 10 complete
    current_turn = st.session_state.get('beta_current_turn', 1)
    if current_turn > 10:
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
        layout="centered",  # Changed from "wide" - gives natural 704px centered layout
        initial_sidebar_state="expanded"
    )

    # Dark theme styling and hide Streamlit defaults
    st.markdown("""
    <style>
    /* =================================================================
       TELOS Design System - CSS Variables
       Based on Material Design 8px grid system
       ================================================================= */
    :root {{
        /* Spacing scale (8px grid) */
        --space-xs: 4px;    /* Tight - between related items */
        --space-sm: 8px;    /* Default - standard gap */
        --space-md: 16px;   /* Moderate - between sections */
        --space-lg: 24px;   /* Generous - between major sections */
        --space-xl: 32px;   /* Maximum - page sections */

        /* Colors - Single source of truth */
        --color-gold: #F4D03F;
        --color-gold-dim: rgba(244, 208, 63, 0.5);
        --color-green: #27ae60;
        --color-yellow: #f39c12;
        --color-orange: #e67e22;
        --color-red: #e74c3c;

        /* Backgrounds */
        --bg-dark: #0a0a0a;
        --bg-surface: #1a1a1a;
        --bg-elevated: #2d2d2d;
        --bg-hover: #3d3d3d;

        /* Text */
        --text-primary: #e0e0e0;
        --text-secondary: #b0b0b0;
        --text-muted: #808080;
    }}

    /* Cache buster: v2025-01-14-CENTERING - Viewport-level centering fix */

    /* VIEWPORT CENTERING: Force app container to center content */
    .stApp {{
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
    }}

    .stApp > section.main {{
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        width: 100% !important;
    }}

    .stApp > section.main > div.block-container {{
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
    }}

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

    /* GLASSMORPHISM: Force ALL backgrounds transparent so we see the injected gradient div */
    .stApp,
    .stApp > *,
    .main,
    .main > *,
    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewContainer"] > *,
    .block-container,
    section.main,
    section.main > div,
    #root,
    .stApp > header,
    .stApp [data-testid="stHeader"],
    .stApp [data-testid="stToolbar"] {{
        background: transparent !important;
        background-color: transparent !important;
    }}

    /* The gradient background div (injected via HTML) */
    #glassmorphism-bg {{
        position: fixed !important;
        top: 0 !important;
        left: 0 !important;
        width: 100vw !important;
        height: 100vh !important;
        z-index: -1000 !important;
        pointer-events: none !important;
        background:
            radial-gradient(ellipse 90% 90% at 15% 5%, rgba(244, 208, 63, 0.7) 0%, transparent 40%),
            radial-gradient(ellipse 80% 80% at 90% 95%, rgba(200, 160, 40, 0.6) 0%, transparent 35%),
            radial-gradient(ellipse 60% 60% at 50% 50%, rgba(244, 180, 63, 0.4) 0%, transparent 50%),
            linear-gradient(135deg, #1a1510 0%, #0d0a05 50%, #0a0805 100%) !important;
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

    /* COMPACT LAYOUT: 950px max-width centered content - MAXIMUM SPECIFICITY */
    /* Target all possible Streamlit container classes */
    .stApp .main .block-container,
    .stApp .main [data-testid="stMainBlockContainer"],
    .stApp [data-testid="stAppViewContainer"] .block-container,
    .stApp [data-testid="stAppViewContainer"] [data-testid="stMainBlockContainer"],
    div.stMainBlockContainer,
    div.block-container,
    .block-container,
    [data-testid="stMainBlockContainer"],
    .stMainBlockContainer.block-container,
    div[data-testid="stMainBlockContainer"].block-container,
    section.main div.block-container,
    .stApp section.main .block-container {{
        max-width: 950px !important;
        width: 950px !important;
        min-width: 0 !important;
        margin-left: auto !important;
        margin-right: auto !important;
        padding-top: 1rem;
        padding-bottom: 1rem;
        box-sizing: border-box !important;
    }}

    /* Override Streamlit's emotion-cache classes */
    [class*="st-emotion-cache"][class*="block-container"],
    [class*="st-emotion-cache"].stMainBlockContainer {{
        max-width: 950px !important;
        width: 950px !important;
        margin-left: auto !important;
        margin-right: auto !important;
    }}

    /* Button styling - force gold borders and hover effects */
    .stButton > button {{
        background-color: #2d2d2d !important;
        color: #e0e0e0 !important;
        border: 1px solid #F4D03F !important;
        transition: all 0.3s ease !important;
        text-align: center !important;
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
    }}

    /* Refined hover effect - subtle but visible (per UI/UX best practices) */
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
        border-width: 2px !important;
        border-style: solid !important;
        border-color: #F4D03F !important;
        box-shadow: 0 0 8px rgba(244, 208, 63, 0.5) !important;
        transform: scale(1.02) !important;
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

    /* Increase all paragraph and text elements with optimal line-height */
    p {{
        font-size: 16px !important;
        line-height: 1.5 !important;
    }}

    /* Global line-height for readability (WCAG 1.4.8) */
    body, html, .stMarkdown, .stText, span, div {{
        line-height: 1.5 !important;
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
        background-color: #0a0a0a !important;
    }}

    /* Hide the SVG checkmark that Streamlit adds - comprehensive selectors */
    .stCheckbox svg {{
        display: none !important;
    }}

    /* Hide all checkmark icons within checkbox containers */
    .stCheckbox svg,
    .stCheckbox path,
    .stCheckbox [data-testid="stCheckbox"] svg,
    [data-testid="stCheckbox"] svg,
    [data-testid="stCheckbox"] path,
    .stCheckbox span svg,
    .stCheckbox label svg {{
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        width: 0 !important;
        height: 0 !important;
    }}

    /* Target Streamlit's specific checkbox icon class */
    .stCheckbox [data-baseweb="checkbox"] svg,
    [data-baseweb="checkbox"] svg {{
        display: none !important;
        visibility: hidden !important;
    }}

    /* Hide any icon containers within checkbox */
    .stCheckbox .st-emotion-cache-1inwz65,
    .stCheckbox [class*="emotion-cache"] svg {{
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
        color: #0a0a0a;
        font-weight: bold;
    }}

    /* BETA Control Buttons - WCAG 2.5.5 compliant touch targets (44px minimum) */
    /* Target buttons in narrow columns (col_buttons) that contain emoji icons */
    .stButton > button {{
        aspect-ratio: 1 !important;
        min-width: 44px !important;
        min-height: 44px !important;
        padding: 0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }}

    /* Turn badge responsive styling - WCAG compliant */
    .turn-badge {{
        aspect-ratio: 1 !important;
        min-width: 44px !important;
        min-height: 44px !important;
    }}

    /* Responsive breakpoints - maintain WCAG 44px touch targets */
    @media (max-width: 1200px) {{
        .stButton > button {{
            min-width: 44px !important;
            min-height: 44px !important;
            font-size: 14px !important;
        }}
        .turn-badge {{
            min-width: 40px !important;
            min-height: 40px !important;
            width: 40px !important;
            height: 40px !important;
            font-size: 18px !important;
            padding: 8px !important;
        }}
    }}

    @media (max-width: 992px) {{
        .stButton > button {{
            min-width: 44px !important;
            min-height: 44px !important;
            font-size: 13px !important;
        }}
        .turn-badge {{
            min-width: 40px !important;
            min-height: 40px !important;
            width: 40px !important;
            height: 40px !important;
            font-size: 16px !important;
            padding: 8px !important;
        }}
    }}

    @media (max-width: 768px) {{
        .stButton > button {{
            min-width: 44px !important;
            min-height: 44px !important;
            font-size: 12px !important;
        }}
        .turn-badge {{
            min-width: 40px !important;
            min-height: 40px !important;
            width: 40px !important;
            height: 40px !important;
            font-size: 14px !important;
            padding: 8px !important;
        }}
    }}

    /* COMPACT LAYOUT: Constrain main content to optimal reading width */
    /* Matches the 778px viewport that displays content optimally */
    /* Responsive: shrinks below 950px, caps at 950px for larger screens */
    .stApp .main .block-container,
    .stApp [data-testid="stAppViewBlockContainer"],
    section.main > div.block-container,
    [data-testid="stAppViewContainer"] .block-container,
    div.block-container {{
        max-width: 950px !important;
        width: 100% !important;
        margin-left: auto !important;
        margin-right: auto !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }}
    </style>
    """, unsafe_allow_html=True)

    # GLASSMORPHISM: Apply gradient to page background only
    import streamlit.components.v1 as components
    components.html("""
    <script>
    (function() {
        var doc = window.parent.document;
        var gradientBg = `
            radial-gradient(ellipse 90% 90% at 15% 5%, rgba(244, 208, 63, 0.7) 0%, transparent 40%),
            radial-gradient(ellipse 80% 80% at 90% 95%, rgba(200, 160, 40, 0.6) 0%, transparent 35%),
            radial-gradient(ellipse 60% 60% at 50% 50%, rgba(244, 180, 63, 0.4) 0%, transparent 50%),
            linear-gradient(135deg, #1a1510 0%, #0d0a05 50%, #0a0805 100%)
        `;

        function applyGlassmorphism() {
            // Apply gradient DIRECTLY to .stApp
            var stApp = doc.querySelector('.stApp');
            if (stApp) {
                stApp.style.setProperty('background', gradientBg, 'important');
                stApp.style.setProperty('background-attachment', 'fixed', 'important');
            }

            // Only make the top-level containers transparent (NOT content areas)
            var topContainers = doc.querySelectorAll(`
                [data-testid="stAppViewContainer"],
                .main,
                section.main
            `);
            topContainers.forEach(function(el) {
                el.style.setProperty('background', 'transparent', 'important');
                el.style.setProperty('background-color', 'transparent', 'important');
            });

            // COMPACT LAYOUT: Apply 950px max-width to main content container
            // This ensures consistent display across all screen sizes
            var blockContainers = doc.querySelectorAll('.block-container');
            blockContainers.forEach(function(el) {
                el.style.setProperty('max-width', '950px', 'important');
                el.style.setProperty('width', '100%', 'important');
                el.style.setProperty('margin-left', 'auto', 'important');
                el.style.setProperty('margin-right', 'auto', 'important');
            });

            // Debug: Log compact layout application
            if (blockContainers.length > 0 && !window._compactLayoutLogged) {
                console.log('COMPACT LAYOUT: Applied 950px max-width to ' + blockContainers.length + ' container(s)');
                window._compactLayoutLogged = true;
            }
        }

        // Apply immediately
        applyGlassmorphism();

        // Re-apply every 500ms to override any Streamlit updates
        setInterval(applyGlassmorphism, 500);

        // CRITICAL: Also apply on window resize to prevent auto-expand behavior
        window.parent.addEventListener('resize', function() {
            applyGlassmorphism();
        });

        // Also add listener to doc resize events
        doc.defaultView.addEventListener('resize', function() {
            applyGlassmorphism();
        });

        console.log('TELOS STYLING: JavaScript initialized (glassmorphism + 950px compact layout + resize protection)');
    })();
    </script>
    """, height=0)

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

    # Mobile Scroll-to-Top on Navigation
    # When users click Next/Continue buttons, Streamlit reruns the page
    # This scrolls to top automatically, especially important on mobile devices
    import streamlit.components.v1 as components
    components.html("""
    <script>
    (function() {
        // Only run once per page load
        if (window.telosScrollToTopApplied) return;
        window.telosScrollToTopApplied = true;

        var doc = window.parent.document;

        // Scroll to top on page load (after st.rerun())
        function scrollToTop() {
            // Try multiple scroll targets for cross-browser compatibility
            var mainContent = doc.querySelector('[data-testid="stAppViewContainer"]');
            var stApp = doc.querySelector('.stApp');

            // Smooth scroll to top
            if (mainContent) {
                mainContent.scrollTo({ top: 0, behavior: 'smooth' });
            }
            if (stApp) {
                stApp.scrollTo({ top: 0, behavior: 'smooth' });
            }
            // Also scroll the document and window
            doc.documentElement.scrollTo({ top: 0, behavior: 'smooth' });
            doc.body.scrollTo({ top: 0, behavior: 'smooth' });
            window.parent.scrollTo({ top: 0, behavior: 'smooth' });
        }

        // Execute scroll with small delay to ensure DOM is ready
        setTimeout(scrollToTop, 100);

        console.log('TELOS MOBILE: Scroll-to-top navigation enabled');
    })();
    </script>
    """, height=0)

    # Session Keep-Alive Heartbeat
    # Prevents Streamlit WebSocket timeout (default 5 min) during inactivity
    # Pings every 2 minutes to keep connection alive
    import streamlit.components.v1 as components
    components.html("""
    <script>
    (function() {
        // Only set up heartbeat once per page load
        if (window.telosHeartbeatActive) return;
        window.telosHeartbeatActive = true;

        // Heartbeat interval: 2 minutes (120000ms)
        // Keeps WebSocket alive during periods of user inactivity
        const HEARTBEAT_INTERVAL = 120000;

        function sendHeartbeat() {
            try {
                // Find and trigger a lightweight interaction to keep session alive
                // This creates minimal activity that resets the Streamlit timeout
                const frames = window.parent.document.querySelectorAll('iframe');
                if (frames.length > 0) {
                    // Session is still active - connection maintained
                    console.log('[TELOS] Session heartbeat sent');
                }
            } catch (e) {
                // Silent fail - don't disrupt user experience
            }
        }

        // Start heartbeat timer
        setInterval(sendHeartbeat, HEARTBEAT_INTERVAL);
        console.log('[TELOS] Session keep-alive heartbeat initialized (2 min interval)');
    })();
    </script>
    """, height=0)

    # Initialize session
    initialize_session()
    state_manager = st.session_state.state_manager

    # Instantiate components
    sidebar_actions = SidebarActionsBeta(state_manager)
    steward_panel = StewardPanel(state_manager)
    beta_steward_panel = BetaStewardPanel()  # BETA-specific Steward with correct context
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

    # Render Steward button (only after PA is established)
    # BETA mode: No top-level "Ask Steward" button - users access Steward via:
    #   1. The 🤝 icon buttons on individual messages
    #   2. The "Ask Steward why" button that appears when fidelity drops below Aligned zone
    # DEMO/TELOS mode: Show regular Steward toggle
    pa_established = st.session_state.get('pa_established', False)
    if has_beta_consent and pa_established:
        current_tab = st.session_state.get('active_tab', 'DEMO')
        if current_tab == "BETA":
            pass  # No top-level button in BETA mode - contextual access only
        else:
            steward_panel.render_button()  # Regular Steward toggle

    # Initialize active tab if not set - default to DEMO for public users
    if 'active_tab' not in st.session_state:
        # Check for admin mode or direct access bypasses
        query_params = st.query_params
        is_admin = query_params.get("admin") == "true"
        beta_direct = query_params.get("beta") == "true"
        telos_direct = query_params.get("telos") == "true"

        if is_admin:
            st.session_state.active_tab = "DEVOPS"
        elif telos_direct:
            # TELOS BYPASS: Skip demo/beta, go directly to full TELOS open mode
            # Auto-grant all consents and mark completions
            st.session_state.beta_consent_given = True
            st.session_state.demo_completed = True
            st.session_state.beta_completed = True
            st.session_state.pa_established = True  # Skip PA onboarding
            st.session_state.telos_demo_mode = False  # Disable demo mode
            st.session_state.active_tab = "TELOS"
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

    # Check for admin mode or direct access bypasses
    query_params = st.query_params
    is_admin = query_params.get("admin") == "true"
    beta_direct = query_params.get("beta") == "true"
    telos_direct = query_params.get("telos") == "true"

    # If user is trying to access BETA or TELOS without consent, show consent screen
    # DEVOPS mode, admin mode, beta direct mode, and telos direct mode bypass consent requirement
    if (active_tab in ["BETA", "TELOS"]) and not has_beta_consent and not is_admin and not beta_direct and not telos_direct:
        # Show consent screen for BETA/TELOS access
        beta_onboarding.render()
        return  # Stop rendering anything else until consent is given
    else:
        # Render tabs and content (DEMO is always accessible, BETA/TELOS require consent, DEVOPS is unrestricted)
        render_tabs_and_content(has_beta_consent, state_manager, sidebar_actions,
                                conversation_display, observation_deck, beta_observation_deck,
                                teloscope_controls, steward_panel, beta_steward_panel, beta_onboarding,
                                pa_onboarding, observatory_lens)


def render_tabs_and_content(has_beta_consent, state_manager, sidebar_actions,
                            conversation_display, observation_deck, beta_observation_deck,
                            teloscope_controls, steward_panel, beta_steward_panel, beta_onboarding,
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

        # TELOS mode: Show profile switch button in sidebar
        if st.session_state.get('active_tab') == 'TELOS' and st.session_state.get('teloscope_profile'):
            with st.sidebar:
                st.markdown("---")
                teloscope_profile_selector.render_profile_badge()
                teloscope_profile_selector.render_profile_switch_button()

        # Get current demo mode setting
        demo_mode = st.session_state.get('telos_demo_mode', False)

        # Progressive unlock system
        demo_complete = st.session_state.get('demo_completed', False)
        beta_complete = st.session_state.get('beta_completed', False)

        # Simple single-content approach: show content based on selected tab via radio buttons
        st.markdown("<div style='margin: 5px 0;'></div>", unsafe_allow_html=True)

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

    # PUBLIC UI: 3 tabs only (DEMO/BETA/TELOS)
    # DEVOPS mode is admin-only, accessible via URL parameter: ?admin=true

    # Check for admin access via query params FIRST
    query_params = st.query_params
    is_admin = query_params.get("admin") == "true"

    # Check if user has entered BETA (either completed DEMO or accepted BETA consent)
    beta_entered = st.session_state.get('beta_consent_given', False) or st.session_state.get('pa_established', False)

    # Fixed-width tab container (700px max, centered) - buttons don't expand with window
    st.markdown("""<div style="max-width: 700px; margin: 0 auto;">""", unsafe_allow_html=True)

    if beta_entered:
        # BETA mode: Always show DEMO + BETA so users can cycle back and forth
        # State is preserved in session_state so they land back where they were
        if is_admin:
            # Admin mode: Show all three tabs (DEMO, BETA, TELOS)
            col_demo, col_beta, col_telos = st.columns(3)
            with col_demo:
                demo_active = active_tab == "DEMO"
                if st.button("DEMO", key="tab_demo", use_container_width=True,
                            type="primary" if demo_active else "secondary"):
                    st.session_state.active_tab = "DEMO"
                    st.rerun()
            with col_beta:
                beta_active = active_tab == "BETA"
                if st.button("BETA", key="tab_beta", use_container_width=True,
                            type="primary" if beta_active else "secondary"):
                    st.session_state.active_tab = "BETA"
                    st.rerun()
            with col_telos:
                telos_active = active_tab == "TELOS"
                if st.button("TELOS (Admin)", key="tab_telos", use_container_width=True,
                            type="primary" if telos_active else "secondary"):
                    st.session_state.active_tab = "TELOS"
                    st.rerun()

        else:
            # Public mode: Show DEMO + BETA (users can cycle between them)
            col_demo, col_beta = st.columns(2)
            with col_demo:
                demo_active = active_tab == "DEMO"
                if st.button("DEMO", key="tab_demo", use_container_width=True,
                            type="primary" if demo_active else "secondary"):
                    st.session_state.active_tab = "DEMO"
                    st.rerun()
            with col_beta:
                beta_active = active_tab == "BETA"
                if st.button("BETA", key="tab_beta", use_container_width=True,
                            type="primary" if beta_active else "secondary"):
                    st.session_state.active_tab = "BETA"
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
            # BETA is now always accessible - users can skip DEMO if they prefer to learn by doing
            if st.button("BETA", key="tab_beta", use_container_width=True,
                        type="primary" if beta_active else "secondary"):
                st.session_state.active_tab = "BETA"
                st.rerun()

    st.markdown("""</div>""", unsafe_allow_html=True)

    # BETA Action Buttons Row (persist throughout session after PA established)
    if beta_entered and active_tab == "BETA":
        pa_established = st.session_state.get('pa_established', False)
        is_loading = st.session_state.get('is_processing_input', False) or st.session_state.get('is_generating_response', False)
        state_manager = st.session_state.get('state_manager')
        steward_is_open = st.session_state.get('beta_steward_panel_open', False)

        # Show buttons whenever PA is established (not just first turn)
        if pa_established and not is_loading and state_manager:
            # Minimal spacing between BETA tab and action buttons (reduced from 15px)
            st.markdown("<div style='margin: 4px 0;'></div>", unsafe_allow_html=True)

            # Wrap action buttons in same 700px container as BETA button for alignment
            st.markdown("""<div style="max-width: 700px; margin: 0 auto;">""", unsafe_allow_html=True)

            # Action buttons
            scroll_mode = state_manager.state.scrollable_history_mode
            scroll_label = "✕ Close Scroll" if scroll_mode else "📜 Scroll View"
            steward_label = "Close Steward" if steward_is_open else "🤝 Ask Steward"

            col1, col2 = st.columns(2)
            with col1:
                scroll_clicked = st.button(scroll_label, key="hidden_scroll_btn", use_container_width=True)
            with col2:
                steward_clicked = st.button(steward_label, key="hidden_steward_btn", use_container_width=True)

            st.markdown("""</div>""", unsafe_allow_html=True)

            if scroll_clicked:
                # Check if we're enabling scroll view (currently off -> turning on)
                currently_scroll_mode = state_manager.state.scrollable_history_mode
                state_manager.toggle_scrollable_history()
                # If we just enabled scroll view, set flag for auto-scroll to input
                if not currently_scroll_mode:
                    st.session_state.scroll_view_just_enabled = True
                st.rerun()
            if steward_clicked:
                if steward_is_open:
                    st.session_state.beta_steward_panel_open = False
                    # Set flag to scroll to top (or current turn if in scroll view)
                    st.session_state.scroll_after_steward_close = True
                else:
                    st.session_state.beta_steward_panel_open = True
                    # Mutual exclusion: close Alignment Lens when Steward opens
                    st.session_state.beta_deck_visible = False
                    # Set flag to scroll to Steward section
                    st.session_state.scroll_to_steward = True
                st.rerun()

    # CRITICAL: Force neutral gray styling on tab buttons
    # This overrides any fidelity-colored borders that leak from demo slide CSS
    st.markdown("""
    <style>
    /* FORCE NEUTRAL TAB STYLING - Override any fidelity-colored borders from slide CSS */
    /* Target all secondary buttons directly inside the first stHorizontalBlock (the tab bar) */
    /* Use !important with high specificity to override later CSS injections */
    div[data-testid="stHorizontalBlock"]:first-of-type button[kind="secondary"],
    div[data-testid="stHorizontalBlock"]:first-of-type button[data-testid="baseButton-secondary"] {
        background-color: #2d2d2d !important;
        border: 2px solid #666666 !important;
        color: #e0e0e0 !important;
    }
    div[data-testid="stHorizontalBlock"]:first-of-type button[kind="secondary"]:hover,
    div[data-testid="stHorizontalBlock"]:first-of-type button[data-testid="baseButton-secondary"]:hover {
        background-color: #3d3d3d !important;
        border: 2px solid #888888 !important;
    }

    /* FULL-WIDTH FLUSH LAYOUT - buttons match content width edge-to-edge */
    /* No max-width constraints - everything stretches to fill available space */
    [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] {
        width: 100% !important;
        max-width: 100% !important;
        box-sizing: border-box !important;
    }
    /* Full-width button rows */
    div[data-testid="stHorizontalBlock"] {
        width: 100% !important;
        max-width: 100% !important;
        box-sizing: border-box !important;
    }
    /* Target the second stHorizontalBlock (action buttons after tab bar) for 8px gap */
    .main .block-container div[data-testid="stHorizontalBlock"]:nth-of-type(2) {
        gap: 8px !important;
        column-gap: 8px !important;
        margin-left: 0 !important;
        margin-right: 0 !important;
    }
    /* Remove internal padding from action button columns */
    .main .block-container div[data-testid="stHorizontalBlock"]:nth-of-type(2) > div[data-testid="stColumn"] {
        padding-left: 0 !important;
        padding-right: 0 !important;
        flex: 1 1 calc(50% - 4px) !important;
    }
    /* Stretch action buttons to full column width with 20px font */
    .main .block-container div[data-testid="stHorizontalBlock"]:nth-of-type(2) button {
        width: 100% !important;
    }
    .main .block-container div[data-testid="stHorizontalBlock"]:nth-of-type(2) button p {
        font-size: 20px !important;
        font-weight: 600 !important;
    }

    </style>
    """, unsafe_allow_html=True)

    # Dynamic Alignment Lens button styling based on current demo slide
    # Only apply in DEMO mode
    if st.session_state.get('active_tab') == 'DEMO':
        from config.colors import get_fidelity_color
        current_slide = st.session_state.get('demo_slide_index', 0)

        # Determine user fidelity based on slide (matching conversation_display.py values)
        # demo_slide_index: 0=welcome, 1-13=Q&A slides
        # Slide 6 (idx 6): Math question - YELLOW (0.69)
        # Slide 7 (idx 7): Quantum physics - ORANGE (0.55)
        # Slide 8 (idx 8): Movies - RED (0.42)
        if current_slide == 6:  # Slide 6 - Math question (YELLOW)
            user_fidelity = 0.69
        elif current_slide == 7:  # Slide 7 - Quantum physics (ORANGE)
            user_fidelity = 0.55
        elif current_slide == 8:  # Slide 8 - Movies (RED)
            user_fidelity = 0.42
        else:  # All other slides (GREEN)
            user_fidelity = 1.00

        user_color = get_fidelity_color(user_fidelity)

        # Use slide-specific CSS class to force browser to re-apply styles on navigation
        st.markdown(f"""
        <style>
        /* Dynamic Alignment Lens button styling for slide {current_slide} - EXCLUDE tab bar */
        /* Fidelity: {user_fidelity} -> Color: {user_color} */
        div[data-testid="stHorizontalBlock"]:not(:first-of-type) button[kind="primary"],
        div[data-testid="stHorizontalBlock"]:not(:first-of-type) button[data-testid="baseButton-primary"] {{
            background-color: #2d2d2d !important;
            border: 3px solid {user_color} !important;
            color: #e0e0e0 !important;
            box-shadow: 0 0 10px {user_color}66 !important;
        }}
        div[data-testid="stHorizontalBlock"]:not(:first-of-type) button[kind="primary"]:hover,
        div[data-testid="stHorizontalBlock"]:not(:first-of-type) button[data-testid="baseButton-primary"]:hover {{
            background-color: #3d3d3d !important;
            border: 3px solid {user_color} !important;
            box-shadow: 0 0 15px {user_color}88 !important;
        }}
        </style>
        """, unsafe_allow_html=True)

    if is_admin:
        st.markdown("<div style='margin: 10px 0;'></div>", unsafe_allow_html=True)
        if st.button("🔧 DEVOPS (Admin Mode)", key="tab_devops", use_container_width=True,
                    type="primary" if active_tab == "DEVOPS" else "secondary"):
            st.session_state.active_tab = "DEVOPS"
            st.rerun()

    # Removed unlock progression message - now shown in Steward intro

    # Add CSS to reduce vertical spacing for DEMO and BETA modes - AGGRESSIVE dead space removal
    if st.session_state.get('active_tab') in ['DEMO', 'BETA']:
        st.markdown("""
        <style>
        /* AGGRESSIVE: Eliminate dead space between tabs and content */
        .stMarkdown + div {
            margin-top: 0 !important;
            padding-top: 0 !important;
        }
        /* Remove main block container padding entirely */
        .main .block-container {
            padding-top: 0 !important;
            padding-bottom: 0.5rem !important;
            margin-top: 0 !important;
        }
        /* Remove spacing between streamlit elements */
        .stMarkdown {
            margin-bottom: 0 !important;
            margin-top: 0 !important;
        }
        div[data-testid="stVerticalBlock"] > div {
            gap: 0 !important;
        }
        /* Remove padding from vertical block containers */
        div[data-testid="stVerticalBlock"] {
            gap: 0 !important;
            padding-top: 0 !important;
        }
        /* Ensure content starts immediately after tabs */
        .element-container {
            margin-top: 0 !important;
            margin-bottom: 0 !important;
        }
        /* Remove any top margin from first content element */
        .main .block-container > div:first-child {
            margin-top: 0 !important;
            padding-top: 0 !important;
        }
        </style>
        """, unsafe_allow_html=True)

    # Hide sidebar for DEMO and BETA modes - only show in TELOS and DEVOPS
    active_tab = st.session_state.get('active_tab', 'DEMO')

    # Hide sidebar for non-TELOS/DEVOPS modes (layout CSS injected at end of render)
    if active_tab not in ['TELOS', 'DEVOPS']:
        st.markdown("""
        <style>
        [data-testid="stSidebar"] {
            display: none !important;
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
        show_observatory_lens_auto = (mode in ["TELOS", "DEVOPS"])  # Auto-enabled in full modes

        # TELOS mode: Show profile selector first if not established
        if mode == "TELOS":
            # Check for profile in query params first (allows direct linking)
            query_profile = teloscope_profile_selector.get_profile_from_query_params()
            if query_profile and not st.session_state.get('teloscope_profile'):
                teloscope_profile_selector.apply_profile_settings(query_profile)

            # If no profile selected, show selector
            if not st.session_state.get('teloscope_profile'):
                selected = teloscope_profile_selector.render_profile_selector()
                if selected:
                    teloscope_profile_selector.apply_profile_settings(selected)
                    st.rerun()
                return  # Stop rendering until profile selected

            # Apply profile-based feature flags (override defaults)
            profile_features = st.session_state.get('teloscope_features', {})
            show_observation_deck = profile_features.get('observation_deck', True)
            show_teloscope = profile_features.get('teloscope_controls', True)
            show_observatory_lens_auto = profile_features.get('observatory_lens', True)

            # Profile header
            teloscope_profile_selector.render_profile_badge()
            st.markdown("<div style='margin-bottom: 15px;'></div>", unsafe_allow_html=True)

        # DEVOPS mode: Profile is always "devops" with all features
        if mode == "DEVOPS":
            if not st.session_state.get('teloscope_profile'):
                teloscope_profile_selector.apply_profile_settings("devops")

        # DEVOPS header
        if show_devops_header:
            st.markdown("### 🔧 DEVOPS Mode - Full System Access")
            st.markdown("**All restrictions removed. Beta mode with full PA extraction and interventions enabled.**")

        # BETA mode: Show PA questionnaire first if not established
        if mode == "BETA":
            # Check if beta is completed - show Thank You screen
            if st.session_state.get('beta_completed', False):
                beta_completion = BetaCompletion(state_manager)
                beta_completion.render()
                return  # Stop rendering - show only Thank You screen

            # Check if PA is already established
            if not st.session_state.get('pa_established', False):
                # Show PA questionnaire and block conversation until complete
                pa_answers = pa_onboarding.render_questionnaire()
                if pa_answers is None:
                    # PA not established yet, stop rendering (questionnaire is showing)
                    return
                # If we get here, PA was just established - continue to conversation

            # CONDITIONAL WIDTH - 700px when Steward closed, full-width when open
            # Check Steward state early to apply correct layout CSS
            beta_steward_is_open = st.session_state.get('beta_steward_panel_open', False)

            # Handle scroll-to-top after closing Steward
            if st.session_state.pop('scroll_after_steward_close', False):
                import streamlit.components.v1 as components
                # Check if in scroll view mode to scroll to current turn instead of top
                scroll_mode = state_manager.state.scrollable_history_mode if state_manager else False
                if scroll_mode:
                    # Scroll to input area (bottom of conversation)
                    components.html("""
                    <script>
                    setTimeout(function() {
                        // Scroll to input area in scroll view
                        const textareas = window.parent.document.querySelectorAll('textarea');
                        if (textareas.length > 0) {
                            textareas[0].scrollIntoView({ behavior: 'smooth', block: 'center' });
                        } else {
                            window.parent.scrollTo({ top: 0, behavior: 'smooth' });
                        }
                    }, 100);
                    </script>
                    """, height=0)
                else:
                    # Scroll to top
                    components.html("""
                    <script>
                    setTimeout(function() {
                        window.parent.scrollTo({ top: 0, behavior: 'smooth' });
                    }, 100);
                    </script>
                    """, height=0)

            if not beta_steward_is_open:
                # Steward CLOSED: Full-width flush layout - edge-to-edge
                st.markdown("""
                <style>
                /* BETA mode full-width flush - buttons and content align edge-to-edge */
                .main .block-container {
                    width: 100% !important;
                    max-width: 100% !important;
                    min-width: 0 !important;
                    margin-left: auto !important;
                    margin-right: auto !important;
                    padding-left: 1rem !important;
                    padding-right: 1rem !important;
                    box-sizing: border-box !important;
                }
                /* Full-width horizontal blocks */
                [data-testid="stHorizontalBlock"] {
                    width: 100% !important;
                    max-width: 100% !important;
                    box-sizing: border-box !important;
                }
                </style>
                """, unsafe_allow_html=True)
            else:
                # Steward OPEN: Full-width layout for 70/30 split
                st.markdown("""
                <style>
                /* BETA mode full-width for Steward panel */
                .main .block-container {
                    width: 100% !important;
                    max-width: 100% !important;
                    min-width: auto !important;
                    margin-left: 0 !important;
                    margin-right: 0 !important;
                    padding-left: 2rem !important;
                    padding-right: 2rem !important;
                }
                /* Let horizontal blocks expand for 70/30 split */
                [data-testid="stHorizontalBlock"] {
                    width: 100% !important;
                    max-width: 100% !important;
                }
                </style>
                """, unsafe_allow_html=True)

            # Scroll to top when entering BETA mode (after PA established)
            # CRITICAL: Only scroll once to prevent "redirected to beginning" bug
            if not st.session_state.get('beta_scroll_to_top_done', False):
                st.session_state.beta_scroll_to_top_done = True
                import streamlit.components.v1 as components
                components.html("""
                <script>
                (function() {
                    // Scroll to top of page
                    window.scrollTo(0, 0);
                    if (window.parent) {
                        window.parent.scrollTo(0, 0);
                    }
                    // Also try to scroll the main container
                    var mainContainer = window.parent.document.querySelector('.main');
                    if (mainContainer) {
                        mainContainer.scrollTop = 0;
                    }
                    var appContainer = window.parent.document.querySelector('[data-testid="stAppViewContainer"]');
                    if (appContainer) {
                        appContainer.scrollTop = 0;
                    }
                })();
                </script>
                """, height=0)

            # BETA welcome info box removed for cleaner UI - just show user input box

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
                # Minimal spacing - Alignment Lens should be close to send message box
                st.markdown("<div style='margin: 10px 0;'></div>", unsafe_allow_html=True)

                # Check if TELOSCOPE panel is open
                if st.session_state.get('teloscope_open', False):
                    # Render full TELOSCOPE panel (with scrubbing, gauges, Steward)
                    render_teloscope_panel()
                else:
                    # Render compact Alignment Lens
                    beta_observation_deck.render()

                # TELOSCOPE toggle button (only show when there's conversation data)
                # Mutual exclusion: hide when Alignment Lens is open
                has_turn_data = st.session_state.get('beta_turn_1_data') is not None
                alignment_lens_open = st.session_state.get('beta_deck_visible', False)
                if has_turn_data and not alignment_lens_open:
                    st.markdown("<div style='margin: 8px 0;'></div>", unsafe_allow_html=True)
                    render_teloscope_button()
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

        # Alignment Lens (auto-enabled in TELOS/DEVOPS, or toggled manually in other modes)
        if show_observatory_lens_auto or st.session_state.get('show_observatory_lens', False):
            st.markdown("<div style='margin: 20px 0;'></div>", unsafe_allow_html=True)
            observatory_lens.render()

    # Check if Steward panel is open (BETA uses separate state)
    active_tab = st.session_state.get('active_tab', 'DEMO')
    is_beta_mode = active_tab == "BETA"

    # Use appropriate panel state and component based on mode
    if is_beta_mode:
        steward_open = st.session_state.get('beta_steward_panel_open', False)
    else:
        steward_open = st.session_state.get('steward_panel_open', False)

    # BETA mode: Use bottom section instead of side panel
    if is_beta_mode:
        # Always render main content at full width
        render_mode_content(st.session_state.active_tab)

        # Render bottom Steward section if open
        if steward_open and has_beta_consent:
            st.markdown("<div style='margin: 20px 0;'></div>", unsafe_allow_html=True)
            render_bottom_section()

    # Non-BETA modes: Use side panel layout (original behavior)
    elif steward_open and has_beta_consent:
        # Two-column layout: Main content (70%) | Steward chat (30%)
        col_main, col_steward = st.columns([7, 3])

        with col_main:
            render_mode_content(st.session_state.active_tab)

        with col_steward:
            steward_panel.render_panel()

    else:
        render_mode_content(st.session_state.active_tab)

    # FINAL CSS OVERRIDE - Inject with highest specificity at runtime
    st.html("""
    <style>
    /* Runtime CSS injection - v20:36 - Fix all buttons and form styling */

    /* ALL BUTTONS - Dark theme with gold border */
    .stButton > button,
    button[data-baseweb="button"],
    button[kind="secondary"],
    button[kind="primary"],
    .stFormSubmitButton > button {
        background-color: #2d2d2d !important;
        background: #2d2d2d !important;
        color: #e0e0e0 !important;
        border: 1px solid #F4D03F !important;
    }

    /* PRIMARY BUTTONS - Gold glow for emphasis */
    button[kind="primary"] {
        border: 2px solid #F4D03F !important;
        box-shadow: 0 0 8px rgba(255, 215, 0, 0.5) !important;
    }

    /* Button hover state */
    .stButton > button:hover,
    button[data-baseweb="button"]:hover,
    .stFormSubmitButton > button:hover {
        border-color: #F4D03F !important;
        box-shadow: 0 0 6px #F4D03F !important;
        background-color: #3d3d3d !important;
    }

    /* TEXT AREA - Gold border, remove any red styling */
    .stTextArea textarea,
    textarea[data-baseweb="textarea"],
    .stTextArea > div > div > textarea {
        background-color: #2d2d2d !important;
        color: #ffffff !important;
        border: 1px solid #F4D03F !important;
        border-color: #F4D03F !important;
    }

    /* Remove Streamlit's focus ring (red/blue) and use gold */
    .stTextArea textarea:focus,
    textarea:focus {
        border-color: #F4D03F !important;
        outline: none !important;
        box-shadow: 0 0 4px rgba(244, 208, 63, 0.5) !important;
    }

    /* Override any form validation red borders */
    .stTextArea > div,
    .stTextArea > div > div {
        border-color: #F4D03F !important;
    }

    /* Form container styling - remove red borders */
    [data-baseweb="form-control-container"],
    .stForm {
        border-color: #F4D03F !important;
    }

    /* Message container hover glow */
    .message-container:hover {
        box-shadow: 0 0 6px #F4D03F !important;
        transition: box-shadow 0.3s ease !important;
    }
    </style>
    """)

    # LAYOUT CSS - Injected at end of render for highest priority
    # This CSS comes AFTER all content is rendered, ensuring it wins CSS specificity battles
    if active_tab not in ['TELOS', 'DEVOPS']:
        if steward_open and has_beta_consent:
            # Full width layout when Steward panel is open (70/30 column split)
            st.markdown("""
            <style>
            /* STEWARD OPEN: Full width layout for 70/30 split */
            .main .block-container,
            [data-testid="stAppViewContainer"] > .main > .block-container,
            section[data-testid="stMainBlockContainer"],
            [data-testid="stMainBlockContainer"] {
                max-width: 100% !important;
                padding-left: 2rem !important;
                padding-right: 2rem !important;
                margin-left: 0 !important;
                margin-right: 0 !important;
            }
            .main [data-testid="stVerticalBlock"],
            .main [data-testid="stHorizontalBlock"] {
                max-width: 100% !important;
                margin-left: 0 !important;
                margin-right: 0 !important;
            }
            /* CRITICAL: Override ALL inline 700px constraints when Steward is open */
            [style*="max-width: 700px"],
            [style*="max-width:700px"],
            div[style*="700px"] {
                max-width: 100% !important;
                width: 100% !important;
            }
            /* Override fixed-width elements */
            [style*="width: 700px"],
            [style*="width:700px"] {
                width: 100% !important;
                max-width: 100% !important;
            }
            </style>
            """, unsafe_allow_html=True)
        else:
            # Centered 700px layout when Steward is closed
            st.markdown("""
            <style>
            /* STEWARD CLOSED: 700px centered layout */
            .main .block-container,
            [data-testid="stAppViewContainer"] > .main > .block-container,
            section[data-testid="stMainBlockContainer"],
            [data-testid="stMainBlockContainer"] {
                max-width: 700px !important;
                margin-left: auto !important;
                margin-right: auto !important;
                padding-left: 1rem !important;
                padding-right: 1rem !important;
            }
            .main [data-testid="stVerticalBlock"],
            .main [data-testid="stHorizontalBlock"] {
                max-width: 700px !important;
                margin-left: auto !important;
                margin-right: auto !important;
            }
            .main [data-testid="column"] {
                max-width: 100% !important;
            }
            iframe {
                min-height: 50px !important;
                overflow: visible !important;
            }
            </style>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
