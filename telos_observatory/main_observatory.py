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
import streamlit.components.v1 as components
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from telos_observatory.mock_data import generate_mock_session
from telos_observatory.observation_deck.deck_interface import render_observation_deck
from telos_observatory.observation_deck.sidebar_deck import render_observation_deck_sidebar
from telos_observatory.teloscope.teloscope_controller import render_teloscope, snap_to_center, snap_to_dock


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
    - deck_expanded: Observation Deck sidebar visibility
    - deck_show_math: Math breakdown section expanded
    - deck_show_counterfactual: Counterfactual section expanded
    - steward_active: Steward chat active
    - steward_api_key: Mistral API key (session only)
    - steward_messages: Steward chat history
    - teloscope_docked: TELOSCOPE docked to bottom
    - teloscope_position: TELOSCOPE position when undocked
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

        # Observation Deck state
        st.session_state.deck_expanded = False
        st.session_state.deck_show_math = False
        st.session_state.deck_show_counterfactual = False

        # Steward state (shared between Observation Deck and TELOSCOPE)
        st.session_state.steward_active = False
        st.session_state.steward_api_key = None
        st.session_state.steward_messages = []

        # TELOSCOPE state
        st.session_state.teloscope_docked = True
        st.session_state.teloscope_position = {'x': 0, 'y': 0}

        # Handle URL parameters for Deep Research links
        handle_url_parameters()


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

    # Main content: Observation Deck (main viewport)
    render_observation_deck()

    # Right sidebar: Observation Deck sidebar (if expanded)
    render_observation_deck_sidebar()

    # Bottom controls: TELOSCOPE Remote
    render_teloscope()

    # Keyboard shortcuts
    render_keyboard_shortcuts()

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
            'session_id': st.session_state.session_data.get('session_id'),
            'deck_expanded': st.session_state.get('deck_expanded', False),
            'teloscope_docked': st.session_state.get('teloscope_docked', True)
        })


def handle_url_parameters():
    """
    Handle URL query parameters for Deep Research links.

    Supports:
    - ?turn=5 : Navigate to specific turn (0-based)
    - ?study=claude_test_1 : Load specific study
    """
    try:
        # Get query parameters
        query_params = st.query_params

        # Handle turn parameter
        if 'turn' in query_params:
            turn_str = query_params['turn']
            try:
                turn_index = int(turn_str)
                # Validate turn index
                turns = st.session_state.session_data.get('turns', [])
                if 0 <= turn_index < len(turns):
                    st.session_state.current_turn = turn_index
                    # Auto-expand Observation Deck for Deep Research
                    st.session_state.deck_expanded = True
            except (ValueError, TypeError):
                pass  # Invalid turn parameter, ignore

        # Handle study parameter (for future Phase 2/2B study loading)
        if 'study' in query_params:
            study_id = query_params['study']
            # TODO: Load specific study from Phase2Loader
            # For now, just store it in session state
            st.session_state.requested_study = study_id

    except Exception as e:
        # Fail silently if URL parameter handling fails
        pass


def render_keyboard_shortcuts():
    """
    Render JavaScript for keyboard shortcuts.

    Shortcuts:
    - Shift+O: Toggle Observation Deck
    - Shift+T: Snap TELOSCOPE to center
    - ESC: Snap TELOSCOPE back to dock
    - Left Arrow: Previous turn
    - Right Arrow: Next turn
    """
    keyboard_js = """
    <script>
    (function() {
        // Prevent multiple bindings
        if (window.telosKeyboardBound) return;
        window.telosKeyboardBound = true;

        document.addEventListener('keydown', function(e) {
            // Get current state
            const activeElement = document.activeElement;
            const isInputFocused = activeElement && (
                activeElement.tagName === 'INPUT' ||
                activeElement.tagName === 'TEXTAREA' ||
                activeElement.contentEditable === 'true'
            );

            // Don't trigger shortcuts if user is typing in an input field
            if (isInputFocused) return;

            // Shift+O: Toggle Observation Deck
            if (e.shiftKey && e.key === 'O') {
                e.preventDefault();
                const deckToggle = document.querySelector('[data-testid="baseButton-secondary"][title*="Observation Deck"]');
                if (deckToggle) deckToggle.click();
            }

            // Shift+T: Snap TELOSCOPE to center
            else if (e.shiftKey && e.key === 'T') {
                e.preventDefault();
                const centerBtn = document.querySelector('[title*="Snap to center"]');
                if (centerBtn) centerBtn.click();
            }

            // ESC: Snap TELOSCOPE back to dock
            else if (e.key === 'Escape') {
                const dockBtn = document.querySelector('[title*="Dock"]');
                if (dockBtn && dockBtn.textContent.includes('📍')) {
                    e.preventDefault();
                    dockBtn.click();
                }
            }

            // Left Arrow: Previous turn
            else if (e.key === 'ArrowLeft') {
                e.preventDefault();
                const prevBtn = document.querySelector('[aria-label="Previous"]') ||
                               document.querySelector('button:has-text("⏮")');
                if (prevBtn) prevBtn.click();
            }

            // Right Arrow: Next turn
            else if (e.key === 'ArrowRight') {
                e.preventDefault();
                const nextBtn = document.querySelector('[aria-label="Next"]') ||
                               document.querySelector('button:has-text("⏭")');
                if (nextBtn) nextBtn.click();
            }
        });
    })();
    </script>
    """

    components.html(keyboard_js, height=0)


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    main()
