"""
TELOSCOPE Controller - Main Control Interface
==============================================

Orchestrates all TELOSCOPE components:
- Navigation controls (Prev/Play/Next)
- Timeline scrubber with markers
- Turn indicator
- Drag-and-drop positioning
- Steward integration
- Deep Research links

Provides frame-by-frame control of conversation analysis.

Keyboard Shortcuts:
- Shift+T: Snap to center
- ESC: Snap back to dock (bottom)
"""

import streamlit as st
import streamlit.components.v1 as components
from telos_observatory.teloscope.navigation_controls import (
    render_navigation_controls,
    handle_autoplay
)
from telos_observatory.teloscope.timeline_scrubber import (
    render_timeline_scrubber,
    render_timeline_legend
)
from telos_observatory.observation_deck.deep_research import render_deep_research_icon_button


def render_teloscope():
    """
    Render complete TELOSCOPE control interface with drag-and-drop.

    Features:
    - Fixed/draggable positioning
    - Snap-to-center (Shift+T)
    - Snap-back to dock (ESC)
    - Steward integration button
    - Deep Research link button

    Reads from st.session_state:
    - current_turn: Active turn index
    - playing: Play state
    - session_data: Session data
    - teloscope_docked: Docked/undocked state
    - teloscope_position: Position when undocked
    """
    # Handle autoplay advancement
    handle_autoplay()

    # Get state
    session_data = st.session_state.get('session_data', {})
    conversation_id = session_data.get('session_id', 'unknown')
    current_turn = st.session_state.get('current_turn', 0)
    docked = st.session_state.get('teloscope_docked', True)

    # TELOSCOPE container styling (docked or floating)
    if docked:
        # Fixed to bottom
        position_style = """
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
        """
    else:
        # Floating/draggable
        pos = st.session_state.get('teloscope_position', {'x': 0, 'y': 0})
        position_style = f"""
            position: fixed;
            bottom: {pos['y']}px;
            left: {pos['x']}px;
            width: 600px;
        """

    st.markdown(f"""
        <style>
        .teloscope-container {{
            {position_style}
            background: rgba(0, 0, 0, 0.9);
            backdrop-filter: blur(15px);
            border: 1px solid rgba(255, 215, 0, 0.3);
            border-radius: {'0' if docked else '12px'};
            padding: 1.5rem 2rem;
            z-index: 1000;
            box-shadow: 0 -4px 20px rgba(0, 0, 0, 0.7);
            {'cursor: move;' if not docked else ''}
            transition: all 0.3s ease;
        }}

        .teloscope-container:hover {{
            border-color: rgba(255, 215, 0, 0.5);
        }}

        .teloscope-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
        }}

        .teloscope-actions {{
            display: flex;
            gap: 0.5rem;
            align-items: center;
        }}

        /* Adjust main content to not be hidden by fixed TELOSCOPE */
        .main .block-container {{
            padding-bottom: {'220px' if docked else '0'} !important;
        }}

        /* Drag handle */
        .drag-handle {{
            cursor: grab;
            padding: 0.5rem;
            border-radius: 4px;
            background: rgba(255, 255, 255, 0.05);
            transition: all 0.2s ease;
        }}

        .drag-handle:hover {{
            background: rgba(255, 255, 255, 0.1);
        }}

        .drag-handle:active {{
            cursor: grabbing;
        }}
        </style>
    """, unsafe_allow_html=True)

    # TELOSCOPE container
    st.markdown('<div class="teloscope-container" id="teloscope">', unsafe_allow_html=True)

    # Header with title and action buttons
    col_title, col_actions = st.columns([3, 2])

    with col_title:
        st.markdown("""
            <div style="font-size: 1.1rem; font-weight: bold; color: #FFD700;">
                🔭 TELOSCOPE Remote
            </div>
        """, unsafe_allow_html=True)

    with col_actions:
        # Action buttons row
        action_cols = st.columns([1, 1, 1, 1, 1])

        # Deep Research button
        with action_cols[0]:
            render_deep_research_icon_button(current_turn, conversation_id)

        # Steward button
        with action_cols[1]:
            if st.button("🤝", key="teloscope_steward", help="Ask Steward"):
                st.session_state.steward_active = not st.session_state.get('steward_active', False)
                st.rerun()

        # Snap to center button
        with action_cols[2]:
            if st.button("⊕", key="teloscope_center", help="Snap to center (Shift+T)"):
                snap_to_center()
                st.rerun()

        # Dock/undock button
        with action_cols[3]:
            dock_icon = "📌" if docked else "📍"
            dock_label = "Undock" if docked else "Dock"
            if st.button(dock_icon, key="teloscope_dock_toggle", help=dock_label):
                st.session_state.teloscope_docked = not docked
                st.rerun()

        # Drag handle (only when undocked)
        with action_cols[4]:
            if not docked:
                st.markdown("""
                    <div class="drag-handle" title="Drag to reposition">
                        ⋮⋮
                    </div>
                """, unsafe_allow_html=True)

    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)

    # Timeline scrubber with markers
    render_timeline_scrubber()

    # Navigation controls
    render_navigation_controls()

    # Legend
    render_timeline_legend()

    st.markdown('</div>', unsafe_allow_html=True)

    # Add drag-and-drop JavaScript (only if undocked)
    if not docked:
        render_drag_script()


def snap_to_center():
    """Snap TELOSCOPE to center of screen."""
    st.session_state.teloscope_docked = False
    # Center position (approximate, will be adjusted by JS)
    st.session_state.teloscope_position = {'x': 300, 'y': 200}


def snap_to_dock():
    """Snap TELOSCOPE back to bottom dock."""
    st.session_state.teloscope_docked = True
    st.session_state.teloscope_position = {'x': 0, 'y': 0}


def render_drag_script():
    """
    Render JavaScript for drag-and-drop functionality.

    Enables dragging TELOSCOPE when undocked.
    """
    drag_js = """
    <script>
    (function() {
        const teloscope = document.getElementById('teloscope');
        if (!teloscope) return;

        let isDragging = false;
        let currentX;
        let currentY;
        let initialX;
        let initialY;
        let xOffset = 0;
        let yOffset = 0;

        teloscope.addEventListener('mousedown', dragStart);
        document.addEventListener('mousemove', drag);
        document.addEventListener('mouseup', dragEnd);

        function dragStart(e) {
            if (e.target.classList.contains('drag-handle') ||
                e.target.closest('.drag-handle')) {
                initialX = e.clientX - xOffset;
                initialY = e.clientY - yOffset;
                isDragging = true;
            }
        }

        function drag(e) {
            if (isDragging) {
                e.preventDefault();
                currentX = e.clientX - initialX;
                currentY = e.clientY - initialY;
                xOffset = currentX;
                yOffset = currentY;

                setTranslate(currentX, currentY, teloscope);
            }
        }

        function dragEnd(e) {
            initialX = currentX;
            initialY = currentY;
            isDragging = false;
        }

        function setTranslate(xPos, yPos, el) {
            el.style.transform = `translate3d(${xPos}px, ${yPos}px, 0)`;
        }
    })();
    </script>
    """

    components.html(drag_js, height=0)


def get_teloscope_status() -> dict:
    """
    Get current TELOSCOPE status for display/debugging.

    Returns:
        Dictionary with current turn, playing state, session info
    """
    session_data = st.session_state.get('session_data', {})
    turns = session_data.get('turns', [])

    return {
        'current_turn': st.session_state.get('current_turn', 0),
        'total_turns': len(turns),
        'playing': st.session_state.get('playing', False),
        'playback_speed': st.session_state.get('playback_speed', 1.0),
        'session_id': session_data.get('session_id', 'unknown'),
        'docked': st.session_state.get('teloscope_docked', True),
        'position': st.session_state.get('teloscope_position', {'x': 0, 'y': 0})
    }
