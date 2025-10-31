"""
TELOSCOPE Navigation Controls
==============================

Prev/Next/Play/Pause buttons for turn-by-turn navigation.

CRITICAL: Follows Streamlit callback pattern:
- Callbacks ONLY update st.session_state
- NO rendering inside callbacks
- st.rerun() after state changes
"""

import streamlit as st
import time


# ============================================================================
# Callbacks (State Updates Only)
# ============================================================================

def on_first_turn():
    """Jump to first turn - callback"""
    st.session_state.current_turn = 0
    st.rerun()


def on_previous_turn():
    """Go to previous turn - callback"""
    if st.session_state.current_turn > 0:
        st.session_state.current_turn -= 1
    st.rerun()


def on_next_turn():
    """Go to next turn - callback"""
    max_turns = len(st.session_state.session_data['turns']) - 1
    if st.session_state.current_turn < max_turns:
        st.session_state.current_turn += 1
    st.rerun()


def on_last_turn():
    """Jump to last turn - callback"""
    max_turns = len(st.session_state.session_data['turns']) - 1
    st.session_state.current_turn = max_turns
    st.rerun()


def on_toggle_play():
    """Toggle play/pause state - callback"""
    st.session_state.playing = not st.session_state.playing
    if st.session_state.playing:
        st.session_state.last_play_time = time.time()
    st.rerun()


# ============================================================================
# Rendering (UI Display Only)
# ============================================================================

def render_navigation_controls():
    """
    Render Prev/Play/Pause/Next navigation buttons.

    Reads from st.session_state:
    - current_turn: Active turn index
    - playing: Play/pause state
    - session_data: Session with turns
    """
    session_data = st.session_state.get('session_data', {})
    turns = session_data.get('turns', [])

    if not turns:
        st.warning("No turns available for navigation")
        return

    current_turn = st.session_state.get('current_turn', 0)
    playing = st.session_state.get('playing', False)
    max_turns = len(turns) - 1

    # Navigation buttons layout
    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])

    with col1:
        st.button(
            "⏮️ First",
            on_click=on_first_turn,
            disabled=(current_turn == 0),
            key="nav_first",
            use_container_width=True
        )

    with col2:
        st.button(
            "◀️ Prev",
            on_click=on_previous_turn,
            disabled=(current_turn == 0),
            key="nav_prev",
            use_container_width=True
        )

    with col3:
        play_label = "⏸️ Pause" if playing else "▶️ Play"
        st.button(
            play_label,
            on_click=on_toggle_play,
            key="nav_play",
            use_container_width=True,
            type="primary" if playing else "secondary"
        )

    with col4:
        st.button(
            "Next ▶️",
            on_click=on_next_turn,
            disabled=(current_turn >= max_turns),
            key="nav_next",
            use_container_width=True
        )

    with col5:
        st.button(
            "Last ⏭️",
            on_click=on_last_turn,
            disabled=(current_turn >= max_turns),
            key="nav_last",
            use_container_width=True
        )


def handle_autoplay():
    """
    Handle automatic turn advancement during play mode.

    Reads/Writes st.session_state:
    - playing: Play state
    - current_turn: Active turn
    - last_play_time: Timestamp of last advance
    - playback_speed: Speed multiplier (default 1.0)
    """
    # Only advance if playing
    if not st.session_state.get('playing', False):
        return

    session_data = st.session_state.get('session_data', {})
    turns = session_data.get('turns', [])
    max_turns = len(turns) - 1
    current_turn = st.session_state.get('current_turn', 0)

    # Check if at end
    if current_turn >= max_turns:
        st.session_state.playing = False
        st.rerun()
        return

    # Check if enough time has passed
    last_play_time = st.session_state.get('last_play_time', 0)
    playback_speed = st.session_state.get('playback_speed', 1.0)
    advance_interval = 2.0 / playback_speed  # 2 seconds per turn at 1x speed

    current_time = time.time()
    time_elapsed = current_time - last_play_time

    if time_elapsed >= advance_interval:
        # Advance to next turn
        st.session_state.current_turn += 1
        st.session_state.last_play_time = current_time
        st.rerun()
