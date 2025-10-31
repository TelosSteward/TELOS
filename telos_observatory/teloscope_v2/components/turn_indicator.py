"""
TELOSCOPE v2 - Turn Indicator Component

Displays current turn information with jump-to functionality.

Features:
- "Turn X / Y" display with current position and total
- Jump-to-turn input (text field or number input)
- Keyboard shortcut support (Enter to jump)
- Validation (bounds checking)
- Visual styling (compact, inline)

Usage:
    from teloscope_v2.components.turn_indicator import render_turn_indicator

    render_turn_indicator(total_turns=50)
"""

import streamlit as st
from typing import Optional
from ..state.teloscope_state import (
    get_current_turn,
    set_current_turn,
    get_teloscope_state,
)


# ============================================================================
# CALLBACKS
# ============================================================================

def _on_jump_to_turn() -> None:
    """
    Handle jump-to-turn action.

    Validates input and updates current_turn.
    Follows callback pattern: state update only, no rendering.
    """
    # Get jump target from widget state
    jump_target = st.session_state.get('turn_jump_input')

    if jump_target is None:
        return

    # Get session bounds
    session_data = st.session_state.get('session_data')
    if not session_data:
        return

    total_turns = len(session_data.get('turns', []))
    if total_turns == 0:
        return

    # Validate bounds (1-indexed input → 0-indexed turn)
    # User enters 1-12, we store 0-11
    turn_index = jump_target - 1
    turn_index = max(0, min(total_turns - 1, turn_index))

    # Update state
    set_current_turn(turn_index, rerun=True)


def _on_first_turn() -> None:
    """Jump to first turn."""
    set_current_turn(0, rerun=True)


def _on_last_turn() -> None:
    """Jump to last turn."""
    session_data = st.session_state.get('session_data')
    if session_data:
        total_turns = len(session_data.get('turns', []))
        set_current_turn(total_turns - 1, rerun=True)


# ============================================================================
# RENDERING
# ============================================================================

def render_turn_indicator(
    total_turns: Optional[int] = None,
    show_jump_controls: bool = True,
    compact: bool = False,
) -> None:
    """
    Render turn indicator with optional jump controls.

    Args:
        total_turns: Total number of turns (auto-detected from session_data if None)
        show_jump_controls: If True, show jump-to input and first/last buttons
        compact: If True, use more compact layout

    Layout:
        Compact:    [Turn 5 / 12] [⏮️] [Input] [⏭️]
        Full:       Turn 5 / 12
                    Jump to: [___] [Go]
                    [First] [Last]
    """
    # Auto-detect total turns
    if total_turns is None:
        session_data = st.session_state.get('session_data')
        if session_data:
            total_turns = len(session_data.get('turns', []))
        else:
            total_turns = 0

    # Get current turn (0-indexed internally, display as 1-indexed)
    current_turn_index = get_current_turn()
    current_turn_display = current_turn_index + 1

    # ===== COMPACT LAYOUT =====
    if compact:
        cols = st.columns([2, 1, 2, 1])

        with cols[0]:
            # Turn display
            st.markdown(
                f"<div style='text-align: center; padding: 8px; font-size: 14px;'>"
                f"<strong>Turn {current_turn_display} / {total_turns}</strong>"
                f"</div>",
                unsafe_allow_html=True
            )

        if show_jump_controls:
            with cols[1]:
                # First button
                st.button(
                    "⏮️",
                    key="turn_indicator_first",
                    on_click=_on_first_turn,
                    disabled=(current_turn_index == 0),
                    help="Jump to first turn",
                    use_container_width=True,
                )

            with cols[2]:
                # Jump input
                st.number_input(
                    "Jump to",
                    min_value=1,
                    max_value=total_turns,
                    value=current_turn_display,
                    key="turn_jump_input",
                    on_change=_on_jump_to_turn,
                    label_visibility="collapsed",
                )

            with cols[3]:
                # Last button
                st.button(
                    "⏭️",
                    key="turn_indicator_last",
                    on_click=_on_last_turn,
                    disabled=(current_turn_index == total_turns - 1),
                    help="Jump to last turn",
                    use_container_width=True,
                )

    # ===== FULL LAYOUT =====
    else:
        # Turn display
        st.markdown(
            f"<div style='text-align: center; padding: 12px; font-size: 16px; font-weight: bold;'>"
            f"Turn {current_turn_display} / {total_turns}"
            f"</div>",
            unsafe_allow_html=True
        )

        if show_jump_controls:
            # Jump input with label
            col1, col2 = st.columns([3, 1])

            with col1:
                st.number_input(
                    "Jump to turn:",
                    min_value=1,
                    max_value=total_turns,
                    value=current_turn_display,
                    key="turn_jump_input_full",
                    on_change=_on_jump_to_turn,
                )

            with col2:
                st.markdown("<div style='padding-top: 30px;'></div>", unsafe_allow_html=True)
                if st.button("Go", key="turn_jump_go", use_container_width=True):
                    _on_jump_to_turn()

            # First/Last buttons
            col3, col4 = st.columns(2)

            with col3:
                st.button(
                    "⏮️ First Turn",
                    key="turn_indicator_first_full",
                    on_click=_on_first_turn,
                    disabled=(current_turn_index == 0),
                    use_container_width=True,
                )

            with col4:
                st.button(
                    "Last Turn ⏭️",
                    key="turn_indicator_last_full",
                    on_click=_on_last_turn,
                    disabled=(current_turn_index == total_turns - 1),
                    use_container_width=True,
                )


def render_turn_indicator_inline(total_turns: Optional[int] = None) -> None:
    """
    Render ultra-compact inline turn indicator (no jump controls).

    Useful for embedding in control strips or headers.

    Args:
        total_turns: Total number of turns (auto-detected if None)
    """
    if total_turns is None:
        session_data = st.session_state.get('session_data')
        if session_data:
            total_turns = len(session_data.get('turns', []))
        else:
            total_turns = 0

    current_turn_index = get_current_turn()
    current_turn_display = current_turn_index + 1

    st.markdown(
        f"<span style='font-size: 13px; color: #888; font-weight: 500;'>"
        f"Turn {current_turn_display} / {total_turns}"
        f"</span>",
        unsafe_allow_html=True
    )


def render_turn_progress_bar(total_turns: Optional[int] = None) -> None:
    """
    Render turn indicator as a progress bar.

    Visual representation of conversation progress.

    Args:
        total_turns: Total number of turns (auto-detected if None)
    """
    if total_turns is None:
        session_data = st.session_state.get('session_data')
        if session_data:
            total_turns = len(session_data.get('turns', []))
        else:
            total_turns = 1  # Avoid division by zero

    current_turn_index = get_current_turn()
    current_turn_display = current_turn_index + 1

    # Calculate progress (0.0 - 1.0)
    progress = (current_turn_display) / total_turns

    # Render progress bar
    st.progress(
        progress,
        text=f"Turn {current_turn_display} / {total_turns}"
    )


# ============================================================================
# KEYBOARD SHORTCUT HELPERS
# ============================================================================

def register_turn_shortcuts() -> None:
    """
    Register keyboard shortcuts for turn navigation.

    Shortcuts:
    - Home: Jump to first turn
    - End: Jump to last turn
    - PageUp: Jump back 10 turns
    - PageDown: Jump forward 10 turns

    Note: Streamlit doesn't natively support keyboard shortcuts,
    so this uses JavaScript injection. Use sparingly.
    """
    # This is a placeholder for future keyboard shortcut implementation
    # Streamlit currently doesn't have native keyboard shortcut support
    # Would require custom component or JavaScript injection

    pass  # TODO: Implement when Streamlit adds keyboard API


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_turn_info() -> dict:
    """
    Get current turn information.

    Returns:
        Dict with current_turn, total_turns, progress
    """
    session_data = st.session_state.get('session_data')
    if not session_data:
        return {
            'current_turn': 0,
            'total_turns': 0,
            'progress': 0.0,
            'display_turn': 1,
        }

    total_turns = len(session_data.get('turns', []))
    current_turn_index = get_current_turn()
    current_turn_display = current_turn_index + 1
    progress = (current_turn_display / total_turns) if total_turns > 0 else 0.0

    return {
        'current_turn': current_turn_index,
        'total_turns': total_turns,
        'progress': progress,
        'display_turn': current_turn_display,
    }


def is_first_turn() -> bool:
    """Check if currently at first turn."""
    return get_current_turn() == 0


def is_last_turn() -> bool:
    """Check if currently at last turn."""
    session_data = st.session_state.get('session_data')
    if not session_data:
        return False

    total_turns = len(session_data.get('turns', []))
    return get_current_turn() == total_turns - 1


def get_relative_position() -> str:
    """
    Get relative position in conversation.

    Returns:
        'beginning' | 'middle' | 'end'
    """
    info = get_turn_info()
    progress = info['progress']

    if progress < 0.2:
        return 'beginning'
    elif progress > 0.8:
        return 'end'
    else:
        return 'middle'


# ============================================================================
# STYLING
# ============================================================================

def get_turn_indicator_css() -> str:
    """
    Get CSS for turn indicator styling.

    Returns:
        CSS string for custom styling
    """
    return """
    <style>
    .turn-indicator {
        display: inline-block;
        padding: 8px 16px;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        font-family: 'SF Mono', 'Monaco', 'Courier New', monospace;
        font-size: 14px;
        font-weight: 600;
        color: #fff;
        text-align: center;
    }

    .turn-indicator-label {
        font-size: 11px;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .turn-indicator-value {
        font-size: 18px;
        font-weight: 700;
        color: #4CAF50;
    }

    .turn-indicator-total {
        font-size: 14px;
        color: #aaa;
    }
    </style>
    """
