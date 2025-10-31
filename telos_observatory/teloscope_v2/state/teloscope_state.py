"""
TELOSCOPE v2 - Centralized State Management

This module provides structured state management for the TELOSCOPE control system.
All TELOSCOPE-related state is organized under st.session_state.teloscope namespace.

State Structure:
    st.session_state.teloscope = {
        'current_turn': int,           # Active turn index (0-based)
        'playing': bool,               # Play/pause state
        'playback_speed': float,       # Speed multiplier (0.5-2.0)
        'last_play_time': float,       # Timestamp for autoplay timing
        'position': str,               # 'fixed-bottom' | 'floating' | 'hidden'
        'float_x': int,                # Floating position X (pixels)
        'float_y': int,                # Floating position Y (pixels)
        'active_tools': dict,          # Tool visibility toggles
        'timeline_markers': list,      # Cached timeline markers
        'initialized': bool,           # Initialization flag
    }

Migration Note:
    This structured state coexists with Phase 1 flat state during transition.
    See: docs/TELOSCOPE_INTEGRATION_RECONCILIATION.md for migration strategy.
"""

import streamlit as st
from typing import Any, Dict, Optional
import time


# ============================================================================
# DEFAULT STATE VALUES
# ============================================================================

DEFAULT_TELOSCOPE_STATE = {
    'current_turn': 0,
    'playing': False,
    'playback_speed': 1.0,
    'last_play_time': 0.0,
    'position': 'fixed-bottom',      # fixed-bottom | floating | hidden
    'float_x': 100,                   # Default floating position
    'float_y': 100,
    'active_tools': {
        'comparison_viewer': False,
        'calculation_window': False,
        'turn_navigator': False,
        'steward_chat': False,
    },
    'timeline_markers': [],           # Cached marker data
    'initialized': True,
}


# ============================================================================
# INITIALIZATION
# ============================================================================

def init_teloscope_state(session_data: Optional[Dict] = None) -> None:
    """
    Initialize TELOSCOPE state namespace.

    Creates st.session_state.teloscope with default values if not present.
    Safe to call multiple times - only initializes once.

    Args:
        session_data: Optional session data to validate turn count

    Usage:
        init_teloscope_state(session_data=mock_session)
    """
    # Only initialize if teloscope namespace doesn't exist
    if 'teloscope' not in st.session_state:
        st.session_state.teloscope = DEFAULT_TELOSCOPE_STATE.copy()

        # Deep copy nested dicts
        st.session_state.teloscope['active_tools'] = DEFAULT_TELOSCOPE_STATE['active_tools'].copy()
        st.session_state.teloscope['timeline_markers'] = []

    # Validate current_turn if session_data provided
    if session_data:
        max_turn = len(session_data.get('turns', [])) - 1
        if st.session_state.teloscope['current_turn'] > max_turn:
            st.session_state.teloscope['current_turn'] = max_turn


def reset_teloscope_state() -> None:
    """
    Reset TELOSCOPE state to defaults.

    Useful for:
    - Loading new session
    - Resetting after error
    - Testing
    """
    if 'teloscope' in st.session_state:
        del st.session_state.teloscope

    init_teloscope_state()


# ============================================================================
# STATE ACCESS
# ============================================================================

def get_teloscope_state(key: Optional[str] = None, default: Any = None) -> Any:
    """
    Get TELOSCOPE state value(s).

    Args:
        key: Specific state key to retrieve. If None, returns entire state dict.
        default: Default value if key not found

    Returns:
        State value or entire state dict

    Usage:
        current_turn = get_teloscope_state('current_turn')
        all_state = get_teloscope_state()
        playing = get_teloscope_state('playing', default=False)
    """
    # Ensure state exists
    if 'teloscope' not in st.session_state:
        init_teloscope_state()

    # Return entire state if no key specified
    if key is None:
        return st.session_state.teloscope

    # Return specific value
    return st.session_state.teloscope.get(key, default)


def update_teloscope_state(key: str, value: Any, rerun: bool = False) -> None:
    """
    Update TELOSCOPE state value.

    Args:
        key: State key to update
        value: New value
        rerun: If True, trigger st.rerun() after update

    Usage:
        update_teloscope_state('current_turn', 5)
        update_teloscope_state('playing', True, rerun=True)
    """
    # Ensure state exists
    if 'teloscope' not in st.session_state:
        init_teloscope_state()

    # Validate key exists in schema
    if key not in DEFAULT_TELOSCOPE_STATE:
        raise ValueError(f"Invalid TELOSCOPE state key: {key}")

    # Update value
    st.session_state.teloscope[key] = value

    # Trigger rerun if requested
    if rerun:
        st.rerun()


def update_teloscope_state_batch(updates: Dict[str, Any], rerun: bool = False) -> None:
    """
    Update multiple TELOSCOPE state values at once.

    Args:
        updates: Dict of {key: value} pairs to update
        rerun: If True, trigger st.rerun() after all updates

    Usage:
        update_teloscope_state_batch({
            'current_turn': 5,
            'playing': False,
        }, rerun=True)
    """
    # Ensure state exists
    if 'teloscope' not in st.session_state:
        init_teloscope_state()

    # Validate all keys
    for key in updates.keys():
        if key not in DEFAULT_TELOSCOPE_STATE:
            raise ValueError(f"Invalid TELOSCOPE state key: {key}")

    # Apply all updates
    st.session_state.teloscope.update(updates)

    # Trigger rerun if requested
    if rerun:
        st.rerun()


# ============================================================================
# NAVIGATION STATE HELPERS
# ============================================================================

def get_current_turn() -> int:
    """Get current turn index."""
    return get_teloscope_state('current_turn', 0)


def set_current_turn(turn: int, rerun: bool = True) -> None:
    """
    Set current turn index.

    Args:
        turn: Turn index (0-based)
        rerun: If True, trigger st.rerun()
    """
    update_teloscope_state('current_turn', turn, rerun=rerun)


def is_playing() -> bool:
    """Check if autoplay is active."""
    return get_teloscope_state('playing', False)


def set_playing(playing: bool, rerun: bool = True) -> None:
    """
    Set play/pause state.

    Args:
        playing: True to start autoplay, False to pause
        rerun: If True, trigger st.rerun()
    """
    updates = {'playing': playing}

    # Record timestamp when starting play
    if playing:
        updates['last_play_time'] = time.time()

    update_teloscope_state_batch(updates, rerun=rerun)


def get_playback_speed() -> float:
    """Get current playback speed multiplier."""
    return get_teloscope_state('playback_speed', 1.0)


def set_playback_speed(speed: float, rerun: bool = False) -> None:
    """
    Set playback speed.

    Args:
        speed: Speed multiplier (0.5-2.0)
        rerun: If True, trigger st.rerun()
    """
    # Clamp to valid range
    speed = max(0.5, min(2.0, speed))
    update_teloscope_state('playback_speed', speed, rerun=rerun)


# ============================================================================
# TOOL STATE HELPERS
# ============================================================================

def is_tool_active(tool_name: str) -> bool:
    """
    Check if a tool is currently active.

    Args:
        tool_name: Tool identifier (e.g., 'comparison_viewer')

    Returns:
        True if tool is active, False otherwise
    """
    active_tools = get_teloscope_state('active_tools', {})
    return active_tools.get(tool_name, False)


def toggle_tool(tool_name: str, rerun: bool = True) -> None:
    """
    Toggle tool visibility.

    Args:
        tool_name: Tool identifier
        rerun: If True, trigger st.rerun()
    """
    active_tools = get_teloscope_state('active_tools', {})

    # Toggle state
    active_tools[tool_name] = not active_tools.get(tool_name, False)

    # Update state
    update_teloscope_state('active_tools', active_tools, rerun=rerun)


def set_tool_active(tool_name: str, active: bool, rerun: bool = True) -> None:
    """
    Set tool visibility directly.

    Args:
        tool_name: Tool identifier
        active: True to show, False to hide
        rerun: If True, trigger st.rerun()
    """
    active_tools = get_teloscope_state('active_tools', {})
    active_tools[tool_name] = active
    update_teloscope_state('active_tools', active_tools, rerun=rerun)


# ============================================================================
# POSITION STATE HELPERS
# ============================================================================

def get_teloscope_position() -> str:
    """
    Get TELOSCOPE position mode.

    Returns:
        'fixed-bottom' | 'floating' | 'hidden'
    """
    return get_teloscope_state('position', 'fixed-bottom')


def set_teloscope_position(position: str, rerun: bool = True) -> None:
    """
    Set TELOSCOPE position mode.

    Args:
        position: 'fixed-bottom' | 'floating' | 'hidden'
        rerun: If True, trigger st.rerun()
    """
    valid_positions = ['fixed-bottom', 'floating', 'hidden']
    if position not in valid_positions:
        raise ValueError(f"Invalid position: {position}. Must be one of {valid_positions}")

    update_teloscope_state('position', position, rerun=rerun)


def get_floating_position() -> tuple[int, int]:
    """
    Get floating position coordinates.

    Returns:
        (x, y) tuple in pixels
    """
    x = get_teloscope_state('float_x', 100)
    y = get_teloscope_state('float_y', 100)
    return (x, y)


def set_floating_position(x: int, y: int, rerun: bool = False) -> None:
    """
    Set floating position coordinates.

    Args:
        x: X coordinate (pixels)
        y: Y coordinate (pixels)
        rerun: If True, trigger st.rerun()
    """
    update_teloscope_state_batch({
        'float_x': x,
        'float_y': y,
    }, rerun=rerun)


# ============================================================================
# TIMELINE CACHE HELPERS
# ============================================================================

def get_timeline_markers() -> list:
    """Get cached timeline markers."""
    return get_teloscope_state('timeline_markers', [])


def set_timeline_markers(markers: list, rerun: bool = False) -> None:
    """
    Cache timeline markers.

    Args:
        markers: List of marker data dicts
        rerun: If True, trigger st.rerun()
    """
    update_teloscope_state('timeline_markers', markers, rerun=rerun)


# ============================================================================
# MIGRATION HELPERS (Phase 1 → Spec)
# ============================================================================

def migrate_from_flat_state() -> bool:
    """
    Migrate from Phase 1 flat state to TELOSCOPE v2 nested state.

    Looks for Phase 1 state variables and copies values to teloscope namespace.
    Safe to call multiple times.

    Returns:
        True if migration performed, False if no migration needed
    """
    migrated = False

    # Only migrate if teloscope namespace doesn't exist yet
    if 'teloscope' not in st.session_state:
        init_teloscope_state()

        # Migrate current_turn
        if 'current_turn' in st.session_state:
            st.session_state.teloscope['current_turn'] = st.session_state.current_turn
            migrated = True

        # Migrate playing
        if 'playing' in st.session_state:
            st.session_state.teloscope['playing'] = st.session_state.playing
            migrated = True

        # Migrate playback_speed
        if 'playback_speed' in st.session_state:
            st.session_state.teloscope['playback_speed'] = st.session_state.playback_speed
            migrated = True

        # Migrate last_play_time
        if 'last_play_time' in st.session_state:
            st.session_state.teloscope['last_play_time'] = st.session_state.last_play_time
            migrated = True

    return migrated


def sync_to_flat_state() -> None:
    """
    Sync TELOSCOPE v2 state back to Phase 1 flat variables.

    Useful during coexistence period when both systems need same values.
    """
    if 'teloscope' not in st.session_state:
        return

    # Sync core navigation state
    st.session_state.current_turn = get_teloscope_state('current_turn', 0)
    st.session_state.playing = get_teloscope_state('playing', False)
    st.session_state.playback_speed = get_teloscope_state('playback_speed', 1.0)
    st.session_state.last_play_time = get_teloscope_state('last_play_time', 0.0)


# ============================================================================
# DEBUG / INSPECTION
# ============================================================================

def get_state_summary() -> Dict[str, Any]:
    """
    Get summary of TELOSCOPE state for debugging.

    Returns:
        Dict with key state values
    """
    if 'teloscope' not in st.session_state:
        return {'error': 'TELOSCOPE state not initialized'}

    return {
        'current_turn': get_current_turn(),
        'playing': is_playing(),
        'playback_speed': get_playback_speed(),
        'position': get_teloscope_position(),
        'active_tools': get_teloscope_state('active_tools', {}),
        'initialized': True,
    }


def print_state_summary() -> None:
    """Print TELOSCOPE state summary (for debugging)."""
    summary = get_state_summary()
    print("\n=== TELOSCOPE State Summary ===")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print("=" * 30 + "\n")
