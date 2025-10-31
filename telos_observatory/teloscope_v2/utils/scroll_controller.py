"""
TELOSCOPE v2 - Scroll Controller

Manages viewport scrolling to synchronize with turn navigation.

Extracted from Phase 1 turn_renderer.py with enhancements:
- Smooth scroll behavior
- Multiple scroll strategies (instant, smooth, delayed)
- Scroll offset configuration (header compensation)
- Scroll position caching
- Viewport visibility detection

Usage:
    from teloscope_v2.utils.scroll_controller import scroll_to_turn

    # Scroll to turn 5 with smooth animation
    scroll_to_turn(turn_index=5, behavior='smooth')

Note:
    Streamlit doesn't have native scroll control, so this module uses
    JavaScript injection via st.components.html() or HTML anchors.

State Management:
    - Uses centralized teloscope_state for navigation state (current_turn)
    - Scroll-specific state (scroll_target, scroll_position) remains flat
    - session_data is intentionally shared between Phase 1 and v2
"""

import streamlit as st
from typing import Literal, Optional

# Import centralized state management
try:
    from ..state.teloscope_state import get_current_turn
except ImportError:
    # Fallback for standalone testing
    def get_current_turn():
        return st.session_state.get('current_turn', 0)


# ============================================================================
# SCROLL STRATEGIES
# ============================================================================

ScrollBehavior = Literal['instant', 'smooth', 'delayed']


def scroll_to_turn(
    turn_index: int,
    behavior: ScrollBehavior = 'smooth',
    offset: int = 100,
    delay_ms: int = 100,
) -> None:
    """
    Scroll viewport to show specified turn.

    Args:
        turn_index: Turn index to scroll to (0-based)
        behavior: 'instant' (immediate), 'smooth' (animated), 'delayed' (wait then scroll)
        offset: Pixels from top of viewport (to account for fixed headers)
        delay_ms: Delay in milliseconds (for 'delayed' behavior)

    Limitations:
        Streamlit doesn't natively support JavaScript execution,
        so this uses HTML anchor method which has limitations.
    """
    # Generate unique anchor for turn
    anchor_id = f"turn_{turn_index}"

    # Store scroll target in session state
    st.session_state.scroll_target = {
        'turn_index': turn_index,
        'anchor_id': anchor_id,
        'behavior': behavior,
        'offset': offset,
        'delay_ms': delay_ms,
    }

    # The actual scrolling happens in render_scroll_anchor()
    # which should be called in the turn rendering function


def render_scroll_anchor(turn_index: int) -> None:
    """
    Render invisible scroll anchor for a turn.

    Should be called during turn rendering to place anchors.

    Args:
        turn_index: Turn index (0-based)

    Usage:
        for idx, turn in enumerate(turns):
            render_scroll_anchor(idx)
            render_turn_content(turn)
    """
    anchor_id = f"turn_{turn_index}"

    # Check if this is the target turn
    scroll_target = st.session_state.get('scroll_target')
    is_target = (scroll_target and scroll_target['turn_index'] == turn_index)

    if is_target:
        # Render visible anchor that browser will scroll to
        st.markdown(
            f'<div id="{anchor_id}" style="scroll-margin-top: {scroll_target["offset"]}px;"></div>',
            unsafe_allow_html=True
        )

        # Use JavaScript to trigger scroll (if behavior is not instant)
        if scroll_target['behavior'] == 'smooth':
            _inject_smooth_scroll(anchor_id, scroll_target['delay_ms'])
        elif scroll_target['behavior'] == 'delayed':
            _inject_delayed_scroll(anchor_id, scroll_target['delay_ms'])

        # Clear target after scrolling
        st.session_state.scroll_target = None

    else:
        # Render invisible anchor for future use
        st.markdown(
            f'<div id="{anchor_id}"></div>',
            unsafe_allow_html=True
        )


def _inject_smooth_scroll(anchor_id: str, delay_ms: int = 0) -> None:
    """
    Inject JavaScript for smooth scrolling.

    Note: Streamlit's security model may block JavaScript execution.
    This is a best-effort implementation.
    """
    script = f"""
    <script>
    setTimeout(function() {{
        const element = document.getElementById('{anchor_id}');
        if (element) {{
            element.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
        }}
    }}, {delay_ms});
    </script>
    """

    # Note: st.markdown doesn't execute <script> tags
    # Would need st.components.html() but that creates iframe
    # For now, use CSS-based approach (scroll-margin-top in render_scroll_anchor)


def _inject_delayed_scroll(anchor_id: str, delay_ms: int) -> None:
    """Inject JavaScript for delayed scrolling."""
    # Similar limitation as _inject_smooth_scroll
    pass


# ============================================================================
# ALTERNATIVE: STREAMLIT ANCHOR METHOD
# ============================================================================

def scroll_to_turn_anchor(turn_index: int) -> None:
    """
    Scroll using Streamlit's query params method.

    This is more reliable than JavaScript injection.

    Args:
        turn_index: Turn index to scroll to

    Usage:
        # After navigation
        scroll_to_turn_anchor(5)

    Note:
        Requires anchor elements rendered with id="turn_5"
    """
    # Set query param to trigger browser scroll
    st.query_params['scroll_to'] = f"turn_{turn_index}"


def get_scroll_target() -> Optional[int]:
    """
    Get turn index from scroll query param.

    Returns:
        Turn index if scroll param present, None otherwise

    Usage:
        target_turn = get_scroll_target()
        if target_turn is not None:
            # Highlight or process target turn
    """
    scroll_param = st.query_params.get('scroll_to')
    if scroll_param and scroll_param.startswith('turn_'):
        try:
            return int(scroll_param.replace('turn_', ''))
        except ValueError:
            return None
    return None


def clear_scroll_target() -> None:
    """Clear scroll query param."""
    if 'scroll_to' in st.query_params:
        del st.query_params['scroll_to']


# ============================================================================
# VIEWPORT VISIBILITY DETECTION
# ============================================================================

def is_turn_in_viewport(turn_index: int, viewport_height: int = 800) -> bool:
    """
    Check if turn is likely visible in viewport.

    This is an approximation since Streamlit doesn't provide
    viewport information.

    Args:
        turn_index: Turn index to check
        viewport_height: Assumed viewport height in pixels

    Returns:
        True if turn is likely visible, False otherwise

    Note:
        This is a rough estimate based on turn count and average turn height.
    """
    # Get current turn from centralized state
    current_turn = get_current_turn()

    # Estimate: turns within ±2 of current are likely visible
    distance = abs(turn_index - current_turn)

    return distance <= 2


def calculate_scroll_offset(
    turn_index: int,
    turn_height: int = 150,
    header_height: int = 80,
    teloscope_height: int = 120,
) -> int:
    """
    Calculate scroll offset for centering a turn.

    Args:
        turn_index: Turn to center
        turn_height: Average height per turn (pixels)
        header_height: Fixed header height
        teloscope_height: Fixed TELOSCOPE height

    Returns:
        Scroll offset in pixels
    """
    # Calculate turn position
    turn_position = turn_index * turn_height

    # Account for fixed elements
    fixed_offset = header_height + teloscope_height

    # Center turn in remaining viewport space
    viewport_center_offset = 200  # Rough estimate

    return turn_position - viewport_center_offset + fixed_offset


# ============================================================================
# SCROLL POSITION TRACKING
# ============================================================================

def save_scroll_position(position: int) -> None:
    """
    Save current scroll position to session state.

    Args:
        position: Scroll position in pixels

    Usage:
        save_scroll_position(window.scrollY)  # From JavaScript
    """
    st.session_state.scroll_position = position


def restore_scroll_position() -> Optional[int]:
    """
    Get saved scroll position.

    Returns:
        Scroll position if saved, None otherwise
    """
    return st.session_state.get('scroll_position')


def clear_scroll_position() -> None:
    """Clear saved scroll position."""
    if 'scroll_position' in st.session_state:
        del st.session_state.scroll_position


# ============================================================================
# EXPERIMENTAL: AUTO-SCROLL DURING PLAYBACK
# ============================================================================

def enable_auto_scroll(enabled: bool = True) -> None:
    """
    Enable/disable auto-scroll during playback.

    When enabled, viewport automatically scrolls to follow active turn.

    Args:
        enabled: True to enable, False to disable
    """
    st.session_state.auto_scroll_enabled = enabled


def is_auto_scroll_enabled() -> bool:
    """Check if auto-scroll is enabled."""
    return st.session_state.get('auto_scroll_enabled', True)


def auto_scroll_to_active_turn() -> None:
    """
    Auto-scroll to active turn if enabled.

    Call this during turn rendering to maintain active turn visibility.

    Usage:
        if is_auto_scroll_enabled():
            auto_scroll_to_active_turn()
    """
    if not is_auto_scroll_enabled():
        return

    # Get current turn from centralized state
    current_turn = get_current_turn()
    scroll_to_turn(current_turn, behavior='smooth')


# ============================================================================
# DIMMING INTEGRATION (from Phase 1)
# ============================================================================

def calculate_turn_opacity(
    turn_index: int,
    active_turn_index: int,
    dimming_enabled: bool = True,
) -> float:
    """
    Calculate opacity for turn based on distance from active turn.

    Extracted from Phase 1 turn_renderer.py.

    Args:
        turn_index: Turn to calculate opacity for
        active_turn_index: Currently active turn
        dimming_enabled: If False, all turns have full opacity

    Returns:
        Opacity value (0.0 - 1.0)

    Dimming Algorithm:
        - Active turn (distance 0): 1.0
        - Adjacent (distance 1): 0.7
        - Two away (distance 2): 0.4
        - Far (distance 3+): 0.2
    """
    if not dimming_enabled:
        return 1.0

    distance = abs(turn_index - active_turn_index)

    if distance == 0:
        return 1.0
    elif distance == 1:
        return 0.7
    elif distance == 2:
        return 0.4
    else:
        return 0.2


def get_turn_border_style(
    turn_index: int,
    active_turn_index: int,
) -> str:
    """
    Get border CSS for turn based on active state.

    Args:
        turn_index: Turn to style
        active_turn_index: Currently active turn

    Returns:
        CSS border string

    Example:
        border_css = get_turn_border_style(5, 5)
        # Returns: "2px solid #FFD700"
    """
    if turn_index == active_turn_index:
        return "2px solid #FFD700"  # Gold border for active
    else:
        return "1px solid rgba(255, 255, 255, 0.1)"  # Subtle border


def get_turn_css_classes(
    turn_index: int,
    active_turn_index: int,
) -> str:
    """
    Get CSS classes for turn element.

    Args:
        turn_index: Turn index
        active_turn_index: Active turn index

    Returns:
        Space-separated CSS classes
    """
    classes = ["turn-message"]

    if turn_index == active_turn_index:
        classes.append("turn-active")

    distance = abs(turn_index - active_turn_index)
    if distance == 1:
        classes.append("turn-adjacent")
    elif distance == 2:
        classes.append("turn-near")
    elif distance >= 3:
        classes.append("turn-far")

    return " ".join(classes)


# ============================================================================
# CSS HELPERS
# ============================================================================

def get_scroll_container_css() -> str:
    """
    Get CSS for scroll container.

    Returns:
        CSS string for smooth scrolling setup
    """
    return """
    <style>
    html {
        scroll-behavior: smooth;
    }

    .scroll-container {
        overflow-y: auto;
        scroll-padding-top: 100px;
    }

    .turn-message {
        scroll-margin-top: 100px;
        transition: opacity 0.3s ease, border 0.3s ease;
    }

    .turn-active {
        border: 2px solid #FFD700 !important;
        box-shadow: 0 0 16px rgba(255, 215, 0, 0.3);
    }

    .turn-adjacent {
        opacity: 0.7;
    }

    .turn-near {
        opacity: 0.4;
    }

    .turn-far {
        opacity: 0.2;
    }
    </style>
    """


# ============================================================================
# DEBUG / TESTING
# ============================================================================

def print_scroll_debug_info() -> None:
    """Print scroll controller debug information."""
    print("\n=== Scroll Controller Debug ===")
    print(f"Scroll Target: {st.session_state.get('scroll_target')}")
    print(f"Scroll Position: {st.session_state.get('scroll_position')}")
    print(f"Auto-Scroll Enabled: {is_auto_scroll_enabled()}")
    print(f"Query Params: {st.query_params}")
    print("=" * 30 + "\n")


def reset_scroll_state() -> None:
    """Reset all scroll-related session state."""
    keys_to_clear = ['scroll_target', 'scroll_position', 'auto_scroll_enabled']
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
