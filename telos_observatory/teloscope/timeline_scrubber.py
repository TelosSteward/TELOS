"""
TELOSCOPE Timeline Scrubber
============================

Interactive timeline slider with visual turn markers.

Markers color-coded by status:
- ✓ Green: Stable turns
- ⚠️ Orange: Drift detected
- ⚡ Yellow: Intervention applied
- ⚙️ Blue: Calibration phase

CRITICAL: Follows Streamlit callback pattern
"""

import streamlit as st


# ============================================================================
# Callbacks (State Updates Only)
# ============================================================================

def on_timeline_change():
    """
    Handle timeline scrubber movement - callback.

    Streamlit auto-syncs slider value to st.session_state.timeline_slider
    because we use key="timeline_slider"
    """
    # Slider value is automatically in st.session_state.timeline_slider
    new_turn = st.session_state.timeline_slider
    st.session_state.current_turn = new_turn
    st.rerun()


# ============================================================================
# Rendering (UI Display Only)
# ============================================================================

def get_turn_marker_html(turns: list, current_turn: int) -> str:
    """
    Generate HTML for turn markers above timeline.

    Args:
        turns: List of turn dictionaries
        current_turn: Active turn index

    Returns:
        HTML string with marker visualization
    """
    if not turns:
        return ""

    # Generate markers
    markers = []
    for idx, turn in enumerate(turns):
        status = turn.get('status', '✓')
        is_active = (idx == current_turn)

        # Status colors
        status_colors = {
            '✓': '#90EE90',  # Light green
            '⚠️': '#FFA500',  # Orange
            '⚡': '#FFD700',  # Gold
            '⚙️': '#6495ED'  # Cornflower blue
        }

        color = status_colors.get(status, '#90EE90')
        size = "12px" if is_active else "8px"
        opacity = "1.0" if is_active else "0.7"

        marker_style = f"""
            width: {size};
            height: {size};
            background: {color};
            border-radius: 50%;
            display: inline-block;
            margin: 0 2px;
            opacity: {opacity};
            transition: all 0.3s ease;
        """

        if is_active:
            marker_style += "box-shadow: 0 0 8px " + color + ";"

        markers.append(f'<span style="{marker_style}" title="Turn {idx + 1}: {status}"></span>')

    return '<div style="text-align: center; margin-bottom: 0.5rem;">' + ''.join(markers) + '</div>'


def render_timeline_scrubber():
    """
    Render interactive timeline scrubber with turn markers.

    Reads from st.session_state:
    - current_turn: Active turn index
    - session_data: Session with turns
    """
    session_data = st.session_state.get('session_data', {})
    turns = session_data.get('turns', [])

    if not turns:
        st.info("No timeline data available")
        return

    current_turn = st.session_state.get('current_turn', 0)
    max_turns = len(turns) - 1

    # Render turn markers
    marker_html = get_turn_marker_html(turns, current_turn)
    st.markdown(marker_html, unsafe_allow_html=True)

    # Timeline slider
    st.slider(
        label="Timeline",
        min_value=0,
        max_value=max_turns,
        value=current_turn,
        key="timeline_slider",
        on_change=on_timeline_change,
        label_visibility="collapsed"
    )

    # Turn counter
    st.markdown(
        f'<div style="text-align: center; color: #888; font-size: 0.85rem; margin-top: -0.5rem;">Turn {current_turn + 1} / {len(turns)}</div>',
        unsafe_allow_html=True
    )


def render_timeline_legend():
    """
    Render legend explaining timeline marker colors.
    """
    legend_html = """
    <div style="
        background: rgba(0, 0, 0, 0.3);
        padding: 0.5rem 1rem;
        border-radius: 4px;
        margin-top: 0.5rem;
    ">
        <div style="font-size: 0.75rem; color: #888; margin-bottom: 0.25rem;">Legend:</div>
        <div style="display: flex; gap: 1rem; flex-wrap: wrap; font-size: 0.75rem;">
            <span><span style="color: #90EE90;">●</span> Stable</span>
            <span><span style="color: #FFA500;">●</span> Drift</span>
            <span><span style="color: #FFD700;">●</span> Intervention</span>
            <span><span style="color: #6495ED;">●</span> Calibration</span>
        </div>
    </div>
    """

    st.markdown(legend_html, unsafe_allow_html=True)
