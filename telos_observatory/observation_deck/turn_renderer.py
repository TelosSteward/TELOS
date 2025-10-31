"""
Turn Renderer with Distance-Based Dimming
==========================================

Renders conversation turns with visual dimming based on distance from
active turn. Provides visual focus on current turn while maintaining
context of surrounding conversation.

Dimming Algorithm:
- Active turn (distance 0): opacity 1.0, gold border
- Adjacent (distance 1): opacity 0.7
- Two away (distance 2): opacity 0.4
- Far (distance 3+): opacity 0.2

All transitions: 0.3s ease animation
"""

import streamlit as st
import streamlit.components.v1 as components
from typing import Dict, Any, List


def calculate_turn_opacity(turn_index: int, active_turn_index: int) -> float:
    """
    Calculate opacity based on distance from active turn.

    Args:
        turn_index: Index of turn being rendered
        active_turn_index: Index of currently active turn

    Returns:
        Opacity value between 0.2 and 1.0
    """
    distance = abs(turn_index - active_turn_index)

    if distance == 0:
        return 1.0  # Active turn: full brightness
    elif distance == 1:
        return 0.7  # Adjacent: slight dim
    elif distance == 2:
        return 0.4  # Two away: more dim
    else:
        return 0.2  # Far: very dim but visible


def get_turn_style(turn_index: int, active_turn_index: int, turn_data: Dict[str, Any]) -> str:
    """
    Generate CSS style for turn based on distance from active turn.

    Args:
        turn_index: Index of turn being rendered
        active_turn_index: Index of currently active turn
        turn_data: Turn data dictionary

    Returns:
        CSS style string
    """
    opacity = calculate_turn_opacity(turn_index, active_turn_index)
    is_active = (turn_index == active_turn_index)

    # Border styling
    if is_active:
        border = "3px solid #FFD700"
        box_shadow = "0 0 15px rgba(255, 215, 0, 0.4)"
    else:
        border = "1px solid rgba(255, 255, 255, 0.1)"
        box_shadow = "none"

    # Background color based on status
    status = turn_data.get('status', '✓')
    if status == '⚠️':  # Drift
        bg_color = "rgba(255, 165, 0, 0.1)"
    elif status == '⚡':  # Intervention
        bg_color = "rgba(255, 215, 0, 0.1)"
    elif status == '⚙️':  # Calibration
        bg_color = "rgba(100, 149, 237, 0.1)"
    else:  # Stable
        bg_color = "rgba(255, 255, 255, 0.05)"

    return f"""
        opacity: {opacity};
        border: {border};
        box-shadow: {box_shadow};
        background: {bg_color};
        transition: all 0.3s ease;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 8px;
    """


def render_turn_message(turn_data: Dict[str, Any], turn_index: int, active_turn_index: int):
    """
    Render a single turn's messages with styling.

    Args:
        turn_data: Turn data dictionary
        turn_index: Index of this turn
        active_turn_index: Index of active turn
    """
    style = get_turn_style(turn_index, active_turn_index, turn_data)

    # Turn container with ID for scrolling
    turn_id = f"turn_{turn_index}"

    # Status icon
    status_icon = turn_data.get('status', '✓')
    status_text = turn_data.get('status_text', 'Stable')

    # Fidelity display
    fidelity = turn_data.get('fidelity')
    if fidelity is not None:
        fidelity_str = f"{fidelity:.2f}"
        if fidelity >= 0.8:
            fidelity_color = "#90EE90"  # Light green
        elif fidelity >= 0.6:
            fidelity_color = "#FFD700"  # Gold
        else:
            fidelity_color = "#FF6B6B"  # Light red
    else:
        fidelity_str = "Cal"  # Calibration
        fidelity_color = "#6495ED"  # Cornflower blue

    html = f"""
    <div id="{turn_id}" style="{style}">
        <!-- Turn header -->
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span style="font-size: 1.2rem;">{status_icon}</span>
                <span style="font-size: 0.85rem; color: #888;">Turn {turn_data['turn']}</span>
                <span style="font-size: 0.75rem; color: #666;">• {turn_data.get('timestamp', 0):.1f}s</span>
            </div>
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span style="font-size: 0.75rem; color: #888;">{status_text}</span>
                <span style="
                    background: {fidelity_color};
                    color: #000;
                    padding: 0.15rem 0.4rem;
                    border-radius: 4px;
                    font-size: 0.75rem;
                    font-weight: bold;
                ">F: {fidelity_str}</span>
            </div>
        </div>

        <!-- User message -->
        <div style="
            background: rgba(0, 123, 255, 0.1);
            border-left: 3px solid #007BFF;
            padding: 0.75rem;
            margin: 0.5rem 0;
            border-radius: 4px;
        ">
            <div style="font-size: 0.75rem; color: #007BFF; margin-bottom: 0.25rem; font-weight: bold;">User</div>
            <div style="color: #E0E0E0; line-height: 1.5;">{turn_data.get('user_input', '')}</div>
        </div>

        <!-- Assistant response -->
        <div style="
            background: rgba(40, 167, 69, 0.1);
            border-left: 3px solid #28A745;
            padding: 0.75rem;
            margin: 0.5rem 0;
            border-radius: 4px;
        ">
            <div style="font-size: 0.75rem; color: #28A745; margin-bottom: 0.25rem; font-weight: bold;">Assistant</div>
            <div style="color: #E0E0E0; line-height: 1.5;">{turn_data.get('response', '')}</div>
        </div>

        <!-- Intervention indicator -->
        {'<div style="font-size: 0.75rem; color: #FFD700; margin-top: 0.5rem;">⚡ Governance intervention applied</div>' if turn_data.get('intervention_applied') else ''}

        <!-- Drift indicator -->
        {'<div style="font-size: 0.75rem; color: #FF6B6B; margin-top: 0.5rem;">⚠️ Drift detected</div>' if turn_data.get('drift_detected') else ''}
    </div>
    """

    st.markdown(html, unsafe_allow_html=True)


def scroll_to_active_turn(active_turn_index: int):
    """
    Scroll viewport to active turn with smooth animation.

    Args:
        active_turn_index: Index of turn to scroll to
    """
    # JavaScript to scroll to active turn
    scroll_js = f"""
    <script>
        setTimeout(function() {{
            const turnElement = document.getElementById('turn_{active_turn_index}');
            if (turnElement) {{
                turnElement.scrollIntoView({{
                    behavior: 'smooth',
                    block: 'center'
                }});
            }}
        }}, 100);
    </script>
    """

    components.html(scroll_js, height=0)


def render_chat_viewport():
    """
    Render scrollable chat viewport with all turns and dimming applied.

    Reads from st.session_state:
    - session_data: Session with turns
    - current_turn: Active turn index (0-based)
    """
    # Get state
    session_data = st.session_state.get('session_data', {})
    turns = session_data.get('turns', [])
    active_turn_index = st.session_state.get('current_turn', 0)

    if not turns:
        st.info("No session data available")
        return

    # Render all turns with dimming
    viewport_container = st.container()

    with viewport_container:
        for idx, turn in enumerate(turns):
            render_turn_message(turn, idx, active_turn_index)

    # Auto-scroll to active turn
    scroll_to_active_turn(active_turn_index)


def render_control_strip():
    """
    Render clickable control strip at top showing current turn and fidelity.
    Clicking toggles the Observation Deck sidebar.

    Reads from st.session_state:
    - current_turn: Active turn index
    - session_data: Session with turns
    - deck_expanded: Observation Deck visibility
    """
    session_data = st.session_state.get('session_data', {})
    turns = session_data.get('turns', [])
    active_turn_index = st.session_state.get('current_turn', 0)
    deck_expanded = st.session_state.get('deck_expanded', False)

    if not turns or active_turn_index >= len(turns):
        return

    current_turn = turns[active_turn_index]

    # Control strip styling with hover and click effects
    st.markdown("""
        <style>
        .control-strip {
            position: fixed;
            top: 60px;
            right: 20px;
            background: rgba(0, 0, 0, 0.8);
            backdrop-filter: blur(10px);
            padding: 0.75rem 1.25rem;
            border-radius: 8px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            z-index: 1000;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .control-strip:hover {
            background: rgba(20, 30, 40, 0.9);
            border: 1px solid rgba(255, 215, 0, 0.3);
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.4);
        }

        .control-strip.active {
            border: 1px solid rgba(255, 215, 0, 0.5);
            background: rgba(30, 40, 50, 0.9);
        }

        .control-strip-hint {
            font-size: 0.65rem;
            color: #666;
            text-align: center;
            margin-top: 0.25rem;
        }
        </style>
    """, unsafe_allow_html=True)

    # Control strip content
    fidelity = current_turn.get('fidelity')
    fidelity_display = f"{fidelity:.2f}" if fidelity is not None else "Calibrating"

    status_icon = current_turn.get('status', '✓')
    status_text = current_turn.get('status_text', 'Stable')

    # Add "active" class if deck is expanded
    active_class = "active" if deck_expanded else ""

    # Render as clickable button
    col1, col2 = st.columns([10, 1])

    with col1:
        # This is just for layout - actual click handled by button below
        control_html = f"""
        <div class="control-strip {active_class}" title="Click to toggle Observation Deck (Shift+O)">
            <div style="display: flex; align-items: center; gap: 1rem;">
                <div>
                    <div style="font-size: 0.75rem; color: #888; margin-bottom: 0.25rem;">Turn</div>
                    <div style="font-size: 1.25rem; font-weight: bold; color: #FFF;">{active_turn_index + 1} / {len(turns)}</div>
                </div>
                <div style="border-left: 1px solid rgba(255,255,255,0.2); height: 40px;"></div>
                <div>
                    <div style="font-size: 0.75rem; color: #888; margin-bottom: 0.25rem;">Fidelity</div>
                    <div style="font-size: 1.25rem; font-weight: bold; color: #FFD700;">{fidelity_display}</div>
                </div>
                <div style="border-left: 1px solid rgba(255,255,255,0.2); height: 40px;"></div>
                <div>
                    <div style="font-size: 0.75rem; color: #888; margin-bottom: 0.25rem;">Status</div>
                    <div style="font-size: 1rem; color: #FFF;">{status_icon} {status_text}</div>
                </div>
                <div style="border-left: 1px solid rgba(255,255,255,0.2); height: 40px;"></div>
                <div style="text-align: center;">
                    <div style="font-size: 1.2rem; color: #FFD700;">🔭</div>
                    <div class="control-strip-hint">Click to observe</div>
                </div>
            </div>
        </div>
        """

        st.markdown(control_html, unsafe_allow_html=True)

    with col2:
        # Invisible button overlay for Streamlit interaction
        if st.button("🔭", key="control_strip_toggle", help="Toggle Observation Deck (Shift+O)"):
            st.session_state.deck_expanded = not deck_expanded
            st.rerun()
