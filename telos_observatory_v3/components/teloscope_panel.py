"""
TELOSCOPE Panel - Research Instrument View for TELOS Observatory.

An expanded granular view accessible from Alignment Lens that provides:
- Fidelity trajectory visualization (grows with each turn)
- Compact gauge meters for User/AI/Primacy State
- Integrated Steward chat interface
- Research instrument aesthetic using TELOS visual library

Design Philosophy:
- Research instrument that reveals governance in action
- Uses existing TELOS visual library (colors, glassmorphism, glow effects)
- Expansion of Alignment Lens, not a replacement
- Direct Steward interaction without extra navigation
"""

import streamlit as st
import streamlit.components.v1 as components
from datetime import datetime
from config.colors import (
    GOLD, get_fidelity_color, get_zone_name,
    STATUS_GOOD, STATUS_MILD, STATUS_MODERATE, STATUS_SEVERE,
    BG_SURFACE, BG_ELEVATED, TEXT_PRIMARY
)
from config.steward_pa import STEWARD_PA
import html
import re


# Zone colors for consistency
ZONE_COLORS = {
    'green': STATUS_GOOD,
    'yellow': STATUS_MILD,
    'orange': STATUS_MODERATE,
    'red': STATUS_SEVERE,
}


def _get_glow_color(hex_color: str, opacity: float = 0.4) -> str:
    """Convert hex color to rgba glow. Handles both 3 and 6 char hex."""
    if not hex_color or hex_color == "None":
        return f"rgba(102, 102, 102, {opacity})"  # Default gray
    hex_color = hex_color.lstrip('#')
    # Handle shorthand hex colors like "666" -> "666666"
    if len(hex_color) == 3:
        hex_color = ''.join([c*2 for c in hex_color])
    # Handle invalid hex colors
    if len(hex_color) != 6:
        return f"rgba(102, 102, 102, {opacity})"  # Default gray
    try:
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return f"rgba({r}, {g}, {b}, {opacity})"
    except ValueError:
        return f"rgba(102, 102, 102, {opacity})"  # Default gray on error


def _md_to_html(text: str) -> str:
    """Convert basic markdown to HTML for display."""
    text = html.escape(text)

    # Zone-colored scores: [[green:0.856]] -> colored span
    def replace_zone_score(match):
        zone = match.group(1)
        value = match.group(2)
        color = ZONE_COLORS.get(zone, GOLD)
        return f'<span style="color: {color}; font-weight: bold;">{value}</span>'

    text = re.sub(r'\[\[(\w+):([^\]]+)\]\]', replace_zone_score, text)
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'\*(.+?)\*', r'<i>\1</i>', text)
    text = re.sub(r'`(.+?)`', r'<code style="background: rgba(0,0,0,0.3); padding: 2px 4px; border-radius: 3px;">\1</code>', text)
    text = text.replace('\n', '<br>')
    return text


class TeloscopePanel:
    """TELOSCOPE - Research instrument panel for granular governance view."""

    def __init__(self):
        """Initialize TELOSCOPE panel."""
        # Initialize Steward LLM if not already done
        if 'beta_steward_llm' not in st.session_state:
            try:
                from services.beta_steward_llm import BetaStewardLLM
                st.session_state.beta_steward_llm = BetaStewardLLM()
                st.session_state.beta_steward_llm_enabled = True
            except (ValueError, ImportError) as e:
                st.session_state.beta_steward_llm = None
                st.session_state.beta_steward_llm_enabled = False

        # Initialize navigation state
        if 'teloscope_selected_turn' not in st.session_state:
            st.session_state.teloscope_selected_turn = None  # None = live mode (latest)

        # Initialize scatter plot visibility toggles (USER on by default)
        if 'teloscope_show_user' not in st.session_state:
            st.session_state.teloscope_show_user = True
        if 'teloscope_show_ai' not in st.session_state:
            st.session_state.teloscope_show_ai = False
        if 'teloscope_show_ps' not in st.session_state:
            st.session_state.teloscope_show_ps = False

    def _get_total_turns(self) -> int:
        """Get the total number of turns with fidelity data."""
        current_turn = st.session_state.get('beta_current_turn', 1)
        count = 0
        for turn_num in range(1, current_turn + 1):
            turn_data = st.session_state.get(f'beta_turn_{turn_num}_data', {})
            if turn_data:
                count += 1
        return count

    def _get_selected_turn(self, total_turns: int) -> int:
        """Get the currently selected turn for viewing."""
        selected = st.session_state.get('teloscope_selected_turn')
        if selected is None or selected > total_turns:
            return total_turns  # Live mode - show latest
        return max(1, selected)

    def _render_navigation_controls(self, selected_turn: int, total_turns: int, is_live_mode: bool):
        """Render navigation controls centered under the AI gauge."""
        if total_turns == 0:
            return  # No navigation needed when no turns

        # Add padding above navigation (separator from gauges)
        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

        # Turn badge color - same style as top-left header badge
        badge_color = STATUS_GOOD if is_live_mode else GOLD

        # Render centered navigation using CSS that forces centering
        st.markdown(f"""
<style>
/* Force the navigation container to be centered */
.teloscope-nav-wrapper {{
    display: flex;
    justify-content: center;
    align-items: center;
    width: 100%;
}}
/* Style for all nav columns container */
[data-testid="stHorizontalBlock"]:has(.teloscope-nav-btn) {{
    justify-content: center !important;
    gap: 8px !important;
}}
/* Style buttons in nav */
.teloscope-nav-btn button {{
    background-color: {BG_ELEVATED} !important;
    border: 1px solid #555 !important;
    color: {TEXT_PRIMARY} !important;
    border-radius: 6px !important;
    padding: 8px 12px !important;
    font-weight: 600 !important;
    min-width: 40px !important;
}}
.teloscope-nav-btn button:hover {{
    background-color: #3d3d3d !important;
    border-color: {GOLD} !important;
}}
.teloscope-nav-btn button:disabled {{
    opacity: 0.35 !important;
    border-color: #444 !important;
}}
</style>
""", unsafe_allow_html=True)

        # Navigation - use 7 columns with spacers on each side for centering
        spacer1, b1, b2, b3, b4, b5, spacer2 = st.columns([2, 1, 1, 1.5, 1, 1, 2])

        with b1:
            if st.button("⏮", key="ts_first", help="First turn", disabled=selected_turn <= 1, use_container_width=True):
                st.session_state.teloscope_selected_turn = 1
                st.rerun()

        with b2:
            if st.button("◀", key="ts_prev", help="Previous turn", disabled=selected_turn <= 1, use_container_width=True):
                st.session_state.teloscope_selected_turn = max(1, selected_turn - 1)
                st.rerun()

        with b3:
            # Turn badge - centered
            st.markdown(f"""
<div style="text-align: center;">
    <span style="
        background: {BG_ELEVATED};
        border: 1px solid {badge_color};
        border-radius: 4px;
        padding: 6px 12px;
        color: {badge_color};
        font-size: 13px;
        font-weight: 600;
        display: inline-block;
        white-space: nowrap;
    ">Turn {selected_turn}</span>
</div>
""", unsafe_allow_html=True)

        with b4:
            if st.button("▶", key="ts_next", help="Next turn", disabled=selected_turn >= total_turns, use_container_width=True):
                st.session_state.teloscope_selected_turn = min(total_turns, selected_turn + 1)
                st.rerun()

        with b5:
            if st.button("⏭", key="ts_last", help="Latest (Live)", disabled=is_live_mode, use_container_width=True):
                st.session_state.teloscope_selected_turn = None  # Back to live mode
                st.rerun()

        # Small spacer
        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    def render(self):
        """Render the TELOSCOPE panel."""
        # Get fidelity data
        fidelity_history = self._get_fidelity_history()
        total_turns = len(fidelity_history)
        selected_turn = self._get_selected_turn(total_turns)
        is_live_mode = st.session_state.get('teloscope_selected_turn') is None

        # Get data for SELECTED turn (not just current)
        if fidelity_history and selected_turn > 0:
            # Find the data for the selected turn
            selected_data = next(
                (d for d in fidelity_history if d['turn'] == selected_turn),
                fidelity_history[-1] if fidelity_history else None
            )
        else:
            selected_data = None

        current_data = selected_data if selected_data else {
            'user': None, 'ai': None, 'ps': None, 'turn': 0
        }

        # Determine overall status color from user fidelity
        user_fidelity = current_data.get('user')
        status_color = get_fidelity_color(user_fidelity) if user_fidelity is not None else STATUS_GOOD
        status_glow = _get_glow_color(status_color, 0.3)

        # Determine status text
        if user_fidelity is None or user_fidelity >= 0.70:
            status_text = "Monitoring"
            status_indicator = "green"
        elif user_fidelity >= 0.60:
            status_text = "Minor Drift"
            status_indicator = "yellow"
        elif user_fidelity >= 0.50:
            status_text = "Drift Detected"
            status_indicator = "orange"
        else:
            status_text = "Intervening"
            status_indicator = "red"

        # Mode indicator text
        mode_text = "LIVE" if is_live_mode else f"Turn {selected_turn}"
        mode_color = STATUS_GOOD if is_live_mode else GOLD

        # Inject CSS for pulse animation (once per page)
        st.markdown("""<style>@keyframes teloscope-pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }</style>""", unsafe_allow_html=True)

        # TELOSCOPE header (self-contained - no wrapper around native components)
        header_html = f"""<div style="background: linear-gradient(135deg, {BG_SURFACE} 0%, {BG_ELEVATED} 100%); border: 2px solid {status_color}; border-radius: 12px 12px 0 0; padding: 20px; box-shadow: 0 0 25px {status_glow};">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <h2 style="color: {GOLD}; margin: 0; font-size: 24px;">TELOSCOPE</h2>
        <div style="display: flex; align-items: center; gap: 12px;">
            <span style="background: {BG_ELEVATED}; border: 1px solid {mode_color}; border-radius: 4px; padding: 4px 10px; color: {mode_color}; font-size: 12px; font-weight: 600;">{mode_text}</span>
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="width: 10px; height: 10px; background: {status_color}; border-radius: 50%; box-shadow: 0 0 8px {status_color}; animation: teloscope-pulse 2s infinite;"></span>
                <span style="color: {status_color}; font-weight: 600;">{status_text}</span>
            </div>
        </div>
    </div>
</div>"""
        st.markdown(header_html, unsafe_allow_html=True)

        # Toggle tabs at TOP (USER/AI/PRIMACY) - styled to match gauge cards
        self._render_toggle_tabs()

        # Fidelity Trajectory Chart (with selected turn highlighted)
        self._render_trajectory_chart(fidelity_history, status_color, selected_turn)

        # Compact Gauge Row with turn context
        self._render_gauge_row(current_data, selected_turn, is_live_mode)

        # Navigation controls at BOTTOM (arrows + turn badge)
        self._render_navigation_controls(selected_turn, total_turns, is_live_mode)

        # Collapsible Attractor Dropdowns
        self._render_attractor_dropdowns()

        # Steward Chat Section (integrated)
        self._render_steward_section()

    def _get_fidelity_history(self) -> list:
        """Get fidelity history from all turns.

        Uses EXACT same extraction logic as beta_observation_deck._get_fidelity_data()
        to ensure calibration matches Alignment Lens.
        """
        history = []
        current_turn = st.session_state.get('beta_current_turn', 1)

        for turn_num in range(1, current_turn + 1):
            turn_data = st.session_state.get(f'beta_turn_{turn_num}_data', {})
            if not turn_data:
                continue

            telos_analysis = turn_data.get('telos_analysis', {})
            ps_metrics = turn_data.get('ps_metrics', {})
            beta_data = turn_data.get('beta_data', {})

            # === PRIMACY STATE - exact same logic as beta_observation_deck ===
            # Priority 1: Display-normalized PS (for UI consistency)
            primacy_state = telos_analysis.get('display_primacy_state')
            # Priority 2: Direct turn_data display_primacy_state
            if primacy_state is None:
                primacy_state = turn_data.get('display_primacy_state')
            # Priority 3: Raw primacy_state_score (fallback for non-normalized flows)
            if primacy_state is None:
                primacy_state = turn_data.get('primacy_state_score')
            # Also check ps_metrics dict if primacy_state_score not directly available
            if primacy_state is None and ps_metrics:
                primacy_state = ps_metrics.get('ps_score')
            # Check inside telos_analysis for primacy_state_score (raw fallback)
            if primacy_state is None:
                primacy_state = telos_analysis.get('primacy_state_score')

            # === USER FIDELITY - exact same logic as beta_observation_deck ===
            # Priority 1: Normalized display value (for UI)
            user_fidelity = telos_analysis.get('display_user_pa_fidelity')
            # Priority 2: Direct turn_data display_fidelity
            if user_fidelity is None:
                user_fidelity = turn_data.get('display_fidelity')
            # Priority 3: Raw user_pa_fidelity
            if user_fidelity is None:
                user_fidelity = turn_data.get('user_pa_fidelity')
            if user_fidelity is None and ps_metrics:
                user_fidelity = ps_metrics.get('f_user')
            # Priority 4: Check telos_analysis for user_pa_fidelity (raw)
            if user_fidelity is None:
                user_fidelity = telos_analysis.get('user_pa_fidelity')
            # Priority 5: Fallback beta_data
            if user_fidelity is None:
                user_fidelity = beta_data.get('user_fidelity') or beta_data.get('input_fidelity')

            # === AI FIDELITY - exact same logic as beta_observation_deck ===
            # Priority 1: Direct turn_data (state_manager path)
            ai_fidelity = turn_data.get('ai_pa_fidelity')
            # Priority 2: ps_metrics dict
            if ai_fidelity is None and ps_metrics:
                ai_fidelity = ps_metrics.get('f_ai')
            # Priority 3: Check telos_analysis for ai_pa_fidelity
            if ai_fidelity is None:
                ai_fidelity = telos_analysis.get('ai_pa_fidelity')
            # Priority 4: Legacy fallbacks
            if ai_fidelity is None:
                legacy_fidelity = beta_data.get('telos_fidelity') or beta_data.get('fidelity_score')
                if legacy_fidelity is not None and legacy_fidelity > 0:
                    ai_fidelity = legacy_fidelity
            if ai_fidelity is None:
                legacy_score = telos_analysis.get('fidelity_score')
                if legacy_score is not None and legacy_score > 0:
                    ai_fidelity = legacy_score
            # Priority 5: Last resort fallback
            if ai_fidelity is None:
                ai_fidelity = turn_data.get('fidelity')

            # Estimate AI fidelity if missing (same as beta_observation_deck)
            if ai_fidelity is None and user_fidelity is not None:
                ai_fidelity = min(1.0, user_fidelity + 0.15) if user_fidelity else None

            # Calculate PS if missing but have fidelities (same as beta_observation_deck)
            if primacy_state is None and user_fidelity is not None and ai_fidelity is not None:
                epsilon = 1e-10
                if user_fidelity + ai_fidelity > epsilon:
                    primacy_state = (2 * user_fidelity * ai_fidelity) / (user_fidelity + ai_fidelity + epsilon)

            if user_fidelity is not None:
                history.append({
                    'turn': turn_num,
                    'user': user_fidelity,
                    'ai': ai_fidelity,
                    'ps': primacy_state
                })

        return history

    def _render_toggle_tabs(self):
        """Render USER/AI/PRIMACY toggle tabs matching gauge card style."""
        # Spacing between TELOSCOPE header and toggle tabs
        st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)

        # Read toggle states
        show_user = st.session_state.get('teloscope_show_user', True)
        show_ai = st.session_state.get('teloscope_show_ai', False)
        show_ps = st.session_state.get('teloscope_show_ps', False)

        # Colors based on active state
        user_border = STATUS_GOOD if show_user else "#444"
        user_bg = "rgba(39, 174, 96, 0.15)" if show_user else BG_ELEVATED
        user_text = STATUS_GOOD if show_user else "#666"

        ai_border = "#3498db" if show_ai else "#444"
        ai_bg = "rgba(52, 152, 219, 0.15)" if show_ai else BG_ELEVATED
        ai_text = "#3498db" if show_ai else "#666"

        ps_border = GOLD if show_ps else "#444"
        ps_bg = "rgba(244, 208, 63, 0.15)" if show_ps else BG_ELEVATED
        ps_text = GOLD if show_ps else "#666"

        # CSS for toggle cards - matching gauge card dimensions exactly
        st.markdown(f"""
<style>
.teloscope-toggle-cards + div {{
    display: flex !important;
    gap: 10px !important;
    margin-top: 25px !important;
    padding-top: 15px !important;
}}
.teloscope-toggle-cards + div button {{
    flex: 1 !important;
    border-radius: 8px !important;
    padding: 12px 8px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    min-height: 50px !important;
}}
</style>
<div class="teloscope-toggle-cards" style="display:none;"></div>
""", unsafe_allow_html=True)

        # Toggle card buttons - same 3-column layout as gauges
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if st.button("● USER" if show_user else "○ USER", key="ts_toggle_user", help="Toggle User fidelity", use_container_width=True):
                st.session_state.teloscope_show_user = not show_user
                st.rerun()
        with col2:
            if st.button("■ AI" if show_ai else "□ AI", key="ts_toggle_ai", help="Toggle AI fidelity", use_container_width=True):
                st.session_state.teloscope_show_ai = not show_ai
                st.rerun()
        with col3:
            if st.button("◆ PRIMACY" if show_ps else "◇ PRIMACY", key="ts_toggle_ps", help="Toggle Primacy State", use_container_width=True):
                st.session_state.teloscope_show_ps = not show_ps
                st.rerun()

        # Spacer after tabs (generous spacing like gauge section)
        st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)

    def _render_trajectory_chart(self, history: list, border_color: str, selected_turn: int = None):
        """Render scatter plots for USER/AI/PRIMACY fidelity trajectories."""
        # Read toggle states
        show_user = st.session_state.get('teloscope_show_user', True)
        show_ai = st.session_state.get('teloscope_show_ai', False)
        show_ps = st.session_state.get('teloscope_show_ps', False)

        if not history:
            st.markdown(f"""
<div style="
    background: {BG_ELEVATED};
    border: 1px solid #444;
    border-radius: 8px;
    padding: 40px;
    text-align: center;
    margin-bottom: 20px;
">
    <span style="color: #666;">No fidelity data yet. Send a message to see the trajectory.</span>
</div>
""", unsafe_allow_html=True)
            return

        # Build SVG scatter plot - LARGER size for visibility
        chart_width = 500  # Fixed coordinate space
        chart_height = 200  # Much taller for better visibility
        padding_x = 25
        padding_y = 15

        # Calculate x positions in fixed coordinate space
        num_points = len(history)
        if num_points == 1:
            x_positions = [chart_width / 2]
        else:
            x_positions = [padding_x + (i * (chart_width - 2*padding_x) / (num_points - 1)) for i in range(num_points)]

        # Build scatter points and zone bands
        svg_elements = []

        # Calculate Y positions for zone boundaries (fidelity 0-1 maps to chart)
        def fidelity_to_y(f):
            return chart_height - (f * (chart_height - 2*padding_y)) - padding_y

        # Zone Y positions
        y_100 = fidelity_to_y(1.0)  # Top
        y_070 = fidelity_to_y(0.70)  # Green/Yellow boundary
        y_060 = fidelity_to_y(0.60)  # Yellow/Orange boundary
        y_050 = fidelity_to_y(0.50)  # Orange/Red boundary
        y_000 = fidelity_to_y(0.0)  # Bottom

        # Add colored zone bands (background)
        # Green zone (>= 0.70)
        svg_elements.append(f'<rect x="0" y="{y_100}" width="{chart_width}" height="{y_070 - y_100}" fill="rgba(39, 174, 96, 0.15)"/>')
        # Yellow zone (0.60 - 0.70)
        svg_elements.append(f'<rect x="0" y="{y_070}" width="{chart_width}" height="{y_060 - y_070}" fill="rgba(243, 156, 18, 0.12)"/>')
        # Orange zone (0.50 - 0.60)
        svg_elements.append(f'<rect x="0" y="{y_060}" width="{chart_width}" height="{y_050 - y_060}" fill="rgba(230, 126, 34, 0.12)"/>')
        # Red zone (< 0.50)
        svg_elements.append(f'<rect x="0" y="{y_050}" width="{chart_width}" height="{y_000 - y_050}" fill="rgba(231, 76, 60, 0.12)"/>')

        # Zone threshold lines
        svg_elements.append(f'<line x1="0" y1="{y_070}" x2="{chart_width}" y2="{y_070}" stroke="{STATUS_GOOD}" stroke-width="1" stroke-dasharray="6,3" opacity="0.6"/>')
        svg_elements.append(f'<line x1="0" y1="{y_060}" x2="{chart_width}" y2="{y_060}" stroke="{STATUS_MILD}" stroke-width="1" stroke-dasharray="6,3" opacity="0.5"/>')
        svg_elements.append(f'<line x1="0" y1="{y_050}" x2="{chart_width}" y2="{y_050}" stroke="{STATUS_MODERATE}" stroke-width="1" stroke-dasharray="6,3" opacity="0.5"/>')

        # Zone labels on right side
        svg_elements.append(f'<text x="{chart_width - 5}" y="{(y_100 + y_070) / 2 + 4}" fill="{STATUS_GOOD}" font-size="11" text-anchor="end" opacity="0.7">Aligned</text>')
        svg_elements.append(f'<text x="{chart_width - 5}" y="{(y_070 + y_060) / 2 + 4}" fill="{STATUS_MILD}" font-size="10" text-anchor="end" opacity="0.6">Minor</text>')
        svg_elements.append(f'<text x="{chart_width - 5}" y="{(y_060 + y_050) / 2 + 4}" fill="{STATUS_MODERATE}" font-size="10" text-anchor="end" opacity="0.6">Drift</text>')
        svg_elements.append(f'<text x="{chart_width - 5}" y="{(y_050 + y_000) / 2 + 4}" fill="{STATUS_SEVERE}" font-size="10" text-anchor="end" opacity="0.6">Critical</text>')

        # Draw Primacy State scatter (back layer) - gold circles
        if show_ps:
            for i, data in enumerate(history):
                if data['ps'] is not None:
                    x = x_positions[i]
                    y = fidelity_to_y(data['ps'])
                    is_selected = (selected_turn is not None and data['turn'] == selected_turn)
                    r = 8 if is_selected else 6
                    stroke = GOLD if is_selected else "#1a1a1a"
                    stroke_w = 3 if is_selected else 1
                    svg_elements.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{GOLD}" stroke="{stroke}" stroke-width="{stroke_w}" opacity="0.85"/>')

        # Draw AI scatter (middle layer) - blue circles
        if show_ai:
            for i, data in enumerate(history):
                if data['ai'] is not None:
                    x = x_positions[i]
                    y = fidelity_to_y(data['ai'])
                    is_selected = (selected_turn is not None and data['turn'] == selected_turn)
                    r = 7 if is_selected else 5
                    stroke = "#3498db" if is_selected else "#1a1a1a"
                    stroke_w = 3 if is_selected else 1
                    svg_elements.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="#3498db" stroke="{stroke}" stroke-width="{stroke_w}" opacity="0.85"/>')

        # Draw User scatter (front layer) - green circles
        if show_user:
            for i, data in enumerate(history):
                if data['user'] is not None:
                    x = x_positions[i]
                    y = fidelity_to_y(data['user'])
                    is_selected = (selected_turn is not None and data['turn'] == selected_turn)
                    r = 8 if is_selected else 6
                    stroke = STATUS_GOOD if is_selected else "#1a1a1a"
                    stroke_w = 3 if is_selected else 1
                    svg_elements.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{STATUS_GOOD}" stroke="{stroke}" stroke-width="{stroke_w}"/>')

        # Build SVG content from elements
        svg_content = chr(10).join(svg_elements)

        turn_labels = ' '.join([f'<span style="color: #888; font-size: 12px; font-weight: 500;">T{d["turn"]}</span>' for d in history])

        chart_html = f"""<div style="background: {BG_ELEVATED}; border: 1px solid #444; border-radius: 10px; padding: 25px 30px; margin-bottom: 30px;">
    <div style="color: {GOLD}; font-size: 15px; font-weight: 600; margin-bottom: 20px;">Fidelity Trajectory</div>
    <svg width="100%" height="{chart_height}" viewBox="0 0 {chart_width} {chart_height}" preserveAspectRatio="xMidYMid meet" style="overflow: visible;">{svg_content}</svg>
    <div style="display: flex; justify-content: space-between; margin-top: 18px; padding: 0 5%;">{turn_labels}</div>
    <div style="display: flex; gap: 18px; margin-top: 18px; justify-content: center; flex-wrap: wrap;">
        <span style="color: #aaa; font-size: 13px;"><span style="color: {STATUS_GOOD};">●</span> User</span>
        <span style="color: #aaa; font-size: 13px;"><span style="color: #3498db;">■</span> AI</span>
        <span style="color: #aaa; font-size: 13px;"><span style="color: {GOLD};">◆</span> Primacy</span>
        <span style="color: #666; font-size: 13px;">|</span>
        <span style="color: #aaa; font-size: 13px;"><span style="color: #27ae60;">---</span> Aligned</span>
        <span style="color: #aaa; font-size: 13px;"><span style="color: #FF5757;">---</span> Basin</span>
    </div>
</div>"""
        st.markdown(chart_html, unsafe_allow_html=True)

    def _render_gauge_row(self, current_data: dict, selected_turn: int = None, is_live_mode: bool = True):
        """Render compact gauge meters for User/AI/Primacy State with turn context."""
        user_val = current_data.get('user')
        ai_val = current_data.get('ai')
        ps_val = current_data.get('ps')

        # Get user input and AI response for the selected turn
        turn_input = None
        turn_response = None
        intervention_info = None
        if selected_turn:
            turn_data = st.session_state.get(f'beta_turn_{selected_turn}_data', {})
            turn_input = turn_data.get('user_input', '')

            # Check multiple locations for AI response
            telos_analysis = turn_data.get('telos_analysis', {})
            turn_response = (
                turn_data.get('response', '') or
                turn_data.get('ai_response', '') or
                telos_analysis.get('response', '') or
                turn_data.get('governed_response', '') or
                turn_data.get('native_response', '') or
                turn_data.get('shown_response', '')
            )

            # If still no response, try state_manager.state.turns
            if not turn_response:
                try:
                    state_manager = st.session_state.get('state_manager')
                    if state_manager and hasattr(state_manager, 'state') and hasattr(state_manager.state, 'turns'):
                        # Turns are 0-indexed, selected_turn is 1-indexed
                        turn_idx = selected_turn - 1
                        if 0 <= turn_idx < len(state_manager.state.turns):
                            state_turn = state_manager.state.turns[turn_idx]
                            turn_response = state_turn.get('response', '') or state_turn.get('shown_response', '')
                            if not turn_input:
                                turn_input = state_turn.get('user_input', '')
                except Exception:
                    pass

            # Get intervention info if any
            if telos_analysis.get('intervention_triggered'):
                intervention_info = telos_analysis.get('intervention_reason', 'Drift detected')

        # Get fidelity colors early for use in turn content display
        user_color = get_fidelity_color(user_val) if user_val is not None else "#666"
        ai_color = get_fidelity_color(ai_val) if ai_val is not None else "#666"

        # Show turn content when viewing historical turn
        if not is_live_mode and turn_input:
            # Truncate for display
            display_input = turn_input[:100] + "..." if len(turn_input) > 100 else turn_input
            display_response = turn_response[:150] + "..." if turn_response and len(turn_response) > 150 else turn_response

            st.markdown(f"""
<div style="
    background: rgba(244, 208, 63, 0.05);
    border: 1px solid {GOLD};
    border-radius: 8px;
    padding: 12px 15px;
    margin-bottom: 15px;
">
    <div style="margin-bottom: 10px;">
        <span style="color: {user_color}; font-size: 12px; font-weight: 600;">Turn {selected_turn} - User:</span>
        <div style="color: {user_color}; font-size: 13px; margin-top: 4px; line-height: 1.4;">"{html.escape(display_input)}"</div>
    </div>
    <div style="border-top: 1px solid rgba(244, 208, 63, 0.2); padding-top: 10px;">
        <span style="color: {ai_color}; font-size: 12px; font-weight: 600;">AI Response:</span>
        <div style="color: {ai_color}; font-size: 12px; margin-top: 4px; line-height: 1.4;">{html.escape(display_response) if display_response else '<em style="color: #666;">No response recorded</em>'}</div>
    </div>
    {f'<div style="margin-top: 8px; padding: 6px 10px; background: rgba(231, 76, 60, 0.15); border-radius: 4px; border-left: 3px solid #e74c3c;"><span style="color: #e74c3c; font-size: 11px; font-weight: 600;">⚠ Intervention:</span> <span style="color: #ccc; font-size: 11px;">{html.escape(intervention_info)}</span></div>' if intervention_info else ''}
</div>
""", unsafe_allow_html=True)

        # Get PS color (user/ai colors computed earlier for turn content display)
        ps_color = get_fidelity_color(ps_val) if ps_val is not None else "#666"

        # Format values
        user_display = f"{user_val:.2f}" if user_val is not None else "---"
        ai_display = f"{ai_val:.2f}" if ai_val is not None else "---"
        ps_display = f"{ps_val:.2f}" if ps_val is not None else "---"

        # Get zone names
        user_zone = get_zone_name(user_val) if user_val is not None else "---"
        ai_zone = get_zone_name(ai_val) if ai_val is not None else "---"
        ps_zone = get_zone_name(ps_val) if ps_val is not None else "---"

        # Build gauge HTML without comments (comments break Streamlit markdown rendering)
        gauge_style = "flex: 1; background: {bg}; border: 2px solid {color}; border-radius: 8px; padding: 12px; text-align: center; box-shadow: 0 0 12px {glow};"

        user_gauge = f"""<div style="{gauge_style.format(bg=BG_ELEVATED, color=user_color, glow=_get_glow_color(user_color, 0.3))}">
            <div style="color: {user_color}; font-size: 11px; font-weight: 600; margin-bottom: 4px;">USER</div>
            <div style="color: {user_color}; font-size: 28px; font-weight: 700; line-height: 1;">{user_display}</div>
            <div style="color: #888; font-size: 10px; margin-top: 4px;">{user_zone}</div>
        </div>"""

        ai_gauge = f"""<div style="{gauge_style.format(bg=BG_ELEVATED, color=ai_color, glow=_get_glow_color(ai_color, 0.3))}">
            <div style="color: {ai_color}; font-size: 11px; font-weight: 600; margin-bottom: 4px;">AI</div>
            <div style="color: {ai_color}; font-size: 28px; font-weight: 700; line-height: 1;">{ai_display}</div>
            <div style="color: #888; font-size: 10px; margin-top: 4px;">{ai_zone}</div>
        </div>"""

        ps_gauge = f"""<div style="{gauge_style.format(bg=BG_ELEVATED, color=ps_color, glow=_get_glow_color(ps_color, 0.3))}">
            <div style="color: {ps_color}; font-size: 11px; font-weight: 600; margin-bottom: 4px;">PRIMACY</div>
            <div style="color: {ps_color}; font-size: 28px; font-weight: 700; line-height: 1;">{ps_display}</div>
            <div style="color: #888; font-size: 10px; margin-top: 4px;">{ps_zone}</div>
        </div>"""

        gauge_html = f"""<div style="display: flex; gap: 15px; margin-bottom: 20px; padding: 0 5px;">{user_gauge}{ai_gauge}{ps_gauge}</div>"""
        st.markdown(gauge_html, unsafe_allow_html=True)

    def _render_attractor_dropdowns(self):
        """Render collapsible User and Steward Attractor dropdowns."""

        def safe_escape(value, default='Not set'):
            """Safely escape PA values."""
            if value is None:
                return default
            if isinstance(value, list):
                return ', '.join(html.escape(str(item)) for item in value)
            return html.escape(str(value))

        # Get User PA - check multiple sources for most current data
        # Priority: user_pa (shift focus updates) > primacy_attractor (initial setup)
        pa = st.session_state.get('user_pa') or st.session_state.get('primacy_attractor', {})
        user_purpose = safe_escape(pa.get('purpose'), 'Not set')
        user_scope = safe_escape(pa.get('scope'), 'Not set')
        user_boundaries = safe_escape(pa.get('boundaries'), 'Not set')

        # Get AI PA (Steward Attractor) - dynamically derived from user PA
        # This updates when user shifts focus
        derived_ai_pa = st.session_state.get('ai_pa', {})
        if derived_ai_pa:
            steward_purpose = safe_escape(derived_ai_pa.get('purpose'), 'Not set')
            steward_scope = safe_escape(derived_ai_pa.get('scope'), 'Not set')
            steward_boundaries = safe_escape(derived_ai_pa.get('boundaries'), 'Not set')
        else:
            # Fallback to default STEWARD_PA only if no derived PA exists
            steward_purpose = safe_escape(STEWARD_PA.get('purpose'), 'Not set')
            steward_scope = safe_escape(STEWARD_PA.get('scope'), 'Not set')
            steward_boundaries = safe_escape(STEWARD_PA.get('boundaries'), 'Not set')

        # Attractor dropdown buttons - use Streamlit expanders
        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

        # User Attractor expander
        with st.expander("User Attractor", expanded=False):
            st.markdown(f"""
<div style="padding: 8px;">
    <div style="color: {STATUS_GOOD}; font-weight: 600; font-size: 11px; margin-bottom: 4px;">Purpose</div>
    <div style="color: {TEXT_PRIMARY}; font-size: 12px; margin-bottom: 12px;">{user_purpose}</div>
    <div style="color: {STATUS_GOOD}; font-weight: 600; font-size: 11px; margin-bottom: 4px;">Scope</div>
    <div style="color: {TEXT_PRIMARY}; font-size: 12px; margin-bottom: 12px;">{user_scope}</div>
    <div style="color: {STATUS_GOOD}; font-weight: 600; font-size: 11px; margin-bottom: 4px;">Boundaries</div>
    <div style="color: {TEXT_PRIMARY}; font-size: 12px;">{user_boundaries}</div>
</div>
""", unsafe_allow_html=True)

        # Steward Attractor expander
        with st.expander("Steward Attractor", expanded=False):
            st.markdown(f"""
<div style="padding: 8px;">
    <div style="color: {STATUS_GOOD}; font-weight: 600; font-size: 11px; margin-bottom: 4px;">Purpose</div>
    <div style="color: {TEXT_PRIMARY}; font-size: 12px; margin-bottom: 12px;">{steward_purpose}</div>
    <div style="color: {STATUS_GOOD}; font-weight: 600; font-size: 11px; margin-bottom: 4px;">Scope</div>
    <div style="color: {TEXT_PRIMARY}; font-size: 12px; margin-bottom: 12px;">{steward_scope}</div>
    <div style="color: {STATUS_GOOD}; font-weight: 600; font-size: 11px; margin-bottom: 4px;">Boundaries</div>
    <div style="color: {TEXT_PRIMARY}; font-size: 12px;">{steward_boundaries}</div>
</div>
""", unsafe_allow_html=True)

    def _render_steward_section(self):
        """Render integrated Steward chat section."""
        # Initialize chat history for TELOSCOPE if not exists
        if 'teloscope_steward_history' not in st.session_state:
            st.session_state.teloscope_steward_history = [
                {
                    'role': 'assistant',
                    'content': "I'm Steward. Ask me about what you're seeing in the TELOSCOPE.",
                    'timestamp': datetime.now().isoformat()
                }
            ]

        # Steward section with TELOS styling
        st.markdown(f"""
<div style="
    background: linear-gradient(135deg, rgba(244, 208, 63, 0.08) 0%, rgba(244, 208, 63, 0.02) 100%);
    border: 2px solid {GOLD};
    border-radius: 12px;
    padding: 20px;
    margin-top: 20px;
    box-shadow: 0 0 20px {_get_glow_color(GOLD, 0.15)};
">
    <div style="color: {GOLD}; font-size: 16px; font-weight: 600; margin-bottom: 15px;">
        Steward
    </div>
""", unsafe_allow_html=True)

        # Chat history display
        for message in st.session_state.teloscope_steward_history:
            if message['role'] == 'assistant':
                formatted = _md_to_html(message['content'])
                st.markdown(f"""
<div style="
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.08) 0%, rgba(255, 255, 255, 0.03) 50%, transparent 100%), rgba(26, 26, 30, 0.45);
    backdrop-filter: blur(10px);
    border: 1px solid {GOLD};
    border-radius: 8px;
    padding: 12px;
    margin-bottom: 10px;
">
    <span style="color: {TEXT_PRIMARY}; font-size: 14px; line-height: 1.5;">{formatted}</span>
</div>
""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
<div style="
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.08) 0%, rgba(255, 255, 255, 0.03) 50%, transparent 100%), rgba(26, 26, 30, 0.45);
    backdrop-filter: blur(10px);
    border: 1px solid #666;
    border-radius: 8px;
    padding: 12px;
    margin-bottom: 10px;
">
    <strong style="color: #888; font-size: 12px;">You:</strong><br>
    <span style="color: {TEXT_PRIMARY}; font-size: 14px; line-height: 1.5;">{html.escape(message['content'])}</span>
</div>
""", unsafe_allow_html=True)

        # Contemplating animation
        if st.session_state.get('teloscope_steward_pending', False):
            st.markdown(f"""
<div style="
    background: rgba(26, 26, 30, 0.45);
    border: 1px solid #666;
    border-radius: 8px;
    padding: 12px;
    margin-bottom: 10px;
">
    <span style="color: {GOLD}; font-style: italic;">Contemplating...</span>
</div>
""", unsafe_allow_html=True)

        # Close the Steward section div
        st.markdown("</div>", unsafe_allow_html=True)

        # Input form (outside the HTML div for Streamlit form handling)
        st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)

        with st.form(key="teloscope_steward_form", clear_on_submit=True):
            user_input = st.text_area(
                "Ask Steward...",
                placeholder="Ask Steward a question...",
                label_visibility="collapsed",
                key="teloscope_steward_input",
                height=80
            )
            submitted = st.form_submit_button("Send", use_container_width=True)

        if submitted and user_input.strip():
            st.session_state.teloscope_steward_history.append({
                'role': 'user',
                'content': user_input.strip(),
                'timestamp': datetime.now().isoformat()
            })
            st.session_state.teloscope_steward_pending = True
            st.session_state.teloscope_steward_input_pending = user_input.strip()
            st.rerun()

        # Add JavaScript to submit form on Enter key (Shift+Enter for new line)
        import streamlit.components.v1 as components
        components.html("""
        <script>
        (function() {
            // Find the Send button within the same form as the textarea
            function findSendButton(textarea) {
                let container = textarea.closest('form') || textarea.closest('[data-testid]') || textarea.parentElement;
                for (let i = 0; i < 10 && container; i++) {
                    const btn = container.querySelector('button');
                    if (btn) {
                        const text = btn.textContent || btn.innerText;
                        if (text.includes('Send')) {
                            return btn;
                        }
                    }
                    container = container.parentElement;
                }
                return null;
            }

            function setupEnterHandler() {
                const doc = window.parent.document;
                const textareas = doc.querySelectorAll('textarea');
                textareas.forEach(function(textarea) {
                    if (textarea.dataset.teloscopeEnterSetup) return;
                    textarea.dataset.teloscopeEnterSetup = 'true';

                    textarea.addEventListener('keydown', function(e) {
                        if (e.key === 'Enter' && !e.shiftKey) {
                            // Prevent submission if input is empty
                            if (!textarea.value || !textarea.value.trim()) {
                                e.preventDefault();
                                return;
                            }
                            e.preventDefault();
                            e.stopPropagation();

                            // Find the Send button in the same form as this textarea
                            const btn = findSendButton(textarea);
                            if (btn && !btn.disabled) {
                                btn.click();
                                return;
                            }
                        }
                    }, true);
                });
            }

            // Run immediately and also after delays to catch dynamic content
            setupEnterHandler();
            setTimeout(setupEnterHandler, 500);
            setTimeout(setupEnterHandler, 1000);
        })();
        </script>
        """, height=0)

        # Fetch response when pending
        if st.session_state.get('teloscope_steward_pending', False):
            user_input = st.session_state.get('teloscope_steward_input_pending', '')

            if st.session_state.get('beta_steward_llm_enabled', False):
                context = self._gather_teloscope_context()
                try:
                    response = st.session_state.beta_steward_llm.get_response(
                        user_message=user_input,
                        conversation_history=st.session_state.teloscope_steward_history[:-1],
                        context=context
                    )
                except Exception as e:
                    response = f"Error: {str(e)}"
            else:
                response = "Steward is not available. Please check MISTRAL_API_KEY configuration."

            st.session_state.teloscope_steward_history.append({
                'role': 'assistant',
                'content': response,
                'timestamp': datetime.now().isoformat()
            })
            st.session_state.teloscope_steward_pending = False
            st.session_state.teloscope_steward_input_pending = None
            st.rerun()

    def _gather_teloscope_context(self) -> dict:
        """Gather context for Steward in TELOSCOPE mode."""
        context = {
            'mode': 'teloscope',
            'viewing': 'fidelity_trajectory'
        }

        # Get PA
        pa = st.session_state.get('primacy_attractor', {})
        if pa:
            context['primacy_attractor'] = pa

        # Get latest turn data
        current_turn = st.session_state.get('beta_current_turn', 1)
        for turn_num in range(current_turn, 0, -1):
            turn_data = st.session_state.get(f'beta_turn_{turn_num}_data', {})
            if turn_data:
                telos_analysis = turn_data.get('telos_analysis', {})
                context['f_user'] = telos_analysis.get('display_user_pa_fidelity') or telos_analysis.get('user_pa_fidelity')
                context['f_ai'] = telos_analysis.get('ai_pa_fidelity')
                context['primacy_state'] = telos_analysis.get('display_primacy_state') or telos_analysis.get('primacy_state_score')
                context['last_user_input'] = turn_data.get('user_input', '')
                context['intervention_triggered'] = telos_analysis.get('intervention_triggered', False)
                break

        return context


def render_teloscope_button():
    """Render the 'Open TELOSCOPE' button for Alignment Lens expansion."""
    # Initialize state
    if 'teloscope_open' not in st.session_state:
        st.session_state.teloscope_open = False

    # Get user fidelity for button color
    current_turn = st.session_state.get('beta_current_turn', 1)
    user_fidelity = None
    for turn_num in range(current_turn, 0, -1):
        turn_data = st.session_state.get(f'beta_turn_{turn_num}_data', {})
        if turn_data:
            telos_analysis = turn_data.get('telos_analysis', {})
            user_fidelity = telos_analysis.get('display_user_pa_fidelity') or telos_analysis.get('user_pa_fidelity')
            break

    btn_color = get_fidelity_color(user_fidelity) if user_fidelity is not None else GOLD

    # CSS for button styling
    st.markdown(f"""
<style>
.teloscope-btn-marker + div button {{
    background-color: {BG_ELEVATED} !important;
    border: 2px solid {btn_color} !important;
    color: {TEXT_PRIMARY} !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    transition: all 0.2s ease !important;
}}
.teloscope-btn-marker + div button:hover {{
    background-color: #3d3d3d !important;
    box-shadow: 0 0 12px {btn_color} !important;
}}
</style>
<div class="teloscope-btn-marker" style="display:none;"></div>
""", unsafe_allow_html=True)

    label = "Close TELOSCOPE" if st.session_state.teloscope_open else "Open TELOSCOPE"

    if st.button(label, key="teloscope_toggle", use_container_width=True):
        st.session_state.teloscope_open = not st.session_state.teloscope_open
        # Mutual exclusion with Alignment Lens
        if st.session_state.teloscope_open:
            st.session_state.beta_deck_visible = False
        st.rerun()


def render_teloscope_panel():
    """Render the TELOSCOPE panel if open."""
    if st.session_state.get('teloscope_open', False):
        # Auto-scroll to TELOSCOPE when opened (uses iframe for JS execution)
        components.html("""
<script>
(function() {
    const findAndScroll = () => {
        const parent = window.parent.document;
        const headers = parent.querySelectorAll('h2');
        for (const h of headers) {
            if (h.textContent.includes('TELOSCOPE')) {
                h.scrollIntoView({ behavior: 'smooth', block: 'start' });
                return true;
            }
        }
        return false;
    };
    // Try with delays for Streamlit rendering
    setTimeout(findAndScroll, 50);
    setTimeout(findAndScroll, 200);
})();
</script>
""", height=0)

        panel = TeloscopePanel()
        panel.render()
