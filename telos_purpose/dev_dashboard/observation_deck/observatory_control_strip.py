"""
Observatory Control Strip - Top-Right Thermometer

Global governance status indicator that stays visible at all times.
Provides ambient awareness without being intrusive.

Components:
1. Turn Counter: Current turn / Total turns
2. Fidelity Gauge: Color-coded fidelity score (green/white/amber/red)
3. Calibration Progress: Only visible during Turns 1-3
4. Pulse Animations: Subtle alerts for drift detection

Design Philosophy:
- "Out of the way" - minimal visual footprint
- Progressive disclosure: thermometer → summary → deep analysis
- Quiet by default, pulses only when attention needed
"""

import streamlit as st
from typing import Dict, Any, Optional


class ObservatoryControlStrip:
    """
    Renders the top-right Observatory Control Strip.

    This is the "global thermometer" showing real-time governance status.
    Always visible regardless of sidebar or deck state.
    """

    # Color thresholds for fidelity gauge
    FIDELITY_COLORS = {
        'high': (0.8, 1.0, '#00ff00'),  # Green: Strongly aligned
        'moderate': (0.6, 0.8, '#ffffff'),  # White: Acceptable
        'low': (0.4, 0.6, '#ffaa00'),  # Amber: Warning
        'critical': (0.0, 0.4, '#ff0000'),  # Red: Drift detected
    }

    def __init__(self, session_manager):
        """
        Initialize Observatory Control Strip.

        Args:
            session_manager: WebSessionManager instance for accessing turn data
        """
        self.session_manager = session_manager

    def render(self, container=None):
        """
        Render the Observatory Control Strip in top-right position.

        Args:
            container: Streamlit container to render in (defaults to st.container)
        """
        if container is None:
            container = st.container()

        with container:
            # Use custom CSS for minimal symbol-based strip
            st.markdown("""
                <style>
                .observatory-control-strip {
                    position: fixed;
                    top: 120px;
                    right: 20px;
                    z-index: 1000;
                    background: rgba(30, 30, 30, 0.8);
                    border: 1px solid rgba(255, 255, 255, 0.15);
                    border-radius: 4px;
                    padding: 4px 8px;
                    display: flex;
                    gap: 8px;
                    align-items: center;
                    font-size: 11px;
                    font-family: monospace;
                }
                .obs-indicator {
                    display: flex;
                    align-items: center;
                    gap: 3px;
                    color: rgba(255, 255, 255, 0.7);
                }
                .obs-dot {
                    width: 6px;
                    height: 6px;
                    border-radius: 50%;
                }
                .pulse-dot {
                    animation: pulse 2s infinite;
                }
                @keyframes pulse {
                    0%, 100% { opacity: 1; }
                    50% { opacity: 0.3; }
                }
                </style>
            """, unsafe_allow_html=True)

            # Get current session data
            turn_data = self._get_current_turn_data()

            # Build HTML for control strip
            strip_html = self._build_control_strip_html(turn_data)

            st.markdown(strip_html, unsafe_allow_html=True)

    def _get_current_turn_data(self) -> Dict[str, Any]:
        """
        Get current turn data from session manager.

        Returns:
            Dictionary with turn number, fidelity, calibration status, drift flag
        """
        try:
            session_data = self.session_manager.get_session_data()
            current_turn = session_data.get('current_turn', 0)
            total_turns = len(session_data.get('turns', []))

            # Get latest turn telemetry
            if total_turns > 0:
                latest_turn = session_data['turns'][-1]
                fidelity = latest_turn.get('fidelity_score', None)
                calibration_phase = latest_turn.get('metadata', {}).get('calibration_phase', False)
                drift_detected = latest_turn.get('governance_drift_flag', False)
            else:
                fidelity = None
                calibration_phase = False
                drift_detected = False

            return {
                'current_turn': current_turn,
                'total_turns': total_turns,
                'fidelity': fidelity,
                'calibration_phase': calibration_phase,
                'drift_detected': drift_detected,
            }
        except Exception as e:
            # Fallback if session manager not available
            return {
                'current_turn': 0,
                'total_turns': 0,
                'fidelity': None,
                'calibration_phase': False,
                'drift_detected': False,
            }

    def _build_control_strip_html(self, turn_data: Dict[str, Any]) -> str:
        """
        Build minimal symbol-based HTML for Observatory Control Strip.

        Args:
            turn_data: Current turn data dictionary

        Returns:
            Compact HTML string
        """
        total_turns = turn_data['total_turns']
        fidelity = turn_data['fidelity']
        drift_detected = turn_data['drift_detected']

        # Turn counter: just the number
        turn_display = f"T{total_turns}" if total_turns > 0 else "T0"

        # Fidelity: colored dot + percentage
        if fidelity is None:
            fidelity_color = "#666"
            fidelity_display = "—"
            pulse = ""
        else:
            fidelity_color = self._get_fidelity_color(fidelity)
            fidelity_display = f"{int(fidelity * 100)}"
            pulse = "pulse-dot" if drift_detected else ""

        # Minimal HTML: T3 ● 95
        html = f"""
            <div class="observatory-control-strip">
                <span class="obs-indicator">{turn_display}</span>
                <span class="obs-indicator">
                    <span class="obs-dot {pulse}" style="background: {fidelity_color};"></span>
                    {fidelity_display}
                </span>
            </div>
        """

        return html

    def _get_fidelity_color(self, fidelity: float) -> str:
        """
        Get color for fidelity score based on thresholds.

        Args:
            fidelity: Fidelity score (0.0-1.0)

        Returns:
            Hex color string
        """
        for level, (min_val, max_val, color) in self.FIDELITY_COLORS.items():
            if min_val <= fidelity < max_val:
                return color

        # Default to critical if outside ranges
        return self.FIDELITY_COLORS['critical'][2]
