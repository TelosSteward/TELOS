"""
Observatory V2 - Control Strip Component
=========================================

Top-right control strip showing turn-by-turn metrics.

Features:
- Turn counter (current / total)
- Fidelity score with gold theming
- Status indicators
- Clickable TELOSCOPE icon to toggle Observation Deck
- Fixed position with hover effects
"""

import streamlit as st
from typing import Dict, Any


class ControlStrip:
    """
    Renders the Observatory control strip.

    Pure component - takes state as input, emits actions through state manager.
    """

    def __init__(self, state_manager):
        """
        Initialize control strip.

        Args:
            state_manager: StateManager instance for reading/updating state
        """
        self.state_manager = state_manager

    def render(self):
        """Render the control strip at top-right of page."""
        # Get current state
        session_info = self.state_manager.get_session_info()
        current_turn_data = self.state_manager.get_current_turn_data()
        deck_expanded = self.state_manager.is_deck_expanded()

        if not current_turn_data:
            return  # No data to display

        # Extract metrics
        fidelity = current_turn_data.get('fidelity', 1.0)
        fidelity_display = f"{fidelity:.2f}" if fidelity is not None else "Cal"

        # Determine status
        status_icon, status_text = self._get_status(fidelity)

        # Render CSS and HTML
        self._render_styles()
        self._render_strip_html(
            current_turn=session_info['current_turn'] + 1,  # Display as 1-based
            total_turns=session_info['total_turns'],
            fidelity=fidelity_display,
            status_icon=status_icon,
            status_text=status_text,
            is_active=deck_expanded
        )

        # Render clickable button
        self._render_toggle_button()

    def _get_status(self, fidelity: float) -> tuple:
        """
        Determine status icon and text based on fidelity.

        Args:
            fidelity: Current fidelity score

        Returns:
            Tuple of (icon, text)
        """
        if fidelity is None:
            return "⚙️", "Calibrating"
        elif fidelity < 0.6:
            return "⚠️", "Drift"
        elif fidelity < 0.8:
            return "⚡", "Watch"
        else:
            return "✓", "Stable"

    def _render_styles(self):
        """Render CSS styles for control strip."""
        st.markdown("""
            <style>
            .control-strip {
                position: fixed;
                top: 60px;
                right: 20px;
                background: rgba(0, 0, 0, 0.85);
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
                background: rgba(20, 30, 40, 0.95);
                border: 1px solid rgba(255, 215, 0, 0.3);
                transform: translateY(-2px);
                box-shadow: 0 6px 12px rgba(0, 0, 0, 0.4);
            }

            .control-strip.active {
                border: 1px solid rgba(255, 215, 0, 0.6);
                background: rgba(30, 40, 50, 0.95);
                box-shadow: 0 0 20px rgba(255, 215, 0, 0.2);
            }

            .control-strip-hint {
                font-size: 0.65rem;
                color: #888;
                text-align: center;
                margin-top: 0.25rem;
            }

            .gold-text {
                color: #FFD700;
            }
            </style>
        """, unsafe_allow_html=True)

    def _render_strip_html(self, current_turn: int, total_turns: int,
                           fidelity: str, status_icon: str, status_text: str,
                           is_active: bool):
        """
        Render the control strip HTML.

        Args:
            current_turn: Current turn number (1-based)
            total_turns: Total number of turns
            fidelity: Fidelity display string
            status_icon: Status emoji
            status_text: Status text
            is_active: Whether Observation Deck is active
        """
        active_class = "active" if is_active else ""

        html = f"""
        <div class="control-strip {active_class}">
            <div style="display: flex; align-items: center; gap: 1rem;">
                <div>
                    <div style="font-size: 0.7rem; color: #888; margin-bottom: 0.25rem; text-transform: uppercase; letter-spacing: 0.5px;">Turn</div>
                    <div style="font-size: 1.25rem; font-weight: bold; color: #FFF;">{current_turn} / {total_turns}</div>
                </div>
                <div style="border-left: 1px solid rgba(255,255,255,0.2); height: 40px;"></div>
                <div>
                    <div style="font-size: 0.7rem; color: #888; margin-bottom: 0.25rem; text-transform: uppercase; letter-spacing: 0.5px;">Fidelity</div>
                    <div style="font-size: 1.25rem; font-weight: bold;" class="gold-text">{fidelity}</div>
                </div>
                <div style="border-left: 1px solid rgba(255,255,255,0.2); height: 40px;"></div>
                <div>
                    <div style="font-size: 0.7rem; color: #888; margin-bottom: 0.25rem; text-transform: uppercase; letter-spacing: 0.5px;">Status</div>
                    <div style="font-size: 1rem; color: #FFF;">{status_icon} {status_text}</div>
                </div>
                <div style="border-left: 1px solid rgba(255,255,255,0.2); height: 40px;"></div>
                <div style="text-align: center;">
                    <div style="font-size: 1.3rem;" class="gold-text">🔭</div>
                    <div class="control-strip-hint">Observatory</div>
                </div>
            </div>
        </div>
        """

        st.markdown(html, unsafe_allow_html=True)

    def _render_toggle_button(self):
        """Render clickable button to toggle Observation Deck."""
        # Add spacing to position button properly
        st.markdown("<div style='margin-top: 110px;'></div>", unsafe_allow_html=True)

        # Button positioned at top-right
        col1, col2 = st.columns([4, 1])
        with col2:
            if st.button("🔭 Observatory", key="control_strip_toggle",
                        help="Toggle Observation Deck", use_container_width=True):
                self.state_manager.toggle_deck()
                st.rerun()
