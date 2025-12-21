"""
Symbolic Flow - Governance Pipeline Visualizer

Visual representation of TELOS governance pipeline showing data flow:
👤 (User Input) → ⚡ (Primacy Attractor) → 🔄 (Fidelity Check) → 🤖 (LLM) → ✓ (Output)

Animation States:
- Quiet: Static symbols, low opacity (no activity)
- Pulse: Gentle pulse (governance active, no drift)
- Strong Pulse: Prominent pulse (drift detected, intervention triggered)

Purpose:
- At-a-glance understanding of governance pipeline
- Early warning system for drift
- Educational visualization of how TELOS works

Used in: Observation Deck Control Strip
"""

import streamlit as st
from typing import Dict, Any


class SymbolicFlow:
    """
    Renders animated symbolic flow visualization of governance pipeline.

    Shows the data flow through TELOS governance with dynamic animations
    based on current governance state.
    """

    # Symbolic flow elements
    FLOW_SYMBOLS = ['👤', '⚡', '🔄', '🤖', '✓']
    FLOW_LABELS = ['Input', 'Attractor', 'Fidelity', 'LLM', 'Output']

    def __init__(self, session_manager):
        """
        Initialize Symbolic Flow visualizer.

        Args:
            session_manager: WebSessionManager instance
        """
        self.session_manager = session_manager

    def render(self, governance_state: Dict[str, Any]):
        """
        Render symbolic flow visualization.

        Args:
            governance_state: Current governance state with telemetry
        """
        # Determine animation state based on governance activity
        animation_state = self._get_animation_state(governance_state)

        # Build CSS for animations
        css = self._build_animation_css(animation_state)
        st.markdown(css, unsafe_allow_html=True)

        # Build HTML for symbolic flow
        html = self._build_flow_html(animation_state)
        st.markdown(html, unsafe_allow_html=True)

    def _get_animation_state(self, governance_state: Dict[str, Any]) -> str:
        """
        Determine animation state based on governance activity.

        Args:
            governance_state: Current governance state

        Returns:
            Animation state: 'quiet', 'pulse', 'strong_pulse'
        """
        drift_detected = governance_state.get('drift_detected', False)
        governance_active = governance_state.get('governance_enabled', False)

        if drift_detected:
            return 'strong_pulse'
        elif governance_active:
            return 'pulse'
        else:
            return 'quiet'

    def _build_animation_css(self, animation_state: str) -> str:
        """
        Build CSS for symbolic flow animations.

        Args:
            animation_state: Current animation state

        Returns:
            CSS string with animation keyframes
        """
        # TODO: Implement animation CSS
        # Will include:
        # - @keyframes for pulse animations
        # - Different intensities for quiet/pulse/strong_pulse
        # - Color transitions based on state
        css = """
            <style>
            .symbolic-flow {
                display: flex;
                align-items: center;
                justify-content: space-around;
                padding: 10px;
                gap: 5px;
            }
            .flow-element {
                display: flex;
                flex-direction: column;
                align-items: center;
                gap: 3px;
            }
            .flow-symbol {
                font-size: 20px;
            }
            .flow-label {
                font-size: 9px;
                color: rgba(255, 255, 255, 0.5);
                text-transform: uppercase;
            }
            .flow-arrow {
                color: rgba(255, 255, 255, 0.3);
                font-size: 14px;
            }
            </style>
        """
        return css

    def _build_flow_html(self, animation_state: str) -> str:
        """
        Build HTML for symbolic flow visualization.

        Args:
            animation_state: Current animation state

        Returns:
            HTML string for flow visualization
        """
        elements_html = []

        for i, (symbol, label) in enumerate(zip(self.FLOW_SYMBOLS, self.FLOW_LABELS)):
            # Flow element
            element_html = f"""
                <div class="flow-element">
                    <div class="flow-symbol">{symbol}</div>
                    <div class="flow-label">{label}</div>
                </div>
            """
            elements_html.append(element_html)

            # Arrow between elements (except after last)
            if i < len(self.FLOW_SYMBOLS) - 1:
                elements_html.append('<div class="flow-arrow">→</div>')

        html = f"""
            <div class="symbolic-flow">
                {''.join(elements_html)}
            </div>
        """

        return html
