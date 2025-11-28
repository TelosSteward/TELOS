"""
Comparison Viewer - TELOS vs Baseline Split View

Side-by-side comparison of TELOS-governed response and baseline (native) response.
Highlights governance interventions with color-coded diffs.

Components:
1. Split View: Left (Baseline) | Right (TELOS)
2. Intervention Highlights: Color-coded sections showing where governance modified output
3. Turn Sync: Updates when turn marker changes

Data Source: CounterfactualBranchManager (dual branches)

Purpose:
- Visual proof of governance impact
- Identify exactly what TELOS changed
- Research tool for analyzing intervention quality
"""

import streamlit as st
from typing import Dict, Any, Optional


class ComparisonViewer:
    """
    Renders side-by-side comparison of TELOS vs Baseline responses.

    Highlights governance interventions and syncs to current turn marker.
    """

    def __init__(self, session_manager):
        """
        Initialize Comparison Viewer.

        Args:
            session_manager: WebSessionManager instance
        """
        self.session_manager = session_manager

    def render(self, turn_number: Optional[int] = None):
        """
        Render comparison view for specified turn.

        Args:
            turn_number: Turn number to display (defaults to current turn)
        """
        st.markdown("### 🔀 Comparison Viewer")
        st.markdown("*TELOS vs Baseline Response*")

        # Get turn data
        turn_data = self._get_turn_data(turn_number)

        if not turn_data:
            st.info("No comparison data available for this turn.")
            return

        # Render split view
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Baseline (Native)**")
            self._render_response(turn_data.get('baseline_response', 'N/A'))

        with col2:
            st.markdown("**TELOS (Governed)**")
            self._render_response(turn_data.get('telos_response', 'N/A'),
                                highlight_interventions=True,
                                interventions=turn_data.get('interventions', []))

        # Show intervention summary
        self._render_intervention_summary(turn_data)

    def _get_turn_data(self, turn_number: Optional[int]) -> Optional[Dict[str, Any]]:
        """
        Get turn data from CounterfactualBranchManager.

        Args:
            turn_number: Turn number to retrieve

        Returns:
            Turn data with baseline and TELOS responses
        """
        # TODO: Wire to CounterfactualBranchManager
        # Will extract:
        # - Baseline response (native LLM)
        # - TELOS response (governed)
        # - Intervention details (what was changed)
        return None

    def _render_response(self, response_text: str, highlight_interventions: bool = False,
                        interventions: list = None):
        """
        Render response text with optional intervention highlights.

        Args:
            response_text: Response text to display
            highlight_interventions: Whether to highlight interventions
            interventions: List of intervention details
        """
        # TODO: Implement response rendering with diff highlights
        st.markdown(response_text)

    def _render_intervention_summary(self, turn_data: Dict[str, Any]):
        """
        Render summary of interventions applied.

        Args:
            turn_data: Turn data with intervention details
        """
        # TODO: Implement intervention summary
        # Shows: Number of interventions, types, reasons
        pass
