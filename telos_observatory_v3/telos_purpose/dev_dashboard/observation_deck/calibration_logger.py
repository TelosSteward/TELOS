"""
Calibration Logger - Mistral Reasoning Visualization (Turns 1-3)

During the first 3 turns, TELOS is calibrating the Primacy Attractor based on user inputs.
This logger displays Mistral's reasoning process as it forms the governance profile.

Components:
1. Mistral Reasoning Logs: Show AI's interpretation of canonical inputs
2. Embedding Evolution: Visualize how primacy attractor embeddings evolve
3. Attractor Formation: Track Purpose, Scope, Boundaries as they emerge

Purpose:
- Educational: Help users understand how TELOS works
- Transparency: Show exactly what governance profile is being created
- Research: Provide data for analyzing calibration quality

Only displayed during Turns 1-3. After calibration, this becomes historical data.
"""

import streamlit as st
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class CalibrationLogger:
    """
    Displays Mistral's reasoning during calibration phase (Turns 1-3).

    Provides full transparency into how the Primacy Attractor is formed.
    """

    def __init__(self, session_manager):
        """
        Initialize Calibration Logger.

        Args:
            session_manager: WebSessionManager instance
        """
        self.session_manager = session_manager

    def render(self):
        """
        Render calibration logger with Mistral reasoning.

        Only shows data for Turns 1-3 (calibration phase).
        """
        st.markdown("### Calibration Logger")

        # Get calibration data from session
        calibration_data = self._get_calibration_data()

        if not calibration_data:
            st.info("No calibration data available. Calibration occurs during PA establishment.")
            return

        # Render calibration progress
        self._render_calibration_progress(calibration_data)

        # Render Mistral reasoning logs
        self._render_mistral_reasoning(calibration_data)

        # Render embedding evolution
        self._render_embedding_evolution(calibration_data)

    def _get_calibration_data(self) -> List[Dict[str, Any]]:
        """
        Get calibration data from session state or session manager.

        Extracts:
        - Canonical inputs per turn
        - Reasoning about Purpose/Scope/Boundaries
        - Embedding evolution

        Returns:
            List of calibration turn data
        """
        calibration_data = []

        # Try to get PA data from session state
        pa_data = st.session_state.get('pa_data', {})
        if pa_data:
            calibration_data.append({
                'turn': 0,
                'type': 'pa_establishment',
                'purpose': pa_data.get('purpose', ''),
                'scope': pa_data.get('scope', ''),
                'template': pa_data.get('template', 'custom'),
                'tau': pa_data.get('tau', 0.5),
                'reasoning': pa_data.get('reasoning', 'PA established from user input'),
            })

        # Try to get from session manager state
        if self.session_manager and hasattr(self.session_manager, 'state'):
            state = self.session_manager.state

            # Get turns that are part of calibration (typically first 3)
            turns = getattr(state, 'turns', [])
            for i, turn in enumerate(turns[:3]):
                if isinstance(turn, dict):
                    turn_data = {
                        'turn': i + 1,
                        'type': 'calibration_turn',
                        'user_input': turn.get('user_input', ''),
                        'fidelity': turn.get('user_fidelity', 0),
                        'display_fidelity': turn.get('display_fidelity', 0),
                        'in_basin': turn.get('layer2_in_basin', True),
                        'reasoning': self._extract_turn_reasoning(turn),
                    }
                    calibration_data.append(turn_data)

        # Try to get from governance trace collector
        try:
            from telos_purpose.core.governance_trace_collector import get_trace_collector
            session_id = st.session_state.get('session_id', 'default')
            collector = get_trace_collector(session_id=session_id)

            # Get early fidelity events
            from telos_purpose.core.evidence_schema import EventType
            fidelity_events = collector.get_events(event_type=EventType.FIDELITY_CALCULATED)

            for event in fidelity_events[:3]:
                if hasattr(event, 'turn_number') and event.turn_number <= 3:
                    existing = next(
                        (d for d in calibration_data if d.get('turn') == event.turn_number),
                        None
                    )
                    if existing:
                        existing['raw_similarity'] = getattr(event, 'raw_similarity', 0)
                        existing['normalized_fidelity'] = getattr(event, 'normalized_fidelity', 0)
                        existing['layer1_hard_block'] = getattr(event, 'layer1_hard_block', False)
                        existing['layer2_outside_basin'] = getattr(event, 'layer2_outside_basin', False)

        except Exception as e:
            logger.debug(f"Could not get trace collector data: {e}")

        return calibration_data

    def _extract_turn_reasoning(self, turn: Dict[str, Any]) -> str:
        """
        Extract reasoning information from a turn.

        Args:
            turn: Turn data dictionary

        Returns:
            Reasoning string
        """
        parts = []

        # Fidelity level
        fidelity_level = turn.get('fidelity_level', 'unknown')
        if fidelity_level:
            parts.append(f"Fidelity classification: {fidelity_level}")

        # Intervention info
        if turn.get('intervention_triggered'):
            reason = turn.get('intervention_reason', 'drift detected')
            parts.append(f"Intervention: {reason}")
        else:
            parts.append("Aligned with purpose (no intervention)")

        # Steward style if available
        steward_style = turn.get('steward_style', {})
        if steward_style:
            band = steward_style.get('band_name', '')
            if band:
                parts.append(f"Steward style: {band}")

        return "; ".join(parts) if parts else "Processing turn..."

    def _render_calibration_progress(self, calibration_data: List[Dict[str, Any]]):
        """
        Render calibration progress indicator.

        Shows: 1/3, 2/3, 3/3 with status indicators

        Args:
            calibration_data: List of calibration turns
        """
        st.markdown("#### Calibration Progress")

        # Count actual calibration turns (not PA establishment)
        calibration_turns = [d for d in calibration_data if d.get('type') == 'calibration_turn']
        num_turns = len(calibration_turns)
        max_turns = 3

        # Progress bar
        progress = min(num_turns / max_turns, 1.0)
        st.progress(progress)

        # Status indicators
        cols = st.columns(3)

        for i, col in enumerate(cols):
            turn_num = i + 1
            turn_data = next(
                (d for d in calibration_turns if d.get('turn') == turn_num),
                None
            )

            with col:
                if turn_data:
                    # Turn completed
                    fidelity = turn_data.get('display_fidelity', turn_data.get('fidelity', 0))
                    in_basin = turn_data.get('in_basin', True)

                    color = "#27ae60" if in_basin else "#e74c3c"
                    status = "Aligned" if in_basin else "Drift"

                    st.markdown(f"""
                    <div style="
                        background: rgba(39, 174, 96, 0.1);
                        border: 1px solid {color};
                        border-radius: 8px;
                        padding: 10px;
                        text-align: center;
                    ">
                        <div style="color: {color}; font-size: 24px;">Turn {turn_num}</div>
                        <div style="color: #888; font-size: 12px;">{status}</div>
                        <div style="color: #F4D03F; font-size: 14px;">{fidelity:.3f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    # Turn pending
                    st.markdown(f"""
                    <div style="
                        background: rgba(136, 136, 136, 0.1);
                        border: 1px solid #444;
                        border-radius: 8px;
                        padding: 10px;
                        text-align: center;
                    ">
                        <div style="color: #888; font-size: 24px;">Turn {turn_num}</div>
                        <div style="color: #666; font-size: 12px;">Pending</div>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("---")

    def _render_mistral_reasoning(self, calibration_data: List[Dict[str, Any]]):
        """
        Render reasoning logs for each calibration turn.

        Shows interpretation of canonical inputs with expandable sections.

        Args:
            calibration_data: List of calibration turns
        """
        st.markdown("#### Reasoning Log")

        if not calibration_data:
            st.info("No reasoning data available yet.")
            return

        for turn_data in calibration_data:
            turn_type = turn_data.get('type', 'unknown')
            turn_num = turn_data.get('turn', 0)

            if turn_type == 'pa_establishment':
                # PA establishment reasoning
                with st.expander("PA Establishment", expanded=True):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("**Purpose:**")
                        st.markdown(f"> {turn_data.get('purpose', 'Not specified')[:200]}")

                    with col2:
                        st.markdown("**Scope:**")
                        st.markdown(f"> {turn_data.get('scope', 'Not specified')[:200]}")

                    st.markdown(f"**Template:** {turn_data.get('template', 'custom')}")
                    st.markdown(f"**Tau (mixture):** {turn_data.get('tau', 0.5):.2f}")

            elif turn_type == 'calibration_turn':
                # Calibration turn reasoning
                fidelity = turn_data.get('display_fidelity', turn_data.get('fidelity', 0))
                color = self._get_fidelity_color(fidelity)

                with st.expander(f"Turn {turn_num} - Fidelity: {fidelity:.3f}", expanded=False):
                    # User input (truncated for privacy)
                    user_input = turn_data.get('user_input', '')
                    if user_input:
                        truncated = user_input[:100] + "..." if len(user_input) > 100 else user_input
                        st.markdown(f"**Input:** {truncated}")

                    # Reasoning
                    st.markdown(f"**Analysis:** {turn_data.get('reasoning', 'Processing...')}")

                    # Technical details
                    if turn_data.get('raw_similarity'):
                        st.markdown(f"""
                        **Technical Details:**
                        - Raw similarity: {turn_data.get('raw_similarity', 0):.4f}
                        - Normalized fidelity: {turn_data.get('normalized_fidelity', 0):.4f}
                        - Layer 1 (hard block): {'Yes' if turn_data.get('layer1_hard_block') else 'No'}
                        - Layer 2 (outside basin): {'Yes' if turn_data.get('layer2_outside_basin') else 'No'}
                        """)

    def _render_embedding_evolution(self, calibration_data: List[Dict[str, Any]]):
        """
        Visualize how primacy attractor embeddings evolve.

        Shows line chart of fidelity scores over calibration turns.

        Args:
            calibration_data: List of calibration turns
        """
        st.markdown("#### Fidelity Evolution")

        # Filter to calibration turns with fidelity data
        turns_with_fidelity = [
            d for d in calibration_data
            if d.get('type') == 'calibration_turn' and (
                d.get('fidelity') or d.get('display_fidelity') or d.get('normalized_fidelity')
            )
        ]

        if not turns_with_fidelity:
            st.info("No fidelity evolution data yet. Complete calibration turns to see embedding evolution.")
            return

        # Try to use Plotly for visualization
        try:
            import plotly.graph_objects as go

            turns = [d.get('turn', 0) for d in turns_with_fidelity]
            fidelities = [
                d.get('display_fidelity') or d.get('normalized_fidelity') or d.get('fidelity', 0)
                for d in turns_with_fidelity
            ]
            raw_sims = [d.get('raw_similarity', 0) for d in turns_with_fidelity]

            fig = go.Figure()

            # Normalized fidelity trace
            fig.add_trace(go.Scatter(
                x=turns,
                y=fidelities,
                mode='lines+markers',
                name='Normalized Fidelity',
                line=dict(color='#F4D03F', width=2),
                marker=dict(size=10),
            ))

            # Raw similarity trace (if available)
            if any(raw_sims):
                fig.add_trace(go.Scatter(
                    x=turns,
                    y=raw_sims,
                    mode='lines+markers',
                    name='Raw Similarity',
                    line=dict(color='#3498db', width=2, dash='dot'),
                    marker=dict(size=8),
                ))

            # Threshold lines
            fig.add_hline(y=0.70, line_dash="dash", line_color="#27ae60", opacity=0.5)
            fig.add_hline(y=0.50, line_dash="dash", line_color="#e67e22", opacity=0.5)

            fig.update_layout(
                xaxis_title="Calibration Turn",
                yaxis_title="Score",
                yaxis=dict(range=[0, 1]),
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(13,17,23,0.95)',
                font=dict(color='#e6edf3'),
                height=250,
                margin=dict(t=20, b=40, l=50, r=20),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )

            st.plotly_chart(fig, use_container_width=True)

        except ImportError:
            # Fallback to text display
            st.markdown("**Fidelity Scores:**")
            for d in turns_with_fidelity:
                turn = d.get('turn', '?')
                fidelity = d.get('display_fidelity') or d.get('fidelity', 0)
                st.markdown(f"- Turn {turn}: {fidelity:.3f}")

    @staticmethod
    def _get_fidelity_color(fidelity: float) -> str:
        """Get color for fidelity value."""
        if fidelity >= 0.70:
            return "#27ae60"
        elif fidelity >= 0.60:
            return "#f39c12"
        elif fidelity >= 0.50:
            return "#e67e22"
        else:
            return "#e74c3c"
