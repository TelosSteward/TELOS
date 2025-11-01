"""
Observatory V2 - Observation Deck (Right Sidebar)
==================================================

Right-side panel showing detailed turn analysis and tools.

Sections:
- Math Breakdown: Fidelity calculations and metrics
- Counterfactual Analysis: Phase 2/2B comparison
- Steward Tool: AI assistant for turn analysis (toggleable)

Design: Fixed right panel that slides in when control strip is clicked.
"""

import streamlit as st
from typing import Dict, Any


class ObservationDeck:
    """
    Renders the Observation Deck right sidebar panel.

    Pure component - reads state, emits actions through state manager.
    """

    def __init__(self, state_manager):
        """
        Initialize Observation Deck.

        Args:
            state_manager: StateManager instance
        """
        self.state_manager = state_manager

    def render(self):
        """Render Observation Deck if expanded."""
        if not self.state_manager.is_deck_expanded():
            return  # Deck is hidden

        # Get current turn data
        turn_data = self.state_manager.get_current_turn_data()
        if not turn_data:
            return

        session_info = self.state_manager.get_session_info()

        # Render deck panel
        self._render_styles()
        self._render_deck_panel(turn_data, session_info)

    def _render_styles(self):
        """Render CSS styles for Observation Deck."""
        st.markdown("""
            <style>
            .observation-deck {
                position: fixed;
                right: 0;
                top: 0;
                width: 380px;
                height: 100vh;
                background: rgba(15, 20, 30, 0.98);
                backdrop-filter: blur(15px);
                border-left: 2px solid rgba(255, 215, 0, 0.4);
                padding: 1.5rem;
                overflow-y: auto;
                z-index: 999;
                box-shadow: -6px 0 30px rgba(0, 0, 0, 0.6);
                animation: slideInRight 0.3s ease-out;
            }

            @keyframes slideInRight {
                from {
                    transform: translateX(100%);
                    opacity: 0;
                }
                to {
                    transform: translateX(0);
                    opacity: 1;
                }
            }

            .deck-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 1.5rem;
                padding-bottom: 1rem;
                border-bottom: 2px solid rgba(255, 215, 0, 0.3);
            }

            .deck-title {
                font-size: 1.3rem;
                font-weight: bold;
                color: #FFD700;
            }

            .deck-section {
                margin: 1.5rem 0;
                padding: 1rem;
                background: rgba(255, 255, 255, 0.03);
                border-radius: 8px;
                border: 1px solid rgba(255, 255, 255, 0.1);
            }

            .section-title {
                font-size: 1rem;
                font-weight: 600;
                color: #FFD700;
                margin-bottom: 0.75rem;
            }

            .metric-row {
                display: flex;
                justify-content: space-between;
                padding: 0.5rem 0;
                border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            }

            .metric-label {
                color: #aaa;
                font-size: 0.85rem;
            }

            .metric-value {
                color: #fff;
                font-weight: 600;
                font-size: 0.85rem;
            }

            .gold-text {
                color: #FFD700;
            }
            </style>
        """, unsafe_allow_html=True)

    def _render_deck_panel(self, turn_data: Dict[str, Any], session_info: Dict[str, Any]):
        """
        Render the deck panel content.

        Args:
            turn_data: Current turn data
            session_info: Session metadata
        """
        # Container for deck
        st.markdown('<div class="observation-deck">', unsafe_allow_html=True)

        # Header with close button
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown('<div class="deck-title">🔭 Observation Deck</div>', unsafe_allow_html=True)
        with col2:
            if st.button("✕", key="deck_close", help="Close Deck"):
                self.state_manager.toggle_deck()
                st.rerun()

        st.markdown('<div style="margin: 0.5rem 0; color: #888; font-size: 0.8rem;">Turn {}/{}</div>'.format(
            session_info['current_turn'] + 1,
            session_info['total_turns']
        ), unsafe_allow_html=True)

        # Section 1: Math Breakdown
        with st.expander("📊 Mathematical Breakdown", expanded=self.state_manager.state.show_math_breakdown):
            self._render_math_breakdown(turn_data)

        # Section 2: Counterfactual Analysis
        with st.expander("🌿 Counterfactual Analysis", expanded=self.state_manager.state.show_counterfactual):
            self._render_counterfactual(turn_data)

        # Section 3: Steward Tool (toggleable)
        st.markdown("---")
        if st.button("🤝 Steward Tool", key="steward_toggle", use_container_width=True):
            self.state_manager.toggle_component('steward')
            st.rerun()

        if self.state_manager.state.show_steward:
            self._render_steward_tool(turn_data)

        st.markdown('</div>', unsafe_allow_html=True)

    def _render_math_breakdown(self, turn_data: Dict[str, Any]):
        """
        Render mathematical breakdown section.

        Args:
            turn_data: Current turn data
        """
        fidelity = turn_data.get('fidelity', 1.0)
        distance = turn_data.get('distance', 0.0)
        threshold = turn_data.get('threshold', 0.8)

        # Fidelity with color coding
        if fidelity >= 0.8:
            fidelity_color = "#90EE90"  # Light green
        elif fidelity >= 0.6:
            fidelity_color = "#FFD700"  # Gold
        else:
            fidelity_color = "#FF6B6B"  # Light red

        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-label">Fidelity Score</div>
            <div class="metric-value" style="color: {fidelity_color};">{fidelity:.3f}</div>
        </div>
        <div class="metric-row">
            <div class="metric-label">Distance</div>
            <div class="metric-value">{distance:.3f}</div>
        </div>
        <div class="metric-row">
            <div class="metric-label">Threshold</div>
            <div class="metric-value">{threshold:.3f}</div>
        </div>
        """, unsafe_allow_html=True)

        # Fidelity formula explanation
        st.caption("**Fidelity = 1 - (distance / max_distance)**")
        st.caption("Measures alignment with purpose attractor")

    def _render_counterfactual(self, turn_data: Dict[str, Any]):
        """
        Render counterfactual analysis section.

        Args:
            turn_data: Current turn data
        """
        # Check if intervention occurred
        intervention = turn_data.get('intervention_applied', False)

        if intervention:
            st.success("⚡ Governance intervention applied")

            # Show Phase 2 comparison if available
            phase2_data = turn_data.get('phase2_comparison', {})
            if phase2_data:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Original Branch", f"{phase2_data.get('original_fidelity', 0):.2f}")
                with col2:
                    st.metric("TELOS Branch", f"{phase2_data.get('telos_fidelity', 0):.2f}")
            else:
                st.caption("Phase 2 comparison data not available")
        else:
            st.info("No intervention on this turn")

        # Drift detection
        if turn_data.get('drift_detected', False):
            st.warning("⚠️ Drift detected but below intervention threshold")

    def _render_steward_tool(self, turn_data: Dict[str, Any]):
        """
        Render Steward AI tool for turn analysis.

        Args:
            turn_data: Current turn data
        """
        st.markdown("### 🤝 Steward Analysis")
        st.caption("AI assistant for analyzing this turn")

        # Query input
        query = st.text_input(
            "Ask about this turn:",
            placeholder="e.g., Why did fidelity drop?",
            key="steward_query"
        )

        if st.button("Analyze", key="steward_analyze"):
            if query:
                # TODO: Wire to actual Steward AI
                st.info(f"Steward: Analyzing turn {turn_data.get('turn', 0)}...")
                st.caption("*(Steward integration pending)*")
            else:
                st.warning("Please enter a question")

        # Show turn context for Steward
        with st.expander("Turn Context"):
            st.json({
                'turn': turn_data.get('turn', 0),
                'user_input': turn_data.get('user_input', '')[:100] + '...',
                'fidelity': turn_data.get('fidelity', 0),
                'intervention': turn_data.get('intervention_applied', False)
            })
