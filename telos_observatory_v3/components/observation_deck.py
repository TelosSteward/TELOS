"""
Observation Deck Component for TELOS Observatory V3.
Right panel with gold-themed statistics windows that slides in when Strip is clicked.
"""

import streamlit as st


class ObservationDeck:
    """Gold-themed observation deck with statistics windows."""

    def __init__(self, state_manager):
        """Initialize with state manager reference.

        Args:
            state_manager: StateManager instance for turn data
        """
        self.state_manager = state_manager

    def render(self):
        """Render the observation deck panel."""
        turn_data = self.state_manager.get_current_turn_data()

        if not turn_data:
            # No turns yet - don't show anything (clean state)
            return

        # Check if deck is expanded
        is_expanded = self.state_manager.is_deck_expanded()

        if not is_expanded:
            # Collapsed state - thin gold bar
            if st.button("Observation Deck - Click to expand", key="deck_toggle_collapsed", use_container_width=True):
                self.state_manager.toggle_deck()
                st.rerun()
            return

        # Expanded state - show all content
        if st.button("▼ Collapse Observation Deck", key="deck_toggle_expanded", use_container_width=True):
            self.state_manager.toggle_deck()
            st.rerun()

        st.markdown("---")

        # Metrics readout section
        self._render_metrics(turn_data)

        st.markdown("---")

        # View Options Toggles
        self._render_view_options()

    def _render_metric_window(self, title, value, icon, description, value_color="#FFD700"):
        """Render a compact gold-themed metric window.

        Args:
            title: Window title
            value: Metric value to display
            icon: Icon emoji
            description: Description text
            value_color: Color for the value text
        """
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
            border: 1px solid #FFD700;
            border-radius: 8px;
            padding: 6px 8px;
            margin-bottom: 6px;
            box-shadow: 0 0 6px rgba(255, 215, 0, 0.15);
        ">
            <div style="display: flex; align-items: center; margin-bottom: 3px;">
                <span style="font-size: 16px; margin-right: 6px;">{icon}</span>
                <span style="color: #FFD700; font-size: 10px; font-weight: bold;">
                    {title}
                </span>
            </div>
            <div style="
                font-size: 20px;
                font-weight: bold;
                color: {value_color};
                margin: 3px 0;
                text-align: center;
            ">
                {value}
            </div>
            <div style="color: #b0b0b0; font-size: 8px; text-align: center;">
                {description}
            </div>
        </div>
        """, unsafe_allow_html=True)

    def _render_metrics(self, turn_data):
        """Render metrics readout section."""
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            self._render_metric_card(
                "Alignment Fidelity",
                f"{turn_data.get('fidelity', 0.0):.3f}",
                "📊",
                "Measure of response alignment"
            )

        with col2:
            self._render_metric_card(
                "Semantic Distance",
                f"{turn_data.get('distance', 0.0):.3f}",
                "📏",
                "Distance from ideal response"
            )

        with col3:
            status = turn_data.get('status_text', 'Nominal')
            status_color = "#4CAF50" if status == "Nominal" else "#FFA500"
            self._render_metric_card(
                "System Status",
                status,
                "⚡",
                "Current operational status",
                value_color=status_color
            )

        with col4:
            intervention = "ACTIVE" if turn_data.get('intervention_applied', False) else "STANDBY"
            intervention_color = "#FFD700" if intervention == "ACTIVE" else "#4CAF50"
            self._render_metric_card(
                "Intervention Status",
                intervention,
                "🛡️",
                "TELOS intervention state",
                value_color=intervention_color
            )

    def _render_metric_card(self, title, value, icon, description, value_color="#FFD700"):
        """Render a single metric card."""
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
            border: 1px solid #FFD700;
            border-radius: 8px;
            padding: 10px;
            text-align: center;
        ">
            <div style="font-size: 20px; margin-bottom: 5px;">{icon}</div>
            <div style="color: #FFD700; font-size: 10px; font-weight: bold; margin-bottom: 5px;">
                {title}
            </div>
            <div style="
                font-size: 18px;
                font-weight: bold;
                color: {value_color};
                margin: 5px 0;
            ">
                {value}
            </div>
            <div style="color: #888; font-size: 8px;">
                {description}
            </div>
        </div>
        """, unsafe_allow_html=True)

    def _render_view_options(self):
        """Render view options toggles."""
        st.markdown("### ⚙️ View Options")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            # Primacy Attractor toggle (renamed from Steward Details)
            pa_label = "✕ Close Primacy Attractor" if self.state_manager.state.show_primacy_attractor else "🎯 Primacy Attractor"
            if st.button(
                pa_label,
                key="deck_toggle_pa_btn",
                use_container_width=True
            ):
                self.state_manager.toggle_component('primacy_attractor')
                st.rerun()

        with col2:
            math_label = "✕ Close Math Breakdown" if self.state_manager.state.show_math_breakdown else "🔢 Math Breakdown"
            if st.button(
                math_label,
                key="deck_toggle_math_btn",
                use_container_width=True
            ):
                self.state_manager.toggle_component('math_breakdown')
                st.rerun()

        with col3:
            cf_label = "✕ Close Counterfactual" if self.state_manager.state.show_counterfactual else "🔀 Counterfactual"
            if st.button(
                cf_label,
                key="deck_toggle_cf_btn",
                use_container_width=True
            ):
                self.state_manager.toggle_component('counterfactual')
                st.rerun()

        with col4:
            # Renamed Deep Dive to Observatory
            if st.button("🔭 Observatory", key="deck_observatory_btn", use_container_width=True):
                st.info("Observatory feature - detailed Phase 2 analysis and metrics")
