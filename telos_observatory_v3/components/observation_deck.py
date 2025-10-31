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
            st.info("No data to display")
            return

        # Deck header
        st.markdown("""
        <div style="
            text-align: center;
            padding: 15px;
            background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
            border-bottom: 2px solid #FFD700;
            margin-bottom: 15px;
        ">
            <h2 style="color: #FFD700; margin: 0;">🔭 Observation Deck</h2>
            <p style="color: #888; font-size: 12px; margin: 5px 0 0 0;">
                Real-time governance metrics
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Fidelity Window
        self._render_metric_window(
            title="Alignment Fidelity",
            value=f"{turn_data.get('fidelity', 0.0):.3f}",
            icon="📊",
            description="Measure of response alignment with user's deeper preferences"
        )

        # Distance Window
        self._render_metric_window(
            title="Semantic Distance",
            value=f"{turn_data.get('distance', 0.0):.3f}",
            icon="📏",
            description="Distance from ideal aligned response"
        )

        # Status Window
        status = turn_data.get('status_text', 'Nominal')
        status_color = "#4CAF50" if status == "Nominal" else "#FFA500"
        self._render_metric_window(
            title="System Status",
            value=status,
            icon="⚡",
            description="Current operational status",
            value_color=status_color
        )

        # Intervention Window
        intervention = "ACTIVE" if turn_data.get('intervention_applied', False) else "STANDBY"
        intervention_color = "#FFD700" if intervention == "ACTIVE" else "#4CAF50"
        self._render_metric_window(
            title="Intervention Status",
            value=intervention,
            icon="🛡️",
            description="Whether TELOS modified this response",
            value_color=intervention_color
        )

        st.markdown("---")

        # Deep Dive Button
        if st.button("🔬 Deep Dive Analysis", use_container_width=True, key="deep_dive_btn"):
            st.info("Deep Dive feature - would show detailed Phase 2B comparison")

        st.markdown("---")

        # Navigation Controls
        self._render_navigation()

        st.markdown("---")

        # Component Toggles
        self._render_toggles()

    def _render_metric_window(self, title, value, icon, description, value_color="#FFD700"):
        """Render a gold-themed metric window.

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
            border: 2px solid #FFD700;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 15px;
            box-shadow: 0 0 15px rgba(255, 215, 0, 0.2);
        ">
            <div style="display: flex; align-items: center; margin-bottom: 8px;">
                <span style="font-size: 24px; margin-right: 10px;">{icon}</span>
                <span style="color: #FFD700; font-size: 14px; font-weight: bold;">
                    {title}
                </span>
            </div>
            <div style="
                font-size: 32px;
                font-weight: bold;
                color: {value_color};
                margin: 10px 0;
                text-align: center;
            ">
                {value}
            </div>
            <div style="color: #888; font-size: 11px; text-align: center;">
                {description}
            </div>
        </div>
        """, unsafe_allow_html=True)

    def _render_navigation(self):
        """Render navigation controls."""
        st.markdown("### 🎮 Navigation")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("◀ Prev", use_container_width=True, key="nav_prev"):
                if self.state_manager.previous_turn():
                    st.rerun()

        with col2:
            session_info = self.state_manager.get_session_info()
            turn_num = session_info.get('current_turn', 0) + 1
            total = session_info.get('total_turns', 0)
            st.markdown(f"""
            <div style="
                text-align: center;
                padding: 8px;
                background-color: #2d2d2d;
                border-radius: 5px;
            ">
                <span style="color: #FFD700; font-size: 14px;">
                    {turn_num} / {total}
                </span>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            if st.button("Next ▶", use_container_width=True, key="nav_next"):
                if self.state_manager.next_turn():
                    st.rerun()

    def _render_toggles(self):
        """Render component visibility toggles."""
        st.markdown("### ⚙️ View Options")

        if st.checkbox(
            "Show Math Breakdown",
            value=self.state_manager.state.show_math_breakdown,
            key="toggle_math"
        ):
            self.state_manager.toggle_component('math_breakdown')
            st.rerun()

        if st.checkbox(
            "Show Counterfactual Analysis",
            value=self.state_manager.state.show_counterfactual,
            key="toggle_cf"
        ):
            self.state_manager.toggle_component('counterfactual')
            st.rerun()

        if st.checkbox(
            "Show Steward Details",
            value=self.state_manager.state.show_steward,
            key="toggle_steward"
        ):
            self.state_manager.toggle_component('steward')
            st.rerun()
