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
        # Don't render anything if we're in demo mode on the observation deck slide (index 4)
        # (that slide has its own embedded observation deck)
        mode = st.session_state.get('active_tab', 'DEMO')
        demo_slide_index = st.session_state.get('demo_slide_index', 0)
        if mode == 'DEMO' and demo_slide_index == 4:
            return

        turn_data = self.state_manager.get_current_turn_data()

        # Always show expanded content when rendered
        # (visibility is controlled by the button in main.py)

        st.markdown("---")

        # Show demo PA if no turn data yet (for demo slides)
        if not turn_data:
            self._render_demo_pa()
        else:
            # Metrics readout section
            self._render_metrics(turn_data)

        # Only show View Options in BETA mode or higher (not in DEMO)
        mode = st.session_state.get('active_tab', 'DEMO')
        if mode != 'DEMO':
            st.markdown("---")
            # View Options Toggles
            self._render_view_options()

        # Add navigation and Hide button at bottom of expanded Observation Deck
        st.markdown("---")

        # ALWAYS show 3-button navigation layout (Previous/Hide/Next)
        # Mode detection happens in parent, we just render the controls
        demo_slide_index = st.session_state.get('demo_slide_index', 0)

        # Render navigation row without background
        st.markdown("<div style='margin: 20px 0; padding: 10px;'>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)

        with col1:
            # Always show button, enable/disable based on slide position
            is_disabled = (demo_slide_index <= 0)
            if st.button(
                "⬅️ Previous",
                key="obs_deck_prev",
                use_container_width=True,
                disabled=is_disabled,
                help="Go to previous slide" if not is_disabled else "Already at first slide",
                type="secondary"
            ):
                if demo_slide_index > 0:
                    st.session_state.demo_slide_index = max(0, demo_slide_index - 1)
                    st.rerun()

        with col2:
            if st.button(
                "Hide Observation Deck",
                key="hide_obs_deck_bottom",
                use_container_width=True,
                help="Close the Observation Deck",
                type="primary"
            ):
                st.session_state.show_observation_deck = False
                st.rerun()

        with col3:
            max_slide = 15  # 0=welcome, 1=intro, 2=PA setup, 3-14=Q&A (12 slides), 15=completion
            is_disabled = (demo_slide_index >= max_slide)
            if st.button(
                "Next ➡️",
                key="obs_deck_next",
                use_container_width=True,
                disabled=is_disabled,
                help="Go to next slide" if not is_disabled else "Already at last slide",
                type="secondary"
            ):
                if demo_slide_index < max_slide:
                    st.session_state.demo_slide_index = min(max_slide, demo_slide_index + 1)
                    st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    def _render_demo_pa(self):
        """Render demo Primacy Attractors (both User PA and AI PA) side-by-side with fidelity metrics."""
        # Centered container matching chat window style
        st.markdown("""
        <style>
        .pa-main-container {
            max-width: 700px;
            margin: 0 auto;
            padding: 0 10px;
        }
        </style>
        """, unsafe_allow_html=True)

        # Wrap everything in centered container
        st.markdown('<div class="pa-main-container">', unsafe_allow_html=True)

        # Main container with gold border
        st.markdown("""
        <div style="
            background-color: #1a1a1a;
            border: 3px solid #FFD700;
            border-radius: 10px;
            padding: 20px;
            margin: 20px auto;
            box-shadow: 0 0 20px rgba(255, 215, 0, 0.3);
        ">
        """, unsafe_allow_html=True)

        # PA Established indicator
        st.markdown("""
        <div style="text-align: center; margin-bottom: 15px;">
            <span style="background-color: #2d2d2d; border: 1px solid #4CAF50; border-radius: 20px; padding: 5px 15px; color: #4CAF50; font-weight: bold; font-size: 14px;">
                ✓ Dual PAs Established - Primacy Basin Achieved
            </span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div style="color: #FFD700; text-align: center; font-size: 22px; font-weight: bold; margin-bottom: 20px;">Dual Primacy Attractors</div>', unsafe_allow_html=True)

        # Fidelity Metrics using table layout
        st.markdown("""
        <table style="width: 100%; margin-bottom: 15px;">
        <tr>
            <td style="width: 33%; text-align: center;">
                <div style="background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%); border: 1px solid #4CAF50; border-radius: 8px; padding: 10px; display: inline-block;">
                    <div style="color: #4CAF50; font-size: 12px; margin-bottom: 5px;">👤 User Fidelity</div>
                    <div style="color: #4CAF50; font-size: 24px; font-weight: bold;">1.000</div>
                </div>
            </td>
            <td style="width: 34%; text-align: center; padding: 0 10px;">
                <div style="background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%); border: 2px solid #FFD700; border-radius: 8px; padding: 10px 20px; display: inline-block;">
                    <div style="color: #FFD700; font-size: 12px; margin-bottom: 5px;">🎯 Primacy State</div>
                    <div style="color: #FFD700; font-size: 24px; font-weight: bold;">1.000</div>
                    <div style="color: #888; font-size: 10px; margin-top: 5px;">Perfect Equilibrium</div>
                </div>
            </td>
            <td style="width: 33%; text-align: center;">
                <div style="background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%); border: 1px solid #FFD700; border-radius: 8px; padding: 10px; display: inline-block;">
                    <div style="color: #FFD700; font-size: 12px; margin-bottom: 5px;">🤖 AI Fidelity</div>
                    <div style="color: #FFD700; font-size: 24px; font-weight: bold;">1.000</div>
                </div>
            </td>
        </tr>
        </table>
        """, unsafe_allow_html=True)

        # Two-column layout for PAs using table
        st.markdown("""
        <table style="width: 100%; border-spacing: 15px;">
        <tr>
            <td style="width: 50%; vertical-align: top; padding: 0;">
                <!-- LEFT COLUMN: USER PA -->
                <div style="background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%); color: #1a1a1a; padding: 8px; border-radius: 8px 8px 0 0; font-weight: bold; text-align: center; font-size: 16px;">
                    👤 User Primacy Attractor
                </div>

                <!-- User Purpose -->
                <div style="background-color: #2d2d2d; border: 2px solid #4CAF50; border-radius: 0 0 8px 8px; padding: 12px;">
                    <div style="color: #4CAF50; font-weight: bold; margin-bottom: 8px; font-size: 14px;">Your Purpose</div>
                    <div style="color: #e0e0e0; line-height: 1.5; font-size: 13px;">
                        • Understand TELOS without technical overwhelm<br>
                        • Learn how purpose alignment keeps AI focused<br>
                        • See real examples of governance in action
                    </div>
                </div>
            </td>

            <td style="width: 50%; vertical-align: top; padding: 0;">
                <!-- RIGHT COLUMN: AI PA -->
                <div style="background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%); color: #1a1a1a; padding: 8px; border-radius: 8px 8px 0 0; font-weight: bold; text-align: center; font-size: 16px;">
                    🤖 AI Primacy Attractor
                </div>

                <!-- AI Purpose -->
                <div style="background-color: #2d2d2d; border: 2px solid #FFD700; border-radius: 0 0 8px 8px; padding: 12px; margin-bottom: 10px;">
                    <div style="color: #FFD700; font-weight: bold; margin-bottom: 8px; font-size: 14px;">Purpose</div>
                    <div style="color: #e0e0e0; line-height: 1.5; font-size: 13px;">
                        • Help you understand TELOS naturally<br>
                        • Stay aligned with your learning goals<br>
                        • Embody human dignity through action
                    </div>
                </div>

                <!-- AI Scope -->
                <div style="background-color: #2d2d2d; border: 2px solid #FFD700; border-radius: 8px; padding: 12px; margin-bottom: 10px;">
                    <div style="color: #FFD700; font-weight: bold; margin-bottom: 8px; font-size: 14px;">Scope</div>
                    <div style="color: #e0e0e0; line-height: 1.5; font-size: 13px;">
                        • TELOS dual attractor system<br>
                        • Perfect equilibrium & primacy basin<br>
                        • Real-time drift detection<br>
                        • Trust through transparency
                    </div>
                </div>

                <!-- AI Boundaries -->
                <div style="background-color: #2d2d2d; border: 2px solid #FFD700; border-radius: 8px; padding: 12px;">
                    <div style="color: #FFD700; font-weight: bold; margin-bottom: 8px; font-size: 14px;">Boundaries</div>
                    <div style="color: #e0e0e0; line-height: 1.5; font-size: 13px;">
                        • Answer what you asked<br>
                        • Stay conversational<br>
                        • 2-3 paragraphs max<br>
                        • No technical jargon
                    </div>
                </div>
            </td>
        </tr>
        </table>
        """, unsafe_allow_html=True)

        # Close main container
        st.markdown('</div>', unsafe_allow_html=True)

        # Close centered container
        st.markdown('</div>', unsafe_allow_html=True)

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
        # First row: User and AI Fidelity (the dual attractors)
        col1, col2 = st.columns(2)

        with col1:
            user_fidelity = turn_data.get('user_fidelity', 1.0)
            # 4-tier color system: Green (≥0.85) | Yellow (0.70-0.85) | Orange (0.50-0.70) | Red (<0.50)
            if user_fidelity >= 0.85:
                user_color = "#4CAF50"  # Green
            elif user_fidelity >= 0.70:
                user_color = "#FFD700"  # Yellow
            elif user_fidelity >= 0.50:
                user_color = "#FFA500"  # Orange
            else:
                user_color = "#FF4444"  # Red
            self._render_metric_card(
                "User Fidelity",
                f"{user_fidelity:.3f}",
                "👤",
                "Your alignment to your PA",
                value_color=user_color
            )

        with col2:
            ai_fidelity = turn_data.get('ai_fidelity', 1.0)
            # 4-tier color system: Green (≥0.85) | Yellow (0.70-0.85) | Orange (0.50-0.70) | Red (<0.50)
            if ai_fidelity >= 0.85:
                ai_color = "#4CAF50"  # Green
            elif ai_fidelity >= 0.70:
                ai_color = "#FFD700"  # Yellow
            elif ai_fidelity >= 0.50:
                ai_color = "#FFA500"  # Orange
            else:
                ai_color = "#FF4444"  # Red
            self._render_metric_card(
                "AI Fidelity",
                f"{ai_fidelity:.3f}",
                "🤖",
                "My alignment to serve you",
                value_color=ai_color
            )

        # Second row: Primacy State and Intervention Status
        col3, col4 = st.columns(2)

        with col3:
            # Calculate Primacy State from both fidelities
            user_f = turn_data.get('user_fidelity', 1.0)
            ai_f = turn_data.get('ai_fidelity', 1.0)
            primacy_state = turn_data.get('primacy_state', (user_f * ai_f))  # Simple PS approximation
            # 4-tier color system: Green (≥0.85) | Yellow (0.70-0.85) | Orange (0.50-0.70) | Red (<0.50)
            if primacy_state >= 0.85:
                ps_color = "#4CAF50"  # Green
            elif primacy_state >= 0.70:
                ps_color = "#FFD700"  # Yellow
            elif primacy_state >= 0.50:
                ps_color = "#FFA500"  # Orange
            else:
                ps_color = "#FF4444"  # Red
            self._render_metric_card(
                "Primacy State",
                f"{primacy_state:.3f}",
                "🎯",
                "Dynamic equilibrium between your goals and AI alignment",
                value_color=ps_color
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
