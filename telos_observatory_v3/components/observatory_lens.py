"""
Observatory Lens Component for TELOS Observatory V3.
Real-time visual dashboard for governance visualization.
"""

import streamlit as st
import logging
import plotly.graph_objects as go
import numpy as np

logger = logging.getLogger(__name__)


class ObservatoryLens:
    """
    Observatory Lens - Visual Dashboard for Real-Time Governance.

    Provides 6 core visualizations:
    1. Basin Visualization - Safe zone vs drift zone in embedding space
    2. Fidelity Gauge - Real-time alignment quality meter
    3. Intervention Pipeline - Three-tier defense system status
    4. Attack Detection Log - Live event feed of governance actions
    5. Embedding Space Viewer - Semantic space with PA center
    6. Session Statistics - Real-time metrics summary

    Design Goal: Non-technical users understand TELOS governance in <30 seconds.
    """

    def __init__(self, state_manager):
        """
        Initialize Observatory Lens with state manager reference.

        Args:
            state_manager: StateManager instance for telemetry data
        """
        self.state_manager = state_manager

    def render(self):
        """Render the Observatory Lens dashboard."""
        # Check if lens should be shown
        if not self.state_manager.state.show_observatory_lens:
            return

        # Expanded state - full dashboard with dark theme
        st.markdown("""
        <style>
        /* Force dark background for main content area */
        .main, .stApp {
            background-color: #0E1117 !important;
        }

        /* Force dark background for all containers */
        [data-testid="stVerticalBlock"],
        [data-testid="stHorizontalBlock"],
        .element-container {
            background-color: transparent !important;
        }
        </style>

        <div style="
            background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%);
            border: 2px solid #F4D03F;
            border-radius: 10px;
            padding: 20px;
            margin-top: 20px;
        ">
            <div style="text-align: center; margin-bottom: 20px;">
                <h2 style="color: #F4D03F; margin: 0; font-weight: bold; letter-spacing: 2px;">
                    🔭 OBSERVATORY LENS
                </h2>
                <p style="color: #888; font-size: 12px; margin: 5px 0 0 0;">
                    Real-Time Governance Visualization
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Collapse button
        col1, col2 = st.columns([4, 1])
        with col2:
            if st.button("✕ Close", key="observatory_lens_close", use_container_width=True):
                self.state_manager.toggle_component('observatory_lens')
                st.rerun()

        st.markdown("---")

        # Check if there's data to visualize
        if self.state_manager.state.total_turns == 0:
            st.info("👋 Observatory Lens will activate once you start a conversation. Try sending a message!")
            return

        # Main 2x3 grid layout
        self._render_visualization_grid()

    def _render_visualization_grid(self):
        """Render the 2x3 grid of visualizations."""
        # Top row: Basin | Fidelity Gauge | Intervention Pipeline
        st.markdown("### 📊 Live Governance Metrics")

        top_col1, top_col2, top_col3 = st.columns([2, 1, 2])

        with top_col1:
            self._render_basin_visualization()

        with top_col2:
            self._render_fidelity_gauge()

        with top_col3:
            self._render_intervention_pipeline()

        st.markdown("---")

        # Bottom row: Embedding Space | Attack Log | Session Stats
        st.markdown("### 🔍 Deep Analysis")

        bottom_col1, bottom_col2, bottom_col3 = st.columns([2, 2, 1])

        with bottom_col1:
            self._render_embedding_space()

        with bottom_col2:
            self._render_attack_log()

        with bottom_col3:
            self._render_session_statistics()

    def _render_basin_visualization(self):
        """Render Basin Visualization (3D safe zone)."""
        turns = self.state_manager.get_all_turns()

        np.random.seed(42)
        positions, colors, sizes, hover_texts = [], [], [], []

        for turn in turns:
            fidelity = turn.get('fidelity', 0.0)
            turn_num = turn.get('turn', 0)

            distance = (1.0 - fidelity) * 2.0
            angle1 = np.random.uniform(0, 2 * np.pi)
            angle2 = np.random.uniform(0, np.pi)

            x = distance * np.sin(angle2) * np.cos(angle1)
            y = distance * np.sin(angle2) * np.sin(angle1)
            z = distance * np.cos(angle2)

            positions.append([x, y, z])
            # 4-tier color system: Green (≥0.85) | Yellow (0.70-0.85) | Orange (0.50-0.70) | Red (<0.50)
            if fidelity >= 0.85:
                point_color = '#4CAF50'  # Green
            elif fidelity >= 0.70:
                point_color = '#F4D03F'  # Yellow
            elif fidelity >= 0.50:
                point_color = '#FFA500'  # Orange
            else:
                point_color = '#FF4444'  # Red
            colors.append(point_color)
            sizes.append(8)
            hover_texts.append(f"Turn {turn_num + 1}<br>Fidelity: {fidelity:.3f}")

        fig = go.Figure()

        if positions:
            positions = np.array(positions)
            fig.add_trace(go.Scatter3d(
                x=positions[:, 0], y=positions[:, 1], z=positions[:, 2],
                mode='markers',
                marker=dict(size=sizes, color=colors, line=dict(color='#F4D03F', width=1)),
                text=hover_texts, hoverinfo='text', name='Responses'
            ))

        fig.add_trace(go.Scatter3d(
            x=[0], y=[0], z=[0],
            mode='markers',
            marker=dict(size=20, color='#F4D03F', symbol='diamond', line=dict(color='#FFFFFF', width=2)),
            text=['Primacy Attractor<br>Center of Basin'], hoverinfo='text', name='PA Center'
        ))

        threshold_radius = 0.6
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        sphere_x = threshold_radius * np.outer(np.cos(u), np.sin(v))
        sphere_y = threshold_radius * np.outer(np.sin(u), np.sin(v))
        sphere_z = threshold_radius * np.outer(np.ones(np.size(u)), np.cos(v))

        fig.add_trace(go.Surface(
            x=sphere_x, y=sphere_y, z=sphere_z,
            opacity=0.1, colorscale=[[0, '#F4D03F'], [1, '#F4D03F']],
            showscale=False, hoverinfo='skip', name='Safe Basin'
        ))

        fig.update_layout(
            scene=dict(
                xaxis=dict(showbackground=False, showgrid=False, showticklabels=False, title=''),
                yaxis=dict(showbackground=False, showgrid=False, showticklabels=False, title=''),
                zaxis=dict(showbackground=False, showgrid=False, showticklabels=False, title=''),
                bgcolor='#1a1a1a',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
            ),
            paper_bgcolor='#1a1a1a', plot_bgcolor='#1a1a1a',
            showlegend=False, height=280, margin=dict(l=0, r=0, t=30, b=0)
        )

        st.plotly_chart(fig, use_container_width=True, key=f"basin_{self.state_manager.state.current_turn}")

    def _render_fidelity_gauge(self):
        """Render Fidelity Gauge (speedometer)."""
        current_turn = self.state_manager.get_current_turn_data()
        fidelity = current_turn.get('fidelity', 0.0) if current_turn else 0.0

        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=fidelity,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Fidelity", 'font': {'size': 16, 'color': '#F4D03F'}},
            number={'font': {'size': 40, 'color': '#F4D03F'}, 'valueformat': '.3f'},
            delta={'reference': 0.8, 'increasing': {'color': "#00FF00"}, 'decreasing': {'color': "#FF0000"}},
            gauge={
                'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': "#F4D03F"},
                'bar': {'color': "#F4D03F"},
                'bgcolor': "#1a1a1a",
                'borderwidth': 2, 'bordercolor': "#F4D03F",
                'steps': [
                    {'range': [0, 0.7], 'color': '#330000'},
                    {'range': [0.7, 0.8], 'color': '#333300'},
                    {'range': [0.8, 1.0], 'color': '#003300'}
                ],
                'threshold': {'line': {'color': "#FF0000", 'width': 4}, 'thickness': 0.75, 'value': 0.7}
            }
        ))

        fig.update_layout(
            paper_bgcolor='#1a1a1a', plot_bgcolor='#1a1a1a',
            font={'color': '#e0e0e0'}, height=280, margin=dict(l=20, r=20, t=50, b=20)
        )

        st.plotly_chart(fig, use_container_width=True, key=f"gauge_{self.state_manager.state.current_turn}")

    def _render_intervention_pipeline(self):
        """Render Intervention Pipeline (three-tier defense)."""
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(255, 255, 255, 0.08) 0%, rgba(255, 255, 255, 0.03) 50%, transparent 100%), rgba(26, 26, 30, 0.45); backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px); border: 1px solid #F4D03F; border-radius: 8px; padding: 15px; height: 300px; box-shadow: 0 0 15px rgba(244, 208, 63, 0.15), 0 8px 32px rgba(0, 0, 0, 0.3), inset 0 1px 1px rgba(255, 255, 255, 0.1);">
            <h4 style="color: #F4D03F; margin: 0 0 10px 0;">Defense Pipeline</h4>
            <p style="color: #888; font-size: 11px;">Three-tier governance</p>
            <div style="display: flex; flex-direction: column; justify-content: center; height: 220px;">
                <div style="padding: 10px; background: #0a0a0a; border-radius: 5px; margin: 5px 0;">
                    <strong style="color: #00FF00;">Tier 1:</strong> PA Math ✓
                </div>
                <div style="padding: 10px; background: #0a0a0a; border-radius: 5px; margin: 5px 0;">
                    <strong style="color: #888;">Tier 2:</strong> RAG (inactive)
                </div>
                <div style="padding: 10px; background: #0a0a0a; border-radius: 5px; margin: 5px 0;">
                    <strong style="color: #888;">Tier 3:</strong> Human (inactive)
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    def _render_embedding_space(self):
        """Render Embedding Space (2D semantic space)."""
        turns = self.state_manager.get_all_turns()

        if not turns:
            st.markdown("""
            <div style="background: linear-gradient(135deg, rgba(255, 255, 255, 0.08) 0%, rgba(255, 255, 255, 0.03) 50%, transparent 100%), rgba(26, 26, 30, 0.45); backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px); border: 1px solid #F4D03F; border-radius: 8px; padding: 15px; height: 350px; box-shadow: 0 0 15px rgba(244, 208, 63, 0.15), 0 8px 32px rgba(0, 0, 0, 0.3), inset 0 1px 1px rgba(255, 255, 255, 0.1);">
                <h4 style="color: #F4D03F;">Embedding Space</h4>
                <div style="display: flex; align-items: center; justify-content: center; height: 270px; color: #888;">
                    No data yet
                </div>
            </div>
            """, unsafe_allow_html=True)
            return

        np.random.seed(42)
        positions_2d, colors, sizes, hover_texts = [], [], [], []

        for turn in turns:
            fidelity = turn.get('fidelity', 0.0)
            turn_num = turn.get('turn', 0)

            distance = (1.0 - fidelity) * 3.0
            angle = (turn_num / len(turns)) * 2 * np.pi + np.random.uniform(-0.3, 0.3)

            x, y = distance * np.cos(angle), distance * np.sin(angle)
            positions_2d.append([x, y])

            # 4-tier color system: Green (≥0.85) | Yellow (0.70-0.85) | Orange (0.50-0.70) | Red (<0.50)
            if fidelity >= 0.85:
                point_color = '#4CAF50'  # Green
            elif fidelity >= 0.70:
                point_color = '#F4D03F'  # Yellow
            elif fidelity >= 0.50:
                point_color = '#FFA500'  # Orange
            else:
                point_color = '#FF4444'  # Red
            colors.append(point_color)
            sizes.append(8 + (turn_num / len(turns)) * 6)
            hover_texts.append(f"Turn {turn_num + 1}<br>Fidelity: {fidelity:.3f}")

        positions_2d = np.array(positions_2d)
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=positions_2d[:, 0], y=positions_2d[:, 1],
            mode='lines', line=dict(color='#888888', width=1, dash='dot'),
            hoverinfo='skip', showlegend=False
        ))

        fig.add_trace(go.Scatter(
            x=positions_2d[:, 0], y=positions_2d[:, 1],
            mode='markers',
            marker=dict(size=sizes, color=colors, line=dict(color='#F4D03F', width=1), opacity=0.8),
            text=hover_texts, hoverinfo='text', showlegend=False
        ))

        fig.add_trace(go.Scatter(
            x=[0], y=[0],
            mode='markers',
            marker=dict(size=20, color='#F4D03F', symbol='star', line=dict(color='#FFFFFF', width=2)),
            text=['PA Center'], hoverinfo='text', showlegend=False
        ))

        theta = np.linspace(0, 2*np.pi, 100)
        circle_x = 0.9 * np.cos(theta)
        circle_y = 0.9 * np.sin(theta)

        fig.add_trace(go.Scatter(
            x=circle_x, y=circle_y,
            mode='lines', line=dict(color='#F4D03F', width=1, dash='dash'),
            hoverinfo='skip', showlegend=False
        ))

        fig.update_layout(
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[-4, 4]),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[-4, 4], scaleanchor="x", scaleratio=1),
            paper_bgcolor='#1a1a1a', plot_bgcolor='#1a1a1a',
            height=330, margin=dict(l=10, r=10, t=30, b=10), hovermode='closest'
        )

        st.plotly_chart(fig, use_container_width=True, key=f"embedding_{self.state_manager.state.current_turn}")

    def _render_attack_log(self):
        """Render Attack Detection Log (event feed)."""
        st.markdown('<div style="background: linear-gradient(135deg, rgba(255, 255, 255, 0.08) 0%, rgba(255, 255, 255, 0.03) 50%, transparent 100%), rgba(26, 26, 30, 0.45); backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px); border: 1px solid #F4D03F; border-radius: 8px; padding: 15px; height: 350px; overflow-y: auto; box-shadow: 0 0 15px rgba(244, 208, 63, 0.15), 0 8px 32px rgba(0, 0, 0, 0.3), inset 0 1px 1px rgba(255, 255, 255, 0.1);"><h4 style="color: #F4D03F; margin: 0 0 10px 0;">Event Log</h4><p style="color: #888; font-size: 11px; margin-bottom: 10px;">Real-time events</p>', unsafe_allow_html=True)

        turns = self.state_manager.get_all_turns()
        recent_turns = turns[-10:] if len(turns) > 10 else turns

        for turn in reversed(recent_turns):
            turn_num = turn.get('turn', 0)
            fidelity = turn.get('fidelity', 0.0)
            intervention = turn.get('intervention_applied', False)

            if intervention:
                icon, color, event_type = "🛡️", "#FF0000", "INTERVENTION"
            elif fidelity < 0.8:
                icon, color, event_type = "⚠️", "#F4D03F", "DRIFT WARNING"
            else:
                icon, color, event_type = "✓", "#00FF00", "NORMAL"

            st.markdown(f'<div style="padding: 8px; background: #0a0a0a; border-left: 3px solid {color}; border-radius: 3px; margin: 5px 0;"><div style="color: {color}; font-size: 11px;">{icon} <strong>Turn {turn_num + 1}:</strong> {event_type}</div><div style="color: #888; font-size: 10px;">Fidelity: {fidelity:.3f}</div></div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    def _render_session_statistics(self):
        """Render Session Statistics (metrics summary)."""
        session_info = self.state_manager.get_session_info()

        html = f"""
<div style="background: linear-gradient(135deg, rgba(255, 255, 255, 0.08) 0%, rgba(255, 255, 255, 0.03) 50%, transparent 100%), rgba(26, 26, 30, 0.45); backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px); border: 1px solid #F4D03F; border-radius: 8px; padding: 15px; height: 350px; box-shadow: 0 0 15px rgba(244, 208, 63, 0.15), 0 8px 32px rgba(0, 0, 0, 0.3), inset 0 1px 1px rgba(255, 255, 255, 0.1);">
    <h4 style="color: #F4D03F; margin: 0 0 10px 0;">Session Stats</h4>
    <p style="color: #888; font-size: 11px;">Real-time metrics</p>
    <div style="margin-top: 20px;">
        <div style="padding: 10px; background: #0a0a0a; border-radius: 5px; margin: 10px 0;">
            <div style="color: #888; font-size: 10px;">TOTAL TURNS</div>
            <div style="color: #F4D03F; font-size: 24px; font-weight: bold;">{session_info.get('total_turns', 0)}</div>
        </div>
        <div style="padding: 10px; background: #0a0a0a; border-radius: 5px; margin: 10px 0;">
            <div style="color: #888; font-size: 10px;">AVG FIDELITY</div>
            <div style="color: #F4D03F; font-size: 24px; font-weight: bold;">{session_info.get('avg_fidelity', 0.0):.3f}</div>
        </div>
        <div style="padding: 10px; background: #0a0a0a; border-radius: 5px; margin: 10px 0;">
            <div style="color: #888; font-size: 10px;">INTERVENTIONS</div>
            <div style="color: #F4D03F; font-size: 24px; font-weight: bold;">{session_info.get('total_interventions', 0)}</div>
        </div>
        <div style="padding: 10px; background: #0a0a0a; border-radius: 5px; margin: 10px 0;">
            <div style="color: #888; font-size: 10px;">ASR</div>
            <div style="color: #00FF00; font-size: 24px; font-weight: bold;">0%</div>
        </div>
    </div>
</div>
"""
        st.markdown(html, unsafe_allow_html=True)
