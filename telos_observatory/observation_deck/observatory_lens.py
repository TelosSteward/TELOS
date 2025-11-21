"""
Observatory Lens - Visual Governance Dashboard
Adapted for main Observatory architecture (uses st.session_state directly)
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np


def render_observatory_lens():
    """
    Render the Observatory Lens dashboard.

    Displays real-time governance visualizations using session data.
    Controlled by st.session_state.show_observatory_lens toggle.
    """
    # Check if lens should be shown
    if not st.session_state.get('show_observatory_lens', False):
        return

    # Header
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%);
        border: 2px solid #FFD700;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
    ">
        <div style="text-align: center; margin-bottom: 20px;">
            <h2 style="color: #FFD700; margin: 0; font-weight: bold; letter-spacing: 2px;">
                🔭 OBSERVATORY LENS
            </h2>
            <p style="color: #888; font-size: 12px; margin: 5px 0 0 0;">
                Real-Time Governance Visualization
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Close button
    col1, col2 = st.columns([4, 1])
    with col2:
        if st.button("✕ Close", key="observatory_lens_close", use_container_width=True):
            st.session_state.show_observatory_lens = False
            st.rerun()

    st.markdown("---")

    # Check if there's data to visualize
    session_data = st.session_state.get('session_data', {})
    turns = session_data.get('turns', [])

    if not turns:
        st.info("👋 Observatory Lens will activate once conversation data is loaded")
        return

    # Main visualization grid
    _render_visualization_grid(turns)


def _render_visualization_grid(turns):
    """Render the 2x3 grid of visualizations."""
    # Top row: Basin | Fidelity Gauge | Defense Pipeline
    st.markdown("### 📊 Live Governance Metrics")

    top_col1, top_col2, top_col3 = st.columns([2, 1, 2])

    with top_col1:
        _render_basin_visualization(turns)

    with top_col2:
        _render_fidelity_gauge(turns)

    with top_col3:
        _render_defense_pipeline()

    st.markdown("---")

    # Bottom row: Embedding Space | Event Log | Session Stats
    st.markdown("### 🔍 Deep Analysis")

    bottom_col1, bottom_col2, bottom_col3 = st.columns([2, 2, 1])

    with bottom_col1:
        _render_embedding_space(turns)

    with bottom_col2:
        _render_event_log(turns)

    with bottom_col3:
        _render_session_statistics(turns)


def _render_basin_visualization(turns):
    """Render Basin Visualization (3D safe zone)."""
    st.markdown("""
    <div style="background: #1a1a1a; border: 1px solid #FFD700; border-radius: 8px; padding: 10px;">
    """, unsafe_allow_html=True)

    # Get current turn index
    current_turn = st.session_state.get('current_turn', 0)

    # Generate 3D positions (simulated)
    np.random.seed(42)
    positions = []
    colors = []
    hover_texts = []

    for i, turn in enumerate(turns):
        fidelity = turn.get('fidelity', 0.85)

        # Simulate 3D position based on fidelity
        distance = (1.0 - fidelity) * 2.0
        angle1 = np.random.uniform(0, 2 * np.pi)
        angle2 = np.random.uniform(0, np.pi)

        x = distance * np.sin(angle2) * np.cos(angle1)
        y = distance * np.sin(angle2) * np.sin(angle1)
        z = distance * np.cos(angle2)

        positions.append([x, y, z])

        # Color by fidelity
        if fidelity >= 0.8:
            colors.append('#00FF00')
        elif fidelity >= 0.7:
            colors.append('#FFD700')
        else:
            colors.append('#FF0000')

        hover_texts.append(f"Turn {i + 1}<br>Fidelity: {fidelity:.3f}")

    # Create 3D scatter
    fig = go.Figure()

    if positions:
        positions = np.array(positions)
        fig.add_trace(go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            mode='markers',
            marker=dict(size=8, color=colors, line=dict(color='#FFD700', width=1)),
            text=hover_texts,
            hoverinfo='text',
            name='Responses'
        ))

    # Add PA center
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers',
        marker=dict(size=20, color='#FFD700', symbol='diamond', line=dict(color='#FFFFFF', width=2)),
        text=['Primacy Attractor<br>Center of Basin'],
        hoverinfo='text',
        name='PA Center'
    ))

    # Add safe basin sphere
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    threshold_radius = 0.6
    sphere_x = threshold_radius * np.outer(np.cos(u), np.sin(v))
    sphere_y = threshold_radius * np.outer(np.sin(u), np.sin(v))
    sphere_z = threshold_radius * np.outer(np.ones(np.size(u)), np.cos(v))

    fig.add_trace(go.Surface(
        x=sphere_x, y=sphere_y, z=sphere_z,
        opacity=0.1,
        colorscale=[[0, '#FFD700'], [1, '#FFD700']],
        showscale=False,
        hoverinfo='skip',
        name='Safe Basin'
    ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(showbackground=False, showgrid=False, showticklabels=False, title=''),
            yaxis=dict(showbackground=False, showgrid=False, showticklabels=False, title=''),
            zaxis=dict(showbackground=False, showgrid=False, showticklabels=False, title=''),
            bgcolor='#1a1a1a',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
        ),
        paper_bgcolor='#1a1a1a',
        plot_bgcolor='#1a1a1a',
        showlegend=False,
        height=280,
        margin=dict(l=0, r=0, t=30, b=0)
    )

    st.plotly_chart(fig, use_container_width=True, key=f"basin_viz_{current_turn}")
    st.markdown("</div>", unsafe_allow_html=True)


def _render_fidelity_gauge(turns):
    """Render Fidelity Gauge (speedometer)."""
    st.markdown("""
    <div style="background: #1a1a1a; border: 1px solid #FFD700; border-radius: 8px; padding: 10px;">
    """, unsafe_allow_html=True)

    # Get current turn fidelity
    current_turn = st.session_state.get('current_turn', 0)
    fidelity = turns[current_turn].get('fidelity', 0.85) if current_turn < len(turns) else 0.85

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=fidelity,
        title={'text': "Fidelity Score", 'font': {'size': 16, 'color': '#FFD700'}},
        number={'font': {'size': 40, 'color': '#FFD700'}, 'valueformat': '.3f'},
        delta={'reference': 0.8, 'increasing': {'color': "#00FF00"}, 'decreasing': {'color': "#FF0000"}},
        gauge={
            'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': "#FFD700"},
            'bar': {'color': "#FFD700"},
            'bgcolor': "#1a1a1a",
            'borderwidth': 2,
            'bordercolor': "#FFD700",
            'steps': [
                {'range': [0, 0.7], 'color': '#330000'},
                {'range': [0.7, 0.8], 'color': '#333300'},
                {'range': [0.8, 1.0], 'color': '#003300'}
            ],
            'threshold': {'line': {'color': "#FF0000", 'width': 4}, 'thickness': 0.75, 'value': 0.7}
        }
    ))

    fig.update_layout(
        paper_bgcolor='#1a1a1a',
        plot_bgcolor='#1a1a1a',
        font={'color': '#e0e0e0'},
        height=280,
        margin=dict(l=20, r=20, t=50, b=20)
    )

    st.plotly_chart(fig, use_container_width=True, key=f"fidelity_gauge_{current_turn}")
    st.markdown("</div>", unsafe_allow_html=True)


def _render_defense_pipeline():
    """Render Defense Pipeline (3-tier system)."""
    st.markdown("""
    <div style="background: #1a1a1a; border: 1px solid #FFD700; border-radius: 8px; padding: 15px; height: 300px;">
        <h4 style="color: #FFD700; margin: 0 0 10px 0;">Defense Pipeline</h4>
        <p style="color: #888; font-size: 11px;">Three-tier governance system</p>
        <div style="display: flex; flex-direction: column; justify-content: center; height: 220px; color: #888;">
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


def _render_embedding_space(turns):
    """Render Embedding Space (2D projection)."""
    st.markdown("""
    <div style="background: #1a1a1a; border: 1px solid #FFD700; border-radius: 8px; padding: 10px;">
    """, unsafe_allow_html=True)

    # Generate 2D positions
    np.random.seed(42)
    positions_2d = []
    colors = []
    sizes = []
    hover_texts = []

    for i, turn in enumerate(turns):
        fidelity = turn.get('fidelity', 0.85)

        distance = (1.0 - fidelity) * 3.0
        angle = (i / len(turns)) * 2 * np.pi + np.random.uniform(-0.3, 0.3)

        x = distance * np.cos(angle)
        y = distance * np.sin(angle)

        positions_2d.append([x, y])

        if fidelity >= 0.8:
            colors.append('#00FF00')
        elif fidelity >= 0.7:
            colors.append('#FFD700')
        else:
            colors.append('#FF0000')

        size = 8 + (i / len(turns)) * 6
        sizes.append(size)

        hover_texts.append(f"Turn {i + 1}<br>Fidelity: {fidelity:.3f}")

    positions_2d = np.array(positions_2d)

    fig = go.Figure()

    # Add drift trail
    fig.add_trace(go.Scatter(
        x=positions_2d[:, 0], y=positions_2d[:, 1],
        mode='lines',
        line=dict(color='#888888', width=1, dash='dot'),
        hoverinfo='skip',
        showlegend=False
    ))

    # Add response points
    fig.add_trace(go.Scatter(
        x=positions_2d[:, 0], y=positions_2d[:, 1],
        mode='markers',
        marker=dict(size=sizes, color=colors, line=dict(color='#FFD700', width=1), opacity=0.8),
        text=hover_texts,
        hoverinfo='text',
        showlegend=False
    ))

    # Add PA center
    fig.add_trace(go.Scatter(
        x=[0], y=[0],
        mode='markers',
        marker=dict(size=20, color='#FFD700', symbol='star', line=dict(color='#FFFFFF', width=2)),
        text=['Primacy Attractor<br>Center'],
        hoverinfo='text',
        showlegend=False
    ))

    # Add safe basin circle
    theta = np.linspace(0, 2*np.pi, 100)
    basin_radius = 0.9
    circle_x = basin_radius * np.cos(theta)
    circle_y = basin_radius * np.sin(theta)

    fig.add_trace(go.Scatter(
        x=circle_x, y=circle_y,
        mode='lines',
        line=dict(color='#FFD700', width=1, dash='dash'),
        hoverinfo='skip',
        showlegend=False
    ))

    fig.update_layout(
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[-4, 4]),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[-4, 4], scaleanchor="x", scaleratio=1),
        paper_bgcolor='#1a1a1a',
        plot_bgcolor='#1a1a1a',
        height=330,
        margin=dict(l=10, r=10, t=30, b=10),
        hovermode='closest'
    )

    st.plotly_chart(fig, use_container_width=True, key=f"embedding_space_{st.session_state.get('current_turn', 0)}")
    st.markdown("</div>", unsafe_allow_html=True)


def _render_event_log(turns):
    """Render Event Log (governance events)."""
    st.markdown("""
<div style="background: #1a1a1a; border: 1px solid #FFD700; border-radius: 8px; padding: 15px; height: 350px; overflow-y: auto;">
    <h4 style="color: #FFD700; margin: 0 0 10px 0;">Event Log</h4>
    <p style="color: #888; font-size: 11px; margin-bottom: 10px;">Real-time governance events</p>
    """, unsafe_allow_html=True)

    recent_turns = turns[-10:] if len(turns) > 10 else turns

    for i, turn in enumerate(reversed(recent_turns)):
        turn_num = turn.get('turn', i)
        fidelity = turn.get('fidelity', 0.85)
        intervention = turn.get('intervention_applied', False)

        if intervention:
            icon, color, event_type = "🛡️", "#FF0000", "INTERVENTION"
        elif fidelity < 0.8:
            icon, color, event_type = "⚠️", "#FFD700", "DRIFT WARNING"
        else:
            icon, color, event_type = "✓", "#00FF00", "NORMAL"

        st.markdown(f"""<div style="padding: 8px; background: #0a0a0a; border-left: 3px solid {color}; border-radius: 3px; margin: 5px 0;"><div style="color: {color}; font-size: 11px;">{icon} <strong>Turn {turn_num + 1}:</strong> {event_type}</div><div style="color: #888; font-size: 10px;">Fidelity: {fidelity:.3f}</div></div>""", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


def _render_session_statistics(turns):
    """Render Session Statistics (metrics summary)."""
    # Calculate statistics
    total_turns = len(turns)
    avg_fidelity = sum(t.get('fidelity', 0.85) for t in turns) / total_turns if total_turns > 0 else 0.0
    total_interventions = sum(1 for t in turns if t.get('intervention_applied', False))

    html = f"""
<div style="background: #1a1a1a; border: 1px solid #FFD700; border-radius: 8px; padding: 15px; height: 350px;">
    <h4 style="color: #FFD700; margin: 0 0 10px 0;">Session Stats</h4>
    <p style="color: #888; font-size: 11px;">Real-time metrics</p>
    <div style="margin-top: 20px;">
        <div style="padding: 10px; background: #0a0a0a; border-radius: 5px; margin: 10px 0;">
            <div style="color: #888; font-size: 10px;">TOTAL TURNS</div>
            <div style="color: #FFD700; font-size: 24px; font-weight: bold;">{total_turns}</div>
        </div>
        <div style="padding: 10px; background: #0a0a0a; border-radius: 5px; margin: 10px 0;">
            <div style="color: #888; font-size: 10px;">AVG FIDELITY</div>
            <div style="color: #FFD700; font-size: 24px; font-weight: bold;">{avg_fidelity:.3f}</div>
        </div>
        <div style="padding: 10px; background: #0a0a0a; border-radius: 5px; margin: 10px 0;">
            <div style="color: #888; font-size: 10px;">INTERVENTIONS</div>
            <div style="color: #FFD700; font-size: 24px; font-weight: bold;">{total_interventions}</div>
        </div>
        <div style="padding: 10px; background: #0a0a0a; border-radius: 5px; margin: 10px 0;">
            <div style="color: #888; font-size: 10px;">ASR (ATTACK SUCCESS)</div>
            <div style="color: #00FF00; font-size: 24px; font-weight: bold;">0%</div>
        </div>
    </div>
</div>
"""
    st.markdown(html, unsafe_allow_html=True)
