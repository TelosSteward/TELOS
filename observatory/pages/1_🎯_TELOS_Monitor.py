"""
TELOS Governance Monitor - Real-time PA tracking, fidelity, interventions.
Multi-page app component for Observatory Developer Suite.
"""

import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.telos_bridge import TelosBridge


# Page configuration
st.set_page_config(
    page_title="TELOS Monitor",
    page_icon="🎯",
    layout="wide"
)

# Apply Observatory styling
st.markdown("""
<style>
/* Match Observatory dark theme */
.stApp {
    background-color: #1a1a1a !important;
}

h1, h2, h3 {
    color: #FFD700 !important;
}

[data-testid="stMetricValue"] {
    color: #FFD700 !important;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<h1 style='text-align: center; color: #FFD700;'>
    🎯 TELOS Governance Monitor
</h1>
<p style='text-align: center; color: #888;'>
    Real-time PA tracking, fidelity metrics, and intervention logs
</p>
<hr style='border: 1px solid #FFD700;'>
""", unsafe_allow_html=True)

# Get state_manager from main app session state
if 'state_manager' in st.session_state:
    bridge = TelosBridge(st.session_state.state_manager)
    session_active = True
else:
    bridge = TelosBridge(None)
    session_active = False

# Session status
if session_active:
    stats = bridge.get_session_stats()
    st.success(f"✅ Active Session: {stats['session_id']}")
else:
    st.warning("⚠️ No active session - Start a conversation in DEVOPS mode to see live data")
    st.stop()

st.markdown("---")

# Three-column minimalist layout
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🎯 Primacy Attractor")

    pa_status = bridge.get_pa_status()

    if pa_status['state'] == 'not_available':
        st.info("No PA data available")
    else:
        # Status indicator
        if pa_status['converged']:
            st.success("✅ Converged")
        else:
            st.warning(f"⏳ Calibrating ({pa_status['current_turn']}/~10)")

        # Metrics
        st.metric("Current Turn", pa_status['current_turn'])
        st.metric("Total Turns", pa_status['total_turns'])

        if not pa_status['converged'] and pa_status['current_turn'] > 0:
            progress = min(pa_status['current_turn'] / 10.0, 1.0)
            st.progress(progress, text=f"{int(progress*100)}% to convergence")

with col2:
    st.markdown("### 📊 Fidelity Tracking")

    metrics = bridge.get_fidelity_metrics()

    if metrics['current'] is None:
        st.info("No fidelity data available")
    else:
        # Current fidelity with color coding
        if metrics['current'] >= 0.8:
            st.success(f"Current: {metrics['current']:.3f}")
        elif metrics['current'] >= 0.6:
            st.warning(f"Current: {metrics['current']:.3f}")
        else:
            st.error(f"Current: {metrics['current']:.3f}")

        st.metric("Average", metrics['average'])
        st.metric("Violations", metrics['violations'],
                 delta="Low" if metrics['violations'] == 0 else "Check logs")

        # Mini trend chart
        if metrics['trend'] and len(metrics['trend']) > 1:
            st.line_chart(metrics['trend'])

with col3:
    st.markdown("### ⚠️ Interventions")

    interventions = bridge.get_intervention_log()
    stats = bridge.get_session_stats()

    st.metric("Total", stats['total_interventions'])

    if interventions:
        latest = interventions[-1]
        st.info(f"""
**Latest**: Turn {latest['turn']}
Type: {latest['type']}
        """)
    else:
        st.success("No interventions needed")

st.markdown("---")

# Detailed intervention log
st.markdown("### 📋 Intervention Log")

interventions = bridge.get_intervention_log()

if not interventions:
    st.info("No interventions recorded in this session")
else:
    # Display as table
    for intervention in reversed(interventions):  # Most recent first
        with st.expander(f"Turn {intervention['turn']} - {intervention['type']}", expanded=False):
            st.markdown(f"""
**Reason**: {intervention['reason']}
**Distance to PA**: {intervention['distance']:.3f if intervention['distance'] else 'N/A'}
            """)

# Auto-refresh option
st.markdown("---")
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("*💡 Tip: Keep this page open while using DEVOPS mode to monitor governance in real-time*")

with col2:
    if st.button("🔄 Refresh", use_container_width=True):
        st.rerun()
