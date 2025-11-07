#!/usr/bin/env python3
"""
LEAN TELOS HUD - Streamlined Dashboard for Steward PM
Focuses on essential metrics and governance without bloat.
"""

import sys
import os
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

# Import health monitor
try:
    from health_monitor import HealthMonitor
    HAS_HEALTH_MONITOR = True
except ImportError:
    HAS_HEALTH_MONITOR = False

# Import essential dashboard components
from components.real_project_analyzer import RealProjectAnalyzer
from components.strategic_view import StrategicView

# Import sync manager
try:
    from telos_sync import DashboardSyncManager
    HAS_SYNC = True
except ImportError:
    HAS_SYNC = False


def apply_custom_styling():
    """Apply lean dark theme with gold accents."""
    st.markdown("""
    <style>
    /* Lean dark theme - minimal and functional */

    /* Hide Streamlit defaults */
    header {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Main app background with light text */
    .stApp {
        background-color: #1a1a1a !important;
        color: #f0f0f0 !important;
    }

    /* All text should be light */
    p, span, label, div {
        color: #e8e8e8 !important;
    }

    /* Sidebar background with light text */
    [data-testid="stSidebar"] {
        background-color: #2a2a2a !important;
        color: #f0f0f0 !important;
    }

    /* Gold border buttons with bright text */
    .stButton > button {
        background-color: #2d2d2d !important;
        color: #ffffff !important;
        border: 1px solid #FFD700 !important;
        transition: all 0.3s ease !important;
    }

    .stButton > button:hover {
        background-color: #3d3d3d !important;
        color: #FFD700 !important;
        box-shadow: 0 0 6px rgba(255, 215, 0, 0.5) !important;
    }

    /* Headers with gold accent */
    h1, h2, h3 {
        color: #FFD700 !important;
    }

    /* Metric cards styling with light text */
    [data-testid="metric-container"] {
        background-color: #2d2d2d !important;
        border: 1px solid #FFD700 !important;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }

    /* Input fields with light text */
    .stTextInput input,
    .stTextArea textarea,
    .stSelectbox select {
        background-color: #2d2d2d !important;
        color: #f0f0f0 !important;
        border: 1px solid #666 !important;
    }

    /* Tabs styling with light text */
    .stTabs [data-baseweb="tab"] {
        color: #f5f5f5 !important;
        background-color: #2d2d2d !important;
        border: 1px solid #FFD700 !important;
    }

    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #3d3d3d !important;
        color: #FFD700 !important;
    }

    /* Progress bars */
    .stProgress > div > div {
        background-color: #FFD700 !important;
    }
    </style>
    """, unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables."""
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = 'real_analysis'

    if 'health_monitor' not in st.session_state:
        if HAS_HEALTH_MONITOR:
            st.session_state.health_monitor = HealthMonitor()

    if 'command_history' not in st.session_state:
        st.session_state.command_history = []

    # Initialize sync manager and perform batch sync on startup
    if 'sync_manager' not in st.session_state and HAS_SYNC:
        st.session_state.sync_manager = DashboardSyncManager()

        # Show sync status in sidebar
        with st.sidebar:
            with st.spinner("🔄 Syncing with Steward PM..."):
                sync_result = st.session_state.sync_manager.initialize_dashboard()

            if sync_result['updates'] > 0:
                st.success(f"✅ Synced {sync_result['updates']} updates")

    # Set synced data flag
    if 'data_synced' not in st.session_state:
        st.session_state.data_synced = True if HAS_SYNC else False


def main():
    """Main application entry point."""
    # Page configuration
    st.set_page_config(
        page_title="TELOS HUD",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Apply custom styling
    apply_custom_styling()

    # Initialize session state
    initialize_session_state()

    # Header
    st.markdown("""
        <h1 style='text-align: center; color: #FFD700;'>
            👁️ TELOS OBSERVATION HUD
        </h1>
        <p style='text-align: center; color: #888;'>
            Pure observation dashboard - Connected to Steward PM
        </p>
        <hr style='border: 1px solid #FFD700;'>
    """, unsafe_allow_html=True)

    # Sidebar navigation - LEAN AND ESSENTIAL ONLY
    with st.sidebar:
        st.markdown("### 🧭 Navigation")

        # Essential views only
        if st.button("🔬 Project Analysis", use_container_width=True):
            st.session_state.active_tab = 'real_analysis'

        if st.button("🎯 Strategic Overview", use_container_width=True):
            st.session_state.active_tab = 'strategic'

        if st.button("🏥 Health Monitor", use_container_width=True):
            st.session_state.active_tab = 'health'

        st.markdown("---")

        # Quick actions
        st.markdown("### ⚡ Quick Actions")

        if st.button("🔄 Refresh Data", use_container_width=True):
            st.rerun()

        if HAS_SYNC:
            if st.button("📥 Force Sync", use_container_width=True):
                sync_result = st.session_state.sync_manager.sync.sync_updates()
                if sync_result['updates'] > 0:
                    st.success(f"Synced {sync_result['updates']} updates")
                    st.rerun()
                else:
                    st.info("Already up to date")

        st.markdown("---")

        # Sync status
        if HAS_SYNC:
            st.markdown("### 📊 Sync Status")
            if st.session_state.data_synced:
                st.success("✅ Connected to Steward")

                # Show last sync time
                sync_state = st.session_state.sync_manager.sync.sync_state
                if sync_state.get('last_sync_time'):
                    st.caption(f"Last sync: {sync_state['last_sync_time'][:19]}")
            else:
                st.warning("⚠️ Not synced")

        # System health indicator
        if HAS_HEALTH_MONITOR:
            st.markdown("---")
            st.markdown("### 🏥 System Health")
            health_summary = st.session_state.health_monitor.get_health_summary()

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Score", f"{health_summary['health_score']}%")
            with col2:
                status_color = {
                    'excellent': '🟢',
                    'good': '🟡',
                    'degraded': '🟠',
                    'critical': '🔴'
                }.get(health_summary['status'], '⚪')
                st.markdown(f"### {status_color} {health_summary['status'].upper()}")

    # Main content area - ONLY ESSENTIAL VIEWS
    if st.session_state.active_tab == 'real_analysis':
        show_real_analysis()

    elif st.session_state.active_tab == 'strategic':
        show_strategic_overview()

    elif st.session_state.active_tab == 'health':
        show_health_monitor()


def show_real_analysis():
    """Display REAL project analysis with actual metrics."""
    analyzer = RealProjectAnalyzer()
    analyzer.render()


def show_strategic_overview():
    """Display strategic overview from Steward PM data."""
    view = StrategicView()
    view.render()


def show_health_monitor():
    """Display health monitoring dashboard."""
    st.markdown("## 🏥 System Health Monitor")

    if HAS_HEALTH_MONITOR:
        health_summary = st.session_state.health_monitor.get_health_summary()

        # Health score gauge
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Health Score",
                f"{health_summary['health_score']}%",
                f"{health_summary['status']}"
            )

        with col2:
            if 'system_metrics' in health_summary:
                metrics = health_summary['system_metrics']
                st.metric("CPU Usage", f"{metrics['cpu']['percent']}%")

        with col3:
            if 'system_metrics' in health_summary:
                metrics = health_summary['system_metrics']
                st.metric("Memory Usage", f"{metrics['memory']['percent']}%")

        st.markdown("---")

        # Detailed metrics
        if 'system_metrics' in health_summary and 'error' not in health_summary['system_metrics']:
            metrics = health_summary['system_metrics']

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 📊 System Resources")
                st.info(f"""
                **CPU:** {metrics['cpu']['percent']}% ({metrics['cpu']['count']} cores)
                **Memory:** {metrics['memory']['available_gb']:.1f}/{metrics['memory']['total_gb']:.1f} GB available
                **Disk:** {metrics['disk']['free_gb']:.1f} GB free ({metrics['disk']['percent']}% used)
                **Process Memory:** {metrics['memory']['process_mb']:.1f} MB
                """)

            with col2:
                st.markdown("### 🌐 Network")
                st.info(f"""
                **Sent:** {metrics['network']['sent_mb']:.2f} MB
                **Received:** {metrics['network']['received_mb']:.2f} MB
                **Platform:** {metrics['system']['platform']}
                **Python:** {metrics['system']['python_version']}
                """)

        # API Status
        if 'api_status' in health_summary:
            st.markdown("### 🌐 API Status")
            for api, status in health_summary['api_status'].items():
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown(f"**{api}**")

                with col2:
                    if status.get('available'):
                        st.success(f"✅ {status['status']}")
                    else:
                        st.error(f"❌ {status['status']}")

                with col3:
                    if 'response_time' in status:
                        st.metric("Response", f"{status['response_time']}s")

        # Refresh button
        if st.button("🔄 Refresh Health Data"):
            st.session_state.health_monitor = HealthMonitor()
            st.rerun()

    else:
        st.error("❌ Health monitor not available")
        st.info("Install psutil: pip install psutil")


if __name__ == "__main__":
    main()