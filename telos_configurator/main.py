"""
TELOS Corpus Configurator MVP - Main Entry Point
=================================================

A comprehensive wizard-based interface for configuring and deploying
three-tier TELOS governance frameworks.

Workflow:
1. Domain Selection - Choose pre-configured template or custom
2. Corpus Upload & Management - Add and embed policy documents
3. PA Configuration - Define Primacy Attractor
4. Threshold Calibration - Set tier boundaries
5. Activation - Activate governance engine
6. Dashboard & Testing - Monitor and test governance

Run:
    streamlit run main.py --server.port 8502

Author: TELOS AI Labs Inc.
Contact: contact@telos-labs.ai
Date: 2026-01-23
"""

import sys
from datetime import datetime

import streamlit as st

# Add to Python path
sys.path.insert(0, '/Users/brunnerjf/Desktop/TELOS_Master/telos_configurator')

# Import configuration and styling
from config.styles import (
    inject_custom_css,
    render_section_header,
    get_glassmorphism_css,
    GOLD, BG_ELEVATED, BG_SURFACE, TEXT_PRIMARY, TEXT_SECONDARY,
    STATUS_GOOD, STATE_PENDING, STATUS_SEVERE, with_opacity
)

# Import state management
from state_manager import (
    initialize_state,
    get_current_step,
    get_selected_domain,
    get_pa_instance,
    get_corpus_engine,
    get_governance_engine,
    is_governance_active,
    get_corpus_stats,
    get_step_status,
    navigate_to_step,
    next_step,
    previous_step,
    save_configuration,
    load_configuration,
    reset_all_state
)

# Import components
from components import (
    render_domain_selector,
    render_corpus_uploader,
    render_corpus_manager,
    render_pa_configurator,
    render_threshold_config,
    render_activation_panel,
    render_dashboard_metrics,
    render_test_query_interface,
    render_audit_panel,
    render_corpus_browser,
    render_retrieval_panel,
    RetrievedChunk
)

# Import engines
from engine.corpus_engine import CorpusEngine
from engine.governance_engine import GovernanceEngine, ThresholdConfig


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="TELOS Corpus Configurator",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================================
# STEP DEFINITIONS
# ============================================================================

STEPS = [
    {
        'id': 0,
        'name': 'Domain Selection',
        'description': 'Choose a pre-configured domain or create custom configuration',
        'icon': '🎯'
    },
    {
        'id': 1,
        'name': 'Corpus Upload',
        'description': 'Upload and manage policy documents',
        'icon': '📚'
    },
    {
        'id': 2,
        'name': 'PA Configuration',
        'description': 'Define the Primacy Attractor',
        'icon': '⚡'
    },
    {
        'id': 3,
        'name': 'Threshold Calibration',
        'description': 'Configure three-tier thresholds',
        'icon': '🎚️'
    },
    {
        'id': 4,
        'name': 'Activation',
        'description': 'Activate the governance engine',
        'icon': '🚀'
    },
    {
        'id': 5,
        'name': 'Dashboard & Testing',
        'description': 'Monitor metrics and test queries',
        'icon': '📊'
    }
]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_step_color(status: str) -> str:
    """Get color for step status."""
    if status == 'completed':
        return STATUS_GOOD
    elif status == 'active':
        return GOLD
    elif status == 'ready':
        return STATE_PENDING
    else:
        return TEXT_SECONDARY


def render_sidebar():
    """Render sidebar with branding, navigation, and quick stats."""
    with st.sidebar:
        # TELOS Branding Header
        st.markdown(f'''
        <div style="{get_glassmorphism_css(GOLD)}; padding: 1.5rem; margin-bottom: 1.5rem; text-align: center;">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">🔮</div>
            <div style="color: {GOLD}; font-size: 1.5rem; font-weight: 700; text-shadow: 0 0 20px {with_opacity(GOLD, 0.3)};">
                TELOS
            </div>
            <div style="color: {TEXT_SECONDARY}; font-size: 0.9rem; margin-top: 0.25rem;">
                Corpus Configurator
            </div>
            <div style="color: {TEXT_SECONDARY}; font-size: 0.75rem; margin-top: 0.5rem; opacity: 0.7;">
                v1.0.0 MVP
            </div>
        </div>
        ''', unsafe_allow_html=True)

        # Step Navigation
        st.markdown(f'''
        <div style="{get_glassmorphism_css()}; padding: 1rem; margin-bottom: 1rem;">
            <h4 style="color: {GOLD}; margin-bottom: 1rem; font-size: 1rem;">Configuration Steps</h4>
        </div>
        ''', unsafe_allow_html=True)

        current_step = get_current_step()

        for step in STEPS:
            step_status = get_step_status(step['id'])
            step_color = get_step_color(step_status)
            is_current = step['id'] == current_step

            # Step button
            border_width = '3px' if is_current else '1px'
            background = with_opacity(step_color, 0.15) if is_current else 'transparent'

            button_html = f'''
            <div style="
                {get_glassmorphism_css(step_color)};
                padding: 0.75rem;
                margin-bottom: 0.5rem;
                cursor: pointer;
                border-width: {border_width};
                background: {background};
            ">
                <div style="display: flex; align-items: center; gap: 0.75rem;">
                    <div style="font-size: 1.2rem;">{step['icon']}</div>
                    <div style="flex: 1;">
                        <div style="color: {step_color}; font-weight: {'700' if is_current else '600'}; font-size: 0.9rem;">
                            {step['name']}
                        </div>
                        <div style="color: {TEXT_SECONDARY}; font-size: 0.75rem; margin-top: 0.15rem;">
                            {step['description']}
                        </div>
                    </div>
                </div>
            </div>
            '''

            st.markdown(button_html, unsafe_allow_html=True)

            # Make clickable
            if st.button(
                f"Go to {step['name']}",
                key=f"nav_step_{step['id']}",
                use_container_width=True,
                type='primary' if is_current else 'secondary'
            ):
                navigate_to_step(step['id'])
                st.rerun()

        # Quick Stats
        st.markdown("---")

        st.markdown(f'''
        <div style="{get_glassmorphism_css()}; padding: 1rem;">
            <h4 style="color: {GOLD}; margin-bottom: 0.75rem; font-size: 0.95rem;">Quick Stats</h4>
        </div>
        ''', unsafe_allow_html=True)

        # Get stats
        corpus_stats = get_corpus_stats()
        pa = get_pa_instance()
        governance_active = is_governance_active()

        # Documents
        st.markdown(f'''
        <div style="margin-bottom: 0.75rem;">
            <div style="color: {TEXT_SECONDARY}; font-size: 0.8rem;">Documents Loaded</div>
            <div style="color: {GOLD if corpus_stats['total_documents'] > 0 else TEXT_SECONDARY}; font-size: 1.3rem; font-weight: 700;">
                {corpus_stats['total_documents']}
            </div>
        </div>
        ''', unsafe_allow_html=True)

        # Embeddings
        st.markdown(f'''
        <div style="margin-bottom: 0.75rem;">
            <div style="color: {TEXT_SECONDARY}; font-size: 0.8rem;">Embedded</div>
            <div style="color: {STATUS_GOOD if corpus_stats['embedded_documents'] > 0 else TEXT_SECONDARY}; font-size: 1.3rem; font-weight: 700;">
                {corpus_stats['embedded_documents']} ({corpus_stats['embedding_percentage']:.0f}%)
            </div>
        </div>
        ''', unsafe_allow_html=True)

        # PA Status
        pa_status = '✓ Configured' if pa else '○ Not Configured'
        pa_color = STATUS_GOOD if pa else TEXT_SECONDARY

        st.markdown(f'''
        <div style="margin-bottom: 0.75rem;">
            <div style="color: {TEXT_SECONDARY}; font-size: 0.8rem;">PA Status</div>
            <div style="color: {pa_color}; font-size: 1rem; font-weight: 600;">
                {pa_status}
            </div>
        </div>
        ''', unsafe_allow_html=True)

        # Governance Status
        gov_status = 'ACTIVE' if governance_active else 'INACTIVE'
        gov_color = STATUS_GOOD if governance_active else TEXT_SECONDARY

        st.markdown(f'''
        <div>
            <div style="color: {TEXT_SECONDARY}; font-size: 0.8rem;">Governance</div>
            <div style="color: {gov_color}; font-size: 1rem; font-weight: 700;">
                {gov_status}
            </div>
        </div>
        ''', unsafe_allow_html=True)

        # Configuration Management
        st.markdown("---")

        st.markdown(f'''
        <div style="{get_glassmorphism_css()}; padding: 1rem; margin-bottom: 0.5rem;">
            <h4 style="color: {GOLD}; margin-bottom: 0.5rem; font-size: 0.95rem;">Configuration</h4>
        </div>
        ''', unsafe_allow_html=True)

        # Save Configuration
        if st.button("Save Configuration", use_container_width=True):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"config_export_{timestamp}.json"
            if save_configuration(filepath):
                st.success(f"Saved to {filepath}")
            else:
                st.error("Failed to save configuration")

        # Load Configuration
        uploaded_config = st.file_uploader(
            "Load Configuration",
            type=['json'],
            help="Upload a previously saved configuration file",
            key="config_loader"
        )

        if uploaded_config is not None:
            if st.button("Load Config File", use_container_width=True):
                temp_path = f"temp_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(temp_path, 'wb') as f:
                    f.write(uploaded_config.getvalue())

                if load_configuration(temp_path):
                    st.success("Configuration loaded")
                    st.rerun()
                else:
                    st.error("Failed to load configuration")

        # Reset All
        if st.button("Reset All", type="secondary", use_container_width=True):
            if st.session_state.get('confirm_reset', False):
                reset_all_state()
                st.session_state.confirm_reset = False
                st.success("All state reset")
                st.rerun()
            else:
                st.session_state.confirm_reset = True
                st.warning("Click again to confirm reset")


def render_main_content():
    """Render main content area based on current step."""
    current_step = get_current_step()

    # Header
    step_info = STEPS[current_step]

    st.markdown(f'''
    <div style="{get_glassmorphism_css(GOLD)}; padding: 1.5rem; margin-bottom: 1.5rem;">
        <div style="display: flex; align-items: center; gap: 1rem;">
            <div style="font-size: 2.5rem;">{step_info['icon']}</div>
            <div style="flex: 1;">
                <div style="color: {GOLD}; font-size: 1.8rem; font-weight: 700;">
                    {step_info['name']}
                </div>
                <div style="color: {TEXT_SECONDARY}; font-size: 1rem; margin-top: 0.25rem;">
                    {step_info['description']}
                </div>
            </div>
            <div style="color: {TEXT_SECONDARY}; font-size: 0.9rem;">
                Step {current_step + 1} of {len(STEPS)}
            </div>
        </div>
    </div>
    ''', unsafe_allow_html=True)

    # Render step-specific content
    if current_step == 0:
        render_step_domain_selection()
    elif current_step == 1:
        render_step_corpus_upload()
    elif current_step == 2:
        render_step_pa_configuration()
    elif current_step == 3:
        render_step_threshold_calibration()
    elif current_step == 4:
        render_step_activation()
    elif current_step == 5:
        render_step_dashboard()

    # Navigation buttons
    render_navigation_buttons()


def render_step_domain_selection():
    """Render Step 0: Domain Selection."""
    render_domain_selector()


def render_step_corpus_upload():
    """Render Step 1: Corpus Upload & Management."""
    col1, col2 = st.columns([1, 1])

    with col1:
        render_corpus_uploader(get_corpus_engine())

    with col2:
        render_corpus_manager(get_corpus_engine())


def render_step_pa_configuration():
    """Render Step 2: PA Configuration."""
    render_pa_configurator()


def render_step_threshold_calibration():
    """Render Step 3: Threshold Calibration."""
    thresholds = render_threshold_config()
    # Store thresholds in session state for activation step
    st.session_state.thresholds = thresholds


def render_step_activation():
    """Render Step 4: Activation."""
    pa = get_pa_instance()
    corpus_engine = get_corpus_engine()
    thresholds = st.session_state.get('thresholds', ThresholdConfig())
    governance_engine = get_governance_engine()

    render_activation_panel(pa, corpus_engine, thresholds, governance_engine)


def render_step_dashboard():
    """Render Step 5: Dashboard & Testing."""
    governance_engine = get_governance_engine()
    corpus_engine = get_corpus_engine()

    # Create tabs for different dashboard views
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Metrics Dashboard",
        "🧪 Test Query Interface",
        "📚 Corpus Browser",
        "📋 Audit Log"
    ])

    with tab1:
        render_dashboard_metrics(governance_engine)

    with tab2:
        render_test_query_interface(governance_engine)

    with tab3:
        # Get corpus documents for browser
        corpus_docs = []

        # First check for healthcare demo corpus
        if st.session_state.get('healthcare_corpus'):
            corpus_docs = st.session_state.healthcare_corpus
        elif corpus_engine:
            # Get from corpus engine
            corpus_docs = corpus_engine.get_all_documents()

        # Get any retrieved chunks from last query
        retrieved_chunks = st.session_state.get('last_retrieved_chunks', [])

        if corpus_docs:
            render_corpus_browser(corpus_docs, retrieved_chunks)
        else:
            st.info("No corpus documents loaded. Load demo data or upload documents in Step 1.")

    with tab4:
        render_audit_panel(governance_engine)


def render_navigation_buttons():
    """Render navigation buttons at bottom of page."""
    st.markdown("---")

    current_step = get_current_step()

    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        if current_step > 0:
            if st.button("◀ Previous", use_container_width=True):
                previous_step()
                st.rerun()

    with col3:
        if current_step < len(STEPS) - 1:
            if st.button("Next ▶", type="primary", use_container_width=True):
                next_step()
                st.rerun()


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application entry point."""

    # Inject custom CSS
    inject_custom_css()

    # Initialize session state
    initialize_state()

    # Render sidebar
    render_sidebar()

    # Render main content
    render_main_content()

    # Footer
    st.markdown("---")
    st.markdown(f'''
    <div style="text-align: center; color: {TEXT_SECONDARY}; font-size: 0.85rem; padding: 1rem;">
        <div>TELOS Corpus Configurator MVP • TELOS AI Labs Inc.</div>
        <div style="margin-top: 0.25rem;">contact@telos-labs.ai</div>
    </div>
    ''', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
