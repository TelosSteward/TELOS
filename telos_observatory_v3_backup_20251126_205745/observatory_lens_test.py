#!/usr/bin/env python3
"""
Observatory Lens - Standalone Test Page
Direct access to Observatory Lens without beta consent or other UI elements.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from telos_observatory_v3.core.state_manager import StateManager
from telos_observatory_v3.components.observatory_lens import ObservatoryLens


def generate_test_data():
    """Generate test session data for Observatory Lens."""
    import random

    turns = []
    for i in range(15):
        # Simulate varying fidelity scores
        if i == 5:  # Intervention at turn 5
            fidelity = 0.65
            intervention = True
        elif i == 12:  # Another intervention
            fidelity = 0.68
            intervention = True
        else:
            fidelity = random.uniform(0.82, 0.96)
            intervention = False

        turn = {
            'turn': i,
            'timestamp': i * 2.5,
            'user_input': f"Test message {i + 1}",
            'response': f"Test response {i + 1} with some content about TELOS governance.",
            'fidelity': fidelity,
            'distance': 1.0 - fidelity,
            'threshold': 0.7,
            'intervention_applied': intervention,
            'drift_detected': fidelity < 0.8,
            'status': "⚠" if intervention else "✓",
            'status_text': "Intervention" if intervention else "Good",
            'in_basin': fidelity >= 0.7,
            'phase2_comparison': None
        }
        turns.append(turn)

    avg_fid = sum(t['fidelity'] for t in turns) / len(turns) if turns else 0.0

    return {
        'session_id': 'test_session_observatory_lens',
        'turns': turns,
        'total_turns': len(turns),
        'current_turn': len(turns) - 1,
        'avg_fidelity': avg_fid,
        'total_interventions': sum(1 for t in turns if t['intervention_applied']),
        'drift_warnings': sum(1 for t in turns if t['drift_detected']),
        'statistics': {
            'avg_fidelity': avg_fid,
            'interventions': sum(1 for t in turns if t['intervention_applied']),
            'drift_warnings': sum(1 for t in turns if t['drift_detected'])
        }
    }


def main():
    """Main test page."""
    st.set_page_config(
        page_title="Observatory Lens - Test Mode",
        page_icon="🔭",
        layout="wide"
    )

    # Dark theme styling with animations
    st.markdown("""
    <style>
    /* Force dark everywhere */
    .stApp {
        background-color: #1a1a1a !important;
    }

    .main {
        background-color: #1a1a1a !important;
    }

    .block-container {
        background-color: #1a1a1a !important;
    }

    /* Kill all white backgrounds */
    * {
        background-color: transparent !important;
    }

    /* Restore specific dark backgrounds */
    .stApp, .main, .block-container {
        background-color: #1a1a1a !important;
    }

    /* Text colors */
    p, span, div, h1, h2, h3, h4 {
        color: #e0e0e0 !important;
    }

    /* Buttons with smooth transitions */
    .stButton > button {
        background-color: #2d2d2d !important;
        color: #e0e0e0 !important;
        border: 1px solid #FFD700 !important;
        transition: all 0.3s ease !important;
    }

    .stButton > button:hover {
        background-color: #3d3d3d !important;
        border: 2px solid #FFD700 !important;
        box-shadow: 0 0 10px rgba(255, 215, 0, 0.5) !important;
        transform: translateY(-2px) !important;
    }

    /* Kill Streamlit's default white styling */
    [data-testid="stMarkdownContainer"] {
        background-color: transparent !important;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #2d2d2d !important;
        color: #e0e0e0 !important;
        transition: background-color 0.3s ease !important;
    }

    .streamlit-expanderHeader:hover {
        background-color: #3d3d3d !important;
    }

    .streamlit-expanderContent {
        background-color: #1a1a1a !important;
    }

    /* Smooth fade-in for visualizations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    [data-testid="stPlotlyChart"] {
        animation: fadeIn 0.5s ease-out;
    }
    </style>
    """, unsafe_allow_html=True)

    # Title
    st.markdown("""
    <div style="text-align: center; padding: 20px;">
        <h1 style="color: #FFD700;">🔭 Observatory Lens - Test Mode</h1>
        <p style="color: #888;">Standalone testing environment (no beta consent required)</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Initialize state manager with test data
    if 'state_manager' not in st.session_state:
        state_manager = StateManager()
        test_data = generate_test_data()
        state_manager.initialize(test_data)
        st.session_state.state_manager = state_manager

    state_manager = st.session_state.state_manager

    # Control panel
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        if st.button("🔄 Regenerate Test Data", use_container_width=True):
            state_manager._initialized = False
            test_data = generate_test_data()
            state_manager.initialize(test_data)
            st.rerun()

    with col2:
        lens_status = "OPEN" if state_manager.state.show_observatory_lens else "CLOSED"
        if st.button(f"🔭 Toggle Lens ({lens_status})", use_container_width=True):
            state_manager.toggle_component('observatory_lens')
            st.rerun()

    with col3:
        st.markdown(f"""
        <div style="
            background: #2d2d2d;
            border: 1px solid #FFD700;
            border-radius: 5px;
            padding: 10px;
            text-align: center;
        ">
            <strong style="color: #FFD700;">Test Data:</strong><br>
            <span style="color: #e0e0e0;">{state_manager.state.total_turns} turns loaded</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Always show lens open by default in test mode
    if 'lens_auto_opened' not in st.session_state:
        state_manager.state.show_observatory_lens = True
        st.session_state.lens_auto_opened = True

    # Render Observatory Lens
    observatory_lens = ObservatoryLens(state_manager)
    observatory_lens.render()

    # Debug info (collapsible)
    with st.expander("🔧 Debug Info"):
        st.json({
            'total_turns': state_manager.state.total_turns,
            'current_turn': state_manager.state.current_turn,
            'avg_fidelity': round(state_manager.state.avg_fidelity, 3),
            'total_interventions': state_manager.state.total_interventions,
            'drift_warnings': state_manager.state.drift_warnings,
            'lens_open': state_manager.state.show_observatory_lens
        })


if __name__ == "__main__":
    main()
