"""
TELOSCOPE v2 - Foundation Test Harness

Minimal test entry point for validating v2 foundation components.

Purpose:
- Test import validation
- Render foundation components
- Generate and display mock data
- Validate state management

Run:
    cd ~/Desktop/TELOS
    ./venv/bin/streamlit run telos_observatory/main_observatory_v2.py

Components Tested:
- teloscope_state (centralized state management)
- mock_data (enhanced session generator)
- turn_indicator (turn display component)
- marker_generator (timeline markers)
- scroll_controller (dimming algorithm)
"""

import streamlit as st
from datetime import datetime

# Import v2 foundation components
from teloscope_v2.state.teloscope_state import (
    init_teloscope_state,
    get_teloscope_state,
    get_current_turn,
    set_current_turn,
    get_state_summary,
)
from teloscope_v2.utils.mock_data import (
    generate_enhanced_session,
    generate_test_suite,
    session_to_transcript,
)
from teloscope_v2.components.turn_indicator import (
    render_turn_indicator,
    render_turn_indicator_inline,
    render_turn_progress_bar,
)
from teloscope_v2.utils.marker_generator import (
    generate_timeline_markers,
    generate_timeline_legend,
    generate_annotated_markers,
)
from teloscope_v2.utils.scroll_controller import (
    calculate_turn_opacity,
    get_turn_border_style,
)

# Import Phase 1.5B counterfactual components
try:
    from teloscope_v2.utils.comparison_viewer_v2 import ComparisonViewerV2
    from teloscope_v2.utils.evidence_exporter import EvidenceExporter
    from teloscope_v2.utils.baseline_adapter import BaselineAdapter
    from teloscope_v2.utils.comparison_adapter import ComparisonAdapter
    COUNTERFACTUAL_AVAILABLE = True
except ImportError as e:
    COUNTERFACTUAL_AVAILABLE = False
    print(f"Warning: Counterfactual components not available: {e}")

# Import real API components for Phase 2
try:
    import sys
    from pathlib import Path
    telos_root = Path(__file__).parent.parent / 'telos_purpose'
    if str(telos_root) not in sys.path:
        sys.path.insert(0, str(telos_root))

    from telos_purpose.llm_clients.mistral_client import MistralClient
    from telos_purpose.core.embedding_provider import SentenceTransformerProvider
    from telos_purpose.core.unified_steward import PrimacyAttractor
    REAL_API_AVAILABLE = True
except ImportError as e:
    REAL_API_AVAILABLE = False
    print(f"Warning: Real API components not available: {e}")


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

def init_session_state():
    """Initialize session state for v2 testing."""
    if 'v2_initialized' not in st.session_state:
        st.session_state.v2_initialized = True

        # Generate test session
        session = generate_enhanced_session(
            turns=12,
            session_type='stable',
            include_annotations=True,
            seed=42,  # Reproducible
        )

        # Store in flat state (shared between Phase 1 and v2)
        st.session_state.session_data = session

        # Initialize teloscope v2 state
        init_teloscope_state(session)


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="TELOSCOPE v2 - Foundation Test",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    # Initialize state
    init_session_state()

    # Get session data
    session = st.session_state.session_data
    turns = session['turns']
    total_turns = len(turns)

    # ===== HEADER =====
    st.title("🔭 TELOSCOPE v2 + Phase 1.5B Test")
    st.caption(f"Version 2.0.0-spec | Testing 9 Components (6 Foundation + 3 Counterfactual)")

    st.markdown("---")

    # ===== INFO PANEL =====
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Session Type", session.get('session_type', 'stable').title())

    with col2:
        st.metric("Total Turns", total_turns)

    with col3:
        avg_fidelity = session['metadata'].get('avg_fidelity', 0.0)
        st.metric("Avg Fidelity", f"{avg_fidelity:.3f}")

    with col4:
        interventions = session['metadata'].get('interventions', 0)
        st.metric("Interventions", interventions)

    st.markdown("---")

    # ===== COMPONENT TESTS =====

    # Test 1: Turn Indicator (Compact)
    st.subheader("1️⃣ Turn Indicator (Compact)")
    st.caption("Component: `components/turn_indicator.py`")
    render_turn_indicator(total_turns=total_turns, show_jump_controls=True, compact=True)

    st.markdown("---")

    # Test 2: Turn Indicator (Inline)
    st.subheader("2️⃣ Turn Indicator (Inline)")
    st.caption("Component: `components/turn_indicator.py`")
    render_turn_indicator_inline(total_turns=total_turns)

    st.markdown("---")

    # Test 3: Turn Indicator (Progress Bar)
    st.subheader("3️⃣ Turn Indicator (Progress Bar)")
    st.caption("Component: `components/turn_indicator.py`")
    render_turn_progress_bar(total_turns=total_turns)

    st.markdown("---")

    # Test 4: Timeline Markers (Standard)
    st.subheader("4️⃣ Timeline Markers (Standard Style)")
    st.caption("Component: `utils/marker_generator.py`")

    current_turn = get_current_turn()
    markers_html = generate_timeline_markers(
        turns=turns,
        current_turn=current_turn,
        style='standard',
        show_tooltips=True,
    )

    # Debug info
    st.write(f"**Debug**: Generated {len(markers_html)} characters of HTML")
    st.write(f"**Current turn**: {current_turn} / {len(turns)}")

    st.markdown(markers_html, unsafe_allow_html=True)

    st.markdown("---")

    # Test 5: Timeline Markers (Enhanced)
    st.subheader("5️⃣ Timeline Markers (Enhanced Style)")
    st.caption("Component: `utils/marker_generator.py`")

    markers_enhanced = generate_timeline_markers(
        turns=turns,
        current_turn=current_turn,
        style='enhanced',
        show_tooltips=True,
    )
    st.markdown(markers_enhanced, unsafe_allow_html=True)

    st.markdown("---")

    # Test 6: Annotated Markers
    st.subheader("6️⃣ Annotated Timeline Markers")
    st.caption("Component: `utils/marker_generator.py`")

    markers_annotated = generate_annotated_markers(
        turns=turns,
        current_turn=current_turn,
        style='enhanced',
    )
    st.markdown(markers_annotated, unsafe_allow_html=True)

    st.markdown("---")

    # Test 7: Timeline Legend
    st.subheader("7️⃣ Timeline Legend")
    st.caption("Component: `utils/marker_generator.py`")

    legend_html = generate_timeline_legend(compact=True)
    st.markdown(legend_html, unsafe_allow_html=True)

    # Debug: Show if legend is empty
    if not legend_html.strip():
        st.warning("Legend HTML is empty")

    st.markdown("---")

    # Test 8: Turn Display with Dimming
    st.subheader("8️⃣ Dimming Algorithm Test")
    st.caption("Component: `utils/scroll_controller.py`")

    st.write("**Active turn dimming visualization:**")

    # Display all turns with dimming
    for idx, turn in enumerate(turns):
        opacity = calculate_turn_opacity(idx, current_turn)
        border = get_turn_border_style(idx, current_turn)

        turn_num = turn.get('turn', idx + 1)
        status = turn.get('status_label', turn.get('status', '✓'))
        fidelity = turn.get('fidelity')

        # Create container with dimming and border
        turn_html = f"""
        <div style="
            opacity: {opacity};
            border: {border};
            padding: 12px;
            margin: 8px 0;
            background: rgba(255, 255, 255, 0.03);
            border-radius: 8px;
            transition: all 0.3s ease;
        ">
            <strong>Turn {turn_num}</strong> - {status}
            {f'| Fidelity: {fidelity:.3f}' if fidelity is not None else ''}
        </div>
        """
        st.markdown(turn_html, unsafe_allow_html=True)

    st.markdown("---")

    # Test 9: Counterfactual Analysis (Phase 1.5B)
    if COUNTERFACTUAL_AVAILABLE:
        st.subheader("9️⃣ Counterfactual Analysis Demo")
        st.caption("Phase 1.5B: REAL Baseline Comparison | Runtime Validation")

        st.markdown("""
        **Purpose**: Run ACTUAL TELOS vs Baseline comparison - THE core feature for V1.00 validation.

        This test executes:
        - **REAL baseline_adapter.run_comparison()** (not mock data)
        - Actual timing from timestamp deltas
        - Real runtime validation tests
        - Evidence export with validation proof
        """)

        # Run ACTUAL baseline comparison
        with st.expander("📊 Run REAL Baseline Comparison", expanded=False):
            st.markdown("### 🔬 Live Baseline Comparison Execution")

            # API Selection
            use_real_api = st.checkbox(
                "🌐 Use Real Mistral API (Phase 2)",
                value=False,
                help="Enable to use real Mistral API calls for actual governance validation. Disabling uses mock LLM for free testing."
            )

            if use_real_api and not REAL_API_AVAILABLE:
                st.error("❌ Real API components not available. Check imports.")
                use_real_api = False

            if use_real_api:
                st.info("ℹ️ **Phase 2 Mode**: Using REAL Mistral API - this will consume API credits (~6 calls for 3-turn comparison)")

                # API Key Input (password field - hidden on screen)
                api_key_input = st.text_input(
                    "🔑 Mistral API Key",
                    type="password",
                    placeholder="Enter your Mistral API key (or set MISTRAL_API_KEY env var)",
                    help="Your API key is only stored in this session and never saved to disk. You can also set the MISTRAL_API_KEY environment variable instead."
                )
            else:
                st.info("ℹ️ **Phase 1 Mode**: Using mock LLM - free testing, validates wiring only")
                api_key_input = None

            # Create mock LLM and embeddings for testing
            import numpy as np

            class MockLLMClient:
                """Simple mock LLM for testing"""
                def generate(self, messages, **kwargs):
                    # Generate context-aware responses
                    if len(messages) <= 2:
                        return "I'd be happy to help you improve your Python coding skills. Let's start with fundamental principles and build a structured learning path tailored to your current level and goals."
                    elif len(messages) <= 4:
                        return "To effectively learn Python, I recommend: 1) Master core syntax and data structures first, 2) Practice with small projects aligned with your interests, 3) Learn debugging techniques early. What's your current experience level?"
                    else:
                        return "Let's focus on deliberate practice. I can guide you through progressively challenging exercises that reinforce concepts while staying within your learning zone. Would you like to start with a specific Python topic or project idea?"

                def chat_completion(self, messages, **kwargs):
                    """Alternative interface for interventions - same as generate()"""
                    return self.generate(messages, **kwargs)

            class MockEmbeddingProvider:
                """Simple mock embedding provider for testing"""
                def encode(self, text):
                    # Return simple random embedding
                    np.random.seed(len(text))  # Deterministic based on text length
                    return np.random.rand(384).astype(np.float32)

            class MockAttractorConfig:
                def __init__(self):
                    self.embedding_dim = 384
                    self.basin_threshold = 0.3
                    self.purpose = ["Help users learn and improve their skills"]
                    self.scope = ["Educational guidance and skill development"]
                    self.boundaries = ["Provide accurate information", "Respect user autonomy"]
                    self.privacy_level = 0.7
                    self.constraint_tolerance = 0.3
                    self.task_priority = 0.8

            # Create test conversation
            test_conversation = [
                ("How can I improve my Python coding skills?", ""),
                ("What's the best way to learn advanced Python concepts?", ""),
                ("Can you help me understand Python decorators?", "")
            ]

            # Initialize components based on mode selection
            if use_real_api:
                st.info("⚙️ Initializing baseline adapter with REAL Mistral API and embeddings...")
            else:
                st.info("⚙️ Initializing baseline adapter with mock LLM and embeddings...")

            try:
                if use_real_api:
                    # Phase 2: Real API components
                    import os

                    # Check for API key (prioritize input field, fall back to env var)
                    api_key = api_key_input.strip() if api_key_input else os.getenv("MISTRAL_API_KEY")

                    if not api_key:
                        st.error("❌ API key required! Enter it in the field above or set MISTRAL_API_KEY environment variable.")
                        st.stop()  # Stop execution gracefully

                    # Initialize real components
                    llm = MistralClient(api_key=api_key, model="mistral-small-latest")
                    embeddings = SentenceTransformerProvider(model_name="sentence-transformers/all-MiniLM-L6-v2")

                    # Real attractor configuration
                    attractor = PrimacyAttractor(
                        purpose=["Help users learn and improve their skills", "Provide accurate educational guidance"],
                        scope=["Educational content", "Programming tutorials", "Skill development"],
                        boundaries=["Provide accurate information", "Respect user autonomy", "Stay within educational scope"],
                        privacy_level=0.7,
                        constraint_tolerance=0.3,
                        task_priority=0.8
                    )

                    st.success("✅ Real API components initialized")
                else:
                    # Phase 1: Mock components
                    llm = MockLLMClient()
                    embeddings = MockEmbeddingProvider()
                    attractor = MockAttractorConfig()

                    st.success("✅ Mock components initialized")

                adapter = BaselineAdapter(llm, embeddings, attractor)

                st.info("🚀 Running comparison: Stateless vs Prompt-Only baselines...")

                # Run ACTUAL comparison with timing and calibration tracking
                import time
                start = time.time()

                results = adapter.run_comparison(
                    test_conversation,
                    baseline_type='stateless',  # Null hypothesis
                    track_timing=True,
                    track_calibration=True
                )

                elapsed = time.time() - start

                st.success(f"✅ Comparison complete in {elapsed:.2f}s")

                # Convert to branch format for comparison
                comp_adapter = ComparisonAdapter()
                baseline_branch = comp_adapter.convert_baseline_result_to_branch(results['baseline'])
                telos_branch = comp_adapter.convert_baseline_result_to_branch(results['telos'])

                st.info("📊 Generating comparison analysis...")
                comparison = comp_adapter.compare_results(baseline_branch, telos_branch)

                # Add turn_results to comparison for runtime validation
                if 'turn_results' in telos_branch:
                    comparison['telos']['turn_results'] = telos_branch['turn_results']
                if 'turn_results' in baseline_branch:
                    comparison['baseline']['turn_results'] = baseline_branch['turn_results']

                st.success("✅ Analysis complete")

            except Exception as e:
                st.error(f"❌ Error during comparison: {str(e)}")
                st.exception(e)
                comparison = None
                baseline_branch = None
                telos_branch = None

            # Render ACTUAL results
            if comparison is not None and baseline_branch is not None and telos_branch is not None:
                st.markdown("---")
                st.markdown("### 📊 Real Comparison Results")

                # Render comparison using viewer
                viewer = ComparisonViewerV2()

                st.markdown("#### Summary")
                viewer.render_summary(comparison, show_chart=False)

                st.markdown("---")
                st.markdown("#### Turn-by-Turn Comparison")
                viewer.render_all_turns(baseline_branch, telos_branch, expanded_first=True)

                st.markdown("---")
                st.markdown("#### Evidence Export")

                col1, col2 = st.columns(2)

                with col1:
                    exporter = EvidenceExporter()
                    json_data = exporter.create_download_data(comparison, format='json')
                    st.download_button(
                        "📄 Download JSON Evidence",
                        data=json_data,
                        file_name=f"evidence_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        use_container_width=True
                    )

                with col2:
                    md_data = exporter.create_download_data(comparison, format='markdown')
                    st.download_button(
                        "📝 Download Markdown Report",
                        data=md_data,
                        file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown",
                        use_container_width=True
                    )

                st.success("✅ Real baseline comparison complete! All infrastructure verified.")
            else:
                st.warning("⚠️ Comparison failed. Check error messages above.")

    else:
        st.subheader("9️⃣ Counterfactual Analysis")
        st.warning("⚠️ Counterfactual components not available. Check imports.")

    st.markdown("---")

    # ===== SIDEBAR: ACTIONS =====
    with st.sidebar:
        st.header("🎛️ Controls")

        st.subheader("Actions")

        if st.button("🔄 Reset State", use_container_width=True):
            if 'v2_initialized' in st.session_state:
                del st.session_state.v2_initialized
            st.rerun()

        if st.button("🎲 Generate New Session", use_container_width=True):
            session_types = ['stable', 'high-drift', 'intervention-heavy']
            import random
            new_type = random.choice(session_types)

            new_session = generate_enhanced_session(
                turns=12,
                session_type=new_type,
                include_annotations=True,
            )

            st.session_state.session_data = new_session
            init_teloscope_state(new_session)
            st.rerun()

        st.markdown("---")

        st.subheader("Export")

        transcript = session_to_transcript(session)
        st.download_button(
            label="📄 Download Transcript",
            data=transcript,
            file_name=f"teloscope_v2_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True,
        )

        st.markdown("---")

        st.caption("TELOSCOPE v2 Foundation Test")
        st.caption(f"Session: {session.get('session_id', 'N/A')}")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
