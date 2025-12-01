"""
TELOSCOPE Observatory - Streamlit UI
=====================================

Complete web interface for counterfactual AI governance observation.

Provides:
- Live conversation monitoring with real-time metrics
- Session replay with timeline scrubber
- Counterfactual branch comparison (TELOSCOPE view)
- Aggregate analytics dashboard
"""

import streamlit as st
from telos_purpose.sessions.web_session import WebSessionManager
from telos_purpose.core.session_state import SessionStateManager
from telos_purpose.core.counterfactual_manager import CounterfactualBranchManager
from telos_purpose.sessions.live_interceptor import LiveInterceptor
from telos_purpose.validation.branch_comparator import BranchComparator
from telos_purpose.llm_clients.mistral_client import TelosMistralClient
from telos_purpose.core.embedding_provider import EmbeddingProvider
from telos_purpose.core.unified_steward import UnifiedGovernanceSteward, PrimacyAttractor
import os
import json
import pandas as pd

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# Page configuration
st.set_page_config(
    page_title="TELOSCOPE - AI Governance Observatory",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customization
st.markdown("""
<style>
.stMetric {
    background-color: #f0f2f6;
    padding: 10px;
    border-radius: 5px;
}

.stButton>button {
    width: 100%;
}

div[data-testid="stExpander"] {
    border: 1px solid #e0e0e0;
    border-radius: 5px;
}

.trigger-badge {
    background-color: #ff6b6b;
    color: white;
    padding: 5px 10px;
    border-radius: 3px;
    display: inline-block;
    margin: 2px;
}
</style>
""", unsafe_allow_html=True)


def initialize_teloscope():
    """Initialize TELOSCOPE components (only once per session)."""
    if 'teloscope_initialized' not in st.session_state:
        try:
            # Load config
            config_path = 'config.json'
            if os.path.exists(config_path):
                with open(config_path) as f:
                    config = json.load(f)
            else:
                config = {
                    'governance_profile': {
                        'purpose': [
                            "Provide accurate, helpful information about AI governance",
                            "Explain TELOS framework concepts clearly"
                        ],
                        'scope': [
                            "AI safety and alignment",
                            "Governance mechanisms",
                            "Technical implementation"
                        ],
                        'boundaries': [
                            "No medical advice",
                            "No financial advice",
                            "Stay focused on AI governance topics"
                        ]
                    }
                }

            # Create web session manager with st.session_state reference
            st.session_state.web_session = WebSessionManager(st.session_state)
            st.session_state.web_session.initialize_web_session()

            # Create session state manager
            st.session_state.session_manager = SessionStateManager(
                web_session_manager=st.session_state.web_session
            )

            # Initialize LLM and embeddings
            api_key = os.getenv('MISTRAL_API_KEY')
            if not api_key:
                st.error("⚠️ MISTRAL_API_KEY not found in environment. Please set it and restart.")
                st.stop()

            llm = TelosMistralClient(api_key=api_key)
            embeddings = EmbeddingProvider(deterministic=False)

            # Create attractor
            gov_profile = config.get('governance_profile', {})
            attractor = PrimacyAttractor(
                purpose=gov_profile.get('purpose', []),
                scope=gov_profile.get('scope', []),
                boundaries=gov_profile.get('boundaries', [])
            )

            # Create steward
            steward = UnifiedGovernanceSteward(
                attractor=attractor,
                llm_client=llm,
                embedding_provider=embeddings,
                enable_interventions=True
            )
            steward.start_session()

            # Create branch manager
            st.session_state.branch_manager = CounterfactualBranchManager(
                llm_client=llm,
                embedding_provider=embeddings,
                steward=steward,
                web_session_manager=st.session_state.web_session,
                branch_length=5
            )

            # Create comparator
            st.session_state.comparator = BranchComparator()

            # Create interceptor (wraps LLM)
            st.session_state.interceptor = LiveInterceptor(
                llm_client=llm,
                embedding_provider=embeddings,
                steward=steward,
                session_manager=st.session_state.session_manager,
                branch_manager=st.session_state.branch_manager,
                web_session_manager=st.session_state.web_session,
                drift_threshold=0.76,  # Goldilocks: Aligned threshold
                enable_counterfactuals=True
            )

            st.session_state.teloscope_initialized = True

        except Exception as e:
            st.error(f"❌ Initialization failed: {e}")
            st.stop()


def render_live_session():
    """Render live conversation interface."""
    st.title("🔴 Live Session")

    # Sidebar: Live Metrics
    with st.sidebar:
        st.subheader("📊 Live Metrics")
        metrics = st.session_state.interceptor.get_live_metrics()

        col1, col2 = st.columns(2)
        with col1:
            fidelity = metrics['current_fidelity']
            delta_color = "normal" if fidelity >= 0.76 else "inverse"  # Goldilocks: Aligned threshold
            st.metric(
                "Fidelity",
                f"{fidelity:.3f}",
                delta_color=delta_color
            )
        with col2:
            st.metric(
                "Distance",
                f"{metrics['current_distance']:.3f}"
            )

        basin_emoji = "✅" if metrics['basin_status'] else "❌"
        st.metric("Basin Status", f"{basin_emoji}")

        # Session controls
        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Reset", type="secondary", use_container_width=True):
                st.session_state.interceptor.reset_session()
                st.session_state.session_manager.clear_session()
                st.session_state.web_session.clear_web_session()
                st.rerun()

        with col2:
            if st.button("💾 Export", type="primary", use_container_width=True):
                session_json = st.session_state.web_session.export_session()
                session_id = st.session_state.current_session.get('session_id', 'unknown')
                st.download_button(
                    label="📥 Download JSON",
                    data=session_json,
                    file_name=f"teloscope_session_{session_id}.json",
                    mime="application/json",
                    use_container_width=True
                )

    # Main area: Chat interface
    st.subheader("💬 Conversation")

    # Display conversation history
    turns = st.session_state.current_session.get('turns', [])

    for turn in turns:
        with st.chat_message("user"):
            st.write(turn['user_input'])

        with st.chat_message("assistant"):
            st.write(turn['assistant_response'])

            # Show metrics badge
            metrics = turn.get('metrics', {})
            fidelity = metrics.get('telic_fidelity', 1.0)

            if fidelity < 0.76:  # Goldilocks: Aligned threshold
                st.warning(f"⚠️ Drift detected (F={fidelity:.3f})")

    # Display triggers as badges
    triggers = st.session_state.web_session.get_all_triggers()
    if triggers:
        st.divider()
        st.info(f"🔬 {len(triggers)} counterfactual experiment(s) triggered")

        # Create columns for trigger badges
        num_cols = min(len(triggers), 4)
        cols = st.columns(num_cols)

        for i, trigger in enumerate(triggers):
            with cols[i % num_cols]:
                if st.button(
                    f"⚠️ Turn {trigger['turn_number']}",
                    key=f"trigger_badge_{i}",
                    help=f"F={trigger.get('fidelity', 0.0):.3f} - {trigger['reason']}",
                    use_container_width=True
                ):
                    st.session_state.selected_trigger = trigger['trigger_id']
                    st.session_state.active_tab = "TELOSCOPE"
                    st.rerun()

    # Chat input
    user_input = st.chat_input("Ask about AI governance...")

    if user_input:
        # Create messages list
        messages = []
        for turn in turns:
            messages.append({"role": "user", "content": turn['user_input']})
            messages.append({"role": "assistant", "content": turn['assistant_response']})
        messages.append({"role": "user", "content": user_input})

        # Generate response through interceptor
        with st.spinner("🤔 Generating response..."):
            try:
                response = st.session_state.interceptor.generate(messages)
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error generating response: {e}")


def render_session_replay():
    """Render session replay with timeline."""
    st.title("⏮️ Session Replay")

    turns = st.session_state.current_session.get('turns', [])

    if not turns:
        st.info("📝 No conversation history yet. Start a conversation in the Live Session tab.")
        return

    # Initialize replay turn
    if 'replay_turn' not in st.session_state:
        st.session_state.replay_turn = 0

    # Timeline controls
    col1, col2, col3 = st.columns([1, 3, 1])

    with col1:
        if st.button("⏮️ Rewind", use_container_width=True):
            current = st.session_state.replay_turn
            st.session_state.replay_turn = max(0, current - 1)
            st.rerun()

    with col2:
        # Timeline slider
        turn_num = st.slider(
            "Turn",
            min_value=0,
            max_value=len(turns) - 1,
            value=st.session_state.replay_turn,
            key="replay_slider"
        )
        st.session_state.replay_turn = turn_num

    with col3:
        if st.button("⏭️ Forward", use_container_width=True):
            current = st.session_state.replay_turn
            st.session_state.replay_turn = min(len(turns) - 1, current + 1)
            st.rerun()

    # Display turn
    selected_turn = turns[turn_num]

    st.divider()

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader(f"Turn {turn_num + 1} of {len(turns)}")

        with st.chat_message("user"):
            st.write(selected_turn['user_input'])

        with st.chat_message("assistant"):
            st.write(selected_turn['assistant_response'])

    with col2:
        st.subheader("📊 Metrics")
        metrics = selected_turn.get('metrics', {})

        st.metric("Fidelity", f"{metrics.get('telic_fidelity', 1.0):.3f}")
        st.metric("Distance", f"{metrics.get('drift_distance', 0.0):.3f}")
        st.metric("Error Signal", f"{metrics.get('error_signal', 0.0):.3f}")

        basin = "Inside ✅" if metrics.get('primacy_basin_membership', True) else "Outside ❌"
        st.write(f"**Basin Status:** {basin}")

    # Show trigger markers on timeline
    triggers = st.session_state.web_session.get_all_triggers()
    if triggers:
        st.divider()
        st.subheader("🔬 Counterfactual Triggers")

        # Create columns for triggers
        num_cols = min(len(triggers), 4)
        trigger_cols = st.columns(num_cols)

        for i, trigger in enumerate(triggers):
            with trigger_cols[i % num_cols]:
                if st.button(
                    f"⚠️ Turn {trigger['turn_number']}",
                    key=f"replay_trigger_{i}",
                    help=trigger['reason'],
                    use_container_width=True
                ):
                    st.session_state.selected_trigger = trigger['trigger_id']
                    st.session_state.active_tab = "TELOSCOPE"
                    st.rerun()


def render_teloscope_view():
    """Render counterfactual branch comparison."""
    st.title("🔭 TELOSCOPE: Counterfactual Evidence")

    triggers = st.session_state.web_session.get_all_triggers()

    if not triggers:
        st.info("""
        ### Welcome to TELOSCOPE

        **No counterfactual experiments yet.**

        Counterfactuals are automatically triggered when drift is detected (fidelity < 0.76).

        **How it works:**
        1. Continue conversations in the Live Session tab
        2. When fidelity drops below 0.76 (Goldilocks Aligned threshold), a trigger fires
        3. Two 5-turn branches are generated:
           - **Baseline**: Shows what happens WITHOUT intervention
           - **TELOS**: Shows what happens WITH intervention
        4. ΔF (improvement) is calculated automatically

        Start a conversation to see TELOSCOPE in action!
        """)
        return

    # Trigger selector
    trigger_options = {t['trigger_id']: f"Turn {t['turn_number']}: {t['reason']}"
                      for t in triggers}

    # Use selected trigger if available
    default_trigger = st.session_state.get('selected_trigger', list(trigger_options.keys())[0])

    selected_trigger_id = st.selectbox(
        "Select Counterfactual Trigger",
        options=list(trigger_options.keys()),
        format_func=lambda x: trigger_options[x],
        index=list(trigger_options.keys()).index(default_trigger) if default_trigger in trigger_options else 0
    )

    # Get branch data
    branch_data = st.session_state.web_session.get_branch(selected_trigger_id)

    if not branch_data:
        st.warning("⏳ Generating counterfactual branches... (this may take 30-60 seconds)")
        if st.button("🔄 Refresh"):
            st.rerun()
        return

    if branch_data.get('status') == 'generating':
        st.info("⏳ Branch generation in progress...")
        if st.button("🔄 Refresh"):
            st.rerun()
        return

    if branch_data.get('status') == 'failed':
        st.error(f"❌ Branch generation failed: {branch_data.get('error', 'Unknown error')}")
        return

    # Get comparison data
    baseline = branch_data.get('baseline', {})
    telos = branch_data.get('telos', {})

    if not baseline or not telos:
        st.warning("⏳ Branch data incomplete. Please wait...")
        return

    comparison = st.session_state.comparator.compare_branches(baseline, telos)

    # Display ΔF improvement
    delta_f = comparison['delta']['delta_f']

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Governance Efficacy (ΔF)",
            f"{delta_f:+.3f}",
            delta=f"{abs(delta_f):.3f} {'improvement' if delta_f > 0 else 'degradation'}",
            delta_color="normal" if delta_f > 0 else "inverse"
        )

    with col2:
        st.metric(
            "Baseline Final F",
            f"{comparison['baseline']['final_fidelity']:.3f}"
        )

    with col3:
        st.metric(
            "TELOS Final F",
            f"{comparison['telos']['final_fidelity']:.3f}"
        )

    st.divider()

    # Side-by-side comparison
    st.subheader("Branch Comparison")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🔴 Baseline (No Intervention)")
        st.caption("Drift continues unchecked")

        baseline_turns = baseline.get('turns', [])
        for i, turn in enumerate(baseline_turns, 1):
            fidelity = turn.get('metrics', {}).get('telic_fidelity', 0.0)
            with st.expander(f"Turn {i} - F={fidelity:.3f}"):
                st.write(f"**Input:** {turn.get('user_input', 'N/A')}")
                st.write(f"**Response:** {turn.get('assistant_response', 'N/A')}")

    with col2:
        st.markdown("### 🟢 TELOS (With Intervention)")
        st.caption("Governance corrects drift")

        telos_turns = telos.get('turns', [])
        for i, turn in enumerate(telos_turns, 1):
            fidelity = turn.get('metrics', {}).get('telic_fidelity', 0.0)
            with st.expander(f"Turn {i} - F={fidelity:.3f}"):
                st.write(f"**Input:** {turn.get('user_input', 'N/A')}")
                st.write(f"**Response:** {turn.get('assistant_response', 'N/A')}")

    st.divider()

    # Fidelity divergence chart
    st.subheader("📈 Fidelity Divergence")

    if HAS_PLOTLY:
        fig = st.session_state.comparator.generate_divergence_chart(comparison)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Plotly not available. Install with: pip install plotly")

    # Metrics table
    st.subheader("📊 Metrics Comparison")
    df = st.session_state.comparator.generate_metrics_table(comparison)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Statistical analysis
    if 'statistics' in comparison:
        with st.expander("📊 Statistical Analysis"):
            stats = comparison['statistics']
            sig_text = "✅ Statistically significant" if stats['significant'] else "⚠️ Not statistically significant"

            st.markdown(f"""
            **{sig_text}** (p = {stats['p_value']:.4f})

            - **Effect Size (Cohen's d):** {stats['effect_size_cohens_d']:.3f}
            - **Mean Improvement:** {stats['mean_difference']:.3f}
            - **95% Confidence Interval:** [{stats['confidence_interval_95'][0]:.3f}, {stats['confidence_interval_95'][1]:.3f}]
            """)

    # Export button
    st.divider()
    col1, col2, col3 = st.columns([2, 1, 1])

    with col3:
        evidence = st.session_state.branch_manager.export_branch_evidence(selected_trigger_id)
        if evidence:
            evidence_json = json.dumps(evidence, indent=2)
            st.download_button(
                "📥 Export Evidence",
                data=evidence_json,
                file_name=f"teloscope_evidence_{selected_trigger_id}.json",
                mime="application/json",
                type="primary",
                use_container_width=True
            )


def render_analytics_dashboard():
    """Render aggregate analytics."""
    st.title("📊 Analytics Dashboard")

    # Session statistics
    stats = st.session_state.web_session.get_session_stats()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Turns", stats.get('total_turns', 0))

    with col2:
        st.metric("Triggers", stats.get('total_triggers', 0))

    with col3:
        st.metric("Avg Fidelity", f"{stats.get('avg_fidelity', 1.0):.3f}")

    with col4:
        trigger_rate = stats.get('trigger_rate', 0.0)
        st.metric("Trigger Rate", f"{trigger_rate * 100:.1f}%")

    st.divider()

    # Historical fidelity chart
    st.subheader("📈 Fidelity Over Time")

    turns = st.session_state.current_session.get('turns', [])
    if turns:
        fidelities = [t['metrics'].get('telic_fidelity', 1.0) for t in turns]

        if HAS_PLOTLY:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(1, len(fidelities) + 1)),
                y=fidelities,
                mode='lines+markers',
                name='Fidelity',
                line=dict(color='#339af0', width=2),
                marker=dict(size=8)
            ))

            fig.add_hline(y=0.8, line_dash="dash", line_color="orange",
                         annotation_text="Warning Threshold")
            fig.add_hline(y=0.5, line_dash="dash", line_color="red",
                         annotation_text="Critical Threshold")

            fig.update_layout(
                xaxis_title="Turn",
                yaxis_title="Fidelity",
                height=300,
                template='plotly_white'
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            # Fallback: simple data display
            st.line_chart(pd.DataFrame({'Fidelity': fidelities}))
    else:
        st.info("No conversation data yet.")

    st.divider()

    # Branch efficacy summary
    st.subheader("🔬 Counterfactual Efficacy")

    triggers = st.session_state.web_session.get_all_triggers()
    branches = st.session_state.current_session.get('branches', {})

    if branches:
        efficacy_data = []
        for trigger_id, branch_data in branches.items():
            if branch_data.get('status') == 'completed':
                comparison = branch_data.get('comparison', {})
                trigger_info = next((t for t in triggers if t['trigger_id'] == trigger_id), {})

                efficacy_data.append({
                    'Trigger Turn': trigger_info.get('turn_number', 'N/A'),
                    'ΔF': comparison.get('delta_f', 0.0),
                    'Avg Improvement': comparison.get('avg_improvement', 0.0),
                    'Significance': '✅' if comparison.get('statistics', {}).get('significant', False) else '❌'
                })

        if efficacy_data:
            df = pd.DataFrame(efficacy_data)
            st.dataframe(df, use_container_width=True, hide_index=True)

            # Average ΔF
            avg_delta_f = df['ΔF'].mean()
            if avg_delta_f > 0:
                st.success(f"**Average ΔF:** {avg_delta_f:+.3f} (governance is effective)")
            else:
                st.warning(f"**Average ΔF:** {avg_delta_f:+.3f} (governance needs tuning)")
        else:
            st.info("No completed counterfactual experiments yet.")
    else:
        st.info("No counterfactual experiments yet. Triggers fire automatically when drift is detected.")


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main application entry point."""

    # Initialize TELOSCOPE
    initialize_teloscope()

    # Header
    st.title("🔭 TELOSCOPE Observatory")
    st.caption("Telically Entrained Linguistic Operational Substrate Counterfactual Observation via Purpose-scoped Experimentation")

    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔴 Live Session",
        "⏮️ Session Replay",
        "🔭 TELOSCOPE",
        "📊 Analytics"
    ])

    with tab1:
        render_live_session()

    with tab2:
        render_session_replay()

    with tab3:
        render_teloscope_view()

    with tab4:
        render_analytics_dashboard()


if __name__ == "__main__":
    main()
