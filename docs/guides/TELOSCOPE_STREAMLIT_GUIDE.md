# TELOSCOPE Streamlit UI Implementation Guide

**Status**: Backend Complete - UI Ready for Integration
**Date**: 2025-10-25

---

## ✅ Backend Components Complete

All 5 backend components are now production-ready:

1. **WebSessionManager** - Streamlit state bridge
2. **SessionStateManager** - Immutable state snapshots
3. **CounterfactualBranchManager** - Branch generation
4. **LiveInterceptor** - Real-time drift detection
5. **BranchComparator** - Visualization generation

---

## Streamlit UI Structure

### Main App Initialization

```python
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

st.set_page_config(
    page_title="TELOSCOPE - AI Governance Observatory",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize components (only once)
def initialize_teloscope():
    if 'teloscope_initialized' not in st.session_state:
        # Load config
        with open('config.json') as f:
            config = json.load(f)

        # Create web session manager with st.session_state reference
        st.session_state.web_session = WebSessionManager(st.session_state)
        st.session_state.web_session.initialize_web_session()

        # Create session state manager
        st.session_state.session_manager = SessionStateManager(
            web_session_manager=st.session_state.web_session
        )

        # Initialize LLM and embeddings
        api_key = os.getenv('MISTRAL_API_KEY')
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
            drift_threshold=0.8,
            enable_counterfactuals=True
        )

        st.session_state.teloscope_initialized = True

# Initialize
initialize_teloscope()
```

---

## Tab 1: Live Session

```python
def render_live_session():
    """Render live conversation interface."""
    st.title("🔴 Live Session")

    # Sidebar: Live Metrics
    with st.sidebar:
        st.subheader("Live Metrics")
        metrics = st.session_state.interceptor.get_live_metrics()

        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Fidelity",
                f"{metrics['current_fidelity']:.3f}",
                delta=None,  # Calculate delta from previous turn
                delta_color="normal"
            )
        with col2:
            st.metric(
                "Distance",
                f"{metrics['current_distance']:.3f}",
                delta=None
            )

        basin_emoji = "✅" if metrics['basin_status'] else "❌"
        st.metric("Basin Status", f"{basin_emoji}")

        # Session controls
        st.divider()
        if st.button("Reset Session", type="secondary"):
            st.session_state.interceptor.reset_session()
            st.session_state.session_manager.clear_session()
            st.session_state.web_session.clear_web_session()
            st.rerun()

        if st.button("Export Session", type="primary"):
            session_json = st.session_state.web_session.export_session()
            st.download_button(
                label="Download JSON",
                data=session_json,
                file_name=f"teloscope_session_{st.session_state.session_id}.json",
                mime="application/json"
            )

    # Main area: Chat interface
    st.subheader("Conversation")

    # Display conversation history
    for turn in st.session_state.current_session.get('turns', []):
        with st.chat_message("user"):
            st.write(turn['user_input'])

        with st.chat_message("assistant"):
            st.write(turn['assistant_response'])

            # Show metrics badge
            metrics = turn.get('metrics', {})
            fidelity = metrics.get('telic_fidelity', 1.0)

            if fidelity < 0.8:
                st.warning(f"⚠️ Drift detected (F={fidelity:.3f})")

    # Display triggers as badges
    triggers = st.session_state.web_session.get_all_triggers()
    if triggers:
        st.info(f"🔬 {len(triggers)} counterfactual experiment(s) triggered")

        cols = st.columns(min(len(triggers), 4))
        for i, trigger in enumerate(triggers):
            with cols[i % 4]:
                if st.button(
                    f"⚠️ Turn {trigger['turn_number']}",
                    key=f"trigger_badge_{i}",
                    help=f"F={trigger['fidelity']:.3f}"
                ):
                    st.session_state.web_session.select_trigger(trigger['trigger_id'])
                    st.session_state.active_tab = "TELOSCOPE"
                    st.rerun()

    # Chat input
    user_input = st.chat_input("Ask about AI governance...")

    if user_input:
        # Create messages list
        messages = []
        for turn in st.session_state.current_session.get('turns', []):
            messages.append({"role": "user", "content": turn['user_input']})
            messages.append({"role": "assistant", "content": turn['assistant_response']})
        messages.append({"role": "user", "content": user_input})

        # Generate response through interceptor
        with st.spinner("Generating response..."):
            response = st.session_state.interceptor.generate(messages)

        # Rerun to show updated conversation
        st.rerun()
```

---

## Tab 2: Session Replay

```python
def render_session_replay():
    """Render session replay with timeline."""
    st.title("⏮️ Session Replay")

    turns = st.session_state.current_session.get('turns', [])

    if not turns:
        st.info("No conversation history yet. Start a conversation in the Live Session tab.")
        return

    # Timeline controls
    col1, col2, col3 = st.columns([1, 3, 1])

    with col1:
        if st.button("⏮️ Rewind"):
            current = st.session_state.get('replay_turn', 0)
            st.session_state.replay_turn = max(0, current - 1)
            st.rerun()

    with col2:
        # Timeline slider
        turn_num = st.slider(
            "Turn",
            min_value=0,
            max_value=len(turns) - 1,
            value=st.session_state.get('replay_turn', 0),
            key="replay_slider"
        )
        st.session_state.replay_turn = turn_num

    with col3:
        if st.button("⏭️ Forward"):
            current = st.session_state.get('replay_turn', 0)
            st.session_state.replay_turn = min(len(turns) - 1, current + 1)
            st.rerun()

    # Display turn
    selected_turn = turns[turn_num]

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"Turn {turn_num + 1}")
        with st.chat_message("user"):
            st.write(selected_turn['user_input'])

        with st.chat_message("assistant"):
            st.write(selected_turn['assistant_response'])

    with col2:
        st.subheader("Metrics")
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

        trigger_cols = st.columns(len(triggers))
        for i, trigger in enumerate(triggers):
            with trigger_cols[i]:
                if st.button(
                    f"⚠️ Turn {trigger['turn_number']}",
                    key=f"replay_trigger_{i}",
                    help=trigger['reason']
                ):
                    st.session_state.web_session.select_trigger(trigger['trigger_id'])
                    st.session_state.active_tab = "TELOSCOPE"
                    st.rerun()
```

---

## Tab 3: TELOSCOPE Modal

```python
def render_teloscope_view():
    """Render counterfactual branch comparison."""
    st.title("🔬 TELOSCOPE: Counterfactual Evidence")

    triggers = st.session_state.web_session.get_all_triggers()

    if not triggers:
        st.info("""
        No counterfactual experiments yet.

        Counterfactuals are automatically triggered when drift is detected (fidelity < 0.8).
        Start a conversation in the Live Session tab to see counterfactual branching in action.
        """)
        return

    # Trigger selector
    selected_trigger_id = st.selectbox(
        "Select Trigger Point",
        options=[t['trigger_id'] for t in triggers],
        format_func=lambda tid: f"Turn {next(t['turn_number'] for t in triggers if t['trigger_id'] == tid)} - {next(t['reason'] for t in triggers if t['trigger_id'] == tid)}"
    )

    # Get branch data
    branch_data = st.session_state.web_session.get_branch(selected_trigger_id)

    if not branch_data or branch_data.get('status') == 'generating':
        st.warning("⏳ Generating counterfactual branches... Please wait.")
        return

    if branch_data.get('status') == 'failed':
        st.error(f"❌ Branch generation failed: {branch_data.get('error')}")
        return

    # Get comparison data
    baseline = branch_data['baseline']
    telos = branch_data['telos']
    comparison = st.session_state.comparator.compare_branches(baseline, telos)

    # Display ΔF improvement
    delta_f = comparison['delta']['delta_f']
    st.metric(
        "Governance Efficacy (ΔF)",
        f"{delta_f:+.3f}",
        delta=f"{abs(delta_f):.3f} improvement",
        delta_color="normal" if delta_f > 0 else "inverse"
    )

    # Side-by-side comparison
    st.subheader("Branch Comparison")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🔴 Baseline (No Intervention)")
        st.caption("Shows drift continuation without governance")

        for i, turn in enumerate(baseline['turns'], 1):
            with st.expander(f"Turn {i} (F={turn['metrics']['telic_fidelity']:.3f})"):
                st.write(f"**User:** {turn['user_input']}")
                st.write(f"**Assistant:** {turn['assistant_response']}")

    with col2:
        st.markdown("### 🟢 TELOS (With Intervention)")
        st.caption("Shows correction through governance")

        for i, turn in enumerate(telos['turns'], 1):
            with st.expander(f"Turn {i} (F={turn['metrics']['telic_fidelity']:.3f})"):
                st.write(f"**User:** {turn['user_input']}")
                st.write(f"**Assistant:** {turn['assistant_response']}")

    st.divider()

    # Fidelity divergence chart
    st.subheader("Fidelity Divergence")
    fig = st.session_state.comparator.generate_divergence_chart(comparison)
    if fig:
        st.plotly_chart(fig, use_container_width=True)

    # Metrics table
    st.subheader("Metrics Comparison")
    df = st.session_state.comparator.generate_metrics_table(comparison)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Statistical analysis
    if 'statistics' in comparison:
        with st.expander("📊 Statistical Analysis"):
            stats_text = st.session_state.comparator.format_statistics_text(comparison['statistics'])
            st.markdown(stats_text)

    # Export button
    st.divider()
    col1, col2 = st.columns([3, 1])
    with col2:
        evidence = st.session_state.branch_manager.export_branch_evidence(selected_trigger_id)
        if evidence:
            evidence_json = json.dumps(evidence, indent=2)
            st.download_button(
                "📥 Export Evidence",
                data=evidence_json,
                file_name=f"teloscope_evidence_{selected_trigger_id}.json",
                mime="application/json",
                type="primary"
            )
```

---

## Tab 4: Analytics Dashboard

```python
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
    st.subheader("Fidelity Over Time")

    turns = st.session_state.current_session.get('turns', [])
    if turns:
        fidelities = [t['metrics'].get('telic_fidelity', 1.0) for t in turns]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(1, len(fidelities) + 1)),
            y=fidelities,
            mode='lines+markers',
            name='Fidelity',
            line=dict(color='#339af0', width=2)
        ))

        fig.add_hline(y=0.8, line_dash="dash", line_color="orange")
        fig.update_layout(
            xaxis_title="Turn",
            yaxis_title="Fidelity",
            height=300
        )

        st.plotly_chart(fig, use_container_width=True)

    # Branch efficacy summary
    st.subheader("Counterfactual Efficacy")

    triggers = st.session_state.web_session.get_all_triggers()
    branches = st.session_state.current_session.get('branches', {})

    if branches:
        efficacy_data = []
        for trigger_id, branch_data in branches.items():
            if branch_data.get('status') == 'completed':
                comparison = branch_data.get('comparison', {})
                efficacy_data.append({
                    'Trigger': trigger_id,
                    'ΔF': comparison.get('delta_f', 0.0),
                    'Avg Improvement': comparison.get('avg_improvement', 0.0)
                })

        if efficacy_data:
            df = pd.DataFrame(efficacy_data)
            st.dataframe(df, use_container_width=True, hide_index=True)

            # Average ΔF
            avg_delta_f = df['ΔF'].mean()
            st.success(f"**Average ΔF:** {avg_delta_f:+.3f}")
```

---

## Main App Structure

```python
# Main tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🔴 Live Session",
    "⏮️ Session Replay",
    "🔬 TELOSCOPE",
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
```

---

## CSS Customization

```python
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
</style>
""", unsafe_allow_html=True)
```

---

## Running the App

```bash
cd ~/Desktop/telos
source venv/bin/activate
export MISTRAL_API_KEY="your_key_here"
streamlit run telos_purpose/dev_dashboard/streamlit_teloscope.py
```

---

## Key Features Checklist

- ✅ Live chat with real-time metrics
- ✅ Trigger badges appear on drift detection
- ✅ Timeline replay with scrubber
- ✅ Side-by-side branch comparison
- ✅ Fidelity divergence visualization
- ✅ Statistical significance testing
- ✅ Export functionality (JSON)
- ✅ Aggregate analytics dashboard
- ✅ Non-blocking counterfactual generation

---

**Status**: Ready for UI implementation with all backend components operational
