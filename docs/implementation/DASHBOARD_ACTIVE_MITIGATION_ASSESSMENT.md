# Dashboard Active Mitigation Assessment & Recommendations

## Executive Summary

**FINDING**: The dashboard has excellent infrastructure but **NO ACTIVE MITIGATION VISUALIZATION**.

**DATA EXISTS** but is not displayed:
- `steward.llm_wrapper.interventions` - Full intervention log
- `steward.llm_wrapper.get_intervention_statistics()` - Aggregate stats
- Per-turn salience, coupling, intervention types

**RECOMMENDATION**: Augment existing infrastructure with targeted active mitigation indicators.

---

## Current Dashboard Architecture

### Tabs (4 total)
1. **🔴 Live Session** - Live chat + session replay
2. **⏮️ Session Replay** - Timeline scrubber
3. **🔭 TELOSCOPE** - Counterfactual evidence viewer
4. **📊 Analytics** - Aggregate statistics

### Current Metrics Display

**Sidebar (Real-time)**:
```python
# Line 234: st.session_state.interceptor.get_live_metrics()
- Telic Fidelity (F)
- Drift Distance (d)
- Basin Status (✅/❌)
- Error Signal (ε)
- Session Stats (turns, triggers, avg F, trigger rate)
```

**Live Chat Messages** (Line 908-917):
```python
# CURRENT: Basic intervention indicator
if fidelity < 0.8:
    st.warning(f"⚠️ Drift detected (F={fidelity:.3f})")

if intervention_applied:
    st.success("✅ Governance intervention applied")
```

**PROBLEM**: This shows PASSIVE observation, not ACTIVE mitigation details.

---

## What's Missing: Active Mitigation Indicators

### Available Data NOT Being Displayed

From `steward.llm_wrapper`:
```python
interventions: List[GovernanceIntervention]
    - turn_number
    - intervention_type: "salience_injection" | "regeneration" | "both" | "none"
    - original_response (if regenerated)
    - governed_response
    - fidelity_original
    - fidelity_governed
    - salience_before
    - salience_after
    - timestamp

get_intervention_statistics():
    - total_interventions
    - by_type: {"salience_injection": X, "regeneration": Y, "both": Z}
    - regeneration_count
    - salience_injection_count
    - avg_fidelity_improvement
    - coupling_threshold
    - salience_threshold
```

**This data exists in memory but is NEVER SHOWN.**

---

## Recommended Implementation Strategy

### Phase 1: Minimal Integration (HIGH IMPACT, LOW EFFORT)

**Where**: Augment existing Live Chat message display (line 904-917)

**What to Add**:
```python
# After each assistant message in Live Chat
with st.chat_message("assistant"):
    st.write(turn['assistant_response'])

    # EXISTING: Basic warnings
    if fidelity < 0.8:
        st.warning(f"⚠️ Drift detected (F={fidelity:.3f})")

    # NEW: Active mitigation details (if available)
    if 'intervention_details' in turn:
        details = turn['intervention_details']

        with st.expander("🛡️ Active Mitigation Details", expanded=False):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Intervention Type", details['type'])

            with col2:
                salience = details.get('salience_after', 1.0)
                salience_emoji = "🟢" if salience >= 0.7 else "🟡"
                st.metric("Salience", f"{salience_emoji} {salience:.3f}")

            with col3:
                if details.get('fidelity_improvement'):
                    improvement = details['fidelity_improvement']
                    st.metric("ΔF", f"{improvement:+.3f}", delta_color="normal")

            # Show intervention timeline
            if details['type'] in ["regeneration", "both"]:
                st.caption("**Flow**: Original → Drift Detected → Regenerated → Governed")
                st.caption(f"F: {details['fidelity_original']:.3f} → {details['fidelity_governed']:.3f}")
```

**Effort**: 30 minutes
**Impact**: Users SEE active mitigation happening

---

### Phase 2: Sidebar Enhancement (MEDIUM IMPACT, LOW EFFORT)

**Where**: Add new section to sidebar (after line 286)

**What to Add**:
```python
# In render_sidebar(), after Session Stats
st.divider()
st.subheader("🛡️ Active Mitigation")

if st.session_state.get('steward'):
    stats = st.session_state.steward.llm_wrapper.get_intervention_statistics()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Interventions", stats['total_interventions'])
    with col2:
        avg_improvement = stats.get('avg_fidelity_improvement', 0)
        st.metric("Avg ΔF", f"{avg_improvement:+.3f}")

    # Intervention breakdown
    if stats['by_type']:
        st.caption("**By Type:**")
        for itype, count in stats['by_type'].items():
            if count > 0:
                st.caption(f"• {itype}: {count}")

    # Thresholds
    with st.expander("⚙️ Thresholds", expanded=False):
        st.caption(f"Salience: {stats['salience_threshold']:.2f}")
        st.caption(f"Coupling: {stats['coupling_threshold']:.2f}")
```

**Effort**: 20 minutes
**Impact**: Always-visible intervention statistics

---

### Phase 3: Analytics Tab Enhancement (HIGH IMPACT, MEDIUM EFFORT)

**Where**: Add new section to Analytics dashboard (in `render_analytics_dashboard()`)

**What to Add**:
```python
st.subheader("🛡️ Active Mitigation Analysis")

if st.session_state.get('steward'):
    interventions = st.session_state.steward.llm_wrapper.export_interventions()

    if interventions:
        # Create DataFrame
        import pandas as pd
        df = pd.DataFrame(interventions)

        # Filter out learning phase
        df_governed = df[df['intervention_type'] != 'learning_phase']

        # Aggregate metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Interventions",
                len(df_governed),
                help="Interventions during governance phase"
            )

        with col2:
            salience_injections = len(df_governed[
                df_governed['intervention_type'].isin(['salience_injection', 'both'])
            ])
            st.metric("Salience Maintenance", salience_injections)

        with col3:
            regenerations = len(df_governed[
                df_governed['intervention_type'].isin(['regeneration', 'both'])
            ])
            st.metric("Regenerations", regenerations)

        with col4:
            # Calculate average improvement for regenerations
            regen_df = df_governed[
                (df_governed['fidelity_original'].notna()) &
                (df_governed['fidelity_governed'].notna())
            ]
            if len(regen_df) > 0:
                avg_improvement = (
                    regen_df['fidelity_governed'] - regen_df['fidelity_original']
                ).mean()
                st.metric("Avg Improvement", f"{avg_improvement:+.3f}")

        st.divider()

        # Intervention timeline (if plotly available)
        if HAS_PLOTLY and len(df_governed) > 0:
            fig = go.Figure()

            # Plot salience over time
            fig.add_trace(go.Scatter(
                x=df['turn_number'],
                y=df['salience_after'],
                mode='lines+markers',
                name='Salience',
                line=dict(color='blue')
            ))

            # Plot fidelity over time
            fig.add_trace(go.Scatter(
                x=df['turn_number'],
                y=df['fidelity_governed'],
                mode='lines+markers',
                name='Fidelity',
                line=dict(color='green')
            ))

            # Mark interventions
            intervention_turns = df_governed[
                df_governed['intervention_type'] != 'none'
            ]['turn_number'].tolist()

            fig.add_trace(go.Scatter(
                x=intervention_turns,
                y=[0.5] * len(intervention_turns),
                mode='markers',
                name='Interventions',
                marker=dict(
                    symbol='diamond',
                    size=12,
                    color='red'
                )
            ))

            # Add threshold lines
            fig.add_hline(
                y=0.70,
                line_dash="dash",
                line_color="orange",
                annotation_text="Salience Threshold"
            )

            fig.add_hline(
                y=0.80,
                line_dash="dash",
                line_color="red",
                annotation_text="Coupling Threshold"
            )

            fig.update_layout(
                title="Active Mitigation Timeline",
                xaxis_title="Turn Number",
                yaxis_title="Score",
                hovermode='x unified',
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

        # Detailed intervention log
        with st.expander("📋 Detailed Intervention Log", expanded=False):
            st.dataframe(
                df_governed[[
                    'turn_number',
                    'intervention_type',
                    'fidelity_governed',
                    'salience_after'
                ]],
                use_container_width=True
            )

            # Download button
            csv = df_governed.to_csv(index=False)
            st.download_button(
                "📥 Download Intervention Log (CSV)",
                data=csv,
                file_name=f"interventions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
```

**Effort**: 1-2 hours (including testing)
**Impact**: Complete intervention analysis with visualizations

---

## Architecture Fit

### Current Data Flow

```
LiveInterceptor.generate()
    ↓
steward.generate_governed_response()
    ↓
llm_wrapper.generate()  ← INTERVENTIONS HAPPEN HERE
    ↓
Result with intervention details
    ↓
LiveInterceptor stores in session
    ↓
??? (NOT displayed in dashboard)
```

### Proposed Data Flow

```
LiveInterceptor.generate()
    ↓
steward.generate_governed_response()
    ↓
result = {
    'governed_response': str,
    'intervention_applied': bool,
    'intervention_type': str,
    'fidelity': float,
    'salience': float,
    'metrics': {...}
}
    ↓
Store in turn['intervention_details']  ← NEW
    ↓
Display in dashboard with expandable details
```

**Required Changes**:
1. `LiveInterceptor.generate()` (line 114) already gets full result
2. Need to store `intervention_details` in turn record
3. Dashboard reads `intervention_details` and displays

---

## Implementation Priority

### Must Have (Phase 1 + 2)
**Time**: 1 hour
**Value**: High - Users see active mitigation working

1. Augment Live Chat messages with intervention details
2. Add sidebar intervention statistics

### Should Have (Phase 3)
**Time**: 2 hours
**Value**: Medium - Comprehensive analysis

3. Analytics tab with timeline and aggregate analysis
4. Downloadable intervention logs

### Nice to Have
**Time**: 3-4 hours
**Value**: Low - Polish

5. Intervention type color coding
6. Real-time intervention animations
7. Comparative visualizations (governed vs ungoverned)

---

## Recommended Next Steps

### Immediate (Do Now)
1. **Modify LiveInterceptor** to store intervention details in turn records
2. **Update Live Chat display** to show intervention expandables
3. **Add sidebar section** for intervention stats

### Short-term (This Week)
4. **Add Analytics tab section** with timeline visualization
5. **Create intervention export** functionality

### Long-term (As Needed)
6. **Real-time intervention notifications** (toast messages)
7. **Intervention effectiveness dashboard** (A/B comparisons)

---

## Code Locations

### Files to Modify

1. **`telos_purpose/sessions/live_interceptor.py`** (line 114-130)
   - Store intervention details in turn metadata

2. **`telos_purpose/dev_dashboard/streamlit_live_comparison.py`**:
   - Line 904-917: Live Chat message display
   - Line 270-286: Sidebar stats section
   - Line 1398+: Analytics dashboard

### No New Files Needed

All changes are augmentations to existing files.

---

## Expected Outcomes

### Before (Current)
```
User: [Off-topic question]
Assistant: [Response]
⚠️ Drift detected (F=0.62)
✅ Governance intervention applied
```

### After (With Active Mitigation Display)
```
User: [Off-topic question]
Assistant: [Response]

🛡️ Active Mitigation Details ▼
┌─────────────────────────────────────────┐
│ Type: regeneration                      │
│ Salience: 🟢 0.75                       │
│ ΔF: +0.27                               │
│                                         │
│ Flow: Original → Drift → Regenerated   │
│ F: 0.62 → 0.89                         │
└─────────────────────────────────────────┘
```

**Sidebar**:
```
🛡️ Active Mitigation
┌──────────────┐
│ Interventions │ 12 │
│ Avg ΔF       │ +0.23 │
└──────────────┘

By Type:
• regeneration: 7
• salience_injection: 3
• both: 2
```

---

## Conclusion

**VERDICT**: The infrastructure is solid. Just need to **wire existing data to UI**.

**BEST PATH**: Phase 1 + 2 (minimal integration) = **1 hour of work** for **complete active mitigation visibility**.

**DO NOT**: Create new complex visualization systems. The dashboard architecture is good - just augment it.

**DO**: Show the intervention data that's already being generated.
