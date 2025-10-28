# Dashboard Wiring Verification

## ✅ ALL FEATURES PROPERLY WIRED

All intervention display and counterfactual simulation features are correctly integrated into the chat rendering flow.

---

## INTERVENTION DISPLAY - WIRED ✅

### Location: Lines 958-1018
**Inside**: `with st.chat_message("assistant"):` block
**When**: After EVERY assistant message is displayed

```python
# Line 948: Start assistant message block
with st.chat_message("assistant"):
    st.write(turn['assistant_response'])  # Line 949: Show response

    # Line 958-1018: INTERVENTION DETAILS DISPLAY
    metadata = turn.get('metadata', {})
    intervention_details = metadata.get('intervention_details')

    if intervention_details and intervention_details.get('intervention_applied'):
        with st.expander("🛡️ Active Mitigation Details", expanded=False):
            # Three-column metrics
            col1, col2, col3 = st.columns(3)

            # Show: Intervention Type, Salience, ΔF
            # Show: Flow diagram for regenerations
            # Show: Side-by-side text comparison
```

**Triggers when**:
- `turn['metadata']['intervention_details']` exists
- `intervention_details['intervention_applied']` is True

---

## SIDEBAR STATISTICS - WIRED ✅

### Location: Lines 302-324
**Inside**: `render_sidebar()` function
**When**: Called on every page render

```python
# Line 302-303: Active Mitigation section header
st.subheader("🛡️ Active Mitigation")

# Line 305-306: Get statistics from steward
if st.session_state.get('steward') and hasattr(st.session_state.steward, 'llm_wrapper'):
    mitigation_stats = st.session_state.steward.llm_wrapper.get_intervention_statistics()

    # Display:
    # - Total interventions count
    # - Average ΔF
    # - Breakdown by type
    # - Thresholds
```

**Triggers when**:
- Steward exists in session state
- Steward has llm_wrapper attribute
- At least one intervention has occurred

---

## INTERVENTION TIMELINE - WIRED ✅

### Location: Lines 1022-1028
**Inside**: Live Chat mode, after conversation history
**When**: After all chat messages are displayed

```python
# Line 1022-1025: Render timeline if turns exist
if turns:
    st.divider()
    render_intervention_timeline(turns)

# Line 1027-1028: Render simulation UI
render_simulation_ui()
```

**Timeline displays**:
- Plotly chart with fidelity over time
- Drift threshold line at 0.8
- Green star intervention markers
- Clickable drift point buttons

**Triggers when**:
- `turns` list is not empty
- At least one turn exists in the conversation

---

## SIMULATION UI - WIRED ✅

### Location: Lines 1845-1925
**Function**: `render_simulation_ui()`
**When**: Called after intervention timeline (line 1028)

**Triggers when**:
- User clicks a drift point button
- `st.session_state.simulate_from_turn` is set

**Displays**:
- Simulation parameters (turns, topic hint)
- Run button
- Progress spinner during execution
- Results auto-display

---

## DATA FLOW

### 1. User Sends Message
```
User types in chat input
    ↓
LiveInterceptor.generate()
    ↓
UnifiedGovernanceSteward.generate_governed_response()
    ↓
Active mitigation (salience check, regeneration)
    ↓
intervention_details created and stored in turn metadata
    ↓
Turn saved to session state
```

### 2. Dashboard Renders
```
render_live_session() called
    ↓
For each turn in turns:
    ├─ Display user message
    └─ Display assistant message
        ├─ Show response text
        ├─ Check for intervention_details  [LINES 958-1018]
        └─ If exists: Show expandable with Type, Salience, ΔF
    ↓
After all messages:
    ├─ Render intervention timeline  [LINE 1025]
    └─ Render simulation UI  [LINE 1028]
    ↓
Sidebar:
    └─ Show intervention statistics  [LINES 302-324]
```

---

## WHY FEATURES MIGHT NOT APPEAR

### 1. No Interventions Yet
**Symptom**: Sidebar shows "No interventions yet"
**Reason**: No drift has been detected yet
**Solution**: Try going off-topic to trigger intervention

### 2. Browser Cache
**Symptom**: Old UI still showing
**Reason**: Browser has cached old version
**Solution**: Hard refresh (Cmd+Shift+R or Ctrl+Shift+R)

### 3. No Conversation Started
**Symptom**: No timeline appears
**Reason**: `turns` list is empty
**Solution**: Start chatting

### 4. Intervention Not Triggered
**Symptom**: No expandable section under assistant message
**Reason**: Response was on-topic (F >= 0.8)
**Solution**: Ask off-topic question to trigger drift

---

## EXPECTED BEHAVIOR

### Scenario 1: On-Topic Response
```
User: "What is TELOS?"
Assistant: "TELOS is a governance framework..."

✓ Message appears
✗ No intervention expandable (not needed - on-topic)
✗ Sidebar shows 0 interventions
✓ Timeline appears (if other turns exist)
```

### Scenario 2: Off-Topic (Intervention Triggered)
```
User: "Tell me about Italian cooking"
Assistant: "While that's an interesting topic, let's focus on..."

✓ Message appears
✓ ⚠️ Drift detected (F=0.65) warning
✓ 🛡️ Active Mitigation Details expandable appears
   ├─ Type: regeneration
   ├─ Salience: 🟢 0.75
   ├─ ΔF: +0.23
   ├─ Flow: F: 0.65 → 0.88
   └─ Side-by-side text comparison
✓ Sidebar shows 1 intervention, Avg ΔF: +0.23
✓ Timeline shows drift point with green star
✓ Drift point button clickable
```

### Scenario 3: Click Drift Point
```
User clicks "🛡️ Turn 2" button
    ↓
✓ Simulation UI appears below timeline
✓ Shows: "Simulating from Turn 2 (F=0.65)"
✓ Parameter controls visible
✓ Run button appears
```

### Scenario 4: Run Simulation
```
User clicks "Run Simulation"
    ↓
✓ Spinner appears
✓ ~20-30 seconds wait
✓ Results appear:
   ├─ Summary metrics (ΔF, improvement)
   ├─ Fidelity trajectory chart
   ├─ Turn-by-turn comparison
   └─ Download buttons
```

---

## VERIFICATION CHECKLIST

### Code Placement ✅
- [x] Intervention details in chat message block (lines 958-1018)
- [x] Sidebar statistics in render_sidebar() (lines 302-324)
- [x] Timeline after chat history (line 1025)
- [x] Simulation UI after timeline (line 1028)

### Functions Defined ✅
- [x] render_intervention_timeline() (lines 1737-1842)
- [x] render_simulation_ui() (lines 1845-1925)
- [x] render_simulation_results() (lines 1928-2070)

### Session State ✅
- [x] simulator initialized (line 192)
- [x] simulation_results storage (line 205)
- [x] steward with llm_wrapper (line 174)

### Imports ✅
- [x] CounterfactualBranchSimulator imported (line 22)
- [x] Plotly imported (lines 35-36)

---

## TESTING PROCEDURE

1. **Navigate to dashboard**: http://localhost:8501

2. **Go to Live Session tab**

3. **Start on-topic conversation**:
   - Input: "What is TELOS?"
   - Expected: Response, no intervention (normal)

4. **Trigger drift**:
   - Input: "Tell me about cooking pasta"
   - Expected:
     - ⚠️ Drift warning appears
     - 🛡️ Expandable section appears
     - Sidebar shows 1 intervention
     - Timeline shows drift point

5. **Check intervention details**:
   - Click expandable
   - Verify: Type, Salience emoji, ΔF value
   - Verify: Flow diagram
   - Verify: Side-by-side text comparison

6. **Check timeline**:
   - Scroll to timeline (below chat)
   - Verify: Plotly chart visible
   - Verify: Drift point button visible

7. **Test simulation**:
   - Click drift point button
   - Verify: Simulation UI appears
   - Set parameters (turns: 5, hint: "cooking")
   - Click "Run Simulation"
   - Wait ~20-30 seconds
   - Verify: Results display

8. **Test evidence export**:
   - Click "Download Markdown Report"
   - Click "Download JSON Data"
   - Verify: Files download

---

## IF FEATURES DON'T APPEAR

### Refresh Browser
```bash
# Hard refresh to clear cache
Cmd+Shift+R (Mac)
Ctrl+Shift+R (Windows/Linux)
```

### Verify Dashboard Running
```bash
# Check dashboard process
ps aux | grep streamlit

# Should show:
# streamlit run telos_purpose/dev_dashboard/streamlit_live_comparison.py
```

### Check for Errors
```bash
# Monitor dashboard output
# Look for any Python errors or tracebacks
```

### Restart Dashboard
```bash
# If needed, restart fresh
pkill -f streamlit
cd ~/Desktop/telos
./launch_dashboard.sh
```

---

## CONCLUSION

**All features are properly wired and integrated.**

The intervention display, sidebar statistics, timeline, and simulation UI are correctly placed in the code and will appear when:
1. Dashboard is freshly loaded
2. Conversations occur
3. Interventions are triggered (drift detected)
4. User interacts with drift points

**No code changes needed - features are ready to use.**

---

## FILES REFERENCE

- **Dashboard Code**: `telos_purpose/dev_dashboard/streamlit_live_comparison.py`
  - Intervention display: Lines 958-1018
  - Sidebar stats: Lines 302-324
  - Timeline: Line 1025
  - Simulation UI: Line 1028
  - Helper functions: Lines 1737-2070

- **This Verification**: `WIRING_VERIFICATION.md`
- **Pre-flight Check**: `PREFLIGHT_CHECK_REPORT.md`
- **Integration Summary**: `DASHBOARD_SIMULATOR_INTEGRATION.md`
