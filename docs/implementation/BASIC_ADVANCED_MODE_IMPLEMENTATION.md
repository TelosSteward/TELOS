# Basic/Advanced Mode Toggle Implementation

**Date**: 2025-10-26
**Status**: ✅ Complete
**File Modified**: `telos_purpose/dev_dashboard/streamlit_live_comparison.py`

## Overview

Implemented a toggleable UI complexity mode with user-friendly terminology for Basic mode and technical terminology for Advanced mode.

## Features Implemented

### 1. Mode Selector

**Location**: Top right of main UI (next to title)

- **Options**: Basic (default) | Advanced
- **Persistence**: Stored in `st.session_state.ui_mode`
- **UI**: Dropdown selector with help text

### 2. Terminology System

**Helper Functions** (lines 100-149):
- `get_mode()`: Returns current UI mode from session state
- `get_terminology(mode)`: Returns terminology dictionary for specified mode

**Terminology Mapping**:

| Concept | Basic Mode | Advanced Mode |
|---------|-----------|---------------|
| Main Metric | Alignment Score (%) | Telic Fidelity (F) |
| Status On | On Purpose ✅ | Inside Basin ✅ |
| Status Off | Off Purpose ⚠️ | Outside Basin ❌ |
| Action | Recalibration | Active Mitigation |
| Result | Realignment | ΔF Improvement |
| Drift Message | "Went off purpose - recalibrating..." | "Drift Detected - Applying Active Mitigation" |
| Success Message | "✅ Realigned to your purpose" | "✅ Intervention successful" |

### 3. Sidebar Metrics (Updated)

**Location**: Sidebar (lines 295-407)

#### Basic Mode Display:
```python
Alignment Score: 92%
Status: On Purpose ✅
Recalibrations: 3
Avg Improvement: +15%
```

#### Advanced Mode Display:
```python
Telic Fidelity (F): 0.920
Drift Distance (d): 0.153
Basin Status: Inside ✅
Error Signal (ε): 0.080
Interventions: 3
Avg ΔF: +0.150
```

### 4. Live Session Messages (Updated)

**Location**: Live Session tab (lines 1041-1144)

#### Drift Detection:
- **Basic**: "⚠️ Went off purpose - recalibrating..."
- **Advanced**: "⚠️ Drift detected (F=0.445)"

#### Intervention Details Expander:
- **Basic**: "🛡️ Recalibration Details"
  - Type, Focus (%), Realignment (+15%)
- **Advanced**: "🛡️ Active Mitigation Details"
  - Intervention Type, Salience (0.750), ΔF (+0.150)

#### Text Comparison:
- **Basic**:
  - "Before Recalibration" | "After Recalibration"
  - "Alignment: 45% ⚠️" | "Alignment: 92% ✅"
- **Advanced**:
  - "Original Response (Drifted)" | "Governed Response (Corrected)"
  - "Fidelity: 0.450 ⚠️" | "Fidelity: 0.920 ✅"

### 5. Mode Selector UI

**Location**: Main header (lines 2194-2218)

```python
col_title, col_mode = st.columns([4, 1])

with col_mode:
    new_mode = st.selectbox(
        "UI Mode",
        options=['Basic', 'Advanced'],
        index=0 if current_mode == 'Basic' else 1,
        help="Basic: User-friendly terminology | Advanced: Technical terminology"
    )
```

## Technical Details

### State Management

- Mode stored in: `st.session_state.ui_mode`
- Default: `'Basic'`
- Change triggers: `st.rerun()` to refresh UI with new terminology

### Conditional Rendering Pattern

All sections follow this pattern:

```python
mode = get_mode()
terms = get_terminology(mode)

if mode == 'Basic':
    # User-friendly display
    st.metric(terms['metric_name'], f"{fidelity * 100:.0f}%")
else:
    # Technical display
    st.metric(terms['metric_name'], f"{fidelity:.3f}")
```

## User Experience

### Basic Mode (Default)
**Target Audience**: Non-technical users, stakeholders, demos

**Benefits**:
- Percentage-based metrics (easier to understand)
- Plain language ("On Purpose" vs "Inside Basin")
- Simple terminology ("Recalibration" vs "Active Mitigation")
- Focused on outcomes rather than mechanics

### Advanced Mode
**Target Audience**: Researchers, developers, technical users

**Benefits**:
- Precise decimal metrics (0.000 - 1.000)
- Technical terminology ("Telic Fidelity", "Basin Membership")
- Full metric visibility (Distance, Error Signal)
- Detailed intervention statistics

## Testing Recommendations

1. **Mode Toggle**: Verify switching between modes updates all UI elements
2. **State Persistence**: Confirm mode persists across reloads within session
3. **Metric Display**: Check both percentage and decimal displays are accurate
4. **Terminology Consistency**: Verify all UI elements use correct terminology
5. **Drift Messages**: Test drift detection shows appropriate message per mode
6. **Intervention Details**: Confirm expander title and metrics match mode

## Future Enhancements

### Potential Additions:
1. **Tooltips**: Mode-specific help text for metrics
2. **Export Formats**: Mode-aware report generation
3. **Visualization**: Basic mode could use simpler charts
4. **Custom Modes**: Allow user-defined terminology sets
5. **Accessibility**: Color-blind friendly status indicators

## Files Modified

- `telos_purpose/dev_dashboard/streamlit_live_comparison.py` (2,247 lines)
  - Added: `get_mode()` and `get_terminology()` functions
  - Updated: Sidebar metrics section
  - Updated: Live session drift messages
  - Updated: Intervention details display
  - Updated: Main UI header with mode selector

## Related Documentation

- `SLIDER_CRASH_FIXED.md`: Previous dashboard bug fix
- `CHAT_INPUT_FIX.md`: Previous UI state management fix
- `DASHBOARD_COMPLETE.md`: Original dashboard implementation

## Notes

- Mode selector uses `st.selectbox()` for clarity over `st.radio()`
- Terminology helper functions are centralized for maintainability
- All percentage displays round to nearest integer for simplicity
- Decimal displays maintain 3 decimal places for precision
- Mode changes trigger immediate UI refresh via `st.rerun()`

## Success Criteria

✅ Mode selector visible and functional
✅ Basic mode uses user-friendly terminology
✅ Advanced mode uses technical terminology
✅ All metrics display correctly in both modes
✅ State persists across interactions
✅ No syntax errors
✅ Consistent terminology throughout UI
