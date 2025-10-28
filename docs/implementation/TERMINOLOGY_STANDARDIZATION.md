# Terminology Standardization: Primacy Attractor

**Date**: 2025-10-27
**Status**: ✅ Complete
**File Modified**: `telos_purpose/dev_dashboard/streamlit_live_comparison.py`

## Overview

Standardized terminology across all UI modes to consistently use "Primacy Attractor" instead of mode-dependent labels like "Your Purpose" in Basic mode.

## Rationale

**Problem**: Inconsistent terminology across modes created confusion and undermined professional branding:
- Basic Mode: "Your Purpose"
- Advanced Mode: "Primacy Attractor"
- Research Mode: "Primacy Attractor"

**Why This Matters**:
1. **Technical Accuracy**: "Primacy Attractor" is the correct technical term established in TELOS framework
2. **Professional Consistency**: Terminology should not vary by user expertise level
3. **Academic Alignment**: Matches terminology used in papers and research
4. **Brand Strength**: Consistent use of proprietary TELOS terminology strengthens IP
5. **User Experience**: Less confusion when switching modes

**Solution**: Use "Primacy Attractor" universally across all modes.

## Changes Made

### Change 1: Terminology Dictionary (Lines 114-136)

**Before**:
```python
if mode == 'Basic':
    return {
        'attractor_name': 'Your Purpose',
        'status_on': 'On Purpose ✅',
        'status_off': 'Off Purpose ⚠️',
        'drift_message': 'Went off purpose - recalibrating...',
        'success_message': '✅ Realigned to your purpose'
    }
```

**After**:
```python
if mode == 'Basic':
    return {
        'attractor_name': 'Primacy Attractor',
        'status_on': 'Aligned ✅',
        'status_off': 'Drifted ⚠️',
        'drift_message': 'Drifted from Primacy Attractor - recalibrating...',
        'success_message': '✅ Realigned to Primacy Attractor'
    }
```

**Impact**:
- `attractor_name`: "Your Purpose" → "Primacy Attractor"
- `status_on`: "On Purpose" → "Aligned" (cleaner, more direct)
- `status_off`: "Off Purpose" → "Drifted" (cleaner, more direct)
- Messages updated to reference "Primacy Attractor" explicitly

### Change 2: Documentation Comments (Lines 114-119)

**Before**:
```python
Returns dict with keys:
- attractor_name: "Your Purpose" or "Primacy Attractor"
- status_on: "On Purpose" or "Inside Basin"
- status_off: "Off Purpose" or "Outside Basin"
```

**After**:
```python
Returns dict with keys:
- attractor_name: Always "Primacy Attractor" (standardized across all modes)
- status_on: "Aligned" or "Inside Basin"
- status_off: "Drifted" or "Outside Basin"
```

### Change 3: Help Text (Line 523)

**Before**:
```python
help="How well aligned with your purpose (0-100%)"
```

**After**:
```python
help="How well aligned with Primacy Attractor (0-100%)"
```

## Terminology Mapping

### Basic Mode (User-Friendly + Professional)
- **Attractor**: "Primacy Attractor" (professional term)
- **Status Aligned**: "Aligned ✅" (simple, clear)
- **Status Drifted**: "Drifted ⚠️" (simple, clear)
- **Metric**: "Alignment Score" (percentage-based)
- **Action**: "Recalibration" (accessible term)
- **Result**: "Realignment" (clear outcome)

### Advanced Mode (Technical)
- **Attractor**: "Primacy Attractor" (same!)
- **Status Aligned**: "Inside Basin ✅" (technical precision)
- **Status Drifted**: "Outside Basin ❌" (technical precision)
- **Metric**: "Telic Fidelity (F)" (decimal-based)
- **Action**: "Active Mitigation" (technical term)
- **Result**: "ΔF Improvement" (mathematical notation)

### Research Mode (Mathematical)
- **Attractor**: "Primacy Attractor" (same!)
- **Status**: Mathematical expressions (d < r_basin)
- **Displays**: Full mathematical observatory

## Before/After Comparison

### Basic Mode Display

**BEFORE**:
```
🎯 Your Purpose
Status: On Purpose ✅
Alignment Score: 92%
⚠️ Went off purpose - recalibrating...
✅ Realigned to your purpose
```

**AFTER**:
```
🎯 Primacy Attractor
Status: Aligned ✅
Alignment Score: 92%
⚠️ Drifted from Primacy Attractor - recalibrating...
✅ Realigned to Primacy Attractor
```

### Advanced Mode Display

**BEFORE** (already correct):
```
🎯 Primacy Attractor
Status: Inside Basin ✅
Telic Fidelity: 0.920
```

**AFTER** (unchanged):
```
🎯 Primacy Attractor
Status: Inside Basin ✅
Telic Fidelity: 0.920
```

## Consistency Maintained

### What Changed:
- Attractor label: "Your Purpose" → "Primacy Attractor"
- Status labels: "On Purpose" / "Off Purpose" → "Aligned" / "Drifted"
- Messages: All references to "your purpose" → "Primacy Attractor"

### What Stayed the Same:
- Metric calculation methods (unchanged)
- Display formats (percentage vs decimal by mode)
- Action terminology ("Recalibration" vs "Active Mitigation" still mode-dependent)
- Result terminology ("Realignment" vs "ΔF" still mode-dependent)

### Why Keep Some Variation:
- **Actions and Results** can use mode-appropriate language (simple vs technical)
- **The Thing Being Governed** (Primacy Attractor) must be consistent
- Analogy: You can call the process "tuning" or "calibration", but the instrument being tuned should have one name

## User Experience Impact

### Benefits:
1. **Professional Tone**: All users learn proper TELOS terminology from start
2. **Mode Switching**: No confusion when toggling between Basic/Advanced
3. **Academic Credibility**: Researchers see consistent terminology they recognize from papers
4. **Brand Strength**: Reinforces "Primacy Attractor" as core TELOS concept
5. **Clarity**: "Aligned" vs "Drifted" are clearer than "On Purpose" vs "Off Purpose"

### Potential Concerns (Addressed):
**Concern**: "Primacy Attractor" might sound too technical for Basic mode
**Response**:
- Status labels ("Aligned" / "Drifted") remain simple
- Metric names ("Alignment Score") remain accessible
- Only the NAME of the concept is standardized
- Users quickly learn the term (appears everywhere consistently)

**Concern**: Users might not know what "Primacy Attractor" means
**Response**:
- Help text explains: "Mathematical representation of conversation purpose"
- Configuration UI shows content (Purpose: X, Scope: Y, Z)
- Progressive extraction messages explain establishment process
- Research Lens provides full mathematical explanation when needed

## Technical Details

### Files Modified:
- `telos_purpose/dev_dashboard/streamlit_live_comparison.py` (Lines 114-136, 523)

### Functions Updated:
- `get_terminology()`: Returns standardized `attractor_name` = "Primacy Attractor" for all modes

### Affected UI Locations:
1. Sidebar configuration display
2. Status indicators
3. Calibration messages (progressive extraction)
4. Drift detection messages
5. Intervention success messages
6. Help text and tooltips
7. Research Mode headers

## Testing Recommendations

1. **Basic Mode**: Verify "Primacy Attractor" appears in all relevant UI locations
2. **Advanced Mode**: Verify no change from previous (already used "Primacy Attractor")
3. **Research Mode**: Verify mathematical headers reference "Primacy Attractor"
4. **Mode Switching**: Toggle between modes and verify no terminology inconsistency
5. **Messages**: Trigger drift and intervention, verify messages use "Primacy Attractor"
6. **Help Text**: Hover over metrics, verify help text updated

## Related Documentation

- `BASIC_ADVANCED_MODE_IMPLEMENTATION.md`: Original mode terminology system
- `DASHBOARD_COMPLETE.md`: Original dashboard implementation
- `INTERVENTION_ANALYSIS_ENHANCEMENT.md`: Recent intervention display update

## Success Criteria

✅ All modes use "Primacy Attractor" as attractor label
✅ Basic mode status simplified to "Aligned" / "Drifted"
✅ Messages reference "Primacy Attractor" explicitly
✅ Help text updated to reference "Primacy Attractor"
✅ Documentation comments updated
✅ Syntax check passed
✅ Dashboard restarted successfully
✅ Terminology consistent across all UI components

## Future Considerations

### Keep Simple in Basic Mode:
- Status: "Aligned ✅" or "Drifted ⚠️" (not overly technical)
- Metrics: Percentage format (accessible)
- Actions: "Recalibration" (simpler than "Active Mitigation")

### Keep Technical in Advanced Mode:
- Status: "Inside Basin ✅" or "Outside Basin ❌" (precise)
- Metrics: Decimal format (precise)
- Actions: "Active Mitigation" (technical accuracy)

### Universal Across All Modes:
- **Primacy Attractor** label (consistent professional terminology)
- Core concept naming (always use TELOS framework terms)
- Research Lens mathematical displays (available to all modes via toggle)

## Notes

- This change strengthens TELOS brand identity by consistent use of proprietary terminology
- "Primacy Attractor" is established in academic literature and should be used consistently
- User feedback will determine if additional explanatory tooltips are needed
- Mode selector help text could be updated to explain terminology differences
