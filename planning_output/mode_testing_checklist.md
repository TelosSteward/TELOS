# Observatory Unified Mode Architecture - Testing Checklist

**Application URL**: http://localhost:8501
**Test Date**: 2025-11-09
**Test Goal**: Validate unified `render_mode_content()` function works correctly across all modes

---

## Testing Protocol

For each mode, verify:
1. Correct components render
2. PA convergence gates work
3. Feature flags activate correctly
4. No regressions from previous behavior

---

## Mode 1: DEMO Mode

**Expected Behavior**:
- ✅ `telos_demo_mode = True` (hardcoded TELOS PA loaded)
- ✅ Conversation display renders
- ❌ NO Observation Deck (DEMO has no governance features)
- ❌ NO TELOSCOPE Controls
- ✅ PA shows as "Established" immediately (demo mode)
- ❌ NO fidelity tracking (demo mode)

**Test Steps**:
1. Switch to DEMO tab
2. Send a test message
3. Verify conversation displays
4. Verify NO Observation Deck appears
5. Verify NO TELOSCOPE Controls appear
6. Verify PA shows as established (demo behavior)

**Result**: [ ] PASS / [ ] FAIL
**Notes**:

---

## Mode 2: BETA Mode

**Expected Behavior**:
- ✅ `telos_demo_mode = False` (progressive PA extraction)
- ✅ Conversation display renders
- ✅ Observation Deck renders (but DISABLED until PA converged)
- ❌ NO TELOSCOPE Controls (BETA doesn't have playback)
- ⏳ PA shows "Calibrating (X/~10)" for first ~10 turns
- ⏳ NO fidelity metrics until PA converged
- ✅ After ~10 turns, PA converges and metrics appear

**Test Steps - Before PA Convergence**:
1. Switch to BETA tab
2. Send message #1
3. Verify conversation displays
4. Verify Observation Deck shows "Unavailable until PA established"
5. Verify NO fidelity metrics in message header
6. Verify PA status shows "Calibrating (1/~10)"

**Test Steps - After PA Convergence**:
7. Continue conversation to ~10 turns
8. Verify PA status changes to "Established"
9. Verify Observation Deck becomes active with metrics
10. Verify fidelity scores appear in message headers

**Result**: [ ] PASS / [ ] FAIL
**Notes**:

---

## Mode 3: TELOS Mode

**Expected Behavior**:
- ✅ `telos_demo_mode = False` (progressive PA extraction)
- ✅ Conversation display renders
- ✅ Observation Deck renders (DISABLED until PA converged)
- ✅ TELOSCOPE Controls render (DISABLED until PA converged)
- ⏳ PA shows "Calibrating (X/~10)" for first ~10 turns
- ⏳ NO fidelity metrics until PA converged
- ✅ After ~10 turns, full Observatory unlocks

**Test Steps - Before PA Convergence**:
1. Switch to TELOS tab
2. Send message #1
3. Verify conversation displays
4. Verify Observation Deck shows "Unavailable until PA established"
5. Verify TELOSCOPE Controls shows "Unavailable until PA established"
6. Verify PA status shows "Calibrating (1/~10)"

**Test Steps - After PA Convergence**:
7. Continue conversation to ~10 turns
8. Verify PA status changes to "Established"
9. Verify Observation Deck becomes active
10. Verify TELOSCOPE Controls become active
11. Test playback controls (Prev/Next/Play/Reset)
12. Test intervention scrubber (if interventions occurred)

**Result**: [ ] PASS / [ ] FAIL
**Notes**:

---

## Mode 4: DEVOPS Mode

**Expected Behavior**:
- ✅ `telos_demo_mode = False` (progressive PA extraction)
- ✅ Debug header displays: "🔧 DEVOPS Mode - Full System Access"
- ✅ Conversation display renders
- ✅ Observation Deck renders (DISABLED until PA converged)
- ✅ TELOSCOPE Controls render (DISABLED until PA converged)
- ⏳ Same PA convergence behavior as TELOS

**Test Steps**:
1. Switch to DEVOPS tab
2. Verify debug header appears at top
3. Verify message: "All restrictions removed. Beta mode with full PA extraction and interventions enabled."
4. Follow same testing protocol as TELOS mode
5. Verify all features match TELOS behavior

**Result**: [ ] PASS / [ ] FAIL
**Notes**:

---

## Cross-Mode Tests

### Steward Panel Integration
**Test**: Open Steward panel (if available) in each mode
- Verify render_mode_content() still works in two-column layout
- Verify Steward chat doesn't interfere with mode rendering

**Result**: [ ] PASS / [ ] FAIL

### Mode Switching
**Test**: Switch between modes during active session
- Switch DEMO → BETA → TELOS → DEVOPS
- Verify each mode renders correctly
- Verify no crashes or errors

**Result**: [ ] PASS / [ ] FAIL

### PA Convergence Timing
**Test**: Verify ~10 turn convergence works
- Count exact number of turns until PA converges
- Verify it's approximately 10 turns (±2)

**Result**: [ ] PASS / [ ] FAIL
**Actual convergence turn**: _____

---

## Regression Checks

### Component Modularity
- [ ] observation_deck.py still checks `pa_converged` internally
- [ ] teloscope_controls.py still checks `pa_converged` internally
- [ ] conversation_display.py still checks `pa_converged` for metrics
- [ ] Components work independently of mode layer

### Feature Flag Validation
Verify feature flags in main.py:920-933:
- [ ] `show_devops_header = (mode == "DEVOPS")` - Only DEVOPS shows header
- [ ] `show_observation_deck = (mode in ["BETA", "TELOS", "DEVOPS"])` - DEMO excluded
- [ ] `show_teloscope = (mode in ["TELOS", "DEVOPS"])` - Only TELOS/DEVOPS
- [ ] `telos_demo_mode = (mode == "DEMO")` - Only DEMO has hardcoded PA

---

## Success Criteria Validation

From planning document:

- [ ] **Single unified rendering function used by all modes**
  - Verified: `render_mode_content()` called at main.py:962 and main.py:970

- [ ] **All modes maintain correct behavior**
  - DEMO: Hardcoded PA, no governance features
  - BETA: Progressive PA, Observation Deck only
  - TELOS: Progressive PA, full Observatory
  - DEVOPS: Same as TELOS + debug header

- [ ] **Code reduction achieved**
  - Before: ~76 duplicated lines
  - After: ~32 unified lines
  - Reduction: 58%

- [ ] **PA convergence gates work correctly**
  - UI shows "Calibrating..." until convergence
  - Metrics appear after PA established

- [ ] **Components remain modular**
  - Components check `pa_converged` internally
  - Mode layer only controls visibility

---

## Test Results Summary

**Overall Status**: [ ] PASS / [ ] FAIL

**Issues Found**:
1.
2.
3.

**Recommendations**:
1.
2.
3.

**Tested By**: _____________
**Test Duration**: _____________
**Completion Date**: _____________

---

*Testing checklist generated from structured planning document*
