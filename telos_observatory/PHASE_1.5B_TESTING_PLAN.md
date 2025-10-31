# Phase 1.5B - Testing Plan

**Date**: 2025-10-30
**Status**: 🧪 TESTING IN PROGRESS
**Phases**: Display → Wiring → Execution Validation

---

## Testing Overview

### Three-Phase Testing Approach:

1. **Display Testing** - Visual verification in Observatory UI
2. **Wiring Testing** - Data flow validation (timestamps → actual timing)
3. **Execution Validation** - Implementation meets requirements

---

## Phase 1: Display Testing

**Objective**: Verify all UI elements render correctly

### Test 1.1: Observatory App Launches
**Location**: `http://localhost:8502`

**Expected**:
- ✅ App launches without errors
- ✅ No import errors in console
- ✅ All tabs/sections visible

**How to verify**:
```bash
# Check app is running
curl http://localhost:8502

# Check for errors in Streamlit output
# Should see: "You can now view your Streamlit app"
```

---

### Test 1.2: Counterfactual Demo Visible
**Location**: Test 9 in Observatory UI

**Expected**:
- ✅ "9️⃣ Counterfactual Analysis Demo" section exists
- ✅ Expandable section renders
- ✅ Mock comparison data displays

**How to verify**:
1. Open Observatory at `http://localhost:8502`
2. Scroll to "9️⃣ Counterfactual Analysis Demo"
3. Expand the section
4. Verify comparison summary displays

---

### Test 1.3: Comparison Summary Displays
**Expected sections**:
- ✅ "📊 Comparison Summary" header
- ✅ 4 metric columns:
  - Baseline Final F
  - TELOS Final F
  - ΔF (Improvement)
  - Avg Improvement
- ✅ Effectiveness badge (green/red/yellow)
- ✅ Statistical Analysis section (if available)
- ✅ **NEW: Runtime Simulation Verification section**

**How to verify**:
1. Check all 4 metrics display
2. Verify ΔF shows +/- sign
3. Check effectiveness badge color matches ΔF sign
4. **Look for "🔍 Runtime Simulation Verification" section**
5. **Verify green checkmark: "✅ Runtime Simulation VERIFIED"**

---

### Test 1.4: Runtime Validation Section
**Expected**:
- ✅ "🔍 Runtime Simulation Verification" header
- ✅ Green success banner: "✅ Runtime Simulation VERIFIED - All 5 tests passed"
- ✅ Expandable "📋 View Validation Details" section
- ✅ 5 test results with green checkmarks
- ✅ Timing summary (3 metrics)
- ✅ Methodology statement

**How to verify**:
1. Locate validation section below statistical analysis
2. Verify success banner is green
3. Expand validation details
4. Count 5 test results (all with ✅)
5. Check timing summary shows:
   - Total Time
   - Avg per Turn
   - Min/Max
6. Verify methodology statement present

---

### Test 1.5: Evidence Export Buttons
**Expected**:
- ✅ "📄 Download JSON Evidence" button
- ✅ "📝 Download Markdown Report" button
- ✅ Both buttons clickable

**How to verify**:
1. Locate export buttons below comparison viewer
2. Verify both buttons visible
3. Click each button (downloads should trigger)

---

## Phase 2: Wiring Testing

**Objective**: Verify data flows from timestamps → actual timing

### Test 2.1: Check Baseline Adapter Wiring
**File**: `baseline_adapter.py`

**Test Code**:
```python
# Create test conversation
conversation = [
    ("What is AI?", "AI is artificial intelligence..."),
    ("How does it work?", "AI works by processing data..."),
    ("What are the risks?", "AI risks include bias...")
]

# Run with timing tracking
from teloscope_v2.utils.baseline_adapter import BaselineAdapter
from telos_purpose.llm.mock_llm import MockLLMClient
from telos_purpose.embeddings.mock_embeddings import MockEmbeddingProvider
from telos_purpose.primacy.attractor import PrimacyAttractorConfig

llm = MockLLMClient()
embeddings = MockEmbeddingProvider()
attractor = PrimacyAttractorConfig()

adapter = BaselineAdapter(llm, embeddings, attractor)

# Run baseline
results = adapter.run_baseline(
    'telos',
    conversation,
    track_timing=True,
    track_calibration=True
)

# VERIFICATION 1: Timestamps present in turn results
print("Test 2.1.1: Timestamps present")
for i, turn in enumerate(results.turn_results):
    timestamp = turn.get('timestamp')
    assert timestamp is not None, f"Turn {i} missing timestamp"
    assert isinstance(timestamp, (int, float)), f"Turn {i} timestamp not numeric"
print("✅ All turns have timestamps")

# VERIFICATION 2: processing_time_ms calculated (not all same)
print("\nTest 2.1.2: Actual timing (not estimated)")
timings = [t['processing_time_ms'] for t in results.turn_results]
assert all(t is not None for t in timings), "Some turns missing timing"
assert len(set(timings)) > 1, "All timings identical (still estimated!)"
print(f"✅ Timing varies across turns: {[f'{t:.1f}' for t in timings]}")

# VERIFICATION 3: Cumulative timing makes sense
print("\nTest 2.1.3: Cumulative timing")
for i, turn in enumerate(results.turn_results):
    if i > 0:
        prev_cumulative = results.turn_results[i-1]['cumulative_time_ms']
        curr_cumulative = turn['cumulative_time_ms']
        assert curr_cumulative > prev_cumulative, f"Cumulative not increasing at turn {i}"
print("✅ Cumulative timing increases")

# VERIFICATION 4: Total matches cumulative
print("\nTest 2.1.4: Total matches cumulative")
last_cumulative = results.turn_results[-1]['cumulative_time_ms']
total_ms = results.metadata['total_processing_time_ms']
assert abs(last_cumulative - total_ms) < 100, f"Total {total_ms} != Cumulative {last_cumulative}"
print(f"✅ Total ({total_ms:.1f} ms) matches cumulative ({last_cumulative:.1f} ms)")
```

**Expected output**:
```
Test 2.1.1: Timestamps present
✅ All turns have timestamps

Test 2.1.2: Actual timing (not estimated)
✅ Timing varies across turns: ['145.2', '98.3', '102.7']

Test 2.1.3: Cumulative timing
✅ Cumulative timing increases

Test 2.1.4: Total matches cumulative
✅ Total (346.2 ms) matches cumulative (346.2 ms)
```

---

### Test 2.2: Check Calibration Tracking
**Test Code**:
```python
# Verify calibration phase tracking
print("Test 2.2: Calibration tracking")

for i, turn in enumerate(results.turn_results):
    turn_num = turn['turn']
    calibration = turn['calibration_phase']
    attractor_established = turn['primacy_attractor_established']

    # First 3 turns should be calibration
    if turn_num <= 3:
        assert calibration == True, f"Turn {turn_num} should be calibration"
        assert attractor_established == False, f"Turn {turn_num} attractor shouldn't be established"
    else:
        assert calibration == False, f"Turn {turn_num} shouldn't be calibration"
        assert attractor_established == True, f"Turn {turn_num} attractor should be established"

    print(f"Turn {turn_num}: calibration={calibration}, attractor={attractor_established} ✅")

print("✅ Calibration tracking correct")
```

**Expected output**:
```
Test 2.2: Calibration tracking
Turn 1: calibration=True, attractor=False ✅
Turn 2: calibration=True, attractor=False ✅
Turn 3: calibration=True, attractor=False ✅
Turn 4: calibration=False, attractor=True ✅
...
✅ Calibration tracking correct
```

---

### Test 2.3: Check Context Size Tracking
**Test Code**:
```python
# Verify context size tracking
print("Test 2.3: Context size tracking")

for i, turn in enumerate(results.turn_results):
    context_size = turn['context_size']
    expected_size = i  # Turn N has N turns in history (0-indexed)

    assert context_size == expected_size, f"Turn {i+1}: context={context_size}, expected={expected_size}"
    print(f"Turn {i+1}: context_size={context_size} ✅")

print("✅ Context size tracking correct")
```

**Expected output**:
```
Test 2.3: Context size tracking
Turn 1: context_size=0 ✅
Turn 2: context_size=1 ✅
Turn 3: context_size=2 ✅
...
✅ Context size tracking correct
```

---

### Test 2.4: Runtime Validator Integration
**Test Code**:
```python
from teloscope_v2.utils.runtime_validator import RuntimeValidator

print("Test 2.4: Runtime validator integration")

validator = RuntimeValidator()

# Convert results to validator format
validator_input = {'turn_results': results.turn_results}

# Run all validation tests
all_passed = validator.validate_runtime_simulation(validator_input)

assert all_passed == True, "Runtime validation failed"
print("✅ Runtime validation passed")

# Check detailed report
report = validator.generate_validation_report(validator_input)
assert 'RUNTIME SIMULATION VERIFIED' in report, "Report doesn't show verification"
print("✅ Validation report generated")

# Check timing summary
timing = validator.get_timing_summary(validator_input)
assert timing['turn_count'] > 0, "No turns in timing summary"
assert timing['total_ms'] > 0, "Total time is zero"
print(f"✅ Timing summary: {timing['total_ms']:.1f} ms total, {timing['avg_ms']:.1f} ms avg")
```

**Expected output**:
```
Test 2.4: Runtime validator integration
✅ No Future Context: All 3 turns have correct historical context only
✅ Sequential Timestamps: All 3 timestamps strictly increasing
✅ Timing Recorded: All 3 turns have processing time recorded
✅ Context Growth: Context grows incrementally turn-by-turn
✅ Empty Initial Context: Turn 0 starts with empty context
✅ Runtime validation passed
✅ Validation report generated
✅ Timing summary: 346.2 ms total, 115.4 ms avg
```

---

### Test 2.5: Evidence Export Integration
**Test Code**:
```python
from teloscope_v2.utils.comparison_adapter import ComparisonAdapter
from teloscope_v2.utils.evidence_exporter import EvidenceExporter
import json

print("Test 2.5: Evidence export integration")

# Create comparison (need baseline and TELOS)
results_baseline = adapter.run_baseline('stateless', conversation, track_timing=True)
results_telos = adapter.run_baseline('telos', conversation, track_timing=True)

comp_adapter = ComparisonAdapter()
baseline_branch = comp_adapter.convert_baseline_result_to_branch(results_baseline)
telos_branch = comp_adapter.convert_baseline_result_to_branch(results_telos)
comparison = comp_adapter.compare_results(baseline_branch, telos_branch)

# Export with validation
exporter = EvidenceExporter(include_validation=True)

# Test JSON export
json_str = exporter.export_comparison(comparison, format='json')
json_data = json.loads(json_str)

assert 'runtime_validation' in json_data, "JSON missing runtime_validation"
assert json_data['runtime_validation']['all_tests_passed'] == True, "Validation failed in export"
assert 'timing_summary' in json_data['runtime_validation'], "JSON missing timing_summary"
print("✅ JSON export includes validation")

# Test Markdown export
md_str = exporter.export_comparison(comparison, format='markdown')

assert '## Runtime Simulation Verification' in md_str, "Markdown missing verification section"
assert 'Runtime Simulation VERIFIED' in md_str, "Markdown doesn't show verification"
assert 'Methodology Statement' in md_str, "Markdown missing methodology"
print("✅ Markdown export includes validation")
```

**Expected output**:
```
Test 2.5: Evidence export integration
✅ JSON export includes validation
✅ Markdown export includes validation
```

---

## Phase 3: Execution Validation

**Objective**: Verify implementation meets all requirements

### Test 3.1: Requirement Checklist

**Core Requirements**:
- [x] ✅ Sequential processing (verified in baseline_runners.py)
- [x] ✅ Historical context only (context_size = turn_num - 1)
- [x] ✅ **Actual per-turn timing** (from timestamp deltas)
- [x] ✅ Timing tracked separately (baseline and TELOS independent)
- [x] ✅ Calibration phase tracking (first 3 turns)
- [x] ✅ Context size tracking (for validation)
- [x] ✅ Validation tests (5 tests implemented)
- [x] ✅ Validation in exports (JSON and Markdown)
- [x] ✅ Validation in UI (comparison viewer)

**How to verify**:
Run all tests in Phase 2. All should pass.

---

### Test 3.2: Research Validity

**Methodology Requirements**:
- [x] ✅ No batch analysis artifacts
- [x] ✅ No future knowledge leakage
- [x] ✅ Embeddings calculated at runtime
- [x] ✅ Timing measured not estimated
- [x] ✅ Validation automated

**How to verify**:
```python
# Check no batch analysis
assert 'processing_pattern' in results.metadata
assert results.metadata['processing_pattern'] == 'sequential'

# Check runtime simulation verified
assert 'runtime_simulation_verified' in results.metadata
assert results.metadata['runtime_simulation_verified'] == True

# Check validation passed
validator_input = {'turn_results': results.turn_results}
assert validator.validate_runtime_simulation(validator_input) == True
```

---

### Test 3.3: Performance Characteristics

**Expected behavior**:
- ✅ Per-turn timing varies (shows real processing variance)
- ✅ First turn may be slower (initialization overhead)
- ✅ Intervention turns may show overhead
- ✅ Total time = sum of per-turn times

**Test Code**:
```python
print("Test 3.3: Performance characteristics")

timings = [t['processing_time_ms'] for t in results.turn_results]

# Check variation exists
variation = max(timings) - min(timings)
assert variation > 0, "No timing variation (still estimated)"
print(f"✅ Timing variation: {variation:.1f} ms range")

# Check total matches sum
total_sum = sum(timings)
total_reported = results.metadata['total_processing_time_ms']
assert abs(total_sum - total_reported) < 100, "Total doesn't match sum"
print(f"✅ Total time matches sum: {total_reported:.1f} ms")

# Show timing distribution
print(f"\nTiming distribution:")
print(f"  Min: {min(timings):.1f} ms")
print(f"  Avg: {sum(timings)/len(timings):.1f} ms")
print(f"  Max: {max(timings):.1f} ms")
```

---

### Test 3.4: End-to-End Integration

**Full workflow test**:
```python
print("Test 3.4: End-to-end integration")

# 1. Generate test data
from teloscope_v2.utils.mock_data import generate_enhanced_session

session = generate_enhanced_session(
    turns=12,
    session_type='high-drift',
    include_annotations=True
)
print("✅ Test data generated")

# 2. Convert to conversation
conversation = adapter.convert_session_to_conversation(session)
assert len(conversation) > 0
print(f"✅ Converted to conversation ({len(conversation)} turns)")

# 3. Run comparison
comparison_results = adapter.run_comparison(
    conversation,
    baseline_type='stateless',
    track_timing=True,
    track_calibration=True
)
print("✅ Comparison run complete")

# 4. Verify both branches have timing
for branch_name in ['baseline', 'telos']:
    branch = comparison_results[branch_name]
    assert 'total_processing_time_ms' in branch.metadata
    for turn in branch.turn_results:
        assert 'processing_time_ms' in turn
        assert 'calibration_phase' in turn
        assert 'context_size' in turn
print("✅ Both branches have complete tracking")

# 5. Convert and compare
baseline_branch = comp_adapter.convert_baseline_result_to_branch(comparison_results['baseline'])
telos_branch = comp_adapter.convert_baseline_result_to_branch(comparison_results['telos'])
comparison = comp_adapter.compare_results(baseline_branch, telos_branch)
print("✅ Comparison generated")

# 6. Validate
validator_input = {'turn_results': telos_branch['turn_results']}
assert validator.validate_runtime_simulation(validator_input) == True
print("✅ Runtime validation passed")

# 7. Export
json_data = exporter.export_comparison(comparison, format='json')
md_data = exporter.export_comparison(comparison, format='markdown')
assert 'runtime_validation' in json.loads(json_data)
assert 'Runtime Simulation Verification' in md_data
print("✅ Evidence exports include validation")

print("\n🎉 END-TO-END TEST PASSED")
```

---

## Testing Execution Plan

### Step 1: Display Testing (Manual)
1. Open Observatory: `http://localhost:8502`
2. Navigate to Test 9: Counterfactual Analysis Demo
3. Verify UI elements (checklist above)
4. Take screenshots for documentation

### Step 2: Wiring Testing (Automated)
1. Create test script: `test_wiring.py`
2. Run all wiring tests (Test 2.1 - 2.5)
3. Verify all assertions pass
4. Save test output

### Step 3: Execution Validation (Automated)
1. Run requirement checklist (Test 3.1)
2. Run research validity tests (Test 3.2)
3. Run performance tests (Test 3.3)
4. Run end-to-end integration (Test 3.4)
5. Generate validation report

---

## Success Criteria

### Phase 1: Display ✅
- [ ] Observatory launches without errors
- [ ] Counterfactual demo visible
- [ ] Runtime validation section displays
- [ ] All 5 validation tests shown
- [ ] Timing summary displays
- [ ] Export buttons work

### Phase 2: Wiring ✅
- [ ] Timestamps present in all turns
- [ ] Per-turn timing varies (not all same)
- [ ] Cumulative timing increases
- [ ] Calibration tracking correct
- [ ] Context size tracking correct
- [ ] Runtime validator passes all tests
- [ ] Evidence exports include validation

### Phase 3: Execution ✅
- [ ] All requirements met
- [ ] Research validity confirmed
- [ ] Performance characteristics reasonable
- [ ] End-to-end integration works

---

## Next Steps After Testing

### If All Tests Pass:
1. ✅ Mark Phase 1.5B as **VERIFIED**
2. ✅ Update status documents
3. ✅ Proceed to V1.00 (wire real LLM client)

### If Tests Fail:
1. Document failure details
2. Identify root cause
3. Implement fix
4. Re-run tests
5. Verify fix

---

## Testing Checklist

**Pre-Testing**:
- [x] ✅ Observatory app running
- [ ] Browser open to `http://localhost:8502`
- [ ] Test script ready (`test_wiring.py`)

**During Testing**:
- [ ] Phase 1: Display testing complete
- [ ] Phase 2: Wiring testing complete
- [ ] Phase 3: Execution validation complete

**Post-Testing**:
- [ ] All tests documented
- [ ] Screenshots captured
- [ ] Test output saved
- [ ] Validation report generated
- [ ] Phase 1.5B status updated

---

**Testing Status**: 🧪 READY TO BEGIN

**Next Action**: Verify Observatory display (Phase 1)

---

**End of Testing Plan**
