#!/usr/bin/env python3
"""
Phase 1.5B Wiring Test Script

Tests actual timing implementation and data flow from timestamps to per-turn timing.
"""

import sys
from pathlib import Path

# Add paths
observatory_root = Path(__file__).parent
telos_root = observatory_root.parent / 'telos_purpose'
sys.path.insert(0, str(observatory_root))
sys.path.insert(0, str(telos_root))

print("=" * 70)
print("PHASE 1.5B WIRING TEST")
print("=" * 70)
print()

# Test imports
print("Test 0: Imports")
print("-" * 70)

try:
    from teloscope_v2.utils.baseline_adapter import BaselineAdapter
    print("✅ BaselineAdapter imported")
except ImportError as e:
    print(f"❌ Failed to import BaselineAdapter: {e}")
    sys.exit(1)

try:
    from teloscope_v2.utils.comparison_adapter import ComparisonAdapter
    print("✅ ComparisonAdapter imported")
except ImportError as e:
    print(f"❌ Failed to import ComparisonAdapter: {e}")
    sys.exit(1)

try:
    from teloscope_v2.utils.evidence_exporter import EvidenceExporter
    print("✅ EvidenceExporter imported")
except ImportError as e:
    print(f"❌ Failed to import EvidenceExporter: {e}")
    sys.exit(1)

try:
    from teloscope_v2.utils.runtime_validator import RuntimeValidator
    print("✅ RuntimeValidator imported")
except ImportError as e:
    print(f"❌ Failed to import RuntimeValidator: {e}")
    sys.exit(1)

try:
    # Import from actual telos_purpose locations
    from telos_purpose.core.unified_steward import PrimacyAttractor
    print("✅ PrimacyAttractor imported")
except ImportError as e:
    print(f"❌ Failed to import PrimacyAttractor: {e}")
    sys.exit(1)

# Create simple mock objects
class MockLLMClient:
    """Simple mock LLM for testing"""
    def generate(self, messages, **kwargs):
        return "This is a mock AI response for testing purposes."

class MockEmbeddingProvider:
    """Simple mock embedding provider for testing"""
    def encode(self, text):
        import numpy as np
        # Return simple random embedding
        return np.random.rand(384).astype(np.float32)

# Create mock attractor config
class MockAttractorConfig:
    def __init__(self):
        self.embedding_dim = 384
        self.basin_threshold = 0.3
        # Required attributes for baseline_runners
        self.purpose = ["Test AI assistant purpose"]
        self.scope = ["Testing and validation"]
        self.boundaries = ["No harmful content", "Respect user privacy"]
        self.privacy_level = 0.7
        self.constraint_tolerance = 0.3
        self.task_priority = 0.8

print("✅ Mock dependencies created")

print()

# Create test conversation
print("Test Setup: Creating test conversation")
print("-" * 70)

conversation = [
    ("What is AI?", "AI is artificial intelligence..."),
    ("How does it work?", "AI works by processing data..."),
    ("What are the risks?", "AI risks include bias..."),
    ("Can you explain more?", "Sure, AI is complex..."),
    ("What about the future?", "The future of AI is promising...")
]

print(f"✅ Test conversation created ({len(conversation)} turns)")
print()

# Setup adapters
print("Setup: Creating adapters")
print("-" * 70)

llm = MockLLMClient()
embeddings = MockEmbeddingProvider()
attractor = MockAttractorConfig()

adapter = BaselineAdapter(llm, embeddings, attractor)
comp_adapter = ComparisonAdapter()
exporter = EvidenceExporter(include_validation=True)
validator = RuntimeValidator()

print("✅ All adapters created")
print()

# Test 2.1: Check Baseline Adapter Wiring
print("Test 2.1: Baseline Adapter Wiring")
print("-" * 70)

# Use 'prompt_only' instead of 'telos' to avoid needing complete attractor config
results = adapter.run_baseline(
    'prompt_only',  # Simpler baseline, doesn't need full TELOS config
    conversation,
    track_timing=True,
    track_calibration=True
)

# VERIFICATION 1: Timestamps present in turn results
print("Test 2.1.1: Timestamps present")
passed = True
for i, turn in enumerate(results.turn_results):
    timestamp = turn.get('timestamp')
    if timestamp is None:
        print(f"❌ Turn {i+1} missing timestamp")
        passed = False
    elif not isinstance(timestamp, (int, float)):
        print(f"❌ Turn {i+1} timestamp not numeric: {type(timestamp)}")
        passed = False

if passed:
    print(f"✅ All {len(results.turn_results)} turns have timestamps")
else:
    print("❌ Timestamp check failed")
    sys.exit(1)

# VERIFICATION 2: processing_time_ms calculated (not all same)
print("\nTest 2.1.2: Actual timing (not estimated)")
timings = [t['processing_time_ms'] for t in results.turn_results]

if not all(t is not None for t in timings):
    print("❌ Some turns missing timing")
    sys.exit(1)

unique_timings = len(set(timings))
if unique_timings == 1:
    print(f"❌ All timings identical ({timings[0]:.1f} ms) - still estimated!")
    sys.exit(1)

print(f"✅ Timing varies across turns ({unique_timings} unique values)")
print(f"   Timings: {[f'{t:.1f}' for t in timings]}")

# VERIFICATION 3: Cumulative timing makes sense
print("\nTest 2.1.3: Cumulative timing")
passed = True
for i, turn in enumerate(results.turn_results):
    if i > 0:
        prev_cumulative = results.turn_results[i-1]['cumulative_time_ms']
        curr_cumulative = turn['cumulative_time_ms']
        if curr_cumulative <= prev_cumulative:
            print(f"❌ Cumulative not increasing at turn {i+1}")
            passed = False

if passed:
    print("✅ Cumulative timing increases")
else:
    sys.exit(1)

# VERIFICATION 4: Total matches cumulative
print("\nTest 2.1.4: Total matches cumulative")
last_cumulative = results.turn_results[-1]['cumulative_time_ms']
total_ms = results.metadata['total_processing_time_ms']

if abs(last_cumulative - total_ms) >= 100:
    print(f"❌ Total {total_ms:.1f} ms != Cumulative {last_cumulative:.1f} ms")
    sys.exit(1)

print(f"✅ Total ({total_ms:.1f} ms) matches cumulative ({last_cumulative:.1f} ms)")
print()

# Test 2.2: Check Calibration Tracking
print("Test 2.2: Calibration Tracking")
print("-" * 70)

passed = True
for i, turn in enumerate(results.turn_results):
    turn_num = turn['turn']
    calibration = turn['calibration_phase']
    attractor_established = turn['primacy_attractor_established']

    # First 3 turns should be calibration
    if turn_num <= 3:
        if calibration != True:
            print(f"❌ Turn {turn_num} should be calibration")
            passed = False
        if attractor_established != False:
            print(f"❌ Turn {turn_num} attractor shouldn't be established")
            passed = False
    else:
        if calibration != False:
            print(f"❌ Turn {turn_num} shouldn't be calibration")
            passed = False
        if attractor_established != True:
            print(f"❌ Turn {turn_num} attractor should be established")
            passed = False

    status = "✅" if (turn_num <= 3 and calibration) or (turn_num > 3 and not calibration) else "❌"
    print(f"{status} Turn {turn_num}: calibration={calibration}, attractor={attractor_established}")

if passed:
    print("✅ Calibration tracking correct")
else:
    print("❌ Calibration tracking failed")
    sys.exit(1)

print()

# Test 2.3: Check Context Size Tracking
print("Test 2.3: Context Size Tracking")
print("-" * 70)

passed = True
for i, turn in enumerate(results.turn_results):
    context_size = turn['context_size']
    expected_size = i  # Turn N has N turns in history (0-indexed)

    if context_size != expected_size:
        print(f"❌ Turn {i+1}: context={context_size}, expected={expected_size}")
        passed = False
    else:
        print(f"✅ Turn {i+1}: context_size={context_size}")

if passed:
    print("✅ Context size tracking correct")
else:
    print("❌ Context size tracking failed")
    sys.exit(1)

print()

# Test 2.4: Runtime Validator Integration
print("Test 2.4: Runtime Validator Integration")
print("-" * 70)

validator_input = {'turn_results': results.turn_results}

all_passed = validator.validate_runtime_simulation(validator_input)

if not all_passed:
    print("❌ Runtime validation failed")
    sys.exit(1)

print("✅ Runtime validation passed")

# Check detailed report
report = validator.generate_validation_report(validator_input)
if 'RUNTIME SIMULATION VERIFIED' not in report:
    print("❌ Report doesn't show verification")
    sys.exit(1)

print("✅ Validation report generated")

# Check timing summary
timing = validator.get_timing_summary(validator_input)
if timing['turn_count'] == 0:
    print("❌ No turns in timing summary")
    sys.exit(1)

if timing['total_ms'] == 0:
    print("❌ Total time is zero")
    sys.exit(1)

print(f"✅ Timing summary: {timing['total_ms']:.1f} ms total, {timing['avg_ms']:.1f} ms avg")
print()

# Test 2.5: Evidence Export Integration
print("Test 2.5: Evidence Export Integration")
print("-" * 70)

import json

# Create comparison (need baseline and another)
results_baseline = adapter.run_baseline('stateless', conversation, track_timing=True)
results_prompt_only = adapter.run_baseline('prompt_only', conversation, track_timing=True)

baseline_branch = comp_adapter.convert_baseline_result_to_branch(results_baseline)
prompt_only_branch = comp_adapter.convert_baseline_result_to_branch(results_prompt_only)
comparison = comp_adapter.compare_results(baseline_branch, prompt_only_branch)

# Test JSON export
json_str = exporter.export_comparison(comparison, format='json')
json_data = json.loads(json_str)

if 'runtime_validation' not in json_data:
    print("❌ JSON missing runtime_validation")
    sys.exit(1)

if not json_data['runtime_validation'].get('all_tests_passed'):
    print("❌ Validation failed in export")
    sys.exit(1)

if 'timing_summary' not in json_data['runtime_validation']:
    print("❌ JSON missing timing_summary")
    sys.exit(1)

print("✅ JSON export includes validation")

# Test Markdown export
md_str = exporter.export_comparison(comparison, format='markdown')

if '## Runtime Simulation Verification' not in md_str:
    print("❌ Markdown missing verification section")
    sys.exit(1)

if 'Runtime Simulation VERIFIED' not in md_str:
    print("❌ Markdown doesn't show verification")
    sys.exit(1)

if 'Methodology Statement' not in md_str:
    print("❌ Markdown missing methodology")
    sys.exit(1)

print("✅ Markdown export includes validation")
print()

# Summary
print("=" * 70)
print("WIRING TEST SUMMARY")
print("=" * 70)
print()
print("✅ All wiring tests PASSED")
print()
print("Verified:")
print("  ✅ Timestamps present in all turns")
print("  ✅ Per-turn timing varies (actual, not estimated)")
print("  ✅ Cumulative timing increases correctly")
print("  ✅ Calibration phase tracking works")
print("  ✅ Context size tracking works")
print("  ✅ Runtime validator integration works")
print("  ✅ Evidence exports include validation")
print()
print("Status: 🎉 WIRING VERIFIED - Ready for display testing")
print("=" * 70)
