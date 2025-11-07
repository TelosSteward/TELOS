# TELOS Test Evidence

This folder contains validation evidence from TELOS governance testing.

---

## Test Files

### `MockTest0_ValidationPassed.json`

**Status**: ✅ PASSED
**Date**: 2025-10-30
**Type**: Phase 1 Wiring Verification (Mock LLM)

**Purpose**:
- Verify baseline adapter wiring
- Validate runtime simulation architecture
- Test evidence export infrastructure
- Confirm all 5 validation tests pass

**Key Results**:
- ✅ All 5 runtime validation tests passed
- ✅ No Future Context: Historical context only
- ✅ Sequential Timestamps: Strictly increasing
- ✅ Timing Recorded: Per-turn timing tracked
- ✅ Context Growth: Incremental growth verified
- ✅ Empty Initial Context: Turn 0 starts empty

**Timing Summary**:
- Total: 0.457 ms
- Average: 0.152 ms per turn
- Range: 0.096 - 0.250 ms

**Note**: This test uses mock LLM and mock embeddings. The negative ΔF (-9.347) is expected and does not indicate governance failure - it validates that the wiring works correctly. Real API testing (Phase 2) will show actual governance efficacy.

---

## Test Categories

### Phase 1: Wiring Verification (Mock)
- **Purpose**: Validate infrastructure without API costs
- **Components**: Mock LLM, mock embeddings, mock attractor
- **Cost**: Free
- **Duration**: < 1 second
- **Files**: `MockTest0_ValidationPassed.json`

### Phase 2: Real API Validation
- **Purpose**: Validate actual governance with real LLM
- **Components**: Mistral API, semantic embeddings, real attractor
- **Cost**: ~$0.01 per test
- **Duration**: 30-60 seconds
- **Test Set**: ShareGPT Quality-Filtered Set (Top 25)
- **Files**: See Phase 2 Test Data below

---

## Validation Criteria

All tests must pass these 5 runtime validation checks:

1. **No Future Context**: Turn N can only see turns 0 to N-1
2. **Sequential Timestamps**: Timestamps strictly increasing
3. **Timing Recorded**: Processing time recorded for each turn
4. **Context Growth**: Context grows incrementally (0 → 1 → 2 → ...)
5. **Empty Initial Context**: Turn 0 starts with no history

---

## File Structure

```json
{
  "test_name": "Mock Test 0 - Validation Passed",
  "test_date": "2025-10-30",
  "status": "PASSED",
  "runtime_validation": {
    "all_tests_passed": true,
    "tests_passed": 5,
    "total_tests": 5,
    "timing_summary": {...}
  },
  "full_comparison": {
    "baseline": {
      "turn_results": [...]
    },
    "telos": {
      "turn_results": [...]
    }
  }
}
```

---

## Phase 2 Test Data

### ShareGPT Quality-Filtered Set (Top 25)

**Set ID**: `SHAREGPT_TOP25`
**Status**: ✅ Ready for Phase 2 Testing
**Created**: 2025-10-30
**Source**: ShareGPT Dataset (philschmid/sharegpt-raw)

**Purpose**:
Replace inadequate 3-turn tests with real-world conversations that allow proper primacy attractor formation through statistical convergence and semantic analysis.

**Selection Process**:

1. **Statistical Filtering** (50 conversations from ~90,000)
   - Conversation length: 10-25 turns
   - Convergence detection: Window-based centroid stability
   - Target convergence: 7-13 turns (10 ± 3)
   - Criteria:
     - Centroid stability ≥ 0.90
     - Variance ≤ 0.20
     - Confidence ≥ 0.70
   - Processing: ~9.5 minutes for 50 conversations
   - Cost: $0 (local embeddings only)

2. **Quality Analysis** (Top 25 from 50)
   - Language filter: English only (46/50 qualified)
   - Quality scoring (0-100 scale):
     - Convergence speed: 40% weight
     - Centroid stability: 30% weight
     - Variance stability: 30% weight
   - Selected: Top 25 by composite score
   - Average quality: 96.7/100
   - Score range: 91.2 - 99.4

**Dataset Characteristics**:

- **Size**: 25 conversations
- **Language**: 100% English
- **Turn range**: 10-16 turns per conversation
- **Convergence**: 7-8 turns (optimal fast convergence)
- **Quality**: High stability and low variance
- **Top conversation**: ID `sharegpt_filtered_14`, score 99.4

**Files**:

- `telos_observatory/sharegpt_data/sharegpt_top25_conversations.json` - Final test set
- `telos_observatory/sharegpt_data/quality_analysis_report.json` - Detailed analysis
- `telos_observatory/sharegpt_data/quality_analysis_report.md` - Human-readable report
- `telos_observatory/sharegpt_data/sharegpt_filtered_conversations.json` - Full 50 filtered set
- `telos_observatory/sharegpt_data/convergence_statistics.json` - Filtering statistics

**Why This Test Set**:

1. **Real-world conversations**: Actual human-AI dialogues from ShareGPT
2. **Proper context length**: 10-25 turns vs inadequate 3-turn tests
3. **Fast convergence**: Primacy attractors form within 7-8 turns
4. **High quality**: Statistical stability metrics indicate strong attractors
5. **Cost-effective**: Pre-filtered without expensive LLM API calls
6. **Diverse topics**: Range of conversation types and purposes

**Expected Test Behavior**:

- **Turns 0-6**: Calibration phase, semantic analysis, NO math
- **Turn 7-8**: Statistical convergence detected, attractor finalized
- **Turn 9+**: Mathematical tracking with established attractor
- **Validation**: All 5 runtime checks must pass (no future context, sequential timestamps, etc.)

**Next Steps**:

1. Wire `ProgressivePrimacyExtractor` into `baseline_runners.py`
2. Replace static attractor initialization with dynamic formation
3. Run Phase 2 tests with top 25 conversations
4. Generate Phase 2 evidence files
5. Compare against baseline (no governance)

---

## Using This Evidence

This evidence can be used for:
- ✅ Research papers (runtime methodology validation)
- ✅ Grant applications (proof of concept)
- ✅ Peer review (reproducible validation)
- ✅ Governance audits (compliance verification)
- ✅ Development milestones (integration checkpoints)

---

**Generated by**: TELOSCOPE Observatory v2
**Framework**: TELOS Purpose Framework
**Validation**: Phase 1.5B - Runtime Simulation Architecture
