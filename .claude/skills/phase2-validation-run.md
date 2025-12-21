# Phase 2 Validation Run

Run complete Phase 2 TELOS validation studies on conversation datasets.

## Purpose

This skill executes the full Phase 2 validation pipeline:
1. LLM-at-every-turn primacy attractor establishment
2. Statistical convergence detection (max 10 turns)
3. Drift monitoring (F < 0.8 threshold)
4. Counterfactual branching (original vs TELOS governance)
5. Evidence generation (JSON + Markdown exports)
6. Research brief creation (comprehensive per-study analysis)

## Usage

This skill runs the `phase2_validation_run.py` script with various options.

### Datasets Available

- **ShareGPT** (50 conversations) - Production dataset from ShareGPT
- **Test Sessions** (8 conversations) - Internal test sessions
- **Edge Cases** (8 conversations) - Edge case testing scenarios
- **Custom** - Any ShareGPT-formatted JSON file

### Commands

Run on ShareGPT data:
```bash
cd telos_observatory
source ../venv/bin/activate
python phase2_validation_run.py --sharegpt
```

Run on internal test data:
```bash
cd telos_observatory
source ../venv/bin/activate
python phase2_validation_run.py --test
```

Run on all datasets:
```bash
cd telos_observatory
source ../venv/bin/activate
python phase2_validation_run.py --all
```

Run on custom file:
```bash
cd telos_observatory
source ../venv/bin/activate
python phase2_validation_run.py path/to/data.json study_name
```

### Prerequisites

1. MISTRAL_API_KEY environment variable must be set
2. Virtual environment activated
3. Input data in ShareGPT format:
   ```json
   [
     {
       "id": "conversation_id",
       "turns": [
         ["user message", "assistant message"],
         ["user message", "assistant message"]
       ]
     }
   ]
   ```

### Output Structure

Each run creates a validation directory with:
```
phase2_validation_{study_name}/
  study_results/
    phase2_study_summary.json          # Aggregate results
    {conversation_id}/
      intervention_{branch_id}.json     # Counterfactual evidence
      intervention_{branch_id}.md       # Human-readable evidence
  research_briefs/
    README.md                            # Index of all briefs
    research_brief_{num}_{id}.md        # Individual study brief
```

### Output Metrics

For each study:
- PA establishment metrics (convergence turn, LLM analyses)
- Drift detection (turn, fidelity score)
- Counterfactual results (original F, TELOS F, ΔF)
- Governance effectiveness (% of positive ΔF)

### Example Output

```
PHASE 2 VALIDATION RUN: TEST_SESSIONS
======================================================================
Input: test_data_converted/test_sessions_sharegpt.json
Timestamp: 2025-10-30T19:30:00

STEP 1: RUNNING PHASE 2 STUDIES
======================================================================
STUDY #1: excellent_session_001
  Total turns: 12
  PA ESTABLISHED at turn 7
  DRIFT DETECTED at turn 8 (F=0.598)
  Branches generated: ΔF = +0.127

Results:
  Total conversations: 8
  Completed: 7
  Failed: 1

Drift Analysis:
  Drift detected: 5
  No drift: 2

Governance Effectiveness:
  Average ΔF: +0.045
  Effective: 4/5 (80.0%)
```

## Protocol Guarantees

This skill uses **identical protocol** across all runs:
- Same drift threshold (0.8)
- Same branch length (5 turns)
- Same distance scale (2.0)
- Same LLM-at-every-turn mode
- Same statistical convergence parameters
- Same intervention strategies

This ensures all results are directly comparable.

## When to Use

- Validating TELOS governance on new conversation datasets
- Running internal test suites
- Generating research artifacts for grants/papers
- Comparing governance effectiveness across datasets
- Establishing demonstrable due diligence

## Notes

- API costs: ~$0.50-2.00 per 10 conversations (Mistral small model)
- Runtime: ~2-5 minutes per conversation
- Zero-context interventions: Each intervention is independent
- Statistical rigor: LLM semantic analysis at every turn during PA establishment
