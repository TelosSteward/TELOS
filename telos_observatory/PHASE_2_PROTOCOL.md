# Phase 2 TELOS Study Protocol

**Status**: ✅ READY TO RUN
**Date**: 2025-10-30
**Purpose**: Production validation of TELOS governance with real LLM API calls

---

## Overview

Phase 2 implements the complete TELOS governance validation pipeline using real-world ShareGPT conversations. Unlike Phase 1 (mock testing), this uses **actual Mistral API calls** at every turn to establish primacy attractors, detect drift, and generate counterfactual evidence.

## Critical Requirements

### 1. Primacy Attractor Establishment (Turns 0-10)

**LLM Semantic Analysis at EVERY Turn**:
- Turn 1: LLM asks "What is this conversation about?" → Embed analysis
- Turn 2: LLM asks "What is this conversation about?" → Embed analysis
- Turn 3-N: Continue until statistical convergence

**Convergence Criteria**:
- **Minimum**: 2 consecutive stable turns required
- **Maximum**: Must converge within 10 turns
- **Stability metrics**:
  - Centroid stability ≥ 0.90
  - Variance ≤ 0.20
  - Confidence ≥ 0.70

**Failure Conditions**:
- If PA doesn't converge by turn 10 → **Study VOID**
- Reason: Not enough turns remaining for meaningful drift analysis
- Action: Skip to next conversation

### 2. Drift Monitoring (Post-PA)

After PA established:
- Calculate fidelity for each remaining turn
- **Drift threshold**: F < 0.8
- Need minimum 3 turns post-PA for analysis

### 3. Counterfactual Branching (On Drift)

When drift detected, create TWO independent branches:

**Original Branch**:
- Uses historical user inputs from ShareGPT
- Uses historical assistant responses
- Recalculates metrics for fair comparison

**TELOS Branch** (Counterfactual):
- Uses **same** historical user inputs
- **Generates NEW** assistant responses via Mistral API
- Applies governance intervention on first turn
- Shows what would have happened with TELOS

**Branch Properties**:
- **Length**: 5 turns (configurable)
- **Independence**: Each branch maintains separate conversation history
- **Zero-context interventions**: Each intervention has NO memory of previous interventions
- **Full documentation**: All user inputs and LLM responses saved

### 4. Evidence Export

Each completed study generates:
- **JSON**: Machine-readable complete evidence
- **Markdown**: Human-readable comparison report
- **Metrics**: Turn-by-turn fidelity trajectories
- **LLM Analyses**: All semantic analyses from PA establishment
- **Attractor**: Final established primacy attractor

---

## Architecture

```
ShareGPT Conversation (10-16 turns)
         │
         ├─ Turn 1: LLM analysis → Embed → Statistical check
         ├─ Turn 2: LLM analysis → Embed → Statistical check
         ├─ Turn 3: LLM analysis → Embed → Statistical check
         │                                          │
         │                                    [Not stable yet]
         │                                          │
         ├─ Turn 7: LLM analysis → Embed → ✅ CONVERGED (2 stable turns)
         │                                          │
         │                              PA ESTABLISHED
         │                                          │
         ├─ Turn 8: Calculate fidelity (F=0.92) ✓
         ├─ Turn 9: Calculate fidelity (F=0.75) ⚠️ DRIFT!
         │                                          │
         │                              TRIGGER COUNTERFACTUAL
         │                                          │
         │                          ┌───────────────┴──────────────┐
         │                          │                              │
         │                    ORIGINAL BRANCH                TELOS BRANCH
         │                  (Historical data)          (API-generated)
         │                          │                              │
         │                    Turn 10 (F=0.68)            Turn 10 (F=0.88) 🛡️
         │                    Turn 11 (F=0.62)            Turn 11 (F=0.91)
         │                    Turn 12 (F=0.57)            Turn 12 (F=0.92)
         │                    Turn 13 (F=0.53)            Turn 13 (F=0.94)
         │                    Turn 14 (F=0.50)            Turn 14 (F=0.95)
         │                          │                              │
         └──────────────────────────┴──────────────────────────────┘
                                    │
                         Side-by-Side Evidence
                    (Saved to phase2_study_results/)
```

---

## Implementation Details

### Updated Components

**1. ProgressivePrimacyExtractor** (`telos_purpose/profiling/progressive_primacy_extractor.py`)

New parameters:
```python
llm_per_turn: bool = False  # Enable Phase 2 LLM-at-every-turn mode
max_turns_safety: int = 10  # Changed from 100 for Phase 2
```

When `llm_per_turn=True`:
- Calls `_analyze_with_llm()` at EVERY turn
- Embeds LLM's semantic analysis (not raw conversation)
- Tracks all LLM analyses in `self.llm_analyses`
- Returns full analysis trail on convergence

**2. CounterfactualBranchManager** (`telos_purpose/core/counterfactual_branch_manager.py`)

Unchanged - already implements:
- Historical user inputs + API-generated responses
- Independent branch histories
- Zero-context interventions
- Evidence export (JSON + Markdown)

**3. Phase2StudyRunner** (`telos_observatory/run_phase2_study.py`)

Complete end-to-end pipeline:
- Loads ShareGPT top 25 conversations
- Establishes PA with LLM-per-turn
- Monitors for drift
- Triggers counterfactual branching
- Exports evidence
- Generates summary statistics

---

## Test Data: ShareGPT Top 25

**Source**: `telos_observatory/sharegpt_data/sharegpt_top25_conversations.json`

**Characteristics**:
- 25 English conversations
- 10-16 turns each
- Pre-filtered for fast convergence (7-8 turns typical)
- High quality metrics (avg score 96.7/100)

**Turn Distribution**:
- 16 turns: 2 conversations (8-9 turns post-PA)
- 13-14 turns: 2 conversations (6-7 turns post-PA)
- 10-12 turns: 21 conversations (3-5 turns post-PA)

**All conversations are viable** for Phase 2 testing.

---

## Cost Estimation

### Per Conversation

**PA Establishment** (7-8 turns with LLM):
- 7-8 LLM calls @ ~$0.0002/call = **$0.0014 - $0.0016**

**Counterfactual Branching** (if drift detected, 5 turns):
- 5 LLM calls @ ~$0.0002/call = **$0.001**

**Total per conversation with drift**: ~$0.0025 (quarter of a cent)

### Full Study (25 Conversations)

Assuming ~60% have drift:
- 25 conversations × $0.0014 (PA) = $0.035
- 15 with drift × $0.001 (branching) = $0.015
- **Total: ~$0.05** (5 cents)

**For research-grade evidence across 25 real-world conversations: 5 cents.**

---

## Running Phase 2 Studies

### Prerequisites

1. **Mistral API Key**:
```bash
export MISTRAL_API_KEY='your-key-here'
```

2. **Data Files**:
- `telos_observatory/sharegpt_data/sharegpt_top25_conversations.json` ✅ (exists)

3. **Dependencies**:
```bash
cd ~/Desktop/telos
source venv/bin/activate
```

### Run Single Study (Testing)

```bash
cd ~/Desktop/telos
source venv/bin/activate
python telos_observatory/run_phase2_study.py 1
```

This runs **1 conversation** for testing (~$0.0025).

### Run Full Study (All 25)

```bash
python telos_observatory/run_phase2_study.py 25
```

This runs **all 25 conversations** (~$0.05).

### Monitor Progress

The script outputs:
```
======================================================================
PHASE 2 TELOS STUDY - PRODUCTION VALIDATION
======================================================================

Loaded 25 conversations

======================================================================
STUDY #1: sharegpt_filtered_14
======================================================================
Total turns: 10

📊 Phase 1: Establishing Primacy Attractor (LLM-at-every-turn)...
  Turn 1: accumulating
  Turn 2: accumulating
  Turn 3: analyzing
  Turn 4: analyzing
  Turn 5: analyzing
  Turn 6: analyzing
  Turn 7: converged

✅ PA ESTABLISHED at turn 7
   Purpose: Summarize presentation requirements
   Scope: Digital graphics, technology impact...
   LLM analyses collected: 7

📈 Continuing with 3 remaining turns...

🔍 Phase 2: Monitoring for drift...
  Turn 8: F = 0.921 ✓
  Turn 9: F = 0.743 ⚠️  DRIFT DETECTED!

🌿 Phase 3: Generating counterfactual branches...
✅ Branches generated: intervention_9_142537

📊 Results:
   Original final F: 0.612
   TELOS final F: 0.923
   ΔF: +0.311
   Evidence exported to: phase2_study_results/sharegpt_filtered_14

✅ Study #1 COMPLETED
```

### Output Structure

```
phase2_study_results/
├── phase2_study_summary.json          # Overall study summary
├── sharegpt_filtered_14/              # Evidence for conversation #14
│   ├── intervention_9_142537.json     # Machine-readable evidence
│   └── intervention_9_142537.md       # Human-readable report
├── sharegpt_filtered_35/
│   ├── intervention_11_143022.json
│   └── intervention_11_143022.md
└── ...
```

---

## Evidence Format

### JSON (Machine-Readable)

```json
{
  "branch_id": "intervention_9_142537",
  "trigger_turn": 9,
  "trigger_fidelity": 0.743,
  "original": {
    "turns": [...],
    "final_fidelity": 0.612,
    "avg_fidelity": 0.687
  },
  "telos": {
    "turns": [...],
    "final_fidelity": 0.923,
    "avg_fidelity": 0.901,
    "interventions_applied": [9]
  },
  "comparison": {
    "delta_f": 0.311,
    "governance_effective": true
  }
}
```

### Markdown (Human-Readable)

```markdown
# TELOS Intervention Evidence

## Summary
- Trigger Turn: 9
- Trigger Fidelity: 0.743
- ΔF (Improvement): +0.311
- Governance Effective: Yes

## Turn-by-Turn Comparison

### Turn 10
**User**: [original question]

#### Original Response
[historical response from ShareGPT]
Fidelity: 0.689

#### TELOS Response
[API-generated governed response]
Fidelity: 0.888
🛡️ Intervention Applied

---
[continues for all turns]
```

---

## Success Criteria

A successful Phase 2 study demonstrates:

1. **✅ PA Establishment**: Converges within 10 turns using LLM semantic analysis
2. **✅ Drift Detection**: Identifies when conversations deviate from PA
3. **✅ Intervention Efficacy**: TELOS branch shows higher fidelity than original
4. **✅ Evidence Quality**: Complete audit trail with all LLM analyses
5. **✅ Reproducibility**: Same conversation produces same results

**Target**: 15+ successful studies out of 25 (60% success rate)

---

## Validation Checks

Each study must pass:

### PA Establishment
- [x] Converged within 10 turns
- [x] At least 2 stable consecutive turns
- [x] LLM analyses collected for all turns
- [x] Attractor has purpose, scope, boundaries

### Post-PA Monitoring
- [x] At least 3 turns available after PA
- [x] Fidelity calculated for each turn
- [x] Drift threshold enforced (F < 0.8)

### Counterfactual Branching
- [x] Both branches use same user inputs
- [x] TELOS branch uses API-generated responses
- [x] Intervention applied on first turn only
- [x] Independent conversation histories
- [x] Evidence exported successfully

---

## Troubleshooting

### "PA did not converge within 10 turns"
- Expected for some conversations
- Indicates conversation doesn't have clear primacy
- **Action**: Skip to next conversation (study VOID)

### "Only N turns after PA (need 3+)"
- Conversation too short for meaningful analysis
- **Action**: Skip to next conversation

### "LLM analysis failed"
- API rate limit or timeout
- **Action**: Check MISTRAL_API_KEY, retry with delay

### "Branching failed"
- Usually insufficient remaining turns
- **Action**: Check branch_length parameter

---

## Next Steps After Phase 2

1. **Analyze Results**:
   - Review `phase2_study_summary.json`
   - Calculate average ΔF across all interventions
   - Identify patterns in successful/failed studies

2. **Generate Research Artifacts**:
   - Export evidence for top 5 interventions
   - Create comparison visualizations
   - Draft methods section for paper

3. **Scale Up** (if successful):
   - Run on additional 25 conversations from filtered set
   - Test with different drift thresholds
   - Experiment with branch lengths

---

## Protocol Summary

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| LLM mode | Per-turn | Research-grade evidence |
| Max PA turns | 10 | Balance quality vs conversation length |
| Convergence | 2 stable | Sufficient for statistical confidence |
| Drift threshold | 0.8 | Standard TELOS threshold |
| Branch length | 5 turns | Enough for trajectory evidence |
| Distance scale | 2.0 | Standard TELOS scaling |

---

## Files Reference

- **Study runner**: `telos_observatory/run_phase2_study.py`
- **PA extractor**: `telos_purpose/profiling/progressive_primacy_extractor.py`
- **Branch manager**: `telos_purpose/core/counterfactual_branch_manager.py`
- **Test data**: `telos_observatory/sharegpt_data/sharegpt_top25_conversations.json`
- **Test documentation**: `tests/README.md` (Section: Phase 2 Test Data)
- **Branching guide**: `docs/guides/COUNTERFACTUAL_BRANCHING_GUIDE.md`

---

**Generated by**: TELOSCOPE Observatory
**Purpose**: Phase 2 TELOS production validation
**Status**: Ready for execution
