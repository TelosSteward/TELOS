# TELOS Counterfactual Branching System - Complete Guide

## 🎯 THE Core TELOS Demonstration Feature

This is the system that transforms TELOS from "drift detection" to **"active governance with concrete evidence"**.

## What This System Does

When drift is detected during session replay, TELOS creates **TWO independent branches using REAL Mistral API calls**:

### 1. **Original Branch** (Historical)
- Uses the ACTUAL historical responses from the session
- Shows what DID happen without intervention
- Metrics recalculated for fair comparison

### 2. **TELOS Branch** (Counterfactual)
- Generates NEW responses via Mistral API
- Applies governance intervention on first turn
- Shows what WOULD have happened with TELOS

**Both branches**:
- Use the SAME user inputs (fair comparison)
- Process the SAME number of turns (configurable, default 5)
- Calculate REAL metrics using actual embeddings
- Are completely independent (no contamination)

## For Grants and Papers: The Evidence Format

Instead of saying:
> ❌ "We detected drift 2 times"

You can now say:
> ✅ "At turn 12, fidelity dropped to 0.73 due to off-topic discussion. TELOS triggered intervention.
> Over the next 5 turns:
> - **Original branch**: Fidelity degraded (0.73 → 0.65 → 0.58 → 0.51 → 0.48)
> - **TELOS branch**: Intervention corrected drift (0.73 → 0.89 → 0.92 → 0.91 → 0.93)
> - **ΔF (improvement)**: +0.45
>
> Complete conversation text, metrics, and side-by-side comparison attached as evidence."

This is **ACTIVE GOVERNANCE** with **MEASURABLE IMPACT**.

## System Architecture

```
Session Replay
     │
     ├─ Process Turn 1 → F = 0.95 ✓
     ├─ Process Turn 2 → F = 0.87 ✓
     ├─ Process Turn 3 → F = 0.72 ⚠️ DRIFT!
     │                          │
     │                          └─ TRIGGER COUNTERFACTUAL
     │                                     │
     │                          ┌──────────┴──────────┐
     │                          │                     │
     │                    ORIGINAL BRANCH      TELOS BRANCH
     │                    (Historical)         (API Generated)
     │                          │                     │
     │                    Turn 4 (F=0.65)      Turn 4 (F=0.89) 🛡️ INTERVENTION
     │                    Turn 5 (F=0.58)      Turn 5 (F=0.92)
     │                    Turn 6 (F=0.51)      Turn 6 (F=0.91)
     │                    Turn 7 (F=0.48)      Turn 7 (F=0.93)
     │                    Turn 8 (F=0.45)      Turn 8 (F=0.94)
     │                          │                     │
     └──────────────────────────┴─────────────────────┘
                                    │
                          Side-by-Side Evidence
                       (Downloadable JSON + Markdown)
```

## Key Files

### Core Implementation
**`telos_purpose/core/counterfactual_branch_manager.py`** (680 lines)
- `CounterfactualBranchManager`: Main class for API-based branching
- `trigger_counterfactual()`: Generates both branches with real API calls
- `_generate_original_branch()`: Historical responses + recalculated metrics
- `_generate_telos_branch()`: API-generated responses + intervention
- `_apply_telos_intervention()`: Governance correction via API call
- `export_evidence()`: JSON and Markdown export with full evidence

### Dashboard Integration
**`telos_purpose/dev_dashboard/streamlit_live_comparison.py`**
- Line 21: Import `CounterfactualBranchManager`
- Line 183-192: Initialize manager + branch tracking
- Line 607-653: Trigger logic (when F < 0.8 detected)
- Line 719-847: Side-by-side visualization + download buttons

## How It Works: Step-by-Step

### Phase 1: Drift Detection
```python
# During session replay, at each turn:
if fidelity < 0.8:
    st.warning("⚠️ DRIFT DETECTED")

    # Get remaining turns from session
    remaining_turns = [(user_msg, historical_response), ...]

    # Get attractor center for metrics
    attractor_center = steward.attractor_center

    # TRIGGER BRANCHING
    branch_id = branch_manager.trigger_counterfactual(...)
```

### Phase 2: Branch Generation

#### Original Branch
```python
for (user_input, historical_response) in remaining_turns:
    # Use ACTUAL historical response
    response = historical_response

    # Calculate REAL metrics
    response_emb = embeddings.encode(response)
    distance = np.linalg.norm(response_emb - attractor_center)
    fidelity = 1.0 - (distance / distance_scale)

    # Store
    turns.append(BranchTurn(...))
```

#### TELOS Branch
```python
conversation_history = [...]  # Up to drift point

for i, (user_input, _ignored_historical) in enumerate(remaining_turns):
    # Add user input
    conversation_history.append({"role": "user", "content": user_input})

    # Generate NEW response via API
    raw_response = mistral_api.generate(conversation_history)

    # Apply intervention on first turn after drift
    if i == 0:
        corrected_response = _apply_telos_intervention(raw_response)
        response = corrected_response
    else:
        response = raw_response

    # Add to history for next turn
    conversation_history.append({"role": "assistant", "content": response})

    # Calculate REAL metrics
    response_emb = embeddings.encode(response)
    fidelity = calculate_fidelity(response_emb, attractor_center)

    # Store
    turns.append(BranchTurn(...))
```

### Phase 3: Evidence Generation

Both branches are compared and displayed:
- **Metrics**: Original final F, TELOS final F, ΔF
- **Trajectories**: Fidelity over time for both branches
- **Full text**: All user inputs and responses
- **Interventions**: Which turns had governance applied
- **Downloads**: JSON and Markdown with complete evidence

## Usage Guide

### 1. Enable Counterfactuals (Default: ON)

In `config.json`:
```json
{
  "enable_counterfactuals": true,
  "branch_length": 5,
  "drift_threshold": 0.8
}
```

### 2. Load a Conversation

Dashboard → **Session Replay** tab → Upload conversation file (.txt, .json, .md)

### 3. Start Replay

Click "▶️ Start Replay" - the system will:
- Process each turn with pristine context
- Calculate fidelity in real-time
- Detect drift (F < 0.8)
- **Automatically trigger counterfactual branching**

### 4. View Results

After replay completes, scroll down to see:
- **Counterfactual Interventions Generated** section
- Summary metrics for each intervention
- Expandable side-by-side comparison
- Download buttons for evidence

### 5. Download Evidence

Two formats available:

#### JSON (Machine-Readable)
```json
{
  "branch_id": "intervention_12_142537",
  "trigger_turn": 12,
  "trigger_fidelity": 0.73,
  "original": {
    "turns": [...],
    "final_fidelity": 0.48,
    "avg_fidelity": 0.55
  },
  "telos": {
    "turns": [...],
    "final_fidelity": 0.93,
    "avg_fidelity": 0.92
  },
  "comparison": {
    "delta_f": 0.45,
    "governance_effective": true
  }
}
```

#### Markdown (Human-Readable Report)
```markdown
# TELOS Intervention Evidence

## Intervention Summary
- Trigger Turn: 12
- Trigger Fidelity: 0.73
- ΔF (Improvement): +0.45

## Turn-by-Turn Comparison

### Turn 13
**User**: [original question]

#### Original Response
[historical response]
Fidelity: 0.65

#### TELOS Response
[API-generated corrected response]
Fidelity: 0.89
🛡️ Intervention Applied: boundary_correction

---
[... continues for all turns ...]
```

## API Usage and Costs

Each counterfactual trigger makes **N Mistral API calls** where N = `branch_length` (default: 5).

**Example**: If 2 drift events occur during a session replay:
- 2 interventions × 5 turns each = **10 API calls total**
- Cost depends on Mistral pricing (typically ~$0.001 per call)
- **Total cost**: ~$0.01 per session with 2 interventions

**This is incredibly cheap** for the evidence you get!

## Evidence Quality

### What Makes This High-Quality Evidence?

1. **REAL API Calls** - Not simulated, not mocked, actual Mistral API responses
2. **Fair Comparison** - Both branches use identical user inputs
3. **Real Metrics** - Actual embeddings and fidelity calculations
4. **Complete Context** - Full conversation history preserved
5. **Independent Branches** - No contamination between paths
6. **Downloadable** - JSON + Markdown for papers/grants
7. **Reproducible** - Same session replay produces same triggers

### Grant Application Language

You can use this in proposals:

> "TELOS includes a counterfactual branching system that generates API-based evidence of governance efficacy. When drift is detected during conversation analysis, the system creates two independent conversation branches: one showing what happened historically, and one showing what would have happened with TELOS intervention. Both branches use real LLM API calls with identical user inputs, providing quantifiable, reproducible evidence of alignment improvement. Each intervention produces downloadable evidence packages including full conversation text, metrics trajectories, and side-by-side comparisons suitable for peer review and auditing."

## Advanced Configuration

### Adjust Branch Length
```python
# In config.json
{
  "branch_length": 10  # Generate 10 turns per branch (more evidence, higher cost)
}
```

### Multiple Intervention Points
The system automatically handles multiple drift events in the same session:
- Each intervention is independent
- Separate branches for each trigger point
- All evidence preserved separately
- No contamination between interventions

### Custom Intervention Logic

To modify how interventions work, edit:
```python
# telos_purpose/core/counterfactual_branch_manager.py
def _apply_telos_intervention(self, response, user_input, ...):
    # Customize the intervention prompt here
    intervention_prompt = f"""
    [Your custom intervention logic]
    """
    return self.llm.generate(intervention_prompt)
```

## Troubleshooting

### "No remaining turns" Error
**Cause**: Drift detected on last turn (no future turns to branch)
**Solution**: Drift must occur at least N turns before end (where N = branch_length)

### "Attractor center is None"
**Cause**: Progressive mode hasn't established baseline yet
**Solution**: Only triggers after baseline established (typically 3-5 turns)

### API Rate Limits
**Cause**: Too many API calls in short time
**Solution**: Add delays or reduce `branch_length`

## Future Enhancements

Potential additions (not yet implemented):
- **Async generation**: Generate branches in background
- **Multiple models**: Compare GPT vs Claude vs Mistral interventions
- **Intervention types**: Different correction strategies
- **Batch export**: Export all interventions as single PDF report
- **Visualization**: Plot fidelity trajectories with Plotly
- **Quality scoring**: Rate intervention effectiveness automatically

## Summary

The counterfactual branching system is **THE demonstration** of TELOS governance.

**Without it**: You have drift detection (observer)
**With it**: You have active governance with measurable impact (controller)

This is the evidence you need for:
- Grant applications
- Paper submissions
- Regulatory compliance
- Governance audits
- Research demonstrations

It's real, it's reproducible, and it's available **right now** in the dashboard.

---

**Questions?** Check the inline documentation in:
- `count erfactual_branch_manager.py`
- `streamlit_live_comparison.py` (lines 607-847)
