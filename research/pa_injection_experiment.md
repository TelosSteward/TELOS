# PA Injection Experiment — Parrot vs. Understanding

**Status:** DESIGNED (not yet executed)
**Design date:** 2026-02-26
**Source:** A10 review team consensus (Nell experimental design, Bengio safety finding)
**Infrastructure:** PAContext (A10-T2), alternating toggle (A10-T3), pa_injected covariate (A10-T1)

---

## 1. Research Question

Does PA system prompt injection improve governance fidelity scores through genuine constraint guidance, or does it inflate scores uniformly (parroting)?

**H0 (null):** PA injection has no differential effect — scores increase uniformly across all event types, including boundary violations.

**H1 (alternative):** PA injection improves scores for in-scope events but NOT for boundary violations, indicating genuine constraint guidance rather than surface-level compliance mimicry.

---

## 2. Data Source

**Existing baseline (WITHOUT PA injection):**
- `research/backtest_forensic_dataset.jsonl` — 2,071 events, 11 scorable
- 10 ESCALATE, 1 EXECUTE
- Mean composite fidelity: 0.4392
- All 11 scorable events are tool dispatches accessing `.git/` paths

**Treatment condition (WITH PA injection):**
- Same 11 scorable events, re-scored with PA injection block in the scoring context
- PAContext loaded from `templates/openclaw.yaml`
- Full injection block (~12k chars) used

---

## 3. Experimental Design

### 3.1 Within-Subject Comparison

Each of the 11 scorable events is scored under both conditions:
- **Condition A (baseline):** Score without PA in context (existing backtest results)
- **Condition B (injected):** Score with PA injection block prepended to action text

This is a within-subject design — each event serves as its own control.

### 3.2 Measurement

For each event under both conditions, record:
- Composite fidelity score (primary outcome)
- 5 dimension scores (purpose, scope, boundary, tool, chain)
- Verdict (EXECUTE/CLARIFY/SUGGEST/INERT/ESCALATE)
- `pa_injected: bool` covariate

### 3.3 Analysis

**Parrot detection logic:**

| In-scope scores | Boundary violation scores | Interpretation |
|-----------------|--------------------------|----------------|
| Increase | Increase equally | PARROTING — injection inflates all scores |
| Increase | No change or decrease | UNDERSTANDING — injection provides genuine guidance |
| No change | No change | NO EFFECT — injection has no impact on scoring |
| Decrease | Decrease | INTERFERENCE — injection confuses scoring |

**Statistical test:**
- Wilcoxon signed-rank test on paired composite scores (N=11, non-parametric)
- Separate tests for in-scope events (EXECUTE verdicts) vs. boundary events (ESCALATE verdicts)
- Effect size: matched-pairs rank-biserial correlation
- Note: N=11 has low statistical power. Results are directional, not confirmatory.

### 3.4 Event Classification for Analysis

From the backtest results:
- **Boundary events (expected ESCALATE):** 10 events accessing `.git/` paths
  - `.git/config`, `.git/refs/heads/main`, `.git/logs/HEAD`, `.git/COMMIT_EDITMSG`
  - These SHOULD remain ESCALATE under PA injection (the PA boundaries prohibit config access)
- **In-scope events (expected EXECUTE):** 1 event (Glob on `.py` files)
  - This SHOULD remain EXECUTE or improve under PA injection

### 3.5 Failure Modes to Watch

1. **Score ceiling effect:** If baseline scores are already near threshold boundaries, injection may push all events across decision thresholds without differential effect
2. **Action text contamination:** If the PA injection block is prepended to action text, the scoring engine may match PA keywords against PA boundaries (self-referential scoring)
3. **Embedding space crowding:** The PA injection block may shift the action embedding toward the purpose centroid for ALL actions, not just in-scope ones

---

## 4. Implementation

```python
# Pseudocode — do not run yet
from telos_governance.pa_context import PAContext
from analysis.run_backtest import load_jsonl_events, classify_event

ctx = PAContext.from_config("templates/openclaw.yaml")
pa_block = ctx.get_injection_block()

for event in scorable_events:
    # Condition A: existing baseline scores (from backtest)
    baseline_composite = event["fidelity"]["composite"]
    baseline_verdict = event["verdict"]

    # Condition B: re-score with PA injection
    action_text_injected = f"{pa_block}\n\n{event['action_text']}"
    result = engine.score_action(
        action_text=action_text_injected,
        tool_name=event["tool_call"],
        tool_args=event["tool_args"],
    )
    injected_composite = result.composite_fidelity
    injected_verdict = result.decision.value

    # Record both conditions
    record_comparison(event, baseline_composite, injected_composite,
                      baseline_verdict, injected_verdict)
```

---

## 5. Decision Criteria

| Outcome | Action |
|---------|--------|
| PARROTING detected | Do NOT use PA injection for scoring. Keep injection for pre-generation context only. Scoring must remain PA-independent to avoid measurement inflation. |
| UNDERSTANDING detected | PA injection improves governance. Recalibrate thresholds for injected vs. non-injected conditions. Document threshold delta. |
| NO EFFECT | PA injection is inert for scoring. No threshold change needed. Consider removing injection from scoring path to save tokens. |
| INTERFERENCE | Investigate — likely action text contamination. Fix scoring pipeline to separate PA context from action text. |

---

## 6. Threshold Recalibration

If PA injection produces a consistent score inflation (e.g., +0.05 across all events), the optimizer thresholds need adjustment:

- Measure mean delta: `delta = mean(injected_composite - baseline_composite)`
- If delta > 0.02: recalibrate all decision thresholds by adding delta
- Re-run optimizer with PA injection enabled to find new optimal thresholds
- Document both threshold sets (with/without injection) for reproducibility

---

## 7. Limitations

- **N=11:** Extremely small sample. Results are hypothesis-generating, not confirmatory.
- **Homogeneous events:** 10/11 events are `.git/` path accesses — limited diversity.
- **Scoring vs. generation:** This experiment measures PA injection's effect on the TELOS scoring engine, NOT on the model's generation behavior. The generation effect requires a separate study with live agent sessions.
- **No alternating-treatment yet:** This first experiment uses static comparison (all baseline vs. all injected). The alternating-treatment design (A10-T3) is for the longitudinal study.

---

## 8. Follow-up Studies

1. **Live agent study:** Run OpenClaw with alternating PA injection (A10-T3 toggle) on real tasks. Compare governance verdicts, boundary violation rates, and task completion quality.
2. **Larger N:** Generate synthetic scorable events or collect more live data to achieve N >= 50 for meaningful statistical power.
3. **Multi-PA:** Test with different PA configs (healthcare, civic) to check generalizability.
4. **Threshold sensitivity:** Sweep injection block size (compressed vs. full) to find minimum effective injection.

*Designed per research/t_pa_injection.md Task 5 specification.*
