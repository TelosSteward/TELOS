# Primacy State Formalization - Hypothesis Testing Framework

**Date:** November 15, 2025
**Purpose:** Structured testing framework for validating PS = ρ_PA · (2·F_user·F_AI)/(F_user+F_AI)
**Status:** Ready for feasibility execution

---

## Overview

This document provides a structured framework for testing 4 key hypotheses about Primacy State formalization. Each hypothesis has:
- Clear statement of what we're testing
- Rationale for why we think it's true
- Specific test methodology
- Data requirements
- Success criteria (quantitative thresholds)
- Implications if true/false

**Success Criteria for GO Decision:** ≥2 of 4 hypotheses supported + computational feasibility

---

## H1: PA Correlation Predicts Primacy State Stability

### Hypothesis Statement

> **H1:** Higher PA correlation (ρ_PA > 0.90) leads to more stable Primacy State, evidenced by ≥30% lower variance in ΔV_dual across turns compared to low correlation (ρ_PA < 0.70) sessions.

### Rationale

**Why we think this is true:**
1. If User PA and AI PA are well-synchronized (high ρ_PA), responses should consistently satisfy both
2. Misaligned attractors create conflicting pulls on system state → unstable trajectories
3. Stable attractor coupling should produce stable energy landscape → predictable dynamics
4. Dual PA validation showed perfect 1.0000 ρ_PA correlation with perfect stability

**Physics analogy:** Like coupled oscillators - when in phase (high correlation), system is stable; when out of phase (low correlation), chaotic

### Test Method

```sql
-- Group sessions by PA correlation level
SELECT
  CASE
    WHEN AVG(pa_correlation) >= 0.90 THEN 'high'
    WHEN AVG(pa_correlation) < 0.70 THEN 'low'
    ELSE 'medium'
  END as correlation_level,
  VARIANCE(delta_v_dual) as stability_variance,
  COUNT(*) as sample_size
FROM (
  SELECT session_id, turn_number, pa_correlation, delta_v_dual
  FROM governance_deltas_ps_test
) AS turn_data
GROUP BY session_id
```

**Analysis:**
1. Calculate ΔV variance for each session
2. Group sessions by mean ρ_PA (high >= 0.90, low < 0.70)
3. Compare variance distributions
4. Compute correlation coefficient between ρ_PA and ΔV variance across all sessions

### Data Required

**From feasibility test:**
- ρ_PA for each session (mean across turns)
- ΔV_dual for each turn (to compute variance)
- Minimum 5 sessions per group (high/low) for statistical power

**From existing dual PA validation:**
- 46 sessions available
- Turn-by-turn dual PA embeddings
- Can compute ρ_PA retroactively

### Success Criterion

**H1 is SUPPORTED if:**
- ✅ Variance reduction ≥ 30% (high correlation vs. low correlation sessions)
- ✅ Correlation coefficient |r| > 0.60 (between ρ_PA and ΔV variance)
- ✅ Statistical significance p < 0.05 (t-test or Mann-Whitney U)

**Quantitative example:**
```
High correlation sessions (ρ_PA >= 0.90): variance = 0.010
Low correlation sessions (ρ_PA < 0.70):  variance = 0.015
Reduction = (0.015 - 0.010) / 0.015 = 33.3% ✓ PASS
```

### Implications if TRUE

**What this enables:**
1. ✅ **PA correlation as early warning indicator**
   - Can predict session stability from initialization
   - Low ρ_PA at session start → flag for monitoring

2. ✅ **Intervention targeting**
   - If ρ_PA drops mid-session, focus on realigning attractors
   - Suggests "PA synchronization" as intervention strategy

3. ✅ **Formulation validation**
   - Confirms ρ_PA should be multiplicative gate in PS formula
   - Justifies computational overhead of correlation tracking

**Strategic value:**
- Strengthens mathematical foundation (dynamical coupling proven)
- Provides predictive capability (not just reactive)
- Good for academic credibility (testable prediction confirmed)

### Implications if FALSE

**What we learn:**
1. ⚠️ **ρ_PA may not be right metric**
   - Perhaps different coupling measure needed
   - Or PA alignment measured differently (angle vs. cosine?)

2. ⚠️ **Simplification opportunity**
   - If ρ_PA doesn't predict stability, can drop from formula
   - PS = (2·F_user·F_AI)/(F_user+F_AI) may be sufficient
   - Reduces computational complexity

3. ⚠️ **Alternative explanations**
   - Stability may depend on other factors (conversation domain, user intent drift)
   - PA correlation may be stable but not predictive

**Doesn't invalidate PS, just suggests formula refinement**

---

## H2: Primacy State Decomposition Enables Earlier Drift Detection

### Hypothesis Statement

> **H2:** PS component failure (F_user < 0.70 OR F_AI < 0.70) occurs 2-5 turns BEFORE overall PS crosses intervention threshold (PS < 0.70), enabling preventive correction in ≥60% of drift cases.

### Rationale

**Why we think this is true:**
1. Dual PA provides two independent drift signals (User PA, AI PA)
2. Conversation drift (User PA) may precede response quality issues
3. Role drift (AI PA) may precede boundary violations
4. Combined PS averages these → may mask early component failures
5. Monitoring components separately should detect drift earlier

**Example scenario:**
```
Turn 15: F_user = 0.68 (user purpose drifting), F_AI = 0.90 (AI fine)
         PS = 0.95 · (2·0.68·0.90)/(0.68+0.90) = 0.77 → MONITOR (no action)

Turn 17: F_user = 0.65, F_AI = 0.85
         PS = 0.95 · (2·0.65·0.85)/(0.65+0.85) = 0.71 → CORRECT

Turn 18: F_user = 0.60, F_AI = 0.80
         PS = 0.95 · (2·0.60·0.80)/(0.60+0.80) = 0.65 → INTERVENE

Component monitoring caught F_user < 0.70 at turn 15 (3 turns earlier)
```

### Test Method

```python
def test_early_detection():
    lead_times = []

    for session in sessions:
        for i, turn in enumerate(session.turns):
            # Find PS intervention events
            if turn.ps_score < 0.70:
                # Look backward for component failure
                for j in range(i-1, max(-1, i-10), -1):
                    prev = session.turns[j]

                    if prev.f_user < 0.70 or prev.f_ai < 0.70:
                        lead_time = i - j
                        lead_times.append(lead_time)
                        break

    mean_lead = np.mean(lead_times)
    early_rate = len([t for t in lead_times if t >= 2]) / len(lead_times)

    return mean_lead, early_rate
```

**Analysis:**
1. Identify all PS < 0.70 intervention events
2. For each event, look backward to find when components first failed
3. Measure lead time (turns between component failure and PS failure)
4. Calculate percentage with ≥2 turn lead time

### Data Required

**From feasibility test:**
- Turn-by-turn F_user, F_AI, PS for all sessions
- Intervention events (PS < 0.70)
- Component-level failure history

**Sample size:**
- Need ≥20 intervention events for statistical validity
- Dual PA validation showed minimal drift → may have few events
- May need to analyze sessions with intentional drift scenarios

### Success Criterion

**H2 is SUPPORTED if:**
- ✅ Mean lead time ≥ 2.5 turns
- ✅ Early detection rate ≥ 60% (component failure 2+ turns before PS failure)
- ✅ Standard deviation shows consistent pattern (not random)

**Quantitative example:**
```
Intervention events: 25
Component early warning: 17
Lead time ≥ 2 turns: 16

Early detection rate = 16/25 = 64% ✓ PASS
Mean lead time = 3.2 turns ✓ PASS
```

### Implications if TRUE

**What this enables:**
1. ✅ **Preventive intervention**
   - Can intervene before combined PS drops critically
   - Reduces severe drift cases (PS < 0.50)

2. ✅ **Component-specific corrections**
   - If F_user failing: Remind user purpose
   - If F_AI failing: Correct AI role behavior
   - More targeted than generic "increase fidelity" correction

3. ✅ **Justifies dual PA complexity**
   - Component monitoring provides measurable benefit
   - Worth computational overhead if enables earlier detection

**Strategic value:**
- Better user experience (fewer severe drift events)
- Stronger regulatory evidence (proactive not reactive)
- Good for validation studies (can demonstrate predictive capability)

### Implications if FALSE

**What we learn:**
1. ⚠️ **Components fail simultaneously**
   - No early warning from decomposition
   - Combined PS and component failures coincide

2. ⚠️ **Still valuable for diagnostics**
   - Even if not predictive, decomposition shows WHAT failed
   - Useful for post-hoc analysis and debugging

3. ⚠️ **May simplify to single metric**
   - If no predictive benefit, could track only combined PS
   - Compute F_user/F_AI on-demand for diagnostics when PS fails

**Doesn't invalidate PS, but reduces justification for real-time component tracking**

---

## H3: Energy-Based Convergence Predicts Intervention Success

### Hypothesis Statement

> **H3:** ΔV_dual < 0 (dual potential energy decreasing) predicts intervention success (fidelity improvement ≥ 0.15) with >75% accuracy, representing ≥10% improvement over single PA ΔV prediction accuracy.

### Rationale

**Why we think this is true:**
1. Dual potential captures more complete system dynamics than single PA
2. Single PA ΔV only tracks distance from one attractor
3. Dual PA ΔV also tracks:
   - Distance from both attractors (V_user + V_AI)
   - Attractor coupling stability (V_coupling)
4. More information → better prediction

**Physics analogy:** Like predicting whether a ball will settle in a valley - dual PA tracks both gravitational potential AND landscape geometry

### Test Method

```python
def test_convergence_prediction():
    predictions = {'correct': 0, 'total': 0}

    for session in sessions:
        for i in range(1, len(session.turns)):
            curr = session.turns[i]
            prev = session.turns[i-1]

            # Check if intervention was successful
            if curr.ps_score - prev.ps_score >= 0.15:
                predictions['total'] += 1

                # Did ΔV_dual predict this?
                if prev.delta_v_dual < 0:  # Convergence predicted
                    predictions['correct'] += 1

    accuracy = predictions['correct'] / predictions['total']
    return accuracy
```

**Comparison to baseline:**
```python
# Also test single PA ΔV (if available)
baseline_accuracy = test_single_pa_delta_v_prediction()
improvement = accuracy - baseline_accuracy
```

### Data Required

**From feasibility test:**
- V_dual and ΔV_dual for each turn
- Pre/post intervention states
- Success metric: PS improvement ≥ 0.15

**Baseline comparison:**
- Single PA ΔV (from original dual PA validation)
- Or use typical accuracy (~65%) from literature

**Sample size:**
- Need ≥30 intervention events for 75% accuracy confidence
- May require broader dataset if few interventions in 46 sessions

### Success Criterion

**H3 is SUPPORTED if:**
- ✅ ΔV_dual prediction accuracy > 75%
- ✅ Improvement over baseline ≥ 10%
- ✅ Statistical significance (binomial test p < 0.05)

**Quantitative example:**
```
Intervention success events: 40
Correctly predicted by ΔV_dual < 0: 32

Accuracy = 32/40 = 80% ✓ PASS
Baseline (single PA ΔV) = 68%
Improvement = 80% - 68% = 12% ✓ PASS
```

### Implications if TRUE

**What this enables:**
1. ✅ **Optimal intervention timing**
   - Can use ΔV_dual to decide when to intervene
   - If ΔV > 0 persistently, intervene sooner
   - If ΔV < 0, system may self-correct

2. ✅ **Intervention success forecasting**
   - Predict whether correction will work before applying
   - If ΔV_dual predicts failure, escalate immediately

3. ✅ **Justifies energy tracking**
   - V_dual computation has measurable predictive value
   - Worth ~10ms overhead if enables better decisions

**Strategic value:**
- More efficient governance (intervene when likely to succeed)
- Reduces unnecessary interventions (if ΔV < 0, wait)
- Good for validation studies (can measure prediction accuracy)

### Implications if FALSE

**What we learn:**
1. ⚠️ **Single PA ΔV sufficient**
   - Dual potential doesn't improve prediction
   - Can drop V_dual tracking, keep only PS score

2. ⚠️ **Alternative predictors needed**
   - Maybe other metrics better predict success (topic shift, user intent change)
   - Or intervention success is inherently unpredictable

3. ⚠️ **Simplification opportunity**
   - Focus on PS score for decisions, skip energy tracking
   - Reduces computational overhead (~10ms savings)

**Doesn't invalidate PS, just suggests dropping V_dual component**

---

## H4: Primacy State Improves Stakeholder Interpretability

### Hypothesis Statement

> **H4:** Non-technical stakeholders (grant reviewers, institutional partners, compliance officers) rate PS-based narratives as "clearer" or "more interpretable" than current fidelity-based narratives, with ≥70% preference rate and ≥4.0/5.0 average clarity rating.

### Rationale

**Why we think this is true:**
1. PS decomposition tells a story: "User purpose drifted" vs. "AI role violated"
2. Current fidelity score is opaque: "0.73" doesn't explain WHAT failed
3. Stakeholders need interpretable evidence for:
   - Grant applications (explain governance clearly)
   - Institutional adoption (troubleshooting guidance)
   - Regulatory compliance (audit trail clarity)

**Example comparison:**
```
Current: "Fidelity score dropped to 0.73 at turn 18."
         → What does 0.73 mean? What failed? Why?

PS:      "Primacy State violated (PS = 0.611). User purpose maintained
         (F_user = 0.90), but AI role drifted (F_AI = 0.50)."
         → Clear: AI behavior is the problem, not conversation topic
```

### Test Method

**Survey Design:**
1. Create 10 narrative pairs (current vs. PS)
2. Blind reviewers to which is which
3. Ask: "Which is clearer?" "Which would you prefer in audit documentation?"
4. Collect 5-point clarity ratings

**Reviewers:**
- 5-7 grant reviewers (mix technical/non-technical)
- 3-5 institutional partners (deployment decision-makers)
- 2-3 compliance/regulatory experts

**Total: 10-15 reviewers for statistical validity**

### Data Required

**From feasibility test:**
- Generate 10 representative narrative pairs
- Cover different failure modes:
  - User PA drift
  - AI PA drift
  - PA misalignment
  - Both PAs failing
  - Perfect alignment

**Sample narratives in:** `test_primacy_state_feasibility.py` (H4 test generates these)

### Success Criterion

**H4 is SUPPORTED if:**
- ✅ Stakeholder preference ≥ 70% for PS narratives
- ✅ Average clarity rating ≥ 4.0/5.0 for PS (vs. < 3.5 for current)
- ✅ No major confusion or misinterpretation reported

**Quantitative example:**
```
Reviewers: 12
Prefer PS narratives: 9
Preference rate = 9/12 = 75% ✓ PASS

Clarity ratings (PS): 4.3/5.0 ✓ PASS
Clarity ratings (current): 3.1/5.0
```

### Implications if TRUE

**What this enables:**
1. ✅ **Stronger grant applications**
   - Can explain governance clearly to reviewers
   - Evidence is interpretable, not just numeric

2. ✅ **Easier institutional adoption**
   - Partners can understand what system is doing
   - Troubleshooting guidance more actionable

3. ✅ **Better regulatory compliance**
   - Audit trails are stakeholder-ready
   - Compliance officers can verify governance without deep technical knowledge

**Strategic value:**
- Reduces "explainability" barrier to adoption
- Positions TELOS as not just mathematically rigorous but also accessible
- Good for GMU partnership (can explain to non-technical stakeholders)

### Implications if FALSE

**What we learn:**
1. ⚠️ **Current narratives sufficient**
   - Stakeholders don't need decomposition
   - Simple fidelity score is clear enough

2. ⚠️ **May need narrative refinement**
   - PS concept is good but language needs improvement
   - Simplify templates, reduce jargon

3. ⚠️ **Dual reporting strategy**
   - Technical audiences: PS decomposition
   - Non-technical audiences: Simplified fidelity narratives
   - Both available, user chooses level of detail

**Suggests PS valuable for internal diagnostics, but external reporting stays simplified**

---

## Summary: Success Criteria Decision Matrix

### GO Decision (Proceed to Implementation)

**Required:**
- ✅ Computational feasibility (p95 < 50ms)
- ✅ At least 2 of 4 hypotheses supported
- ✅ No critical failures (numerical instability, severe performance issues)

**Ideal (Strong GO):**
- ✅ 3-4 hypotheses supported
- ✅ Stakeholder feedback very positive
- ✅ Clear improvements over current approach

### NO-GO Decision (Do Not Implement)

**Reasons:**
- ❌ Computational infeasibility (p95 > 100ms)
- ❌ 0-1 hypotheses supported
- ❌ Critical technical failures
- ❌ Stakeholder confusion increases

### DEFER Decision (Needs More Work)

**Reasons:**
- ⚠️ 2 hypotheses supported but marginal (just barely passing thresholds)
- ⚠️ Computational borderline (50-75ms)
- ⚠️ Mixed stakeholder feedback
- ⚠️ Suggests alternative formulation may work better

**Action:** Refine PS formula, try different approaches, re-test

---

## Timeline for Hypothesis Testing

### Week 1: Feasibility Test Execution
- Run `test_primacy_state_feasibility.py` on 46 sessions
- Generate computational performance metrics
- Compute H1, H2, H3 results automatically

### Week 2: Stakeholder Survey (H4)
- Distribute narrative pairs to 10-15 reviewers
- Collect clarity ratings and preferences
- Analyze qualitative feedback

### Week 2: Decision Point
- Compile all results
- Apply decision matrix
- Make GO / NO-GO / DEFER decision based on evidence

---

## Next Steps After Hypothesis Testing

### If GO:
1. Week 3-4: Supabase schema + state manager implementation
2. Week 4: Delta interpreter + narrative templates
3. Week 5: Whitepaper Section 2.3
4. Week 6: BETA integration decision

### If NO-GO:
1. Document lessons learned
2. Update Memory MCP (research status: Abandoned, rationale: insufficient evidence)
3. Consider alternatives (simplified formulas, different approaches)

### If DEFER:
1. Analyze which hypotheses failed and why
2. Propose alternative formulations
3. Design refined tests
4. Re-run feasibility phase with adjustments

---

**Document Status:** COMPLETE - Ready for testing
**Next Action:** Execute `test_primacy_state_feasibility.py`
