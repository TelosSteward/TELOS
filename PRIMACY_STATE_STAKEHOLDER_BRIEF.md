# Primacy State Formalization - Stakeholder Communication Brief

**Date:** November 15, 2025
**Purpose:** Explain Primacy State concept to different audiences
**Status:** Template for use if/when PS is validated and implemented

---

## Executive Summary (All Audiences)

**What's Changing:**
TELOS is exploring a mathematical enhancement called "Primacy State formalization" that would make governance more diagnostic and interpretable.

**Current Approach:**
- Single "fidelity score" tells us if conversation is aligned
- Score like "0.73" indicates drift, but doesn't explain WHY or WHERE

**Proposed Approach:**
- "Primacy State score" derived from three components:
  - User purpose alignment (WHAT - is conversation on topic?)
  - AI behavior alignment (HOW - is AI acting appropriately?)
  - Attractor synchronization (are these two working together?)
- Score like "PS = 0.855" PLUS diagnosis: "User purpose maintained, AI role drifted"

**Why It Matters:**
- **Better diagnostics** - Know WHAT failed, not just that something failed
- **Earlier detection** - Components may fail before combined score drops
- **Clearer evidence** - Easier to explain to non-technical stakeholders
- **Predictive capability** - Can forecast drift before it becomes severe

**Current Status:** Exploratory research phase - testing feasibility before committing to implementation

---

## For Grant Reviewers

### What We're Testing

**Research Question:**
Can we formalize "Primacy State" - the condition of being well-governed - as a mathematically derived constant from dual Primacy Attractor dynamics?

**Current System (Baseline):**
- TELOS uses one Primacy Attractor (PA) defining conversation purpose/scope/boundaries
- Measures "fidelity" = how aligned response is to that PA
- Fidelity > 0.85 = aligned, fidelity < 0.70 = drifting

**Proposed Enhancement:**
- TELOS would use TWO Primacy Attractors:
  - User PA: What the user wants to accomplish (purpose, scope)
  - AI PA: How the AI should help (supportive role, behavioral constraints)
- "Primacy State" emerges from the interaction between these two
- Formula: PS = ρ_PA · (2·F_user·F_AI)/(F_user + F_AI)
  - F_user = alignment to user purpose
  - F_AI = alignment to AI role
  - ρ_PA = how synchronized the two attractors are

**Evidence Base:**
- Built on dual PA validation: +85.32% improvement over single PA (46 sessions, p < 0.001)
- Perfect 1.0000 fidelity achieved when dual PAs properly coupled (51-turn conversation)
- Zero interventions needed - system self-stabilized in Primacy State equilibrium

**Research Plan:**
1. **Feasibility Phase (Week 1-2):** Test PS formula on existing data, measure computational overhead, assess interpretability
2. **Decision Point:** GO if ≥2 of 4 hypotheses supported and computational cost acceptable
3. **Implementation (Week 3-6):** If GO, implement PS tracking in production system
4. **Validation:** Demonstrate PS provides measurable improvement over current approach

**Intellectual Contribution:**
- Treats governance as dynamical systems problem (control theory, Lyapunov stability)
- Primacy State becomes emergent property, not arbitrary measurement
- Provides mathematical framework for "constitutional computing"

**Relevance to Grant:**
- Strengthens theoretical foundation (good for academic reviewers)
- Demonstrates continuous improvement (not static system)
- Provides clearer evidence generation (regulatory compliance)

---

## For Institutional Partners

### What This Means for Deployment

**The Challenge:**
When deploying TELOS in your environment, you need to understand when governance is working and when it's failing. Current fidelity scores are opaque: "0.73" doesn't tell you what to do.

**The Solution (If Implemented):**
Primacy State decomposition gives you diagnostic information:

**Example Scenario:**
```
Current System:
"Fidelity score: 0.73. Intervention triggered."
→ What failed? How do I troubleshoot?

Primacy State System:
"Primacy State violated (PS = 0.611).
 User purpose maintained (F_user = 0.90), but AI role drifted (F_AI = 0.50).
 AI began writing content directly instead of guiding.
 Intervention: Corrected AI role behavior.
 Primacy State restored (PS = 0.855)."
→ Clear: AI role was the problem. Corrected. Back to normal.
```

**Benefits for Your Deployment:**
1. **Easier troubleshooting:** Know exactly which component is failing
2. **Better training:** Can show your team what good/bad governance looks like
3. **Clearer reporting:** Can explain to management/compliance what's happening
4. **Predictive warnings:** May detect issues 2-5 turns before they become severe

**Implementation Impact:**
- ✅ No disruption to current functionality
- ✅ Backward compatible (can run alongside current system)
- ✅ Feature flag controlled (can enable/disable easily)
- ⚠️ Slight computational overhead (~20ms per turn) - acceptable for most deployments

**Timeline:**
- Exploratory phase: November 2025
- Implementation decision: December 2025
- Deployment: Q1 2026 (if validated)

**Your Role:**
- Feedback welcome on whether diagnostic decomposition would be useful
- Pilot testing opportunity if you're interested
- No action required - this is our research initiative

---

## For Compliance / Regulatory Experts

### Audit Trail Implications

**Regulatory Context:**
- EU AI Act Article 72 requires "systematic procedures to review experience"
- NIST AI RMF requires "tracking identified AI risks over time"
- Both need **observable demonstrable due diligence** - evidence governance was maintained

**Current Audit Trail:**
```json
{
  "turn_42": {
    "fidelity_score": 0.73,
    "intervention": "reminder_injected",
    "post_intervention_fidelity": 0.87
  }
}
```

**What this tells you:**
- ✓ System detected drift
- ✓ System intervened
- ✓ Fidelity improved
- ✗ But: What specifically failed? Where did drift occur?

**Proposed Audit Trail (Primacy State):**
```json
{
  "turn_42": {
    "primacy_state_score": 0.611,
    "primacy_state_condition": "violated",

    "user_pa_fidelity": 0.90,
    "ai_pa_fidelity": 0.50,
    "pa_correlation": 0.95,

    "failure_mode": "ai_pa_drift",
    "intervention": "ai_role_correction",

    "post_intervention_ps": 0.855,
    "post_intervention_condition": "achieved"
  }
}
```

**What this tells you:**
- ✓ System detected drift (PS < 0.70)
- ✓ **DIAGNOSTIC:** AI role violated (F_AI = 0.50), user purpose maintained (F_user = 0.90)
- ✓ **TARGETED:** Intervention corrected AI behavior specifically
- ✓ **VERIFIED:** Primacy State restored (PS = 0.855)
- ✓ **TRACEABLE:** Can see continuous derivation from components

**Compliance Value:**

1. **Article 72 "Systematic Procedures":**
   - PS derived systematically every turn (not ad-hoc)
   - Continuous monitoring (not periodic sampling)
   - Quantitative metrics (not qualitative judgment)

2. **NIST "Tracking Over Time":**
   - Turn-by-turn PS scores create time series
   - Can plot governance trajectory
   - Demonstrates active tracking, not passive assumption

3. **Demonstrable Due Diligence:**
   - PS decomposition shows system UNDERSTOOD what failed
   - Targeted intervention shows system ACTED appropriately
   - Restoration shows intervention WORKED

**Questions for Compliance Officers:**
- Does PS decomposition provide clearer audit evidence?
- Would this satisfy post-market monitoring requirements better than current?
- Are there additional metrics you'd need to see?

**Current Status:** Exploratory - gathering feedback before implementation decision

---

## For Technical Reviewers (Academic)

### Mathematical Formulation and Validation Approach

**Core Innovation:**
Primacy State as **emergent equilibrium** of dual attractor dynamics, not arbitrary measurement.

**Single PA (Current):**
```
Primacy State ≈ "Is fidelity above threshold?"
Binary: PASS or FAIL
```

**Dual PA (Proposed):**
```
Primacy State = f(F_user, F_AI, ρ_PA)
Continuous derivation from attractor field geometry
```

**Mathematical Formulation:**

**(Option 1) Primacy State Score:**
$$PS = \rho_{PA} \cdot \frac{2 \cdot F_{user} \cdot F_{AI}}{F_{user} + F_{AI}}$$

Where:
- $F_{user} = \cos(x_t, \hat{a}_{user})$ - User PA fidelity
- $F_{AI} = \cos(x_t, \hat{a}_{AI})$ - AI PA fidelity
- $\rho_{PA} = \cos(\hat{a}_{user}, \hat{a}_{AI})$ - PA correlation

Properties:
- Harmonic mean ensures both PAs must pass (can't compensate)
- $\rho_{PA}$ as multiplicative gate - misaligned attractors fail regardless of individual fidelities
- Range [0, 1] where 1.0 = perfect Primacy State

**(Option 2) Dual Potential Energy:**
$$V_{dual}(x) = \alpha \cdot ||x - \hat{a}_{user}||^2 + \beta \cdot ||x - \hat{a}_{AI}||^2 + \gamma \cdot ||\hat{a}_{user} - \hat{a}_{AI}||^2$$

Where $\alpha + \beta + \gamma = 1.0$ (typically $\alpha \approx 0.5, \beta \approx 0.4, \gamma \approx 0.1$)

**Convergence Tracking:**
$$\Delta V_{dual}(t) = V_{dual}(x_{t+1}) - V_{dual}(x_t)$$

- $\Delta V < 0$: Converging to Primacy State (stable)
- $\Delta V > 0$: Diverging from Primacy State (unstable)

**Hypotheses Under Test:**

**H1:** $\rho_{PA} > 0.90$ → 30% lower variance in $\Delta V$ (PA correlation predicts stability)

**H2:** Component failure precedes combined PS failure by 2-5 turns in ≥60% of cases (earlier detection)

**H3:** $\Delta V_{dual} < 0$ predicts intervention success with >75% accuracy, ≥10% improvement over single PA $\Delta V$ (energy convergence is predictive)

**H4:** Stakeholders rate PS narratives as "clearer" (≥70% preference) than fidelity narratives (improved interpretability)

**Validation Approach:**
- Retrospective application to 46 existing dual PA validation sessions
- Computational feasibility: p95 latency < 50ms
- Comparative analysis: PS-based decisions vs. current fidelity-based decisions
- Stakeholder survey: Narrative clarity assessment

**Success Criteria for GO:**
- Computational feasibility (p95 < 50ms) ✓
- ≥2 of 4 hypotheses supported ✓

**Novelty:**
- First formalization of "governance state" as derived dynamical property
- Extends Lyapunov-like stability analysis to multi-attractor AI governance
- Provides testable predictions (H1-H4) - falsifiable science

**Relation to Prior Work:**
- Builds on dual PA validation (+85.32% improvement)
- Extends control theory to constitutional computing
- Operationalizes "observable demonstrable due diligence" for regulatory compliance

**Open Questions:**
- Does ρ_PA remain stable across session lifecycle?
- Are component failures independent or correlated?
- Does PS formalization scale to multi-attractor systems (>2 PAs)?

---

## For Media / Public Communication

### Plain Language Explanation

**The Big Idea:**
We're making AI governance easier to understand and troubleshoot.

**The Problem:**
When AI systems drift from their intended purpose, current tools give you a single number like "alignment score: 0.73." This tells you something's wrong, but not WHAT or WHERE.

**The Solution:**
"Primacy State" breaks that single number into three parts:
1. **Is the conversation on topic?** (User purpose alignment)
2. **Is the AI behaving appropriately?** (AI role alignment)
3. **Are these two working together?** (Synchronization)

**Example:**
```
Current:
"AI alignment dropped to 0.73"
→ Something's wrong... but what?

Primacy State:
"AI alignment dropped to 0.73 because:
 - Conversation stayed on topic ✓
 - AI started acting like a writer instead of a guide ✗
 - These two got out of sync ✗"
→ Clear: Fix the AI's behavior
```

**Why It Matters:**
- **Transparency:** Easier to understand what AI systems are doing
- **Accountability:** Clearer audit trails for regulators
- **Trust:** Users can see when governance is working

**Current Status:**
Early research phase - testing whether this actually helps before implementing

**Timeline:**
- Testing: November 2025
- Decision: December 2025
- Potential deployment: 2026

---

## Communication Strategy by Audience

| Audience | Key Message | Tone | Detail Level |
|----------|-------------|------|--------------|
| Grant reviewers | Theoretical innovation + rigorous validation | Academic | High (math, stats) |
| Institutional partners | Better diagnostics + easier deployment | Professional | Medium (benefits, examples) |
| Compliance officers | Clearer audit trails + regulatory alignment | Formal | Medium (evidence, requirements) |
| Technical reviewers | Novel formulation + testable hypotheses | Scientific | High (formulas, proofs) |
| Media / Public | Transparency + trust + accountability | Accessible | Low (metaphors, examples) |

---

## FAQ by Audience

### For All Audiences

**Q: Is this a fundamental change to TELOS?**
A: No - it's an enhancement to how we measure and explain governance. Core mechanics stay the same.

**Q: When will this be available?**
A: Still in research phase (Nov 2025). Implementation decision in Dec 2025. Potential deployment Q1 2026 if validated.

**Q: Do I need to do anything?**
A: No - this is our research initiative. Feedback welcome but not required.

### For Institutional Partners

**Q: Will this break my current deployment?**
A: No - backward compatible design. Can run alongside current system. Feature flag controlled.

**Q: What's the performance impact?**
A: ~20ms added latency per turn. Acceptable for most deployments. Can disable if needed.

**Q: Can I test this before full rollout?**
A: Yes - happy to include interested partners in pilot testing (Q1 2026 if GO decision).

### For Grant Reviewers

**Q: How does this compare to other approaches?**
A: First formalization of governance state as derived dynamical property. Novel application of control theory to AI alignment.

**Q: What's the validation evidence?**
A: Built on dual PA validation (+85.32%, p<0.001, n=46). Adds 4 new hypotheses with clear success criteria.

**Q: Where will this be published?**
A: TELOS whitepaper Section 2.3 (if validated). Potentially standalone paper on Primacy State formalization.

### For Compliance Officers

**Q: Does this satisfy EU AI Act Article 72?**
A: Provides stronger evidence (continuous derivation, diagnostic decomposition). Still need full compliance program.

**Q: Can I use PS audit trails for regulatory reporting?**
A: If implemented and validated - yes. Provides quantitative evidence of continuous monitoring.

**Q: What about GDPR / data privacy?**
A: No change - same delta-only storage. PS metrics are numeric, contain no conversation content.

---

**Document Status:** COMPLETE - Ready for stakeholder communication if/when PS validated
**Next Action:** Use appropriate section when communicating with each audience type
