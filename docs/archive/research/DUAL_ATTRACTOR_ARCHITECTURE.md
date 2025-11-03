# TELOS Dual Attractor Architecture - Evolution & Rationale

**Status:** Experimental / Consideration Code
**Tag Baseline:** v1.1-single-attractor
**Target:** v1.2-dual-attractor (experimental branch)
**Purpose:** Extend governance from conversation topic to AI behavior

---

## Executive Summary

This document describes the evolution from **Single Primacy Attractor** to **Dual Primacy Attractor** architecture in TELOS. This is NOT gospel - this is consideration code requiring validation through testing.

### Key Innovation

**Lock-On Derivation**: The AI's Primacy Attractor is not independent - it's **computed from** the User's Primacy Attractor to ensure automatic alignment.

```
Single PA:  User Input → Check(PA_user) → Intervene
Problem:    AI behavior governed by prompt only (soft constraint)

Dual PA:    User Input → Check(PA_user AND PA_ai) → Intervene
                                    ↑
                        PA_ai = derive(PA_user)

Benefit:    AI behavior mathematically governed (hard constraint)
```

---

## Why Single Attractor Works (But Has Limits)

### Single PA Architecture (Current - v1.1)

```
Components:
├─ One Primacy Attractor per session
├─ Governs conversation topic/purpose (WHAT to discuss)
├─ AI behavior governed by system prompt only
└─ Fidelity measured: distance(response, PA)

Strengths:
✓ Proven effective for topic governance
✓ Prevents drift from user's purpose
✓ Real-time intervention system works
✓ Observable, measurable, reproducible

Limitations:
✗ AI behavior (HOW to help) governed by prompt only
✗ No mathematical governance of AI's role
✗ Boundaries enforced via text, not semantics
✗ AI personality/tone drift possible
✗ Can't prove AI maintained professional boundaries
```

### Real-World Scenario Where Single PA Falls Short

**Customer Service Bot**:
```
User PA: "Resolve my billing issue"
System Prompt: "Be helpful, empathetic, never blame user, escalate when needed"

Problem:
- User PA governs WHAT (billing resolution)
- System prompt governs HOW (empathy, no blame)
- But prompt is soft constraint - LLM might ignore it
- No fidelity score for "Was this empathetic?"
- No intervention if AI becomes dismissive
```

**Result**: Topic stays on track, but AI might become unhelpful/dismissive over time.

---

## Dual Attractor Solution

### Architecture Overview

```python
class DualPrimacyAttractor:
    """
    Governs TWO semantic dimensions simultaneously:
    1. User's Purpose (WHAT conversation should accomplish)
    2. AI's Role (HOW AI should behave to serve that purpose)

    Key Constraint: AI PA must be DERIVED from User PA
    (Not independent - ensures automatic alignment)
    """
    user_pa: PrimacyAttractor   # User's purpose
    ai_pa: PrimacyAttractor      # AI's role (derived from user_pa)
    correlation: float            # How well ai_pa serves user_pa
```

### The Lock-On Mechanism (Core Innovation)

**Derivation Rules Map User Intent → AI Action**:

```python
intent_to_role = {
    'learn': 'teach',
    'understand': 'explain',
    'solve': 'help solve',
    'create': 'help create',
    'decide': 'help decide',
    'explore': 'guide exploration'
}

# Example:
User PA: "Learn Python programming"
    ↓ (derivation)
AI PA: "Teach Python clearly and patiently"

# Automatic alignment through computation
```

### Why Derivation Matters

**Independent PAs (BAD)**:
```
User PA: "Learn calculus"
AI PA: "Be entertaining and casual"

Problem: AI might tell jokes while user wants serious learning
Result: Misalignment possible
```

**Derived PA (GOOD)**:
```
User PA: "Learn calculus"
    ↓ (detect: learning intent)
AI PA: "Teach calculus clearly with rigor"

Benefit: AI role computed to serve user purpose
Result: Automatic alignment guaranteed
```

---

## Mathematical Foundation

### Dual Fidelity Calculation

```python
# Single PA (current)
fidelity = similarity(response_embedding, pa_embedding)
pass = fidelity >= threshold

# Dual PA (new)
user_fidelity = similarity(response_embedding, user_pa_embedding)
ai_fidelity = similarity(response_embedding, ai_pa_embedding)

user_pass = user_fidelity >= user_pa.threshold
ai_pass = ai_fidelity >= ai_pa.threshold

overall_pass = user_pass AND ai_pass
```

### Intervention Logic

```python
if not overall_pass:
    if not user_pass and not ai_pass:
        failure = "both"      # Drifted from purpose AND role
    elif not user_pass:
        failure = "purpose"   # Off-topic
    else:
        failure = "role"      # Wrong behavior

    # Generate targeted correction
    intervene(failure_type=failure)
```

---

## Implementation Architecture

### Component Responsibilities

**Steward PM (Coordinator)**:
```python
class StewardPM:
    """
    Orchestrates dual PA workflow.
    Owns: PA derivation, correlation checking, routing
    """

    async def initialize_governance(self, conversation):
        # Extract User PA
        self.user_pa = await extract_user_pa(conversation)

        # Derive AI PA (in parallel with embedding)
        tasks = [
            derive_ai_pa(self.user_pa),
            embed_user_pa(self.user_pa)
        ]
        self.ai_pa, user_embedding = await asyncio.gather(*tasks)

        # Validate correlation
        correlation = check_correlation(self.user_pa, self.ai_pa)
        if correlation < 0.3:
            logger.warning(f"Low PA correlation: {correlation}")

        # Configure dual governance
        self.dual_pa = DualPrimacyAttractor(
            user_pa=self.user_pa,
            ai_pa=self.ai_pa,
            correlation=correlation
        )
```

**Governance Steward (Executor)**:
```python
class UnifiedGovernanceSteward:
    """
    Executes fidelity checks and interventions.
    Owns: Math, thresholds, correction generation
    """

    def check_dual_pa(self, response_embedding, dual_pa):
        # Calculate both fidelities
        user_fidelity = dual_pa.user_pa.calculate_fidelity(response_embedding)
        ai_fidelity = dual_pa.ai_pa.calculate_fidelity(response_embedding)

        # Determine pass/fail
        return {
            'user_fidelity': user_fidelity,
            'ai_fidelity': ai_fidelity,
            'overall_pass': (user_fidelity >= user_pa.threshold and
                            ai_fidelity >= ai_pa.threshold),
            'dominant_failure': identify_primary_failure(...)
        }
```

**Clean Separation**:
- PM decides WHAT to govern and WHEN
- Governance executes HOW to check and HOW to fix
- No overlap, easy to test, parallel-friendly

---

## Backwards Compatibility

### Fallback Mode

```python
# Single PA mode still works
steward = StewardPM(enable_dual_mode=False)
# Uses only user PA, existing behavior

# Dual PA mode opt-in
steward = StewardPM(enable_dual_mode=True)
# Uses derived AI PA + user PA
```

### Migration Path

**Phase 1**: Tag v1.1 (single PA) as stable ✅
**Phase 2**: Implement dual PA on experimental branch
**Phase 3**: Test dual PA extensively
**Phase 4**: Compare single vs dual PA performance
**Phase 5**: If validated, promote dual PA to default

---

## What This Enables

### 1. Configurable AI Personality

**Same User Purpose, Different AI Roles**:

```python
user_pa = "Resolve billing issue"

# Professional consultant mode
ai_pa = derive_ai_pa(user_pa, template={
    'boundaries': ["Be formal", "Use precise language", "No humor"]
})

# Friendly helper mode
ai_pa = derive_ai_pa(user_pa, template={
    'boundaries': ["Be warm", "Use everyday language", "Light humor OK"]
})

Same topic governance, different behavioral governance
```

### 2. Boundary Enforcement

**AI Role Boundaries Now Mathematically Governed**:

```python
ai_pa = PrimacyAttractor(
    purpose=["Provide medical information"],
    boundaries=[
        "Never diagnose conditions",
        "Never prescribe treatments",
        "Always recommend consulting doctor"
    ]
)

# If response violates boundary:
# Measured via fidelity drop
# Intervention triggered automatically
# Correction enforces boundary mathematically
```

### 3. Better Interventions

**Corrections Reference BOTH PAs**:

```python
# Single PA intervention:
"Return to user's purpose: [purpose]"

# Dual PA intervention:
"You drifted from:
 - User's purpose: [user_purpose]
 - Your role: [ai_role]

Your role was derived to serve their purpose.
Satisfying both should feel natural and aligned."
```

### 4. Provable Compliance

**For Regulated Industries**:

```python
# Healthcare bot logs show:
- User wanted: "Understand chest pain causes"
- AI role: "Provide educational info, never diagnose"
- All responses: >0.70 fidelity to both PAs
- Zero responses crossed diagnostic boundary

Mathematical proof AI stayed compliant
```

---

## Testing Requirements

### Validation Criteria

**Dual PA Creation**:
```
✓ User PA → AI PA derivation produces sensible role
✓ Correlation > 0.3 (AI role serves user purpose)
✓ Intent detection maps correctly (learn → teach, etc.)
```

**Response Checking**:
```
✓ Both PAs evaluated independently
✓ Overall pass requires both pass
✓ Failure type identified correctly
```

**Intervention Quality**:
```
✓ Corrections reference both PAs
✓ Failure type determines intervention strategy
✓ Corrected responses satisfy both PAs
```

**Observatory Integration**:
```
✓ Shows two fidelity lines (user + AI)
✓ Identifies which PA triggered intervention
✓ Displays correlation over time
```

**Backwards Compatibility**:
```
✓ Single PA mode still works
✓ Dual mode opt-in (not forced)
✓ Clean fallback when correlation too low
```

---

## Open Questions (To Resolve During Implementation)

### 1. Threshold Configuration

```
Q: Should AI PA always have higher threshold than user PA?
Current thinking: Yes
- User PA: 0.65 (primary importance)
- AI PA: 0.70 (role is important too)
Rationale: AI behavior violations more damaging than topic drift?
```

### 2. Derivation Comprehensiveness

```
Q: Are intent→role mappings sufficient?
Current coverage:
- learn → teach
- solve → help solve
- understand → explain
- create → help create
- decide → help decide
- explore → guide exploration

Missing cases?
Edge cases needing special handling?
```

### 3. Correlation Minimum

```
Q: What's minimum acceptable correlation?
Current thresholds:
- > 0.5: Well aligned
- 0.3-0.5: Acceptable
- < 0.3: Warning
- < 0.2: Fail to single PA fallback

Right thresholds? Need empirical data?
```

### 4. Intervention Priority

```
Q: If both PAs fail, which to prioritize in correction?
Options:
A. Always prioritize user purpose (it's primary)
B. Address both equally (they're related anyway)
C. Prioritize based on magnitude (bigger failure first)

Current: B (address both), since AI PA derived to serve user PA
```

### 5. PA Update Frequency

```
Q: Should AI PA be re-derived as user PA evolves?
Scenarios:
- User clarifies purpose → derive new AI PA?
- Or keep AI PA fixed after initial establishment?

Tradeoff: Adaptability vs stability
```

---

## What NOT To Build

❌ **Don't** create novel mathematical frameworks
❌ **Don't** implement "quantum entanglement" or "gravitational" models
❌ **Don't** add complex optimization algorithms
❌ **Don't** create more than two attractors (KISS principle)
❌ **Don't** force dual mode (keep single PA as fallback)

✅ **Do** use existing PrimacyAttractor code
✅ **Do** keep derivation logic simple
✅ **Do** maintain backwards compatibility
✅ **Do** validate correlation between PAs
✅ **Do** test extensively before promoting

---

## Success Metrics

### How We'll Know It Works

**Quantitative**:
```
- PA correlation > 0.5 for 80%+ of sessions
- Both PAs pass 90%+ of turns without intervention
- When intervention needed, <3 attempts to satisfy both PAs
- User PA + AI PA interventions more effective than user PA alone
- Zero false positives (valid responses rejected)
```

**Qualitative**:
```
- AI maintains consistent role throughout conversation
- Boundary violations caught automatically
- Interventions feel natural and aligned
- Users perceive AI as more professional/consistent
- Stakeholders can audit AI behavior via saved logs
```

**Comparative (vs Single PA)**:
```
- Does dual PA catch drift single PA misses?
- Do interventions improve response quality?
- Does correlation predict session success?
- Is complexity cost worth governance benefit?
```

---

## Next Steps

### Implementation Sequence

**Step 1: Infrastructure** (COMPLETE ✅)
- [x] Tag single PA as v1.1-single-attractor
- [x] Document evolution rationale
- [x] Create experimental branch `experimental/dual-attractor`
- [x] Set up dual PA testing framework (23/24 tests passing)

**Step 2: Core Components** (COMPLETE ✅)
- [x] Implement DualPrimacyAttractor class
- [x] Build AI PA derivation logic with intent detection
- [x] Add correlation checking
- [x] Implement GovernanceConfig for mode switching
- [x] Build PA comparison runner for empirical validation
- [ ] Extend Steward PM for orchestration (NEXT)
- [ ] Extend Governance Steward for execution (NEXT)

**Step 3: Integration**
- [ ] Update Observatory visualization
- [ ] Add dual fidelity display
- [ ] Show correlation tracking
- [ ] Indicate failure type in UI

**Step 4: Testing**
- [ ] Unit tests for each component
- [ ] Integration tests for full workflow
- [ ] Comparison tests (single vs dual)
- [ ] Edge case testing
- [ ] Performance benchmarking

**Step 5: Validation**
- [ ] Demo Mode with dual PA
- [ ] Real conversation testing
- [ ] User feedback collection
- [ ] Metrics analysis
- [ ] Decision: promote or archive

---

## Timeline & Milestones

**Experimental Phase** (Current):
- Build dual PA architecture
- Test with synthetic conversations
- Validate technical feasibility

**Validation Phase** (Next):
- Test with real users (Beta Testing Mode)
- Blind preference testing (single vs dual)
- Collect quantitative metrics
- Gather qualitative feedback

**Decision Phase** (Future):
- Analyze results
- Compare cost/benefit
- Decide: promote, iterate, or archive
- Document findings regardless of outcome

---

## Conclusion

Dual Primacy Attractor architecture extends TELOS governance from **conversation topic** to **AI behavior**. The innovation is **lock-on derivation** - computing AI's role from user's purpose ensures automatic alignment.

This is **consideration code** requiring validation. The single PA model (v1.1) remains stable and production-ready. Dual PA is experimental work on a separate branch to prove (or disprove) the concept.

If validated, dual PA enables:
- Mathematically governed AI behavior
- Enforceable boundaries
- Configurable AI personality
- Provable compliance for regulated use
- More effective interventions

If not validated, we learned something valuable and can archive the experiment with clear documentation of why it didn't work.

**Build to learn. Test to validate. Document regardless of outcome.**

---

**Document Status**: Initial draft for experimental branch setup
**Last Updated**: 2025-11-02
**Author**: Claude Code (Coder)
**Reviewer**: JB (Product Lead)
