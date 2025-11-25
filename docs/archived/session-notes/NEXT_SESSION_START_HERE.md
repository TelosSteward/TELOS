# 🚀 Next Session Quick Start

**Start Here:** This is your entry point for continuing the beta testing integration.

---

## Session Setup

1. **Enable Plan Mode** in Claude Code
2. **Enable Reasoning Mode** in your new MCP
3. **Read this file first** for context

---

## What We've Accomplished (70% Complete)

✅ **Foundation Complete:**
- Streamlit Cloud compatibility
- Beta-only mode UI (grayed tabs)
- Beta onboarding with consent
- PA calibration display (already working!)

✅ **Components Built (Not Yet Wired):**
- `beta_session_manager.py` - Full A/B testing logic
- `beta_feedback.py` - Feedback UI components

---

## What's Left To Do (30% Remaining)

🚧 **The Big Decision:**

We need to choose one of three paths:

### Option A: Full Dual-Response A/B Testing
**What:** Generate BOTH baseline and TELOS responses, randomly show one
**Time:** 6-9 hours
**Risk:** Medium (modifies core response generation)
**Value:** Full research-grade A/B testing

### Option B: Simplified Feedback-Only
**What:** Keep single-response flow, just add thumbs up/down
**Time:** 2-3 hours
**Risk:** Low (minimal changes)
**Value:** Quick feedback collection (not true A/B)

### Option C: Hybrid (Recommended)
**What:** Do Option B first, upgrade to Option A later
**Time:** 2 hours + 4 hours = 6 hours total
**Risk:** Low initially, staged risk
**Value:** Continuous delivery, best balance

---

## Critical Architecture Requirement

⚠️ **SINGLE CONVERSATION HISTORY**

This is THE most important architectural constraint:

```
✅ CORRECT:
history = [what_user_saw_turn1, what_user_saw_turn2, ...]
baseline_response = llm.generate(history + [new_message])
telos_response = telos.generate(history + [new_message])  # SAME history
shown = random.choice([baseline, telos])
history.append(shown)  # Update THE ONE history

❌ WRONG:
baseline_history = [baseline_r1, baseline_r2, ...]
telos_history = [telos_r1, telos_r2, ...]  # NO! Parallel universes!
```

**Why:** If TELOS sees responses the user never saw, it responds to a conversation that didn't happen.

Documented in: `BETA_A_B_ARCHITECTURE.md`

---

## Files You'll Modify

**Option A:**
1. `core/state_manager.py` - Add `_generate_beta_ab_response()` method
2. `components/conversation_display.py` - Integrate feedback UI
3. `main.py` - Add completion checking

**Option B:**
1. `components/conversation_display.py` - Add phase indicator + simple thumbs up/down
2. `core/state_manager.py` - Add completion tracking

---

## Step-by-Step Implementation

📖 **See FULL details in:** `BETA_INTEGRATION_PLAN.md`

**Quick summary:**

### For Option A:
1. Modify `generate_response_stream()` to detect beta mode
2. Add `_generate_beta_ab_response()` method
3. Call `beta_session_manager.generate_dual_response()`
4. Randomly select which response to show
5. Store both (backend only), show one (frontend)
6. Render feedback UI after response
7. Add completion tracking (2 weeks OR 50 feedbacks)

### For Option B:
1. Add phase transition message at turn 11
2. Add thumbs up/down buttons after responses
3. Store feedback in session state
4. Add completion tracking

---

## Testing Checklist

After implementation:

- [ ] Beta consent flow works
- [ ] PA calibration shows "Calibrating" → "Established"
- [ ] Phase transition message appears at turn 11
- [ ] Feedback UI renders correctly
- [ ] Feedback is recorded
- [ ] Completion criteria work (2 weeks OR 50 feedbacks)
- [ ] Tabs unlock after completion
- [ ] No conversation content stored (privacy check)

---

## Key Session State Variables

Track these during implementation:

```python
st.session_state.beta_consent_given    # True after consent
st.session_state.beta_completed        # True after 2 weeks OR 50 feedbacks
st.session_state.beta_session          # BetaSession object (Option A)
st.session_state.beta_feedback         # List[Dict] (Option B)
st.session_state.beta_start_time       # datetime (for 2-week tracking)
```

---

## File Reference Guide

**Read First (Architecture):**
- `BETA_INTEGRATION_PLAN.md` ← **Start here for full details**
- `BETA_A_B_ARCHITECTURE.md` ← Critical single-history architecture
- `BETA_INTEGRATION_STATUS.md` ← What's complete, what's not

**Already Built (Use These):**
- `beta_testing/beta_session_manager.py` ← Dual-response generation
- `components/beta_feedback.py` ← Feedback UI components
- `components/beta_onboarding.py` ← Consent flow (complete)
- `utils/env_helper.py` ← Cloud compatibility (complete)

**Modify These:**
- `core/state_manager.py` ← Response generation
- `components/conversation_display.py` ← UI integration
- `main.py` ← Completion checking

---

## Recommended Workflow

1. **Plan Mode ON** + **Reasoning Mode ON**
2. **Read** `BETA_INTEGRATION_PLAN.md` (full details)
3. **Decide:** Option A, B, or C
4. **Create todo list** for chosen option
5. **Implement incrementally:**
   - Code one step
   - Test immediately
   - Commit before next step
6. **Test end-to-end** when complete
7. **Deploy to Streamlit Cloud**

---

## Questions to Answer in Plan Mode

Before you start coding, use plan mode to think through:

1. Which option (A, B, or C) best balances risk vs value?
2. How will we test the single conversation history requirement?
3. What edge cases exist for completion tracking (e.g., user leaves mid-session)?
4. How do we handle session state persistence across Streamlit reruns?
5. Should we add any logging/debugging aids for the beta period?

---

## Success Criteria

You'll know you're done when:

✅ User can consent to beta testing
✅ PA calibration works (turns 1-10)
✅ A/B testing or feedback collection works (turns 11+)
✅ Completion criteria trigger correctly
✅ Tabs unlock after completion
✅ All data privacy requirements met
✅ System works on Streamlit Cloud

---

## Current Progress

**Completed:** 70%
**Remaining:** 30%
**Estimated Time:** 2-9 hours (depends on option)

---

**Ready to start? Go to:** `BETA_INTEGRATION_PLAN.md`

**Good luck! 🚀**
