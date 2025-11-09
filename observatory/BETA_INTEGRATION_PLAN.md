# Beta Testing Integration - Complete Implementation Plan

**Status:** 70% complete - Foundation built, awaiting integration decision
**Created:** 2025-01-08
**Next Session:** Start here with plan mode + reasoning mode enabled

---

## Executive Summary

TELOS Observatory Beta testing framework is 70% complete with all foundation components built. The remaining work involves integrating the two-phase beta flow (PA calibration → A/B testing) into the response generation pipeline. This document provides a complete roadmap for finishing the integration.

---

## ✅ Completed Components (Phases 1-4)

### Phase 1: Streamlit Cloud Compatibility ✓
**Files Created:**
- `observatory/utils/env_helper.py` - Cloud detection, secrets management, data directory helpers
- `.streamlit/config.toml` - Streamlit deployment configuration
- `.streamlit/secrets.toml` (template) - API key storage template
- `.gitignore` (updated) - Added beta data privacy exclusions

**Key Functions:**
- `is_streamlit_cloud()` - Detects cloud vs local environment
- `get_api_key(service)` - Gets API keys from secrets or environment
- `get_data_dir(subdir)` - Returns cloud-compatible data paths

**Status:** ✓ Complete and tested

### Phase 2: Beta-Only Mode UI ✓
**File Modified:** `observatory/main.py:554-659`

**Implementation:**
```python
is_beta_only = not st.session_state.get('beta_completed', False)

# Grayed tabs with disabled state
button(disabled=is_beta_only)

# Progress message
if is_beta_only:
    st.markdown("Complete beta testing to unlock DEMO and TELOS tabs")
```

**Status:** ✓ Complete - tabs gray out until beta completion

### Phase 3: Beta Onboarding ✓
**File Modified:** `observatory/components/beta_onboarding.py`

**Key Changes:**
- Updated messaging for preference testing focus
- Privacy-first: "We collect your feedback and preferences — not your conversations"
- Consent tracked via `st.session_state.beta_consent_given`
- Consent logging to `beta_consents/consent_log.json`

**Status:** ✓ Complete - consent flow working

### Phase 4: PA Calibration Display ✓
**File:** `observatory/components/conversation_display.py:380-409`

**Already Working:**
- Shows "Calibrating" (orange) for turns 1-7
- Shows "Established" (green) after convergence
- Displays fidelity scores with color coding
- ΔF (Delta Fidelity) if available

**Status:** ✓ Complete - user confirmed working correctly

---

## 📦 Built But Not Wired

### Beta Session Manager
**File:** `observatory/beta_testing/beta_session_manager.py`

**Available Methods:**
```python
class BetaSessionManager:
    def start_session(user_id) -> BetaSession
    def assign_test_condition(session) -> TestCondition  # 40/40/20 split
    def generate_dual_response(user_message, state_manager, turn_number) -> Dict
    def record_feedback(session, feedback)
    def record_runtime_fidelity(session, turn_number, fidelity_data)
    def end_session(session)
    def export_sessions(output_file) -> str
    def get_session_stats() -> Dict
```

**Key Feature:** `generate_dual_response()`
- Generates BOTH baseline and TELOS responses
- Returns both plus fidelity metrics
- Uses single conversation history (critical architecture requirement)

### Beta Feedback UI
**File:** `observatory/components/beta_feedback.py`

**Available UI Components:**
```python
def render_single_blind_feedback(turn_number, response_source)
    # Thumbs up/down buttons
    # Optional text feedback

def render_head_to_head_feedback(turn_number, response_a, response_b)
    # Side-by-side comparison
    # Preference selection

def render_researcher_dashboard(session_manager)
    # Session statistics
    # Data export buttons
```

---

## 🎯 Critical Architecture Decision

### The Single Conversation History Requirement

**User's Insight (from previous conversation):**
> "shouldn't see that it actually responded previously so that should be wiped from memory and every context resets yes? Because former responses may actually impact the current context window and decision making for the current turn."

**This is CRITICAL and CORRECT.**

**The Architecture:**
- **One source of truth:** Single conversation history containing only what user actually saw
- **Both models stateless:** Each turn, both baseline and TELOS generate responses fresh using same context
- **No parallel universes:** Don't track separate histories for baseline/TELOS paths
- **TELOS is adaptive:** Can guide toward PA from any conversation state, doesn't need "pure" path

**Example Flow:**
```python
# Turn 11: Show baseline
history = [turn1_baseline, turn2_telos, ..., turn10_baseline]  # What user saw
baseline_response = llm.generate(history + [user_message_11])
telos_response = telos.process(history + [user_message_11])
shown_response = baseline_response  # Random choice
history.append(shown_response)  # Update THE SINGLE history

# Turn 12: Show TELOS
history = [turn1_baseline, ..., turn11_baseline]  # Includes turn 11 baseline
baseline_response = llm.generate(history + [user_message_12])  # Sees baseline from turn 11
telos_response = telos.process(history + [user_message_12])  # Sees same history
shown_response = telos_response  # Random choice
history.append(shown_response)  # Update history with TELOS response
```

**Documented in:** `observatory/BETA_A_B_ARCHITECTURE.md`

---

## 🚧 Integration Options - DECISION REQUIRED

### Current Challenge

**Existing Flow (`state_manager.py:generate_response_stream()`):**
1. Generate ONE baseline response from LLM
2. Pass that response to TELOS for **evaluation**
3. TELOS evaluates and returns **metrics** (not a new response)
4. Show the original baseline response to user

**What A/B Testing Needs:**
1. Generate baseline response from LLM
2. Generate **separate TELOS response** (not just evaluation)
3. Randomly choose which to show
4. Update single conversation history with what was shown
5. Log both for research, show feedback UI

### Option A: Full Dual-Response A/B Testing

**Approach:** Modify core response generation to use `beta_session_manager.generate_dual_response()`

**Changes Required:**
1. **Modify:** `state_manager.py:generate_response_stream()`
2. **Add:** Beta mode detection (check `beta_consent_given` and turn number)
3. **Add:** Test condition assignment (once per session)
4. **Add:** Random response selection logic
5. **Modify:** Conversation history to only include shown responses
6. **Add:** Backend logging of both responses
7. **Integrate:** Feedback UI rendering after responses

**Pros:**
- True A/B testing with real comparison
- Full feature set as originally envisioned
- Valuable preference data for research

**Cons:**
- Complex integration into core response logic
- Higher risk of bugs in critical path
- More testing required before deployment

**Estimated Time:** 4-6 hours development + 2-3 hours testing

### Option B: Simplified Feedback-Only

**Approach:** Keep existing single-response flow, add feedback UI only

**Changes Required:**
1. **Add:** Phase indicator after turn 10: "PA Established! Beta preference testing active"
2. **Add:** Feedback UI rendering after responses (turns 11+)
3. **Add:** Feedback storage in session state
4. **Add:** Beta completion tracking
5. **Keep:** Existing response generation unchanged

**Pros:**
- Low risk, minimal changes to core logic
- Quick path to working beta system
- Immediate feedback collection
- Can upgrade to Option A later

**Cons:**
- Not true A/B testing (no comparison)
- Less valuable research data initially
- May need to explain to users why "preference testing" without choices

**Estimated Time:** 1-2 hours development + 1 hour testing

### Hybrid Option C: Progressive Enhancement

**Approach:** Start with Option B, upgrade to Option A incrementally

**Phase 1 (Quick Win):**
- Implement feedback-only (Option B)
- Deploy and start collecting data
- Get beta testers onboarded

**Phase 2 (Full A/B):**
- Add dual-response generation
- Migrate existing beta users smoothly
- Enhanced research data

**Pros:**
- Fastest path to deployed beta
- De-risks complex integration
- Continuous user value

**Cons:**
- Two development cycles
- Data schema may need migration

**Estimated Time:** 2 hours (Phase 1) + 4 hours (Phase 2)

---

## 📋 Detailed Implementation Steps

### Option A: Full Dual-Response Integration

#### Step 1: Modify Response Generation
**File:** `observatory/core/state_manager.py`

**Current location:** `generate_response_stream()` at line 694

**Changes:**
```python
def generate_response_stream(self, message: str, turn_idx: int):
    # ... existing initialization ...

    # Check if beta mode active
    beta_mode = st.session_state.get('beta_consent_given', False)
    beta_completed = st.session_state.get('beta_completed', False)

    if beta_mode and not beta_completed:
        # Beta testing flow - two-phase approach
        if turn_idx < 10:
            # Phase 1: PA calibration - baseline only
            full_response = yield from self._generate_calibration_response(
                message, turn_idx, conversation_history
            )
        else:
            # Phase 2: A/B testing - dual responses
            full_response = yield from self._generate_beta_ab_response(
                message, turn_idx, conversation_history
            )
    else:
        # Normal Observatory mode (existing code)
        full_response = yield from self._generate_normal_response(
            message, turn_idx, conversation_history
        )
```

**Add new method:**
```python
def _generate_beta_ab_response(self, message: str, turn_idx: int, history: List[Dict]):
    """Generate dual responses for A/B testing."""
    # Initialize beta session manager if needed
    if not hasattr(self, '_beta_manager'):
        from observatory.beta_testing.beta_session_manager import BetaSessionManager
        self._beta_manager = BetaSessionManager()

        # Start session and assign test condition
        if 'beta_session' not in st.session_state:
            session = self._beta_manager.start_session()
            test_condition = self._beta_manager.assign_test_condition(session)
            st.session_state.beta_session = session

    # Generate both responses
    dual_result = self._beta_manager.generate_dual_response(
        user_message=message,
        state_manager=self,
        turn_number=turn_idx + 1
    )

    baseline_response = dual_result['baseline_response']
    telos_response = dual_result['telos_response']

    # Get test condition
    session = st.session_state.beta_session
    test_condition = session.test_condition

    # Select which response to show based on test condition
    if test_condition == "single_blind_baseline":
        shown_response = baseline_response
        response_source = "baseline"
    elif test_condition == "single_blind_telos":
        shown_response = telos_response
        response_source = "telos"
    else:  # head_to_head
        # For head-to-head, we'll show both in the UI
        shown_response = None  # Special handling in display layer
        response_source = "both"

    # Store BOTH responses for research (backend only)
    turn_data = {
        'shown_response': shown_response,
        'baseline_response_hidden': baseline_response,
        'telos_response_hidden': telos_response,
        'response_source_hidden': response_source,
        'test_condition': test_condition,
        'baseline_fidelity': dual_result['baseline_fidelity'],
        'telos_fidelity': dual_result['telos_fidelity'],
        'drift_detected': dual_result['drift_detected'],
        'intervention_applied': dual_result['intervention_applied']
    }

    # Yield the shown response
    if shown_response:
        yield shown_response
    else:
        # For head-to-head, yield both (display layer will handle)
        yield f"__HEAD_TO_HEAD__{baseline_response}__SPLIT__{telos_response}"

    # Update turn with data
    self.state.turns[turn_idx].update(turn_data)
    self.state.turns[turn_idx]['needs_feedback'] = True  # Flag for UI
```

#### Step 2: Integrate Feedback UI
**File:** `observatory/components/conversation_display.py`

**Location:** After rendering response in `_render_current_turn_only()`

**Changes:**
```python
def _render_current_turn_only(self, current_turn_idx: int, all_turns: list):
    # ... existing rendering ...

    # After response is shown, check if feedback needed
    if turn_data.get('needs_feedback', False) and not turn_data.get('feedback_collected', False):
        from observatory.components.beta_feedback import (
            render_single_blind_feedback,
            render_head_to_head_feedback
        )

        test_condition = turn_data.get('test_condition')

        if test_condition in ['single_blind_baseline', 'single_blind_telos']:
            render_single_blind_feedback(
                turn_number=turn_number,
                response_source=turn_data.get('response_source_hidden')  # Hidden from user
            )
        elif test_condition == 'head_to_head':
            render_head_to_head_feedback(
                turn_number=turn_number,
                response_a=turn_data.get('baseline_response_hidden'),
                response_b=turn_data.get('telos_response_hidden')
            )
```

#### Step 3: Add Beta Completion Tracking
**File:** `observatory/core/state_manager.py`

**Add method:**
```python
def check_beta_completion(self):
    """Check if beta testing criteria are met."""
    if st.session_state.get('beta_completed', False):
        return  # Already completed

    session = st.session_state.get('beta_session')
    if not session:
        return

    # Criteria 1: 2 weeks elapsed
    from datetime import datetime, timedelta
    start_time = datetime.fromisoformat(session.start_time)
    two_weeks_elapsed = datetime.now() - start_time >= timedelta(days=14)

    # Criteria 2: 50 feedback items collected
    feedback_count = len(session.feedback_items)
    fifty_feedbacks = feedback_count >= 50

    # Complete if either criterion met
    if two_weeks_elapsed or fifty_feedbacks:
        st.session_state.beta_completed = True

        # Save final session data
        if hasattr(self, '_beta_manager'):
            self._beta_manager.end_session(session)

        # Show completion message
        st.success("🎉 Beta testing complete! Full TELOS features now unlocked.")
```

#### Step 4: Update Main Entry Point
**File:** `observatory/main.py`

**Add after beta consent check:**
```python
if beta_consent_given and not beta_completed:
    # Check completion criteria each session
    state_manager.check_beta_completion()
```

### Option B: Simplified Feedback-Only

#### Step 1: Add Phase Indicator
**File:** `observatory/components/conversation_display.py`

**After turn 10, show message:**
```python
if turn_number == 11 and st.session_state.get('beta_consent_given', False):
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
                border: 1px solid #FFD700; border-radius: 8px; padding: 20px; margin: 20px 0;">
        <h3 style="color: #FFD700; margin-top: 0;">🎯 PA Established!</h3>
        <p style="color: #e0e0e0; font-size: 19px;">
            Beta preference testing is now active. Please rate responses to help improve TELOS.
        </p>
    </div>
    """, unsafe_allow_html=True)
```

#### Step 2: Add Simple Feedback UI
**After each response (turns 11+):**
```python
if turn_number >= 11 and st.session_state.get('beta_consent_given', False):
    col1, col2, col3 = st.columns([1, 1, 8])
    with col1:
        if st.button("👍", key=f"thumbs_up_{turn_number}"):
            # Record feedback
            if 'beta_feedback' not in st.session_state:
                st.session_state.beta_feedback = []
            st.session_state.beta_feedback.append({
                'turn': turn_number,
                'rating': 'thumbs_up',
                'timestamp': datetime.now().isoformat()
            })
            st.rerun()
    with col2:
        if st.button("👎", key=f"thumbs_down_{turn_number}"):
            # Record feedback
            if 'beta_feedback' not in st.session_state:
                st.session_state.beta_feedback = []
            st.session_state.beta_feedback.append({
                'turn': turn_number,
                'rating': 'thumbs_down',
                'timestamp': datetime.now().isoformat()
            })
            st.rerun()
```

#### Step 3: Add Completion Tracking
**Same as Option A Step 3**, but check `len(st.session_state.get('beta_feedback', []))` instead of session feedback items.

---

## 🧪 Testing Plan

### Test Cases for Both Options

1. **Beta Consent Flow**
   - [ ] First-time user sees onboarding
   - [ ] Consent properly recorded
   - [ ] DEMO/TELOS tabs are grayed out

2. **PA Calibration Phase (Turns 1-10)**
   - [ ] Shows "Calibrating" status
   - [ ] Fidelity scores display
   - [ ] No feedback UI shown yet

3. **A/B Testing Phase (Turns 11+)**
   - [ ] Phase transition message shown at turn 11
   - [ ] Feedback UI renders correctly
   - [ ] Feedback properly recorded

4. **Beta Completion**
   - [ ] 2-week timer works correctly
   - [ ] 50-feedback threshold works
   - [ ] Completion message shows
   - [ ] Tabs unlock (`beta_completed = True`)

5. **Data Privacy**
   - [ ] Only feedback/preferences stored
   - [ ] No conversation content in logs
   - [ ] Proper .gitignore exclusions

### Additional Tests for Option A

6. **Dual Response Generation**
   - [ ] Both baseline and TELOS responses generated
   - [ ] Random selection works (40/40/20 distribution)
   - [ ] Correct response shown to user
   - [ ] Both logged for research

7. **Single Conversation History**
   - [ ] Only shown responses in history
   - [ ] Both models see same context
   - [ ] No context contamination

8. **Head-to-Head Mode**
   - [ ] Both responses display side-by-side
   - [ ] Preference selection works
   - [ ] Correct choice recorded

---

## 📁 File Reference

### Files to Modify (Option A)
1. `observatory/core/state_manager.py` - Add dual-response logic
2. `observatory/components/conversation_display.py` - Integrate feedback UI
3. `observatory/main.py` - Add completion check

### Files to Modify (Option B)
1. `observatory/components/conversation_display.py` - Add phase indicator and simple feedback
2. `observatory/core/state_manager.py` - Add completion tracking

### Files Already Complete (No Changes Needed)
- `observatory/beta_testing/beta_session_manager.py`
- `observatory/components/beta_feedback.py`
- `observatory/components/beta_onboarding.py`
- `observatory/utils/env_helper.py`
- `.streamlit/config.toml`
- `.streamlit/secrets.toml` (template)

### Configuration Files
- `.gitignore` - Updated with beta data exclusions
- `observatory/BETA_A_B_ARCHITECTURE.md` - Architecture documentation
- `observatory/BETA_INTEGRATION_STATUS.md` - Detailed status
- `observatory/BETA_INTEGRATION_PLAN.md` - This file

---

## 🎯 Recommendation

**For Next Session:**

1. **Start in Plan Mode** with reasoning mode enabled
2. **Review this document** and architecture docs
3. **Make decision:** Option A, B, or C
4. **If Option A:** Start with Step 1 (modify response generation)
5. **If Option B:** Start with Step 1 (add phase indicator)
6. **Test incrementally** after each step
7. **Deploy to Streamlit Cloud** once working

**Recommended Path:** Option C (Hybrid)
- Implement Option B first (1-2 hours) → deploy → collect feedback
- Upgrade to Option A (4 hours) → enhanced data collection
- Best balance of speed, risk, and value

---

## 📊 Current Progress Tracking

**Completed:** 70%
- ✅ Phase 1: Cloud compatibility
- ✅ Phase 2: Beta-only mode
- ✅ Phase 3: Onboarding
- ✅ Phase 4: PA calibration display
- ⏸️ Phase 5: A/B testing integration (awaiting decision)
- ⏸️ Phase 6: Completion tracking (depends on Phase 5)
- ⏸️ Phase 7: End-to-end testing

**Remaining Work:**
- Phase 5: 15-20% of total project
- Phase 6: 5% of total project
- Phase 7: 5% of total project

**Estimated Time to Completion:**
- Option A: 6-9 hours
- Option B: 2-3 hours
- Option C: 2 hours (Phase 1) → 4 hours (Phase 2) = 6 hours total

---

## 🔑 Key Session State Variables

**Track these in next session:**

```python
# Beta status flags
st.session_state.beta_consent_given  # True after consent
st.session_state.beta_completed      # True after 2 weeks OR 50 feedbacks

# Beta session data (Option A)
st.session_state.beta_session        # BetaSession object
st.session_state.beta_session.test_condition  # "single_blind_baseline" | "single_blind_telos" | "head_to_head"
st.session_state.beta_session.feedback_items  # List[FeedbackData]

# Beta feedback (Option B - simplified)
st.session_state.beta_feedback       # List[Dict] with ratings
st.session_state.beta_start_time     # datetime for 2-week tracking
```

---

## 🚀 Next Session Checklist

- [ ] Review BETA_INTEGRATION_PLAN.md (this file)
- [ ] Review BETA_A_B_ARCHITECTURE.md
- [ ] Review BETA_INTEGRATION_STATUS.md
- [ ] Enable plan mode
- [ ] Enable reasoning mode in MCP
- [ ] Decide: Option A, B, or C
- [ ] Create implementation todo list
- [ ] Begin integration work
- [ ] Test after each step
- [ ] Deploy when complete

---

**End of Integration Plan - Ready for Next Session**
