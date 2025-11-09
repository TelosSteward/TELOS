# Beta Testing Integration - Quick Start Guide

**Use this guide to start implementation immediately.**

---

## 🚀 Before You Start (5 minutes)

### Read These Files (in order):
1. ✅ `EXECUTIVE_SUMMARY.md` (you're reading it) - Overview
2. ✅ `NEXT_VERSION_PLAN.md` - Complete plan
3. ✅ `BETA_A_B_ARCHITECTURE.md` - Critical constraints

### Setup Checklist:
- [ ] Git status clean (commit current work)
- [ ] Create backup branch: `git checkout -b beta-integration-backup`
- [ ] Create feature branch: `git checkout -b beta-phase1-implementation`
- [ ] Enable plan mode in Claude Code
- [ ] Enable reasoning mode in MCP
- [ ] Open these files in editor:
  - `/Users/brunnerjf/Desktop/TELOS_CLEAN/observatory/components/conversation_display.py`
  - `/Users/brunnerjf/Desktop/TELOS_CLEAN/observatory/main.py`

---

## 📋 Phase 1 Implementation Checklist

**Estimated Time: 3 hours**

### Step 1: Phase Transition Indicator (30 min)

**File:** `conversation_display.py`

**What to do:**
1. Find the `_render_main_chat()` method
2. Add this new method to the class:

```python
def _show_beta_phase_transition(self, turn_number: int):
    """Show phase transition message when PA calibration completes."""
    if turn_number == 11 and st.session_state.get('beta_consent_given', False):
        if st.session_state.get('beta_phase_transition_shown'):
            return

        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
            border: 2px solid #FFD700;
            border-radius: 12px;
            padding: 25px;
            margin: 20px 0;
            text-align: center;
            box-shadow: 0 0 20px rgba(255, 215, 0, 0.2);
        ">
            <div style="font-size: 48px; margin-bottom: 10px;">🎯</div>
            <h3 style="color: #FFD700; margin: 10px 0;">PA Established!</h3>
            <p style="color: #e0e0e0; font-size: 18px; line-height: 1.6; margin: 10px 0;">
                Your conversation purpose is now calibrated.<br>
                Beta preference testing is active - please rate responses below.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.session_state.beta_phase_transition_shown = True
```

3. Call this method in `_render_main_chat()` after rendering each turn

**Test:**
```bash
streamlit run observatory/main.py
# Have a conversation for 11 turns
# Check message appears at turn 11 only
```

**Commit:** `git commit -m "Add phase transition indicator at turn 11"`

---

### Step 2: Simple Feedback UI (45 min)

**File:** `conversation_display.py`

**What to do:**
1. Add these two new methods to the class:

```python
def _render_simple_beta_feedback(self, turn_number: int):
    """Render simple thumbs up/down feedback for beta testing."""
    if turn_number < 11:
        return

    if not st.session_state.get('beta_consent_given', False):
        return

    feedback_key = f"beta_feedback_{turn_number}"
    if st.session_state.get(feedback_key):
        st.markdown("""
        <div style="color: #4CAF50; font-size: 14px; margin: 10px 0;">
            ✓ Thank you for your feedback!
        </div>
        """, unsafe_allow_html=True)
        return

    st.markdown("<div style='margin: 15px 0;'></div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 6])

    with col1:
        if st.button("👍", key=f"thumbs_up_{turn_number}",
                     use_container_width=True, help="Helpful response"):
            self._record_simple_feedback(turn_number, "thumbs_up")
            st.session_state[feedback_key] = True
            st.rerun()

    with col2:
        if st.button("👎", key=f"thumbs_down_{turn_number}",
                     use_container_width=True, help="Not helpful"):
            self._record_simple_feedback(turn_number, "thumbs_down")
            st.session_state[feedback_key] = True
            st.rerun()

    with col3:
        st.markdown("""
        <div style="color: #888; font-size: 14px; padding-top: 8px;">
            Rate this response
        </div>
        """, unsafe_allow_html=True)

def _record_simple_feedback(self, turn_number: int, rating: str):
    """Record simple feedback to session state."""
    from datetime import datetime

    if 'beta_feedback' not in st.session_state:
        st.session_state.beta_feedback = []

    feedback_item = {
        'turn': turn_number,
        'rating': rating,
        'timestamp': datetime.now().isoformat()
    }

    st.session_state.beta_feedback.append(feedback_item)

    if len(st.session_state.beta_feedback) == 1:
        st.session_state.beta_start_time = datetime.now().isoformat()

    import logging
    logging.getLogger(__name__).info(f"Feedback: turn {turn_number} = {rating}")
```

2. Call `_render_simple_beta_feedback(turn_number)` after rendering each response

**Test:**
```bash
# Continue conversation past turn 11
# Check thumbs up/down buttons appear
# Click each button and verify:
#   - Feedback is recorded
#   - Confirmation message shows
#   - Can't vote twice on same turn
```

**Commit:** `git commit -m "Add simple feedback UI for turns 11+"`

---

### Step 3: Beta Completion Tracking (30 min)

**File:** `main.py`

**What to do:**
1. Add these two functions before the `main()` function:

```python
def check_beta_completion():
    """Check if beta testing is complete and unlock full access."""
    if not st.session_state.get('beta_consent_given', False):
        return False

    if st.session_state.get('beta_completed', False):
        return True

    from datetime import datetime, timedelta

    start_time_str = st.session_state.get('beta_start_time')
    if not start_time_str:
        return False

    start_time = datetime.fromisoformat(start_time_str)
    elapsed = datetime.now() - start_time
    two_weeks_elapsed = elapsed >= timedelta(days=14)

    feedback_items = st.session_state.get('beta_feedback', [])
    fifty_feedbacks = len(feedback_items) >= 50

    if two_weeks_elapsed or fifty_feedbacks:
        st.session_state.beta_completed = True
        st.balloons()
        st.success("""
        🎉 **Beta Testing Complete!**

        Thank you for helping improve TELOS! Full Observatory features are now unlocked.
        """)
        return True

    return False

def show_beta_progress():
    """Show beta progress in sidebar."""
    if not st.session_state.get('beta_consent_given', False):
        return

    if st.session_state.get('beta_completed', False):
        return

    from datetime import datetime, timedelta

    start_time_str = st.session_state.get('beta_start_time')
    if not start_time_str:
        return

    start_time = datetime.fromisoformat(start_time_str)
    elapsed = datetime.now() - start_time
    days_elapsed = elapsed.days
    days_remaining = max(0, 14 - days_elapsed)

    feedback_items = st.session_state.get('beta_feedback', [])
    feedback_count = len(feedback_items)
    feedbacks_remaining = max(0, 50 - feedback_count)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Beta Progress")
    st.sidebar.markdown(f"""
    **Completion Criteria** (either one):
    - ⏰ Days: {days_elapsed}/14 ({days_remaining} remaining)
    - 📊 Feedback: {feedback_count}/50 ({feedbacks_remaining} remaining)
    """)
```

2. In `main()`, find where beta onboarding is checked (after `BetaOnboarding.render()`)
3. Add these calls:

```python
# After beta onboarding check
check_beta_completion()

# In sidebar section (find where sidebar is being populated)
show_beta_progress()
```

**Test:**
```bash
# Provide 50 feedback items (or fast-forward clock)
# Verify completion message appears
# Verify balloons animation
# Verify progress updates correctly
```

**Commit:** `git commit -m "Add beta completion tracking and progress display"`

---

### Step 4: Beta-Only Mode UI (30 min)

**File:** `main.py`

**What to do:**
1. Find where tabs are rendered (search for `st.tabs` or tab creation code)
2. Add this code BEFORE the tab rendering:

```python
# Check beta status for UI restrictions
beta_only_mode = (
    st.session_state.get('beta_consent_given', False) and
    not st.session_state.get('beta_completed', False)
)

if beta_only_mode:
    # Gray out non-BETA tabs
    st.markdown("""
    <style>
    /* Gray out DEMO and TELOS tabs during beta */
    button[data-baseweb="tab"]:nth-of-type(1),
    button[data-baseweb="tab"]:nth-of-type(2) {
        opacity: 0.3 !important;
        pointer-events: none !important;
        cursor: not-allowed !important;
    }

    /* Highlight BETA tab */
    button[data-baseweb="tab"]:nth-of-type(3) {
        border: 2px solid #FFD700 !important;
        background: rgba(255, 215, 0, 0.1) !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Show unlock message
    feedback_count = len(st.session_state.get('beta_feedback', []))
    st.info(f"""
    🔒 **Complete beta testing to unlock full Observatory access**

    Progress: {feedback_count}/50 feedback items OR 2 weeks elapsed
    """)
```

**Test:**
```bash
# Start fresh session, give consent
# Verify DEMO and TELOS tabs are grayed out
# Verify BETA tab is highlighted
# Verify unlock message shows
# Complete beta testing
# Verify tabs become clickable
```

**Commit:** `git commit -m "Add beta-only mode UI with tab locking"`

---

### Step 5: End-to-End Testing (1 hour)

**Test Scenarios:**

#### Scenario 1: Fresh User Journey
```
1. Start app → See beta onboarding
2. Give consent
3. Verify BETA tab auto-selected and highlighted
4. Verify DEMO/TELOS tabs grayed out
5. Start conversation
6. Turns 1-10: No feedback UI
7. Verify PA calibration shows
8. Turn 11: Phase transition message appears
9. Turn 11+: Feedback UI appears
10. Give feedback
11. Verify progress updates
```

#### Scenario 2: Completion
```
1. Give 50 feedback items (or fast-forward)
2. Verify completion message
3. Verify balloons
4. Verify tabs unlock
5. Verify beta progress no longer shows
```

#### Scenario 3: Edge Cases
```
1. Refresh during beta → State persists
2. Give feedback twice on same turn → Prevented
3. Phase transition shows twice → Prevented
4. Non-beta user → No beta UI
```

**Commit:** `git commit -m "Phase 1 complete - tested and validated"`

---

## 🎯 Phase 1 Complete!

### What You've Built:
- ✅ Phase transition indicator at turn 11
- ✅ Simple thumbs up/down feedback UI
- ✅ Beta completion tracking (2 weeks OR 50 feedback)
- ✅ Beta-only mode with tab locking
- ✅ Progress display in sidebar

### What Users Get:
- Clear beta flow with PA calibration
- Simple feedback collection
- Visible progress toward completion
- Feature unlock after completion

### Next Steps:
1. Deploy to Streamlit Cloud
2. Test with real usage
3. Monitor for 1-2 weeks
4. Collect user feedback
5. Decide on Phase 2 timing

---

## 🚢 Deploy to Streamlit Cloud

### Deployment Checklist:
- [ ] All tests passing
- [ ] Code committed to main branch
- [ ] `.streamlit/secrets.toml` configured
- [ ] `MISTRAL_API_KEY` added to Streamlit Cloud secrets
- [ ] Deploy via Streamlit Cloud dashboard
- [ ] Test deployed version

### Streamlit Cloud Setup:
1. Go to https://share.streamlit.io
2. Connect GitHub repo
3. Select branch and file (`observatory/main.py`)
4. Add secrets in Settings:
   ```toml
   MISTRAL_API_KEY = "your-key-here"
   ```
5. Deploy!

---

## 📊 Monitoring Phase 1

### What to Watch:
- Number of beta consents
- Feedback submission rate
- Progress toward completion
- User drop-off points
- Error logs

### Success Metrics (Week 1):
- [ ] 5+ users give consent
- [ ] 20+ feedback items collected
- [ ] <5% error rate
- [ ] Average session >15 minutes
- [ ] Users return for multiple sessions

### Decision Point:
After 1-2 weeks of Phase 1, decide:
- **Proceed to Phase 2?** If yes, implement full A/B testing
- **Stay with Phase 1?** If current data sufficient
- **Iterate Phase 1?** If improvements needed first

---

## 🔧 Troubleshooting

### Issue: Phase transition doesn't show
**Check:**
- Turn number is exactly 11
- `beta_consent_given` is True
- `beta_phase_transition_shown` not already set

### Issue: Feedback not recording
**Check:**
- `beta_feedback` list exists in session_state
- Timestamp format is correct
- Logger shows feedback recording

### Issue: Tabs not unlocking
**Check:**
- `beta_completed` is set to True
- Completion criteria met (50 feedback OR 2 weeks)
- CSS not overriding unlock

### Issue: Progress not updating
**Check:**
- `beta_start_time` is set
- Feedback count is accurate
- Datetime calculations correct

---

## 🎓 Key Learnings for Phase 2

After Phase 1, you'll understand:
1. How Streamlit state behaves with your beta flow
2. Where users get confused or drop off
3. What edge cases exist in real usage
4. How to structure beta state management
5. Performance characteristics

These learnings make Phase 2 much easier!

---

## 📞 Need Help?

### Resources:
- Full plan: `NEXT_VERSION_PLAN.md`
- Architecture: `BETA_A_B_ARCHITECTURE.md`
- Status: `BETA_INTEGRATION_STATUS.md`
- Implementation: `BETA_INTEGRATION_PLAN.md`

### Common Questions:
- **Q: Can I skip Phase 1 and go straight to Phase 2?**
  A: You can, but it's riskier. Phase 1 validates the flow.

- **Q: How do I test completion without waiting 2 weeks?**
  A: Give 50 feedback items, or temporarily reduce threshold.

- **Q: What if users are confused by the UI?**
  A: Iterate on messaging, add tooltips, improve copy.

---

**Good luck! You've got this. 🚀**
