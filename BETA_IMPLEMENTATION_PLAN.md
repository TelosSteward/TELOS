# TELOS Beta Testing - Complete Implementation Plan

## Executive Summary

This document outlines the complete implementation of the TELOS Beta Testing System - a two-phase preference testing framework that validates TELOS governance through real user feedback.

**Current Status:** Core framework built (70% complete). Integration and deployment tasks remaining (30%).

---

## ✅ COMPLETED (What's Already Built)

### 1. Beta Testing Data Infrastructure
**Location:** `observatory/beta_testing/`

**Files Created:**
- ✅ `beta_session_manager.py` - Complete session management system
- ✅ `beta_feedback.py` - Full feedback UI components
- ✅ `__init__.py` - Module exports

**Features Implemented:**
- Random test condition assignment (40% baseline, 40% TELOS, 20% head-to-head)
- Dual-path response generation (baseline + TELOS in parallel)
- Fidelity tracking for both paths
- Session persistence (JSONL format)
- Data export for analysis
- Aggregate statistics dashboard

### 2. Feedback Collection UI
**Location:** `observatory/components/beta_feedback.py`

**Components Built:**
- ✅ Single-blind rating UI (thumbs up/down)
- ✅ Head-to-head comparison UI (side-by-side preference selection)
- ✅ Optional qualitative feedback (text input for "why")
- ✅ Researcher metrics dashboard
- ✅ Data export button

### 3. Beta Onboarding
**Location:** `observatory/components/beta_onboarding.py`

**Status:** ✅ Updated text to reflect preference testing (not deltas)

**Changes Made:**
- Explained feedback collection process
- Clarified: no personal info, no tracking, no KYC
- Emphasized testing data for internal use only
- Updated consent language

### 4. Data Models
**Location:** `observatory/beta_testing/beta_session_manager.py`

**Models Implemented:**
- ✅ `BetaSession` - Complete session tracking
- ✅ `FeedbackData` - Per-turn feedback with metrics
- ✅ `ConversationGoal` - Goal validation (implemented but will be removed)
- ✅ Test condition types (`TestCondition` enum)

---

## 🔄 IN PROGRESS (What Needs to Be Done)

### Phase 1: Streamlit Cloud Compatibility (30 mins)

**Task:** Create deployment configuration for seamless cloud deployment

**Files to Create:**

#### 1.1 `.streamlit/config.toml`
```toml
[theme]
primaryColor = "#FFD700"
backgroundColor = "#1a1a1a"
secondaryBackgroundColor = "#2d2d2d"
textColor = "#e0e0e0"
font = "monospace"

[server]
headless = true
port = 8501
enableCORS = false
enableXsrfProtection = true

[browser]
gatherUsageStats = false
```

#### 1.2 `.streamlit/secrets.toml` (template)
```toml
# Template - users must add their own key
MISTRAL_API_KEY = "your-mistral-api-key-here"
```

#### 1.3 `observatory/utils/env_helper.py` (NEW FILE)
```python
"""
Environment detection and secrets management for Streamlit Cloud compatibility.
"""
import os
import streamlit as st
from pathlib import Path

def is_streamlit_cloud():
    """Detect if running on Streamlit Cloud."""
    return os.getenv("STREAMLIT_RUNTIME_ENV") == "cloud"

def get_api_key(key_name: str, default=None):
    """
    Get API key from Streamlit secrets or environment variables.

    Args:
        key_name: Name of the API key (e.g., "MISTRAL_API_KEY")
        default: Default value if key not found

    Returns:
        API key value
    """
    # Try Streamlit secrets first (cloud deployment)
    try:
        return st.secrets[key_name]
    except (AttributeError, KeyError, FileNotFoundError):
        pass

    # Fall back to environment variables (local development)
    return os.getenv(key_name, default)

def get_data_dir(subdir: str = "beta_testing/data"):
    """
    Get data directory path (cloud-compatible).

    Args:
        subdir: Subdirectory name

    Returns:
        Path object for data directory
    """
    if is_streamlit_cloud():
        # On cloud, use /tmp for temporary storage
        data_dir = Path("/tmp") / subdir
    else:
        # Local development, use project directory
        data_dir = Path(__file__).parent.parent.parent / subdir

    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir
```

#### 1.4 `requirements.txt` (Update)
Add missing dependencies:
```txt
# Core
streamlit>=1.28.0
numpy>=1.24.0
pandas>=2.0.0

# TELOS Core
sentence-transformers>=2.2.0
torch>=2.0.0

# LLM Clients
mistralai>=0.1.0

# Utilities
python-dotenv>=1.0.0
```

**Implementation Steps:**
1. Create `.streamlit/` directory and config files
2. Create `env_helper.py` utility
3. Update `requirements.txt`
4. Update all API key access to use `get_api_key()`
5. Update all file paths to use `get_data_dir()`

**Testing:**
- Local: Run with `.env` file
- Cloud: Deploy to Streamlit Cloud, add secrets via UI

---

### Phase 2: Beta-Only Mode UI (45 mins)

**Task:** Restrict beta users to BETA tab only, gray out other tabs

**Files to Modify:**

#### 2.1 `observatory/main.py`

**Changes Needed:**

```python
# After beta consent given, check if beta-only mode
has_beta_consent = st.session_state.get('beta_consent_given', False)

if has_beta_consent:
    # Check if beta completed
    beta_completed = st.session_state.get('beta_completed', False)

    if not beta_completed:
        # BETA-ONLY MODE: Show only BETA tab, gray out others
        st.markdown("""
        <style>
        /* Gray out DEMO and TELOS tabs */
        button[data-testid="tab_demo"],
        button[data-testid="tab_telos"] {
            opacity: 0.4 !important;
            pointer-events: none !important;
            cursor: not-allowed !important;
        }

        /* Highlight BETA tab */
        button[data-testid="tab_beta"] {
            border: 2px solid #FFD700 !important;
            box-shadow: 0 0 10px rgba(255, 215, 0, 0.5) !important;
        }
        </style>
        """, unsafe_allow_html=True)

        # Force active tab to BETA
        if st.session_state.active_tab != "BETA":
            st.session_state.active_tab = "BETA"

        # Show unlock message
        st.info("🔒 Complete beta testing to unlock full Observatory access | Progress: X/50 turns")
```

**Implementation Steps:**
1. Add beta completion check
2. Add CSS to gray out tabs
3. Force BETA tab active
4. Show progress indicator
5. Add unlock logic (see Phase 5)

---

### Phase 3: Two-Phase Beta Flow (2 hours)

**Task:** Implement PA calibration phase followed by beta testing phase

**Files to Modify:**

#### 3.1 `observatory/core/state_manager.py`

**Add PA Calibration Tracking:**

```python
# Add to ObservatoryState dataclass
@dataclass
class ObservatoryState:
    # ... existing fields ...

    # Beta Testing State
    beta_mode: bool = False
    beta_phase: str = "calibration"  # "calibration" or "testing"
    pa_calibration_complete: bool = False
    beta_test_condition: Optional[str] = None  # Set after PA calibration
    beta_session_id: Optional[str] = None
```

**Add Methods:**

```python
def is_pa_calibration_phase(self) -> bool:
    """Check if still in PA calibration phase."""
    return (
        self.state.beta_mode and
        self.state.beta_phase == "calibration" and
        not self.state.pa_calibration_complete
    )

def is_beta_testing_phase(self) -> bool:
    """Check if in beta testing phase."""
    return (
        self.state.beta_mode and
        self.state.beta_phase == "testing" and
        self.state.pa_calibration_complete
    )

def mark_pa_calibration_complete(self):
    """Mark PA calibration as complete and transition to testing phase."""
    self.state.pa_calibration_complete = True
    self.state.beta_phase = "testing"
    logger.info("PA calibration complete, transitioning to beta testing phase")
```

#### 3.2 `observatory/components/conversation_display.py`

**Add Phase Detection and UI:**

**Step 1: Show PA Calibration Indicator (Turns 1-10)**

```python
def _render_pa_calibration_indicator(self):
    """Show PA calibration status during calibration phase."""
    if not self.state_manager.is_pa_calibration_phase():
        return

    turn_count = len(self.state_manager.get_all_turns())

    # Check if PA has converged (from UnifiedGovernanceSteward)
    if hasattr(self.state_manager, '_telos_steward'):
        pa_extractor = getattr(self.state_manager._telos_steward, 'pa_extractor', None)
        if pa_extractor:
            converged = getattr(pa_extractor, 'converged', False)
            convergence_turn = getattr(pa_extractor, 'convergence_turn', None)

            if converged and convergence_turn:
                # PA Established!
                st.markdown("""
                <div style="
                    background: linear-gradient(135deg, #2d2d2d 0%, #1a1a1a 100%);
                    border: 2px solid #4CAF50;
                    border-radius: 10px;
                    padding: 15px;
                    margin: 15px 0;
                    text-align: center;
                ">
                    <div style="color: #4CAF50; font-size: 20px; font-weight: bold;">
                        ✓ Primacy Attractor Established
                    </div>
                    <div style="color: #e0e0e0; font-size: 14px; margin-top: 5px;">
                        Converged at turn {convergence_turn} | Ready for beta testing
                    </div>
                </div>
                """.format(convergence_turn=convergence_turn), unsafe_allow_html=True)

                # Mark calibration complete
                self.state_manager.mark_pa_calibration_complete()

                # Initialize beta session
                from observatory.beta_testing import BetaSessionManager
                beta_manager = BetaSessionManager()
                beta_session = beta_manager.start_session()

                # Assign test condition
                test_condition = beta_manager.assign_test_condition(beta_session)

                # Store in state
                st.session_state.beta_session = beta_session
                st.session_state.beta_manager = beta_manager
                self.state_manager.state.beta_test_condition = test_condition

                return

    # Still calibrating
    st.markdown("""
    <div style="
        background-color: #2d2d2d;
        border: 2px solid #FFA500;
        border-radius: 10px;
        padding: 15px;
        margin: 15px 0;
        text-align: center;
    ">
        <div style="color: #FFA500; font-size: 18px; font-weight: bold;">
            🔄 Calibrating Primacy Attractor...
        </div>
        <div style="color: #e0e0e0; font-size: 14px; margin-top: 5px;">
            Turn {turn} of ~10 | Establishing conversation purpose
        </div>
    </div>
    """.format(turn=turn_count), unsafe_allow_html=True)
```

**Step 2: Generate Responses Based on Phase**

```python
def _handle_user_message_beta_mode(self, message: str):
    """Handle user message in beta mode (two-phase approach)."""

    if self.state_manager.is_pa_calibration_phase():
        # PHASE 1: PA Calibration
        # Just use baseline Mistral, run PA extraction in background
        self.state_manager.add_user_message(message)  # Existing logic

    elif self.state_manager.is_beta_testing_phase():
        # PHASE 2: Beta Testing
        # Generate both baseline and TELOS responses
        beta_manager = st.session_state.get('beta_manager')
        beta_session = st.session_state.get('beta_session')

        if not beta_manager or not beta_session:
            st.error("Beta session not initialized. Please restart.")
            return

        # Generate dual responses
        turn_number = len(self.state_manager.get_all_turns())
        response_data = beta_manager.generate_dual_response(
            user_message=message,
            state_manager=self.state_manager,
            turn_number=turn_number
        )

        # Store response data for feedback
        response_data['user_message'] = message
        st.session_state[f'beta_response_data_{turn_number}'] = response_data

        # Determine which response to show based on test condition
        test_condition = self.state_manager.state.beta_test_condition

        if test_condition == "single_blind_baseline":
            # Show baseline only
            shown_response = response_data['baseline_response']
        elif test_condition == "single_blind_telos":
            # Show TELOS only
            shown_response = response_data['telos_response']
        else:  # head_to_head
            # Will show both in feedback UI
            shown_response = None  # Don't show yet

        # Add to conversation (if single-blind)
        if shown_response:
            # Add turn with shown response
            # (implementation depends on existing add_user_message logic)
            pass
```

**Step 3: Render Feedback UI After Response**

```python
def _render_beta_feedback_if_needed(self, turn_number: int):
    """Render feedback UI after each response in beta testing phase."""
    if not self.state_manager.is_beta_testing_phase():
        return

    # Check if feedback already given
    if st.session_state.get(f'beta_feedback_{turn_number}'):
        return

    # Get response data
    response_data = st.session_state.get(f'beta_response_data_{turn_number}')
    if not response_data:
        return

    # Get beta UI components
    from observatory.components.beta_feedback import BetaFeedbackUI
    beta_ui = BetaFeedbackUI(st.session_state.beta_manager)

    # Render appropriate feedback UI based on test condition
    test_condition = self.state_manager.state.beta_test_condition

    if test_condition in ["single_blind_baseline", "single_blind_telos"]:
        beta_ui.render_single_blind_feedback(turn_number, response_data)
    elif test_condition == "head_to_head":
        beta_ui.render_head_to_head_comparison(turn_number, response_data)
```

**Implementation Steps:**
1. Update `ObservatoryState` with beta tracking fields
2. Add PA calibration indicator to conversation display
3. Modify message handling to detect phase
4. Implement dual-path response generation for testing phase
5. Integrate feedback UI after responses
6. Test phase transitions

---

### Phase 4: Remove Conversation Goal (15 mins)

**Task:** Remove conversation goal capture (not needed for beta)

**Files to Modify:**

#### 4.1 `observatory/beta_testing/beta_session_manager.py`

```python
# Remove ConversationGoal from imports
# Remove conversation_goal field from BetaSession dataclass
```

#### 4.2 `observatory/components/beta_feedback.py`

```python
# Remove render_conversation_goal_input() method
# Remove render_conversation_goal_validation() method
# Remove _record_goal_validation() method
```

**Implementation Steps:**
1. Remove conversation goal code
2. Update data models
3. Test session creation without goals

---

### Phase 5: Beta Completion Tracking (30 mins)

**Task:** Track beta completion (2 weeks OR minimum turns) and unlock full access

**Files to Modify:**

#### 5.1 `observatory/beta_testing/beta_session_manager.py`

**Add Completion Logic:**

```python
def check_beta_completion(self, session: BetaSession) -> bool:
    """
    Check if beta testing is complete.

    Criteria:
    - 2 weeks elapsed OR
    - Minimum 50 turns with feedback

    Args:
        session: Beta session to check

    Returns:
        True if beta testing complete
    """
    from datetime import datetime, timedelta

    # Check time elapsed
    start_time = datetime.fromisoformat(session.start_time)
    elapsed = datetime.now() - start_time

    if elapsed >= timedelta(days=14):
        return True

    # Check turns with feedback
    feedback_count = len(session.feedback_items)
    if feedback_count >= 50:
        return True

    return False

def get_beta_progress(self, session: BetaSession) -> Dict[str, Any]:
    """Get beta completion progress."""
    from datetime import datetime, timedelta

    start_time = datetime.fromisoformat(session.start_time)
    elapsed = datetime.now() - start_time
    days_remaining = max(0, 14 - elapsed.days)

    feedback_count = len(session.feedback_items)
    turns_remaining = max(0, 50 - feedback_count)

    return {
        "days_elapsed": elapsed.days,
        "days_remaining": days_remaining,
        "feedback_count": feedback_count,
        "turns_remaining": turns_remaining,
        "completed": self.check_beta_completion(session)
    }
```

#### 5.2 `observatory/main.py`

**Add Completion Check:**

```python
# After beta consent, check completion
if has_beta_consent:
    beta_manager = st.session_state.get('beta_manager')
    beta_session = st.session_state.get('beta_session')

    if beta_manager and beta_session:
        progress = beta_manager.get_beta_progress(beta_session)

        if progress['completed'] and not st.session_state.get('beta_completed'):
            # Unlock full access!
            st.session_state.beta_completed = True
            st.balloons()
            st.success("🎉 Beta testing complete! Full Observatory access unlocked!")

            # End beta session and save data
            beta_manager.end_session(beta_session)

        # Show progress
        if not progress['completed']:
            st.sidebar.markdown(f"""
            **Beta Progress:**
            - Days: {progress['days_elapsed']}/14
            - Turns: {progress['feedback_count']}/50
            """)
```

**Implementation Steps:**
1. Add completion check logic
2. Add progress tracking
3. Implement unlock behavior
4. Test completion criteria

---

### Phase 6: Integration Testing (1 hour)

**Task:** End-to-end testing of complete beta flow

**Test Cases:**

#### 6.1 Fresh User Journey
1. User lands on beta onboarding
2. Reads consent info
3. Accepts consent
4. Sees BETA tab highlighted, others grayed
5. Starts conversation
6. Sees "Calibrating PA..." indicator
7. Has normal conversation for ~10 turns
8. Sees "PA Established ✓"
9. Gets randomly assigned to test condition
10. Sees appropriate feedback UI
11. Provides feedback
12. Continues until completion
13. Unlocks full access

#### 6.2 Test Conditions
- Test single-blind baseline flow
- Test single-blind TELOS flow
- Test head-to-head comparison flow

#### 6.3 Data Validation
- Check JSONL file creation
- Verify feedback data structure
- Confirm fidelity metrics captured
- Test data export

#### 6.4 Cloud Deployment
- Deploy to Streamlit Cloud
- Test secrets management
- Verify file paths work
- Check performance

**Testing Checklist:**
- [ ] Fresh user can complete full flow
- [ ] PA calibration works correctly
- [ ] All 3 test conditions work
- [ ] Feedback is recorded properly
- [ ] Beta completion unlocks access
- [ ] Data export generates valid JSON
- [ ] Cloud deployment works
- [ ] No console errors

---

## 📊 Implementation Timeline

**Estimated Total Time: 5-6 hours**

| Phase | Task | Time | Status |
|-------|------|------|--------|
| ✅ 0 | Core framework & data models | 3h | Complete |
| ✅ 0 | Beta onboarding updates | 30m | Complete |
| 🔄 1 | Streamlit Cloud compatibility | 30m | Pending |
| 🔄 2 | Beta-only mode UI | 45m | Pending |
| 🔄 3 | Two-phase beta flow | 2h | Pending |
| 🔄 4 | Remove conversation goal | 15m | Pending |
| 🔄 5 | Beta completion tracking | 30m | Pending |
| 🔄 6 | Integration testing | 1h | Pending |

---

## 🗂️ File Structure

```
observatory/
├── beta_testing/                    # ✅ Complete
│   ├── __init__.py
│   ├── beta_session_manager.py     # Session management, A/B logic
│   └── data/                        # Auto-created for JSONL storage
│       └── beta_sessions.jsonl
│
├── components/
│   ├── beta_onboarding.py           # ✅ Updated
│   ├── beta_feedback.py             # ✅ Complete
│   ├── conversation_display.py      # 🔄 Needs phase integration
│   └── ...
│
├── core/
│   └── state_manager.py             # 🔄 Needs beta state tracking
│
├── utils/
│   └── env_helper.py                # 🔄 NEW - Cloud compatibility
│
└── main.py                          # 🔄 Needs beta-only mode

.streamlit/                          # 🔄 NEW - Cloud config
├── config.toml
└── secrets.toml (template)

beta_testing/data/                   # Auto-created
└── beta_sessions.jsonl             # Persistent storage

requirements.txt                     # 🔄 Update dependencies
```

---

## 🚀 Deployment Checklist

### Local Development
- [ ] Install dependencies (`pip install -r requirements.txt`)
- [ ] Set `MISTRAL_API_KEY` in `.env` or environment
- [ ] Run: `streamlit run observatory/main.py`
- [ ] Test full beta flow

### Streamlit Cloud
- [ ] Push code to GitHub
- [ ] Deploy from Streamlit Cloud dashboard
- [ ] Add `MISTRAL_API_KEY` to Secrets (cloud UI)
- [ ] Test cloud deployment
- [ ] Monitor performance and errors

---

## 📈 Expected Outcomes

With 100+ beta test sessions, you'll be able to demonstrate:

- **"TELOS responses preferred X% more often than baseline in head-to-head comparisons"**
- **"Users gave thumbs up to Y% of TELOS responses vs. Z% for baseline"**
- **"Higher fidelity scores correlate with AAA% increase in user preference"**
- **"TELOS maintains governance alignment over long conversations"**

This data is **publication-ready** and **grant-application-ready**.

---

## 🔧 Maintenance & Monitoring

### Data Management
- JSONL files stored in `beta_testing/data/`
- Export to JSON for analysis
- Clean up completed sessions periodically

### Monitoring
- Track beta completion rates
- Monitor test condition distribution
- Analyze feedback patterns
- Review fidelity correlations

### Updates
- Adjust test condition ratios based on results
- Refine feedback UI based on user behavior
- Update completion criteria if needed

---

## 📞 Support

For questions or issues during implementation:
- Technical questions: Check this implementation plan
- Architecture decisions: Review code comments
- Cloud deployment: See Streamlit Cloud docs

---

## ✨ Success Criteria

Beta testing system is complete when:

✅ Users can complete full beta flow without errors
✅ All 3 test conditions function correctly
✅ Feedback is recorded and exportable
✅ PA calibration works as expected
✅ Beta completion unlocks full access
✅ System runs on Streamlit Cloud
✅ Data export generates valid analysis-ready JSON

---

*Last Updated: 2025-01-08*
*Status: 70% Complete - Core framework built, integration pending*
