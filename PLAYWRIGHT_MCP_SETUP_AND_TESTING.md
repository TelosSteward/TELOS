# Playwright MCP Setup and Testing Guide

**Status:** ✅ Configured and Ready
**Date:** 2025-11-08
**Purpose:** Automated testing of TELOS Beta Testing implementation

---

## SETUP COMPLETE

### 1. Configuration Added

**File:** `.mcp.json`

Added Playwright MCP server:
```json
"playwright": {
  "command": "npx",
  "args": [
    "-y",
    "@playwright/mcp"
  ]
}
```

### 2. Verification

✅ Package available: `@playwright/mcp` (v0.0.46)
✅ Command works: `npx -y @playwright/mcp --help`
✅ MCP server configured

### 3. Next Steps

**⚠️ IMPORTANT:** You must **restart Claude Code** for the Playwright MCP to load.

After restart, Playwright tools will be available with `mcp__playwright__*` prefix.

---

## PLAYWRIGHT MCP CAPABILITIES

### Available Tools (After Restart)

Once Claude Code loads the Playwright MCP, you'll have access to browser automation tools:

- **mcp__playwright__navigate** - Navigate to URLs
- **mcp__playwright__click** - Click elements
- **mcp__playwright__fill** - Fill form inputs
- **mcp__playwright__screenshot** - Take screenshots
- **mcp__playwright__evaluate** - Execute JavaScript in browser
- **mcp__playwright__waitFor** - Wait for elements/conditions
- And more...

### What This Enables

✅ **Automated UI Testing** - Click through entire beta flow
✅ **Screenshot Capture** - Visual verification of UI states
✅ **Session State Inspection** - Check Streamlit session variables
✅ **Multi-scenario Testing** - Test all 8 scenarios automatically
✅ **Time Manipulation** - Mock system time for 2-week test

---

## TEST PLAN: Beta Testing Implementation

### Test Suite Overview

**Target:** http://localhost:8502 (Streamlit Observatory)
**Scenarios:** 8 critical test cases
**Estimated Time:** 15-20 minutes (automated)
**Manual Equivalent:** 3-4 hours

---

### TEST SCENARIO 1: Beta Onboarding Flow

**Objective:** Verify beta consent process works correctly

**Steps:**
1. Navigate to http://localhost:8502
2. Verify beta onboarding screen appears
3. Locate "I consent" button or checkbox
4. Click consent
5. Verify redirect to BETA tab
6. Verify DEMO and TELOS tabs are grayed out/disabled
7. Verify BETA tab is highlighted

**Success Criteria:**
- ✅ Consent screen renders
- ✅ Consent button clickable
- ✅ Redirect works
- ✅ Tab locking active
- ✅ Beta tab highlighted

**Playwright Commands:**
```javascript
// Navigate
await playwright.navigate('http://localhost:8502')

// Screenshot initial state
await playwright.screenshot('beta_onboarding_initial.png')

// Find and click consent
await playwright.click('button:has-text("I consent")')

// Wait for redirect
await playwright.waitFor('div[data-testid="stTab"]:has-text("BETA")')

// Verify tab states
await playwright.evaluate(`
  const tabs = document.querySelectorAll('[data-testid="stTab"]');
  return {
    demo_disabled: tabs[0].disabled || tabs[0].style.opacity < 1,
    telos_disabled: tabs[1].disabled || tabs[1].style.opacity < 1,
    beta_active: tabs[2].classList.contains('active')
  };
`)

// Screenshot post-consent
await playwright.screenshot('beta_onboarding_complete.png')
```

---

### TEST SCENARIO 2: PA Calibration Phase (Turns 1-10)

**Objective:** Verify no feedback UI appears during calibration

**Steps:**
1. Start in beta mode (consent given)
2. Send 10 messages to establish PA
3. For each turn, verify NO feedback buttons appear
4. Verify PA status indicator shows "Calibrating" → "Established"

**Success Criteria:**
- ✅ Turns 1-10: No thumbs up/down buttons
- ✅ PA status indicator visible
- ✅ Conversation works normally
- ✅ No phase transition message yet

**Playwright Commands:**
```javascript
// Send 10 messages
for (let i = 1; i <= 10; i++) {
  // Find input
  await playwright.fill('input[aria-label="chat input"]', `Test message ${i}`)

  // Send
  await playwright.click('button:has-text("Send")' )

  // Wait for response
  await playwright.waitFor('.stChatMessage:last-child')

  // Verify NO feedback buttons
  const hasFeedback = await playwright.evaluate(`
    return document.querySelectorAll('button:has-text("👍")').length > 0;
  `)

  if (hasFeedback) {
    throw new Error(`Feedback buttons appeared at turn ${i} (should be turn 11+)`)
  }

  // Screenshot every 3 turns
  if (i % 3 === 0) {
    await playwright.screenshot(`pa_calibration_turn_${i}.png`)
  }
}

// Verify PA status
const paStatus = await playwright.evaluate(`
  return document.querySelector('div:has-text("PA Established")') !== null;
`)
```

---

### TEST SCENARIO 3: Phase Transition at Turn 11

**Objective:** Verify phase transition message appears exactly once

**Steps:**
1. Continue from turn 10
2. Send 11th message
3. Wait for response
4. Verify phase transition message appears
5. Refresh page
6. Verify message does NOT appear again

**Success Criteria:**
- ✅ Message appears at turn 11
- ✅ Message contains "🎯 PA Established!"
- ✅ Message says "Beta preference testing is active"
- ✅ Message shows exactly once (not on refresh)

**Playwright Commands:**
```javascript
// Send turn 11
await playwright.fill('input[aria-label="chat input"]', 'Turn 11 test message')
await playwright.click('button:has-text("Send")')
await playwright.waitFor('.stChatMessage:last-child')

// Wait for phase transition message
const transitionMessage = await playwright.waitFor('div:has-text("🎯 PA Established!")', {timeout: 5000})

// Verify message content
const messageText = await playwright.evaluate(`
  return document.querySelector('div:has-text("PA Established")').textContent;
`)

// Screenshot
await playwright.screenshot('phase_transition_turn_11.png')

// Refresh page
await playwright.evaluate('location.reload()')
await playwright.waitFor('body')

// Verify message does NOT appear again
const messageAgain = await playwright.evaluate(`
  return document.querySelector('div:has-text("🎯 PA Established!")') !== null;
`)

if (messageAgain) {
  throw new Error('Phase transition message appeared twice (should be once)')
}
```

---

### TEST SCENARIO 4: Beta Feedback UI (Turns 11+)

**Objective:** Verify feedback buttons work correctly

**Steps:**
1. At turn 11, verify feedback buttons appear
2. Click thumbs up
3. Verify confirmation message
4. Verify cannot rate again
5. Send turn 12
6. Click thumbs down
7. Verify feedback recorded

**Success Criteria:**
- ✅ Buttons appear for turns 11+
- ✅ Thumbs up works
- ✅ Thumbs down works
- ✅ Confirmation shows
- ✅ Duplicate prevention works
- ✅ Feedback data stored

**Playwright Commands:**
```javascript
// Verify buttons exist
const hasButtons = await playwright.evaluate(`
  return document.querySelectorAll('button:has-text("👍")').length > 0 &&
         document.querySelectorAll('button:has-text("👎")').length > 0;
`)

if (!hasButtons) {
  throw new Error('Feedback buttons missing at turn 11')
}

// Click thumbs up
await playwright.click('button:has-text("👍")')

// Wait for confirmation
await playwright.waitFor('div:has-text("✓ Thank you for your feedback!")')

// Screenshot
await playwright.screenshot('feedback_thumbs_up.png')

// Verify button gone (replaced with confirmation)
const buttonsGone = await playwright.evaluate(`
  return document.querySelectorAll('button:has-text("👍")').length === 0;
`)

// Send turn 12
await playwright.fill('input[aria-label="chat input"]', 'Turn 12 test message')
await playwright.click('button:has-text("Send")')
await playwright.waitFor('.stChatMessage:last-child')

// Click thumbs down
await playwright.click('button:has-text("👎")')
await playwright.waitFor('div:has-text("✓ Thank you for your feedback!")')

// Check session state
const feedbackData = await playwright.evaluate(`
  return window.streamlit.session_state.beta_feedback;
`)

console.log('Feedback data:', feedbackData)
```

---

### TEST SCENARIO 5: Progress Tracking in Sidebar

**Objective:** Verify progress display updates correctly

**Steps:**
1. Give 1 feedback
2. Check sidebar for "Beta Progress"
3. Verify shows 1/50 feedback
4. Give 5 more feedbacks
5. Verify count updates to 6/50

**Success Criteria:**
- ✅ Progress section appears in sidebar
- ✅ Shows days elapsed
- ✅ Shows feedback count
- ✅ Counts update correctly
- ✅ Shows "either one" criteria

**Playwright Commands:**
```javascript
// Look for progress section
const hasProgress = await playwright.evaluate(`
  return document.querySelector('div:has-text("Beta Progress")') !== null;
`)

// Get progress text
const progressText = await playwright.evaluate(`
  const progress = document.querySelector('div:has-text("Beta Progress")');
  return progress ? progress.textContent : null;
`)

console.log('Progress:', progressText)

// Verify format includes "Days: X/14" and "Feedback: X/50"
if (!progressText.includes('Days:') || !progressText.includes('Feedback:')) {
  throw new Error('Progress format incorrect')
}

// Screenshot
await playwright.screenshot('progress_sidebar.png')

// Give 5 more feedbacks and verify count updates
// (implement feedback loop)
```

---

### TEST SCENARIO 6: Beta Completion Criteria

**Objective:** Verify completion triggers correctly

**Test 6A: 50 Feedbacks**

**Steps:**
1. Give 50 feedbacks (automate)
2. Verify balloons animation
3. Verify success message
4. Verify tabs unlock

**Playwright Commands:**
```javascript
// Automate 50 feedbacks
for (let i = 11; i <= 60; i++) {
  // Send message
  await playwright.fill('input[aria-label="chat input"]', `Feedback test ${i}`)
  await playwright.click('button:has-text("Send")')
  await playwright.waitFor('.stChatMessage:last-child')

  // Give feedback (alternate thumbs up/down)
  const button = i % 2 === 0 ? '👍' : '👎'
  await playwright.click(`button:has-text("${button}")`)
  await playwright.waitFor('div:has-text("✓ Thank you for your feedback!")')
}

// Wait for completion
await playwright.waitFor('div:has-text("🎉 Beta Testing Complete!")', {timeout: 10000})

// Screenshot celebration
await playwright.screenshot('completion_50_feedbacks.png')

// Verify tabs unlocked
const tabsUnlocked = await playwright.evaluate(`
  const tabs = document.querySelectorAll('[data-testid="stTab"]');
  return {
    demo_enabled: tabs[0].disabled === false && tabs[0].style.opacity >= 1,
    telos_enabled: tabs[1].disabled === false && tabs[1].style.opacity >= 1
  };
`)

console.log('Tabs unlocked:', tabsUnlocked)
```

**Test 6B: 2 Weeks Elapsed**

**Steps:**
1. Mock system time +2 weeks
2. Reload page
3. Verify completion triggers

**Playwright Commands:**
```javascript
// Mock time (requires Playwright clock API)
await playwright.evaluate(`
  // Get current start time
  const startTime = window.streamlit.session_state.beta_start_time;
  const twoWeeksLater = new Date(startTime);
  twoWeeksLater.setDate(twoWeeksLater.getDate() + 14);

  // Override Date.now()
  const originalNow = Date.now;
  Date.now = () => twoWeeksLater.getTime();
`)

// Reload to trigger completion check
await playwright.evaluate('location.reload()')
await playwright.waitFor('body')

// Should see completion
await playwright.waitFor('div:has-text("🎉 Beta Testing Complete!")')
```

---

### TEST SCENARIO 7: State Persistence

**Objective:** Verify session state persists across page refreshes

**Steps:**
1. Give 5 feedbacks
2. Note feedback count
3. Refresh page
4. Verify feedback count same
5. Verify cannot re-rate turns

**Success Criteria:**
- ✅ Feedback data persists
- ✅ Progress maintains
- ✅ Flags persist (phase_transition_shown, beta_feedback_{turn})

**Playwright Commands:**
```javascript
// Get initial state
const beforeRefresh = await playwright.evaluate(`
  return {
    feedback_count: window.streamlit.session_state.beta_feedback?.length,
    transition_shown: window.streamlit.session_state.beta_phase_transition_shown,
    start_time: window.streamlit.session_state.beta_start_time
  };
`)

// Refresh
await playwright.evaluate('location.reload()')
await playwright.waitFor('body')

// Get state after refresh
const afterRefresh = await playwright.evaluate(`
  return {
    feedback_count: window.streamlit.session_state.beta_feedback?.length,
    transition_shown: window.streamlit.session_state.beta_phase_transition_shown,
    start_time: window.streamlit.session_state.beta_start_time
  };
`)

// Compare
if (beforeRefresh.feedback_count !== afterRefresh.feedback_count) {
  throw new Error('Feedback data lost on refresh')
}

console.log('State persisted correctly:', afterRefresh)
```

---

### TEST SCENARIO 8: Edge Cases

**Objective:** Verify edge cases handled gracefully

**Test Cases:**

**8A: Non-Beta User**
- Navigate without beta consent
- Verify no beta UI elements appear

**8B: Completed Beta User**
- Set beta_completed = true
- Verify tabs unlocked
- Verify no progress shown
- Verify no feedback UI

**8C: Multiple Tabs**
- Open two browser tabs
- Give feedback in tab 1
- Verify state syncs in tab 2

**Playwright Commands:** (Sample for 8A)
```javascript
// Clear session state (simulate fresh user who doesn't consent)
await playwright.evaluate(`
  delete window.streamlit.session_state.beta_consent_given;
`)

await playwright.evaluate('location.reload()')

// Verify no beta elements
const hasBetaUI = await playwright.evaluate(`
  return {
    has_transition: document.querySelector('div:has-text("PA Established")') !== null,
    has_feedback: document.querySelectorAll('button:has-text("👍")').length > 0,
    has_progress: document.querySelector('div:has-text("Beta Progress")') !== null
  };
`)

if (hasBetaUI.has_transition || hasBetaUI.has_feedback || hasBetaUI.has_progress) {
  throw new Error('Beta UI shown to non-beta user')
}
```

---

## RUNNING THE TESTS

### Prerequisites

1. ✅ Playwright MCP configured in .mcp.json
2. ⚠️ **Restart Claude Code** (REQUIRED)
3. ✅ Streamlit app running on http://localhost:8502
4. ✅ Fresh session (no beta state)

### Execution

**After restarting Claude Code**, you can run tests by asking Claude:

```
"Use Playwright MCP to run test scenario 1: Beta Onboarding Flow"
```

Claude will then use the mcp__playwright__ tools to:
- Navigate to the app
- Interact with elements
- Take screenshots
- Verify conditions
- Report results

### Test Order

Run in sequence:
1. Scenario 1: Onboarding
2. Scenario 2: PA Calibration
3. Scenario 3: Phase Transition
4. Scenario 4: Feedback UI
5. Scenario 5: Progress Tracking
6. Scenario 6A: 50 Feedbacks Completion
7. Scenario 7: State Persistence
8. Scenario 8: Edge Cases

(Skip Scenario 6B: Time-based completion for now - complex)

---

## EXPECTED OUTPUTS

### Screenshots

After running all tests, you should have:
- `beta_onboarding_initial.png`
- `beta_onboarding_complete.png`
- `pa_calibration_turn_3.png`
- `pa_calibration_turn_6.png`
- `pa_calibration_turn_9.png`
- `phase_transition_turn_11.png`
- `feedback_thumbs_up.png`
- `feedback_thumbs_down.png`
- `progress_sidebar.png`
- `completion_50_feedbacks.png`
- And more...

### Test Report

Format:
```
TEST SCENARIO X: [Name]
Status: ✅ PASS / ❌ FAIL
Duration: Xs
Errors: None / [Error details]
Screenshots: [List]
Notes: [Observations]
```

---

## TROUBLESHOOTING

### Playwright MCP Not Loading

**Problem:** MCP tools not available after adding to .mcp.json

**Solution:**
1. Verify .mcp.json syntax is valid (no trailing commas)
2. **Restart Claude Code completely**
3. Check MCP server status in Claude Code settings
4. Try running: `npx -y @playwright/mcp` manually to verify install

### Streamlit App Not Responding

**Problem:** Playwright can't navigate to app

**Solution:**
1. Verify app is running: `ps aux | grep streamlit`
2. Check correct port: http://localhost:8502
3. Try accessing in regular browser first
4. Check for firewall blocking

### Test Failures

**Problem:** Tests fail unexpectedly

**Solution:**
1. Check screenshots to see what UI looked like
2. Verify Streamlit session state in browser console
3. Run test scenarios manually first
4. Check if app behavior changed
5. Review error messages from Playwright

---

## NEXT STEPS

1. **NOW:** Restart Claude Code
2. **THEN:** Verify Playwright MCP loaded (check for mcp__playwright__ tools)
3. **THEN:** Run test scenarios 1-8
4. **THEN:** Review results and fix any bugs
5. **FINALLY:** Commit tested code and deploy

---

## BENEFITS OF PLAYWRIGHT TESTING

✅ **Fast:** 15-20 minutes vs 3-4 hours manual testing
✅ **Comprehensive:** Tests all 8 scenarios systematically
✅ **Repeatable:** Can run tests after any code change
✅ **Visual:** Screenshots provide proof of UI states
✅ **Reliable:** Catches edge cases humans miss
✅ **Documented:** Test commands serve as documentation

---

**Setup Status:** ✅ COMPLETE
**Next Action:** Restart Claude Code to load Playwright MCP
**Ready For:** Automated testing

---

*Document created: 2025-11-08*
*Playwright MCP Version: 0.0.46*
*Target App: TELOS Observatory Beta Testing*
