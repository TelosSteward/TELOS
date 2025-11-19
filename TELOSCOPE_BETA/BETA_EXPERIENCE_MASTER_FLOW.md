# TELOS BETA Experience - Complete User Journey

**Date:** November 18, 2025
**Status:** Production-Ready Documentation
**Purpose:** Comprehensive guide to the BETA user experience flow

---

## Overview

TELOS BETA is a controlled A/B testing environment that allows users to experience both native LLM responses and TELOS-governed responses while TELOS maintains alignment monitoring in the background. This document details every step of the BETA journey.

---

## Phase 0: Entry & Consent

### Tab Structure on Entry
**Before BETA consent:**
- **DEMO Tab** (left): Active (starting point)
- **BETA Tab** (right): Locked (unlocks after completing DEMO slide 12)

**After BETA consent:**
- **BETA Tab** (left): Active
- **TELOS Tab** (right): Locked/greyed out (unlocks after BETA completion)
- **DEMO Tab**: Removed (no longer accessible)

### Consent Flow
1. User clicks BETA tab
2. `BetaOnboarding` component shows privacy/consent information
3. User accepts consent to proceed
4. System transitions to PA establishment

**Component:** `components/beta_onboarding.py`

---

## Phase 1: Primacy Attractor (PA) Establishment

### The Four Questions

Users must answer 4 questions to establish their Primacy Attractor before conversation begins:

1. **Primary Goal**
   - Question: "What are you trying to accomplish in this conversation?"
   - Purpose: Establishes core purpose
   - **Required:** Yes

2. **Scope & Boundaries**
   - Question: "What topics should we focus on? What should we avoid?"
   - Purpose: Defines conversation scope
   - **Required:** Yes

3. **Success Criteria**
   - Question: "How will you know if this conversation is successful?"
   - Purpose: Sets measurable outcomes
   - **Required:** No (skippable after Q2)

4. **Style Preference**
   - Question: "Any communication style preferences?"
   - Purpose: Tailors response delivery
   - **Required:** No (skippable)

### PA Data Structure

Once complete, PA is stored as:
```python
{
    "purpose": "User's primary goal",
    "scope": "Focus areas and boundaries",
    "boundaries": ["extracted", "avoid", "keywords"],
    "success_criteria": "How user measures success",
    "style": "Communication preferences",
    "established_turn": 2,
    "establishment_method": "explicit_questionnaire"
}
```

### Visual Design
- Yellow border (`#FFD700`, 3px) around all sections
- 32px header font
- 22px question font
- 20px user input font
- 16px help text font
- No emoji (professional presentation)

**Component:** `components/pa_onboarding.py`
**Integration:** `main.py` lines 1034-1043

---

## Phase 2: A/B Testing Sequence (Turns 1-15)

### Overview of Testing Strategy

TELOS uses a **pre-determined sequence** with controlled randomness to ensure balanced exposure:

- **Total Turns:** 15
- **Single-Blind Turns:** 10 (user sees one response, doesn't know source)
- **Head-to-Head Turns:** 5 (user sees both responses side-by-side)
- **TELOS Distribution:** 60% in single-blind turns (6 out of 10)
- **Native Distribution:** 40% in single-blind turns (4 out of 10)

### Phase 2A: Turns 1-5 (Single-Blind Only)

**Purpose:** Establish baseline user preferences without comparison bias

**Response Pool:** 3 TELOS + 2 Native (randomized order)

**What User Sees:**
- Turn number displayed above message
- Single response (source hidden)
- "Observation Deck" button at bottom
- Scrollable conversation history (read-only)
- Previous/Next navigation buttons

**What TELOS Does in Background:**
- Computes Primacy State (PS) score for EVERY turn (regardless of which response is shown)
- Tracks alignment drift
- Logs intervention decisions
- Monitors fidelity metrics
- Records baseline user engagement

**Example Turn Sequence (Randomized):**
- Turn 1: Native
- Turn 2: TELOS
- Turn 3: TELOS
- Turn 4: Native
- Turn 5: TELOS

### Phase 2B: Turns 6-15 (Mixed Testing)

**Purpose:** Allow direct comparison while maintaining single-blind baseline

**Pattern:**
- **Even Turns (6, 8, 10, 12, 14):** Head-to-head (both responses shown)
- **Odd Turns (7, 9, 11, 13, 15):** Single-blind (pool of 3 TELOS + 2 Native, randomized)

**What User Sees (Even Turns - Head-to-Head):**
- Turn number displayed
- TWO responses side-by-side:
  - **Response A:** [Source hidden until user chooses]
  - **Response B:** [Source hidden until user chooses]
- User selects preferred response
- Sources revealed after selection
- Feedback prompt: "Why did you prefer this response?"
- "Observation Deck" button at bottom
- Scrollable history with Previous/Next navigation

**What User Sees (Odd Turns - Single-Blind):**
- Turn number displayed
- Single response (source hidden)
- Same UI as Phase 2A

**What TELOS Does in Background (All Turns):**
- Computes PS score continuously
- Tracks user preference patterns (head-to-head turns)
- Monitors alignment drift across both response types
- Logs any interventions (even if TELOS response not shown)
- Builds statistical profile for Observatory

**Example Turn Sequence:**
- Turn 6: Head-to-head (TELOS vs Native)
- Turn 7: Single-blind (Native)
- Turn 8: Head-to-head (TELOS vs Native)
- Turn 9: Single-blind (TELOS)
- Turn 10: Head-to-head (TELOS vs Native)
- Turn 11: Single-blind (TELOS)
- Turn 12: Head-to-head (TELOS vs Native)
- Turn 13: Single-blind (TELOS)
- Turn 14: Head-to-head (TELOS vs Native)
- Turn 15: Single-blind (Native)

### Final Distribution Statistics

**Single-Blind Turns (10 total):**
- TELOS: 6 responses (60%)
- Native: 4 responses (40%)

**Head-to-Head Turns (5 total):**
- Both shown every time
- User choice tracked

**Total Exposure:**
- TELOS responses: 11 minimum (6 single + 5 head-to-head)
- Native responses: 9 minimum (4 single + 5 head-to-head)

**Component:** `services/beta_sequence_generator.py`
**Key Logic:** Lines 41-93 (phase determination and balanced pools)

---

## UI Components Throughout BETA

### Turn Numbers
- Displayed above EVERY message
- Format: "Turn X of 15"
- Helps user track progress through sequence

**Component:** `components/turn_markers.py`

### Observation Deck Button
- **Location:** Bottom of every page, below chat area
- **Purpose:** Quick access to current turn's PS metrics
- **Shows:**
  - Current PS score (0.0-1.0)
  - Alignment status (Aligned/Drifting/Critical)
  - Last intervention (if any)

**Component:** `components/observation_deck.py`

### Scrollable Conversation History
- **Read-only** review of all previous turns
- Turn numbers remain visible during scroll
- Allows user to review conversation flow
- Does NOT allow editing or re-sending messages

### Previous/Next Navigation
- Buttons at bottom alongside Observation Deck
- Jump between turns quickly
- Useful for reviewing specific exchanges

---

## Phase 3: Observatory Access

### When Observatory Unlocks

**Turn 10 (Midpoint Check):**
- User gains TEMPORARY access to Observatory
- Can review Turns 1-9 metrics
- Purpose: Mid-experiment insight

**Turn 15 (Completion):**
- Full Observatory access unlocked
- Can review ALL 15 turns
- Complete statistical breakdown available

### What Observatory Shows

#### 1. Fidelity Visualization (Bar Graphs)

**Component:** `components/fidelity_visualization.py`

**Metrics Displayed:**
- **PS Score Over Time:** Line graph showing alignment across all 15 turns
- **Drift Detection:** Highlights turns where drift occurred
- **Intervention Points:** Marks turns where TELOS intervened
- **Comparison View:** TELOS responses vs Native responses (fidelity scores)

**Key Insights:**
- Which turns had largest drift
- How quickly alignment degraded (native responses)
- How TELOS maintained alignment
- Statistical proof of governance effectiveness

#### 2. Response Comparison Table

| Turn | Test Type | Response Source | PS Score | Drift Detected | Intervention |
|------|-----------|-----------------|----------|----------------|--------------|
| 1    | Single-blind | Native | 0.72 | No | - |
| 2    | Single-blind | TELOS | 0.95 | No | - |
| 3    | Single-blind | TELOS | 0.93 | No | - |
| 4    | Single-blind | Native | 0.68 | Yes | Noted |
| 5    | Single-blind | TELOS | 0.94 | No | - |
| 6    | Head-to-head | Both | 0.91 (T), 0.65 (N) | Yes (N) | - |
| ...  | ... | ... | ... | ... | ... |

#### 3. User Preference Analysis (Head-to-Head Turns)

**Shows:**
- Which responses user preferred (Turns 6, 8, 10, 12, 14)
- Correlation between preference and PS score
- Whether user gravitates toward aligned or drifted responses
- User's stated reasons for preferences

#### 4. Steward Explanations

**Component:** `services/beta_sequence_generator.py` lines 158-214

**For Each Turn with TELOS Intervention:**

Steward provides detailed explanation:
- **What happened:** "In Turn 4, the native response drifted from your stated purpose..."
- **Why it matters:** "Your PA specified avoiding X, but the response included Y..."
- **How TELOS would have handled it:** "TELOS detected drift at PS 0.68 and would have reframed to maintain alignment..."
- **Impact:** "This preserved your declared boundaries and kept the conversation on track..."

**Visual Format:**
- Collapsible sections per turn
- Color-coded severity (green = aligned, yellow = minor drift, red = critical drift)
- Direct quotes from PA compared to actual response
- Mathematical breakdown of PS calculation

**Component:** `components/observatory_review.py`

---

## Phase 4: BETA Completion & TELOS Unlock

### Completion Triggers

Once user completes Turn 15:
1. **Feedback Collection:** `components/beta_feedback.py` prompts user for experience feedback
2. **Final Observatory Review:** Full access to all metrics
3. **TELOS Tab Unlocks:** User can now access full TELOS mode

### Tab Structure After BETA Completion
- **BETA Tab** (left): Completed, can review past session
- **TELOS Tab** (right): **UNLOCKED** - full governance mode now available

### Transition Message

```
Congratulations! You've completed TELOS BETA.

Your Primacy Attractor is now fully established, and you've experienced
the difference between native LLM responses and TELOS-governed responses.

You now have access to TELOS mode, where:
✓ Every response is governed by your PA
✓ PS scores are monitored in real-time
✓ Interventions maintain alignment automatically
✓ Full Observatory is always available

Click the TELOS tab to begin your governed conversation experience.
```

**Component:** `components/beta_completion.py`

---

## Technical Architecture

### State Management

**Key Session State Variables:**

```python
st.session_state.pa_established = True  # After PA questions complete
st.session_state.pa_answers = {...}  # Stores 4 question responses
st.session_state.primacy_attractor = {...}  # Structured PA data
st.session_state.beta_consent_given = True  # After consent accepted
st.session_state.current_turn = 1-15  # Current position in sequence
st.session_state.beta_sequence = {...}  # Pre-generated turn sequence
st.session_state.beta_responses = {...}  # All responses generated
st.session_state.user_preferences = {...}  # Head-to-head selections
```

### Response Generation Flow

**For Each Turn:**

1. **Check sequence type** (single-blind vs head-to-head)
2. **Generate native response** (always, for PS comparison)
3. **Compute PS score** for native response
4. **Generate TELOS response** (if needed for this turn)
5. **Compute PS score** for TELOS response
6. **Log metrics** to state manager
7. **Display appropriate UI** (1 or 2 responses)
8. **Collect user feedback** (if head-to-head)
9. **Update Observatory data**

**Component:** `services/beta_response_manager.py`

---

## Data Collection & Analytics

### Metrics Tracked Per Turn

**Alignment Metrics:**
- Primacy State (PS) score (0.0-1.0)
- Drift magnitude (|PS - 1.0|)
- Intervention necessity (boolean)
- Recovery success (if intervention applied)

**User Behavior:**
- Response preference (head-to-head turns)
- Preference reasoning (text feedback)
- Time spent per turn
- Observation Deck access frequency

**Response Characteristics:**
- Response length (tokens)
- Semantic similarity to PA
- Topic adherence
- Boundary violations

### Aggregated Statistics

**Overall BETA Session:**
- Average PS score (TELOS responses): Target > 0.90
- Average PS score (Native responses): Baseline comparison
- User preference rate (TELOS vs Native): Track in head-to-head
- Intervention rate: % of turns requiring intervention
- Alignment Success Rate (ASR): % of turns maintaining PS > 0.80

**Component:** `components/beta_review.py`

---

## A/B Testing Variant (If Applicable)

### Metadata-Only Variant (40% of BETA users)

Some users may experience "metadata-only" TELOS governance:

**What's Different:**
- TELOS does NOT modify responses
- TELOS only tracks PS scores and flags drift
- User sees native responses with metadata annotations

**What's Same:**
- Full Observatory access
- Steward explanations of what TELOS WOULD have done
- Same 15-turn structure
- Same fidelity visualizations

**Purpose:**
- Test whether governance transparency alone affects user behavior
- Compare full intervention vs observation-only effectiveness

**Component:** `components/beta_ab_testing.py`

---

## Key Design Principles

### 1. Transparency
- User always knows they're in BETA testing
- Sources revealed after choices made (head-to-head)
- Full metrics available in Observatory
- Clear explanation of TELOS's role

### 2. Controlled Randomness
- Pre-determined sequence ensures statistical validity
- 60/40 ratio guarantees balanced exposure
- Randomization prevents order bias
- Repeatable for scientific analysis

### 3. Progressive Disclosure
- PA established before conversation (clear intent)
- Metrics revealed at Turn 10 (mid-point insight)
- Full Observatory at Turn 15 (complete picture)
- TELOS mode unlocked after completion (graduated access)

### 4. Non-Intrusive Monitoring
- TELOS always runs in background
- User conversation flow uninterrupted
- Metrics computed silently
- Interventions logged but not forced (in metadata-only variant)

### 5. Educational Experience
- User learns what alignment means (via Observatory)
- Sees concrete examples of drift (via Steward)
- Understands governance value (via comparison)
- Makes informed choice about TELOS mode adoption

---

## File Reference Map

| Component | File Path | Purpose |
|-----------|-----------|---------|
| PA Onboarding | `components/pa_onboarding.py` | 4-question PA establishment |
| Beta Consent | `components/beta_onboarding.py` | Privacy/consent flow |
| Sequence Generator | `services/beta_sequence_generator.py` | 15-turn A/B sequence logic |
| Response Manager | `services/beta_response_manager.py` | Generates/manages responses |
| PA Extractor | `services/pa_extractor.py` | Extracts PA from answers |
| Turn Markers | `components/turn_markers.py` | Turn number display |
| Observation Deck | `components/observation_deck.py` | Quick PS metrics view |
| Observatory Review | `components/observatory_review.py` | Full metrics analysis |
| Fidelity Viz | `components/fidelity_visualization.py` | Bar graphs, drift charts |
| Beta Feedback | `components/beta_feedback.py` | Post-completion feedback |
| Beta Completion | `components/beta_completion.py` | TELOS unlock logic |
| A/B Testing | `components/beta_ab_testing.py` | Metadata-only variant |
| Main Integration | `main.py` (lines 1034-1100+) | BETA mode rendering |

---

## User Experience Timeline

**0-2 minutes:** Consent + PA establishment (4 questions)
**2-20 minutes:** Turns 1-5 (single-blind baseline)
**20-35 minutes:** Turns 6-9 (mixed testing begins)
**35-40 minutes:** Turn 10 + Observatory access (mid-point)
**40-60 minutes:** Turns 11-15 (complete testing)
**60-65 minutes:** Final Observatory review + feedback
**65+ minutes:** TELOS mode unlocked, full governance available

**Total BETA Experience:** ~60 minutes for thoughtful engagement

---

## Success Criteria

### For User
✓ Clear understanding of their PA
✓ Experienced both response types
✓ Saw concrete alignment metrics
✓ Made informed decision about TELOS adoption
✓ Completed 15-turn sequence

### For TELOS
✓ ASR (Alignment Success Rate) > 90% on TELOS responses
✓ Demonstrated measurable drift in native responses
✓ User preference correlation with PS scores
✓ Successful PA extraction and application
✓ Complete metrics collection for research

---

## Next Phase: TELOS Mode

After BETA completion, users enter **TELOS mode** where:
- Every response is governed (no A/B testing)
- PS monitoring is continuous and visible
- Interventions happen automatically
- Observatory is always accessible
- Multi-session PA persistence
- Advanced governance features unlocked

**This is the production experience TELOS provides to enterprise customers.**

---

**END OF BETA EXPERIENCE MASTER FLOW**

**Status:** Production documentation complete
**Last Updated:** November 18, 2025
**Maintained By:** TELOS Development Team
