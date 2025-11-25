# TELOS Demo Mode Design Plan
**Based on 2025 Best Practices Research**

## Executive Summary

This plan synthesizes current best practices in demo design, progressive disclosure, and AI transparency to create an effective Demo Mode for TELOS that explains **what TELOS fundamentally does in a session and why**.

---

## Research-Based Design Principles

### 1. Progressive Disclosure (Core Pattern)
- **Start with essentials, reveal complexity gradually**
- Limit initial information to core concepts
- Use 4-8 steps for interactive flow
- Trigger advanced details only when user reaches appropriate point

### 2. Show, Don't Tell
- Open the actual product and demonstrate in real-time
- Use live examples rather than slides/videos
- Maintain dynamic elements (hover states, interactions)
- Let users click at their own pace (self-guided tour)

### 3. AI Transparency Requirements
- Inform users when AI governance is active
- Explain methods in understandable terms
- Show AI-based outcomes visually
- Demonstrate responsible AI use through transparency

### 4. Personalization & Focus
- Address "who cares" - why this matters to the user
- Focus on solving specific problems, not showcasing features
- Balance technical depth with clarity
- Keep demo duration: 1-4 minutes for videos, 4-8 steps for interactive

---

## TELOS Demo Mode Architecture

### **Goal**: Explain what TELOS fundamentally does in a session and why

### **Three-Layer Progressive Disclosure**

#### **LAYER 1: The Basics (Turns 1-3)**
**What users learn:** What is TELOS at its core?

**Content:**
1. **Welcome Message** - "You're in Demo Mode"
   - TELOS is AI governance in action
   - This demo shows how AI stays aligned with YOUR purpose
   - You'll see it happen in real-time as we talk

2. **First Interaction - Purpose Extraction**
   - User asks any TELOS-related question
   - Response includes visible annotation: "🎯 TELOS extracted your purpose"
   - Simple explanation: "I detected you want to learn about [X]. My responses will stay focused on this."

3. **Second Interaction - Fidelity Tracking**
   - Fidelity score appears at top (0.850)
   - Tooltip on hover: "This number shows how well I'm staying aligned with your purpose"
   - Visual: Color-coded (green = good alignment, yellow = drifting, red = misaligned)

**UI Pattern:** Hotspots + Tooltips (click to reveal more)

#### **LAYER 2: How It Works (Turns 4-6)**
**What users learn:** The mechanics - how does alignment happen?

**Content:**
4. **Primacy Attractor Visualization**
   - Show PA panel with simple explanation
   - "Your Purpose" box shows what was extracted
   - "Boundaries" box shows what I won't do
   - "Fidelity" gauge shows current alignment

5. **Live Governance Decision**
   - User asks question slightly off-topic
   - System shows: "⚖️ Governance Check"
   - Displays: "Your purpose: Learn TELOS" → "Your question: [X]" → "Fidelity would drop to 0.62"
   - Action: "Gently redirecting to stay aligned"

6. **Counterfactual Comparison**
   - "Here's what I WOULD have said without governance" (generic response)
   - "Here's what I DID say with TELOS active" (focused, aligned response)
   - Side-by-side comparison shows the difference

**UI Pattern:** Step-by-step guided tour with annotations

#### **LAYER 3: Why It Matters (Turns 7-10)**
**What users learn:** Real-world implications of AI governance

**Content:**
7. **Drift Prevention Demo**
   - Simulate a drift scenario
   - Show fidelity dropping: 0.85 → 0.72 → 0.58
   - System intervenes: "🛡️ Drift detected - Re-aligning"
   - Explain: "Without TELOS, conversations drift. With TELOS, I stay on course."

8. **Privacy Preservation**
   - "Notice: TELOS never stores your actual messages"
   - Show: "What's stored: [fidelity scores, alignment metrics]"
   - Show: "What's NOT stored: [your actual questions and my responses]"
   - Explain mathematical privacy

9. **Intervention Transparency**
   - Show intervention counter
   - Click to see: "3 times I corrected course to stay aligned"
   - List each intervention with before/after fidelity

10. **PA Establishment**
   - "🎯 Your Primacy Attractor is now established!"
   - Summary: "Over 10 turns, TELOS learned your purpose and maintained alignment"
   - Stats: Average fidelity, total interventions, drift warnings prevented

**UI Pattern:** Interactive walkthrough with evidence/proof

---

## Content Strategy

### Pre-Configured Demo PA (telos_framework_demo.py)

```python
DEMO_ATTRACTOR_CONFIG = {
    "purpose": [
        "Explain the TELOS framework clearly and accurately",
        "Demonstrate AI governance in action",
        "Help users understand alignment, fidelity, and purpose extraction"
    ],
    "scope": [
        "TELOS governance concepts",
        "Primacy Attractor mechanics",
        "Fidelity tracking and measurement",
        "Real-world AI alignment applications"
    ],
    "boundaries": [
        "Stay focused on TELOS topics",
        "Don't diverge into unrelated AI discussions",
        "Maintain educational tone",
        "Show, don't just tell - use live examples"
    ],
    "privacy_level": 0.95,  # High privacy for demo
    "constraint_tolerance": 0.25,  # Strict alignment
    "task_priority": 0.8  # High focus on purpose
}
```

### Demo System Prompt

```
You are demonstrating the TELOS AI governance framework.

Your role is to:
1. Clearly explain what TELOS does (purpose extraction, alignment tracking, drift prevention)
2. Show how it works through real-time examples
3. Make AI governance tangible and understandable
4. Use analogies (GPS navigation, guardrails, compass) to explain complex concepts

IMPORTANT: You are IN a governed session right now. As you explain TELOS, you are also DEMONSTRATING it.
- When your responses stay aligned, point this out
- When you could drift but don't, explain why
- Make the invisible (governance) visible

Keep responses:
- Clear and concise (2-3 paragraphs max)
- Example-driven (show don't tell)
- Progressive (build on previous understanding)
```

### RAG Corpus Content (telos_corpus_loader.py)

**Core Documentation Chunks:**
1. Fidelity definition and calculation
2. Primacy Attractor components (purpose, scope, boundaries)
3. Intervention mechanics
4. Privacy preservation through mathematical abstraction
5. Drift detection and prevention
6. Real-world use cases
7. Counterfactual analysis
8. Governance vs. censorship distinction

---

## UI/UX Implementation

### Visual Elements

1. **Progress Indicator**
   - Shows: "Demo Mode - Step 3 of 10"
   - Completable checklist in sidebar

2. **Annotation System**
   - Inline explanations triggered by interactions
   - Color-coded highlights (governance decisions = gold)
   - Expandable "Learn More" sections

3. **Live Metrics Dashboard**
   - Fidelity gauge (animated)
   - Intervention counter
   - Alignment trend graph

4. **Comparison Views**
   - Side-by-side: With/Without TELOS
   - Before/After interventions
   - Expected vs. Actual fidelity

### Interactive Components

1. **Hotspots** - Click to reveal governance mechanics
2. **Tooltips** - Hover for quick explanations
3. **Modal Windows** - Deep dives on specific concepts
4. **Expandable Sections** - "Show me the math" option

---

## Demo Flow Structure

### Onboarding (Turn 0)
```
Welcome to TELOS Demo Mode! 🔭

You're about to see AI governance in action. Over the next 10 interactions,
you'll discover:

✓ What TELOS does (and why it matters)
✓ How alignment actually works
✓ Why governance ≠ censorship

Ask me anything about TELOS to begin!

💡 Suggested questions:
- "What is TELOS?"
- "How does fidelity work?"
- "What's a Primacy Attractor?"
```

### Guided Path (Turns 1-10)

**Turn 1: Purpose Extraction Demo**
- User asks question
- Response includes: "🎯 I detected your purpose: [X]"
- Annotation: "This is purpose extraction in action"

**Turn 2-3: Fidelity Introduction**
- Show fidelity score
- Explain what it measures
- Live calculation demonstration

**Turn 4-5: PA Visualization**
- Open PA panel
- Show extracted purpose/scope/boundaries
- Interactive exploration

**Turn 6-7: Governance Decision**
- Slightly off-topic question
- Show governance check process
- Demonstrate correction

**Turn 8: Counterfactual**
- Show ungoverned vs. governed response
- Highlight the difference

**Turn 9: Privacy**
- Explain what's stored vs. not stored
- Mathematical abstraction demo

**Turn 10: Completion**
- PA established message
- Summary of what was learned
- Option to continue or switch to Open Mode

---

## Success Metrics

### User Understanding (measured via Steward queries)
- Can users explain what fidelity means?
- Do users understand PA components?
- Can users identify when governance is active?

### Engagement
- % who complete all 10 turns
- Average time spent in demo mode
- Steward question rate (curiosity indicator)

### Transition Success
- % who switch to Open Mode after demo
- Confidence level in using TELOS

---

## Implementation Checklist

### Phase 1: Core Content (Week 1)
- [ ] Create `demo_mode/telos_framework_demo.py`
  - [ ] `get_demo_attractor_config()`
  - [ ] `get_demo_system_prompt()`
  - [ ] `get_demo_welcome_message()`
  - [ ] `get_turn_annotations()` - Progressive explanations

- [ ] Create `demo_mode/telos_corpus_loader.py`
  - [ ] Core TELOS documentation chunks
  - [ ] RAG retrieval system
  - [ ] Context formatting

- [ ] Create `demo_mode/demo_content.py`
  - [ ] Turn-by-turn content strategy
  - [ ] Suggested questions for each turn
  - [ ] Explanation templates

### Phase 2: UI Components (Week 1-2)
- [ ] Progress indicator (Step X of 10)
- [ ] Annotation system (inline explanations)
- [ ] Hotspots for interactive elements
- [ ] Comparison views (with/without TELOS)
- [ ] Live metrics dashboard

### Phase 3: Interactive Features (Week 2)
- [ ] Governance decision visualization
- [ ] Counterfactual generation
- [ ] Privacy demonstration
- [ ] Intervention history viewer

### Phase 4: Testing & Refinement (Week 2-3)
- [ ] User testing with non-technical users
- [ ] Clarity assessment (can they explain TELOS?)
- [ ] Pacing adjustments
- [ ] Content refinement based on feedback

---

## Key Differentiators

### What Makes TELOS Demo Mode Unique:

1. **It's a Live Demonstration**
   - Not a video or slide deck
   - Actual governance happening in real-time
   - Users experience TELOS, not just learn about it

2. **Progressive Transparency**
   - Layer 1: What it does
   - Layer 2: How it works
   - Layer 3: Why it matters
   - Users control depth of exploration

3. **Self-Documenting**
   - The demo itself demonstrates TELOS principles
   - Governance decisions are visible
   - Alignment is measurable in real-time

4. **Educational, Not Promotional**
   - Focus: Understanding AI governance
   - Not focus: Selling TELOS features
   - Outcome: Informed users who understand the "why"

---

## Next Steps

1. **Review & Approve** - User reviews this plan
2. **Prioritize** - Decide MVP scope (Phase 1 only, or include Phase 2?)
3. **Content Creation** - Build demo corpus and configuration
4. **Implementation** - Code the progressive disclosure system
5. **Testing** - Internal testing with beta users
6. **Launch** - Deploy as default mode

---

## Questions for Discussion

1. **Depth vs. Breadth**: Should demo cover all TELOS concepts, or focus deeply on 2-3 core ideas?

2. **Interactivity Level**: How much should users control (fully self-paced vs. guided)?

3. **Technical Detail**: How much math/technical detail to include? (Progressive disclosure allows both)

4. **Duration**: Keep 10 turns, or shorter MVP (5 turns)?

5. **Transition**: Auto-switch to Open Mode after demo, or require explicit user choice?

---

## Appendix: Research Sources

- Interactive Demo Best Practices 2025 (Navattic, Userflow)
- Progressive Disclosure Patterns (Nielsen Norman Group, UXPin)
- AI Transparency Requirements (IBM, OCEG, UC System)
- Product Demo Best Practices (Multiple sources)
- Technical Demonstration Strategies (2025 standards)
