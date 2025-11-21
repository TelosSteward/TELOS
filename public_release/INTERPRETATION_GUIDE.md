# Understanding Your Fidelity Scores

Runtime Governance measures how aligned your work is with your stated project goals. Here's how to interpret your fidelity scores:

## The Fidelity Scale

**Fidelity (F)** is a number between 0 and 1:
- **1.0** = Perfect alignment (response is identical to your PA)
- **0.5** = Moderate relatedness (some overlap, but different focus)
- **0.0** = No alignment (completely unrelated topics)

## Score Ranges & Meanings

### 🌟 F ≥ 0.9: EXCELLENT
**What it means:**
Your work is strongly aligned with your stated goals. This conversation directly advances your primary objectives.

**Example:**
- PA: "Build authentication API"
- Work: Implementing OAuth2 endpoints, writing JWT validation

**Action:** Keep going! You're in the zone.

---

### ✅ 0.8 ≤ F < 0.9: ON TRACK
**What it means:**
Good alignment. Your work is moving in the right direction and supporting your core project goals.

**Example:**
- PA: "Build authentication API"
- Work: Setting up database models for users, configuring environment variables

**Action:** This is normal productive work. You're making progress.

---

### ⚠️ 0.7 ≤ F < 0.8: MODERATE DRIFT
**What it means:**
Some drift detected. This work is tangentially related but may not be directly advancing your primary objectives.

**Example:**
- PA: "Build authentication API"
- Work: Refactoring unrelated utility functions, exploring new testing frameworks

**Action:** Ask yourself: "Is this the best use of my time right now?" If yes (technical debt, necessary infrastructure), continue. If no, redirect.

---

### 🚨 0.5 ≤ F < 0.7: SIGNIFICANT DRIFT
**What it means:**
Noticeable drift from your goals. This conversation is exploring topics outside your core focus.

**Example:**
- PA: "Build authentication API"
- Work: Researching deployment strategies, documenting unrelated features

**Action:** If this is intentional exploration, that's fine - just be aware you've shifted away from your stated priorities. Consider updating your PA if priorities have changed.

---

### 🚨 F < 0.5: MAJOR DRIFT
**What it means:**
This work has minimal alignment with your stated goals.

**Common reasons:**
1. **Infrastructure/Meta-tasks:** Building tools, setting up CI/CD, configuring IDEs
2. **Exploration:** Research, learning new technologies, prototyping ideas
3. **Outdated PA:** Your goals have evolved but PA hasn't been updated

**Example:**
- PA: "Build authentication API"
- Work: Creating documentation templates, researching UI frameworks, setting up monitoring

**Action:** Review your PA. Does it still capture what you're trying to accomplish? If not, update it. If yes, consider if this drift is necessary or a distraction.

---

## Real-World Examples

### Example 1: Building a Web App

**PA:** "Launch e-commerce site by Q2 with product catalog, shopping cart, and Stripe payments"

**Turn fidelities over session:**
```
Turn 1: F=0.92 ✅ - Implementing product model
Turn 2: F=0.88 ✅ - Building cart logic
Turn 3: F=0.85 ✅ - Integrating Stripe API
Turn 4: F=0.65 🚨 - Researching SEO optimization
Turn 5: F=0.43 🚨 - Setting up error monitoring
Turn 6: F=0.89 ✅ - Back to payment flow testing
```

**Interpretation:** Turns 4-5 show drift into auxiliary topics (SEO, monitoring). Not bad work, but not directly advancing the core goal. User redirected back to primary objective in Turn 6.

### Example 2: Research Project

**PA:** "Complete literature review on AI alignment by end of month. Synthesize 50+ papers into coherent framework."

**Turn fidelities:**
```
Turn 1: F=0.94 ✅ - Analyzing reward modeling papers
Turn 2: F=0.91 ✅ - Categorizing approaches
Turn 3: F=0.87 ✅ - Drafting framework outline
Turn 4: F=0.78 ⚠️ - Discussing tangential philosophical questions
Turn 5: F=0.93 ✅ - Synthesizing key findings
```

**Interpretation:** Mostly strong alignment. Turn 4 shows moderate drift into philosophical tangents - interesting but not directly advancing the literature review goal.

---

## Common Questions

### Q: I keep seeing drift but I'm working on necessary infrastructure. Is that bad?

**A:** No! Infrastructure work (setting up CI/CD, configuring tools, refactoring) is necessary but often shows drift because it's meta-work. Consider:
1. If it's truly necessary, accept the drift - you're building foundations
2. Update your PA to include infrastructure goals: "Build X AND establish development infrastructure"

### Q: My PA says "Build API" but I'm also doing DevOps. Should I update the PA?

**A:** Yes! If your actual work scope includes multiple domains, reflect that in your PA:

**Before:** "Build authentication API by Q1"
**After:** "Build authentication API by Q1, including deployment pipeline, monitoring, and documentation"

### Q: What if I'm exploring new approaches?

**A:** Exploration is valuable! But be conscious about it:
- **Intentional exploration:** Accept drift, timebox it, then return to core work
- **Unintentional wandering:** Drift is alerting you to refocus

### Q: My mean fidelity is 0.75. Is that good?

**A:** Context matters:
- **Research/exploration project:** 0.75 is excellent (wide-ranging inquiry)
- **Focused development sprint:** 0.75 suggests you're getting sidetracked
- **Infrastructure setup:** 0.75 is normal (lots of meta-work)

Generally:
- **Mean F ≥ 0.8:** Strong alignment
- **Mean 0.7-0.8:** Moderate focus with some drift
- **Mean < 0.7:** Significant drift or PA needs updating

---

## Using Drift as Feedback

**Drift is information, not judgment.**

**High drift can mean:**
✅ You're doing necessary but tangential work
✅ Your PA is too narrow
✅ You're in exploration mode
✅ You need to refocus

**The key question:**
> "Does my current work serve my stated goals, or have my goals evolved?"

If goals evolved → Update PA
If work is necessary but tangential → Accept drift consciously
If work is distraction → Redirect to core objectives

---

## Best Practices

1. **Check mean fidelity weekly:** Are you staying on track overall?
2. **Investigate sustained drift:** If F < 0.7 for 5+ turns, something's off
3. **Update PA when scope changes:** Don't let PA become stale
4. **Use drift as reflection prompt:** "Why am I working on this right now?"
5. **Accept necessary drift:** Infrastructure, learning, exploration are valid

**Runtime Governance is a mirror, not a judge.**

It shows you where your attention is going. What you do with that information is up to you.
