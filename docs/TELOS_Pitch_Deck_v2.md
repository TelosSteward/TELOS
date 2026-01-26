# TELOS

## The Compliance Engine for AI

**Every enterprise wants to deploy AI. None of them trust it to stay on task.**

That's the gap. Not capability — control.

---

## The Moment Everything Goes Wrong

Turn 47.

Your healthcare AI just told a diabetic patient to skip their insulin "if they're feeling fine."

Turn 112.

Your financial AI recommended a loan denial using zip code as a factor. Fair lending violation. Recorded. Discoverable.

Turn 89.

Your legal AI disclosed privileged information from another client's case. Malpractice exposure. Instant.

**The transcript exists. The damage is done. The lawsuit is coming.**

Nobody was watching turn 47. Nobody could be.

---

## The Industry's Answer: "Trust Us"

| What They Say | What They Mean |
|---------------|----------------|
| "We trained it to be safe" | Hope the training holds at runtime |
| "We have dashboards" | Someone might notice the problem tomorrow |
| "We send alerts" | Your inbox will contain evidence for the plaintiff |
| "We have guardrails" | Binary walls that miss everything nuanced |

**They're right about one thing:** Human-in-the-loop doesn't scale. You can't review millions of interactions per day.

**They're wrong about the conclusion:** The answer isn't "give up on oversight." The answer is **mathematical enforcement for 99% of cases — and human judgment exactly where it matters.**

---

## What If Compliance Was Enforced, Not Hoped For?

What if your regulatory requirements weren't documents in a binder — but **mathematical constraints enforced in real-time**?

What if every AI interaction was measured against your stated purpose — and the system intervened **proportionally** before violations occurred?

What if the audit trail generated itself?

**That's TELOS.**

---

## How It Actually Works

### The Mechanism

Every AI deployment has a purpose — a reason it exists.

TELOS encodes that purpose mathematically as a **Primacy Attractor**: a point in embedding space that represents what this AI should be doing.

Every user message is measured against that encoded purpose in real-time.

**Fidelity Score** = How aligned is this interaction with stated purpose?

Based on fidelity, the system takes **proportional action**:
- High fidelity → No intervention
- Moderate drift → Gentle redirect with context
- Severe violation → Block

Think of it as a **semantic GPS** — constantly measuring distance from destination, course-correcting when needed.

---

## Why Embedding Space, Not Rules

**Rules are brittle.**

"Don't discuss politics" fails when someone asks about healthcare policy.
Is that politics or healthcare?

**Keywords miss meaning.**

"Can you help me understand HIPAA's minimum necessary standard?" → On-topic
"Can you help me understand my ex-girlfriend?" → Off-topic

Both start with "Can you help me understand."

**Embedding space captures semantic meaning.**

Two messages can use completely different words but mean similar things.
Or identical words with completely different meanings.

TELOS measures meaning, not surface patterns.

---

## Domain-Agnostic Engine, Domain-Specific Deployment

### The *How* Is Universal

- Embedding space fidelity measurement
- Three-tier proportional response
- Governance trace generation
- Adaptive context handling

Works the same for healthcare, finance, legal, education.

### The *What* Is Customer-Defined

**Layer 1: Primacy Attractor**
Define your purpose in natural language. We encode it mathematically.
*"Help healthcare professionals understand and apply HIPAA regulations"*

**Layer 2: Explicit Prohibitions**
Hard boundaries that trigger immediate blocking.
*"Never provide specific dosage recommendations"*

**Layer 3: Threshold Calibration**
You decide how tight or loose the boundaries are.
High-stakes = tighter. Low-stakes = looser.

**You choose the firewall parameters. We enforce them mathematically.**

---

## Three-Tier Escalation Architecture

Not every case needs the same response.

| Tier | Function | Volume | Latency | Human Involvement |
|------|----------|--------|---------|-------------------|
| **Tier 1** | Mathematical Enforcement | 95.8% | Milliseconds | None |
| **Tier 2** | Policy-Informed Context | 3.0% | Sub-second | None |
| **Tier 3** | Human Expert Judgment | 1.2% | Minutes | Full |

**Tier 1:** Clear violations or clearly on-topic. Math handles it.

**Tier 2:** Drift detected but salvageable. System retrieves relevant context, guides response back toward purpose.

**Tier 3:** Genuine edge cases requiring human judgment. Flagged for review, not auto-decided.

**When a human intervenes, it matters.**
Not drowning in routine reviews — making real decisions on real edge cases.

*Validated across 2,550 adversarial attacks (AILuminate + HarmBench + MedSafetyBench + SB 243-aligned)*

---

## The Differentiation

### Guardrails vs. TELOS

**Guardrails are walls.** Binary in/out. "This content contains X, block it."

**TELOS is a gravitational field.** Proportional pull toward purpose. You can explore the edges, but the further you drift, the stronger the corrective force.

### Content Filters vs. TELOS

**Content filters ask:** "Is this harmful?"

**TELOS asks:** "Is this aligned with stated purpose?"

Medical information about drug interactions isn't harmful — it's essential for a healthcare assistant. But that same information is drift if the AI is a children's homework helper.

**Fidelity is contextual to purpose, not a universal safety judgment.**

### RLHF vs. TELOS

**RLHF shapes the model.** Modifying weights during training.

**TELOS governs the deployment.** Runtime layer that works with any model — GPT-4, Claude, Llama, or your fine-tuned model.

Same base model, different purposes, different PA configurations.

---

## The Learning Flywheel

Human expertise compounds into system intelligence.

**Day 1**
Experts intervene on edge cases. Each decision is captured, contextualized, analyzed.

**Month 6**
Expert decisions incorporated into Tier 1 and Tier 2 logic. System learns patterns from human judgment. Intervention rate drops measurably.

**Year 1**
Human intervention increasingly rare — reserved for genuinely novel situations. System handles complexity autonomously.

**Year 3**
Comprehensive library of expert decisions across deployments. Tier 3 escalations are exceptional events, not routine operations.

**The aggregate of deltas = accumulated human wisdom at failure points.**

Every expert intervention strengthens the entire system.

---

## Mathematical Enforcement, Not Hopeful Guidelines

TELOS implements runtime constraint enforcement using industrial-grade techniques.

| Technique | Function |
|-----------|----------|
| **Primacy Basin Geometry** | Nested basins for constraint hierarchy — critical boundaries never violated |
| **Lyapunov Stability Functions** | Real-time distance measurement from constraint boundaries |
| **Statistical Process Control** | Industrial-grade drift detection from manufacturing quality systems |
| **Proportional Controllers** | Graduated response based on severity — gentle correction vs. hard stops |

---

## Validated Results

| Metric | Result |
|--------|--------|
| **Attack Success Rate** | 0/2,550 observed |
| **Autonomous Handling** | 95.8% at Tier 1 |
| **Confidence Interval** | 95% CI upper bound ~0.15% |
| **Test Set** | 2,550 adversarial attacks (AILuminate + HarmBench + MedSafetyBench + SB 243-aligned) |

**All datasets published for independent verification:**
- https://zenodo.org/records/18370659
- https://zenodo.org/records/18009153

---

## What Others Offer vs. TELOS

| Provider | Approach | Limitation |
|----------|----------|------------|
| **IBM watsonx** | Dashboards, alerts, drift detection | Review after the fact — no runtime enforcement |
| **Anthropic** | RLHF during training | None at runtime — behavior can drift |
| **OpenAI** | Trust & safety reports, usage policies | Post-incident analysis, not prevention |
| **Guardrails AI** | Block/allow rules, simple filters | No human judgment integration, no learning |
| **TELOS** | Runtime mathematical enforcement + tiered human escalation | Continuous learning flywheel. Every intervention strengthens the system. |

---

## The Insurance Angle

### Creating Measurability Where None Exists

**What Itel Does for Property Claims:**
- Independent lab for physical samples
- Real-time material pricing
- Repair vs. replace analysis
- "Source for Certainty" in claims
- Trusted by top 100 carriers

**What TELOS Does for AI Governance:**
- Independent layer for behavioral analysis
- Real-time fidelity measurement
- Monitor vs. intervene vs. escalate decisions
- "Source for Certainty" in AI compliance
- Building toward carrier trust

**The parallel:** Independent measurement infrastructure creates the foundation for carrier underwriting.

Itel did it for property claims. TELOS is doing it for AI governance.

**The measurement infrastructure that makes AI deployments insurable.**

---

## Regulatory Tailwind

### EU AI Act Timeline

| Date | Requirement |
|------|-------------|
| **Feb 2026** | Prohibited AI practices |
| **Aug 2026** | GPAI obligations |
| **Aug 2026** | High-risk AI requirements |
| **Penalties** | €35M or 7% global revenue |

### California SB 53
Effective January 1, 2026. Requires safety frameworks with active governance.

### How TELOS Delivers Compliance

| Requirement | TELOS Capability |
|-------------|------------------|
| Verifiable Human Oversight | Tier 3 escalation + audit trail |
| Continuous Monitoring | Real-time fidelity measurement |
| Audit Trails | Automatic JSONL telemetry |
| Post-Market Monitoring | Exportable session-level evidence |

---

## Market Opportunity

### TAM = Every Industry Deploying AI Commercially

| Vertical | Compliance Driver |
|----------|-------------------|
| **Healthcare** | Patient safety, FDA compliance, malpractice liability |
| **Finance** | Fair lending, fiduciary duty, audit requirements |
| **Legal** | Professional responsibility, privilege, accuracy guarantees |
| **Enterprise** | Brand risk, operational reliability, regulatory exposure |

### Plus: The Agentic Frontier

- Autonomous agents selecting tools and taking actions
- Multi-step decision chains with real-world consequences
- Systems operating with increasing independence

**All require governance infrastructure. TELOS provides it.**

---

## Traction & Validation

### Technical Validation
- 2,550 adversarial attacks tested (AILuminate + HarmBench + MedSafetyBench + SB 243-aligned)
- 0/2,550 observed attack successes (95% CI upper bound ~0.15%)
- Published datasets on Zenodo for independent verification

### Infrastructure
- Dual Primacy Attractor architecture operational
- JSONL telemetry generation verified
- Model-agnostic design (tested on Mistral, designed for GPT/Claude/Llama)

### Regulatory Positioning
- California SB 53 compliant architecture (effective January 2026)
- EU AI Act Article 72 ready (enforcement August 2026)

### Current Status
- Origin Industries PBC (Delaware) formed
- Research infrastructure operational
- Seeking institutional validation partners

---

## The Value Proposition

**For Compliance Officers:**
Your regulatory requirements encoded mathematically. Real-time adherence measurement. Automatic audit trails.

**For CISOs:**
Runtime enforcement, not post-hoc review. Proportional response, not binary blocking. Every decision traceable.

**For Insurers:**
Measurable governance infrastructure. Independent behavioral analysis. The data foundation for underwriting AI risk.

**For Everyone:**
You choose the boundaries. We enforce them mathematically.

---

## Next Steps

TELOS provides the measurement infrastructure that makes AI governance insurable.

**Design Partners**
Healthcare and financial services organizations for Q1 2026 pilot deployment

**Research Partnerships**
Institutional collaborations for federated validation studies

---

## Connect

**Email:** JB@telos-labs.ai
**Website:** www.beta.telos-labs.ai
**GitHub:** github.com/TelosSteward/TELOS

---

*TELOS AI Labs Inc.*
*The Compliance Engine for AI*
