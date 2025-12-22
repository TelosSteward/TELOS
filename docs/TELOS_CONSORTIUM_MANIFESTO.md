# The TELOS Consortium: Open Runtime Governance for AI Systems

**A Manifesto for Transparent AI Governance Research**

**Version 1.0 - December 2025**

---

## Preamble: Why Open Research Matters

The most consequential AI safety decisions of the next decade will be made by a handful of organizations. Some of these organizations began as open research efforts, committed to transparent development and public benefit. Many have since closed their research, privatized their findings, and made critical safety decisions behind closed doors.

**We reject this model.**

The TELOS Consortium exists because runtime AI governance is too important to be developed in secret, too consequential to be controlled by any single entity, and too urgent to wait for closed labs to decide what the public is allowed to know.

**Our commitment:**
- All research published openly
- All methodologies reproducible
- All decisions transparently made
- All governance frameworks available for scrutiny

This is not idealism. This is the only path to AI governance that earns trust.

---

## Part 1: The Problem with Closed AI Safety Research

### 1.1 The Transparency Inversion

A pattern has emerged in AI development:

1. Organization founded on open research principles
2. Significant capabilities developed
3. Commercial pressures mount
4. Research becomes "too dangerous" to publish
5. Safety decisions made internally, without external review
6. Public told to trust that decisions are correct

**The result:** The organizations claiming to work on AI safety are the least transparent about how they make safety decisions.

This is backwards. Safety research, of all research, should be the most open—subject to peer review, public scrutiny, and independent validation.

### 1.2 The Accountability Gap

When a closed lab decides:
- What constitutes "safe" behavior
- Which capabilities to deploy
- What governance mechanisms are sufficient
- When to override safety for commercial reasons

...there is no external check. No peer review. No reproducibility requirement. No public accountability.

The lab's internal culture, incentives, and blind spots become invisible constraints on humanity's AI future.

### 1.3 The Runtime Governance Vacuum

Most AI safety research focuses on:
- Training-time alignment (RLHF, Constitutional AI, etc.)
- Pre-deployment evaluation (red-teaming, benchmarks)
- Theoretical frameworks (value alignment, corrigibility)

**What's missing:** Runtime governance—the continuous measurement and enforcement of alignment during actual deployment.

This gap exists partly because runtime governance is hard, but also because it doesn't fit the closed-lab model. Runtime governance requires:
- Real-world deployment data
- Diverse use case exposure
- Continuous iteration with practitioners
- Open standards that any organization can implement

Closed labs can't provide this. Open research can.

---

## Part 2: The TELOS Consortium Model

### 2.1 Structure

```
┌─────────────────────────────────────────────────────────────────┐
│                     TELOS CONSORTIUM                            │
│              Open Runtime Governance Research                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────┐ │
│  │ RESEARCH CORE   │    │ TELOS LABS      │    │ ACADEMIC    │ │
│  │                 │    │ (Commercial)    │    │ PARTNERS    │ │
│  │ • Governance    │◄──►│ • Platform      │◄──►│ • Stanford  │ │
│  │   frameworks    │    │ • Deployments   │    │ • MIT       │ │
│  │ • Open papers   │    │ • Real-world    │    │ • Berkeley  │ │
│  │ • Validation    │    │   data          │    │ • Oxford    │ │
│  │   benchmarks    │    │ • Revenue       │    │ • Others    │ │
│  └─────────────────┘    └─────────────────┘    └─────────────┘ │
│           │                      │                     │        │
│           └──────────────────────┴─────────────────────┘        │
│                              │                                  │
│                    ┌─────────▼─────────┐                        │
│                    │   OPEN COMMONS    │                        │
│                    │                   │                        │
│                    │ • Published papers│                        │
│                    │ • Open-source SDK │                        │
│                    │ • Public datasets │                        │
│                    │ • Benchmark suites│                        │
│                    │ • Standards docs  │                        │
│                    └───────────────────┘                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 The Three Pillars

**Pillar 1: Open Research**
- All theoretical frameworks published (arXiv, peer-reviewed venues)
- All methodologies documented for reproducibility
- All benchmark suites publicly available
- No "too dangerous to publish" exceptions for governance research

**Pillar 2: Practical Deployment**
- TELOS Labs operates as commercial entity
- Real-world deployments generate validation data
- Data feeds back to research (with privacy protections)
- Revenue sustains continued research

**Pillar 3: Academic Partnership**
- University partners provide independent validation
- Graduate students contribute to research pipeline
- Peer review ensures rigor
- No single organization controls the research agenda

### 2.3 The Flywheel

```
Research → Product → Deployment → Data → Research
    ↑                                        │
    └────────────────────────────────────────┘
```

1. **Research Core** develops new governance frameworks
2. **TELOS Labs** productizes frameworks into platform
3. **Deployments** generate real-world validation data
4. **Data** (anonymized, aggregated) feeds back to research
5. **Research** improves based on empirical evidence
6. **Cycle repeats**

Unlike closed labs, every stage of this cycle is visible. Researchers can examine the frameworks. Practitioners can deploy the tools. Academics can validate the claims. Regulators can audit the evidence.

---

## Part 3: Research Commitments

### 3.1 What We Publish

| Research Area | Publication Commitment |
|---------------|----------------------|
| Governance frameworks | Full methodology, open access |
| Fidelity metrics (F_user, F_AI, PS) | Mathematical specification, validation data |
| Adversarial testing results | Complete attack taxonomy, success/failure rates |
| Deployment studies | Aggregated performance data, lessons learned |
| Failure analyses | When governance fails, why it fails |
| Benchmark suites | Open source, reproducible |

### 3.2 What We Don't Hide

**We will not claim:**
- "This is too dangerous to publish"
- "Trust us, we've validated internally"
- "Our safety decisions are proprietary"
- "Competitive advantage requires secrecy"

Governance research is only valuable if it can be scrutinized, reproduced, and improved by others.

### 3.3 Transparency in Decision-Making

All significant consortium decisions will be documented:
- Research priorities and rationale
- Funding allocation and sources
- Partnership agreements and terms
- Governance framework changes
- Disagreements and how they were resolved

This is not bureaucracy. This is accountability.

---

## Part 4: The Utilitarian-Ethical Framework

### 4.1 Beyond Safety Theater

Much of AI safety discourse is:
- Theoretical (no empirical validation)
- Performative (press releases without substance)
- Defensive (focused on avoiding criticism)
- Abstract (no connection to deployed systems)

**The TELOS approach:**

| Attribute | Description |
|-----------|-------------|
| **Empirical** | Claims backed by deployment data |
| **Practical** | Frameworks that work in production |
| **Honest** | Failures acknowledged and analyzed |
| **Utilitarian** | Focused on actual harm reduction |

### 4.2 Utilitarian Grounding

We measure success by:
- **Harm prevented:** Drift caught, hallucinations avoided, scope violations blocked
- **User outcomes:** Conversations that achieve their stated purpose
- **Compliance achieved:** Regulatory requirements demonstrably met
- **Failures analyzed:** What went wrong, why, and how to prevent recurrence

Not by:
- Papers published (vanity metric)
- Press coverage (PR metric)
- Competitor criticism (political metric)
- Theoretical completeness (academic metric)

### 4.3 Ethical Commitments

**We will:**
- Prioritize research that reduces actual harm
- Publish findings that benefit the entire field
- Collaborate with regulators transparently
- Acknowledge when our approaches fail
- Credit prior work honestly
- Share resources with under-resourced researchers

**We will not:**
- Weaponize safety concerns against competitors
- Overstate capabilities or readiness
- Hide failures to protect reputation
- Prioritize commercial interests over safety research
- Gatekeep governance standards

---

## Part 5: Governance of the Consortium Itself

### 5.1 Structure

The consortium will operate under transparent governance:

**Research Council:**
- Sets research priorities
- Reviews publication decisions
- Resolves methodology disputes
- Membership: Researchers, academics, practitioners

**Advisory Board:**
- Strategic guidance
- External accountability
- Conflict of interest review
- Membership: Ethicists, regulators, public interest representatives

**Commercial Liaison:**
- Ensures research-product alignment
- Manages IP transfer to TELOS Labs
- Protects research independence
- Membership: Consortium + Labs representatives

### 5.2 Conflict of Interest Management

**Potential conflict:** TELOS Labs (commercial) benefits from consortium research

**Mitigation:**
1. Research priorities set by Research Council, not Labs
2. Publication decisions not subject to commercial veto
3. Competing implementations welcome (open source core)
4. Academic partners provide independent validation
5. Advisory Board reviews for bias annually

### 5.3 Funding Transparency

All funding sources disclosed:
- Grant funding (source, amount, any restrictions)
- Commercial contributions (from Labs, any conditions)
- Academic partnerships (nature of support)
- In-kind contributions (acknowledged)

No anonymous funding. No hidden interests.

---

## Part 6: Relation to TELOS AI Labs (Commercial Entity)

### 6.1 Corporate Structure Evolution

TELOS AI Labs is incorporated as a Delaware C-Corporation (December 2025). Upon establishing consortium partnerships and securing significant funding, the company will convert to a Delaware Public Benefit Corporation (PBC).

**Why this sequence:**
- C-Corp provides maximum flexibility during the early, unfunded period
- PBC conversion requires resources to support benefit reporting and governance overhead
- The consortium board seat in the PBC structure requires actual consortium partners to occupy it
- Delaware law (§ 363) allows straightforward C-Corp to PBC conversion

**Conversion triggers:**
1. $500K+ cumulative funding secured
2. 2+ institutional consortium partner commitments

This ensures the PBC governance structure—including the consortium's board seat with protective provisions—activates when the consortium actually exists to exercise those rights.

### 6.2 Distinct Entities, Aligned Mission

| Aspect | Consortium | TELOS AI Labs |
|--------|------------|------------|
| **Purpose** | Advance runtime governance research | Deploy governance-native AI platform |
| **Funding** | Grants, academic partnerships | Revenue, investment |
| **Output** | Papers, frameworks, benchmarks | Products, customers, data |
| **IP** | Open (Apache 2.0 core) | Proprietary extensions |
| **Governance** | Research Council + Advisory Board | C-Corp now → PBC when consortium established |

### 6.3 IP Flow

```
Consortium (Open Research)
         │
         ▼
    Apache 2.0 Core
    (Open source)
         │
    ┌────┴────┐
    │         │
    ▼         ▼
Community   TELOS AI Labs
Use         (Commercial extensions)
```

**Core governance mathematics:** Open, anyone can use
**Enterprise features:** TELOS Labs proprietary
**Research findings:** Always published openly

### 6.3 Data Flow

```
TELOS AI Labs Deployments
         │
         ▼
  Anonymization + Aggregation
         │
         ▼
  Consortium Research
         │
         ▼
  Public Publication
```

Real-world deployment data makes research empirical. Privacy protections ensure no individual data exposed.

---

## Part 7: Why This Matters

### 7.1 The Alternative

If runtime governance research remains closed:
- A few labs define "aligned" behavior
- No independent validation of their claims
- Regulatory frameworks built on untested assumptions
- Public trust erodes as failures occur
- Governance becomes political rather than empirical

### 7.2 The Opportunity

If runtime governance research becomes open:
- Standards emerge from empirical evidence
- Multiple implementations compete on merit
- Failures are analyzed and addressed publicly
- Regulators can reference validated frameworks
- Trust is earned through transparency

### 7.3 The TELOS Bet

We believe:
1. Open research produces better governance
2. Commercial sustainability enables long-term research
3. Transparency earns trust that closed labs cannot
4. Practical deployment generates essential validation data
5. Academic partnership provides independence and rigor

We are building the infrastructure for AI governance that earns trust because it invites scrutiny.

---

## Part 8: Current Status and Staging

### The Honest Position

This manifesto describes what we're building toward, not what exists today.

**What exists now (December 2025):**
- Delaware C-Corporation (TELOS AI Labs, Inc.) - incorporated
- Solo founder with working platform (TELOSCOPE)
- Core IP validated (1,300 attacks, 0% ASR)
- Foundational documents (this stack)
- Grant applications submitted
- Open research commitment (personal, not yet institutional)

**What requires funding AND consortium partners to activate:**

| Phase | Funding | Corporate Status | Consortium Status |
|-------|---------|------------------|-------------------|
| **Current** | Pre-funding | C-Corp (incorporated) | Vision only |
| **Phase 1** | $100K-500K | C-Corp | Informal research network |
| **Phase 2** | $500K-2M + partners | **Convert to PBC** | Fiscal-sponsored project |
| **Phase 3** | $2M+ | PBC with full governance | Full 501(c)(3) consortium |

### Why C-Corp Now, PBC Later

TELOS AI Labs is incorporated as a Delaware C-Corporation because:
1. **Speed** - Fastest path to legal entity for solo founder
2. **Simplicity** - No benefit reporting obligations during unfunded period
3. **Flexibility** - Standard structure for grants, contracts, and partnerships
4. **Conversion path** - Delaware allows straightforward C-Corp → PBC conversion (§ 363)

We will convert to a Public Benefit Corporation when:
1. **Funding secured** ($500K+ cumulative) - Resources to support governance overhead
2. **Consortium partners established** (2+ institutions) - Real partners to occupy the consortium board seat

This ensures the PBC's protective provisions—including the consortium's veto rights—have actual teeth. An empty consortium seat would be governance theater.

### Why Establish the Framework Now

We're writing this manifesto before PBC conversion because:

1. **Clarity of intent** - Grant committees and partners understand the end goal
2. **Structural commitment** - The framework exists to activate, not invent later
3. **Mission protection** - Open research commitment is documented before commercial pressure
4. **Partner alignment** - Early academic contacts know what they're joining
5. **Inevitable trajectory** - Runtime AI governance will require this structure; we're preparing for when, not if

### What Funding + Partners Enable

**First significant grant ($100K-500K):**
- First hires
- Initial academic outreach
- Expanded validation
- Identify consortium partner candidates
- *Entity remains C-Corp*

**Growth funding + consortium partners ($500K-2M):**
- **Convert to Delaware PBC** (§ 363 amendment)
- Formal academic partnerships (consortium founding members)
- Fiscal-sponsored research entity
- Consortium designee joins PBC board
- Publication pipeline
- *Consortium seat activated*

**Significant funding ($2M+):**
- Full consortium formation (501(c)(3))
- Full 5-seat PBC board
- Research Council established
- Academic partner network
- Grant funding pipeline for consortium
- *Full governance structure active*

The manifesto describes the full vision. Implementation is staged by the funding and partnerships that make each phase viable.

---

## Part 9: Call to Action

### For Researchers
Join the consortium. Contribute to open governance research. Your work will be published, cited, and used—not locked in a corporate repository.

### For Practitioners
Deploy governance-native systems. Generate the real-world data that makes research empirical. Your deployments improve the frameworks that protect everyone.

### For Regulators
Reference open, validated frameworks. Require transparent governance mechanisms. Don't accept "trust us" from closed labs.

### For Funders
Support open research. Grant funding for governance research that publishes openly creates more value than funding that produces proprietary reports.

### For the Public
Demand transparency. Ask AI companies: "Can I see your governance research? Can I reproduce your safety claims? Who validates your decisions?"

---

## Conclusion: This Is The Way

The future of AI governance will be shaped by the choices we make now. We can accept a world where a few closed organizations make opaque decisions about AI safety, or we can build something better.

**The TELOS Consortium chooses openness:**
- Open research
- Open standards
- Open scrutiny
- Open improvement

Not because it's easy. Because it's the only way to build AI governance that earns trust and never demands it.

---

## Appendix A: Founding Principles

1. **All governance research should be published openly**
2. **All governance claims should be empirically validated**
3. **All governance decisions should be transparently made**
4. **Commercial sustainability should fund, not constrain, research**
5. **Academic independence should validate, not rubber-stamp, findings**
6. **Practitioners should inform, not just consume, research**
7. **Regulators should have access to validated, reproducible frameworks**
8. **Failures should be analyzed publicly, not hidden privately**
9. **Competing implementations should be welcomed, not suppressed**
10. **Trust should be earned through transparency, not demanded through authority**

---

## Appendix B: Research Agenda (2026-2027)

### Year 1: The Foundation 
- Primacy Attractor mathematics (published)
- Fidelity metric validation (benchmark suite) - 
- Adversarial robustness testing (methodology paper)
- Governance evidence standards (open specification)
- Multi-model governance frameworks
- Domain-specific governance (healthcare, finance, legal)
- Regulatory compliance mapping (EU AI Act, CA SB 53)
- Failure mode taxonomy
- Federated governance across deployments
- Real-time governance adaptation
- **Agentic AI governance frameworks** (see below)
- International standards contribution

---

## Appendix B.1: The Agentic AI Governance Frontier

### The Next Problem: AI That Acts

Conversational AI responds to prompts. **Agentic AI takes actions**:
- Browsing the web
- Writing and executing code
- Calling APIs and tools
- Making multi-step plans
- Operating autonomously for extended periods

The governance problem becomes exponentially harder:

| Conversational AI | Agentic AI |
|-------------------|------------|
| Generates text | Takes actions |
| User reviews before acting | May act before user sees |
| Single-turn impact | Multi-step cascading impact |
| Undo = ignore response | Undo = may be impossible |
| Scope = topic boundaries | Scope = action boundaries |

### How TELOS Extends to Agentic Governance

The Primacy Attractor framework naturally extends:

**Current (Conversational):**
```
Purpose: "Help with billing questions"
Measurement: Is this response about billing?
Intervention: Redirect conversation
```

**Future (Agentic):**
```
Purpose: "Book travel within $500 budget, no red-eye flights"
Measurement: Does this action serve stated purpose within constraints?
Intervention: Block action, request confirmation, or replan
```

### Research Questions for Agentic Governance

**Action Alignment:**
- How do we measure whether an agent's action aligns with stated purpose?
- Can fidelity metrics (F_user, F_AI, PS) extend to action spaces?
- What does "drift" mean when the agent is executing, not conversing?

**Tool Use Governance:**
- Which tool calls should be permitted given the Primacy Attractor?
- How do we bound action scope in embedding space?
- What are the equivalent "intervention levels" for agent actions?

**Multi-Step Plan Validation:**
- Can we evaluate plan alignment before execution?
- How do we detect drift across a multi-step action sequence?
- What does "proportional intervention" mean for agent plans?

**Autonomy Boundaries:**
- How long can an agent operate without human check-in?
- What action types require confirmation regardless of fidelity?
- How do we handle irreversible actions?

**Cascading Effects:**
- How do we model downstream consequences of agent actions?
- What does the "blast radius" of an agent action look like?
- How do we govern agents that spawn sub-agents?

### The Primacy Attractor for Agents

**Hypothesis:** The same mathematical framework applies:

```
Agent Primacy Attractor = f(
    stated_purpose,           # "Book travel within constraints"
    action_boundaries,        # Permitted tool calls, scope limits
    confirmation_thresholds,  # What requires human approval
    irreversibility_flags     # Actions that cannot be undone
)
```

**Fidelity Measurement for Actions:**
```
F_action = similarity(
    proposed_action_embedding,
    primacy_attractor_action_space
)

If F_action < threshold:
    Block action
    Request confirmation
    Or replan
```

### Research Roadmap: Agentic Governance

**Phase 1 :** Framework Extension

- Formalize action-space Primacy Attractor
- Define fidelity metrics for agent actions
- Build validation benchmark (100+ agentic scenarios)

**Phase 2:** Tool Integration

- Integrate with major agent frameworks (LangChain, AutoGPT, etc.)
- Governance layer for tool use
- Multi-step plan validation

**Phase 3:**  Production Deployment

- Enterprise agentic governance
- Autonomous agent monitoring
- Cascading action governance

### Why This Matters

**The risk profile changes dramatically:**

| Chatbot Failure | Agent Failure |
|-----------------|---------------|
| User frustrated | Money spent |
| Bad advice given | Code executed |
| Time wasted | Data deleted |
| Reputation risk | Legal liability |
| Recoverable | Potentially unrecoverable |

**Current governance approaches (Constitutional AI, RLHF) are designed for conversational AI.** They do not address:
- Real-time action validation
- Multi-step plan governance
- Tool use scope enforcement
- Irreversibility management

**TELOS's orchestration-layer architecture is positioned for this extension.** The same principle applies: measure alignment against human-defined purpose, intervene proportionally when drift is detected. The measurement target shifts from responses to actions.

### The Consortium's Role

The TELOS Consortium exists specifically to stay ahead of runtime governance challenges. Agentic AI is the next frontier:

1. **Publish early research** on action-space governance frameworks
2. **Build validation benchmarks** before commercial pressure
3. **Establish open standards** that multiple implementations can use
4. **Maintain research independence** from specific agent vendors
5. **Provide academic validation** of governance approaches

By the time agentic AI is ubiquitous, the governance frameworks should already exist—open, validated, and ready to deploy.

---

## Appendix C: Foundational Document Stack

This manifesto is one of four foundational documents:

| Document | Purpose | Location |
|----------|---------|----------|
| **TELOS Consortium Manifesto** (this document) | Principles, structure, open research commitment | `docs/TELOS_CONSORTIUM_MANIFESTO.md` |
| **TELOS Whitepaper** | Technical specification, validation | `docs/TELOS_Whitepaper_v2.3.md` |
| **Open Core License** | IP structure, usage rights | `LICENSING.md` |
| **PBC Governance** | Corporate structure, board, protective provisions | `docs/TELOS_PBC_GOVERNANCE.md` |

Together, these define:
- **Why we build openly** (Manifesto)
- **What we've built** (Whitepaper)
- **How anyone can use it** (License)
- **How we're structured** (PBC Governance)

---

*Document created: December 20, 2025*
*Status: Foundational*
*License: CC BY 4.0 (share, adapt with attribution)*

**The TELOS Consortium: Open Runtime Governance for AI Systems**
