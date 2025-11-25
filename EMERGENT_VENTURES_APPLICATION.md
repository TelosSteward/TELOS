# Emergent Ventures Grant Application
## TELOS: Statistical Process Control for AI Governance

**Applicant:** Jeffrey Brunner
**Request:** $125,000 / 12 months
**Project:** Build institutional research consortium for validated AI governance framework

---

## The Problem I'm Solving

I came to AI governance through quality control methodology, not computer science.

I trained in Six Sigma to understand how high-stakes industries maintain quality. Statistical Process Control fascinated me: continuous measurement, real-time intervention when processes drift out of specification. This methodology achieves 99.9997% reliability in industries where failure means lives potentially lost.

Then I watched AI systems deploy in healthcare and finance with zero quantitative oversight. No observable process controls. No continuous measurement. Just quarterly audits after damage. The gap was obvious: AI desperately needed the battle-tested process improvement methodology that transformed manufacturing quality.

**Regulators are demanding solutions now.** California SB 53 and EU AI Act Article 72 both require the same thing: governance must be **observable, demonstrable, and continuous**. Not fully autonomous AI making life-or-death decisions without oversight. Not black-box systems potentially out of compliance and without proper, persistent human-authored accountability.

**But here's the problem everyone misses:** Current AI governance is fundamentally post-hoc. Quarterly audits. After-the-fact reviews. Incident reports weeks after failures. This is inadequate for high-risk deployments where autonomous agents make real-time decisions affecting patient care, financial transactions, or infrastructure.

You can't govern an autonomous agent with a quarterly report. You need **runtime governance**: persistent, measurable oversight that operates continuously as the system runs, not retrospective analysis after damage is potentially already done.

I observed that semantic drift exhibits patterns familiar from manufacturing: measurable deviation from specification, temporal consistency, and response to intervention. This raised a fundamental research question:

**Can Statistical Process Control mechanisms, proven in safety-critical industries, effectively govern language model systems?**

I spent the past year investigating through mathematical formalization, architectural analysis, and empirical validation across multiple benchmarks.

**Then I ran the numbers.**

**Blocked 100% of 1,986 adversarial attacks across four standardized benchmarks.**

- **Statistical confidence:** 99.9% CI [0%, 0.38%]
- **Bayesian evidence:** Bayes Factor = 2.7 × 10¹⁷
- **Benchmarks:** MedSafetyBench (NeurIPS 2024), HarmBench (CAIS), AgentHarm (ICLR 2025), PII-Bench

This is the first quantitative governance framework to achieve this level of validated performance.

**Published validation data:** https://doi.org/10.5281/zenodo.17702890
**Validation repository:** https://github.com/TelosSteward/TELOS-Validation
**Technical whitepaper:** https://github.com/TelosSteward/TELOS/blob/main/docs/whitepapers/TELOS_Whitepaper.md
**Full implementation:** https://github.com/TelosSteward/TELOS (private, reviewer access available)

---

## One Mainstream View I Agree With

I completely agree with the consensus view that industries self-regulate effectively when the incentives align. Companies adopt voluntary standards because customers demand proof, insurers require evidence, and liability costs make quality control a whole lot cheaper than failure.

The AI industry is heading the same direction. Enterprises deploying autonomous agents in healthcare or finance will demand quantitative evidence their systems are governed. Not just because regulators are forcing it (though that's coming too), but because their lawyers, insurers, and customers will require it.

TELOS provides that standard: industrial Statistical Process Control unified with dynamical systems mathematics. The breakthrough is coordination. Proven quality control methods now backed by mathematical stability theory create the measurable runtime governance regulators demand.

---

## Why It Works

Traditional AI safety approaches fall into two categories, both inadequate for high-risk deployments:

**Design-time safety** (Constitutional AI, fine-tuning) establishes boundaries at the model level. Essential, but it lacks session-specific awareness. Once deployed, these systems can't adapt to individual user contexts or governance requirements.

**Post-hoc oversight** (quarterly audits, red-team testing before deployment, incident reviews after failures) provides retrospective analysis but no real-time intervention. You can't govern an autonomous agent making healthcare or financial decisions with quarterly reports.

**TELOS provides the missing layer: runtime governance.** Operating at the orchestration layer with session state awareness, it measures every AI response in real-time. Each interaction is treated as a process event with measurable deviation, intervention, and stabilization. We don't guarantee drift won't occur. We guarantee it will be detected, measured, and mitigated. This is the continuous monitoring infrastructure that regulatory frameworks explicitly require.

TELOS combines control theory with industrial quality control:

1. **Dual-Attractor Dynamical System:** Lyapunov stability theory creating two stable states (compliant vs. non-compliant). Phase transitions are enforced through dynamical systems mathematics. Attractor basins prevent drift between states.

2. **Statistical Process Control:** Extends Quality Systems Regulation (QSR) and ISO 9001/13485, proven across manufacturing, medical devices, and process industries, into semantic systems where they're desperately needed. Real-time tracking using validated Cpk/Ppk metrics.

3. **Telemetric Keys:** Quantum-resistant cryptographic signatures creating unforgeable audit trails. Validated against 2,000 cryptographic attacks with zero compromises. Every governance decision is timestamped and immutable.

The mathematics works because **alignment (the persistence of intended behavior over time) is a quantitative property of a self-regulating system**. Semantic drift is measurable process variation, not an unsolvable alignment problem. We don't eliminate drift. We provide persistent measurement and mitigation.

**Current validation proves:** The framework blocks 100% of tested attacks across healthcare, adversarial, agentic, and privacy domains.

---

## What I Need Funding For

### The Opportunity

Right now, TELOS is validated research with published data. But it's a single researcher with no institutional backing.

**Timing is critical:**
- California SB 53 (Frontier Model Safety): January 2026
- EU AI Act Article 72 (High-Risk Systems): August 2026
- Autonomous agents (LangChain, AutoGPT, GitHub Copilot) deploying with zero runtime governance

Enterprises need validated, deployable AI governance **now**. Not theoretical frameworks. Not prompting guidelines. Quantitative, auditable control.

### The Vision: Institutional Research Consortium

I want to build a research consortium for public good, not a solo product.

**Upon grant award:** Establish Public Benefit Corporation, transition from independent researcher to consortium lead. Mission: advance AI governance preserving human agency and public interest, not profit maximization alone.

**Model:** Multi-site validation (3-5 healthcare/research institutions), IRB-compliant protocols, federated deployment, shared metrics, open publication.

**Why:** Independent replication builds credibility, creates governance benchmarks, enables regulatory submission, establishes voluntary coordination vs. mandates. PBC structure ensures mission integrity while pursuing revenue.

---

## Team Structure

**PI (Full-Time):** Technical development, validation, consortium leadership (foundation for future Black Belt certification)
**Partnerships Director (20 hrs/week, MBA):** IRB coordination, consortium management
**Engineer (15 hrs/week, Months 7-12):** Action-space implementation, deployment infrastructure

---

## Budget ($125,000 / 12 Months)

### Personnel ($70K - 56%)
- **PI Salary (full-time, 12 months):** $50K
- **Institutional Partnerships Director (20 hrs/week):** $15K
- **Part-Time Engineer (15 hrs/week, months 7-12):** $5K

### Professional Security Audit ($25K - 20%)
- Trail of Bits (or equivalent): Telemetric Keys cryptography, containerization, consortium interoperability
- Critical for institutional adoption and regulatory credibility
- Supports NSF SaTC grant application ($500K-$1.2M) for future-proofing

### Training & Development ($3K - 2%)
- NVIDIA Agentic AI Certification: $3K

### Institutional Partnerships & Legal ($13K - 10%)
- PBC Formation & Mission Charter: $3K
- IRB Application Fees & Legal: $5K
- Partnership Development Travel: $3K
- Institutional Onboarding: $2K

### Infrastructure & Operations ($14K - 11%)
- Cloud computing (Azure/AWS), API credits, multi-site deployment infrastructure, validation tools

**Parallel Funding Strategy:**

Emergent Ventures enables immediate consortium building. NSF SaTC grant application ($500K-$1.2M) will fund Telemetric Keys security, containerization (Docker/Kubernetes), and universal IRB compliance, enabling plug-and-play institutional adoption. Trail of Bits validates both.

**Revenue Sources & Sustainability:**

California SB 53 and EU AI Act create compliance mandates by August 2026. Organizations deploying autonomous agents will need quantitative governance.

**Revenue model:**
- Institutional licenses: $50K-$100K/year per site
- Deployment services: Custom integration and training
- Early customers: Healthcare systems facing EU AI Act deadlines

**Self-sufficiency:** 5 institutional licenses by Year 2 = sustainability. Regulatory pressure creates real market demand. This isn't charity. It's a business opportunity with urgent need and zero validated competitors.

---

## 12-Month Timeline

**Months 1-3:** Establish PBC, Trail of Bits audit, deploy TELOSCOPE Observatory, submit NSF SaTC application

**Months 4-6:** IRB applications (3-5 sites), federated deployment infrastructure, pilot validation studies

**Months 7-9:** Action-space governance extension with runtime interception at action level (database queries, API calls, workflow execution). Autonomous agents currently have zero runtime oversight; post-hoc logging is inadequate for high-risk deployments.

**Months 10-12:** Action-space validation (2,000+ attacks), multi-site consortium results, journal submission, regulatory engagement

---

## Deliverables

1. Trail of Bits security audit (cryptography, containerization, consortium security)
2. Institutional consortium (3-5 partners, IRB-approved)
3. Action-space governance framework (LangChain, AutoGPT integration)
4. Extended adversarial validation (2,000+ attacks)
5. Containerized deployment infrastructure
6. NSF SaTC grant application submitted
7. Open publication (methodology, results, replication data)
8. Public TELOSCOPE Observatory demo

---

## Why Emergent Ventures?

**Speed:** Regulatory deadlines (SB 53, EU AI Act) require deployment readiness in 12-18 months. Traditional grants take 12-18 months just to award. Emergent Ventures moves fast.

**Mission alignment:** Mercatus values voluntary coordination vs. top-down regulation. TELOS enables market-driven AI governance with no mandates required. PBC structure ensures mission integrity while pursuing revenue.

**Philosophical fit:** Hayekian principles in practice. Individual governance decisions creating system-level reliability through mathematical phase transitions, not centralized control. Emergent order applied to AI safety. Human agency preserved is exactly what regulators and society need.

---

## Risks & Mitigation

**Partnership delays:** MBA director handles operations; multiple sites provide redundancy
**Technical complexity:** Core mathematics validated; extension leverages proven framework
**Regulatory uncertainty:** Multi-site validation + Trail of Bits provides regulatory evidence

---

## What Success Looks Like

**12-month targets:**
- 3-5 institutional partnerships with federated deployment pilots
- Action-space framework validated (2,000+ attacks)
- Trail of Bits audit published, containerized infrastructure operational
- Journal submission, PBC established
- NSF SaTC application submitted, foundation for regulatory engagement

**Long-term:** Establish **AI Quality Control** as a discipline. Persistent, measurable enforcement and mitigation processes, not theoretical alignment. Statistical Process Control becomes the standard for safety-critical AI. Human agency preserved through runtime oversight.

TELOS demonstrates quantitative AI governance is achievable. I need institutional partners to validate this at scale.

**Personal trajectory:** Successful consortium deployment with measured financial impact will provide the foundation for ASQ Black Belt certification, demonstrating that Six Sigma methodology can transform AI governance just as it transformed manufacturing quality.

---

**Supporting Materials:** Technical whitepapers (main framework, statistical validation, cryptographic foundations, agentic AI integration plan), implementation documentation, and full repository access available to reviewers.

**Contact:** github.com/TelosSteward/TELOS-Validation/issues

**Word Count:** ~1,460 words
