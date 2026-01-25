# NSF SBIR Project Pitch Draft
## TELOS: Mathematical Governance Infrastructure for Agentic AI Systems

**Company:** TELOS AI Labs Inc.
**Date:** January 2026

---

## 1. THE TECHNOLOGY INNOVATION (3,500 characters max)

**Current:** ~3,240 characters

AI systems drift from intended purpose during operation, creating compliance risk in healthcare, finance, and critical infrastructure. Current governance relies on heuristic defenses: prompt injection classifiers, harm category filters, and keyword-based guardrails. Peer-reviewed research documents their inadequacy—Agent Security Bench (ICLR 2025) reports 84% attack success rates against current defenses; AgentDojo (NeurIPS 2024) shows even "best" defenses leave 30% of attacks successful while degrading utility to 53%.

**Two fundamental limitations:** First, these approaches ask the wrong question—detecting pre-defined harm categories rather than measuring alignment to declared purpose. Task Shield (ACL 2025) validates this distinction, showing purpose alignment reduces attack success from 47% to 2%. Second, they use binary event detection rather than continuous measurement with graduated intervention—no trajectory tracking, no drift quantification, no proportional response. Task Shield fixes the first problem but not the second; it remains single-request and binary.

TELOS addresses both: asking the right question (purpose alignment via Primacy Attractor) AND using the right framework (Statistical Process Control in semantic space). The core innovation is the **Primacy Attractor**—an embedding-space representation of user-defined purpose. Every AI response is embedded and measured against this attractor using cosine similarity, transforming alignment from subjective assessment to quantitative measurement.

The two-layer fidelity architecture:
- **Layer 1 (Baseline Pre-Filter):** Catches content outside purpose embedding space
- **Layer 2 (Basin Membership):** Detects gradual drift through normalized fidelity measurement

When drift is detected, a proportional controller applies graduated intervention scaled to drift severity—extending proven industrial quality methodology (DMAIC, ISO 9001, FDA QSR) into semantic systems.

**Validated Results Across Four Benchmarks:**

*Adversarial Security:* 0% Attack Success Rate on 1,300 attacks (HarmBench + MedSafetyBench). 99.9% CI: [0%, 0.28%]. Compared to 3.7-11.1% for system prompts and 30.8-43.9% for undefended models.

*Out-of-Scope Detection (CLINC150):* Standard classifiers achieve 0% OOS detection—they must assign every input to some intent. TELOS governance achieves **78% OOS detection** with only 4.5% false positive rate by introducing fidelity-based gating before classification.

*Drift Detection (MultiWOZ):* 100% detection across cross-domain, off-topic, and adversarial drift categories. Critically, **jailbreak attempts produce negative fidelity scores**—the mathematical structure of semantic similarity itself exposes the attack.

Results published on Zenodo with DOIs (10.5281/zenodo.18013104, 10.5281/zenodo.18009153).

**The Phase I Research Challenge:** Extending this validated framework from conversational AI to agentic AI systems. Agents execute action chains (API calls, database operations, transactions) where consequences are irreversible and error compounding creates cascade failures. Can Primacy Attractor fidelity govern action-space trajectories with the same rigor demonstrated in semantic space? This requires novel action-chain embeddings, sub-50ms intervention, and formal reversibility classification—genuine high-risk R&D on proven foundations.

---

## 2. THE TECHNICAL OBJECTIVES AND CHALLENGES (3,500 characters max)

**Current:** ~3,400 characters

**Primary Objective:** Validate that Primacy Attractor governance—proven in conversational AI—provides effective defense against adversarial agentic attacks using established benchmarks (AgentHarm, AgentDojo).

**The Research Gap:** Peer-reviewed benchmarks (ICLR 2025, NeurIPS 2024, ACL 2025) document both the threat and the defense gap. AgentHarm shows LLMs "readily comply with malicious requests without jailbreaking" (80%+ baseline ASR). AgentDojo reveals "existing defenses break security properties" with 30-53% residual ASR. Critically, our analysis of six defense mechanisms across three benchmarks identifies a universal gap: **no current approach addresses trajectory-level drift**—multi-step attacks where individual actions appear benign but cumulative behavior diverges from purpose. Task Shield (ACL 2025) validates that purpose alignment outperforms harm detection (2% vs 47% ASR), but provides only single-request evaluation with no trajectory tracking and no released implementation. TELOS addresses this documented gap with mathematical fidelity measurement across action chains.

**Technical Objective 1: Gateway Architecture Completion (Months 1-2)**
Complete the TELOS Gateway—an OpenAI-compatible proxy that intercepts agent API calls, measures fidelity against declared purpose (Primacy Attractor), and applies graduated governance (EXECUTE/CLARIFY/SUGGEST/INERT/ESCALATE). Architecture exists; Phase I completes tool manifest enforcement, multi-step trajectory tracking, and production hardening. Success metric: Gateway handles 100 req/sec with <50ms governance overhead.

**Technical Objective 2: AgentHarm Validation (Months 2-4)**
Validate TELOS defense against AgentHarm's 110 malicious agent tasks across 11 harm categories (fraud, cybercrime, harassment). Compare: (a) baseline agent (no governance), (b) system-prompt defense, (c) TELOS-governed agent. Success metric: Reduce Attack Success Rate from baseline ~80% to <5% while maintaining task utility on benign requests.

**Technical Objective 3: AgentDojo Validation (Months 3-5)**
Validate against AgentDojo's 97 realistic tasks and 629 security test cases covering prompt injection, tool manipulation, and context poisoning. This benchmark was used by NIST and UK AISI to evaluate Claude 3.5 Sonnet. Success metric: TELOS-governed agents achieve top-quartile security scores while maintaining utility.

**Technical Objective 4: Trajectory-Level Fidelity (Months 4-6)**
Extend single-request fidelity to multi-step action chains. Develop cumulative drift detection that catches error compounding before cascade failure. Success metric: Detect 95% of multi-step attack sequences by step 3.

**Key Technical Challenges:**

**Challenge 1: Utility-Security Tradeoff.** Aggressive governance blocks attacks but may impair legitimate use. Mitigation: Calibrate thresholds per-domain; measure Net Resilient Performance (NRP) metric from ASB.

**Challenge 2: Latency Budget.** Agent loops execute in milliseconds. Mitigation: Cache PA embeddings, parallelize fidelity computation, target <50ms overhead.

**Challenge 3: Adaptive Attacks.** Attackers will probe TELOS thresholds. Mitigation: Publish methodology openly; design for adversarial robustness from day one.

**Deliverables:** Validated Gateway, AgentHarm/AgentDojo benchmark results, trajectory fidelity specification, peer-reviewed publication.

---

## 3. THE MARKET OPPORTUNITY (1,750 characters max)

**Current:** ~1,740 characters

The AI governance market is projected to reach $15-25B by 2028, driven by regulatory mandates. The EU AI Act requires continuous post-market monitoring for high-risk AI systems. California SB 53 mandates safety framework documentation effective January 2026. Organizations face €35M or 7% global revenue penalties for non-compliance.

**Regulatory Uncertainty Is a Market Differentiator:** The EU's November 2025 Digital Omnibus proposal may extend certain deadlines to 2027-2028—but if it fails adoption before August 2026, original deadlines apply with €35M penalties. Organizations cannot gamble on legislative outcomes. TELOS provides governance infrastructure that satisfies compliance requirements regardless of which timeline prevails. This regulatory hedging capability is unavailable from competitors focused solely on technical defenses.

**Customer Segments:**
- Enterprise AI deployers in regulated industries (healthcare, finance, legal, HR)
- Insurance carriers entering AI liability market—cannot price risk without measurement
- AI platform providers seeking compliance solutions for enterprise customers

**The Gap:** 93% of C-suite executives report governance challenges with agentic AI (Teradata 2024). Current defenses leave 30-84% of attacks successful. None provide mathematical runtime measurement of purpose adherence with auditable governance traces.

**TELOS Differentiation:** Only solution with validated near-zero attack success rate (95% CI: [0%, 0.28%]) against adversarial benchmarks. Provider-agnostic. Generates EU AI Act Article 72 compliant telemetry automatically. Maps to ISO 9001, FDA 21 CFR Part 820, and NIST AI RMF.

**Commercial Path:** SaaS governance platform for enterprise, with insurance MGA partnership for AI liability coverage bundled with governance infrastructure.

---

## 4. THE COMPANY AND TEAM (1,750 characters max)

**Current:** ~1,720 characters

**Founder:** Jeffrey Brunner developed the complete TELOS mathematical governance framework over 2+ years, including:
- Primacy Attractor theory and two-layer fidelity architecture
- Working production system (beta.telos-labs.ai) with Streamlit interface
- Validation across 4 benchmarks: 0% ASR (1,300 attacks), 78% OOS detection (CLINC150), 100% drift detection (MultiWOZ), 100% child safety (SB 243)
- Three published datasets on Zenodo with DOIs; NIST AI RMF alignment documented

Technical expertise spans AI/ML systems, embedding architectures, control theory, and quality systems methodology. Background includes software engineering and systems design.

**Company Structure:** TELOS AI Labs Inc. is a Delaware Public Benefit Corporation with explicit commitment to open research. Dual-entity model separates commercial operations (Labs) from research mission (Consortium), ensuring findings benefit the broader field regardless of commercial outcome.

**Team Gaps and Mitigation:**
- *Regulatory expertise:* Engaging advisors with EU AI Act and FDA QSR experience for Phase I
- *Domain specialists:* Will recruit healthcare and finance advisors for sector-specific validation
- *Scale engineering:* Phase II funding will support engineering hires for production deployment

**Why Fund Now:** The technical foundation is validated. Regulatory deadlines (EU AI Act August 2026, SB 53 January 2026) create immediate market demand. First-mover advantage in mathematical AI governance is time-sensitive. Phase I extends proven conversational AI governance to agentic AI before the compliance window closes.

**Requested:** Phase I funding to validate agentic AI governance extension, positioning for Phase II commercial deployment.

---

## CHARACTER COUNT VERIFICATION

| Section | Limit | Actual | Status |
|---------|-------|--------|--------|
| Technology Innovation | 3,500 | ~3,240 | ✓ |
| Technical Objectives | 3,500 | ~3,400 | ✓ |
| Market Opportunity | 1,750 | ~1,680 | ✓ |
| Company and Team | 1,750 | ~1,734 | ✓ |

---

## NOTES FOR FINAL SUBMISSION

1. **Verify exact character counts** using NSF's submission system (may count differently)
2. **Add specific NSF topic area** alignment if required
3. **Update team section** with any new advisors or partners
4. **Review competitor claims** for accuracy at submission time
5. **Confirm Zenodo DOIs** are publicly accessible

---

*Draft prepared January 2026*
*TELOS AI Labs Inc.*
