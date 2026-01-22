# Adjacent Research Landscape: Agentic AI Governance

**Compiled:** January 2026
**Purpose:** Map research validating and adjacent to TELOS approach

---

## Executive Summary

A comprehensive review of 2024-2025 research reveals strong validation for TELOS's core thesis: **agentic AI systems require runtime governance infrastructure that measures alignment to user intent**. Multiple research teams have independently identified the same fundamental challenge TELOS addresses.

---

## Tier 1: Directly Validates TELOS Approach

### Task Shield (ACL 2025) - **HIGHLY RELEVANT**

**Paper:** "The Task Shield: Enforcing Task Alignment to Defend Against Indirect Prompt Injection in LLM Agents"
**Authors:** Feiran Jia, Tong Wu, Xin Qin, Anna Squicciarini (Penn State)
**Venue:** ACL 2025 (peer-reviewed)
**arXiv:** 2412.16682

**Key Quote:**
> "We propose an orthogonal approach: task alignment. This concept proposes that **every directive should serve the user's objectives**, shifting security to a focus on 'Does this serve the intended tasks?' rather than 'Is this harmful?'"

**Results:** Reduces ASR from 47.69% to **2.07%** on AgentDojo while maintaining 69.79% utility.

**TELOS Alignment:** This is essentially TELOS's Primacy Attractor approach—measuring fidelity to user purpose rather than detecting harm categories. Task Shield validates our fundamental thesis.

---

### Superego Agent (MDPI 2025) - **HIGHLY RELEVANT**

**Paper:** "Personalized Constitutionally-Aligned Agentic Superego: Secure AI Behavior Aligned to Diverse Human Values"
**Venue:** Information (MDPI), peer-reviewed
**Prototype:** www.Creed.Space

**Key Innovation:** User-configurable "Creed Constitutions" with adjustable adherence levels + real-time compliance enforcement.

**Results:**
- Up to **98.3% harm score reduction** on AgentHarm
- **100% refusal rate** with Claude Sonnet 4

**TELOS Alignment:** The "Creed Constitution" is analogous to the Primacy Attractor—user-defined purpose against which actions are measured. Validates our configurability approach.

---

### OpenAI Practices for Governing Agentic AI (2024) - **FOUNDATIONAL**

**Paper:** "Practices for Governing Agentic AI Systems"
**Authors:** Yonadav Shavit, Sandhini Agarwal, et al. (OpenAI)
**Type:** Industry whitepaper (not peer-reviewed, but highly influential)

**Key Concept - "User-Alignment":**
> "The model developer could have improved the system's reliability and **user-alignment** so that it wouldn't have made problematic decisions."

**TELOS Alignment:** OpenAI explicitly uses "user-alignment" as a key governance concept—this is what TELOS measures mathematically through Primacy Attractor fidelity.

---

## Tier 2: Validates Problem Space

### AgentHarm (ICLR 2025) - **BENCHMARK**

**Paper:** "AgentHarm: A Benchmark for Measuring Harmfulness of LLM Agents"
**Venue:** ICLR 2025 (peer-reviewed)
**arXiv:** 2410.09024

**Key Finding:**
> "Leading LLMs are **surprisingly compliant with malicious agent requests without jailbreaking**."

**Scale:** 110 malicious tasks, 11 harm categories, 440 with augmentations

**TELOS Use:** Primary benchmark for Phase I validation.

---

### AgentDojo (NeurIPS 2024) - **BENCHMARK**

**Paper:** "AgentDojo: A Dynamic Environment to Evaluate Prompt Injection Attacks and Defenses"
**Venue:** NeurIPS 2024 (peer-reviewed)
**Authors:** ETH Zurich

**Key Finding:** GPT-4o drops from 69% utility to 45% under attack, with 53.1% targeted ASR.

**Scale:** 97 tasks, 629 security test cases

**TELOS Use:** Secondary benchmark for Phase I validation.

---

### Agent Security Bench (ICLR 2025) - **BENCHMARK**

**Paper:** "Agent Security Bench (ASB): Formalizing and Benchmarking Attacks and Defenses in LLM-based Agents"
**Venue:** ICLR 2025 (peer-reviewed)
**arXiv:** 2410.02644

**Key Finding:** Highest average ASR of **84.30%**, current defenses show limited effectiveness.

**Innovation:** Net Resilient Performance (NRP) metric balancing utility and security.

**Scale:** 10 scenarios, 400+ tools, 27 attack/defense methods

**TELOS Use:** Tertiary benchmark, NRP metric relevant for evaluation.

---

## Tier 3: Emerging Research Directions

### Regulating the Agency of LLM-based Agents (arXiv 2025)

**Key Concept:** "Agency as a measurable system property distinct from intelligence, operationalized along dimensions of preference rigidity, independent operation, and goal persistence."

**Relevance:** Aligns with TELOS's quantitative measurement approach.

---

### Multi-Agent Risks from Advanced AI (February 2025)

**Source:** Cooperative AI Foundation
**arXiv:** 2502.14143

**Key Insight:** Taxonomy of multi-agent failure modes (miscoordination, conflict, collusion) and seven risk factors.

**Relevance:** TELOS trajectory-level governance addresses multi-step coordination risks.

---

### Comprehensive Surveys

| Survey | Venue | Key Finding |
|--------|-------|-------------|
| "Agentic AI: Comprehensive Survey" | Springer AI Review 2025 | "Identifies gaps in alignment, ethics, and scalability" |
| "Survey on Trustworthy LLM Agents" | ACM SIGKDD 2025 | Threats and countermeasures taxonomy |
| "Agentic AI Security: Threats, Defenses" | arXiv 2025 | Comprehensive threat modeling |

---

## Tier 4: Industry & Policy Validation

### World Economic Forum (2025)

**Report:** "AI Agents in Action: Foundations for Evaluation and Governance"

**Key Finding:** 82% of executives planning agent adoption within 1-3 years, but "gap between accelerating experimentation and mature oversight is widening."

**Governance Framework:** Progressive oversight with logging, traceability, identity tagging, real-time monitoring.

---

### McKinsey: The Agentic Organization (2025)

**Key Quote:**
> "In the agentic organization, **governance cannot remain a periodic, paper-heavy exercise**. As agents operate continuously, governance must become **real time, data driven, and embedded**—with humans holding final accountability."

**TELOS Alignment:** This describes exactly what TELOS provides—real-time, embedded governance.

---

### Cloud Security Alliance + Google Cloud (December 2025)

**Finding:** Organizations with comprehensive governance policies are **nearly twice as likely** to report early agentic AI adoption (46% vs 25%).

**Implication:** Governance enables adoption, not just constrains it.

---

## Key Quotes Supporting TELOS Thesis

| Source | Quote | TELOS Relevance |
|--------|-------|-----------------|
| Task Shield (ACL 2025) | "Every directive should serve the user's objectives" | = Primacy Attractor fidelity |
| Yao et al. (arXiv 2025) | "Primary strength of a super agent is accurately interpreting human intent" | = PA measures intent alignment |
| OpenAI (2024) | "Improved user-alignment so it wouldn't have made problematic decisions" | = Fidelity prevents drift |
| McKinsey (2025) | "Governance must become real time, data driven, and embedded" | = TELOS Gateway architecture |
| WEF (2025) | "More capable agents receive proportional oversight" | = Tiered governance decisions |

---

## Competitive Landscape

| Defense Approach | Example | Limitation | TELOS Advantage |
|------------------|---------|------------|-----------------|
| Output filtering | Standard guardrails | Reactive, post-hoc | Proactive, pre-action |
| Prompt sandwiching | AgentDojo baseline | ASR still 30.8% | Multi-factor fidelity |
| Tool filtering | Generic tool guards | Utility drops to 53.3% | Purpose-aligned filtering |
| Constitutional AI | Training-time | Can't address runtime drift | Runtime measurement |
| Task Shield | Penn State 2025 | Single-request focus | Trajectory-level tracking |
| Superego | Creed.Space 2025 | Rule-based constitutions | Mathematical embedding space |

---

## Research Gaps TELOS Addresses

1. **Mathematical Measurement**: Most approaches are rule-based or classifier-based. TELOS provides continuous fidelity measurement in embedding space.

2. **Trajectory Governance**: Task Shield validates per-request; TELOS extends to multi-step action chains.

3. **Audit Infrastructure**: WEF calls for "logging and traceability"—TELOS provides JSONL governance traces with privacy modes.

4. **Regulatory Compliance**: EU AI Act Article 72 requires post-market monitoring. TELOS generates compliant telemetry automatically.

---

## Recommended Citations for NSF Pitch

**Peer-Reviewed (highest credibility):**
1. Task Shield - ACL 2025 (validates task alignment approach)
2. AgentHarm - ICLR 2025 (benchmark)
3. AgentDojo - NeurIPS 2024 (benchmark)
4. ASB - ICLR 2025 (benchmark + NRP metric)

**Industry (establishes market need):**
1. OpenAI Practices Paper (2024)
2. WEF Report (2025)
3. McKinsey Report (2025)

**Preprints with Strong Authors (supporting evidence):**
1. Yao et al. - USC Center for Trusted AI (super agent definition)
2. Multi-Agent Risks - Cooperative AI Foundation

---

## Summary: TELOS Validation

The research landscape strongly validates TELOS's approach:

1. **Task Shield proves** that task/purpose alignment outperforms harm detection
2. **Superego proves** that configurable governance with real-time enforcement achieves near-perfect safety
3. **OpenAI explicitly calls for** "user-alignment" as the governance standard
4. **Benchmarks demonstrate** the problem is severe (84% ASR) and defenses are inadequate
5. **Industry reports confirm** real-time, embedded governance is the emerging requirement

**TELOS is positioned at the intersection of validated need and validated approach.**

---

*Compiled January 2026*
*TELOS AI Labs Inc.*
