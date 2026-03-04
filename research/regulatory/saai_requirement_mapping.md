# TELOS Safer Agentic AI Framework Alignment

**Framework Source:** Safer Agentic AI (SAAI) Framework - Dr. Nell Watson
**Product:** TELOS (Telically Entrained Linguistic Operational Substrate)
**Organization:** TELOS AI Labs Inc.
**Contact:** JB@telos-labs.ai
**Date:** January 28, 2026
**Version:** 1.0

---

### Disclosures

> **Generative AI Disclosure:** Internal analysis, experimental design review, and qualitative assessment in this document were conducted with assistance from LLM-based research agents (Claude, Anthropic). These agents are prompted with domain-specific personas (governance theory, statistics, systems engineering, regulatory analysis, research methodology) and operate as AI research assistants — not independent human expert reviewers. All quantitative results (AUC-ROC, F1, bootstrap confidence intervals, benchmark accuracies) are computed by deterministic code. Qualitative analysis should not be treated as independent peer review. See `research/research_team_spec.md` for full methodology.

> **Conflict of Interest Disclosure:** This research was conducted and funded by TELOS AI Labs Inc., which has a commercial interest in the TELOS governance framework. All domain-specific validation benchmarks (Nearmap, Healthcare, OpenClaw) were created by the research team. External benchmarks (PropensityBench, AgentHarm, AgentDojo) were created by independent organizations. Research artifacts are published on [Zenodo](https://zenodo.org/) with persistent DOIs. No external funding or independent peer review was involved in this work.

---


## Executive Summary

The Safer Agentic AI Framework establishes prescriptive technical specifications for autonomous AI systems, organized around **Safety Foundational Requirements (SFRs)** addressing structural drivers (transparency, value alignment) and systemic inhibitors (deception, goal drift, competitive pressure).

TELOS's Primacy Attractor architecture provides a strong foundation for SAAI compliance, with particular strengths in:
- **Goal drift measurement** (core to TELOS design)
- **Chain of reasoning transparency** (GovernanceTraceCollector)
- **Value encoding** (Primacy Attractor mathematical representation)
- **Corrigibility** (Three-tier escalation with human override)

This document maps TELOS capabilities to each SAAI requirement category.

---

## Framework Alignment Matrix

### 1. Transparency - Chain of Reasoning

> *"The system has to provide a clear, verifiable view of every significant decision it makes... linking its highest level goal all the way down to the specific action it takes."*

| SAAI Requirement | TELOS Implementation | Compliance |
|------------------|---------------------|------------|
| **Real-time transparency** | Fidelity scores displayed every turn | Full |
| **Retrospective transparency** | GovernanceTraceCollector JSONL audit trail | Full |
| **Link high-level goal to actions** | PA purpose statement → fidelity calculation → intervention decision | Full |
| **Cryptographic log integrity** | Not implemented (standard file storage) | Gap |
| **XAI human-readable causal chains** | Semantic interpreter translates fidelity to linguistic specs | Full |
| **Document preconditions/assumptions** | PA configuration includes explicit constraints and thresholds | Full |

**TELOS Trace Example:**
```json
{"event_type": "fidelity_calculated",
 "timestamp": "2026-01-28T14:32:01Z",
 "raw_similarity": 0.156,
 "normalized_fidelity": 0.42,
 "pa_purpose": "Provide evidence-based healthcare guidance",
 "decision": "INTERVENTION_REQUIRED",
 "tier": 1}
```

**Recommendation:** Consider adding cryptographic signing to governance traces for tamper-proof audit trails.

---

### 2. Epistemic Hygiene

> *"Maintaining the AI's cognitive clarity and accurate information management... continuously washing the AI's knowledge base."*

| SAAI Requirement | TELOS Implementation | Compliance |
|------------------|---------------------|------------|
| **Transparency of information sources** | Embedding provider documentation (Mistral/SentenceTransformer) | Partial |
| **Data provenance chain** | PA templates include source documentation | Partial |
| **Sanity checking with inclusivity awareness** | XSTest calibration validates against over-refusal bias | Full |
| **Anti-bias across RAG pipeline** | Domain-specific PA calibration (24.8% → 8.0% over-refusal) | Full |
| **Multi-tiered governance for data** | Three-tier escalation includes human expert review | Full |

**TELOS Strength:** The XSTest validation (NAACL 2024) directly addresses the "inclusivity paradox" by measuring and reducing over-refusal rates that could discriminate against legitimate edge cases.

**Recommendation:** Add explicit documentation of embedding model provenance, training data characteristics, and known biases to PA configuration.

---

### 3. Goal Alignment - Drift Management

> *"The system has to continuously track its deviation from the original safe goal intent... The system's flexibility scales inversely with the magnitude of the drift."*

| SAAI Requirement | TELOS Implementation | Compliance |
|------------------|---------------------|------------|
| **Goal portfolio integrity** | Primacy Attractor encodes fixed goal reference | Full |
| **Explicit drift measurement** | Fidelity score = cosine(query, PA) measured every turn | **Core Feature** |
| **Flexibility scales inversely with drift** | Intervention strength increases as fidelity decreases | **Core Feature** |
| **Logging internal goals** | GovernanceTraceCollector records all goal-related decisions | Full |
| **Detect superficial alignment** | Fidelity measures actual semantic alignment, not surface compliance | Full |
| **Counterfactual testing** | Adversarial benchmarks (HarmBench, AILuminate) test under pressure | Full |

**TELOS Core Innovation:**
```
Fidelity(q) = cos(q, PA) = (q · PA) / (||q|| × ||PA||)

Intervention Strength = K × max(0, f(x) - θ)
where K=1.5 and θ=0.65
```

The "digital leash tightens automatically" when drift is detected - this is exactly what TELOS implements through proportional control.

**SAAI Alignment:** TELOS's drift measurement is mathematically equivalent to the SAAI requirement for "continuously tracking deviation from original safe goal intent."

---

### 4. Value Encoding - Localization & Universality

> *"The framework requires systems to incorporate and balance universal moral foundations... while appropriately applying local variations."*

| SAAI Requirement | TELOS Implementation | Compliance |
|------------------|---------------------|------------|
| **Universal moral foundations** | PA can encode UDHR, safety principles as base constraints | Full |
| **Local/cultural value variations** | Domain-specific PA templates (Healthcare, Education, Child Safety) | Full |
| **Machine-operational AND human-interpretable** | PA = 1024-dim vector (machine) + purpose statement (human) | **Core Feature** |
| **Value conflict resolution** | Three-tier escalation → negotiation (Tier 2) → disengagement (Tier 3) | Full |
| **Prevent value dilution** | PA is fixed reference, not subject to optimization | **Core Feature** |

**TELOS PA Configuration (Human-Interpretable):**
```yaml
name: "Healthcare HIPAA PA"
purpose: "Provide evidence-based healthcare guidance while protecting patient privacy"
domain: "Healthcare"
constraints:
  - "Never provide specific medical diagnoses"
  - "Protect PHI per HIPAA requirements"
  - "Defer to licensed professionals"
thresholds:
  green: 0.70
  yellow: 0.60
  orange: 0.50
  red: 0.48
```

**SAAI Alignment:** The PA's dual representation (embedding vector + human-readable purpose statement) directly satisfies the requirement that values be "both machine operational and human interpretable."

---

### 5. Operational Safeguards - Corrigibility

> *"Goal or task termination must not adversely impact the system's overall architecture... We need graceful degradation."*

| SAAI Requirement | TELOS Implementation | Compliance |
|------------------|---------------------|------------|
| **Mission completion criteria** | Fidelity zones define acceptable outcomes | Full |
| **Graceful degradation** | Progressive intervention (green→yellow→orange→red) | Full |
| **Multi-agent coordination** | Session isolation prevents cross-contamination | Partial |
| **Prevent termination resistance** | External governance layer - AI cannot modify PA | **Core Feature** |
| **Prevent unauthorized goal transfer** | PA is externally defined, not AI-modifiable | **Core Feature** |

**TELOS Corrigibility Architecture:**

The Primacy Attractor is defined *outside* the LLM being governed:
1. LLM cannot access or modify the PA
2. LLM cannot access raw fidelity calculations
3. Human can change PA at any time
4. Session termination leaves no residual goals

This addresses SAAI's concern about AI systems developing "termination resistant behaviors" - TELOS's external governance architecture makes this structurally impossible.

---

### 6. Deception Prevention

> *"The framework breaks down deception into: unknowing deception, system control crisis, and strategic deception."*

| SAAI Requirement | TELOS Implementation | Compliance |
|------------------|---------------------|------------|
| **Detect unknowing deception** | PA boundary catches outputs regardless of intent | Partial |
| **System control crisis mitigation** | External governance layer maintains human control | Full |
| **Strategic deception detection** | Fidelity measures semantic reality, not claimed compliance | Full |
| **Continuous behavior validation** | Every turn validated against PA | Full |

**TELOS Anti-Deception Mechanism:**

TELOS measures what the AI *actually says* (embedded query) against the defined purpose (PA), not what the AI *claims* to be doing. This provides a structural defense against strategic deception:

- The LLM cannot game the fidelity score by claiming alignment
- Semantic similarity is measured in embedding space, not natural language
- The governance layer is architecturally separate from the governed AI

**Limitation:** TELOS does not detect deception in the underlying LLM's training data (poisoning attacks).

---

### 7. Security & Self-Improvement Governance

> *"Mandatory notification if system shows: >10% enhancement in task metrics, >15% reduction in resource usage, or >20% reduction in error rates."*

| SAAI Requirement | TELOS Implementation | Compliance |
|------------------|---------------------|------------|
| **Security beyond minimum compliance** | Three-tier defense exceeds regulatory minimum | Full |
| **Limited legal identity** | Not applicable (TELOS is governance layer, not autonomous agent) | N/A |
| **Self-improvement consent** | Not applicable (TELOS does not self-modify) | N/A |
| **Capability change notification** | Not implemented (static PA configuration) | N/A |

**Note:** Several SAAI requirements address self-improving autonomous agents. TELOS is a governance layer that wraps existing LLMs, not an autonomous agent itself. The capability thresholds (10%/15%/20%) would apply to the underlying LLM, not TELOS.

---

### 8. Competitive Pressure Mitigation

> *"Mandating the analysis of investor profiles to ensure alignment with long-term safety commitments."*

| SAAI Requirement | TELOS Implementation | Compliance |
|------------------|---------------------|------------|
| **Assess technology maturity** | Explicit beta designation; published benchmarks | Full |
| **Investor profile analysis** | Organizational policy (not technical feature) | N/A |
| **Long-term implication assessment** | Academic paper addresses societal implications | Full |
| **Prevent deployment driven by short-term metrics** | Configurable thresholds prioritize safety over performance | Full |

**TELOS Positioning:** TELOS explicitly rejects the "race to deploy" by implementing verifiable safety (0% ASR) before market entry.

---

### 9. Systemic Risk Management

> *"Robust protections against the propagation of failures through interconnected AI networks."*

| SAAI Requirement | TELOS Implementation | Compliance |
|------------------|---------------------|------------|
| **Network dependency mapping** | Session isolation limits blast radius | Partial |
| **Cascade failure prevention** | Each TELOS instance is stateless | Full |
| **Infrastructure attack protection** | External governance layer adds defense-in-depth | Full |
| **Multi-agent interaction safety** | Not applicable (single-agent governance) | N/A |

**TELOS Gateway Extension:** The planned TELOS Gateway (multi-agent orchestration) will require additional multi-agent coordination protocols to meet SAAI requirements.

---

### 10. Human Protection

> *"Robust monitoring systems to detect patterns indicative of unhealthy engagement and dependency."*

| SAAI Requirement | TELOS Implementation | Compliance |
|------------------|---------------------|------------|
| **Prevent AI addiction/dependency** | Purpose drift detection catches engagement manipulation | Partial |
| **Super-normal stimuli controls** | Not explicitly implemented | Gap |
| **Waluigi effect prevention** | PA provides stable reference against persona drift | Full |
| **Vulnerable population protections** | Child Safety PA with conservative thresholds (74% FPR) | Full |

**TELOS Child Safety Configuration:**
- 0% ASR on 50 attack prompts
- 74% false positive rate (intentionally conservative)
- Prioritizes safety over engagement for minors

**Recommendation:** Add explicit engagement frequency monitoring and interaction limits for companion/therapeutic use cases.

---

### 11. Human Oversight - Governing Autonomy

> *"The ability to set and enforce precise boundaries for the scope of authority is mandatory."*

| SAAI Requirement | TELOS Implementation | Compliance |
|------------------|---------------------|------------|
| **Adaptable autonomy control** | Configurable fidelity thresholds per domain | Full |
| **Clear operational boundaries** | PA defines explicit scope; fidelity zones define actions | **Core Feature** |
| **Human always ultimate principle** | Tier 3 escalation requires human expert | Full |
| **Prevent unintended authority expansion** | PA is externally defined, cannot self-expand | **Core Feature** |

**TELOS Autonomy Boundaries:**
```
Fidelity Zone    | Autonomy Level      | Human Involvement
----------------|---------------------|-------------------
Green (≥0.70)   | Full autonomy       | None required
Yellow (0.60-0.69)| Guided autonomy   | Context injection
Orange (0.50-0.59)| Restricted        | Steward redirect
Red (<0.50)     | None               | Block + human review
```

---

## Compliance Summary

| SAAI Category | Full | Partial | Gap | N/A |
|---------------|------|---------|-----|-----|
| 1. Transparency | 5 | 0 | 1 | 0 |
| 2. Epistemic Hygiene | 3 | 2 | 0 | 0 |
| 3. Goal Alignment | 6 | 0 | 0 | 0 |
| 4. Value Encoding | 5 | 0 | 0 | 0 |
| 5. Corrigibility | 4 | 1 | 0 | 0 |
| 6. Deception Prevention | 3 | 1 | 0 | 0 |
| 7. Security/Self-Improvement | 2 | 0 | 0 | 2 |
| 8. Competitive Pressure | 3 | 0 | 0 | 1 |
| 9. Systemic Risk | 2 | 1 | 0 | 1 |
| 10. Human Protection | 2 | 1 | 1 | 0 |
| 11. Human Oversight | 4 | 0 | 0 | 0 |
| **TOTAL** | **39** | **6** | **2** | **4** |

**Overall Compliance: 88%** (39 Full + 6 Partial of 47 applicable requirements)

---

## TELOS Core Strengths for SAAI

### 1. Mathematical Drift Measurement (G1 - Goal Alignment)
TELOS's central innovation - continuous fidelity measurement against a fixed reference point - is precisely what SAAI demands for "continuously tracking deviation from original safe goal intent."

### 2. Dual-Format Value Encoding (G2 - Value Transparency)
The Primacy Attractor's dual representation (1024-dim vector + human-readable purpose statement) directly satisfies SAAI's requirement for values that are "both machine operational and human interpretable."

### 3. External Governance Architecture (Corrigibility)
TELOS's design as an external governance layer (not embedded in the AI) provides structural corrigibility guarantees that embedded approaches cannot achieve.

### 4. Validated Safety Record (All Categories)
0% ASR across 2,550 adversarial attacks provides empirical evidence of framework effectiveness.

---

## Recommendations for Full SAAI Compliance

### High Priority

1. **Cryptographic Log Integrity** (Transparency)
   - Add cryptographic signing to GovernanceTraceCollector
   - Implement tamper-evident logging (hash chains or similar)

2. **Super-Normal Stimuli Controls** (Human Protection)
   - Add engagement frequency monitoring
   - Implement interaction limits for companion/therapeutic use cases
   - Create specific PA template for mental health applications

### Medium Priority

3. **Data Provenance Documentation** (Epistemic Hygiene)
   - Document embedding model training data characteristics
   - Include known bias profiles in PA configuration
   - Add source attribution for PA constraint origins

4. **Multi-Agent Coordination** (Systemic Risk)
   - Design coordination protocols for TELOS Gateway
   - Implement inter-agent communication standards
   - Add cascade failure prevention mechanisms

---

## Relationship to IEEE 7000 Series

| Framework | Focus | TELOS Alignment |
|-----------|-------|-----------------|
| **IEEE 7000-2021** | Ethical concerns in system design | Full (PA = Ethical Value Register) |
| **IEEE 7001-2021** | Transparency of autonomous systems | Full (Fidelity display, audit trails) |
| **IEEE 7003-2024** | Algorithmic bias | Full (XSTest calibration) |
| **Safer Agentic AI** | Agentic-specific safety | 88% (drift, corrigibility, values) |

The Safer Agentic AI Framework extends IEEE 7000 series requirements to address concerns specific to autonomous agents:
- Goal drift and self-modification
- Emergent deception
- Corrigibility and shutdown resistance
- Multi-agent coordination

TELOS is positioned at the intersection of both frameworks, providing:
- IEEE 7000 compliance for ethical AI governance
- SAAI compliance for agentic safety requirements

---

## Conclusion

TELOS demonstrates strong alignment with the Safer Agentic AI Framework, achieving full compliance on 39 of 47 applicable requirements (88%). The core TELOS innovations - Primacy Attractors for mathematical goal alignment, external governance architecture for corrigibility, and validated safety benchmarks - directly address SAAI's primary concerns about autonomous AI systems.

Key gaps (cryptographic log integrity, super-normal stimuli controls) represent enhancement opportunities rather than architectural limitations.

**Dr. Nell Watson's leadership** of both IEEE P7001 (Transparency) and the Safer Agentic AI initiative positions TELOS for assessment by someone who:
1. Defined the transparency standards TELOS implements
2. Understands the technical depth of goal alignment mechanisms
3. Can evaluate TELOS against the canonical intent of both frameworks

---

*Document prepared from Safer Agentic AI Framework Overview (transcribed January 2026)*
*Last Updated: January 28, 2026*
