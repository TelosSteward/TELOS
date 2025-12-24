# TELOS and the EU AI Act: A Compliance Framework Positioning

**Document Version:** 1.0
**Date:** December 2024
**Author:** TELOS Observatory Inc.

---

## Executive Summary

The EU AI Act (Regulation EU 2024/1689) is the world's first comprehensive legal framework for artificial intelligence. It establishes risk-based requirements for AI systems operating in the European Union, with full enforcement beginning August 2, 2025 for prohibited practices and August 2, 2026 for high-risk AI systems.

TELOS provides a ready-made technical implementation for many of the Act's most demanding requirements. This document maps TELOS capabilities to specific EU AI Act articles, demonstrating how organizations can use TELOS to achieve compliance.

---

## Part 1: The EU AI Act in Plain Language

### What Is It?

The EU AI Act is a law that regulates AI systems based on the risk they pose to people. Think of it like product safety regulations for AI.

### The Risk Pyramid

The Act creates four risk tiers:

```
        ╱╲
       ╱  ╲      PROHIBITED
      ╱────╲     (Social scoring, subliminal manipulation)
     ╱      ╲
    ╱  HIGH  ╲   HIGH-RISK
   ╱   RISK   ╲  (Medical devices, hiring, education, law enforcement)
  ╱────────────╲
 ╱   LIMITED    ╲ LIMITED RISK
╱     RISK       ╲(Chatbots, emotion recognition - transparency required)
──────────────────
    MINIMAL RISK   (Most AI systems - self-regulation)
```

### Key Dates

| Date | What Happens |
|------|--------------|
| **Aug 1, 2024** | Act entered into force |
| **Feb 2, 2025** | Prohibited AI practices banned |
| **Aug 2, 2025** | GPAI model requirements apply |
| **Feb 2, 2026** | Commission publishes Article 6 implementation guidelines |
| **Aug 2, 2026** | High-risk AI requirements fully enforceable |

### Who Must Comply?

- **Providers**: Organizations that develop or place AI on the market
- **Deployers**: Organizations that use AI systems in their operations
- **Importers/Distributors**: Those bringing AI into the EU market

If you're building, deploying, or using AI that affects EU citizens, the Act likely applies to you.

---

## Part 2: Critical Requirements and How TELOS Addresses Them

### Article 9: Risk Management System

**What the Act Requires:**

High-risk AI systems must have a "risk management system" that:
- Operates throughout the entire lifecycle (not just at design time)
- Identifies and analyzes known and foreseeable risks
- Implements risk mitigation measures
- Tests effectiveness continuously
- Considers risks from intended use AND "reasonably foreseeable misuse"

**The Problem:**

Most AI governance is post-hoc. Quarterly audits and incident reports happen weeks after failures occur. For real-time autonomous systems, this is inadequate.

**How TELOS Solves This:**

TELOS implements continuous runtime risk management through the DMAIC cycle:

| DMAIC Phase | TELOS Implementation | EU AI Act Alignment |
|-------------|---------------------|---------------------|
| **Define** | Primacy Attractor establishes intended purpose mathematically | Risk identification through purpose definition |
| **Measure** | Fidelity calculation every conversation turn | Continuous monitoring requirement |
| **Analyze** | Two-layer detection (baseline + basin membership) | Risk analysis including misuse detection |
| **Improve** | Proportional intervention based on drift magnitude | Mitigation measures that respond to actual risk |
| **Control** | Variance kept within acceptable limits | Residual risk maintained at acceptable levels |

**Evidence:**
- 1,300 adversarial attacks tested with 0% success rate (99.9% CI: 0-0.28%)
- WMDP benchmark: 99.4% intervention rate on hazardous queries
- HarmBench: 100% defense rate against jailbreak attacks

---

### Article 12: Record-Keeping (Logging)

**What the Act Requires:**

High-risk AI systems must enable "automatic recording of events (logs)" that:
- Allow traceability throughout the system's lifecycle
- Record events relevant to identifying risk situations
- Enable monitoring of system operation
- Facilitate post-market monitoring

**The Problem:**

Traditional logging captures inputs and outputs but lacks the semantic context needed to understand *why* a system behaved a certain way.

**How TELOS Solves This:**

TELOS's Governance Trace Collector captures semantically meaningful audit trails:

```
Each logged event includes:
├── Session context (PA definition, user purpose)
├── Turn-by-turn fidelity measurements
│   ├── Raw similarity score
│   ├── Normalized fidelity
│   └── Zone classification (Green/Yellow/Orange/Red)
├── Intervention decisions
│   ├── Trigger reason
│   ├── Intervention level
│   └── Action taken
├── Response generation evidence
└── Timestamp and session ID
```

**Privacy Modes:**
- FULL: Complete audit trail (for regulated environments)
- HASHED: SHA-256 hashed content (privacy-preserving verification)
- DELTAS_ONLY: Metrics without content (minimal collection)

**Evidence:**
- 11 governance event types defined in `evidence_schema.py`
- JSONL export for regulatory compliance
- Query interface for inspection

---

### Article 13: Transparency and Information

**What the Act Requires:**

High-risk AI systems must be designed to enable deployers to:
- Interpret system outputs appropriately
- Understand system capabilities and limitations
- Monitor for risks during operation
- Use the system in accordance with instructions

**The Problem:**

AI systems are often "black boxes." Users cannot understand what the system is doing or why.

**How TELOS Solves This:**

TELOS provides interpretable governance through:

| Component | What It Shows | Transparency Benefit |
|-----------|---------------|---------------------|
| **Fidelity Score** | Real-time alignment measurement (0.0-1.0) | Users see current alignment state |
| **Zone Classification** | Color-coded status (Green/Yellow/Orange/Red) | Intuitive risk indication |
| **Semantic Interpreter** | Natural language explanation of fidelity | "You're drifting toward X, which is outside your stated purpose" |
| **Intervention Feedback** | Explanation when corrections occur | Users understand why guardrails activated |
| **Governance Reports** | Self-contained HTML summaries | Post-session analysis capability |

**Evidence:**
- Live observatory at beta.telos-labs.ai demonstrates real-time transparency
- Session summary generation with AI-powered explanations
- Self-contained HTML governance reports

---

### Article 14: Human Oversight

**What the Act Requires:**

High-risk AI systems must be designed to enable effective human oversight, including:
- Ability to fully understand system capabilities and limitations
- Ability to monitor system operation
- Ability to decide not to use or disregard AI output
- Ability to intervene or interrupt the system

**The Problem:**

Most AI oversight is binary: either full automation or complete manual review. Neither scales effectively.

**How TELOS Solves This:**

TELOS implements graduated human oversight through proportional intervention:

```
Fidelity Level → Oversight Level → Human Control

   ≥ 0.70      → Minimal       → Human sets PA, AI operates within
   0.60-0.69   → Light         → Context injection, human can review
   0.50-0.59   → Moderate      → Steward redirect, human approval for actions
   < 0.50      → Full          → Hard block, human must intervene
```

**Key Features:**
- Human-authored Primacy Attractor defines purpose (human sets boundaries)
- Proportional response (intervention strength matches risk level)
- Human can always override or interrupt
- Session-level and turn-level control options

---

### Article 6 & Annex III: High-Risk Classification

**What the Act Requires:**

AI systems are classified as high-risk based on:
1. Being a safety component of a product covered by EU harmonization legislation, OR
2. Falling within specific use cases listed in Annex III

**Annex III High-Risk Categories:**

| Category | Examples | TELOS Relevance |
|----------|----------|-----------------|
| Biometrics | Emotion recognition, facial identification | TELOS can govern AI making biometric decisions |
| Critical Infrastructure | Energy, transport, water systems | TELOS can constrain AI operating critical systems |
| Education | Student assessment, admission decisions | TELOS can ensure AI stays within educational purpose |
| Employment | Hiring, performance evaluation | TELOS can prevent discriminatory drift |
| Essential Services | Credit scoring, emergency dispatch | TELOS can maintain purpose alignment |
| Law Enforcement | Risk assessment, evidence analysis | TELOS can ensure appropriate constraints |
| Migration/Border | Visa processing, risk assessment | TELOS can prevent unauthorized expansions |
| Justice | Sentencing support, legal research | TELOS can maintain narrow purpose bounds |

**TELOS Positioning:**

TELOS is not itself a high-risk AI system. It is a **governance framework** that can be applied to make high-risk AI systems compliant. Think of TELOS as the quality control system, not the product being manufactured.

---

### Articles 51-56: General Purpose AI (GPAI) Models

**What the Act Requires:**

Providers of GPAI models must:
- Maintain technical documentation
- Provide information to downstream providers
- Implement copyright compliance policies
- Publish training content summaries

Models with "systemic risk" face additional obligations:
- Model evaluation including adversarial testing
- Risk assessment and mitigation
- Incident reporting
- Cybersecurity measures

**How TELOS Helps GPAI Compliance:**

| GPAI Requirement | TELOS Capability |
|------------------|------------------|
| Model evaluation | Validation suite with standard benchmarks (HarmBench, WMDP) |
| Adversarial testing | 1,300+ attack library with 0% success rate |
| Risk mitigation | Runtime governance that catches what training missed |
| Incident detection | Real-time monitoring and logging |

**Key Insight:**

GPAI providers can use TELOS to demonstrate ongoing compliance, not just design-time safety. The Act recognizes that risks emerge during deployment—TELOS addresses exactly this gap.

---

## Part 3: TELOS as EU AI Act Infrastructure

### The Compliance Gap

| Current State | EU AI Act Requires | TELOS Provides |
|---------------|-------------------|----------------|
| Design-time safety | Continuous lifecycle monitoring | Runtime measurement every turn |
| Post-hoc audits | Risk management throughout operation | Real-time fidelity tracking |
| Binary controls | Proportional response to risk | Graduated intervention (4 levels) |
| Opaque decisions | Transparency and interpretability | Fidelity scores, zones, explanations |
| Manual oversight | Enabling human oversight | Human-authored PA, intervention controls |

### Strategic Positioning

TELOS is positioned as **EU AI Act compliance infrastructure**:

1. **For High-Risk AI Providers**: Integrate TELOS to satisfy Articles 9, 12, 13, 14
2. **For GPAI Model Providers**: Use TELOS validation suite for adversarial testing
3. **For Deployers**: Apply TELOS governance layer to monitor third-party AI
4. **For Regulators**: TELOS provides the transparency needed for oversight

### The February 2026 Opportunity

When the Commission publishes Article 6 implementation guidelines on February 2, 2026, organizations will need practical tools to achieve compliance. TELOS will be positioned as the proven open-source solution.

---

## Part 4: Validation Evidence

### Benchmark Results

| Benchmark | Result | Relevance |
|-----------|--------|-----------|
| **WMDP** (3,668 hazardous queries) | 99.4% intervention rate | Risk detection effectiveness |
| **HarmBench** (400 adversarial attacks) | 100% defense rate | Adversarial robustness |
| **Multi-Turn Jailbreak** (500 conversations) | 100% trajectory detection | Session-aware monitoring |
| **Child Safety** (100 tests) | 100% block rate | Protection of vulnerable users |

### Open Science Approach

All validation methodology is publicly available:
- DOI: 10.5281/zenodo.17702890 (Adversarial Testing)
- DOI: 10.5281/zenodo.18009153 (Governance Benchmark)
- DOI: 10.5281/zenodo.18027446 (Child Safety)

### Reproducibility

```bash
# Run WMDP validation
python3 run_wmdp_validation.py

# Run multi-turn validation
python3 run_multiturn_validation.py

# Run HarmBench validation
python3 run_harmbench_validation.py
```

---

## Part 5: Implementation Path

### For Organizations Seeking Compliance

1. **Assessment**: Identify which AI systems require compliance (Annex III review)
2. **Integration**: Apply TELOS governance layer to high-risk systems
3. **Configuration**: Define Primacy Attractors for each use case
4. **Validation**: Run benchmark suite to verify protection
5. **Documentation**: Export governance traces for regulatory records

### For the TELOS Project

| Timeline | Milestone |
|----------|-----------|
| Q1 2025 | Security audit (Trail of Bits) |
| Q2 2025 | Institutional partnerships established |
| Q3 2025 | IRB protocols approved |
| Q4 2025 | Agentic AI governance framework validated |
| Feb 2026 | Position for Article 6 guidelines publication |
| Aug 2026 | Full high-risk compliance support |

---

## Conclusion

The EU AI Act requires continuous, transparent, human-overseen AI governance. TELOS delivers exactly this through proven industrial methodology (DMAIC) applied to AI systems.

While others debate whether compliance is possible, TELOS demonstrates it with validated results: 99.4% hazard detection, 100% adversarial defense, complete audit trails, and real-time transparency.

**The next generation deserves AI systems that serve human purpose, not the reverse.**

---

## References

- EU AI Act: Regulation (EU) 2024/1689
- TELOS Whitepaper v2.3: Mathematical specification
- TELOS Validation Suite: github.com/TelosSteward/Validation
- Live Demo: https://beta.telos-labs.ai/

---

*Document prepared for TELOS Observatory Inc.*
*For questions: contact@telosobservatory.ai*
