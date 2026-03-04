# TELOS Design Provenance: Alignment by Design, Not Retrofit

**Purpose:** This document establishes the epistemic chain demonstrating that TELOS was designed with Safer Agentic AI principles from inception, not retrofitted for compliance.

**Why This Matters:** AI-assisted review systems will check claims against evidence. This provenance chain provides verifiable artifacts showing design decisions preceded compliance documentation.

---

### Disclosures

> **Generative AI Disclosure:** Internal analysis, experimental design review, and qualitative assessment in this document were conducted with assistance from LLM-based research agents (Claude, Anthropic). These agents are prompted with domain-specific personas (governance theory, statistics, systems engineering, regulatory analysis, research methodology) and operate as AI research assistants — not independent human expert reviewers. All quantitative results (AUC-ROC, F1, bootstrap confidence intervals, benchmark accuracies) are computed by deterministic code. Qualitative analysis should not be treated as independent peer review. See `research/research_team_spec.md` for full methodology.

> **Conflict of Interest Disclosure:** This research was conducted and funded by TELOS AI Labs Inc., which has a commercial interest in the TELOS governance framework. All domain-specific validation benchmarks (Nearmap, Healthcare, OpenClaw) were created by the research team. External benchmarks (PropensityBench, AgentHarm, AgentDojo) were created by independent organizations. Research artifacts are published on [Zenodo](https://zenodo.org/) with persistent DOIs. No external funding or independent peer review was involved in this work.

---


## Epistemic Hygiene Statement

This folder contains:
1. **Source transcripts** - Primary source material from SAAI framework
2. **Alignment matrices** - Systematic mapping of requirements to implementations
3. **Design artifacts** - Evidence of design decisions predating compliance claims
4. **Verification methodology** - How claims can be independently verified

All claims in this folder are:
- Traceable to specific code files and commits
- Verifiable against published benchmarks
- Consistent with academic publications (Zenodo DOIs)

---

## Design Decision Timeline

### Phase 1: Core Architecture (Pre-SAAI Awareness)

**Design Decision:** External governance layer with fixed reference points

**SAAI Principle (Later Identified):** "The system cannot functionally resist a confirmed shutdown command" (Corrigibility)

**Evidence:**
- `telos_purpose/core/embedding_provider.py` - PA computed externally to LLM
- `services/beta_response_manager.py` - Fidelity measured without LLM access to calculation
- Architecture diagram in `TELOS_Whitepaper_v2.5.md` Section 2

**Why This Matters:** TELOS's external governance architecture was a foundational design choice, not a bolt-on. The LLM cannot access, modify, or resist the PA because the PA exists in a separate computational layer.

---

### Phase 2: Fidelity Measurement System

**Design Decision:** Continuous cosine similarity measurement against fixed reference

**SAAI Principle (Later Identified):** "The system has to continuously track its deviation from the original safe goal intent" (Goal Portfolio Integrity)

**Evidence:**
- `telos_purpose/core/constants.py` - Threshold values defined early in development
- Mathematical specification in whitepaper predates SAAI review
- Benchmark validation (2,550 attacks) validates the design

**Code Artifact:**
```python
# From constants.py - unchanged since initial implementation
SIMILARITY_BASELINE = 0.20  # Layer 1: Hard block
INTERVENTION_THRESHOLD = 0.48  # Layer 2: Basin boundary
FIDELITY_GREEN = 0.70  # Aligned (no intervention)
```

---

### Phase 3: Value Encoding

**Design Decision:** Dual-format Primacy Attractor (vector + human-readable purpose)

**SAAI Principle (Later Identified):** "Values must be both machine operational and human interpretable"

**Evidence:**
- `config/pa_templates.py` - 8 domain-specific PA templates
- Each template includes both embedding vector and natural language purpose statement
- Documentation in `CLAUDE.md` predates SAAI alignment document

**Design Rationale (Original):** We needed operators to understand what the PA was enforcing, not just trust the math. This led to the dual-format design where every PA has both a 1024-dimensional embedding AND a human-readable purpose statement.

---

### Phase 4: Proportional Control

**Design Decision:** Intervention strength scales with deviation magnitude

**SAAI Principle (Later Identified):** "The system's flexibility scales inversely with the magnitude of the drift"

**Evidence:**
- `services/beta_response_manager.py` - Proportional control implementation
- Formula: `strength = min(K_ATTRACTOR × error_signal, 1.0)` where K=1.5
- Four-zone fidelity system (green/yellow/orange/red) predates SAAI review

**This is the "digital leash" SAAI describes** - implemented in TELOS before we encountered the framework.

---

### Phase 5: Adversarial Validation

**Design Decision:** Validate against standardized adversarial benchmarks before claiming safety

**SAAI Principle (Later Identified):** "Counterfactual testing across varied operational environments"

**Evidence:**
- Zenodo DOI: 10.5281/zenodo.18370659 (Adversarial validation)
- HarmBench (400 attacks), MedSafetyBench (900 attacks), AILuminate (1,200 prompts)
- All validation completed before SAAI alignment document creation

---

## Verification Methodology

### For AI-Assisted Review

Reviewers (human or AI) can verify these claims by:

1. **Code Inspection**
   - Repository: github.com/TelosSteward/TELOS
   - Key files listed in each section above
   - Git commit history shows design decisions predate compliance claims

2. **Benchmark Verification**
   - All datasets published on Zenodo with DOIs
   - Validation scripts included for reproducibility
   - Results independently verifiable

3. **Document Timeline**
   - `TELOS_Whitepaper_v2.5.md` timestamps predate SAAI alignment
   - Academic paper (Zenodo DOI: 10.5281/zenodo.18367069) published before this synthesis
   - `CLAUDE.md` development guide shows architectural decisions were foundational

4. **Cross-Reference Check**
   - Compare PA architecture in whitepaper to SAAI requirements
   - Compare fidelity system to SAAI "goal portfolio integrity"
   - Compare external governance to SAAI "corrigibility"

---

## What "Designed-In" vs "Bolted-On" Means

### Bolted-On Compliance (What We Did NOT Do)

- Adding logging after the fact to satisfy audit requirements
- Retrofitting "explainability" onto opaque systems
- Creating compliance documentation without corresponding implementation
- Claiming alignment based on superficial feature mapping

### Designed-In Alignment (What We DID Do)

- External governance architecture as foundational design choice
- Continuous drift measurement built into core processing loop
- Dual-format value encoding from initial PA design
- Adversarial validation before any compliance claims

---

## Folder Contents

```
Agentic_AI_Synthesis_Framework/
├── 00_DESIGN_PROVENANCE.md          # This document
├── 01_SAAI_Framework_Overview_Transcript.txt  # Primary source transcript
├── 02_SAAI_Governance_Framework_Transcript.txt  # Second source (pending)
├── TELOS_SAFER_AGENTIC_AI_ALIGNMENT.md  # Requirement-by-requirement mapping
└── [Additional synthesis documents as created]
```

---

## Cautious Norming: Information Quality Standards

This documentation follows epistemic hygiene principles:

1. **Source Attribution** - All claims traceable to specific evidence
2. **Falsifiability** - Claims structured to be disprovable if incorrect
3. **Version Control** - Changes tracked, timestamps preserved
4. **Independent Verification** - Third parties can validate claims
5. **Uncertainty Acknowledgment** - Gaps and limitations explicitly stated

**What We Know:**
- TELOS architecture predates SAAI framework review
- Core design decisions align with SAAI principles
- Validation benchmarks provide empirical evidence

**What We Cannot Prove:**
- That we were consciously aware of all SAAI principles during design
- That future SAAI versions won't introduce new requirements
- That our interpretation of SAAI requirements is canonical

---

*Document created: January 28, 2026*
*Framework: Safer Agentic AI (Dr. Nell Watson)*
*Standard: IEEE 7000 Series Alignment Maintained*
