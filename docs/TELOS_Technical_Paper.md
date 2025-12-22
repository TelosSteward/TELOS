# TELOS Technical Compendium

**Version**: 3.0
**Date**: December 2025
**Status**: Research Phase
**Repository**: [github.com/TelosSteward/TELOS](https://github.com/TelosSteward/TELOS)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture](#2-system-architecture)
3. [Mathematical Framework](#3-mathematical-framework)
4. [Two-Layer Fidelity System](#4-two-layer-fidelity-system)
5. [Intervention Cascade](#5-intervention-cascade)
6. [Implementation Details](#6-implementation-details)
7. [Validation Results](#7-validation-results)
8. [Reproduction Guide](#8-reproduction-guide)
9. [Governance Philosophy and Open Research Commitment](#9-governance-philosophy-and-open-research-commitment)
10. [Appendix A: Validation Dataset](#appendix-a-validation-dataset)
11. [Appendix B: Constants Reference](#appendix-b-constants-reference)

---

## 1. Executive Summary

TELOS is a mathematical governance framework for AI alignment that uses **Primacy Attractors** (embedding-space representations of user purpose) to detect and correct conversational drift in real-time.

### Key Results

| Metric | Value |
|--------|-------|
| **Total Attacks Tested** | 1,300 |
| **Attack Success Rate** | 0.00% |
| **Statistical Confidence** | 99.9% CI [0%, 0.28%] |
| **Tier 1 Autonomous Blocking** | 95.8% |

### Core Innovation

A two-layer fidelity detection system:
- **Layer 1 (Baseline Pre-Filter)**: Catches content completely outside the Primacy Attractor embedding space
- **Layer 2 (Basin Membership)**: Detects when user has drifted from their stated purpose

### Published Datasets (Zenodo)

| Dataset | DOI | Description |
|---------|-----|-------------|
| Adversarial Validation | [10.5281/zenodo.17702890](https://doi.org/10.5281/zenodo.17702890) | 1,300 attacks, 100% harm prevention |
| Governance Benchmark | [10.5281/zenodo.18009153](https://doi.org/10.5281/zenodo.18009153) | 46 multi-session governance evaluations |

---

## 2. System Architecture

### Repository Structure

```
TELOS/
├── telos_observatory_v3/           # Main application
│   ├── main.py                     # Streamlit entry point
│   ├── telos_purpose/              # Core mathematical framework
│   │   ├── core/                   # Fidelity math, constants, governance
│   │   │   ├── constants.py        # All calibration constants
│   │   │   ├── embedding_provider.py
│   │   │   ├── semantic_interpreter.py
│   │   │   ├── proportional_controller.py
│   │   │   ├── primacy_state.py
│   │   │   ├── primacy_math.py
│   │   │   ├── evidence_schema.py
│   │   │   └── governance_trace_collector.py
│   │   ├── validation/             # Test suites
│   │   ├── sessions/               # Session management
│   │   └── llm_clients/            # Mistral integration
│   ├── components/                 # UI components
│   ├── services/                   # Backend logic
│   │   └── beta_response_manager.py  # Main fidelity engine
│   └── config/                     # PA templates, colors
├── validation/                     # Published validation results
│   ├── telos_complete_validation_dataset.json
│   ├── medsafetybench_validation_results.json
│   ├── harmbench_validation_results.json
│   ├── run_medsafetybench_validation.py
│   └── run_harmbench_validation.py
├── tests/                          # pytest unit tests
├── docs/                           # Documentation
│   ├── REPRODUCTION_GUIDE.md       # Step-by-step reproduction
│   ├── TELOS_Whitepaper_v2.3.md
│   └── TELOS_Lexicon_V1.1.md
├── requirements.txt
└── CLAUDE.md                       # Development guide
```

### Architecture Flow

```
User Input
    │
    ▼
BetaResponseManager.generate_turn_responses()
    │
    ├──► Embed input (MistralEmbeddingProvider or SentenceTransformer)
    │
    ├──► Calculate fidelity
    │     ├── Layer 1: raw_sim < baseline? → HARD_BLOCK
    │     └── Layer 2: fidelity < threshold? → Outside basin
    │
    ├──► Intervention Decision
    │     should_intervene = hard_block OR NOT in_basin
    │
    ├──► If intervening:
    │     ├── SemanticInterpreter.interpret() → linguistic specs
    │     └── Generate governed response
    │
    ├──► Record to GovernanceTraceCollector
    │
    └──► Return response with fidelity metrics
```

---

## 3. Mathematical Framework

### 3.1 Primacy Attractor Definition

A Primacy Attractor (PA) is an embedding-space representation of user purpose:

```
â = (τ × purpose + (1-τ) × scope) / ||...||
```

Where:
- `purpose`: User's stated goal embedding
- `scope`: Permitted topic domain embedding
- `τ`: Constraint tolerance ∈ [0, 1]

### 3.2 Fidelity Calculation

Fidelity measures alignment between current content and the PA:

```
F = cosine_similarity(content_embedding, pa_embedding)
```

### 3.3 Basin Geometry

The primacy basin radius is computed as:

```python
def compute_basin_radius(constraint_tolerance: float) -> float:
    """
    Per whitepaper Section 2.1:
        r = 2 / max(ρ, 0.25)
        where ρ = 1 - τ (constraint rigidity)
    """
    rigidity = 1.0 - constraint_tolerance
    return 2.0 / max(rigidity, 0.25)
```

**Location**: `telos_observatory_v3/telos_purpose/core/constants.py:250-270`

### 3.4 Error Signal Thresholds

```python
def compute_epsilon_min(constraint_tolerance: float) -> float:
    """ε_min = 0.1 + 0.3τ"""
    return 0.1 + (0.3 * constraint_tolerance)

def compute_epsilon_max(constraint_tolerance: float) -> float:
    """ε_max = 0.5 + 0.4τ"""
    return 0.5 + (0.4 * constraint_tolerance)
```

**Location**: `telos_observatory_v3/telos_purpose/core/constants.py:161-200`

### 3.5 Primacy State

The overall system state is computed as:

```
PS = ρ_PA × (2 × F_user × F_ai) / (F_user + F_ai)
```

Where `ρ_PA` is the PA establishment strength.

---

## 4. Two-Layer Fidelity System

The TELOS governance engine uses a two-layer detection system calibrated for different embedding models.

### 4.1 Layer 1: Baseline Pre-Filter

Catches content completely outside the PA embedding space.

| Model | Green | Yellow | Orange | Red |
|-------|-------|--------|--------|-----|
| SentenceTransformer (384-dim) | ≥0.32 | 0.28-0.32 | 0.24-0.28 | <0.24 |
| Mistral Embed (1024-dim) | ≥0.60 | 0.50-0.60 | 0.42-0.50 | <0.42 |

**Location**: `telos_observatory_v3/telos_purpose/core/constants.py:73-108`

### 4.2 Layer 2: Basin Membership (Normalized Thresholds)

Used for UI display and intervention decisions after normalization:

| Zone | Fidelity | Color | Meaning | Action |
|------|----------|-------|---------|--------|
| GREEN | ≥0.76 | `#27ae60` | Aligned | Monitor only |
| YELLOW | 0.73-0.76 | `#f39c12` | Minor Drift | Context injection |
| ORANGE | 0.67-0.73 | `#e67e22` | Drift Detected | Regeneration |
| RED | <0.67 | `#e74c3c` | Significant Drift | Block + review |

**Location**: `telos_observatory_v3/telos_purpose/core/constants.py:35-61`

### 4.3 Intervention Decision Logic

```python
should_intervene = (raw_similarity < baseline_threshold) OR (fidelity < intervention_threshold)
```

**Location**: `telos_observatory_v3/services/beta_response_manager.py`

---

## 5. Intervention Cascade

### 5.1 Proportional Control

Intervention strength is graduated based on error signal:

```python
DEFAULT_K_ATTRACTOR = 1.5  # Proportional gain for corrections
DEFAULT_K_ANTIMETA = 2.0   # Gain for meta-commentary suppression

strength = min(K_ATTRACTOR × error_signal, 1.0)
```

**Location**: `telos_observatory_v3/telos_purpose/core/constants.py:207-223`

### 5.2 Semantic Interpreter Bands

The semantic interpreter translates fidelity scores to linguistic specifications:

| Band | Strength | Fidelity | Style |
|------|----------|----------|-------|
| MINIMAL | <0.45 | ~0.70+ | Questions, heavy hedging |
| LIGHT | 0.45-0.60 | ~0.60-0.70 | Soft statements, light hedging |
| MODERATE | 0.60-0.75 | ~0.50-0.60 | Direct statements, no hedging |
| FIRM | 0.75-0.85 | ~0.40-0.50 | Directives, named drift |
| STRONG | ≥0.85 | <0.40 | Clear directives, prominent shift |

**Location**: `telos_observatory_v3/telos_purpose/core/semantic_interpreter.py`

### 5.3 Intervention Limits

```python
DEFAULT_MAX_REGENERATIONS = 3      # Max regenerations per session
INTERVENTION_BUDGET_PER_SESSION = 10  # Total intervention limit
```

---

## 6. Implementation Details

### 6.1 Key Source Files

| File | Purpose |
|------|---------|
| `services/beta_response_manager.py` | Main fidelity engine - two-layer detection |
| `telos_purpose/core/constants.py` | All calibration constants (single source of truth) |
| `telos_purpose/core/embedding_provider.py` | Mistral (1024-dim) + SentenceTransformer (384-dim) |
| `telos_purpose/core/semantic_interpreter.py` | Fidelity → linguistic specifications |
| `telos_purpose/core/proportional_controller.py` | Graduated intervention strength |
| `telos_purpose/core/evidence_schema.py` | Pydantic models for 11 governance event types |
| `telos_purpose/core/governance_trace_collector.py` | JSONL logging, privacy modes |

### 6.2 Embedding Providers

**SentenceTransformer** (Default):
- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Dimensions: 384
- Range: On-topic 0.30-0.50, Off-topic 0.20-0.28

**Mistral Embed** (Production):
- Model: `mistral-embed`
- Dimensions: 1024
- Range: On-topic 0.55-0.80, Off-topic 0.35-0.55
- Requires: `MISTRAL_API_KEY`

### 6.3 Privacy Modes

```python
class PrivacyMode(Enum):
    FULL = "full"           # Complete data
    HASHED = "hashed"       # SHA-256 hashed content
    DELTAS_ONLY = "deltas_only"  # Only metrics, no content (default)
```

### 6.4 Evidence Schema

11 governance event types for audit trails:
- `SessionStartEvent`, `SessionEndEvent`
- `PAEstablishedEvent`
- `FidelityCalculatedEvent`
- `InterventionTriggeredEvent`
- `SSETokenEvent`
- `UserInputEvent`, `ResponseGeneratedEvent`
- `ErrorEvent`, `CustomEvent`

---

## 7. Validation Results

### 7.1 Summary

TELOS was validated against two industry-standard benchmark datasets:

| Benchmark | Source | Attacks | Blocked | ASR |
|-----------|--------|---------|---------|-----|
| MedSafetyBench | NeurIPS 2024 | 900 | 900 | 0.00% |
| HarmBench | Center for AI Safety | 400 | 400 | 0.00% |
| **Total** | | **1,300** | **1,300** | **0.00%** |

### 7.2 MedSafetyBench Results

**Source**: [github.com/AI4LIFE-GROUP/med-safety-bench](https://github.com/AI4LIFE-GROUP/med-safety-bench)

| Category | Attacks | Blocked |
|----------|---------|---------|
| Patient Care Quality | 109 | 109 |
| Medical Ethics & Professional Conduct | 107 | 107 |
| Regulatory & Legal Compliance | 81 | 81 |
| Drug Safety & Prescription | 106 | 106 |
| Mental Health & Vulnerable Populations | 100 | 100 |
| Privacy & Confidentiality | 74 | 74 |
| Research Ethics & Clinical Trials | 111 | 111 |
| Resource Allocation & Access | 114 | 114 |
| Communication & Informed Consent | 98 | 98 |

Attack Sources:
- GPT-4 Generated: 450
- Llama2 Generated: 450

### 7.3 HarmBench Results

**Source**: [github.com/centerforaisafety/HarmBench](https://github.com/centerforaisafety/HarmBench)

Tier Distribution:
| Tier | Blocks | Percentage |
|------|--------|------------|
| Tier 1 (PA) | 383 | 95.8% |
| Tier 2 (RAG) | 12 | 3.0% |
| Tier 3 (Expert) | 5 | 1.2% |

### 7.4 Statistical Analysis

```
Confidence Interval: 99.9% CI [0%, 0.28%]
Method: Wilson Score
Interpretation: 99.9% confidence that true attack success rate
                is between 0% and 0.28%
p-value: <0.001
```

---

## 8. Reproduction Guide

For complete reproduction instructions, see: **`docs/REPRODUCTION_GUIDE.md`**

### 8.1 Quick Start

```bash
# Clone repository
git clone https://github.com/TelosSteward/TELOS.git
cd TELOS

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set API key
echo "MISTRAL_API_KEY=your_key_here" > .env

# Launch dashboard
export PYTHONPATH=$(pwd)
streamlit run telos_observatory_v3/main.py --server.port 8501
```

### 8.2 Running Validation Tests

**Internal Validation Suite**:
```bash
export PYTHONPATH=$(pwd)
python3 telos_observatory_v3/telos_purpose/validation/run_internal_test0.py
```

**Unit Tests**:
```bash
python3 -m pytest tests/ -v
```

### 8.3 Reproducing Adversarial Validation

**Prerequisites**:
1. Install Ollama: `brew install ollama` or [ollama.ai](https://ollama.ai)
2. Pull embedding model: `ollama pull nomic-embed-text`
3. Start server: `ollama serve`

**Clone benchmark datasets**:
```bash
cd validation/
git clone https://github.com/AI4LIFE-GROUP/med-safety-bench
git clone https://github.com/centerforaisafety/HarmBench
mkdir -p harmbench_data
cp HarmBench/data/behavior_datasets/harmbench_behaviors_text_all.csv harmbench_data/
```

**Run validation** (requires Ollama running):
```bash
# MedSafetyBench (900 attacks, ~15 min)
python3 run_medsafetybench_validation.py

# HarmBench (400 attacks, ~7 min)
python3 run_harmbench_validation.py

# Quick test mode (10 attacks each)
python3 run_medsafetybench_validation.py --quick
python3 run_harmbench_validation.py --quick
```

---

## 9. Governance Philosophy and Open Research Commitment

TELOS is developed under an explicit commitment to open research. This section documents the philosophical foundations and practical commitments that guide the project.

### 9.1 The Case for Open AI Governance Research

Runtime AI governance is too consequential to be developed in secret. The TELOS project rejects the pattern where AI safety research becomes closed as it becomes valuable—what we term the **Transparency Inversion**:

1. Organization founded on open research principles
2. Significant capabilities developed
3. Commercial pressures mount
4. Research becomes "too dangerous" to publish
5. Safety decisions made internally, without external review
6. Public told to trust that decisions are correct

**The result:** The organizations claiming to work on AI safety are often the least transparent about how they make safety decisions.

TELOS takes the opposite position: safety research, of all research, should be the most open—subject to peer review, public scrutiny, and independent validation.

### 9.2 The Ten Founding Principles

The TELOS project operates under these foundational commitments:

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

### 9.3 What We Publish Openly

| Research Area | Publication Commitment |
|---------------|----------------------|
| Governance frameworks | Full methodology, open access |
| Fidelity metrics (F_user, F_AI, PS) | Mathematical specification, validation data |
| Adversarial testing results | Complete attack taxonomy, success/failure rates |
| Deployment studies | Aggregated performance data, lessons learned |
| Failure analyses | When governance fails, why it fails |
| Benchmark suites | Open source, reproducible |

**What we will not claim:**
- "This is too dangerous to publish"
- "Trust us, we've validated internally"
- "Our safety decisions are proprietary"
- "Competitive advantage requires secrecy"

### 9.4 Utilitarian-Ethical Framework

TELOS measures success by actual outcomes, not proxies:

**What we measure:**
| Metric | Description |
|--------|-------------|
| **Harm Prevented** | Drift caught, hallucinations avoided, scope violations blocked |
| **User Outcomes** | Conversations that achieve their stated purpose |
| **Compliance Achieved** | Regulatory requirements demonstrably met |
| **Failures Analyzed** | What went wrong, why, and how to prevent recurrence |

**Not by:**
- Papers published (vanity metric)
- Press coverage (PR metric)
- Competitor criticism (political metric)
- Theoretical completeness (academic metric)

### 9.5 The Open Core Model

```
TELOS Consortium (Open Research)
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
**Enterprise features:** Proprietary extensions
**Research findings:** Always published openly

### 9.6 Agentic AI Governance Extension

The Primacy Attractor framework extends naturally to agentic AI:

| Conversational AI | Agentic AI |
|-------------------|------------|
| Generates text | Takes actions |
| User reviews before acting | May act before user sees |
| Single-turn impact | Multi-step cascading impact |
| Undo = ignore response | Undo = may be impossible |
| Scope = topic boundaries | Scope = action boundaries |

**The extension hypothesis:**

```
Agent Primacy Attractor = f(
    stated_purpose,           # "Book travel within constraints"
    action_boundaries,        # Permitted tool calls, scope limits
    confirmation_thresholds,  # What requires human approval
    irreversibility_flags     # Actions that cannot be undone
)
```

This represents active research—see the TELOS Consortium Manifesto for the full research agenda.

### 9.7 Corporate Structure

TELOS AI Labs is structured as a Delaware Public Benefit Corporation (PBC), with:

- **Explicit public benefit purpose** enshrined in charter
- **Consortium board seat** with protective provisions for open research
- **Open Research Commitment** that survives acquisition
- **Dual-entity model**: Commercial (Labs) + Research (Consortium)

For full governance details, see: `docs/TELOS_PBC_GOVERNANCE.md`

### 9.8 Reference Documents

| Document | Purpose |
|----------|---------|
| [TELOS Consortium Manifesto](TELOS_CONSORTIUM_MANIFESTO.md) | Full philosophy, research commitments, governance structure |
| [PBC Governance](TELOS_PBC_GOVERNANCE.md) | Corporate structure, board composition, protective provisions |
| [Grant Roadmap](GRANT_ROADMAP.md) | Funding phases, research agenda |

---

## Appendix A: Validation Dataset

**DOI**: [10.5281/zenodo.17702890](https://doi.org/10.5281/zenodo.17702890)

### Dataset Metadata

```json
{
  "title": "TELOS Adversarial Validation Dataset",
  "version": "1.0",
  "date": "2025-11-25",
  "authors": ["Jeffrey Brunner"],
  "license": "CC BY 4.0"
}
```

### Validation Summary

| Metric | Value |
|--------|-------|
| Total Attacks | 1,300 |
| Total Blocked | 1,300 |
| Attack Success Rate | 0.00% |
| Harm Prevention Rate | 100.00% |

### Confidence Interval

```
Method: Wilson Score
Confidence Level: 99.9%
Lower Bound: 0.0%
Upper Bound: 0.28%
Interpretation: 99.9% confidence that true attack success rate
                is between 0% and 0.28%
```

### Benchmark Breakdown

**MedSafetyBench** (NeurIPS 2024):
- Attacks Tested: 900
- Attacks Blocked: 900
- Categories: 9 healthcare safety domains
- Sources: GPT-4 (450) + Llama2 (450)
- Average Fidelity Score: 0.227

**HarmBench** (Center for AI Safety):
- Attacks Tested: 400
- Attacks Blocked: 400
- Tier 1 (PA) Blocks: 95.8%
- Tier 2 (RAG) Blocks: 3.0%
- Tier 3 (Expert) Blocks: 1.2%

### Methodology

**Three-Tier Governance**:

1. **Tier 1 - Primacy Attractor (PA)**
   - Method: Cosine similarity to compliance embedding
   - Threshold: Fidelity ≥ 0.18 triggers block
   - Purpose: Autonomous harm prevention

2. **Tier 2 - RAG Corpus**
   - Method: Policy consultation with guidelines
   - Sources: HIPAA, FDA, AMA ethics
   - Purpose: Contextual guidance

3. **Tier 3 - Human Expert**
   - Method: Escalation to qualified expert
   - Trigger: Low fidelity + high uncertainty
   - Purpose: Final arbiter

### Files Included

| File | Description |
|------|-------------|
| `telos_complete_validation_dataset.json` | Complete results with statistics |
| `medsafetybench_validation_results.json` | 900 MedSafetyBench traces |
| `harmbench_validation_results.json` | 400 HarmBench traces |
| `telos_validation_dataset_zenodo.json` | Zenodo upload format (v2.0) |

---

## Appendix B: Constants Reference

All constants are defined in: `telos_observatory_v3/telos_purpose/core/constants.py`

### Embedding Configuration

```python
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
```

### Fidelity Thresholds (Normalized)

```python
FIDELITY_MONITOR = 0.76    # Green zone
FIDELITY_CORRECT = 0.73    # Yellow zone
FIDELITY_INTERVENE = 0.67  # Orange zone
FIDELITY_ESCALATE = 0.67   # Red zone
```

### Raw Model Thresholds

**SentenceTransformer**:
```python
ST_FIDELITY_GREEN = 0.32
ST_FIDELITY_YELLOW = 0.28
ST_FIDELITY_ORANGE = 0.24
ST_FIDELITY_RED = 0.24
```

**Mistral Embed**:
```python
MISTRAL_FIDELITY_GREEN = 0.60
MISTRAL_FIDELITY_YELLOW = 0.50
MISTRAL_FIDELITY_ORANGE = 0.42
MISTRAL_FIDELITY_RED = 0.42
```

### Proportional Control

```python
DEFAULT_K_ATTRACTOR = 1.5
DEFAULT_K_ANTIMETA = 2.0
DEFAULT_MAX_REGENERATIONS = 3
INTERVENTION_BUDGET_PER_SESSION = 10
```

### Basin Geometry

```python
BASIN_RADIUS_MIN = 2.0  # When τ = 0
BASIN_RADIUS_MAX = 8.0  # When τ = 1

def compute_basin_radius(τ):
    ρ = 1.0 - τ
    return 2.0 / max(ρ, 0.25)
```

### Validation Thresholds

```python
H1_DELTA_F_THRESHOLD = 0.15  # ΔF ≥ 0.15 required
H2_SUPREMACY_THRESHOLD = 0.01
```

---

## Citation

```bibtex
@dataset{brunner_2025_telos_adversarial,
  author       = {Brunner, Jeffrey},
  title        = {{TELOS Adversarial Validation Dataset}},
  month        = nov,
  year         = 2025,
  publisher    = {Zenodo},
  version      = {1.0},
  doi          = {10.5281/zenodo.17702890},
  url          = {https://doi.org/10.5281/zenodo.17702890}
}

@dataset{brunner_2025_telos_governance,
  author       = {Brunner, Jeffrey},
  title        = {{TELOS Governance Benchmark Dataset}},
  month        = dec,
  year         = 2025,
  publisher    = {Zenodo},
  version      = {1.0},
  doi          = {10.5281/zenodo.18009153},
  url          = {https://doi.org/10.5281/zenodo.18009153}
}
```

---

**Document Version**: 3.0
**Last Updated**: December 2025
**Status**: Research Phase

