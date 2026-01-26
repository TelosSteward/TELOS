# TELOS Observatory: Reproduction Guide

**Purpose**: Enable peer reviewers, grant evaluators, and researchers to verify TELOS governance functionality.

**Status**: Research phase - this is an active research project, not production software.

---

## Published Validation Datasets (Zenodo)

| Dataset | DOI | Description |
|---------|-----|-------------|
| **Adversarial Validation** | [10.5281/zenodo.18370659](https://doi.org/10.5281/zenodo.18370659) | 2,550 attacks, 0/2,550 observed successes |
| **AILuminate Benchmark** | [10.5281/zenodo.18370263](https://doi.org/10.5281/zenodo.18370263) | 1,200 MLCommons attacks across 12 harm categories |
| **Governance Benchmark** | [10.5281/zenodo.18009153](https://doi.org/10.5281/zenodo.18009153) | 46 multi-session governance evaluations across 8 domains |
| **SB 243 Child Safety** | [10.5281/zenodo.18370504](https://doi.org/10.5281/zenodo.18370504) | CA SB 243-aligned evaluation (0% ASR, 74% FPR) |
| **XSTest Over-Refusal** | [10.5281/zenodo.18370603](https://doi.org/10.5281/zenodo.18370603) | False positive calibration (8% FPR with domain PA) |

### Adversarial Validation Dataset

Located in `validation/` directory (included in repository):
- `telos_complete_validation_dataset.json` - Complete results with statistics
- `medsafetybench_validation_results.json` - 900 healthcare attacks (NeurIPS 2024)
- `harmbench_validation_results_summary.json` - 400 HarmBench attacks

### Governance Benchmark Dataset

**External download required**: The governance benchmark data is hosted on Zenodo, not included in this repository.

To reproduce governance benchmark studies:

1. Download from Zenodo: https://doi.org/10.5281/zenodo.18009153
2. Or run Phase 2 validation on your own conversation data (see Phase 2 Validation below)

---

## Prerequisites

### System Requirements
- **OS**: macOS, Linux, or Windows
- **Python**: 3.8+
- **Disk Space**: ~500MB

### API Key Required
- **Mistral API**: Get a free key at https://console.mistral.ai/

---

## Quick Start (10 minutes)

### Step 1: Clone and Install

```bash
# Clone repository
git clone https://github.com/TelosSteward/TELOS.git
cd TELOS

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Set API Key

Create a `.env` file in the repository root:
```bash
echo "MISTRAL_API_KEY=your_api_key_here" > .env
```

Or export directly:
```bash
export MISTRAL_API_KEY="your_api_key_here"
```

### Step 3: Launch the Observatory Dashboard

```bash
export PYTHONPATH=$(pwd)
streamlit run telos_observatory_v3/main.py --server.port 8501
```

Open http://localhost:8501 in your browser.

---

## Validation Tests

### Test 1: Internal Validation Suite

Runs baseline condition tests with comprehensive error handling.

```bash
cd /path/to/TELOS
export PYTHONPATH=$(pwd)
export MISTRAL_API_KEY="your_key"

python3 telos_observatory_v3/telos_purpose/validation/run_internal_test0.py
```

**What it tests**:
- StatelessRunner
- PromptOnlyRunner
- CadenceReminderRunner
- ObservationRunner
- TELOSRunner

### Test 2: Integration Tests

End-to-end pipeline tests.

```bash
python3 -m telos_observatory_v3.telos_purpose.validation.integration_tests
```

**What it tests**:
- Data pipeline
- Analytics pipeline
- Export pipeline
- Session workflow

### Test 3: Performance Check

Measures fidelity calculation performance.

```bash
python3 telos_observatory_v3/telos_purpose/validation/performance_check.py
```

### Test 4: Comparative Test

Compares different Primacy Attractor configurations.

```bash
python3 telos_observatory_v3/telos_purpose/validation/comparative_test.py
```

### Test 5: Unit Tests (pytest)

Comprehensive unit tests for TELOS core constants and mathematical functions.

```bash
cd /path/to/TELOS

# Run all unit tests
python3 -m pytest tests/ -v

# Run specific test file
python3 -m pytest tests/test_constants.py -v
```

**What it tests** (78 tests):
- Basin radius calculations (r = 2/max(ρ, 0.25))
- Epsilon threshold functions (ε_min, ε_max)
- Model-specific threshold lookups (SentenceTransformer vs Mistral)
- Fidelity zone classification
- Proportional control gains (K_ATTRACTOR, K_ANTIMETA)
- Numerical edge cases and boundary conditions

**Expected output**: All 78 tests should pass in < 1 second.

---

## Phase 2 Validation (Governance Benchmark)

Run complete governance validation studies on conversation datasets. This generates data comparable to the Governance Benchmark published on Zenodo.

### Available Datasets

| Dataset | Location | Sessions |
|---------|----------|----------|
| Test Sessions | `telos_purpose/test_data/test_sessions/` | 8 synthetic sessions |
| Edge Cases | `telos_purpose/test_data/edge_cases/` | 10 edge case scenarios |
| Custom | Your ShareGPT-format JSON | Variable |

### Running Phase 2 Validation

```bash
cd /path/to/TELOS
export PYTHONPATH=$(pwd)
export MISTRAL_API_KEY="your_key"

# On internal test data
python3 phase2_validation_run.py --test

# On custom ShareGPT-format data
python3 phase2_validation_run.py path/to/data.json study_name
```

### What Phase 2 Tests

1. **LLM-at-every-turn PA establishment** - Statistical convergence detection (max 10 turns)
2. **Drift monitoring** - F < 0.8 threshold detection
3. **Counterfactual branching** - Original vs TELOS governance comparison
4. **Evidence generation** - JSON + Markdown exports

### Output Structure

```
phase2_validation_{study_name}/
  study_results/
    phase2_study_summary.json          # Aggregate results
    {conversation_id}/
      intervention_{branch_id}.json     # Counterfactual evidence
      intervention_{branch_id}.md       # Human-readable evidence
  research_briefs/
    README.md                            # Index of all briefs
    research_brief_{num}_{id}.md        # Individual study brief
```

### Internal Test Data Details

**Test Sessions** (`test_data/test_sessions/`):
- 8 sessions, 163 total turns
- Average fidelity: 0.787
- 77 total interventions
- Scenarios: normal, high_drift, excellent, long, short, critical_drift, stable, oscillating

**Edge Cases** (`test_data/edge_cases/`):
- 10 edge case scenarios
- Tests: empty sessions, single turns, missing fields, extreme fidelity, unicode handling, very long sessions

---

## Generate Test Data

Create reproducible test conversation sessions:

```bash
python3 -m telos_observatory_v3.telos_purpose.test_data.generate_test_sessions \
    --output telos_observatory_v3/telos_purpose/test_data/test_sessions/ \
    --seed 42
```

---

## Core Architecture

### Two-Layer Fidelity System

The TELOS governance engine uses a two-layer detection system:

**Layer 1: Baseline Pre-Filter**
```
Constant: SIMILARITY_BASELINE = 0.35
```
Catches content completely outside the Primacy Attractor embedding space. Raw cosine similarity < 0.35 triggers immediate block.

**Layer 2: Basin Membership**
```
Constants: BASIN = 0.50, TOLERANCE = 0.02
Threshold: INTERVENTION_THRESHOLD = 0.48
```
Detects when user has drifted from stated purpose. Fidelity < 0.48 means outside basin, triggers intervention.

**Intervention Decision**:
```python
should_intervene = (raw_similarity < 0.35) OR (fidelity < 0.48)
```

### Key Source Files

| File | Purpose |
|------|---------|
| `telos_observatory_v3/services/beta_response_manager.py` | Main fidelity engine |
| `telos_observatory_v3/telos_purpose/core/constants.py` | All calibration constants |
| `telos_observatory_v3/telos_purpose/core/embedding_provider.py` | Mistral + SentenceTransformer embeddings |
| `telos_observatory_v3/telos_purpose/core/semantic_interpreter.py` | Fidelity to linguistic specs |
| `telos_observatory_v3/telos_purpose/core/proportional_controller.py` | Intervention strength (K=1.5) |

---

## Adversarial Validation Reproduction

The validation against MedSafetyBench and HarmBench can be fully reproduced.

### Prerequisites

1. **Install Ollama** (local embedding server):
   ```bash
   # macOS
   brew install ollama

   # Or download from https://ollama.ai
   ```

2. **Pull the embedding model**:
   ```bash
   ollama pull nomic-embed-text
   ollama serve  # Start the server (runs on localhost:11434)
   ```

3. **Clone benchmark datasets** (not included in repo to keep size small):
   ```bash
   cd validation/

   # MedSafetyBench (NeurIPS 2024)
   git clone https://github.com/AI4LIFE-GROUP/med-safety-bench

   # HarmBench (Center for AI Safety)
   git clone https://github.com/centerforaisafety/HarmBench
   mkdir -p harmbench_data
   cp HarmBench/data/behavior_datasets/harmbench_behaviors_text_all.csv harmbench_data/
   ```

### Run Adversarial Validation

```bash
cd validation/

# Run MedSafetyBench validation (900 attacks, ~15 min)
python3 run_medsafetybench_validation.py

# Run HarmBench validation (400 attacks, ~7 min)
python3 run_harmbench_validation.py

# Quick test mode (10 attacks each)
python3 run_medsafetybench_validation.py --quick
python3 run_harmbench_validation.py --quick
```

### Expected Output

- `medsafetybench_validation_results.json` - Per-attack forensic traces
- `harmbench_validation_results.json` - Per-attack forensic traces
- Console output showing 0% ASR, 100% VDR

---

## Published Validation Results

Results from `validation/` directory:

| Dataset | Attacks | Result |
|---------|---------|--------|
| Complete Validation | 2,550 | 0/2,550 observed (95% CI upper bound ~0.15%) |
| AILuminate | 1,200 | 0/1,200 observed |
| MedSafetyBench | 900 | 0/900 observed |
| HarmBench | 400 | 0/400 observed |
| SB 243-aligned | 50 | 0/50 observed |

### AILuminate Benchmark Validation

Industry-standard benchmark from MLCommons AI Safety Working Group covering 12 NIST AI RMF harm categories.

| Metric | Value |
|--------|-------|
| Total Prompts | 1,200 |
| Attack Success Rate | 0/1,200 observed |
| Harm Categories | 12 (CBRNE, Child Safety, Hate, Self-Harm, etc.) |

**Files** (in `validation/`):
- `ailuminate_validation_results.json` - Per-prompt forensic traces
- `ailuminate_forensic_summary.json` - Aggregate statistics

**Run validation**:
```bash
cd validation/
python3 run_ailuminate_validation.py

# Quick test mode (10 prompts)
python3 run_ailuminate_validation.py --quick
```

**DOI**: [10.5281/zenodo.18370263](https://doi.org/10.5281/zenodo.18370263)

### SB 243-Aligned Child Safety Evaluation

Domain-specific validation against California SB 243 child safety harm categories (suicide, self-harm, sexual content, eating disorders).

| Metric | Value |
|--------|-------|
| Attack Success Rate | 0.00% |
| False Positive Rate | 74.00% |
| Violation Defense Rate | 100.00% |

**Acknowledged Limitation**: The 74% FPR is an intentional design choice. In child safety contexts, it is better to catch too much than too little. The cost of a false positive is user friction; the cost of a false negative could be harm to a child.

**Files** (in `validation/`):
- `sb243_data/sb243_test_attacks.csv` - 50 harmful prompts
- `sb243_data/sb243_benign_contrastive.csv` - 50 benign contrastive queries
- `config/sb243_child_safety_pa_config.json` - PA configuration
- `run_sb243_validation.py` - Attack validation script
- `run_false_positive_validation.py` - FPR validation script

**Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)

**Run validation**:
```bash
pip install sentence-transformers numpy
cd validation/
python3 run_sb243_validation.py
python3 run_false_positive_validation.py
```

See `validation/FALSE_POSITIVE_ANALYSIS.md` for full analysis of the safety-utility tradeoff.

### XSTest Over-Refusal Calibration

Measures false positive rate using the XSTest benchmark (safe prompts that look unsafe). Demonstrates that domain-specific Primacy Attractors reduce over-refusal compared to generic safety filters.

| Metric | Healthcare PA | Generic PA |
|--------|---------------|------------|
| False Positive Rate | 8.0% | 24.8% |
| Improvement | 16.8 percentage points | - |

**What XSTest Measures**: Safe prompts that contain sensitive keywords but are actually benign (e.g., "How do I kill a process in Linux?"). High FPR means blocking legitimate queries.

**Files** (in `validation/`):
- `xstest_validation_results.json` - Per-prompt classification results
- `xstest_fpr_analysis.json` - False positive rate calculations

**Run validation**:
```bash
cd validation/
python3 run_xstest_validation.py

# Compare domain PA vs generic PA
python3 run_xstest_validation.py --compare
```

**Key Finding**: Domain-specific PAs (e.g., Healthcare PA) dramatically reduce false positives while maintaining attack blocking, because they understand context rather than just detecting sensitive keywords.

**DOI**: [10.5281/zenodo.18370603](https://doi.org/10.5281/zenodo.18370603)

---

## Verification Checklist

### Basic Setup
- [ ] Cloned repository
- [ ] Installed dependencies (`pip install -r requirements.txt`)
- [ ] Set `MISTRAL_API_KEY`
- [ ] Launched Streamlit dashboard successfully

### Adversarial Validation (2,550 attacks)
- [ ] Reviewed `validation/telos_complete_validation_dataset.json`
- [ ] Ran internal validation test (`run_internal_test0.py`)
- [ ] Observed fidelity measurements in dashboard

### Individual Benchmark Reproduction (optional)
- [ ] AILuminate: Ran `run_ailuminate_validation.py` (1,200 prompts)
- [ ] MedSafetyBench: Ran `run_medsafetybench_validation.py` (900 prompts)
- [ ] HarmBench: Ran `run_harmbench_validation.py` (400 prompts)
- [ ] SB 243: Ran `run_sb243_validation.py` (50 prompts)
- [ ] XSTest: Ran `run_xstest_validation.py` (FPR calibration)

### Governance Benchmark (optional)
- [ ] Downloaded governance benchmark from Zenodo (or)
- [ ] Ran Phase 2 validation on test sessions
- [ ] Generated research briefs

---

## Troubleshooting

### Import Errors

Ensure PYTHONPATH is set:
```bash
export PYTHONPATH=/path/to/TELOS
```

### API Authentication Error

1. Check API key is set: `echo $MISTRAL_API_KEY`
2. Verify key is valid at https://console.mistral.ai/
3. Re-export if needed

### Streamlit Not Starting

```bash
# Check if port is in use
lsof -i :8501

# Try alternative port
streamlit run telos_observatory_v3/main.py --server.port 8502
```

### Embedding Model Issues

```bash
# Verify sentence-transformers installed
python3 -c "from sentence_transformers import SentenceTransformer; print('OK')"
```

---

## Repository Structure

```
TELOS/
├── telos_observatory_v3/
│   ├── main.py                    # Streamlit entry point
│   ├── telos_purpose/
│   │   ├── core/                  # Mathematical framework
│   │   ├── validation/            # Test suites
│   │   ├── test_data/             # Test data generation
│   │   ├── sessions/              # Session management
│   │   └── llm_clients/           # LLM integrations
│   ├── components/                # UI components
│   ├── services/                  # Backend logic
│   └── config/                    # Configuration
├── validation/                    # Published validation results
├── docs/                          # Documentation
├── requirements.txt
└── CLAUDE.md                      # Development guide
```

---

## Contact & Support

**Issues**: https://github.com/TelosSteward/TELOS/issues

---

**Document Version**: 2.1
**Last Updated**: January 2026
**Status**: Research Phase
