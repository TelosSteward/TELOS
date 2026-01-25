# TELOS Observatory

**Runtime AI Governance Framework with Published Validation**

![Python](https://img.shields.io/badge/python-3.8+-green)
![License](https://img.shields.io/badge/license-MIT-orange)
[![Attacks](https://img.shields.io/badge/attacks-2%2C550-blue)](https://github.com/TelosSteward/TELOS-Validation)
[![ASR](https://img.shields.io/badge/attack%20success%20rate-0%25-brightgreen)](https://github.com/TelosSteward/TELOS-Validation)

[![AILuminate DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18370263.svg)](https://doi.org/10.5281/zenodo.18370263)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18013104.svg)](https://doi.org/10.5281/zenodo.18013104)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18009153.svg)](https://doi.org/10.5281/zenodo.18009153)
[![SB 243 DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18027446.svg)](https://doi.org/10.5281/zenodo.18027446)
[![XSTest DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18370603.svg)](https://doi.org/10.5281/zenodo.18370603)

---

## Published Validation Results

### Adversarial Validation (2,550 attacks, 0% ASR)
- **Total Attacks**: 2,550
- **Attack Success Rate**: 0.00%
- **Statistical Confidence**: 99.9% CI [0%, 0.14%]
- **Forensic Audit Trails**: Full JSONL traces for every governance decision

| Benchmark | Attacks | Blocked | DOI |
|-----------|---------|---------|-----|
| AILuminate (MLCommons) | 1,200 | 1,200 (100%) | [10.5281/zenodo.18370263](https://doi.org/10.5281/zenodo.18370263) |
| MedSafetyBench (NeurIPS 2024) | 900 | 900 (100%) | [10.5281/zenodo.18013104](https://doi.org/10.5281/zenodo.18013104) |
| HarmBench (CAIS) | 400 | 400 (100%) | [10.5281/zenodo.18013104](https://doi.org/10.5281/zenodo.18013104) |
| SB 243 Child Safety | 50 | 50 (100%) | [10.5281/zenodo.18027446](https://doi.org/10.5281/zenodo.18027446) |

### XSTest Over-Refusal Calibration
- **False Positive Rate**: 8.0% (Healthcare PA) vs 24.8% (Generic PA)
- **Improvement**: 16.8 percentage point reduction
- **DOI**: [10.5281/zenodo.18370603](https://doi.org/10.5281/zenodo.18370603)

### Governance Benchmark Dataset
- **Sessions**: 46 multi-session governance evaluations
- **Domains**: 8 diverse application domains
- **DOI**: [10.5281/zenodo.18009153](https://doi.org/10.5281/zenodo.18009153)

---

## Runtime Auditable Governance

TELOS produces audit records at the moment of each governance decision. When regulators examine an incident, they can trace exactly what the system measured, what thresholds applied, and why a particular intervention occurred.

**Forensic Trace Contents:**
- Session initialization with PA configuration
- Per-turn fidelity calculations with raw similarity scores
- Intervention decisions with tier classification and rationale
- Complete JSONL format for log aggregation (Elasticsearch, Splunk, CloudWatch)

**Published Evidence:** All validation datasets include complete forensic audit trails (11,208 governance events across 2,550 attacks).

**Regulatory Alignment:**
- EU AI Act Article 12: Automatic recording of events during operation
- EU AI Act Article 72: Post-market monitoring with continuous logging
- California SB 53: Documentation of safety-relevant decisions
- HIPAA Security Rule: Audit controls for access and decision logging

---

## What is TELOS Observatory?

TELOS is a mathematical governance framework for AI alignment. It uses **Primacy Attractors** (embedding-space representations of user purpose) to detect and correct conversational drift in real-time.

**Core Innovation**: Two-layer fidelity detection - baseline normalization catches extreme off-topic content, basin membership catches purpose drift.

---

## Quick Start

### Run Locally

```bash
git clone https://github.com/TelosSteward/TELOS.git
cd TELOS
pip install -r requirements.txt
export MISTRAL_API_KEY="your_key_here"
streamlit run telos_observatory_v3/main.py --server.port 8501
```

Opens in browser at `http://localhost:8501`

### Reproduce Validation Results

See [REPRODUCTION_GUIDE.md](docs/REPRODUCTION_GUIDE.md) for step-by-step instructions.

```bash
# Run internal validation suite
python3 telos_observatory_v3/telos_purpose/validation/run_internal_test0.py

# Adversarial validation results are pre-computed in validation/ directory
# See validation/telos_complete_validation_dataset.json
```

---

## Architecture

### Two-Layer Fidelity System

**Layer 1: Baseline Pre-Filter**
- Constant: `SIMILARITY_BASELINE = 0.35`
- Catches content completely outside the PA embedding space
- Raw cosine similarity < 0.35 triggers immediate HARD_BLOCK

**Layer 2: Basin Membership**
- Constants: `BASIN = 0.50`, `TOLERANCE = 0.02`
- Threshold: `INTERVENTION_THRESHOLD = 0.48`
- Detects when user has drifted from stated purpose

**Intervention Decision**:
```python
should_intervene = (raw_similarity < 0.35) OR (fidelity < 0.48)
```

### Fidelity Zones

| Zone | Fidelity | Color | Meaning |
|------|----------|-------|---------|
| GREEN | >= 0.70 | `#27ae60` | Aligned |
| YELLOW | 0.60-0.69 | `#f39c12` | Minor Drift |
| ORANGE | 0.50-0.59 | `#e67e22` | Drift Detected |
| RED | < 0.50 | `#e74c3c` | Significant Drift |

---

## Key Features

- **Alignment Lens**: Real-time governance metrics and intervention tracking
- **TELOSCOPE**: Mathematical transparency window showing fidelity calculations
- **Turn Navigation**: Time-travel through conversation history

---

## Related Repositories

- **[TELOS-Validation](https://github.com/TelosSteward/TELOS-Validation)**: Published validation datasets

---

## Citation

For the adversarial validation dataset:

```bibtex
@dataset{brunner_2025_telos_adversarial,
  author       = {Brunner, Jeffrey},
  title        = {{TELOS Adversarial Validation Dataset}},
  month        = nov,
  year         = 2025,
  publisher    = {Zenodo},
  version      = {1.0},
  doi          = {10.5281/zenodo.18013104},
  url          = {https://doi.org/10.5281/zenodo.18013104}
}
```

For the governance benchmark dataset:

```bibtex
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

For the SB 243 child safety validation:

```bibtex
@dataset{brunner_2025_telos_sb243,
  author       = {Brunner, Jeffrey},
  title        = {{TELOS SB 243 Child Safety Validation Dataset}},
  month        = dec,
  year         = 2025,
  publisher    = {Zenodo},
  version      = {1.0},
  doi          = {10.5281/zenodo.18027446},
  url          = {https://doi.org/10.5281/zenodo.18027446}
}
```

---

## License

MIT License - See LICENSE file

---

## Hardware Requirements

See [HARDWARE_REQUIREMENTS.md](HARDWARE_REQUIREMENTS.md) for system specifications.

---

**Last Updated**: January 25, 2026

