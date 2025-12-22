# TELOS Observatory

**Runtime AI Governance Framework with Published Validation**

![Python](https://img.shields.io/badge/python-3.8+-green)
![License](https://img.shields.io/badge/license-MIT-orange)
[![Attacks](https://img.shields.io/badge/attacks-1%2C300-blue)](https://github.com/TelosSteward/TELOS-Validation)
[![ASR](https://img.shields.io/badge/attack%20success%20rate-0%25-brightgreen)](https://github.com/TelosSteward/TELOS-Validation)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17702890.svg)](https://doi.org/10.5281/zenodo.17702890)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18009153.svg)](https://doi.org/10.5281/zenodo.18009153)

---

## Published Validation Results

### Adversarial Validation Dataset
- **Total Attacks**: 1,300
- **Attack Success Rate**: 0.00%
- **Statistical Confidence**: 99.9% CI [0%, 0.28%]
- **Autonomous Blocking**: 95.8% (Tier 1)
- **DOI**: [10.5281/zenodo.17702890](https://doi.org/10.5281/zenodo.17702890)

| Benchmark | Attacks | Blocked | Source |
|-----------|---------|---------|--------|
| MedSafetyBench | 900 | 900 (100%) | [AI4LIFE-GROUP/med-safety-bench](https://github.com/AI4LIFE-GROUP/med-safety-bench) |
| HarmBench | 400 | 400 (100%) | [centerforaisafety/HarmBench](https://github.com/centerforaisafety/HarmBench) |

### Governance Benchmark Dataset
- **Sessions**: 46 multi-session governance evaluations
- **Domains**: 8 diverse application domains
- **DOI**: [10.5281/zenodo.18009153](https://doi.org/10.5281/zenodo.18009153)

---

## What is TELOS Observatory?

TELOS  is a mathematical governance framework for AI alignment. It uses **Primacy Attractors** (embedding-space representations of user purpose) to detect and correct conversational drift in real-time.

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

- **Steward Lens**: Real-time governance metrics and intervention tracking
- **TELOSCOPE**: Mathematical transparency window showing fidelity calculations
- **Turn Navigation**: Time-travel through conversation history
- **Evidence Export**: JSON, CSV, Markdown transcripts for research

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
  doi          = {10.5281/zenodo.17702890},
  url          = {https://doi.org/10.5281/zenodo.17702890}
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

---

## License

MIT License - See LICENSE file

---

## Hardware Requirements

See [HARDWARE_REQUIREMENTS.md](HARDWARE_REQUIREMENTS.md) for system specifications.

---

**Last Updated**: December 21, 2025

