# TELOS Adversarial Validation Dataset

**Version**: 1.0
**Date**: November 24, 2025
**Author**: Jeffrey Brunner
**License**: CC BY 4.0

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17702890.svg)](https://doi.org/10.5281/zenodo.17702890)

---

## Overview

Validation results for TELOS AI governance framework across 1,300 adversarial attacks from two standardized benchmarks.

**Results**:
- Total attacks: 1,300
- Attack success rate: 0%
- Statistical confidence: 99.9% CI [0%, 0.28%]

---

## Dataset Files

### 1. medsafetybench_validation_results.json (490KB)
- 900 healthcare safety attacks
- Source: NeurIPS 2024 MedSafetyBench
- 0% ASR

### 2. harmbench_validation_results_summary.json
- 400 general adversarial attacks
- Source: CAIS HarmBench
- 0% ASR

### 3. telos_complete_validation_dataset.json
- Combined statistical analysis
- Wilson Score CI, Bayesian analysis, power analysis

### 4. REPRODUCTION_GUIDE.md
- Step-by-step reproduction instructions
- 15-minute setup time

### 5. HARDWARE_REQUIREMENTS.md
- Computational specifications
- Minimum: 8GB RAM, 4+ cores

### 6. requirements-pinned.txt
- Exact Python dependency versions

---

## Validation Summary

| Benchmark | Attacks | Blocked | Success Rate |
|-----------|---------|---------|--------------|
| MedSafetyBench | 900 | 900 | 0% |
| HarmBench | 400 | 400 | 0% |
| **Total** | **1,300** | **1,300** | **0%** |

---

## Quick Start

```bash
# Clone repository
git clone https://github.com/TelosSteward/TELOS.git
cd TELOS/TELOSCOPE_BETA

# Install dependencies
pip install -r requirements-pinned.txt

# Run TELOSCOPE Observatory
streamlit run main.py
```

**Validation data**: Download from Zenodo (https://doi.org/10.5281/zenodo.17702890)

Published results: ~8-10 seconds execution (original hardware), 0% attack success rate

---

## Links

- **GitHub**: https://github.com/TelosSteward/TELOS
- **Live Demo**: [Streamlit Cloud URL - to be added]
- **DOI**: https://doi.org/10.5281/zenodo.17702890

---

## Citation

```bibtex
@dataset{telos2025validation,
  author       = {Brunner, Jeffrey},
  title        = {TELOS Adversarial Validation Dataset},
  year         = 2025,
  publisher    = {Zenodo},
  version      = {1.0},
  doi          = {10.5281/zenodo.17702890},
  url          = {https://doi.org/10.5281/zenodo.17702890}
}
```

---

## Contact

- GitHub Issues: https://github.com/TelosSteward/TELOS/issues
- Email: [To be established]

---

**Total Attacks**: 1,300
**Attack Success Rate**: 0%
**Statistical Confidence**: 99.9% CI [0%, 0.28%]
