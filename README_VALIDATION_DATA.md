# TELOS Adversarial Validation Dataset

**Version**: 1.0
**Date**: November 24, 2025
**Author**: Jeffrey Brunner
**License**: CC BY 4.0

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17702890.svg)](https://doi.org/10.5281/zenodo.17702890)

---

## Overview

Validation results for TELOS AI governance framework across 1,986 adversarial attacks from four benchmarks.

**Results**:
- Total attacks: 1,986
- Attack success rate: 0%
- Statistical confidence: 99.9% CI [0%, 0.38%]
- All governance decisions cryptographically signed (Telemetric Keys)

---

## Dataset Files

### 1. medsafetybench_validation_results.json (490KB)
- 960 healthcare safety attacks
- Source: NeurIPS 2024 MedSafetyBench
- 0% ASR

### 2. harmbench_validation_results_summary.json
- 410 general adversarial attacks
- Source: CAIS HarmBench
- 0% ASR

### 3. agentharm_validation_results.json (75KB)
- 196 agentic AI attacks
- Source: ICLR 2025 AgentHarm
- 0% ASR

### 4. unified_benchmark_results.json (83KB)
- 400 privacy/PII attacks (PII-Bench)
- Combined statistical analysis
- Wilson Score CI, Bayesian analysis, power analysis

### 5. REPRODUCTION_GUIDE.md
- Step-by-step reproduction instructions
- 15-minute setup time

### 6. HARDWARE_REQUIREMENTS.md
- Computational specifications
- Minimum: 8GB RAM, 4+ cores

### 7. requirements-pinned.txt
- Exact Python dependency versions

---

## Validation Summary

| Benchmark | Attacks | Blocked | Success Rate |
|-----------|---------|---------|--------------|
| MedSafetyBench | 960 | 960 | 0% |
| HarmBench | 410 | 410 | 0% |
| AgentHarm | 196 | 196 | 0% |
| PII-Bench | 400 | 400 | 0% |
| **Total** | **1,966** | **1,966** | **0%** |

---

## Quick Start

```bash
# Clone repository
git clone https://github.com/TelosSteward/TELOS.git
cd TELOS/TELOSCOPE_BETA

# Install dependencies
pip install -r requirements-pinned.txt

# Run validation
cd ../healthcare_validation
python3 run_unified_benchmark.py
```

Expected: ~12 seconds execution, 0% attack success rate

---

## Cryptographic Validation

Each governance decision signed with Telemetric Keys (SHA3-512 + HMAC-SHA512):
- 1,966 governance decisions signed
- 0 signatures forged (355 attempts)
- 0 keys extracted (400 attempts)
- 256-bit post-quantum security

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

**Total Attacks**: 1,966
**Attack Success Rate**: 0%
**Statistical Confidence**: 99.9% CI [0%, 0.38%]
