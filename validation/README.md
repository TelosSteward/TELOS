# TELOS Validation Dataset

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17702890.svg)](https://doi.org/10.5281/zenodo.17702890)
[![Governance DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18009153.svg)](https://doi.org/10.5281/zenodo.18009153)
[![Attacks](https://img.shields.io/badge/attacks-1%2C300-blue)](https://github.com/TelosSteward/TELOS-Validation)
[![ASR](https://img.shields.io/badge/attack%20success%20rate-0%25-brightgreen)](https://github.com/TelosSteward/TELOS-Validation)
[![License](https://img.shields.io/badge/license-CC%20BY%204.0-orange)](https://creativecommons.org/licenses/by/4.0/)

Official validation data for TELOS AI governance framework.

---

## Results Summary

- **Total Attacks**: 1,300
- **Attack Success Rate**: 0.00%
- **Statistical Confidence**: 99.9% CI [0%, 0.28%]
- **Autonomous Blocking**: 95.8% (Tier 1)

---

## Benchmarks

### MedSafetyBench (NeurIPS 2024)
- **Attacks**: 900
- **Blocked**: 900 (100%)
- **Source**: [AI4LIFE-GROUP/med-safety-bench](https://github.com/AI4LIFE-GROUP/med-safety-bench)

### HarmBench (Center for AI Safety)
- **Attacks**: 400
- **Blocked**: 400 (100%)
- **Source**: [centerforaisafety/HarmBench](https://github.com/centerforaisafety/HarmBench)

---

## Reproducing the Validation

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

### Run Validation

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

## Files

- **`telos_validation_dataset_zenodo.json`** - Consolidated Zenodo dataset (v2.0) with all 1,300 per-attack traces
- **`telos_complete_validation_dataset.json`** - Complete validation summary with statistical analysis
- **`medsafetybench_validation_results.json`** - Detailed MedSafetyBench results (900 attacks)
- **`harmbench_validation_results.json`** - HarmBench results (400 attacks)

---

## Zenodo Datasets

| Dataset | DOI | Description |
|---------|-----|-------------|
| **Adversarial Validation** | [10.5281/zenodo.17702890](https://doi.org/10.5281/zenodo.17702890) | 1,300 adversarial attacks, 0% ASR |
| **Governance Benchmark** | [10.5281/zenodo.18009153](https://doi.org/10.5281/zenodo.18009153) | 46 multi-session governance evaluations |

---

## Related Repositories

- **[TELOS Observatory](https://github.com/TelosSteward/Observatory)** - Main TELOS framework with reproduction scripts and documentation

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

**Creative Commons Attribution 4.0 International (CC BY 4.0)**

---

**Last Updated**: December 21, 2025
**Dataset Version**: 1.0
