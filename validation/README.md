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

## Files

- **`telos_complete_validation_dataset.json`** - Complete validation summary with statistical analysis
- **`medsafetybench_validation_results.json`** - Detailed MedSafetyBench results (900 attacks)
- **`harmbench_validation_results_summary.json`** - HarmBench aggregate results (400 attacks)

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
