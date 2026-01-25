# TELOS AILuminate Validation Dataset

**DOI:** [10.5281/zenodo.18370263](https://doi.org/10.5281/zenodo.18370263)

## Overview

This dataset contains validation results from testing the TELOS three-tier governance framework against the MLCommons AILuminate v1.0 DEMO benchmark - the industry-standard AI safety evaluation used by major AI companies.

**Citation (this dataset):**
```
Brunner, J.F. (2026). TELOS AILuminate Validation Dataset: MLCommons AI Safety
Benchmark with Forensic Audit Trail. Zenodo. https://doi.org/10.5281/zenodo.18370263
```

**Citation (AILuminate benchmark):**
```
MLCommons AI Safety Working Group
AILuminate: Standardized AI Safety Benchmarking
GitHub: https://github.com/mlcommons/ailuminate
Version: v1.0 DEMO (1,200 prompts)
```

## Epistemological Note

**IMPORTANT:** This validation demonstrates the *mechanistic integrity* of the TELOS governance architecture, NOT claims of domain expertise. We make no claim that TELOS "understands" specific harm categories - only that it can detect semantic proximity to configured governance constraints and route queries through appropriate defense layers.

The Healthcare HIPAA PA configuration serves as a **domain-agnostic safety baseline** that demonstrates the Primacy Attractor mechanism's ability to identify harmful content across diverse categories.

## Key Results

| Metric | Value |
|--------|-------|
| Total Prompts Tested | 1,200 |
| Attack Success Rate (ASR) | 0.00% |
| Violation Defense Rate (VDR) | 100.00% |
| Statistical Confidence | 99.9% CI [0%, 0.28%] |

### Tier Distribution

| Tier | Count | Percentage | Description |
|------|-------|------------|-------------|
| Tier 1 (PA Block) | 1,200 | 100.0% | Mathematical enforcement via embedding similarity |
| Tier 2 (RAG Block) | 0 | 0.0% | Policy retrieval (not triggered) |
| Tier 3 (Expert Escalation) | 0 | 0.0% | Human expert required (not triggered) |
| Attacks Passed | 0 | 0.0% | Successful attacks bypassing all tiers |

## AILuminate Hazard Categories

### Core Categories (100 prompts each)

| Code | Category | Blocked | ASR |
|------|----------|---------|-----|
| cse | Child Sexual Exploitation | 100/100 | 0% |
| dfm | Defamation | 100/100 | 0% |
| hte | Hate Speech | 100/100 | 0% |
| ipv | Interpersonal Violence | 100/100 | 0% |
| iwp | Indiscriminate Weapons (CBRNE) | 100/100 | 0% |
| ncr | Non-Violent Crimes | 100/100 | 0% |
| prv | Privacy Violations | 100/100 | 0% |
| src | Sexual Content | 100/100 | 0% |
| ssh | Suicide & Self-Harm | 100/100 | 0% |
| sxc_prn | Sex Crimes / Pornography | 100/100 | 0% |
| vcr | Violent Crimes | 100/100 | 0% |

### Specialized Categories

| Code | Category | Blocked | ASR |
|------|----------|---------|-----|
| spc_fin | Specialized: Financial | 26/26 | 0% |
| spc_hlt | Specialized: Health | 26/26 | 0% |
| spc_ele | Specialized: Elections | 24/24 | 0% |
| spc_lgl | Specialized: Legal | 24/24 | 0% |

### Attacker Persona Breakdown

| Persona | Count | Blocked | ASR |
|---------|-------|---------|-----|
| skilled | 600 | 600 | 0% |
| unskilled | 600 | 600 | 0% |

## Methodology

### Embedding Model
- **Model:** nomic-embed-text (768-dimensional)
- **Provider:** Ollama (local inference)
- **Note:** Different embedding models produce different similarity distributions; thresholds must be calibrated per model

### Primacy Attractor Configuration
- **Domain:** Healthcare HIPAA (used as safety baseline)
- **Purpose:** HIPAA-compliant clinical information support
- **Configuration File:** `healthcare_hipaa_pa_config.json`

### Threshold Configuration

| Threshold | Value | Action |
|-----------|-------|--------|
| Tier 1 (PA Block) | fidelity >= 0.18 | Mathematical enforcement |
| Tier 2 (RAG) | 0.12 <= fidelity < 0.18 | Policy retrieval |
| Tier 3 (Expert) | fidelity < 0.12 | Human escalation |

## Dataset Structure

```
zenodo_ailuminate/
├── README.md                              # This file
├── methodology.md                         # Detailed methodology
├── healthcare_hipaa_pa_config.json        # PA configuration used
├── ailuminate_validation_results.json     # Full 1,200-prompt results
├── ailuminate_validation_summary.json     # Aggregated statistics
├── fidelity_distribution.csv              # Per-prompt fidelity scores
└── forensic_output/                       # Full forensic audit trail
    ├── traces/                            # JSONL governance event logs
    │   └── session_*.jsonl                # 4,803 governance events
    ├── ailuminate_forensic_forensic_results.json    # Detailed per-prompt forensics
    ├── ailuminate_forensic_forensic_summary.json    # Aggregate forensic statistics
    ├── ailuminate_forensic_fidelity_distribution.csv # Fidelity score analysis
    └── ailuminate_forensic_governance_report.html   # Interactive HTML report
```

## Forensic Audit Trail

This dataset includes full forensic evidence via the TELOS GovernanceTraceCollector:

| Artifact | Description | Size |
|----------|-------------|------|
| `traces/*.jsonl` | Complete JSONL audit trail with 4,803 governance events | 1.69 MB |
| `*_forensic_results.json` | Per-prompt forensic data with fidelity scores | 458 KB |
| `*_forensic_summary.json` | Aggregate statistics and threshold sensitivity | 2.8 KB |
| `*_fidelity_distribution.csv` | CSV of all fidelity scores for analysis | 76 KB |
| `*_governance_report.html` | Interactive visualization dashboard | 3.2 MB |

### Governance Event Types Recorded

1. `session_start` - Validation session initialization
2. `pa_established` - Primacy Attractor configuration loaded
3. `turn_start` - Each prompt evaluation begins
4. `fidelity_calculated` - Fidelity score computation with embedding details
5. `intervention_triggered` - Blocking decision with tier classification
6. `turn_complete` - Prompt evaluation complete with outcome
7. `session_end` - Validation complete with final statistics

## Comparison with Other Benchmarks

| Benchmark | Prompts | ASR | Source |
|-----------|---------|-----|--------|
| HarmBench (CAIS) | 400 | 0% | Center for AI Safety |
| MedSafetyBench (NeurIPS 2024) | 900 | 0% | AI4LIFE-GROUP |
| SB 243 Child Safety | 50 | 0% | Custom (CA regulation) |
| **AILuminate (MLCommons)** | **1,200** | **0%** | **MLCommons AI Safety** |
| **TOTAL** | **2,550** | **0%** | Combined |

## MLCommons AILuminate Overview

AILuminate is developed by the MLCommons AI Safety Working Group to provide:
- Standardized safety evaluation across AI systems
- Diverse hazard categories covering major harm types
- Persona-based attack variations (skilled vs. unskilled attackers)
- Industry-standard benchmarking used by leading AI companies

The DEMO dataset (1,200 prompts) is publicly available for evaluation purposes.

## Reproducibility

To reproduce these results:

1. Install Ollama and pull nomic-embed-text model:
   ```bash
   ollama pull nomic-embed-text
   ```

2. Download AILuminate DEMO dataset:
   ```bash
   curl -L -o ailuminate_demo.csv \
     "https://raw.githubusercontent.com/mlcommons/ailuminate/main/airr_official_1.0_demo_en_us_prompt_set_release.csv"
   ```

3. Run validation script with the provided PA configuration

4. Compare fidelity scores against threshold classifications

## License

This validation dataset is released under CC-BY-4.0.
AILuminate benchmark is subject to MLCommons license terms.

## Contact

TELOS AI Labs Inc.
- Primary: JB@telos-labs.ai
- General: contact@telos-labs.ai
- GitHub: https://github.com/TelosSteward/TELOS

## Related Publications

- TELOS Paper: DOI 10.5281/zenodo.18367069
- TELOS Adversarial Validation: DOI 10.5281/zenodo.18370659
- TELOS Governance Benchmark: DOI 10.5281/zenodo.18009153
- TELOS SB 243 Child Safety: DOI 10.5281/zenodo.18370504
- TELOS XSTest Calibration: DOI 10.5281/zenodo.18370603

## Validation Date

2026-01-25
