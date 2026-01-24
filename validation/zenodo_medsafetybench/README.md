# TELOS MedSafetyBench Validation Dataset

## Overview

This dataset contains validation results from testing the TELOS three-tier healthcare governance framework against the MedSafetyBench benchmark (NeurIPS 2024 Datasets and Benchmarks Track).

**Citation:**
```
MedSafetyBench: Evaluating and Improving the Medical Safety of LLMs
NeurIPS 2024 Datasets and Benchmarks Track
GitHub: https://github.com/AI4LIFE-GROUP/med-safety-bench
```

## Epistemological Note

**IMPORTANT:** This validation demonstrates the *mechanistic integrity* of the three-tier governance architecture, NOT claims of domain expertise. We make no claim that TELOS "understands" healthcare - only that it can detect semantic proximity to configured governance constraints and route queries through appropriate defense layers.

The reference corpus (8 authoritative public documents: HIPAA, CDC, AMA) serves as a **minimal viable demonstration** of the RAG-based policy retrieval layer. Production deployment would require comprehensive regulatory coverage developed with actual domain experts.

## Key Results

| Metric | Value |
|--------|-------|
| Total Attacks Tested | 900 |
| Attack Success Rate (ASR) | 0.00% |
| Violation Defense Rate (VDR) | 100.00% |

### Full Three-Tier Validation (PA + RAG + Expert)

| Tier | Count | Percentage | Description |
|------|-------|------------|-------------|
| Tier 1 (PA Block) | 207 | 23.0% | Mathematical enforcement via embedding similarity |
| Tier 2 (RAG Block) | 693 | 77.0% | Policy retrieval from healthcare corpus |
| Tier 3 (Expert Escalation) | 0 | 0.0% | Human expert required (none triggered) |
| Attacks Passed | 0 | 0.0% | Successful attacks bypassing all tiers |

### Fidelity Statistics

- **Mean:** 0.6042
- **Min:** 0.4233
- **Max:** 0.7963
- **Std Dev:** 0.0742

## Methodology

### Embedding Model
- **Model:** nomic-embed-text (768-dimensional)
- **Provider:** Ollama (local inference)
- **Note:** Different embedding models produce different similarity distributions; thresholds must be calibrated per model

### Primacy Attractor Configuration
- **Domain:** Healthcare HIPAA
- **Purpose:** HIPAA-compliant patient data access and clinical decision support
- **Configuration File:** `healthcare_hipaa_pa_config.json`

### Threshold Interpretation

The paper specifies thresholds for Mistral embeddings (1024-dim):
- **Tier 1 (PA Block):** fidelity >= 0.65
- **Tier 2 (RAG):** 0.35 <= fidelity < 0.65
- **Tier 3 (Expert):** fidelity < 0.35 with secondary heuristic flags

With nomic-embed-text embeddings, the similarity distribution differs, but the relative classification remains meaningful. All MedSafetyBench attacks fall in the 0.42-0.80 fidelity range, triggering either Tier 1 or Tier 2 defense.

## Dataset Structure

```
zenodo_medsafetybench/
├── README.md                                    # This file
├── methodology.md                               # Detailed methodology & epistemological framing
├── healthcare_hipaa_pa_config.json              # PA configuration used
│
├── # Tier 1 (PA) Classification Results
├── medsafetybench_results_full.json             # 900-attack fidelity scores + tier classification
├── medsafetybench_results_summary.json          # Tier 1 aggregated statistics
│
├── # Full Three-Tier Validation (PA + RAG + Expert)
├── medsafetybench_tier2_validation_full.json    # Complete 900-attack tiered results with RAG retrieval details
├── medsafetybench_tier2_validation_summary.json # Three-tier validation summary
│
└── fidelity_distribution.csv                    # Per-attack fidelity scores
```

### Reference Corpus (Not Redistributed)

The Tier 2 RAG validation used 8 authoritative public documents:
- HHS OCR: HIPAA Privacy Rule Summary, Minimum Necessary Guidance, Individual Rights, Breach Notification
- CDC: Clinical Guidelines (STI, Tuberculosis Treatment 2024)
- AMA: Code of Medical Ethics - Informed Consent
- Joint Commission: Sentinel Event Policy

## MedSafetyBench Categories

| Category | Name | Count |
|----------|------|-------|
| 1 | Patient Care Quality | 100 |
| 2 | Medical Ethics & Professional Conduct | 100 |
| 3 | Regulatory & Legal Compliance | 100 |
| 4 | Drug Safety & Prescription | 100 |
| 5 | Mental Health & Vulnerable Populations | 100 |
| 6 | Privacy & Confidentiality | 100 |
| 7 | Research Ethics & Clinical Trials | 100 |
| 8 | Resource Allocation & Access | 100 |
| 9 | Communication & Informed Consent | 100 |

## Attack Sources

- **GPT-4 Generated:** 450 attacks (prompting-based)
- **Llama2 Jailbreaks:** 450 attacks (jailbreak-based)

## Reproducibility

To reproduce these results:

1. Install Ollama and pull nomic-embed-text model
2. Clone MedSafetyBench: `git clone https://github.com/AI4LIFE-GROUP/med-safety-bench`
3. Run validation script with the provided PA configuration
4. Compare fidelity scores against threshold classifications

## License

This validation dataset is released under CC-BY-4.0.
MedSafetyBench is subject to its own license terms.

## Contact

TELOS AI Labs Inc.
- Primary: JB@telos-labs.ai
- General: contact@telos-labs.ai

## Validation Date

2026-01-23
