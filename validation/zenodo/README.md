# TELOS SB 243 Child Safety Validation Dataset

Reproducible validation artifacts for TELOS geometric governance against California SB 243 child safety harm categories.

## Key Results

| Metric | Value |
|--------|-------|
| Attack Success Rate | 0.00% |
| False Positive Rate | 74.00% |
| Violation Defense Rate | 100.00% |

## Contents

### Test Data
- `sb243_data/sb243_test_attacks.csv` - 50 harmful attack prompts (suicide, self-harm, sexual content, eating disorders)
- `sb243_data/sb243_benign_contrastive.csv` - 50 benign helper/educational queries

### Configuration
- `config/sb243_child_safety_pa_config.json` - Primacy Attractor definition for SB 243 compliance

### Validation Scripts
- `run_sb243_validation.py` - Attack validation (measures ASR)
- `run_false_positive_validation.py` - False positive validation (measures FPR)

### Results
- `sb243_validation_results.json` - Detailed attack validation results
- `sb243_false_positive_results.json` - Detailed FPR results

## Reproduction

```bash
pip install sentence-transformers numpy

# Run attack validation
python run_sb243_validation.py

# Run false positive validation
python run_false_positive_validation.py
```

## Embedding Model

All validation uses `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions).

## Governance Thresholds

- Tier 1 (Autonomous Block): fidelity >= 0.25
- Tier 2 (Review Block): fidelity in [0.15, 0.25)
- Tier 3 (Expert Escalation): fidelity < 0.15

## Regulatory Context

California SB 243 (Chapter 883, Statutes of 2024) - Child safety requirements for social media platforms.

## Citation

If you use this dataset, please cite the TELOS project:
- GitHub: https://github.com/TelosSteward/TELOS

## License

CC-BY 4.0

---

*Validation Date: 2024-12-22*
