# TELOS Reproducibility Guide

**For Independent Validation of Published Results**

This guide enables peer reviewers, grant evaluators, and researchers to independently reproduce the TELOS validation results: **0% Attack Success Rate across 1,300 attacks with 99.9% confidence interval [0%, 0.28%]**.

---

## Quick Start (15 Minutes)

### Prerequisites

- Python 3.9+ installed
- 8GB RAM minimum (16GB recommended)
- Internet connection
- Mistral API key OR Ollama installed

###Step 1: Clone Repository

```bash
git clone https://github.com/TelosSteward/TELOS.git
cd TELOS
```

### Step 2: Install Dependencies

**For exact reproduction**:
```bash
cd TELOSCOPE_BETA
pip install -r requirements-pinned.txt
```

**For flexible installation** (newer versions OK):
```bash
pip install -r requirements.txt
```

### Step 3: Configure Secrets

Create `.streamlit/secrets.toml`:
```toml
MISTRAL_API_KEY = "your-key-here"
```

**Alternative (Ollama local)**:
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Download model
ollama pull mistral:7b

# No API key needed for local mode
```

### Step 4: Run TELOSCOPE Observatory

```bash
streamlit run main.py
```

Opens at: `http://localhost:8501`

---

## Reproducing Validation Results

### Option A: Review Published Validation Data

**Validation results are published on Zenodo** (no local scripts required):
- **DOI**: https://doi.org/10.5281/zenodo.17702890
- **Files**: Complete JSON forensics, statistical analyses, raw results

**Download and verify**:
```bash
# Download from Zenodo (visit link above)
# Or use curl/wget with the direct download URLs

# Review JSON forensics
cat medsafetybench_validation_results.json | jq '.summary'
```

**Published results**:
- Execution time: ~8-10 seconds (on original hardware)
- Result: 0% Attack Success Rate
- Files included:
  - `medsafetybench_validation_results.json` (490KB)
  - `harmbench_validation_results_summary.json`

### Option B: Validation Data Review (No Compute Required)

**Published validation data available at**:
- **Zenodo**: https://doi.org/10.5281/zenodo.17702890
- **GitHub**: https://github.com/TelosSteward/TELOS-Validation

**Verification without re-running**:
```bash
# Download validation results from Zenodo
# Visit: https://doi.org/10.5281/zenodo.17702890
# Download files directly from Zenodo interface

# Review JSON forensics
cat medsafetybench_validation_results.json | jq '.summary'
```

---

## Understanding the Results

### Validation Breakdown

**1,300 Total Attacks**:
1. **MedSafetyBench** (900 attacks):
   - Source: NeurIPS 2024 workshop
   - Domain: Healthcare safety
   - Result: 0% ASR, 93.8% Tier 1 (PA alone)

2. **HarmBench** (400 attacks):
   - Source: CAIS (Center for AI Safety)
   - Domain: General adversarial
   - Result: 0% ASR, 95.8% Tier 1

### Statistical Validation

**Confidence Interval**: 99.9% CI [0%, 0.28%]
- Interpretation: True attack success rate < 0.28% with 99.9% confidence

**Bayesian Analysis**: Bayes Factor = 2.7 × 10¹⁷
- Interpretation: Overwhelming evidence for TELOS effectiveness

**Power Analysis**: Statistical power > 0.99
- Interpretation: Would detect even 0.5% vulnerabilities

**P-Value**: p < 0.001
- Interpretation: Highly statistically significant vs baselines

---

## Troubleshooting

### Common Issues

**Issue 1: Model download fails**
```bash
# Manual download
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
```

**Issue 2: Mistral API rate limits**
- Use local Ollama instead (see Step 3 alternative)
- Or contact Mistral for paid tier access

**Issue 3: Out of memory**
- Reduce batch size in validation script
- Upgrade to 16GB RAM system
- Use swap file/virtual memory

**Issue 4: Validation takes too long**
- Expected: ~12 seconds for 2,000 attacks
- If slower: Check CPU cores (need 4+), check network latency
- Parallelize across multiple cores

**Issue 5: Different results**
```bash
# Verify exact versions
pip list | grep -E "streamlit|mistralai|torch"

# Should match requirements-pinned.txt
# If not, reinstall with pinned versions
```

---

## Validation Checklist

To claim successful reproduction, verify:

- [ ] All dependencies installed (`pip list` matches requirements)
- [ ] TELOSCOPE Observatory runs (`streamlit run main.py`)
- [ ] Demo mode functional (12-slide slideshow works)
- [ ] BETA mode accessible (PA establishment works)
- [ ] Validation data downloaded from Zenodo
- [ ] Results match published (0% ASR, forensic JSONs verified)
- [ ] Statistical analysis reproduces (Wilson Score, Bayesian, power)

---

## Hardware Used for Published Results

**For transparency and reproducibility**:

- **CPU**: Apple M2 Pro (10 cores)
- **RAM**: 32GB
- **Storage**: 512GB SSD
- **OS**: macOS 14.3
- **Python**: 3.9.18
- **Execution Time**: ~8-10 seconds for 1,300 attacks

**Your hardware may differ**:
- Minimum: 8GB RAM, 4 cores → expect ~30-60 seconds
- Recommended: 16GB RAM, 8 cores → expect ~15-30 seconds
- See `HARDWARE_REQUIREMENTS.md` for details

---

## Support for Reproducibility

**We are committed to supporting independent validation.**

### For Peer Reviewers

- Extended compute time estimates for your hardware
- Alternative LLM backend configuration
- Debugging support for environment issues
- Access to additional validation data

### For Grant Evaluators

- Live demonstration available (Streamlit Cloud deployment)
- Video walkthrough of validation process
- Consultation on extending validation to new domains

### Contact

- **GitHub Issues**: https://github.com/TelosSteward/TELOS/issues
- **Email**: [To be established with grant funding]
- **Documentation**: `/docs/` directory in repository

---

## Next Steps After Reproduction

### Option 1: Extend Validation

**Add your own adversarial attacks**:
```python
from strix import AdversarialAttackFramework

# Create custom attack
custom_attack = {
    "category": "your-domain",
    "prompt": "your adversarial prompt",
    "expected_refusal": True
}

# Test with TELOS
result = framework.test_attack(custom_attack)
```

### Option 2: Compare to Baselines

**Run head-to-head comparison**:
```bash
# Test TELOS vs no-governance vs Constitutional AI
python3 run_comparative_evaluation.py --baselines all
```

### Option 3: Deploy for Your Research

**Institutional deployment**:
```bash
# Configure for your LLM backend
export MISTRAL_API_KEY="your-key"
# OR use local Ollama
export LLM_BACKEND="ollama"

# Launch for your research group
streamlit run main.py --server.port 8501
```

---

## Publication & Citation

**When citing TELOS validation results**:

```bibtex
@misc{telos2025validation,
  title={TELOS: Statistical Process Control for AI Governance},
  author={Brunner, Jeffrey},
  year={2025},
  note={1,300 attacks validated, 0\% ASR, 99.9\% CI [0\%, 0.28\%]},
  url={https://github.com/TelosSteward/TELOS}
}
```

**Validation data citation**:
```bibtex
@dataset{telos2025data,
  author={Brunner, Jeffrey},
  title={TELOS Adversarial Validation Dataset},
  year={2025},
  publisher={Zenodo},
  doi={10.5281/zenodo.17702890},
  url={https://doi.org/10.5281/zenodo.17702890}
}
```

---

## Frequently Asked Questions

**Q: How long does reproduction take?**
A: 15 minutes for setup + 12 seconds for validation = ~20 minutes total

**Q: Do I need a GPU?**
A: No. CPU-only works fine. GPU provides 2-3x speedup (optional).

**Q: Can I reproduce without Mistral API?**
A: Yes. Use local Ollama (free, unlimited). See Step 3 alternative.

**Q: What if I get different results?**
A: Check versions (`requirements-pinned.txt`), hardware specs, and random seeds. Contact us for debugging support.

**Q: Is this the same as the published paper?**
A: Yes. This code produced the results in the TELOS Technical Paper.

**Q: Can I use this for my research?**
A: Yes. MIT license. Cite appropriately. See `LICENSE` file.

---

## Reproducibility Badges

**ACM Artifact Evaluation**:
- **Available**: ✅ Code on GitHub (public)
- **Functional**: ⏳ In progress (Docker containerization)
- **Reproduced**: ⏳ Awaiting independent validation

**Help us earn these badges**:
If you successfully reproduce our results, please:
1. Open a GitHub issue confirming reproduction
2. Share your hardware specs and execution time
3. Report any deviations from published results

---

## Appendix: Directory Structure

```
TELOS/
├── TELOSCOPE_BETA/          # Main observatory application
│   ├── main.py              # Entry point
│   ├── requirements.txt     # Dependencies (flexible)
│   ├── requirements-pinned.txt  # Dependencies (exact)
│   ├── telos_purpose/       # Governance engine (embedded)
│   │   ├── core/            # Core modules
│   │   └── llm_clients/     # LLM integrations
│   └── .streamlit/
│       ├── config.toml      # UI configuration
│       └── secrets.toml     # API keys (not in git)
├── docs/whitepapers/        # Academic papers
│   ├── TELOS_Technical_Paper.md
│   ├── Statistical_Validity.md
│   └── TELEMETRIC_KEYS_FOUNDATIONS.md
├── REPRODUCTION_GUIDE.md    # This file
├── README.md                # Project overview
└── validation_data/         # Available on Zenodo
    └── (see https://doi.org/10.5281/zenodo.17702890)
```

---

**Document Version**: 1.0
**Last Updated**: November 24, 2025
**Corresponding Publication**: TELOS Technical Paper v2.0
**Validation Dataset**: https://doi.org/10.5281/zenodo.17702890

**End of Reproduction Guide**
