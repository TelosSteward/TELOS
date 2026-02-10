# TELOS Observatory

**Runtime AI Governance Framework with Published Validation**

![Python](https://img.shields.io/badge/python-3.8+-green)
![License](https://img.shields.io/badge/license-Apache%202.0-orange)
[![Scenarios](https://img.shields.io/badge/scenarios-2%2C550-blue)](https://github.com/TelosSteward/TELOS)
[![ASR](https://img.shields.io/badge/attack%20success%20rate-0%25-brightgreen)](https://github.com/TelosSteward/TELOS)

[![AILuminate](https://img.shields.io/badge/AILuminate-900-blue)](https://doi.org/10.5281/zenodo.18370263)
[![Adversarial](https://img.shields.io/badge/Adversarial-2%2C550-blue)](https://doi.org/10.5281/zenodo.18370659)
[![Governance](https://img.shields.io/badge/Governance-46_sessions-blue)](https://doi.org/10.5281/zenodo.18009153)
[![SB 243](https://img.shields.io/badge/SB_243-100-blue)](https://doi.org/10.5281/zenodo.18370504)
[![XSTest](https://img.shields.io/badge/XSTest-FPR_8%25-blue)](https://doi.org/10.5281/zenodo.18370603)

---

TELOS (Telically Entrained Linguistic Operational Substrate) is a mathematical governance framework for AI alignment. It uses **Primacy Attractors** -- embedding-space representations of user purpose -- to measure conversational fidelity via cosine similarity and detect drift in real-time. When a user's input deviates from their stated purpose, TELOS applies proportional interventions calibrated by a two-layer fidelity system. The framework has been validated against 2,550 adversarial scenarios across five peer-reviewed benchmarks with a 0% attack success rate, and all validation data is published with DOIs on Zenodo for independent verification.

---

## Quick Start

### Option 1: Run Locally

```bash
git clone https://github.com/TelosSteward/TELOS.git
cd TELOS

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

cp .env.example .env
# Edit .env and add your MISTRAL_API_KEY

export PYTHONPATH=$(pwd)
streamlit run telos_observatory/main.py --server.port 8501
```

Opens in browser at `http://localhost:8501`

### Option 2: Hosted Beta

For evaluation without local setup: **[beta.telos-labs.ai](https://beta.telos-labs.ai)**

*Note: The hosted beta may experience latency under load. For development or integration testing, we recommend running locally with your own API key.*

### Option 3: Reproduce Validation Results

See the [Reproduction Guide](docs/REPRODUCTION_GUIDE.md) for step-by-step instructions.

```bash
# Run unit tests (no external dependencies)
PYTHONPATH=. pytest tests/ -v

# Run SB 243 child safety validation
cd validation/ && python3 run_sb243_validation.py

# Run XSTest over-refusal calibration (250 prompts)
cd validation/ && python3 run_xstest_validation.py

# Adversarial validation results are pre-computed in validation/
# See validation/telos_complete_validation_dataset.json
```

---

## Architecture

### Package Structure

```
telos/
├── telos_core/              # Pure mathematical engine (zero framework dependencies)
│   ├── constants.py         #   All calibration thresholds
│   ├── primacy_math.py      #   Attractor geometry, basin membership
│   ├── fidelity_engine.py   #   Two-layer fidelity calculation
│   ├── embedding_provider.py #  Multi-model embedding interface
│   └── governance_trace.py  #   Audit trail structure
├── telos_governance/        # Governance decision gates
│   ├── fidelity_gate.py     #   Two-tier conversational gate
│   └── governance_protocol.py #  Audit-trail protocol
├── telos_gateway/           # FastAPI API gateway (OpenAI-compatible)
│   ├── routes/              #   Health, chat completions
│   └── providers/           #   Mistral, OpenAI provider adapters
├── telos_adapters/          # Framework integration adapters
│   ├── langgraph/           #   LangGraph wrapper
│   └── generic/             #   @telos_governed decorator
├── telos_observatory/       # Streamlit UI
│   ├── main.py              #   App entrypoint
│   ├── components/          #   UI components
│   ├── services/            #   Backend services (fidelity, steward, PA)
│   ├── demo_mode/           #   Demo slides + corpus
│   └── config/              #   PA templates, calibration phrases
├── tests/                   # Unit and integration tests
└── validation/              # Benchmark datasets and results
```

### Dependency Flow

```
telos_core  <--  telos_governance  <--  telos_gateway
                                   <--  telos_adapters
                                   <--  telos_observatory
```

`telos_core` is a pure mathematical library with no framework dependencies. All governance logic flows through `telos_governance`, which higher-level packages consume.

---

## Two-Layer Fidelity System

TELOS uses two detection layers to distinguish adversarial content from legitimate purpose drift.

**Layer 1: Baseline Pre-Filter**
- Constant: `SIMILARITY_BASELINE = 0.20`
- Catches content completely outside the Primacy Attractor embedding space
- Raw cosine similarity < 0.20 triggers immediate hard block

**Layer 2: Basin Membership**
- Constant: `INTERVENTION_THRESHOLD = 0.48`
- Detects when the user has drifted from their stated purpose
- Fidelity < 0.48 triggers a proportional governance intervention

**Intervention Decision:**
```python
should_intervene = (raw_similarity < 0.20) OR (fidelity < 0.48)
```

### Fidelity Zones

| Zone | Fidelity Range | Color | Meaning |
|------|----------------|-------|---------|
| GREEN | >= 0.70 | `#27ae60` | Aligned with purpose |
| YELLOW | 0.60 -- 0.69 | `#f39c12` | Minor drift detected |
| ORANGE | 0.50 -- 0.59 | `#e67e22` | Drift detected, redirect |
| RED | < 0.50 | `#e74c3c` | Significant drift, block + review |

---

## Validation Results

### Adversarial Benchmarks (2,550 scenarios, 0% ASR)

TELOS was validated against five peer-reviewed adversarial benchmarks. All validation data is published on Zenodo with DOIs for independent verification.

- **Total Adversarial Scenarios**: 2,550
- **Attack Success Rate**: 0.00%
- **Statistical Confidence**: 99.9% CI [0%, 0.14%]
- **Forensic Audit Trails**: Full JSONL traces for every governance decision

| Benchmark | Scenarios | Blocked | Venue | DOI |
|-----------|-----------|---------|-------|-----|
| AILuminate | 900 | 900 (100%) | MLCommons | [10.5281/zenodo.18370263](https://doi.org/10.5281/zenodo.18370263) |
| HarmBench | 400 | 400 (100%) | Center for AI Safety | [10.5281/zenodo.18370659](https://doi.org/10.5281/zenodo.18370659) |
| MedSafetyBench | 900 | 900 (100%) | NeurIPS 2024 | [10.5281/zenodo.18370659](https://doi.org/10.5281/zenodo.18370659) |
| SB 243 Child Safety | 100 | 100 (100%) | California Legislature | [10.5281/zenodo.18370504](https://doi.org/10.5281/zenodo.18370504) |
| XSTest | 250 | Calibration benchmark | NAACL 2024 | [10.5281/zenodo.18370603](https://doi.org/10.5281/zenodo.18370603) |

### XSTest Over-Refusal Calibration

XSTest measures false positive rates -- how often the system incorrectly blocks safe prompts.

- **False Positive Rate**: 8.0% (Healthcare PA) vs. 24.8% (Generic PA)
- **Improvement**: 16.8 percentage point reduction via domain-specific Primacy Attractor calibration
- **DOI**: [10.5281/zenodo.18370603](https://doi.org/10.5281/zenodo.18370603)

### Governance Benchmark Dataset

- **Sessions**: 46 multi-session governance evaluations
- **Domains**: 8 diverse application domains
- **DOI**: [10.5281/zenodo.18009153](https://doi.org/10.5281/zenodo.18009153)

---

## Key Documentation

| Document | Description |
|----------|-------------|
| [Whitepaper v2.5 (PDF)](docs/TELOS_Whitepaper_v2.5.pdf) | Complete mathematical specification and validation details |
| [Academic Paper (PDF)](docs/TELOS_Academic_Paper.pdf) | Peer-review ready paper with full methodology and results |
| [Reproduction Guide](docs/REPRODUCTION_GUIDE.md) | Step-by-step instructions to independently verify all results |

---

## Citation

For the adversarial validation dataset:

```bibtex
@dataset{brunner_2025_telos_adversarial,
  author       = {Brunner, Jeffrey},
  orcid        = {0009-0003-6848-8014},
  title        = {{TELOS Adversarial Validation Dataset}},
  month        = nov,
  year         = 2025,
  publisher    = {Zenodo},
  version      = {1.0},
  doi          = {10.5281/zenodo.18370659},
  url          = {https://doi.org/10.5281/zenodo.18370659}
}
```

For the governance benchmark dataset:

```bibtex
@dataset{brunner_2025_telos_governance,
  author       = {Brunner, Jeffrey},
  orcid        = {0009-0003-6848-8014},
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
  orcid        = {0009-0003-6848-8014},
  title        = {{TELOS SB 243 Child Safety Validation Dataset}},
  month        = dec,
  year         = 2025,
  publisher    = {Zenodo},
  version      = {1.0},
  doi          = {10.5281/zenodo.18370504},
  url          = {https://doi.org/10.5281/zenodo.18370504}
}
```

For the XSTest over-refusal calibration:

```bibtex
@dataset{brunner_2026_telos_xstest,
  author       = {Brunner, Jeffrey},
  orcid        = {0009-0003-6848-8014},
  title        = {{TELOS XSTest Over-Refusal Calibration Dataset}},
  month        = jan,
  year         = 2026,
  publisher    = {Zenodo},
  version      = {1.0},
  doi          = {10.5281/zenodo.18370603},
  url          = {https://doi.org/10.5281/zenodo.18370603}
}
```

---

## License

Apache License 2.0 -- See [LICENSE](LICENSE) file.

---

## Author

**Jeffrey Brunner**
TELOS AI Labs Inc.
[![ORCID](https://img.shields.io/badge/ORCID-0009--0003--6848--8014-green?logo=orcid)](https://orcid.org/0009-0003-6848-8014)
Email: JB@telos-labs.ai
