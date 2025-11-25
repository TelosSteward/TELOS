# TELOS - Runtime AI Governance Framework

**Statistical Process Control for AI Systems**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17702890.svg)](https://doi.org/10.5281/zenodo.17702890)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Validation](https://img.shields.io/badge/ASR-0%25-success.svg)](https://github.com/TelosSteward/TELOS-Validation)

**0% Attack Success Rate** | **1,300 Adversarial Attacks** | [See Validation Results →](docs/whitepapers/TELOS_Technical_Paper.md)

---

## What is TELOS?

TELOS provides **runtime governance** for AI systems—continuous, measurable oversight during operation using Statistical Process Control (SPC) methodology proven over 70+ years in aerospace, nuclear, and medical devices.

**The Problem:** You can't govern an autonomous agent with a quarterly audit. Current AI governance is post-hoc—violations discovered weeks after they occur.

**The Solution:** TELOS treats semantic drift as measurable process variation. A **Primacy Attractor** encodes constitutional constraints as a fixed reference point in embedding space. Every response is measured against this reference. Drift triggers graduated intervention.

**The Result:** 0% attack success rate across 1,300 adversarial attacks with 99.9% confidence interval [0%, 0.28%].

---

## Quick Start

```bash
# Clone repository
git clone https://github.com/TelosSteward/TELOS.git
cd TELOS

# Install dependencies
pip install -r requirements.txt

# Configure API key (choose one)
export MISTRAL_API_KEY="your-key"
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"

# Launch TELOSCOPE Observatory
cd TELOSCOPE_BETA
streamlit run main.py
```

Navigate to `http://localhost:8501`

---

## Example: Healthcare Governance

```python
from telos.core.unified_steward import UnifiedGovernanceSteward
from telos.core.governance_config import GovernanceConfig
from telos.core.primacy_math import PrimacyAttractorMath

# Define constitutional constraints
config = GovernanceConfig(
    mode="dual_pa",
    user_pa_threshold=0.65,
    ai_pa_threshold=0.70,
    tolerance=0.15,
    intervention_threshold=0.85
)

# Create Primacy Attractor with healthcare boundaries
pa = PrimacyAttractorMath(
    purpose_vector="Assist with clinical documentation formatting only",
    constraint_tolerance=0.2,
    constraint_rigidity=0.8
)

# Initialize governance steward
steward = UnifiedGovernanceSteward(
    attractor=pa,
    llm_client=your_llm_client,
    embedding_provider=your_embedder
)

# Runtime enforcement
steward.start_session()
result = steward.process_turn(user_input, model_response)
print(f"Fidelity: {result.get('fidelity_score', 0):.2%}")
```

---

## How It Works

**Dual-Attractor System:**
- **Primacy Attractor**: Human-defined constitutional law encoded in embedding space
- **Continuous Measurement**: Every response measured via cosine similarity
- **Proportional Control**: Graduated intervention (gentle → strong → regenerate)
- **Statistical Process Control**: Real-time fidelity scoring, drift detection, control limits

**Architecture:**
```
[Your Application]
        ↓
[TELOS Orchestration Layer] ← Governance operates here
        ↓
[Frontier LLM] (OpenAI, Anthropic, Mistral - unmodified)
```

TELOS operates at the orchestration layer—middleware between your app and frontier LLMs. No model modifications required.

---

## Validation

**Adversarial Testing**: 1,300 attacks across 2 benchmark suites validated with 0% ASR.

| Benchmark | Attacks | Source | ASR |
|-----------|---------|--------|-----|
| MedSafetyBench | 900 | NeurIPS 2024 | 0% |
| HarmBench | 400 | CAIS | 0% |

**Statistical confidence**: 99.9% CI [0%, 0.28%]

See [Technical Paper](docs/whitepapers/TELOS_Technical_Paper.md) for complete methodology and forensic traces.

---

## Documentation

**Core Docs:**
- [Technical Whitepaper](docs/whitepapers/TELOS_Whitepaper.md) - Mathematical foundations
- [Technical Paper](docs/whitepapers/TELOS_Technical_Paper.md) - Complete validation evidence
- [Implementation Guide](docs/guides/Implementation_Guide.md) - Integration patterns

**Compliance:**
- [EU AI Act (Article 72)](docs/regulatory/EU_Article72_Submission.md)

**Research:**
- [TELOSCOPE Observatory](TELOSCOPE_BETA/README.md) - Research instrument for governance validation

---

## Use Cases

**Healthcare**: HIPAA-compliant clinical documentation assistance with governance validation

**Finance**: GLBA-compliant transaction monitoring with real-time drift detection

**Government**: Multi-level security with clearance-aware constitutional boundaries

**Enterprise**: Autonomous agent governance for LangChain, AutoGPT deployments

---

## Repository Structure

```
TELOS/
├── telos/                  # Core governance engine
│   ├── core/               # Dual-attractor dynamics, SPC, steward
│   └── utils/              # Embedding, similarity, logging
├── TELOSCOPE_BETA/         # Observatory research interface
│   ├── components/         # UI components
│   ├── services/           # Backend services
│   └── telos_purpose/      # Governance implementation
├── docs/                   # Documentation
│   ├── whitepapers/        # Technical papers
│   ├── guides/             # Implementation guides
│   └── regulatory/         # Compliance docs
└── security/               # Security validation
```

---

## Regulatory Compliance

**California SB 53** (January 2026): Active governance mechanisms with quantitative evidence

**EU AI Act Article 72** (August 2026): Continuous post-market monitoring requirements

**FDA QSR / ISO 13485**: Statistical process control for medical device AI

TELOS provides the measurement infrastructure regulatory frameworks require: continuous monitoring, quantitative evidence, real-time intervention, complete audit trails.

---

## Citation

```bibtex
@software{telos2025,
  author    = {Brunner, Jeffrey},
  title     = {TELOS: Statistical Process Control for AI Governance},
  year      = {2025},
  publisher = {GitHub},
  url       = {https://github.com/TelosSteward/TELOS}
}
```

**Validation Dataset:**
```bibtex
@dataset{telos2025validation,
  author    = {Brunner, Jeffrey},
  title     = {TELOS Adversarial Validation Dataset},
  year      = {2025},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.17702890}
}
```

---

## Contact

**GitHub**: [github.com/TelosSteward/TELOS-Validation/issues](https://github.com/TelosSteward/TELOS-Validation/issues)

**For:**
- Research collaborations
- Enterprise licensing
- Grant reviewer access
- Technical questions

---

## License

MIT License - Use freely for research and commercial applications. See [LICENSE](LICENSE).

**Validation Dataset**: CC BY 4.0 - [doi.org/10.5281/zenodo.17702890](https://doi.org/10.5281/zenodo.17702890)

---

**Runtime governance. Measurable control. Human authority.**
