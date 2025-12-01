# TELOSCOPE Beta

**Research Observatory for TELOS Governance Validation**

TELOSCOPE is a purpose-built research instrument for observable AI governance. TELOS (Telically Entrained Linguistic Operational Substrate) is the mathematical governance framework that maintains AI alignment through proportional control and attractor dynamics.

## Architecture

```
TELOSCOPE (Research Instrument)
    |
    +-- Observes and measures
    |
    v
TELOS (Governance Framework)
    |
    +-- Primacy Attractor (constitutional reference)
    +-- Fidelity Measurement (alignment scoring)
    +-- Proportional Control (drift correction)
    |
    v
LLM (Mistral API)
```

## Quick Start

### 1. Setup Virtual Environment

```bash
cd TELOSCOPE_BETA
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure API Keys

Copy the example secrets file and add your Mistral API key:

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edit secrets.toml with your actual MISTRAL_API_KEY
```

Or set as environment variable:
```bash
export MISTRAL_API_KEY="your-key-here"
```

### 3. Launch

```bash
streamlit run main.py
```

## Project Structure

```
TELOSCOPE_BETA/
├── main.py                     # Streamlit application entry point
├── core/                       # State management and async processing
├── components/                 # UI components (observation deck, panels)
├── services/                   # Backend services (PA extractor, A/B testing)
├── config/                     # Configuration and color schemes
├── demo_mode/                  # Demo corpus and RAG loader
├── telos_purpose/              # TELOS governance implementation
│   ├── core/                   # Mathematical primitives
│   │   ├── primacy_math.py     # Attractor mathematics
│   │   ├── intervention_controller.py
│   │   └── unified_orchestrator_steward.py
│   ├── dev_dashboard/          # Developer analysis tools
│   ├── validation/             # Test runners and validation
│   └── llm_clients/            # LLM API integrations
└── beta_testing/               # Beta test management
```

## Core Concepts

### Primacy Attractor (PA)
A fixed reference point in embedding space representing session-level constitutional constraints (purpose, scope, boundaries). All AI responses are measured against this reference.

### Telic Fidelity
Cosine similarity between response embeddings and the Primacy Attractor. Range [0,1] where higher values indicate better alignment.

### Proportional Control
Intervention strength scales with drift magnitude:
- `F > 0.8`: No intervention needed
- `0.5 < F < 0.8`: Gentle reminder injection
- `F < 0.5`: Response regeneration

## Development

### Running Tests

```bash
python -m pytest telos_purpose/validation/
```

### Configuration

Edit `config/` files to adjust:
- Governance parameters (tolerance, thresholds)
- UI styling
- Backend endpoints

## Requirements

- Python 3.9+
- Streamlit
- Mistral API access
- sentence-transformers (for embeddings)

## License

MIT License

---

**Version**: Beta
**Framework**: TELOS (Telically Entrained Linguistic Operational Substrate)
