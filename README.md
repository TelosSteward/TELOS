# TELOS - Runtime AI Governance System

**Telically Entrained Linguistic Operational Substrate**

[![Status](https://img.shields.io/badge/status-production--ready-green)]()
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)]()
[![License](https://img.shields.io/badge/license-proprietary-red)]()

> Industrial-grade AI alignment through runtime governance and statistical process control

---

## Overview

TELOS is a runtime AI governance system that applies **Lean Six Sigma DMAIC** methodology and **Statistical Process Control (SPC)** to AI alignment. Using a novel **Dual Primacy Attractor** architecture, TELOS provides real-time measurement and proportional intervention to maintain AI system alignment during operation.

### Key Innovation: Dual Primacy Attractor Architecture

- **User PA**: User's conversational purpose (vector in embedding space)
- **AI PA**: AI's role constraints (derived via lock-on formula)
- **Fidelity Measurement**: F = cos(response_embedding, PA)
- **Proportional Intervention**: Scaled to deviation magnitude

### Validated Performance

- **+85.32%** improvement over baseline (60+ validation studies)
- Real-time alignment monitoring and correction
- Production-tested with Observatory interface

---

## Project Structure

```
TELOS/
├── telos/                    # Core Python package
│   ├── core/                 # Runtime governance engine
│   ├── crypto/               # Telemetric Keys cryptography
│   ├── llm/                  # LLM client adapters
│   ├── profiling/            # PA extraction & analysis
│   └── storage/              # Storage adapters
│
├── observatory/              # Observatory UI (Streamlit)
├── dev_dashboard/            # Development dashboard
├── steward/                  # Steward orchestration
│   ├── steward_pm.py        # Project manager assistant
│   └── steward.py           # Main steward
│
├── tests/                    # Comprehensive test suite
├── docs/                     # Documentation
│   └── audit/               # OPUS audit findings
├── scripts/                  # Utility scripts
└── config/                   # Configuration files
```

---

## Quick Start

### Prerequisites

- Python 3.9+
- OpenAI API key (for embeddings)
- Anthropic API key (optional, for Claude integration)

### Installation

```bash
# Clone repository
git clone https://github.com/TelosSteward/TELOS.git
cd TELOS

# Install dependencies
pip install -r requirements.txt

# Set up configuration
cp config/governance_config.example.json config/governance_config.json
# Edit config/governance_config.json with your API keys
```

### Launch Observatory

```bash
./scripts/launch_dashboard.sh
```

### Launch Development Dashboard

```bash
./scripts/launch_dev_dashboard.sh
```

---

## Core Components

### 1. Runtime Governance (`telos/core/`)

- **Dual Attractor System** - User PA + AI PA alignment
- **Proportional Controller** - Scaled intervention logic
- **Embedding Provider** - OpenAI/Anthropic integration
- **Session State Management** - Turn-by-turn governance
- **Fidelity Calculation** - Cosine similarity measurement

### 2. Cryptography (`telos/crypto/`)

- **Telemetric Keys** - Novel cryptographic system using session telemetry
- **Forward Secrecy** - Turn-by-turn key rotation
- **Delta Extraction** - Privacy-preserving federated learning (future)

### 3. Observatory (`observatory/`)

- Real-time telemetry visualization
- Beta user onboarding
- Steward PM assistant (Mistral-powered)
- Conversation analysis

### 4. Steward Orchestration (`steward/`)

- Multi-layer governance
- Project management assistance
- LLM adapter integration

---

## Documentation

- **[OPUS Audit Findings](docs/audit/OPUS_FINDINGS.md)** - Security & code quality audit
- **[Architecture Overview](docs/architecture/)** - System design
- **[API Documentation](docs/api/)** - Developer reference
- **[Deployment Guide](docs/deployment/)** - Production setup

---

## Development

### Running Tests

```bash
# Unit tests
pytest tests/unit/

# Integration tests
pytest tests/integration/

# Validation suite
pytest tests/validation/
```

### Code Quality

Post-OPUS audit hardening includes:
- ✅ Zero vector division handling
- ✅ NaN/Inf validation
- ✅ Cryptographic entropy strengthening
- ✅ Async/await corrections

---

## License

**Proprietary and Confidential**

Copyright © 2025 TELOS Labs. All Rights Reserved.

See [COPYRIGHT.md](COPYRIGHT.md) for details.

---

## Contact

**TELOS Labs**
- Email: telos.steward@gmail.com
- Website: [Coming Soon]

For licensing inquiries or institutional partnerships, please contact us directly.

---

## Institutional Partnerships

TELOS is designed for deployment in research institutions and production environments. We are actively partnering with:
- George Mason University (GMU)
- University of Oxford
- UC Berkeley

For partnership inquiries: telos.steward@gmail.com

---

**Built with precision. Governed with purpose.**
