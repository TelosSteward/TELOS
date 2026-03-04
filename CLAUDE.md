# TELOS Development Guide

> Architecture reference, development conventions, and technical context
> for contributors working on the TELOS governance framework.

## Architecture Overview

TELOS (Telically Entrained Linguistic Operational Substrate) is a runtime
governance framework for AI agents. The architecture has three tiers:

### Tier 1: Core Mathematical Engine (`telos_core/`)
- **Primacy Attractors** — mathematically defined purpose vectors in embedding space
- **Fidelity Scoring** — cosine similarity between agent actions and purpose attractor
- **Embedding Provider** — pluggable embedding backends (OpenAI, ONNX, MLX, Mistral)

### Tier 2: Governance Protocol (`telos_governance/`)
- **Two-Gate Architecture** — Gate 1 (fidelity scoring) + Gate 2 (boundary corpus)
- **PA Constructor** — builds primacy attractors from YAML configuration
- **Tool Semantics** — classifies agent tool calls into governance-relevant categories
- **Boundary Corpus** — hard constraints enforced structurally (not behaviorally)
- **SetFit Classifier** — learned classification for ambiguous boundary cases
- **Scoring Cascade** — L0 (embedding) → L1 (SetFit) → L1.5 (context) → L2 (aggregate)

### Tier 3: Integration Layer
- **Adapters** (`telos_adapters/`) — Generic, LangGraph, OpenClaw (11-module)
- **Gateway** (`telos_gateway/`) — OpenAI-compatible HTTP governance proxy
- **Observatory** (`telos_observatory/`) — Streamlit dashboard for governance visualization
- **TKeys** (`telos_privacy/`) — Ed25519 signing, HMAC-SHA512 receipts, hash chains

## Key Thresholds

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Intervention trigger | fidelity < 0.70 OR raw < 0.20 | Balances safety with operational freedom |
| EXECUTE | fidelity >= 0.70, no boundary violation | Full purpose alignment confirmed |
| CLARIFY | 0.50 <= fidelity < 0.70 | Uncertain — request clarification |
| SUGGEST | 0.35 <= fidelity < 0.50 | Low alignment — suggest alternative |
| INERT | fidelity < 0.35, no boundary hit | Too far from purpose — block silently |
| ESCALATE | boundary violation detected | Hard constraint — escalate to human |

## Development Conventions

- **Tests before merge** — full suite must pass (`python -m pytest tests/ -x -q`)
- **Benchmark regression gate** — Cat A >= 80%, overall >= 83%, FPR <= 10%
- **Governance changes require advisory review** — see `GOVERNANCE.md`
- **Cryptographic layer changes require security review** — see `SECURITY.md`

## Project Structure

```
telos_core/           Core math: primacy attractors, fidelity, embeddings
telos_governance/     Governance protocol: scoring, boundaries, PA constructor
telos_adapters/       Integration adapters (generic, langgraph, openclaw)
telos_gateway/        OpenAI-compatible governance gateway
telos_observatory/    Streamlit governance dashboard
telos_privacy/        TKeys cryptographic signing layer
templates/            YAML PA configuration templates
validation/           7 benchmarks, 5,212 scenarios
tests/                Unit, integration, validation, scenario tests
demos/                Live governance demos
docs/                 Papers, regulatory mappings, technical briefs
research/             Research artifacts and methodology
```

## Research Program

TELOS development follows a structured research program with:
- **Adversarial validation** across 7 benchmark families
- **Formal hypothesis testing** (governed vs ungoverned agent behavior)
- **Regulatory mapping** (NIST AI RMF, EU AI Act, Berkeley CLTC, IEEE 7000)
- **Independent measurement** via TELOSCOPE observation instrument
- **SAAI Framework compliance** (Dr. Nell Watson)

See `research/research_team_spec.md` for the full research methodology,
including generative AI and conflict of interest disclosures.

## Governance Visualization

The Observatory dashboard (`telos_observatory/`) provides:
- Real-time fidelity trajectory with color-coded zones (green/yellow/orange/red)
- Verdict distribution charts
- Intervention timeline
- Session-level governance health indicators
- Raw event data export

Zone colors map to the threshold table above.

## Quick Start

```bash
git clone https://github.com/TelosSteward/TELOS.git
cd TELOS && python3 -m venv venv && source venv/bin/activate
pip install -e ".[dev]"
python -m pytest tests/ -x -q          # Run tests
telos score "Assess roof condition"     # Score a request
telos demo nearmap                      # Run live demo
```

---

**TELOS AI Labs** | [GitHub](https://github.com/TelosSteward/TELOS) | Built with purpose alignment.
