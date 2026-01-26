# TELOS Observatory V3 - Development Guide

## Quick Reference

**Company:** TELOS AI Labs Inc. | **Email:** JB@telos-labs.ai
**Working Directory:** `/Users/brunnerjf/Desktop/TELOS_Master`
**Launch:** `streamlit run telos_observatory_v3/main.py --server.port 8501`
**Required:** `MISTRAL_API_KEY` in `.env`

---

## What is TELOS?

TELOS (Telically Entrained Linguistic Operational Substrate) is a mathematical governance framework for AI alignment using **Primacy Attractors** (embedding-space representations of user purpose) to detect and correct conversational drift in real-time.

**Core Innovation:** Two-layer fidelity detection - baseline normalization (Layer 1) catches extreme off-topic, basin membership (Layer 2) catches purpose drift.

---

## Two-Layer Fidelity System

**Single source of truth:** `telos_purpose/core/constants.py`

| Threshold | Value | Purpose |
|-----------|-------|---------|
| SIMILARITY_BASELINE | 0.20 | Layer 1: Hard block (extreme off-topic) |
| INTERVENTION_THRESHOLD | 0.48 | Layer 2: Basin boundary |
| FIDELITY_GREEN | 0.70 | Aligned (no intervention) |
| FIDELITY_YELLOW | 0.60 | Minor drift (context injection) |
| FIDELITY_ORANGE | 0.50 | Drift detected (Steward redirect) |
| FIDELITY_RED | <0.50 | Significant drift (block + review) |

**Intervention Decision:**
```python
should_intervene = (raw_similarity < 0.20) OR (fidelity < 0.70)
```

---

## Project Structure

```
telos_observatory_v3/
├── main.py                    # Streamlit entry point
├── telos_purpose/core/        # Fidelity math, constants, governance
├── components/                # UI components (27 files)
├── services/                  # Backend logic (16 files)
├── config/                    # PA templates, colors, steward config
└── core/                      # State management
```

---

## Key Files

| File | Purpose |
|------|---------|
| `services/beta_response_manager.py` | **Main fidelity engine** - two-layer detection |
| `telos_purpose/core/constants.py` | All calibration constants |
| `telos_purpose/core/embedding_provider.py` | Mistral (1024-dim) + SentenceTransformer (384-dim) |
| `telos_purpose/core/semantic_interpreter.py` | Fidelity → linguistic specs |
| `telos_purpose/core/governance_trace_collector.py` | JSONL logging, privacy modes |
| `telos_purpose/core/adaptive_context.py` | Multi-tier context buffer, phase detection |
| `config/pa_templates.py` | 8 pre-configured Primacy Attractors |
| `config/colors.py` | `get_fidelity_color()` zone mapping |
| `core/state_manager.py` | Session state orchestration |

---

## Architecture Flow

```
User Input → Embed → Raw Similarity → Layer 1 Check (0.20)
    → Adaptive Context → Intervention Decision (0.70)
    → [If intervening] SemanticInterpreter → Steward Response
    → Record to GovernanceTraceCollector → Return
```

---

## Development Patterns

**Modifying Fidelity Logic:**
1. Edit `telos_purpose/core/constants.py`
2. Update `services/beta_response_manager.py` if logic changes
3. Test with off-topic queries (Layer 1) and drifted queries (Layer 2)

**Adding UI Components:**
1. Create in `components/`
2. Use border-style pattern: `border: 2px solid {get_fidelity_color(fidelity)};`

**Modifying Steward:**
- `config/steward_pa.py` - Purpose attractor
- `config/steward_styles.py` - Visual styling
- `services/beta_steward_llm.py` - LLM integration

---

## Quick Debugging

| Issue | Check |
|-------|-------|
| Fidelity always 0.5 | `MISTRAL_API_KEY` set? Embedding provider initializing? |
| Interventions not triggering | `INTERVENTION_THRESHOLD` in constants.py, `should_intervene` logic |
| UI not updating | `st.rerun()` calls, session state keys |
| Trace collector not recording | `TRACE_COLLECTOR_AVAILABLE` flag, session_id consistency |

---

## Documentation

| Document | Purpose |
|----------|---------|
| `docs/TELOS_Academic_Paper.tex` | **Canonical academic paper** (LaTeX source, arXiv-ready) |
| `docs/TELOS_Academic_Paper.pdf` | Academic paper PDF (14 pages, two-column, with figures) |
| `docs/TELOS_Academic_Paper.md` | Academic paper markdown version |
| `docs/TELOS_Whitepaper_v2.3.md` | Mathematical specification |
| `docs/TELOS_Lexicon_V1.1.md` | Terminology definitions |
| `docs/GRANT_ROADMAP.md` | Funding phases, consortium architecture |

## Academic Paper Figures

Standalone TikZ figures in `docs/diagrams/`:

| Figure | File | Section | Description |
|--------|------|---------|-------------|
| Fig 1 | `fig1_three_tier_governance.tex/.pdf` | Section 4 | Three-Tier Defense Architecture (full-width) |
| Fig 2 | `fig2_primacy_attractor.tex/.pdf` | Section 3 | PA basin geometry in embedding space |
| Fig 3 | `fig3_fidelity_pipeline.tex/.pdf` | Section 2 | Two-Layer Fidelity Architecture |
| Fig 4 | `fig4_governance_trace.tex/.pdf` | Section 6 | Governance Trace event flow |

Figures are included in paper via `\includegraphics`. To recompile standalone figures:
```bash
cd docs/diagrams && pdflatex fig1_three_tier_governance.tex
```

---

## Formulas (Reference)

- **Primacy State:** `PS = ρ_PA × (2 × F_user × F_ai) / (F_user + F_ai)`
- **Intervention Strength:** `strength = min(K_ATTRACTOR × error_signal, 1.0)` where K=1.5

---

*Last updated: 2026-01-25*
