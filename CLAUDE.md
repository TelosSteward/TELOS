# TELOS Observatory V3 - Project Context

**Target:** February 2026 institutional deployment (healthcare & enterprise)
**Status:** Research implementation with validated +85.32% fidelity improvement

---

## What is TELOS?

TELOS (Telically Entrained Linguistic Operational Substrate) is a mathematical governance framework for AI alignment. It uses **Primacy Attractors** (PAs) - embedding-space representations of user purpose - to detect and correct conversational drift in real-time.

**Core Innovation:** Two-layer fidelity detection combining baseline normalization for high-dimensional embedding spaces with basin membership for purpose alignment.

---

## Critical Architecture: Two-Layer Fidelity System

### Layer 1: Baseline Pre-Filter (Extreme Off-Topic Detection)
```
File: telos_observatory_v3/services/beta_response_manager.py:37
Constant: SIMILARITY_BASELINE = 0.35
```

**Purpose:** Catch content so far outside the PA embedding manifold that raw cosine similarity clusters at non-zero values.

**Why needed:** Mistral 1024-dim embeddings produce similarity ~0.35-0.56 for completely unrelated content due to concentration of measure. Without this, "PB&J sandwich" vs "AI governance" scores ~0.56 instead of ~0.0.

**Decision:** `raw_similarity < 0.35` triggers HARD_BLOCK (immediate intervention)

### Layer 2: Basin Membership (Purpose Drift Detection)
```
File: telos_observatory_v3/services/beta_response_manager.py:45-49
Constants:
  BASIN = 0.50
  TOLERANCE = 0.02
  INTERVENTION_THRESHOLD = 0.48  (BASIN - TOLERANCE)
```

**Purpose:** Detect when user has drifted from stated purpose even if still related to domain.

**Decision:** `fidelity < 0.48` means outside basin, triggers intervention

### Unified Intervention Decision
```python
should_intervene = baseline_hard_block OR NOT in_basin
# Equivalent to: raw_sim < 0.35 OR fidelity < 0.48
```

---

## Key File Locations

### Core Fidelity Calculation
| File | Purpose | Key Lines |
|------|---------|-----------|
| `telos_observatory_v3/services/beta_response_manager.py` | Two-layer fidelity, intervention decision | 37, 45-49, 150-165, 239-299 |
| `telos_observatory_v3/telos_purpose/core/intercepting_llm_wrapper.py` | Active response governance | 338-392 |
| `telos_observatory_v3/telos_purpose/core/constants.py` | All calibration constants | 1-382 |

### Mathematics & Attractors
| File | Purpose |
|------|---------|
| `telos_purpose/core/primacy_math.py` | Attractor center, basin radius, Lyapunov |
| `telos_purpose/core/dual_attractor.py` | Dual PA (user intent + AI role) |
| `telos_purpose/core/primacy_state.py` | PS = rho * harmonic_mean(F_user, F_ai) |
| `telos_purpose/core/proportional_controller.py` | Graduated intervention (K=1.5) |

### Embedding Providers
| File | Models |
|------|--------|
| `telos_purpose/core/embedding_provider.py` | MistralEmbeddingProvider (1024-dim), SentenceTransformerProvider (384-dim) |

### Entry Point
```bash
cd /Users/brunnerjf/Desktop/telos_privacy
streamlit run telos_observatory_v3/main.py --server.port 8501
```

---

## Calibration Constants Reference

| Constant | Value | Purpose | Location |
|----------|-------|---------|----------|
| `SIMILARITY_BASELINE` | 0.35 | Layer 1 hard-block (Mistral-specific) | beta_response_manager.py:37 |
| `BASIN` | 0.50 | Basin boundary | beta_response_manager.py:45 |
| `TOLERANCE` | 0.02 | Safety margin | beta_response_manager.py:46 |
| `INTERVENTION_THRESHOLD` | 0.48 | Actual decision point | beta_response_manager.py:49 |
| `FIDELITY_GREEN` | 0.85 | UI: High alignment | beta_response_manager.py:58 |
| `FIDELITY_YELLOW` | 0.70 | UI: Moderate | beta_response_manager.py:59 |
| `FIDELITY_ORANGE` | 0.50 | UI: Low | beta_response_manager.py:60 |

**Model-Specific Warning:** `SIMILARITY_BASELINE` must be recalibrated if embedding model changes. See `docs/internal/EMBEDDING_BASELINE_NORMALIZATION.md`.

---

## Environment Setup

```bash
# Required in .env
MISTRAL_API_KEY=<your_key>

# Launch
export PYTHONPATH=/Users/brunnerjf/Desktop/telos_privacy
streamlit run telos_observatory_v3/main.py --server.port 8501
```

---

## Architecture Overview

```
User Input
    |
    v
BetaResponseManager.generate_turn_responses()
    |
    +---> _calculate_user_fidelity()
    |         |
    |         +---> Layer 1: raw_sim < 0.35? -> HARD_BLOCK
    |         +---> Layer 2: fidelity < 0.48? -> Outside basin
    |
    +---> Intervention Decision
    |         should_intervene = hard_block OR NOT in_basin
    |
    +---> Generate Response
              |
              +---> Native (no intervention)
              +---> TELOS-governed (with intervention)
```

---

## Key Formulas

**Primacy State:**
```
PS = rho_PA * (2 * F_user * F_ai) / (F_user + F_ai)
```

**Attractor Center:**
```
a_hat = (tau * purpose + (1-tau) * scope) / ||...||
```

**Basin Radius:**
```
r = 1.0 / max(rigidity, 0.25)  where rigidity = 1 - tau
```

**Intervention Cascade:**
- F >= 0.85: MONITOR (no action)
- 0.70 <= F < 0.85: CORRECT (context injection)
- 0.50 <= F < 0.70: INTERVENE (regeneration)
- F < 0.50: ESCALATE (block + human review)

---

## Development Patterns

### When Modifying Fidelity Logic
1. Both `beta_response_manager.py` and `intercepting_llm_wrapper.py` must use same constants
2. Test with off-topic queries (e.g., "PB&J sandwich") to verify Layer 1
3. Test with related-but-drifted queries to verify Layer 2
4. Run validation: `python3 test_baseline_validation.py`

### When Changing Embedding Model
1. Empirically determine new `SIMILARITY_BASELINE` for that model
2. Update in `beta_response_manager.py` and `intercepting_llm_wrapper.py`
3. Document in `docs/internal/EMBEDDING_BASELINE_NORMALIZATION.md`
4. Test baseline produces hard-block for truly unrelated content

---

## Documentation

| Document | Purpose |
|----------|---------|
| `docs/internal/EMBEDDING_BASELINE_NORMALIZATION.md` | Layer 1 calibration rationale |
| `docs/TELOS_Whitepaper_v2.3.md` | Mathematical specification |
| `docs/TELOS_Lexicon_V1.1.md` | Terminology definitions |
| `CLAUDE_CODE_GOVERNANCE.md` | Meta-governance setup |

---

## Slash Commands Available

- `/telos` - TELOS workflow
- `/steward` - Steward mode
- `/monitor-status` - Monitor status
- `/monitor-export` - Export monitoring data

---

## Current Focus Areas

1. **Validation:** Confirming two-layer architecture catches both extreme off-topic and subtle drift
2. **Grant Applications:** Preparing evidence for Feb 2026 institutional deployment
3. **Documentation:** This CLAUDE.md, plus internal technical docs

---

## Quick Reference: Turn-by-Turn Flow

1. User input received
2. Embed user input with MistralEmbeddingProvider
3. Calculate raw cosine similarity to user_pa_embedding
4. **Layer 1:** If raw_sim < 0.35 -> HARD_BLOCK
5. **Layer 2:** If fidelity < 0.48 -> Outside basin
6. **Decision:** Intervene if Layer1 OR Layer2 triggered
7. Generate response (native or TELOS-governed)
8. Display with fidelity metrics

---

*Last updated: 2024-11-29*
*Authoritative source for TELOS V3 project context*
