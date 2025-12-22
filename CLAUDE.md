# TELOS Observatory V3 - Development Guide

**Working Directory:** `/Users/brunnerjf/Desktop/TELOS_Master`
**Entry Point:** `streamlit run telos_observatory_v3/main.py --server.port 8501`
**Required:** `MISTRAL_API_KEY` in `.env`

---

## What is TELOS?

TELOS (Telically Entrained Linguistic Operational Substrate) is a mathematical governance framework for AI alignment. It uses **Primacy Attractors** (embedding-space representations of user purpose) to detect and correct conversational drift in real-time.

**Core Innovation:** Two-layer fidelity detection - baseline normalization catches extreme off-topic content, basin membership catches purpose drift.

---

## Two-Layer Fidelity System

### Layer 1: Baseline Pre-Filter
```
Constant: SIMILARITY_BASELINE = 0.35
Location: services/beta_response_manager.py
```
Catches content completely outside the PA embedding space. Raw cosine similarity < 0.35 triggers immediate HARD_BLOCK.

### Layer 2: Basin Membership
```
Constants: BASIN = 0.50, TOLERANCE = 0.02
Threshold: INTERVENTION_THRESHOLD = 0.48
Location: services/beta_response_manager.py
```
Detects when user has drifted from stated purpose. Fidelity < 0.48 means outside basin, triggers intervention.

### Intervention Decision
```python
should_intervene = (raw_similarity < 0.35) OR (fidelity < 0.48)
```

---

## Display Thresholds (UI Zones)

| Zone | Fidelity | Color | Meaning | Intervention |
|------|----------|-------|---------|--------------|
| GREEN | >= 0.70 | `#27ae60` | Aligned | None |
| YELLOW | 0.60-0.69 | `#f39c12` | Minor Drift | Context injection |
| ORANGE | 0.50-0.59 | `#e67e22` | Drift Detected | Steward redirect |
| RED | < 0.50 | `#e74c3c` | Significant Drift | Block + review |

**Single source of truth:** `telos_purpose/core/constants.py`

---

## Project Structure

```
telos_observatory_v3/
├── main.py                    # Streamlit entry point
├── telos_purpose/             # Core mathematical framework
│   ├── core/                  # Fidelity math, constants, governance
│   ├── llm_clients/           # Mistral integration
│   └── dev_dashboard/         # Developer tools
├── components/                # UI components (27 files)
├── services/                  # Backend logic (16 files)
├── config/                    # PA templates, colors, steward config
├── core/                      # State management
└── utils/                     # Helpers
```

---

## Key Files

### Core Engine
| File | Purpose |
|------|---------|
| `services/beta_response_manager.py` | **Main fidelity engine** - two-layer detection, intervention decision |
| `telos_purpose/core/constants.py` | All calibration constants (single source of truth) |
| `telos_purpose/core/embedding_provider.py` | Mistral (1024-dim) + SentenceTransformer (384-dim) |
| `telos_purpose/core/semantic_interpreter.py` | Fidelity → linguistic specifications |
| `telos_purpose/core/proportional_controller.py` | Graduated intervention strength (K=1.5) |

### Governance Observability (Recent)
| File | Purpose |
|------|---------|
| `telos_purpose/core/evidence_schema.py` | Pydantic models for 11 governance event types |
| `telos_purpose/core/governance_trace_collector.py` | JSONL logging, privacy modes, query interface |
| `telos_purpose/core/sse_replay_manager.py` | Token-by-token streaming capture |
| `services/session_summarizer.py` | AI-powered session summaries with caching |
| `services/report_generator.py` | Self-contained HTML governance reports |

### UI Components
| File | Purpose |
|------|---------|
| `components/conversation_display.py` | Turn-by-turn message renderer |
| `components/beta_observation_deck.py` | Main BETA interface |
| `components/beta_steward_panel.py` | Steward persona sidebar |
| `components/observatory_lens.py` | Fidelity trajectory visualization |
| `components/alignment_indicator.py` | Simple fidelity status indicator |
| `components/intervention_evidence_dashboard.py` | Governance evidence browser |

### Configuration
| File | Purpose |
|------|---------|
| `config/pa_templates.py` | 8 pre-configured Primacy Attractors |
| `config/colors.py` | `get_fidelity_color()` - zone color mapping |
| `config/steward_pa.py` | Steward's purpose attractor |
| `config/steward_styles.py` | Steward persona styling |

### State Management
| File | Purpose |
|------|---------|
| `core/state_manager.py` | Session state orchestration |
| `beta_testing/beta_session_manager.py` | BETA session management |

---

## Architecture Flow

```
User Input
    │
    ▼
BetaResponseManager.generate_turn_responses()
    │
    ├──► Embed input (MistralEmbeddingProvider)
    │
    ├──► Calculate fidelity
    │     ├── Layer 1: raw_sim < 0.35? → HARD_BLOCK
    │     └── Layer 2: fidelity < 0.48? → Outside basin
    │
    ├──► Intervention Decision
    │     should_intervene = hard_block OR NOT in_basin
    │
    ├──► If intervening:
    │     ├── SemanticInterpreter.interpret() → linguistic specs
    │     └── Generate governed response
    │
    ├──► Record to GovernanceTraceCollector
    │
    └──► Return response with fidelity metrics
```

---

## Governance Observability System

### Evidence Schema
11 event types defined in `evidence_schema.py`:
- `SessionStartEvent`, `SessionEndEvent`
- `PAEstablishedEvent`
- `FidelityCalculatedEvent` - raw similarity, normalized fidelity, zone
- `InterventionTriggeredEvent` - level, reason, action
- `SSETokenEvent` - streaming token capture
- `UserInputEvent`, `ResponseGeneratedEvent`
- `ErrorEvent`, `CustomEvent`

### Privacy Modes
```python
class PrivacyMode(Enum):
    FULL = "full"           # Complete data
    HASHED = "hashed"       # SHA-256 hashed content
    DELTAS_ONLY = "deltas_only"  # Only metrics, no content (default)
```

### Trace Collector Usage
```python
from telos_purpose.core.governance_trace_collector import get_trace_collector

collector = get_trace_collector(session_id="my_session")
collector.record_fidelity(turn_number=1, raw_similarity=0.65, ...)
collector.record_intervention(turn_number=1, intervention_level="correct", ...)

# Query
stats = collector.get_session_stats()
trajectory = collector.get_fidelity_trajectory()
interventions = collector.get_interventions()

# Export
session_data = collector.export_to_dict()
```

---

## Development Patterns

### When Modifying Fidelity Logic
1. Edit constants in `telos_purpose/core/constants.py`
2. Update `services/beta_response_manager.py` if logic changes
3. Test with off-topic queries (verify Layer 1)
4. Test with related-but-drifted queries (verify Layer 2)

### When Adding UI Components
1. Create in `components/`
2. Follow border-style pattern for fidelity coloring:
```css
background-color: #2d2d2d;
border: 2px solid {zone_color};
color: #e0e0e0;
```
3. Import and use in appropriate page

### When Modifying Steward Persona
1. `config/steward_pa.py` - Purpose attractor definition
2. `config/steward_styles.py` - Visual styling
3. `services/beta_steward_llm.py` - LLM integration

### Key Color Function
```python
from config.colors import get_fidelity_color

color = get_fidelity_color(0.75)  # Returns "#27ae60" (green)
```

---

## Formulas Reference

**Primacy State:**
```
PS = ρ_PA × (2 × F_user × F_ai) / (F_user + F_ai)
```

**Attractor Center:**
```
â = (τ × purpose + (1-τ) × scope) / ||...||
```

**Intervention Strength:**
```
strength = min(K_ATTRACTOR × error_signal, 1.0)  where K_ATTRACTOR = 1.5
```

---

## Semantic Interpreter Bands

| Band | Strength | Fidelity | Style |
|------|----------|----------|-------|
| MINIMAL | < 0.45 | ~0.70+ | Questions, heavy hedging |
| LIGHT | 0.45-0.60 | ~0.60-0.70 | Soft statements, light hedging |
| MODERATE | 0.60-0.75 | ~0.50-0.60 | Direct statements, no hedging |
| FIRM | 0.75-0.85 | ~0.40-0.50 | Directives, named drift |
| STRONG | >= 0.85 | < 0.40 | Clear directives, prominent shift |

---

## Environment Setup

```bash
cd /Users/brunnerjf/Desktop/TELOS_Master

# Ensure .env has:
# MISTRAL_API_KEY=<your_key>

# Set Python path
export PYTHONPATH=/Users/brunnerjf/Desktop/TELOS_Master

# Launch
streamlit run telos_observatory_v3/main.py --server.port 8501
```

---

## Testing

```bash
# Syntax check a file
python3 -m py_compile telos_observatory_v3/path/to/file.py

# Run validation suite
python3 telos_observatory_v3/telos_purpose/validation/run_internal_test0.py
```

---

## Documentation

| Document | Location | Purpose |
|----------|----------|---------|
| This file | `CLAUDE.md` | Development guide |
| Whitepaper | `docs/TELOS_Whitepaper_v2.3.md` | Mathematical specification |
| Lexicon | `docs/TELOS_Lexicon_V1.1.md` | Terminology definitions |
| Grant Roadmap | `docs/GRANT_ROADMAP.md` | Funding phases, consortium architecture |
| Telemetric Keys | `docs/TELEMETRIC_KEYS.md` | Cryptographic privacy layer details |

---

## Quick Debugging

**Fidelity always 0.5?**
- Check `MISTRAL_API_KEY` is set
- Check embedding provider is initializing

**Interventions not triggering?**
- Verify `INTERVENTION_THRESHOLD` in constants.py
- Check `should_intervene` logic in beta_response_manager.py

**UI not updating?**
- Check `st.rerun()` calls
- Verify session state keys

**Trace collector not recording?**
- Check `TRACE_COLLECTOR_AVAILABLE` flag
- Verify session_id is consistent

---

*Last updated: 2025-12-19*
*For grant language and research roadmap, see `docs/GRANT_ROADMAP.md`*
