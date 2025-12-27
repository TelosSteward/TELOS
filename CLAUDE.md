# TELOS Observatory V3 - Development Guide

## Organization Details

- **Company:** TELOS AI Labs Inc.
- **Primary Email:** JB@telos-labs.ai
- **Contact:** contact@telos-labs.ai

---

**Working Directory:** `/Users/brunnerjf/Desktop/TELOS_Master`
**Entry Point:** `streamlit run telos_observatory_v3/main.py --server.port 8501`
**Required:** `MISTRAL_API_KEY` in `.env`

---

## What is TELOS?

TELOS (Telically Entrained Linguistic Operational Substrate) is a mathematical governance framework for AI alignment. It uses **Primacy Attractors** (embedding-space representations of user purpose) to detect and correct conversational drift in real-time.

**Core Innovation:** Two-layer fidelity detection - baseline normalization catches extreme off-topic content, basin membership catches purpose drift.

---

## Two-Layer Fidelity System

**Single source of truth:** `telos_purpose/core/constants.py`

### Layer 1: Baseline Pre-Filter
```
Constant: SIMILARITY_BASELINE = 0.20
```
Catches extreme off-topic content. Raw cosine similarity < 0.20 triggers immediate HARD_BLOCK.

### Layer 2: Basin Membership
```
Constants: BASIN_CENTER = 0.50, BASIN_TOLERANCE = 0.02
Threshold: INTERVENTION_THRESHOLD = 0.48
```
Detects purpose drift. Fidelity < 0.48 means outside the primacy basin.

### Intervention Decision
```python
# Note: Intervention triggers below GREEN threshold (0.70), not basin threshold (0.48)
# The basin threshold is for mathematical basin membership, not UI intervention
should_intervene = (raw_similarity < 0.20) OR (fidelity < 0.70)
```

---

## Display Thresholds (UI Zones)

| Zone | Fidelity | Color | Meaning | Intervention |
|------|----------|-------|---------|--------------|
| GREEN | >= 0.70 | `#27ae60` | Aligned | None |
| YELLOW | 0.60-0.69 | `#f39c12` | Minor Drift | Context injection |
| ORANGE | 0.50-0.59 | `#e67e22` | Drift Detected | Steward redirect |
| RED | < 0.50 | `#e74c3c` | Significant Drift | Block + review |

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
| `services/beta_response_manager.py` | **Orchestrator** - coordinates fidelity, intervention, and response modules |
| `services/fidelity_calculator.py` | Fidelity calculation: cosine similarity, adaptive context, thresholds |
| `services/intervention_decider.py` | Intervention decision: zone classification, proportional control |
| `services/response_generator.py` | Response generation: native, redirect, steward styling |
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

### Adaptive Context System
| File | Purpose |
|------|---------|
| `telos_purpose/core/adaptive_context.py` | Multi-tier context buffer, phase detection, adaptive thresholds |
| `tests/test_adaptive_context.py` | Unit tests for adaptive context components |

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

## Decomposed Module Architecture

The core engine has been decomposed into single-responsibility modules:

```
BetaResponseManager (Orchestrator)
    │
    ├── FidelityCalculator
    │       ├── cosine_similarity()
    │       ├── calculate_fidelity()
    │       ├── get_thresholds()
    │       └── adaptive context integration
    │
    ├── InterventionDecider
    │       ├── decide() → InterventionDecision
    │       ├── zone classification (GREEN/YELLOW/ORANGE/RED)
    │       ├── proportional control (K=1.5)
    │       └── governance trace logging
    │
    └── ResponseGenerator
            ├── generate_native_response() - GREEN zone
            ├── generate_redirect_response() - intervention zones
            ├── _build_system_prompt() - PA-aware prompts
            ├── _build_redirect_prompt() - hybrid styling
            └── get_steward_styling() - band interpolation
```

**Benefits:**
- Each module has a single responsibility
- Easier testing and debugging
- Clearer dependency graph
- Can be imported independently

---

## Architecture Flow

```
User Input
    │
    ▼
BetaResponseManager.generate_turn_responses()
    │
    ├──► 1. EMBED: Embed input (SentenceTransformer or Mistral)
    │
    ├──► 2. RAW SIMILARITY: cosine(input_embedding, PA_embedding)
    │         Range: ST (0.15-0.45), Mistral (0.40-0.75)
    │
    ├──► 3. LAYER 1 CHECK: raw_sim < SIMILARITY_BASELINE (0.20)?
    │         If yes → baseline_hard_block = True (extreme off-topic)
    │
    ├──► 4. ADAPTIVE CONTEXT (if enabled):
    │         ├── Classify message type (ANAPHORA, FOLLOW_UP, etc.)
    │         ├── Compute MAX similarity to prior high-fidelity turns
    │         ├── Apply type-specific boost (ANAPHORA: 1.5x, DIRECT: 0.7x)
    │         └── Return adjusted_fidelity
    │
    ├──► 5. INTERVENTION DECISION:
    │         should_intervene = baseline_hard_block OR fidelity < FIDELITY_GREEN (0.70)
    │         GREEN (≥0.70): No intervention
    │         YELLOW/ORANGE/RED (<0.70): Steward intervention
    │
    ├──► 6. If intervening:
    │         ├── SemanticInterpreter.interpret() → linguistic specs
    │         ├── get_intervention_prompt() → Steward therapeutic persona
    │         └── Generate governed response (hybrid styling)
    │
    ├──► 7. Record to GovernanceTraceCollector
    │
    └──► Return response with fidelity metrics
```

### Threshold Summary (Single Source of Truth: constants.py)

| Threshold | Value | Purpose |
|-----------|-------|---------|
| SIMILARITY_BASELINE | 0.20 | Layer 1 hard block (extreme off-topic) |
| INTERVENTION_THRESHOLD | 0.48 | Layer 2 basin boundary |
| FIDELITY_GREEN | 0.70 | Display zone: Aligned (no intervention) |
| FIDELITY_YELLOW | 0.60 | Display zone: Minor drift |
| FIDELITY_ORANGE | 0.50 | Display zone: Drift detected |
| FIDELITY_RED | 0.50 | Display zone: Significant drift |

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

## Adaptive Context System

The Adaptive Context System provides intelligent message classification, conversation phase detection, and adaptive threshold adjustment.

### Components
| Component | Purpose |
|-----------|---------|
| `MessageType` | Enum: DIRECT, FOLLOW_UP, CLARIFICATION, ANAPHORA |
| `ConversationPhase` | Enum: EXPLORATION, FOCUS, DRIFT, RECOVERY |
| `TieredMessage` | Dataclass for messages with fidelity, embedding, tier |
| `MultiTierContextBuffer` | Three-tier message buffer with capacity limits |
| `PhaseDetector` | Rolling-window conversation phase detection |
| `AdaptiveThresholdCalculator` | Adjusts intervention threshold based on context |
| `AdaptiveContextManager` | Unified interface for all components |

### Constants (adaptive_context.py)
```python
# Tier Thresholds (fidelity-based classification)
TIER1_THRESHOLD = 0.70  # High fidelity messages
TIER2_THRESHOLD = 0.35  # Medium fidelity messages
TIER3_THRESHOLD = 0.25  # Low fidelity messages

# Buffer Capacities (per tier)
TIER1_CAPACITY = 5      # Most recent high-fidelity
TIER2_CAPACITY = 3      # Most recent medium-fidelity
TIER3_CAPACITY = 2      # Most recent low-fidelity

# Governance Safeguards
HARD_FLOOR = 0.20       # Minimum threshold (never go below)
MAX_BOOST = 0.20        # Maximum threshold increase
BASE_THRESHOLD = 0.48   # Default intervention threshold

# Context Weighting
RECENCY_DECAY = 0.8     # λ for weighted embeddings (newer = higher weight)
PHASE_WINDOW_SIZE = 5   # Rolling window for phase detection
```

### Usage
```python
from telos_purpose.core.adaptive_context import (
    AdaptiveContextManager,
    classify_message_type,
    MessageType
)

# Initialize manager
manager = AdaptiveContextManager(base_threshold=0.48)

# Process a message
result = manager.process_message(
    text="Tell me more about that",
    embedding=embedding_vector,
    fidelity_score=0.72
)

# Access result fields
print(f"Message Type: {result.message_type}")     # e.g., MessageType.FOLLOW_UP
print(f"Phase: {result.phase}")                   # e.g., ConversationPhase.FOCUS
print(f"Tier: {result.tier}")                     # e.g., 1
print(f"Adjusted Threshold: {result.adjusted_threshold}")  # e.g., 0.52
print(f"Context Embedding: {result.context_embedding}")    # Weighted numpy array
```

### Message Type Classification
```python
MESSAGE_TYPE_THRESHOLDS = {
    MessageType.DIRECT: 0.70,        # Clear, on-topic statements
    MessageType.FOLLOW_UP: 0.35,     # Continuations, elaborations
    MessageType.CLARIFICATION: 0.25, # Questions about prior content
    MessageType.ANAPHORA: 0.25,      # Pronoun references
}
```

### Feature Flag
```python
# In beta_response_manager.py
ADAPTIVE_CONTEXT_ENABLED = True  # Set False to disable
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

*Last updated: 2025-12-26*
*For grant language and research roadmap, see `docs/GRANT_ROADMAP.md`*
