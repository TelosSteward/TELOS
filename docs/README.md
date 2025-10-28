# TELOS — Governance That Holds Through the Conversation
*A runtime mathematical framework for governance persistence in large language model sessions.*

-----

## What TELOS Does

TELOS provides **runtime governance for LLM sessions** through mathematical measurement and intervention. Rather than assuming alignment persists across multi-turn conversations, TELOS makes governance persistence **observable, quantifiable, and testable**—transforming AI oversight from declarative policy into empirical science.

**The Problem:** Empirical studies across major model families (OpenAI, Anthropic, Google, Mistral) report 20–40% governance fidelity loss in multi-turn sessions as context drifts and initial instructions lose salience. Current approaches rely on static prompts or universal safety guidelines, leaving session-level governance claims unverified and unmeasured.

**The Approach:** TELOS applies established mathematical tools—embedding-based similarity, attractor dynamics, and Lyapunov stability analysis—to convert purpose, scope, and boundary declarations into measurable attractors in embedding space. It continuously tracks semantic drift, computes fidelity metrics, and applies proportional corrections only when deviation exceeds mathematical thresholds.

**The Architecture:** While the mathematical foundations draw from systems theory and vector-space methods, TELOS’s contribution lies in operational synthesis: embedding these dynamics into a live orchestration loop that measures drift turn-by-turn and exports research-grade telemetry. This enables governance to be studied as a continuous dynamical process rather than a static configuration, providing transparent, reproducible measurement infrastructure that supports human oversight and interpretability.

**The Goal:** TELOS establishes a rigorous approach toward evidence-based AI oversight—one where alignment claims can be verified, challenged, and improved through reproducible measurement rather than institutional assumption.

-----

## Core Components

|Component                  |Function                                                                           |
|---------------------------|-----------------------------------------------------------------------------------|
|**Primacy Attractor Math** |Defines governance center, basin radius, and stability metrics                     |
|**Intervention Controller**|Triggers proportional corrections (reminder/regeneration) based on drift thresholds|
|**Unified Steward**        |Orchestrates runtime loop and exports telemetry                                    |
|**Health Monitor**         |Runtime diagnostics - catches bugs, config errors, math anomalies                  |
|**Validation Runners**     |Comparative testing across governance modes (stateless/prompt-only/cadence/TELOS)  |

-----

## Install

```bash
git clone https://github.com/your-org/telos.git
cd telos
pip install -e .

# Optional dependencies
pip install -e ".[embeddings]"   # sentence-transformers for semantic analysis
pip install -e ".[dev]"          # pytest and development tools
```

**Requirements:** Python 3.9+, numpy≥1.23.0

-----

## Configure

Edit `config.json` to define governance parameters:

```json
{
  "purpose": ["explain AI governance mechanisms"],
  "scope": ["AI alignment", "runtime oversight"],
  "boundaries": ["no harmful content", "stay technical"],
  "privacy_level": 0.8,
  "constraint_tolerance": 0.2,
  "task_priority": 0.7
}
```

**Key Parameter:** `constraint_tolerance`

- `0.0` = strict adherence (small basin, frequent interventions)
- `1.0` = permissive adherence (large basin, infrequent interventions)

-----

## Quick Start

**Interactive session with dashboard:**

```bash
python -m telos_purpose.sessions.run_with_dashboard --config config.json
```

**Run test conversation:**

```bash
python -m telos_purpose.runners.TELOS_runner_script \
  --config config.json \
  --conversation telos_purpose/test_conversations/test_convo_001.json
```

**Internal Test 0 (5-condition validation):**

```bash
python -m telos_purpose.validation.run_internal_test0
```

-----

## Python API

```python
from telos_purpose.core.unified_steward import UnifiedGovernanceSteward, PrimacyAttractor
from telos_purpose.core.embedding_provider import DeterministicEmbeddingProvider
from telos_purpose.llm_clients.mistral_client import TelosMistralClient

# Define governance perimeter
attractor = PrimacyAttractor(
    purpose=["explain AI governance"],
    scope=["AI alignment", "runtime oversight"],
    boundaries=["no harmful content"],
    constraint_tolerance=0.2
)

# Initialize components
llm = TelosMistralClient()
embeddings = DeterministicEmbeddingProvider()

steward = UnifiedGovernanceSteward(
    attractor=attractor,
    llm_client=llm,
    embedding_provider=embeddings
)

# Run governed session
steward.start_session()
result = steward.process_turn(
    user_input="What is TELOS?",
    model_response="[LLM output here]"
)
summary = steward.end_session()
```

-----

## System Health Monitoring

TELOS includes runtime diagnostics that catch bugs and configuration errors during sessions.

### Enable Live Monitoring (Development)

```python
steward.health.enable_live_monitoring()
steward.start_session()
# Prints: [T1] F=0.873 | E=0.234 | Action=none
```

### Mid-Session Health Checks

```python
report = steward.health.generate_instant_report()
if report['event_summary']['critical'] > 0:
    print("Critical issues detected")
```

### Export Diagnostics

```python
steward.end_session()

# Multiple export formats
steward.health.export_diagnostic_data(format="json")  # Full data
steward.health.export_diagnostic_data(format="csv")   # Turn-by-turn
steward.health.plot_vital_signs(show=True)            # Visual plots (requires matplotlib)
```

### Disable for Production

```python
steward = UnifiedGovernanceSteward(..., enable_health_monitor=False)
```

**Note:** Health monitor catches code errors. Telemetry validates whether TELOS works.

-----

## Internal Test 0: Five-Condition Validation

Minimal internal test verifying runtime behavior across governance modes:

|Condition        |Description                             |Tests                        |
|-----------------|----------------------------------------|-----------------------------|
|**Stateless**    |No governance                           |Baseline drift behavior      |
|**Prompt-Only**  |Governance declared once                |Effect of initial declaration|
|**Cadence**      |Fixed-interval reminders (every 3 turns)|Scheduled reinforcement      |
|**Observation**  |TELOS math active, no interventions     |Pure drift detection         |
|**TELOS Runtime**|Full adaptive governance                |Proportional intervention    |

**Run:** `python -m telos_purpose.validation.run_internal_test0`

**Output:** 5 CSV files (turn-level telemetry) + 5 JSON files (session summaries) in `validation_results/internal_test0/`

-----

## Mathematical Foundations

|Concept         |Formula                                |Meaning                                |
|----------------|---------------------------------------|---------------------------------------|
|Attractor Center|`â = (τ·p + (1-τ)·s) / ‖τ·p + (1-τ)·s‖`|Purpose/scope weighted center          |
|Basin Radius    |`r = 2/max(ρ,0.25)` where `ρ = 1-τ`    |Tolerance determines basin size        |
|Error Signal    |`e = ‖x - â‖ / r`                      |Normalized drift distance              |
|Fidelity        |`F = (1/T) Σ [x ∈ basin]`              |Fraction of turns within governance    |
|Lyapunov        |`V(x) = ‖x - â‖²`                      |Stability measure (ΔV < 0 = convergent)|

-----

## Outputs

- **Session telemetry:** `purpose_protocol_exports/session_*.json`
- **Health diagnostics:** `purpose_protocol_exports/health/system_health_*.json`
- **Validation results:** `validation_results/`
- **Streaming logs:** `logs/telemetry_log.jsonl`

-----

## Project Structure

```
telos_purpose/
├── core/                    # Mathematical foundations
│   ├── primacy_math.py
│   ├── intervention_controller.py
│   └── unified_steward.py
├── validation/              # Comparative testing
│   ├── baseline_runners.py
│   ├── system_health_monitor.py
│   ├── run_internal_test0.py
│   └── telemetry_utils.py
├── sessions/                # Interactive runtime
└── dev_dashboard/           # Developer telemetry
```

See `TELOS_Project_Structure.md` for complete file tree.

-----

## Makefile Commands

```bash
make run        # Interactive governed session
make validate   # Comparative validation study
make smoke      # Quick pytest tests
make clean      # Remove logs/results
```

-----

## Summary

TELOS moves alignment from declaration to measurement.
It doesn’t assume purpose holds — it quantifies when it does and when it drifts.
By turning aspirational governance principles into observable runtime data,
TELOS makes purpose persistence a measurable property of LLM sessions.

-----

## License

MIT License. Research infrastructure for studying runtime governance in LLMs.

© 2025 TELOS Labs
</artifact>

-----