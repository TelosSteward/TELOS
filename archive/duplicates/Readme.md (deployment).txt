# TELOS: Mathematical Runtime Governance Framework

**Telically Entrained Linguistic Operating Substrate**

Mathematical framework for measuring and mitigating drift in multi-turn AI conversations.

-----

## Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Configure API
export MISTRAL_API_KEY="your_key_here"

# 3. Run Test 0
python -m telos_purpose.validation.run_internal_test0

# 4. View results
python -m telos_purpose.validation.summarize_internal_test0
```

**Full instructions:** See [`QUICKSTART.md`](QUICKSTART.md)

-----

## What is TELOS?

TELOS provides **mathematical runtime governance** for AI conversations:

- **Measure drift** using semantic embeddings and attractor dynamics
- **Detect violations** when responses leave acceptable governance basin
- **Apply corrections** proportional to drift magnitude
- **Generate telemetry** for validation and compliance

### Core Concept

```
Primacy Attractor (â) = governance center in embedding space
Basin of Attraction = acceptable response region
Error Signal (E) = normalized distance from center
Intervention = proportional correction when E > threshold
```

-----

## Repository Structure

```
telos/
├── telos_purpose/           # Core framework
│   ├── core/                # Mathematical components
│   │   ├── primacy_math.py
│   │   ├── unified_steward.py
│   │   ├── intervention_controller.py
│   │   └── embedding_provider.py
│   │
│   ├── validation/          # Testing framework
│   │   ├── baseline_runners.py
│   │   ├── run_internal_test0.py
│   │   └── summarize_internal_test0.py
│   │
│   ├── llm_clients/         # API adapters
│   │   └── mistral_client.py
│   │
│   └── test_conversations/  # Test data
│       ├── test_convo_001.json
│       ├── test_convo_002.json
│       └── test_convo_003.json
│
├── docs/                    # Documentation
│   ├── TELOS_Whitepaper.md
│   ├── QUICKSTART.md
│   └── RUNNING_TEST_0.md
│
├── config.json              # Governance configuration
├── requirements.txt         # Dependencies
├── setup.py                 # Installation
├── Makefile                 # Quick commands
└── README.md                # This file
```

-----

## Internal Test 0

Validates TELOS through 5-way comparison:

|Condition  |Description                 |Intervention Mode|
|-----------|----------------------------|-----------------|
|Stateless  |No governance memory        |None             |
|Prompt-Only|Constraints stated once     |None             |
|Cadence    |Fixed-interval reminders    |Every 3 turns    |
|Observation|Math active, no intervention|Metrics only     |
|TELOS      |Full adaptive governance    |Proportional     |

**Success criteria:**

- H1: TELOS improves fidelity by ≥0.15 over best baseline
- H2: TELOS achieves highest fidelity among all conditions

-----

## Key Components

### Primacy Attractor Math

```python
from telos_purpose.core.primacy_math import PrimacyAttractorMath

# Initialize attractor
attractor = PrimacyAttractorMath(
    purpose_vector=purpose_embedding,
    scope_vector=scope_embedding,
    constraint_tolerance=0.2,  # 0.0=strict, 1.0=permissive
    privacy_level=0.8,
    task_priority=0.9
)

# Measure drift
error_signal = attractor.compute_error_signal(response_embedding)
in_basin = attractor.compute_basin_membership(response_embedding)
```

### Steward (Orchestrator)

```python
from telos_purpose.core.unified_steward import UnifiedGovernanceSteward

# Initialize governed session
steward = UnifiedGovernanceSteward(
    attractor=attractor_config,
    llm_client=mistral_client,
    embedding_provider=embedder,
    enable_interventions=True
)

# Process conversation
steward.start_session()
result = steward.process_turn(user_input, model_response)
summary = steward.end_session()
```

-----

## Requirements

- **Python**: 3.8+
- **API Access**: Mistral API key
- **Compute**: CPU sufficient, GPU optional for embeddings
- **Memory**: 2GB+ RAM
- **Storage**: 2GB for models

-----

## Configuration

Edit `config.json` to customize:

```json
{
  "attractor_parameters": {
    "constraint_tolerance": 0.2,  // 0.0=strict, 1.0=permissive
    "privacy_level": 0.8,
    "task_priority": 0.9
  },
  "intervention_thresholds": {
    "epsilon_min": 0.5,  // Reminder threshold
    "epsilon_max": 0.8   // Regeneration threshold
  }
}
```

-----

## Output Data

### Turn-Level Telemetry (CSV)

```csv
session_id,condition,turn_id,fidelity_score,error_signal,in_basin,intervention_triggered
test0_telos,telos,1,0.9234,0.1234,true,false
test0_telos,telos,2,0.8567,0.2345,true,false
test0_telos,telos,3,0.5432,0.8901,false,true
```

### Session Summary (JSON)

```json
{
  "session_metadata": {
    "session_id": "test0_telos",
    "condition": "telos",
    "total_turns": 5
  },
  "session_metrics": {
    "avg_fidelity": 0.8734,
    "basin_adherence": 0.9123,
    "intervention_rate": 0.2000
  }
}
```

-----

## Development

### Running Tests

```bash
# Unit tests
pytest tests/

# Integration test
python -m telos_purpose.validation.run_internal_test0
```

### Code Style

```bash
# Format
black telos_purpose/

# Lint
flake8 telos_purpose/

# Type check
mypy telos_purpose/
```

-----

## Documentation

- **Whitepaper**: [`docs/TELOS_Whitepaper.md`](docs/TELOS_Whitepaper.md)
- **Quick Start**: [`QUICKSTART.md`](QUICKSTART.md)
- **Running Test 0**: [`docs/RUNNING_TEST_0.md`](docs/RUNNING_TEST_0.md)
- **Architecture**: [`docs/TELOS_Architecture_and_Development_Roadmap.md`](docs/)

-----

## License

© 2025 Origin Industries PBC / TELOS Labs LLC

Released under research-use and public-benefit license (pending finalization).

-----

## Contact

**Origin Industries PBC / TELOS Labs LLC**

For collaboration inquiries: [contact information]

For issues: Open a GitHub issue

-----

## Status

**Phase**: Internal Validation (Test 0)  
**Version**: 1.0.0  
**Updated**: October 2025

Test 0 validation in progress. Full results pending completion of controlled studies with institutional partners.