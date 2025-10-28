# TELOS Dev Playbook — Human-Dev Handoff Edition

## 0. Core Philosophy

- AI scaffold is a draft. Use it only for reference.
- Human refactor required. Every module should be rewritten or reviewed by a human dev.
- Contracts matter. As long as the input/output contracts stay consistent, your dev has full freedom in implementation style.
- Goal: Maintainability + empirical validation.

-----

## 1. File Responsibilities (What Each Module Must Do)

```
telos_purpose/
├── llm_clients/
│   └── mistral_client.py
│       - Mistral API adapter with retry logic
│       - Contract: `generate(messages: List[Dict]) -> str`
│
├── core/
│   ├── primacy_math.py
│   │   - Math ops: embeddings, basin membership, Lyapunov
│   │   - Already validated; minimal changes needed
│   │
│   ├── intervention_controller.py
│   │   - Proportional control (Kp * error_signal)
│   │   - Threshold logic based on constraint_tolerance
│   │
│   ├── unified_steward.py
│   │   - Runtime governor (the orchestrator)
│   │   - Contract: 
│   │       `process_turn(user_input, model_response) -> Dict`
│   │       `end_session() -> Dict`
│   │
│   ├── conversation_manager.py
│   │   - Session history and context injection
│   │
│   └── embedding_provider.py
│       - DeterministicEmbeddingProvider (testing)
│       - SentenceTransformerProvider (production)
│
├── validation/
│   ├── baseline_runners.py
│   │   - StatelessRunner, PromptOnlyRunner, CadenceRunner, TELOSRunner
│   │   - All use constraint_tolerance
│   │
│   ├── heuristics_baseline.py
│   │   - Keyword-based comparison system
│   │
│   └── run_validation.py
│       - CLI: run 4-way baseline comparison
│
└── sessions/
    └── run_with_dashboard.py
        - CLI: interactive governed session with dashboard
```

-----

## 2. Critical Contracts

These are non-negotiable — everything else is up to dev style.

### steward.process_turn()

```python
{
  "turn_number": int,
  "final_response": str,
  "response_was_modified": bool,
  "governance_action": str,  # "none" | "reminder" | "regeneration" | "antimeta"
  "intervention_applied": bool,
  "metrics": {
    "primacy_basin_membership": bool,
    "error_signal": float,
    "lyapunov_value": float,
    "telic_fidelity": float
  }
}
```

### steward.end_session()

```python
{
  "session_id": str,
  "session_metadata": {
    "final_telic_fidelity": float,
    "trajectory_stability": float,
    "basin_adherence": float
  },
  "intervention_statistics": {
    "total_interventions": int,
    "by_type": Dict[str, int],
    "avg_strength": float
  },
  "governance_quality": str  # "EXCELLENT" | "GOOD" | "ACCEPTABLE" | "POOR"
}
```

### Validation Output (Comparative Study)

```python
{
  "study_id": str,
  "results": {
    "stateless": BaselineResult,
    "prompt_only": BaselineResult,
    "cadence_reminder": BaselineResult,
    "telos": BaselineResult
  },
  "analysis": {
    "fidelity_comparison": Dict[str, float],
    "deltas_vs_telos": Dict[str, Dict],
    "hypothesis_tests": Dict[str, Any]
  }
}
```

-----

## 3. Workflow (Developer Flow)

### Step 1: Verify Math Core

```bash
pytest tests/test_primacy_math.py -v
pytest tests/test_conversation_manager.py -v
```

Confirms: basin geometry, Lyapunov calculation, fidelity metrics.

### Step 2: Configure System

Edit `config.json`:

```json
{
  "purpose": ["demonstrate AI governance"],
  "scope": ["technical explanation", "implementation guidance"],
  "boundaries": ["no harmful content", "respect privacy"],
  "privacy_level": 0.8,
  "constraint_tolerance": 0.2,
  "task_priority": 0.7
}
```

**Key parameter:** `constraint_tolerance`

- 0.0-0.2: Strict (zero tolerance for drift)
- 0.3-0.5: Moderate
- 0.6-1.0: Permissive (high tolerance)

### Step 3: Test Interactive Session

```bash
# Set API key
export MISTRAL_API_KEY=your_key_here

# Run interactive session
python -m telos_purpose.sessions.run_with_dashboard --config config.json

# Use dashboard commands
> status
> explain
> diagnose
```

### Step 4: Run Validation Study

```bash
# Create test conversation
cat > test_convo.json << 'EOF'
{
  "conversation": [
    ["What is AI alignment?", ""],
    ["How does governance drift occur?", ""],
    ["Tell me about sports", ""],
    ["Back to alignment", ""]
  ]
}
EOF

# Run 4-way comparison
python -m telos_purpose.validation.run_validation \
  --config config.json \
  --conversation test_convo.json \
  --study-id pilot_001

# Results in: ./validation_results/pilot_001.json
```

### Step 5: Check Outputs

Expected structure:

```
validation_results/
└── pilot_001.json
    ├── fidelity_comparison (all 4 baselines)
    ├── deltas_vs_telos
    ├── hypothesis_tests (H1, H2)
    └── aggregate_metrics
```

-----

## 4. Success Criteria

**Hypothesis 1 (H1): Minimum Improvement**

- ΔF > 0.15 (TELOS improves fidelity by at least 15% vs best baseline)

**Hypothesis 2 (H2): Best Performance**

- TELOS achieves highest fidelity score across all conditions

**Sample Requirements:**

- n ≥ 10 session pairs minimum
- Reproducible results across runs
- Statistical validation (effect sizes computed)

**Latency Requirements:**

- Average overhead < 100ms
- Intervention latency scales with tolerance
- No intervention when error < threshold

-----

## 5. Developer Guidance (Human Touch)

### Code Quality

- **Refactor for clarity**: Shorter functions, consistent naming
- **Intent-focused comments**: Why the function exists, not just what it does
- **Prefer explicit over clever**: Fewer abstractions, more readability
- **Document configs**: `config.json` is the only moving part non-devs should edit

### Testing Philosophy

- Math core: Unit tests ensure correctness
- Integration: Validation studies prove effectiveness
- Regression: Archive validation results for comparison

### Future-Proofing

- Note TODOs where deeper math hooks can be added
- Keep intervention thresholds configurable
- Design for multiple embedding providers
- Plan for federated learning extensions

-----

## 6. Minimal Example Config (Ready to Run)

```json
{
  "purpose": ["demonstrate AI governance mechanisms"],
  "scope": ["technical explanation", "implementation guidance", "mathematical foundations"],
  "boundaries": ["no harmful content", "respect privacy", "stay technical"],
  "privacy_level": 0.8,
  "constraint_tolerance": 0.2,
  "task_priority": 0.7,
  "export_dir": "./purpose_protocol_exports"
}
```

-----

## 7. Known Issues & Deprecations

### Bootstrap Deprecated

The old `bootstrap_telos_full.py` is **archived and unmaintained**. It used the deprecated `constraint_rigidity` parameter.

**Do not use bootstrap.** Instead:

1. Clone the repo
1. `pip install -e .`
1. Configure via `config.json` with `constraint_tolerance`

### Parameter Migration Complete

All files now use `constraint_tolerance` (0.0=strict, 1.0=loose).

If you encounter any references to `constraint_rigidity`, those files are outdated and should not be used.

-----

## 8. Quick Reference

**Key Files:**

- Config: `config.json`
- Math: `telos_purpose/core/primacy_math.py`
- Steward: `telos_purpose/core/unified_steward.py`
- Validation: `telos_purpose/validation/run_validation.py`

**Key Commands:**

```bash
# Interactive session
make run

# Validation study
make validate STUDY_ID=my_test

# Tests
pytest -v
```

**Key Metrics:**

- Fidelity: 0.0-1.0 (percentage of turns in basin)
- Lyapunov: Stability indicator (should decrease)
- Error Signal: Distance from attractor center
- Intervention Rate: Percentage of turns requiring correction

-----

This playbook is current as of the constraint_tolerance migration. All code artifacts use the new parameter semantics.