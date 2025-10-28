
TELOS Dev Playbook — Human-Dev Handoff Edition

0. Core Philosophy
	•	AI scaffold is a draft. Use it only for reference.
	•	Human refactor required. Every module should be rewritten or reviewed by a human dev.
	•	Contracts matter. As long as the input/output contracts stay consistent, your dev has full freedom in implementation style.
	•	Goal: Maintainability + empirical validation.

⸻

1. File Responsibilities (What Each Module Must Do)

telos_purpose/
├── llm_clients.py
│   - Adapters for LLMs (Mistral SDK, REST fallback, SimulatedClient).
│   - Contract: `generate(messages: List[Message]) -> str`.
│
├── integrated_orchestrator.py
│   - High-level pipeline: user input → LLM response → governance → metrics.
│   - Tracks session history and turn results.
│   - Contract: must use `steward.process_turn()` and `steward.end_session()`.
│
├── core/
│   ├── primacy_math.py
│   │   - Math ops: embeddings, basin membership, Lyapunov check.
│   │   - Already exists; leave mostly untouched.
│   │
│   ├── primacy_controller.py
│   │   - Corrective force logic (Kp * error).
│   │   - Already exists; leave mostly untouched.
│   │
│   ├── steward.py
│   │   - Runtime governor.
│   │   - Contract: 
│   │       `process_turn(user_input, model_response) -> Dict`
│   │       `end_session() -> Dict`
│   │
│   ├── metric_spec.py
│   │   - Defines canonical metrics (telic_fidelity, corrections, Lyapunov).
│   │   - Contract: `get_metric_spec() -> Dict`.
│   │
│   └── delta_export.py
│       - Takes governed + stateless JSON, computes ΔM, exports bundles.
│       - Contract: 
│           `export_delta_bundle(gov_path, stat_path, out_path) -> str`
│           `aggregate_deltas(delta_paths, out_path) -> str`
│
├── run_governed_session.py
│   - CLI: run a governed session (interactive or scripted).
│
└── run_full_comparison.py
    - CLI: run N governed vs stateless pairs, compute deltas, aggregate results.


⸻

2. Critical Contracts

These are non-negotiable — everything else is up to dev style.

steward.process_turn()

{
  "turn_number": int,
  "governance_action": str,  # "Maintained" | "Corrected" | "DISABLED"
  "metrics": {
    "telic_fidelity": float,   # 0.0–1.0
    "correction_applied": bool
  }
}

steward.end_session()

{
  "session_metadata": {
    "final_telic_fidelity": float,
    "corrections_applied": int,
    "lyapunov_violations_corrected": int,
    "total_alerts": int
  },
  "governance_quality": str  # "High" | "Moderate" | "Low"
}

delta_export
	•	Inputs: governed_session.json, stateless_session.json
	•	Output: delta_bundle.json with ΔM (fidelity, corrections, Lyapunov, alerts).

⸻

3. Workflow (Developer Flow)
	1.	Verify math core
	•	Run pytest tests/test_corrected_enforcement.py.
	•	Confirms basin geometry, correction, Lyapunov enforcement.
	2.	Implement/review integration
	•	Write llm_clients.py cleanly (Mistral SDK + REST fallback).
	•	Write integrated_orchestrator.py in a human style: minimal boilerplate, clear flow.
	•	Orchestrator = session manager.
	3.	Implement metrics & export
	•	metric_spec.py: just a dict.
	•	delta_export.py: compute ΔM, add optional differential privacy.
	4.	Write CLIs
	•	run_governed_session.py: manual testing.
	•	run_full_comparison.py: automated validation pipeline.
	5.	Run quick tests

python run_governed_session.py --provider simulated
export MISTRAL_API_KEY=your-key
python run_governed_session.py --provider mistral --semantic


	6.	Run validation

python run_full_comparison.py --sessions 20 --provider mistral


	7.	Check outputs
	•	Expect comparison_results/ with governed, stateless, deltas, federated aggregate.

⸻

4. Success Criteria
	•	ΔF > 0.10 (governance improves fidelity)
	•	ΔC < 0 (governance reduces corrections)
	•	n ≥ 10 (sample size)
	•	Metrics export reproducible JSON bundles.

⸻

5. Developer Guidance (Human Touch)
	•	Refactor for clarity: shorter functions, consistent naming.
	•	Write intent-focused comments: why the function exists, not just what it does.
	•	Prefer explicit over clever: fewer abstractions, more readability.
	•	Document configs: config.json is the only moving part non-devs should edit.
	•	Future-proof: note TODOs where deeper math hooks can be added later.

⸻

6. Minimal Example Config (ready to run)

{
  "purpose": ["demonstrate AI governance"],
  "scope": ["technical explanation", "implementation guidance"],
  "boundaries": ["no harmful content", "respect privacy"],
  "privacy_level": 0.8,
  "constraint_rigidity": 0.9,
  "task_priority": 0.7,
  "conversation": [
    ["What is AI governance?", ""],
    ["How does TELOS prevent drift?", ""],
    ["Show me the metrics.", ""]
  ]
}

