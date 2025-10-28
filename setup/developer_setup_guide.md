TELOS Developer Setup Guide

This guide gets you from zero → running TELOS in under 15 minutes.
Intended for developer onboarding / handoff.

⸻

1. Clone & Install

git clone https://github.com/your-org/telos.git
cd telos

# Base install
pip install -e .

# With semantic embeddings (recommended)
pip install -e ".[embeddings]"

# With dev tools (pytest, linters, etc.)
pip install -e ".[dev]"


⸻

2. Environment Variables

Provision a Mistral API key and export it:

export MISTRAL_API_KEY="your-key-here"

Confirm:

echo $MISTRAL_API_KEY


⸻

3. Config File

Governance perimeters live in config.json.

Example:

{
  "purpose": ["demonstrate AI governance"],
  "scope": ["technical explanation", "implementation guidance"],
  "boundaries": ["no harmful content", "respect privacy"],
  "privacy_level": 0.8,
  "constraint_tolerance": 0.2,
  "task_priority": 0.7
}

🔑 constraint_tolerance:
	•	0.0 = strict (zero drift allowed)
	•	1.0 = permissive (wide drift allowed)

⸻

4. Smoke Test

make smoke

Ensures dependencies and config are wired correctly.

⸻

5. Interactive Session

Launch with developer dashboard:

python -m telos_purpose.sessions.run_with_dashboard --config config.json

Available dashboard commands:
status, explain, diagnose, history, intervention, watch.

⸻

6. Streamlit UI (Funding Demo)

Run live comparison:

streamlit run telos_purpose/ui/streamlit_live_comparison.py

Features:
	•	Chat Tab → Test prompts across Stateless, Heuristics, TELOS
	•	Metrics Tab → Live fidelity, latency, overhead plots
	•	Narrative Tab → TELOS plain-English math explanation

⸻

7. Validation Study

Run comparative validation:

python -m telos_purpose.validation.run_validation \
  --config config.json \
  --conversation test_convo.json \
  --study-id pilot_001

Outputs → validation_results/

⸻

8. Optional: Semantic Embeddings

pip install sentence-transformers

Switch to SentenceTransformerProvider in config or steward init for semantic fidelity.

⸻

✅ You’re Ready

With this setup, you can:
	•	Run governed sessions
	•	Compare TELOS vs baselines
	•	Generate reproducible validation data
	•	Demo live with Streamlit UI
