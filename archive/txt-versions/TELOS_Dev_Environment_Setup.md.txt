⚙️ Dev Environment Setup

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

Provision a Mistral API key and export it in your shell:

export MISTRAL_API_KEY="your-key-here"

Confirm it’s set:

echo $MISTRAL_API_KEY


⸻

3. Config File

Governance parameters live in config.json. Example:

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

4. Quick Validation

Run a smoke test to check dependencies + config:

make smoke


⸻

5. Interactive Session

Launch with live developer dashboard:

python -m telos_purpose.sessions.run_with_dashboard --config config.json

Use dashboard commands (status, explain, diagnose, etc.) during session.

⸻

6. Streamlit UI (Live Comparison)

For demos/funding pitch:

streamlit run telos_purpose/ui/streamlit_live_comparison.py

This opens a browser UI with:
	•	Chat Tab → Send prompts to Stateless, Heuristics, and TELOS
	•	Metrics Tab → Live plots of fidelity, latency, overhead
	•	Narrative Tab → Plain-English explanation of TELOS math

⸻

7. Validation Study

Run comparative validation:

python -m telos_purpose.validation.run_validation \
  --config config.json \
  --conversation test_convo.json \
  --study-id pilot_001

Results saved in validation_results/.

⸻

8. Optional: Embeddings

For higher fidelity semantic analysis:

pip install sentence-transformers

Switch embedding provider in config.json or in the steward init.

⸻

✅ With this setup your dev has everything:
	•	Mistral key in place
	•	Streamlit UI running
	•	Governed sessions + validation reproducible
