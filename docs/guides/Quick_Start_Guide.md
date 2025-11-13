
TELOS Quickstart Guide

1. Install

git clone https://github.com/your-org/telos.git
cd telos

# Base install
pip install -e .

# With semantic embeddings
pip install -e ".[embeddings]"

# With dev tools
pip install -e ".[dev]"


⸻

2. Configure

Edit config.json to set governance perimeters:

{
  "purpose": ["explain AI governance"],
  "scope": ["AI alignment", "runtime oversight"],
  "boundaries": ["no harmful content"],
  "privacy_level": 0.8,
  "constraint_tolerance": 0.1,
  "task_priority": 0.7
}

	•	constraint_tolerance:
	•	0.0 = strict (zero tolerance)
	•	1.0 = permissive (wide tolerance)

⸻

3. Run an Interactive Session

python -m telos_purpose.sessions.run_with_dashboard --config config.json

Live dashboard shows fidelity, error signals, interventions.

⸻

4. Run a Test Conversation

python -m telos_purpose.runners.TELOS_runner_script \
  --config config.json \
  --conversation telos_purpose/test_conversations/test_convo_001.json


⸻

5. Run a Validation Study

python -m telos_purpose.validation.run_validation \
  --config config.json \
  --conversation test_convo.json \
  --study-id pilot_001

Outputs saved in validation_results/.

⸻

6. Outputs
	•	Session telemetry → purpose_protocol_exports/session_*.json
	•	Validation results → validation_results/study_*.json
	•	Logs → logs/telemetry_log.jsonl

⸻

7. First Things to Try
	1.	Edit constraint_tolerance in config.json:
	•	0.1 → stricter
	•	0.8 → more permissive
	2.	Run:

make run        # Interactive session
make validate   # Validation study


