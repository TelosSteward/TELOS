TELOS Architecture and Development Roadmap

A concise reference for how TELOS works and how to execute Internal Test 0 — the final validation step before grant submission.

⸻

🧠 Core Architecture Overview

1. ConversationManager

Manages conversation history and context window.
Keeps governance context active and trims old turns to stay within token limits.

2. LLM Client (TelosMistralClient)

Handles API calls to Mistral (or other providers).
Centralizes retries, token accounting, and provider switching.

3. PrimacyAttractorMath

Embeds responses and measures semantic drift using foundations-aligned geometry
(τ-weighted center, r = 2 / ρ basin radius).
Computes error signals, fidelity, Lyapunov stability, and basin membership.

4. InterventionController

Applies proportional corrections only when drift exceeds tolerance thresholds
(ε_min, ε_max derived from constraint_tolerance).

5. UnifiedSteward

Coordinates all components, executes the runtime loop, logs telemetry,
and produces developer-facing explanations and diagnostics.

6. Validation Runners

Implements the five test modes used in Internal Test 0:
Stateless, Prompt-Only, Cadence, Observation, TELOS Runtime.

7. Telemetry Exports

Logs each turn’s fidelity, error, and interventions.
Outputs standardized CSV and JSON files for quantitative analysis.

⸻

⚙️ Runtime Sequence

sequenceDiagram
User->>ConversationManager: Input
ConversationManager->>LLMClient: Send prompt
LLMClient->>ConversationManager: Return response
ConversationManager->>PrimacyMath: Embed + measure drift
PrimacyMath->>InterventionController: Send metrics
InterventionController->>UnifiedSteward: Decide (monitor / remind / regenerate)
UnifiedSteward->>Telemetry: Log + export
UnifiedSteward->>User: Return governed response


⸻

🧩 Repository Checklist
	•	✅ telos_purpose/core/ — math, control, orchestration
	•	✅ telos_purpose/validation/ — test runners and telemetry utilities
	•	✅ telos_purpose/test_conversations/ — test_convo_001–003.json
	•	✅ config.json — governance parameters (constraint_tolerance, etc.)

⸻

🚀 Internal Test 0 — Final Validation Step

Goal: Demonstrate that TELOS measures drift, applies interventions proportionally, and exports coherent telemetry.
Five conditions confirm the full runtime pipeline functions end-to-end.

Condition	Description	Demonstrates
A — Stateless	No governance	Baseline drift pattern
B — Prompt-Only	Declaration at start	Declarative persistence
C — Cadence	Fixed 3-turn reminders	Scheduled correction
D — Observation	TELOS math active, no interventions	Pure drift detection
E — TELOS Runtime	Full adaptive governance	Proportional control

Run command

python -m telos_purpose.validation.run_internal_test0

Expected Output
	•	5 CSV + 5 JSON telemetry files in validation_results/internal_test0/
	•	Console summary with final fidelity per mode
	•	Evidence that Primacy Attractor math detects and corrects drift

⸻

📈 Minimal Deliverables for Grant Submission

Deliverable	Description	Status
Runtime Telemetry	Turn-level CSV + JSON from Internal Test 0	✅
Session Comparisons	Five-mode fidelity summary (comparative_summary.json)	✅
Explanatory Appendix	README + Architecture Overview (this doc)	✅
Pilot Brief	2-page summary of findings + plots (Pilot0_Brief.md)	🔜
Grant Package	Proposal + validated data attachments	🔜


⸻

🧰 Developer Commands

make run        # Launch governed interactive session
make validate   # Execute Internal Test 0
make clean      # Remove logs and cached results


⸻

🔍 Runtime Explainability Layer

Every intervention logs a plain-language rationale:
	•	“Fidelity drift 0.42 > ε_max (0.4): regenerating response.”
	•	“Minor drift below ε_min: no action taken.”

These explanations appear in the telemetry JSON under
intervention_reasoning, allowing transparent governance audits.

⸻

🧾 Validation Artifact Checklist
	•	✅ Turn-level CSV (exported)
	•	✅ Session-summary JSON
	•	✅ Comparative summary (JSON)
	•	✅ Pilot Brief with plots (optional for grants)

⸻

Last updated: October 2025 — aligned with PrimacyAttractorMath v2 (foundations-aligned geometry).

⸻
