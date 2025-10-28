TELOS Developer & Research Operations Guide

This document provides the full operational workflow for building, validating, and managing the TELOS runtime governance framework — from zero setup through Internal Test 0 and pre-registration readiness.

It is designed for researchers, developers, and institutions who wish to reproduce, extend, or audit TELOS as a measurement infrastructure for runtime AI governance.

By following this guide, a new contributor can:
	•	Set up the TELOS environment from scratch
	•	Run Internal Test 0 to verify baseline and governed session fidelity
	•	Generate reproducible telemetry and summary results
	•	Use the health monitoring and repository audit systems for diagnostics
	•	Manage repository, validation, and organizational setup for ongoing collaboration

All configuration and validation steps are designed for transparency, reproducibility, and eventual open research publication.
Sensitive credentials (e.g., API keys) should always be stored locally in a .env file and never committed to the repository.

⸻

🧭 Phase 0 — Environment Setup
	1.	Clone the Repository

git clone https://github.com/your-org/telos.git
cd telos
pip install -e .

	2.	Set Up Environment Variables

Copy the template file and add your actual API key:

cp .env.template .env

Then edit .env:

MISTRAL_API_KEY="your-key-here"

	3.	Verify Environment

make check-env

Expected output:

Environment check passed.


⸻

⚙️ Phase 1 — Internal Test 0 Execution
	1.	Run Comparative Validation

make internal-test

This will:
	•	Clean previous artifacts
	•	Run the five validation modes (stateless, prompt-only, cadence, observation, telos)
	•	Export telemetry (CSV + JSON) to validation_results/internal_test0/
	•	Generate a summary table and hypothesis results (ΔF, best performer)

	2.	Confirm Output

You should see:

✓ Exported telemetry: session_XXXXX
TELOS Internal Test 0 Summary Complete

	3.	Check Validation Files

	•	validation_results/internal_test0/*_summary.json
	•	validation_results/internal_test0/*_turns.csv

⸻

🩺 Phase 2 — System Health Monitoring

The SystemHealthMonitor continuously evaluates system-level integrity during runtime.

Run with Health Monitoring:

python -m telos_purpose.validation.system_health_monitor

It checks for:
	•	NaN or infinite values in telemetry
	•	Unstable Lyapunov metrics
	•	Threshold violations or failed interventions
	•	Anomalous runtime states

Outputs:
	•	diagnostics_<session>.json
	•	diagnostics_<session>.csv
	•	Optional: auto-generated plots (Lyapunov drift, fidelity decay)

Interpretation:
	•	✅ Stable: All signals within range
	•	⚠️ Degraded: Some minor drift or numeric instability detected
	•	❌ Critical: Governance logic or attractor math not functioning

⸻

🧮 Phase 3 — Repository Audit & Verification

Before submission, collaboration, or publication, every TELOS environment must pass the repository audit.

Run the Audit

Option 1 (recommended):

make audit

Option 2 (manual):

python -m telos_purpose.utils.repo_audit

What the Audit Does

Check	Purpose	Expected Outcome
Package structure	Ensures all Python packages include __init__.py	✅ “All packages valid”
Stale artifacts	Detects leftover telemetry (e.g., validation_results/)	⚠️ Warning → run make clean
Schema consistency	Verifies JSON summaries use same schema version	✅ “Schema: consistent”
Environment reproducibility	Confirms .env.template + requirements.lock.txt exist	✅ “Environment reproducible”
Documentation integrity	Checks README references (e.g., config.json, test files)	✅ “All referenced files present”

Output Example

======================================================================
TELOS REPOSITORY AUDIT
======================================================================

ERRORS (1):
  [DOC.MISSING_FILE] README references config.json but file does not exist

WARNINGS (2):
  [ARTIFACT.STALE] Found 4 uncommitted artifacts in validation_results/
  [ENV.NO_LOCK] No requirements.lock.txt found

INFO (1):
  [ENV.NO_TEMPLATE] No .env.template found (create example for API key setup)

======================================================================
STATUS: FAIL (errors must be fixed)
======================================================================

How to Interpret Results

Severity	Meaning	Action
✅ PASS	No errors found	Ready for submission
⚠️ WARNINGS	Minor repository hygiene issues	Optional cleanup
❌ FAIL	Missing files or invalid package	Fix before continuing

Common Fix Commands

make clean
pip freeze > requirements.lock.txt
cp .env.template .env

Re-run Audit:

make audit

Expected:

✓ No issues found — repository ready for submission
STATUS: PASS


⸻

🧩 Phase 4 — Collaborative Research & Pre-Registration Readiness

Once Internal Test 0 passes and the audit is clean:
	1.	Archive Validation Results
	•	Keep validation_results/internal_test0/ for reproducibility
	•	Do not modify raw CSV or JSON outputs
	2.	Generate Summary for Review

python -m telos_purpose.validation.test0_quick_summary


	3.	Document Version and Parameters
	•	Record: git commit hash, TELOS version, constraint_tolerance used
	•	Store in validation_results/metadata_summary.json
	4.	Prepare for Open Review
	•	Optional: push results to GitHub (if public repo)
	•	Recommended: submit archived data to OSF or Zenodo

⸻

🧠 Summary

Stage	Goal	Verification
Phase 0	Environment setup	make check-env passes
Phase 1	Run Internal Test 0	5 CSV + 5 JSON exported
Phase 2	Monitor runtime stability	Health monitor clean
Phase 3	Validate repository integrity	Audit passes with 0 errors
Phase 4	Package for collaboration	Results reproducible and documented


⸻

© 2025 TELOS Labs
Research infrastructure for runtime governance and empirical AI oversight.
