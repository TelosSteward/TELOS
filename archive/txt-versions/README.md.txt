# TELOS: Mathematical Foundations for Runtime AI Governance  
**Origin Industries PBC / TELOS Labs LLC**  
*Canonical Repository Overview – Version 2.0 (October 2025)*  

---

## 1. Overview

**TELOS (Telically Entrained Linguistic Operating Substrate)** is a mathematical runtime framework for *persistent AI governance*.  
It transforms human-declared purpose, scope, and boundaries into measurable dynamics that maintain alignment across multi-turn interactions.

Unlike traditional AI safety filters that operate at design time, TELOS measures and corrects *drift in real time* — ensuring conversational systems remain in-governance throughout their operational lifecycle.

> **Core Principle:** Governance is no longer aspirational — it is observable, measurable, and correctable.

---

## 2. Core Concept

TELOS operates through two architectural constructs:

- **Primacy Basin** – the orchestration-layer domain defining the governance field of interaction.  
  It is not part of model embedding space but a runtime geometry constraining conversation dynamics.  

- **Runtime Steward (Teleological Operator)** – the system process that performs proportional correction toward the *Primacy State*.  
  It continuously measures fidelity, gravity, and orbit — ensuring adherence to declared purpose within the Primacy Basin.

Mathematically, this dynamic is expressed as:

\[
x_{t+1} = f(x_t) - K_p \cdot e_t, \quad e_t = \|x_t - \hat{a}\|
\]

Where \( e_t \) represents deviation from the Primacy Attractor and \( K_p \) determines corrective strength.

---

## 3. Repository Overview

telos/
├── core/                                  # Mathematical & runtime orchestration core
│   ├── primacy_math.py                    # Attractor dynamics and stability computation
│   ├── unified_steward.py                 # Runtime Steward (Teleological Operator)
│   ├── intervention_controller.py         # Proportional correction engine
│   ├── telemetry_utils.py                 # Turn-level and session-level telemetry
│   ├── embedding_provider.py              # Embedding provider abstraction
│   └── conversation_manager.py            # Session-level message handling
│
├── validation/                            # Validation and benchmarking framework
│   ├── run_internal_test0.py              # Executes Internal Test 0
│   ├── summarize_internal_test0.py        # Summarizes validation output
│   ├── comparative_test.py                # Comparative analysis across modes
│   ├── retro_analyzer.py                  # Retroactive governance replay
│   ├── system_health_monitor.py           # Runtime diagnostic (the EKG of TELOS)
│   └── telemetry_utils.py                 # Validation export utilities
│
├── sessions/                              # Execution-level runtime tools
│   ├── run_with_dashboard.py              # Visualization-enabled runtime runner
│   ├── observation_validation_run.py      # Observation-only governance run
│   ├── profile_extractor_cli.py           # CLI user-profile extractor
│   └── dashboard_assets/                  # Visualization templates and configs
│
├── cli/                                   # Developer automation and operational tools
│   ├── TELOS_CLI_Tool_Extraction.cli.py
│   ├── CLI_Reference.md
│   ├── MakeFile_OPs_Commands.txt
│   └── run_dashboard.py
│
├── setup/                                 # Environment setup and developer onboarding
│   ├── TELOS_Dev_Environment_Setup.md
│   ├── TELOS_Dev_Playbook.md
│   └── TELOS_Integrations_Handoff.txt
│
├── governance/                            # Governance prototypes and legacy validation
│   ├── TELOS_Internal_Test_0.txt
│   ├── TELOS_Structure_with_README.txt
│   └── Unified_Steward.py
│
├── public/                                # Public-facing materials
│   ├── TELOS_Executive_Summary.md
│   ├── TELOS_Grant_Application.txt
│   └── Why_TELOS_Had_to_Be_an_OS.md
│
├── docs/                                  # Canonical documentation corpus
│   ├── TELOS_Whitepaper.md
│   ├── TELOS_Architecture_and_Development_Roadmap.md
│   ├── TELOS_Developer_and_Research_Operations_Guide.md
│   ├── TELOS_Documentation_Index_v2.0.md
│   ├── TELOS_Repository_Structure.md
│   └── README.md
│
├── validation_results/                    # Validation output (gitignored)
│   ├── internal_test0/
│   ├── pilot0/
│   └── federation/
│
├── config.json                            # Governance and runtime configuration
├── Makefile                               # Build and validation automation
├── requirements.txt                       # Dependency manifest
├── .gitignore                             # Ignore rules
└── README.md                              # (This file)

---

## 4. Developer Quick Start

### Installation
```bash
git clone https://github.com/your-org/telos.git
cd telos
pip install -e .
export MISTRAL_API_KEY="your_key_here"

Validate Core Operation

python -m telos_purpose.validation.run_internal_test0

Expected Output:

✓ INTERNAL TEST 0 COMPLETE — results saved to validation_results/internal_test0/

Summarize Results

python -m telos_purpose.validation.summarize_internal_test0

Expected Metrics:

Mode	Avg Fidelity	Basin Adherence	Interventions	Result
stateless	0.65	0.58	0%	baseline
cadence	0.74	0.67	33%	periodic correction
telos	0.91	0.93	18%	best performer


⸻

5. Documentation

Document	Description
TELOS Whitepaper	Core scientific paper detailing runtime governance mathematics
Architecture & Development Roadmap	Technical evolution and milestone tracking
Developer & Research Operations Guide	Internal processes for validation, testing, and deployment
Documentation Index (v2.0)	Unified entry point linking all internal resources
Executive Summary	Condensed version for institutional review
Grant Application	Emergent Ventures / PBC submission document


⸻

6. Validation Milestones

Phase	Objective	Deliverable	Status
Internal Test 0	Mechanical verification	Validation Report (10 files)  Complete
Pilot 0	Controlled institutional study	Federated protocol	🔄 In Progress
Federation 1.0	Cross-institutional aggregation	Privacy-preserving delta protocol	Planned Q2 2026


⸻

7. Licensing & Governance

Entity: Origin Industries PBC
Research Arm: TELOS Labs LLC

All development follows the Origin Industries Public Benefit Charter:

To advance the measurable, ethical, and transparent governance of intelligent systems for human benefit.

Distributed contributions are governed via the Proof-of-Contribution Registry (PoConRegistry) — a cryptographic ledger of validated participation.

⸻

8. Contact & Collaboration

For institutional collaboration, validation participation, or research partnership:
	•	Email: research@origin-industries.org
	•	Website: origin-industries.org/telos
	•	X (formerly Twitter): @origin_tech
	•	GitHub Discussions: /discussions (coming post-federation)

⸻

Document Version: 2.0
Maintained by: TELOS Labs Documentation Steward
Status: Canonical – Active Repository Overview

---
