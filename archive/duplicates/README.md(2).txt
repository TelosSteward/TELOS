# TELOS: Mathematical Foundations for Runtime AI Governance

**Origin Industries PBC / TELOS Labs LLC**  
Canonical Repository Overview – Version 2.0 (October 2025)  
Maintainer: @TelosSteward

---

## 1 | Overview

TELOS (Telically Entrained Linguistic Operating Substrate) is a mathematical runtime framework for persistent AI governance.
It transforms human-declared purpose, scope, and boundaries into measurable dynamics that maintain alignment across multi-turn interactions.

Unlike static AI safety filters that operate only at design time, TELOS measures and corrects drift in real time — ensuring conversational systems remain in-governance throughout their operational lifecycle.

**Core Principle:** Governance is no longer aspirational — it is observable, measurable, and correctable.

---

## 2 | Core Concept

TELOS operates through two architectural constructs:

- **Primacy Basin** – the orchestration-layer domain defining the governance field of interaction. It is not part of model embedding space but a runtime geometry constraining conversational dynamics.

- **Runtime Steward (Teleological Operator)** – the system process that performs proportional correction toward the Primacy State. It continuously measures fidelity, gravity, and orbit — ensuring adherence to declared purpose within the Primacy Basin.

Mathematically, this dynamic is expressed as:

x_{t+1} = f(x_t) - K_p · e_t,    where e_t = ||x_t - â||

Where `e_t` represents deviation from the Primacy Attractor and `K_p` determines corrective strength.
Observation and intervention are unified into a single continuous operation — the Teleological Operator.

---

## 3 | Repository Structure (v2.0)

telos/
├── telos_purpose/                  # Primary package
│   ├── core/                       # Mathematical & runtime orchestration
│   │   ├── primacy_math.py
│   │   ├── unified_steward.py
│   │   ├── intervention_controller.py
│   │   ├── embedding_provider.py
│   │   └── conversation_manager.py
│   │
│   ├── validation/                 # Empirical verification framework
│   │   ├── run_internal_test0.py
│   │   ├── summarize_internal_test0.py
│   │   ├── comparative_test.py
│   │   ├── retro_analyzer.py
│   │   ├── system_health_monitor.py
│   │   └── telemetry_utils.py
│   │
│   ├── sessions/                   # Runtime execution & visualization
│   │   ├── run_with_dashboard.py
│   │   ├── observation_validation_run.py
│   │   └── dashboard_assets/
│   │
│   ├── llm_clients/                # Model adapters (e.g., Mistral)
│   ├── test_conversations/         # Validation datasets
│   └── dev_dashboard/              # Optional live dashboards
│
├── docs/                           # Canonical documentation corpus
│   ├── TELOS_Whitepaper.md
│   ├── TELOS_Architecture_and_Development_Roadmap.md
│   ├── TELOS_Developer_and_Research_Operations_Guide.md
│   ├── TELOS_Repository_Structure_v2.0.md
│   └── TELOS_Documentation_Index_v2.0.md
│
├── public/                         # Public & regulatory materials
│   ├── TELOS_Executive_Summary.md
│   └── TELOS_Grant_Application.txt
│
├── setup/                          # Environment setup & integration handoff
│   └── TELOS_Integrations_Handoff.txt
│
├── validation_results/             # Validation outputs (gitignored)
│
├── config.json                     # Governance configuration
├── Makefile                        # Build / validation automation
├── requirements.txt                # Dependency manifest
├── setup.py
└── .gitignore

📄 **Full directory map:** docs/TELOS_Repository_Structure_v2.0.md  
📚 **Documentation index:** docs/TELOS_Documentation_Index_v2.0.md

---

## 4 | Developer Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/your-org/telos.git
cd telos

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set API key (example: Mistral)
export MISTRAL_API_KEY="your_key_here"

# 4. Verify core imports
python -c "from telos_purpose.core import primacy_math, unified_steward; print(' TELOS core ready')"

# 5. Run Internal Test 0
python -m telos_purpose.validation.run_internal_test0
python -m telos_purpose.validation.summarize_internal_test0

Expected output → validation_results/internal_test0/ (5 CSV + 5 JSON)
Success criteria → ΔF ≥ 0.15 and highest fidelity for TELOS mode.

➡️ For full operational handoff and tagging procedures, see setup/TELOS_Integrations_Handoff.txt.

⸻

5 | Validation Metrics & Results (Expected Pattern - Pending Internal Test 0)

Note: The table below shows expected patterns based on system design. Actual results will be updated after Internal Test 0 completes.

Mode	Avg Fidelity	Basin Adherence	Interventions	Result
stateless	0.65	0.58	0%	baseline
cadence	0.74	0.67	33%	periodic correction
telos	0.91	0.93	18%	best performer

Interpretation (expected): TELOS should improve governance persistence by ~17% over cadence while reducing intervention frequency by ~45%. Each turn produces structured telemetry (fidelity, error_signal, ΔV, intervention_type, latency_ms).

⸻

6 | Governance Charter & Licensing

Entity: Origin Industries PBC
Research Arm: TELOS Labs LLC

All development follows the Origin Industries Public Benefit Charter:

To advance the measurable, ethical, and transparent governance of intelligent systems for human benefit.

Distributed contributions are recorded via the Proof-of-Contribution Registry (PoConRegistry) — a cryptographic ledger of validated participation.

© 2025 Origin Industries PBC / TELOS Labs LLC
Released under a research-use and public-benefit license (pending finalization).

⸻

7 | Documentation References

Document	Purpose
docs/TELOS_Whitepaper.md	Foundational mathematical and philosophical framework
docs/TELOS_Architecture_and_Development_Roadmap.md	System architecture and release timeline
docs/TELOS_Developer_and_Research_Operations_Guide.md	Developer setup and validation procedures
docs/TELOS_Repository_Structure_v2.0.md	Canonical repository layout and directory definitions
docs/TELOS_Documentation_Index_v2.0.md	Comprehensive documentation map
public/TELOS_Executive_Summary.md	Concise overview for stakeholders and regulators


⸻

8 | Contact & Collaboration

For institutional collaboration, validation participation, or research partnership:
	•	Contact information available post-validation
	•	GitHub Discussions: Available after Federation Phase 1

For urgent inquiries, open a GitHub issue.

⸻

9 | Version & Status

Phase	Objective	Deliverable	Status
Internal Test 0	Mechanical verification	Validation Report (10 files)	⏳ In Progress
Pilot 0	Controlled institutional study	Federated protocol	📋 Planned
Federation 1.0	Cross-institutional aggregation	Privacy-preserving delta protocol	📋 Q2 2026 planned


⸻

Document Version: 2.0
Maintained by: @TelosSteward
Status: Canonical – Active Repository Overview
Last Updated: October 2025
