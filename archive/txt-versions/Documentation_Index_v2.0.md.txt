TELOS Documentation Index (v2.0)

Canonical Condensed Developer & Research Reference
Origin Industries PBC / TELOS Labs LLC
Maintainer: @TelosSteward
Date: October 2025

⸻

0 | Purpose

This index defines the canonical documentation hierarchy for TELOS.
It establishes the relationships between public, developer, and research-facing materials, ensuring synchronization between conceptual, mathematical, and operational documentation.

Scope:
	•	Reflects all documentation stored under /docs/ and /public/.
	•	Maps directly to repository files listed in TELOS_Repository_Structure_v2.0.md.
	•	Provides reference order for institutional review, developer onboarding, and version-controlled publication.

⸻

1 | Document Hierarchy Overview

Category	File	Description
Conceptual Foundations	TELOS_Whitepaper.md	Canonical statement of TELOS theory: attractor dynamics, Primacy Basin geometry, and the teleological operator model for runtime governance.
	TELOS_Executive_Summary.md	Concise high-level overview for regulators, investors, and institutional partners.
	Why_TELOS_Had_to_Be.md	Philosophical essay outlining why runtime governance required a purpose-driven architectural design.
Technical Documentation	TELOS_Architecture_and_Development_Roadmap.md	System development milestones, feature evolution, and engineering architecture.
	TELOS_Developer_and_Research_Operations_Guide.md	Operational guide for developers and researchers to run, validate, and extend TELOS components.
	TELOS_Repository_Structure_v2.0.md	Complete mapping of the repository’s organizational and dependency hierarchy.
Validation & Regulatory Context	TELOS_Grant_Application.txt	Institutional funding document aligning TELOS validation goals to post-market monitoring standards (EU AI Act Art. 72, NIST RMF).
	TELOS_Documentation_Index_v2.0.md	(This file) — meta-index defining canonical order and cross-reference map.


⸻

2 | Conceptual Layer

Primary Documents:
	•	TELOS_Whitepaper.md — Defines TELOS as a measurable governance framework, detailing the Primacy Basin, attractor dynamics, and teleological feedback model.
	•	Why_TELOS_Had_to_Be.md — Provides philosophical rationale for runtime governance as a persistent alignment mechanism.
	•	TELOS_Executive_Summary.md — Summarizes the framework in accessible regulatory and institutional language.

Conceptual Cohesion:
These works collectively define the teleological and mathematical logic that underpins the runtime framework and articulate its ethical intent.

⸻

3 | Technical Layer

Primary Documents:
	•	TELOS_Architecture_and_Development_Roadmap.md — Engineering roadmap showing system evolution from Internal Test 0 → Pilot 0 → Federation 1.0.
	•	TELOS_Developer_and_Research_Operations_Guide.md — Complete usage instructions, including imports, validation workflow, and telemetry schema.
	•	TELOS_Repository_Structure_v2.0.md — Canonical reference of repository layout and component interdependencies.

Alias & Concept Integration:
This layer formalizes TELOS runtime components:
	•	TeleologicalOperator → runtime alias for UnifiedGovernanceSteward
	•	Primacy Basin → orchestration domain where governance dynamics operate
	•	Attractor Dynamics → continuous measurement-correction triad (fidelity, gravity, orbit)

⸻

4 | Validation & Regulatory Layer

Primary Documents:
	•	TELOS_Grant_Application.txt — Defines institutional and regulatory value, framing TELOS validation as applied compliance science.
	•	Validation artifacts (from validation_results/) — provide empirical evidence for Internal Test 0 and Pilot 0 results.

Purpose:
To demonstrate continuous runtime governance under real-world conditions and generate reproducible, regulator-ready telemetry.

⸻

5 | Cross-Reference Map

Area	Primary Source	Supporting File	Outcome
Mathematical Foundation	TELOS_Whitepaper.md	core/primacy_math.py	Defines attractor geometry, basin limits, and drift dynamics.
Runtime Orchestration	TELOS_Architecture_and_Development_Roadmap.md	core/unified_steward.py	Describes Runtime Steward ↔ TeleologicalOperator mapping.
Validation & Empiricism	TELOS_Developer_and_Research_Operations_Guide.md	validation/run_internal_test0.py, summarize_internal_test0.py	Quantifies fidelity improvements and proportional correction.
Governance Philosophy	Why_TELOS_Had_to_Be.md	TELOS_Whitepaper.md	Establishes ethical rationale for persistent governance.
Regulatory Relevance	TELOS_Grant_Application.txt	TELOS_Executive_Summary.md	Aligns runtime measurement to regulatory monitoring standards.


⸻

6 | Versioning & Update Policy
	•	Documentation follows semantic versioning: vMAJOR.MINOR (e.g., v2.0 → major revision to structure or scope).
	•	Any new file added to /docs/ or /public/ must be registered here with:
	1.	A one-line summary
	2.	Dependency or conceptual linkage
	3.	Category classification

Maintainer: @TelosSteward
Update Cadence: After each milestone — Internal Test 0, Pilot 0, Federation 1.0.

⸻

7 | Canonical File Map

/docs/
├── TELOS_Whitepaper.md
├── TELOS_Architecture_and_Development_Roadmap.md
├── TELOS_Developer_and_Research_Operations_Guide.md
├── TELOS_Repository_Structure_v2.0.md
├── TELOS_Documentation_Index_v2.0.md
└── README.md

/public/
├── TELOS_Executive_Summary.md
├── TELOS_Grant_Application.txt
└── Why_TELOS_Had_to_Be.md


⸻

8 | Summary

The TELOS documentation ecosystem functions as a layered evidence system:
	1.	Conceptual — defines theoretical and ethical foundations.
	2.	Technical — details implementation and system architecture.
	3.	Empirical — demonstrates measurable validation.
	4.	Regulatory — connects empirical outcomes to compliance frameworks.

Each layer reinforces the next, forming a closed documentation loop that mirrors TELOS’s teleological feedback model: purpose → measurement → correction → evidence.

⸻

Status: Canonical Documentation Index — Version 2.0
Maintainer: @TelosSteward
Last Updated: October 2025
