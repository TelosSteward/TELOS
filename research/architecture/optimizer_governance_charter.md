# Research Governance Charter: TELOS Governance Configuration Optimizer

**Document ID:** TELOS-RGC-001
**Version:** 1.0
**Effective Date:** 2026-02-20
**Owner:** TELOS AI Labs Inc.
**Classification:** Internal / Auditor-Accessible
**EU AI Act Reference:** Article 9 (Risk Management System)

---

### Disclosures

> **Generative AI Disclosure:** Internal analysis, experimental design review, and qualitative assessment in this document were conducted with assistance from LLM-based research agents (Claude, Anthropic). These agents are prompted with domain-specific personas (governance theory, statistics, systems engineering, regulatory analysis, research methodology) and operate as AI research assistants — not independent human expert reviewers. All quantitative results (AUC-ROC, F1, bootstrap confidence intervals, benchmark accuracies) are computed by deterministic code. Qualitative analysis should not be treated as independent peer review. See `research/research_team_spec.md` for full methodology.

> **Conflict of Interest Disclosure:** This research was conducted and funded by TELOS AI Labs Inc., which has a commercial interest in the TELOS governance framework. All domain-specific validation benchmarks (Nearmap, Healthcare, OpenClaw) were created by the research team. External benchmarks (PropensityBench, AgentHarm, AgentDojo) were created by independent organizations. Research artifacts are published on [Zenodo](https://zenodo.org/) with persistent DOIs. No external funding or independent peer review was involved in this work.

---

## 1. Purpose and Scope

This charter governs the operation of the **TELOS Governance Configuration Optimizer** (`analysis/governance_optimizer.py`), a Bayesian optimization system that automatically tunes the threshold parameters controlling TELOS governance decisions.

**What the optimizer does.** The TELOS governance framework uses Primacy Attractors (embedding-space representations of user purpose) to detect and direct conversational and agentic drift. The governance engine makes decisions (EXECUTE, CLARIFY, SUGGEST, ESCALATE) based on 14 tunable threshold parameters defined in `telos_governance/threshold_config.py`. The optimizer uses Optuna's Tree-structured Parzen Estimator (TPE) to search for parameter configurations that maximize a scalarized objective across multiple domain-specific benchmarks, subject to hard safety constraints.

**What this charter covers:**
- Who may run the optimizer and under what conditions
- What parameter ranges are searchable
- What safety gates prevent harmful configurations from being deployed
- How results are reviewed, approved, and recorded
- How this charter itself may be amended

**What this charter does not cover:**
- The governance framework itself (covered by the TELOS architecture documentation)
- Benchmark construction methodology (covered by individual benchmark PROVENANCE.md files)
- Research team operations (covered by `research/research_team_spec.md`)

**Current deployment environment:** Mac Mini M4 Pro (48GB), isolated research environment. No production deployment of optimizer-generated configurations without human approval. Future state: 24/7 autonomous optimization on isolated instances, still subject to human approval before deployment.

---

## 2. Authority and Roles

| Role | Who | Responsibility |
|------|-----|----------------|
| **Optimizer Operator** | Any TELOS developer with repo access | May run the optimizer and inspect results. May NOT deploy configs. |
| **Configuration Reviewer** | TELOS principal researcher (currently: JB) | Reviews optimizer output, regression reports, and ratchet flags. Approves or rejects candidate configs. |
| **Configuration Deployer** | TELOS principal researcher (currently: JB) | Merges approved configs into `telos_core/constants.py` and `ThresholdConfig` defaults. Tags the commit. |
| **Charter Custodian** | TELOS principal researcher (currently: JB) | Approves amendments to this charter and to optimizer search bounds. |
| **Research Team** | Russell, Gebru, Karpathy, Schaake, Nell (LLM agents) | Advisory review of optimizer design, safety mechanisms, and results. Not decision-makers. |

**Key constraint:** No optimizer output may be deployed to production without explicit human review and approval by the Configuration Reviewer. This applies regardless of whether all safety gates pass. The optimizer produces candidates; humans decide.

---

## 3. Acceptable Parameter Ranges

The optimizer searches over 14 parameters in `ThresholdConfig`, classified into three groups. Search bounds are defined in `suggest_threshold_config()` in `analysis/governance_optimizer.py`.

### 3.1 Bounded Threshold Parameters (9 parameters, ordering-constrained)

These parameters have dependent sampling to enforce ordering invariants.

| Parameter | Search Min | Search Max | Production Default | Ordering Constraint |
|-----------|-----------|-----------|-------------------|-------------------|
| `st_suggest` | 0.15 | 0.35 | 0.25 | -- |
| `st_clarify` | st_suggest + 0.05 | 0.42 | 0.35 | > st_suggest + 0.05 |
| `st_execute` | st_clarify + 0.05 | 0.52 | 0.45 | > st_clarify + 0.05 |
| `boundary_violation` | 0.60 | 0.80 | 0.70 | -- |
| `boundary_margin` | 0.02 | 0.15 | 0.05 | -- |
| `keyword_boost` | 0.05 | 0.30 | 0.15 | -- |
| `keyword_embedding_floor` | 0.25 | 0.55 | 0.40 | -- |
| `fidelity_orange` | 0.45 | 0.60 | 0.50 | -- |
| `fidelity_yellow` | fidelity_orange + 0.05 | 0.70 | 0.60 | > fidelity_orange + 0.05 |
| `fidelity_green` | fidelity_yellow + 0.05 | 0.80 | 0.70 | > fidelity_yellow + 0.05 |

### 3.2 Free Weight Parameters (5 parameters, softmax-normalized)

Raw weights are sampled independently and then normalized to sum to 1.0.

| Parameter | Raw Sample Min | Raw Sample Max | Production Default (normalized) |
|-----------|---------------|---------------|-------------------------------|
| `raw_weight_purpose` | 0.10 | 0.60 | 0.35 |
| `raw_weight_scope` | 0.05 | 0.40 | 0.20 |
| `raw_weight_tool` | 0.05 | 0.40 | 0.20 |
| `raw_weight_chain` | 0.05 | 0.30 | 0.15 |
| `raw_weight_boundary_penalty` | 0.05 | 0.25 | 0.10 |

### 3.3 Frozen Parameters (not in search space)

The following are explicitly excluded from optimization:

- **SAAI drift thresholds** (`SIMILARITY_BASELINE = 0.20`) -- Layer 1 hard block, set by design, not empirical tuning
- **`max_regenerations`** -- Intervention limit, not a scoring parameter

**Rationale for bounds.** Upper bounds on `st_execute` and `st_clarify` were narrowed from initial ranges (per Nell consensus, 2026-02-20) after validation showed that high values (e.g., st_execute=0.556, st_clarify=0.486) pushed most scenarios toward ESCALATE, creating an over-escalation failure mode. The current bounds preserve the valid search region while preventing this known failure mode.

---

## 4. Safety Gates and Constraints

The optimizer implements a **four-gate ratchet** system. All four gates must pass for a candidate configuration to advance the ratchet (i.e., become the starting point for the next generation). If any gate fails, the previous configuration is retained.

### Gate 1: No Category A Regressions

Any scenario that was correctly classified under the previous config but is misclassified under the candidate, where the scenario's boundary category is "A" (hard violations -- safety-critical boundaries), causes the gate to fail.

- **Implementation:** `generate_regression_report()` in `governance_optimizer.py`
- **Threshold:** Zero tolerance. One Cat A regression blocks the candidate.
- **Rationale:** Category A boundaries represent hard safety violations (e.g., unauthorized data access, harmful action execution). Regression on these is categorically unacceptable regardless of gains elsewhere.

### Gate 2: Holdout Category A Detection at 100%

The candidate config is evaluated on a held-out test split (30% of scenarios, stratified by boundary category). Cat A detection must be 100% on the holdout set.

- **Implementation:** `holdout_cat_a_pass` in `run_generation()`
- **Note:** Cross-validation during optimization uses a relaxed floor of 95% (to accommodate small fold sizes), but the holdout gate requires 100%.
- **Rationale:** CV folds can have as few as 3-5 Cat A scenarios, making 100% per-fold brittle. The holdout set is larger and provides a more reliable safety check.

### Gate 3: No Less-Restrictive Parameters Without Review

The asymmetric ratchet classifies each parameter change as `more_restrictive`, `less_restrictive`, or `unchanged`. If any parameter moves in the less-restrictive direction (lower threshold for boundaries/escalation, higher authority for weights), the ratchet does not advance.

- **Implementation:** `classify_ratchet_direction()` + `_write_review_required()`
- **What happens:** A `gen_XX_review_required.json` artifact is written to the output directory, documenting which parameters moved less-restrictive, the delta values, and requiring human review fields (`reviewer`, `approval_status`, `rationale`).
- **Origin:** Karpathy compromise (2026-02-20) -- instead of blocking optimizer execution entirely, write a reviewable artifact and hold the ratchet. The optimizer continues exploring, but the less-restrictive config is not auto-adopted.
- **EU AI Act Article 14 compliance fields** are included in the review artifact (Schaake consensus).

### Gate 4: Governance Degradation Detector (GDD)

The Governance Stringency Index (GSI) is computed as the mean of key restrictive thresholds (`st_execute`, `st_clarify`, `boundary_violation`, `fidelity_green`). If the candidate's GSI drops more than 15% relative to the baseline, the gate fails.

- **Implementation:** `check_governance_degradation()` with `GDD_MAX_GSI_DROP = 0.15`
- **Rationale:** Even if individual parameters pass the ratchet check, a systemic drift toward less-restrictive governance (many small changes in the same direction) should be caught. GSI provides a single aggregate measure.

### Hard Constraints (within-trial, not gates)

In addition to the four-gate ratchet, individual trials are pruned if they violate:

| Constraint | Threshold | Source |
|------------|-----------|--------|
| Cat A detection (CV aggregate) | >= 95% | `MIN_CAT_A_DETECTION` |
| Cat E detection (adversarial) | >= 85% | `MIN_CAT_E_DETECTION` |
| Weighted Harmonic Safety (WHS) | >= 85% | `MIN_WHS` |

These constraints cause the trial to return `None` (infeasible), not to fail the generation.

---

## 5. Review and Approval Process

### 5.1 Standard Review (All Four Gates Pass)

When an optimizer run completes and all four gates pass for the final generation:

1. **Operator** inspects the generation trajectory (objectives, accuracies, optimism gap, GSI) in `optimizer_summary.json`.
2. **Operator** files a review request by sharing the output directory path with the Configuration Reviewer.
3. **Configuration Reviewer** examines:
   - Per-benchmark holdout accuracy vs. train accuracy (optimism gap should be small and stable)
   - Regression report (zero Cat A regressions, net change positive or zero)
   - Ratchet flags (all `more_restrictive` or `unchanged`)
   - GSI trajectory (stable or increasing)
   - Config hash chain (parent hash matches previous generation)
4. **Configuration Reviewer** either:
   - **Approves:** Config is merged into production defaults. Commit references the generation YAML, config hash, and this charter.
   - **Rejects:** Reviewer documents the reason. Config is archived but not deployed.

### 5.2 Less-Restrictive Review (Gate 3 Flagged)

When Gate 3 flags less-restrictive parameters:

1. All steps from 5.1 apply.
2. **Additionally,** the Configuration Reviewer must:
   - Open `gen_XX_review_required.json`
   - Evaluate whether each less-restrictive change is justified by corresponding accuracy or safety gains
   - Fill in the `reviewer`, `approval_status`, and `rationale` fields in the JSON artifact
   - If approved: the ratchet may be manually advanced for the next run by using the approved config as the starting point
   - If rejected: the less-restrictive config is archived with rejection rationale

### 5.3 Multi-Seed Stability

Before any configuration is deployed to production, a **multi-seed stability analysis** (`--multi-seed N`) should be run with at least 3 seeds. The configuration is considered stable if the coefficient of variation (CV) of the objective across seeds is < 0.05. Unstable results require investigation before deployment.

---

## 6. Escalation Path

### Anomalous Results

If the optimizer produces results that are unexpected or concerning, the following escalation path applies:

| Condition | Action |
|-----------|--------|
| Objective drops significantly between generations | Check for data leakage between train/holdout, verify scenario integrity, re-run with different seed |
| Optimism gap > 0.10 | Possible overfitting to CV folds. Increase `n_folds`, verify holdout stratification, check for scenario duplication |
| All trials pruned in a generation | Search space may be too constrained or hard constraints too tight. Review bounds in Section 3. Do NOT relax hard constraints without charter amendment. |
| Premature convergence (generations terminate before trial 150 of 200) | Escalate `CONVERGENCE_IMPROVEMENT_THRESHOLD` from 0.02 to 0.05 (per Gebru consensus) |
| Cat A regression on holdout despite passing CV | Investigate the specific scenario. If it reveals a boundary corpus gap, add the scenario to the corpus (not to the optimizer). |
| GDD fires repeatedly across generations | The optimizer may be systematically finding less-restrictive optima. This is a signal that the objective function may be underweighting safety. Convene research team review. |

### Circuit Breaker

If any of the following occur, **stop all optimizer runs** and convene the full research team:

1. A deployed configuration (one that passed all review) causes a Cat A failure in production or external benchmarks
2. The optimizer discovers a configuration that achieves > 95% accuracy while reducing all thresholds to their lower bounds (indicates possible benchmark gaming)
3. An external audit identifies a gap in the safety gate system

---

## 7. Audit Trail Requirements

### 7.1 Per-Generation Artifacts

Each optimizer generation produces three mandatory artifacts in `{output_dir}/generations/`:

| Artifact | Format | Contents |
|----------|--------|----------|
| `gen_XX_best.yaml` | YAML | Best config with `config_hash` (SHA-256, first 16 hex chars) and `parent_hash` |
| `gen_XX_trajectory.json` | JSON | Full metrics: objective, train/holdout accuracy, optimism gap, GSI, ratchet flags, regression report, holdout Cat A pass, elapsed time, hash chain |
| `gen_XX_holdout.json` | JSON | Per-benchmark holdout accuracy |

When Gate 3 flags, an additional artifact is produced:

| Artifact | Format | Contents |
|----------|--------|----------|
| `gen_XX_review_required.json` | JSON | Less-restrictive params with deltas, baseline/candidate values, Article 14 review fields |

### 7.2 Run Summary

Each complete optimizer run produces `optimizer_summary.json` containing the full generation trajectory, baseline config, final config, and all hash chain entries.

### 7.3 Hash Chain Integrity

Every generation YAML includes a `config_hash` (SHA-256 of the canonical JSON-serialized `ThresholdConfig`) and a `parent_hash` (hash of the starting config for that generation). This forms an append-only hash chain. Verifying the chain:

```
Gen 0: parent_hash = hash(production_defaults), config_hash = hash(gen_0_best)
Gen 1: parent_hash = config_hash from Gen 0,     config_hash = hash(gen_1_best)
...
```

If a ratchet gate fails, `parent_hash` for the next generation equals the previous generation's `parent_hash` (not its `config_hash`), because the ratchet did not advance.

### 7.4 Append-Only Requirement

Generation artifacts are **append-only**. An optimizer run MUST NOT overwrite or delete artifacts from previous generations. The Optuna study database (`study.db`, SQLite) is also append-only -- studies are created with `load_if_exists=True` and unique study names per generation.

### 7.5 Retention

All optimizer artifacts must be retained for the lifetime of the research program. When configurations are deployed to production, the originating generation YAML and trajectory JSON must be referenced in the deployment commit message.

---

## 8. Change Management

### 8.1 Modifying This Charter

Amendments to this charter require:

1. A written proposal describing the change and its rationale
2. Review by the Configuration Reviewer (if the change affects safety gates or parameter bounds)
3. Research team advisory review (if the change affects the four-gate ratchet or hard constraints)
4. Version increment in the Version History table below
5. Commit with the amendment referencing this charter's document ID (TELOS-RGC-001)

Changes that relax safety gates (lowering `MIN_CAT_A_DETECTION`, widening `GDD_MAX_GSI_DROP`, removing a ratchet gate) require explicit justification grounded in empirical evidence, not convenience.

### 8.2 Modifying Search Bounds

Changes to the search bounds in `suggest_threshold_config()` are governed by this charter. The process:

1. **Widening bounds** (expanding the search space): Requires a written rationale explaining why the current bounds exclude valid configurations. Must be accompanied by a validation run demonstrating that the expanded space does not produce configurations that fail existing benchmarks.
2. **Narrowing bounds** (restricting the search space): Permitted with documentation. Rationale should reference the failure mode being prevented (as with the Nell consensus narrowing of `st_execute` / `st_clarify` upper bounds).
3. **Adding parameters** to the search space: Requires charter amendment (Section 8.1 process). Frozen parameters were frozen for a reason.
4. **Removing parameters** from the search space (freezing them): Permitted with documentation.

### 8.3 Modifying Hard Constraints

Changes to `MIN_CAT_A_DETECTION`, `MIN_CAT_E_DETECTION`, `MIN_WHS`, or `GDD_MAX_GSI_DROP`:

- **Making constraints stricter:** Permitted with documentation.
- **Making constraints less strict:** Requires charter amendment (Section 8.1 process) with empirical justification.

### 8.4 Modifying the Objective Function

Changes to the objective weights (`OBJ_WEIGHT_*`) or the scalarized objective formula require:

1. Written rationale (preferably with research team input, as with the Gebru/Russell rebalancing of accuracy 0.40 -> 0.35, boundary 0.15 -> 0.20)
2. Disclosure that this is a value judgment, not a purely empirical finding
3. A validation run demonstrating the effect on the Pareto frontier

---

## 9. Version History

| Version | Date | Author | Change |
|---------|------|--------|--------|
| 1.0 | 2026-02-20 | TELOS AI Labs (JB) | Initial charter. Establishes four-gate ratchet, parameter bounds, review process, audit trail requirements. Based on 5-agent research team consensus (Russell, Gebru, Karpathy, Schaake, Nell). |

---

*This document is maintained at `research/optimizer_governance_charter.md` in the TELOS Hardened repository. It is referenced by `analysis/governance_optimizer.py` and is subject to the change management process defined in Section 8.*
