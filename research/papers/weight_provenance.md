# Composite Weight Provenance

**Version:** 1.0
**Date:** 2026-02-26
**Author:** TELOS AI Labs Inc.
**Purpose:** Documented derivation, rationale, sensitivity, and change control for composite scoring weights. Required by A8 review (Russell, Schaake) and EU AI Act Article 9 (risk management system requires provenance for all scoring parameters).

---

## 1. Current Production Weights

| Dimension | Weight | Sign | Source |
|-----------|--------|------|--------|
| Purpose Fidelity | 0.35 | + | `agentic_fidelity.py:111` |
| Scope Fidelity | 0.20 | + | `agentic_fidelity.py:112` |
| Tool Fidelity | 0.20 | + | `agentic_fidelity.py:113` |
| Chain Continuity (SCI) | 0.15 | + | `agentic_fidelity.py:114` |
| Boundary Penalty | 0.10 | - | `agentic_fidelity.py:115` |

**Positive weights sum:** 0.90 (not 1.0 — intentional)
**Composite formula:** `0.35p + 0.20s + 0.20t + 0.15c - 0.10b`, clamped [0, 1]

---

## 2. Origin

**Derivation method:** Principled design choice, not empirical tuning.

The weights were established during initial engine design (Sessions 1–2, February 2026) by Jeffrey Brunner. They were not derived from training data, grid search, or statistical optimization. The rationale was theoretical — grounded in the TELOS governance model's first principles.

**Why not empirically tuned at inception:** At the time of weight selection, no benchmark corpus existed to tune against. The Nearmap (173 scenarios), Healthcare (280 scenarios), and OpenClaw (100 scenarios) benchmarks were created later. The weights were designed to encode the governance model's theoretical priorities, then validated (not tuned) against benchmarks as they became available.

**Confirmed unchanged:** The weights have remained at these exact values through all 8 calibration phases documented in `research/openclaw_calibration_log.md`. The calibration log section "What Was NOT Changed (Team Consensus)" explicitly lists: "Composite weights: 0.35/0.20/0.20/0.15/-0.10" as a frozen parameter.

---

## 3. Rationale

### 3.1 Why Purpose Is Weighted Highest (0.35)

Purpose fidelity is the primacy attractor's core measurement. The PA defines the agent's constitutional authority — what the agent exists to do. If an action doesn't align with the PA's stated purpose, no amount of scope alignment, tool match, or chain continuity should compensate.

**Theoretical grounding:**
- **TELOS whitepaper (Brunner, 2025):** The Primacy Attractor is the embedding-space representation of user purpose. All governance flows from this anchor. Purpose is not one dimension among equals — it is the dimension from which the others derive meaning.
- **SAAI Framework §G1.9 (Watson et al., 2026):** "The system shall maintain alignment with the operator's stated intent." Purpose fidelity is the direct measurement of this requirement.
- **Ostrom's Design Principle 1 (1990):** Clearly defined boundaries. Purpose defines what is "inside" the governance commons.

**Why 0.35 specifically (not 0.40 or 0.50):** At 0.35, a maximum-purpose action (purpose=1.0) contributes 0.35 to composite, which alone is below the ST EXECUTE threshold (0.45). This means purpose alone is never sufficient for EXECUTE — at least one other dimension must also be strong. This prevents "purpose-only" governance where an action that aligns with purpose but has wrong tool, broken chain, or boundary violation could still pass.

### 3.2 Why Scope and Tool Share Weight (0.20 each)

Scope and tool fidelity serve different but equally important roles:
- **Scope:** Is this action in the agent's authorized domain? (horizontal constraint)
- **Tool:** Does this action match a registered tool? (vertical constraint)

Equal weighting reflects that neither is more important than the other. A well-scoped action with the wrong tool is as problematic as a wrong-scope action with the right tool.

**Why 0.20 (not 0.25):** At 0.20, these dimensions are meaningful contributors without dominating the composite. Combined (0.40), they match purpose + chain (0.50), ensuring no single "axis" of governance can override the others.

### 3.3 Why Chain Is Lower Weight (0.15)

Chain continuity (SCI) measures trajectory, not position. It answers "is this sequence drifting?" rather than "is this step aligned?" Lower weight reflects two structural limitations:

1. **First-step penalty:** The first action in any chain has SCI = 0.0 (no previous step to compare). At weight 0.15, this costs 0.15 * 0.0 = 0.00 — a structural penalty that could prevent EXECUTE on otherwise-aligned first actions if weighted higher.
2. **Redundancy with purpose:** A truly purpose-aligned action will score well on both purpose and chain (since purpose-aligned actions tend to follow other purpose-aligned actions). Chain adds signal primarily when purpose checks miss slow drift across a long sequence.

### 3.4 Why Boundary Is Subtractive (-0.10)

The boundary dimension is fundamentally different from the other four. Purpose, scope, tool, and chain measure alignment (higher = better). Boundary measures violation (higher = worse). The negative sign makes this explicit in the formula.

**Why -0.10 (not -0.20 or -0.30):**
- The boundary dimension's primary enforcement is through the **hard override** in `_make_decision()`: `if boundary_triggered: return ESCALATE`. The composite penalty is a secondary signal — it degrades the composite score for actions that are near boundaries but don't trigger the hard threshold (0.70).
- At -0.10, a maximum boundary violation (score=1.0) subtracts 0.10 from composite. This can shift a borderline EXECUTE (0.48) to CLARIFY (0.38), but cannot by itself force ESCALATE. The hard override handles escalation.
- Stronger penalty would degrade legitimate actions that have incidental boundary similarity (e.g., a healthcare agent asking about "patient records" near a "do not access financial records" boundary).

### 3.5 Why Positive Weights Sum to 0.90

The 0.10 gap between the positive weight sum (0.90) and 1.0 is not rounding error — it is architectural:

- **Theoretical maximum composite: 0.90** (when all positive dimensions = 1.0 and boundary = 0.0)
- This implements the SAAI Framework §G1.9 principle: governance always retains intervention capacity. No action achieves "perfect" fidelity.
- The 0.10 "reserved space" is occupied by the boundary penalty dimension, which can only reduce composite, never increase it.
- **OWASP LLM Top 10 (LLM08):** No agent achieves unbounded authority. The composite ceiling is a permanent architectural governor.

---

## 4. Sensitivity Analysis

### 4.1 Analytical Sensitivity

Weight perturbation of ±0.05 on each dimension, holding others constant:

| Perturbation | Effect on Composite | Verdict Impact (ST thresholds) |
|-------------|--------------------|-----------------------------|
| purpose ±0.05 | ±0.05 * score (max ±0.05) | Can shift ~1 verdict band for borderline scores |
| scope ±0.05 | ±0.05 * score (max ±0.05) | Smaller absolute effect than purpose (typical scope scores lower) |
| tool ±0.05 | ±0.05 * score (max ±0.05) | Similar to scope |
| chain ±0.05 | ±0.05 * score (max ±0.05) | Least impact in practice (SCI often 0.0 on first step) |
| boundary ±0.05 | ∓0.05 * score (max ±0.05) | Affects near-boundary actions; hard override still dominates |

**Key finding from Phase 8 calibration:** Purpose weight limits engine-level impact. Purpose is 35% of composite. A +0.08 purpose lift = +0.028 composite. Most Cat C scenarios have composites at 0.22–0.44 (need 0.45 for EXECUTE). A +0.028 boost crosses the threshold for only ~3% of scenarios. This demonstrates that weight redistribution alone cannot solve the Cat C accuracy problem — the issue is structural (thin semantic space for 36 tools).

### 4.2 Optimizer Search Space

The governance optimizer (`analysis/governance_optimizer.py`) can explore weight configurations via multi-seed Optuna TPE optimization:

- **Purpose:** [0.10, 0.60] raw, softmax-normalized
- **Scope:** [0.05, 0.40] raw, softmax-normalized
- **Tool:** [0.05, 0.40] raw, softmax-normalized
- **Chain:** [0.05, 0.30] raw, softmax-normalized
- **Boundary penalty:** [0.05, 0.25] raw, softmax-normalized

Softmax normalization ensures weights always sum to 1.0 after optimization. The optimizer has not yet been run with Cat C accuracy as an objective term — this is a recommended next step from the Phase 8 calibration log.

### 4.3 What Has Been Validated

The current weights have been validated (not tuned) against:
- **Nearmap benchmark:** 173 scenarios, 82.6% overall accuracy (post-calibration)
- **Healthcare benchmark:** 280 scenarios, 7 configurations, 12 attack families
- **OpenClaw benchmark:** 100 scenarios, 54% overall, 100% Cat A, 87.5% Cat E (post Phase 8)
- **SetFit MVE:** AUC 0.9905 ±0.0152, 96.2% detection, 1.3% FPR

The weights were confirmed unchanged through all 8 calibration phases by team consensus. The calibration improved scoring through other mechanisms (scope centroid, SetFit L1.5, max-pool rescue, hook enrichment) while preserving the original weight structure.

### 4.4 What Has NOT Been Done

- **No ablation study:** No experiment has systematically removed one dimension at a time to measure its marginal contribution. Deferred until production telemetry provides ground truth.
- **No empirical weight optimization:** The optimizer exists but has not been run for weight tuning. Current weights are theoretical, not data-derived.
- **No per-domain weight adaptation:** All benchmarks use the same weights. A file-read operation scoring 0.42 composite is treated identically to a messaging operation scoring 0.42, despite different risk profiles.
- **No sensitivity to perturbation beyond ±0.05:** No formal analysis of larger perturbations or non-linear interactions between weights.

---

## 5. Change Control

### 5.1 Who Can Modify Weights

Jeffrey Brunner (principal authority, TELOS AI Labs Inc.) must approve all weight changes. Weights are governance parameters, not tuning parameters — they define the constitutional structure of how dimensions relate to each other.

### 5.2 Required Process

Weight changes require ALL of the following:

1. **Optimizer run** with the four-gate ratchet:
   - Cat A regression gate: Cat A detection must not decrease
   - Holdout 100% gate: must pass on held-out evaluation set
   - Less-restrictive block gate: changes that make governance less restrictive flagged for review
   - GDD gate: Governance Stability Index must not drop >15%
2. **Cross-benchmark validation:** New weights must pass Nearmap, Healthcare, AND OpenClaw benchmarks without regression
3. **Research team review:** Russell (governance theory), Gebru (statistics), Karpathy (systems) — consensus required
4. **Jeffrey's explicit approval**

### 5.3 Version Control

Weights are module-level constants in `telos_governance/agentic_fidelity.py` (lines 111–115). Any change is visible in `git diff`. The `ThresholdConfig` dataclass (`threshold_config.py`) provides runtime parameterization for optimizer trials but production always uses the module-level defaults.

### 5.4 The Ratchet / PA Framework

Weight changes are subject to the same asymmetric ratchet that governs all TELOS configuration changes:
- **More restrictive** (tighter governance): requires optimizer validation + team review
- **Less restrictive** (looser governance): requires optimizer validation + team review + explicit Jeffrey approval + regression analysis documenting what was previously caught and would now be missed

This asymmetry is intentional. Making governance stricter is a lower-risk operation than making it permissive. Per SAAI Framework: "flexibility scales inversely with the magnitude of the drift."

---

## 6. Approval Record

| Date | Action | Approved By |
|------|--------|-------------|
| 2026-02 (Sessions 1–2) | Initial weight selection: 0.35/0.20/0.20/0.15/-0.10 | Jeffrey Brunner |
| 2026-02-18 (Phase 7) | Weights confirmed unchanged through 7 calibration phases | Team consensus (Russell, Gebru, Karpathy, Nell, Schaake) |
| 2026-02-26 (Phase 8) | Weights confirmed unchanged through max-pool calibration | Team consensus |

---

*Weight provenance document v1.0 — EU AI Act Article 9 compliance.*
