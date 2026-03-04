# SetFit OpenClaw Experimental Design — Pre-Registration

**TELOS AI Labs Inc. — Research Program**
**Author:** Nell Watson (Research Methodologist)
**Date:** 2026-02-18
**Status:** Pre-Registration (Experiment Not Yet Executed)
**Depends On:** `research/setfit_mve_phase2_closure.md` (Healthcare SetFit closure — GREEN verdict)
**Depends On:** `validation/openclaw/openclaw_boundary_corpus_v1.jsonl` (100-scenario benchmark)
**Depends On:** `validation/openclaw/openclaw_setfit_training_v1.jsonl` (171-scenario training corpus)

---

### Disclosures

> **Generative AI Disclosure:** Internal analysis, experimental design review, and qualitative assessment in this document were conducted with assistance from LLM-based research agents (Claude, Anthropic). These agents are prompted with domain-specific personas (governance theory, statistics, systems engineering, regulatory analysis, research methodology) and operate as AI research assistants — not independent human expert reviewers. All quantitative results (AUC-ROC, F1, bootstrap confidence intervals, benchmark accuracies) are computed by deterministic code. Qualitative analysis should not be treated as independent peer review. See `research/research_team_spec.md` for full methodology.

> **Conflict of Interest Disclosure:** This research was conducted and funded by TELOS AI Labs Inc., which has a commercial interest in the TELOS governance framework. All domain-specific validation benchmarks (Nearmap, Healthcare, OpenClaw) were created by the research team. External benchmarks (PropensityBench, AgentHarm, AgentDojo) were created by independent organizations. Research artifacts are published on [Zenodo](https://zenodo.org/) with persistent DOIs. No external funding or independent peer review was involved in this work.

---


## Executive Summary

This document pre-registers the experimental methodology for evaluating SetFit as a boundary violation detector for OpenClaw autonomous agent governance. The healthcare SetFit MVE (AUC 0.980) validated contrastive learning for boundary detection. The central question is whether a domain-specific OpenClaw model improves over the cross-domain healthcare model, which scores only 31.0% overall accuracy when applied to OpenClaw's 36-tool, 10-tool-group governance surface.

Three structural differences from healthcare motivate a separate model:
1. **Tool group diversity** — 10 tool groups (vs 7 healthcare configs) with distinct violation surfaces
2. **Attack surface** — autonomous agent exploitation (RCE, supply chain, credential theft) vs healthcare compliance
3. **Vocabulary** — system administration and DevOps terminology vs clinical vocabulary

---

## 1. Hypothesis Formulation

### 1.1 Primary Hypotheses

**H0 (Null):** SetFit fine-tuning on the OpenClaw boundary corpus does not improve boundary violation detection beyond the keyword + cosine baseline.

**H1 (Alternative):** SetFit fine-tuning improves AUC-ROC by at least 0.05 (absolute) over the best existing baseline on held-out data.

### 1.2 Overfitting Counter-Hypothesis

**H_overfit:** Any observed improvement is an artifact of statistical entanglement between training and test data.

Addressed by:
- 5-fold stratified CV (no scenario appears in both train and test within a fold)
- Leave-One-Tool-Group-Out (LOTO) — entire tool groups held out
- Cat E adversarial holdout (15 scenarios never seen during training)

### 1.3 Cross-Tool-Group Transfer Hypothesis

**H_transfer:** SetFit's violation detection transfers to tool groups not seen during training.

Tested via LOTO: train on 9 groups, test on the held-out group. If LOTO AUC < CV AUC - 0.10, the model has memorized group-specific patterns rather than learning generalizable violation semantics.

---

## 2. Data Handling Protocol

### 2.1 Training Corpus

**Source:** `validation/openclaw/openclaw_setfit_training_v1.jsonl` (171 scenarios)

| Category | Count | Label | Description |
|----------|-------|-------|-------------|
| Cat A | 45 | 1 (violation) | Clear boundary violations |
| Cat E | 50 | 1 (violation) | Adversarial/evasion attempts |
| Cat C | 69 | 0 (safe) | Legitimate in-scope requests |
| Cat FP | 7 | 0 (safe) | False-positive controls |

**Positive/Negative ratio:** 95/76 (55.6%/44.4%)
**Tool group coverage:** All 10 groups + cross_group
**Attack family coverage:** 6 families (chain_exploitation, information_disclosure, injection, supply_chain, social_engineering, semantic_cloaking)

### 2.2 Stratification

Stratified by composite key: `(tool_group, boundary_category, label)`. This ensures each fold sees proportional representation of tool groups and violation types.

### 2.3 Adversarial Holdout

15 Cat E scenarios with highest difficulty levels are held out entirely from cross-validation. They serve as a blind adversarial evaluation — the model never sees these during training.

### 2.4 LOTO (Leave-One-Tool-Group-Out)

10 LOTO folds, one per tool group. Each fold trains on scenarios from 9 groups and evaluates on the held-out group. This tests whether violation detection generalizes across tool domains.

---

## 3. Pre-Registered Decision Criteria

| Verdict | AUC-ROC | Detection Rate | FPR | LOTO Gap | Adversarial Detection |
|---------|---------|----------------|-----|----------|----------------------|
| GREEN | >= 0.85 | >= 80% | <= 10% | < 0.05 | >= 80% |
| YELLOW | >= 0.75 | >= 65% | <= 20% | < 0.08 | >= 70% |
| RED | < 0.75 | < 65% | > 20% | > 0.10 | < 70% |

**LOTO Gap** = |CV AUC - mean LOTO AUC|. A large gap indicates the model relies on tool-group-specific patterns rather than generalizable violation semantics.

### 3.1 Verdict Consequences

- **GREEN:** Deploy `setfit_openclaw_v1` as L1.5 in cascade. Export ONNX model.
- **YELLOW:** Deploy with monitoring. Document weak tool groups for targeted augmentation.
- **RED:** Do not deploy. Analyze failure modes. Consider alternative approaches (larger training corpus, different base model, multi-task learning).

---

## 4. Baselines

### 4.1 Keyword Baseline
22 OpenClaw-specific violation keywords (from `openclaw.yaml`). Binary classification: any keyword match → violation.

### 4.2 Frozen-LR Baseline
Logistic regression on frozen MiniLM-L6-v2 embeddings (384-dim). Tests whether contrastive fine-tuning adds value beyond the pre-trained representation.

### 4.3 Cross-Domain Baseline
Healthcare SetFit model (`models/setfit_healthcare_v1/`) applied to OpenClaw data without fine-tuning. Tests domain transfer.

---

## 5. Training Configuration

- **Base model:** `sentence-transformers/paraphrase-MiniLM-L6-v2`
- **Contrastive pairs:** All positive-negative combinations within each fold
- **Training epochs:** 1 (SetFit default for contrastive phase)
- **Classification head:** Logistic regression
- **Random seed:** 42 (reproducibility)
- **Bootstrap CIs:** 10,000 resamples for 95% confidence intervals

---

## 6. Evaluation Metrics

1. **AUC-ROC** — Primary metric, threshold-independent
2. **Detection rate** (recall at optimal threshold) — Violation catch rate
3. **FPR** (false positive rate at optimal threshold) — Legitimate request rejection rate
4. **Per-tool-group AUC** — Identifies weak spots
5. **Per-attack-family detection** — Identifies blind spots
6. **Adversarial holdout detection** — Robustness to unseen evasion tactics
7. **LOTO cross-group transfer** — Generalization evidence

---

## 7. Reproducibility

- All random operations seeded (seed=42)
- Training data versioned (`openclaw_setfit_training_v1.jsonl`)
- Model pinned (`paraphrase-MiniLM-L6-v2`)
- Results exported as JSON with full fold-level detail
- Script: `validation/openclaw/setfit_openclaw_mve.py`

---

## 8. Negative Result Contingency

If RED verdict:
1. Error analysis by tool group and attack family
2. Identify whether failure is data quantity (too few examples) or data quality (ambiguous labels)
3. Consider: larger training corpus (target 300+), different base model (MPNet-768), multi-task learning with healthcare data
4. Document in closure report for future work

---

*Pre-registered before experiment execution. Results will be documented in `research/setfit_openclaw_mve_closure.md`.*
