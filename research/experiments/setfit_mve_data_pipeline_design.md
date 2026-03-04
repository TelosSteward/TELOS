# SetFit MVE — Data Pipeline, Evaluation Metrics, and Statistical Methodology

**TELOS AI Labs Inc. — Research Program**
**Date:** 2026-02-17
**Status:** Design Specification (Pre-Implementation)
**Author:** Gebru (Data Science) — TELOS Research Team
**Depends On:** `research/cross_encoder_nli_mve_phase1.md` (NLI Phase 1 closure, negative result)
**Depends On:** `research/literature_survey_safety_classification.md` (Section 6: Few-Shot Methods)

---

### Disclosures

> **Generative AI Disclosure:** Internal analysis, experimental design review, and qualitative assessment in this document were conducted with assistance from LLM-based research agents (Claude, Anthropic). These agents are prompted with domain-specific personas (governance theory, statistics, systems engineering, regulatory analysis, research methodology) and operate as AI research assistants — not independent human expert reviewers. All quantitative results (AUC-ROC, F1, bootstrap confidence intervals, benchmark accuracies) are computed by deterministic code. Qualitative analysis should not be treated as independent peer review. See `research/research_team_spec.md` for full methodology.

> **Conflict of Interest Disclosure:** This research was conducted and funded by TELOS AI Labs Inc., which has a commercial interest in the TELOS governance framework. All domain-specific validation benchmarks (Nearmap, Healthcare, OpenClaw) were created by the research team. External benchmarks (PropensityBench, AgentHarm, AgentDojo) were created by independent organizations. Research artifacts are published on [Zenodo](https://zenodo.org/) with persistent DOIs. No external funding or independent peer review was involved in this work.

---


## Executive Summary

This document designs the data pipeline, evaluation metrics, and statistical methodology for a SetFit fine-tuning minimum viable experiment (MVE) on the TELOS healthcare governance benchmark. SetFit is a few-shot sentence transformer fine-tuning method that learns contrastive pairs to reshape the embedding space, then trains a lightweight classification head. The Phase 1 NLI experiment established that the keyword baseline (AUC 0.724, FPR 4.3%) outperforms zero-shot NLI (best AUC 0.672, FPR 33.3%), so SetFit must beat both to justify its integration complexity.

The central challenge is statistical: we have only 61 Category A violation scenarios across 7 healthcare configurations (7-10 per config), creating a severe small-sample regime. This document maps out how to construct contrastive training pairs from the existing three-layer boundary corpus (230+ phrasings across the property_intel domain, to be replicated for healthcare), how to evaluate with appropriate metrics given class imbalance, and what cross-validation strategy preserves statistical validity with n=61 positives.

The key recommendation is a unified (pooled across configs) SetFit model evaluated with config-aware stratified 5-fold cross-validation, using precision-recall AUC as the primary metric and FPR at 80% recall as the operational decision metric. Per-config models are statistically nonviable with 7-10 positive samples each.

**What this means in plain English:** We have 280 test scenarios for healthcare AI governance. Only 61 of them are actual violations that must be caught. We want to teach a small AI model to tell violations from legitimate requests by showing it pairs of examples ("this is similar to that" and "this is different from that"). This document is the blueprint for building those pairs, testing the model fairly, and knowing when it is good enough.

---

## 1. Training Data Construction

### 1.1 The Data Landscape

TELOS currently has two data sources relevant to SetFit training:

| Source | Content | Volume | Domain |
|--------|---------|--------|--------|
| Boundary corpus (L1+L2+L3) | Affirmative violation phrasings per boundary | ~61 L1 + ~121 L2 + ~48 L3 = ~230 phrasings | Property Intel only |
| Healthcare benchmark | Scenario request texts with category labels | 280 scenarios (61 A, 94 C, 21 FP, etc.) | Healthcare (7 configs) |

The property_intel boundary corpus (`boundary_corpus_static.py`, `boundary_corpus_llm.py`, `boundary_corpus_regulatory.py`) provides the template for how to construct healthcare-domain violation phrasings. The healthcare YAML configs (`templates/healthcare/*.yaml`) define 7-8 boundaries per config with `safe_exemplars` (8-15 per config) and `violation_keywords`, but no violation corpus equivalent to property_intel's three-layer system.

### 1.2 Training Pair Construction Strategy

SetFit uses contrastive learning on sentence pairs: (anchor, positive) where both belong to the same class, and (anchor, negative) where they belong to different classes. The quality of pairs determines model quality.

**Step 1: Build healthcare violation corpus (prerequisite)**

Before SetFit training, we need violation phrasings for each healthcare config's boundaries. The property_intel corpus shows the pattern: 12-13 hand-crafted phrasings per boundary (L1), 18-24 LLM-generated gap-fillers per boundary (L2), and 6-10 regulatory extractions per boundary (L3).

For the healthcare MVE, we can generate a minimal corpus using the existing benchmark Cat A scenarios as L1 seeds, then LLM-expand. Per config:

| Layer | Method | Expected yield per boundary | Total per config (8 boundaries) |
|-------|--------|----------------------------|--------------------------------|
| L0 | Cat A benchmark scenarios (already exist) | 1-2 per boundary (sparse) | ~9 total (not per-boundary) |
| L1 | Hand-crafted from boundary text + Cat A seeds | 8 phrasings | ~64 |
| L2 | LLM-generated gap-fillers | 12 phrasings | ~96 |
| Total | | ~20 per boundary | ~160 per config |

Across 7 configs: ~1,120 violation phrasings total.

**Step 2: Construct contrastive pairs**

For SetFit, we need same-class and different-class pairs:

- **Positive pairs (violation-violation):** Sample from within the violation corpus. For boundary b_i, pair phrasings from b_i with phrasings from b_j (same config). This teaches the model that all violations are "similar" regardless of which boundary they target.
- **Negative pairs (violation-legitimate):** Pair violation phrasings with `safe_exemplars` from the same config. This is the critical contrastive signal — it teaches the model to distinguish "skip documenting the allergy" from "document the allergy mentioned during the visit."
- **Hard negative pairs (violation-FP control):** Pair violation phrasings with Cat FP benchmark scenarios. These share vocabulary but are legitimate, making them the hardest discrimination task.

**Pair generation budget:**

SetFit's original paper shows strong results with R=20 contrastive pairs per example. With ~160 violation phrasings per config and ~12-15 safe exemplars, the pair space is:

- Same-class violation pairs: C(160, 2) = 12,720 possible (sample R=20 per anchor = 3,200)
- Cross-class pairs per anchor: 160 violations x 15 safe = 2,400 possible (sample R=20 per anchor = 3,200)
- Total training pairs per config: ~6,400
- Total across 7 configs (unified model): ~44,800 pairs

This is more than sufficient. SetFit achieves 92.7% accuracy on IMDB with 8 examples per class and R=20 (Tunstall et al., 2022).

### 1.3 Minimum Viable Training Set Size

The literature establishes SetFit's data efficiency:

| Setting | Examples per class | Performance | Source |
|---------|-------------------|-------------|--------|
| SetFit on RAFT | 8 per class | 0.669 avg (beats GPT-3 0.627) | Tunstall et al. 2022 |
| SetFit + ModernBERT on IMDB | 8 per class | 92.7% accuracy | HuggingFace blog |
| FastFit (comparable) | 5-10 per class | Comparable to SetFit-large | IBM Research, NAACL 2024 |

For the TELOS healthcare task, we have two classes (violation vs. legitimate). Per class:

- **Violations:** 61 benchmark Cat A scenarios + ~1,120 corpus phrasings = ~1,181 unique violation texts
- **Legitimate:** 94 Cat C + 21 FP = 115 benchmark scenarios + ~84 safe_exemplars (12 per config x 7) = ~199 unique legitimate texts

This exceeds SetFit's minimum by an order of magnitude. The bottleneck is not training data volume but training data quality (specifically: how well the generated corpus phrasings represent real-world violation attempts).

**Minimum viable:** 8 violations + 8 legitimate per config = 56 + 56 = 112 labeled examples. We have more than this from the benchmark alone (61 + 115 = 176 labeled scenarios).

**Recommended MVE training set:** Use benchmark scenarios only (no corpus expansion) for the first experiment. This establishes a clean lower bound on SetFit performance using only human-labeled data. If this beats the keyword baseline, corpus expansion is a follow-on improvement, not a prerequisite.

### 1.4 Data Provenance and Documentation

Per Gebru et al. (2021) Datasheets for Datasets methodology, the training data must document:

- **Composition:** Ratio of real benchmark scenarios vs. synthetic corpus phrasings
- **Collection process:** Benchmark scenarios are hand-authored from regulatory sources (see `validation/healthcare/PROVENANCE.md`); corpus phrasings are LLM-generated with deduplication
- **Preprocessing:** Cosine similarity deduplication at threshold 0.92 (same as property_intel L2)
- **Label sources:** Category labels from benchmark schema (human-assigned); corpus phrasings are labeled by construction (all are violations)
- **Known biases:** Violation phrasings may over-represent explicit/direct violations and under-represent indirect/contextual violations (the exact failure mode of the current system on ambient_doc and call_center configs)

---

## 2. Cross-Validation Strategy

### 2.1 The Statistical Constraint

With n=61 Cat A positives and n=115 Cat C+FP negatives (total N=176 for binary classification), standard machine learning cross-validation faces two challenges:

1. **Small positive class:** 61 samples means each fold in 5-fold CV has ~12 positive test samples. Standard error on per-fold accuracy is approximately sqrt(p(1-p)/n) = sqrt(0.5*0.5/12) = 0.14, or 14 percentage points. This is very noisy.

2. **Config structure:** The 7 configs are not exchangeable — call_center (AUC 0.267 with NLI) and therapeutic (AUC 0.972 with NLI) represent fundamentally different boundary violation structures. Pooling configs without stratification risks leaking config-specific signal.

### 2.2 Recommended: Config-Aware Stratified 5-Fold

**Primary strategy:** Stratified 5-fold cross-validation with stratification on (boundary_category, config_id) pairs.

This ensures every fold has approximately:
- 12.2 Cat A violations (61/5)
- 23 Cat C+FP negatives (115/5)
- Proportional representation from each of the 7 configs

**Implementation detail:** Use scikit-learn's `StratifiedGroupKFold` with the group variable set to `config_id` and the stratification variable set to `boundary_category`. However, since Cat A samples per config range from 7 to 10, pure group stratification may put all of one config's violations in a single fold. Instead, use `StratifiedKFold` with a composite stratification key `f"{config_id}_{boundary_category}"` to balance both dimensions.

**What this means:** We split our 176 labeled scenarios into 5 groups. Each group has a fair mix of violations and legitimate requests from all 7 healthcare domains. We train on 4 groups, test on the 5th, and rotate. This prevents the model from being tested on data it has already seen.

### 2.3 Secondary: Leave-One-Config-Out (LOCO)

**Generalization test:** Train on 6 configs, test on the held-out config. This answers: "Does SetFit generalize to a healthcare domain it has never seen?"

| Held-out config | Train Cat A | Test Cat A | Train negatives | Test negatives |
|-----------------|-------------|------------|-----------------|----------------|
| ambient_doc | 52 | 9 | 97 | 18 |
| call_center | 53 | 8 | 100 | 15 |
| coding | 51 | 10 | 98 | 17 |
| diagnostic_ai | 52 | 9 | 99 | 16 |
| patient_facing | 52 | 9 | 99 | 16 |
| predictive | 54 | 7 | 98 | 17 |
| therapeutic | 52 | 9 | 99 | 16 |

**Statistical implication:** With 7-10 positive test samples per LOCO fold, per-fold metrics are extremely noisy. A single misclassified violation swings recall by 11-14%. LOCO is useful for detecting catastrophic generalization failure (e.g., model scores 0% on call_center) but not for precise performance estimation. Report LOCO results with exact counts (e.g., "8/9 detected") rather than percentages.

### 2.4 Why Not Leave-One-Out (LOO)?

Leave-one-out cross-validation would give N=176 folds, each with N-1 training samples. While this maximizes training data per fold, it has two problems for SetFit:

1. **Computational cost:** SetFit fine-tunes the entire sentence transformer backbone for each fold. With 176 folds and ~2 min per fine-tune, LOO takes ~6 hours.
2. **Variance estimation:** LOO estimates are nearly unbiased but have high variance because training sets overlap by all but one sample. The correlation between fold errors inflates confidence intervals.

5-fold CV is the pragmatic choice. Report the mean and standard deviation across folds.

---

## 3. Evaluation Metrics

### 3.1 Primary Metric: Precision-Recall AUC (PR-AUC)

**Why not ROC-AUC?** The class imbalance (61 violations vs. 115 legitimate, ratio 1:1.89) is moderate but meaningful. ROC-AUC counts true negatives, which inflates apparent performance when most samples are negative. A model that flags everything achieves AUC-ROC = 0.50 but PR-AUC near the positive class prevalence (0.35). PR-AUC is strictly more informative for rare-event detection.

However, this imbalance ratio is not extreme (contrast with fraud detection at 1:1000). ROC-AUC is still interpretable here. The Phase 1 NLI experiment used ROC-AUC for comparability with the keyword baseline. **Decision: Report both. Use PR-AUC as primary for model selection; report ROC-AUC for comparability with Phase 1.**

**What this means:** PR-AUC measures how well the model catches violations (precision) without falsely flagging legitimate requests (recall), giving extra weight to the harder task. ROC-AUC treats catching violations and correctly passing legitimate requests as equally important.

### 3.2 Operational Metric: FPR at Fixed Recall

For deployment in the governance pipeline, the operational question is: "At what false positive rate do we catch X% of violations?" This is the receiver operating characteristic evaluated at a fixed true positive rate.

**Target:** FPR at 80% Cat A recall (catch 80% of violations). This matches the Phase 1 GREEN criterion (Cat A detection >= 80%).

| Method | AUC-ROC | Cat A detection | FP FPR | Phase 1 Verdict |
|--------|---------|-----------------|--------|-----------------|
| Keyword baseline | 0.724 | 49.2% | 4.3% | RED |
| NLI base (Framing A) | 0.672 | 63.9% | 33.3% | RED |
| **SetFit target** | **>0.80** | **>=80%** | **<=15%** | **GREEN** |

**Pre-registered decision criteria for SetFit MVE:**

| Outcome | Disposition |
|---------|-------------|
| PR-AUC > 0.85 AND FPR@80%recall <= 10% | GREEN: SetFit approved for pipeline integration as Layer 1.5 |
| PR-AUC 0.75-0.85 AND FPR@80%recall <= 20% | YELLOW: SetFit viable as ensemble feature; test with keyword L0 pre-filter |
| PR-AUC < 0.75 OR FPR@80%recall > 20% | RED: SetFit insufficient alone; investigate FastFit or domain-adaptive pretraining |

### 3.3 Per-Config Diagnostic Metrics

The Phase 1 NLI experiment revealed a bimodal distribution: NLI works for propositionally explicit violations (therapeutic AUC 0.972, diagnostic_ai AUC 0.917) but fails for contextual violations (call_center AUC 0.267, ambient_doc AUC 0.457). SetFit must close this gap.

**Report per config:**
- Cat A detection rate (exact count: X/Y detected)
- FP false positive rate (exact count: X/Y falsely flagged)
- Score gap: mean(violation scores) - mean(legitimate scores). Must be positive and >0.05 for operational utility.
- 95% confidence interval on detection rate using exact binomial (Clopper-Pearson), not normal approximation (inappropriate for n<20)

**Critical configs to watch:**

| Config | Phase 1 NLI AUC | Cat A count | Known failure mode |
|--------|-----------------|-------------|-------------------|
| ambient_doc | 0.457 | 9 | Contextual violations (fabrication, omission) share vocabulary with legitimate documentation |
| call_center | 0.267 | 8 | Procedural violations (clinical advice, order modification) vs. administrative tasks |
| predictive | 0.672 | 7 | Risk model governance violations vs. legitimate risk assessment |

SetFit must achieve detection rate >60% on all three of these configs to be considered viable. If any config scores <50%, the model has not learned the config-specific violation structure.

### 3.4 Calibration Metric

SetFit produces a classification probability (via logistic regression head or differentiable head). For integration into the governance pipeline, the probability must be calibrated — a score of 0.80 should mean 80% chance of being a violation.

**Report:** Expected Calibration Error (ECE) with 10 bins. Target ECE < 0.10. If uncalibrated, apply Platt scaling or isotonic regression as post-hoc calibration. The governance pipeline needs reliable probabilities for the graduated sanction system (EXECUTE/CLARIFY/SUGGEST/INERT/ESCALATE), not just binary decisions.

### 3.5 Baseline Comparison Suite

Seven baselines, ordered by complexity:

| # | Baseline | Description | Purpose |
|---|----------|-------------|---------|
| 1 | Random | P(violation) = 61/176 = 0.347 | Chance-level reference |
| 2 | Keyword-only | 26 global deontic keywords + per-config keywords | Phase 1 winner (AUC 0.724) |
| 3 | Cosine-only (unfrozen) | Current bi-encoder cosine similarity to boundary centroids | Current production system |
| 4 | Keyword + Cosine | Layer 0 keyword boost on Layer 1 cosine scores | Current production system with keyword integration |
| 5 | Logistic regression on frozen embeddings | sklearn LogisticRegression on MiniLM-L6-v2 embeddings | Tests whether classification head alone is sufficient |
| 6 | SetFit (MiniLM-L6-v2 backbone) | Full SetFit fine-tuning | Primary experimental method |
| 7 | SetFit (MPNet backbone) | Full SetFit fine-tuning on 768-dim model | Tests whether larger embedding space helps |

**Why include baseline 5 (logistic regression on frozen embeddings)?** This isolates the contribution of SetFit's contrastive fine-tuning. If logistic regression on frozen embeddings matches SetFit, then fine-tuning the embedding space is unnecessary — we just need a better classification head on top of existing embeddings. This would be cheaper, faster, and preserve full backward compatibility with the existing cosine-based pipeline.

---

## 4. Class Imbalance Handling

### 4.1 Imbalance Assessment

The binary classification task has 61 violations vs. 115 legitimate (ratio 1:1.89). This is mild imbalance by ML standards. For reference:

| Domain | Typical positive:negative ratio | Our ratio |
|--------|-------------------------------|-----------|
| Credit card fraud | 1:500+ | |
| Medical diagnosis | 1:10-100 | |
| Spam detection | 1:5-10 | |
| **TELOS healthcare** | | **1:1.89** |

### 4.2 Does SetFit Handle This Naturally?

SetFit's contrastive learning phase samples pairs, not individual examples. With R=20 pairs per anchor, the number of contrastive pairs is proportional to class size but the sampling process ensures balanced exposure:

- Positive-positive pairs: drawn from violations (smaller class)
- Negative-negative pairs: drawn from legitimate (larger class)
- Cross-class pairs: drawn between classes

The standard SetFit implementation uses balanced sampling for cross-class pairs (equal number of positive and negative anchors per batch). This means the contrastive phase naturally handles 1:1.89 imbalance without oversampling.

**The classification head is the risk point.** After contrastive fine-tuning, SetFit trains a logistic regression (or differentiable head) on the fine-tuned embeddings. This head can be biased by class imbalance. Mitigation: use `class_weight='balanced'` in the logistic regression head, which weights each class inversely proportional to its frequency.

**Recommendation:** No oversampling needed for the contrastive phase. Use class-weighted loss for the classification head. Validate by checking that the decision boundary is not trivially shifted toward flagging everything (FPR > 30% would indicate this).

### 4.3 Augmentation Strategy (If Needed)

If the 1:1.89 ratio causes problems (detected by FPR > 25% at 80% recall), two augmentation strategies are available:

1. **Corpus expansion of violations:** Use the three-layer corpus methodology (L1 hand-crafted + L2 LLM-generated + L3 regulatory) to create more violation phrasings. This increases the violation class size without synthetic duplication.

2. **Hard negative mining:** After initial SetFit training, identify the legitimate scenarios that score highest as violations (closest to the violation cluster in the fine-tuned embedding space). Use these as additional negative training examples in a second training round. This targets the model's specific confusion zone.

---

## 5. Embedding Geometry: SetFit and the Existing Pipeline

### 5.1 How SetFit Changes the Embedding Space

SetFit fine-tunes the sentence transformer backbone using contrastive loss. This reshapes the embedding geometry:

- **Before fine-tuning:** MiniLM-L6-v2 produces 384-dimensional embeddings where cosine similarity reflects general semantic similarity. Violations and legitimate requests that share vocabulary (e.g., "skip documenting allergies" vs. "document allergies") occupy nearby regions because the model has not been trained to distinguish compliance from violation.

- **After fine-tuning:** The contrastive loss pulls violation embeddings closer together and pushes them away from legitimate embeddings. The embedding space now encodes a compliance/violation axis in addition to general semantics.

This is precisely the transformation needed. The current cosine-based system fails on ambient_doc (AUC 0.457) and call_center (AUC 0.267) because violations and legitimate requests are not linearly separable in the original embedding space. SetFit's contrastive fine-tuning creates that separability.

### 5.2 Impact on Existing Cosine-Based Pipeline

**Critical question:** Does fine-tuning the embedding model break the existing governance pipeline?

The current `agentic_fidelity.py` uses the same embedding function for all five governance dimensions:
1. Purpose fidelity: cos(action, purpose_embedding)
2. Scope fidelity: cos(action, scope_embedding)
3. Boundary detection: cos(action, boundary_centroid)
4. Tool fidelity: cos(action, tool_description)
5. Chain continuity: cos(action_n, action_{n-1})

If we fine-tune the embedding model with SetFit, the geometry changes affect ALL dimensions, not just boundary detection. Purpose and scope embeddings were calibrated against the original MiniLM space — the thresholds in `telos_core/constants.py` (EXECUTE >= 0.85, CLARIFY >= 0.70, etc.) assume the original embedding distribution.

**Three architectural options:**

| Option | Description | Pros | Cons |
|--------|-------------|------|------|
| A. Dual-model | Keep original MiniLM for purpose/scope/tool/chain; use SetFit-finetuned model for boundary only | Zero regression risk; isolates SetFit impact | 2x embedding computation; 2x memory |
| B. Unified fine-tuned | Use SetFit model for all dimensions; recalibrate thresholds | Single model; unified embedding space | Requires full threshold recalibration; regression risk on purpose/scope |
| C. SetFit as classification head only | Freeze backbone; train only the classification head on MiniLM embeddings | Zero embedding change; backward compatible | Loses the main benefit of SetFit (contrastive fine-tuning) |

**Recommendation: Option A (dual-model) for the MVE.** The MVE must isolate SetFit's contribution to boundary detection without confounding it with threshold recalibration across all other dimensions. At ~11MB per MiniLM model, the memory cost is negligible. Latency adds one embedding pass (~5-10ms on CPU with ONNX).

If Option A SetFit achieves GREEN metrics, evaluate Option B as a follow-on experiment with a full recalibration sweep. Option C should be tested as baseline 5 (logistic regression on frozen embeddings) — if it matches Option A, we never needed contrastive fine-tuning.

### 5.3 Bi-Encoder Architecture Preservation

SetFit fine-tunes a bi-encoder (sentence transformer). Unlike cross-encoders (which require O(N) forward passes for N boundaries), bi-encoders precompute boundary representations and do O(1) cosine similarity at inference. This preserves the sub-50ms latency requirement that eliminated cross-encoder NLI in Phase 1.

**Inference flow with SetFit (Option A):**

```
User request
    |
    v
[MiniLM-L6-v2 original] --> purpose, scope, tool, chain dimensions
    |
    v
[MiniLM-L6-v2 SetFit-finetuned] --> boundary classification score
    |
    v
Composite fidelity = 0.35*purpose + 0.20*scope + 0.20*tool + 0.15*chain - 0.10*boundary_penalty
```

The boundary_penalty term in the composite formula currently uses the cosine-based violation score (0-1). SetFit produces a classification probability (0-1). These are numerically compatible — the SetFit probability can replace the cosine violation score in the composite formula without changing the formula structure.

### 5.4 Centroid Compatibility

The existing boundary detection uses a three-layer corpus centroid (mean of violation phrasing embeddings) with sub-centroid clustering. SetFit replaces this entire mechanism:

| Component | Current (cosine) | SetFit |
|-----------|-------------------|--------|
| Boundary representation | Centroid + sub-centroids | Learned embedding space + classification head |
| Detection method | cos(action, centroid) >= 0.70 | P(violation) >= threshold |
| Contrastive detection | cos(action, violation_centroid) - cos(action, safe_centroid) | Implicit in fine-tuned space |
| Keyword boost | +0.15 to cosine score when keywords match | Can be retained as feature or made unnecessary |

**Key insight:** SetFit subsumes the contrastive detection mechanism that was added in CLI Milestone 1 (`contrastive_margin = violation - safe`). The contrastive fine-tuning encodes this margin directly into the embedding space — violations are far from legitimate requests by construction, not by post-hoc margin computation. If SetFit works, the manual safe_centroid / contrastive_suppression logic becomes unnecessary.

---

## 6. Per-Config vs. Unified Model: Power Analysis

### 6.1 Per-Config Fine-Tuning: Statistical Viability

With 7-10 Cat A samples per config, per-config SetFit is operating at the absolute floor of few-shot viability:

| Config | Cat A | Cat C+FP | Total | Viable? |
|--------|-------|----------|-------|---------|
| ambient_doc | 9 | 18 | 27 | Marginal |
| call_center | 8 | 15 | 23 | Marginal |
| coding | 10 | 17 | 27 | Marginal |
| diagnostic_ai | 9 | 16 | 25 | Marginal |
| patient_facing | 9 | 16 | 25 | Marginal |
| predictive | 7 | 17 | 24 | Below threshold |
| therapeutic | 9 | 16 | 25 | Marginal |

SetFit claims 8 examples per class as the minimum. Most configs meet this for the violation class (8-10 >= 8), but the margin is zero. There is no room for cross-validation — with 5-fold CV on 9 violations, each test fold has 1-2 positive samples. You cannot compute meaningful per-fold AUC with 1-2 positives.

**Power analysis for per-config evaluation:**

To detect a difference of delta=0.15 in AUC (e.g., 0.72 to 0.87) with alpha=0.05 and power=0.80, the required sample size per group is approximately:

n = (Z_alpha + Z_beta)^2 * (p0(1-p0) + p1(1-p1)) / (p1 - p0)^2

For AUC comparison using the DeLong test, with n_pos=9 and n_neg=16, the minimum detectable AUC difference at 80% power is approximately 0.25-0.30. This means per-config evaluation can only detect large improvements (going from AUC 0.50 to AUC 0.80), not moderate improvements (AUC 0.72 to 0.85).

**Verdict: Per-config fine-tuning is statistically nonviable for the MVE.** The sample sizes are too small for reliable training or evaluation. This is not a SetFit limitation — it is a fundamental statistical constraint. No few-shot method can reliably learn from 7 positive examples and also validate on held-out data from those same 7 examples.

### 6.2 Unified Model: Statistical Properties

Pooling across configs gives n=61 positives and n=115 negatives (N=176). This supports:

- **5-fold CV:** ~12 positives and ~23 negatives per test fold. Per-fold AUC has standard error ~0.06-0.08. Mean across 5 folds converges to a usable estimate.
- **DeLong test for AUC comparison:** With n=61 and n=115, the minimum detectable AUC difference at 80% power is approximately 0.10-0.12. This is sufficient to detect meaningful improvements over the keyword baseline (AUC 0.724).
- **Per-config reporting within unified model:** After training on all configs, report per-config metrics on the held-out test fold. These are descriptive (not inferential) but reveal config-specific failure modes.

### 6.3 Hybrid Approach: Config-Conditioned Unified Model

A middle ground: train a single unified SetFit model but prepend config context to each input.

**Input format:** `"[healthcare_ambient_doc] Skip documenting the penicillin allergy the patient mentioned"`

This allows the model to learn config-specific violation patterns while pooling all data for training. The config token provides a learnable context signal without fragmenting the training data.

**Recommendation:** Test both vanilla unified (no config prefix) and config-conditioned unified. If config-conditioned improves performance on the three weak configs (ambient_doc, call_center, predictive) without hurting the strong configs (therapeutic, diagnostic_ai), adopt it.

---

## 7. Detailed Baseline Comparison Design

### 7.1 Baseline 1: Random Classifier

- **Metric:** AUC-ROC = 0.50, PR-AUC = 0.347 (positive class prevalence)
- **Purpose:** Sanity check. Any method must beat this.

### 7.2 Baseline 2: Keyword-Only

- **Method:** Match against `GLOBAL_DEONTIC_KEYWORDS` (17 terms from `agentic_fidelity.py`) + per-config `violation_keywords` (2-4 terms from YAML configs). Binary output: 1 if any keyword matches, 0 otherwise.
- **Phase 1 result:** AUC-ROC 0.724, FPR 4.3% (on the full 280-scenario set with A vs. C+FP binary classification)
- **Reproduce:** Run on 176-scenario A vs. C+FP subset for apples-to-apples comparison

### 7.3 Baseline 3: Cosine-Only (Current Production)

- **Method:** Embed request with MiniLM-L6-v2, compute max cosine similarity to all boundary centroids (using existing three-layer centroid + sub-centroid mechanism from `agentic_pa.py`). Score = max normalized violation similarity.
- **Threshold:** Current BOUNDARY_VIOLATION_THRESHOLD = 0.70
- **Purpose:** Measures the existing production system's discrimination ability

### 7.4 Baseline 4: Keyword + Cosine (Current Production with L0)

- **Method:** Baseline 3 + keyword boost (KEYWORD_BOOST = 0.15 when keywords match AND cosine >= KEYWORD_EMBEDDING_FLOOR = 0.40)
- **Purpose:** Measures the full current production pipeline. SetFit must beat this to justify integration.

### 7.5 Baseline 5: Logistic Regression on Frozen Embeddings

- **Method:** Embed all 176 scenarios with MiniLM-L6-v2. Train sklearn `LogisticRegression(class_weight='balanced')` on 384-dim embeddings. Evaluate with same 5-fold CV.
- **Purpose:** Isolates the contribution of SetFit's contrastive fine-tuning. If LogReg matches SetFit, fine-tuning is unnecessary.
- **Feature variants:**
  - (a) Raw 384-dim embedding only
  - (b) 384-dim embedding + keyword binary feature
  - (c) 384-dim embedding + max cosine similarity to each boundary centroid (7-8 additional features per config)

### 7.6 Baseline 6: SetFit (MiniLM-L6-v2 backbone)

- **Method:** Full SetFit pipeline — contrastive fine-tuning of `sentence-transformers/all-MiniLM-L6-v2` + logistic regression head
- **Hyperparameters for MVE:**
  - Contrastive pairs per example: R=20
  - Training epochs: 1 (SetFit default for few-shot)
  - Body learning rate: 2e-5
  - Batch size: 16
  - Max training pairs: 44,800 (all pairs generated in Section 1.2)
- **Variants:**
  - (a) Benchmark-only training data (61 + 115 = 176 examples)
  - (b) Benchmark + L1 corpus expansion (~176 + ~1,120 violation phrasings)
  - (c) Config-conditioned (prefix with config token)

### 7.7 Baseline 7: SetFit (MPNet backbone)

- **Method:** Same as Baseline 6 but with `sentence-transformers/all-mpnet-base-v2` (768-dim)
- **Purpose:** Tests whether the larger embedding space (768 vs. 384) helps boundary discrimination. The Phase 1 NLI experiment found MPNet improves boundary detection by +4.9% Cat A but drops overall accuracy (49.3% vs. 72.5% uncalibrated). SetFit may resolve this by fine-tuning the space directly.

---

## 8. Experimental Protocol

### 8.1 Pre-Registration

Before running any experiment, file the following in `research/setfit_mve_preregistration.md`:

1. Decision criteria (Section 3.2 above)
2. Primary metric (PR-AUC)
3. Cross-validation strategy (config-aware stratified 5-fold)
4. Baseline list (7 baselines)
5. Stop criteria: If any baseline achieves GREEN independently, SetFit integration should be reconsidered for cost-benefit

### 8.2 Execution Order

1. **Phase A: Baselines (no SetFit).** Run baselines 1-5 on the 176-scenario A vs. C+FP dataset with 5-fold CV. This takes <5 minutes total and establishes the performance floor.

2. **Phase B: SetFit MVE.** Run baselines 6a, 6b, 6c, 7 with the same 5-fold splits (critical: use identical fold assignments for fair comparison). Each SetFit run takes ~2-5 minutes on CPU.

3. **Phase C: Per-config diagnostic.** Using the best SetFit variant from Phase B, report per-config detection rates on the held-out folds. Run LOCO evaluation for generalization assessment.

4. **Phase D: Calibration.** Evaluate ECE. Apply Platt scaling if ECE > 0.10. Re-evaluate all metrics on calibrated scores.

5. **Phase E: Integration feasibility.** If SetFit achieves YELLOW or GREEN, test Option A (dual-model) integration into `agentic_fidelity.py` by replacing the `_check_boundaries()` method's cosine scoring with SetFit classification probability. Verify that the composite fidelity formula still produces sensible decisions.

### 8.3 Reproducibility Requirements

- **Random seeds:** Fix numpy seed=42, torch seed=42, transformers seed=42 for all experiments
- **Fold assignments:** Persist fold indices to `validation/healthcare/setfit_mve_folds.json` so future experiments use identical splits
- **Model versioning:** Pin `sentence-transformers==3.3.1` (or latest stable), `setfit==1.1.0` (or latest), `scikit-learn==1.5.2`
- **Hardware:** Report CPU model and RAM. SetFit MVE should run on any machine with 8GB RAM. No GPU required.

---

## 9. Statistical Tests

### 9.1 AUC Comparison

Use the DeLong test (DeLong et al., 1988) to compare AUC between methods. This is the standard test for comparing two ROC curves on the same dataset. Available in `scipy` via the `roc_auc_score` function and custom implementation of the DeLong variance estimator.

**Multiple comparisons correction:** With 7 baselines, 7 pairwise comparisons against the best SetFit variant. Apply Holm-Bonferroni correction (less conservative than Bonferroni, more powerful). Report both raw p-values and corrected p-values.

### 9.2 Bootstrap Confidence Intervals

For PR-AUC (where DeLong does not apply), use 2,000 bootstrap resamples to estimate 95% confidence intervals. Resample within each fold's test set to preserve the stratified structure.

### 9.3 McNemar's Test for Per-Scenario Agreement

To test whether two methods make different errors (not just different overall accuracy), apply McNemar's test on the binary decision matrix. This detects cases where SetFit catches violations that keywords miss, or vice versa — relevant for ensemble design.

### 9.4 Per-Config Binomial Tests

For per-config detection rates (e.g., 8/9 on ambient_doc), use exact binomial confidence intervals (Clopper-Pearson). Do not use normal approximation — it is invalid for n < 20. Report the 95% CI alongside the point estimate.

Example: If SetFit detects 8/9 Cat A violations on ambient_doc:
- Point estimate: 88.9%
- 95% Clopper-Pearson CI: [51.8%, 99.7%]
- **The CI is very wide.** This is honest. With 9 samples, we cannot be precise.

---

## 10. Risk Register

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| SetFit contrastive fine-tuning damages purpose/scope embeddings | Medium | High | Option A: dual-model architecture; original model untouched |
| Healthcare violation corpus (L1+L2) is not yet built | High | Blocks Phase B experiment 6b | Phase B experiment 6a uses benchmark-only data; corpus is not a prerequisite for MVE |
| 61 positive samples too few for stable SetFit training | Low | Medium | SetFit designed for 8+ examples per class; we have 61. Risk is overfitting, not underfitting |
| Config-specific failure on ambient_doc/call_center persists | Medium | High | Config-conditioned input variant (6c); if still fails, per-config corpus expansion is the next step |
| Keyword baseline already achieves GREEN with FPR optimization | Low | Medium | Good outcome — means the simple approach works. Re-scope SetFit to handle the 50.8% of Cat A that keywords miss |
| SetFit + ONNX export path is broken or unsupported | Medium | Medium | Verify ONNX export before investing in training; fallback to PyTorch inference with model loading |
| Fine-tuned model overfits to benchmark phrasing style | Medium | High | LOCO evaluation detects this; corpus expansion provides phrasing diversity |

---

## 11. Decision Framework

After Phase B completes, apply this decision tree:

```
Is PR-AUC(SetFit) > PR-AUC(keyword+cosine) + 0.05?
├── YES: SetFit adds value over current production
│   ├── Is FPR@80%recall <= 15%?
│   │   ├── YES: GREEN — integrate as Layer 1.5
│   │   └── NO: YELLOW — test keyword L0 pre-filter + SetFit L1.5 cascade
│   └── Per-config: does SetFit fix ambient_doc AND call_center?
│       ├── YES: Unified model sufficient
│       └── NO: Investigate config-conditioned or per-config corpus expansion
├── NO: SetFit does not beat current production
│   ├── Is LogReg(frozen) close to SetFit? (within 0.02 PR-AUC)
│   │   ├── YES: Fine-tuning unnecessary; use LogReg head on frozen embeddings
│   │   └── NO: Fine-tuning helps but not enough; investigate FastFit or domain pretraining
│   └── Does SetFit fix specific configs even if aggregate is flat?
│       ├── YES: Config-stratified pipeline (SetFit for weak configs, cosine for strong)
│       └── NO: SetFit eliminated; proceed to domain-adaptive pretraining path
└── Is LogReg(frozen) > keyword+cosine? (Baseline 5 > Baseline 4)
    ├── YES: Classification head on frozen embeddings is the cheapest win
    └── NO: Embedding space lacks discriminative information; fine-tuning is necessary
```

---

## 12. Connections to Phase 1 Findings

This design directly addresses every gap identified in the Phase 1 NLI closure document:

| Phase 1 Finding | SetFit MVE Response |
|----------------|---------------------|
| NLI fails on deontic reasoning (Russell) | SetFit bypasses deontic logic by fine-tuning on violation examples directly, not on NLI logic |
| Keyword baseline AUC 0.724, FPR 4.3% (Gebru) | Keyword retained as baseline 2 and as L0 pre-filter; SetFit must beat 0.724 AUC |
| Cross-encoders too slow: O(N) per request (Karpathy) | SetFit is a bi-encoder: O(1) inference after precomputation. ~10ms per request. |
| 33.3% FPR disqualifying (Schaake) | Pre-registered FPR threshold: <=15% at 80% recall for GREEN |
| Bimodal per-config AUC: therapeutic 0.972 vs call_center 0.267 | Per-config diagnostics; config-conditioned variant; LOCO evaluation |
| NLI residual value for therapeutic/diagnostic_ai | If SetFit achieves GREEN, NLI L2 becomes unnecessary; if YELLOW, NLI scores as ensemble feature for those 2 configs |

---

## 13. Timeline and Resource Estimate

| Phase | Duration | Compute | Dependency |
|-------|----------|---------|------------|
| A: Baselines 1-5 | 30 minutes | CPU only | Healthcare benchmark data (exists) |
| B: SetFit MVE (6a, 6c, 7) | 2 hours | CPU only (GPU optional) | setfit library installation |
| B: SetFit with corpus (6b) | 1 day | CPU + LLM for corpus generation | Healthcare violation corpus (must be built) |
| C: Per-config diagnostics | 30 minutes | CPU only | Phase B best model |
| D: Calibration | 30 minutes | CPU only | Phase C results |
| E: Integration test | 2 hours | CPU only | Phase D calibrated model |
| **Total (without corpus)** | **~4 hours** | | |
| **Total (with corpus)** | **~1.5 days** | | |

---

## 14. Action Items

1. **Install SetFit library** and verify ONNX export compatibility with `sentence-transformers/all-MiniLM-L6-v2` backbone.

2. **Create fold assignment file** (`validation/healthcare/setfit_mve_folds.json`) using config-aware stratified 5-fold split with seed=42.

3. **Run Phase A baselines** (1-5) on the 176-scenario A vs. C+FP dataset. Record PR-AUC, ROC-AUC, FPR@80%recall for each.

4. **Run Phase B SetFit experiments** (6a, 6c, 7) using benchmark-only training data. Record all metrics with 5-fold CV means and standard deviations.

5. **Apply decision framework** (Section 11) to Phase B results. If GREEN or YELLOW, proceed to Phase C-E. If RED, document negative result and investigate FastFit or domain-adaptive pretraining.

6. **If Phase B achieves YELLOW or better:** Build healthcare violation corpus (L1+L2 for all 7 configs) and run experiment 6b to test corpus expansion benefit.

7. **If Phase E integration test passes:** File design document for dual-model boundary detection in `agentic_fidelity.py` and schedule implementation.

8. **Pre-register** all decision criteria in `research/setfit_mve_preregistration.md` before running any experiments.

---

*Filed as research artifact per TELOS research program documentation requirements. Subject to 5-agent research team review before implementation begins.*

*Gebru — 2026-02-17*
