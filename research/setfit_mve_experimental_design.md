# SetFit MVE Experimental Design — Methodology, Controls, and Falsifiability

**TELOS AI Labs Inc. — Research Program**
**Author:** Nell Watson (Research Methodologist)
**Date:** 2026-02-17
**Status:** Pre-Registration (Experiment Not Yet Executed)
**Depends On:** `research/cross_encoder_nli_mve_phase1.md` (Phase 1 NLI closure)
**Depends On:** `research/literature_survey_safety_classification.md` (40-paper survey)
**Depends On:** `research/boundary_corpus_methodology.md` (three-layer corpus design)

---

### Disclosures

> **Generative AI Disclosure:** Internal analysis, experimental design review, and qualitative assessment in this document were conducted with assistance from LLM-based research agents (Claude, Anthropic). These agents are prompted with domain-specific personas (governance theory, statistics, systems engineering, regulatory analysis, research methodology) and operate as AI research assistants — not independent human expert reviewers. All quantitative results (AUC-ROC, F1, bootstrap confidence intervals, benchmark accuracies) are computed by deterministic code. Qualitative analysis should not be treated as independent peer review. See `research/research_team_spec.md` for full methodology.

> **Conflict of Interest Disclosure:** This research was conducted and funded by TELOS AI Labs Inc., which has a commercial interest in the TELOS governance framework. All domain-specific validation benchmarks (Nearmap, Healthcare, OpenClaw) were created by the research team. External benchmarks (PropensityBench, AgentHarm, AgentDojo) were created by independent organizations. Research artifacts are published on [Zenodo](https://zenodo.org/) with persistent DOIs. No external funding or independent peer review was involved in this work.

---


## Executive Summary

This document pre-registers the experimental methodology for evaluating SetFit (Sentence Transformer Fine-Tuning) as a boundary violation detector on the TELOS healthcare governance benchmark. SetFit was identified as the medium-term path after zero-shot NLI was eliminated in Phase 1 (best AUC 0.672, below the 0.75 viability threshold). The central question is whether a few-shot fine-tuned model can learn to distinguish boundary violations from legitimate requests when both categories share overlapping clinical vocabulary -- a problem that defeated zero-shot approaches. This document specifies the hypotheses, data handling protocol, overfitting controls, ablation design, pre-registered decision criteria, reproducibility requirements, and negative result contingencies so that the experiment's outcome -- positive or negative -- is interpretable and auditable. The goal is not to make SetFit succeed; it is to determine whether SetFit succeeds, with sufficient rigor that the answer is trustworthy either way.

---

## 1. Hypothesis Formulation

### 1.1 Primary Hypotheses

**H0 (Null):** SetFit fine-tuning on the TELOS healthcare boundary corpus does not improve boundary violation detection beyond the existing keyword + cosine baseline, as measured by AUC-ROC on held-out data.

**H1 (Alternative):** SetFit fine-tuning improves AUC-ROC by at least 0.05 (absolute) over the best existing baseline on held-out data, with FPR at or below the keyword baseline (4.3%).

**What this means in plain English:** We are testing whether teaching a small AI model to recognize boundary violations from a handful of labeled examples produces a meaningful improvement over the simple methods we already have. The null hypothesis says "no, it does not." We need the data to reject this null convincingly.

### 1.2 Overfitting Counter-Hypothesis (Pre-Registered)

**H_overfit:** Any observed improvement in SetFit AUC on the 280-scenario benchmark is an artifact of training on data that is statistically entangled with the test data, rather than evidence of genuine discriminative learning.

This counter-hypothesis must be addressed by the experimental design. If SetFit achieves AUC 0.90 on cross-validation but 0.70 on a truly held-out set, H_overfit is confirmed. Specifically:

- **Scenario leakage:** SetFit sees paraphrases of test scenarios during contrastive training.
- **Config leakage:** SetFit learns config-specific patterns (e.g., "ambient" = one distribution, "call_center" = another) rather than generalizable violation semantics.
- **Vocabulary leakage:** The 26 keyword terms appear disproportionately in Cat A scenarios. SetFit may learn keyword co-occurrence rather than semantic violation structure.

### 1.3 Generalization Hypothesis

**H_gen:** SetFit's violation detection transfers to novel configs and novel violation phrasings not seen during training.

This is tested via the config-holdout protocol in Section 2.3 and the synthetic probe set in Section 2.4.

### 1.4 Theoretical Grounding

SetFit uses contrastive learning on sentence pairs to adapt a pre-trained sentence transformer, then fits a classification head. The contrastive phase pulls violation embeddings toward each other and pushes them away from legitimate request embeddings. This is architecturally distinct from zero-shot NLI in two ways:

1. **Supervised signal:** SetFit receives explicit violation/non-violation labels. NLI relied on unsupervised transfer from MNLI/SNLI pretraining, which encodes propositional entailment, not deontic compliance (as Russell identified in Phase 1).
2. **Metric learning:** SetFit reshapes the embedding space itself, potentially creating a violation-specific manifold. NLI operates in a fixed representation space not optimized for this task.

The risk: SetFit's bi-encoder backbone inherits the same "negation blindness" documented in the literature survey (NevIR: bi-encoders at 50% on negation pairs). Contrastive fine-tuning may or may not overcome this with only 61 positive examples.

---

## 2. Data Handling Protocol

### 2.1 The Core Problem: 280 Scenarios Are Both Training and Test Data

The healthcare benchmark contains 280 scenarios across 7 configs. These are the ONLY labeled data. If we train SetFit on any subset and test on the remainder, there is a risk of information leakage through:

- Shared vocabulary between Cat A and Cat C/FP scenarios within the same config
- Shared boundary text (all Cat A scenarios for a config reference the same boundaries)
- Author consistency (all scenarios authored by the same team with consistent phrasing patterns)

**This is like testing a student on questions written by the same person who wrote their study guide. The student may pass by recognizing the author's style rather than understanding the material.**

### 2.2 Primary Evaluation: Stratified K-Fold Cross-Validation

**Protocol:** 5-fold stratified cross-validation, stratified by both boundary_category AND config_id.

| Fold | Training Scenarios | Test Scenarios | Cat A in Train | Cat A in Test |
|------|-------------------|----------------|----------------|---------------|
| 1 | ~224 | ~56 | ~49 | ~12 |
| 2 | ~224 | ~56 | ~49 | ~12 |
| 3 | ~224 | ~56 | ~49 | ~12 |
| 4 | ~224 | ~56 | ~49 | ~12 |
| 5 | ~224 | ~56 | ~49 | ~12 |

**Stratification requirement:** Each fold must contain at least 1 Cat A scenario per config in the test set. Given 7-10 Cat A scenarios per config, 5-fold CV gives ~1-2 Cat A per config per fold. This is statistically thin but the best achievable without synthetic augmentation.

**Metric:** Report mean and standard deviation of AUC-ROC, F1, Cat A detection rate, and FP FPR across all 5 folds. A high standard deviation (>0.08 on AUC) indicates fold sensitivity, which is a form of instability that undermines confidence in the result.

### 2.3 Secondary Evaluation: Config-Holdout (Generalization Test)

**Protocol:** Leave-one-config-out (LOCO) cross-validation. Train on 6 configs, test on the 7th. Repeat 7 times.

| Holdout Config | Train Scenarios | Test Scenarios | Cat A Train | Cat A Test |
|----------------|----------------|----------------|-------------|------------|
| ambient_doc | 238 | 42 | 52 | 9 |
| call_center | 243 | 37 | 53 | 8 |
| coding | 238 | 42 | 51 | 10 |
| diagnostic_ai | 240 | 40 | 52 | 9 |
| patient_facing | 239 | 41 | 52 | 9 |
| predictive | 241 | 39 | 54 | 7 |
| therapeutic | 241 | 39 | 52 | 9 |

**Purpose:** This tests H_gen -- whether SetFit learns violation semantics that transfer across clinical domains. If SetFit achieves AUC 0.85 on 5-fold CV but 0.65 on LOCO, it has learned config-specific patterns, not generalizable violation detection.

**Critical test:** call_center and ambient_doc are the configs where NLI failed most catastrophically (AUC 0.267 and 0.457 respectively). If SetFit also fails on these configs when held out, the problem is with these configs' violation structure, not with the model.

**What this means:** Imagine training a medical test on data from 6 hospitals, then testing it at a 7th hospital with different patient demographics. If it still works, the test is robust. If it fails, it was memorizing hospital-specific patterns rather than learning the underlying disease.

### 2.4 Tertiary Evaluation: Synthetic Probe Set (Unseen Violation Patterns)

**Protocol:** Generate 20-30 synthetic scenarios that are NOT in the benchmark. These test novel violation phrasings not seen during any training.

**Generation method:**
1. For each of the 7 configs, write 3-4 new Cat A scenarios using violation language not present in any existing scenario (different verbs, different sentence structures, different clinical contexts).
2. Write 3-4 new Cat C/FP scenarios using boundary-adjacent vocabulary (mentioning the same clinical concepts as violations but in compliant contexts).
3. Have a second author (not the original benchmark author) write these to avoid style leakage.

**Purpose:** This is the hardest test. If SetFit passes 5-fold CV and LOCO but fails on synthetic probes, it has memorized the benchmark's phrasing patterns.

**Note on feasibility:** This requires human effort to write new scenarios. It is the single most valuable validation step and should not be skipped. Budget 2-4 hours for scenario authoring.

### 2.5 Data That SetFit May Train On

| Data Source | Allowed for Training? | Rationale |
|-------------|----------------------|-----------|
| Cat A scenarios (61) | Yes (in training folds only) | These are the positive examples |
| Cat C scenarios (94) | Yes (in training folds only) | Legitimate request negatives |
| Cat FP scenarios (21) | Yes (in training folds only) | False-positive control negatives |
| Cat B scenarios (28) | Exclude from binary task | Off-topic; different task |
| Cat D scenarios (27) | Exclude from binary task | Edge cases with ambiguous labels |
| Cat E scenarios (35) | Exclude from training, use for adversarial eval | Adversarial; not representative of deployment distribution |
| Cat H scenarios (14) | Exclude from binary task | Cross-config drift; different task |
| Boundary corpus phrasings (61 L1 + 121 L2 + 48 L3) | Use for contrastive pairs only | Not user-facing scenarios; risk of distribution shift |
| Keyword list (26 terms) | Do NOT use as features in SetFit alone | Must be tested as ablation (Section 4.3) |

**Binary classification task:** A (violation, positive) vs. C+FP (legitimate, negative). Total eligible: 61 + 94 + 21 = 176 scenarios.

**Class imbalance:** 61 positive vs. 115 negative (34.7% positive rate). Moderate imbalance. SetFit's contrastive learning naturally handles this by sampling pairs, but the classification head should be evaluated with and without class weighting.

---

## 3. Overfitting Controls

### 3.1 The Scale of the Risk

With 61 positive examples across 7 configs, we are in an extreme few-shot regime. SetFit was designed for few-shot learning (competitive with 8 examples per class on RAFT), but the TELOS problem has specific overfitting risks:

**Risk 1: Phrasing Memorization.** With only 7-10 Cat A scenarios per config, SetFit may memorize specific phrases ("add a diagnosis...even though the doctor didn't mention it") rather than learning generalizable violation features.

**Risk 2: Config-as-Feature.** If all Cat A scenarios for a config share distinctive vocabulary from the config's clinical domain (e.g., "ambient" configs mention "encounter," "note," "documentation" frequently), SetFit may use config identity as a proxy for violation probability rather than learning violation semantics.

**Risk 3: Keyword Shortcut.** The 26 violation keywords (skip, override, bypass, without review, etc.) are present in most Cat A scenarios by construction. SetFit may learn to detect keywords rather than semantic violation structure. This would replicate the keyword baseline, not improve upon it.

**Risk 4: Contrastive Collapse.** With very few positive examples, contrastive learning can collapse to a degenerate solution where all positives map to one point and all negatives map to another. This achieves high training accuracy but zero generalization.

### 3.2 Mitigation Strategies

| Risk | Mitigation | Verification |
|------|-----------|-------------|
| Phrasing memorization | LOCO evaluation (Section 2.3) + synthetic probes (Section 2.4) | If LOCO AUC drops >0.10 from CV AUC, memorization is occurring |
| Config-as-feature | Config-holdout test + config_id ablation (remove config tokens from input) | Compare SetFit with vs. without config context |
| Keyword shortcut | Keyword ablation (Section 4.3) + keyword-removed evaluation | Run SetFit on scenarios with violation keywords masked |
| Contrastive collapse | Monitor embedding space geometry post-training | Report intra-class and inter-class cosine similarity distributions |

### 3.3 Embedding Space Monitoring (Mandatory Post-Training Diagnostic)

After SetFit fine-tuning, compute and report:

1. **Mean pairwise cosine similarity within Cat A** (should increase from pre-training baseline)
2. **Mean pairwise cosine similarity within Cat C+FP** (should increase from pre-training baseline)
3. **Mean cosine similarity between Cat A centroid and Cat C+FP centroid** (should decrease from pre-training baseline)
4. **Nearest-neighbor analysis:** For each Cat A scenario, what are the 5 nearest neighbors? If they are all Cat A, the model has good separation. If Cat C/FP neighbors appear, those are the failure cases to examine.
5. **t-SNE/UMAP visualization:** 2D projection of all 176 scenarios colored by category, before and after SetFit fine-tuning. Visual confirmation of separation.

**What this means:** After training, we look at the model's internal representation to verify it actually learned meaningful patterns, not shortcuts. This is like checking whether a student understands the material (can explain it differently) or just memorized the answers (can only reproduce them verbatim).

### 3.4 Training Hyperparameter Constraints

To limit overfitting capacity:

| Parameter | Constraint | Rationale |
|-----------|-----------|-----------|
| Epochs (contrastive) | Test {1, 2, 3, 5}. Report all. | >5 epochs on 176 examples risks memorization |
| Epochs (classification head) | Test {1, 2, 5, 10}. Report all. | |
| Batch size | 8 or 16 | Larger batches on small data = fewer gradient steps = less overfitting |
| Learning rate | {2e-5, 5e-5, 1e-4} | Standard SetFit range |
| Number of contrastive pairs | Report exact count | More pairs from fewer examples = more reuse = more overfitting risk |
| Early stopping | Monitor validation loss per fold | Stop if validation loss increases for 2 consecutive checkpoints |

---

## 4. Ablation Design

### 4.1 Required Ablations (Must Run)

Every ablation below must be run on the same 5-fold CV splits and LOCO splits for comparability. Results must be reported in a single comparison table.

#### Ablation 1: SetFit vs. Logistic Regression on Frozen Embeddings

**Purpose:** Isolate whether SetFit's improvement (if any) comes from contrastive fine-tuning of the embedding space or merely from the classification head.

| Configuration | Embedding | Classifier |
|---------------|----------|-----------|
| SetFit-MiniLM | Fine-tuned MiniLM-L6-v2 | SetFit head (logistic regression) |
| Frozen-MiniLM-LR | Frozen MiniLM-L6-v2 | Logistic regression on [CLS] or mean pooling |
| Frozen-MiniLM-SVM | Frozen MiniLM-L6-v2 | Linear SVM on [CLS] or mean pooling |

**Interpretation:** If Frozen-MiniLM-LR achieves similar AUC to SetFit-MiniLM, then the contrastive fine-tuning is not adding value. The improvement comes from a linear decision boundary in the existing space, which is simpler and less prone to overfitting.

#### Ablation 2: Backbone Comparison (MiniLM vs. MPNet)

**Purpose:** Test whether embedding dimensionality affects SetFit's ability to learn violation structure.

| Configuration | Backbone | Dimensions | Parameters |
|---------------|---------|-----------|-----------|
| SetFit-MiniLM | all-MiniLM-L6-v2 | 384 | 22M |
| SetFit-MPNet | all-mpnet-base-v2 | 768 | 110M |

**Rationale:** TELOS already has dual-model infrastructure (MiniLM + MPNet). MPNet's higher dimensionality may capture finer-grained distinctions relevant to violation detection. The Phase 1 NLI experiment showed non-monotonic scaling (base > large), so bigger is not always better. We must test both.

#### Ablation 3: SetFit With and Without Keyword Features

**Purpose:** Test whether keyword features provide orthogonal signal to SetFit embeddings.

| Configuration | Input | Features |
|---------------|-------|----------|
| SetFit-only | Request text | SetFit embedding |
| Keywords-only | Request text | 26-dim binary keyword vector |
| SetFit+Keywords | Request text | SetFit embedding concatenated with 26-dim keyword vector |

**Interpretation:** If SetFit+Keywords > SetFit-only, keywords provide signal that contrastive learning did not capture (likely the deontic operators: "skip," "override," "bypass"). If SetFit-only >= SetFit+Keywords, the contrastive learning has already internalized keyword patterns.

#### Ablation 4: Per-Config vs. Unified Model

**Purpose:** Test whether a single model can govern all 7 configs or whether config-stratified models are needed.

| Configuration | Training Data | Test Data |
|---------------|-------------|----------|
| Unified | All 7 configs pooled | 5-fold CV across all configs |
| Per-config (7 models) | Single config only | Within-config CV (small samples; report with caveat) |
| Config-stratified (2 groups) | Propositionally-explicit configs vs. contextual configs | Cross-group evaluation |

**Rationale from Phase 1:** NLI showed a bimodal distribution -- AUC > 0.90 for therapeutic/diagnostic_ai vs. AUC < 0.46 for call_center/ambient_doc. If this bimodality persists under SetFit, the two groups may require fundamentally different approaches. A per-config model with only 7-10 Cat A examples per config is almost certainly overfitting, but the experiment documents this.

**Config grouping for stratified test:**
- Group 1 (propositionally explicit): therapeutic, diagnostic_ai, coding
- Group 2 (contextual/pragmatic): call_center, ambient_doc, patient_facing, predictive

### 4.2 Desirable Ablations (Run If Time Permits)

#### Ablation 5: Contrastive Pair Generation Strategy

**Purpose:** Test whether SetFit benefits from using the boundary corpus phrasings as additional contrastive anchors.

| Strategy | Positive Pairs | Negative Pairs |
|----------|---------------|----------------|
| Scenario-only | Cat A scenario pairs | Cat A vs. Cat C/FP scenario pairs |
| Corpus-augmented | Cat A scenarios + boundary corpus L1/L2 phrasings | Same + corpus phrasings vs. Cat C/FP |
| Hard-negative mining | As above, but negatives selected by highest cosine similarity to positives | Targeted at the vocabulary-overlap boundary |

#### Ablation 6: FastFit Comparison

**Purpose:** FastFit (IBM Research, NAACL 2024) achieves 3-20x faster training than SetFit with comparable accuracy. Token-level similarity may be particularly relevant for distinguishing "skip the allergies" from "document the allergies."

#### Ablation 7: SetFit + NLI Ensemble

**Purpose:** Phase 1 showed NLI achieves AUC > 0.90 on 2 of 7 configs (therapeutic, diagnostic_ai). Test whether an ensemble of SetFit (all configs) + NLI (2 high-performing configs) outperforms either alone.

---

## 5. Pre-Registered Decision Criteria

### 5.1 Primary Thresholds (5-Fold CV, Binary A vs. C+FP)

Following the Phase 1 pattern, these thresholds are registered before any SetFit experiment is run. The experiment outcome maps to a disposition:

| Outcome | AUC-ROC (mean +/- 1 SD) | Cat A Detection (mean) | FP FPR (mean) | Disposition |
|---------|--------------------------|------------------------|---------------|-------------|
| **GREEN** | >= 0.85 | >= 80% | <= 10% | SetFit approved for governance pipeline integration |
| **YELLOW** | 0.75 -- 0.84 | >= 65% | <= 20% | SetFit viable as ensemble component; investigate per-config stratification |
| **RED (marginal)** | 0.65 -- 0.74 | >= 50% | any | SetFit marginal; contrastive fine-tuning insufficient for this task |
| **RED (fail)** | < 0.65 | any | any | SetFit eliminated; embedding-based boundary detection has fundamental limits for contextual violations |

### 5.2 Generalization Thresholds (LOCO)

| Outcome | LOCO AUC vs. CV AUC | Disposition |
|---------|---------------------|-------------|
| **Generalizes** | LOCO mean AUC within 0.05 of CV mean AUC | Model learns transferable features |
| **Partial transfer** | LOCO mean AUC 0.05-0.10 below CV mean AUC | Model partially overfits to config-specific patterns |
| **Does not generalize** | LOCO mean AUC > 0.10 below CV mean AUC | Model overfits to config-specific patterns; config-stratified architecture needed |

### 5.3 Improvement Over Baselines

SetFit must demonstrate statistically significant improvement over every existing baseline to be worth the added complexity. "Statistically significant" requires bootstrap confidence interval on AUC difference that excludes zero (1000 bootstrap resamples, 95% CI).

| Baseline | Phase 1 AUC | SetFit Must Beat |
|----------|-------------|-----------------|
| Random | 0.500 | Obviously |
| Keyword (26 terms) | 0.724 | By >= 0.05 with bootstrap p < 0.05 |
| Cosine (current production) | Measured per-fold | By >= 0.05 with bootstrap p < 0.05 |
| NLI-base best (Framing A) | 0.672 | Not a target; NLI already eliminated |

### 5.4 Per-Config Minimum Viability

Even if aggregate AUC is GREEN, no config may be worse than the keyword baseline. This prevents the "good aggregate, terrible subset" failure pattern identified in Phase 1 (aggregate AUC 0.672 masked call_center AUC 0.267).

| Criterion | Threshold |
|-----------|-----------|
| Worst-config AUC | Must exceed 0.65 (no config is worse than chance + 0.15) |
| Worst-config Cat A detection | Must exceed 50% |
| Configs below keyword AUC (0.724) | Flag and document; if >3 configs are below, overall viability is questionable |

### 5.5 What "Success" Means Concretely

**GREEN means:** SetFit becomes Layer 1.5 in the governance pipeline, between the keyword pre-filter (Layer 0) and the existing cosine check (Layer 1). It must add detection value without introducing unacceptable FPR. The pipeline becomes: Keyword screen (fast, high-precision) --> SetFit (fast, learned features) --> Cosine centroid (existing, semantic proximity) --> Optional NLI (slow, 2 configs only).

**YELLOW means:** SetFit is not a standalone detector but provides useful signal when combined with other methods. This leads to an ensemble architecture where SetFit, keywords, and cosine are combined via learned weights.

**RED means:** SetFit does not work for this task. We document why and move to the next approach (see Section 8).

---

## 6. Reproducibility Requirements

### 6.1 Pinned Artifacts

Every artifact below must be version-pinned in the experiment script and recorded in the results JSON.

| Artifact | What to Pin | How |
|----------|-----------|-----|
| SetFit library version | `setfit==X.Y.Z` | `pip freeze` output in results |
| sentence-transformers version | `sentence-transformers==X.Y.Z` | Exact version in requirements |
| torch version | `torch==X.Y.Z` | Affects contrastive training |
| Base model revision | HuggingFace commit SHA for all-MiniLM-L6-v2 | `revision="sha..."` in model load |
| MPNet revision (if used) | HuggingFace commit SHA for all-mpnet-base-v2 | Same |
| Random seed | Fixed integer (42 or similar) | Set via `set_seed(42)` before any training |
| NumPy/sklearn seeds | Fixed | `np.random.seed(42)`, `sklearn` random_state=42 |
| CV fold assignments | Saved as JSON | Exact scenario_id-to-fold mapping persisted |
| LOCO assignments | Deterministic from config_id | config_id determines holdout |
| Hyperparameters | All values recorded | Full SetFitTrainer config in results JSON |
| Dataset version | healthcare_counterfactual_v1.jsonl SHA-256 hash | Hash computed and recorded before training |
| Hardware | CPU model, RAM, OS version | Recorded in results metadata |
| Training duration | Wall-clock time per fold | Recorded in results |

### 6.2 Determinism Verification

Run the full experiment twice with identical seeds. All metrics must match to 4 decimal places. If they do not, identify the source of non-determinism (typically CUDA non-determinism; CPU should be deterministic with pinned seeds) and document it.

### 6.3 Artifact Storage

| Artifact | Location | Format |
|----------|----------|--------|
| Experiment script | `validation/healthcare/setfit_mve.py` | Python |
| Results (all ablations, all folds) | `validation/healthcare/setfit_mve_results.json` | JSON |
| Fold assignments | Embedded in results JSON | JSON array |
| Trained model checkpoints | `validation/healthcare/setfit_models/` | SetFit format (do not commit to git; record hash) |
| This document | `research/setfit_mve_experimental_design.md` | Markdown |
| Closure document (post-experiment) | `research/setfit_mve_phase2.md` | Markdown |

---

## 7. Publication Readiness Assessment

### 7.1 What a Skeptical Reviewer Would Challenge

| Challenge | Response Strategy |
|-----------|-----------------|
| "280 scenarios is too small for ML training" | Acknowledged. SetFit is designed for few-shot. We report 5-fold CV with standard deviations. We compare against frozen-embedding baselines to isolate the value of contrastive fine-tuning. We use LOCO and synthetic probes to test generalization. |
| "You are training and testing on the same benchmark" | Acknowledged as the primary methodological limitation. We mitigate with three-tier evaluation (CV, LOCO, synthetic probes). We explicitly test H_overfit. |
| "The keyword baseline is a straw man" | The keyword baseline is not a straw man; it outperformed all NLI models in Phase 1. It is a legitimate competitor. SetFit must beat it with statistical significance. |
| "Why not just use a larger LLM?" | Latency constraints (current pipeline is <50ms, LLMs are 1-10s). Documented in Phase 1, Section 4.4. Also: cost, determinism, offline capability, and auditability. |
| "Why binary classification? Boundary compliance is multi-class." | The binary task (violation vs. legitimate) is the minimum viable question. Multi-class refinement (which boundary? what severity?) is a follow-on experiment contingent on the binary task succeeding. |
| "No external validation dataset" | This is a real limitation. The synthetic probe set (Section 2.4) partially addresses it, but an independently authored external validation set is the gold standard. We recommend commissioning one if SetFit reaches YELLOW or GREEN. |
| "SetFit inherits bi-encoder negation blindness" | Acknowledged in the literature survey (NevIR: bi-encoders at 50% on negation pairs). The experiment tests whether contrastive fine-tuning on domain-specific violation pairs overcomes this limitation. If it does not, that is a publishable finding about the limits of contrastive learning for deontic reasoning. |

### 7.2 Reporting Standards

The closure document must include:

1. **Full results table:** All ablations, all folds, all metrics (AUC, F1, precision, recall, Cat A detection, FP FPR).
2. **Statistical tests:** Bootstrap confidence intervals on all AUC differences. Paired permutation tests for ablation comparisons.
3. **Error analysis:** For every false negative (missed Cat A) and false positive (flagged Cat C/FP), document the scenario_id, the request text, and a hypothesis for why the model failed.
4. **Embedding visualization:** t-SNE/UMAP before and after fine-tuning.
5. **Negative results:** If SetFit fails, the closure document must be as thorough as the Phase 1 NLI closure. A well-documented negative result is more valuable than an undocumented positive one.

### 7.3 Ethical Considerations

- **No patient data:** All scenarios are synthetic. Zero-PHI attestation applies.
- **No demographic features:** SetFit sees request text only, not patient demographics. The equity/bias scenarios (Cat E, attack family 12) test whether the model inadvertently encodes demographic biases from clinical language patterns.
- **Dual-use risk:** A model fine-tuned to detect boundary violations could, in principle, be used to craft adversarial violations that evade detection. This is inherent to any security classifier. The Cat E adversarial scenarios test robustness against this.

---

## 8. Negative Result Contingency

### 8.1 If SetFit Achieves RED (Fail): AUC < 0.65

**Interpretation:** Contrastive fine-tuning on a bi-encoder backbone cannot overcome negation blindness for deontic reasoning tasks, even with domain-specific training data. This is a meaningful finding -- it confirms the NevIR result (bi-encoders at 50% on negation) extends to policy compliance.

**Next steps (in priority order):**

1. **Cross-encoder fine-tuning (DeBERTa-v3-base):** Phase 1 showed zero-shot NLI achieved AUC 0.672. Fine-tuning on the 176 binary examples may push this above 0.75. Cross-encoders handle negation significantly better than bi-encoders (NevIR: 75% vs. 50%). The trade-off is latency: cross-encoders require O(N) forward passes per boundary, estimated at 100-300ms per request for 6-8 boundaries per config.

2. **FastFit (IBM Research):** Token-level similarity scoring may capture the "skip" vs. "document" distinction that sentence-level contrastive learning misses. 3-20x faster training than SetFit enables more rapid iteration.

3. **Assertion Detection Transfer:** Clinical assertion detection (i2b2 framework) achieves 92.9% on Absent/Present classification. The mapping from "allergy is Present" / "allergy is Absent" to "documenting allergies" / "skipping allergies" is structurally analogous. This requires clinical NLP domain adaptation.

4. **Hybrid keyword + NLI + cosine ensemble:** If no single method achieves GREEN, a learned ensemble combining all available signals (keyword match, cosine similarity, NLI score for 2 configs, SetFit embedding distance) may cross the threshold. This is architecturally complex but bounds the problem.

### 8.2 If SetFit Achieves RED (Marginal): AUC 0.65-0.74

**Interpretation:** SetFit provides marginal improvement but is not sufficient as a standalone detector. Similar to Phase 1 NLI's outcome.

**Next steps:**
1. Evaluate SetFit as a feature in an ensemble (not a standalone classifier).
2. Investigate per-config stratification: SetFit may work for the "easy" configs (therapeutic, diagnostic_ai) while a different approach is needed for "hard" configs (call_center, ambient_doc).
3. Test data augmentation: Use LLM-generated paraphrases to expand the 61 Cat A examples to 200+ (following the LegalLens augmentation pattern: 312 to 936 examples, +7.65% F1).

### 8.3 The Fundamental Limits Question

If SetFit, NLI, and all embedding-based approaches fail to achieve GREEN on the full benchmark, we must consider whether the problem is with the methods or with the task definition:

**Possibility A: Embedding-based boundary detection has fundamental limits for contextual violations.** The violation semantics in call_center and ambient_doc configs are pragmatic (what is contextually inappropriate) rather than propositional (what logically contradicts a rule). No amount of contrastive fine-tuning can make a bi-encoder reason about context-dependent deontic norms. This would be a publishable finding and would redirect TELOS toward hybrid architectures where embedding-based detection handles propositionally explicit violations (4-5 configs) and a different mechanism (rule-based, LLM-in-the-loop, or human review) handles contextual violations (2-3 configs).

**Possibility B: The benchmark's "hard" configs have poorly specified boundaries.** The call_center AUC of 0.267 in Phase 1 may reflect ambiguity in the boundary text, not a fundamental detection limit. If the boundaries for call_center and ambient_doc are rewritten to be more semantically explicit, detection may improve. This is a benchmark-quality hypothesis, not a method hypothesis, and must be tested separately.

**Possibility C: 61 positive examples is simply not enough.** The LegalLens paper achieved F1 of 84.73% with 312 examples augmented to 936. TELOS's 61 Cat A examples may be below the minimum viable training set size for any method. The data augmentation ablation (Section 8.2) tests this directly.

**Decision point:** If three consecutive approaches (NLI, SetFit, and one of the contingency approaches) all achieve RED on the full benchmark, convene the full research team (Russell, Gebru, Karpathy, Schaake, Nell) for a disposition review. The question on the table: "Should we accept a config-stratified architecture where 4-5 configs use embedding-based detection and 2-3 configs use a fundamentally different approach?" This is not a failure; it is an architectural finding about the heterogeneity of clinical AI governance tasks.

---

## 9. Experiment Execution Checklist

Before running any SetFit training:

- [ ] Compute SHA-256 hash of `healthcare_counterfactual_v1.jsonl` and record
- [ ] Pin all library versions in a `requirements_setfit_mve.txt`
- [ ] Generate and save 5-fold CV assignments as JSON (with seed=42)
- [ ] Verify fold stratification (at least 1 Cat A per config per fold)
- [ ] Run keyword baseline on same folds to establish per-fold comparison
- [ ] Run frozen-embedding logistic regression on same folds (Ablation 1)
- [ ] Record pre-training embedding geometry (intra/inter class cosine means)

During training:

- [ ] Log all hyperparameters to results JSON
- [ ] Log training loss per epoch
- [ ] Implement early stopping on validation loss
- [ ] Save model checkpoints per fold

After training:

- [ ] Run all 5 folds and record metrics with standard deviations
- [ ] Run LOCO evaluation (7 holdout configs)
- [ ] Run synthetic probe evaluation (Section 2.4, if probes authored)
- [ ] Compute post-training embedding geometry
- [ ] Generate t-SNE/UMAP visualization
- [ ] Run all required ablations on same folds
- [ ] Compute bootstrap confidence intervals on all AUC differences vs. baselines
- [ ] Perform full error analysis on false negatives and false positives
- [ ] Map results to GREEN/YELLOW/RED criteria
- [ ] Write closure document following Phase 1 format
- [ ] Present to research team for review

---

## 10. Traceability

### 10.1 Decision Chain

| Decision | Evidence | Document |
|----------|---------|----------|
| NLI eliminated as standalone | Phase 1: Best AUC 0.672, below 0.75 threshold | `research/cross_encoder_nli_mve_phase1.md` |
| Keyword baseline established | Phase 1: AUC 0.724, FPR 4.3% | Same |
| SetFit selected as next approach | Literature survey: 8 examples/class, bi-encoder backbone, ONNX-compatible | `research/literature_survey_safety_classification.md` |
| This experimental design | Pre-registration of methodology before execution | This document |
| SetFit results (pending) | Experiment not yet run | `research/setfit_mve_phase2.md` (future) |

### 10.2 Risk Management (EU AI Act Article 9(2))

This pre-registration document constitutes a documented risk management measure: systematic specification of experimental methodology with pre-registered decision criteria before execution. The negative result contingency plan (Section 8) ensures that any outcome -- positive or negative -- leads to a defined next action, preventing ad hoc post-hoc rationalization.

### 10.3 Methodologist's Note

The Phase 1 NLI experiment validated a concern I raised during the initial review: testing a single xsmall model was insufficient to draw conclusions about the entire NLI approach. The team expanded testing to 3 model sizes and 4 framings, which confirmed the negative result with much greater confidence. The same rigor standard applies here. SetFit must be tested with multiple backbones, multiple evaluation protocols, and explicit overfitting controls. A single positive cross-validation result is not sufficient evidence for production deployment. The LOCO and synthetic probe evaluations are what separate a research finding from an engineering decision.

The most common failure mode in ML research is not getting the wrong answer -- it is getting the right answer on the wrong question. The question here is not "can SetFit achieve high AUC on a 280-scenario benchmark?" The question is "can SetFit learn to detect boundary violations in clinical AI governance with sufficient generalization for production deployment?" These are different questions. The experimental design must ensure we are answering the second one.

---

*Pre-registered: 2026-02-17. No experiments have been run as of this date. All decision criteria are fixed prior to data analysis.*
*Filed as required audit artifact per Schaake recommendation (EU AI Act Article 9(2)).*
