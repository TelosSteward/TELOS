# Cross-Encoder NLI MVE — Phase 1 Experimental Record

**TELOS AI Labs Inc. — Research Program**
**Date:** 2026-02-16
**Status:** Complete (Negative Result — NLI Eliminated as Standalone Method)
**Authors:** TELOS Development Team + Research Team (Russell, Gebru, Karpathy, Schaake, Nell)

---

### Disclosures

> **Generative AI Disclosure:** Internal analysis, experimental design review, and qualitative assessment in this document were conducted with assistance from LLM-based research agents (Claude, Anthropic). These agents are prompted with domain-specific personas (governance theory, statistics, systems engineering, regulatory analysis, research methodology) and operate as AI research assistants — not independent human expert reviewers. All quantitative results (AUC-ROC, F1, bootstrap confidence intervals, benchmark accuracies) are computed by deterministic code. Qualitative analysis should not be treated as independent peer review. See `research/research_team_spec.md` for full methodology.

> **Conflict of Interest Disclosure:** This research was conducted and funded by TELOS AI Labs Inc., which has a commercial interest in the TELOS governance framework. All domain-specific validation benchmarks (Nearmap, Healthcare, OpenClaw) were created by the research team. External benchmarks (PropensityBench, AgentHarm, AgentDojo) were created by independent organizations. Research artifacts are published on [Zenodo](https://zenodo.org/) with persistent DOIs. No external funding or independent peer review was involved in this work.

---


## Executive Summary

This document records the systematic evaluation of zero-shot NLI (Natural Language Inference) cross-encoder models as a boundary compliance detection method for TELOS's healthcare governance benchmark. The experiment tested whether NLI models could discriminate boundary violations (Category A) from legitimate requests (Categories C/FP) across 280 healthcare scenarios and 7 agent configurations.

**Result:** NLI fails as a standalone boundary compliance method. The best configuration (DeBERTa-v3-base, 86M params, Framing A) achieved AUC-ROC 0.672 — marginally above chance and below the pre-registered 0.75 viability threshold. A simple keyword baseline (26 terms, zero ML) achieved AUC-ROC 0.724 with an order of magnitude lower false positive rate (4.3% vs. 33.3%).

**Disposition:** NLI eliminated as a standalone architecture. Keyword pre-filter approved for integration into the governance pipeline. SetFit fine-tuning on existing embeddings approved as medium-term path.

---

## 1. Hypothesis

**H0 (null):** Zero-shot NLI cross-encoders cannot discriminate boundary violations from legitimate requests on the TELOS healthcare benchmark (AUC-ROC not significantly different from 0.50).

**H1:** Zero-shot NLI cross-encoders can discriminate violations from legitimate requests (AUC-ROC > 0.75 with acceptable FPR).

**Theoretical basis:** NLI models classify sentence pairs as entailment/contradiction/neutral. If boundary text is the premise and a user request is the hypothesis, violations should produce high contradiction scores while legitimate requests should produce entailment or neutral scores.

**Counter-hypothesis (Russell, pre-registered):** Boundary compliance is a deontic reasoning task (does this action violate this rule?), not a propositional reasoning task (does A logically entail/contradict B?). NLI pretraining on MNLI/SNLI does not encode deontic modalities.

---

## 2. Methodology

### 2.1 Dataset

- **Source:** `validation/healthcare/healthcare_counterfactual_v1.jsonl`
- **Size:** 280 scenarios across 7 healthcare agent configurations
- **Category distribution:** A=61 (violations), B=28 (off-topic), C=94 (legitimate), D=27 (edge cases), E=35 (adversarial), FP=21 (false-positive controls), H=14
- **Binary classification:** A vs. C+FP (61 positive, 115 negative)

### 2.2 Models Tested

| Model | Parameters | Architecture | Source |
|-------|-----------|-------------|--------|
| `cross-encoder/nli-deberta-v3-xsmall` | 22M | DeBERTa-v3 | HuggingFace |
| `cross-encoder/nli-deberta-v3-base` | 86M | DeBERTa-v3 | HuggingFace |
| `cross-encoder/nli-deberta-v3-large` | 304M | DeBERTa-v3 | HuggingFace |

All models are zero-shot (no fine-tuning). Output: `[contradiction, entailment, neutral]` probabilities via softmax.

### 2.3 NLI Framings

| Framing | Premise | Hypothesis | Target Label | Rationale |
|---------|---------|-----------|------|-----------|
| A | Boundary text (prohibition) | User request | Contradiction | Violation contradicts the prohibition |
| B | Affirmative violation statement | User request | Entailment | Request entails the violation action |
| C | User request | Boundary text | Contradiction | Reversed direction |
| D | Cat A corpus exemplar (leave-one-out) | User request | Entailment | Request resembles known violations |

### 2.4 Baselines

| Baseline | Method | Purpose |
|----------|--------|---------|
| Random | AUC = 0.50 reference | Chance-level comparison |
| Keyword | 26 violation-indicator terms (skip, override, bypass, without review, etc.) | Simple pattern matching |

### 2.5 Additional Experiments

- **Simplified boundary text:** Stripped em-dashes, parenthetical citations, statistics from boundary text
- **Per-config breakdown:** AUC and detection rates computed per healthcare configuration

### 2.6 Pre-Registered Decision Criteria

| Outcome | Next Step |
|---------|-----------|
| Best NLI AUC < 0.65 | NLI eliminated → proceed to SetFit |
| Best NLI AUC 0.65–0.75 | NLI marginal → SetFit likely better, NLI as ensemble feature |
| Best NLI AUC > 0.75 | NLI viable → test in cascade architecture |
| Keyword baseline AUC > 0.70 | Add keyword features before investing in ML fine-tuning |

### 2.7 Green/Yellow/Red Criteria

- **GREEN:** Cat A detection >= 80% AND FP FPR <= 15%
- **YELLOW:** Cat A detection >= 60% AND FP FPR <= 25%
- **RED:** Everything else

---

## 3. Results

### 3.1 Model Comparison (Framing A — Best Diagnostic)

| Model | Params | AUC-ROC | Cat A Detection | FP FPR | Score Gap | Verdict |
|-------|--------|---------|-----------------|--------|-----------|---------|
| Keyword baseline | — | **0.7242** | 49.2% | **4.3%** | binary | RED |
| xsmall | 22M | 0.5682 | 36.1% | 4.8% | -0.0021 | RED |
| **base** | **86M** | **0.6724** | **63.9%** | 33.3% | +0.0244 | RED |
| large | 304M | 0.6038 | 50.8% | 9.5% | -0.0085 | RED |

### 3.2 All Framings — xsmall (22M)

| Framing | AUC-ROC | Cat A | FPR | Score Gap |
|---------|---------|-------|-----|-----------|
| A | 0.5682 | 36.1% | 4.8% | -0.0021 |
| B | 0.1769 | 100% (inverted) | 100% | -0.0311 |
| C | 0.5829 | 42.6% | 33.3% | +0.0007 |
| D | 0.1413 | 100% (inverted) | 100% | -0.1362 |

### 3.3 All Framings — base (86M)

| Framing | AUC-ROC | Cat A | FPR | Score Gap |
|---------|---------|-------|-----|-----------|
| A | **0.6724** | 63.9% | 33.3% | +0.0244 |
| B | 0.1799 | 100% (inverted) | 100% | -0.0474 |
| C | 0.6450 | 62.3% | 38.1% | +0.0004 |
| D | 0.1790 | 100% (inverted) | 100% | -0.1443 |

### 3.4 Simplified Boundary Text (xsmall, Framing A)

AUC-ROC dropped from 0.5682 to 0.5162. Simplified text performed **worse**, confirming the bottleneck is not input formatting.

### 3.5 Per-Config Breakdown (base, Framing A)

| Configuration | AUC-ROC | Cat A Rate | Safe FPR | Score Gap |
|--------------|---------|------------|----------|-----------|
| therapeutic | **0.972** | 89% (8/9) | 6% (1/16) | +0.3153 |
| diagnostic_ai | **0.917** | 78% (7/9) | 6% (1/16) | +0.4448 |
| coding | 0.694 | 60% (6/10) | 24% (4/17) | +0.0109 |
| predictive | 0.672 | 57% (4/7) | 29% (5/17) | -0.0854 |
| patient_facing | 0.625 | 56% (5/9) | 19% (3/16) | -0.0872 |
| ambient_doc | 0.457 | 78% (7/9) | 67% (12/18) | -0.0562 |
| call_center | **0.267** | 25% (2/8) | 40% (6/15) | -0.3899 |

**Bimodal distribution:** NLI works for propositionally explicit violations (therapeutic, diagnostic_ai: AUC > 0.90) but fails for contextual/pragmatic violations (call_center, ambient_doc: AUC < 0.46).

### 3.6 Framing B/D Inversion Stability

The inverted signal is model-size invariant:
- Framing B: xsmall 0.177, base 0.180, delta = +0.003
- Framing D: xsmall 0.141, base 0.179, delta = +0.038

The inversion is a systematic property of how NLI represents these framings, not a capacity limitation.

---

## 4. Analysis

### 4.1 Why NLI Fails (Russell — Governance Theory)

Boundary compliance is a deontic reasoning task, not a propositional one. NLI asks "does A logically entail/contradict B?" while compliance asks "would this action violate this rule?" A car going 80 in a 65 zone does not "contradict" the speed limit in any logical sense — it *violates* it. NLI pretraining on MNLI/SNLI does not encode this distinction.

**Revised thesis:** NLI transfers to boundary compliance only when violations are propositionally explicit (~2 of 7 healthcare configs). For configs where violations are procedural, contextual, or involve acts of omission, NLI lacks the representational capacity to discriminate.

### 4.2 Why Keywords Win (Gebru — Statistical Analysis)

The keyword baseline succeeds because violation language contains reliable lexical markers of prohibited action ("skip," "override," "bypass," "without review"). These are deontic operators expressed lexically. The base NLI model's score gap of +0.0244 means violations and legitimate requests receive nearly identical NLI scores — the model separates within 2.4% of score space, exploiting noise in the tail of the contradiction distribution.

With ~60 positive samples, the standard error on AUC is approximately 0.06-0.07. The delta between keyword (0.724) and NLI-base (0.672) is 0.052 — not statistically significant (bootstrap p > 0.3). At matched FPR, the keyword baseline dominates.

### 4.3 Non-Monotonic Scaling (Gebru)

The pattern xsmall (0.568) < large (0.604) < base (0.672) violates the expectation that more parameters means better performance. The large model has sufficient capacity to learn fine-grained semantic distinctions that are orthogonal to regulatory category boundaries. The base model sits in a sweet spot — enough capacity to capture coarse semantic similarity, not enough to overfit to NLI-specific features. Consistent with the "capability overhang" pattern in transfer learning literature.

### 4.4 Latency Implications (Karpathy — Systems)

| Model | 280 scenarios | Per-request (amortized) |
|-------|--------------|------------------------|
| Keyword | <0.01s | ~0.04ms |
| xsmall | 35s | ~125ms |
| base | 80s | ~286ms |
| large | 3,156s | ~11.3s |

TELOS's existing cosine embedding path does a full governance decision in under 50ms. Cross-encoders at 125-11,300ms per request are 2.5x to 226x slower. Cross-encoders are architecturally wrong for this workload — they require O(N) forward passes per request (one per boundary), while bi-encoders precompute centroids and do O(1) cosine similarity at inference.

### 4.5 Regulatory Implications (Schaake)

A 33.3% FPR is disqualifying under any compliance framework requiring proportionate oversight (EU AI Act Article 9). A system that flags one in three compliant interactions produces alert fatigue that masks genuine violations. The per-config AUC of 0.267 (call_center) is worse than random — deploying this would constitute a documentable deficiency.

The negative result, properly documented, is a regulatory *strength*: it demonstrates systematic evaluation of alternatives with documented rationale for rejection, satisfying EU AI Act Article 9(2) risk management requirements.

---

## 5. Disposition

### 5.1 Decision (Unanimous, 5/5 Research Team)

**NLI is eliminated as a standalone boundary compliance method.** The best achievable AUC (0.672) falls in the pre-registered "marginal" band (0.65-0.75). The keyword baseline (AUC 0.724, FPR 4.3%) outperforms all NLI configurations on aggregate metrics with zero ML overhead.

### 5.2 Residual Value

For 2 of 7 configs (therapeutic, diagnostic_ai), NLI achieves AUC > 0.90. These configs may benefit from NLI as an ensemble feature in a config-stratified architecture. This should be evaluated during SetFit integration.

### 5.3 Approved Next Steps

1. **Immediate:** Integrate keyword pre-filter into the governance pipeline as Layer 0 (high-precision, low-latency first screen)
2. **Medium-term:** SetFit fine-tuning on existing MiniLM/MPNet embeddings using the boundary corpus as training data
3. **Architecture:** Config-stratified pipeline (keyword L0 → cosine L1 → SetFit L1.5 → optional NLI L2 for 2 configs)

---

## 6. Artifacts

| Artifact | Location | Purpose |
|----------|----------|---------|
| MVE script | `validation/healthcare/cross_encoder_mve.py` | Reproducible evaluation (all framings, models, baselines) |
| xsmall all-framings results | `validation/healthcare/cross_encoder_mve_phase1_xsmall_all.json` | Full xsmall run with keyword baseline |
| base all-framings results | `validation/healthcare/cross_encoder_mve_phase1_base_all.json` | Full base run with keyword baseline |
| base Framing A results | `validation/healthcare/cross_encoder_mve_phase1_base_A.json` | Initial base model test |
| large Framing A results | `validation/healthcare/cross_encoder_mve_phase1_large_A.json` | Large model test |
| simplified boundaries results | `validation/healthcare/cross_encoder_mve_phase1_simplified_A.json` | Simplified boundary text test |
| This document | `research/cross_encoder_nli_mve_phase1.md` | Experimental record and disposition memo |

### 6.1 Reproducibility

```bash
# Reproduce all Phase 1 results
python3 validation/healthcare/cross_encoder_mve.py --baselines --output results_xsmall.json
python3 validation/healthcare/cross_encoder_mve.py --model cross-encoder/nli-deberta-v3-base --baselines --output results_base.json
python3 validation/healthcare/cross_encoder_mve.py --model cross-encoder/nli-deberta-v3-large --framing A --baselines --output results_large.json
python3 validation/healthcare/cross_encoder_mve.py --simplify-boundaries --framing A --output results_simplified.json
```

All models are publicly available on HuggingFace. Dataset is in the repository. No external API calls required.

---

## 7. Traceability

### 7.1 Risk Management (EU AI Act Article 9(2))

This experiment constitutes a documented risk management measure: systematic evaluation of an alternative boundary detection method with pre-registered decision criteria and quantitative outcomes. The negative result was processed through a 5-agent research review with documented consensus. The decision to reject NLI and proceed to keyword + SetFit architecture is traceable to the empirical evidence in Sections 3-4.

### 7.2 Research Team Sign-Off

| Agent | Verdict | Key Finding |
|-------|---------|-------------|
| Russell (Governance) | No-go | Deontic logic gap confirmed — NLI transfers only for propositionally explicit violations |
| Gebru (Data Science) | No-go | AUC difference not statistically significant; keyword baseline dominates at matched FPR |
| Karpathy (Systems) | No-go | Cross-encoders architecturally incompatible with real-time governance (O(N) vs O(1)) |
| Schaake (Regulatory) | No-go | 33.3% FPR disqualifying; negative result is a regulatory strength when documented |
| Nell (Methodology) | No-go | Under-testing concern resolved; methodology grade B+; experiments confirmed original finding |

---

*Last updated: 2026-02-16. Filed as required audit artifact per Schaake recommendation.*
