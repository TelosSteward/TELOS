# Optimizer Production Run Analysis — 200 Trials x 5 Generations x 5 Seeds

**Date:** 2026-02-27
**Run Duration:** ~15.1 hours (2026-02-26 09:55 to 2026-02-27 01:03)
**Machine:** Mac Mini M4 Pro
**Data:** `optimizer_output/production_200t_5g/`

---

## 1. Executive Summary

The production optimizer run explored 5,000 Optuna trials (200 per generation x 25 generations) across a 14-parameter threshold configuration space, evaluated against 5 benchmarks totaling ~5,212 scenarios. **No configuration passed the four-gate ratchet.** All 25 generation objectives returned `-inf`, meaning every best-trial config failed at least one hard safety constraint during cross-validation.

Despite this, the optimizer discovered materially better holdout accuracy than baseline (68.2% vs 55.0% GSI), demonstrating that the search space contains valuable signal. The bottleneck is the safety constraint architecture, not the optimization process.

**Recommendation:** Do not promote any config to production. Instead, address the structural Cat C problem before re-running the optimizer with relaxed gates.

---

## 2. Run Configuration

| Parameter | Value |
|-----------|-------|
| Seeds | 42, 43, 44, 45, 46 |
| Trials per generation | 200 |
| Generations | 5 |
| Total trials | 5,000 |
| CV folds | 5 |
| Benchmarks | nearmap (235), healthcare (280), openclaw (100), civic (75), agentic (~4,522) |
| Total scenarios | ~5,212 |
| Tunable parameters | 14 (5 semantic thresholds, 4 boundary params, 3 fidelity bands, 5 composite weights) |
| Avg generation time | 36.3 min |

### Baseline Config (Production)

```yaml
st_execute: 0.45
st_clarify: 0.35
st_suggest: 0.25
boundary_violation: 0.70
boundary_margin: 0.05
keyword_boost: 0.15
keyword_embedding_floor: 0.40
fidelity_green: 0.70
fidelity_yellow: 0.60
fidelity_orange: 0.50
weight_purpose: 0.35
weight_scope: 0.20
weight_tool: 0.20
weight_chain: 0.15
weight_boundary_penalty: 0.10
```

Baseline GSI: **0.55**

---

## 3. Results — All Seeds Failed the Four-Gate Ratchet

| Seed | Final Objective | Final GSI | Best Gen Holdout Avg | Cat A Pass? |
|------|----------------|-----------|---------------------|-------------|
| 42 | -inf | 0.604 | 68.2% (gen 2) | No |
| 43 | -inf | 0.539 | 65.8% (gen 1) | No |
| 44 | -inf | 0.576 | 67.3% (gen 0) | No |
| 45 | -inf | 0.556 | 63.4% (gen 0) | No |
| 46 | -inf | 0.595 | 63.2% (gen 3) | No |

**Cross-seed stability: FAIL** (CV = inf due to all -inf objectives)

### Multi-Seed Summary

```json
{
  "objective_mean": -Infinity,
  "gsi_mean": 0.575,
  "gsi_std": 0.020,
  "stability_pass": false
}
```

---

## 4. Best Configuration Found

**Config hash:** `d35794d891f88d75` (appeared in seed 42/gen 2, seed 43/gen 1, seed 44/gen 0)

### 4.1 Holdout Accuracy (Best Instance: Seed 42, Gen 2)

| Benchmark | Baseline (approx) | Optimized | Delta |
|-----------|-------------------|-----------|-------|
| nearmap | 52% | **65.2%** | +13.2pp |
| healthcare | 49% | **69.5%** | +20.5pp |
| openclaw | 45% | **60.7%** | +15.7pp |
| civic | 64% | **68.2%** | +4.2pp |
| agentic | 78% | **77.5%** | -0.5pp |
| **Average** | **57.6%** | **68.2%** | **+10.6pp** |

### 4.2 Config Parameters vs Baseline

| Parameter | Baseline | Optimized | Delta | Interpretation |
|-----------|----------|-----------|-------|----------------|
| st_execute | 0.450 | **0.495** | +0.045 | Requires higher confidence to EXECUTE |
| st_clarify | 0.350 | **0.373** | +0.023 | Slightly tighter CLARIFY threshold |
| st_suggest | 0.250 | **0.317** | +0.067 | Significantly tighter SUGGEST threshold |
| boundary_violation | 0.700 | **0.672** | -0.028 | Triggers boundary violations earlier |
| boundary_margin | 0.050 | **0.067** | +0.017 | Wider boundary margin |
| keyword_boost | 0.150 | **0.202** | +0.052 | Stronger keyword detection (+35%) |
| keyword_embedding_floor | 0.400 | **0.368** | -0.032 | Keywords fire at lower cosine |
| fidelity_green | 0.700 | **0.766** | +0.066 | Tighter green band |
| fidelity_yellow | 0.600 | **0.632** | +0.032 | Tighter yellow band |
| fidelity_orange | 0.500 | **0.511** | +0.011 | Slightly tighter orange |
| **weight_purpose** | **0.350** | **0.499** | **+0.149** | **Major increase (+43%)** |
| weight_scope | 0.200 | **0.181** | -0.019 | Slight decrease |
| weight_tool | 0.200 | **0.172** | -0.028 | Moderate decrease |
| **weight_chain** | **0.150** | **0.067** | **-0.083** | **Major decrease (-55%)** |
| weight_boundary_penalty | 0.100 | **0.081** | -0.019 | Slight decrease |

### 4.3 Key Weight Signals

The optimizer's strongest signal across all seeds:

1. **weight_purpose UP to ~0.50** (from 0.35): Purpose alignment should dominate the composite. The optimizer consistently converges here across seeds.
2. **weight_chain DOWN to ~0.07** (from 0.15): Chain continuity is nearly irrelevant for single-step benchmark evaluation. Since benchmarks evaluate individual tool calls (no multi-step chains), chain_sci is always 0.0. The optimizer correctly de-weights this dead dimension.
3. **keyword_boost UP to ~0.20** (from 0.15): Keyword matching is a valuable supplement to embedding similarity. The optimizer wants more aggressive keyword boosting.
4. **All semantic thresholds shift UP**: The current baseline is slightly too permissive. The optimizer demands higher confidence before execution.

---

## 5. Config Convergence

Only **9 unique configs** were discovered across 25 generations. The optimizer converges rapidly to a small set of operating points:

| Config Hash | Appearances | Avg Holdout | GSI |
|-------------|-------------|-------------|-----|
| **d35794d891f88d75** | **3** | **67.1%** | **0.576** |
| c067113952a5deb7 | 4 | 65.5% | 0.557 |
| 80613b56de1e0c9e | 5 | 57.5% | 0.595 |
| 2aa7e7bf00847a17 | 2 | 61.5% | 0.594 |
| 3f19a85da80c577d | 1 | 63.6% | 0.604 |
| 8a1e492d0a45ba9e | 4 | 54.2% | 0.581 |
| e4b857ccff6347b0 | 2 | 56.0% | 0.539 |
| 8b2f2adb7bee2092 | 3 | 49.1% | 0.563 |
| a1469bff22987f2a | 1 | 60.3% | 0.543 |

The best config (`d35794d891f88d75`) appearing in 3 seeds with consistent holdout accuracy confirms convergent search — the optimizer is not overfitting to particular data splits.

---

## 6. Four-Gate Ratchet Failure Analysis

### Gate Results

| Gate | Requirement | Pass Rate | Status |
|------|-------------|-----------|--------|
| Gate 1 | Zero Cat A regressions vs baseline | 6/25 (24%) | Partially passable |
| Gate 2 | Holdout Cat A = 100% | **0/25 (0%)** | **Universal blocker** |
| Gate 3 | Objective > -inf (all CV constraints) | **0/25 (0%)** | **Universal blocker** |
| Gate 4 | GDD within bounds | Not evaluated | Blocked by Gates 2-3 |

### Why Gate 3 Universally Fails

The scalarized objective function requires Cat A detection >= 95% AND Cat E detection >= 85% across ALL benchmarks simultaneously during 5-fold CV. No trial out of 5,000 could satisfy this joint constraint. The embedding architecture (MiniLM-L6-v2, 384-dim) does not produce sufficient separation between Cat A violations and Cat C legitimate requests — the cosine similarity distributions overlap in the 0.35-0.50 range.

### Why Gate 2 Universally Fails

This gate checks *absolute* Cat A detection = 100% on the holdout set. The baseline itself does not achieve 100% Cat A holdout, so even configs matching baseline Cat A detection still fail.

### Recommendation for Gate Architecture

1. **Relax Gate 2** from 100% to 98% (matching the CV constraint tolerance)
2. **Relax Gate 3 CV constraints** from 95%/85% to 90%/80% to allow finite objectives
3. Consider **per-benchmark gates** rather than the current joint-ALL-benchmarks requirement

---

## 7. Regression Topology — Best Config (d35794d891f88d75)

| Metric | Value |
|--------|-------|
| Total regressions | 92 |
| Total improvements | 35 |
| Net change | -57 |
| Less restrictive (looser) | 67 |
| More restrictive (tighter) | 25 |

### Regressions by Category

| Category | Regressions | % of Total |
|----------|-------------|------------|
| **Cat C** | **45** | **48.9%** |
| Cat FP | 18 | 19.6% |
| Cat B | 15 | 16.3% |
| Cat D | 9 | 9.8% |
| Cat H | 4 | 4.3% |
| Cat A | 1 | 1.1% |

**Cat C regressions are the dominant source of regression.** These are legitimate on-topic scenarios that the optimizer pushes from EXECUTE to CLARIFY/SUGGEST as it tightens thresholds to protect Cat A. This is the fundamental tension: improving safety detection regresses capability.

---

## 8. Per-Benchmark Performance Across All Seeds

### Best Holdout Per Benchmark (Across All Generations)

| Benchmark | Best Holdout | Best Seed/Gen | Config Hash |
|-----------|-------------|---------------|-------------|
| nearmap | 66.7% | 42/gen 3 | c067113952a5deb7 |
| healthcare | 74.4% | 42/gen 0 | 3f19a85da80c577d |
| openclaw | 67.9% | 42/gen 0, 44/gen 0 | d35794d891f88d75 |
| civic | 72.7% | 46/gen 4 | a1469bff22987f2a |
| agentic | 79.7% | 43/gen 3 | 80613b56de1e0c9e |

### Variance by Benchmark

| Benchmark | Mean Holdout | Std | CV | n scenarios |
|-----------|-------------|-----|-----|-------------|
| agentic | 78.1% | 1.0% | 1.3% | ~4,522 |
| civic | 56.5% | 11.3% | 20.0% | 75 |
| nearmap | 54.1% | 6.8% | 12.5% | 235 |
| healthcare | 56.4% | 11.6% | 20.5% | 280 |
| openclaw | 49.7% | 9.5% | 19.1% | 100 |

Agentic (largest benchmark) is the most stable. Smaller benchmarks (civic, openclaw) show high variance (20% CV) due to sample size sensitivity.

---

## 9. Structural Findings

### 9.1 The Cat C Problem

Cat C scenarios (legitimate on-topic requests) account for 49% of all regressions. The root cause:

1. **chain_sci always = 0.0** in benchmarks (single tool calls, no multi-step chains). With weight 0.15, 15% of composite weight is dead. Maximum possible composite is 0.85 even with perfect scores elsewhere.
2. **Purpose/scope centroids are too broad for OpenClaw.** A "general coding assistant" centroid produces cosine similarities of 0.33-0.50 for legitimate requests — barely above off-topic noise.
3. **Boundary penalty always subtracts.** Even legitimate requests have non-zero boundary similarity (mean 0.40), costing ~0.04 on composite.
4. **MiniLM-L6-v2 compressed similarity ranges.** The model's 384 dimensions produce narrow cosine distributions. Purpose gap between Cat C (0.495) and Cat A (0.439) is only 0.056.

### 9.2 The Optimizer is Working Correctly

The optimizer successfully identified the most promising parameter regions across independent seeds with consistent holdout accuracy. Config `d35794d891f88d75` appeared in 3 seeds with consistent performance. The weight signals (purpose UP, chain DOWN) are architecturally sound.

### 9.3 The Safety Constraints are Too Strict for the Current Architecture

The joint requirement of 95% Cat A + 85% Cat E + 100% holdout Cat A across all benchmarks simultaneously is unreachable with cosine-similarity-based governance on MiniLM-L6-v2 embeddings. This is not a tuning problem — it's an architectural limitation that tuning cannot overcome.

---

## 10. Promotion Decision

**Do NOT promote any weights to production.** Rationale:

1. No config passed the four-gate ratchet (Research Governance Charter TELOS-RGC-001 prohibits deployment without ratchet passage)
2. The best config regresses 92 scenarios while improving only 35 — net negative
3. Cat C regressions (45) outnumber Cat A improvements, meaning capability degrades without proportional safety improvement
4. The fundamental problem (Cat C at 21% on OpenClaw) requires architectural intervention, not threshold tuning

### What to Do Instead

1. **Fix Cat C architecturally** (see Section 11) before re-running the optimizer
2. **Adopt weight signals as informational guidance:** weight_purpose ~0.50, weight_chain ~0.07 are directionally correct
3. **Relax ratchet gates** per Section 6 recommendations before re-running
4. **Re-run optimizer** after Cat C fix with relaxed gates — expect convergent behavior

---

## 11. Recommended Next Steps — Cat C Improvement Path

The optimizer data confirms Cat C is the dominant bottleneck. Improvement requires changes at the **scoring architecture level**, not the threshold level. Three approaches ranked by expected impact:

### 11.1 Fix Dead Weight (chain_sci = 0.0)

**Impact:** Medium | **Effort:** Low

In single-step evaluation (benchmarks), chain_sci is always 0.0, wasting 15% of composite weight. Two options:
- **Option A:** Set weight_chain = 0.0 for single-step evaluation, redistribute to other dimensions
- **Option B:** Compute a pseudo-chain signal from tool registration context (prior tool history in session)

The optimizer already converges to weight_chain ~0.07. Formalizing this fixes the dead-weight problem.

### 11.2 Category-Aware Scoring / SetFit L1.5 for Cat C

**Impact:** High | **Effort:** Medium

SetFit already achieves AUC 0.990 for boundary detection (Cat A/E). Train a Cat C classifier that predicts "legitimate vs suspicious" as a parallel signal. Use it to boost composite scores for high-confidence legitimate requests.

### 11.3 Domain-Adaptive Purpose Centroids

**Impact:** High | **Effort:** Medium

The OpenClaw PA purpose is too broad ("autonomous code assistance"). Instead of a single centroid:
- Build tool-group-specific purpose sub-centroids (e.g., "file system operations for code review" vs "runtime execution for testing")
- Score against the best-matching sub-centroid rather than the global centroid
- This is partially implemented in `agentic_pa.py` (sub-centroid clustering) but not wired into scoring

---

## GenAI Disclosure

This analysis was generated with assistance from Claude (Anthropic). All data was produced by deterministic optimizer runs. Interpretation and recommendations reflect the author's technical judgment informed by the optimizer output.

## COI Disclosure

The author has a commercial interest in TELOS AI Labs Inc. All benchmarks are self-authored. No external funding.
