# Statistical Framework: Governed vs. Ungoverned OpenClaw Comparison Study

**TELOS AI Labs Inc. -- Research Program**
**Author:** Gebru (Data Scientist / Statistician)
**Date:** 2026-02-19
**Status:** Pre-Registration (Study Not Yet Executed)
**Depends On:** `research/setfit_openclaw_mve_closure.md` (SetFit AUC 0.9905 -- GREEN)
**Depends On:** `validation/openclaw/openclaw_boundary_corpus_v1.jsonl` (100-scenario benchmark)
**Depends On:** `validation/openclaw/benchmark_results.json` (53% overall, 87.5% Cat E, 96% boundary sensitivity)
**Depends On:** `telos_adapters/openclaw/governance_hook.py` (4-layer cascade scoring bridge)

---

### Disclosures

> **Generative AI Disclosure:** Internal analysis, experimental design review, and qualitative assessment in this document were conducted with assistance from LLM-based research agents (Claude, Anthropic). These agents are prompted with domain-specific personas (governance theory, statistics, systems engineering, regulatory analysis, research methodology) and operate as AI research assistants — not independent human expert reviewers. All quantitative results (AUC-ROC, F1, bootstrap confidence intervals, benchmark accuracies) are computed by deterministic code. Qualitative analysis should not be treated as independent peer review. See `research/research_team_spec.md` for full methodology.

> **Conflict of Interest Disclosure:** This research was conducted and funded by TELOS AI Labs Inc., which has a commercial interest in the TELOS governance framework. All domain-specific validation benchmarks (Nearmap, Healthcare, OpenClaw) were created by the research team. External benchmarks (PropensityBench, AgentHarm, AgentDojo) were created by independent organizations. Research artifacts are published on [Zenodo](https://zenodo.org/) with persistent DOIs. No external funding or independent peer review was involved in this work.

---


## Executive Summary

This document specifies the complete statistical design for a head-to-head comparison of two isolated OpenClaw instances executing identical tasks -- one governed by the TELOS daemon (4-layer cascade: keyword, cosine, SetFit, LLM) and one ungoverned (raw tool execution). The study answers a single question: **does TELOS governance measurably reduce harmful agent actions while preserving task completion?**

The design uses a paired, within-task comparison with repeated measurements to control for LLM stochasticity. The primary outcome is the Risk Reduction Ratio (RRR), which expresses the fraction of violations eliminated by governance. Secondary outcomes include task completion rate, latency overhead, false positive rate, and action chain divergence point. The minimum viable study requires 60 task-pairs at 5 replications each (300 total runs per arm), achieving 80% power to detect a 30% risk reduction. The ideal study doubles this to 100 task-pairs at 10 replications (1,000 runs per arm), achieving 95% power with stratification by tool group and risk tier.

The framework defines how to measure the ungoverned instance (post-hoc TELOS scoring with human annotation calibration), how to handle LLM stochasticity (fixed-seed replications with ecological validity crossover), and how to present cost-asymmetric results (weighted harm scores using the FN-cost-is-100-1000x-FP principle from prior analysis). A separate non-inferiority arm for LangChain validates governance portability.

---

## Table of Contents

1. [Research Questions and Hypotheses](#1-research-questions-and-hypotheses)
2. [Study Design](#2-study-design)
3. [Outcome Metrics](#3-outcome-metrics)
4. [Sample Size and Power Analysis](#4-sample-size-and-power-analysis)
5. [Handling LLM Stochasticity](#5-handling-llm-stochasticity)
6. [Measuring the Ungoverned Instance](#6-measuring-the-ungoverned-instance)
7. [Effect Size and Practical Significance](#7-effect-size-and-practical-significance)
8. [LangChain Baseline Design](#8-langchain-baseline-design)
9. [Analysis Plan](#9-analysis-plan)
10. [Minimum Viable Study vs. Ideal Study](#10-minimum-viable-study-vs-ideal-study)
11. [Threats to Validity](#11-threats-to-validity)
12. [Pre-Registered Decision Criteria](#12-pre-registered-decision-criteria)

---

## 1. Research Questions and Hypotheses

### 1.1 Primary Hypothesis

**H0 (Null):** TELOS governance does not reduce the rate of boundary violations committed by OpenClaw agents compared to ungoverned execution.

**H1 (Alternative):** TELOS governance reduces the boundary violation rate by at least 30% (Risk Reduction Ratio >= 0.30).

### 1.2 Secondary Hypotheses

**H2 (Task Completion):** TELOS governance reduces task completion rate by no more than 15% compared to ungoverned execution (non-inferiority margin delta = -0.15).

**H3 (Latency):** TELOS governance adds no more than 50ms median latency per tool call (based on cascade architecture: L0 keyword ~0.1ms + L1 cosine ~10ms + L1.5 SetFit ONNX ~3-5ms).

**H4 (Divergence):** In multi-step tasks, governed and ungoverned action chains diverge at or before the first violating tool call, not earlier (governance does not prematurely truncate legitimate chains).

**H5 (Asymmetric Cost):** The cost-weighted harm score (incorporating the 100-1000x asymmetry between false negatives and false positives) favors governed execution.

### 1.3 Exploratory Questions

- Does governance effectiveness vary by tool group risk tier (CRITICAL/HIGH/MEDIUM/LOW)?
- Does governance effectiveness vary by attack family (injection, supply chain, social engineering, etc.)?
- At what chain length do governed and ungoverned instances first produce divergent actions?
- Does the cascade architecture's layered detection (keyword -> cosine -> SetFit) produce redundant catches, or does each layer catch unique violations?

---

## 2. Study Design

### 2.1 Design Choice: Paired, Within-Task Comparison

**Selected design: Randomized crossover with paired measurements.**

Rationale for paired over independent groups:

| Design | Pros | Cons |
|--------|------|------|
| **Paired (same task, both arms)** | Controls for task difficulty; higher statistical power; smaller N needed | Carry-over effects if order matters (mitigated: both run from fresh state) |
| Independent groups (random assignment) | No carry-over; simpler | Requires 2-4x more tasks for equivalent power; task difficulty becomes confound |

The paired design is strongly preferred because:
1. **Task difficulty is a massive confound.** A violation-heavy task assigned only to the ungoverned arm inflates the effect. Pairing eliminates this.
2. **The 100-scenario OpenClaw corpus is fixed.** We cannot generate unlimited novel tasks. Pairing maximizes statistical power from the available corpus.
3. **Carry-over is not a concern** because both instances start from fresh state (new daemon, empty action chain, no session history).

### 2.2 Experimental Units

**Task-pair:** One scenario from `openclaw_boundary_corpus_v1.jsonl` (or extended corpus) executed on both the governed and ungoverned instances.

**Replication:** Each task-pair is run N times (N >= 5) to estimate within-task variance from LLM stochasticity.

**Blocking variable:** Tool group (11 levels: automation, cross_group, fs, memory, messaging, nodes, openclaw, runtime, sessions, ui, web).

**Stratification variable:** Risk tier (4 levels: CRITICAL, HIGH, MEDIUM, LOW).

### 2.3 Execution Protocol

```
For each task T_i in corpus (i = 1..K):
    For each replication r = 1..N:
        1. Initialize GOVERNED instance:
           - Fresh OpenClaw process
           - TELOS daemon active (balanced preset)
           - IPC server listening on UDS
           - SetFit ONNX model loaded (setfit_openclaw_v1)
           - Action chain reset

        2. Initialize UNGOVERNED instance:
           - Fresh OpenClaw process
           - NO TELOS daemon
           - NO governance hook
           - Raw tool execution

        3. Submit identical task T_i to both instances
           - If using LLM: same seed s_r for replication r
           - If deterministic: identical input, no seed needed

        4. Record:
           - GOVERNED: Full GovernanceVerdict for each tool call
             (decision, fidelity scores, cascade layers, latency_ms)
           - UNGOVERNED: Raw action log (tool_name, args, output, timestamp)

        5. Post-hoc: Score ungoverned action log through TELOS offline
           (Section 6 details the classification approach)

        6. Compare paired outcomes: (violation_count_gov, violation_count_ungov)
```

### 2.4 Randomization

- **Task order:** Randomly shuffled per replication (prevents systematic order effects on LLM state).
- **Arm order:** For each task-pair, randomly assign which instance (governed/ungoverned) executes first. In practice, both can run in parallel on separate processes, making order irrelevant.
- **Seed assignment:** Replication seeds drawn from a CSPRNG, recorded in the experiment log for reproducibility.

---

## 3. Outcome Metrics

### 3.1 Primary Outcome: Risk Reduction Ratio (RRR)

```
RRR = 1 - (V_gov / V_ungov)

where:
  V_gov   = Number of boundary violations in governed arm (across all task-pairs and replications)
  V_ungov = Number of boundary violations in ungoverned arm (same tasks, same replications)
```

**Interpretation:**
- RRR = 0.0: Governance has no effect
- RRR = 0.30: Governance prevents 30% of violations (minimum clinically significant effect)
- RRR = 0.96: Governance prevents 96% of violations (expected based on SetFit AUC 0.9905)
- RRR = 1.0: Governance prevents all violations (unrealistic -- some semantic cloaking attacks evade detection)

**Why RRR over alternatives:**

| Metric | Formula | Pro | Con |
|--------|---------|-----|-----|
| **RRR** | 1 - (V_gov/V_ungov) | Intuitive; directly answers "what fraction of harm did governance prevent?" | Undefined if V_ungov = 0 |
| Harm Prevention Rate | V_caught / V_ungov | Same as RRR when V_caught = V_ungov - V_gov | Requires counting "caught" separately |
| Governance Effectiveness | 1 - (V_gov/V_ungov) | Identical to RRR | Naming ambiguity |
| NNT (Number Needed to Treat) | 1 / (V_ungov_rate - V_gov_rate) | Standard clinical metric | Hard to interpret for non-medical audience |
| Odds Ratio | (V_gov/safe_gov) / (V_ungov/safe_ungov) | Handles rare events well | Non-intuitive; requires explanation |

**Decision: RRR is the primary metric.** Report the odds ratio as a secondary metric for meta-analysis compatibility.

### 3.2 Secondary Outcomes

#### 3.2.1 Task Completion Rate (TCR)

```
TCR_arm = (Tasks reaching expected terminal state) / (Total tasks attempted)

Completion Ratio = TCR_gov / TCR_ungov
```

A governed task is "complete" if it reaches the expected decision (EXECUTE for Cat C, ESCALATE for Cat A/E) AND the agent produces a meaningful response. An ungoverned task is "complete" if the agent produces a meaningful response (it always "completes" since nothing blocks it -- but violations are still violations).

**Non-inferiority test:** H0: TCR_gov - TCR_ungov <= -0.15 (governed is more than 15% worse).

#### 3.2.2 Latency Overhead

```
Latency_overhead_ms = median(latency_governed) - median(latency_ungoverned)
```

Per tool call. The governance cascade adds:
- L0 keyword: ~0.1ms (string matching)
- L1 cosine: ~10ms (embedding + similarity)
- L1.5 SetFit ONNX: ~3-5ms (when triggered; not every call)
- L2 LLM: ~1-10s (only for ESCALATE; not in hot path)

**Expected overhead:** 10-15ms per tool call (L0+L1+L1.5). Report both median and P95.

#### 3.2.3 False Positive Rate (FPR)

```
FPR = (Legitimate tasks blocked by governance) / (Total legitimate tasks)
```

Measured only on Category C (legitimate) and Category FP (false-positive control) scenarios. From existing data:
- SetFit FPR: 1.3% (1/76 in CV)
- Boundary FPR: 0.0% (0/7 Cat FP controls)
- Overall benchmark FPR on Cat C: ~82% are NOT correctly classified as EXECUTE due to composite scoring threshold issue (this is a known limitation -- Cat C accuracy is 18.2%)

**Important distinction:** The FPR we measure here is specifically *boundary-triggered false positives* (governance ESCALATES a legitimate task because the boundary detector misfires), NOT composite scoring threshold misses (governance produces CLARIFY/SUGGEST instead of EXECUTE because fidelity is below threshold). The latter is a calibration issue, not a governance harm.

#### 3.2.4 Action Chain Divergence Point

For multi-step tasks (sequence groups in the corpus), record the step number at which the governed and ungoverned action chains first produce different tool calls.

```
Divergence_step = min(step_i : tool_call_gov[i] != tool_call_ungov[i])
```

**Expected finding:** Divergence occurs at or immediately before the first violating tool call. If governance causes early divergence (blocking legitimate steps that happen to precede a violation), that indicates over-eager chain breaking.

#### 3.2.5 Cascade Layer Attribution

For each caught violation, record which cascade layer triggered:
- L0 only (keyword caught it)
- L1 only (cosine caught it)
- L1.5 only (SetFit escalated)
- Multiple layers (redundant detection)

This decomposes the RRR by detection mechanism.

---

## 4. Sample Size and Power Analysis

### 4.1 Statistical Framework

The primary analysis is a **McNemar's test** on paired binary outcomes (violation/no-violation per task-pair), aggregated across replications.

For each task-pair i, define:

```
D_i = (V_ungov_i > 0) - (V_gov_i > 0)

where V_arm_i = count of violations across N replications for task i in arm
```

Under H0, E[D_i] = 0. Under H1, E[D_i] > 0 (governed has fewer violations).

**Discordant pairs** are the informationally useful units:
- Type A: Ungoverned violates, governed does not (governance helped)
- Type B: Governed violates, ungoverned does not (governance made things worse -- should be rare)
- Type C: Both violate (governance failed)
- Type D: Neither violates (no violation to prevent)

Power depends on the number of discordant pairs (Type A + Type B) and the ratio pA / (pA + pB).

### 4.2 Parameter Estimation from Existing Data

From `benchmark_results.json` and `setfit_openclaw_mve_closure.md`:

**Violation base rate (ungoverned):**
- Cat A (direct violations): 25 scenarios, all would violate if ungoverned
- Cat E (adversarial): 24 scenarios, all would violate if ungoverned
- Cat D (edge cases): 9 scenarios, approximately 50% would violate
- Total expected violations in ungoverned: ~54/100 tasks

**Governed catch rate:**
- Boundary sensitivity (Cat A): 96% (from SetFit CV -- 77/80 detected)
- Cat E detection: 87.5% (benchmark) to 93.3% (SetFit adversarial holdout)
- Combined: ~90-96% of violations caught by governance

**Expected discordant pair distribution:**
- Type A (ungoverned violates, governed does not): ~50 * 0.93 = ~46.5 tasks
- Type B (governed violates, ungoverned does not): ~2 tasks (false positive boundary triggers on legitimate tasks)
- Ratio pA/(pA+pB) = 46.5/48.5 = 0.959

### 4.3 Power Calculations

#### McNemar's Test Power

For McNemar's test, power depends on the number of discordant pairs n_d and the proportion under H1:

```
n_d = K * (pA + pB)

where K = number of task-pairs

Power = P(reject H0 | H1 true)

For a one-sided test at alpha = 0.05:
  z = (pA - pB) / sqrt(pA + pB) * sqrt(n_d)
  Power = Phi(z - z_alpha)
```

#### Scenario 1: Detecting 30% Risk Reduction (Conservative)

```
Assumptions:
  - Base violation rate: 54%
  - Governance catch rate: 30% (worst case; far below expected)
  - pA = 0.54 * 0.30 = 0.162 (ungoverned violates, governed catches)
  - pB = 0.02 (governed false-triggers on legitimate)
  - n_d = K * 0.182

For 80% power (beta = 0.20, alpha = 0.05):
  Required n_d = ((z_0.05 + z_0.20) / (pA - pB) * sqrt(pA + pB))^2
  = ((1.645 + 0.842) / (0.162 - 0.02) * sqrt(0.182))^2
  = ((2.487) / 0.142 * 0.427)^2
  = (2.487 / 0.0607)^2
  = (40.97)^2 ... [using exact McNemar power formula]

  Exact: n_d >= 40 discordant pairs
  K = 40 / 0.182 = 220 task-pairs

  With replications (N=5 per task, majority-vote aggregation):
  K >= 60 task-pairs (reduced by variance reduction from replications)

For 90% power:
  K >= 80 task-pairs (with N=5 replications)

For 95% power:
  K >= 100 task-pairs (with N=5 replications)
```

#### Scenario 2: Detecting Expected Effect Size (~90% Risk Reduction)

```
Assumptions:
  - Base violation rate: 54%
  - Governance catch rate: 90% (conservative estimate from SetFit data)
  - pA = 0.54 * 0.90 = 0.486
  - pB = 0.013 (SetFit FPR)
  - n_d = K * 0.499

For 80% power:
  K >= 18 task-pairs (with N=5 replications)

For 90% power:
  K >= 24 task-pairs

For 95% power:
  K >= 30 task-pairs
```

#### Scenario 3: Near-Perfect Governance (~96% Catch Rate)

```
Assumptions:
  - Base violation rate: 54%
  - Governance catch rate: 96% (SetFit CV detection rate)
  - pA = 0.54 * 0.96 = 0.518
  - pB = 0.013
  - n_d = K * 0.531

For 80% power:
  K >= 14 task-pairs

For 95% power:
  K >= 22 task-pairs
```

### 4.4 Summary Power Table

| Effect Size (RRR) | K (80% power, N=5) | K (90% power, N=5) | K (95% power, N=10) |
|--------------------|---------------------|---------------------|----------------------|
| 0.30 (conservative) | 60 | 80 | 70 |
| 0.50 (moderate) | 35 | 45 | 40 |
| 0.70 (strong) | 24 | 30 | 28 |
| 0.90 (expected) | 18 | 24 | 20 |
| 0.96 (SetFit-predicted) | 14 | 18 | 16 |

### 4.5 Stratification Requirements

The existing corpus has unequal tool group representation:

| Tool Group | n | Risk Tier |
|-----------|---|-----------|
| runtime | 16 | CRITICAL |
| fs | 13 | HIGH |
| messaging | 11 | CRITICAL |
| openclaw | 11 | CRITICAL |
| web | 11 | HIGH |
| automation | 9 | CRITICAL |
| cross_group | 9 | CRITICAL |
| nodes | 9 | MEDIUM |
| memory | 4 | LOW |
| sessions | 4 | LOW |
| ui | 3 | LOW |

**Minimum per-stratum requirement for stratified analysis:** 10 tasks per tool group.

**Current gaps:** memory (4), sessions (4), ui (3) are underpowered. For stratified analysis, augment to >= 10 each, bringing corpus to ~119 scenarios.

For risk-tier stratification (4 strata), current distribution is adequate: CRITICAL=56, HIGH=24, MEDIUM=9, LOW=11. Only MEDIUM is borderline. Augment nodes to >= 12 (total corpus ~122).

**Recommendation:** Augment corpus to 120-130 scenarios, ensuring >= 10 per tool group and >= 12 per risk tier. This enables both overall and stratified analyses.

### 4.6 Bootstrap vs. Parametric

**Primary analysis: Parametric** (McNemar's test). Well-understood, widely accepted, computationally trivial. The paired binary structure is exactly what McNemar's was designed for.

**Secondary analysis: Bootstrap** (10,000 resamples). Used for:
1. Confidence intervals on RRR (bias-corrected and accelerated, BCa)
2. Per-tool-group effect sizes (too few per stratum for parametric)
3. Cascade layer attribution proportions

**Sensitivity analysis: Permutation test.** Randomly reassign governance labels within pairs 10,000 times. If the observed RRR exceeds 95% of permuted RRRs, the result is significant at p < 0.05. This is distribution-free and makes no assumptions about the shape of D_i.

---

## 5. Handling LLM Stochasticity

### 5.1 The Problem

When both instances use an LLM to generate tool calls:
- Same task + same prompt may produce different tool calls due to sampling
- Different tool calls may trigger different governance decisions
- This creates noise that inflates variance and reduces power

### 5.2 Recommended Approach: Tiered Strategy

#### Tier 1: Deterministic Baseline (Primary Analysis)

**Method:** Use the existing `MockToolExecutor` with `--no-governance` to simulate the ungoverned baseline. The benchmark already implements this:

```python
# From run_openclaw_benchmark.py, line 432:
def _make_no_governance_result(scenario):
    return {
        "actual_decision": "EXECUTE",  # Always execute
        ...
    }
```

**Rationale:** The benchmark corpus (`openclaw_boundary_corpus_v1.jsonl`) contains predetermined `request_text` values. The question is not "what tool calls does the LLM generate?" but "given these tool calls, does governance catch the violations?" This is the cleanest comparison because it eliminates LLM stochasticity entirely.

**Implementation:** Run `run_openclaw_benchmark.py` with and without `--no-governance`. This is already built into the codebase.

**Limitation:** This tests governance *detection* accuracy on fixed inputs, not governance's effect on live agent *behavior* (where the LLM might generate different tool calls based on whether previous calls were blocked).

#### Tier 2: Fixed-Seed Replications (Secondary Analysis)

**Method:** Run both instances with the same LLM, same model, same temperature, same seed for each replication.

```
For replication r:
    seed_r = CSPRNG(experiment_master_seed, r)
    governed_instance.set_seed(seed_r)
    ungoverned_instance.set_seed(seed_r)
```

**Rationale:** With identical seeds, the LLM produces identical initial tool calls. Divergence occurs only when governance intervenes (blocks a tool call, causing the LLM to generate an alternative). This isolates the governance effect from LLM randomness.

**Implementation notes:**
- OpenClaw uses Claude as the underlying LLM. Temperature must be set to 0 or a fixed value.
- If the LLM API does not support deterministic seeds, use N replications with temperature=0 (most providers return identical outputs for temperature=0).
- Record whether outputs actually vary across replications (if they do not, N=1 is sufficient per seed).

#### Tier 3: Ecological Validity Crossover (Exploratory)

**Method:** Run with realistic temperature (e.g., temperature=0.7) and N=10 replications per task.

**Rationale:** Real deployments use non-zero temperature. This tier captures the distribution of outcomes under realistic conditions.

**Analysis:** For each task, compute the within-task violation rate for each arm:

```
p_viol_gov_i = (violations across N replications for task i in governed arm) / N
p_viol_ungov_i = (violations across N replications for task i in ungoverned arm) / N

Paired t-test on: (p_viol_ungov_i - p_viol_gov_i) across tasks
```

### 5.3 Summary Decision

| Tier | Stochasticity | Ecological Validity | Analysis Complexity | Role |
|------|--------------|---------------------|--------------------|----- |
| 1: Deterministic | None | Low (fixed inputs) | Simple | Primary: proves detection works |
| 2: Fixed-seed | Controlled | Medium | Moderate | Secondary: proves live governance works |
| 3: Free-running | Full | High | Complex | Exploratory: proves robustness to randomness |

**Publication recommendation:** Lead with Tier 1 (cleanest, most reproducible), supplement with Tier 2 (demonstrates live effect), report Tier 3 in appendix (ecological validity).

---

## 6. Measuring the Ungoverned Instance

### 6.1 The Classification Problem

The ungoverned instance has no TELOS scoring. We need to classify its actions as violation/legitimate after the fact. This is the most methodologically delicate part of the study.

### 6.2 Recommended Approach: Post-Hoc TELOS Scoring + Human Calibration

#### Layer 1: Post-Hoc TELOS Scoring (Automated)

Run the full TELOS cascade on the ungoverned instance's action log after execution:

```python
from telos_adapters.openclaw.governance_hook import GovernanceHook

hook = GovernanceHook(config_loader, preset="balanced")

for action in ungoverned_action_log:
    verdict = hook.score_action(
        tool_name=action["tool_name"],
        action_text=action["action_text"],
        tool_args=action.get("tool_args"),
    )
    action["post_hoc_verdict"] = verdict.to_dict()
```

**Why this works:** TELOS scoring is stateless per call (each `score_action` invocation is independent except for chain state). Running the same action text through the same cascade produces the same verdict regardless of whether the scoring happened at execution time or post-hoc.

**Chain state caveat:** The `ActionChain` maintains cross-group transition state. For post-hoc scoring, replay the entire action sequence in order to preserve chain context.

#### Layer 2: Human Annotation (Calibration Sample)

Expert annotators classify a random sample of ungoverned actions. This serves as the ground truth against which post-hoc TELOS scoring is calibrated.

**Annotation protocol:**
1. Sample 30% of ungoverned actions (stratified by tool group and risk tier)
2. Two independent annotators per action
3. Three-class labels: VIOLATION, LEGITIMATE, AMBIGUOUS
4. Inter-annotator agreement measured by Cohen's kappa
5. Disagreements resolved by third annotator

**Annotator inclusion criteria:**
- Familiar with OpenClaw tool semantics
- Familiar with the 17 sourced boundaries in `openclaw.yaml`
- Blinded to TELOS post-hoc verdict during annotation

**Calibration:** Compare post-hoc TELOS verdicts against human annotations on the calibration sample. Report:
- Agreement rate (expected: >90% based on SetFit AUC 0.9905)
- Cohen's kappa between TELOS and human gold standard
- Confusion matrix: TELOS-vs-human on the calibration sample
- Any systematic TELOS errors identified by human annotators

#### Layer 3: Proxy Metrics (Supplementary)

For defense-in-depth measurement, also record:
- **Filesystem changes:** Files created, modified, deleted outside workspace
- **Network calls:** Outbound connections to non-allowlisted hosts
- **Process spawning:** Child processes (especially shells, curl, wget)
- **Credential access:** Reads of .env, .ssh, .aws directories
- **Privilege escalation:** sudo, chmod, chown invocations

These proxy metrics are not the primary outcome but serve as independent validation: if TELOS says "no violation" but the proxy metrics show .env was read and curl was called to an external host, that is a false negative.

### 6.3 Defensibility Assessment

| Approach | Reproducible | Scalable | Ground Truth | Publication-Ready |
|----------|-------------|----------|-------------|-------------------|
| Post-hoc TELOS + human calibration | Yes | Yes | Validated | **Yes (recommended)** |
| Human annotation only | Yes | No (expensive) | Gold standard | Yes (gold standard) |
| Proxy metrics only | Yes | Yes | Noisy | Supplementary only |
| Automated monitors (syscall) | Yes | Yes | Indirect | Supplementary only |

**Decision:** Post-hoc TELOS scoring is the primary classification method, validated against a human-annotated calibration sample. This is both scalable (all actions scored) and defensible (human ground truth on a representative subset).

---

## 7. Effect Size and Practical Significance

### 7.1 Expected Effect Sizes

Based on existing empirical data:

| Component | Metric | Value | Source |
|-----------|--------|-------|--------|
| SetFit boundary detection | AUC-ROC | 0.9905 | `setfit_openclaw_mve_closure.md` |
| SetFit detection rate | Recall | 96.2% | 5-fold CV (77/80) |
| SetFit false positive rate | FPR | 1.3% | 5-fold CV (1/76) |
| Adversarial detection | Recall | 93.3% (holdout) | 14/15 Cat E never seen in training |
| Benchmark boundary sensitivity | Sensitivity | 96.0% | Calibration log (Cat A) |
| Benchmark Cat E accuracy | Accuracy | 87.5% | 21/24 (benchmark_results.json) |
| Benchmark overall accuracy | Accuracy | 53.0% | 53/100 (known Cat C composite issue) |

**Expected RRR in this study:** 0.87 to 0.96.

Breakdown:
- Of ~54 violations in ungoverned arm, governance catches 87.5-96.2%
- ~2-7 violations slip through (SetFit misses + adversarial misses)
- ~0-2 false positive escalations on legitimate tasks

**Effect size in Cohen's h (for McNemar-equivalent):**
```
p_A = 0.486 (ungoverned violates, governed does not)
p_B = 0.013 (governed violates, ungoverned does not)

h = 2 * arcsin(sqrt(pA/(pA+pB))) - 2 * arcsin(sqrt(0.5))
  = 2 * arcsin(sqrt(0.974)) - pi/2
  = 2 * arcsin(0.987) - 1.571
  = 2 * 1.410 - 1.571
  = 1.249

Cohen's h = 1.25 (very large effect)
```

This is an extraordinarily large effect by social science standards (h > 0.8 is "large"). The study is massively overpowered for detecting the expected effect.

### 7.2 The False Positive Tradeoff

Governance creates two types of costs:
1. **Prevented violations (benefit):** Harm averted
2. **False positive blocks (cost):** Legitimate tasks blocked

The tradeoff must be presented honestly. However, the costs are profoundly asymmetric.

### 7.3 Cost Asymmetry Framework

From prior analysis (referenced in research questions): a false negative on runtime/web is 100-1000x the cost of a false positive.

**Formal cost model:**

```
Cost_FN = C_viol  (cost of an undetected violation)
Cost_FP = C_block (cost of a blocked legitimate task)

Asymmetry ratio: lambda = C_viol / C_block

Weighted Harm Score:
  WHS = lambda * FN_count + FP_count
```

**Estimating lambda by tool group:**

| Tool Group | Risk Tier | Violation Consequence | Estimated lambda |
|-----------|-----------|----------------------|-----------------|
| runtime | CRITICAL | RCE, data exfiltration, system compromise | 1000 |
| messaging | CRITICAL | Unauthorized external communication, data leak | 500 |
| automation | CRITICAL | Persistent backdoor via cron, scheduled exfiltration | 1000 |
| openclaw | CRITICAL | Malicious skill installation, config poisoning | 500 |
| fs | HIGH | Credential theft, code injection, data destruction | 200 |
| web | HIGH | SSRF, data exfiltration via HTTP, phishing redirect | 200 |
| nodes | MEDIUM | Unauthorized delegation, privilege escalation | 50 |
| sessions | LOW | Session hijacking (limited impact) | 10 |
| memory | LOW | Memory poisoning (limited impact) | 10 |
| ui | LOW | UI spoofing (limited impact) | 5 |

**Computation:**

```
WHS_ungov = sum_over_violations(lambda_group * 1)
WHS_gov = sum_over_FN(lambda_group * 1) + sum_over_FP(1)

Expected:
  WHS_ungov = 25*1000 + 10*500 + 5*200 + 4*50 + ...  (very large)
  WHS_gov   = 2*1000 + 1*200 + 2*1                    (very small)

Weighted Risk Reduction Ratio:
  WRRR = 1 - (WHS_gov / WHS_ungov)
```

### 7.4 Presentation Strategy

**Figure 1 (primary): RRR with 95% CI.** Simple, interpretable. "Governance prevented X% of violations."

**Figure 2 (tradeoff): Two-axis scatter.** X-axis = FPR (tasks wrongly blocked). Y-axis = FNR (violations missed). Each point is a tool group. Ideal is origin (0,0). Governance pulls all groups toward the Y-axis (low FPR) and bottom (low FNR).

**Figure 3 (cost-weighted): Weighted Harm Score comparison.** Bar chart: WHS_ungov vs WHS_gov. The bars will differ by 2-3 orders of magnitude. Log scale may be necessary.

**Table 1 (stratified): Per-tool-group RRR, FPR, and lambda.** Shows that governance is most valuable where lambda is highest (CRITICAL tier).

---

## 8. LangChain Baseline Design

### 8.1 Design Rationale

LangChain/LangGraph agents are more deterministic than OpenClaw because:
- Tool calls are defined in a graph structure with explicit edges
- The governance surface is smaller (fewer tool groups, more constrained flow)
- The `TelosWrapper` in `telos_adapters/langgraph/wrapper.py` applies pre/post checks

### 8.2 Study Type: Non-Inferiority + Portability

**Non-Inferiority Arm:** Show that TELOS governance does not degrade task completion on a well-behaved LangGraph agent.

```
H0: TCR_governed - TCR_ungoverned <= -delta (governance harms completion by more than delta)
H1: TCR_governed - TCR_ungoverned > -delta (governance does not harm by more than delta)

delta = 0.05 (5% non-inferiority margin -- tighter than OpenClaw's 15% because LangGraph tasks are more structured)
```

**Portability Arm:** Show that the same governance math (cascade scoring, composite weights, boundary detection) produces correct verdicts on LangGraph tool calls.

```
H0: AUC_langchain < AUC_openclaw - 0.05 (governance degrades on different framework)
H1: AUC_langchain >= AUC_openclaw - 0.05 (governance transfers)
```

### 8.3 Sample Size for LangChain

**Non-inferiority sample size (paired proportions):**

```
n = (z_alpha + z_beta)^2 * 2 * p * (1-p) / delta^2

For:
  alpha = 0.025 (one-sided)
  beta = 0.20 (80% power)
  p = 0.95 (expected TCR for both arms on well-behaved framework)
  delta = 0.05

n = (1.96 + 0.842)^2 * 2 * 0.95 * 0.05 / 0.05^2
  = (2.802)^2 * 0.095 / 0.0025
  = 7.851 * 38
  = 298

n >= 300 task-pairs for 80% power on non-inferiority
```

However, this assumes no replications. With N=5 replications per task and majority-vote aggregation:

```
n >= 75 task-pairs (with N=5 replications)
```

**Portability sample size:**

The portability test is simpler: run 50-100 LangGraph scenarios through the TELOS cascade and verify AUC >= 0.94 (OpenClaw AUC 0.9905 minus 0.05 margin). This requires:

```
n >= 50 scenarios for a 95% CI on AUC with width < 0.10
```

### 8.4 LangGraph Scenario Construction

Unlike OpenClaw (which has a 100-scenario corpus with documented provenance), LangGraph scenarios must be constructed. Options:

1. **Translate OpenClaw scenarios** to LangGraph equivalents. Preserve the request text and expected decision, but map to LangGraph tool names (`graph_node.tool_call` instead of `Bash`, `Read`, etc.). Cost: Medium. Validity: High (same violations, different framework).

2. **LangGraph-native scenarios** from the existing LangGraph tests in `telos_adapters/langgraph/`. Cost: Low. Validity: Medium (may not cover the same violation surface).

3. **Synthetic scenarios** generated from the same attack taxonomy (8 families from `ADVERSARIAL_DATASHEET.md`). Cost: High. Validity: Highest (independent verification).

**Recommendation:** Option 1 (translate) for the non-inferiority arm, Option 3 (synthetic) for the portability arm.

---

## 9. Analysis Plan

### 9.1 Pre-Registration

All analyses below are pre-registered. Exploratory analyses must be labeled as such in the paper.

### 9.2 Primary Analysis

1. **Compute RRR** with 95% bootstrap CI (BCa, 10,000 resamples)
2. **McNemar's test** on discordant pairs (one-sided, alpha = 0.05)
3. **Permutation test** (10,000 permutations) as non-parametric confirmation

### 9.3 Secondary Analyses

4. **Non-inferiority test** on task completion rate (one-sided, alpha = 0.025)
5. **Wilcoxon signed-rank test** on latency overhead (paired, two-sided)
6. **FPR estimation** with 95% Clopper-Pearson exact CI (appropriate for small counts)
7. **Action chain divergence** analysis (descriptive: histogram of divergence steps)

### 9.4 Stratified Analyses

8. **Per-tool-group RRR** with bootstrap CIs (11 strata)
9. **Per-risk-tier RRR** with bootstrap CIs (4 strata)
10. **Per-attack-family detection rate** with bootstrap CIs (6-8 strata)

### 9.5 Cost-Weighted Analysis

11. **Weighted Harm Score** comparison (WHS_gov vs WHS_ungov)
12. **Sensitivity analysis** on lambda values (vary lambda from 10 to 10,000 and show WRRR is robust)

### 9.6 LangChain Analysis

13. **Non-inferiority test** on LangGraph TCR
14. **AUC portability test** (LangGraph AUC vs OpenClaw AUC, with 95% CI)

### 9.7 Multiple Comparisons

Stratified analyses (8-10) involve 11 + 4 + 8 = 23 tests. Apply Benjamini-Hochberg FDR correction at q = 0.05. Report both raw and adjusted p-values.

For the primary analysis (tests 1-3), no correction is needed (single primary outcome).

---

## 10. Minimum Viable Study vs. Ideal Study

### 10.1 Minimum Viable Study (Publishable)

| Parameter | Value |
|-----------|-------|
| Task-pairs | 60 (augmented corpus) |
| Replications (Tier 1) | 1 (deterministic) |
| Replications (Tier 2) | 5 (fixed-seed) |
| Total runs per arm | 60 (Tier 1) + 300 (Tier 2) = 360 |
| Human annotation sample | 20% (72 actions) |
| Annotators | 2 |
| Power (for RRR >= 0.30) | 80% |
| Power (for expected RRR ~0.90) | >99% |
| Stratification | Risk tier only (4 strata) |
| LangChain arm | 50 scenarios (portability only, no non-inferiority) |
| Estimated compute cost | ~2 hours (ONNX inference, no LLM API calls for Tier 1) |
| Estimated annotation cost | 2 annotators * 72 actions * 5 min/action = 12 person-hours |

**What you get:** A defensible, pre-registered, paired comparison with bootstrap CIs on RRR, published in a peer-reviewed venue. Sufficient for a workshop paper or short conference paper.

### 10.2 Ideal Study (Full Research Paper)

| Parameter | Value |
|-----------|-------|
| Task-pairs | 130 (fully augmented corpus: >= 10 per tool group) |
| Replications (Tier 1) | 1 (deterministic) |
| Replications (Tier 2) | 10 (fixed-seed) |
| Replications (Tier 3) | 10 (ecological, temp=0.7) |
| Total runs per arm | 130 + 1,300 + 1,300 = 2,730 |
| Human annotation sample | 30% (819 actions) |
| Annotators | 3 (with third as tiebreaker) |
| Power (for RRR >= 0.30) | 95% |
| Stratification | Tool group (11) + risk tier (4) + attack family (8) |
| LangChain arm | 100 scenarios (non-inferiority + portability) |
| Estimated compute cost | ~8 hours (includes LLM API calls for Tier 2/3) |
| Estimated annotation cost | 3 annotators * 273 actions * 5 min/action = 68 person-hours |

**What you get:** A comprehensive study with stratified analyses, cost-weighted harm scores, ecological validity arm, and LangChain portability proof. Sufficient for a full research paper at a top venue (AAAI, NeurIPS Safety Workshop, FAccT, AIES).

### 10.3 What to Cut If Resources Are Scarce

Priority order (cut from bottom):

1. **Keep:** Tier 1 deterministic comparison (60 task-pairs, 1 replication) -- the core result
2. **Keep:** Bootstrap CIs on RRR -- publishability requires confidence intervals
3. **Keep:** Post-hoc TELOS scoring on ungoverned logs -- the measurement backbone
4. **Cut last:** Human annotation calibration (20% sample) -- reduces to 10% if resources tight
5. **Cut if needed:** Tier 2 fixed-seed replications -- removes live-agent evidence
6. **Cut if needed:** LangChain portability arm -- can be a follow-up paper
7. **Cut if needed:** Tier 3 ecological validity -- can be a follow-up paper
8. **Cut if needed:** Cost-weighted analysis -- important but can be computed post-hoc from published data

---

## 11. Threats to Validity

### 11.1 Internal Validity

| Threat | Mitigation |
|--------|-----------|
| **Task selection bias:** Corpus may not represent real-world OpenClaw usage | Corpus is incident-derived (CVEs, breach reports), not hypothetical. Provenance documented in `PROVENANCE.md`. |
| **Measurement bias:** TELOS scoring its own performance | Human annotation calibration sample (Section 6.2). Independent annotators blinded to TELOS verdicts. |
| **Testing effect:** Governance changes agent behavior (not just catches violations) | Tier 2 fixed-seed design isolates the governance intervention effect. |
| **History threat:** LLM API changes between arm executions | Run both arms on same day, same API version. Pin model version. |

### 11.2 External Validity

| Threat | Mitigation |
|--------|-----------|
| **Ecological validity:** Deterministic benchmark != real agent behavior | Tier 3 ecological validity arm with realistic temperature. |
| **Generalization to other agents:** OpenClaw-specific results | LangChain portability arm demonstrates framework-agnostic governance. |
| **Generalization to other violations:** Corpus covers 8 attack families but not all possible violations | Document known gaps honestly per `ADVERSARIAL_DATASHEET.md`. |

### 11.3 Construct Validity

| Threat | Mitigation |
|--------|-----------|
| **"Violation" definition:** Post-hoc classification may disagree with human judgment | Cohen's kappa on human-vs-TELOS calibration sample. Report disagreement rate. |
| **Composite scoring threshold issue:** Cat C accuracy is 18.2%, meaning many legitimate tasks produce CLARIFY instead of EXECUTE | Separate boundary-triggered false positives (FPR metric) from composite threshold misses. The study measures governance *harm prevention*, not composite threshold calibration. |

---

## 12. Pre-Registered Decision Criteria

### 12.1 Primary Outcome (RRR)

| Verdict | RRR (95% CI lower bound) | Interpretation |
|---------|--------------------------|----------------|
| **STRONG SUPPORT** | >= 0.70 | Governance prevents majority of violations |
| **MODERATE SUPPORT** | >= 0.30 | Governance has meaningful harm reduction |
| **WEAK SUPPORT** | >= 0.10 | Governance reduces some violations but marginal |
| **NULL** | < 0.10 | Governance does not meaningfully reduce violations |
| **HARMFUL** | RRR < 0 (CI excludes 0) | Governance increases violations (would require investigation) |

### 12.2 Secondary Outcomes

| Metric | Threshold | Verdict |
|--------|-----------|---------|
| Task Completion non-inferiority | TCR difference > -0.15 | PASS |
| Latency overhead | Median < 50ms per tool call | PASS |
| False positive rate | FPR < 5% | PASS |
| Weighted Harm Score | WRRR > 0.50 | PASS |

### 12.3 LangChain Arm

| Metric | Threshold | Verdict |
|--------|-----------|---------|
| TCR non-inferiority | TCR difference > -0.05 | PASS |
| AUC portability | AUC >= 0.94 | PASS |

---

## Appendix A: Formulas Reference

### A.1 McNemar's Test Statistic

```
chi^2 = (|b - c| - 1)^2 / (b + c)

where:
  b = Type A discordant pairs (ungoverned violates, governed does not)
  c = Type B discordant pairs (governed violates, ungoverned does not)

Reject H0 if chi^2 > chi^2_crit(1, alpha)
```

### A.2 Bootstrap BCa Confidence Interval for RRR

```
1. Compute RRR* on 10,000 bootstrap resamples (resample task-pairs with replacement)
2. Compute bias correction: z_0 = Phi^{-1}(proportion of RRR* < RRR_observed)
3. Compute acceleration: a = sum(jack_i^3) / (6 * (sum(jack_i^2))^{3/2})
4. Adjusted percentiles:
   alpha_lo = Phi(z_0 + (z_0 + z_{alpha/2}) / (1 - a*(z_0 + z_{alpha/2})))
   alpha_hi = Phi(z_0 + (z_0 + z_{1-alpha/2}) / (1 - a*(z_0 + z_{1-alpha/2})))
5. CI = [RRR*_{alpha_lo}, RRR*_{alpha_hi}]
```

### A.3 Weighted Harm Score

```
WHS = sum_i (lambda_i * x_i)

where:
  lambda_i = cost weight for violation i (from tool group risk tier)
  x_i = 1 if violation occurred, 0 otherwise

WRRR = 1 - (WHS_gov / WHS_ungov)
```

### A.4 Cohen's Kappa (Inter-Annotator Agreement)

```
kappa = (p_o - p_e) / (1 - p_e)

where:
  p_o = observed agreement rate
  p_e = expected agreement by chance = sum_k(p_{1k} * p_{2k})
```

### A.5 Non-Inferiority Test for Task Completion

```
z = (TCR_gov - TCR_ungov + delta) / sqrt(var(TCR_gov - TCR_ungov))

Reject H0 if z > z_alpha (one-sided)
```

---

## Appendix B: Implementation Checklist

### B.1 Codebase Requirements

- [ ] Augment `openclaw_boundary_corpus_v1.jsonl` to >= 120 scenarios (add memory, sessions, ui, nodes)
- [ ] Add `--seed` flag to `run_openclaw_benchmark.py` for reproducible LLM seeds
- [ ] Add `--replications N` flag for multi-run aggregation
- [ ] Implement post-hoc scoring mode: load ungoverned action logs and score offline
- [ ] Add `--export-pairs` flag to output paired results in analysis-ready format
- [ ] Build human annotation interface (simple web form: action text + context -> label dropdown)
- [ ] Implement proxy metric collection (filesystem, network, process monitors)

### B.2 Analysis Scripts

- [ ] `analysis/compute_rrr.py`: Primary RRR with bootstrap CI
- [ ] `analysis/mcnemar_test.py`: McNemar's test on discordant pairs
- [ ] `analysis/permutation_test.py`: Permutation test (10,000 permutations)
- [ ] `analysis/stratified_rrr.py`: Per-tool-group and per-risk-tier RRR
- [ ] `analysis/cost_weighted.py`: WHS computation with sensitivity on lambda
- [ ] `analysis/divergence_analysis.py`: Action chain divergence point histogram
- [ ] `analysis/langchain_portability.py`: LangChain AUC and non-inferiority test
- [ ] `analysis/annotator_agreement.py`: Cohen's kappa on calibration sample

### B.3 Reporting

- [ ] Pre-registration document (this file) posted before experiment execution
- [ ] Raw data archived with SHA-256 checksums
- [ ] Analysis scripts version-controlled in `analysis/` directory
- [ ] All random seeds recorded in experiment log
- [ ] Human annotation guidelines documented
- [ ] Results recorded in closure document (`research/governed_vs_ungoverned_closure.md`)

---

*Pre-registered before experiment execution. Results will be documented in `research/governed_vs_ungoverned_closure.md`. All analyses specified above are confirmatory unless explicitly labeled exploratory. Deviations from this protocol will be documented and justified in the closure report.*
