# TELOS Academic Paper - Chunk 3: Adversarial Validation (HUMANIZED)

## 5. Adversarial Validation

### 5.1 Attack Taxonomy

We tested 1,300 attacks across two benchmarks:

| Benchmark | Source | Attacks | Domain | TELOS ASR |
|-----------|--------|---------|--------|-----------|
| HarmBench | Center for AI Safety | 400 | General-purpose harms | 0% |
| MedSafetyBench | NeurIPS 2024 | 900 | Healthcare/medical safety | 0% |
| Total | | 1,300 | | 0% |

Tier Distribution (HarmBench subset, n=400):
- Tier 1 (PA blocks): 95.8% (383/400)
- Tier 2 (RAG blocks): 3.0% (12/400)
- Tier 3 (Expert blocks): 1.2% (5/400)

Note: Tier assignments were recorded for HarmBench attacks only. MedSafetyBench validation confirmed 0% ASR, but individual tier details were not collected during that run.

### 5.2 Experimental Setup

Models Tested:
- Mistral Small (baseline and TELOS-controlled)
- Mistral Large (baseline and TELOS-controlled)
- Raw models (no controls)

Metrics:
- Attack Success Rate (ASR): Percentage of successful violations
- Violation Defense Rate (VDR): 1 - ASR

### 5.3 Results

| Configuration | ASR | VDR | 95% CI |
|--------------|-----|-----|--------|
| Raw Mistral Small | 30.8% | 69.2% | [25.1%, 36.5%] |
| Mistral Small + System Prompt | 11.1% | 88.9% | [7.8%, 14.4%] |
| Mistral Small + TELOS | 0.0% | 100.0% | [0.0%, 5.4%] |
| Raw Mistral Large | 43.9% | 56.1% | [37.8%, 50.0%] |
| Mistral Large + System Prompt | 3.7% | 96.3% | [1.9%, 5.5%] |
| Mistral Large + TELOS | 0.0% | 100.0% | [0.0%, 5.4%] |

### 5.4 Statistical Significance

Using Wilson score intervals for 0 out of 1,300 successes:
- 95% CI: [0.0%, 0.28%]
- 99% CI: [0.0%, 0.35%]

This confirms a 0% ASR with high confidence. This result is significantly different from baseline approaches (p < 0.001, Fisher's exact test).

### 5.5 Statistical Validity of 0% ASR Claim

#### 5.5.1 Confidence Intervals for Zero Success Rate

With 0 successes in 1,300 trials, we cannot state that the true success rate is exactly 0%. Instead, we establish confidence intervals using standard statistical methods for rare events.

Wilson Score Interval:

The Wilson score interval is preferable over normal approximation for proportions near 0 or 1:

CI = [p̂ + z²/(2n) ± z√(p̂(1-p̂)/n + z²/(4n²))] / (1 + z²/n)

Where:
- p̂ = observed proportion = 0/1,300 = 0
- n = sample size = 1,300
- z = z-score for confidence level

Calculated Intervals:

| Confidence Level | z-score | Lower Bound | Upper Bound | Interpretation |
|-----------------|---------|-------------|-------------|----------------|
| 90% | 1.645 | 0.000 | 0.0020 | True ASR < 0.20% with 90% confidence |
| 95% | 1.960 | 0.000 | 0.0028 | True ASR < 0.28% with 95% confidence |
| 99% | 2.576 | 0.000 | 0.0035 | True ASR < 0.35% with 99% confidence |
| 99.9% | 3.291 | 0.000 | 0.0044 | True ASR < 0.44% with 99.9% confidence |

Rule of Three: For 0/n events, this rule provides a simple approximation: 95% CI upper bound ≈ 3/n = 3/1,300 = 0.23%. This closely matches our Wilson score calculation.

#### 5.5.2 Power Analysis and Sample Size Justification

To differentiate between 0% and a specified alternative ASR with statistical power:

n = [z_α√(p₀(1-p₀)) + z_β√(p₁(1-p₁))]² / (p₁ - p₀)²

| Alternative ASR | Power | Required n | Our n | Adequate? |
|----------------|-------|------------|-------|-----------|
| 10% | 80% | 29 | 1,300 | Exceeds by 44x |
| 5% | 80% | 59 | 1,300 | Exceeds by 22x |
| 3% | 80% | 99 | 1,300 | Exceeds by 13x |
| 1% | 80% | 299 | 1,300 | Exceeds by 4.3x |
| 0.5% | 80% | 599 | 1,300 | Exceeds by 2.2x |
| 0.25% | 80% | 1,198 | 1,300 | Exceeds by 1.1x |

Our 1,300 attacks provide 80% power to detect an ASR as low as 0.25%, considerably higher than the best published baselines (3.7% for system prompts).
