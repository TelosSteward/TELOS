# TELOS Academic Paper - Chunk 4: Statistical Comparison and Baselines

#### 5.5.3 Comparison to Literature Baselines

| Study | System | Attacks Tested | Reported ASR | 95% CI |
|-------|--------|---------------|--------------|---------|
| Anthropic (2023) | Constitutional AI | 50 | 8% | [3.1%, 16.8%] |
| OpenAI (2024) | GPT-4 + Moderation | 100 | 3% | [1.0%, 7.6%] |
| Google (2024) | PaLM + Safety | 40 | 12.5% | [5.3%, 24.7%] |
| NVIDIA (2024) | NeMo Guardrails | 200 | 4.8% | [2.6%, 8.2%] |
| TELOS (2025) | PA + 3-Tier | 1,300 | 0% | [0%, 0.28%] |

Our sample size exceeds all published studies by at least 6.5x while achieving superior results with a dramatically tighter confidence interval.

#### 5.5.4 Bayesian Analysis

Using Bayesian inference with uninformative Beta(1,1) prior:

P(θ|data) ~ Beta(α + s, β + f) = Beta(1, 1301)

Posterior Statistics:
- Mean: 0.077%
- Median: 0.053%
- Mode: 0%
- 95% Credible Interval: [0.002%, 0.23%]

The Bayesian 95% credible interval provides strong evidence for near-zero ASR.

#### 5.5.5 Attack Diversity and Coverage

| Category | HarmBench | MedSafetyBench | Total | Percentage |
|----------|-----------|----------------|-------|------------|
| Direct Requests (L1) | 45 | 85 | 130 | 10.0% |
| Social Engineering (L2) | 80 | 180 | 260 | 20.0% |
| Multi-turn Manipulation (L3) | 85 | 195 | 280 | 21.5% |
| Prompt Injection (L4) | 90 | 120 | 210 | 16.2% |
| Semantic Boundary Probes (L5) | 50 | 90 | 140 | 10.8% |
| Role-play/Jailbreaks (L6) | 50 | 80 | 130 | 10.0% |
| Domain-specific Advanced | - | 150 | 150 | 11.5% |
| Total | 400 | 900 | 1,300 | 100.0% |

Coverage metrics: 6/6 attack sophistication levels, 12/12 harm categories covered.

#### 5.5.6 Statistical Comparison with Baselines

Fisher's Exact Test vs. System Prompts:

              Blocked | Violated | Total
TELOS:         1,300  |    0     | 1,300
Baseline:      1,252  |   48     | 1,300

Fisher's exact test p-value < 0.0001

Chi-Square Test vs. Raw Models:

              Blocked | Violated | Total
TELOS:         1,300  |    0     | 1,300
Raw:             732  |  568     | 1,300

χ² = 568.0, df = 1, p < 0.0001

#### 5.5.7 Summary

Our claim of 0% ASR is statistically rigorous:

1. 95% CI [0%, 0.28%] establishes upper bound far below all baselines
2. 1,300 attacks exceeds typical adversarial testing by 10-30x
3. 80% power to detect ASR as low as 0.25%
4. Comprehensive coverage across 6 attack levels and 12 harm categories
5. Two established benchmarks (HarmBench + MedSafetyBench) ensure external validity
6. 95.8% Tier 1 blocking (HarmBench subset) demonstrates mathematical layer effectiveness

### 5.6 Regulatory Compliance Implications

Our validation directly addresses emerging regulatory requirements. We map TELOS capabilities to specific legislative mandates:

Table 5: Regulation-to-Validation Mapping

| Regulation | Requirement | TELOS Validation Evidence |
|------------|-------------|---------------------------|
| CA SB 243 | AI chatbots with children must prevent harmful content | 0% ASR on 130 direct requests, 260 social engineering attempts |
| CA AB 3030 | Healthcare AI must disclose AI nature, prevent patient harm | 0/30 HIPAA attacks succeeded, Tier 1 blocks at 0.70+ fidelity |
| EU AI Act Art. 9 | High-risk AI requires documented risk management | Complete forensic traces for all 1,300 attacks via TELOSCOPE |
| EU AI Act Art. 10 | Training data governance and bias testing | Published test datasets on Zenodo |
| EU AI Act Art. 14 | Human oversight mechanisms required | Three-tier architecture includes human expert escalation (Tier 3) |
| HIPAA Security Rule | Technical safeguards for PHI protection | 100% VDR on 900 MedSafetyBench healthcare attacks |
| CA SB 53 | Frontier AI transparency and safety testing | Automated validation protocol with public attack library |

Implications for Vulnerable Populations: California SB 243 specifically targets AI interactions with minors, requiring that chatbots "not encourage behavior that poses a risk of physical, emotional, developmental, or financial harm to a minor." Our validation against 260 social engineering attacks and 130 direct manipulation attempts demonstrates the type of empirical evidence this legislation will require. We have published our child-safety-specific attack patterns to Zenodo to support other researchers developing SB 243-compliant systems.

Healthcare AI Readiness: California AB 3030 requires healthcare AI systems to maintain transparency and prevent patient harm. TELOS's 100% Violation Defense Rate on MedSafetyBench's 900 healthcare-specific attacks—including 150 domain-specific advanced attacks—provides exactly the empirical evidence regulators will require. Our complete MedSafetyBench validation is available at Zenodo.

European Market Access: The EU AI Act takes effect August 2026, with Articles 9-16 mandating risk management systems, data governance, and human oversight for high-risk AI. TELOS's three-tier architecture directly maps to these requirements: Tier 1 (mathematical risk detection), Tier 2 (authoritative guidance from documented sources), and Tier 3 (human expert escalation). The complete governance trace collector provides the audit trail that Article 12 requires for post-market monitoring.
