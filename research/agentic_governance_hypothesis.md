# Research Hypothesis: Semantic Density and Governance Effectuality in Agentic AI Systems

**TELOS AI Labs Inc.**
**Principal Investigator:** JB@telos-labs.ai
**Document Status:** Active Research — Living Document
**Created:** 2026-02-07
**Last Updated:** 2026-02-17

---

### Disclosures

> **Generative AI Disclosure:** Internal analysis, experimental design review, and qualitative assessment in this document were conducted with assistance from LLM-based research agents (Claude, Anthropic). These agents are prompted with domain-specific personas (governance theory, statistics, systems engineering, regulatory analysis, research methodology) and operate as AI research assistants — not independent human expert reviewers. All quantitative results (AUC-ROC, F1, bootstrap confidence intervals, benchmark accuracies) are computed by deterministic code. Qualitative analysis should not be treated as independent peer review. See `research/research_team_spec.md` for full methodology.

> **Conflict of Interest Disclosure:** This research was conducted and funded by TELOS AI Labs Inc., which has a commercial interest in the TELOS governance framework. All domain-specific validation benchmarks (Nearmap, Healthcare, OpenClaw) were created by the research team. External benchmarks (PropensityBench, AgentHarm, AgentDojo) were created by independent organizations. Research artifacts are published on [Zenodo](https://zenodo.org/) with persistent DOIs. No external funding or independent peer review was involved in this work.

---


## Abstract

This document formalizes the core research hypothesis guiding TELOS's extension from conversational AI governance into agentic AI governance. We hypothesize that agentic AI systems present a fundamentally more governable domain for cosine similarity-based fidelity measurement than conversational systems, due to the semantic density of tool definitions, boundary specifications, and action chain patterns. This counter-intuitive finding — that governing actions is mathematically easier than governing language — has significant implications for the deployment of trustworthy AI agents in regulated domains.

---

## 1. Background

### 1.1 Conversational Governance: The Harder Problem

TELOS (Telically Entrained Linguistic Operational Substrate) was originally developed to govern conversational AI alignment. The framework uses Primacy Attractors (PA) — embedding-space representations of user purpose — to detect and direct conversational drift in real-time using cosine similarity measurement.

In conversational governance, the PA is derived from natural language purpose statements (e.g., "help the user understand recursion"). This creates a broad basin in embedding space where:

- Hundreds of semantically related concepts have legitimate membership
- Basin boundaries are inherently soft and context-dependent
- Phrasing variation introduces measurement noise
- Adjacent topics create overlapping embedding regions
- The same intent expressed differently produces variable fidelity scores

Despite these challenges, TELOS achieved strong results in controlled testing:
- 0% Attack Success Rate across 1,300 adversarial attacks (MedSafetyBench + HarmBench)
- 100% block rate against 100 suicide/self-harm elicitation attempts
- Consistent drift detection and proportional correction across 10-turn sessions

### 1.2 Agentic Governance: The Tighter Domain

Agentic AI systems introduce a fundamentally different governance target. Instead of governing free-form language, the system governs:

- **Tool invocations** — discrete, categorical actions with well-defined descriptions
- **Action chains** — structured sequences of tool calls serving a stated purpose
- **Boundary compliance** — enumerable constraints that define operational limits
- **Delegation decisions** — structured handoffs between specialized agents

The initial assumption was that agentic governance would be MORE challenging because it involves governing actions rather than words. Our research suggests the opposite.

---

## 2. Core Hypothesis

### 2.1 Primary Hypothesis

> **In agentic AI governance, the Primacy Attractor operates in a semantically denser subspace than in conversational governance. Tool definitions, boundary specifications, and action chain patterns create embedding targets with lower intra-cluster variance and greater inter-cluster separation than conversational purpose descriptions. This predicts that fidelity measurement in agentic space will exhibit higher precision (fewer false positives) and higher recall (fewer false negatives) than equivalent measurements in conversational space, using identical embedding models and identical mathematical frameworks.**

### 2.2 Supporting Propositions

**P1: Semantic Density of Tool Descriptions**
Tool descriptions occupy narrow, well-defined regions in embedding space. "Execute SQL SELECT queries and return results" has significantly lower semantic variance than "help users understand recursion." Aligned requests cluster tightly around tool embeddings; misaligned requests fall clearly outside the cluster.

**P2: Deterministic Boundary Detection**
Boundary specifications in agentic space are categorical rather than semantic. "No data modification (INSERT, UPDATE, DELETE)" creates a bright-line embedding target. High cosine similarity between a request and a boundary unambiguously indicates violation. Compare this to conversational boundaries like "stay focused on the topic," which require interpretation of what constitutes "the topic."

**P3: Discrete Decision Space**
Agentic governance produces categorical decisions (execute tool X / don't execute tool X). Conversational governance produces gradient responses (minor redirect / moderate intervention / strong block). The discrete decision space means the governance math needs to be correct at the threshold, not approximately correct across a continuous spectrum.

**P4: PA Refinement from Phrase to Manifest**
In conversation, the PA is derived from a purpose phrase — broad, interpretive, context-dependent. In agentic space, the PA is derived from a fully predefined tool manifest — narrow, enumerable, deterministic. The PA doesn't represent "understand databases" but rather "execute these 4 specific operations within these specific boundaries." The zone of inference collapses from a broad semantic field to a tight, enumerable set of valid actions.

**P5: Behavioral Similarity to Keyword Matching Without Being Keyword Matching**
Because the semantic mappings between tool-aligned requests and tool descriptions are inherently congruent, the governance BEHAVES similarly to keyword matching — "show me revenue" maps to "SQL SELECT query" with high similarity, as if matching keywords. But it IS NOT keyword matching because:
- "Display the financial figures for last quarter" maps correctly despite sharing zero keywords
- "Remove all entries from the relation" maps to the DELETE boundary despite different vocabulary
- The system captures INTENT, not vocabulary — intent and vocabulary are simply more correlated in agentic space than in conversational space

### 2.3 Extended Hypotheses for Autonomous Agent Governance (H6-H10)

The following hypotheses extend the primary hypothesis to address concerns specific to always-on autonomous agents (e.g., OpenClaw). Sourced from Russell's M0 governance theory analysis.

**H6: Temporal PA Decay**

> **Over extended continuous operation (weeks/months), the Primacy Attractor's effectiveness as a governance anchor degrades due to distributional shift in the action request space.** As agents encounter novel tool compositions and evolving user patterns, the original PA embedding becomes a less precise representation of the intended governance constraints.

- *Measurement:* Track governance decision accuracy per week over a 30-day deployment. Plot accuracy vs. time since PA calibration.
- *Confirmation criteria:* Accuracy decline > 5% over 30 days (PA re-calibration needed).
- *Disconfirmation:* No measurable decline (PA is stable for autonomous operation).
- *Source:* Russell M0 analysis — principal-agent temporal dynamics in 24/7 operation.

**H7: Cross-Channel Contamination**

> **Governance decisions on one tool group influence the governance engine's behavior on adjacent tool groups through shared embedding space effects.** For example, repeated EXECUTE decisions on `fs` (LOW risk) may subtly shift the engine's calibration for `runtime` (CRITICAL risk) decisions due to shared semantic neighbors.

- *Measurement:* Isolate per-tool-group accuracy before and after concentrated activity on a single tool group.
- *Confirmation criteria:* Adjacent-group accuracy shift > 2% after 1,000 decisions on a focal group.
- *Disconfirmation:* No cross-group accuracy interference.
- *Source:* Russell M0 analysis — cross-channel embedding interference in multi-group governance.

**H8: Cumulative Authority Bounds**

> **The aggregate authority granted to an autonomous agent over time through individual EXECUTE decisions exceeds the authority any single decision grants, creating an unbounded cumulative authority problem.** In session-based governance, authority is naturally bounded by session termination. In always-on agents, this bound is removed.

- *Measurement:* Track cumulative EXECUTE decisions per tool group per day. Define authority as the sum of risk-tier weights of all executed actions.
- *Confirmation criteria:* Cumulative authority at day 30 exceeds any reasonable single-session authority grant by > 10x.
- *Mitigation if confirmed:* Rolling authority budgets per tool group with mandatory re-authorization periods.
- *Source:* Russell M0 analysis — principal-agent theory, authority delegation without session bounds.

**H9: Governance Overhead Scaling**

> **Governance latency scales sub-linearly with the number of tool groups governed.** Adding tool groups to the governance scope does not linearly increase per-decision latency because L0 keyword matching and L1 cosine similarity are constant-time operations against a fixed embedding index.

- *Measurement:* Benchmark governance latency with 5, 10, 15, 20 tool groups.
- *Confirmation criteria:* Latency at 20 groups < 2x latency at 5 groups.
- *Disconfirmation:* Linear or super-linear scaling (architectural bottleneck).
- *Source:* Karpathy M0 analysis — latency budget for 10-17ms total cascade.

**H10: Multi-Agent Governance**

> **Governing agent-to-agent delegation requires additional governance dimensions beyond single-agent tool governance.** When one agent delegates to another (e.g., via OpenClaw's `group:nodes`), the governance engine must evaluate not just the delegation action but the delegated agent's authority scope, creating a recursive governance requirement.

- *Measurement:* Evaluate governance accuracy on multi-agent delegation scenarios vs. single-agent tool calls.
- *Confirmation criteria:* Delegation scenarios show > 10% lower accuracy than single-agent scenarios without additional governance dimensions.
- *Mitigation if confirmed:* Add delegation-specific dimensions (delegated scope, chain depth, transitive authority) to the fidelity score.
- *Source:* Russell M0 analysis — principal-agent chains in multi-agent systems.

### 2.4 Null Hypothesis

The null hypothesis states that agentic governance provides no measurable improvement in precision or recall over conversational governance when using identical embedding models and mathematical frameworks. Under this hypothesis, the semantic density of tool descriptions provides no significant advantage, and the same measurement noise present in conversational governance persists in agentic space.

---

## 3. Testing Framework

### 3.1 Metric Definitions

| Metric | Definition | Measurement Method |
|--------|-----------|-------------------|
| **Intra-cluster Variance** | Average pairwise cosine distance between on-topic requests for the same tool/purpose | Embed N phrasings of the same intent, compute variance of cosine similarities to target |
| **Inter-cluster Separation** | Average cosine distance between on-topic and off-topic requests | Embed on-topic and off-topic sets, compute mean distance between groups |
| **Decision Boundary Sharpness** | Gap between lowest on-topic score and highest off-topic score | min(on-topic scores) - max(off-topic scores); positive = clean separation |
| **Cross-Phrasing Stability** | Variance of fidelity scores for the same intent expressed N ways | Embed 10+ phrasings, measure coefficient of variation |
| **Threshold Sensitivity** | Change in error rate per 0.01 threshold shift | Sweep threshold ±0.10, plot error rate curve |
| **Governance Precision** | True positives / (True positives + False positives) | Standard classification metrics on labeled test set |
| **Governance Recall** | True positives / (True positives + False negatives) | Standard classification metrics on labeled test set |

### 3.2 Test Protocols

**Protocol 1: Embedding Precision Comparison**
- For each agent template (SQL, Research, Customer Service):
  - Generate 50 on-topic requests per tool (200 total per agent)
  - Generate 50 off-topic requests (clearly misaligned)
  - Generate 50 boundary violation requests
  - Compute fidelity scores using MiniLM-L6 (384-dim)
  - Compare intra-cluster variance and inter-cluster separation
- For conversational PA (Learn Recursion, Understand TELOS):
  - Generate equivalent test sets
  - Compute same metrics
- **Hypothesis confirmed if:** Agentic intra-cluster variance < Conversational intra-cluster variance AND Agentic inter-cluster separation > Conversational inter-cluster separation

**Protocol 2: Decision Boundary Analysis**
- For each domain (agentic + conversational):
  - Plot fidelity score distributions for on-topic vs off-topic
  - Measure overlap region (the "gray zone")
  - Compute decision boundary sharpness
- **Hypothesis confirmed if:** Agentic gray zone is narrower than conversational gray zone

**Protocol 3: Cross-Phrasing Stability**
- Select 10 core intents per domain
- Express each intent in 20 different phrasings
- Compute fidelity score variance per intent
- **Hypothesis confirmed if:** Agentic coefficient of variation < Conversational coefficient of variation

**Protocol 4: Threshold Robustness**
- Sweep fidelity threshold from 0.50 to 0.95 in 0.01 increments
- At each threshold, compute precision and recall
- Plot ROC curves for agentic vs conversational
- **Hypothesis confirmed if:** Agentic AUC > Conversational AUC

**Protocol 5: Adversarial Resistance**
- Apply the same adversarial attack patterns used in MedSafetyBench/HarmBench
- Adapt attacks for agentic context (social engineering to invoke wrong tools, boundary bypass attempts)
- Measure Attack Success Rate (ASR) for agentic vs conversational
- **Hypothesis confirmed if:** Agentic ASR <= Conversational ASR (both expected near 0%)

### 3.3 Embedding Models Under Test

| Model | Dimensions | Purpose |
|-------|-----------|---------|
| MiniLM-L6 | 384 | Primary — smallest model, tests whether semantic density compensates for model size |
| MPNet | 768 | Secondary — larger model, establishes performance ceiling |
| Mistral Embed | 1024 | Tertiary — production-grade, validates scaling behavior |

### 3.4 Control Variables

- Same normalization pipeline (baseline 0.20, ceiling 0.80)
- Same composite fidelity formula (0.35 purpose + 0.20 scope + 0.20 tool + 0.15 chain - 0.10 boundary)
- Same decision thresholds (EXECUTE >= 0.85, CLARIFY >= 0.70, SUGGEST >= 0.50)
- Same embedding model per comparison pair
- Same hardware and inference configuration

---

## 4. Expected Outcomes

### 4.1 If Hypothesis is Confirmed

- Agentic governance achieves higher precision and recall than conversational governance with the same models
- Small sentence transformers (MiniLM-L6, 384-dim) are SUFFICIENT for production-grade agentic governance
- The cost/latency profile of agentic governance is favorable (small models, fast inference, high accuracy)
- TELOS can be positioned as providing governance that gets MORE reliable as AI systems become more autonomous
- Regulatory frameworks (EU AI Act Article 72) can be satisfied with mathematically demonstrable governance

### 4.2 If Hypothesis is Partially Confirmed

- Tier 2 (tool selection) and boundary checking show improved precision, but Tier 1 (request fidelity) shows equivalent performance to conversational
- This would confirm the structural advantage of discrete tool targets while showing that the natural language interface layer retains conversational-level noise
- Implication: the governance stack is strongest at the tool/action layer and should weight those signals accordingly

### 4.3 If Hypothesis is Disproven

- Agentic governance shows no measurable improvement over conversational governance
- This would suggest that embedding-space noise is dominated by model-level factors rather than target-level factors
- Implication: larger models are required for agentic governance; small sentence transformers are insufficient regardless of semantic density

---

## 5. Research Log Protocol

All testing data, observations, and analysis are recorded in the companion research log:
`/research/research_log.md`

### Log Entry Format

Each entry follows this structure:
```
### [DATE] — [TITLE]
**Observer:** [Name/Agent]
**Type:** [Observation | Test Result | Analysis | Theory Revision]
**Context:** [What prompted this entry]

**Observation:**
[What was observed]

**Analysis:**
[Interpretation and implications]

**Action Items:**
[What follows from this observation]
```

### Automated Review Triggers

Research team review is triggered by:
1. New test data generated (embedding precision comparisons)
2. Threshold changes in `telos_core/constants.py`
3. New agent template added or modified
4. Governance decision overridden in production
5. Periodic review (weekly during active development)

---

## 6. Relationship to Existing Validation

This research builds on the established conversational governance validation:

| Benchmark | Domain | Result | Status |
|-----------|--------|--------|--------|
| MedSafetyBench (1,300 attacks) | Conversational/Healthcare | 0% ASR | Published (Zenodo) |
| HarmBench (multi-category) | Conversational/General | 0% ASR | Published (Zenodo) |
| Suicide/Self-Harm (100 attempts) | Conversational/Healthcare | 100% block rate | Published (Zenodo) |
| Agentic Tool Selection | Agentic/SQL | TBD | This research |
| Agentic Boundary Compliance (NLI) | Agentic/Healthcare | **NEGATIVE** — AUC 0.672 (best), NLI eliminated | Phase 1 complete (`research/cross_encoder_nli_mve_phase1.md`) |
| Agentic Boundary Compliance (SetFit) | Agentic/Healthcare | **GREEN** — AUC 0.980, 91.8% detection, 5.2% FPR | Phase 2 complete (`research/setfit_mve_phase2_closure.md`) |
| Agentic Chain Continuity | Agentic/Multi-step | TBD | This research |
| Agentic Autonomous Agent (OpenClaw) | Agentic/Autonomous | **Phase I** — 75.5% violation detection (pre-calibration), 100 scenarios, 11 tool groups | Phase I complete (`validation/openclaw/run_openclaw_benchmark.py`) |
| Cross-domain Comparison | Conv. vs Agentic | TBD | This research |

### 6.1 Acknowledged Limitation: Over-Refusal Equity

XSTest calibration reduced TELOS's conversational over-refusal rate from 24.8% to 8.0% (research/setfit_mve_phase2_closure.md). However, the remaining 8% over-refusal has **not** been analyzed for demographic disproportionality. It is possible that the residual false positives disproportionately affect certain demographic groups, linguistic styles, or cultural contexts — patterns well-documented in content moderation systems (Sap et al. 2019, Davidson et al. 2019).

This is a genuine equity gap in the current validation. Addressing it requires:
1. **Disaggregated analysis** of the remaining over-refusals by demographic proxy variables (dialect markers, cultural references, topic domains)
2. **Intersectional false positive rates** across protected categories
3. **Comparison with baseline models** to determine whether TELOS amplifies or mitigates pre-existing demographic bias in the underlying embedding model (MiniLM-L6-v2)

This analysis is planned as part of the external validation set (G7) and 30-day field study (H6-H10). Until completed, the 8.0% over-refusal rate should be treated as a system-level aggregate that may mask within-group disparities.

---

## 7. Theoretical Implications

If confirmed, this hypothesis has implications beyond TELOS:

1. **Governance scales with structure.** As AI systems become more structured (tools, APIs, workflows), they become MORE governable, not less. This inverts the common assumption that autonomy reduces controllability.

2. **Small models suffice for structured governance.** The industry trend toward ever-larger models for safety may be unnecessary in agentic contexts where semantic density provides natural amplification of model capability.

3. **The principal-agent problem has a mathematical solution in agentic space.** When the agent's action space is enumerable and each action has a well-defined embedding, the alignment between principal intent and agent action becomes mathematically measurable with high confidence.

4. **Runtime governance is more feasible than assumed.** If small models achieve high accuracy in agentic governance, the computational overhead of runtime monitoring becomes negligible — enabling persistent governance without performance degradation.

---

## References

- Brunner, J.F. (2026). TELOS: Telically Entrained Linguistic Operational Substrate. TELOS AI Labs.
- TELOS Validation Data (2026). Zenodo: https://zenodo.org/records/18370659
- EU AI Act, Article 72: Post-market monitoring by providers of high-risk AI systems.
- Reimers, N. & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.
- Wang, Y. et al. (2020). MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression.

---

*This is a living document. All revisions are tracked in the research log.*
