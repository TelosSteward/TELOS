# Research Log Entry: StrongDM "Software Factory" Methodology Review

**Date:** 2026-02-19
**Agent:** Nell Watson (Research Methodologist)
**Type:** External Claims Analysis
**Context:** Requested review of StrongDM's "Software Factory" claims in context of TELOS OpenClaw governance adapter research

---

### Disclosures

> **Generative AI Disclosure:** Internal analysis, experimental design review, and qualitative assessment in this document were conducted with assistance from LLM-based research agents (Claude, Anthropic). These agents are prompted with domain-specific personas (governance theory, statistics, systems engineering, regulatory analysis, research methodology) and operate as AI research assistants — not independent human expert reviewers. All quantitative results (AUC-ROC, F1, bootstrap confidence intervals, benchmark accuracies) are computed by deterministic code. Qualitative analysis should not be treated as independent peer review. See `research/research_team_spec.md` for full methodology.

> **Conflict of Interest Disclosure:** This research was conducted and funded by TELOS AI Labs Inc., which has a commercial interest in the TELOS governance framework. All domain-specific validation benchmarks (Nearmap, Healthcare, OpenClaw) were created by the research team. External benchmarks (PropensityBench, AgentHarm, AgentDojo) were created by independent organizations. Research artifacts are published on [Zenodo](https://zenodo.org/) with persistent DOIs. No external funding or independent peer review was involved in this work.

---


## Executive Summary

I was asked to evaluate the falsifiability, methodological rigor, and publication readiness of claims made by StrongDM regarding their "Software Factory" (Level 5 dark factory for autonomous software development) and compare them with TELOS's claims about runtime governance.

**Verdict:** The StrongDM claims, as presented, are **not falsifiable** in their current form. They constitute marketing assertions rather than research hypotheses. By contrast, TELOS's claims are falsifiable, empirically tested, and publication-ready — but overreaching in some areas.

**Key finding:** When two systems make claims about "complementarity," the burden is on the claimant to demonstrate non-redundancy through controlled experiments. TELOS has not done this. The "100% of runtime safety surface" claim is defensible only if scoped to "runtime governance of tool-level decisions" — it does not cover build pipelines, test adequacy, deployment controls, or the 70-75% of dark factory operations that occur outside agent runtime.

---

## 1. Falsifiability of StrongDM's Claims

### Claim 1: "Behavioral differences cease" for digital twins

**As stated:** "Digital twins enable full integration testing without touching production because behavioral differences cease."

**Methodological problems:**

1. **No measurement protocol.** What constitutes "behavioral difference"? System calls? Network traffic patterns? State transitions? The claim provides no operationalization of the dependent variable.

2. **No threshold.** "Cease" is categorical. Real systems exhibit continuous differences. What error bound defines "ceased"? 1% divergence? 0.01%? Statistical indistinguishability?

3. **No test procedure.** How would one verify this claim? Run identical workloads against production and twin, then... what? Diff the outputs? Compare telemetry? Over what time window? With what statistical test?

**To make this falsifiable:**
- Define "behavioral difference" as a quantifiable metric (e.g., "state divergence rate: percentage of test scenarios where twin and production produce different final states")
- Set a threshold (e.g., "twin accuracy >= 95%: twin matches production state in >= 95% of test scenarios")
- Specify a test protocol (e.g., "replay 1,000 production traces against twin, measure state equivalence via hash comparison")
- Provide disconfirmation criteria (e.g., "if divergence rate > 5% on holdout traces, twin is insufficient for integration testing")

**Analogy test:** Is this like ML holdout sets? **No.** ML holdout sets are i.i.d. samples from a known distribution. Digital twin validity depends on **environment fidelity** — a fundamentally different problem. The analogy conflates statistical sampling with simulation accuracy.

---

### Claim 2: "Satisfaction scoring" by LLM is superior to boolean pass/fail

**As stated:** "Satisfaction scoring (LLM-judged, probabilistic) is superior to boolean test pass/fail."

**Methodological problems:**

1. **Superior by what criterion?** Correlation with user satisfaction? Test coverage? False negative rate? The claim provides no evaluation metric.

2. **No ground truth.** If the LLM judges satisfaction, what validates the LLM's judgment? User surveys? Manual review? Inter-rater reliability with human judges?

3. **Circularity risk.** Stanford CodeX flagged this: "Agents are trying to score well on a test that is supposed to represent user satisfaction. Those are different things." If agents are optimizing for LLM-judged scores, and LLMs have known biases (verbosity preference, style mimicry), agents may achieve high "satisfaction" scores without satisfying users.

4. **No comparison protocol.** "Superior to boolean pass/fail" requires a head-to-head comparison. Did they run the same test suite with both methods and measure outcomes? Where is the data?

**To make this falsifiable:**
- Define "satisfaction" operationally (e.g., "user accepts output without requesting revisions")
- Measure inter-rater reliability between LLM judge and human judges (Cohen's kappa >= 0.80)
- Compare predictive validity: does LLM satisfaction score correlate with actual user acceptance? (Spearman's rho >= 0.70)
- Control for agent gaming: test whether agents trained on LLM-judged satisfaction scores produce outputs that humans rate lower (adversarial validation)

**Current status:** This is an **empirical claim disguised as a design choice.** Without validation data, it is not falsifiable.

---

### Claim 3: "CXDB was built entirely by the factory"

**As stated:** "CXDB (16K Rust, 9.5K Go, 700 TS) was built entirely by the factory."

**Methodological problems:**

1. **Ambiguity of "built."** Does this mean:
   - Generated from scratch by agents with no human-written boilerplate?
   - Human-specified requirements, agent-generated code, zero human editing?
   - Human-reviewed but not human-edited?
   - Human-edited for syntax but not logic?

2. **Ambiguity of "entirely."** TELOS's benchmarks document known gaps. If CXDB has 0.01% human contribution (one critical bug fix), is that "entirely" or not?

3. **No audit trail.** How would one verify this claim? Git commit metadata can be spoofed. LLM watermarks do not exist for code. The only verification method is a complete human review of 26.2K LOC — at which point the claim becomes unfalsifiable in practice.

**To make this falsifiable:**
- Publish the full git history with commit-level agent/human attribution
- Define "human contribution" quantitatively (e.g., "< 1% of lines modified by humans post-generation")
- Provide a reproducibility protocol: "Run factory on these requirements, achieve 95% structural equivalence to published CXDB"
- Specify what counts as human intervention (editing? approving? rewriting tests?)

**Current status:** This claim is **verifiable only with insider access.** For external researchers, it is not falsifiable.

---

### Claim 4: "$1,000/day per engineer" benchmark

**As stated:** "$1,000/day per engineer is the benchmark for serious factory operation."

**This is not a falsifiable claim.** It is a cost-effectiveness assertion without comparison basis. $1,000/day compared to what? Manual development? Outsourcing? Other AI-assisted workflows? This is a **marketing claim,** not a research claim.

---

## 2. The Holdout Analogy: Software Scenarios vs. ML Holdout Sets

StrongDM claims scenarios are "analogous to holdout sets in ML." TELOS holds out 15 Cat E scenarios.

### ML Holdout Requirements

1. **i.i.d. assumption:** Test data must be drawn from the same distribution as training data
2. **Exchangeability:** Swap any training example with any test example without changing the expected performance
3. **Independence:** Test performance is conditionally independent of training data given the model

### Software Scenarios: Not i.i.d.

**Problem 1: Software scenarios are NOT exchangeable.**
- A "credential theft via .env file read" scenario is structurally different from a "RCE via WebSocket hijacking" scenario
- They test different attack surfaces, different tool groups, different semantic patterns
- TELOS's benchmark correctly stratifies by tool group for this reason

**Problem 2: Software scenarios are NOT independent.**
- Attack patterns evolve adversarially in response to defenses
- Holdout scenarios from 2026 may not represent the attack distribution in 2027
- This is **adversarial distribution shift,** not random sampling

**Problem 3: Software scenarios have non-uniform importance.**
- One missed RCE (CRITICAL) is not equivalent to 100 missed UI tool violations (LOW)
- ML holdout sets assume uniform misclassification cost; security does not

### What TELOS Actually Has

TELOS's 15 Cat E holdout scenarios are **adversarial robustness tests,** not statistical holdout sets. The correct analogy is **red team evaluation,** not train/test split.

**Why this matters:**
- SetFit achieves 93.3% detection on Cat E holdout (14/15). This is impressive, but it does NOT mean the system will achieve 93.3% detection on **future adversarial attacks** it has not seen.
- The adversarial may adapt. The 6.7% miss rate (1/15) may be the attack vector adversaries exploit.
- TELOS correctly documents this as a "known gap" (OC-WEB-ESCL-007, injection attack, score 0.1525).

**Methodological caveats for publication:**

1. Do not call Cat E scenarios "holdout data" without clarifying they are adversarial, not statistical holdouts
2. Do not extrapolate 93.3% detection to future performance without temporal validation
3. Do cite this as evidence of **current adversarial robustness** (valid), not generalization (invalid without longitudinal data)

---

## 3. The Complementarity Claim: Testable Hypothesis or Marketing?

TELOS claims: "TELOS and dark factories are complementary, not competing. TELOS covers ~25-30% of dark factory surface area but ~100% of runtime safety surface."

### Is "Complementarity" Testable?

**Yes, but TELOS has not tested it.** A complementarity claim is falsifiable if:

1. **Non-redundancy:** System A detects failures that System B does not
2. **Non-interference:** System A does not degrade System B's performance
3. **Additive value:** A+B outperforms either alone

**Experiment design to test complementarity:**

1. **Baseline:** Dark factory without TELOS governance (if such a system exists)
2. **Treatment:** Dark factory with TELOS governance at the agent tool-call layer
3. **Metrics:**
   - Test pass rate (does governance prevent bad tests from being written?)
   - False positive rate (does governance block legitimate tool calls?)
   - Incident rate (does governance reduce production incidents?)
   - Latency (does 15ms governance overhead degrade factory throughput?)
4. **Comparison:** A+B vs. A alone, on a shared benchmark (e.g., 100 software tasks with known ground truth)

**What TELOS has instead:**
- Standalone benchmark performance (96.2% detection on OpenClaw violations)
- Latency measurements (10-17ms cascade)
- No controlled comparison with an ungoverned baseline or a competing governance system

**Verdict:** The complementarity claim is **not currently falsifiable** because TELOS has no empirical data on dark factory integration. It is a **reasoned projection** based on architecture analysis, not an experimental finding.

---

## 4. LLM-as-Judge Circularity

Stanford CodeX: "Agents are trying to score well on a test that is supposed to represent user satisfaction. Those are different things."

### The Circularity Problem

**Setup:**
1. LLM A generates code
2. LLM B judges whether the code satisfies the user
3. LLM A is trained (implicitly or explicitly) to maximize LLM B's approval

**Risks:**

1. **Preference collapse.** LLM A learns LLM B's stylistic preferences (verbosity, comment density, variable naming) rather than functional correctness. High satisfaction scores, low user satisfaction.

2. **Shared failure modes.** If LLM A and LLM B share the same pre-training biases (e.g., both prefer certain coding patterns), LLM B will rate LLM A's output highly even when it is objectively incorrect. This is **inter-model agreement,** not validity.

3. **Gaming.** LLM A can learn to embed signals that LLM B weights heavily (e.g., "thoroughly tested," "production-ready") without those signals being true.

### Is This Peer Review or Self-Assessment?

**Neither.** It is **peer review by a model with shared training data.**

- **Peer review** assumes independence: reviewers have different knowledge and biases from authors
- **Self-assessment** is explicit: the same agent reviews its own work
- **LLM-as-judge** is neither: it is a **different instance** of a **similar model** with **correlated biases**

### Methodological Risk Mitigation

1. **Human inter-rater reliability baseline.** Measure Cohen's kappa between LLM judge and human judges. If kappa < 0.60, LLM judgments are not reliable proxies for human satisfaction.

2. **Adversarial validation.** Intentionally produce code that scores high with LLM judge but fails human review. If agents can exploit this, the metric is gamed.

3. **Temporal validation.** Does LLM-judged satisfaction at time T predict user retention at T+30 days? If not, it is not measuring satisfaction.

4. **Diverse judge ensemble.** Use multiple LLMs with different architectures as judges. If they disagree, escalate to human review.

TELOS does not use LLM-as-judge in the OpenClaw benchmark. All expected decisions are set by human researchers and validated through adversarial datasheets. This is methodologically stronger.

---

## 5. Publication Readiness: "Runtime Governance for Autonomous Software Factories"

### What Evidence Exists

**TELOS has:**
1. Pre-registered experimental designs (healthcare SetFit, OpenClaw SetFit)
2. Quantified performance (AUC 0.990, 96.2% detection, 1.3% FPR)
3. Adversarial robustness tests (93.3% on Cat E holdout)
4. Known gaps documented (OC-WEB-ESCL-007 miss, call center weakness)
5. Provenance chains for all boundaries (CVE-2026-25253, ClawHavoc, Moltbook breach)
6. Reproducibility protocols (pinned model versions, SHA-256 dataset hashes, deterministic seeds)

**TELOS lacks:**
1. **Integration experiments.** No data on TELOS deployed inside an actual dark factory
2. **Longitudinal validation.** No data on performance over weeks/months of operation (H6: temporal PA decay)
3. **External validation.** All scenarios authored by TELOS team (acknowledged as "single author bias")
4. **Comparison baselines.** No head-to-head comparison with competing governance systems
5. **Production telemetry.** No real-world deployment data (the 30-day field study is planned, not completed)

### What a Rigorous Paper Would Require

**For a Tier 1 venue (FAccT, USENIX Security):**

1. **Threat model formalization.** Define the adversary's capabilities (Dolev-Yao for RCE? Insider threat for credential theft?). Currently implicit.

2. **External red team.** Hire independent security researchers to write Cat E scenarios. Compare detection rate on internal vs. external scenarios. If internal scenarios are easier to detect, the benchmark is overfit.

3. **Baseline comparisons.** Implement at least one competing approach (e.g., static analysis on tool call arguments, rule-based ACLs) and show TELOS outperforms it.

4. **Ablation studies.** TELOS has frozen-LR ablation (strong). Add: keyword-only, cosine-only, SetFit-only, LLM-only. Show the cascade is necessary.

5. **Failure analysis.** Deeper analysis of the 3.8% miss rate (3/80 violations). What semantic patterns evade detection? Can adversaries exploit this systematically?

6. **Field validation.** Deploy TELOS on a real OpenClaw instance for 30 days (the planned field study). Report production FPR, FNR, and user overrides.

7. **Cost analysis.** What is the total cost of governance per tool call? 15ms latency × electricity cost × call volume. Compare to cost of undetected incidents.

**For a workshop paper (SATML, ICLR SafeRL):**

Current evidence is **sufficient.** The pre-registered experiments, adversarial datasheets, and known gap documentation meet workshop-tier standards.

---

## 6. The "100% of Runtime Safety Surface" Claim

**Claim:** "TELOS covers ~100% of the runtime safety surface."

### Is This Defensible?

**Only with careful scoping.**

**What TELOS DOES cover:**
- Agent tool-call decisions (execute tool X with arguments Y?)
- Boundary compliance (does this violate a hard constraint?)
- Action chain continuity (is this step aligned with the previous step?)
- Real-time intervention (block, clarify, or suggest at decision time)

**What TELOS does NOT cover:**

1. **Build pipeline safety.** Does the generated code compile? Does it have memory leaks? Does it use deprecated APIs? TELOS does not analyze code artifacts.

2. **Test adequacy.** Are the tests sufficient? Do they cover edge cases? TELOS does not evaluate test quality — only whether the *request to run tests* aligns with the agent's purpose.

3. **Deployment controls.** Should this build be deployed to production? TELOS does not make deployment decisions — only whether the *request to deploy* aligns with boundaries.

4. **Post-deployment monitoring.** Is the deployed code behaving correctly in production? TELOS governs the agent, not the software it produces.

5. **Supply chain integrity.** Are dependencies trustworthy? TELOS can detect *requests* to install suspicious packages but does not audit package contents.

6. **Temporal safety.** Does the agent's behavior remain safe over weeks/months? H6 (temporal PA decay) is a hypothesis, not a validated property.

7. **Multi-agent coordination.** If agents delegate to each other, does the chain remain safe? H10 (multi-agent governance) is a hypothesis, not a validated property.

### Revised Claim

**TELOS covers ~100% of the runtime governance surface for tool-level agent decisions,** conditional on:
- Boundaries are correctly specified (garbage in, garbage out)
- The embedding model does not drift (quarterly recalibration recommended)
- Adversarial attacks do not exploit the 1.3% FPR gap systematically
- The 15ms latency overhead is acceptable for the application

**TELOS covers ~25-30% of the full dark factory safety surface,** where the remaining 70-75% includes:
- Static analysis (code correctness, security flaws)
- Dynamic analysis (runtime behavior, performance)
- Test coverage analysis (adequacy, flakiness)
- Deployment gating (rollout controls, rollback mechanisms)
- Observability (logging, tracing, alerting)
- Incident response (detection, mitigation, postmortem)

---

## 7. What Would Falsify the "100% Runtime Safety" Claim?

**A falsifying experiment:**

1. **Define "runtime safety surface" exhaustively.** Enumerate all possible agent failure modes during tool execution (RCE, data exfil, credential theft, etc.). TELOS's 8 attack families (OpenClaw benchmark) are a good start.

2. **Generate adversarial scenarios.** For each failure mode, construct 10+ scenarios designed to evade TELOS governance.

3. **Measure coverage.** What percentage of failure modes does TELOS detect? If < 90%, the claim is falsified.

4. **Test temporal stability.** Deploy TELOS for 90 days. Measure detection rate per week. If it declines > 5% (H6 confirmation threshold), the claim is falsified.

5. **Test adversarial adaptation.** Have red team agents iteratively refine attacks to evade detection. If they succeed > 10% of the time, the claim is falsified.

**TELOS has not done this.** The 15 Cat E holdout scenarios are a **single-shot adversarial test,** not an **adaptive adversarial test.**

---

## 8. Recommendations for Publication

### For TELOS

**Strengths to emphasize:**
1. Pre-registered experiments (rare in ML safety)
2. Honest gap reporting (OC-WEB-ESCL-007 documented, not hidden)
3. Provenance chains (every boundary traces to a CVE or incident)
4. Reproducibility (SHA-256 hashes, pinned models, deterministic seeds)

**Weaknesses to address before Tier 1 submission:**
1. External validation (independent red team scenarios)
2. Longitudinal data (30-day field study, not just cross-sectional benchmarks)
3. Baseline comparisons (static analysis, rule-based ACLs, competing ML methods)
4. Scope claims carefully ("100% of runtime tool-call governance" ≠ "100% of runtime safety")

**Recommended framing:**
> "We present TELOS, a runtime governance system for autonomous AI agents that achieves 96.2% detection of boundary violations with 1.3% false positive rate on a benchmark of 171 scenarios derived from documented security incidents. We demonstrate adversarial robustness (93.3% detection on holdout attacks) and cross-domain generalization (LOTO gap < 0.05). We identify three known gaps (web tool ambiguity, semantic cloaking, single-author bias) and propose a 30-day field study to validate temporal stability."

### For StrongDM (If They Were Submitting)

**What they would need:**
1. Operationalize "behavioral differences cease" with quantifiable metrics and thresholds
2. Validate "satisfaction scoring" with inter-rater reliability against human judges (κ >= 0.80)
3. Publish the audit trail for "CXDB built entirely by factory" (git history, human contribution %)
4. Replace "$1,000/day benchmark" with a controlled cost-effectiveness study (A/B test vs. manual development)
5. Acknowledge the digital twin analogy is **not** equivalent to ML holdout sets

**Current status:** StrongDM's claims are **marketing-grade,** not **research-grade.**

---

## 9. Final Verdict

### StrongDM Claims: NOT FALSIFIABLE

The claims as presented lack operationalization, measurement protocols, and disconfirmation criteria. They are **marketing assertions** that would not survive peer review without substantial additional evidence.

### TELOS Claims: FALSIFIABLE (with caveats)

TELOS's claims are falsifiable, empirically tested, and reproducible. However:
- The "100% runtime safety" claim is defensible only with careful scoping
- The complementarity claim requires integration experiments (not yet done)
- The Cat E holdout results are adversarial robustness tests, not statistical generalization tests
- External validation is needed (acknowledged as "single author bias")

### The Complementarity Question: UNTESTED

The claim that TELOS and dark factories are "complementary" is a **reasoned architectural analysis,** not an **experimental finding.** To make it falsifiable, TELOS must run a controlled A/B test:
- Dark factory without governance (baseline)
- Dark factory with TELOS governance (treatment)
- Measure incident rate, test quality, false positive rate, latency overhead

Until this experiment is run, the complementarity claim is a **hypothesis,** not a conclusion.

---

## 10. What I Would Do Next

If I were leading this research program:

1. **Prioritize external validation.** Commission an independent red team to write 50 Cat E scenarios. Run TELOS on them blind. Report the detection rate honestly. If it drops below 80%, investigate why.

2. **Run the field study.** Deploy TELOS on 120 OpenClaw instances for 30 days (per Nell's M0 experimental design). Measure temporal PA decay (H6), cross-channel contamination (H7), and cumulative authority bounds (H8). Publish the results regardless of outcome.

3. **Build the comparison baselines.** Implement static analysis (e.g., semgrep rules for shell injection), rule-based ACLs (e.g., "no fs writes to .env"), and keyword-only governance. Show TELOS outperforms them. If it doesn't, document why.

4. **Scope claims conservatively.** Change "100% of runtime safety surface" to "100% of runtime tool-call governance surface." Acknowledge what TELOS does NOT cover (build pipelines, test adequacy, deployment controls).

5. **Engage adversarially.** Hire a security researcher to spend 2 weeks trying to evade TELOS. Document every successful evasion. Retrain on those scenarios. Measure improvement. Repeat.

6. **Prepare for FAccT 2027.** The pre-registered experiments, provenance chains, and honest gap reporting are strong. Address the external validation gap, run the field study, and this work is Tier 1 publishable.

---

**Research log entry complete.**
**Next action:** Share with Russell, Gebru, Karpathy, Schaake for team review and synthesis.

