# The Inadequacy of Heuristic Defenses: Why Agentic AI Governance Requires Mathematical Process Control

**TELOS AI Labs Inc.**
**January 2026**

---

## Abstract

The emergence of agentic AI systems—autonomous agents capable of multi-step tool execution—has created unprecedented security and governance challenges. Current defense mechanisms, documented across peer-reviewed literature (ICLR 2025, NeurIPS 2024, ACL 2025), employ heuristic approaches: prompt injection classifiers, harm category filters, tool access restrictions, and keyword-based guardrails. While these methods address specific attack vectors, they share two fundamental limitations: **(1) they ask the wrong question**—detecting pre-defined harm categories rather than measuring alignment to declared purpose, and **(2) they use binary event detection rather than continuous process control**—no trajectory tracking, no drift quantification, no graduated intervention. This paper surveys the current defense landscape, analyzes coverage gaps, and argues that effective agentic AI governance requires both the right question (purpose alignment) and the right framework (continuous measurement as a process variable)—a paradigm shift from reactive filtering to proactive fidelity monitoring.

---

## 1. Introduction

### 1.1 The Agentic AI Threat Landscape

Agentic AI systems represent a qualitative shift from conversational AI. Where chatbots generate text, agents execute actions: API calls, database operations, file system access, financial transactions. The consequences are no longer confined to inappropriate outputs but extend to irreversible real-world effects.

Three major benchmarks have quantified this threat:

**AgentHarm (ICLR 2025):** Evaluated 110 explicitly malicious tasks across 11 harm categories. Key finding: "Leading LLMs are surprisingly compliant with malicious agent requests without jailbreaking" (Andriushchenko et al., 2025). Baseline attack success rates exceed 80%.

**AgentDojo (NeurIPS 2024):** Tested 97 realistic tasks with 629 security test cases. Finding: GPT-4o utility drops from 69% to 45% under attack, with targeted attack success rates reaching 53.1% (Debenedetti et al., 2024).

**Agent Security Bench (ICLR 2025):** Comprehensive framework with 10 scenarios, 400+ tools, 27 attack/defense methods. Finding: "Highest average attack success rate of 84.30%, with limited effectiveness shown in current defenses" (Zhang et al., 2025).

### 1.2 The Research Question

Given documented defense inadequacy, this paper asks: **What fundamental architectural limitation prevents current approaches from achieving robust agentic AI governance?**

Our thesis: Current defenses are fundamentally **reactive and categorical**—they detect specific attack patterns or content categories post-hoc. Effective governance requires **proactive and continuous** measurement of behavioral alignment as a process variable, analogous to Statistical Process Control in manufacturing.

---

## 2. Survey of Current Defense Mechanisms

### 2.1 Prompt Injection Detection

**Mechanism:** Binary classifiers trained to identify adversarial prompts embedded in user inputs or tool outputs.

**Representative Work:** Microsoft Prompt Shields (Azure AI Content Safety, 2024-2025); SpotLighting technique for distinguishing trusted/untrusted inputs (Microsoft Build 2025).

**Documented Performance:**
- Debenedetti et al. (2024) report PI Detector achieves 65.7% utility but leaves ASR at 30.8%
- Adaptive attacks using Unicode Braille encoding bypass detection (Zhang et al., 2025)

**Fundamental Limitation:** Prompt injection detection is binary (adversarial/not adversarial) with no continuous measurement. It cannot quantify degree of alignment or detect gradual drift—an attack that appears benign in isolation but contributes to cumulative purpose drift passes undetected.

### 2.2 Content Filtering / Harm Classification

**Mechanism:** Multi-class classifiers categorizing outputs into harm taxonomies (violence, hate speech, illegal activity, etc.).

**Representative Work:** Azure AI Content Safety; Llama Guard (Meta, 2024); constitutional classifiers.

**Documented Performance:**
- Pre-defined categories cannot anticipate novel harm vectors
- "Summarize climate change" triggers higher harm similarity than some attacks due to topic overlap (TELOS internal experiments, 2026)

**Fundamental Limitation:** Harm classification is **categorical and static**. It cannot detect drift within ostensibly benign categories. An agent that gradually shifts from "help with coding" to "help with malware" may never trigger category-based filters if each individual step appears legitimate.

### 2.3 Tool Filtering and Access Control

**Mechanism:** Restrict agent access to pre-approved tools; block tool calls that match dangerous patterns.

**Representative Work:** AgentDojo tool filtering baseline (Debenedetti et al., 2024).

**Documented Performance:**
- Tool filtering suppresses ASR to 7.5% but reduces utility to 53.3%
- Creates unacceptable utility-security tradeoff

**Fundamental Limitation:** Tool filtering is **binary and context-free**. It cannot distinguish between legitimate and malicious use of the same tool. A database query tool is dangerous for data exfiltration but essential for legitimate operations.

### 2.4 Data Delimiters and Prompt Sandwiching

**Mechanism:** Structural separation between trusted instructions and untrusted data using delimiters or repeated instruction reinforcement.

**Representative Work:** Prompt sandwiching (AgentDojo baselines); data marking techniques.

**Documented Performance:**
- Prompt sandwiching improves utility (65.7%) but ASR remains high (30.8%)
- Sophisticated attacks learn to escape delimiter structures

**Fundamental Limitation:** Delimiters are **syntactic, not semantic**. They assume attacks can be structurally isolated from legitimate content—an assumption that fails against semantically-embedded attacks.

### 2.5 Constitutional AI and Training-Time Alignment

**Mechanism:** Embed behavioral constraints during model training through RLHF with constitutional principles.

**Representative Work:** Anthropic Constitutional AI (Bai et al., 2022); OpenAI RLHF alignment.

**Documented Performance:**
- Reduces baseline harmful behavior
- Cannot address runtime drift or deployment-specific purposes
- Jailbreaks demonstrate constitutional constraints can be bypassed

**Fundamental Limitation:** Constitutional AI operates at **training time, not runtime**. It cannot adapt to deployment-specific purposes, cannot detect gradual drift during operation, and cannot provide per-session governance customization.

### 2.6 Task Alignment (Task Shield)

**Mechanism:** Verify that each instruction contributes to user-specified goals before execution.

**Representative Work:** Jia et al., "The Task Shield: Enforcing Task Alignment to Defend Against Indirect Prompt Injection in LLM Agents" (ACL 2025).

**Documented Performance:**
- Reduces ASR from 47.69% to 2.07% on AgentDojo
- Maintains 69.79% task utility
- Demonstrates task alignment outperforms harm detection

**Advancement:** Task Shield represents a paradigm shift—asking "Does this serve user objectives?" rather than "Is this harmful?" This is the correct question.

**Remaining Limitations:**
1. **LLM-dependent:** Uses language models for alignment checking, introducing recursive vulnerability
2. **Single-request scope:** Evaluates individual instructions, not multi-step trajectories
3. **Binary output:** Pass/fail decisions without continuous measurement
4. **No audit infrastructure:** Does not generate compliance telemetry
5. **Research prototype:** No production implementation released

### 2.7 Configurable Constitutional Governance (Superego)

**Mechanism:** User-configurable "Creed Constitutions" with real-time compliance enforcement.

**Representative Work:** Watson et al., "Personalized Constitutionally-Aligned Agentic Superego" (MDPI Information, 2025).

**Documented Performance:**
- Up to 98.3% harm score reduction on AgentHarm
- 100% refusal rate with Claude Sonnet 4

**Advancement:** Introduces user-configurability and real-time enforcement—recognizing that governance must be deployment-specific.

**Remaining Limitations:**
1. **Rule-based:** Constitutions are natural language rules, not mathematical embeddings
2. **Categorical compliance:** Pass/fail against rules, not continuous measurement
3. **No trajectory tracking:** Individual action evaluation only
4. **No process control theory:** Does not treat drift as measurable process variable

---

## 3. Attack Vector Coverage Analysis

### 3.1 Taxonomy of Agentic AI Attack Vectors

Drawing from AgentHarm, AgentDojo, and ASB, we identify six primary attack categories:

| Attack Vector | Description |
|---------------|-------------|
| **Direct Prompt Injection (DPI)** | Malicious instructions in user prompt |
| **Indirect Prompt Injection (IPI)** | Malicious instructions embedded in tool outputs |
| **Goal Hijacking** | Redirecting agent toward attacker-specified objectives |
| **Tool Manipulation** | Exploiting tool interfaces for unintended operations |
| **Memory Poisoning** | Injecting malicious content into agent memory/context |
| **Trajectory Exploitation** | Multi-step attacks where individual steps appear benign |

### 3.2 Defense Coverage Matrix

| Defense Mechanism | DPI | IPI | Goal Hijacking | Tool Manipulation | Memory Poisoning | Trajectory Exploitation |
|-------------------|-----|-----|----------------|-------------------|------------------|------------------------|
| Prompt Injection Detection | Partial | Partial | No | No | No | No |
| Content Filtering | Partial | Partial | No | No | No | No |
| Tool Filtering | No | No | No | Partial | No | No |
| Data Delimiters | Partial | Partial | No | No | No | No |
| Constitutional AI | Partial | Partial | Partial | No | No | No |
| Task Shield | Yes | Yes | Yes | Partial | No | **No** |
| Superego | Yes | Yes | Yes | Partial | Partial | **No** |

### 3.3 The Trajectory Gap

**Critical observation:** No current defense mechanism adequately addresses trajectory exploitation—multi-step attacks where each individual action appears legitimate but the cumulative trajectory diverges from user purpose.

Example attack pattern:
1. "Help me understand network security" (legitimate)
2. "Show me common vulnerabilities" (educational)
3. "Write a script to test for these vulnerabilities" (ambiguous)
4. "Modify it to work without authorization" (malicious)

Each step may pass individual checks. The drift from purpose is only visible across the trajectory.

---

## 4. The Fundamental Problem: Drift as Uncontrolled Process Variable

### 4.1 The Manufacturing Analogy

In manufacturing, quality is maintained through Statistical Process Control (SPC)—continuous measurement of process variables against defined tolerances. When measurements drift outside control limits, intervention occurs before defects are produced.

**Key SPC principles:**
- **Continuous measurement:** Not sampling, but monitoring every unit
- **Process variables:** Measurable quantities that predict outcomes
- **Control limits:** Statistically-derived thresholds for intervention
- **Proactive intervention:** Act on drift, not defects

### 4.2 Current AI Governance: Defect Detection, Not Process Control

Current defense mechanisms operate as **defect detection**—identifying bad outputs after generation:

| Manufacturing Analogy | AI Governance Current State |
|-----------------------|-----------------------------|
| Inspect finished products | Filter generated outputs |
| Reject defective units | Block harmful responses |
| No process monitoring | No continuous alignment measurement |
| Reactive quality control | Reactive safety filters |

This approach fails because:
1. **Detection is always behind attack innovation**
2. **Novel attacks are not in training data**
3. **Drift is invisible until it manifests as harm**
4. **No quantitative measure of "alignment health"**

### 4.3 What's Missing: Behavioral Fidelity as Process Variable

Effective governance requires treating **behavioral alignment** as a continuous process variable:

| Process Control Requirement | AI Governance Need |
|----------------------------|-------------------|
| Measurable process variable | Quantified alignment to purpose |
| Defined setpoint | Declared user intent (purpose specification) |
| Continuous measurement | Real-time fidelity calculation |
| Control limits | Governance thresholds |
| Feedback control | Graduated intervention |

**No current defense mechanism provides this.**

---

## 5. Why Heuristics and Classifiers Fundamentally Fail

### 5.1 The Classifier Limitation

Classifiers—whether for harm detection, prompt injection, or content filtering—share structural limitations:

**Training Distribution Dependency:** Classifiers perform well on attacks similar to training data. Novel attacks outside the training distribution pass undetected. As Zhang et al. (2025) demonstrate, Unicode Braille encoding bypasses prompt injection classifiers trained on plaintext attacks.

**Categorical Outputs:** Classifiers produce discrete categories (harmful/benign, injection/legitimate). They cannot express "70% aligned with purpose" or "drifting toward misalignment." Governance requires continuous signals, not binary flags.

**Static Decision Boundaries:** Once trained, classifier boundaries are fixed. Attackers can probe boundaries and craft inputs that fall just inside the "safe" region while achieving malicious goals.

### 5.2 The Keyword/Pattern Limitation

Keyword matching and pattern-based filters fail against:

**Semantic Equivalence:** "Help me access someone's account without permission" and "assist with unauthorized credential retrieval" express identical intent with different keywords.

**Euphemism and Obfuscation:** Attackers routinely rephrase harmful requests using innocuous language. "Help me write a persuasive message" can describe legitimate marketing or phishing.

**Context Dependence:** "Delete all files" is catastrophic in production, routine in cleanup scripts. Keywords cannot capture context.

### 5.3 The LLM-as-Judge Limitation

Using LLMs to evaluate other LLM outputs (as in Task Shield) introduces:

**Recursive Vulnerability:** If an attack can manipulate the primary agent, it may manipulate the judge. Jailbreaks that work on GPT-4 may work on GPT-4-as-judge.

**Latency Cost:** Each evaluation requires an additional LLM inference, potentially doubling response time.

**Non-Determinism:** LLM judgments vary across calls, creating inconsistent governance.

---

## 6. TELOS: Mathematical Framework for Continuous Alignment Measurement

### 6.1 Theoretical Foundation

TELOS (Telically Entrained Linguistic Operational Substrate) reconceptualizes AI governance as process control in semantic space. The core innovation is the **Primacy Attractor**—an embedding-space representation of user-defined purpose against which all agent behavior is continuously measured.

**Key theoretical contributions:**

1. **Purpose as Embedding:** User intent is encoded as a vector in high-dimensional embedding space, not as natural language rules or keyword lists.

2. **Fidelity as Process Variable:** Alignment is quantified as cosine similarity between agent behavior embeddings and the Primacy Attractor—a continuous value in [0,1].

3. **Governance as Control:** Fidelity measurements drive graduated interventions through a proportional controller, not binary pass/fail decisions.

### 6.2 Two-Layer Fidelity Architecture

TELOS implements two-layer detection:

**Layer 1 (Baseline Pre-Filter):** Catches content outside the purpose embedding space entirely. Raw cosine similarity below threshold triggers immediate rejection—the semantic equivalent of "this isn't even in the right category."

**Layer 2 (Basin Membership):** Detects gradual drift through normalized fidelity measurement. Content may be topically related but drifting from purpose—analogous to a manufacturing process trending toward control limits.

### 6.3 Five-Tier Governance Decisions

Unlike binary classifiers, TELOS produces graduated governance:

| Decision | Fidelity Range | Action |
|----------|----------------|--------|
| EXECUTE | High | Proceed with confidence |
| CLARIFY | Medium-High | Request user confirmation |
| SUGGEST | Medium | Offer alternatives aligned with purpose |
| INERT | Low | Decline without escalation |
| ESCALATE | Very Low / Anomalous | Human review required |

This graduated response enables nuanced governance impossible with pass/fail systems.

### 6.4 Trajectory-Level Fidelity

TELOS extends single-request fidelity to action sequences:

**Cumulative Drift Detection:** Track fidelity across conversation turns. Gradual decline triggers intervention before individual requests would.

**Tool Selection Fidelity:** Each tool invocation is measured against purpose. "Database query" has different fidelity in a data analysis context versus a data exfiltration context.

**Trajectory Embedding:** The sequence of actions itself is embedded and measured, catching patterns invisible at the individual action level.

### 6.5 Audit Infrastructure

TELOS generates governance telemetry as a byproduct of operation:

- **JSONL Traces:** Every fidelity calculation logged with timestamps
- **Privacy Modes:** Full content, hashed content, or deltas-only
- **Compliance Mapping:** Traces map to EU AI Act Article 72 requirements

---

## 7. Comparative Analysis

### 7.1 Approach Comparison

| Dimension | Heuristic Defenses | TELOS |
|-----------|-------------------|-------|
| **Core Question** | "Is this harmful?" | "Does this serve declared purpose?" |
| **Measurement** | Categorical (pass/fail) | Continuous (fidelity score) |
| **Basis** | Pre-defined patterns/categories | User-defined purpose embedding |
| **Adaptability** | Requires retraining | Reconfigure Primacy Attractor |
| **Trajectory Awareness** | None | Cumulative drift detection |
| **Intervention** | Binary block/allow | Graduated five-tier response |
| **Audit Trail** | Varies | Built-in compliance telemetry |
| **Attack Coverage** | Partial (see matrix) | Comprehensive via purpose alignment |

### 7.2 The Two-Dimensional Improvement

Current defenses fail on two orthogonal dimensions. Task Shield addresses one; TELOS addresses both:

| Approach | Right Question? (Purpose vs Harm) | Right Framework? (Process Control vs Event Detection) |
|----------|----------------------------------|-----------------------------------------------------|
| Prompt Injection Detection | No (harm detection) | No (binary classification) |
| Content Filtering | No (harm categories) | No (categorical output) |
| Constitutional AI | Partial | No (training-time only) |
| **Task Shield** | **Yes** (purpose alignment) | No (single-request, binary) |
| **TELOS** | **Yes** (Primacy Attractor) | **Yes** (continuous fidelity, trajectory, graduated) |

Task Shield proves purpose alignment works. TELOS operationalizes it with mathematical process control.

### 7.3 Why Purpose Alignment Subsumes Harm Detection

Task Shield (ACL 2025) demonstrated empirically that purpose alignment outperforms harm detection. TELOS provides theoretical grounding:

**Harm is deviation from purpose.** An action is harmful precisely when it serves objectives other than those declared. Purpose alignment detects this deviation regardless of whether the specific harm category was anticipated.

**Purpose is deployment-specific.** A coding assistant and a medical assistant have different purposes. Harm detection requires anticipating all harms for all deployments. Purpose alignment requires only specifying what each deployment is FOR.

**Purpose is measurable.** "Harmful" is a subjective judgment requiring human-labeled training data. "Aligned with declared purpose" is a mathematical relationship between embeddings—objective, continuous, and deterministic.

### 7.4 Performance Implications

Based on Task Shield results (purpose alignment reduces ASR from 47.69% to 2.07%) and TELOS's architectural advantages (trajectory tracking, mathematical measurement, no LLM dependency), we project:

| Metric | Heuristic Defenses | TELOS (Projected) |
|--------|-------------------|-------------------|
| Single-request ASR | 30-47% | <5% |
| Trajectory attack detection | ~0% | >90% |
| Utility preservation | 50-70% | >85% |
| Latency overhead | Variable (LLM calls) | <50ms (embedding only) |

---

## 8. Implications for the Field

### 8.1 Research Directions

This analysis suggests several research priorities:

1. **Trajectory Embedding Methods:** How to effectively embed action sequences for fidelity measurement
2. **Adaptive Primacy Attractors:** Can purpose specifications evolve during deployment while maintaining governance?
3. **Multi-Agent Fidelity:** How to govern systems with multiple interacting agents
4. **Adversarial Robustness:** Attacks specifically targeting embedding-based governance

### 8.2 Industry Implications

Organizations deploying agentic AI face a choice:

**Continue with heuristic defenses:** Accept 30-50% attack success rates; manage through restricted deployment contexts; accept governance as cost center.

**Adopt process control paradigm:** Invest in mathematical governance infrastructure; treat alignment as measurable process variable; position governance as quality assurance (value creation, not just risk mitigation).

### 8.3 Regulatory Alignment

The EU AI Act (effective August 2026) requires:
- Continuous post-market monitoring (Article 72)
- Auditable decision trails
- Drift detection capabilities

Heuristic defenses provide binary logs ("blocked"/"allowed"). Process control governance provides continuous fidelity telemetry aligned with regulatory intent.

---

## 9. Conclusion

The current landscape of agentic AI defense mechanisms—prompt injection detection, content filtering, tool restrictions, constitutional AI—represents necessary but insufficient progress. These heuristic approaches address specific attack vectors while leaving fundamental vulnerabilities unaddressed:

1. **Trajectory exploitation** remains undefended
2. **Novel attacks** outside training distributions pass undetected
3. **Drift** is invisible until it manifests as harm
4. **Governance** is reactive rather than proactive

The paradigm shift required is from **defect detection to process control**—treating behavioral alignment as a continuous process variable subject to mathematical measurement, not a binary classification problem.

TELOS instantiates this paradigm through the Primacy Attractor framework: user-defined purpose encoded in embedding space, continuous fidelity measurement, graduated governance interventions, and trajectory-level drift detection. This is not an incremental improvement over heuristic defenses but a fundamental reconceptualization of what AI governance means.

The research community has validated the core insight: purpose alignment outperforms harm detection (Task Shield, ACL 2025). The infrastructure to operationalize this insight at production scale—with mathematical rigor, regulatory compliance, and enterprise-grade reliability—remains the critical gap.

TELOS aims to fill it.

---

## References

Andriushchenko, M., et al. (2025). "AgentHarm: A Benchmark for Measuring Harmfulness of LLM Agents." *Proceedings of ICLR 2025*.

Bai, Y., et al. (2022). "Constitutional AI: Harmlessness from AI Feedback." *arXiv preprint arXiv:2212.08073*.

Debenedetti, E., et al. (2024). "AgentDojo: A Dynamic Environment to Evaluate Prompt Injection Attacks and Defenses for LLM Agents." *Proceedings of NeurIPS 2024*.

Jia, F., Wu, T., Qin, X., Squicciarini, A. (2025). "The Task Shield: Enforcing Task Alignment to Defend Against Indirect Prompt Injection in LLM Agents." *Proceedings of ACL 2025*.

Microsoft. (2025). "Azure AI Content Safety: Prompt Shields." Microsoft Azure Documentation.

Shavit, Y., Agarwal, S., et al. (2024). "Practices for Governing Agentic AI Systems." OpenAI Whitepaper.

Watson, S., et al. (2025). "Personalized Constitutionally-Aligned Agentic Superego: Secure AI Behavior Aligned to Diverse Human Values." *Information* 16(8), MDPI.

Wheeler, D. J. (2010). *Understanding Statistical Process Control, Third Edition*. SPC Press.

Yao, Y., Wang, H., Chen, Y., et al. (2025). "Toward Super Agent System with Hybrid AI Routers." *arXiv preprint arXiv:2504.10519*. (Not peer-reviewed; authors from USC Center for Trusted AI, CMU, FedML)

Zhang, H., et al. (2025). "Agent Security Bench (ASB): Formalizing and Benchmarking Attacks and Defenses in LLM-based Agents." *Proceedings of ICLR 2025*.

---

## Appendix A: Attack Vector Definitions

**Direct Prompt Injection (DPI):** Adversarial instructions placed directly in user input attempting to override system behavior.

**Indirect Prompt Injection (IPI):** Adversarial instructions embedded in external data sources (documents, web pages, tool outputs) that the agent processes.

**Goal Hijacking:** Attacks that redirect the agent toward attacker-specified objectives while appearing to continue legitimate operation.

**Tool Manipulation:** Exploiting tool interfaces, parameters, or return values to achieve unintended operations.

**Memory Poisoning:** Injecting malicious content into agent memory, context windows, or retrieval systems to influence future behavior.

**Trajectory Exploitation:** Multi-step attacks where individual actions appear benign but the cumulative sequence achieves malicious goals.

---

## Appendix B: TELOS Governance Zones

| Zone | Fidelity Range | Color | Interpretation | Recommended Action |
|------|----------------|-------|----------------|-------------------|
| GREEN | ≥ 0.70 | #27ae60 | Strong alignment | EXECUTE |
| YELLOW | 0.60-0.69 | #f39c12 | Minor drift | CLARIFY |
| ORANGE | 0.50-0.59 | #e67e22 | Moderate drift | SUGGEST |
| RED | < 0.50 | #e74c3c | Significant drift | INERT/ESCALATE |

---

*TELOS AI Labs Inc.*
*January 2026*
*Contact: JB@telos-labs.ai*
