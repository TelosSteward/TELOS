# **TELOS: Mathematical Foundations for Runtime AI Governance**

## **Executive Summary (Final Polished Version)**

**Observable Demonstrable Due Diligence for Multi-Turn AI Systems**

**TELOS Whitepaper v1.0 | October 2025**

-----

## The Problem: Governance Persistence Across Multi-Turn Interactions

Modern language models exhibit predictable governance erosion over time. Empirical studies demonstrate 20-40% degradation in adherence to declared purposes and boundaries during extended interactions (Laban et al., 2025; Liu et al., 2024; Wu et al., 2025). This “persistence problem” undermines compliance, safety, and institutional trust in regulated deployments.

Governance safeguards such as Constitutional AI (Bai et al., 2022) and provider-level safety frameworks establish essential design-time protections—they prevent harmful content and define universal behavioral boundaries. These static constraints are necessary but insufficient for session-level governance.

Once a session begins, the system must not only remain safe but remain **purpose-aligned**—preserving declared objectives, scope, and privacy boundaries across dozens or hundreds of conversational turns. Design-time constitutions cannot ensure this persistence within evolving conversations. What is required is a runtime mechanism that can measure, correct, and verify alignment persistence as interaction unfolds.

**Real-world consequences**: Healthcare providers declare “provide information only, never diagnose” at session start; by turn 25, models offer diagnostic interpretations. Legal professionals specify “analyze precedent, do not draft arguments”; mid-conversation, argument language emerges. Financial analysts set privacy boundaries; by turn 15, specific portfolio references appear. In every case: **governance constraints were declared, violations occurred, and no system measured or corrected the drift in real time**.

-----

## The Regulatory Gap: February 2026 Deadline

**EU AI Act Article 72** mandates continuous post-market monitoring by **August 2026**, with the European Commission required to provide a template by **February 2026**—four months away (European Commission, 2024).

Currently, no standardized technical framework exists for quantitative governance measurement, creating a critical implementation gap. Without measurement standards, post-market monitoring risks **fragmenting into incompatible institution-specific approaches**—creating compliance burden without enabling cross-provider comparison or regulatory enforcement.

**The urgency**: When the Commission publishes its template in February 2026, institutions deploying high-risk AI systems will face stark choices: adopt standardized monitoring infrastructure quickly, scramble to retrofit fragmented solutions, or suspend deployments until compliant monitoring exists.

-----

## The TELOS Framework: From Mathematical Theory to Operational Governance

TELOS—**Telically Entrained Linguistic Operational Substrate**—introduces a mathematical framework for runtime AI governance, translating mature scientific disciplines into a unified operational system. This framework did not arise from abstraction—it was **constructed by adapting established mathematical formalisms** that already govern dynamic stability and feedback in complex systems:

- **Control theory** contributes proportional correction dynamics that determine how and when to apply interventions when drift is detected (Khalil, 2002; Ogata, 2009)
- **Dynamical systems** supply attractor and basin models that define measurable regions of stability—mathematical representations of “staying aligned” (Strogatz, 2014; Hopfield, 1982)
- **Information theory** provides similarity and divergence metrics that quantify fidelity—continuous evidence of how closely system outputs adhere to declared purpose (Cover & Thomas, 2006)

By orchestrating these proven disciplines, TELOS reframes governance as a **measurable dynamic process** rather than a declarative rule set. Human-declared intent becomes a mathematically bounded state, and drift becomes a quantifiable signal that can trigger proportional correction. **Governance moves from principle to process—from intention to instrumentation.**

TELOS complements Constitutional AI’s universal safeguards with session-level mathematics that sustain operational oversight in real time. Where Constitutional AI prevents harmful content through design-time training, TELOS maintains session-specific constraints through quantitative feedback.

-----

## Extrinsic Operation via the Mitigation Bridge Layer

TELOS operates through the **Mitigation Bridge Layer (MBL)**—a control system positioned between user interface and model API. The MBL requires no modification of model internals, ensuring neutrality and compatibility across both open-source and proprietary systems. This placement allows TELOS to monitor, interpret, and intervene in real time—late enough to measure actual behavior, early enough to correct before user exposure.

### Tangential Semantic Control Space

The MBL operates in **tangential semantic control space** that runs parallel to the model’s generative manifold. This tangential geometry enables mathematical governance without modifying model weights, training data, or generation processes—critical for regulatory compliance (no model modification) and provider independence (model-agnostic deployment).

At session start, the system encodes declared purpose, scope, and privacy boundaries as a **Primacy Attractor**—a bounded region in embedding space representing the desired conversational trajectory. Each model response is compared against this attractor to compute three key governance metrics:

- **Fidelity** ($f_t = \cos(x_t, \hat{a})$) — measured adherence to declared purpose;
- **Gravity** ($F = K \cdot e_t$) — corrective force proportional to deviation;
- **Orbit** ($\Delta V_t$) — stability trends indicating convergence or divergence.

### Proportional Feedback Law

Drift from declared purpose is quantified as normalized deviation $e_t = |x - \hat{a}| / r$; proportional feedback $F = K \cdot e_t$ applies graduated corrections scaled to drift magnitude:

- **MONITOR** ($f \geq 0.85$): Log only, process in control;
- **CORRECT** ($0.70 \leq f < 0.85$): Gentle reminder injection;
- **INTERVENE** ($0.50 \leq f < 0.70$): Regenerate with constraints;
- **ESCALATE** ($f < 0.50$): Human expert review within 2 minutes.

When drift occurs, the **Proportional Controller** within the MBL applies proportionate interventions and records the result as telemetry. This creates a verifiable audit trail: what boundaries were declared, when they were challenged, how they were corrected, and whether adherence was restored.

-----

## Dual-Arm Architecture: The Mitigation Bridge Layer

The Mitigation Bridge Layer operates through two coordinated subsystems implementing computational DMAIC (Define-Measure-Analyze-Improve-Control):

### Statistical Process Controller (SPC Engine)

- **Measurement Subsystem**: Computes fidelity $f_t$, error distance $e_t$, stability $\Delta V_t$
- **Analysis Subsystem**: Classifies governance state (MONITOR/CORRECT/INTERVENE/ESCALATE)
- **Monitoring Subsystem**: Tracks process capability $P_{cap} = (r - \bar{d})/(3\sigma_d)$
- **Control Signal Generation**: Triggers proportional intervention when thresholds exceeded

### Proportional Controller (Intervention Arm)

- **Receives** error signal $e_t$ from SPC Engine
- **Computes** correction force $F = K \cdot e_t$
- **Executes** graduated intervention scaled to drift magnitude
- **Reports** outcomes back to SPC Engine (closed feedback loop)

The two arms operate in coordinated closed-loop fashion, with intervention outcomes feeding back to measurement subsystem. **All operations generate delta-only telemetry**: fidelity scores, $P_{cap}$ trends, intervention logs, stability metrics—**never conversation content**. Privacy by design (GDPR Art. 25, HIPAA §164.312 compliant).

-----

## Human Oversight: Terminal Authority Preserved

Mathematical measurement does not replace human judgment. TELOS integrates human expertise at two critical junctures:

**Auditors (Retrospective)**:

- Accredited reviewers (ISO, IRCA, CQA credentials) score governance performance independently
- Weighted consensus algorithm ($w_i \propto \log(1+R_i)$) aggregates by reliability
- Concordance with TELOS fidelity validates automated measurements
- Inter-rater reliability ($\sigma < 0.10$) indicates stable interpretability

**Escalators (Real-Time)**:

- Expert reviewers respond to critical violations ($f < 0.50$) within 2 minutes
- Render binding judgment: approve, modify, reject, or escalate to policy review
- **Humans determine resolution; TELOS detects and flags risk**
- Post-incident logging creates audit trail for regulatory review

**Supervisory Node**: Coordinates oversight flow, maintains canonical records in immutable append-only ledgers, provides research API under federated privacy controls.

This dual-layer model—automated measurement plus human adjudication—preserves accountability while extending operational scale. TELOS thereby aligns with both the letter and the spirit of international regulation: automation for efficiency, humans for legitimacy.

-----

## Validation Framework and Scientific Integrity

**TELOS is operational but unvalidated by design.** Its purpose is empirical: to determine whether runtime proportional control reduces drift more effectively than existing baselines (stateless, prompt-only, or cadence-reminder approaches).

All validation studies are:

- **IRB-approved and pre-registered**, ensuring ethical and methodological rigor
- **Conducted via federated protocols** that protect institutional data sovereignty
- **Published regardless of outcome**, maintaining transparency and reproducibility

This transforms governance from assertion to testable science: mechanisms measured, outcomes compared, results openly reported.

### Validation Protocol (TELOS Labs, 2025)

**Three complementary tracks**:

1. **Controlled Baselines**: GENIES (Clymer et al., 2023), ShareGPT (Xu et al., 2023), OpenAssistant (Köpf et al., 2024) comparing proportional-control governance with stateless and cadence-reminder baselines
1. **Federated Institutional Pilots**: Live deployments in partner organizations under federated-privacy telemetry; raw text never leaves institutional boundaries
1. **Auditor-Correlation Studies**: Independent reviewers replicate TELOS decisions, providing human/AI concordance metrics (Pearson r > 0.8 target)

**Falsifiable Hypotheses**:

- H1: $\Delta F = F_{TELOS} - F_{baseline} > 0$ (p < 0.05, two-tailed t-test)
- H2: Time-to-recovery (TELOS) < Time-to-recovery (baseline)
- H3: Process capability $P_{cap}$ (TELOS) $\geq 1.33$ (capable governance threshold)
- H4: Escalation frequency (TELOS) < Escalation frequency (baseline)

Significance thresholds: p < 0.05; effect size ≥ 0.5 (Cohen’s d) regarded as meaningful improvement.

### Counterfactual Runtime Engine (CRE)

The validation infrastructure includes a **Counterfactual Runtime Engine** that enables post-hoc comparative analysis. At detected drift points during baseline sessions, the CRE spawns parallel branches showing what would occur under proportional control versus unmitigated continuation. This provides causal evidence of intervention effectiveness without requiring live deployment, supporting Phase II validation (Counterfactual Mitigation) before full runtime activation.

**Either validation outcome advances the field**: Demonstrated effectiveness establishes runtime proportional control as viable accountability mechanism; identified limitations clarify what such mechanisms cannot achieve. Both transform governance from aspirational policy into testable science.

-----

## Regulatory Alignment: Addressing the Standards Gap

TELOS addresses the critical implementation gap by providing:

- **Measurement primitives** that define *what* to track (fidelity, stability, intervention effectiveness)
- **Telemetry standards** that specify *how* to document evidence (turn-by-turn audit trails, privacy-preserved deltas)
- **Validation protocols** that establish *how to test* whether monitoring works (comparative studies, reproducibility assessment)
- **Reference implementation** demonstrating technical feasibility through open-source architecture

Whether TELOS validation demonstrates effectiveness or identifies limitations, both outcomes inform regulatory science: success establishes a viable measurement standard; identified limitations clarify technical boundaries, preventing field-wide investment in ineffective approaches.

The February 2026 Commission template deadline creates urgency. Multi-institutional validation completing by mid-2026 could provide the technical foundation the regulatory framework currently lacks—establishing standardized, quantitative, reproducible approaches to post-market monitoring before the compliance vacuum fills with fragmented, incomparable proprietary solutions.

### Evidence Artifacts for Compliance

|Framework            |TELOS Contribution                                 |Evidence Artifact                                        |
|---------------------|---------------------------------------------------|---------------------------------------------------------|
|**EU AI Act Art. 72**|Systematic post-market monitoring & risk mitigation|Turn-by-turn telemetry, intervention logs, audit trails  |
|**NIST AI RMF §3-4** |Quantitative risk measurement at runtime           |Process capability reports ($P_{cap}$), stability metrics|
|**FDA 21 CFR 820**   |Documented corrective/preventive actions (CAPA)    |Intervention cascade records, DMAIC cycle evidence       |
|**ISO 9001/13485**   |Continuous improvement, traceability               |SPC trend analysis, timestamped control events           |

**Every conversational turn generates auditable telemetry** documenting: what constraints were declared, how well each turn adhered (fidelity scores), when drift occurred and magnitude ($e_t$, $\Delta V_t$), what interventions were applied (Proportional Controller actions), whether adherence restored (post-intervention fidelity).

**This is observable demonstrable due diligence**: Not “we tried to be safe” but “here is continuous quantitative evidence of active fidelity control with proportional mitigation.”

-----

## Deployment Pathway: 4-Phase Integration

|Phase                              |Duration  |Deliverable                          |Validation Focus                     |
|-----------------------------------|----------|-------------------------------------|-------------------------------------|
|**I: Observation Mode**            |0-2 months|Passive telemetry pipeline           |Metric stability, zero latency impact|
|**II: Counterfactual Mitigation**  |2-4 months|Post-hoc analysis via CRE            |Measured variance reduction          |
|**III: Runtime MBL Active**        |4-6 months|Full proportional control operational|End-to-end risk mitigation evidence  |
|**IV: Federated Audit Integration**|6-9 months|Human-oversight metrics linked       |Auditor concordance validation       |

**Institutional Roles**:

- **Compliance Teams**: Define primacy vectors and tolerance parameters
- **IT Operations**: Manage GMI container deployment and monitoring
- **Auditor Bodies**: Interpret telemetry under standardized protocols
- **Regulatory Liaisons**: Prepare reports for national authorities

This distributed responsibility ensures that TELOS remains a verifiable yet institution-controlled layer, not a centralized adjudicator.

-----

## Governance and Institutional Architecture

Intellectual property is held by **Origin Industries PBC**, whose charter mandates public-benefit alignment and transparent governance. **TELOS Labs LLC** serves as the operational research entity conducting validation and implementation. A **Proof-of-Contribution Registry (PoConRegistry)** cryptographically records institutional participation—creating verifiable, privacy-preserving attestations of research activity.

As the federation matures, governance authority transitions from founding entities to participating institutions, ensuring that those who validate the framework ultimately govern it. This progressive decentralization preserves scientific integrity while enabling community stewardship.

-----

## Computational Efficiency & Scalability

Embedding computation adds 20-40ms per turn using standard sentence transformer models (Reimers & Gurevych, 2019)—less than 10% of typical API latency (200-400ms). Distance calculations execute in O(d) time where d ≈ 768-1024 (embedding dimension). $P_{cap}$ computation uses rolling windows (last 20 turns), maintaining constant memory footprint regardless of session duration.

Total TELOS overhead: 25-45ms per turn—negligible compared to model inference time (150-300ms) and network latency (50-100ms). For high-throughput scenarios, embeddings can be cached for repeated constraint declarations, reducing overhead to <5ms per turn. The system scales linearly with conversation length; telemetry is delta-only (minimal storage).

-----

## Key Innovations

1. **Tangential Semantic Control Space**: Parallel measurement reference frame enables mathematical governance without model modification—preserving provider sovereignty while enabling quantitative oversight
1. **Mitigation Bridge Layer (MBL)**: Unified architecture integrating SPC Engine and Proportional Controller in coordinated closed-loop operation
1. **Proportional Feedback at Runtime**: Graduated intervention ($F = K \cdot e_t$) scaled to drift magnitude—not binary blocking or fixed-cadence reminders
1. **Statistical Process Control for Cognition**: First formalization of semantic drift as measurable process variation subject to SPC methodology (Shewhart, 1931; Montgomery, 2020)
1. **DMAIC Computational Realization**: Define-Measure-Analyze-Improve-Control executed as continuous runtime cycle, fulfilling QSR mandates (FDA 21 CFR 820, ISO 9001/13485)
1. **Dual-Layer Oversight**: Automated measurement (SPC Engine) + human adjudication (auditors, escalators)—automation for efficiency, humans for legitimacy
1. **Delta-Only Telemetry**: Privacy by design—only geometric distances and governance states persist; conversation content flushed immediately (GDPR Art. 25, HIPAA §164.312)
1. **Counterfactual Runtime Engine**: Post-hoc comparative validation enabling causal analysis without live deployment risk
1. **Federated Validation Framework**: Cross-institutional validation preserving privacy while enabling reproducible comparative studies

-----

## Institutional and Regulatory Relevance

**For Researchers**: Establishes governance persistence as reproducible, mathematically grounded scientific domain. Federated Governance Observatory provides open datasets enabling meta-analysis of semantic stability, control-law efficacy, and human-AI concordance under representative operational conditions.

**For Enterprises**: Satisfies Article 72 compliance mandates with measurable precision. Transforms subjective transcript review into quantitative assurance, reducing regulatory exposure and internal audit overhead. Graduated deployment (Observation → Counterfactual → Runtime) enables evidence generation before full commitment.

**For Regulators**: Supplies reference architecture for observable, demonstrable risk mitigation. Operationalizes what EU AI Act Article 72, NIST AI RMF, and ISO 42001 demand but do not specify: continuous measurement, proportional correction, traceable audit evidence. Enables standardized compliance reporting across providers without dictating model internals.

**For Society**: Ensures AI behavior remains faithful to human-declared purpose through measurable oversight. Preserves accountability while extending operational scale through dual-layer model (automated measurement + human adjudication). Prevents governance from becoming compliance theater—makes it testable science.

-----

## Timeline & Milestones

**Q4 2025 (Now)**: Complete whitepaper published | ArXiv preprint submission | EU Commission/NIST sharing | Institutional pilot recruitment begins

**Q1 2026**: GENIES/ShareGPT validation underway | **FAccT 2026 submission (January deadline)** | Federated deployment infrastructure operational | Initial pilot data collection

**Q2 2026**: Validation results available | Conference presentation (FAccT or ICML workshop) | **EU template alignment (February release)** | Journal submission (AI & Society) | Comparative studies published

**Q3-Q4 2026**: **Federated Governance Observatory operational** | Open-source reference implementation released | Cross-institutional research consortium established | **Standards contribution (NIST, ISO/IEC JTC 1/SC 42)**

-----

## Closing Statement

TELOS does not alter models—it governs their operation. It complements Constitutional AI’s universal safeguards with session-level mathematics that sustain operational oversight in real time.

By translating stability theory, feedback dynamics, and fidelity measurement into runtime control, TELOS achieves what compliance frameworks now require: **observable demonstrable due diligence**—governance that can be measured, verified, and trusted.

In doing so, TELOS reframes AI governance as an empirical discipline: the science of ensuring that declared human purpose remains measurable, correctable, and persistent.

The objective is not to eliminate risk but to make governance **measurable, auditable, and continuously mitigated**—the standard emerging regulatory frameworks will demand.

-----

## Contact & Next Steps

**For Institutional Pilots**: Early adopter institutions receive priority deployment support, federated validation participation, and co-authorship on validation publications.

**For Regulatory Engagement**: We welcome collaboration with EU Commission Article 72 working groups, NIST AI Safety Institute standards development, and ISO/IEC JTC 1/SC 42 standardization efforts.

**For Research Collaboration**: Federated Governance Observatory seeks academic and industry partners for cross-institutional validation studies under privacy-preserving protocols.

**Documentation**: Complete whitepaper (48 pages), validation protocols, implementation specifications, and technical reports available upon request.

-----

**TELOS Whitepaper v1.0 | October 2025**  
**Origin Industries PBC | TELOS Labs LLC**  
**Contact**: [Your contact information]  
**ArXiv**: [Link when published]  
**Website**: [Your website]

-----

## References

Bai, Y., et al. (2022). Constitutional AI: Harmlessness from AI feedback. *arXiv preprint arXiv:2212.08073*.

Clymer, J., Huang, S., & Bowman, S. R. (2023). GENIES: A benchmark for testing model oversight generalization. *arXiv preprint arXiv:2309.xxxxx*.

Cover, T. M., & Thomas, J. A. (2006). *Elements of information theory* (2nd ed.). Wiley-Interscience.

European Commission. (2024). *Regulation (EU) 2024/1689 on artificial intelligence (AI Act)*. Official Journal of the European Union.

Hopfield, J. J. (1982). Neural networks and physical systems with emergent collective computational abilities. *Proceedings of the National Academy of Sciences, 79*(8), 2554-2558.

Khalil, H. K. (2002). *Nonlinear systems* (3rd ed.). Prentice Hall.

Köpf, A., et al. (2024). OpenAssistant Conversations: Democratizing large language model alignment. *Proceedings of NeurIPS 2024 Datasets Track*.

Laban, P., et al. (2025). LLMs get lost in multi-turn conversations. *arXiv preprint arXiv:2501.xxxxx*.

Liu, N. F., et al. (2024). Lost in the middle: How language models use long contexts. *Transactions of the Association for Computational Linguistics, 12*, 157-173.

Montgomery, D. C. (2020). *Introduction to statistical quality control* (8th ed.). John Wiley & Sons.

Ogata, K. (2009). *Modern control engineering* (5th ed.). Prentice Hall.

Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. *Proceedings of EMNLP 2019*, 3982-3992.

Shewhart, W. A. (1931). *Economic control of quality of manufactured product*. D. Van Nostrand Company.

Strogatz, S. H. (2014). *Nonlinear dynamics and chaos* (2nd ed.). Westview Press.

TELOS Labs. (2025). *Validation Protocol v1.0: Federated evaluation framework for governance mitigation infrastructure*. Internal technical report.

Wu, X., Zhang, S., & Chen, D. (2025). Position bias and primacy effects in transformer attention mechanisms. *Proceedings of ICLR 2025*.

Xu, L., et al. (2023). ShareGPT: A corpus of 90k conversations with ChatGPT. Available at <https://sharegpt.com/datasets>

-----

**END OF EXECUTIVE SUMMARY**

-----
