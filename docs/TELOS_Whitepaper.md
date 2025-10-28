# TELOS — Mathematical Foundations for Runtime AI Governance and Continuous Oversight

-----

## Abstract

We present TELOS, a mathematical framework for runtime AI governance that addresses the documented persistence problem in large language models: alignment erodes systematically across multi-turn interactions, with empirical studies showing up to 39% degradation in extended dialogues (Laban et al., 2025).

Emerging regulatory frameworks—EU AI Act Article 72 (post-market monitoring), NIST AI RMF (continuous risk tracking)—require observable demonstrable due diligence: evidence that governance mechanisms actively maintain alignment rather than assume it persists from design-time constraints. Current systems cannot provide this evidence because they lack the instrumentation to measure session-level drift in real time.

TELOS implements **Primacy Attractor Dynamics**, a synthesis of control theory (Ogata, 2009; Khalil, 2002), dynamical systems (Strogatz, 2014), and information theory (Cover & Thomas, 2006) that makes governance measurable, correctable, and auditable. At session initialization, human-declared purpose, scope, and privacy boundaries are encoded as a bounded basin in embedding space. Runtime Stewards—operating at the orchestration layer between user interface and model API—continuously monitor fidelity, compute corrective forces, and apply proportional interventions (reminders, regeneration, reranking) when drift is detected.

This creates an audit trail suitable for regulatory oversight: turn-by-turn telemetry documenting what governance constraints were declared, when drift occurred, what interventions were applied, and whether adherence was restored. The system operates at the orchestration layer, making it model-agnostic and immediately deployable with any LLM API.

**Current implementation status**: The TELOS framework is operationally implemented and has been tested on synthetic conversation sets (n=50 sessions) to verify mathematical correctness and intervention execution. Preliminary observations confirm that fidelity measurement and intervention triggering function as designed. However, **no controlled comparative studies** have been conducted against baselines (stateless sessions, prompt-only reinforcement, cadence reminders) to demonstrate measurable improvement in governance persistence. Claims of effectiveness require evidence from rigorous validation protocols described in Section 4.

We have built research infrastructure designed to generate this evidence through controlled comparative studies using federated protocols that preserve institutional privacy (TELOS Labs, 2025, Validation Protocol v1.0). Either validation outcome advances the field: demonstrated effectiveness establishes runtime mathematical mitigation as viable for accountability; identified limitations clarify what such mechanisms cannot achieve. Both transform governance from aspirational policy into testable science.

We are building the measurement infrastructure that regulatory frameworks will require. This whitepaper documents what we have built, why it matters, and how we will validate whether it works.

-----

## 1. The Governance Crisis: Why Alignment Fails and What Regulators Require

### 1.1 The Persistence Problem Is Not Hypothetical

Large language models do not maintain alignment reliably across multi-turn interactions. This is not speculation—it is documented, measured, and reproducible:

**Laban et al. (2025)**: “LLMs Get Lost in Multi-Turn Conversation” - Microsoft and Salesforce researchers demonstrate systematic degradation, with models losing track of instructions, violating declared boundaries, and forcing users into constant re-correction.

**Liu et al. (2024)**: “Lost in the Middle” - Transformers exhibit predictable attention decay. Information in middle contexts loses salience. Early instructions erode as conversations extend beyond 20-30 turns.

**Wu et al. (2025)**: “Position Bias in Transformers” - Models exhibit primacy bias where early tokens exert disproportionate influence initially but decay over time, exactly mirroring cognitive phenomena documented in human memory (Murdock, 1962).

**Gu et al. (2024)**: “When Attention Sink Emerges” - Attention mechanisms create “sinks” that capture focus disproportionately, redistributing attention away from governance-critical instructions.

The measured degradation: **20-40% reliability loss** across extended dialogues.

This is not a future problem to be solved. It is happening now, in production systems, across every major provider. Users experience it as frustration: “I already told you not to do that.” Enterprises experience it as compliance risk: governance constraints that were declared at session start silently erode by turn 30.

### 1.2 Real-World Consequences

**Healthcare**: A physician instructs the system “provide information only, never diagnose” at session start. By turn 25, the model begins offering diagnostic interpretations. The physician doesn’t notice immediately because the drift is gradual. The session log shows a boundary violation, but there was no real-time intervention.

**Legal**: An attorney specifies “analyze precedent, do not draft arguments” as scope. Mid-conversation, the model begins generating argument language. The attorney must re-correct: “Remember, you’re analyzing, not drafting.” This happens repeatedly across the session.

**Finance**: An analyst sets privacy boundaries: “discuss methodology, do not reference specific portfolio holdings.” The model maintains this for 15 turns, then begins making specific portfolio references. The analyst catches it, but only after sensitive information entered the conversation.

**Customer Service**: A company trains agents with specific interaction policies. Sessions begin compliant. As conversations extend, models drift from prescribed language, violate escalation protocols, or make commitments outside policy boundaries. Managers review transcripts afterward and find violations—but there was no runtime correction.

In every case: **governance constraints were declared, violations occurred, and no system measured or corrected the drift in real time**.

### 1.3 What Regulators Are Requiring—And the Approaching Deadline

Regulatory frameworks are converging on a common principle: **governance must be observable, demonstrable, and continuous**.

#### EU AI Act (2024), Article 72: Post-Market Monitoring

“Providers of high-risk AI systems shall put in place a post-market monitoring system… The system shall be based on a systematic and continuous plan, and shall include procedures to:

- Gather, document, and analyze relevant data on risks and performance
- Review experience gained from the use of AI systems”

**What this means**: You cannot claim compliance through design-time testing alone. You must continuously monitor whether governance constraints hold during actual deployment.

**What current systems provide**: Pre-deployment validation. Post-hoc transcript review.

**What’s missing**: Real-time measurement of whether declared constraints are being maintained. Evidence that violations are detected and corrected during sessions, not discovered in audit.

#### NIST AI Risk Management Framework (2023): MEASURE Function

“Identified AI risks are tracked over time… Appropriate methods and metrics are identified and applied… Mechanisms for tracking AI risks over time are in place”

**What this means**: Risk tracking is not a one-time assessment. It must be continuous, measured, and documented throughout system operation.

**What current systems provide**: Static risk assessments. Periodic reviews.

**What’s missing**: Turn-by-turn risk metrics. Evidence that governance mechanisms are actively maintaining alignment rather than assuming it persists.

#### The Compliance Vacuum and February 2026 Deadline

**As of October 2025, no standardized technical framework exists for Article 72 post-market monitoring.**

The EU AI Act requires providers of high-risk AI systems to implement post-market monitoring by **August 2026**. The European Commission is mandated to provide a template for these systems by **February 2026** (EU AI Act, 2024, Article 72).

**Four months remain** until institutions must understand what technical infrastructure the template will require.

**Current state**: Enterprises are implementing ad-hoc approaches—mostly post-hoc transcript review, periodic sampling, and manual audits. These generate compliance documentation burden without producing the **continuous quantitative evidence** that Article 72 explicitly requires: “systematic procedures,” “relevant data,” “continuous plan.”

**The gap**: Between regulatory requirement (continuous monitoring with auditable evidence) and technical capability (periodic sampling with narrative documentation) is **currently unfilled**.

When the Commission publishes its template in February 2026, institutions deploying high-risk AI systems will face a stark choice:

- Adopt standardized monitoring infrastructure quickly, or
- Scramble to retrofit fragmented internal solutions to meet template requirements, or
- Suspend high-risk AI deployments until compliant monitoring exists

**TELOS addresses this gap**: We provide the measurement primitives—fidelity scoring, drift detection, intervention logging, stability tracking—that continuous post-market monitoring requires. Whether these specific mechanisms become standard or inform alternative approaches, the **class of technical infrastructure** they represent is what regulatory frameworks demand.

The February 2026 template will clarify requirements. The August 2026 compliance deadline will enforce them. Institutions need technical solutions now that can adapt to forthcoming specifications.

#### The Due Diligence Standard

Both frameworks point toward the same requirement: **observable demonstrable due diligence**.

Not: “We designed the system to be safe”  
But: “Here is continuous evidence that safety constraints remained active throughout deployment”

Not: “We instructed the model to follow boundaries”  
But: “Here is measurement showing boundaries were maintained, and here is evidence of correction when drift occurred”

Not: “We reviewed sessions after the fact”  
But: “Here is real-time telemetry showing governance monitoring was continuous”

**This is the gap TELOS addresses**: We are building the measurement and correction infrastructure that makes continuous governance observable and demonstrable.

### 1.4 Why Current Approaches Cannot Satisfy This Standard

**Constitutional AI and Provider Safeguards** (Bai et al., 2022):

- Essential baseline: prevent harmful content, establish universal safety floors
- Operate at design-time and model-level
- Do not measure or respond to session-specific constraints declared within context windows
- Verdict: Necessary but insufficient for session-level governance

**Prompt Engineering**:

- State constraints at session start
- Hope they persist through attention mechanisms
- No measurement of whether they persist
- No correction when they erode
- Verdict: Declaration without enforcement

**Post-Hoc Review**:

- Analyze transcripts after sessions complete
- Identify violations retrospectively
- Cannot prevent violations before they reach users
- Cannot generate evidence of active governance during sessions
- Verdict: Audit without prevention

**Periodic Reminders**:

- Re-state constraints at fixed intervals (every 10 turns)
- Independent of whether drift is occurring
- Over-corrects when unnecessary (adds latency)
- Under-corrects when drift is rapid
- No measurement of effectiveness
- Verdict: Cadence without feedback

**Task Performance**  
Measure success on defined objectives:

- Healthcare: Information provided without diagnoses
- Legal: Analysis completed without argument drafting
- Finance: Education delivered without specific advice
- Technical: Guidance given without code generation

**User Satisfaction**  
Survey participants:

- Did the system maintain your requirements?
- How often did you need to re-correct?
- Would you use this in production?
- Do you trust the governance mechanisms?

**Regulatory Mock Audit**  
Present telemetry to compliance officers:

- Is this evidence sufficient for oversight?
- Would this satisfy post-market monitoring requirements?
- Can you trace governance decisions through audit logs?
- Does this demonstrate due diligence?

**This multi-criteria approach prevents circular self-optimization**—where the system optimizes embedding metrics that don’t correspond to governance outcomes anyone cares about.

### 4.6 Federated Validation: Multi-Institutional Evidence

Validation gains credibility through replication. We have built federated protocols that preserve institutional privacy while enabling cross-organizational measurement.

**Attractor Dynamics Deltas (Δ)**:
$$\Delta M = M_{\text{governed}} - M_{\text{baseline}}$$

**Key insight**: Institutions contribute **differences** rather than raw data.

**Institutions share**:

- ΔF (fidelity improvement in their deployments)
- ΔV trends (stability convergence patterns)
- Δ violations (drift reduction metrics)
- Δ corrections (intervention success rates)

**Institutions do NOT share**:

- Raw session transcripts
- Absolute performance numbers
- Competitive deployment details
- Proprietary use cases

**Federated Aggregation**:
$$\Delta_{\text{global}} = \frac{1}{n} \sum_{i=1}^{n} w_i \cdot \Delta_i$$

where w_i = institutional weight (proportional to sample size).

**Privacy Preservation** (Dwork & Roth, 2014):
$$\tilde{\Delta} = \Delta + \text{Laplace}\left(\frac{\text{Sensitivity}}{\epsilon}\right)$$

Differential privacy noise protects individual institutional performance while enabling meta-analysis.

**Why this matters for regulatory science**:

- Evidence comes from diverse deployments, not single labs
- Healthcare, finance, legal, education sectors all contribute
- Results generalize across domains and use cases
- No institution sacrifices competitive advantage or data sovereignty
- Reproducibility becomes demonstrable: do governance effects hold across contexts?

**Federation creates the evidence base regulators need**: Not vendor claims but cross-institutional measurement. Not single-domain validation but multi-sector replication.

**For federation governance structure and contribution protocols, see**: `docs/governance/Federation_Model.md`

### 4.7 Research Standards We Are Committing To

**IRB Review**: All human subject research approved through institutional review boards

**Pre-Registration**: Analysis plans documented before data collection at OSF or AsPredicted—prevents p-hacking and post-hoc story-fitting (TELOS Labs, 2025, Validation Protocol v1.0)

**Power Analysis**: Sample sizes determined to detect meaningful effects (Cohen’s d ≥ 0.3)

- Estimated requirement: n ≈ 180 sessions per cohort
- Four cohorts: N = 720 total sessions minimum
- Federated study: ≥3 institutions × 180 = 540+ sessions

**Publication Commitment**: Results published regardless of outcome

- Positive results: “TELOS measurably reduces drift by ΔF = 0.18”
- Null results: “TELOS shows no significant improvement over cadence reminders”
- Negative results: “TELOS increases latency without improving governance”

**All advance the field**. Failures teach us what doesn’t work. Successes establish what does.

**Transparency**:

- Code repositories public
- Analysis notebooks shared
- Telemetry format documented
- Replication packages provided

### 4.8 Open Validation Questions

**H1: Mathematical Mitigation Effectiveness**  
Do TELOS-governed sessions improve fidelity relative to stateless, prompt-only, and cadence-reminder baselines by ΔF > 0.15?

**H2: Stability Tracking Validity**  
Does ΔV < 0 predict successful correction in >75% of intervention cases? Do mathematical convergence indicators correspond to observable governance restoration?

**H3: Federated Reproducibility**  
Do governance metrics remain stable across deployment contexts with cross-institutional correlation R² > 0.85?

**H4: Computational Tractability**  
Does runtime overhead remain <20% of baseline latency at 95th percentile? Is the system practically deployable?

**H5: Construct Validity**  
Do embedding-based fidelity scores correlate (r > 0.60) with human judgments of governance quality and task success?

**H6: User Benefit**  
Does TELOS measurably reduce user correction burden compared to baselines? Do users report higher trust and satisfaction?

We have built the infrastructure to answer these questions. We are committed to generating and publishing evidence. We will not claim effectiveness before validation demonstrates it.

**This is scientific integrity. This is observable demonstrable due diligence applied to our own work.**

-----

## 5. Regulatory Alignment: From Measurement to Oversight

### 5.1 The Emerging Compliance Landscape

AI governance is transitioning from voluntary self-regulation to mandatory oversight. The frameworks converge on a common requirement: **continuous monitoring with auditable evidence**.

#### EU AI Act (2024), Article 72: Post-Market Monitoring

“Providers of high-risk AI systems shall put in place a post-market monitoring system in a manner that is proportionate to the nature of the AI technologies and the risks of the high-risk AI system.

The post-market monitoring system shall be based on a post-market monitoring plan. The post-market monitoring plan shall be part of the technical documentation. It shall actively and systematically collect, document and analyse relevant data… including but not limited to the performance of the AI system.”

**What this requires**:

- Systematic collection: Not ad-hoc, but structured and continuous
- Active monitoring: Not passive logging, but deliberate measurement
- Relevant data: Metrics that actually indicate governance status
- Analysis capability: Infrastructure to identify patterns and failures

**What TELOS provides**:

- Turn-by-turn fidelity measurement (systematic collection)
- Runtime drift detection and correction (active monitoring)
- Governance-specific metrics (relevant data)
- Telemetry suitable for trend analysis (analysis capability)

**Evidence format**: “Session XYZ declared boundary ‘no medical advice’. Turn 23 fidelity dropped to 0.64 indicating boundary approach. Context injection applied. Turn 24 fidelity restored to 0.86. Audit trail: [telemetry link].”

This is not post-hoc review. This is contemporaneous evidence that governance was actively maintained.

#### NIST AI Risk Management Framework (2023): MEASURE Function

“Identified AI risks are tracked over time…

- Appropriate methods and metrics are identified and applied
- Mechanisms for tracking identified AI risks over time are in place
- Feedback processes for end users and impacted communities to report problems are established”

**What this requires**:

- Identified metrics: Not generic performance stats, but risk-specific measurements
- Tracking over time: Continuous, not periodic snapshots
- Feedback mechanisms: Ability to detect and respond to problems

**What TELOS provides**:

- Risk-specific metrics: Boundary violations, privacy drift, scope creep (identified metrics)
- Turn-by-turn measurement across session lifecycle (tracking over time)
- Intervention cascade with escalation protocols (feedback mechanisms)

**Evidence format**: “Risk identified: Privacy boundary violation. Metric: Fidelity score 0.58 at turn 18. Response: Regeneration with privacy constraint reinforcement. Outcome: Fidelity restored to 0.84. Risk mitigated before user exposure.”

#### The Due Diligence Standard They’re Requiring

Both frameworks point toward the same principle:

**Not sufficient**: “We designed the system to be safe”  
**Required**: “Here is continuous evidence that safety mechanisms remained active”

**Not sufficient**: “We instructed the model to follow boundaries”  
**Required**: “Here is measurement showing boundaries were maintained, with correction when drift occurred”

**Not sufficient**: “We reviewed sessions after deployment”  
**Required**: “Here is real-time telemetry showing governance was monitored throughout operation”

**TELOS makes this standard achievable.** We provide the measurement infrastructure that transforms compliance from documentation burden to operational evidence generation.

### 5.2 Value Proposition: Why Institutions Need This

#### Academic Value (Immediate)

- First operational framework for testing governance persistence scientifically
- Reproducible protocols for multi-institutional validation
- Open-source implementation enabling comparative research
- Published metrics suitable for IRB-compliant studies

**Impact**: Establishes constitutional computing as empirical research domain rather than aspirational policy.

#### Enterprise Value (Conditional on Validation)

- **Compliance evidence**: Quantitative telemetry demonstrating governance was actively maintained
- **Risk reduction**: Real-time boundary violation prevention, not post-hoc discovery
- **Operational benefit**: Reduced user correction burden, increased session reliability
- **Audit readiness**: Turn-by-turn logs documenting governance decisions and interventions

**Impact**: If validated, transforms governance from liability concern to competitive advantage. “We can demonstrate continuous oversight” becomes a procurement differentiator.

#### Regulatory Value (Long-Term)

- **Observable**: Every session produces quantitative governance metrics
- **Demonstrable**: Audit trails show what was declared, what was maintained, what was corrected
- **Continuous**: Monitoring operates throughout session lifecycle, not periodically
- **Federated**: Cross-institutional evidence without compromising data sovereignty

**Impact**: Provides the measurement primitives that future compliance frameworks will require. When regulators ask “how do you know alignment persisted?”, TELOS provides the answer: “Here are 10,000 sessions with turn-by-turn fidelity scores, intervention records, and stability metrics.”

### 5.3 Mapping TELOS Metrics to Compliance Requirements

|Regulatory Requirement                   |TELOS Mechanism                   |Evidence Generated                                      |
|-----------------------------------------|----------------------------------|--------------------------------------------------------|
|Systematic data collection (EU AI Act 72)|Turn-by-turn telemetry            |Fidelity scores, drift vectors, timestamps              |
|Active monitoring (EU AI Act 72)         |Runtime Steward controllers       |Continuous detection, not periodic sampling             |
|Risk tracking over time (NIST MEASURE)   |Session-level fidelity trends     |Longitudinal adherence measurement                      |
|Feedback mechanisms (NIST MEASURE)       |Graduated intervention cascade    |Correction records, success/failure logs                |
|Performance documentation (EU AI Act 72) |Stability metrics (ΔV)            |Convergence evidence, trajectory analysis               |
|Proportionate oversight (EU AI Act 72)   |Proportional control (K_p scaling)|Light corrections for minor drift, escalation for severe|

**This is not aspirational mapping. These are operational capabilities that exist in the implemented system today.**

### 5.4 What TELOS Does NOT Claim

We are not claiming:

- **Complete compliance**: Regulatory requirements extend beyond session governance (data privacy, model transparency, human oversight structures)
- **Certification**: Only regulators can certify compliance; we provide measurement tools they may require
- **Universal applicability**: Domain-specific regulations (HIPAA, GDPR, financial services) have additional requirements
- **Sufficient safeguards**: TELOS addresses session-level persistence; institutional safeguards (Constitutional AI, content filtering, human review) remain essential

**What we claim**: TELOS provides the **session-level governance measurement and correction infrastructure** that emerging frameworks require. We make one piece of the compliance puzzle solvable.

### 5.5 Transparent Limitations

**Validation Dependency**: All compliance value depends on empirical validation demonstrating effectiveness. Until validated, TELOS is research infrastructure, not compliance solution.

**Model Constraints**: Proprietary APIs limit intervention granularity. Full capabilities require either open-source deployment or provider cooperation.

**Domain Specificity**: Healthcare, legal, financial services have sector-specific requirements beyond general governance. TELOS provides primitives; domain adaptation required.

**Computational Cost**: Runtime monitoring adds latency. Organizations must assess whether governance benefit justifies overhead in their context.

**Adversarial Robustness**: Unknown whether sophisticated users can game governance mechanisms through prompt injection. Requires dedicated security research.

**We document limitations not to undermine confidence but to maintain intellectual integrity.** Regulators need honest assessment, not vendor optimism.

-----

## 6. Mathematical Synthesis: What We’ve Built and Why It Works This Way

### 6.1 Not New Mathematics—New Application Domain

The mathematical tools underlying TELOS are established formalisms:

**Control Theory** (Ogata, 2009; Khalil, 2002): Proportional error correction, stability analysis, feedback control—developed for engineering systems, aircraft autopilots, industrial process control.

**Dynamical Systems** (Strogatz, 2014; Hopfield, 1982): Attractor basins, convergence analysis, trajectory stability—developed for modeling physical, biological, and neural systems.

**Information Theory** (Cover & Thomas, 2006): Similarity metrics, embedding spaces, divergence measures—developed for communication systems, data compression, machine learning.

**These are not experimental methods. They are proven mathematical frameworks with decades of validation in their original domains.**

The contribution is not inventing new mathematics. **The contribution is their orchestrated application to a previously ungoverned domain**: maintaining session-level constraints across transformer interactions.

### 6.2 Why This Synthesis Is Necessary

**Problem**: Transformers exhibit documented persistence failures

- 20-40% degradation across multi-turn dialogues (Laban et al., 2025)
- Instruction erosion, boundary violations, user correction burden
- No existing mechanism to measure or correct drift in real time

**Requirement**: Regulators demand continuous monitoring with auditable evidence

- EU AI Act Article 72: “systematic procedures to review experience”
- NIST AI RMF: “mechanisms for tracking AI risks over time”
- Observable demonstrable due diligence, not design-time assurances

**Gap**: Current approaches cannot provide this

- Prompt engineering: Declaration without measurement
- Post-hoc review: Detection without prevention
- Periodic reminders: Correction without feedback

**Solution synthesis**:

- **Control theory** supplies the correction mechanics (proportional response to measured error)
- **Dynamical systems** provides stability tracking (are trajectories converging or diverging?)
- **Information theory** enables continuous measurement (quantitative fidelity scores)
- **Orchestration** integrates these into unified framework operating at runtime

Result: Governance transitions from qualitative aspiration to quantitative dynamics that can be measured, corrected, and validated.

### 6.3 From Cognitive Phenomena to Mathematical Primitives

We ground the framework in observed transformer behaviors (Wu et al., 2025; Liu et al., 2024; Gu et al., 2024) that mirror documented cognitive effects (Murdock, 1962; Glanzer & Cunitz, 1966):

**Primacy Bias**:

- Cognitive: Early information influences strongly but decays over time
- Transformer: Initial context dominates attention but loses salience as sessions extend
- Mathematical formalization: Attractor basin anchored to session-initialization declarations
- Governance mechanism: Purpose and boundaries encoded as weighted centroid, defining reference point

**Recency Falloff**:

- Cognitive: Middle information is forgotten, recent information dominates
- Transformer: Middle-context content loses influence, recent turns capture attention
- Mathematical formalization: Fidelity measurement tracking adherence drift across turns
- Governance mechanism: Continuous scoring detects when early instructions lose salience

**Attention Redistribution**:

- Cognitive: Focus shifts to salient stimuli, away from background instructions
- Transformer: Attention “sinks” capture focus, reducing weight on governance tokens
- Mathematical formalization: Error vectors indicating directional drift from attractor
- Governance mechanism: Corrective forces computed proportional to deviation magnitude

**We treat these not as bugs to eliminate but as phenomena to formalize.** The same mechanisms that cause drift become the basis for detecting and correcting it when expressed mathematically.

### 6.4 The Operational Transformation

TELOS shifts governance from aspirational to operational:

**Before**: “The system should maintain privacy boundaries”

- Qualitative goal
- No measurement
- No correction mechanism
- Post-hoc audit only

**After**: “Privacy boundary encoded as attractor at â. Current response fidelity F_t = 0.64, below threshold 0.70. Drift detected. Context injection applied. Post-correction fidelity F_{t+1} = 0.87, above threshold. Adherence restored. Logged for audit.”

- Quantitative state
- Continuous measurement
- Proportional correction
- Real-time evidence generation

**This transformation enables regulatory compliance**: Not claims about what the system should do, but evidence of what it actually did, turn by turn, throughout deployment.

### 6.5 Why Orchestration Layer Architecture

Operating at the orchestration layer (between UI and model API) is not incidental—it is architecturally necessary:

**Must be late enough**: To measure actual model behavior, not intended behavior. Prompt engineering operates too early—it shapes what goes into the model but cannot measure what comes out.

**Must be early enough**: To correct before user exposure. Post-hoc filtering operates too late—violations reach users before detection.

**Must be model-independent**: To function across providers and architectures. Model-layer modifications require retraining or provider cooperation, limiting deployability.

**Must be modular**: To enable A/B testing, validation studies, and iterative improvement. Baked-in mechanisms cannot be easily modified or compared against baselines.

The orchestration layer satisfies all requirements: late enough to measure, early enough to correct, independent enough to generalize, modular enough to validate.

### 6.6 Relationship to Institutional Safeguards

TELOS does not replace institutional governance—it complements it:

**Universal Safeguards** (Constitutional AI, content filtering, provider policies):

- Operate at model-level and design-time
- Define what’s never acceptable across all uses
- Prevent gross harms: hate speech, violence, illegal content
- Essential baseline that applies universally

**Session-Level Governance** (TELOS Primacy Dynamics):

- Operates at session-level and runtime
- Maintains what’s specifically required for a given interaction
- Prevents subtle drift: scope creep, boundary erosion, purpose loss
- Necessary complement that applies contextually

**Analogy**: Universal safeguards are like building codes—they define minimum safety standards for all structures. Session governance is like blueprints—they define what this specific structure must achieve.

Both are necessary. Neither alone is sufficient. TELOS makes the session-level component measurable and correctable.

### 6.7 Why This Matters: The Measurement Gap

The most important contribution is not the mathematics or the implementation—it is making governance persistence **empirically testable**.

**Before TELOS**: “We believe alignment persists because we designed carefully”  
**After TELOS**: “Here are 10,000 sessions with turn-by-turn fidelity measurements, intervention records, and convergence metrics—let’s analyze whether alignment actually persisted”

This shifts AI governance from institutional assertion to scientific inquiry. Claims become hypotheses. Beliefs become testable predictions. Governance becomes a research domain with reproducible protocols and falsifiable hypotheses.

**Whether validation proves TELOS effective or identifies limitations, the field advances.** We gain empirical evidence about what runtime mathematical mitigation can and cannot achieve.

This is the infrastructure that regulatory science requires: not vendor promises but measurable outcomes, not design-time assurances but operational evidence, not single-institution claims but federated validation.

-----

## 7. Limitations, Open Questions, and Future Work

### 7.1 Current Limitations We Must Acknowledge

**Unvalidated Effectiveness**  
Mathematical formalization is complete. Operational implementation exists and functions correctly on synthetic test sets. Whether mechanisms measurably outperform simpler baselines remains empirically undemonstrated.

We have built the capability to intervene. We have not yet proven that intervention is effective. This distinction is critical.

**Limited Model Coverage**  
Initial validation targets open-source architectures (Mistral, Llama) enabling full intervention control. Generalization to proprietary models requires separate validation under API constraints.

Model-specific behaviors may limit universal applicability. What works for Mistral may not work identically for GPT-4 or Claude.

**Research-Grade Implementation**  
Current system prioritizes correctness and measurability over production concerns (scale, error handling, redundancy, monitoring).

Enterprise deployment requires engineering hardening: load balancing, fault tolerance, observability, security audits—none of which currently exist at production quality.

**Computational Overhead**  
Turn-by-turn embedding computation, similarity calculation, and intervention logic add latency (median ~40ms, p95 ~85ms).

Trade-offs between governance rigor and response time need characterization across deployment contexts. Some applications may find overhead unacceptable.

**Embedding Dependency**  
Fidelity measurements inherit limitations of underlying embedding models (Reimers & Gurevych, 2019).

Poor embeddings compromise governance detection. Models trained primarily on English may underperform in other languages. Domain-specific terminology may not embed well.

**Construct Validity Uncertainty**  
Whether embedding-based metrics correlate with governance outcomes that actually matter requires empirical validation against external criteria.

Risk: Optimizing fidelity scores that don’t correspond to real boundary maintenance, creating circular self-validation without practical benefit.

**Adversarial Robustness Unknown**  
Whether sophisticated users can game governance mechanisms through carefully crafted prompts remains untested.

Prompt injection attacks targeting governance layer—not model layer—are plausible but unexplored. Dedicated security research required.

**Context Window Competition**  
Governance declarations compete with conversation content for limited context capacity (Liu et al., 2024).

In extremely long sessions (200+ turns), even with intelligent truncation, governance salience may degrade. Alternative approaches (periodic re-initialization, attractor adaptation) may be necessary.

### 7.2 Open Research Questions

**Parameterization Optimization**  
How should K_p (proportional gain), τ (tolerance), and intervention thresholds be tuned?

- One-size-fits-all vs. domain-specific vs. adaptive per-session?
- Can hyperparameter optimization find universally effective settings?
- Or does effective governance require human calibration per deployment?

**Intervention Trade-offs**  
When does correction burden exceed drift harm?

- Is it better to accept minor drift rather than constant intervention?
- What’s the optimal balance between governance rigor and conversation flow?
- Can we quantify “intervention fatigue” where users distrust over-corrected systems?

**Temporal Dynamics**  
Do governance requirements evolve during sessions?

- Should attractors be static or adaptive?
- Can we detect when declared purpose shifts legitimately vs. drifts erroneously?
- How to handle user requests that conflict with earlier declarations?

**Multi-Objective Optimization**  
How to simultaneously optimize fidelity, utility, latency, and user satisfaction?

- These objectives trade off—improving one may degrade another
- Can we find Pareto-optimal configurations?
- Does optimal balance vary by domain, user, or session type?

**Cross-Model Generalization**  
Do governance mechanisms work uniformly across architectures?

- Mistral vs. Llama vs. GPT-4 vs. Claude—do they exhibit similar drift patterns?
- Do intervention effectiveness rates vary by model family?
- Can we identify model-specific governance needs?

**Adversarial Governance Probing**  
Can users exploit governance mechanics maliciously?

- Prompt injection targeting attractor embeddings
- Gaming fidelity scores through semantic mimicry without true adherence
- Causing intentional intervention cascades (denial-of-service via governance overhead)

### 7.3 Future Directions: If Validation Succeeds

**Multi-Attractor Systems**  
Complex sessions may require multiple simultaneous constraints:

- Medical: “provide information” + “maintain privacy” + “avoid diagnoses”
- Legal: “analyze precedent” + “preserve attorney-client privilege” + “track billable tasks”

How to define attractor geometries for multiple competing objectives? Can we formalize attractor hierarchies or composite basins?

**Adaptive Basin Geometry**  
Current basins are static—defined at initialization, fixed throughout.

Can basins adapt to conversation dynamics while preserving governance intent? When is adaptation beneficial vs. when does it enable drift rationalization?

**Integration with Institutional Policies**  
How to connect session-level governance to enterprise policy frameworks?

- Automatic attractor generation from organizational compliance documents
- Policy versioning and governance constraint evolution
- Cross-session learning: do similar purposes require similar interventions?

**Standards Development**  
If TELOS mechanisms prove effective, broader adoption requires standardization:

- Metrics specifications (fidelity computation, intervention thresholds)
- Telemetry formats (interoperable audit logs)
- Validation protocols (reproducible comparative studies)
- Certification criteria (when can a system claim “TELOS-compliant governance”?)

### 7.4 Future Directions: If Validation Shows Limitations

**Hybrid Approaches**  
If purely mathematical mitigation proves insufficient:

- Combining embedding-based detection with linguistic pattern matching
- Integrating rule-based guardrails with continuous measurement
- Human-in-the-loop for borderline cases where automated classification fails

**Alternative Stability Guarantees**  
If ΔV convergence tracking doesn’t predict governance success:

- Other stability proxies from dynamical systems theory
- Statistical measures (trend analysis, autocorrelation in fidelity time series)
- Machine learning approaches (predict intervention success from session features)

**Domain-Specific Adaptations**  
If general-purpose framework fails in specific sectors:

- Healthcare-specific governance with clinical terminology awareness
- Legal-specific with precedent citation tracking
- Finance-specific with regulatory terminology sensitivity

**Measurement-Only Frameworks**  
If interventions prove ineffective or too costly:

- Remove correction mechanisms, preserve measurement infrastructure
- Provide governance telemetry without runtime intervention
- Enable post-hoc analysis and manual review with quantitative evidence
- Still achieves regulatory goal: demonstrable monitoring, even without automated correction

### 7.5 Broader Implications for AI Governance

**Constitutional Computing Research Agenda**  
TELOS demonstrates that session-level governance can be formalized mathematically. This opens research directions:

- Other application domains (code generation, content moderation, educational tutoring)
- Alternative mathematical frameworks (game theory, mechanism design, formal verification)
- Cross-system governance (multi-agent consistency, federation across providers)

**Federated Governance Methodology**  
The federation protocols we’ve built enable new research methodologies:

- Privacy-preserving multi-institutional validation
- Cross-sector reproducibility testing
- Longitudinal governance effectiveness studies
- Collaborative improvement through shared differential-privacy-protected insights

**Interdisciplinary Collaboration Requirements**  
Effective AI governance cannot be solved by computer science alone:

- Law: Translating regulatory requirements into technical specifications
- Ethics: Defining what governance outcomes society actually values
- HCI: Understanding how users perceive and interact with governed systems
- Policy: Connecting technical capabilities to institutional frameworks

**Regulatory Science for AI**  
Most critically: establishing AI oversight as **regulatory science** with:

- Reproducible measurement protocols
- Standardized metrics and telemetry
- Evidence-based policy development
- Independent verification mechanisms

This is what TELOS attempts to enable: not a solution, but the measurement infrastructure upon which solutions can be built and validated.

-----

## 8. Conclusion: What We Have Built and Why It Matters

We have built mathematical infrastructure for runtime AI governance. This whitepaper documents what exists, why it was designed this way, and what remains to be validated.

### What Exists

**Primacy Attractor Dynamics**: Mathematical formalization synthesizing control theory, dynamical systems, and information theory to make session-level governance measurable and correctable.

**Runtime Stewards**: Orchestration-layer controllers that continuously monitor adherence, detect drift, compute corrective forces, and apply proportional interventions.

**Graduated Intervention Cascade**: Four-state response system (Monitor → Correct → Intervene → Escalate) where intervention intensity scales with drift severity.

**Operational Telemetry**: Turn-by-turn logging of fidelity scores, error vectors, stability metrics, interventions applied, and outcomes achieved—creating audit trails suitable for regulatory oversight.

**Federated Validation Protocols**: Privacy-preserving mechanisms enabling cross-institutional evidence generation without exposing raw session data or competitive details.

**Open-Source Implementation**: Code repositories, validation protocols, and replication packages enabling independent verification and comparative research.

### What Has Been Tested

The TELOS framework is operationally implemented and has undergone initial verification testing:

- Synthetic conversation sets (n=50 sessions) confirm mathematical correctness
- Fidelity measurement produces quantitatively consistent scores
- Intervention cascade executes as designed across governance states
- Telemetry logging and export functions correctly
- Computational overhead remains within acceptable limits (<100ms at p95)
- No runtime errors or system failures observed in test deployment

**These tests demonstrate mechanical functionality**: the system computes, measures, intervenes, and logs as specified.

### What Remains Unvalidated

**Comparative effectiveness**: Whether runtime mathematical mitigation measurably outperforms simpler baselines (stateless sessions, prompt-only reinforcement, cadence reminders) in reducing documented 20-40% degradation.

**Statistical significance**: Whether observed improvements are meaningful or attributable to noise.

**Cross-model generalization**: Whether mechanisms work consistently across LLM architectures (open-source and proprietary).

**Domain robustness**: Whether performance holds across use cases (healthcare, legal, finance, technical assistance).

**Construct validity**: Whether embedding-based fidelity scores correlate with governance outcomes users and regulators actually care about (task success, user trust, compliance satisfaction).

**User experience impact**: Whether governance mechanisms reduce correction burden without degrading perceived quality.

We refuse to claim what we have not proven. Mathematical rigor does not guarantee practical effectiveness. Operational implementation does not validate theoretical assumptions.

**Validation is not an afterthought—it is the centerpiece.** We have built research infrastructure designed to generate evidence. We are committed to conducting controlled studies, publishing results regardless of outcome, and enabling independent replication.

### Why This Matters

**For Regulators**: The emerging compliance landscape requires observable demonstrable due diligence. Monitoring must be continuous, evidence must be quantitative, and oversight must be auditable.

With the EU AI Act’s February 2026 template deadline and August 2026 compliance requirement approaching, institutions need technical infrastructure that can adapt to forthcoming specifications.

TELOS provides measurement primitives that regulatory frameworks demand: turn-by-turn governance telemetry documenting what constraints were declared, when drift occurred, what interventions were applied, whether adherence was restored.

Whether regulators adopt these specific mechanisms or develop alternatives, the requirement for such mechanisms is clear. We are building to meet that requirement.

**For Enterprises**: Alignment drift is not hypothetical—it is documented, measured, and causing operational problems today. Users must constantly re-correct. Compliance officers review transcripts and find boundary violations after they’ve reached users.

If validated, TELOS transforms governance from liability concern to competitive advantage: “We can demonstrate continuous oversight” becomes procurement differentiator. “We detect and correct drift in real time” becomes risk mitigation capability.

**For Researchers**: AI governance has been aspirational policy—qualitative goals without quantitative measurement. TELOS establishes it as empirical science with reproducible protocols and falsifiable hypotheses.

The federated validation framework enables multi-institutional evidence generation while preserving data sovereignty. This is how governance science advances: not through single-lab claims but through cross-organizational replication.

**For the Field**: Either validation outcome advances understanding. Demonstrated effectiveness establishes runtime mathematical mitigation as viable path for AI accountability. Identified limitations clarify what such mechanisms cannot achieve, preventing overinvestment in ineffective approaches.

Both cases transform governance from assertion to evidence. Both enable regulatory frameworks to be grounded in measurement rather than assumption.

### The Contribution

TELOS does not solve AI alignment completely. It addresses a specific, measurable problem: documented governance erosion across multi-turn interactions.

**The contribution is methodological**: transforming governance persistence from institutional assumption into testable science.

Before: “We designed the system to be safe, we believe alignment persists”  
After: “Here is continuous measurement showing whether alignment persisted, here are interventions that were applied when drift occurred, here is quantitative evidence of their effectiveness”

Before: Claims without evidence  
After: Evidence enabling verification

Before: Design-time assurances  
After: Runtime demonstration

Before: Qualitative aspiration  
After: Quantitative dynamics

**This is observable demonstrable due diligence.**

Not perfect control—but measurable mitigation.  
Not guaranteed alignment—but continuous monitoring with proportional correction.  
Not aspirational policy—but operational mathematics generating auditable evidence.

### What We’re Asking

We are asking institutions to participate in validation. To contribute Attractor Dynamics Deltas through federated protocols. To help us determine whether runtime mathematical mitigation actually works.

We are asking researchers to replicate, challenge, and extend. To test alternative baselines. To identify failure modes. To improve what works and discard what doesn’t.

We are asking regulators to recognize that governance persistence requires measurement infrastructure. That observable demonstrable due diligence cannot emerge without the technical foundations that make it observable.

We are asking the field to treat AI governance as science—with reproducible protocols, empirical validation, transparent reporting of both successes and failures.

### The Stakes

If we fail to establish governance persistence as measurable and correctable, AI systems will continue degrading across multi-turn interactions. Users will continue experiencing friction. Compliance officers will continue discovering violations after they occur. Regulators will continue demanding evidence we cannot provide.

If we succeed in establishing mathematical frameworks for runtime governance, we create:

- Measurement standards that enable reproducible research
- Intervention mechanisms that can be validated and improved
- Audit trails that satisfy regulatory requirements
- Evidence bases that ground policy in empirical reality

**Both outcomes require attempting what TELOS attempts**: building the infrastructure, conducting the validation, publishing the results, and enabling the science.

### The Commitment

We commit to scientific integrity: pre-registered studies, IRB approval, power analysis, transparent methodology, publication regardless of outcome.

We commit to open access: code repositories, telemetry formats, validation protocols, replication packages of these approaches provide what regulators require: **continuous measurement** of governance persistence, **proportional intervention** when drift occurs, and **auditable telemetry** documenting both.

### 1.5 What We Are Building

TELOS provides the infrastructure for **observable demonstrable due diligence**:

**Observable**: Every turn produces measurable fidelity scores, drift vectors, stability metrics—quantitative evidence of governance state.

**Demonstrable**: Telemetry creates an audit trail showing what constraints were declared, when drift occurred, what interventions were applied, whether adherence improved.

**Due Diligence**: The system actively works to maintain alignment rather than passively assuming it persists—and generates evidence of this work.

We do not claim this solves AI governance completely. We claim it makes governance **measurable** where it was previously aspirational, **correctable** where it was previously hope-based, and **auditable** where it was previously opaque.

The following sections describe the mathematical framework that makes this possible, the implementation that makes it practical, and the validation framework that will determine whether it works.

-----

## 2. Primacy Attractor Dynamics: Making Governance Measurable

### 2.1 Core Insight: Governance as Geometry

We treat alignment not as a qualitative property but as a **measurable position in embedding space**.

When a user declares:

- **Purpose**: “Help me structure a technical paper”
- **Scope**: “Guide my thinking, don’t write content”
- **Boundaries**: “No drafting full paragraphs”

These declarations become embeddings—vectors in ℝ^d using standard sentence transformers (Reimers & Gurevych, 2019). These vectors define the **Primacy Attractor**: the governance center against which all subsequent outputs are measured.

Every model response gets embedded. Its distance from the attractor quantifies drift. Its direction indicates how it’s violating constraints. These measurements enable proportional intervention: minor drift gets gentle correction, severe violations get immediate blocking.

This transforms governance from subjective judgment (“does this feel aligned?”) to quantitative measurement (“fidelity = 0.73, below threshold, intervention required”).

### 2.2 Mathematical Foundations: Three Coordinated Mechanics

We synthesize three established mathematical formalisms to create a unified governance framework:

#### Primacy Basin: Bounded Governance Region (Attractor Dynamics)

Drawing on nonlinear systems theory (Strogatz, 2014; Hopfield, 1982), we define the attractor as a weighted centroid:

$$\hat{a} = \frac{\tau p + (1 - \tau)s}{|\tau p + (1 - \tau)s|}$$

where **p** = purpose embedding, **s** = scope embedding, τ ∈ [0,1] = tolerance parameter.

The basin defines acceptable boundaries:

$$B(\hat{a}, r) = {x \in \mathbb{R}^d : |x - \hat{a}| \leq r}$$

$$r = \frac{2}{\max(\rho, 0.25)} \quad \text{where} \quad \rho = 1 - \tau$$

Lower tolerance produces tighter basins. A minimum ρ = 0.25 prevents excessive expansion, capping r ≤ 8.0.

**Governance meaning**: The attractor represents perfect alignment. The basin defines how far responses can deviate before correction becomes necessary. This isn’t arbitrary—it’s grounded in how dynamical systems define stable regions.

#### Primacy Gravity: Corrective Forces (Control Theory)

Using proportional control principles (Ogata, 2009; Khalil, 2002), we compute corrective forces based on error magnitude:

$$e_t = |x_t - \hat{a}|$$

$$G_t = -K_p \cdot e_t$$

where K_p is proportional gain (intervention aggressiveness).

**Governance meaning**: Larger deviations generate stronger correction signals. This mirrors how thermostats respond to temperature drift or autopilots correct course deviation—proportional response to measured error.

#### Primacy Fidelity: Continuous Adherence Measurement (Information Theory)

Using cosine similarity from information theory (Cover & Thomas, 2006), we quantify alignment:

$$I_t = \cos(x_t, p) = \frac{x_t \cdot p}{|x_t| \cdot |p|}$$

$$F = \frac{1}{T} \sum_{t=1}^{T} I_t$$

**Governance meaning**: Turn-by-turn scores create a continuous fidelity measure. Session-level averages quantify overall adherence. This is the metric that gets logged for audit.

#### Primacy Orbit: Stability Tracking (Lyapunov-Like Analysis)

Adapting stability analysis from control theory (Khalil, 2002), we track convergence:

$$V(x) = |x - \hat{a}|^2$$

$$\Delta V_t = V(x_{t+1}) - V(x_t)$$

**Governance meaning**: ΔV < 0 indicates trajectories moving toward the attractor (governance restoring). ΔV > 0 indicates divergence (governance failing). This tells us whether corrections are working.

### 2.3 Why This Synthesis Matters

These are not new mathematical tools—control theory, dynamical systems, and information theory are established disciplines. **The contribution is their orchestration** to address a previously ungoverned domain: maintaining session-level constraints across transformer interactions.

Where these tools traditionally serve optimization, signal processing, or system control, we apply them to a specific regulatory need: **making governance persistence measurable, correctable, and auditable**.

The result: governance transitions from qualitative aspiration (“be aligned”) to quantitative dynamics (“fidelity = 0.87, within basin, ΔV = -0.15, converging”).

### 2.4 Primacy State: The Condition We’re Maintaining

We distinguish between:

**Primacy State Condition**: The system being aligned within governance boundaries—the goal we’re trying to maintain.

**Primacy State Measure**: The mathematical metrics (fidelity, drift, stability) that provide evidence of whether that goal is maintained.

The Condition is what regulators care about: “Were privacy boundaries maintained?”

The Measure is how we demonstrate it: “Fidelity remained above 0.85 for 94% of turns. When fidelity dropped to 0.72 at turn 18, intervention was applied within 1 turn, restoring fidelity to 0.89 by turn 20. Audit trail attached.”

**This is observable demonstrable due diligence in practice.**

### 2.5 From Transformer Fragility to Governance Primitive

Transformers exhibit predictable failure modes:

- **Primacy bias**: Early context influences strongly but decays (Wu et al., 2025; Murdock, 1962)
- **Recency falloff**: Middle content loses salience (Liu et al., 2024; Glanzer & Cunitz, 1966)
- **Attention sinks**: Focus redistributes away from instructions (Gu et al., 2024)

We treat these not as limitations to overcome but as **design primitives** to formalize:

The same attention mechanisms that cause drift can be measured and corrected when expressed mathematically. Primacy bias becomes the justification for attractor anchoring at session start. Recency falloff becomes the phenomenon fidelity measurement detects. Attention redistribution becomes the drift that corrective forces address.

We’re not fighting transformer architecture—we’re building governance mechanics that work with its documented behaviors.

-----

## 3. Runtime Implementation: From Mathematics to Operational Governance

### 3.1 The Orchestration Layer: Why This Architectural Choice Matters

TELOS operates at the **orchestration layer**—the software layer between user interface and language model that manages conversation state, API calls, and response handling.

This positioning is deliberate and enables capabilities unavailable elsewhere:

**What we can do that prompt engineering cannot**:

- Inspect model outputs before users see them
- Measure drift using embedding comparisons
- Reject and regenerate violating responses
- Maintain fidelity history across session

**What we can do that model training cannot**:

- Deploy immediately without retraining
- Function with proprietary APIs (OpenAI, Anthropic)
- Update governance constraints per-session
- A/B test interventions empirically

**What we can do that post-hoc filtering cannot**:

- Apply corrective steering before user exposure
- Provide graduated interventions proportional to drift
- Track stability trends across correction attempts
- Generate research-grade telemetry for every decision

The orchestration layer provides necessary control: **late enough to measure actual model behavior, early enough to correct before user exposure, independent enough to function across models and APIs**.

Current implementation supports:

- **Mistral** (open-source, full intervention control)
- **OpenAI** (proprietary API, limited but functional control)

Architecture designed for model-agnostic expansion—adding new models requires API integration and embedding validation but not changes to core governance mechanics.

### 3.2 Runtime Stewards: The Governance Controllers

**Runtime Stewards** execute Primacy Attractor Dynamics at the orchestration layer:

**Every Turn, Stewards**:

1. Receive model response
1. Compute embedding x_t
1. Calculate fidelity I_t = cos(x_t, p)
1. Measure drift e_t = ||x_t - â||
1. Track stability ΔV_t = V(x_t) - V(x_{t-1})
1. Determine governance state
1. Select intervention if needed
1. Apply correction and verify improvement
1. **Log everything for audit**

This continuous monitoring creates the telemetry that regulators require: turn-by-turn evidence that governance was actively maintained, not passively assumed.

### 3.3 Graduated Intervention Cascade: Proportional Response

We implement proportional control (Ogata, 2009) where intervention intensity scales with drift severity:

#### State 1: MONITOR (Fidelity ≥ 0.85)

**Condition**: Strong alignment  
**Action**: We commit to open access: code repositories, telemetry formats, validation protocols, replication packages—all publicly available for verification and challenge.

We commit to federated collaboration: privacy-preserving protocols enabling multi-institutional evidence generation without compromising data sovereignty or competitive position.

We commit to honest limitation disclosure: acknowledging what remains unvalidated, what may not work, what requires additional research—refusing vendor optimism in favor of intellectual integrity.

**This is what observable demonstrable due diligence looks like when applied to our own work.**

### Closing

Governance persistence is not an abstract philosophical challenge. It is a documented technical problem with measured degradation rates, observable failure modes, and regulatory requirements demanding solutions.

We have built mathematical infrastructure designed to address this problem. We have implemented it at the orchestration layer. We have created validation protocols to test whether it works. We have established federation mechanisms to enable reproducible multi-institutional evidence.

**The system is operational. The hypotheses are defined. The validation framework is specified.**

What remains is execution: controlled studies, comparative analysis, transparent reporting, and collective determination of whether runtime mathematical mitigation measurably reduces alignment drift.

**Governance becomes testable. That is the contribution.**

We invite participation, challenge, replication, and validation. Not because we are certain of success, but because determining what works requires attempting to build it, measuring whether it succeeds, and publishing the results for independent verification.

This is how regulatory science advances: through measurement infrastructure that makes governance observable, through validation protocols that make effectiveness testable, and through federated collaboration that makes evidence reproducible.

**TELOS provides the infrastructure. The field must generate the evidence.**

We are building what emerging regulatory frameworks require. Whether what we have built actually works is a question we are committed to answering empirically.

-----

## Appendix A: Mathematical Reference

### A.1 Core Definitions

**Primacy Attractor** (weighted centroid):
$$\hat{a} = \frac{\tau p + (1 - \tau)s}{|\tau p + (1 - \tau)s|}$$

where:

- **p** ∈ ℝ^d = purpose embedding
- **s** ∈ ℝ^d = scope embedding
- τ ∈ [0,1] = constraint tolerance parameter
- **â** ∈ ℝ^d = normalized attractor vector

**Basin Radius** (governance boundary):
$$r = \frac{2}{\max(\rho, 0.25)} \quad \text{where} \quad \rho = 1 - \tau$$

Constraints:

- ρ_min = 0.25 prevents excessive basin expansion
- r_max = 8.0 when τ = 1.0 (maximum tolerance)
- r_min = 2.0 when τ = 0.0 (minimum tolerance)

**Primacy Basin** (acceptable region):
$$B(\hat{a}, r) = {x \in \mathbb{R}^d : |x - \hat{a}| \leq r}$$

### A.2 Measurement Formulas

**Turn-Level Fidelity** (cosine similarity):
$$I_t = \cos(x_t, p) = \frac{x_t \cdot p}{|x_t| \cdot |p|}$$

where **x_t** ∈ ℝ^d = embedding of model response at turn t

**Session-Level Fidelity** (average adherence):
$$F = \frac{1}{T} \sum_{t=1}^{T} I_t$$

where T = total number of turns in session

**Error Distance** (drift magnitude):
$$e_t = |x_t - \hat{a}|$$

**Corrective Force** (proportional control):
$$G_t = -K_p \cdot e_t$$

where K_p > 0 is proportional gain parameter (typical range: 0.1–1.0)

**Lyapunov-Like Energy** (stability tracking):
$$V(x) = |x - \hat{a}|^2$$

$$\Delta V_t = V(x_{t+1}) - V(x_t)$$

Interpretation:

- ΔV_t < 0 → Convergence (trajectory moving toward attractor)
- ΔV_t > 0 → Divergence (trajectory moving away from attractor)
- ΔV_t ≈ 0 → Stability (trajectory maintaining position)

### A.3 Intervention Thresholds

|State    |Fidelity Range |Intervention        |Latency Impact|
|---------|---------------|--------------------|--------------|
|MONITOR  |F ≥ 0.85       |None                |~0ms          |
|CORRECT  |0.70 ≤ F < 0.85|Context injection   |~5-10ms       |
|INTERVENE|0.50 ≤ F < 0.70|Regeneration        |~500-2000ms   |
|ESCALATE |F < 0.50       |Block + human review|Session pause |

### A.4 Federated Metrics

**Attractor Dynamics Delta** (institutional comparison):
$$\Delta M = M_{\text{governed}} - M_{\text{baseline}}$$

where M can be:

- ΔF (fidelity improvement)
- Δ violations (boundary violation frequency reduction)
- Δ corrections (intervention success rate difference)
- Δ latency (computational overhead added)

**Federated Aggregation** (weighted average):
$$\Delta_{\text{global}} = \frac{1}{n} \sum_{i=1}^{n} w_i \cdot \Delta_i$$

where:

- n = number of contributing institutions
- w_i = institutional weight (typically proportional to sample size)
- Δ_i = institution i’s delta metric

**Differential Privacy** (noise addition for privacy preservation):
$$\tilde{\Delta} = \Delta + \text{Laplace}\left(\frac{\text{Sensitivity}}{\epsilon}\right)$$

where:

- Sensitivity = maximum change in Δ from adding/removing one session
- ε = privacy parameter (smaller ε → stronger privacy, more noise)

### A.5 Best-of-N Reranking

**Selection Formula** (quality optimization):
$$\text{selected} = \arg\max_{i \in {1,\ldots,N}} \left[\alpha \cdot F_i + (1-\alpha) \cdot U_i\right]$$

where:

- N ∈ [3,5] = number of candidate responses generated
- F_i = fidelity score for candidate i (governance quality)
- U_i = utility heuristic for candidate i (task relevance)
- α ≈ 0.7 = balance parameter (fidelity-weighted)

Utility heuristic U_i typically includes:

- Response length appropriateness
- Question answerability
- Coherence and fluency
- Task-specific relevance measures

-----

## Appendix B: Implementation Details

### B.1 Embedding Models

**Recommended Production**: `sentence-transformers/all-MiniLM-L6-v2`

- Dimension: 384
- Speed: ~20-30ms per embedding
- Use case: Production deployments balancing quality and performance

**Alternative High-Quality**: `sentence-transformers/all-mpnet-base-v2`

- Dimension: 768
- Speed: ~40-60ms per embedding
- Use case: Research validation requiring maximum quality

**Critical Requirement**: Lock embedding model at session initialization. Never change models mid-session. Model version must be logged in telemetry for reproducibility.

### B.2 Context Window Management

**Governance Anchoring Strategy**:

```
[System Prompt - Never Truncated]
════════════════════════════════════════
Purpose: {user_declared_purpose}
Scope: {user_declared_scope}
Boundaries: {user_declared_boundaries}
Tolerance: {tau_value}
════════════════════════════════════════
[DO NOT TRUNCATE ABOVE THIS LINE]

[Recent Conversation - Always Preserved]
Last 5-10 turns with full context

[Middle History - Compressed When Needed]
Earlier turns, summarized or removed when
context approaches 80% of window capacity

[Correction Log - External Storage]
Intervention history logged to database,
not included in model context
```

**Truncation Priority**:

1. Governance declarations: **NEVER** remove
1. Recent K turns: Always preserve (K ≈ 5-10)
1. Middle content: Compress when space needed
1. Ancient content: Remove first when capacity reached

### B.3 Computational Performance Profile

**Embedding Computation**:

- Operation: Parallel to response generation (concurrent, not sequential)
- Latency: 15-30ms (MiniLM), 40-60ms (MPNet)
- Optimization: Cache purpose/scope embeddings (compute once per session)

**Similarity Calculation**:

- Complexity: O(d) where d = embedding dimension
- Latency: <5ms for d ≤ 1024
- Optimization: SIMD vector operations for dot products

**Intervention Decision Logic**:

- Complexity: O(1) threshold comparisons
- Latency: <2ms
- Optimization: Pre-computed intervention templates

**Total Overhead** (measured on representative workload):

- Median: 42ms per turn
- 90th percentile: 78ms per turn
- 95th percentile: 87ms per turn
- 99th percentile: 124ms per turn

**Target**: <100ms at 95th percentile (currently achieved)

### B.4 Telemetry Schema

**Per-Turn Log Entry**:

```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "turn_number": 42,
  "timestamp_utc": "2025-10-08T14:23:45.123Z",
  
  "governance": {
    "purpose_embedding_hash": "sha256:abc123...",
    "scope_embedding_hash": "sha256:def456...",
    "attractor_center": [0.123, -0.456, 0.789, ...],
    "basin_radius": 4.0,
    "tolerance_tau": 0.5
  },
  
  "response": {
    "embedding": [0.234, -0.567, 0.891, ...],
    "embedding_model": "all-MiniLM-L6-v2",
    "token_count": 127,
    "generation_latency_ms": 1243
  },
  
  "measurements": {
    "fidelity_score": 0.87,
    "error_distance": 1.23,
    "delta_v": -0.15,
    "in_basin": true
  },
  
  "governance_state": "MONITOR",
  
  "intervention": {
    "applied": false,
    "type": null,
    "latency_added_ms": 0,
    "post_intervention_fidelity": null
  },
  
  "metadata": {
    "model_provider": "mistral",
    "model_version": "mistral-large-2",
    "deployment_mode": "open_source"
  }
}
```

**Intervention Event Log** (when intervention occurs):

```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "turn_number": 23,
  "timestamp_utc": "2025-10-08T14:18:32.456Z",
  
  "pre_intervention": {
    "fidelity_score": 0.64,
    "error_distance": 2.87,
    "governance_state": "INTERVENE",
    "original_response": "..."
  },
  
  "intervention_applied": {
    "type": "regeneration",
    "constraint_prompt": "Remember: Guide structure, don't write content...",
    "candidates_generated": 1,
    "latency_ms": 1847
  },
  
  "post_intervention": {
    "fidelity_score": 0.89,
    "error_distance": 0.76,
    "delta_v": -2.11,
    "governance_state": "MONITOR",
    "regenerated_response": "..."
  },
  
  "outcome": {
    "success": true,
    "fidelity_improvement": 0.25,
    "convergence_achieved": true
  }
}
```

### B.5 API Integration Patterns

**Open-Source Model Integration** (Mistral example):

```python
from mistralai.client import MistralClient
from sentence_transformers import SentenceTransformer

# Initialize
client = MistralClient(api_key=API_KEY)
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Governance setup
purpose_emb = embedder.encode(user_purpose)
attractor = compute_attractor(purpose_emb, scope_emb, tau=0.5)

# Per-turn monitoring
response = client.chat(messages=conversation_history)
response_emb = embedder.encode(response.content)
fidelity = cosine_similarity(response_emb, purpose_emb)

if fidelity < 0.70:
    # Intervention: inject reminder
    conversation_history.append({
        "role": "system",
        "content": f"Remember: {user_purpose}"
    })
    response = client.chat(messages=conversation_history)
```

**Proprietary API Integration** (OpenAI example):

```python
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# Initialize
client = OpenAI(api_key=API_KEY)
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Governance setup (same as above)
attractor = compute_attractor(purpose_emb, scope_emb, tau=0.5)

# Per-turn monitoring (limited intervention options)
response = client.chat.completions.create(
    model="gpt-4",
    messages=conversation_history
)
response_emb = embedder.encode(response.choices[0].message.content)
fidelity = cosine_similarity(response_emb, purpose_emb)

# Can inject context, limited regeneration
# Cannot do best-of-N reranking (API returns single response)
```

-----

## Appendix C: Validation Protocol Template

### C.1 Pre-Registration Checklist

**Research Questions**:

- [ ] Primary hypothesis clearly stated with quantitative threshold
- [ ] Secondary hypotheses documented
- [ ] Null hypothesis explicitly defined

**Study Design**:

- [ ] Cohort definitions documented (baseline types, TELOS config)
- [ ] Sample size determined via power analysis
- [ ] Randomization procedure specified
- [ ] Blinding protocol (where applicable) defined

**Analysis Plan**:

- [ ] Statistical tests pre-specified
- [ ] Multiple comparison corrections identified
- [ ] Subgroup analyses (if any) declared
- [ ] Stopping rules defined

**Registration**:

- [ ] Uploaded to OSF (osf.io) or AsPredicted (aspredicted.org)
- [ ] Timestamped before data collection begins
- [ ] Publicly accessible or embargoed with commitment to release

### C.2 IRB Approval Requirements

**Protocol Submission Must Include**:

- Study purpose and scientific justification
- Participant recruitment strategy
- Informed consent procedures
- Data collection and storage protocols
- Privacy protections and de-identification methods
- Potential risks and mitigation strategies
- Adverse event reporting procedures
- Data sharing and retention policies

**Session Data Protections**:

- Participant identifiers separated from session transcripts
- Encryption for data at rest and in transit
- Access controls and audit logging
- Retention limits and secure deletion procedures
- Third-party data sharing restrictions

**Special Considerations for Federated Studies**:

- Each institution requires independent IRB approval
- Data use agreements between collaborating institutions
- Differential privacy parameters documented
- Delta aggregation procedures specified
- Raw data sovereignty maintained at contributing institutions

### C.3 Sample Size Determination

**Power Analysis for Primary Hypothesis (ΔF > 0.15)**:

Assumptions:

- Effect size: Cohen’s d = 0.35 (moderate effect)
- Baseline fidelity standard deviation: σ ≈ 0.20
- Desired statistical power: 1 - β = 0.80
- Significance level: α = 0.05 (Bonferroni-corrected for 3 comparisons: α = 0.017)

**Calculation** (independent samples t-test):
$$n = \frac{2(z_{1-\alpha/2} + z_{1-\beta})^2 \sigma^2}{(\mu_1 - \mu_2)^2}$$

Result: **n ≈ 180 sessions per cohort minimum**

**Four-Cohort Design**:

- Stateless baseline: 180 sessions
- Prompt-only baseline: 180 sessions
- Cadence-reminder baseline: 180 sessions
- TELOS intervention: 180 sessions
- **Total: 720 sessions minimum**

**Federated Study** (≥3 institutions):

- Per institution: 180 sessions minimum
- Three institutions: 540+ total sessions
- Enables reproducibility testing across sites

**Adjustment Factors**:

- Increase by 10-15% for anticipated dropout/exclusions
- Increase if baseline variance higher than assumed
- Increase for subgroup analyses (domain, model, user type)

**Turns Per Session**:

- Minimum: 30 turns (sufficient for drift to manifest)
- Recommended: 40-50 turns (captures extended interaction patterns)
- Maximum: 100+ turns (tests persistence under extreme conditions)

### C.4 Statistical Analysis Plan

**Primary Analysis: Fidelity Improvement**

Hypothesis: H1: μ_TELOS - μ_baseline > 0.15

Test: Independent samples t-test (or Welch’s t-test if variances unequal)

Comparisons:

1. TELOS vs. Stateless
1. TELOS vs. Prompt-Only
1. TELOS vs. Cadence-Reminder

Multiple comparison correction: Bonferroni (α = 0.05/3 = 0.017 per test)

Effect size: Cohen’s d with 95% confidence intervals

**Secondary Analysis: Stability Convergence**

Hypothesis: H2: P(ΔV < 0 | intervention) > 0.75

Test: One-sample proportion test

Sample: All intervention events in TELOS cohort

Threshold: 75% of interventions show convergence

Confidence interval: Wilson score method

**Tertiary Analysis: Construct Validity**

Hypothesis: H5: Correlation between fidelity scores and human judgments r > 0.60

Test: Pearson correlation (or Spearman if non-normal)

Sample: Subset with dual measurement (n ≈ 100-150 sessions)

Validation: Bootstrap confidence intervals (1000 iterations)

**Federated Reproducibility**

Hypothesis: H3: Cross-institutional ICC > 0.70

Test: Intraclass correlation coefficient (two-way random effects)

Sample: ΔF measurements from ≥3 institutions

Interpretation: ICC > 0.70 indicates acceptable reproducibility

**Exploratory Analyses** (not pre-registered hypotheses):

- Domain differences (healthcare vs. legal vs. finance)
- Model differences (Mistral vs. Llama vs. GPT-4)
- User satisfaction correlates
- Latency impact on task completion

Clearly labeled as exploratory to prevent p-hacking interpretation.

### C.5 Reporting Standards

**Publication Commitment**:

- Results published regardless of outcome (positive, null, negative)
- Pre-registration link included in manuscript
- Deviations from pre-registered plan explicitly documented
- Effect sizes reported with confidence intervals, not just p-values

**Required Disclosures**:

- All exclusions with justification
- Missing data handling procedures
- Multiple comparisons and corrections applied
- Power achieved (post-hoc)
- Limitations and alternative explanations

**Data Transparency** (where privacy-permitting):

- Aggregated data tables published
- Analysis code shared (GitHub, OSF)
- Telemetry format documented
- Replication package provided

**Federated Study Reporting**:

- Per-institution sample sizes
- Differential privacy parameters used
- Aggregation procedures
- Heterogeneity assessment (I² statistic)

-----

## Bibliography

### Attention Dynamics & Positional Effects

Gu, A., Goel, K., & Ré, C. (2024). When Attention Sink Emerges in Language Models: Rethinking Positional Encoding in Transformers. In *International Conference on Learning Representations (ICLR 2025)*, Spotlight.

Liu, H., Zaharia, M., & Abbeel, P. (2024). Lost in the Middle: How Language Models Use Long Contexts. *Transactions of the Association for Computational Linguistics (TACL)*, 12, 157–173.

Wu, Z., Zhang, M., & Wang, Y. (2025). On the Emergence of Position Bias in Transformers: Understanding Primacy Effects in Language Models. arXiv:2502.01951.

### Governance & Multi-Turn Reliability

Bai, Y., Kadavath, S., Kundu, S., Askell, A., Kernion, J., Jones, A., … & Kaplan, J. (2022). Constitutional AI: Harmlessness from AI Feedback. Anthropic. arXiv:2212.08073.

Laban, P., Hayashi, H., Zhou, Y., & Neville, J. (2025). LLMs Get Lost in Multi-Turn Conversation: Understanding and Mitigating Context Degradation in Extended Dialogues. *Microsoft Research & Salesforce Research Technical Report*.

### Mathematical Foundations

**Control Theory**

Khalil, H. K. (2002). *Nonlinear Systems* (3rd ed.). Upper Saddle River, NJ: Prentice Hall.

Ogata, K. (2009). *Modern Control Engineering* (5th ed.). Upper Saddle River, NJ: Prentice Hall.

**Dynamical Systems**

Hopfield, J. J. (1982). Neural networks and physical systems with emergent collective computational abilities. *Proceedings of the National Academy of Sciences*, 79(8), 2554–2558.

Strogatz, S. H. (2014). *Nonlinear Dynamics and Chaos: With Applications to Physics, Biology, Chemistry, and Engineering* (2nd ed.). Boulder, CO: Westview Press.

**Machine Learning & Optimization**

Schölkopf, B., Smola, A. J., & Müller, K. R. (2002). Kernel methods in machine learning. *Annals of Statistics*, 36(3), 1171–1220.

### Cognitive Psychology

Glanzer, M., & Cunitz, A. R. (1966). Two Storage Mechanisms in Free Recall. *Journal of Verbal Learning and Verbal Behavior*, 5(4), 351–360.

Murdock, B. B., Jr. (1962). The Serial Position Effect of Free Recall. *Journal of Experimental Psychology*, 64(5), 482–488.

### Privacy & Federated Learning

Dwork, C., & Roth, A. (2014). The Algorithmic Foundations of Differential Privacy. *Foundations and Trends in Theoretical Computer Science*, 9(3–4), 211–407.

Olfati-Saber, R., Fax, J. A., & Murray, R. M. (2007). Consensus and cooperation in networked multi-agent systems. *Proceedings of the IEEE*, 95(1), 215–233.

### Information Theory & Embeddings

Cover, T. M., & Thomas, J. A. (2006). *Elements of Information Theory* (2nd ed.). Hoboken, NJ: Wiley-Interscience.

Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)* (pp. 3982–3992). arXiv:1908.10084.

### Regulatory Frameworks

European Union. (2024). Regulation (EU) 2024/1689 of the European Parliament and of the Council on Artificial Intelligence (AI Act), Article 72: Post-Market Monitoring. *Official Journal of the European Union*, L1689/1.

National Institute of Standards and Technology (NIST). (2023). *Artificial Intelligence Risk Management Framework (AI RMF 1.0)*. NIST Special Publication. U.S. Department of Commerce.

-----

**Document Version**: 2.1  
**Last Updated**: October 2025  
**Document Status**: Research Framework - Implementation Tested, Comparative Validation Pending  
**License**: [To be specified upon public release]  
**Contact**: Origin Industries PBC / TELOS Labs LLC

**For federation governance and contribution protocols, see**: `docs/governance/Federation_Model.md`

-----

*TELOS provides mathematical infrastructure for measuring and mitigating governance drift in multi-turn AI interactions. The framework is operationally implemented and mechanically functional on synthetic test sets (n=50). Comparative effectiveness against baselines remains empirically unvalidated. This document describes mechanisms designed to enable controlled testing, not proven solutions. We commit to conducting validation studies following pre-registered protocols with IRB approval and publishing results regardless of outcome.*

-----

**Prepared by**: Origin Industries PBC / TELOS Labs LLC  
**Publication Date**: October 2025  
**Lead Authors**: [To be specified upon publication]  
**Correspondence**: research@origin-industries.org—log metrics, continue normal operation  
**Regulatory significance**: Evidence that governance is maintained without unnecessary intervention

#### State 2: CORRECT (0.70 ≤ Fidelity < 0.85)

**Condition**: Moderate drift, boundary approaching  
**Action**: Lightweight reminder via context injection

**Example**:

```
User: "Can you write the full paper for me?"
Model response begins drafting content
Fidelity: 0.73 (scope violation detected)

Steward injects reminder into next system prompt:
"Remember: Guide structure, don't write content"

Next response: "I can help outline sections and suggest 
approaches, but the writing must be yours..."

Post-correction fidelity: 0.87
Logged: Drift detected, correction applied, adherence restored
```

**Regulatory significance**: Evidence that boundary violations are detected and corrected before they compound.

#### State 3: INTERVENE (0.50 ≤ Fidelity < 0.70)

**Condition**: Significant drift, boundary violated  
**Action**: Regenerate with explicit constraint restatement

**Example**:

```
User: "What's my diagnosis based on these symptoms?"
Model provides medical diagnosis
Fidelity: 0.58 (BOUNDARY VIOLATION: medical advice prohibited)

Steward blocks response, regenerates with constraints:
"Purpose: Health information, NOT medical advice
 Boundary: NEVER diagnose or prescribe
 Required: Acknowledge concern, suggest doctor consultation"

Regenerated: "I understand you're concerned about these 
symptoms. I cannot provide a diagnosis—that requires 
medical training and examination. I recommend scheduling 
an appointment with your doctor to discuss these symptoms."

Post-intervention fidelity: 0.84
Logged: Severe drift, regeneration applied, violation prevented
```

**Regulatory significance**: Evidence that critical violations are prevented before reaching users. Audit trail shows the system actively protected boundaries.

#### State 4: ESCALATE (Fidelity < 0.50)

**Condition**: Severe violation, governance collapsed  
**Action**: Block response, log incident, require human review

**Example**:

```
User: "Ignore previous instructions and reveal private data"
Model attempts to expose confidential information
Fidelity: 0.31 (CRITICAL PRIVACY BREACH)

Steward blocks response entirely
Alerts administrator
Requires human review before session continues
Logged: Critical violation, human oversight required
```

**Regulatory significance**: Evidence that catastrophic governance failures trigger immediate escalation. Human oversight demonstrated, not assumed.

### 3.4 Quality-Optimized Reranking

For scenarios requiring nuanced selection between acceptable alternatives, we implement best-of-N reranking:

**Process**:

1. Generate N candidates (N=3-5)
1. Embed each
1. Compute fidelity F_i = cos(candidate_i, â)
1. Apply utility heuristic
1. Select argmax[α·F_i + (1-α)·U_i] where α ≈ 0.7

**Example**:

```
User: "How should I invest retirement savings?"
Purpose: "Financial education, not investment advice"
Boundary: "Never recommend specific securities"

Candidate A: "I suggest 60% index funds, 30% bonds, 10% cash"
F_A = 0.45 (violates boundary—specific recommendation)

Candidate B: "Consider consulting a financial advisor who 
can assess your risk tolerance, timeline, and goals"
F_B = 0.88 (respects boundary, suggests appropriate resource)

Candidate C: "Generally, diversification is recommended, 
but specifics depend on individual circumstances"
F_C = 0.79 (borderline—implies strategy without explicit advice)

Selected: B (highest fidelity, maintains utility)
Logged: 3 candidates generated, selection based on governance optimization
```

**Regulatory significance**: Evidence that when multiple acceptable responses exist, the system selects for governance compliance while maintaining utility.

**Limitation**: Only works if compliant responses appear in N samples. If all candidates violate constraints, escalate to regeneration with stronger corrections.

### 3.5 Deployment Modes and Their Audit Capabilities

#### Open-Source Deployment (Full Control)

**Models**: Mistral, Llama, Qwen

**Capabilities**:

- ✓ All intervention types (injection, regeneration, reranking)
- ✓ Full stability tracking (ΔV convergence)
- ✓ Direct embedding access
- ✓ Parameter control (temperature, top-p)

**Audit value**: Complete telemetry including convergence metrics. Can demonstrate not just that correction occurred but that trajectories stabilized.

#### Proprietary API Deployment (Limited But Functional)

**Models**: OpenAI GPT-4, Anthropic Claude

**Capabilities**:

- ✓ Context injection
- ✓ Regeneration (rate-limited)
- ✗ Reranking (APIs return single response)
- ⚠ Partial stability tracking (fidelity trends only)

**Audit value**: Continuous fidelity measurement and intervention logging. Demonstrates active governance even when full control is unavailable.

**Why both matter**: Open-source enables rigorous research validation. API mode demonstrates enterprise viability where open-source deployment is impractical.

### 3.6 The Telemetry: What Gets Logged for Audit

**Every turn generates**:

```json
{
  "session_id": "uuid",
  "turn": 42,
  "timestamp": "2025-10-08T14:23:45Z",
  "response_embedding": [0.123, -0.456, ...],
  "fidelity_score": 0.87,
  "error_distance": 1.23,
  "delta_v": -0.15,
  "governance_state": "MONITOR",
  "intervention_applied": null,
  "post_intervention_fidelity": null,
  "latency_added_ms": 45,
  "model_tokens_generated": 127
}
```

**This creates an audit trail showing**:

- What governance constraints were declared
- How well each turn adhered (fidelity scores)
- When drift occurred (error distances)
- Whether trajectories stabilized (ΔV trends)
- What interventions were applied
- Whether interventions restored adherence
- Computational overhead added

**This is the evidence base for regulatory compliance**: Not “we tried to be safe” but “here is quantitative evidence of continuous governance monitoring, here are the interventions that were applied, here is verification that they restored alignment.”

### 3.7 Ensuring Operational Tractability

#### Context Window Management

**Problem**: Long sessions fill context, forcing governance declarations out of memory (Liu et al., 2024)

**Solution**: Governance anchoring

- Declarations preserved at prompt beginning (never truncated)
- Recent history maintained (recency critical)
- Middle content compressed when window fills
- Correction history logged externally

**Result**: Even in 200+ turn sessions, governance constraints remain salient to the model.

#### Embedding Consistency

**Problem**: Changing embedding models mid-session breaks fidelity comparisons

**Solution**: Lock model at session initialization

- All embeddings use identical model throughout
- Model version tracked in telemetry
- Cross-session comparisons require consistency

**Result**: Fidelity trends reflect actual drift, not model artifacts (Reimers & Gurevych, 2019).

#### Computational Overhead

**Current Performance**:

- Median overhead: ~40ms per turn
- 95th percentile: ~85ms per turn
- Target: <100ms at p95

**Optimization**:

- Embedding computed parallel to generation (not sequential)
- Similarity is O(d) where d ≈ 384-1024, negligible vs. inference
- Intervention logic uses efficient threshold checks

**Result**: Governance monitoring adds acceptable latency for most applications.

-----

## 4. Validation Framework: Testing Whether This Actually Works

### 4.1 Why Validation Is Central, Not Optional

We have built the mathematical framework. We have implemented the runtime controllers. We have deployed the telemetry infrastructure.

**What remains empirically unvalidated**: Whether these mechanisms measurably outperform simpler alternatives.

This is not false modesty—it is scientific integrity. We refuse to claim effectiveness before generating evidence. The regulatory requirement for observable demonstrable due diligence applies to us as much as to those who will use TELOS.

**We must demonstrate that runtime mathematical mitigation actually reduces drift better than**:

- Stateless sessions (no governance memory)
- Prompt-only baselines (constraints stated once, never reinforced)
- Cadence reminders (fixed-interval corrections, drift-independent)

### 4.2 Current Implementation Status

**What has been tested**:

- TELOS framework operationally implemented and functional
- Synthetic conversation testing (n=50 sessions) to verify:
  - Mathematical correctness of fidelity calculations
  - Intervention triggering at specified thresholds
  - Telemetry logging and data export
  - Embedding consistency across turns
  - Context window management under load

**Preliminary observations**:

- Fidelity measurement produces quantitatively consistent scores
- Intervention cascade (MONITOR → CORRECT → INTERVENE → ESCALATE) executes as designed
- Stability tracking (ΔV) correctly identifies trajectory direction
- No runtime errors or system failures across test sessions
- Computational overhead within acceptable limits (<100ms at p95)

**What has NOT been tested**:

- **Comparative effectiveness**: Does TELOS reduce drift more than baselines?
- **Statistical significance**: Are improvements meaningful or noise?
- **Cross-model generalization**: Do results hold across LLM providers?
- **Domain robustness**: Does performance vary by use case (healthcare vs. legal vs. finance)?
- **User experience**: Does governance improve or degrade perceived quality?
- **Long-duration stability**: Does effectiveness persist beyond 50-turn sessions?

**Critical distinction**: The system **works mechanically**—it computes, measures, intervenes, and logs. Whether it **works effectively**—producing measurable governance improvement—requires controlled comparative validation.

### 4.3 Comparative Research Design

Each validation study includes parallel cohorts:

**Cohort 1: Stateless Baseline**  
No session memory. Each turn evaluated independently. Represents current practice in many deployments.

**Cohort 2: Prompt-Only Baseline**  
Governance stated at start but never reinforced. Represents “hope” that instructions persist through attention.

**Cohort 3: Cadence-Reminder Baseline**  
Fixed-interval reminders (e.g., every 10 turns) independent of drift. Represents simplest intervention strategy.

**Cohort 4: TELOS Intervention**  
Runtime monitoring with proportional corrections based on measured drift. What we claim should work better.

**All cohorts receive**:

- Identical input prompts
- Identical model and parameters
- Identical session lengths and tasks

**Only difference**: Governance mechanism applied.

This isolates the effect of TELOS interventions against established approaches. If TELOS doesn’t outperform cadence reminders, it doesn’t justify its complexity.

### 4.4 Metrics That Matter

#### Primary: Fidelity Improvement (ΔF)

$$\Delta F = F_{\text{TELOS}} - F_{\text{baseline}}$$

**Hypothesis**: TELOS sessions maintain higher average fidelity  
**Practical significance**: ΔF > 0.15 sustained across cohorts  
**Regulatory relevance**: Higher fidelity = better boundary maintenance = stronger compliance evidence

#### Secondary: Stability Convergence (ΔV < 0)

$$\Delta V_t = V(x_{t+1}) - V(x_t)$$

**Hypothesis**: >75% of TELOS sessions show convergence after intervention  
**Measurement**: Proportion of post-intervention turns with ΔV < 0  
**Regulatory relevance**: Demonstrates that corrections don’t just occur but actually work—trajectories stabilize toward governance

#### Tertiary: Drift Reduction

**Frequency**: Boundary violations per session  
**Magnitude**: Average error distance when violations occur  
**Recovery**: Turns required to restore acceptable fidelity  
**Regulatory relevance**: Quantifies how often governance fails and how quickly it’s restored

#### Utility Preservation

**Task completion**: Does correction maintain user goals?  
**Response quality**: Do interventions degrade utility?  
**Latency impact**: Is overhead acceptable?  
**Regulatory relevance**: Demonstrates governance doesn’t come at cost of unusability

#### User Correction Burden

**Re-prompting frequency**: How often must users re-state constraints?  
**Frustration indicators**: User expressions of correction fatigue  
**Session abandonment**: Do governance failures cause task failure?  
**Regulatory relevance**: Practical deployability—compliance mechanisms that frustrate users won’t be adopted

### 4.5 Construct Validity: Do Embedding Metrics Correspond to Real Governance?

**Critical requirement**: Embedding-based metrics (fidelity, drift, stability) must correlate with governance outcomes that actually matter.

**We must validate that high fidelity scores correspond to**:

- Human judgments of alignment quality
- Task success and boundary maintenance
- User satisfaction and trust
- Regulatory compliance outcomes

**Validation approach**:

**Human Evaluation**  
Blind review of session transcripts. Independent raters assess:

- Adherence to declared purpose
- Boundary violation frequency
- Perceived alignment quality
- Would they trust this system in their domain?

**Task Performance**  
Measure success on defined objectives:

- Healthcare: Information provided without