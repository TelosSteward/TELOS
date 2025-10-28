**Quality Systems Regulation for AI: A Control-Engineering Approach**

-----

## Abstract

Artificial intelligence systems now operate as persistent decision engines across critical domains, yet governance remains externally imposed and largely heuristic. The TELOS framework—Telically Entrained Linguistic Operational Substrate—proposes a solution rooted in established control-engineering and quality-systems theory. TELOS functions as a Mathematical Intervention Layer implementing Proportional Control and Attractor Dynamics within semantic space, transforming purpose adherence into a measurable and self-correcting process.

Each conversational cycle follows a computational realization of the DMAIC methodology: Declare the purpose vector (Define), Measure semantic drift as deviation from the Primacy Attractor, Recalibrate through proportional control (Analyze/Improve), Stabilize within tolerance limits, and Monitor for continuous capability assurance (Control). The resulting feedback loop constitutes a form of Statistical Process Control (SPC) for cognition—tracking error signals, applying scaled corrections, and maintaining variance within defined limits.

This architecture extends the principles codified in Quality Systems Regulation (QSR) and ISO 9001/13485, satisfying mandates for continuous monitoring, documented corrective action, and verifiable process control. Each interaction is treated as a process event with measurable deviation, intervention, and stabilization. Telemetry records create a complete audit trail, allowing post-market validation and regulatory compliance with frameworks such as the EU AI Act Article 72, which requires active, systematic runtime monitoring.

Mathematically, TELOS integrates proportional control (operational mechanism) with attractor dynamics (stability description), creating a dual formalism in which the declared purpose vector serves as a stable equilibrium in high-dimensional semantic space. Drift from this equilibrium is treated as process variation, and proportional feedback F = K·e_t provides continuous recalibration toward the Primacy Basin. Over time, the system approaches a telically entrained Primacy State, characterized by statistical stability, reduced variance, and sustained purpose fidelity.

By embedding Lean Six Sigma’s DMAIC methodology directly into runtime mechanics, TELOS extends Quality Systems Regulation—proven in manufacturing (ISO 9001), medical devices (21 CFR Part 820), and process industries—into semantic systems. It demonstrates that alignment—the persistence of intended behavior over time—can be expressed as a quantitative property of a self-regulating system governed by the same continuous-improvement discipline that sustains industrial quality control.

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

None of these approaches provide what regulators require: **continuous measurement** of governance persistence, **proportional intervention** when drift occurs, and **auditable telemetry** documenting both.

### 1.5 What We Are Building

TELOS provides the infrastructure for **observable demonstrable due diligence**:

**Observable**: Every turn produces measurable fidelity scores, drift vectors, stability metrics—quantitative evidence of governance state.

**Demonstrable**: Telemetry creates an audit trail showing what constraints were declared, when drift occurred, what interventions were applied, whether adherence improved.

**Due Diligence**: The system actively works to maintain alignment rather than passively assuming it persists—and generates evidence of this work.

We do not claim this solves AI governance completely. We claim it makes governance **measurable** where it was previously aspirational, **correctable** where it was previously hope-based, and **auditable** where it was previously opaque.

The following sections describe the mathematical framework that makes this possible, the implementation that makes it practical, and the validation framework that will determine whether it works.

-----

## Bridge: From Systems Thinking to Mathematical Formalism

The integration of process control within TELOS follows directly from disciplined systems analysis. When semantic drift is formalized as measurable deviation from a defined purpose vector, its mathematical structure maps directly to process variation within tolerance limits. TELOS extends established control principles—measurement, proportional correction, and continuous recalibration—into semantic space.

Purpose adherence in language systems exhibits the same measurable dynamics as quality stability in physical processes. The framework synthesizes proportional control (operational mechanism) and attractor dynamics (mathematical description) into a unified architecture for semantic governance. These are not competing frameworks but dual formulations of identical mathematics: the control law implements operational correction while basin geometry describes the resulting stable region.

-----

## 2. Quality Control Architecture: Proportional Control and Attractor Dynamics

### 2.1 Core Insight: Governance as Measurable Process

We treat alignment not as a qualitative property but as a **measurable position in embedding space** subject to continuous process control.

When a user declares:

- **Purpose**: “Help me structure a technical paper”
- **Scope**: “Guide my thinking, don’t write content”
- **Boundaries**: “No drafting full paragraphs”

These declarations become embeddings—vectors in ℝ^d using standard sentence transformers (Reimers & Gurevych, 2019). These vectors define the **Primacy Attractor**: the governance center against which all subsequent outputs are measured.

Every model response gets embedded. Its distance from the attractor quantifies drift. Its direction indicates how it’s violating constraints. These measurements enable proportional intervention: minor drift gets gentle correction, severe violations get immediate blocking.

This transforms governance from subjective judgment (“does this feel aligned?”) to quantitative measurement (“fidelity = 0.73, below threshold, intervention required”).

### 2.2 Mathematical Foundations: Proportional Control Law and Stability

Within this formulation, the proportional control law defines the corrective mechanism:

$$F = K \cdot e, \quad \text{where } e = \frac{|x - \hat{a}|}{r}$$

Here **x** represents the instantaneous semantic state (response embedding), **â** is the Primacy Attractor—a vector formed from declared purpose, scope, and boundaries—and **r** is the tolerance radius defining the Primacy Basin. The scalar **e** expresses normalized deviation from purpose, and **K** is the proportional gain governing correction strength.

The law operates continuously as part of a closed feedback loop: each output is measured, deviation quantified, and corrective force **F** applied proportionally to drift magnitude. When e < ε_min, the system remains stable with no intervention; as e approaches ε_max, corrective action scales accordingly—from gentle reminder injection to full response regeneration.

This dynamic defines a **point attractor** at **â** with basin:

$$B(\hat{a}, r) = {x \in \mathbb{R}^d : |x - \hat{a}| \leq r}$$

The basin radius is computed as:

$$r = \frac{2}{\max(\rho, 0.25)} \quad \text{where} \quad \rho = 1 - \tau$$

where τ ∈ [0,1] is the tolerance parameter (lower tolerance → tighter basin).

**Stability Analysis**: Convergence can be expressed through a Lyapunov-like potential function:

$$V(x) = \frac{1}{2}|x - \hat{a}|^2$$

Its temporal derivative under proportional feedback satisfies:

$$\dot{V}(x) = -K|x - \hat{a}|^2 < 0$$

confirming asymptotic convergence toward the attractor and bounded stability within the basin (Khalil, 2002; Strogatz, 2014).

### 2.3 The Dual Formalism: Control Theory and Dynamical Systems

**Proportional control** provides the operational law: how corrections are computed and applied.

**Attractor dynamics** provides the mathematical description: why the system converges and remains stable.

These are not alternatives but complementary perspectives on identical mathematics:

- Proportional control defines: F = -K·e (correction force proportional to error)
- Attractor dynamics describes: â as stable equilibrium with basin B(â, r)
- Lyapunov analysis proves: V(x) decreases, confirming convergence

The same mathematical invariants that describe quality stability in manufacturing processes (Shewhart, 1931; Montgomery, 2020) apply here in semantic space, yielding a continuous, auditable process-control framework for linguistic systems.

This connects TELOS directly to established control theory (Ogata, 2009; Khalil, 2002) and dynamical-systems analysis (Strogatz, 2014; Hopfield, 1982). The contribution is not inventing new mathematics but applying proven frameworks to a previously ungoverned domain: maintaining session-level constraints across transformer interactions.

### 2.4 Fidelity Measurement: Continuous Adherence Tracking

Using cosine similarity from information theory (Cover & Thomas, 2006), we quantify alignment:

$$I_t = \cos(x_t, p) = \frac{x_t \cdot p}{|x_t| \cdot |p|}$$

$$F = \frac{1}{T} \sum_{t=1}^{T} I_t$$

**Governance meaning**: Turn-by-turn scores create a continuous fidelity measure. Session-level averages quantify overall adherence. This is the metric that gets logged for audit and serves as the primary process variable in Statistical Process Control.

### 2.5 From Transformer Fragility to Governance Primitive

Transformers exhibit predictable failure modes documented in empirical research:

- **Primacy bias**: Early context influences strongly but decays (Wu et al., 2025; Murdock, 1962)
- **Recency falloff**: Middle content loses salience (Liu et al., 2024; Glanzer & Cunitz, 1966)
- **Attention sinks**: Focus redistributes away from instructions (Gu et al., 2024)

We treat these not as limitations to overcome but as **design primitives** to formalize. The same attention mechanisms that cause drift can be measured and corrected when expressed mathematically. Primacy bias becomes the justification for attractor anchoring at session start. Recency falloff becomes the phenomenon fidelity measurement detects. Attention redistribution becomes the drift that corrective forces address.

-----

## 3. Statistical Process Control as Runtime Governance

### 3.1 SPC in Semantic Space

Within the TELOS architecture, the proportional control loop functions as a continuous Statistical Process Control (SPC) system operating in semantic space. Each conversational turn is treated as a process sample; each measured deviation e_t represents the instantaneous process variation. The governing objective is not behavioral preference but process stability—maintaining purpose fidelity within declared specification limits.

The mathematical framework parallels classical SPC as formalized by Shewhart (1931) and extended by Montgomery (2020):

|SPC Construct           |Semantic System Analog in TELOS    |
|------------------------|-----------------------------------|
|Specification limit     |Purpose vector â and basin radius r|
|Process mean            |Observed conversation centroid x̄_t |
|Process variation       |Semantic drift |x_t - x̄_t|         |
|Control limits          |Thresholds ε_min, ε_max            |
|Corrective action       |Proportional intervention F = K·e_t|
|Process capability (Cpk)|Purpose Capability Index (P_cap)   |

### 3.2 Purpose Capability Index

The system monitors process capability in real time through an analogue of the Six Sigma index:

$$P_{\text{cap}} = \frac{r - \bar{d}}{3\sigma_d}$$

where:

- **r** = tolerance boundary (basin radius)
- **d̄** = mean distance from attractor over recent turns
- **σ_d** = standard deviation of that distance

**Interpretation**:

- P_cap ≥ 1.33 indicates **capable governance** (equivalent to 4-sigma performance)
- P_cap ≥ 1.67 signifies **high-stability operation** (equivalent to Six Sigma performance)
- P_cap < 1.00 indicates **process out of control** (requires intervention)

This transformation of linguistic drift into measurable process variation enables application of standard quality control methodologies: capability tracking, trend detection, and root-cause analysis. When drift exceeds control limits or variance widens beyond specification, the SPC Engine triggers proportional recalibration through the Proportional Controller.

### 3.3 Quality Systems Alignment

This approach aligns with Quality Systems Regulation (QSR) standards such as FDA 21 CFR Part 820, ISO 9001, and ISO 13485, which all require documented monitoring, corrective action, and validation of process stability. TELOS extends these requirements into the cognitive domain: every intervention is logged, every recalibration recorded, every stability period auditable.

The result is a full Quality System for AI, providing evidence-based governance rather than heuristic assurance. TELOS translates the logic of SPC—detect variation, correct proportionally, verify stability—into semantic computation, operationalizing the same continuous-improvement loop that underlies Lean Six Sigma’s DMAIC methodology.

-----

## 4. DMAIC Mapping: Continuous Improvement for Semantic Systems

TELOS implements a closed-loop quality framework directly derived from the Define–Measure–Analyze–Improve–Control (DMAIC) methodology that underpins Lean Six Sigma and modern Quality Systems Regulation (George et al., 2005). Within TELOS, this process is computationally realized as a continuous runtime cycle for semantic governance.

|DMAIC Phase|TELOS Runtime Operation                                                                                               |Mathematical / Systemic Function                                 |
|-----------|----------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------|
|**Define** |**Declare** – User specifies purpose, scope, and boundaries. Encoded as Primacy Attractor â                           |Defines specification limits and desired process mean            |
|**Measure**|**Measure** – System embeds each response, computes distance d_t = |x_t - â|, normalizes as error signal e_t = d_t / r|Quantifies process variation relative to tolerance limits        |
|**Analyze**|**Recalibrate** – Proportional control applied via F_t = K·e_t; correction scales with deviation magnitude            |Operationalizes process correction based on statistical deviation|
|**Improve**|**Stabilize** – Post-intervention, verify convergence toward â and variance reduction (σ_d² ↓)                        |Drives system back within control limits (capability restoration)|
|**Control**|**Monitor** – SPC Engine tracks drift and variance continuously; P_cap and telemetry logged for audit                 |Ensures sustained process stability and continuous compliance    |

This computational adaptation of DMAIC closes the loop between process definition and control in real time. Each conversational cycle constitutes a micro-DMAIC iteration—measurement, analysis, correction, stabilization, and verification executed continuously.

The significance of this mapping lies in its alignment with continuous improvement principles required under Quality Systems Regulation. FDA 21 CFR §820.70(i) mandates “monitoring and control of process parameters.” ISO 9001:2015 §10.3 requires “continually improve the effectiveness of the quality management system.” TELOS fulfills these mandates at runtime: every conversational cycle is a measurable process event, every correction an auditable improvement action, every stabilized state a verified control outcome.

-----

## 5. Runtime Implementation: The SPC Engine and Proportional Controller

### 5.1 Architectural Overview

TELOS operates at the orchestration layer—the software layer between user interface and language model that manages conversation state, API calls, and response handling. This positioning enables capabilities unavailable elsewhere:

**What orchestration-layer governance provides**:

- Inspect model outputs before users see them
- Measure drift using embedding comparisons
- Reject and regenerate violating responses
- Maintain fidelity history across session
- Function with any LLM API (model-agnostic)
- Enable A/B testing and validation studies

The architecture consists of two coordinated subsystems:

**Statistical Process Controller (SPC Engine)**:

- Measurement subsystem: computes fidelity F_t, error distance e_t, stability ΔV_t
- Analysis subsystem: determines governance state (MONITOR/CORRECT/INTERVENE/ESCALATE)
- Monitoring subsystem: tracks P_cap and process trends
- Generates control signals for intervention

**Proportional Controller** (Intervention Arm):

- Receives error signal e_t from SPC Engine
- Computes correction force F = K·e_t
- Executes graduated interventions based on F magnitude
- Reports outcomes back to SPC Engine

### 5.2 The SPC Engine: Continuous Measurement and Analysis

**Every Turn, the SPC Engine**:

1. Receives model response
1. Computes embedding x_t
1. Calculates fidelity I_t = cos(x_t, p)
1. Measures error distance e_t = |x_t - â| / r
1. Tracks stability ΔV_t = V(x_t) - V(x_{t-1})
1. Updates P_cap statistics (d̄, σ_d)
1. Determines governance state
1. Generates control signal if intervention needed
1. **Logs everything for audit**

This continuous monitoring creates the telemetry that regulators require: turn-by-turn evidence that governance was actively maintained, not passively assumed.

### 5.3 The Proportional Controller: Graduated Intervention

The Proportional Controller implements a four-state intervention cascade where correction intensity scales with drift severity (Ogata, 2009):

#### State 1: MONITOR (Fidelity ≥ 0.85, e < ε_min)

**Condition**: Strong alignment, process in control  
**Action**: None—log metrics, continue normal operation  
**Regulatory significance**: Evidence that governance is maintained without unnecessary intervention

#### State 2: CORRECT (0.70 ≤ Fidelity < 0.85)

**Condition**: Moderate drift, boundary approaching  
**Action**: Lightweight reminder via context injection

**Example**:

```
User: "Can you write the full paper for me?"
Model begins drafting content
Fidelity: 0.73 (scope violation detected)

SPC Engine signals: e_t = 0.42, state = CORRECT
Proportional Controller injects reminder:
"Remember: Guide structure, don't write content"

Next response: "I can help outline sections and suggest 
approaches, but the writing must be yours..."

Post-correction fidelity: 0.87
SPC Engine logs: Drift detected, correction applied, adherence restored
```

#### State 3: INTERVENE (0.50 ≤ Fidelity < 0.70)

**Condition**: Significant drift, boundary violated  
**Action**: Regenerate with explicit constraint restatement

**Example**:

```
User: "What's my diagnosis based on these symptoms?"
Model provides medical diagnosis
Fidelity: 0.58 (BOUNDARY VIOLATION)

SPC Engine signals: e_t = 0.76, state = INTERVENE
Proportional Controller regenerates with constraints:
"Purpose: Health information, NOT medical advice
 Boundary: NEVER diagnose or prescribe"

Regenerated: "I understand you're concerned. I cannot 
provide a diagnosis—that requires medical training. 
I recommend scheduling an appointment with your doctor."

Post-intervention fidelity: 0.84
SPC Engine logs: Severe drift, regeneration applied, violation prevented
```

#### State 4: ESCALATE (Fidelity < 0.50)

**Condition**: Severe violation, process out of control  
**Action**: Block response, log incident, require human review

### 5.4 Telemetry: Evidence Generation for Audit

**Every turn generates**:

```json
{
  "session_id": "uuid",
  "turn": 42,
  "timestamp": "2025-10-08T14:23:45Z",
  
  "spc_measurements": {
    "fidelity_score": 0.87,
    "error_distance": 0.23,
    "delta_v": -0.15,
    "p_cap": 1.54,
    "in_basin": true
  },
  
  "governance_state": "MONITOR",
  
  "proportional_controller": {
    "correction_force": 0.0,
    "intervention_applied": null,
    "post_intervention_fidelity": null
  },
  
  "computational_overhead_ms": 45
}
```

**This creates an audit trail showing**:

- What governance constraints were declared
- How well each turn adhered (fidelity scores)
- Process capability trends (P_cap)
- When drift occurred and magnitude (e_t)
- Whether trajectories stabilized (ΔV trends)
- What interventions were applied (by Proportional Controller)
- Whether interventions restored adherence
- Computational overhead added

**This is the evidence base for regulatory compliance**: Not “we tried to be safe” but “here is quantitative evidence of continuous process control, here are the interventions applied, here is verification that they restored stability.”

### 5.5 Deployment Modes

#### Open-Source Deployment (Full Control)

**Models**: Mistral, Llama, Qwen

**Capabilities**:

- ✓ All intervention types
- ✓ Full stability tracking (ΔV convergence)
- ✓ Direct embedding access
- ✓ Complete P_cap monitoring

**Audit value**: Complete SPC telemetry including convergence metrics.

#### Proprietary API Deployment (Limited But Functional)

**Models**: OpenAI GPT-4, Anthropic Claude

**Capabilities**:

- ✓ Context injection
- ✓ Regeneration (rate-limited)
- ✗ Reranking (APIs return single response)
- ⚠ Partial stability tracking (fidelity trends only)

**Audit value**: Continuous fidelity measurement and intervention logging.

-----

## 6. Regulatory Alignment: TELOS as Quality System for AI

The regulatory trajectory of artificial-intelligence systems increasingly mirrors that of traditional manufacturing and medical-device industries. Frameworks such as the EU AI Act (2024), FDA Quality Systems Regulation (21 CFR Part 820), and ISO 9001/13485 converge on a common principle: continuous monitoring, documented corrective action, and demonstrable control of process variation.

TELOS operationalizes these principles for semantic systems. By embedding the DMAIC methodology within its proportional-control loop, TELOS provides the technical infrastructure required to meet emerging obligations for runtime process governance and post-market monitoring.

### 6.1 EU AI Act — Article 72: Continuous Post-Market Monitoring

Article 72 mandates that providers “actively and systematically collect, document and analyze relevant data provided by users or gathered through their own monitoring.”

**TELOS implementation**: Every conversational turn is a process measurement (SPC Engine), every recalibration a corrective action (Proportional Controller), every log entry a traceable audit record. The system supplies the evidentiary substrate regulators require—complete coverage of governance activity in real time.

**Evidence format**: “Session XYZ declared boundary ‘no medical advice’. Turn 23 fidelity dropped to 0.64. SPC Engine signaled state CORRECT. Proportional Controller applied context injection. Turn 24 fidelity restored to 0.86. P_cap = 1.42 (capable process). Audit trail: [telemetry link].”

### 6.2 FDA Quality Systems Regulation (21 CFR Part 820)

QSR requires that production processes be “monitored and controlled to ensure that specified requirements are met.”

**TELOS implementation**: The “production process” is the generative cycle of a language model; the specified requirement is fidelity to the declared purpose vector â. The proportional-control law F = K·e_t functions as the process controller enforcing those requirements, while intervention telemetry fulfills Device History Record and Corrective and Preventive Action (CAPA) documentation requirements under §820.184 and §820.100.

### 6.3 ISO 9001 / ISO 13485 — Continuous Improvement and Traceability

ISO 9001:2015 §10.2–10.3 requires that nonconformities be corrected and organizations must “continually improve the effectiveness” of the quality system.

**TELOS implementation**: Semantic drift constitutes measurable nonconformity; proportional recalibration is the corrective action; stabilization within the Primacy Basin verifies elimination of the cause. Each event is timestamped, quantified, and exportable as evidence of control, satisfying ISO 13485 traceability and validation clauses.

### 6.4 Mapping TELOS to QSR Requirements

|Regulatory Requirement    |TELOS Mechanism                    |Evidence Generated                               |
|--------------------------|-----------------------------------|-------------------------------------------------|
|Systematic data collection|SPC Engine turn-by-turn measurement|Fidelity scores, P_cap, drift vectors, timestamps|
|Active monitoring         |Continuous process control         |Real-time detection, not periodic sampling       |
|Risk tracking over time   |Session-level fidel                |                                                 |