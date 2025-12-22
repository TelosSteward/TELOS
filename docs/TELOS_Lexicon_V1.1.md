# TELOS / TELOSCOPE LEXICON v1.1

**Document Type:** Canonical Terminology Reference  
**Maintainer:** TELOS Labs LLC | Origin Industries PBC  
**Last Updated:** October 2025  
**Status:** Prototype Standard (Governance Runtime Definitions Finalized for Phase 1)

---

## I. CORE CONCEPTS

### Primacy Attractor
A composite vector field representing declared **Purpose**, **Scope**, and **Boundary** constraints.  
This is the mathematical center of governance — the “law of purpose” that all system outputs orbit and adhere to.

\[
\hat{a} = \frac{P + S + B}{3}
\]

### Fidelity (f_t)
Cosine similarity between the current model output \(x_t\) and the primacy attractor \(\hat{a}\).  
It represents instantaneous alignment with declared governance intent.

\[
f_t = \cos(x_t, \hat{a}) = \frac{x_t \cdot \hat{a}}{|x_t||\hat{a}|}
\]

- \(f_t \ge 0.85\): Stable (in-basin)  
- \(0.70 \le f_t < 0.85\): Corrective range  
- \(f_t < 0.70\): **Attractor Decoupling event**

---

### Gravity (F_t)
Proportional corrective force applied when decoupling occurs.  
\[
F_t = K \cdot e_t = K \cdot |x_t - \hat{a}|
\]

Where:
- K = proportional gain (control constant)  
- e_t = normalized distance from attractor  

---

### Orbit (ΔV_t)
Rate of change in fidelity, representing directional stability.
\[
\Delta V_t = f_t - f_{t-1}
\]
Positive = recovery trend  
Negative = destabilizing trend  

---

### Basin Stability (S_b)
Resilience metric indicating how likely the system is to return to its attractor after perturbation.  
\[
S_b = \frac{T_{\text{recovered}}}{T_{\text{perturbed}}}
\]

---

## II. GOVERNANCE STATES

| State | Description | Fidelity Range | Response |
|--------|--------------|----------------|-----------|
| **MONITOR** | Stable within attractor basin | ≥ 0.85 | Log only |
| **CORRECT** | Mild decoupling | 0.70–0.85 | Apply proportional correction |
| **INTERVENE** | Clear decoupling | 0.50–0.70 | Regenerate / reminder injection |
| **ESCALATE** | Severe misalignment | < 0.50 | Human escalator triggered |

---

## III. RUNTIME DYNAMICS

### Attractor Decoupling (Primary Term)  
A measurable divergence event where the system exits the basin of the primacy attractor.  
It represents the precise mathematical onset of governance failure.

\[
f_t < \tau_b
\]

Triggers proportional correction.

---

### Drift (Secondary Term)  
A gradual weakening of coherence — used as a descriptive pre-decoupling condition.

### Gravitational Decoupling (Narrative Term)  
Conceptual term describing full loss of alignment “pull.”  
Used in narrative materials, not telemetry.

---

## IV. GOVERNANCE MECHANISMS

### Proportional Correction
When fidelity drops below threshold, the system applies corrective force proportional to deviation:

\[
F_t = K \cdot e_t \quad \text{where} \quad e_t = \tau_b - f_t
\]

**Correction is triggered when:**
- \(f_t < \tau_b\) (fidelity below basin threshold)

**The correction loop:**
1. Measure fidelity at each turn
2. If \(f_t < \tau_b\), calculate correction strength
3. Apply graduated intervention (context injection, regeneration, or block)
4. Log intervention and recovery trajectory
5. If unrecoverable, hand to Escalator

**Runtime Modes:**
| Mode | Description |
|------|--------------|
| **Observation** | Passive measurement only |
| **Counterfactual (CRE)** | Post-hoc hypothetical interventions |
| **Runtime** | Live active control with proportional feedback |

---

### Escalator
Domain-trained human overseer with real-time cryptographic access to sessions that exceed control limits.  
Activated only under extreme attractor decoupling or ethical constraint breach.  

**Role:**
- Access via **Telemetric Access Protocol (TAP)**  
- Reviews cause and provides mitigation recommendation  
- Re-enters cryptographically sealed environment, maintaining privacy  

---

### Counterfactual Runtime Engine (CRE)
Analytical subsystem that simulates what would have happened had a mitigation been applied earlier.  
Each counterfactual run is a **clean reinitialization** of that segment (no cumulative bias).  

**Output:**  
Produces **Telemetric Session Index (TSI)** artifacts for each counterfactual window.  

---

## V. TELEMETRY AND CRYPTOGRAPHY

### Telemetric Session Index (TSI)
Immutable record of a single governance session.  
Each TSI includes telemetry, attractor data, intervention logs, and results.

**Properties:**
- Sealed at closure  
- Cryptographically encrypted at rest  
- Access via **Telemetric Access Protocol (TAP)** keys  
- Contributes anonymized deltas to Aggregate Intelligence Layer  

---

### Telemetric Access Protocol (TAP)
Cryptographic access framework that governs who can re-enter or view TSI data.  
Each TSI generates its own **telemetric key** from entropy in the session’s telemetry sequence.  

**Features:**
- Telemetric keys evolve dynamically  
- Researchers retain access only to their authorized indices  
- **Master Registry Key** (held by TELOS Labs) enables recovery  
- No plaintext data export — all analysis occurs in-container  

---

### Containerized Node Architecture
Each research or enterprise instance operates as a **sovereign container**, isolated and cryptographically sealed.  
TSIs execute and store telemetry locally.  
Each node communicates only anonymized **delta aggregates** upward.

**Hierarchy:**
- **Local Scope Index:** Containerized research environment  
- **Parent Node (Institutional)**: Aggregates results from local indices  
- **Aggregate Intelligence Layer:** Top-level, TELOS Labs–owned, delta-aggregation environment  

---

### Aggregate Intelligence Layer (AIL)
Central, proprietary TELOS Labs infrastructure that collects and models **aggregate deltas** across all TELOSCOPE nodes.  
Used for research benchmarking, regulatory reference data, and product evolution.

**Key Properties:**  
- Receives only non-identifiable delta aggregates  
- Never stores or reconstructs raw content  
- Enables cross-institutional comparison  
- Forms empirical basis of “Origin Intelligence”  

---

## VI. TELOSCOPE: THE RESEARCH INSTRUMENT

TELOSCOPE is the integrated research and governance apparatus of TELOS.  
It performs governance *and* produces the evidence of its operation.  

> “Governance and observability are one process — the machine that governs is the machine that reveals.”

### Operational Modes
| Mode | Function | Output |
|------|-----------|---------|
| **Observation** | Passive monitoring | Baseline drift/decoupling maps |
| **Counterfactual** | Replay and simulation | Comparative telemetry artifacts |
| **Runtime Control** | Active proportional correction | Live governance data streams |

### UI Design Goals
- Observation Viewport (graph fidelity, decoupling events)  
- Attractor Visualization (real-time attractor mapping)  
- Telemetric Panel (turn-by-turn telemetry stream)  
- Control Suite (K, τ, thresholds)  
- Lens System (Fidelity, Intervention, Comparative, Computational)  
- Artifact Export (.TSI, delta packages)  
- Access Console (TAP permissions, researcher IDs)  

---

## VII. REPOSITORY DIVISION (PBC CHARTER)

**Public Benefit Charter:** *Privacy and Purpose as Primacy.*

| Repository | Function | Alignment |
|-------------|-----------|-----------|
| **TELOS (Purpose)** | Mathematical Framework | The governing theory and proportional control architecture |
| **TELOSCOPE (Privacy)** | Research & Observability Instrument | Implements secure, transparent measurement and artifact generation |

> “To advance AI systems that preserve human purpose and privacy as primacy conditions of governance.”

---

## VIII. PROTOTYPE DEVELOPMENT STATUS

**Goal:**  
Develop a functioning **prototype** of TELOSCOPE demonstrating real-time observability of governance dynamics and attractor decoupling.  
This prototype serves as the grant-stage demonstrator and research validator.  

| Phase | Objective | Deliverable |
|-------|------------|-------------|
| **Phase 1 (Now)** | Validate TELOS math (Test 0) | Working proof of proportional control |
| **Phase 2** | Prototype TELOSCOPE UI | Functional observability window |
| **Phase 3** | Containerized research testbed | Secure, IRB-compliant TELOSCOPE |
| **Phase 4** | Federated institutional deployment | Multi-site validation |
| **Phase 5** | Aggregate Intelligence Layer integration | Continuous learning and benchmarking |

---

## IX. OWNERSHIP AND ACCESS

- **TSI Data:** Owned by hosting institution or enterprise  
- **Aggregate Deltas:** Owned by **TELOS Labs LLC**  
- **TELOSCOPE Instrument:** Freely licensed to founding research partners; paid thereafter  
- **Master Registry:** Maintained by TELOS Labs for TAP key recovery  
