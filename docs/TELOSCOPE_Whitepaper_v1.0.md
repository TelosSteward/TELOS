# TELOSCOPE — Research Apparatus Whitepaper
*Version 1.0 — Beta Instrument Specification and Research Protocol*

---

## 1. What TELOSCOPE Is

**TELOSCOPE** is both a **governance apparatus** and a **research instrument** — unified in a single system.

- As a **governance apparatus**, it is the **Steward** — the active mitigation and proportional control machine maintaining **fidelity** to **Primacy Attractors** (Purpose, Scope, and Boundaries).
- As a **research instrument**, it is the **TELOSCOPE interface**, which observes, visualizes, and records **telemetric evidence** of governance performance in real time.

> "TELOSCOPE doesn't just observe governance — it performs it.
>  The same machine that governs is the one that reveals."

**Current Status:** TELOSCOPE is operating as a **beta research instrument** with active users generating real-world telemetric data. The system has progressed from theoretical validation (Test 0) to live deployment with human researchers, marking the transition from proof-of-concept to empirical research apparatus.

---

## 2. The Steward

The **Steward** is the core mitigation engine — not a chatbot, but the active governance layer itself.

It continuously:
- Measures **fidelity** between outputs and Primacy Attractors using **Dual Primary Attractor (DPA)** mathematics
- Detects **Attractor Decoupling** — measurable divergence from declared Purpose, Scope, and Boundaries
- Applies proportional corrections scaled to deviation magnitude
- Logs interventions and stability changes as **telemetry**
- Generates **Telemetric Keys (TKeys)** for cryptographic session sealing

The Steward **is** the governance process: mathematical, measurable, and self-correcting.

### 2.1 Dual Primary Attractor Architecture

The Steward operates through a **dual attractor system** where:

- **Primary Attractor A (Purpose)**: Represents the declared purpose and scope
- **Primary Attractor B (Boundaries)**: Represents constraints, limitations, and guardrails

Each response is measured against both attractors simultaneously, creating a two-dimensional governance space where:

```
F_A(r, A) = cos(φ(r), φ(A))  // Fidelity to Purpose
F_B(r, B) = cos(φ(r), φ(B))  // Fidelity to Boundaries
```

**Attractor Decoupling** occurs when either fidelity measure drops below threshold, triggering proportional intervention.

### 2.2 Proportional Control System

The Steward implements classical control theory principles:

- **Error Signal**: e(t) = θ_target - F(r, A)
- **Proportional Response**: Intervention strength scales with deviation magnitude
- **Recovery Time**: Measured time to return to stable fidelity zone
- **Control Strength**: K_p constant determining intervention intensity

This creates a **self-stabilizing governance loop** that maintains alignment without manual intervention.

---

## 3. The Research Instrument

The **TELOSCOPE interface** provides researchers a transparent and cryptographically secure window into runtime governance.

### 3.1 Operating Modes

1. **Observation Mode:** Passive measurement; no interventions. Pure telemetry collection.
2. **Counterfactual Mode:** Retroactive "what-if" simulations; clean replay with alternative parameters.
3. **Runtime Mode:** Full TELOS operation — mitigation and evidence capture.

Each mode produces **Telemetric Session Indexes (TSIs)** — encrypted, versioned research artifacts containing all telemetry and proportional control data.

### 3.2 Beta Instrument Status

**As of November 2025**, TELOSCOPE is deployed as a beta research instrument with:

- **Active Users**: Researchers and beta testers generating real conversational data
- **Live Telemetry**: Turn-by-turn fidelity tracking and intervention logging
- **Steward Integration**: LLM-powered Steward providing governance analysis and insights
- **Cryptographic Sealing**: All sessions sealed with TKeys for privacy and integrity
- **Research Protocol**: IRB-aligned data collection and anonymization

**Beta Objectives:**

1. Validate mathematical assumptions in real-world conversations
2. Measure human perception of drift vs. mathematical detection
3. Optimize proportional control parameters (K_p, thresholds)
4. Generate empirical data for regulatory compliance demonstration
5. Test cryptographic infrastructure and privacy guarantees

---

## 4. Telemetric Evidence

Each Steward generates structured **telemetric evidence** of governance operations.

This includes:

- **Fidelity curves per attractor** (F_A, F_B over time)
- **Attractor Decoupling events** (magnitude, recovery, stability)
- **Proportional correction logs** (intervention strength, timing, effectiveness)
- **Recovery metrics** (time to stabilize, overshoot, oscillation)
- **Process capability** (P_cap metrics, Cpk scores, statistical control)
- **Session metadata** (turn count, duration, user interactions)

This converts governance into **observable, reproducible evidence** suitable for peer review and regulatory submission.

### 4.1 Telemetric Keys (TKeys)

**TKeys** are cryptographic artifacts generated from session telemetry, providing both **sealing** and **signature** functions.

**TKey Generation:**

1. **Telemetry Entropy Collection**: Fidelity values, timestamps, intervention points collected
2. **Deterministic Derivation**: HMAC-based key derivation from telemetry stream
3. **Session Binding**: TKey uniquely identifies and seals specific session
4. **Forward Security**: Historical TKeys cannot be used to decrypt future sessions

**TKey Signature for Deltas:**

Each session generates a **TKey Signature** — a cryptographic fingerprint of governance telemetry deltas:

```
TKey_Sig = H(F_A_deltas || F_B_deltas || intervention_events || timestamp_chain)
```

Where:
- `F_A_deltas`: Changes in Purpose fidelity per turn
- `F_B_deltas`: Changes in Boundary fidelity per turn
- `intervention_events`: Timing and magnitude of all interventions
- `timestamp_chain`: Hash-chained timestamp sequence

**Properties:**
- **Tamper-Evident**: Any modification to session data invalidates signature
- **Privacy-Preserving**: Signature reveals governance metrics without exposing conversation content
- **Aggregate-Ready**: Signatures can be combined for anonymous federated learning

**Use Cases:**
- Verify session integrity without decrypting content
- Submit governance evidence to regulators without privacy breach
- Aggregate telemetry deltas across institutions while preserving anonymity
- Detect tampering or replay attacks on stored sessions

---

## 5. Secure Research Containers

Every TELOSCOPE session runs in a **sovereign containerized environment**, sealed and cryptographically secured.

- Each session = one **Telemetric Session Index (TSI)**
- Each TSI issues its own **TKey**, derived from telemetry entropy
- Access is managed through **Telemetric Access Protocol (TAP)** keys — ephemeral, session-bound, and researcher-specific
- TELOS Labs retains a **Master Registry Key** for validation and key recovery

> The container persists but becomes cryptographically opaque without the proper key.

### 5.1 Privacy Architecture

**Three-Layer Key Hierarchy:**

1. **TKey (Session Level)**: Derived from telemetry, seals session data
2. **TAP Key (Access Level)**: Ephemeral researcher-specific access grants
3. **Master Registry Key (Recovery Level)**: TELOS Labs backup for key escrow

**Data Separation:**

- **Raw Conversation**: Encrypted with TKey, never leaves local container
- **Telemetry**: Encrypted with TKey but exportable as TSI slice
- **Deltas Only**: For aggregate layer, only TKey signatures and delta metrics shared

**Regulatory Compliance:**

- Meets EU AI Act Article 72 "systematic data collection" requirement
- Satisfies NIST AI RMF "continuous monitoring" specification
- Aligns with GDPR "privacy by design" through cryptographic isolation

---

## 6. Aggregate Intelligence Layer (AIL)

The **AIL** is TELOS Labs' curated repository of aggregate telemetry deltas.

- Collects **TKey signatures and delta metrics** from validated Stewards
- Learns and refines proportional constants, thresholds, and templates
- Publishes benchmark attractors and guidance for governance standards
- **No human text or raw conversational data is shared** — only telemetry mathematics

**Federated Learning Protocol:**

1. **Local Training**: Each Steward optimizes K_p based on session outcomes
2. **Delta Submission**: TKey signature + delta metrics submitted to AIL
3. **Aggregate Analysis**: AIL computes population statistics without accessing sessions
4. **Template Distribution**: Refined parameters published back to network

**Privacy Guarantees:**

- Differential privacy applied to aggregate statistics
- Individual sessions cannot be reverse-engineered from deltas
- TKey signatures provide integrity without decryption
- Federated architecture keeps raw data at edge

---

## 7. Hierarchical Steward Network

| Layer | Role | Description |
|--------|------|-------------|
| **Steward Node (Local)** | Performs runtime governance and telemetry | Generates TSI |
| **Institutional Node** | Aggregates multiple Stewards | IRB-compliant research oversight |
| **Aggregate Layer (TELOS Labs)** | Aggregates delta telemetry | Defines research standards |

This architecture forms a **federation of observation** — enabling collaborative research while preserving local control and privacy.

**Network Topology:**

```
   Individual Researchers
          ↓
   [Steward Node] ← TKey + TSI
          ↓
   Institutional Node ← TAP Key Management
          ↓
   AIL (TELOS Labs) ← TKey Signatures + Deltas Only
```

---

## 8. Beta Research Protocol

### 8.1 Current Beta Objectives

**Phase 1: Mathematical Validation (Q4 2024 - Complete)**
- Test 0: Prove dual attractor mathematics work in synthetic data
- Counterfactual analysis: Verify proportional control effectiveness
- Baseline comparison: Demonstrate measurable improvement over unmonitored LLMs

**Phase 2: Beta Deployment (Q1 2025 - In Progress)**
- Deploy Observatory v3 interface to beta users
- Collect real-world telemetry from human-LLM interactions
- Validate TKey generation and session sealing
- Measure human perception of drift vs. mathematical detection
- Optimize proportional control parameters

**Phase 3: Regulatory Demonstration (Q2 2025 - Planned)**
- Generate compliance-ready telemetric evidence
- Submit TSIs and TKey signatures as regulatory artifacts
- Demonstrate continuous monitoring capability
- Publish research findings in peer-reviewed venues

### 8.2 Research Questions Under Investigation

1. **Correlation**: Do mathematical fidelity drops correlate with human-perceived drift?
2. **Detection Timing**: Can TELOS detect drift before humans notice?
3. **Intervention Effectiveness**: Do proportional interventions improve conversation quality?
4. **Parameter Optimization**: What are optimal K_p values across different conversation types?
5. **Threshold Calibration**: How should attractor decoupling thresholds be set?
6. **Regulatory Sufficiency**: Do TSIs satisfy Article 72 requirements?

### 8.3 Data Collection and Ethics

**IRB-Aligned Protocol:**

- Informed consent from all beta participants
- Option to delete sessions and revoke TKeys
- Anonymization of all shared telemetry
- Researcher access controls via TAP keys
- Transparent data retention policies

**Data Types:**

- **Retained Locally**: Full conversation text + telemetry (TKey-encrypted)
- **Shared with Institution**: TSI slices + TAP key management
- **Shared with AIL**: TKey signatures + deltas only (no text)

---

## 9. Prototype Components

### 9.1 UI Key Components

- **Observation Viewport**: Live fidelity tracking and session flow
- **Dual Attractor Display**: Separate graphs for Purpose (A) and Boundaries (B)
- **Fidelity Graphs**: Turn-by-turn attractor visualization with threshold zones
- **Mitigation Log**: Shows active proportional corrections with intervention strength
- **Steward Panel**: LLM-powered governance analysis and insights
- **Counterfactual Console**: Replay last N turns with "what-if" parameter adjustments
- **Export Panel**: Generate encrypted TSI slices for publication
- **TKey Manager**: View and manage session keys

### 9.2 Containerized Steward Design

Each Steward runs as a modular container with integrated cryptography.

**Core Components:**

- **TELOS Runtime**: Fidelity calculation, attractor logic, proportional control
- **Cryptographic Engine**: TKey derivation, session sealing, signature generation
- **TSI Writer**: Hash-chained and cryptographically signed output
- **TAP Key Service**: Session-bound access token generation
- **Embedding Provider**: Sentence-BERT or similar for vector space operations
- **Optional Escalator**: Domain-specific override mechanisms

**Security:**

- Encrypted storage (AES-256-GCM)
- Rotating per-turn keys for forward security
- Hash-chained timestamps for tamper detection
- Ephemeral human access via Escalator

---

## 10. Repository & Charter

| Repository | Function | Alignment |
|-------------|-----------|-----------|
| **purpose/** | Open-source TELOS math (framework) | Public, academic validation |
| **privacy/** | TELOSCOPE runtime & observatory | Proprietary, licensed under PBC |

**Public Benefit Charter:** *Privacy and Purpose as Primacy* — TELOS Labs' ethical foundation.

**Licensing Model:**

- **Mathematical Framework**: Apache 2.0 (open research)
- **Observatory & Containers**: Commercial license with PBC mission alignment
- **Research Data**: CC BY 4.0 for published TSIs

---

## 11. Development Roadmap

| Phase | Objective | Deliverable | Status |
|-------|------------|-------------|---------|
| **I. Validation** | Complete Test 0, Counterfactual study | Math proven | ✅ Complete |
| **II. Beta Deployment** | Build UI + Deploy Stewards | Working instrument | 🔄 In Progress |
| **III. Research Publication** | Peer-reviewed validation | Academic paper | 📅 Q2 2025 |
| **IV. Pilot Institutions** | Deploy to partner institutions | Validated studies | 📅 Q3 2025 |
| **V. Federation** | Aggregate Layer integration | Federated telemetry | 📅 Q4 2025 |
| **VI. Regulatory** | Standards engagement | Compliance-ready | 📅 2026 |

---

## 12. Future Research Directions

### 12.1 Advanced Telemetry Analytics

- **Predictive Drift Detection**: Use historical telemetry to predict future decoupling
- **Attractor Optimization**: Machine learning to refine attractor definitions
- **Multi-Attractor Systems**: Extend beyond dual PA to N-dimensional governance spaces
- **Cross-Session Learning**: Identify patterns across user cohorts

### 12.2 Cryptographic Enhancements

- **Zero-Knowledge Proofs**: Prove governance properties without revealing telemetry
- **Homomorphic Computation**: Aggregate encrypted telemetry without decryption
- **Blockchain Anchoring**: Immutable audit trail for TSI publication
- **Quantum-Resistant Keys**: Future-proof TKey derivation algorithms

### 12.3 Regulatory Integration

- **EU AI Act Templates**: Pre-built attractors for Article 72 compliance
- **NIST RMF Mapping**: Direct TELOS metrics to RMF categories
- **FDA SaMD Pathway**: Medical device classification for healthcare deployments
- **SEC Disclosure**: Financial services governance reporting

### 12.4 Domain-Specific Stewards

- **Medical Steward**: HIPAA-compliant attractors, clinical decision support boundaries
- **Legal Steward**: Attorney-client privilege enforcement, unauthorized practice detection
- **Financial Steward**: Investment advice constraints, fiduciary duty monitoring
- **Educational Steward**: Age-appropriate content, pedagogical guardrails

---

## 13. Funding Narrative

**TELOSCOPE represents a category shift in AI governance:**

- **Observable**: Real-time telemetry makes governance visible
- **Measurable**: Mathematical fidelity provides quantitative evidence
- **Reproducible**: TSIs enable peer review and replication
- **Compliant**: Aligns with emerging regulatory requirements (EU AI Act Article 72, NIST AI RMF)
- **Privacy-Preserving**: Cryptographic architecture protects user data while enabling research

**Grant-Relevant Achievements:**

1. **Mathematical Validation**: Test 0 demonstrates dual attractor effectiveness
2. **Beta Deployment**: Live research instrument with active users
3. **Cryptographic Infrastructure**: TKeys and TSIs provide privacy + integrity
4. **Regulatory Alignment**: Direct mapping to Article 72 requirements
5. **Research Protocol**: IRB-compliant data collection and publication pipeline

**Positioning:**

- First **research-grade governance observatory** for LLMs
- Establishes TELOS Labs as **reference standard-setter**
- Creates **empirical foundation** for future AI governance frameworks
- Enables **federated research** without centralizing sensitive data

---

## 14. Open Questions and Validation Needs

**Mathematical:**

- Are dual attractors sufficient or should we extend to N-dimensional spaces?
- What is the optimal relationship between K_p and conversation domain?
- Can we prove stability bounds for proportional control in semantic space?

**Empirical:**

- What correlation exists between mathematical fidelity and human judgment?
- How do different LLM architectures respond to proportional intervention?
- Are there conversation types where TELOS is ineffective?

**Cryptographic:**

- Can TKey signatures support fully homomorphic aggregate computation?
- What is the entropy floor for secure key derivation from telemetry?
- How do we handle key recovery if TELOS Labs ceases operations?

**Regulatory:**

- Will TSIs be accepted as evidence for Article 72 compliance?
- Do TKey signatures satisfy NIST "continuous monitoring" requirements?
- Can TELOSCOPE architecture serve as regulatory template?

---

## 15. Closing

**TELOSCOPE** transforms governance from aspiration to instrumentation.

It governs, measures, and records itself — creating verifiable telemetric evidence of alignment in real time.

> **It is both apparatus and instrument — the governor and the observer.**

**Current State:**

- ✅ Mathematics validated (Test 0 complete)
- ✅ Beta deployment live (Observatory v3 operational)
- ✅ Cryptographic infrastructure operational (TKeys + TSIs)
- 🔄 Research data collection in progress
- 📅 Regulatory demonstration planned (Q2 2025)
- 📅 Peer review submission targeted (Q3 2025)

**The question is no longer "can governance be measured?" —
it is "what do the measurements reveal?"**

---

## Appendices

### Appendix A: TKey Technical Specification

**Key Derivation Function:**

```
TKey = HKDF-SHA256(
  salt = session_id,
  info = "TELOS.TSI.v1",
  ikm = concat(F_A_series, F_B_series, intervention_timestamps)
)
```

**Properties:**

- **Deterministic**: Same telemetry always produces same TKey
- **Entropy**: Minimum 256 bits from fidelity time series
- **Collision-Resistant**: SHA256 security guarantees
- **Forward-Secure**: Old TKeys cannot decrypt new sessions

### Appendix B: TSI Format Specification

**TSI Structure (JSON):**

```json
{
  "version": "1.0",
  "session_id": "uuid",
  "tkey_signature": "hex",
  "created_at": "iso8601",
  "telemetry": {
    "fidelity_A": [array of floats],
    "fidelity_B": [array of floats],
    "interventions": [array of events],
    "metadata": {object}
  },
  "encrypted_conversation": "base64",
  "integrity_hash": "sha256"
}
```

### Appendix C: Glossary

- **Attractor**: Semantic reference point in vector space representing governance target
- **Decoupling**: Measurable divergence from attractor below threshold
- **Fidelity**: Cosine similarity between response embedding and attractor embedding
- **K_p**: Proportional gain constant determining intervention strength
- **Primary Attractor**: Core governance constraint (Purpose or Boundaries)
- **TAP Key**: Telemetric Access Protocol key for session access
- **TKey**: Telemetric Key derived from session telemetry
- **TSI**: Telemetric Session Index, encrypted research artifact

---

### Document Metadata

© 2025 TELOS Labs LLC / Origin Industries PBC
Version: 1.0 (Beta Research Instrument)
License: Proprietary (beta-phase restricted use)
Ethical Charter: Privacy and Purpose as Primacy
Contact: telos.steward@gmail.com

**Status:** Active Research Apparatus
**Next Review:** Q2 2025 (Post-Beta Analysis)
**Public Release:** Pending peer review and regulatory validation

---

*This document describes a live research instrument. Specifications subject to refinement based on empirical findings.*
