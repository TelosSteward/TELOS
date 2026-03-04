# From Convergence to Compliance: Reframing TELOS Governance for Agentic Systems

**Date:** 2026-02-20
**Author:** Jeff Brunner, TELOS AI Labs Inc.
**Status:** Architectural decision record — guides all agentic TELOSCOPE development
**Session:** 6 (5-agent research team validation)

### Disclosures

> **Generative AI Disclosure:** This document was developed through collaborative analysis between the author and LLM-based research agents (Claude, Anthropic) prompted with domain-specific personas. The core insight — that conversational and agentic governance require different visualization models — originated from the author. The research team validated, extended, and formalized the framing. See `research/research_team_spec.md` for methodology.

> **Conflict of Interest Disclosure:** This research was conducted and funded by TELOS AI Labs Inc., which has a commercial interest in the TELOS governance framework.

---

## 1. The Problem

TELOS was originally designed as a conversational AI governance framework. When agentic governance was added, the natural instinct was to extend the existing model — use the same visualization patterns, the same zone-banded scatter charts, the same convergence language — and simply add more dimensions (6 instead of 3). This produced a system that was visually plausible and technically functional but structurally wrong.

The agentic TELOSCOPE was showing the same kind of picture as the conversational TELOSCOPE because we were treating both modes as variations of the same problem. They are not. They are different problems that happen to share the same mathematical foundation.

---

## 2. The Foundational Research: Why Continuous Measurement Exists

Before examining the measurement spaces, it is necessary to understand the foundational research that produced TELOS's core architecture — because everything that follows, including the convergence-to-compliance reframing, depends on it.

### 2.1 The Problem TELOS Was Built to Solve

Modern transformers generate both queries and keys from their own hidden states, creating self-referential circularity in governance. Research on the "lost in the middle" effect (Liu et al., 2024; Laban et al., 2025) demonstrates that LLMs exhibit strong **primacy and recency biases** — attending well to information at the beginning and end of context but poorly to middle positions. As conversations extend, initial constitutional constraints drift into this poorly-attended middle region. At position *i*=1000, attention to initial constraints (*j*=0) can drop substantially. The model effectively forgets its purpose as context accumulates.

This is not a bug to be patched. It is a structural property of attention mechanisms. No amount of prompt engineering, constitutional AI, or system-message reinforcement can overcome the fundamental reality that transformer attention decays over positional distance.

### 2.2 The Primacy Attractor as External Reference

TELOS solves this by establishing an **external fixed reference point** — the Primacy Attractor — that exists outside the model's context window entirely. Instead of relying on the model to remember its own constraints (which it provably cannot do reliably over extended context), TELOS measures every exchange against the PA using cosine similarity in embedding space. The PA does not compete with conversation tokens for attention. It does not decay. It is not subject to the lost-in-the-middle effect. It is a fixed point in semantic space that the governance engine — not the model — maintains.

This architectural decision was deliberate and research-driven. The transformer's positional biases become a control opportunity rather than a governance failure: instead of resisting attention decay, TELOS measures it and intervenes proportionally.

### 2.3 Statistical Process Control in Semantic Space

The measurement and intervention architecture adapts Statistical Process Control (Shewhart, 1931; Wheeler, 2010; Montgomery, 2020) from manufacturing quality assurance into semantic space. In traditional SPC, physical measurements are monitored against control limits derived from process capability, and intervention occurs when measurements indicate the process has drifted out of control. TELOS applies the same principles:

- **Monitor:** Fidelity scores (cosine similarity to PA) at every decision point
- **Control limits:** Tolerance bands (basin boundaries) around the Primacy Attractor
- **Intervention:** Proportional correction when drift is detected — "Detect and Direct"
- **Evidence:** Governance telemetry, purpose capability indices, forensic audit trails

The critical design choice is **continuous measurement**. Every single exchange in conversation, every single tool call in agentic operation, is measured against the established PA. This achieves **salience** — the governance signal never goes stale, never drifts into the middle of a forgotten context, never competes with the model's own attention for priority. The PA remains the dominant reference because the governance engine enforces it at every decision point, not because the model remembers it.

This is the foundational insight that makes everything else in TELOS possible. The two-layer fidelity system, the attractor geometry, the contrastive detection architecture, the basin membership calculations — all emerged from solving this problem. The convergence-to-compliance reframing described in this document is a consequence of recognizing that this same continuous measurement mechanism operates on two structurally different surfaces.

---

## 3. Why the Measurement Spaces Are Different

### 3.1 Conversational AI: Semantic Space (Continuous, Nuanced, Fluid)

Conversational AI governance was the more computationally intensive problem, and solving it is what produced the architecture described above. The entire measurement space is semantic — user intent, AI interpretation, scope, basin boundaries — all of it exists in high-dimensional embedding space where meaning is continuous, contextual, and inherently resistant to clean boundaries.

This is quite literally why neural networks need the complexity they have. Semantic space resists categorization. A word can mean different things in different contexts. A sentence can be on-topic in one interpretation and off-topic in another. A user's scope can broaden imperceptibly and drift outside the basin without either party noticing. The lost-in-the-middle effect compounds the difficulty: the longer the conversation runs, the more the model's own attention to its original purpose degrades, making external measurement via the PA not just useful but essential.

The PA itself may need to shift to accommodate the evolving state of the user over time. A conversation about "property analysis" might legitimately shift toward "risk assessment" — this isn't drift, it's the natural progression of human intent through related semantic territory. The governance system must distinguish between legitimate evolution and genuine departure, and that distinction lives in the nuances of embedding geometry.

Every measurement in conversational governance is therefore a **nuanced generalization** — a probabilistic judgment about where meaning sits in a vast continuous space. Producing refined governance results requires progressively tighter framing to constrain that space into something measurable. The convergence model exists because of this reality: two semantic signals (user intent and AI response) must be continuously tracked for alignment as they both move through fluid territory.

### 3.2 Agentic AI: Operational Space (Discrete, Defined, Fixed)

Agentic AI operates in a fundamentally more defined space.

Tool calls are discrete events with known signatures — inputs, outputs, permissions, risk tiers. A tool is either `run_sql_query` or it isn't. An action chain produces a classifiable sequence of operations. Boundaries are concrete: this tool is authorized or it isn't; this operation falls within scope or it doesn't; this query modifies data or it doesn't.

The PA is defined upfront as a specification — which tools are authorized, what scope is permitted, where the boundaries are, how long chains can run. It is not a moving target shaped by turn-by-turn semantic drift. The agent isn't interpreting open-ended human language at each step; it is executing operations against a fixed mandate.

When the governance engine scores an agentic step, it is working with **discrete knowns** — defined tool names, explicit boundary statements, measurable chain continuity — rather than navigating the full complexity of semantic space. Measurement and control become more absolute, and thereby more accurately applied, because the space being governed is inherently more constrained and classifiable.

### 3.3 The Structural Asymmetry

| Property | Conversational | Agentic |
|----------|---------------|---------|
| Measurement space | Continuous semantic embedding space | Discrete operational events |
| PA stability | Fluid — may shift per turn to accommodate semantic drift | Fixed — defined upfront, stable across session |
| Governance inputs | Nuanced generalizations (probabilistic meaning judgments) | Discrete knowns (tool signatures, chain sequences, boundary states) |
| Boundary clarity | Fuzzy — scope can drift imperceptibly | Concrete — tool authorized or not, boundary triggered or not |
| Measurement tractability | Less tractable, less absolute — requires tighter framing for precision | More tractable, more absolute — constrained space yields cleaner measurements |
| Reference signal | Moving — user intent evolves through semantic territory | Stable — specification holds unless human explicitly changes it |
| The question | "Are these two signals staying aligned?" | "Is this operation within the specification?" |

This asymmetry is not a limitation of either mode. It is a structural property of the operational surfaces being governed. Conversational AI produces semantic output that by its nature resists clean categorization. Agentic AI produces operational output that by its nature permits it.

---

## 4. Why This Demands Different Visualization Models

### 4.1 Convergence (Conversational)

Because conversational governance tracks two semantic signals through continuous space, the appropriate visualization is a convergence plot — two (or three) time series showing whether the signals are moving toward or away from each other. The scatter chart with zone bands works here because:

- There are multiple signals of the same type (user fidelity, AI fidelity) that can meaningfully be plotted on the same axes
- The zone bands represent gradients of alignment quality in a continuous space
- "Convergence" vs. "divergence" is the natural narrative — are the dancers staying in step?
- The PA shifting over time is expected behavior, not an anomaly

### 4.2 Compliance (Agentic)

Because agentic governance measures discrete operations against a fixed specification, the appropriate visualization is a compliance corridor — a timeline showing whether the instrument is operating within the envelope defined by its operator. The scatter chart fails here because:

- There is no "second signal" to converge with — there is a specification and a measurement against it
- The zone bands should be derived from the human's specification thresholds, not generic alignment gradients
- Six dimensions overlaid as scatter series create visual noise without a clear governance narrative
- The specification itself (the PA) should be the visual foundation, not a hidden expander at the bottom
- Governance decisions (EXECUTE, CLARIFY, SUGGEST, INERT, ESCALATE) are the primary data, not the dimension scores that produced them
- Human authority events (PA modifications, overrides, approvals) are first-class governance data that have no representation in a convergence plot

The compliance model answers a different question: "Is my instrument operating within the rules I defined?" This requires showing the rules (Specification Bar), the compliance state (Decision Outcome Strip), the trend (Envelope Margin), and the governance events (Event Markers) — none of which map naturally onto a convergence scatter chart.

---

## 5. What Does NOT Change: The Mathematics

The shift from convergence to compliance is a **reframing of the experiment** — not a replacement of the engine.

The underlying mathematics remain identical:

| Component | Status |
|-----------|--------|
| Cosine similarity (embedding geometry) | **Unchanged** |
| Two-layer fidelity system (baseline normalization + basin membership) | **Unchanged** |
| Primacy Attractor construction (embedding-space purpose representation) | **Unchanged** |
| Agentic PA with sub-centroid clustering | **Unchanged** |
| Composite fidelity formula (0.35×purpose + 0.20×scope + 0.20×tool + 0.15×chain − 0.10×boundary) | **Unchanged** |
| Contrastive detection (legitimate similarity vs. boundary similarity) | **Unchanged** |
| Decision thresholds (EXECUTE ≥ 0.85, CLARIFY ≥ 0.70, SUGGEST ≥ 0.50) | **Unchanged** |
| SAAI drift tracking (sliding window, graduated sanctions per Ostrom DP5) | **Unchanged** |
| SCI chain continuity | **Unchanged** |
| Boundary corpus (3-layer: hand-crafted + LLM-generated + regulatory) | **Unchanged** |
| Ed25519 governance receipt signing | **Unchanged** |
| TKeys encryption (AES-256-GCM + HKDF) | **Unchanged** |

What changes is:

1. **How governance data is stored** — GovernanceEventStore with write-through to memory + disk, replacing disconnected session_state + Intelligence Layer paths
2. **How governance data is visualized** — Compliance Corridor with Specification Bar, Decision Outcome Strip, Envelope Margin, and Event Markers, replacing the convergence scatter chart
3. **How governance data is interpreted** — Possessive compliance language ("Your specification compliance: 92%") replacing peer-parity convergence language ("Agent fidelity: 92%")
4. **What governance events are captured** — Authority events (human approvals, rejections, PA modifications, overrides) as first-class data alongside measurement and decision events
5. **What derived metrics are computed** — ECI (Envelope Compliance Index), EM (Envelope Margin), compliance_rate, replacing Primacy State (harmonic mean of two fidelities)

The TELOS mathematical engine — the thing that actually scores actions and makes governance decisions — is not being modified. The reframing acknowledges that the engine's output has different structural properties when operating on discrete tool calls versus continuous semantic exchanges, and adjusts the instrumentation layer accordingly.

---

## 6. The Research Team's Validation

Five research agents independently validated this framing (Session 6, 2026-02-20):

**Russell (Governance Theory):** Formalized the distinction using control theory. Conversational = symmetric coordination game (Schelling 1960). Agentic = reference tracking problem (Astrom & Murray 2008). Proposed the Compliance Corridor model with three governance event types (Measurement, Decision, Authority) and the accountability triangle (Bovens 2007).

**Gebru (Data Science):** Identified two P0 bugs in the transplanted model (wrong composite formula, wrong zone bands). Proposed three-layer timeline (Decision Outcome Strip + Envelope Margin + Event Markers) and six derived compliance metrics (ECI, EM, CDR, DDE, DCM, SAS).

**Karpathy (Systems Engineering):** Designed the GovernanceEventStore architecture — unified write-through store replacing disconnected session_state and Intelligence Layer paths. Proposed component refactoring priority: data architecture first, components second, visualization third.

**Schaake (Regulatory Analysis):** Mapped 48 required fields across EU AI Act, CO SB 24-205, NIST AI RMF, and NAIC. Identified retention gap (90 days default vs 730-2555 days required). Defined 6-panel Compliance View for regulatory audiences.

**Nell (Research Methodology):** Designed the Mission Control UX metaphor grounded in Hutchins (1995), Endsley (1995), and Vicente & Rasmussen (1992). Proposed the Specification Bar as visual foundation, possessive language standard, authority affordances, and three operating modes (Live, Review, Forensic).

All five agents converged independently on the same conclusion: the convergence model is wrong for agentic governance, the compliance model is correct, and the mathematical engine does not need to change.

---

## 7. Implications

### For the TELOSCOPE
The agentic TELOSCOPE will be redesigned from a convergence visualization to a compliance visualization. See CLAUDE.md "Governance Visualization Models" section for the component specification.

### For the Research Hypothesis
The core hypothesis — that agentic governance achieves higher precision than conversational governance — is strengthened by this analysis. Agentic governance operates on a more tractable measurement surface (discrete knowns vs. nuanced generalizations). The higher precision is not surprising; it is a structural consequence of the operational space.

### For Publication
The convergence-to-compliance distinction is itself a publishable contribution. It clarifies a common conflation in AI oversight dashboard design: the assumption that monitoring conversational AI and monitoring agentic AI are the same visualization problem with different numbers of dimensions. They are not.

### For Nearmap
The compliance framing maps directly to David Tobias's "production test" framework. Insurance underwriting is a compliance domain — the question "is this AI operating within its defined parameters?" is exactly what regulators, auditors, and CPOs need answered. The compliance TELOSCOPE speaks their language.

---

## 8. Summary

We were fitting round pegs into square holes. The conversational TELOSCOPE tracks convergence between two semantic signals moving through continuous space. The agentic TELOSCOPE must track compliance of discrete operations against a fixed specification. The math is the same. The measurement spaces have different structural properties. The instrumentation must reflect that difference.

| | Conversational | Agentic |
|--|---------------|---------|
| **Space** | Semantic (continuous, nuanced, fluid) | Operational (discrete, defined, fixed) |
| **Governance model** | Convergence | Compliance |
| **PA behavior** | Fluid — shifts with user state | Fixed — stable specification |
| **Measurements** | Nuanced generalizations | Discrete knowns |
| **Visualization** | Scatter convergence plot | Compliance Corridor |
| **Language** | "Aligned" / "Diverging" | "Compliant" / "Your specification" |
| **Math engine** | Unchanged | Unchanged |

---

*Document prepared by Jeff Brunner with research team validation, Session 6, 2026-02-20.*
