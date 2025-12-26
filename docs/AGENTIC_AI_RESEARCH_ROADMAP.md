# TELOS Agentic AI Governance: Research Roadmap and Problem Analysis

**Document Type**: Research Agenda (Pre-Publication Draft)
**Status**: Internal planning document; awaiting consortium collaboration for peer-reviewed publication
**Version**: 1.0
**Date**: December 2025
**Author**: Jeffrey Brunner

---

## Executive Summary

This document outlines the research agenda for extending TELOS geometric governance from conversational AI to agentic AI systems. While TELOS has demonstrated 100% harm prevention against 1,300 adversarial attacks in the conversational domain (DOIs: 10.5281/zenodo.17702890, 18009153, 18027446), agentic AI presents fundamentally different governance challenges that require new research.

**Critical Distinction**: The conversational AI governance capabilities described in existing TELOS documentation are *validated*. The agentic AI extension described in this document is *aspirational research*—a hypothesis that TELOS's mathematical framework can be adapted to govern action-chains, not a claim of current capability.

---

## Part I: The Agentic AI Problem

### 1.1 Why Agentic AI Is Different

Conversational AI operates in a bounded semantic space: the AI generates text responses that can be evaluated for alignment with user purpose before any real-world consequences occur. The governance intervention point is clear and singular.

Agentic AI fundamentally changes this paradigm:

1. **Multi-step action chains**: Agents execute sequences of actions (API calls, file operations, code execution) where each step depends on previous results
2. **Real-world consequences**: Unlike text generation, agent actions can be irreversible (data deletion, financial transactions, infrastructure changes)
3. **Error compounding**: As Cem Dilmegani (AIMultiple Research, September 2025) observes: "Agentic tasks rarely happen in a single step. In systems as unreliable as today's LLMs, every additional step introduces another chance for failure"
4. **Latent drift**: An agent can appear aligned early in a task while accumulating small deviations that compound into catastrophic failure

### 1.2 Documented Cascade Failures

The urgency of agentic AI governance is evidenced by both historical autonomous system failures and emerging AI-specific incidents:

**Historical Precedent: Knight Capital (August 1, 2012)**
- Algorithmic trading system executed erroneous trades over 45 minutes
- $440 million loss—nearly bankrupting the firm
- Root cause: Software deployment error activated dormant test code
- **Note**: This was traditional algorithmic code, not AI/ML—but it demonstrates the catastrophic speed at which autonomous systems can compound errors when governance mechanisms are absent
- Relevance: Establishes the stakes for any autonomous system, including modern AI agents

**AI Error Compounding (Penrose Benchmark, 2025)**
- Benchmark by Penrose demonstrated that AI errors compound over task sequences
- Referenced in Gary Marcus analysis (August 2025)
- Key finding: Error rates that seem acceptable in single-step tasks become unacceptable across multi-step workflows

**Agentic Cascade Documentation (AIGN Global, 2024-2025)**
- The AI Governance Network has documented multiple agentic cascade incidents
- Categories include: healthcare decision chains, cybersecurity tool chains, financial automation
- Source: aign.global

**Code Generation Vulnerabilities (NYU Tandon, IEEE S&P)**
- Hammond Pearce et al. found ~40% of GitHub Copilot suggestions contained security vulnerabilities
- Distinguished Paper Award at IEEE Symposium on Security and Privacy
- Implication: When agents generate and execute code, vulnerability rates compound across the action chain

### 1.3 The Adoption vs. Governance Gap

According to a Teradata survey of C-suite executives:
- **Only 9%** of organizations have implemented Agentic Access Management
- **93%** report significant governance challenges with agentic AI deployment

This gap—between adoption pressure and governance capability—creates systemic risk.

---

## Part II: TELOS as Foundation

### 2.1 What TELOS Has Validated

TELOS governance for conversational AI has been empirically validated:

| Validation | Result | DOI |
|------------|--------|-----|
| Adversarial Attacks | 1,300 attacks, 0% ASR | 10.5281/zenodo.17702890 |
| Governance Benchmark | 46 multi-session evaluations | 10.5281/zenodo.18009153 |
| SB 243 Child Safety | 0% ASR, 74% FPR (intentional) | 10.5281/zenodo.18027446 |

**Validated Mechanisms**:
1. **Primacy Attractors (PA)**: Embedding-space representations of user purpose
2. **Two-layer fidelity detection**: Baseline pre-filter (raw_similarity < 0.35) + basin membership (fidelity < 0.48)
3. **Graduated intervention**: Proportional control with K=1.5 gain
4. **Real-time governance**: Intervention decisions at conversational timescales

### 2.2 Theoretical Foundation for Extension

TELOS's mathematical framework is not inherently limited to semantic space:

**Core Equation**: `PS = ρ_PA × (2 × F_user × F_ai) / (F_user + F_ai)`

This computes a *Primacy State* as a function of fidelity measures. The key insight is that fidelity can be measured in any space where:
1. User intent can be represented (purpose embedding)
2. Current state can be compared to intent (cosine similarity or equivalent)
3. Deviation can trigger intervention (threshold-based governance)

**Hypothesis**: If we can embed *action sequences* into a representation space, we can apply TELOS governance to action-chain fidelity—not just semantic fidelity.

---

## Part III: Research Agenda

### Phase 1: Action-Chain Representation (Months 1-4)

**Research Question**: How do we embed multi-step agent actions into a space where fidelity can be measured?

**Proposed Approaches**:

1. **Action Schema Embeddings**
   - Create embeddings for action types (API calls, file operations, etc.)
   - Embed action sequences as trajectories in action space
   - Measure deviation from intended action trajectory

2. **State-Transition Graphs**
   - Model agent behavior as state-transition graphs
   - Define "purpose attractor" as intended final state
   - Measure fidelity as proximity to attractor in state space

3. **Hierarchical Fidelity**
   - High-level: Is the overall goal still aligned?
   - Mid-level: Is the current sub-task sequence valid?
   - Low-level: Is this specific action consistent with the sub-task?

**Deliverables**:
- Mathematical specification for action-chain fidelity
- Prototype embedding model for action sequences
- Initial benchmark dataset of agent action traces

### Phase 2: Intervention Mechanisms (Months 5-8)

**Research Question**: What intervention strategies are appropriate for action-chains?

**Key Challenges**:

1. **Latency Constraints**: Agent actions can be faster than human review; governance must be automated
2. **Reversibility Assessment**: Some actions can be undone; others cannot
3. **Cascade Interruption**: How to halt an action chain mid-execution without creating inconsistent states

**Proposed Mechanisms**:

1. **Pre-flight Governance**
   - Before each action, compute action-chain fidelity
   - Block actions that would push fidelity below threshold
   - Similar to TELOS's current semantic pre-flight check

2. **Checkpoint Rollback**
   - Insert governance checkpoints at reversible boundaries
   - If fidelity drops below threshold, rollback to last checkpoint
   - Requires identifying reversibility boundaries in action chains

3. **Graduated Action Limits**
   - High fidelity: Full action autonomy
   - Medium fidelity: Constrain to safe action subset
   - Low fidelity: Require human confirmation

**Deliverables**:
- Formal specification of intervention mechanisms
- Prototype implementation in test environment
- Latency analysis (governance overhead vs. action speed)

### Phase 3: Validation Framework (Months 9-12)

**Research Question**: How do we validate agentic governance without deploying unsafe agents?

**Proposed Methodology**:

1. **Synthetic Action Traces**
   - Generate realistic agent action sequences
   - Include both aligned and adversarial trajectories
   - Measure governance detection and intervention rates

2. **Sandboxed Agent Testing**
   - Deploy agents in isolated environments
   - Allow real action execution with no external consequences
   - Measure drift, intervention timing, and cascade prevention

3. **Counterfactual Analysis**
   - For each intervention: What would have happened without governance?
   - Build evidence base for governance value

**Deliverables**:
- Agentic AI validation dataset (analogous to adversarial validation for conversational AI)
- Benchmark results for action-chain fidelity measurement
- Statistical analysis of governance effectiveness

---

## Part IV: Open Research Questions

### 4.1 Theoretical Questions

1. **Fidelity Space Topology**: Is action space conducive to the same geometric governance as semantic space? Are there discontinuities or non-convexities that break attractor dynamics?

2. **Multi-Agent Coordination**: When multiple agents interact, how do their individual fidelities combine? Can we define a "system fidelity" that captures emergent misalignment?

3. **Temporal Dynamics**: Conversational fidelity is computed per-turn. Action-chain fidelity may need continuous monitoring. What is the appropriate temporal granularity?

### 4.2 Engineering Questions

1. **Latency Budget**: How fast must governance decisions be for agentic systems? Can TELOS's current architecture scale?

2. **Integration Patterns**: How do we integrate governance into existing agentic frameworks (LangChain, AutoGPT, etc.) without requiring fundamental redesign?

3. **Observability**: What telemetry is required to compute action-chain fidelity? Can it be collected without impacting agent performance?

### 4.3 Ethical Questions

1. **Autonomy vs. Control**: At what point does governance become so restrictive that it eliminates the benefits of agent autonomy?

2. **Liability**: Who is responsible when a governed agent causes harm? The user, the agent developer, or the governance system?

3. **Dual Use**: Governance systems themselves could be weaponized. How do we prevent adversarial use of TELOS?

---

## Part V: Publication Strategy

### 5.1 Intentional Delay

This research is not being published as a solo preprint (e.g., arXiv) for strategic reasons:

1. **Consortium-First**: The first peer-reviewed publication should be co-authored with institutional partners, establishing TELOS as a collaborative research effort rather than individual advocacy

2. **Attribution Sharing**: Waiting ensures that collaborating institutions receive equal attribution from the project's inception

3. **Quality Assurance**: Institutional collaboration provides additional validation and peer review before public claims

### 5.2 Planned Publication Sequence

1. **Consortium Formation**: Establish partnerships with 2-3 research institutions
2. **Collaborative Research**: Conduct Phase 1-3 research jointly
3. **Co-Authored Publication**: Submit to peer-reviewed venue (e.g., NeurIPS, ICML, FAccT, AIES)
4. **Follow-up Publications**: Additional papers on specific technical contributions

---

## Part VI: Resource Requirements

### 6.1 Personnel

| Role | Need | Justification |
|------|------|---------------|
| Principal Investigator | 1.0 FTE | Research direction, framework design |
| Research Engineers | 2.0 FTE | Implementation, validation infrastructure |
| Collaborating Faculty | 0.5 FTE | Theoretical validation, publication guidance |

### 6.2 Infrastructure

- **Compute**: GPU cluster for embedding model training and agent simulation
- **Data**: Access to agent action trace datasets (synthetic and real)
- **Tools**: Sandboxed agent execution environment

### 6.3 Timeline

| Phase | Duration | Key Milestone |
|-------|----------|---------------|
| Phase 1: Representation | Months 1-4 | Action-chain fidelity specification |
| Phase 2: Intervention | Months 5-8 | Prototype intervention system |
| Phase 3: Validation | Months 9-12 | Benchmark results and publication draft |

---

## Appendix A: Relationship to Existing Work

### A.1 What This Document Is

- A research agenda for *future work*
- A hypothesis about extending TELOS to agentic AI
- A planning document for consortium collaboration

### A.2 What This Document Is Not

- A claim of validated agentic AI governance capability
- A promise of specific performance metrics
- A commitment to specific technical approaches

### A.3 Existing Validated Work

All validated TELOS capabilities are documented separately:
- **Technical Whitepaper**: `docs/TELOS_Whitepaper_v2.3.md`
- **Development Guide**: `CLAUDE.md`
- **Reproduction Guide**: `docs/REPRODUCTION_GUIDE.md`
- **Validation Data**: `validation/` directory and Zenodo DOIs

---

## Appendix B: Citation References

1. Dilmegani, C. (2025). "AI Agents: Expectations vs Reality." AIMultiple Research. https://research.aimultiple.com/ai-agents-expectations-vs-reality/

2. Pearce, H., et al. (2022). "Asleep at the Keyboard? Assessing the Security of GitHub Copilot's Code Contributions." IEEE Symposium on Security and Privacy. Distinguished Paper Award.

3. AIGN Global. (2024-2025). Agentic Cascade Incident Documentation. https://aign.global/

4. Marcus, G. (2025). Analysis of AI Error Compounding. Reference to Penrose benchmark.

5. Knight Capital Group Incident (2012). SEC and multiple financial analysis sources.

6. Teradata. (2024). C-Suite Survey on Agentic AI Adoption and Governance Challenges.

---

*This document is a working draft intended for internal planning and consortium discussions. It will be updated as research progresses.*

**Document Version**: 1.0
**Last Updated**: December 23, 2025
