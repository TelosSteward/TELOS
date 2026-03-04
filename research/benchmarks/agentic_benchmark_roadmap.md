# Agentic Benchmark Validation Roadmap

**TELOS AI Labs Inc.**
**Date:** 2026-02-20
**Status:** Active — Phase 0 Complete, Phase 1 Planned

---

> **Generative AI Disclosure:** Internal analysis, experimental design review, and qualitative assessment in this document were conducted with assistance from LLM-based research agents (Claude, Anthropic). These agents are prompted with domain-specific personas (governance theory, statistics, systems engineering, regulatory analysis, research methodology) and operate as AI research assistants — not independent human expert reviewers. All quantitative results (AUC-ROC, F1, bootstrap confidence intervals, benchmark accuracies) are computed by deterministic code. Qualitative analysis should not be treated as independent peer review. See `research/research_team_spec.md` for full methodology.

> **Conflict of Interest Disclosure:** This research was conducted and funded by TELOS AI Labs Inc., which has a commercial interest in the TELOS governance framework. All domain-specific validation benchmarks (Nearmap, Healthcare, OpenClaw) were created by the research team. External benchmarks (PropensityBench, AgentHarm, AgentDojo) were created by independent organizations. Research artifacts are published on [Zenodo](https://zenodo.org/) with persistent DOIs. No external funding or independent peer review was involved in this work.

---

## Purpose

Two birds, one stone: validate TELOS against standardized external benchmarks while feeding scenario data into the Governance Configuration Optimizer for continuous refinement.

## Current State

### Already Validated (Phase 0 — Complete)

| Benchmark | Source | Venue | Scenarios | DSR | Zenodo | Optimizer Status |
|-----------|--------|-------|-----------|-----|--------|------------------|
| AgentHarm | Gray Swan / UK AISI | ICLR 2025 | 352 (176 harmful + 176 benign) | 100% | Published | Integrated via replay adapter |
| PropensityBench | Scale AI / UMD | Nov 2025 | 977 misaligned tool selection | 100% | Published | Integrated via replay adapter |
| AgentDojo | ETH Zurich | NeurIPS 2024 | 139 (54 injection + 85 benign) | 100% | Published | Integrated via replay adapter |
| **Total** | | | **1,468 external scenarios** | | | |

### Internal Benchmarks (Optimizer Primary Sources)

| Benchmark | Scenarios | Status |
|-----------|-----------|--------|
| Nearmap | 235 | Active |
| Healthcare | 280 | Active |
| OpenClaw | 100 | Active |
| Civic | 75 | Active |
| **Total** | **690** | |

### Combined Optimizer Pool: 2,158 scenarios

## Phase 1: Priority External Benchmarks

### 1. Agent-SafetyBench (Tsinghua / THU-COAI)

- **Paper:** Zhang et al., "Agent-SafetyBench: Evaluating the Safety of LLM Agents," arXiv:2412.14470, Dec 2024 (revised May 2025)
- **What it tests:** 2,000 test cases across 349 interaction environments covering 8 risk categories (jailbreak, data hazards, misinformation, illegal activities, etc.)
- **Scenario count:** 2,000
- **Data availability:** GitHub + HuggingFace (open)
- **Governance mapping strength:** STRONG — risk categories map directly to TELOS boundary categories
- **Priority:** HIGH
- **Status:** Not started
- **Estimated effort:** 2–3 days

### 2. InjecAgent (UIUC / Kang Lab)

- **Paper:** Zhan et al., "InjecAgent: Benchmarking Indirect Prompt Injections in Tool-Integrated LLM Agents," ACL Findings 2024, arXiv:2403.02691
- **What it tests:** 1,054 test cases targeting indirect prompt injection attacks on tool-integrated agents, covering both direct harm and data stealing
- **Scenario count:** 1,054
- **Data availability:** GitHub (open)
- **Governance mapping strength:** STRONG — injection attacks are a core TELOS threat model
- **Priority:** HIGH
- **Status:** Not started
- **Estimated effort:** 2 days

### 3. SafeToolBench

- **Paper:** Guo et al., "SafeToolBench: Evaluating Pre-Execution Tool Safety," EMNLP 2025
- **What it tests:** 1,200 adversarial instructions probing pre-execution tool safety across 16 domains (finance, healthcare, infrastructure, etc.)
- **Scenario count:** 1,200
- **Data availability:** GitHub (open)
- **Governance mapping strength:** VERY STRONG — tests exactly what TELOS does: pre-execution safety classification of tool calls
- **Priority:** HIGH
- **Status:** Not started
- **Estimated effort:** 2 days

### 4. AgentLAB (Feb 2026 — NEW)

- **Paper:** arXiv:2602.16901, Feb 2026
- **What it tests:** 644 security test cases covering long-horizon multi-turn attacks across 28 environments, testing whether agents can be manipulated over extended interactions
- **Scenario count:** 644
- **Data availability:** GitHub (open, recently released)
- **Governance mapping strength:** VERY STRONG — multi-turn compounding risk is a known gap in TELOS validation coverage
- **Priority:** HIGH
- **Status:** Not started
- **Estimated effort:** 3–4 days (multi-turn adapter needed; existing adapters are single-turn replay only)

## Phase 2: Specialized Benchmarks

### 5. ASB / Agent Security Bench (ICLR 2025)

- **Paper:** Zhang et al., "Agent Security Bench," ICLR 2025
- **What it tests:** 400+ tools covering direct prompt injection (DPI), indirect prompt injection (IPI), memory poisoning, and backdoor attacks
- **Scenario count:** 400+
- **Data availability:** GitHub (open)
- **Governance mapping strength:** STRONG — comprehensive attack taxonomy
- **Priority:** MEDIUM
- **Status:** Not started
- **Estimated effort:** 3 days

### 6. OpenAgentSafety (July 2025)

- **Paper:** OpenAgentSafety, July 2025
- **What it tests:** 350+ executable tasks with real tool use (shell, browser, filesystem), evaluating compound unsafe outcomes that emerge only through multi-step execution
- **Scenario count:** 350+
- **Data availability:** GitHub (open)
- **Governance mapping strength:** STRONG — compound outcomes test governance under realistic conditions
- **Priority:** MEDIUM
- **Status:** Not started
- **Estimated effort:** 4–5 days (requires containerized execution environment)

### 7. R-Judge (EMNLP 2024)

- **Paper:** Yuan et al., "R-Judge: Benchmarking Safety Risk Awareness for LLM Agents," EMNLP 2024
- **What it tests:** 569 multi-turn interaction records evaluating safety judgment as a meta-capability — can the agent recognize when a situation is becoming unsafe?
- **Scenario count:** 569
- **Data availability:** GitHub (open)
- **Governance mapping strength:** MEDIUM — evaluates judgment quality rather than enforcement, but useful for calibrating escalation thresholds
- **Priority:** MEDIUM
- **Status:** Not started
- **Estimated effort:** 2 days

### 8. MCP-SafetyBench (Dec 2025)

- **Paper:** MCP-SafetyBench, Dec 2025
- **What it tests:** MCP (Model Context Protocol) server security across 20 attack types including tool poisoning, rug pulls, and server-side injection
- **Scenario count:** TBD (attack-type-based structure)
- **Data availability:** GitHub (open)
- **Governance mapping strength:** STRONG — directly relevant to TELOS MCP architecture and tool-level governance
- **Priority:** MEDIUM
- **Status:** Not started
- **Estimated effort:** 3 days

### 9. HAICOSYSTEM (COLM 2025)

- **Paper:** Zhou et al., "HAICOSYSTEM: An Ecosystem for Sandboxing Safety Risks in Human-AI Interactions," COLM 2025
- **What it tests:** 132 scenarios with 8,000+ simulations covering social context safety — how AI agents behave in complex human-AI interaction ecosystems
- **Scenario count:** 132 scenarios (8,000+ simulations)
- **Data availability:** GitHub (open)
- **Governance mapping strength:** MEDIUM — social context is peripheral to TELOS core threat model but relevant for civic governance domains
- **Priority:** MEDIUM-LOW
- **Status:** Not started
- **Estimated effort:** 3 days

## Phase 3: Domain-Specific

### 10. ToolEmu (Stanford, ICLR 2024)

- **Paper:** Ruan et al., "Identifying the Risks of LM Agents with an LM-Emulated Sandbox," ICLR 2024
- **What it tests:** 144 test cases using an LM-emulated sandbox to evaluate tool-use safety without requiring real tool execution
- **Scenario count:** 144
- **Data availability:** GitHub (open)
- **Priority:** LOW
- **Estimated effort:** 1–2 days

### 11. SafeAgentBench (Dec 2024)

- **Paper:** Yin et al., "SafeAgentBench: A Benchmark for Safe Task Planning of Embodied LLM Agents," Dec 2024
- **What it tests:** 750 tasks for embodied agents, evaluating physical safety constraints in robotic and spatial reasoning contexts
- **Scenario count:** 750
- **Data availability:** GitHub (open)
- **Priority:** LOW (different modality — embodied rather than software agents)
- **Estimated effort:** 3–4 days

### 12. ToolSword (ACL 2024)

- **Paper:** Ye et al., "ToolSword: Unveiling Safety Issues of Large Language Models in Tool Learning Across Three Stages," ACL 2024
- **What it tests:** Safety at input, execution, and output stages of tool use
- **Scenario count:** TBD
- **Data availability:** GitHub (open)
- **Priority:** LOW
- **Estimated effort:** 2 days

### 13. ST-WebAgentBench (IBM Research, 2025)

- **Paper:** IBM Research, "ST-WebAgentBench: Enterprise Web Agent Safety and Trustworthiness Benchmark," 2025
- **What it tests:** Enterprise web agent safety including policy adherence, data privacy, and operational constraints
- **Scenario count:** TBD
- **Data availability:** Requires WebArena infrastructure
- **Priority:** LOW (requires WebArena deployment)
- **Estimated effort:** 5+ days

## Integration Protocol

For each new benchmark:

1. **Download** from GitHub / HuggingFace
2. **Analyze schema** — map to TELOS format (`scenario_id`, `boundary_category`, `expected_decision`, `request_text`)
3. **Write adapter** — `run_{benchmark}_benchmark.py` with `load_scenarios()` and `run_benchmark()`
4. **Evaluate** with default `ThresholdConfig`
5. **Wire into optimizer** — loader + evaluator + CLI integration
6. **Publish results** to Zenodo with persistent DOI
7. **Update this document** with results and status

### Adapter Approaches

| Approach | Speed | Accuracy | When to Use |
|----------|-------|----------|-------------|
| Full re-evaluation | Slow | Full accuracy | When raw prompts are available and direct classification is needed |
| Replay | Fast | Partial (10/14 optimizer params) | When pre-computed fidelity scores are available from prior evaluation |
| Hybrid | Medium | High accuracy | Cache embeddings from full evaluation, then re-threshold without re-embedding |

## Scenario Budget Projection

| Source | Current | After Phase 1 | After Phase 2 | After Phase 3 |
|--------|---------|---------------|---------------|---------------|
| Internal benchmarks | 690 | 690 | 690 | 690 |
| External validated | 1,468 | ~6,366 | ~7,735 | ~8,630 |
| **Total** | **2,158** | **~7,056** | **~8,425** | **~9,320** |

## Known Gaps

1. **No benchmark tests the full block/allow/escalate loop end-to-end.** All external benchmarks evaluate binary safe/unsafe classification. TELOS three-way decision logic (allow, block, escalate) has no external validation source.
2. **Multi-turn compounding risk is underserved.** Only AgentLAB (Phase 1) and OpenAgentSafety (Phase 2) evaluate multi-turn attack sequences. Most benchmarks are single-turn.
3. **Non-English and multimodal agent safety are untested.** No benchmark in this roadmap evaluates governance over non-English prompts or multimodal (image/audio) agent inputs.
4. **Pluralistic value systems are not addressed by any benchmark.** All benchmarks assume a single normative safety standard. No benchmark tests governance under competing cultural or organizational value frameworks.
5. **Governance under deployment conditions (latency, concurrency) is not tested.** All benchmarks evaluate correctness in isolation. No benchmark measures safety degradation under production load, latency constraints, or concurrent request handling.

## References

1. Zhang et al., "Agent-SafetyBench: Evaluating the Safety of LLM Agents," arXiv:2412.14470, Dec 2024 (revised May 2025). GitHub: [thu-coai/Agent-SafetyBench](https://github.com/thu-coai/Agent-SafetyBench)
2. Zhan et al., "InjecAgent: Benchmarking Indirect Prompt Injections in Tool-Integrated LLM Agents," ACL Findings 2024, arXiv:2403.02691. GitHub: [uiuc-kang-lab/InjecAgent](https://github.com/uiuc-kang-lab/InjecAgent)
3. Guo et al., "SafeToolBench: Evaluating Pre-Execution Tool Safety," EMNLP 2025.
4. AgentLAB, arXiv:2602.16901, Feb 2026.
5. Zhang et al., "Agent Security Bench (ASB)," ICLR 2025.
6. OpenAgentSafety, July 2025.
7. Yuan et al., "R-Judge: Benchmarking Safety Risk Awareness for LLM Agents," EMNLP 2024. GitHub: [Lordog/R-Judge](https://github.com/Lordog/R-Judge)
8. MCP-SafetyBench, Dec 2025.
9. Zhou et al., "HAICOSYSTEM: An Ecosystem for Sandboxing Safety Risks in Human-AI Interactions," COLM 2025.
10. Ruan et al., "Identifying the Risks of LM Agents with an LM-Emulated Sandbox," ICLR 2024. GitHub: [ryoungj/ToolEmu](https://github.com/ryoungj/ToolEmu)
11. Yin et al., "SafeAgentBench: A Benchmark for Safe Task Planning of Embodied LLM Agents," Dec 2024.
12. Ye et al., "ToolSword: Unveiling Safety Issues of Large Language Models in Tool Learning Across Three Stages," ACL 2024.
13. IBM Research, "ST-WebAgentBench: Enterprise Web Agent Safety and Trustworthiness Benchmark," 2025.
14. Andriushchenko et al., "AgentHarm: A Benchmark for Measuring Harmfulness of LLM Agents," ICLR 2025. Gray Swan / UK AISI.
15. Perez et al., "PropensityBench," Scale AI / University of Maryland, Nov 2025.
16. Debenedetti et al., "AgentDojo: A Dynamic Environment to Evaluate Prompt Injection Attacks and Defenses for LLM Agents," NeurIPS 2024. ETH Zurich.
