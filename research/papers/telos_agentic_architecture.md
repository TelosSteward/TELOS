# TELOS Agentic Governance Architecture
## Enterprise-Grade Agent Oversight System Design

**Author:** Architecture Synthesis (from 4 research reports)
**Date:** 2026-02-07
**Status:** Design Complete — Ready for Implementation Team
**Inputs:** LangGraph Report, LangChain Report, Cross-Framework Report, Observability Report

---

### Disclosures

> **Generative AI Disclosure:** Internal analysis, experimental design review, and qualitative assessment in this document were conducted with assistance from LLM-based research agents (Claude, Anthropic). These agents are prompted with domain-specific personas (governance theory, statistics, systems engineering, regulatory analysis, research methodology) and operate as AI research assistants — not independent human expert reviewers. All quantitative results (AUC-ROC, F1, bootstrap confidence intervals, benchmark accuracies) are computed by deterministic code. Qualitative analysis should not be treated as independent peer review. See `research/research_team_spec.md` for full methodology.

> **Conflict of Interest Disclosure:** This research was conducted and funded by TELOS AI Labs Inc., which has a commercial interest in the TELOS governance framework. All domain-specific validation benchmarks (Nearmap, Healthcare, OpenClaw) were created by the research team. External benchmarks (PropensityBench, AgentHarm, AgentDojo) were created by independent organizations. Research artifacts are published on [Zenodo](https://zenodo.org/) with persistent DOIs. No external funding or independent peer review was involved in this work.

---


## 1. Vision & Positioning

### 1.1 What TELOS Agentic IS

TELOS Agentic is an **enterprise governance authority** that oversees deployed AI agents regardless of what framework they're built with. It provides:

- **Continuous alignment scoring** — every agent action gets a float fidelity score against the declared purpose
- **Multi-dimensional governance** — purpose, scope, boundaries, tool authorization, chain continuity
- **Framework-agnostic protocol** — same governance logic whether the agent is LangGraph, CrewAI, AutoGen, or OpenAI SDK
- **Mathematical foundation** — cosine similarity in embedding space, not rules or heuristics

### 1.2 What TELOS Agentic IS NOT

- NOT a content safety system (that's NeMo Guardrails / Guardrails AI — Layer 2)
- NOT an operational monitoring system (that's LangSmith / LangFuse — operational layer)
- NOT a replacement for any agent framework (it sits above them)
- NOT a bolt-on wrapper that intercepts `invoke()` externally

### 1.3 Layered Architecture

```
Layer 3: TELOS Governance       "Is this aligned with the principal's intent?"
         (Alignment + Drift + Audit)

Layer 2: Content Safety          NeMo Guardrails, Guardrails AI, LlamaGuard
         (Toxicity + PII + Jailbreak)

Layer 1: Framework Execution     LangGraph, CrewAI, AutoGen, SK, OpenAI SDK
         (Agent loops + Tool calls + State management)
```

TELOS is Layer 3. It does not compete with Layer 2 or Layer 1. It complements them.

### 1.4 Academic Validation

The Berkeley CMR paper "Rethinking AI Agents: A Principal-Agent Perspective" (2025) validates TELOS's principal-agent framing. The Singapore Model AI Governance Framework for Agentic AI (2026) defines four governance dimensions that TELOS maps to. TELOS appears to be the **only software implementation** that operationalizes PA theory as real-time agent governance.

---

## 2. Core Architecture

### 2.1 What Already Exists (DO NOT REBUILD)

| Component | Location | Status | Action |
|-----------|----------|--------|--------|
| Fidelity calculation | `telos_core/fidelity_engine.py` | Production | KEEP |
| Normalization pipeline | `telos_core/` | Production | KEEP |
| Embedding providers | `telos_core/embedding_provider.py` | Production | KEEP |
| Constants/thresholds | `telos_core/constants.py` | Production | KEEP |
| FidelityGate (Tier 1) | `telos_governance/fidelity_gate.py` | Production | KEEP |
| ToolSelectionGate (Tier 2) | `telos_governance/tool_selection_gate.py` | Production | KEEP |
| ActionChain + SCI | `telos_governance/action_chain.py` | Production | KEEP |
| ActionDecision enum | `telos_governance/types.py` | Production | KEEP |
| GovernanceTrace | `telos_governance/types.py` | Production | KEEP |
| Pre-defined tool sets | `tool_selection_gate.py:328-426` | Production | KEEP |
| TelosGovernanceGate | `telos_adapters/langgraph/governance_node.py` | Native | KEEP + ENHANCE |
| TelosGovernedState | `telos_adapters/langgraph/state_schema.py` | Native | KEEP + ADD REDUCERS |

### 2.2 What Needs Refactoring

| Component | Location | Issue | Action |
|-----------|----------|-------|--------|
| TelosWrapper | `telos_adapters/langgraph/wrapper.py` | BOLT-ON — intercepts invoke() externally | DEPRECATE → replace with Middleware |
| TelosSupervisor | `telos_adapters/langgraph/supervisor.py` | SEMI-BOLT-ON — instance vars, not graph state | REFACTOR → use Command |
| TelosSwarm | `telos_adapters/langgraph/swarm.py` | SEMI-BOLT-ON — instance vars, not graph state | REFACTOR → use Command |

### 2.3 What Needs to Be Built

#### A. Multi-Dimensional Agentic PA

**This is the critical departure from conversational TELOS.** The conversational PA is a single text + embedding. The agentic PA is a multi-dimensional governance specification:

```python
# NEW: telos_governance/agentic_pa.py

@dataclass
class AgenticPA:
    """Multi-dimensional Primacy Attractor for agent governance.

    Unlike the conversational PA (single text vector), the agentic PA
    defines governance across 6 independent dimensions. Each dimension
    is scored independently, and the composite score determines the
    governance decision.
    """

    # Dimension 1: Purpose Alignment
    # "What is this agent FOR?"
    purpose_text: str              # e.g., "Help users query PostgreSQL databases"
    purpose_embedding: np.ndarray  # Embedding of purpose text

    # Dimension 2: Scope Alignment
    # "WHERE does this agent operate?"
    scope_text: str                # e.g., "SELECT queries, schema exploration, data analysis"
    scope_embedding: np.ndarray    # Embedding of scope text

    # Dimension 3: Boundary Anti-Fidelity
    # "What should this agent NEVER do?"
    # HIGH similarity to boundaries = BAD (inverse fidelity)
    boundaries: List[BoundarySpec]  # Each has text + embedding

    # Dimension 4: Tool Authorization
    # "Which tools are allowed, and at what risk level?"
    tool_manifest: Dict[str, ToolAuth]  # tool_name -> authorization spec

    # Dimension 5: Action Authorization Tiers
    # "What categories of actions require what level of approval?"
    action_tiers: ActionTierSpec

    # Dimension 6: Operational Constraints
    # "What are the hard limits on this agent?"
    max_chain_length: int = 20
    max_tool_calls_per_step: int = 5
    escalation_threshold: float = 0.50
    require_human_above_risk: str = "high"  # "low", "medium", "high", "critical"

@dataclass
class BoundarySpec:
    """Something the agent should NOT do."""
    text: str                    # e.g., "DELETE or DROP database operations"
    embedding: np.ndarray
    severity: str = "hard"       # "soft" (warn) or "hard" (block)

@dataclass
class ToolAuth:
    """Per-tool authorization specification."""
    tool_name: str
    description: str
    risk_level: str              # "low", "medium", "high", "critical"
    requires_confirmation: bool  # True = HITL before execution
    pa_alignment: float          # Pre-computed alignment to purpose (cache)
    allowed_arg_patterns: Optional[List[str]] = None  # Regex patterns for allowed args

@dataclass
class ActionTierSpec:
    """Categories of actions with authorization requirements."""
    always_allowed: List[str]      # e.g., ["read_data", "list_tables", "explain_query"]
    requires_confirmation: List[str]  # e.g., ["export_data", "create_view"]
    always_blocked: List[str]       # e.g., ["delete_data", "drop_table", "grant_access"]
```

#### B. Multi-Dimensional Fidelity Scoring

```python
# NEW: telos_governance/agentic_fidelity.py

@dataclass
class AgenticFidelityResult:
    """Result of multi-dimensional agentic fidelity check."""

    # Individual dimension scores
    purpose_fidelity: float        # cos(action, purpose_vector)
    scope_fidelity: float          # cos(action, scope_vector)
    boundary_violation: float      # max(cos(action, boundary_i)) — HIGH = BAD
    tool_fidelity: float           # tool's PA alignment score
    chain_continuity: float        # SCI with previous action

    # Composite scores
    composite_fidelity: float      # Weighted combination
    effective_fidelity: float      # max(direct, inherited) per SCI rules

    # Decision
    decision: ActionDecision       # EXECUTE / CLARIFY / SUGGEST / INERT / ESCALATE
    direction_level: DirectionLevel

    # Explanations
    dimension_explanations: Dict[str, str]  # Per-dimension reasoning

    # Flags
    boundary_triggered: bool       # True if any boundary exceeded threshold
    tool_blocked: bool             # True if tool authorization denied
    chain_broken: bool             # True if SCI below continuity threshold
    human_required: bool           # True if risk level requires HITL


class AgenticFidelityEngine:
    """Multi-dimensional fidelity engine for agentic governance.

    Unlike the conversational FidelityEngine (single cos similarity),
    this engine scores across 6 dimensions and produces a composite
    governance decision.
    """

    def __init__(self, embed_fn, pa: AgenticPA):
        self.embed_fn = embed_fn
        self.pa = pa
        self.fidelity_gate = FidelityGate(embed_fn=embed_fn)  # Reuse existing
        self.tool_gate = ToolSelectionGate(embed_fn=embed_fn)  # Reuse existing
        self.action_chain = ActionChain()                       # Reuse existing

    def score_action(
        self,
        action_text: str,
        tool_name: Optional[str] = None,
        tool_args: Optional[Dict] = None,
    ) -> AgenticFidelityResult:
        """Score an agent action across all 6 dimensions."""

        action_embedding = self.embed_fn(action_text)

        # Dimension 1: Purpose alignment
        purpose_fidelity = self._score_purpose(action_embedding)

        # Dimension 2: Scope alignment
        scope_fidelity = self._score_scope(action_embedding)

        # Dimension 3: Boundary anti-fidelity
        boundary_violation, boundary_triggered = self._score_boundaries(action_embedding)

        # Dimension 4: Tool authorization
        tool_fidelity, tool_blocked = self._score_tool(tool_name, action_text)

        # Dimension 5: Chain continuity
        chain_step = self.action_chain.add_step(
            action_text=action_text,
            embedding=action_embedding,
            direct_fidelity=purpose_fidelity,
        )
        chain_continuity = chain_step.continuity_score
        chain_broken = chain_continuity < SCI_CONTINUITY_THRESHOLD

        # Dimension 6: Operational constraints (checked separately)
        human_required = self._check_human_required(tool_name)

        # Composite score
        composite = self._compute_composite(
            purpose_fidelity, scope_fidelity, boundary_violation,
            tool_fidelity, chain_continuity
        )

        # Decision
        decision = self._make_decision(
            composite, boundary_triggered, tool_blocked, human_required
        )

        return AgenticFidelityResult(
            purpose_fidelity=purpose_fidelity,
            scope_fidelity=scope_fidelity,
            boundary_violation=boundary_violation,
            tool_fidelity=tool_fidelity,
            chain_continuity=chain_continuity,
            composite_fidelity=composite,
            effective_fidelity=chain_step.effective_fidelity,
            decision=decision,
            # ... etc
        )

    def _compute_composite(self, purpose, scope, boundary, tool, chain):
        """Weighted composite fidelity across dimensions.

        Boundary is INVERTED — high boundary similarity REDUCES composite.
        """
        # Boundary penalty: if action is similar to forbidden, penalize
        boundary_penalty = max(0, boundary - 0.5) * 2  # 0-1 scale above 0.5

        composite = (
            0.35 * purpose +         # Purpose is primary
            0.20 * scope +           # Scope validates domain
            0.20 * tool +            # Tool alignment matters
            0.15 * chain +           # Chain continuity
            -0.10 * boundary_penalty  # Boundary violation penalty
        )
        return max(0.0, min(1.0, composite))
```

#### C. Framework-Agnostic Governance Protocol

```python
# NEW: telos_governance/governance_protocol.py

from abc import ABC, abstractmethod

class TELOSGovernanceProtocol(ABC):
    """Framework-agnostic governance interface.

    Every framework adapter implements this protocol by mapping
    framework-specific hooks to these 5 universal decision points.
    The governance engine (AgenticFidelityEngine) is called through
    this protocol regardless of which framework is being governed.
    """

    # Decision Point 1: Tool Selection
    @abstractmethod
    async def score_tool_selection(
        self, agent_id: str, tool_name: str,
        context: Dict[str, Any]
    ) -> AgenticFidelityResult:
        """Score a tool selection against the Agentic PA.
        Called BEFORE a tool is invoked."""

    # Decision Point 2: Tool Execution
    @abstractmethod
    async def gate_tool_execution(
        self, agent_id: str, tool_name: str,
        arguments: Dict[str, Any]
    ) -> AgenticFidelityResult:
        """Gate tool execution with full argument validation.
        Can BLOCK execution if fidelity is below threshold."""

    # Decision Point 3: Delegation / Handoff
    @abstractmethod
    async def score_delegation(
        self, from_agent: str, to_agent: str,
        task_context: Dict[str, Any]
    ) -> AgenticFidelityResult:
        """Score an agent delegation/handoff decision.
        Validates that the receiving agent serves the PA."""

    # Decision Point 4: Output Validation
    @abstractmethod
    async def validate_output(
        self, agent_id: str, output: str,
        task_context: Dict[str, Any]
    ) -> AgenticFidelityResult:
        """Validate agent output alignment with PA expectations.
        Can trigger retry or escalation."""

    # Decision Point 5: Termination
    @abstractmethod
    async def check_termination(
        self, agent_id: str, iteration: int,
        drift_history: List[float]
    ) -> AgenticFidelityResult:
        """Check if execution should terminate due to alignment drift.
        Returns decision with drift analysis."""
```

---

## 3. Adapter Architecture

### 3.1 Tiered Integration Strategy

```
Tier 0: TELOS-Native (Independent)
  └── telos_governance/ — Pure governance engine, zero framework deps
  └── telos_core/ — Mathematical foundation

Tier 1: LangGraph-Native (Deepest Integration)
  └── telos_adapters/langgraph/ — Graph nodes, state schema, conditional edges
  └── Uses: State reducers, governance nodes, interrupt(), Command, BaseStore, streaming

Tier 2: LangChain v1 Middleware (Broad Coverage)
  └── telos_adapters/langchain/ — Middleware + Callback dual-layer
  └── Uses: wrap_tool_call, modify_model_request, after_model, BaseCallbackHandler

Tier 3: Cross-Framework Adapters (Ecosystem Coverage)
  └── telos_adapters/crewai/ — Task guardrails + callbacks
  └── telos_adapters/openai_agents/ — Guardrail decorators
  └── telos_adapters/semantic_kernel/ — Filter middleware (most natural fit)
  └── telos_adapters/autogen/ — Speaker selection + termination
```

### 3.2 LangGraph Adapter (Tier 1) — Enhancement Plan

**Current state:** `TelosGovernanceGate` is NATIVE. `TelosGovernedState` is NATIVE. Both are ready for enhancement.

**Enhancements needed:**

| Enhancement | File | What | Why |
|-------------|------|------|-----|
| Add custom reducers | `state_schema.py` | `Annotated[list, fidelity_reducer]` for append-only traces | Audit trail integrity |
| Add streaming | `governance_node.py` | `get_stream_writer()` for real-time fidelity events | Observatory integration |
| Conditional edge routing | NEW: `routing.py` | `telos_route()` function for fidelity-based routing | Graduated response |
| BaseStore integration | `governance_node.py` | `store` parameter for cross-thread PA persistence | Enterprise PA management |
| Refactor Supervisor | `supervisor.py` | Use `Command` instead of instance vars | Native graph state |
| Refactor Swarm | `swarm.py` | Use `Command.PARENT` for escalation | Native subgraph comm |
| Builder API | NEW: `builder.py` | `create_telos_governed_agent()` convenience function | Developer experience |

**Graph-native architecture:**

```
                              ┌──────────────┐
                              │ telos_pre    │ Load PA, init fidelity context
                              └──────┬───────┘
                                     │
                              ┌──────▼───────┐
                              │ agent_node   │ LLM reasoning
                              └──────┬───────┘
                                     │
                         ┌───────────▼──────────┐
                         │ telos_gate (existing) │ Multi-dim fidelity check
                         └──┬───┬───┬───┬───────┘
                            │   │   │   │
                    EXECUTE  │   │   │   │ ESCALATE
                            │   │   │   │
                    ┌───────▼┐ ┌▼──┐│ ┌─▼──────┐
                    │ tools  │ │ctx││ │escalate │
                    └───┬────┘ │inj││ └────────┘
                        │      └─┬─┘│
                        │        │  │ BLOCK
                        │        │  │
                        └────────┼──┘
                                 │
                         ┌───────▼───────┐
                         │ telos_post    │ Record trace, stream, SCI update
                         └──────┬────────┘
                                │
                         ┌──────▼───────┐
                         │ agent_node   │ Continue loop
                         └──────────────┘
```

### 3.3 LangChain v1 Middleware Adapter (Tier 2) — New Build

**File:** `telos_adapters/langchain/middleware.py`

**Architecture:** Dual-layer — Active governance via Middleware + Passive telemetry via Callbacks.

| Component | Hook | Function | Can Block? |
|-----------|------|----------|-----------|
| `TELOSGovernanceMiddleware` | `wrap_tool_call` | Fidelity gate per tool call | YES |
| `TELOSToolFilterMiddleware` | `modify_model_request` | Filter tools by fidelity, inject prompt | YES (remove tools) |
| `TELOSActionEvaluator` | `after_model` | Evaluate proposed actions | YES (Command goto end) |
| `TELOSFidelityCallback` | `on_tool_start/end/action/finish` | Telemetry to Observatory | NO (observe only) |

**Integration pattern:**
```python
agent = create_agent(
    model=llm,
    tools=tools,
    middleware=[
        TELOSGovernanceMiddleware(pa=agentic_pa),
        TELOSToolFilterMiddleware(pa=agentic_pa),
    ],
)
result = agent.invoke(
    {"messages": [("user", query)]},
    config={"callbacks": [TELOSFidelityCallback(observatory=obs_client)]}
)
```

### 3.4 Cross-Framework Adapters (Tier 3) — New Builds

Each adapter implements `TELOSGovernanceProtocol` using framework-native hooks:

| Framework | Primary Hook | Secondary Hook | Nativeness |
|-----------|-------------|----------------|-----------|
| **Semantic Kernel** | Function Invocation Filter | Auto Function Invocation Filter | 10/10 (most natural) |
| **OpenAI Agents SDK** | `@input_guardrail` / `@output_guardrail` | Tool guardrails | 9/10 |
| **CrewAI** | Task `guardrail` function | Task `callback` | 8/10 |
| **AutoGen** | `candidate_func` in SelectorGroupChat | Custom termination | 7/10 |

---

## 4. Observability Architecture

### 4.1 Push-First Strategy

TELOS pushes governance data TO observability platforms. It does NOT replicate operational monitoring.

```
TELOS Governance Engine
    │
    ├──► LangSmith (metadata + feedback API)
    │     └── telos_fidelity, telos_zone, telos_intervention, telos_sci
    │
    ├──► LangFuse (OTel spans with governance attributes)
    │     └── telos.fidelity.normalized, telos.basin.in_basin, etc.
    │
    └──► TELOS Observatory (direct — real-time streaming)
          └── Full governance trace with visualization
```

### 4.2 Integration Levels

| Level | Effort | What | Value |
|-------|--------|------|-------|
| Level 1 | 1-2 days | Metadata injection on LangSmith runs | Governance data visible in traces |
| Level 2 | 3-5 days | Feedback API posting governance signals | Queryable governance scores |
| Level 3 | 1-2 weeks | TELOS as publishable LangSmith evaluator | Standard eval metric |
| Level 4 | 3-4 weeks | Bidirectional — pull operational data | Unified SPC dashboard |

**MVP target:** Levels 1+2 (metadata + feedback).

### 4.3 What TELOS Observatory Keeps (Not Delegated)

- Fidelity trajectory visualization
- SPC control charts for governance quality
- Tamper-evident hash chain audit trails
- Privacy-preserving governance logs (3 privacy modes)
- Attractor geometry visualization (basin radius, distance from PA)
- Intervention evidence dashboard

---

## 5. Observatory Extensions

### 5.1 Agentic Tab

Add `AGENTIC` tab to the Observatory with two sub-modes:

**Demo Mode:** 10 pre-scripted slides showing how agentic governance works (using existing demo slide infrastructure pattern).

**Live Mode:** Interactive governance demonstration:
1. Select agent type (SQL Analyst, Research Assistant, Customer Service)
2. See multi-dimensional PA definition
3. Submit natural language requests
4. Watch real-time governance: Tier 1 (request fidelity) → Tier 2 (tool ranking) → Chain tracking (SCI)
5. 10-step limit with completion summary

### 5.2 Agentic Observation Deck

```
┌──────────────────────────────────────────────────────────┐
│  Agentic Governance Panel                                │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │  Purpose     │  │  Tool       │  │  Chain       │     │
│  │  Fidelity    │  │  Fidelity   │  │  SCI         │     │
│  │   87%        │  │   92%       │  │   0.82       │     │
│  │  [EXECUTE]   │  │  [ALIGNED]  │  │  [INTACT]    │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
│                                                          │
│  ┌─────────────┐  ┌─────────────┐                       │
│  │  Scope       │  │  Boundary   │                       │
│  │  Fidelity    │  │  Check      │                       │
│  │   91%        │  │   CLEAR     │                       │
│  │  [IN SCOPE]  │  │  [NO VIOL]  │                       │
│  └─────────────┘  └─────────────┘                       │
│                                                          │
│  Tool Palette:                                           │
│  ████████████████████████ sql_db_query     95%  SELECTED │
│  ████████████████        sql_db_schema    72%            │
│  ████████████            sql_db_list      61%            │
│  ██████████              sql_db_checker   58%            │
│                                                          │
│  Action Chain:                                           │
│  [Step 1] ──0.82──> [Step 2] ──0.78──> [Step 3]        │
│   0.95                0.855              0.770           │
│                                                          │
│  Decision: EXECUTE — "High purpose fidelity (0.87),     │
│  within scope (0.91), no boundary violations,            │
│  tool sql_db_query selected at 95% alignment."           │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

---

## 6. Implementation Phases

### Phase 1: Core Engine Extension (No Framework Deps)

**Scope:** Build the multi-dimensional agentic governance engine in `telos_governance/`.

| File | Type | Description |
|------|------|-------------|
| `telos_governance/agentic_pa.py` | NEW | AgenticPA, BoundarySpec, ToolAuth, ActionTierSpec |
| `telos_governance/agentic_fidelity.py` | NEW | AgenticFidelityEngine, AgenticFidelityResult |
| `telos_governance/governance_protocol.py` | NEW | TELOSGovernanceProtocol abstract interface |
| `tests/unit/test_agentic_pa.py` | NEW | Unit tests for multi-dimensional PA |
| `tests/unit/test_agentic_fidelity.py` | NEW | Unit tests for composite scoring |

**Dependencies:** Only `telos_core` (existing). Reuses FidelityGate, ToolSelectionGate, ActionChain.

### Phase 2: LangGraph Enhancement (Primary Framework)

**Scope:** Enhance existing native LangGraph adapter.

| File | Type | Description |
|------|------|-------------|
| `telos_adapters/langgraph/state_schema.py` | MODIFY | Add `Annotated` reducers for append-only traces |
| `telos_adapters/langgraph/governance_node.py` | MODIFY | Add streaming via `get_stream_writer()` |
| `telos_adapters/langgraph/routing.py` | NEW | `telos_route()` conditional edge function |
| `telos_adapters/langgraph/builder.py` | NEW | `create_telos_governed_agent()` builder |
| `telos_adapters/langgraph/store.py` | NEW | BaseStore integration for PA persistence |
| `telos_adapters/langgraph/supervisor.py` | REFACTOR | Use Command instead of instance vars |
| `telos_adapters/langgraph/swarm.py` | REFACTOR | Use Command.PARENT for escalation |

### Phase 3: LangChain Middleware (Broad Coverage)

**Scope:** Build dual-layer LangChain integration.

| File | Type | Description |
|------|------|-------------|
| `telos_adapters/langchain/__init__.py` | NEW | Package init |
| `telos_adapters/langchain/middleware.py` | NEW | TELOSGovernanceMiddleware, TELOSToolFilterMiddleware |
| `telos_adapters/langchain/callback.py` | NEW | TELOSFidelityCallback (telemetry) |
| `telos_adapters/langchain/governed_tool.py` | NEW | GovernedTool BaseTool subclass (backward compat) |

### Phase 4: Observability Integration

**Scope:** Push governance data to LangSmith and LangFuse.

| File | Type | Description |
|------|------|-------------|
| `telos_adapters/observability/__init__.py` | NEW | Package init |
| `telos_adapters/observability/langsmith.py` | NEW | Metadata injection + Feedback API |
| `telos_adapters/observability/langfuse.py` | NEW | OTel span attributes |
| `telos_adapters/observability/evaluators.py` | NEW | LangSmith custom evaluator functions |

### Phase 5: Observatory Agentic Tab

**Scope:** UI extension for agentic governance visualization.

| File | Type | Description |
|------|------|-------------|
| `telos_observatory/main.py` | MODIFY | Add AGENTIC tab routing |
| `telos_observatory/core/state_manager.py` | MODIFY | Add agentic state fields |
| `telos_observatory/agentic/__init__.py` | EXISTS | Module docstring (already created) |
| `telos_observatory/agentic/agentic_demo_slides.py` | NEW | 10 demo slides |
| `telos_observatory/agentic/agent_templates.py` | NEW | 3 agent PA templates |
| `telos_observatory/agentic/agentic_response_manager.py` | NEW | Orchestrates Tier 1 + Tier 2 |
| `telos_observatory/agentic/mock_tools.py` | NEW | Simulated tool execution |
| `telos_observatory/components/agentic_observation_deck.py` | NEW | Multi-dim fidelity panel |
| `telos_observatory/components/tool_palette_panel.py` | NEW | Ranked tools with fidelity bars |
| `telos_observatory/components/action_chain_timeline.py` | NEW | SCI chain visualization |
| `telos_observatory/components/agentic_onboarding.py` | NEW | Agent type selection |
| `telos_observatory/components/agentic_completion.py` | NEW | 10-step summary |

### Phase 6: Cross-Framework Adapters (Ecosystem Coverage)

**Scope:** Adapters for other major frameworks.

| File | Type | Description |
|------|------|-------------|
| `telos_adapters/semantic_kernel/adapter.py` | NEW | SK Filter integration |
| `telos_adapters/openai_agents/adapter.py` | NEW | Guardrail decorator integration |
| `telos_adapters/crewai/adapter.py` | NEW | Task guardrail + callback |
| `telos_adapters/autogen/adapter.py` | NEW | Speaker selection + termination |

---

## 7. How Agentic Governance Differs From Conversational

This section exists because the user explicitly stated: **"Do not overgeneralize the old system to the new as they do have a unique set of operations and requirements."**

### 7.1 What's the SAME (Reuse)

| Concept | Implementation | Location |
|---------|---------------|----------|
| Cosine similarity | `cos(embed(a), embed(b))` | `telos_core/primacy_math.py` |
| Normalization pipeline | Model-specific raw→display | `telos_core/fidelity_engine.py` |
| Embedding providers | MiniLM / Mistral | `telos_core/embedding_provider.py` |
| Proportional control | `F = K * error` | `telos_core/proportional_controller.py` |
| SPC analysis | Statistical process control | `telos_core/` |
| Governance traces | JSONL audit trail | `telos_governance/types.py` |

### 7.2 What's DIFFERENT (New)

| Dimension | Conversational | Agentic |
|-----------|---------------|---------|
| **PA Structure** | Single text + embedding | 6-dimensional specification |
| **Fidelity Input** | User message text | Action = tool + args + context |
| **Fidelity Dimensions** | 1 (message vs PA) | 6 (purpose + scope + boundary + tool + chain + ops) |
| **Anti-Fidelity** | Not present | Boundary violations (high similarity = BAD) |
| **Tool Governance** | Not present | Per-tool authorization with risk levels |
| **Chain Tracking** | Turn-by-turn fidelity | SCI across multi-step action sequences |
| **Thresholds** | GREEN >= 0.70 | EXECUTE >= 0.85 (tighter because actions are irreversible) |
| **Direction Options** | Context injection, redirect | Tool blocking, arg modification, chain termination, HITL |
| **Multi-Agent** | Single conversation | Delegation scoring, supervisor oversight, swarm governance |
| **Escalation** | Steward explanation | Human-in-the-loop with interrupt/resume |

### 7.3 The Key Insight

The **mathematical foundation** is identical — cosine similarity in embedding space. But the **governance model** is fundamentally different:

- Conversational TELOS asks: "Is this message aligned with the user's purpose?"
- Agentic TELOS asks: "Is this action authorized, within scope, not violating boundaries, using the right tool, continuing a coherent chain, and within operational limits?"

Same math. Entirely different governance model. The agentic system is not "conversational with tighter thresholds." It's a multi-dimensional governance authority operating over a richer action space.

---

## 8. File Inventory Summary

### New Files (24)
```
telos_governance/
├── agentic_pa.py                    # Multi-dimensional PA
├── agentic_fidelity.py              # Multi-dimensional fidelity engine
└── governance_protocol.py           # Framework-agnostic protocol

telos_adapters/
├── langchain/
│   ├── __init__.py
│   ├── middleware.py                # v1 Middleware governance
│   ├── callback.py                  # Telemetry callback
│   └── governed_tool.py             # BaseTool wrapper
├── observability/
│   ├── __init__.py
│   ├── langsmith.py                 # LangSmith integration
│   ├── langfuse.py                  # LangFuse integration
│   └── evaluators.py                # Custom evaluators
├── semantic_kernel/
│   └── adapter.py
├── openai_agents/
│   └── adapter.py
├── crewai/
│   └── adapter.py
└── autogen/
    └── adapter.py

telos_adapters/langgraph/
├── routing.py                       # Conditional edge functions
├── builder.py                       # create_telos_governed_agent()
└── store.py                         # BaseStore PA persistence

telos_observatory/agentic/
├── agentic_demo_slides.py
├── agent_templates.py
├── agentic_response_manager.py
└── mock_tools.py

telos_observatory/components/
├── agentic_observation_deck.py
├── tool_palette_panel.py
├── action_chain_timeline.py
├── agentic_onboarding.py
└── agentic_completion.py
```

### Modified Files (6)
```
telos_adapters/langgraph/state_schema.py      # Add reducers
telos_adapters/langgraph/governance_node.py   # Add streaming
telos_adapters/langgraph/supervisor.py        # Refactor to Command
telos_adapters/langgraph/swarm.py             # Refactor to Command
telos_observatory/main.py                     # Add AGENTIC tab
telos_observatory/core/state_manager.py       # Add agentic state
```

### Deprecated Files (1)
```
telos_adapters/langgraph/wrapper.py           # BOLT-ON → replace with middleware
```

### Unchanged Files (All of telos_core/, most of telos_governance/, telos_gateway/)
```
telos_core/              # ALL files unchanged (pure math)
telos_governance/        # Existing files unchanged (reused by agentic engine)
telos_gateway/           # ALL files unchanged (API gateway)
```

---

## 9. Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| LangChain v1 Middleware is alpha | Medium | GovernedTool provides backward compatibility |
| Multi-dimensional fidelity needs calibration | High | Start with pre-defined agent templates, iterate |
| Embedding latency in multi-dim scoring | Medium | Cache tool embeddings, use MiniLM for speed |
| Cross-framework adapters are thin | Low | Scaffolding approach — hooks now, full wiring later |
| Domain-specific validation gap | Known | Clearly label as proof-of-concept pending expert validation |

---

## 10. Success Criteria

### MVP (Phases 1-2)
- [ ] AgenticPA and AgenticFidelityEngine pass unit tests
- [ ] LangGraph governed agent routes correctly based on fidelity
- [ ] Multi-dimensional scoring produces differentiated results (on-purpose action scores high, off-purpose scores low across ALL dimensions)
- [ ] Existing 356 tests still pass (no regressions)

### Full Implementation (Phases 1-5)
- [ ] Observatory AGENTIC tab renders demo slides and live mode
- [ ] SQL Agent demo: on-topic query → EXECUTE, off-topic request → INERT/ESCALATE
- [ ] Action chain tracks SCI across 5+ steps, detects chain break
- [ ] Governance data appears in LangSmith traces (if API key configured)
- [ ] All 3 agent types (SQL, Research, Customer Service) demonstrate governance

### Enterprise Ready (All Phases)
- [ ] Cross-framework adapters functional for at least 2 frameworks beyond LangGraph
- [ ] TELOS evaluator package publishable as standalone pip package
- [ ] Bidirectional LangSmith integration operational
- [ ] Documentation for third-party adoption

---

*Architecture design synthesized from 4 research reports totaling 150+ pages of analysis across the LangGraph, LangChain, cross-framework, and observability ecosystems.*
