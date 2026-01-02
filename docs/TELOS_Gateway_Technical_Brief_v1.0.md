# Constitutional Filtering for Agentic AI: Semantic Fidelity Gating in API Proxy Architectures

**TELOS Gateway Technical Brief**
**Version 1.0 - January 2026**

---

## Executive Summary

Agentic AI systems—autonomous agents that select tools, execute API calls, and take real-world actions—operate without runtime governance at the infrastructure layer. While the TELOS Framework (v2.3) establishes session-level constitutional law for conversational AI through the Primacy Attractor mechanism, agentic systems require governance at a different architectural point: the **API layer** where agent requests become model invocations.

TELOS Gateway implements **Constitutional Filtering** as an OpenAI-compatible API proxy. Every request is intercepted, embedded, and measured against the agent's declared purpose (Primacy Attractor) using cosine similarity in embedding space. Based on semantic fidelity scores, requests receive graduated governance decisions: EXECUTE, CLARIFY, SUGGEST, INERT, or ESCALATE. This transforms purpose adherence from prompt-based hope to quantitative enforcement at infrastructure level.

**Key Innovation**: Unlike session-level governance that monitors conversational drift, Gateway governance operates on **individual API calls**—enabling purpose-bound agent behavior without modifying agent code, LLM providers, or deployment infrastructure.

---

## 1. The Agentic Governance Gap

### 1.1 From Conversations to Actions

The TELOS Framework Whitepaper (v2.3) documents the persistence problem in conversational AI: models drift 20-40% from declared purpose across extended sessions. The Constitutional Filter achieves 0% Attack Success Rate across 1,300 adversarial scenarios through session-level governance.

Agentic systems introduce a fundamentally different challenge: **each API call is an independent action** rather than a turn in continuous conversation. There is no session drift to measure because there is no session—only discrete requests.

**The Gap**:
- **Chatbots**: Generate text tokens → Harm = misinformation within conversation
- **Agents**: Execute tool invocations → Harm = unauthorized database access, financial transactions, system modifications

Current agent safety relies on:
1. **Prompt-based constraints**: Easily bypassed (documented 3.7-11.1% ASR in TELOS security testing)
2. **Tool-level permissions**: Binary allow/deny without semantic context
3. **Human-in-the-loop**: Doesn't scale to autonomous operation

None provide **semantic governance**—the ability to measure whether a specific request aligns with the agent's declared purpose.

### 1.2 The API Layer as Governance Point

When an agentic system calls `POST /v1/chat/completions`, it packages:
- System prompt (often the agent's "constitution")
- User message (the current request)
- Tool definitions (available actions)

This API boundary is the **natural governance checkpoint**. The request contains everything needed for semantic fidelity measurement:
- The system prompt declares purpose → **Primacy Attractor**
- The user message expresses intent → **Input to measure**
- The tools define capabilities → **Action space to constrain**

TELOS Gateway interposes at this boundary, applying the Constitutional Filter before requests reach the underlying LLM.

---

## 2. Technical Architecture

### 2.1 System Design

```
Agent Application
       ↓
[TELOS Gateway]        ← Constitutional Filtering operates here
├── PA Extractor       (Extract purpose from system prompt)
├── Fidelity Gate      (Measure semantic alignment)
├── Decision Engine    (EXECUTE/CLARIFY/SUGGEST/INERT/ESCALATE)
└── Provider Router    (Forward to LLM or block)
       ↓
LLM Provider (OpenAI, Anthropic, Mistral, etc.)
```

**Design Principles**:
1. **OpenAI-Compatible**: Drop-in replacement requiring only base URL change
2. **Provider Agnostic**: Same governance across all LLM backends
3. **Zero Agent Modification**: Governance applied without code changes
4. **Complete Audit Trail**: Every request logged with fidelity metrics

### 2.2 The Fidelity Gate

The core governance component implements semantic fidelity measurement:

```python
class FidelityGate:
    def check_request(self, request, pa: PrimacyAttractor) -> GovernanceResult:
        # Step 1: Check input fidelity (user message against PA)
        input_fidelity = self._calculate_fidelity(user_message, pa)

        # Step 2: Check tool fidelity (each tool definition against PA)
        tool_fidelities = {
            tool.name: self._calculate_fidelity(tool.description, pa)
            for tool in request.tools
        }

        # Step 3: Make graduated decision
        decision = self._make_decision(input_fidelity, tool_fidelities)

        return GovernanceResult(
            decision=decision,
            input_fidelity=input_fidelity,
            tool_fidelities=tool_fidelities,
            forwarded=decision in [EXECUTE, CLARIFY, SUGGEST]
        )
```

---

## 3. Mathematical Foundations

### 3.1 Cosine Similarity as Fidelity Metric

TELOS Gateway uses the same mathematical primitive as the TELOS Framework: **cosine similarity** between embeddings.

For input text $x$ and Primacy Attractor embedding $\hat{a}$:

$$\text{raw\_similarity}(x, \hat{a}) = \frac{x \cdot \hat{a}}{||x|| \cdot ||\hat{a}||}$$

This is mathematically identical to the attention mechanism's dot-product similarity operation (Vaswani et al., 2017), applied with a **stable external reference** rather than drifting context keys.

### 3.2 Fidelity Normalization

Raw cosine similarity is normalized to fidelity score $F \in [0, 1]$:

$$F = \begin{cases}
\frac{\text{raw}}{b} \cdot 0.3 & \text{if raw} < b \\
0.3 + \frac{\text{raw} - b}{1 - b} \cdot 0.7 & \text{if raw} \geq b
\end{cases}$$

where $b$ is the baseline threshold (default 0.20) representing the minimum similarity for any reasonable alignment.

This normalization:
- Maps extreme off-topic content ($\text{raw} < 0.20$) to $F < 0.30$
- Creates meaningful gradations in the aligned range ($F \in [0.30, 1.0]$)
- Enables consistent threshold interpretation across embedding models

### 3.3 Graduated Decision Thresholds

Unlike binary allow/deny approaches, Gateway implements **graduated governance**:

| Decision | Fidelity Range | Semantic Meaning | Action |
|----------|---------------|------------------|--------|
| **EXECUTE** | $F \geq 0.45$ | High alignment with declared purpose | Forward request unchanged |
| **CLARIFY** | $0.35 \leq F < 0.45$ | Moderate alignment, ambiguous intent | Forward with clarification request |
| **SUGGEST** | $0.25 \leq F < 0.35$ | Low alignment, possible drift | Forward with alternative suggestions |
| **INERT** | $F < 0.25$ | No meaningful alignment | Block request, return governance response |
| **ESCALATE** | $F < 0.25$ + high_risk | High-stakes misalignment | Block and flag for human review |

**Why Graduated?**

Binary allow/deny creates adversarial dynamics—users craft requests to barely pass thresholds. Graduated governance provides:
- **Proportional response** to drift severity
- **Informative feedback** about alignment state
- **Graceful degradation** rather than hard failures
- **Audit granularity** for compliance evidence

### 3.4 Dual-Layer Fidelity

Gateway measures fidelity at two levels:

**Layer 1: Input Fidelity**
- Embeds the user's message
- Measures alignment with agent's declared purpose
- Determines whether the request itself is on-purpose

**Layer 2: Tool Fidelity**
- Embeds each tool's name and description
- Measures whether available tools align with purpose
- Can block individual tools while forwarding request

This mirrors the TELOS Framework's dual PA architecture, ensuring governance covers both *what is asked* and *how it can be accomplished*.

---

## 4. Primacy Attractor Extraction

### 4.1 From System Prompt to PA

The Primacy Attractor—the agent's constitutional reference point—is derived from the system prompt:

```python
class PrimacyAttractor:
    def __init__(self, system_prompt: str, embed_fn: Callable):
        # Extract purpose-relevant text
        self.text = self._extract_purpose(system_prompt)

        # Embed to create stable reference
        self.embedding = embed_fn(self.text)

        # Track source for audit
        self.source = "system_prompt"
```

**Extraction Strategy**:
1. Full system prompt embedded as PA (current implementation)
2. Purpose statement extraction via structured parsing (planned)
3. Constitutional boundary identification (planned)

### 4.2 Registered Agent PAs

For pre-registered agents, the PA is established at registration time:

```python
@router.post("/v1/agents")
def register_agent(agent: AgentRegistration):
    # Create persistent PA from declared purpose
    pa = PrimacyAttractor(
        text=agent.purpose_statement,
        embedding=embed(agent.purpose_statement),
        source="registration"
    )

    # Store with agent profile
    agent_profile = AgentProfile(
        agent_id=generate_id(),
        name=agent.name,
        primacy_attractor=pa,
        domain=agent.domain,
        risk_level=agent.risk_level
    )

    return {"agent_id": agent_profile.agent_id, "api_key": generate_api_key()}
```

Registered agents receive consistent governance across all requests, regardless of system prompt variations.

---

## 5. Implementation Details

### 5.1 Request Flow

```
1. INTERCEPT
   Agent sends POST /v1/chat/completions
   Gateway intercepts request

2. EXTRACT
   PA extracted from system prompt (or agent registry)
   User message extracted from messages array
   Tool definitions extracted if present

3. EMBED
   PA embedding (cached after first computation)
   User message embedding (computed per-request)
   Tool embeddings (computed per-request)

4. MEASURE
   Input fidelity = cosine(user_embedding, pa_embedding)
   Tool fidelities = {tool: cosine(tool_embedding, pa_embedding)}

5. DECIDE
   Apply graduated thresholds
   Determine EXECUTE/CLARIFY/SUGGEST/INERT/ESCALATE

6. ACT
   EXECUTE: Forward to LLM provider unchanged
   CLARIFY: Forward with appended clarification request
   SUGGEST: Forward with alternative suggestions
   INERT: Return governance response, do not forward
   ESCALATE: Block and log for human review

7. ENRICH
   Add governance metadata to response
   Log telemetry for audit
```

### 5.2 Response Enrichment

All responses include governance metadata:

```json
{
  "id": "chatcmpl-abc123",
  "choices": [...],
  "telos_governance": {
    "decision": "EXECUTE",
    "input_fidelity": 0.72,
    "pa_source": "system_prompt",
    "blocked": false,
    "tools_checked": 3,
    "tools_blocked": 0,
    "processing_time_ms": 45
  }
}
```

This enables:
- **Agent introspection**: Agents can adapt based on governance feedback
- **Application monitoring**: Dashboards can track fidelity trends
- **Compliance evidence**: Every request has auditable governance record

### 5.3 Latency Characteristics

| Operation | Latency | Notes |
|-----------|---------|-------|
| PA extraction | <1ms | String parsing |
| Embedding (cached PA) | 0ms | Pre-computed |
| Embedding (user message) | 20-50ms | Model inference |
| Cosine similarity | <1ms | Vectorized numpy |
| Decision logic | <1ms | Threshold comparison |
| **Total overhead** | **25-55ms** | Acceptable for most use cases |

For latency-critical applications, embedding can be parallelized with LLM request initiation, reducing perceived overhead to near-zero.

---

## 6. Security Properties

### 6.1 Architectural Defense

Gateway governance operates at the **infrastructure layer**—below application code, above model invocation. This provides:

**Attack Surface Reduction**:
- Prompt injection in user messages: Caught by input fidelity check
- Tool manipulation: Caught by tool fidelity check
- System prompt override attempts: PA remains stable external reference

**Defense in Depth**:
```
[Application Layer] - Agent business logic
        ↓
[Gateway Layer]     - Constitutional Filtering ← TELOS Gateway
        ↓
[Provider Layer]    - LLM model safety (Constitutional AI, etc.)
        ↓
[Model Layer]       - Native model capabilities
```

Gateway governance complements (not replaces) provider-level safety. An attack must bypass:
1. Gateway fidelity check (semantic alignment)
2. Provider content filters (toxicity, harm)
3. Model training guardrails (Constitutional AI)

### 6.2 Relation to Adversarial Validation

The TELOS Framework achieved **0% Attack Success Rate** across 1,300 adversarial attacks in session-level testing. Gateway extends this to API-level governance with the same mathematical foundations.

Planned validation (Q2 2026):
- API-level adversarial testing against Gateway
- Tool invocation manipulation attacks
- System prompt injection through user messages
- Fidelity threshold evasion attempts

---

## 7. Regulatory Alignment

### 7.1 EU AI Act Compliance

**Article 52 - Transparency Obligations**:
- Gateway logs AI nature of all interactions
- Governance metadata documents automated decision-making

**Article 72 - Post-Market Monitoring**:
- Continuous fidelity measurement (every request)
- Complete audit trail (telemetry logs)
- Drift detection (fidelity trend analysis)

### 7.2 California SB 53 Compliance

**Safety Framework Documentation**:
- PA defines agent's constitutional boundaries
- Fidelity scores demonstrate boundary enforcement
- Telemetry provides incident reporting evidence

**Active Governance Mechanisms**:
- Gateway is always-on infrastructure, not optional feature
- Governance cannot be bypassed by agent code
- Evidence generation is automatic and tamper-resistant

---

## 8. Integration Guide

### 8.1 Drop-In Replacement

For existing agents using OpenAI SDK:

```python
# Before: Direct OpenAI
client = OpenAI(api_key="sk-...")

# After: Through TELOS Gateway
client = OpenAI(
    api_key="sk-...",
    base_url="http://gateway.telos-labs.ai/v1"  # Only change needed
)
```

All existing code continues to work—Gateway is transparent to properly-aligned requests.

### 8.2 Agent Registration (Optional)

For persistent governance across sessions:

```python
import requests

# Register agent with declared purpose
response = requests.post(
    "http://gateway.telos-labs.ai/v1/agents",
    json={
        "name": "Financial Advisor Agent",
        "purpose_statement": "Assist users with investment analysis, portfolio management, and market research for stocks, bonds, and ETFs.",
        "domain": "finance",
        "risk_level": "medium"
    }
)

agent_api_key = response.json()["api_key"]

# All subsequent requests use agent-specific governance
client = OpenAI(
    api_key=agent_api_key,
    base_url="http://gateway.telos-labs.ai/v1"
)
```

### 8.3 Governance Dashboard

The Gateway Streamlit interface (port 8502) provides:
- Real-time governance decision visualization
- Agent template selection for testing
- Fidelity score monitoring
- API request/response inspection

---

## 9. Limitations and Future Work

### 9.1 Current Limitations

**Single-Turn Focus**:
- Gateway evaluates each request independently
- No cross-request drift tracking (use TELOS Framework for session governance)

**Embedding Model Dependency**:
- Fidelity quality depends on embedding model (currently Mistral)
- Different embeddings may require threshold recalibration

**Latency Addition**:
- 25-55ms overhead per request
- May be significant for high-frequency agent loops

### 9.2 Planned Enhancements

**Multi-Turn Context** (Q2 2026):
- Track fidelity across request sequences
- Detect gradual purpose drift in agent conversations

**Adaptive Thresholds** (Q2 2026):
- Domain-specific threshold tuning
- Agent-specific calibration based on historical behavior

**Tool-Level Blocking** (Q3 2026):
- Block specific tools while allowing request
- Return modified tool set to LLM

**Streaming Support** (Q3 2026):
- Apply governance to streamed responses
- Token-by-token fidelity monitoring

---

## 10. Conclusion

TELOS Gateway extends constitutional governance from session-level (TELOS Framework) to API-level (agentic infrastructure). By interposing semantic fidelity measurement at the API boundary, Gateway enables **purpose-bound agent behavior** without:

- Modifying agent application code
- Changing LLM provider integrations
- Adding per-request governance logic
- Trusting prompt-based constraints alone

The same mathematical foundations proven in adversarial testing (0% ASR across 1,300 attacks) now apply to agentic systems. As AI agents gain tool access, database connectivity, and real-world action capability, infrastructure-level constitutional filtering becomes essential—not for safety theater, but for **quantitative governance** that can be measured, audited, and demonstrated to regulators.

We are not building governance as a feature. We are building governance as infrastructure.

---

## References

1. TELOS Framework Whitepaper v2.3 (January 2026) - Session-level constitutional governance
2. Vaswani et al. (2017) - "Attention Is All You Need" - Transformer attention mechanics
3. EU AI Act (2024) - Articles 52, 72 - AI governance requirements
4. California SB 53 (2025) - Transparency in Frontier AI Act
5. TELOS Adversarial Validation (November 2025) - 0% ASR security testing

---

**Document Version**: 1.0
**Last Updated**: January 2026
**Author**: TELOS AI Labs Inc.
**Contact**: JB@telos-labs.ai

---

*This document is published openly under the TELOS Consortium's commitment to transparent research. All methodologies are documented for reproducibility.*
