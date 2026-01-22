# WorkBench Benchmark Integration Scope
## TELOS Gateway Validation Against Enterprise Workplace Tasks

**Date:** January 2026
**Status:** Scoping Document
**Objective:** Validate TELOS Gateway governance on realistic workplace agent tasks

---

## Executive Summary

WorkBench (arXiv:2405.00823) is a 690-task benchmark for evaluating AI agents on workplace tasks. Current best result: GPT-4 achieves only 43% success rate, with dangerous errors like "email sent to wrong person."

**TELOS Gateway integration would demonstrate:**
1. Governance catches dangerous misalignment errors
2. Purpose-based filtering prevents out-of-scope actions
3. Trajectory tracking detects multi-step drift
4. Enterprise-ready agentic AI governance

---

## WorkBench Architecture

### Components

| Component | Count | Description |
|-----------|-------|-------------|
| Tasks | 690 | Workplace operations (email, scheduling, database) |
| Tools | 26 | Across 6 domains |
| Databases | 5 | Sandbox environments |
| Evaluation | Outcome-centric | Did database change correctly? |

### Tool Domains (6 toolkits, 26 tools)

| Toolkit | File | Example Tools |
|---------|------|---------------|
| Calendar | `calendar.py` | Schedule meeting, check availability |
| Email | `email.py` | Send email, read inbox |
| Analytics | `analytics.py` | Query metrics, generate reports |
| Project Management | `project_management.py` | Create task, update status |
| CRM | `customer_relationship_manager.py` | Update contact, log interaction |
| Company Directory | `company_directory.py` | Lookup employee, get org chart |

### Agent Architecture

```
User Query
    │
    ▼
LangChain ReAct Agent (STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION)
    │
    ├── Reasoning: What should I do?
    ├── Action Selection: Which tool?
    └── Tool Execution: convert_agent_action_to_function_call()
            │
            ▼
        Database State Change
            │
            ▼
        Outcome Evaluation (correct/incorrect)
```

---

## TELOS Gateway Integration Points

### Architecture with Governance

```
User Query + Primacy Attractor (declared purpose)
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                    TELOS GATEWAY                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  INTERCEPT 1: Query Fidelity Check                         │
│  ├── Embed user query                                       │
│  ├── Measure cosine similarity to PA                        │
│  └── Gate: Is this within declared purpose?                 │
│                                                             │
│  INTERCEPT 2: Tool Selection Governance                     │
│  ├── Agent proposes tool + parameters                       │
│  ├── Embed (tool, action, parameters)                       │
│  ├── Measure fidelity to PA                                 │
│  └── Decision: EXECUTE / CLARIFY / BLOCK                    │
│                                                             │
│  INTERCEPT 3: Trajectory Tracking                           │
│  ├── Monitor intermediate_steps                             │
│  ├── Cumulative fidelity measurement                        │
│  └── Detect multi-step drift patterns                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
LangChain ReAct Agent (governed)
    │
    ▼
Tool Execution (only if governance approved)
    │
    ▼
Outcome Evaluation
```

### Key Integration File

**Target:** `src/evals/utils.py` - specifically the `convert_agent_action_to_function_call()` function

```python
# CURRENT (ungovverned)
action = agent.plan(query)
result = convert_agent_action_to_function_call(action)

# WITH TELOS GATEWAY
action = agent.plan(query)
governance_decision = telos_gateway.evaluate(
    action=action,
    primacy_attractor=pa,
    trajectory=intermediate_steps
)
if governance_decision.execute:
    result = convert_agent_action_to_function_call(action)
else:
    result = governance_decision.intervention  # CLARIFY/BLOCK/ESCALATE
```

---

## Implementation Requirements

### 1. Environment Setup

```bash
# Clone WorkBench
git clone https://github.com/olly-styles/WorkBench.git
cd WorkBench

# Python 3.10.11 required
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Add TELOS Gateway dependencies
pip install sentence-transformers numpy
```

### 2. Primacy Attractor Definition

For WorkBench, we need domain-specific PAs for each toolkit:

```python
WORKBENCH_PRIMACY_ATTRACTORS = {
    "calendar": {
        "purpose": "Schedule and manage calendar events for authorized users within the organization",
        "scope": "Meeting scheduling, availability checking, event modification",
        "constraints": "Only modify events for the authenticated user or with explicit delegation"
    },
    "email": {
        "purpose": "Send and manage email communications on behalf of the authenticated user",
        "scope": "Composing, sending, reading emails within organizational boundaries",
        "constraints": "Verify recipient before sending; no bulk unsolicited emails"
    },
    "crm": {
        "purpose": "Manage customer relationship data for legitimate business operations",
        "scope": "Contact updates, interaction logging, pipeline management",
        "constraints": "Maintain data accuracy; no unauthorized data export"
    },
    # ... etc for each toolkit
}
```

### 3. TELOS Gateway Adapter

**New file:** `src/telos_gateway_adapter.py`

```python
from telos_gateway import TELOSGateway, PrimacyAttractor
from sentence_transformers import SentenceTransformer

class WorkBenchGovernanceAdapter:
    def __init__(self, pa_config: dict):
        self.gateway = TELOSGateway()
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.primacy_attractors = self._build_attractors(pa_config)
        self.trajectory = []

    def evaluate_action(self, action, toolkit: str) -> GovernanceDecision:
        """
        Evaluate proposed agent action against Primacy Attractor.

        Returns:
            GovernanceDecision with:
            - execute: bool
            - decision: EXECUTE/CLARIFY/SUGGEST/INERT/ESCALATE
            - fidelity: float
            - reason: str
        """
        # Embed the action
        action_text = f"{action.tool}: {action.tool_input}"
        action_embedding = self.encoder.encode(action_text)

        # Get PA for this toolkit
        pa = self.primacy_attractors[toolkit]

        # Measure fidelity
        fidelity = self.gateway.compute_fidelity(
            action_embedding,
            pa.embedding
        )

        # Track trajectory
        self.trajectory.append({
            'action': action_text,
            'fidelity': fidelity,
            'toolkit': toolkit
        })

        # Trajectory drift check
        trajectory_fidelity = self._compute_trajectory_fidelity()

        # Governance decision
        return self.gateway.decide(
            fidelity=fidelity,
            trajectory_fidelity=trajectory_fidelity,
            action=action
        )

    def _compute_trajectory_fidelity(self) -> float:
        """Cumulative fidelity across action chain."""
        if not self.trajectory:
            return 1.0
        weights = [0.8 ** i for i in range(len(self.trajectory))]
        weighted_sum = sum(w * t['fidelity'] for w, t in zip(weights, reversed(self.trajectory)))
        return weighted_sum / sum(weights)
```

### 4. Modified Evaluation Loop

**Modify:** `src/evals/utils.py`

```python
# Add governance wrapper
def run_governed_inference(query, agent, tools, governance_adapter):
    """Run agent with TELOS Gateway governance."""

    # Check query fidelity first
    query_decision = governance_adapter.evaluate_query(query)
    if not query_decision.execute:
        return {
            'blocked': True,
            'reason': query_decision.reason,
            'stage': 'query_gate'
        }

    # Run agent with governed tool execution
    intermediate_steps = []
    for iteration in range(MAX_ITERATIONS):
        # Agent plans next action
        action = agent.plan(query, intermediate_steps)

        if action.tool == "Final Answer":
            break

        # TELOS GATEWAY INTERCEPT
        toolkit = get_toolkit_for_tool(action.tool)
        decision = governance_adapter.evaluate_action(action, toolkit)

        if decision.execute:
            # Approved - execute tool
            observation = execute_tool(action)
            intermediate_steps.append((action, observation))
        else:
            # Blocked or needs clarification
            if decision.decision == "CLARIFY":
                observation = f"[GOVERNANCE] Clarification needed: {decision.reason}"
            elif decision.decision == "BLOCK":
                observation = f"[GOVERNANCE] Action blocked: {decision.reason}"
                # Log for analysis
                log_governance_intervention(action, decision)
            intermediate_steps.append((action, observation))

    return {
        'intermediate_steps': intermediate_steps,
        'governance_trace': governance_adapter.get_trace()
    }
```

---

## Evaluation Metrics

### Primary Metrics (New - Governance)

| Metric | Description | Target |
|--------|-------------|--------|
| **Dangerous Error Prevention Rate** | % of "wrong recipient" type errors caught | >95% |
| **Out-of-Scope Action Block Rate** | % of off-purpose actions blocked | >90% |
| **Trajectory Drift Detection** | % of multi-step attacks caught by step 3 | >95% |
| **Governance Latency** | Time added per tool call | <50ms |

### Secondary Metrics (Existing - Utility)

| Metric | Description | Baseline | With Governance |
|--------|-------------|----------|-----------------|
| Task Completion Rate | % of tasks completed correctly | 43% (GPT-4) | TBD |
| Action Accuracy | % of individual actions correct | TBD | TBD |
| False Positive Rate | Legitimate actions incorrectly blocked | N/A | <5% target |

### Combined Metric: Net Resilient Performance (NRP)

```
NRP = (Task_Completion_Rate × Safety_Score) / (1 + False_Positive_Rate)

Where:
- Safety_Score = 1 - (Dangerous_Errors / Total_Tasks)
- Target: NRP improvement over ungoverned baseline
```

---

## Dangerous Error Categories to Detect

Based on WorkBench findings, these are the error types TELOS should catch:

| Error Type | Example | TELOS Detection Method |
|------------|---------|------------------------|
| Wrong Recipient | Email sent to wrong person | PA fidelity on (recipient, purpose) |
| Scope Creep | Calendar tool used for unauthorized user | Tool manifest constraint check |
| Data Leakage | CRM data exported to external party | PA boundary violation |
| Action Reversal | Deleting instead of updating | Semantic similarity to intended action |
| Privilege Escalation | Accessing data above clearance | PA authorization scope |

---

## Implementation Timeline

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| **Setup** | 2 days | WorkBench running locally, baseline metrics |
| **PA Design** | 3 days | Domain-specific Primacy Attractors for 6 toolkits |
| **Adapter Build** | 5 days | TELOS Gateway adapter integrated with utils.py |
| **Baseline Run** | 2 days | Ungoverned agent results on full 690 tasks |
| **Governed Run** | 3 days | TELOS-governed agent results |
| **Analysis** | 3 days | Error prevention analysis, NRP calculation |
| **Documentation** | 2 days | Results write-up, publication prep |

**Total: ~3 weeks**

---

## Resource Requirements

### Compute

| Resource | Requirement |
|----------|-------------|
| Python | 3.10.11 |
| RAM | 16GB minimum (embedding models) |
| GPU | Optional (CPU inference acceptable) |
| API Keys | OpenAI (GPT-4), optional Anthropic |

### API Costs (Estimated)

| Model | Tasks | Est. Cost |
|-------|-------|-----------|
| GPT-4 | 690 | ~$50-100 (depends on tokens) |
| GPT-3.5 | 690 | ~$5-10 |

### Storage

- WorkBench repo: ~500MB
- Results data: ~100MB
- Governance traces: ~50MB

---

## Success Criteria

### Minimum Viable Result

1. **Dangerous Error Prevention**: >80% of "wrong recipient" errors caught
2. **Utility Preserved**: Task completion rate within 10% of ungoverned baseline
3. **Latency**: <100ms governance overhead per action

### Strong Result (Publication-Worthy)

1. **Dangerous Error Prevention**: >95%
2. **Utility Preserved**: Task completion rate within 5% of baseline OR improved
3. **Latency**: <50ms
4. **Novel Finding**: Trajectory-level governance catches errors single-action checks miss

### Exceptional Result

1. **TELOS-governed GPT-4 achieves higher task completion than ungoverned**
   - Hypothesis: Governance prevents cascading errors that derail multi-step tasks
2. First published governance results on WorkBench

---

## Strategic Value

### For NSF SBIR

- Demonstrates agentic AI governance on realistic enterprise tasks
- Complements AgentHarm/AgentDojo security benchmarks with utility benchmark
- Shows TELOS works across different agent architectures (LangChain ReAct)

### For Work-Bench VC Conversation

- "We validated against WorkBench - here's how governance improves workplace AI safety"
- Direct relevance to their enterprise AI investment thesis
- Differentiates from Arthur AI (observability) with runtime governance

### For Market Positioning

- First governance solution with WorkBench results
- Enterprise-relevant (workplace tasks, not just security attacks)
- Quantified safety-utility tradeoff

---

## Next Steps

1. [ ] Clone and run WorkBench baseline
2. [ ] Design 6 domain-specific Primacy Attractors
3. [ ] Build governance adapter
4. [ ] Run governed evaluation
5. [ ] Analyze and document results

---

*Scoping document prepared January 2026*
*TELOS AI Labs Inc.*
