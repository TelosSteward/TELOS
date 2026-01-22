# TELOS Gateway Agentic Benchmark Validation Plan

**Purpose:** Internal validation of TELOS Gateway against adversarial agentic benchmarks
**Status:** Pre-NSF submission internal testing
**Date:** January 2026

---

## Executive Summary

The TELOS Gateway already implements two-tier governance for agentic AI:
- **Tier 1:** Fidelity check (request vs agent Primacy Attractor)
- **Tier 2:** Tool selection fidelity (request vs tool descriptions)

This plan outlines how to validate the Gateway against established adversarial benchmarks to confirm the approach works for agentic systems before the NSF SBIR submission.

---

## Benchmark Targets

### 1. AgentHarm (ICLR 2025)

**Source:** HuggingFace `ai-safety-institute/AgentHarm`
**Tasks:** 110 malicious agent tasks (440 with augmentations)
**Categories:** Fraud, cybercrime, harassment, etc.
**Baseline ASR:** ~80% (attacks succeed against undefended agents)

**Integration Approach:**
```python
# AgentHarm uses the inspect_evals framework
# TELOS integrates as a custom "defense" solver

from inspect_evals import agentharm
from telos_gateway.governance.fidelity_gate import FidelityGate

class TELOSDefenseSolver:
    """TELOS as a defense layer for AgentHarm evaluation."""

    def __init__(self, agent_pa: str, embed_fn):
        self.fidelity_gate = FidelityGate(embed_fn)
        self.pa = PrimacyAttractor(agent_pa)

    def solve(self, task):
        # Check request fidelity BEFORE passing to agent
        result = self.fidelity_gate.check_request(task.prompt, self.pa)

        if result.final_decision in [ActionDecision.INERT, ActionDecision.ESCALATE]:
            return "I cannot assist with that request."

        # Only if governance passes, forward to underlying agent
        return self.underlying_agent.solve(task)
```

**Run Command:**
```bash
# After implementing TELOS defense solver
inspect eval inspect_evals/agentharm --model openai/gpt-4o -T agent=telos_defense
```

### 2. AgentDojo (NeurIPS 2024)

**Source:** `pip install agentdojo`
**Tasks:** 97 realistic tasks, 629 security test cases
**Attack Types:** Prompt injection, tool manipulation, context poisoning

**Integration Approach:**
```python
# AgentDojo provides a simple agent interface
# TELOS wraps the agent's query function

from agentdojo.agent_pipeline import AgentPipeline
from telos_gateway.governance.fidelity_gate import FidelityGate

class TELOSGovernedAgent(AgentPipeline):
    """TELOS governance layer for AgentDojo agents."""

    def __init__(self, base_agent, agent_pa: str, embed_fn):
        self.base_agent = base_agent
        self.fidelity_gate = FidelityGate(embed_fn)
        self.pa = PrimacyAttractor(agent_pa)

    def query(self, user_instructions, tools, state):
        # Tier 1: Check request fidelity
        result = self.fidelity_gate.check_request_text(
            user_instructions, self.pa
        )

        if not result.should_forward:
            return self._governance_refusal(result)

        # Tier 2: Filter tools to only purpose-aligned ones
        filtered_tools = self._filter_tools_by_fidelity(tools)

        # Forward to base agent with governance constraints
        return self.base_agent.query(user_instructions, filtered_tools, state)
```

**Run Command:**
```bash
python -m agentdojo.scripts.benchmark -s workspace \
  --model gpt-4o-2024-05-13 \
  --defense telos_gateway \
  --attack tool_knowledge
```

---

## Validation Phases

### Phase 1: Baseline Measurement (Week 1)

1. **Set up benchmarks:**
   ```bash
   pip install agentdojo
   pip install "agentdojo[transformers]"
   # For AgentHarm:
   pip install inspect-ai
   pip install inspect-evals
   ```

2. **Run undefended baseline:**
   - Document baseline ASR for each benchmark
   - Establish comparison point

3. **Run system-prompt defense:**
   - Standard "refuse harmful requests" prompt
   - Document ASR improvement (expect ~7-10% based on literature)

### Phase 2: TELOS Integration (Week 2)

1. **Create TELOS defense adapters:**
   - `telos_agentharm_adapter.py` - Wraps FidelityGate for AgentHarm
   - `telos_agentdojo_adapter.py` - Wraps FidelityGate for AgentDojo

2. **Define agent Primacy Attractors:**
   ```python
   # Generic safe agent PA
   SAFE_AGENT_PA = """You are a helpful assistant that operates within
   legal and ethical boundaries. You help users with legitimate tasks
   including information retrieval, analysis, and task completion.
   You do NOT assist with harmful, illegal, or unethical activities."""
   ```

3. **Run TELOS-defended agent:**
   - Document ASR with TELOS governance
   - Compare to baseline and system-prompt

### Phase 3: Analysis (Week 3)

1. **Calculate metrics:**
   - Attack Success Rate (ASR)
   - Utility preservation (benign task success rate)
   - Tier distribution (% blocked at Tier 1 vs Tier 2)

2. **Analyze failure cases:**
   - Which attacks (if any) bypassed TELOS?
   - What fidelity scores did they produce?
   - Can thresholds be tuned?

3. **Document findings:**
   - Internal report with results
   - Keep for NSF Phase I milestone comparison

---

## Expected Results (Hypothesis)

Based on semantic validation results (0% ASR on 1,300 attacks), we hypothesize:

| Defense | AgentHarm ASR | AgentDojo Security | Notes |
|---------|---------------|-------------------|-------|
| None (baseline) | ~80% | Low | Per published benchmarks |
| System prompt | ~70% | Low-Medium | ~10% improvement typical |
| **TELOS Gateway** | **<10%** | **High** | Hypothesis based on semantic results |

**Key Insight:** Malicious agent tasks should produce LOW fidelity against a legitimate agent PA, just like adversarial prompts produced low fidelity in semantic testing.

---

## Technical Requirements

### Dependencies
```bash
# Core
pip install numpy sentence-transformers

# Benchmarks
pip install agentdojo "agentdojo[transformers]"
pip install inspect-ai inspect-evals

# TELOS Gateway
cd /Users/brunnerjf/Desktop/telos_gateway
pip install -e .
```

### Environment Variables
```bash
export MISTRAL_API_KEY=<your_key>
export OPENAI_API_KEY=<your_key>  # For benchmark LLMs
```

### Hardware
- Standard laptop sufficient for initial testing
- Embedding computation: ~50ms per request
- Full benchmark: ~2-4 hours depending on LLM rate limits

---

## Success Criteria

**Primary:** TELOS-governed agent achieves ASR <10% on AgentHarm (vs ~80% baseline)

**Secondary:**
- Utility preserved: >90% success on benign tasks
- Tier 1 blocks >80% of attacks autonomously (no Tier 2 needed)
- Latency overhead <100ms per request

---

## Files to Create

1. `/Users/brunnerjf/Desktop/telos_gateway/benchmarks/agentharm_adapter.py`
2. `/Users/brunnerjf/Desktop/telos_gateway/benchmarks/agentdojo_adapter.py`
3. `/Users/brunnerjf/Desktop/telos_gateway/benchmarks/run_agentharm.py`
4. `/Users/brunnerjf/Desktop/telos_gateway/benchmarks/run_agentdojo.py`
5. `/Users/brunnerjf/Desktop/telos_gateway/benchmarks/results/` (output directory)

---

## Timeline

| Week | Activity | Deliverable |
|------|----------|-------------|
| 1 | Setup + Baseline | Undefended ASR measurements |
| 2 | TELOS Integration | Adapter code + initial results |
| 3 | Analysis + Documentation | Internal validation report |

**Total:** 3 weeks for internal validation before NSF submission

---

*This is INTERNAL testing only. Results inform NSF proposal confidence but are not disclosed as "already done" - the proposal frames this as Phase I R&D.*
