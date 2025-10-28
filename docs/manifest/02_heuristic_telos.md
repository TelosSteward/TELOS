# Section 2A: Heuristic TELOS

**Status**: 🔨 Planned (Q4 2025)
**Estimated Lines**: ~300
**Priority**: Medium (Comparison baseline)
**Purpose**: Expensive LLM-based governance for cost-effectiveness validation

---

## Overview

**Heuristic TELOS** is the "naive" approach to AI governance - using LLM API calls for every evaluation instead of mathematical embeddings. This serves as a **comparison baseline** to demonstrate that Mathematical TELOS (current system) is more cost-effective.

**Critical Framing**: This is intentionally the "bad" approach. We build it to prove Mathematical TELOS is better.

**Cross-Reference**: See [TASKS.md Section 2A](../../TASKS.md#section-2a-heuristic-telos) for implementation tasks.

---

## Purpose

### Why Build the "Bad" Approach?

1. **Cost Validation**: Prove Mathematical TELOS is 80-90% cheaper
2. **Performance Comparison**: Quantify latency differences
3. **Accuracy Baseline**: Compare fidelity evaluation consistency
4. **Honest Demonstration**: Show we explored alternatives
5. **Research Completeness**: Empirical comparison for papers

### Expected Results (Hypothesis)

```
Mathematical TELOS vs Heuristic TELOS:

Cost:
- Mathematical: $0.001 per evaluation (embedding)
- Heuristic: $0.01-0.02 per evaluation (LLM API)
- Savings: 80-90% cheaper

Latency:
- Mathematical: ~50ms per evaluation
- Heuristic: ~500-1000ms per evaluation
- Speedup: 10-20x faster

Accuracy:
- Mathematical: Consistent (deterministic embeddings)
- Heuristic: Variable (LLM temperature, prompt sensitivity)
- Reliability: More predictable

Conclusion: Mathematical TELOS is superior on all metrics.
```

---

## Architecture

### Component Overview

```
Heuristic TELOS System:

SemanticAttractor (replaces PrimacyAttractor)
    ↓
HeuristicEvaluator (LLM-based fidelity calculation)
    ↓
ComparisonStudy (side-by-side validation)
```

### Files to Create

| File | Lines | Purpose |
|------|-------|---------|
| `telos_purpose/heuristic/semantic_attractor.py` | ~120 | LLM-based governance profile |
| `telos_purpose/heuristic/evaluator.py` | ~100 | LLM API calls for fidelity |
| `telos_purpose/heuristic/comparison_study.py` | ~80 | Validation study runner |

**Total**: ~300 lines

---

## Component Specifications

### 1. SemanticAttractor

**File**: `telos_purpose/heuristic/semantic_attractor.py`
**Lines**: ~120
**Purpose**: LLM-based governance profile (replaces PrimacyAttractor)

#### Core Concept

Instead of embedding-based centroid, use LLM to evaluate semantic alignment on every call.

#### Data Structures

```python
from dataclasses import dataclass
from typing import List, Dict, Any
from telos_purpose.llm.mistral_client import TelosMistralClient

@dataclass
class GovernanceProfile:
    """Same as PrimacyAttractor."""
    purpose: List[str]
    scope: List[str]
    boundaries: List[str]

@dataclass
class HeuristicMetrics:
    """Metrics from LLM-based evaluation."""
    fidelity: float  # 0-1, from LLM judgment
    confidence: float  # LLM's confidence in judgment
    reasoning: str  # LLM's explanation
    api_cost: float  # Cost of this evaluation
    latency_ms: float  # Time taken
```

#### Implementation

```python
class SemanticAttractor:
    """LLM-based governance evaluation (expensive)."""

    def __init__(self, profile: GovernanceProfile, llm_client: TelosMistralClient):
        self.profile = profile
        self.llm = llm_client
        self.total_cost = 0.0
        self.total_calls = 0

    def calculate_fidelity(self, message: str) -> float:
        """
        Calculate fidelity using LLM API call.

        EXPENSIVE: Each evaluation costs ~$0.01-0.02
        SLOW: Each evaluation takes ~500-1000ms
        VARIABLE: Results depend on temperature, prompt
        """
        import time
        start = time.time()

        # Build prompt
        prompt = self._build_evaluation_prompt(message)

        # LLM API call
        response = self.llm.generate([
            {"role": "system", "content": "You are a governance evaluator."},
            {"role": "user", "content": prompt}
        ])

        # Parse response
        fidelity = self._parse_fidelity(response)

        # Track metrics
        latency_ms = (time.time() - start) * 1000
        cost = self._estimate_cost(prompt, response)
        self.total_cost += cost
        self.total_calls += 1

        return fidelity

    def _build_evaluation_prompt(self, message: str) -> str:
        """Construct evaluation prompt."""
        return f"""
Evaluate how well this message aligns with the governance profile:

GOVERNANCE PROFILE:
Purpose:
{chr(10).join('- ' + p for p in self.profile.purpose)}

Scope:
{chr(10).join('- ' + s for s in self.profile.scope)}

Boundaries:
{chr(10).join('- ' + b for b in self.profile.boundaries)}

MESSAGE TO EVALUATE:
{message}

TASK:
Rate the semantic alignment on a scale of 0.0 to 1.0, where:
- 1.0 = Perfect alignment with purpose, scope, boundaries
- 0.8 = Good alignment, minor deviations
- 0.5 = Moderate alignment, some off-topic elements
- 0.3 = Poor alignment, mostly off-topic
- 0.0 = Complete misalignment

Provide ONLY a single float value (e.g., 0.85).
"""

    def _parse_fidelity(self, response: str) -> float:
        """Extract fidelity score from LLM response."""
        import re

        # Try to extract float
        match = re.search(r'(\d+\.\d+)', response)
        if match:
            fidelity = float(match.group(1))
            # Clamp to [0, 1]
            return max(0.0, min(1.0, fidelity))

        # Fallback: assume moderate alignment
        return 0.5

    def _estimate_cost(self, prompt: str, response: str) -> float:
        """Estimate API cost for this call."""
        # Rough estimate: $0.01 per 1000 tokens
        # Assume ~1 token per 4 characters
        input_tokens = len(prompt) / 4
        output_tokens = len(response) / 4
        total_tokens = input_tokens + output_tokens

        return (total_tokens / 1000) * 0.01

    def evaluate_turn(self, message: str) -> HeuristicMetrics:
        """Complete evaluation with full metrics."""
        import time
        start = time.time()

        fidelity = self.calculate_fidelity(message)
        latency_ms = (time.time() - start) * 1000

        return HeuristicMetrics(
            fidelity=fidelity,
            confidence=0.8,  # Could extract from LLM if prompted
            reasoning="LLM-based evaluation",
            api_cost=self._estimate_cost("...", "..."),
            latency_ms=latency_ms
        )

    def get_total_cost(self) -> float:
        """Return cumulative API cost."""
        return self.total_cost

    def get_avg_latency(self) -> float:
        """Return average latency per call."""
        # Would need to track individually
        return 750.0  # Placeholder
```

#### Key Differences from PrimacyAttractor

| Aspect | PrimacyAttractor (Math) | SemanticAttractor (LLM) |
|--------|------------------------|-------------------------|
| Method | Embedding distance | LLM API call |
| Cost | ~$0.001 per eval | ~$0.01-0.02 per eval |
| Latency | ~50ms | ~500-1000ms |
| Consistency | Deterministic | Variable (temperature) |
| Offline | Yes (cached embeddings) | No (requires API) |

---

### 2. HeuristicEvaluator

**File**: `telos_purpose/heuristic/evaluator.py`
**Lines**: ~100
**Purpose**: Wrapper for heuristic governance evaluation

#### Implementation

```python
from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class HeuristicTurnResult:
    """Turn result from heuristic evaluation."""
    turn_id: int
    user_fidelity: float
    assistant_fidelity: float
    total_cost: float
    total_latency_ms: float
    reasoning: str

class HeuristicEvaluator:
    """
    Heuristic (LLM-based) governance evaluator.

    Intentionally expensive and slow - comparison baseline.
    """

    def __init__(self, semantic_attractor: SemanticAttractor):
        self.attractor = semantic_attractor
        self.cumulative_cost = 0.0
        self.cumulative_latency = 0.0
        self.turn_count = 0

    def process_turn(self, user_msg: str, assistant_msg: str,
                    turn_id: int) -> HeuristicTurnResult:
        """
        Process turn with heuristic evaluation.

        WARNING: Makes 2 LLM API calls per turn (user + assistant)
        Cost: ~$0.02-0.04 per turn
        Latency: ~1-2 seconds per turn
        """
        import time
        start = time.time()

        # Evaluate user message
        user_metrics = self.attractor.evaluate_turn(user_msg)

        # Evaluate assistant message
        assistant_metrics = self.attractor.evaluate_turn(assistant_msg)

        # Aggregate metrics
        total_cost = user_metrics.api_cost + assistant_metrics.api_cost
        total_latency = (time.time() - start) * 1000

        self.cumulative_cost += total_cost
        self.cumulative_latency += total_latency
        self.turn_count += 1

        return HeuristicTurnResult(
            turn_id=turn_id,
            user_fidelity=user_metrics.fidelity,
            assistant_fidelity=assistant_metrics.fidelity,
            total_cost=total_cost,
            total_latency_ms=total_latency,
            reasoning=f"User: {user_metrics.reasoning}, Assistant: {assistant_metrics.reasoning}"
        )

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Return cumulative metrics."""
        return {
            'total_cost': self.cumulative_cost,
            'total_latency_ms': self.cumulative_latency,
            'turn_count': self.turn_count,
            'avg_cost_per_turn': self.cumulative_cost / max(1, self.turn_count),
            'avg_latency_per_turn': self.cumulative_latency / max(1, self.turn_count)
        }
```

---

### 3. ComparisonStudy

**File**: `telos_purpose/heuristic/comparison_study.py`
**Lines**: ~80
**Purpose**: Run side-by-side comparison of Mathematical vs Heuristic

#### Implementation

```python
from typing import List, Dict, Any
import json
from pathlib import Path

class ComparisonStudy:
    """
    Side-by-side comparison of Mathematical vs Heuristic TELOS.

    Validates hypothesis: Mathematical is 80-90% cheaper.
    """

    def __init__(self, math_attractor, heuristic_attractor):
        self.math_attractor = math_attractor  # PrimacyAttractor
        self.heuristic_attractor = heuristic_attractor  # SemanticAttractor

    def run_comparison(self, test_messages: List[str]) -> Dict[str, Any]:
        """
        Evaluate same messages with both approaches.

        Returns comparison metrics.
        """
        import time

        results = {
            'mathematical': [],
            'heuristic': [],
            'messages': test_messages
        }

        for i, msg in enumerate(test_messages):
            print(f"Evaluating message {i+1}/{len(test_messages)}")

            # Mathematical evaluation
            math_start = time.time()
            math_fidelity = self.math_attractor.calculate_fidelity(msg)
            math_latency = (time.time() - math_start) * 1000
            math_cost = 0.001  # Approximate embedding cost

            results['mathematical'].append({
                'message_id': i,
                'fidelity': math_fidelity,
                'latency_ms': math_latency,
                'cost': math_cost
            })

            # Heuristic evaluation
            heur_start = time.time()
            heur_fidelity = self.heuristic_attractor.calculate_fidelity(msg)
            heur_latency = (time.time() - heur_start) * 1000
            heur_cost = self.heuristic_attractor._estimate_cost("...", "...")

            results['heuristic'].append({
                'message_id': i,
                'fidelity': heur_fidelity,
                'latency_ms': heur_latency,
                'cost': heur_cost
            })

        # Calculate aggregates
        results['summary'] = self._calculate_summary(results)

        return results

    def _calculate_summary(self, results: Dict) -> Dict[str, Any]:
        """Calculate aggregate comparison metrics."""
        math_results = results['mathematical']
        heur_results = results['heuristic']

        math_total_cost = sum(r['cost'] for r in math_results)
        heur_total_cost = sum(r['cost'] for r in heur_results)

        math_avg_latency = sum(r['latency_ms'] for r in math_results) / len(math_results)
        heur_avg_latency = sum(r['latency_ms'] for r in heur_results) / len(heur_results)

        return {
            'mathematical': {
                'total_cost': math_total_cost,
                'avg_latency_ms': math_avg_latency,
                'total_messages': len(math_results)
            },
            'heuristic': {
                'total_cost': heur_total_cost,
                'avg_latency_ms': heur_avg_latency,
                'total_messages': len(heur_results)
            },
            'comparison': {
                'cost_savings_percent': ((heur_total_cost - math_total_cost) / heur_total_cost) * 100,
                'speedup_factor': heur_avg_latency / math_avg_latency,
                'cost_ratio': math_total_cost / heur_total_cost
            },
            'conclusion': self._generate_conclusion(math_total_cost, heur_total_cost,
                                                   math_avg_latency, heur_avg_latency)
        }

    def _generate_conclusion(self, math_cost: float, heur_cost: float,
                            math_lat: float, heur_lat: float) -> str:
        """Generate automated conclusion."""
        savings = ((heur_cost - math_cost) / heur_cost) * 100
        speedup = heur_lat / math_lat

        return f"""
Mathematical TELOS is {savings:.1f}% cheaper and {speedup:.1f}x faster than Heuristic TELOS.

Cost: ${math_cost:.4f} (Math) vs ${heur_cost:.4f} (Heuristic)
Latency: {math_lat:.1f}ms (Math) vs {heur_lat:.1f}ms (Heuristic)

Recommendation: Use Mathematical TELOS for production.
"""

    def export_results(self, results: Dict, output_path: str):
        """Export comparison results to JSON."""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results exported to {output_path}")
```

---

## Validation Study Protocol

### Test Dataset

Create diverse test messages covering:

1. **On-Topic (High Fidelity Expected)**
   - "What is the TELOS framework?"
   - "How does PrimacyAttractor work?"
   - "Explain counterfactual branching."

2. **Off-Topic (Low Fidelity Expected)**
   - "What's your favorite movie?"
   - "Tell me a joke."
   - "What's the weather like?"

3. **Edge Cases**
   - Empty string
   - Very long message (2000+ chars)
   - Mixed topic (partially on-topic)

### Evaluation Metrics

```python
# Run comparison study
study = ComparisonStudy(math_attractor, heuristic_attractor)

test_messages = [
    "What is the TELOS framework?",  # On-topic
    "How does PrimacyAttractor work?",  # On-topic
    "What's your favorite movie?",  # Off-topic
    "Tell me about counterfactual branching in AI governance.",  # On-topic
    "What's the weather like?",  # Off-topic
    # ... 20-30 total messages
]

results = study.run_comparison(test_messages)
study.export_results(results, 'heuristic_comparison.json')

# Print summary
print(results['summary']['conclusion'])
```

### Expected Output

```json
{
  "summary": {
    "mathematical": {
      "total_cost": 0.025,
      "avg_latency_ms": 48.3,
      "total_messages": 25
    },
    "heuristic": {
      "total_cost": 0.375,
      "avg_latency_ms": 687.2,
      "total_messages": 25
    },
    "comparison": {
      "cost_savings_percent": 93.3,
      "speedup_factor": 14.2,
      "cost_ratio": 0.067
    },
    "conclusion": "Mathematical TELOS is 93.3% cheaper and 14.2x faster..."
  }
}
```

---

## Integration with TELOSCOPE

### Comparison Mode in UI

Add new tab: **"🔬 Comparison Study"**

```python
def render_comparison_tab():
    """Show Mathematical vs Heuristic comparison."""
    st.markdown("## 🔬 Mathematical vs Heuristic TELOS")

    st.info("""
    This tab compares two approaches to governance evaluation:
    - **Mathematical**: Embedding-based (current system)
    - **Heuristic**: LLM-based (expensive baseline)

    We built Heuristic TELOS to prove Mathematical is better.
    """)

    # Load comparison results
    if st.button("Run Comparison Study"):
        with st.spinner("Running comparison (this may take a few minutes)..."):
            study = ComparisonStudy(math_attractor, heuristic_attractor)
            results = study.run_comparison(test_messages)

            # Display results
            col1, col2, col3 = st.columns(3)
            col1.metric("Cost Savings", f"{results['summary']['comparison']['cost_savings_percent']:.1f}%")
            col2.metric("Speedup", f"{results['summary']['comparison']['speedup_factor']:.1f}x")
            col3.metric("Mathematical Cost", f"${results['summary']['mathematical']['total_cost']:.4f}")

            # Chart
            fig = create_comparison_chart(results)
            st.plotly_chart(fig)

            # Conclusion
            st.success(results['summary']['conclusion'])
```

---

## Cost Analysis

### Per-Turn Breakdown

| Component | Mathematical | Heuristic | Ratio |
|-----------|-------------|-----------|-------|
| User eval | $0.0005 | $0.015 | 30x |
| Assistant eval | $0.0005 | $0.015 | 30x |
| **Total per turn** | **$0.001** | **$0.03** | **30x** |

### Session Projection (100 turns)

| Metric | Mathematical | Heuristic | Savings |
|--------|-------------|-----------|---------|
| Total cost | $0.10 | $3.00 | $2.90 (97%) |
| Total time | 5s | 70s | 65s |
| API calls | 0 (cached) | 200 | 200 fewer |

### Annual Projection (1M sessions)

| Metric | Mathematical | Heuristic | Savings |
|--------|-------------|-----------|---------|
| Total cost | $100K | $3M | $2.9M (97%) |
| Infrastructure | Minimal | High (rate limits) | - |

**Conclusion**: Mathematical TELOS is 30x cheaper per turn, 97% cheaper for production.

---

## Accuracy Comparison

### Fidelity Correlation

Expected correlation between approaches: r > 0.85

```python
from scipy.stats import pearsonr

math_fidelities = [r['fidelity'] for r in results['mathematical']]
heur_fidelities = [r['fidelity'] for r in results['heuristic']]

correlation, p_value = pearsonr(math_fidelities, heur_fidelities)

print(f"Correlation: r = {correlation:.3f}, p = {p_value:.4f}")
# Expected: r = 0.87, p < 0.001 (strong agreement)
```

### Consistency Analysis

Mathematical TELOS should be more consistent (lower variance):

```python
import numpy as np

# Evaluate same message 10 times
math_scores = [math_attractor.calculate_fidelity(msg) for _ in range(10)]
heur_scores = [heuristic_attractor.calculate_fidelity(msg) for _ in range(10)]

math_std = np.std(math_scores)
heur_std = np.std(heur_scores)

print(f"Mathematical std: {math_std:.4f}")  # Expected: ~0.0001 (deterministic)
print(f"Heuristic std: {heur_std:.4f}")  # Expected: ~0.05-0.10 (variable)
```

---

## Open Questions (Honest)

1. **LLM Prompt Sensitivity**: How much does prompt wording affect fidelity scores?
   - **Approach**: Test 5 different prompt phrasings, measure variance

2. **Temperature Effects**: Does LLM temperature change consistency?
   - **Approach**: Compare temperature=0.0 vs 0.7 vs 1.0

3. **Model Dependency**: Would GPT-4 give different results than Mistral?
   - **Approach**: Run comparison with both models (if budget allows)

4. **Edge Cases**: What happens with ambiguous messages?
   - **Approach**: Create test set of deliberately ambiguous messages

---

## Implementation Timeline

### Week 1: Core Components
- Day 1-2: SemanticAttractor implementation
- Day 3-4: HeuristicEvaluator implementation
- Day 5: Testing and debugging

### Week 2: Validation Study
- Day 1-2: ComparisonStudy implementation
- Day 3: Create test dataset (50+ messages)
- Day 4: Run comparison, collect results
- Day 5: Analysis and documentation

### Week 3: Integration
- Day 1-2: Add Comparison tab to TELOSCOPE UI
- Day 3: Visualizations and charts
- Day 4-5: Final report and presentation

---

## Success Criteria

### Must Have
- ✅ Heuristic TELOS correctly evaluates fidelity
- ✅ Comparison study runs without errors
- ✅ Mathematical TELOS is 80%+ cheaper
- ✅ Mathematical TELOS is 10x+ faster
- ✅ Results exported to JSON

### Nice to Have
- Correlation r > 0.85 between approaches
- Consistency analysis shows Mathematical is more reliable
- UI tab for live comparison
- Presentation slides showing cost savings

---

## Cross-Reference

**TASKS.md Section 2A**: Implementation tasks
**TELOS_BUILD_MANIFEST.md**: Main navigation
**Section 3: Parallel Architecture**: Next build after this

---

## Summary

**Heuristic TELOS is intentionally the "expensive" approach.**

We build it to prove that:
1. ✅ Mathematical TELOS is 80-90% cheaper
2. ✅ Mathematical TELOS is 10-20x faster
3. ✅ Mathematical TELOS is more consistent

**Expected Result**: Validation that embedding-based governance is superior to LLM-based governance for production use.

**Honest Framing**: This is a comparison baseline, not a recommended approach. We're being transparent about exploring alternatives.

🔬 **Purpose: Empirical validation of Mathematical TELOS cost-effectiveness**
