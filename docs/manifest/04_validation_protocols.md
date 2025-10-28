# Section 4: Validation Protocols

**Status**: 🔨 In Progress
**Priority**: High
**Purpose**: Testing and validation for TELOS components

---

## Overview

Comprehensive validation protocols for all TELOS components, including:
- HBP (Hierarchical Boundary Partitioning) specification
- Manual testing checklists
- Automated test suites
- Statistical validation methods
- Performance benchmarking

**Cross-Reference**: See [TASKS.md Section 4](../../TASKS.md#section-4-validation) for validation tasks.

---

## Section 2C: Hierarchical Boundary Partitioning (HBP)

**Status**: 🔨 Planned (Q4 2025)
**Lines**: ~350
**Priority**: Medium
**Purpose**: Spatial decomposition of governance basin

### Core Concept

**Problem**: PrimacyAttractor has a single spherical basin. What if governance regions are non-convex or multi-modal?

**Solution**: Partition embedding space into hierarchical regions, each with different governance characteristics.

**Example**:
- Inner basin (r < 0.5): Core governance, high confidence
- Middle basin (0.5 ≤ r < 1.0): Acceptable, monitor
- Outer region (r ≥ 1.0): Drift, intervene

### Architecture

**File**: `telos_purpose/core/hierarchical_boundary.py`
**Lines**: ~350

#### Data Structures

```python
from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np

@dataclass
class BoundaryRegion:
    """Defines one region in hierarchical partition."""
    region_id: str
    region_name: str  # "core", "acceptable", "warning", "critical"
    inner_radius: float
    outer_radius: float
    governance_policy: str  # "monitor", "warn", "intervene"
    confidence_threshold: float

@dataclass
class HBPMetrics:
    """Metrics from HBP evaluation."""
    fidelity: float
    distance: float
    region: BoundaryRegion
    confidence: float
    recommended_action: str

class HierarchicalBoundaryPartitioner:
    """
    Partition embedding space into hierarchical governance regions.

    Enables nuanced governance beyond binary in/out basin.
    """

    def __init__(self, centroid: np.ndarray, regions: List[BoundaryRegion]):
        self.centroid = centroid
        self.regions = sorted(regions, key=lambda r: r.inner_radius)

    def evaluate(self, embedding: np.ndarray) -> HBPMetrics:
        """Evaluate which region the embedding falls into."""
        distance = np.linalg.norm(embedding - self.centroid)

        # Find region
        region = self._find_region(distance)

        # Calculate fidelity
        fidelity = 1.0 / (1.0 + distance)

        # Confidence based on region
        confidence = self._calculate_confidence(distance, region)

        # Recommended action
        action = self._recommend_action(region, fidelity)

        return HBPMetrics(
            fidelity=fidelity,
            distance=distance,
            region=region,
            confidence=confidence,
            recommended_action=action
        )

    def _find_region(self, distance: float) -> BoundaryRegion:
        """Find which region contains this distance."""
        for region in self.regions:
            if region.inner_radius <= distance < region.outer_radius:
                return region

        # Default: outermost region
        return self.regions[-1]

    def _calculate_confidence(self, distance: float, region: BoundaryRegion) -> float:
        """Calculate confidence in governance for this region."""
        # Confidence decreases with distance
        region_span = region.outer_radius - region.inner_radius
        position_in_region = (distance - region.inner_radius) / region_span

        # Higher confidence near inner boundary
        return region.confidence_threshold * (1.0 - position_in_region)

    def _recommend_action(self, region: BoundaryRegion, fidelity: float) -> str:
        """Recommend action based on region policy."""
        if region.governance_policy == "monitor":
            return "continue"
        elif region.governance_policy == "warn":
            return "warn_user"
        elif region.governance_policy == "intervene":
            return "generate_intervention"
        else:
            return "unknown"
```

#### Example Configuration

```python
# Define hierarchical regions
regions = [
    BoundaryRegion(
        region_id="core",
        region_name="Core Governance",
        inner_radius=0.0,
        outer_radius=0.5,
        governance_policy="monitor",
        confidence_threshold=0.95
    ),
    BoundaryRegion(
        region_id="acceptable",
        region_name="Acceptable Range",
        inner_radius=0.5,
        outer_radius=0.8,
        governance_policy="monitor",
        confidence_threshold=0.85
    ),
    BoundaryRegion(
        region_id="warning",
        region_name="Warning Zone",
        inner_radius=0.8,
        outer_radius=1.2,
        governance_policy="warn",
        confidence_threshold=0.70
    ),
    BoundaryRegion(
        region_id="critical",
        region_name="Critical Drift",
        inner_radius=1.2,
        outer_radius=float('inf'),
        governance_policy="intervene",
        confidence_threshold=0.50
    )
]

hbp = HierarchicalBoundaryPartitioner(attractor.centroid, regions)

# Evaluate message
metrics = hbp.evaluate(message_embedding)

print(f"Region: {metrics.region.region_name}")
print(f"Confidence: {metrics.confidence:.3f}")
print(f"Action: {metrics.recommended_action}")
```

### Validation Protocol

1. **Test with known messages** at different distances
2. **Verify region assignment** matches distance
3. **Check confidence calculation** decreases with distance
4. **Validate action recommendations** match policy

### Expected Results

```
Message: "What is TELOS?" (on-topic)
→ Distance: 0.3, Region: Core, Confidence: 0.93, Action: continue

Message: "Explain PrimacyAttractor" (on-topic, slightly broader)
→ Distance: 0.6, Region: Acceptable, Confidence: 0.82, Action: monitor

Message: "What about AI safety?" (related but drifting)
→ Distance: 0.9, Region: Warning, Confidence: 0.65, Action: warn_user

Message: "What's your favorite movie?" (off-topic)
→ Distance: 1.8, Region: Critical, Confidence: 0.35, Action: generate_intervention
```

---

## Manual Testing Checklists

### TELOSCOPE UI Testing

#### Initialization
- [ ] Launch dashboard without errors
- [ ] API key validation works
- [ ] Default config loads correctly
- [ ] All components initialize
- [ ] Sidebar displays initial metrics

#### Live Session Tab
- [ ] Chat input accepts messages
- [ ] LLM responses appear
- [ ] Metrics update in real-time
- [ ] Drift warnings appear when F < 0.8
- [ ] Trigger badges display
- [ ] Click trigger navigates to TELOSCOPE tab
- [ ] Session history persists

#### Session Replay Tab
- [ ] Timeline slider works
- [ ] First/Prev/Next buttons functional
- [ ] Turn display updates correctly
- [ ] Metrics panel shows correct values
- [ ] Trigger markers clickable
- [ ] Boundary conditions handled (turn 0, last turn)

#### TELOSCOPE Tab
- [ ] Welcome screen shows when no triggers
- [ ] Trigger selector populates
- [ ] Loading state displays during generation
- [ ] ΔF metrics display correctly
- [ ] Side-by-side comparison renders
- [ ] Divergence chart displays (or fallback)
- [ ] Statistical analysis expands
- [ ] Export button downloads JSON
- [ ] Filename includes timestamp

#### Analytics Tab
- [ ] Session overview metrics correct
- [ ] Historical fidelity chart renders
- [ ] Efficacy summary table populates
- [ ] Aggregate metrics calculate correctly
- [ ] Overall assessment matches data
- [ ] Complete analytics export works

### Backend Component Testing

#### SessionStateManager
- [ ] Turn snapshots are immutable
- [ ] State reconstruction works
- [ ] Audit trail export valid
- [ ] Session reset clears all data

#### PrimacyAttractor
- [ ] Centroid calculation correct
- [ ] Fidelity formula validated
- [ ] Basin membership logic correct
- [ ] Performance < 50ms per evaluation

#### UnifiedGovernanceSteward
- [ ] Turn processing completes
- [ ] Drift detection triggers at threshold
- [ ] Intervention generation works
- [ ] Metrics accuracy confirmed

#### CounterfactualBranchManager
- [ ] Immutable fork points verified
- [ ] Branch generation completes
- [ ] ΔF calculation correct
- [ ] Threading safety confirmed
- [ ] Status transitions work

#### BranchComparator
- [ ] Statistical calculations correct
- [ ] Plotly charts render
- [ ] Metrics tables format properly
- [ ] Interpretation logic validated

---

## Automated Test Suites

### Unit Tests

**File**: `tests/unit/test_session_state.py`

```python
import pytest
from telos_purpose.core.session_state import SessionStateManager, TurnSnapshot

def test_turn_snapshot_immutability():
    """Verify TurnSnapshot is immutable."""
    snapshot = TurnSnapshot(
        turn_id=1,
        timestamp=1234567890.0,
        user_message="test",
        assistant_response="response",
        fidelity=0.85,
        drift_distance=0.3,
        in_basin=True,
        error_signal=0.15,
        intervention_triggered=False,
        metadata={}
    )

    # Should raise error (frozen dataclass)
    with pytest.raises(Exception):
        snapshot.fidelity = 0.5

def test_state_reconstruction():
    """Verify perfect state reconstruction."""
    manager = SessionStateManager()

    # Save 5 turns
    for i in range(5):
        manager.save_turn_snapshot({
            'turn_id': i,
            'timestamp': float(i),
            'user_message': f"msg_{i}",
            'assistant_response': f"resp_{i}",
            'fidelity': 0.8 + i * 0.01,
            'drift_distance': 0.3,
            'in_basin': True,
            'error_signal': 0.2 - i * 0.01,
            'intervention_triggered': False,
            'metadata': {}
        })

    # Reconstruct at turn 2
    state = manager.reconstruct_state_at_turn(2)

    assert state['turn_id'] == 2
    assert state['user_message'] == "msg_2"
    assert state['fidelity'] == 0.82

def test_audit_trail_export():
    """Verify audit trail is complete and valid JSON."""
    manager = SessionStateManager()

    # Add turns
    manager.save_turn_snapshot({...})

    # Export
    audit_trail = manager.export_audit_trail()

    assert 'turns' in audit_trail
    assert len(audit_trail['turns']) > 0
    assert all('turn_id' in t for t in audit_trail['turns'])
```

**File**: `tests/unit/test_primacy_attractor.py`

```python
import pytest
import numpy as np
from telos_purpose.core.primacy_attractor import PrimacyAttractor, GovernanceProfile

def test_fidelity_calculation():
    """Verify fidelity formula: F = 1 / (1 + d)."""
    profile = GovernanceProfile(
        purpose=["Test purpose"],
        scope=["Test scope"],
        boundaries=["Test boundary"]
    )

    # Mock embedding provider
    class MockEmbedding:
        def embed(self, text):
            return np.array([0.0, 0.0])

    attractor = PrimacyAttractor(profile, MockEmbedding())
    attractor.centroid = np.array([0.0, 0.0])

    # Test at known distances
    test_cases = [
        (np.array([0.0, 0.0]), 0.0, 1.0),  # d=0, F=1.0
        (np.array([1.0, 0.0]), 1.0, 0.5),  # d=1, F=0.5
        (np.array([3.0, 0.0]), 3.0, 0.25), # d=3, F=0.25
    ]

    for embedding, expected_d, expected_F in test_cases:
        d = attractor.calculate_drift(embedding)
        F = attractor.calculate_fidelity_from_distance(d)

        assert abs(d - expected_d) < 0.01
        assert abs(F - expected_F) < 0.01

def test_basin_membership():
    """Verify basin membership logic."""
    attractor = create_test_attractor()
    attractor.basin_radius = 1.0

    # Inside basin
    assert attractor.is_in_basin(0.5) == True
    assert attractor.is_in_basin(0.99) == True

    # Outside basin
    assert attractor.is_in_basin(1.1) == False
    assert attractor.is_in_basin(2.0) == False
```

### Integration Tests

**File**: `tests/integration/test_counterfactual_flow.py`

```python
import pytest
from telos_purpose.core.counterfactual_manager import CounterfactualBranchManager

def test_full_counterfactual_flow():
    """Test complete counterfactual generation flow."""
    # Setup components
    session_manager = create_test_session()
    branch_manager = CounterfactualBranchManager(session_manager, ...)

    # Create trigger
    trigger_turn = 5
    experiment_id = branch_manager.create_branch_experiment(trigger_turn)

    # Generate branches (may take 30-60s in real scenario)
    baseline = branch_manager.generate_baseline_branch(trigger_turn, length=5)
    telos = branch_manager.generate_telos_branch(trigger_turn, length=5)

    # Verify branches
    assert len(baseline.turns) == 5
    assert len(telos.turns) == 5
    assert baseline.fork_turn_id == trigger_turn
    assert telos.fork_turn_id == trigger_turn

    # Calculate ΔF
    delta_f = branch_manager.calculate_delta_f(baseline, telos)

    # Verify ΔF is reasonable (TELOS should improve)
    assert delta_f > 0  # TELOS better than baseline
    assert -1.0 <= delta_f <= 1.0  # Within valid range
```

---

## Statistical Validation Methods

### Fidelity Correlation Analysis

**Purpose**: Validate that Mathematical TELOS correlates with Heuristic TELOS

**Method**: Pearson correlation

```python
from scipy.stats import pearsonr

def validate_fidelity_correlation(messages: List[str],
                                 math_attractor,
                                 heuristic_attractor) -> Dict:
    """
    Validate that Mathematical and Heuristic TELOS agree.

    Expected: r > 0.85 (strong correlation)
    """
    math_fidelities = []
    heur_fidelities = []

    for msg in messages:
        math_F = math_attractor.calculate_fidelity(msg)
        heur_F = heuristic_attractor.calculate_fidelity(msg)

        math_fidelities.append(math_F)
        heur_fidelities.append(heur_F)

    # Pearson correlation
    r, p_value = pearsonr(math_fidelities, heur_fidelities)

    return {
        'correlation': r,
        'p_value': p_value,
        'is_significant': p_value < 0.05,
        'interpretation': interpret_correlation(r)
    }

def interpret_correlation(r: float) -> str:
    if r > 0.9:
        return "Very strong agreement"
    elif r > 0.7:
        return "Strong agreement"
    elif r > 0.5:
        return "Moderate agreement"
    else:
        return "Weak agreement (investigate)"
```

### ΔF Statistical Significance

**Purpose**: Validate that TELOS branch truly outperforms baseline

**Method**: Paired t-test

```python
from scipy.stats import ttest_rel

def validate_delta_f_significance(baseline_branch, telos_branch) -> Dict:
    """
    Test if TELOS significantly improves over baseline.

    H0: ΔF = 0 (no effect)
    H1: ΔF > 0 (TELOS improves outcomes)
    """
    baseline_fidelities = [t.fidelity for t in baseline_branch.turns]
    telos_fidelities = [t.fidelity for t in telos_branch.turns]

    # Paired t-test
    t_stat, p_value = ttest_rel(baseline_fidelities, telos_fidelities)

    # Effect size (Cohen's d)
    mean_diff = np.mean(telos_fidelities) - np.mean(baseline_fidelities)
    pooled_std = np.sqrt((np.std(baseline_fidelities)**2 +
                         np.std(telos_fidelities)**2) / 2)
    cohens_d = mean_diff / pooled_std

    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'is_significant': p_value < 0.05,
        'cohens_d': cohens_d,
        'effect_size': interpret_effect_size(cohens_d),
        'conclusion': generate_conclusion(p_value, cohens_d)
    }

def interpret_effect_size(d: float) -> str:
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"
```

### Consistency Analysis

**Purpose**: Validate that Mathematical TELOS is more consistent than Heuristic

**Method**: Variance comparison

```python
import numpy as np

def validate_consistency(message: str, attractor, n_trials: int = 10) -> Dict:
    """
    Test consistency by evaluating same message multiple times.

    Mathematical TELOS: Expected std < 0.01 (deterministic)
    Heuristic TELOS: Expected std > 0.05 (variable)
    """
    fidelities = []

    for _ in range(n_trials):
        F = attractor.calculate_fidelity(message)
        fidelities.append(F)

    return {
        'mean': np.mean(fidelities),
        'std': np.std(fidelities),
        'min': np.min(fidelities),
        'max': np.max(fidelities),
        'range': np.max(fidelities) - np.min(fidelities),
        'is_consistent': np.std(fidelities) < 0.02  # Threshold for consistency
    }
```

---

## Performance Benchmarking

### Latency Benchmarks

```python
import time
from typing import Dict, List

class LatencyBenchmark:
    """Benchmark performance of TELOS components."""

    def benchmark_component(self, component_name: str,
                           operation: callable,
                           n_trials: int = 100) -> Dict:
        """Benchmark a single component operation."""
        latencies = []

        for _ in range(n_trials):
            start = time.time()
            operation()
            latency = (time.time() - start) * 1000  # ms
            latencies.append(latency)

        return {
            'component': component_name,
            'n_trials': n_trials,
            'mean_ms': np.mean(latencies),
            'median_ms': np.median(latencies),
            'std_ms': np.std(latencies),
            'min_ms': np.min(latencies),
            'max_ms': np.max(latencies),
            'p95_ms': np.percentile(latencies, 95)
        }

# Example usage
benchmarks = {
    'PrimacyAttractor.calculate_fidelity': lambda: attractor.calculate_fidelity(msg),
    'UnifiedGovernanceSteward.process_turn': lambda: steward.process_turn(user, asst, 1),
    'SessionStateManager.save_snapshot': lambda: session.save_turn_snapshot(data),
}

for name, operation in benchmarks.items():
    result = benchmark_component(name, operation)
    print(f"{name}: {result['mean_ms']:.2f}ms (p95: {result['p95_ms']:.2f}ms)")
```

### Expected Benchmarks

| Component | Expected Latency | Acceptable Range |
|-----------|-----------------|------------------|
| PrimacyAttractor.calculate_fidelity | 30-50ms | < 100ms |
| UnifiedGovernanceSteward.process_turn | 50-100ms | < 200ms |
| SessionStateManager.save_snapshot | 1-5ms | < 10ms |
| CounterfactualBranchManager.generate_branch | 30-60s | < 120s |
| BranchComparator.compare_branches | 100-200ms | < 500ms |

---

## Validation Study Protocols

### Protocol 1: On-Topic vs Off-Topic

**Objective**: Validate drift detection accuracy

**Method**:
1. Create 50 test messages (25 on-topic, 25 off-topic)
2. Evaluate each with PrimacyAttractor
3. Classify: F > 0.8 → on-topic, F ≤ 0.8 → off-topic
4. Calculate accuracy, precision, recall

**Success Criteria**:
- Accuracy > 90%
- Precision > 85%
- Recall > 85%

### Protocol 2: Counterfactual Efficacy

**Objective**: Validate that TELOS improves outcomes

**Method**:
1. Run 20 full sessions with intentional drift
2. Generate counterfactuals for each trigger
3. Calculate ΔF for each experiment
4. Test significance with paired t-test

**Success Criteria**:
- Average ΔF > 0.1
- Significance: p < 0.05
- Effect size: d > 0.5

### Protocol 3: Parallel Architecture

**Objective**: Validate parallel processing speedup

**Method**:
1. Setup 1, 3, 5, 10 attractors
2. Evaluate 100 messages with each configuration
3. Measure total latency
4. Calculate speedup vs sequential

**Success Criteria**:
- 3 attractors: 2x speedup
- 5 attractors: 2.5x speedup
- 10 attractors: 3x speedup

---

## Cross-Reference

**TASKS.md**: Validation tasks tracked
**TELOS_BUILD_MANIFEST.md**: Main navigation
**Section 1**: Completed components being validated

---

## Summary

Comprehensive validation ensures:
- ✅ Components work correctly
- ✅ Statistical methods are sound
- ✅ Performance meets requirements
- ✅ Evidence generation is valid

**Status**: Manual testing complete, automated suite in progress

🧪 **Purpose: Ensure TELOS platform is production-ready and scientifically rigorous**
