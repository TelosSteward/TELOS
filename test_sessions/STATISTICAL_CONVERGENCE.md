# Statistical Convergence Implementation

## Overview

The ProgressivePrimacyExtractor has been completely rewritten to use **statistical convergence detection** instead of arbitrary turn limits. This provides a data-driven, mathematically rigorous approach to determining when a conversation has converged to a stable primacy attractor.

## Key Changes from Previous Implementation

### ❌ OLD APPROACH (Removed)
- **Arbitrary turn limits**: `min_turns=3`, `baseline_turns=5`
- **Fixed processing window**: Process exactly N turns, no more
- **No convergence detection**: Blind extraction after fixed turns
- **No adaptivity**: Same parameters for all conversations

### ✅ NEW APPROACH (Statistical)
- **NO arbitrary limits**: Processes as many turns as needed
- **Statistical convergence**: Detects when conversation stabilizes
- **Rolling window analysis**: Continuous monitoring of stability
- **Multi-metric confidence**: Combines multiple stability indicators
- **Data-driven parameters**: Derive optimal settings from real conversations

## Mathematical Foundation

### 1. Rolling Window Centroid Comparison

The system maintains a sliding window of embeddings and compares the centroid of the current window to the previous window.

**Centroid Stability (Cosine Similarity)**:
```
current_window = embeddings[-window_size:]
previous_window = embeddings[-2*window_size : -window_size]

current_centroid = mean(current_window)
previous_centroid = mean(previous_window)

centroid_stability = 1 - cosine_distance(current_centroid, previous_centroid)
```

**Threshold**: `centroid_stability >= 0.95` (95% similarity required)

### 2. Variance Stability

Low variance within the current window indicates the conversation has settled into a stable region of embedding space.

**Relative Variance**:
```
variances = var(current_window, axis=0)
mean_variance = mean(variances)
centroid_norm = ||current_centroid||

relative_variance = mean_variance / centroid_norm²
variance_stability = 1 - min(relative_variance / threshold, 1.0)
```

**Threshold**: `relative_variance <= 0.15` (low variance required)

### 3. Confidence Scoring

Weighted combination of multiple stability metrics:

```
confidence = (
    0.4 * centroid_stability +      # Primary signal
    0.3 * variance_stability +      # Secondary signal
    0.3 * data_confidence          # Sufficient data
)
```

**Threshold**: `confidence >= 0.75` (75% confidence required)

### 4. Consecutive Stable Turns

To avoid premature convergence, the system requires N consecutive turns to meet all stability thresholds.

**Default**: `consecutive_stable_turns = 3`

## Convergence Criteria

Convergence is declared when:

1. **Sufficient data**: `turn_count >= 2 * window_size` (need 2 windows to compare)
2. **Centroid stable**: `centroid_stability >= 0.95`
3. **Variance stable**: `relative_variance <= 0.15`
4. **High confidence**: `confidence >= 0.75`
5. **Sustained stability**: Stable for `consecutive_stable_turns` in a row

## Multi-Session Analysis

The `ConvergenceAnalyzer` class enables statistical analysis across multiple conversations to derive optimal parameters.

### Features

**Statistics Computed**:
- Mean, median, std of convergence turns
- 25th, 75th, 90th, 95th percentiles
- 95% confidence intervals
- Convergence rate (% of sessions that converged)
- Stability metrics (centroid and variance)

**Parameter Recommendations**:
- `window_size`: `median_convergence + 1 * std`
- `confidence_threshold`: `mean_confidence - 0.5 * std`
- `centroid_threshold`: `mean_centroid_stability * 1.1`
- `variance_threshold`: `mean_variance_stability * 1.1`

### Workflow

```python
from telos_purpose.profiling.convergence_analyzer import ConvergenceAnalyzer

# Create analyzer
analyzer = ConvergenceAnalyzer()

# Process multiple sessions
for session in sessions:
    extractor = ProgressivePrimacyExtractor(...)

    for turn in session.turns:
        extractor.add_turn(...)
        if extractor.converged:
            break

    # Get convergence record
    record = extractor.get_convergence_record(session_id)
    analyzer.add_record(record)

# Generate statistics
stats = analyzer.compute_statistics()
recommendations = analyzer.recommend_parameters()

# Export report
analyzer.export_report('convergence_report.json')
analyzer.print_summary()
```

## Usage Example

### Basic Usage

```python
from telos_purpose.profiling.progressive_primacy_extractor import ProgressivePrimacyExtractor
from telos_purpose.core.embedding_providers import get_embedding_function

# Initialize with statistical convergence
embedding_fn = get_embedding_function(provider='mistral')
extractor = ProgressivePrimacyExtractor(
    embedding_function=embedding_fn,
    llm_analyzer=your_llm,
    window_size=8,
    centroid_stability_threshold=0.95,
    variance_stability_threshold=0.15,
    confidence_threshold=0.75,
    consecutive_stable_turns=3,
)

# Process conversation progressively
for turn in conversation:
    extractor.add_turn(
        speaker=turn.speaker,
        message=turn.content
    )

    # Check convergence
    if extractor.converged:
        print(f"Converged at turn {extractor.convergence_turn}")
        break

# Get final attractor
status = extractor.get_status()
attractor = status.get('attractor')
```

### Advanced: Parameter Tuning

```python
# Run on multiple test conversations
analyzer = ConvergenceAnalyzer()

for session_file in test_sessions:
    session = SessionLoader().load(session_file)

    extractor = ProgressivePrimacyExtractor(
        embedding_function=embedding_fn,
        llm_analyzer=llm,
    )

    for turn in session.turns:
        extractor.add_turn(turn.speaker, turn.content)
        if extractor.converged:
            break

    record = extractor.get_convergence_record(session.id)
    analyzer.add_record(record)

# Get data-driven recommendations
recommendations = analyzer.recommend_parameters()

print(f"Recommended window_size: {recommendations['recommended_window_size']}")
print(f"Recommended confidence: {recommendations['recommended_confidence_threshold']}")

# Use these parameters for production
extractor = ProgressivePrimacyExtractor(
    embedding_function=embedding_fn,
    llm_analyzer=llm,
    window_size=recommendations['recommended_window_size'],
    confidence_threshold=recommendations['recommended_confidence_threshold'],
)
```

## Testing

Run the comprehensive test suite:

```bash
cd ~/Desktop/telos
source venv/bin/activate
python test_sessions/test_statistical_convergence.py
```

This will:
1. Test statistical convergence on real conversations
2. Generate convergence records
3. Perform multi-session analysis
4. Produce data-driven parameter recommendations
5. Export comprehensive JSON report

## Benefits for Grant Materials

### Scientific Rigor
- **Mathematical foundation**: Cosine similarity, variance analysis, confidence scoring
- **Statistical evidence**: 95% CIs, percentiles, distribution analysis
- **Data-driven**: Parameters derived from empirical data, not guesses
- **Reproducible**: Clear metrics and thresholds

### Adaptivity
- **No arbitrary limits**: System adapts to conversation complexity
- **Conversation-specific**: Short conversations converge quickly, complex ones take longer
- **Parameter tuning**: Optimize based on your specific use case

### Empirical Validation
- **Multi-session analysis**: Analyze convergence patterns across N conversations
- **Statistical reporting**: Generate comprehensive reports with evidence
- **Distribution visualization**: Show convergence turn distributions
- **Performance metrics**: Convergence rate, stability metrics, quality indicators

## Files Modified/Created

### Core Implementation
- `telos_purpose/profiling/progressive_primacy_extractor.py` (rewritten, 609 lines)
  - Statistical convergence detection
  - Rolling window analysis
  - Confidence scoring
  - Convergence record generation

### Analysis Framework
- `telos_purpose/profiling/convergence_analyzer.py` (new, 270 lines)
  - Multi-session analysis
  - Statistical computations
  - Parameter recommendations
  - Report generation

### Testing
- `test_sessions/test_statistical_convergence.py` (new, 280 lines)
  - Comprehensive test suite
  - Multiple conversation analysis
  - Parameter recommendation demo

## Parameters Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `window_size` | 8 | Number of turns in rolling window |
| `centroid_stability_threshold` | 0.95 | Required cosine similarity between windows |
| `variance_stability_threshold` | 0.15 | Maximum allowed relative variance |
| `confidence_threshold` | 0.75 | Minimum confidence score to converge |
| `consecutive_stable_turns` | 3 | Required consecutive stable turns |
| `safety_limit` | 500 | Maximum turns to prevent infinite loops |

## Future Enhancements

- **Adaptive window sizing**: Automatically adjust window based on conversation dynamics
- **Early stopping**: Detect divergent conversations that won't converge
- **Quality prediction**: Predict attractor quality before full convergence
- **Visualization tools**: Real-time convergence plots, embedding space visualization
- **Benchmark suite**: Standard test conversations for parameter validation

---

**Questions or issues?** See the test script for working examples or consult the inline documentation in the source files.
