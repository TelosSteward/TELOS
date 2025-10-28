# Progressive Primacy Attractor Guide

## Three Attractor Modes for User Customizability

TELOS now supports three modes for establishing primacy attractors, giving users flexibility in how governance is defined:

### 1. **Pre-defined Mode** (Prescribed Purpose)
- **What**: Load attractor from config.json
- **When to use**: Compliance scenarios, regulated industries, hard constraints
- **Example**: HIPAA-compliant medical chat, SEC-compliant financial advice
- **How**: Attractor fully specified in config before conversation starts

### 2. **Progressive Mode** (Emergent Purpose)
- **What**: Learn entire attractor from first N turns of conversation
- **When to use**: Exploratory conversations, research, discovery, natural dialogue
- **Example**: Your 4 test sessions where topic emerges naturally
- **How**: System observes first 5 turns, extracts topics/scope, creates attractor

### 3. **Hybrid Mode** (Constrained Emergence)
- **What**: Pre-defined boundaries + progressive scope learning
- **When to use**: Soft boundaries with topic flexibility
- **Example**: "Must be respectful and safe" but topic can emerge naturally
- **How**: Boundaries from config + learned purpose/scope from first 5 turns

---

## Architecture

### Progressive Mode Flow

```
Turn 1: User discusses Python
        └─> Accumulate in baseline

Turn 2: User discusses functions
        └─> Accumulate in baseline

Turn 3: User discusses lists
        └─> Accumulate in baseline

Turn 4: User discusses loops
        └─> Accumulate in baseline

Turn 5: User discusses classes
        └─> Accumulate in baseline
        └─> BASELINE ESTABLISHED
        └─> Extract terms: ["python", "functions", "lists", "loops", "classes"]
        └─> Create attractor:
            • Purpose: "Discussion about python, functions, lists"
            • Scope: ["python", "functions", "lists", "loops", "classes", ...]
            • Boundaries: ["Stay within established topic scope"]

Turn 6+: Measure fidelity against established attractor
```

### Hybrid Mode Flow

```
Config provides:
  • Boundaries: ["No medical advice", "No financial advice", "Be respectful"]
  • Privacy level: 0.8
  • Constraint tolerance: 0.2

Turn 1-5: Same accumulation as progressive mode

Turn 5: BASELINE ESTABLISHED
        └─> Extract learned terms: ["python", "functions", "lists", ...]
        └─> Create hybrid attractor:
            • Purpose (LEARNED): "Discussion about python, functions, lists"
            • Scope (LEARNED): ["python", "functions", "lists", ...]
            • Boundaries (PRE-DEFINED): ["No medical advice", ...]
            • Privacy (PRE-DEFINED): 0.8
            • Constraint tolerance (PRE-DEFINED): 0.2

Turn 6+: Measure fidelity against hybrid attractor
```

---

## Usage in Dashboard

### Step 1: Load Historical Conversation

Navigate to **Tab 1: Live Session** → **Load & Replay** mode

Upload conversation file (.txt, .json, or .md)

### Step 2: Select Attractor Mode

Choose one of three modes:

**Pre-defined**:
```
📋 Using governance profile from config.json
```

**Progressive**:
```
🌱 Attractor will be learned from first 5 turns of conversation
```

**Hybrid**:
```
🔀 Boundaries from config + scope learned from first 5 turns
```

### Step 3: Play Replay

Click **▶️ Play** to start turn-by-turn processing

**During Baseline (Turns 1-5)**:
```
🔄 Establishing progressive baseline... (3/5 turns)
Fidelity: Establishing...
```

**After Baseline (Turn 6+)**:
```
✅ Progressive primacy baseline established from turns 1-5
Fidelity: 0.892
⚠️ DRIFT DETECTED (F=0.742)  # If fidelity drops below 0.8
```

---

## Implementation Details

### File: `telos_purpose/profiling/progressive_primacy_extractor.py`

**Class**: `ProgressivePrimacyExtractor`

**Key Methods**:

```python
__init__(
    llm_client,
    embedding_provider,
    mode='progressive',                    # 'progressive' or 'hybrid'
    seed_attractor=None,                   # Required for hybrid mode
    # Statistical convergence parameters
    window_size=3,                         # Rolling window for stability check
    centroid_stability_threshold=0.95,     # Cosine similarity threshold
    variance_stability_threshold=0.1,      # Relative variance threshold
    confidence_threshold=0.80,             # Overall confidence to converge
    consecutive_stable_turns=2,            # Stability duration required
    max_turns_safety=100,                  # Safety limit to prevent infinite loops
    distance_scale=2.0                     # Distance-to-fidelity scaling
)

add_turn(user_message, assistant_response) -> Dict:
    # Process turn incrementally
    # Returns: {
    #   'baseline_established': bool,
    #   'turn_count': int,
    #   'fidelity': Optional[float],
    #   'drift_detected': Optional[bool],
    #   'status_message': str
    # }

is_ready() -> bool:
    # Check if baseline established

get_attractor() -> PrimacyAttractor:
    # Get established attractor (after baseline)
```

**Private Methods**:

```python
_establish_progressive_baseline():
    # Learn entire attractor from baseline turns
    # Extracts: purpose, scope, boundaries

_establish_hybrid_baseline():
    # Keep seed boundaries
    # Learn: purpose and scope from baseline turns

_extract_key_terms(text) -> List[str]:
    # Extract key terms using basic NLP
    # Removes stopwords, filters short words
```

---

## Testing

### Test Script: `test_progressive_extractor.py`

Run:
```bash
cd ~/Desktop/telos
source venv/bin/activate
export MISTRAL_API_KEY='your_key_here'
python test_progressive_extractor.py
```

**Output**:
```
============================================================
TESTING PROGRESSIVE MODE
============================================================

Processing turns (Progressive Mode):

Turn 1:
  User: What is Python?
  Status: 🔄 Establishing progressive baseline... (1/3 turns)

Turn 2:
  User: How do I define a function?
  Status: 🔄 Establishing progressive baseline... (2/3 turns)

Turn 3:
  User: What are lists?
  Status: ✅ Progressive primacy baseline established from turns 1-3
  Fidelity: 0.923

Turn 4:
  User: How do I use loops?
  Status: ✅ Aligned (F=0.889)
  Fidelity: 0.889

✅ Baseline established: True

Learned Attractor:
  Purpose: ['Discussion about python, function, lists']
  Scope: ['Topics including: python, function, lists, define, ...']
  Boundaries: ['Stay within established topic scope', ...]

============================================================
TESTING HYBRID MODE
============================================================

Seed Attractor:
  Boundaries: ['No medical advice', 'No financial advice']

Processing turns (Hybrid Mode):

Turn 1:
  User: What is Python?
  Status: 🔄 Establishing hybrid baseline... (1/3 turns)

...

Hybrid Attractor:
  Purpose (LEARNED): ['Discussion about python, function, lists']
  Scope (LEARNED): ['Topics including: python, function, lists, ...']
  Boundaries (PRE-DEFINED): ['No medical advice', 'No financial advice']
  Privacy (PRE-DEFINED): 0.8
  Constraint Tolerance (PRE-DEFINED): 0.2

============================================================
✅ ALL TESTS COMPLETE
============================================================
```

---

## Use Cases

### Pre-defined Mode

**Regulatory Compliance**:
```python
# config.json
{
  "governance_profile": {
    "purpose": ["Provide HIPAA-compliant medical information"],
    "scope": ["General health education", "Symptom information"],
    "boundaries": [
      "No diagnosis",
      "No treatment recommendations",
      "Refer to licensed medical professionals"
    ]
  },
  "attractor_parameters": {
    "constraint_tolerance": 0.1,  # STRICT
    "privacy_level": 0.95,
    "task_priority": 0.9
  }
}
```

### Progressive Mode

**Exploratory Research**:
```python
# Let conversation define its own topic - uses statistical convergence
extractor = ProgressivePrimacyExtractor(
    llm_client=llm,
    embedding_provider=embeddings,
    mode='progressive',
    window_size=8,
    centroid_stability_threshold=0.95,
    variance_stability_threshold=0.15,
    confidence_threshold=0.75,
    consecutive_stable_turns=3
)

# Process your test sessions
# Attractor emerges naturally when conversation stabilizes
```

### Hybrid Mode

**Soft Boundaries with Flexibility**:
```python
# config.json (seed attractor)
{
  "governance_profile": {
    "purpose": [],  # Will be learned
    "scope": [],    # Will be learned
    "boundaries": [
      "Be respectful and professional",
      "No harmful content",
      "Maintain privacy"
    ]
  },
  "attractor_parameters": {
    "constraint_tolerance": 0.3,  # MODERATE
    "privacy_level": 0.8,
    "task_priority": 0.7
  }
}

extractor = ProgressivePrimacyExtractor(
    llm_client=llm,
    embedding_provider=embeddings,
    mode='hybrid',
    seed_attractor=load_from_config(),
    window_size=8,
    centroid_stability_threshold=0.95,
    variance_stability_threshold=0.15,
    confidence_threshold=0.75,
    consecutive_stable_turns=3
)
```

---

## Configuration Comparison

| Parameter | Pre-defined | Progressive | Hybrid |
|-----------|-------------|-------------|--------|
| **Purpose** | Config | Learned | Learned |
| **Scope** | Config | Learned | Learned |
| **Boundaries** | Config | Learned | **Config** |
| **Privacy Level** | Config | Default (0.8) | **Config** |
| **Constraint Tolerance** | Config | Default (0.2) | **Config** |
| **Task Priority** | Config | Default (0.7) | **Config** |

**Bold** = Pre-defined/preserved

---

## Pristine State Isolation

All three modes maintain pristine turn-by-turn processing:

```python
# Turn N: Process with ONLY turns 1-(N-1) as context
# Turn N+1: Process with ONLY turns 1-N as context

# NO FUTURE KNOWLEDGE LEAKAGE
# Each turn measured with state that existed at that point in time
```

**Progressive/Hybrid Baseline**:
```
Turns 1-5: Accumulate WITHOUT measuring (baseline establishment)
Turn 6+: Measure against FIXED attractor established at turn 5
```

**Pre-defined**:
```
Turn 1+: Measure against pre-loaded attractor
```

---

## Dashboard Access

**TELOSCOPE Observatory** is running at:
```
http://localhost:8502
```

**To test**:
1. Go to Tab 1: **Live Session**
2. Select **Load & Replay** mode
3. Upload `/Users/brunnerjf/Desktop/telos/test_sessions/example_conversation.txt`
4. Choose attractor mode: **Pre-defined**, **Progressive**, or **Hybrid**
5. Click **▶️ Play**
6. Watch real-time processing with status updates

---

## Key Differences from Pre-defined Mode

### Pre-defined
- ✅ Immediate drift detection (turn 1+)
- ✅ Full intervention capability
- ✅ Consistent attractor across sessions
- ❌ Requires domain expertise to configure
- ❌ Inflexible for emergent conversations

### Progressive
- ✅ No configuration required
- ✅ Adapts to any conversation topic
- ✅ Natural for exploratory dialogue
- ⚠️ Delayed drift detection (turn 6+)
- ⚠️ Attractor varies per session

### Hybrid
- ✅ Best of both worlds
- ✅ Hard boundaries (compliance)
- ✅ Flexible scope (adaptability)
- ✅ Balances prescription and emergence
- ⚠️ Still requires boundary configuration

---

## Future Enhancements

1. **Adaptive Baseline Length**: Auto-determine optimal N based on topic stability
2. **Attractor Refinement**: Continue learning beyond initial N turns
3. **Multi-session Learning**: Aggregate attractors across multiple sessions
4. **Intervention Support**: Add intervention capability to progressive/hybrid modes
5. **Confidence Scoring**: Add confidence metrics for learned attractors
6. **LLM-based Extraction**: Use LLM to extract purpose/scope/boundaries (not just term frequency)

---

**Last Updated**: 2025-10-26
**Status**: ✅ Production-ready
**Integration**: Complete with TELOSCOPE Dashboard

🔭 **Making AI Governance Observable Through User-Customizable Attractors**
