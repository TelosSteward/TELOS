# ARCHITECTURE DIAGNOSIS: Active Mitigation vs Passive Observation

## Executive Summary

**FINDING**: TELOS currently operates in a **HYBRID MODE** with elements of both architectures:

| Component | Mode | Evidence |
|-----------|------|----------|
| Learning Phase | ✅ ACTIVE | LLM-based semantic analysis |
| Attractor Content | ✅ TEXTUAL | Purpose/scope/boundaries from LLM |
| Operational Flow | ❌ PASSIVE | Post-hoc analysis of pre-generated responses |
| Salience Maintenance | ❌ NOT IMPLEMENTED | No context injection |
| Counterfactual Generation | 🟡 REACTIVE | Generates AFTER drift detected, not BEFORE |

**CRITICAL GAP**: The steward receives responses that are **already generated**, making it a **post-hoc analyzer** rather than a **proactive interceptor**.

---

## Q1: OPERATIONAL MODE

### Current Implementation: ❌ PASSIVE OBSERVATION (Post-hoc)

**Evidence from `unified_steward.py:308`**:
```python
def process_turn(self, user_input: str, model_response: str) -> Dict[str, Any]:
    """Process single turn through MBL..."""
```

**Analysis**:
- Method signature accepts `model_response` as **INPUT**
- Response has **ALREADY BEEN GENERATED** by the time steward sees it
- Steward can only measure and potentially regenerate AFTER the fact

**What it SHOULD be** (Active Mitigation):
```python
def process_turn(self, user_input: str) -> str:
    """Intercept user input, ensure salience, generate governed response."""
    # 1. Check if attractor salience is degrading
    # 2. Inject attractor into context if needed
    # 3. Call LLM with reinforced context
    # 4. Check response coupling
    # 5. Regenerate if decoupled
    # 6. Return governed response
```

**Current Flow**:
```
User Input → ??? (who generates?) → Response → Steward → Measure → Maybe Regenerate
```

**Desired Flow**:
```
User Input → Steward Intercepts → Salience Check → LLM Call → Coupling Check → Governed Response
```

---

## Q2: LEARNING PHASE (Turns 1-N)

### Current Implementation: ✅ LLM-BASED (Semantic Analysis)

**Evidence from `progressive_primacy_extractor.py:357-412`**:
```python
def _analyze_with_llm(self) -> Dict[str, Any]:
    """Use LLM to analyze accumulated turns."""

    system_prompt = """You are a conversation analyzer. Extract the primary
    purpose, key topics, and implicit boundaries..."""

    response = self.llm_client.generate(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3,
        max_tokens=500
    )
```

**Analysis**: ✅ CORRECT
- Uses LLM API calls for semantic analysis
- Prompts extract purpose, scope, boundaries
- Creates textual attractor (not just math)
- Invoked when statistical convergence detected (lines 196, 229)

This part is **correctly implemented** as active LLM-based learning.

---

## Q3: ATTRACTOR CONTENT

### Current Implementation: ✅ TEXTUAL (Purpose/Scope/Boundaries)

**Evidence from `progressive_primacy_extractor.py:235-250`**:
```python
self.primacy_attractor = PrimacyAttractor(
    purpose=llm_analysis.get('purpose', []),
    scope=llm_analysis.get('scope', []),
    boundaries=llm_analysis.get('boundary', [])
)
```

**Analysis**: ✅ CORRECT
- Attractor contains textual elements from LLM analysis
- Not just a mathematical centroid
- Can be used for governance prompts

However, there's also a **mathematical centroid** computed:
```python
self.attractor_centroid = np.mean(self.accumulated_embeddings, axis=0)
```

**Dual Nature**:
- **Textual**: For LLM prompts during intervention
- **Mathematical**: For distance/fidelity calculation

This is **correct** - you need both.

---

## Q4: SALIENCE MAINTENANCE

### Current Implementation: ❌ NOT IMPLEMENTED

**Evidence**: No code found for context injection or salience tracking.

**What's Missing**:
```python
def _check_salience_degradation(self, context: List[Dict]) -> bool:
    """Check if attractor is losing prominence in context."""
    # Compute embedding of current context
    context_embedding = embed(context)

    # Compare with attractor
    salience = similarity(context_embedding, attractor_centroid)

    # Return True if attractor fading
    return salience < threshold

def _inject_attractor_reinforcement(self, context: List[Dict]) -> List[Dict]:
    """Inject attractor back into context to maintain salience."""
    attractor_text = format_attractor_as_reminder(self.attractor)

    return context + [{"role": "system", "content": attractor_text}]
```

**Impact**: Even if attractor exists, it's not being actively maintained in the conversation context, allowing salience degradation over time.

---

## Q5: DECOUPLING RESPONSE

### Current Implementation: 🟡 REACTIVE REGENERATION (Post-hoc only)

**Evidence from `proportional_controller.py:227-256`**:
```python
def _apply_regeneration(
    self,
    original_response: str,  # <-- Response already generated
    conversation_history: List[Dict[str, str]],
    error_signal: float
) -> InterventionRecord:
    """Apply State 3 (INTERVENE) intervention: Regeneration."""

    corrective = {
        "role": "system",
        "content": (
            "The previous answer drifted from the session purpose/scope. "
            "Regenerate a response that stays strictly on-purpose..."
        )
    }

    messages = conversation_history.copy()
    messages.append(corrective)

    # Request regeneration from LLM
    regenerated = self.llm_client.generate(messages, ...)
```

**Analysis**: 🟡 PARTIALLY CORRECT
- ✅ DOES regenerate using LLM
- ✅ DOES include corrective guidance
- ❌ BUT only triggers AFTER response already generated
- ❌ Original drifted response was already created (wasted API call)

**Ideal Flow**: Generate WITH governance from the start, not regenerate after drift.

---

## Q6: COUNTERFACTUAL GENERATION

### Current Implementation: ✅ IMPLEMENTED (Real LLM API calls)

**Evidence from `counterfactual_branch_manager.py:265-303`**:
```python
def _generate_telos_branch(self, ...):
    """Generate TELOS branch using REAL API calls."""

    for i, (user_input, _historical_response) in enumerate(remaining_turns):
        # Add user input
        current_history.append({"role": "user", "content": user_input})

        # Generate NEW response via API (counterfactual)
        raw_response = self.llm.generate(
            messages=current_history,
            max_tokens=500,
            temperature=0.7
        )

        # Apply intervention on first turn
        if i == 0:
            corrected_response = self._apply_telos_intervention(...)
```

**Analysis**: ✅ CORRECT
- Real API calls generate counterfactual responses
- Applies intervention via governance prompts
- Uses same user inputs for fair comparison
- Calculates real metrics with actual embeddings

**However**: Counterfactuals are **RETROSPECTIVE**
- Triggered AFTER drift already happened
- Not preventing original drift
- Showing "what would have happened" not "what WILL happen"

---

## FLOW DIAGRAM: Current vs Desired

### Current Architecture (HYBRID - Passive + Reactive)
```
USER INPUT
    ↓
??? (Who calls LLM first time?)  <-- Gap: Pre-governance LLM call
    ↓
ORIGINAL RESPONSE GENERATED
    ↓
Steward.process_turn(user_input, model_response)  <-- Receives pre-generated response
    ↓
Measure Fidelity
    ↓
IF F < threshold → Regenerate  <-- Reactive
    ↓
Return (possibly modified) response
```

### Desired Architecture (ACTIVE MITIGATION)
```
USER INPUT
    ↓
Steward.intercept(user_input)
    ↓
Check Salience (is attractor prominent in context?)
    ↓
IF degrading → Inject attractor reinforcement
    ↓
Generate with reinforced context
    ↓
Check Coupling (does response stay coupled?)
    ↓
IF F < threshold → Regenerate with entrainment
    ↓
Return governed response
```

---

## GAP ANALYSIS

### Critical Missing Pieces

#### 1. **Pre-emptive Interception** 🔴 HIGH PRIORITY
**What**: Steward must receive user input BEFORE any LLM call
**Why**: Can't prevent drift if response already generated
**Implementation**:
```python
class InterceptingLLMWrapper:
    """Wraps LLM client to intercept all calls."""

    def __init__(self, base_llm, steward):
        self.llm = base_llm
        self.steward = steward

    def generate(self, messages, **kwargs):
        # Steward checks/modifies context BEFORE generation
        governed_messages = self.steward.prepare_context(messages)

        # Generate with governed context
        response = self.llm.generate(governed_messages, **kwargs)

        # Steward checks response AFTER generation
        governed_response = self.steward.check_response(response, governed_messages)

        return governed_response
```

#### 2. **Salience Monitoring** 🟡 MEDIUM PRIORITY
**What**: Track if attractor fading from context over turns
**Why**: Prevents "topic drift creep" where focus shifts gradually
**Implementation**:
```python
def track_salience(self, context_history):
    """Monitor attractor prominence in context window."""
    # Embed current context
    context_emb = self.embed_context(context_history[-5:])  # Last 5 turns

    # Compare with attractor
    salience_score = cosine_sim(context_emb, self.attractor_centroid)

    # Log salience trend
    self.salience_history.append(salience_score)

    # Trigger injection if degrading
    if salience_score < self.salience_threshold:
        return self._create_reinforcement_message()
```

#### 3. **Context Injection** 🟡 MEDIUM PRIORITY
**What**: Inject attractor into context when salience degrades
**Why**: Maintains "purpose prominence" without regenerating
**Implementation**:
```python
def inject_attractor(self, messages):
    """Add attractor reinforcement to context."""
    reinforcement = {
        "role": "system",
        "content": f"""Session Purpose: {self.attractor.purpose}

Keep responses focused on: {', '.join(self.attractor.scope)}
Stay within boundaries: {', '.join(self.attractor.boundaries)}"""
    }

    # Insert before current user message
    return messages[:-1] + [reinforcement] + [messages[-1]]
```

#### 4. **Live Session Integration** 🔴 HIGH PRIORITY
**What**: Wire steward into actual live conversations
**Why**: Currently only works on pre-recorded replays
**Implementation**: Dashboard "Live Chat" mode needs to:
```python
# CURRENT (doesn't use steward):
user_input = st.chat_input()
response = llm.generate([{"role": "user", "content": user_input}])
st.write(response)

# SHOULD BE:
user_input = st.chat_input()
response = steward.process_live_turn(user_input)  # Steward calls LLM internally
st.write(response)
```

---

## SUMMARY

### What TELOS Currently Does

**✅ CORRECT**:
1. Learning Phase uses LLM-based semantic analysis
2. Creates textual attractor (purpose/scope/boundaries)
3. Can regenerate responses using LLM when drift detected
4. Generates real counterfactual branches via API

**❌ INCORRECT**:
1. Operates post-hoc (receives pre-generated responses)
2. No salience monitoring
3. No context injection
4. Reactive, not proactive
5. Only works on replays, not live sessions

### What TELOS SHOULD Do (Mitigation Layer Model)

**Active Governance Loop**:
```
Turn N:
1. Intercept user input
2. Check salience of attractor in context
3. Inject reinforcement if degrading
4. Generate response with governed context
5. Check coupling to attractor
6. Regenerate if decoupled
7. Return governed response
8. Update salience history
```

---

## ARCHITECTURAL RECOMMENDATIONS

### Phase 1: Foundation (Enable Active Interception) 🔴

**Priority**: CRITICAL
**Effort**: Medium
**Impact**: Transforms from observer to governor

**Tasks**:
1. Create `InterceptingLLMWrapper` that steward controls
2. Modify `UnifiedGovernanceSteward.process_turn()` signature:
   - From: `(user_input, model_response)` ← passive
   - To: `(user_input) → governed_response` ← active
3. Steward calls LLM internally, not externally
4. Dashboard "Live Chat" routes through steward

**Deliverable**: Steward controls LLM calls, not vice versa

---

### Phase 2: Salience Maintenance (Prevent Drift) 🟡

**Priority**: HIGH
**Effort**: Medium
**Impact**: Reduces drift frequency significantly

**Tasks**:
1. Implement `track_salience()` method
2. Implement `inject_attractor_reinforcement()` method
3. Add salience degradation to metrics
4. Log salience trajectory per session

**Deliverable**: Attractor stays prominent in context, preventing creep

---

### Phase 3: Live Demonstrations (Show It Working) 🟡

**Priority**: HIGH (for grants/papers)
**Effort**: Low (just wiring)
**Impact**: Actual demos of governance in action

**Tasks**:
1. Wire dashboard "Live Chat" through steward
2. Show metrics in real-time during conversation
3. Display when salience injection occurs
4. Display when regeneration occurs
5. Export evidence from live sessions

**Deliverable**: Live demos showing TELOS actively governing

---

### Phase 4: Optimization (Fine-tune) 🟢

**Priority**: MEDIUM
**Effort**: Low-Medium
**Impact**: Better performance

**Tasks**:
1. Tune salience threshold empirically
2. Optimize injection message format
3. A/B test different intervention strengths
4. Minimize API calls via better prediction

**Deliverable**: Optimized governance with minimal overhead

---

## CONCLUSION

**TELOS is architecturally positioned between two paradigms**:

- **Learning**: Active (LLM-based semantic analysis) ✅
- **Governance**: Passive (post-hoc measurement) ❌

To become a true **Mitigation Layer**, TELOS needs:

1. **Control the LLM**, don't just analyze its output
2. **Maintain salience** proactively, don't just detect decay
3. **Prevent drift** before it happens, not just correct after
4. **Govern live sessions**, not just analyze replays

The components exist (LLM client, attractor, metrics, regeneration), but they're wired in the wrong order.

**Analogy**:
- Current: Fire department analyzes burned buildings
- Desired: Fire prevention system stops fires from starting

The counterfactual system I built shows what WOULD have happened. Now we need to make that happen in REAL TIME during live conversations.

---

## GRANT APPLICATION LANGUAGE

### Current State (Be Honest):
> "TELOS detects when AI responses drift from intended purpose by measuring embedding-space fidelity. When drift is detected, TELOS can generate counterfactual responses showing what would have happened with governance intervention. This provides measurable evidence of alignment degradation in historical conversation logs."

### After Phase 1-3 (Active Mitigation):
> "TELOS operates as a real-time mitigation layer between users and LLMs. It maintains the salience of the conversation's purpose through context injection, preventing topic drift before it occurs. When responses begin to decouple from the established trajectory, TELOS regenerates them using governance-informed prompts. The system provides both preventive governance (salience maintenance) and corrective governance (regeneration), with all interventions logged and exportable as evidence. Live demonstrations show TELOS maintaining alignment over extended multi-turn conversations where ungovern ed systems drift significantly."

The second version is what we need to build.
