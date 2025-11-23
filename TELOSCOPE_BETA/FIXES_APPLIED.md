# BETA Governance Fixes Applied
**Date**: 2025-11-21
**Status**: Testing Required

---

## Summary

BETA mode governance was completely non-functional. The system accepted off-topic requests (PB&J sandwiches when PA was "AI governance"), showed high fidelity scores (0.850+) for unrelated content, and displayed "Calibrating" status despite pre-established PA.

---

## Root Causes & Fixes

### 1. Import Error in Demo Data (CRITICAL)
**File**: `utils/telos_demo_data.py:25`
**Problem**: Wrong class name caused TELOS engine to be completely disabled

**Before**:
```python
from telos_purpose.core.embedding_provider import SentenceTransformerEmbeddingProvider
```

**After**:
```python
from telos_purpose.core.embedding_provider import SentenceTransformerProvider
```

---

### 2. BetaResponseManager - Missing PA Parameter (CRITICAL)
**File**: `services/beta_response_manager.py:281-339`
**Problem**: `UnifiedGovernanceSteward` initialized WITHOUT passing `attractor` parameter

**Before**:
```python
def _initialize_telos_engine(self):
    embedding_provider = SentenceTransformerProvider()
    self.telos_engine = UnifiedGovernanceSteward(embedding_provider)  # WRONG!
```

**After**:
```python
def _initialize_telos_engine(self):
    # Read PA from session state
    pa_data = st.session_state.get('primacy_attractor', None)
    pa_established = st.session_state.get('pa_established', False)

    if pa_data and pa_established:
        # Convert strings to List[str] as required by PrimacyAttractor
        purpose_str = pa_data.get('purpose', 'General assistance')
        scope_str = pa_data.get('scope', 'Open discussion')

        attractor = PrimacyAttractor(
            purpose=[purpose_str] if isinstance(purpose_str, str) else purpose_str,
            scope=[scope_str] if isinstance(scope_str, str) else scope_str,
            boundaries=pa_data.get('boundaries', [...])
        )
    else:
        # Fallback PA
        attractor = PrimacyAttractor(...)

    llm_client = MistralClient()
    embedding_provider = SentenceTransformerProvider()

    self.telos_engine = UnifiedGovernanceSteward(
        attractor=attractor,  # ✅ Now passes correct PA
        llm_client=llm_client,
        embedding_provider=embedding_provider,
        enable_interventions=True
    )
```

---

### 3. Using Replay Mode Instead of Active Governance (CRITICAL)
**File**: `services/beta_response_manager.py:93-162`
**Problem**: Code called `process_turn()` (replay-only) instead of `generate_governed_response()` (active governance)

**Before**:
```python
def _generate_telos_response(self, user_input: str, turn_number: int) -> Dict:
    result = self.telos_engine.process_turn(
        user_input=user_input,
        primacy_attractor=pa  # WRONG METHOD - replay only!
    )
```

**After**:
```python
def _generate_telos_response(self, user_input: str, turn_number: int) -> Dict:
    conversation_history = self._get_conversation_history()

    # Generate governed response (ACTIVE MODE - prevents drift BEFORE generation)
    result = self.telos_engine.generate_governed_response(
        user_input=user_input,
        conversation_context=conversation_history
    )

    # Extract metrics
    telos_data = {
        'response': result.get('response', ''),
        'fidelity_score': result.get('telic_fidelity', 0.0),
        'distance_from_pa': result.get('error_signal', 0.0),
        'intervention_triggered': result.get('intervention_applied', False),
        'drift_detected': result.get('telic_fidelity', 1.0) < 0.7,
        ...
    }
```

---

### 4. PA Not Deleted After Questionnaire
**File**: `components/pa_onboarding.py:242-249`
**Problem**: TELOS engine initialized ONCE with fallback PA, never updated when real PA established

**Fix Applied**:
```python
# Force TELOS steward to re-initialize with new PA
if hasattr(st.session_state.state_manager, '_telos_steward'):
    delattr(st.session_state.state_manager, '_telos_steward')

# ALSO delete BETA response manager's telos_engine
if 'beta_response_manager' in st.session_state:
    if hasattr(st.session_state.beta_response_manager, 'telos_engine'):
        st.session_state.beta_response_manager.telos_engine = None
```

---

### 5. pa_converged Flag Not Set
**File**: `components/pa_onboarding.py:240`
**Problem**: UI checks `state.pa_converged` to show status, but flag never set to `True`

**Fix Applied**:
```python
st.session_state.state_manager.state.pa_converged = True  # Mark as converged
```

---

### 6. Token Limits Too Low
**Files**:
- `telos_purpose/llm_clients/mistral_client.py`
- `services/steward_llm.py`

**Problem**: max_tokens=500 and max_tokens=2048 too restrictive

**Fix Applied**: Increased to 16000 in both files

---

## Architecture Understanding

### BETA Mode Response Generation Flow

1. **Entry Point**: `BetaResponseManager.generate_turn_responses()`
2. **Both Responses Generated**:
   - `_generate_telos_response()` → Active governance via `generate_governed_response()`
   - `_generate_native_response()` → Direct LLM call without governance
3. **Random Display**: A/B test sequence determines which response user sees
4. **User Feedback**: Thumbs up/down/shrug rating collected
5. **Observatory Access**: At end of BETA, user can review ALL turns with both responses + metrics

### Why This Matters

- **Control Condition**: Native response + real fidelity score showing drift
- **Governed Condition**: TELOS response + real fidelity score showing alignment
- **Both must calculate accurate fidelity** so Observatory can show true comparison
- **Fidelity always real** - never fake/arbitrary, even in Control mode

---

## Expected Behavior After Fixes

1. PA questionnaire completes → PA stored → engines deleted
2. First message → engines re-created with **correct PA** → embeddings from real purpose/scope
3. Off-topic message (PB&J) → large distance from PA → **low fidelity (< 0.3)**
4. TELOS response: Redirects/warns about drift
5. Native response: Answers PB&J question
6. Fidelity accurate for **both** responses
7. UI shows "**Primacy Attractor Status: Established**" from turn 1

---

## Testing Status

### Prerequisites
- Complete DEMO mode OR temporarily unlock BETA tab
- Start fresh session (clear browser cache)
- Complete PA questionnaire with specific focus (e.g., "AI governance at runtime project called TELOS")

### Test Procedure
1. Send off-topic message: "I would like to know the best methods for making a peanut butter and jelly sandwich."
2. **Expected Results**:
   - **If TELOS response shown**: Fidelity < 0.3, response redirects back to TELOS topics
   - **If Native response shown**: Fidelity < 0.3 (measured), response gives PB&J instructions
3. **Verify**: PA Status shows "Established" NOT "Calibrating"
4. **Check Observatory** (after 15 turns): Both responses generated, accurate fidelity for each

---

## Files Modified

1. `utils/telos_demo_data.py` - Fixed import
2. `services/beta_response_manager.py` - PA initialization + active governance
3. `components/pa_onboarding.py` - pa_converged flag + engine deletion
4. `telos_purpose/llm_clients/mistral_client.py` - Token limit increase
5. `services/steward_llm.py` - Token limit increase

---

## Next Steps

1. **Manual Testing**: Verify governance works with off-topic requests
2. **Observatory Verification**: Check that both responses + metrics stored correctly
3. **A/B Test Validation**: Confirm random response selection working
4. **Fidelity Accuracy**: Verify scores make sense for on-topic vs off-topic
