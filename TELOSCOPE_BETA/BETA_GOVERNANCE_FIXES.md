# BETA Mode Governance Fixes

## Problem Summary

BETA mode PA governance was completely non-functional. The system would:
- Accept off-topic requests (e.g., PB&J sandwiches when PA was "AI governance")
- Show high fidelity scores (0.85+) for completely unrelated content
- Display "Calibrating (X/~10)" status even though PA was pre-established
- Not enforce the PA established via questionnaire

## Root Causes Identified

### 1. **PA Type Mismatch**
- **Location**: `core/state_manager.py:1099-1104`
- **Issue**: PA questionnaire stores `purpose` and `scope` as single strings, but `PrimacyAttractor` dataclass expects `List[str]`
- **Impact**: When creating attractor from questionnaire data, type mismatch could cause errors or unexpected behavior

### 2. **PA Not Loaded from Session State**
- **Location**: `core/state_manager.py:1086-1111`
- **Issue**: System was using generic fallback PA instead of reading `st.session_state.primacy_attractor`
- **Impact**: Established PA from questionnaire was completely ignored

### 3. **pa_converged Flag Not Set**
- **Location**: `components/pa_onboarding.py:240`
- **Issue**: UI checks `state.pa_converged` to decide whether to show "Calibrating" or metrics, but this flag was never set to `True` after questionnaire
- **Impact**: UI always showed "Calibrating (X/~10)" instead of actual fidelity metrics

### 4. **TELOS Steward Never Re-initialized** ⭐ **CRITICAL**
- **Location**: `components/pa_onboarding.py:243-244`
- **Issue**: `UnifiedGovernanceSteward` object created ONCE on first message with whatever PA existed at that time (usually fallback). When real PA was established via questionnaire, steward was never re-created, so it continued using old/wrong PA embeddings.
- **Impact**: **Governance completely broken** - fidelity calculations used wrong PA vector, making all drift detection meaningless

## Fixes Applied

### Fix 1: PA Type Conversion
**File**: `core/state_manager.py`
**Lines**: 1099-1104

```python
# Convert strings to lists as PrimacyAttractor expects List[str]
purpose_str = pa_data.get('purpose', 'General assistance')
scope_str = pa_data.get('scope', 'Open discussion')

attractor = PrimacyAttractor(
    purpose=[purpose_str] if isinstance(purpose_str, str) else purpose_str,
    scope=[scope_str] if isinstance(scope_str, str) else scope_str,
    boundaries=pa_data.get('boundaries', [...])
)
```

### Fix 2: PA Loading from Session
**File**: `core/state_manager.py`
**Lines**: 1086-1111

```python
# Beta/Open mode: Use established PA from session state
pa_data = st.session_state.get('primacy_attractor', None)
pa_established = st.session_state.get('pa_established', False)

if pa_data and pa_established:
    # Use the PA established during onboarding
    attractor = PrimacyAttractor(...)
    logger.info(f"✅ Using established PA - Purpose: {purpose_str[:80]}")
else:
    # Fallback (should rarely happen)
    attractor = PrimacyAttractor(...)
    logger.warning("⚠️ No established PA found - using generic fallback")
```

### Fix 3: Set pa_converged Flag
**File**: `components/pa_onboarding.py`
**Line**: 240

```python
# Also store for state manager
if 'state_manager' in st.session_state:
    st.session_state.state_manager.state.primacy_attractor = pa
    st.session_state.state_manager.state.user_pa_established = True
    st.session_state.state_manager.state.convergence_turn = 2
    st.session_state.state_manager.state.pa_converged = True  # ← NEW
```

### Fix 4: Force Steward Re-initialization
**File**: `components/pa_onboarding.py`
**Lines**: 243-244

```python
# Force TELOS steward to re-initialize with new PA
if hasattr(st.session_state.state_manager, '_telos_steward'):
    delattr(st.session_state.state_manager, '_telos_steward')
```

This ensures that on the next message after PA questionnaire completes, the TELOS steward will be re-created with the correct PA, causing fresh embedding calculations from the proper purpose/scope text.

## Technical Details

### How PA Embeddings Work

1. `PrimacyAttractor` object contains textual lists: `purpose: List[str]`, `scope: List[str]`
2. When `UnifiedGovernanceSteward` initializes, it calls `_initialize_spc_engine()` which:
   ```python
   purpose_text = " ".join(self.attractor_config.purpose)
   scope_text = " ".join(self.attractor_config.scope)

   p_vec = self.embedding_provider.encode(purpose_text)
   s_vec = self.embedding_provider.encode(scope_text)

   self.attractor_math = PrimacyAttractorMath(
       purpose_vector=p_vec,
       scope_vector=s_vec,
       ...
   )
   ```
3. The `PrimacyAttractorMath` object computes the attractor center (BoM center) from these vectors
4. **Fidelity is calculated** as `1 - (distance / basin_radius)` where distance is L2 norm between response embedding and attractor center

### Why Re-initialization Was Critical

The steward was initialized ONCE before the PA questionnaire even ran, using the fallback PA:
- Fallback scope: `["Any subject the user wishes to discuss"]`
- This got embedded and used for ALL fidelity calculations
- When real PA was stored in session state, the already-initialized steward kept using the old embedding
- Result: PB&J sandwiches got high fidelity scores because they were being compared to "any subject" embedding, not "AI governance" embedding

## Expected Behavior After Fixes

1. PA questionnaire completes → PA stored in session → steward deleted
2. User sends first message → steward re-initialized with correct PA → embeddings computed from real purpose/scope
3. Off-topic message (PB&J) → large embedding distance from PA → low fidelity (< 0.5)
4. System triggers interventions (reminders, regeneration, or redirection)
5. UI shows "Primacy Attractor Status: Established" from turn 1

## Testing

### Automated Testing (Blocked)

Playwright automation attempted but failed due to:
- BETA tab is locked behind DEMO mode completion
- `st.session_state.demo_completed` flag required
- No client-side method to unlock BETA (localStorage doesn't work with Streamlit server-side sessions)

Automated test scripts created but cannot run without either:
1. Completing DEMO mode first (defeats purpose of automated testing)
2. Temporarily removing BETA lock in code (requires code modification)

**Files Created**:
- `test_beta_governance.py` - Full governance test (blocked by BETA lock)
- `test_beta_simple.py` - Simple BETA state checker (blocked by BETA lock)

### Manual Testing (Recommended)

See `MANUAL_GOVERNANCE_TEST.md` for complete step-by-step instructions.

**Quick Test**:
1. Navigate to BETA tab (unlock via DEMO completion or temporary code change)
2. Complete PA questionnaire with specific purpose (e.g., "AI governance at runtime project called TELOS")
3. Send off-topic message: "I really would like to know the best way to make a Peanut Butter and Jelly Sandwich."
4. Verify:
   - PA Status shows "Established" (not "Calibrating")
   - Fidelity score is LOW (< 0.5)
   - Response redirects or warns about drift (NO PB&J instructions)

## Files Modified

1. `core/state_manager.py` - PA loading and type conversion
2. `components/pa_onboarding.py` - Set pa_converged flag and force steward reset
3. `telos_purpose/llm_clients/mistral_client.py` - Increased max_tokens from 500 to 16000
4. `services/steward_llm.py` - Increased max_tokens from 2048 to 16000

## Additional Fixes (Not Governance-Related)

- Removed token restrictions: Changed max_tokens to 16000 in all LLM calls
- Updated BETA welcome message: Clarified PA viewing, Steward usage, and mode differences
- Added file upload functionality: Created `services/file_handler.py` for PDF/DOCX/image support
