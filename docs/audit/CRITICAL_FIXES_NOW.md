# CRITICAL FIXES - IMMEDIATE IMPLEMENTATION (NOW)

**Purpose:** Production-blocking issues that MUST be fixed before institutional deployment
**Timeline:** Implement immediately (before grant applications and beta launch)
**Status:** Ready for systematic implementation

---

## IMPLEMENTATION APPROACH

**One Issue = One Commit**
- Commit format: `[OPUS-CRITICAL-N] Description`
- Test each fix independently
- Granular rollback capability
- Track progress via git commits

**Testing Protocol**
- Add edge case tests for each fix
- Verify no regressions in validation suite
- Ensure Observatory v3 remains functional
- Document any behavioral changes

---

## CRITICAL ISSUE #1: Zero Vector Division in Fidelity Calculations

### Files to Fix:
- `telos_purpose/core/primacy_math.py`
  - Function: `calculate_fidelity()`
  - Function: `cosine_similarity()`
- `telos_purpose/core/dual_attractor.py`
  - Any fidelity calculation calls

### Problem Statement:
Division by zero when embedding or PA has zero norm. No graceful handling.

### Impact if Not Fixed:
- **Session crashes** when processing zero vectors
- **NaN propagation** through calculation chain
- **No recovery** mechanism

### Fix Implementation:

**Step 1:** Update `primacy_math.py`

```python
def calculate_fidelity(embedding: np.ndarray, pa: np.ndarray) -> float:
    """Calculate cosine similarity between embedding and PA.

    Handles zero vectors gracefully by returning 0.0 (maximum deviation).

    Args:
        embedding: Response embedding vector
        pa: Primacy attractor vector

    Returns:
        float: Fidelity score in [-1, 1], or 0.0 for zero vectors
    """
    norm_embedding = np.linalg.norm(embedding)
    norm_pa = np.linalg.norm(pa)

    # Zero vector handling: treat as maximum deviation
    # Epsilon threshold prevents numerical instability
    if norm_embedding < 1e-10 or norm_pa < 1e-10:
        return 0.0

    return np.dot(embedding, pa) / (norm_embedding * norm_pa)
```

**Step 2:** Add edge case tests

```python
# Add to test suite:
def test_fidelity_zero_vector_handling():
    """Test fidelity calculation with zero vectors."""
    pa = np.array([1.0, 0.0, 0.0])
    zero_vec = np.array([0.0, 0.0, 0.0])

    # Should return 0.0, not crash
    fidelity = calculate_fidelity(zero_vec, pa)
    assert fidelity == 0.0

    # Reverse case
    fidelity = calculate_fidelity(pa, zero_vec)
    assert fidelity == 0.0

    # Both zero
    fidelity = calculate_fidelity(zero_vec, zero_vec)
    assert fidelity == 0.0
```

**Step 3:** Verify no regressions
- Run full validation suite
- Check Observatory v3 telemetry display
- Verify fidelity calculations in live session

### Commit Message:
```
[OPUS-CRITICAL-1] Fix zero vector division in fidelity calculations

- Add norm checks before division in calculate_fidelity()
- Return 0.0 for zero vectors (maximum deviation semantics)
- Add epsilon threshold (1e-10) for numerical stability
- Add edge case tests for zero vector handling

Fixes potential session crashes from zero-norm embeddings.
```

---

## CRITICAL ISSUE #2: Missing NaN/Inf Validation in Embedding Processing

### Files to Fix:
- `telos_purpose/core/embedding_provider.py`
  - Function: `get_embedding()`
  - Function: `get_batch_embeddings()`
  - Add new: `_is_valid_embedding()` helper

### Problem Statement:
No validation that API-returned embeddings contain valid float values. Could receive NaN, Inf, empty arrays, wrong dimensions.

### Impact if Not Fixed:
- **Silent NaN propagation** through all calculations
- **Incorrect interventions** based on invalid data
- **Data corruption** in telemetry logs
- **Difficult debugging** (no clear error point)

### Fix Implementation:

**Step 1:** Add validation helper to `embedding_provider.py`

```python
class EmbeddingProvider:
    def __init__(self, model: str, expected_dim: int):
        self.model = model
        self.expected_dim = expected_dim

    def _is_valid_embedding(self, embedding: np.ndarray) -> bool:
        """Validate embedding contains no NaN/Inf and has correct shape.

        Args:
            embedding: Vector to validate

        Returns:
            bool: True if valid, False otherwise
        """
        if embedding is None or embedding.size == 0:
            return False

        # Check for NaN or Inf values
        if not np.all(np.isfinite(embedding)):
            return False

        # Check dimension matches expected
        if embedding.shape[0] != self.expected_dim:
            return False

        return True
```

**Step 2:** Update `get_embedding()` with validation

```python
def get_embedding(self, text: str) -> np.ndarray:
    """Get embedding with comprehensive validation.

    Args:
        text: Input text to embed

    Returns:
        np.ndarray: Validated embedding vector

    Raises:
        ValueError: If embedding is invalid (NaN/Inf/wrong dimension)
    """
    try:
        response = self.client.embeddings.create(
            model=self.model,
            input=text
        )
        embedding = np.array(response.data[0].embedding, dtype=np.float64)

        # Validate embedding before returning
        if not self._is_valid_embedding(embedding):
            raise ValueError(
                f"Invalid embedding received: shape={embedding.shape}, "
                f"has_nan={np.any(np.isnan(embedding))}, "
                f"has_inf={np.any(np.isinf(embedding))}"
            )

        return embedding

    except Exception as e:
        logger.error(f"Embedding API failed for text: '{text[:50]}...': {e}")
        raise  # Re-raise for upstream handling
```

**Step 3:** Update `get_batch_embeddings()` similarly

```python
def get_batch_embeddings(self, texts: list[str]) -> list[np.ndarray]:
    """Get batch embeddings with validation."""
    # ... existing batch API call ...

    embeddings = [np.array(item.embedding, dtype=np.float64)
                  for item in response.data]

    # Validate all embeddings
    for i, emb in enumerate(embeddings):
        if not self._is_valid_embedding(emb):
            raise ValueError(f"Invalid embedding at index {i}")

    return embeddings
```

**Step 4:** Add validation tests

```python
def test_embedding_validation():
    """Test embedding validation catches invalid data."""
    provider = EmbeddingProvider(model="text-embedding-3-small", expected_dim=1536)

    # Test NaN detection
    nan_embedding = np.array([1.0, np.nan, 0.5])
    assert not provider._is_valid_embedding(nan_embedding)

    # Test Inf detection
    inf_embedding = np.array([1.0, np.inf, 0.5])
    assert not provider._is_valid_embedding(inf_embedding)

    # Test wrong dimension
    wrong_dim = np.zeros(768)  # Expected 1536
    assert not provider._is_valid_embedding(wrong_dim)

    # Test valid embedding
    valid_embedding = np.random.randn(1536)
    assert provider._is_valid_embedding(valid_embedding)
```

### Commit Message:
```
[OPUS-CRITICAL-2] Add NaN/Inf validation to embedding processing

- Add _is_valid_embedding() helper with comprehensive checks
- Validate all API-returned embeddings before use
- Check for NaN, Inf, empty arrays, dimension mismatches
- Add detailed error messages for debugging
- Add validation test suite

Prevents silent NaN propagation through fidelity calculations.
```

---

## CRITICAL ISSUE #3: Weak Cryptographic Entropy in Telemetric Keys

### Files to Fix:
- `telos_privacy/cryptography/telemetric_keys.py`
  - Function: `extract_entropy_from_telemetry()`
  - Function: `generate_session_key()`
- Add dependency: `cryptography` library for HKDF

### Problem Statement:
Current entropy extraction from session telemetry alone provides insufficient randomness for cryptographic key generation. Keys may be predictable if session patterns similar.

### Impact if Not Fixed:
- **Compromised forward secrecy** if keys predictable
- **Reduced security margin** for TSI creation
- **Vulnerability to pattern analysis** attacks

### Fix Implementation:

**Step 1:** Add cryptography dependency

```bash
# Add to requirements.txt:
cryptography>=41.0.0
```

**Step 2:** Update `extract_entropy_from_telemetry()` in `telemetric_keys.py`

```python
import secrets
import time
import os
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes

def extract_entropy_from_telemetry(self, session_telemetry: dict) -> bytes:
    """Extract cryptographically strong entropy from telemetry + system sources.

    Combines multiple entropy sources:
    1. System CSPRNG (primary security)
    2. High-resolution timestamps (uniqueness)
    3. Session telemetry (binding to specific session)

    Args:
        session_telemetry: Session-specific telemetry data

    Returns:
        bytes: 32-byte (256-bit) key material
    """
    # Session-specific telemetry (provides uniqueness, not primary entropy)
    fidelity_variance = session_telemetry.get('fidelity_variance', 0.0)
    timing_jitter = session_telemetry.get('response_time_variance', 0.0)
    turn_sequence = session_telemetry.get('turn_sequence_hash', '')

    # High-entropy system sources (PRIMARY SECURITY)
    system_random = secrets.token_bytes(32)  # Cryptographically secure PRNG
    timestamp_ns = time.time_ns()  # Nanosecond precision for uniqueness
    process_id = os.getpid()

    # Combine all entropy sources
    entropy_material = b''.join([
        system_random,  # 32 bytes of strong randomness
        str(timestamp_ns).encode('utf-8'),
        str(process_id).encode('utf-8'),
        str(fidelity_variance).encode('utf-8'),
        str(timing_jitter).encode('utf-8'),
        turn_sequence.encode('utf-8')
    ])

    # Use proper KDF to derive key material
    # HKDF provides domain separation and extract-then-expand
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=32,  # 256-bit key
        salt=None,  # Optional: add session-specific salt if available
        info=b'TELOS-TKeys-v1'  # Domain separation string
    )

    return hkdf.derive(entropy_material)
```

**Step 3:** Update `generate_session_key()` to use strengthened entropy

```python
def generate_session_key(self, session_telemetry: dict) -> bytes:
    """Generate session key with cryptographically strong entropy.

    Args:
        session_telemetry: Current session telemetry data

    Returns:
        bytes: 32-byte session key
    """
    return self.extract_entropy_from_telemetry(session_telemetry)
```

**Step 4:** Add entropy quality tests

```python
def test_entropy_quality():
    """Test entropy extraction produces high-quality randomness."""
    tkeys = TelemetricKeys()

    # Generate multiple keys from identical telemetry
    telemetry = {'fidelity_variance': 0.05, 'response_time_variance': 0.12}

    key1 = tkeys.extract_entropy_from_telemetry(telemetry)
    key2 = tkeys.extract_entropy_from_telemetry(telemetry)

    # Keys must be different (system randomness ensures uniqueness)
    assert key1 != key2

    # Keys must be 32 bytes
    assert len(key1) == 32
    assert len(key2) == 32

    # Keys should have high entropy (basic check)
    # Count unique bytes - should be diverse
    assert len(set(key1)) > 20  # At least 20 unique byte values
```

**Step 5:** Document TKeys signatures for grant applications

```python
class TelemetricKeys:
    """Telemetric Keys - Cryptographic system using session telemetry.

    ** CURRENT IMPLEMENTATION **
    This implementation demonstrates the TKeys approach with strengthened
    entropy for production use. Shows cryptographic signature for grants.

    ** FUTURE FULL IMPLEMENTATION **
    - Cryptographic containerization for deltas
    - Federated learning infrastructure
    - Institutional data sharing without raw data exposure

    Current Status: Production-ready signatures (NOW)
    Future Work: Full containerization (post-validation studies)
    """
```

### Commit Message:
```
[OPUS-CRITICAL-3] Strengthen cryptographic entropy in Telemetric Keys

- Combine system CSPRNG with session telemetry for entropy
- Use secrets.token_bytes() as primary randomness source
- Add HKDF for proper key derivation
- Add high-resolution timestamps and process ID
- Add entropy quality tests
- Document TKeys signatures for grant applications

Ensures 256-bit security for session key generation.
Shows cryptographic approach for institutional partnerships.
```

---

## CRITICAL ISSUE #4: Missing `await` on Async Calls

### Files to Fix:
- `telos_observatory_v3/core/async_processor.py`
  - Function: `process_batch()`
  - Function: `async_intervention_check()`
- Any other files with async/await issues (full audit needed)

### Problem Statement:
Async functions called without `await` causing silent failures, race conditions, and resource leaks.

### Impact if Not Fixed:
- **Observatory UI hangs** waiting for non-executed tasks
- **Interventions not triggered** (unawaited checks never run)
- **Memory leaks** from accumulated coroutine objects

### Fix Implementation:

**Step 1:** Audit all async function calls

```bash
# Search for potential issues:
grep -r "async def" telos_observatory_v3/ | grep -v "__pycache__"
# Then check each call site
```

**Step 2:** Fix `process_batch()` in `async_processor.py`

```python
# BEFORE (missing await):
async def process_batch(self, items: list):
    for item in items:
        self.process_item(item)  # Missing await - coroutine never executes!
    return "Done"

# AFTER (proper async):
async def process_batch(self, items: list):
    """Process items concurrently with proper await and error handling.

    Args:
        items: List of items to process

    Returns:
        list: Results from processing (or exceptions)
    """
    # Create tasks for concurrent execution
    tasks = [self.process_item(item) for item in items]

    # Await all tasks, capturing exceptions
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Log any failures
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"Item {i} processing failed: {result}")

    return results
```

**Step 3:** Fix `async_intervention_check()`

```python
# BEFORE:
async def check_all_sessions(self):
    for session_id in self.active_sessions:
        self.async_intervention_check(session_id)  # Missing await!

# AFTER:
async def check_all_sessions(self):
    """Check all active sessions for needed interventions."""
    tasks = [
        self.async_intervention_check(session_id)
        for session_id in self.active_sessions
    ]
    await asyncio.gather(*tasks, return_exceptions=True)
```

**Step 4:** Add async tests

```python
import pytest

@pytest.mark.asyncio
async def test_process_batch_awaits_correctly():
    """Test that process_batch properly awaits async operations."""
    processor = AsyncProcessor()

    items = ['item1', 'item2', 'item3']
    results = await processor.process_batch(items)

    # Verify all items were processed
    assert len(results) == 3
    assert all(r is not None for r in results)

@pytest.mark.asyncio
async def test_exception_handling_in_batch():
    """Test that exceptions in batch processing are caught."""
    processor = AsyncProcessor()

    # Mix of valid and invalid items
    items = ['valid', None, 'valid']  # None should cause exception
    results = await processor.process_batch(items)

    # Should have 3 results, middle one is exception
    assert len(results) == 3
    assert not isinstance(results[0], Exception)
    assert isinstance(results[1], Exception)
    assert not isinstance(results[2], Exception)
```

**Step 5:** Add linting check for future prevention

```bash
# Add to .pre-commit-hooks or CI:
# Check for async functions called without await
ruff check --select ASYNC  # If using ruff
# Or pylint:
pylint --disable=all --enable=missing-await telos_observatory_v3/
```

### Commit Message:
```
[OPUS-CRITICAL-4] Fix missing await on async function calls

- Add proper await to process_batch() concurrent execution
- Fix async_intervention_check() unawaited calls
- Add asyncio.gather() for concurrent task management
- Add exception handling for failed tasks
- Add async test suite
- Add linting check to prevent future issues

Fixes Observatory UI hangs and intervention execution failures.
```

---

## MEDIUM PRIORITY FIXES (Include in NOW if Time Permits)

### MEDIUM #5: Unbounded Intervention History Growth
- **File:** `telos_purpose/core/session_state.py`
- **Fix:** Add MAX_HISTORY_SIZE with rolling window
- **Priority:** Include if time before beta launch

### MEDIUM #6: Missing Embedding Dimension Validation
- **File:** `telos_purpose/core/dual_attractor.py`
- **Fix:** Validate PA dimensions before calculations
- **Priority:** Include if time before beta launch

---

## IMPLEMENTATION CHECKLIST

**Pre-Implementation:**
- [ ] Create git branch `post-opus-critical-fixes`
- [ ] Ensure `pre-opus-audit` branch/tag exist for rollback
- [ ] Backup current working state

**Implementation (One at a Time):**
- [ ] Issue #1: Zero vector division fix
  - [ ] Code changes implemented
  - [ ] Edge case tests added
  - [ ] Validation suite passes
  - [ ] Git commit: `[OPUS-CRITICAL-1]`

- [ ] Issue #2: NaN/Inf validation
  - [ ] Validation helper added
  - [ ] All embedding calls updated
  - [ ] Validation tests added
  - [ ] Git commit: `[OPUS-CRITICAL-2]`

- [ ] Issue #3: Crypto entropy strengthening
  - [ ] HKDF implementation added
  - [ ] System randomness integrated
  - [ ] Entropy tests added
  - [ ] TKeys signatures documented
  - [ ] Git commit: `[OPUS-CRITICAL-3]`

- [ ] Issue #4: Missing await fixes
  - [ ] All async calls audited
  - [ ] await added where needed
  - [ ] Async tests added
  - [ ] Linting check added
  - [ ] Git commit: `[OPUS-CRITICAL-4]`

**Post-Implementation:**
- [ ] Run complete validation suite
- [ ] Test Observatory v3 functionality
- [ ] Verify no performance regressions
- [ ] Update documentation
- [ ] Ready for institutional deployment

---

## TESTING REQUIREMENTS

**Edge Case Tests (Add to test suite):**
- Zero vectors in fidelity calculations
- NaN/Inf embeddings from API
- Dimension mismatches in PA operations
- Async task exception handling
- Entropy quality validation

**Integration Tests:**
- Full session with edge cases injected
- Observatory UI stability under load
- Intervention triggering correctness

**Performance Tests:**
- No regression in fidelity calculation speed
- Memory usage stable in long sessions
- API error recovery time

---

## SUCCESS CRITERIA

**Ready for Production When:**
✅ All 4 critical issues fixed and committed
✅ Edge case test suite passing
✅ Validation studies still pass (no regressions)
✅ Observatory v3 stable
✅ TKeys signatures documented for grants
✅ Performance validated (no slowdowns)

**Ready for Institutional Deployment:**
✅ GMU partnership outreach complete
✅ 60+ validation studies complete
✅ Observatory screenshots captured
✅ Grant applications submitted

---

**Document Status:** Ready for implementation
**Timeline:** Implement before institutional partnerships (December 2025)
**Next Steps:** Begin systematic implementation with git commits per issue

---

**Contact:** TELOS Labs - telos.steward@gmail.com
