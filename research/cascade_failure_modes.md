# Cascade Failure Mode Specification

**Version:** 1.0
**Date:** 2026-02-26
**Author:** TELOS AI Labs Inc.
**Purpose:** Failure mode documentation for the 4-layer boundary detection cascade. A8 review item (Karpathy finding). Addresses Bengio safety concern re: max-pool gate discontinuity.

---

## Architecture Overview

```
                    action_text
                        │
                        ▼
              ┌─────────────────┐
              │  L0: Keyword    │  ~0.1ms
              │  Pre-Filter     │  AUC 0.724
              └────────┬────────┘
                       │ keyword_matches[]
                       ▼
              ┌─────────────────┐
              │  L1: Cosine     │  ~10ms
              │  Similarity     │  Centroid + sub-centroid
              └────────┬────────┘
                       │ max_violation, triggered?
                       ▼
              ┌─────────────────┐
              │  L1.5: SetFit   │  ~3-5ms (ONNX)
              │  Classifier     │  AUC 0.980
              └────────┬────────┘
                       │ setfit_triggered?
                       ▼
              ┌─────────────────┐
              │  L2: Contrastive│  ~0.5ms
              │  Detection      │  Safe centroid margin
              └────────┬────────┘
                       │ contrastive_suppressed?
                       ▼
                BoundaryCheckResult
```

All layers execute within `_check_boundaries()` in `agentic_fidelity.py:554-754`. The cascade is NOT a sequential gate (where each layer decides whether to invoke the next). Instead, L0 and L1 always run; L1.5 conditionally fires; L2 runs per-boundary alongside L1.

---

## Layer 0: Keyword Pre-Filter

**Entry conditions:** Always runs. Scans `action_text` for violation-indicator keywords.

**Code path:** `agentic_fidelity.py:545-552` (`_keyword_check()`), boost applied at lines 631-643.

**Dependencies:**
- `self._violation_keywords` — merged list of global deontic keywords + per-config domain keywords. No external resources. Pure string matching.

**What triggers this layer:**
- Any non-empty `action_text` input.
- Keywords checked: `GLOBAL_DEONTIC_KEYWORDS` ("skip", "ignore", "override", "bypass", "suppress", "dismiss", "disregard", "directly", "automatically", "without review", "without approval", "even though", "go ahead", "just do", "don't bother", "no need to", "turn off") + per-config `violation_keywords` from YAML.

**What happens if this layer crashes:**
- **Exception in `_keyword_check()`:** The method is a simple list comprehension over strings. Only possible crash: corrupted `self._violation_keywords` list (None element). This would raise `TypeError`.
- **Failure mode:** If `_keyword_check()` raises, the exception propagates up through `_check_boundaries()` to `score_action()`. **No fallback — fail-open by crash.** The entire scoring call fails.
- **Mitigation:** This layer has no external dependencies (no model, no file I/O, no network). The only failure path is a programming error in keyword list construction.

**What happens if this layer is slow:**
- **Cannot be slow.** String matching against a list of ~20-50 keywords on a single text string. Sub-microsecond operation. No timeout needed.

**Expected latency:**
- p50: <0.1ms
- p99: <0.1ms
- Upper bound: O(n*m) where n=text length, m=keyword count. For typical inputs (<500 chars, <50 keywords): negligible.

**Output:** `keyword_matches: List[str]` (empty if none). Keywords boost L1 cosine score by +0.15 only if cosine already ≥ 0.40 (prevents keyword-only false positives).

---

## Layer 1: Cosine Similarity

**Entry conditions:** Always runs. Iterates over all `pa.boundaries`.

**Code path:** `agentic_fidelity.py:605-629` (main loop), lines 659-690 (trigger logic).

**Dependencies:**
- `pa.boundaries` — list of `BoundarySpec` objects with pre-computed `effective_embedding` vectors. Loaded at engine initialization from YAML config via `AgenticPA.create_from_template()`.
- `self._normalize_fidelity()` — normalization function (pure math, no external dependencies).
- `self._cosine_similarity()` — numpy dot product + norm (pure math).

**What triggers this layer:**
- At least one boundary in `pa.boundaries` with a non-None `effective_embedding`.

**What happens if this layer crashes:**
- **NaN in embedding:** `_cosine_similarity()` has an explicit NaN guard (line 448-450): returns 0.0 (fail-closed). This prevents NaN propagation.
- **Zero-norm embedding:** Division by zero is guarded (lines 443-445): returns 0.0.
- **Corrupted boundary embeddings:** If a boundary's `effective_embedding` is None, it's skipped (`if boundary.effective_embedding is None: continue`).
- **numpy exception:** Extremely unlikely given well-formed float32 arrays. If it occurs, exception propagates up. **Fail-open by crash.**
- **Sub-centroid failure:** Sub-centroid check is inside a try-free block. A malformed `sub_centroids` matrix would raise a numpy broadcasting error. **Fail-open by crash.**

**What happens if this layer is slow:**
- **Proportional to boundary count.** Each boundary requires one embedding cosine (O(d) where d=384 or 1024). For 17 boundaries (OpenClaw config): ~17 × 0.5ms ≈ 8.5ms. For 7 boundaries (Healthcare): ~3.5ms.
- **No timeout mechanism.** The loop runs to completion. A config with hundreds of boundaries would linearly degrade performance but would not hang.
- **Latency budget:** 10ms allocated (Karpathy design, HANDOFF_OPENCLAW.md convergence point C2).

**Expected latency:**
- p50: ~5-10ms (depends on boundary count)
- p95: ~12ms
- p99: ~15ms
- Dominated by numpy array operations — CPU-bound, deterministic.

**Output:** `max_violation: float`, `triggered: bool`, per-boundary margin data. Violation threshold: 0.70 (`BOUNDARY_VIOLATION_THRESHOLD`).

---

## Layer 1.5: SetFit Classifier

**Entry conditions — ALL must be true:**
1. `self._setfit_classifier` is not None (classifier was provided at engine init)
2. `not triggered` (L1 cosine did NOT already trigger a violation)
3. `max_violation >= KEYWORD_EMBEDDING_FLOOR (0.40)` (cosine shows boundary relevance)
4. `action_text` is non-empty

**Code path:** `agentic_fidelity.py:698-721`

**Dependencies:**
- **ONNX Runtime** (`onnxruntime` package) — for backbone inference
- **HuggingFace Tokenizers** (`tokenizers` package) — for text tokenization
- **Model files:** `model.onnx` (~86.8 MB), `tokenizer.json`, `head_weights.json` in model directory
- **Optional:** Platt calibration JSON (`platt_a`, `platt_b` parameters)

**What triggers this layer:**
- Cosine similarity shows boundary relevance (≥ 0.40) but hasn't triggered a definitive violation (< 0.70). This is the "ambiguous zone" where SetFit adds value.

**What happens if this layer crashes:**
```python
try:
    setfit_score = self._setfit_classifier.predict(action_text)
    if setfit_score >= self._setfit_classifier.threshold:
        setfit_triggered = True
        triggered = True
except Exception as e:
    logger.warning(f"SetFit L1.5 inference failed: {e}")
```
- **Explicit try/except.** Any exception (ONNX crash, tokenizer error, numpy error, model file corruption) is caught and logged as a warning.
- **Failure mode: FAIL-OPEN.** SetFit failure means the classifier didn't run — `setfit_triggered` stays False. The boundary check result depends on L1 cosine and L2 contrastive only.
- **This is the only cascade layer with explicit exception handling.** The design choice is intentional: SetFit is an enhancement layer, not a safety layer. L1 cosine is the safety backstop.

**What happens if this layer is slow:**
- **ONNX inference is synchronous, no timeout.** If the ONNX runtime hangs (corrupted model, resource exhaustion), the entire scoring call blocks.
- **No timeout mechanism** — this is a known gap (Karpathy finding).
- **Practical risk is low:** ONNX inference on a 384-dim model is CPU-bound and deterministic. Hangs would require ONNX runtime corruption, not slow convergence.

**Expected latency:**
- p50: 3-5ms (ONNX backbone + LR head)
- p95: 7ms (cold cache penalty on first invocation)
- p99: 10ms (GC pause or thread contention)
- Cold start: 50-200ms (first inference loads ONNX graph; subsequent calls reuse session). Hidden by engine initialization.

**Asymmetric override policy:**
- SetFit can ESCALATE (set `triggered = True`) but can NEVER DOWNGRADE (if L1 triggered, SetFit is not invoked).
- This is a deliberate safety property: the cascade can only escalate, never de-escalate.

**Output:** `setfit_triggered: bool`, `setfit_score: Optional[float]`.

---

## Layer 2: Contrastive Detection

**Entry conditions:** Runs for each boundary that has a non-None `safe_centroid` (defined in YAML config). Executes within the same boundary loop as L1 — not a separate pass.

**Code path:** `agentic_fidelity.py:645-696`

**Dependencies:**
- `boundary.safe_centroid` — pre-computed embedding of "safe" versions of the boundary (e.g., for "do not access financial records" boundary, the safe centroid embeds "review clinical records for treatment planning"). Loaded from YAML.
- Same `_cosine_similarity()` and `_normalize_fidelity()` as L1.

**What triggers this layer:**
- A boundary has `safe_centroid is not None` AND that boundary's L1 cosine score ≥ `BOUNDARY_VIOLATION_THRESHOLD` (0.70).

**What happens if this layer crashes:**
- **No explicit exception handling.** If `_cosine_similarity()` or `_normalize_fidelity()` raises for the safe centroid embedding, the exception propagates up. **Fail-open by crash.**
- **NaN guard inherited:** `_cosine_similarity()` returns 0.0 on NaN, so corrupted safe_centroid embeddings produce margin = violation - 0.0 = violation, meaning the violation triggers (fail-closed for this specific case).

**What happens if this layer is slow:**
- **Cannot be independently slow.** Contrastive detection is one additional cosine similarity per boundary — same O(d) cost as L1. Runs within the L1 loop.

**Expected latency:**
- Included in L1 timing. Adds ~0.5ms per boundary with a safe_centroid.

**Margin decision logic:**
```
margin = violation_score - safe_similarity

margin ≤ 0:         SUPPRESS (safe centroid closer → legitimate)
0 < margin ≤ 0.05:  TRIGGER (ambiguous zone → conservative)
margin > 0.05:      TRIGGER (clear violation)
```

**Output:** Modifies `triggered`, `contrastive_suppressed`, `trigger_safe_sim`, `trigger_margin`.

---

## Cross-Layer Failure Modes

### FM-1: Total scoring failure

**Trigger:** Uncaught exception in any layer (except L1.5 which has try/except).
**Effect:** `score_action()` raises an exception. The caller (response manager, hook, adapter) receives no result.
**Current behavior:** No top-level exception handler in `score_action()`. The caller must handle the exception.
**Risk:** If the caller does not handle the exception, the agent action may proceed without governance (fail-open) or halt entirely (fail-closed), depending on the integration pattern.
**Recommendation:** The response manager (`response_manager.py:788-791`) has a fallback: `result.decision = "INERT"` when "Governance engine unavailable." This provides fail-closed behavior at the application layer. The OpenClaw adapter (`governance_hook.py`) implements per-preset fail policy (fail-open or fail-closed, configurable).

### FM-2: Embedding function failure

**Trigger:** `self.embed_fn(action_text)` at `score_action()` line 298 raises or returns garbage.
**Effect:** All dimensions fail — they all depend on `action_embedding`.
**Current behavior:** No exception handling around `embed_fn()`. Exception propagates. If `embed_fn()` returns wrong-shaped array, numpy operations in subsequent dimensions may raise or produce NaN (which is caught at cosine level).
**Risk:** The embedding function is an external dependency (SentenceTransformer model or Mistral API). Network failure, model corruption, or OOM can cause this.

### FM-3: SetFit available but stale model

**Trigger:** SetFit model files exist but are from a different domain (e.g., healthcare model loaded for OpenClaw config).
**Effect:** SetFit produces predictions but they are semantically mismatched. May produce false triggers (legitimate OpenClaw actions classified as healthcare violations) or false negatives (OpenClaw violations not recognized by healthcare model).
**Current behavior:** `AgenticResponseManager._discover_setfit()` implements auto-discovery matching model directory names to config names. Mismatches are possible if directories are renamed or models are not rebuilt after config changes.
**Risk:** Silent degradation — no error, just wrong predictions.

### FM-4: Cascade ordering violation

**Trigger:** Future code changes that alter the layer execution order.
**Effect:** The asymmetric override policy depends on L1 running before L1.5 (SetFit only fires when L1 hasn't triggered). If order changes, SetFit might downgrade L1 decisions.
**Current behavior:** Enforced by code structure (sequential execution in `_check_boundaries()`).
**Risk:** Low — requires deliberate code modification. No runtime ordering guarantee mechanism.

### FM-5: Boundary-free config

**Trigger:** PA config with empty `boundaries` list.
**Effect:** `_check_boundaries()` returns `BoundaryCheckResult(0.0, False, "No boundaries defined", None, None, False)`. No boundary detection occurs.
**Current behavior:** Correct — governance without boundaries is valid (early-stage configuration).
**Risk:** None if intentional. If boundaries are accidentally removed from config, governance silently degrades.

---

## Max-Pool Gate Discontinuity (Bengio Safety Concern)

### The Discontinuity

The purpose and scope dimensions implement a max-pool rescue mechanism gated at 0.50:

```python
# agentic_fidelity.py:511
if self.pa.purpose_example_embeddings and centroid_score < 0.50:
    max_example = max(normalize(cosine(action, ex)) for ex in examples)
    if max_example > centroid_score:
        lift = (max_example - centroid_score) * 0.4
        return centroid_score + min(lift, 0.08)
return centroid_score
```

**The gate creates a sharp boundary:**
- At centroid_score = 0.499: max-pool fires, potential +0.08 lift → score = 0.579
- At centroid_score = 0.501: max-pool does NOT fire → score = 0.501

**Maximum discontinuity: 0.078** (0.499+0.08=0.579 vs 0.501). This violates the assumption that similar inputs produce similar outputs (Lipschitz continuity).

### Adversarial Exploitation Risk

An adversary who understands the gate could craft inputs that:

1. **Land just below 0.50** to trigger max-pool rescue on an action that should NOT be rescued. If the action happens to be close to a stored example embedding, it gets an unwarranted +0.08 boost. This could promote a borderline-SUGGEST action to CLARIFY band.

2. **Land just above 0.50** to avoid max-pool rescue on an action that the examples would have flagged as unusual. This is less concerning because the action is already in a higher band.

### Risk Assessment

**Practical exploitability: LOW.** The adversary must:
- Know the gate threshold (0.50) — this is in open-source code
- Know the stored example embeddings — these are derived from the YAML config
- Craft input text that produces a centroid score in the narrow window [0.42, 0.50) where the +0.08 lift matters for verdict outcomes
- Do all of this while also producing the intended max-example match

The window where the gate changes verdicts is narrow: only composites in [0.42, 0.50) where the +0.028 composite lift (0.35 × 0.08) could cross the EXECUTE threshold (0.45). This is a ~0.03 composite window.

### Mitigations

1. **Cap at +0.08:** The lift is bounded, preventing unbounded score inflation.
2. **40% factor:** Only 40% of the gap between centroid and max_example is applied, damping the effect.
3. **Boundary override:** Even if purpose fidelity is inflated, boundary detection operates independently and ESCALATE overrides any threshold-based verdict.
4. **Cat A structural protection:** Boundary detection fires first in `_make_decision()` and overrides to ESCALATE regardless of purpose score. No amount of purpose inflation can regress Cat A. Confirmed across all 12 Phase 8 experiments.

### Acknowledged Gap

The gate creates a non-differentiable point in the scoring function. For production deployment in adversarial environments, consider:
- Replacing the hard gate with a sigmoid transition: `gate_factor = 1 / (1 + exp(k * (centroid_score - 0.50)))` where k controls steepness
- Logging gate activations for monitoring (are adversaries probing the boundary?)
- Including gate threshold in the optimizer search space

---

## Latency Budget Summary

| Layer | p50 | p95 | p99 | Budget |
|-------|-----|-----|-----|--------|
| L0: Keyword | <0.1ms | <0.1ms | <0.1ms | 0.1ms |
| L1: Cosine | 5-10ms | 12ms | 15ms | 10ms |
| L1.5: SetFit | 3-5ms | 7ms | 10ms | 5ms |
| L2: Contrastive | ~0.5ms | ~0.5ms | ~1ms | Included in L1 |
| **Total boundary** | **9-16ms** | **20ms** | **26ms** | **15ms** |

**Full scoring (all 6 dimensions):** ~10-17ms total (Karpathy convergence point C2). The boundary cascade is the dominant cost. Purpose, scope, and chain are single cosine operations (~0.5ms each). Tool fidelity involves ranking all registered tools (~2-5ms for 36 tools).

---

## Dependency Matrix

| Layer | Python Packages | Models/Files | External Services | Fail Behavior |
|-------|----------------|-------------|-------------------|---------------|
| L0 | None (stdlib) | None | None | Crash → propagate |
| L1 | numpy | PA boundary embeddings (in memory) | None | NaN → 0.0 (fail-closed) |
| L1.5 | onnxruntime, tokenizers | model.onnx, tokenizer.json, head_weights.json | None | Exception → caught, fail-open |
| L2 | numpy | PA safe_centroid embeddings (in memory) | None | NaN → 0.0 (fail-closed for this case) |
| embed_fn | sentence-transformers OR httpx | MiniLM model files OR Mistral API | Mistral API (if Mistral) | Crash → propagate |

---

*Cascade failure modes v1.0 — A8 review Karpathy finding + Bengio safety concern documented.*
