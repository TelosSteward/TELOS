# TELOS Observatory V3 - Supabase Audit Summary

## Key Findings at a Glance

### Status: FOUNDATION COMPLETE, INTEGRATION INCOMPLETE

The Supabase integration exists and is well-architected, but only a fraction of available data is being persisted.

---

## File Inventory

| File | Lines | Status | Purpose |
|------|-------|--------|---------|
| `services/backend_client.py` | 506 | COMPLETE | Primary Supabase interface |
| `services/turn_tracker.py` | 147 | PARTIAL | Turn lifecycle tracking |
| `main.py` | 150 (relevant) | PARTIAL | App initialization |
| `core/state_manager.py` | 300+ | PARTIAL | State management |
| `services/beta_response_manager.py` | 1500+ | PARTIAL | Main governance engine |
| `components/beta_pa_establishment.py` | 500+ | PARTIAL | PA setup |
| `components/beta_ab_testing.py` | 200+ | PARTIAL | A/B testing |
| `components/beta_review.py` | 100+ | PARTIAL | Session review |

---

## Supabase Tables & Operations

### Tables Confirmed to Exist (6)

| Table | Rows | Frequency | Status |
|-------|------|-----------|--------|
| `governance_deltas` | Per turn | INSERT, UPSERT | PRIMARY - Main data table |
| `beta_consent_log` | Once/session | INSERT | AUDIT TRAIL |
| `primacy_attractor_configs` | Once/PA | INSERT | METADATA ONLY |
| `beta_sessions` | Once/session | INSERT, UPDATE | SESSION TRACKING |
| `beta_turns` | Per turn | INSERT, UPDATE | TURN TRACKING |
| `session_summaries` | Once/session | UPSERT | AGGREGATES |

### Critical Gap: Missing Tables (5)

These should exist but don't:
- `governance_trace_events` - For detailed event logging
- `turn_snapshots` - For complete turn context
- `adaptive_context_decisions` - For context manager decisions
- `user_feedback` - For feedback integration
- `ab_test_results` - For structured A/B test data

---

## What's Being Saved

### PER TURN (✓ Implemented)
```
✓ session_id (UUID)
✓ turn_number (int)
✓ fidelity_score (0.0-1.0)
✓ distance_from_pa (float)
✓ intervention_triggered (bool)
✓ intervention_type (string)
✓ mode ('demo', 'beta', 'open')
```

### SESSION LEVEL (✓ Implemented)
```
✓ total_turns
✓ avg_fidelity
✓ total_interventions
✓ completion timestamp
```

### CRITICAL MISSING DATA

1. **Turn Embeddings** ✗
   - User input embeddings (384 or 1024 dims)
   - Response embeddings
   - Attractor center vector

2. **Fidelity Details** ✗
   - Raw similarity score (before normalization)
   - Layer 1 hard block check result
   - Layer 2 basin membership
   - Fidelity zone (GREEN/YELLOW/ORANGE/RED)
   - Previous fidelity delta

3. **Intervention Semantics** ✗
   - Semantic interpretation band
   - Behavioral specifications applied
   - Context injection prompt used
   - Intervention strength (0.0-1.0)

4. **Adaptive Context** ✗
   - Message type classification
   - Conversation phase detection
   - Tier classification (1-3)
   - Threshold adjustments

5. **AI Response Fidelity** ✗
   - AI response raw similarity
   - AI response normalized fidelity
   - Intervention applied to AI (yes/no)
   - Regeneration attempts & results

6. **Session Configuration** ✗
   - Embedding model used
   - PA establishment method
   - Feature flags enabled
   - Browser/environment info

---

## Active Call Sites

### 2-3 Call Sites Found:

1. **`services/beta_response_manager.py`** - _store_turn_data()
   - Calls: `backend.transmit_delta()`
   - Frequency: Per turn
   - Status: ACTIVE (if backend.enabled)

2. **`services/turn_tracker.py`** - Multiple methods
   - Calls: `initiate_turn()`, `mark_calculating_pa()`, `complete_turn()`
   - Frequency: Multiple per turn
   - Status: ACTIVE IF TurnTracker IS USED

3. **`main.py`** - check_beta_completion()
   - Calls: `transmit_delta()`
   - Frequency: Once per session
   - Status: ACTIVE

### Mostly Inactive:

- `components/beta_review.py` - Query operations
- `components/beta_ab_testing.py` - Insert/update operations
- `components/beta_pa_establishment.py` - Session creation

---

## Critical Gaps Ranked by Impact

### 🔴 HIGH IMPACT

1. **Governance Trace Events Missing**
   - Evidence schema exists but doesn't persist to Supabase
   - Local JSONL only → data lost when session ends
   - Need: `transmit_governance_events()` method

2. **Turn Cache Not Persisted**
   - Exists in `st.session_state` but never saved
   - Contains: embeddings, normalized scores, decision context
   - Problem: Can't reconstruct turns from Supabase

3. **Intervention Semantics Lost**
   - Semantic interpreter results computed but not stored
   - Semantic bands, behavioral specs unknown after session
   - Can't analyze intervention effectiveness

### 🟡 MEDIUM IMPACT

4. **Adaptive Context Decisions Unmeasured**
   - Message type classification per turn unknown
   - Phase transitions unmeasured
   - Threshold adjustments not logged

5. **AI Response Fidelity Uncaptured**
   - AI response quality independent of user input unknown
   - Regeneration attempts hidden
   - Can't measure AI drift separately from user drift

6. **A/B Test Results Unstandardized**
   - Saved as nested JSON with turn_number=999 marker
   - Not queryable as first-class data
   - Need: dedicated `ab_test_results` table

### 🟢 LOW IMPACT

7. **Session Metadata Incomplete**
   - PA establishment method not recorded
   - Feature flags not logged
   - Can't correlate session config with outcomes

---

## Data Flow Comparison

### Current (Incomplete)
```
User Input
  ├─→ Calculate Fidelity [NOT SAVED DETAILS]
  ├─→ Decide Intervention [BASIC INFO ONLY]
  ├─→ Generate Response [NOT TRACKED]
  └─→ transmit_delta() → governance_deltas table
       [Only fidelity score + intervention type, no context]
```

### Should Be (Complete)
```
User Input
  ├─→ Calculate Fidelity → record_fidelity_event()
  │   [Save raw score, normalized, zone, layer checks]
  ├─→ Decide Intervention → record_intervention_event()
  │   [Save reason, strength, semantic band]
  ├─→ Generate Response → record_response_event()
  │   [Save source, length, generation time, AI fidelity]
  ├─→ Analyze Context → record_context_decision()
  │   [Save message type, phase, tier, adjustments]
  └─→ transmit_delta() → COMPLETE turn record
      [All events + delta data in atomic record]
```

---

## Privacy Protection Status

### ✓ What's Protected
- No conversation content ever transmitted (enforced)
- Consent audit trail maintained
- Graceful degradation if backend unavailable
- Session-specific tracking by default

### ⚠️ Potential Concerns
- `intervention_reason` field could leak content if not careful
- User feedback text may be sensitive
- Element counts might reveal intent
- Semantic interpretation could be reversed to infer content

### Recommendations
- Add DATA_CLASSIFICATION to all tables (public/internal/sensitive)
- Implement row-level security (RLS) policies
- Regular privacy audits of field values
- Exclude sensitive fields by default

---

## Recommended Implementation Priority

### Phase 1: EVENT INFRASTRUCTURE (2-3 days)
```python
# Add to BackendService
def record_fidelity_event(session_id, turn, raw_sim, normalized_fidelity, zone, ...)
def record_intervention_event(session_id, turn, level, reason, strength, band, ...)
def record_response_event(session_id, turn, source, length, ai_fidelity, ...)
def batch_transmit_events(session_id, events)  # Efficient bulk insert
```

### Phase 2: GOVERNANCE TRACE INTEGRATION (1-2 days)
```python
# Connect evidence_schema.py to backend_client
# Sync JSONL events to Supabase on session end
# Create governance_trace_events table
```

### Phase 3: TURN CONTEXT PERSISTENCE (2-3 days)
```python
# Define turn_cache schema
# Persist turn snapshots with embeddings
# Create turn_snapshots table
```

### Phase 4: ADAPTIVE CONTEXT LOGGING (1-2 days)
```python
# Save message type + phase + tier per turn
# Track threshold adjustments
# Create adaptive_context_decisions table
```

### Phase 5: A/B TEST STANDARDIZATION (1 day)
```python
# Replace turn_number=999 marker pattern
# Create ab_test_results table
# Update query interfaces
```

---

## Files That Should Be Modified

**To Add:**
- `services/backend_client.py` (extend with new methods)
- `services/governance_trace_sync.py` (new - sync JSONL to Supabase)

**To Integrate:**
- `services/beta_response_manager.py` (call new backend methods)
- `services/turn_tracker.py` (record all lifecycle events)
- `telos_purpose/core/governance_trace_collector.py` (add backend export)
- `telos_purpose/core/adaptive_context.py` (persist decisions)

---

## Key Metrics Not Being Captured

### Per Turn
- Raw vs. normalized fidelity
- Fidelity zone (GREEN/YELLOW/ORANGE/RED)
- Layer 1 & Layer 2 check results
- AI response fidelity
- Message type classification
- Conversation phase
- Semantic band (intervention strength level)

### Per Session
- PA establishment method used
- Embedding model used
- Feature flags enabled
- Conversation goals and accomplishment
- User preference (baseline vs TELOS)
- Response generation latency per component

---

## Summary Statistics

| Metric | Value |
|--------|-------|
| Primary Supabase file | `backend_client.py` (506 lines) |
| Methods defined | 14 |
| Methods actively used | 2-3 |
| Tables created | 6 |
| Tables needed | 5+ |
| Data fields persisted | ~10 |
| Data fields available but not saved | ~30 |
| Call sites in codebase | 7 |
| Active call sites | 2-3 |
| Privacy-sensitive fields | 3 |
| Estimated integration completeness | 20-30% |

---

## Next Steps

1. **Review** this report with the team
2. **Prioritize** which gaps are most important for your research goals
3. **Implement** Phase 1 (Event Infrastructure) first
4. **Test** with a sample beta session
5. **Validate** data quality before Phase 2

For detailed implementation guidance, see the full report:
**`SUPABASE_AUDIT_REPORT.txt`** (46KB - comprehensive reference)

