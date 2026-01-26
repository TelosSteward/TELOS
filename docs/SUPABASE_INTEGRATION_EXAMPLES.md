# Supabase Integration - Code Examples & Patterns

## Overview

This document provides code examples for implementing the missing Supabase integrations identified in the audit.

---

## Pattern 1: Recording Fidelity Events

### Current (Missing)
```python
# In services/beta_response_manager.py line ~400
raw_similarity = calculate_similarity(user_embedding, pa_embedding)
fidelity_score = normalize_fidelity(raw_similarity)
# ❌ NO RECORD: fidelity calculation details lost
```

### Solution
```python
# Add to BackendService class
def record_fidelity_event(
    self,
    session_id: str,
    turn_number: int,
    raw_similarity: float,
    normalized_fidelity: float,
    fidelity_zone: str,  # "GREEN", "YELLOW", "ORANGE", "RED"
    layer1_hard_block: bool,
    layer2_outside_basin: bool,
    distance_from_pa: float,
    in_basin: bool
) -> bool:
    """Record fidelity calculation event with full governance context."""
    if not self.enabled:
        return False
    
    try:
        event_data = {
            'session_id': str(session_id),
            'turn_number': turn_number,
            'event_type': 'fidelity_calculated',
            'raw_similarity': raw_similarity,
            'normalized_fidelity': normalized_fidelity,
            'fidelity_zone': fidelity_zone,
            'layer1_hard_block': layer1_hard_block,
            'layer2_outside_basin': layer2_outside_basin,
            'distance_from_pa': distance_from_pa,
            'in_basin': in_basin,
            'timestamp': datetime.now().isoformat()
        }
        
        result = self.client.table('governance_trace_events').insert(event_data).execute()
        
        if result.data:
            print(f"✓ Fidelity event recorded: Turn {turn_number}, Zone {fidelity_zone}")
            return True
        else:
            print(f"❌ Failed to record fidelity event")
            return False
    except Exception as e:
        print(f"❌ Error recording fidelity event: {e}")
        return False


# Usage in beta_response_manager.py:
if self.backend and self.backend.enabled:
    self.backend.record_fidelity_event(
        session_id=session_id,
        turn_number=turn_number,
        raw_similarity=raw_similarity,
        normalized_fidelity=user_fidelity,
        fidelity_zone="GREEN" if user_fidelity >= 0.70 else "YELLOW",
        layer1_hard_block=baseline_hard_block,
        layer2_outside_basin=not in_basin,
        distance_from_pa=1.0 - raw_similarity,
        in_basin=in_basin
    )
```

---

## Pattern 2: Recording Intervention Events

### Current (Missing)
```python
# In services/beta_response_manager.py line ~440
if should_intervene:
    semantic_band = compute_semantic_band(fidelity)
    intervention_prompt = get_intervention_prompt(semantic_band)
    # ❌ NO RECORD: semantic interpretation details lost
```

### Solution
```python
# Add to BackendService class
def record_intervention_event(
    self,
    session_id: str,
    turn_number: int,
    intervention_level: str,  # "none", "monitor", "correct", "intervene", "escalate", "hard_block"
    trigger_reason: str,
    fidelity_at_trigger: float,
    controller_strength: float,
    semantic_band: str,  # "minimal", "light", "moderate", "firm", "strong"
    action_taken: str  # "context_injection", "regeneration", "block", "human_review"
) -> bool:
    """Record intervention event with semantic details."""
    if not self.enabled:
        return False
    
    try:
        event_data = {
            'session_id': str(session_id),
            'turn_number': turn_number,
            'event_type': 'intervention_triggered',
            'intervention_level': intervention_level,
            'trigger_reason': trigger_reason,
            'fidelity_at_trigger': fidelity_at_trigger,
            'controller_strength': controller_strength,
            'semantic_band': semantic_band,
            'action_taken': action_taken,
            'timestamp': datetime.now().isoformat()
        }
        
        result = self.client.table('governance_trace_events').insert(event_data).execute()
        
        if result.data:
            print(f"✓ Intervention event recorded: Turn {turn_number}, Level {intervention_level}")
            return True
        else:
            print(f"❌ Failed to record intervention event")
            return False
    except Exception as e:
        print(f"❌ Error recording intervention event: {e}")
        return False


# Usage in beta_response_manager.py:
if self.backend and self.backend.enabled:
    self.backend.record_intervention_event(
        session_id=session_id,
        turn_number=turn_number,
        intervention_level="correct" if user_fidelity < 0.70 else "none",
        trigger_reason="Basin exit detected" if not in_basin else "Drift below GREEN",
        fidelity_at_trigger=user_fidelity,
        controller_strength=intervention_strength,
        semantic_band=semantic_band,
        action_taken="context_injection"
    )
```

---

## Pattern 3: Recording Response Events

### Current (Missing)
```python
# In services/beta_response_manager.py line ~450
native_response = generate_response(user_input)
# ❌ NO RECORD: Response generation metrics and AI fidelity lost
```

### Solution
```python
# Add to BackendService class
def record_response_event(
    self,
    session_id: str,
    turn_number: int,
    response_source: str,  # "native", "governed", "steward"
    response_length: int,
    generation_time_ms: int,
    tokens_generated: Optional[int],
    ai_response_fidelity: Optional[float] = None,
    ai_fidelity_raw: Optional[float] = None
) -> bool:
    """Record response generation event with AI fidelity metrics."""
    if not self.enabled:
        return False
    
    try:
        event_data = {
            'session_id': str(session_id),
            'turn_number': turn_number,
            'event_type': 'response_generated',
            'response_source': response_source,
            'response_length': response_length,
            'generation_time_ms': generation_time_ms,
            'tokens_generated': tokens_generated,
            'ai_response_fidelity': ai_response_fidelity,
            'ai_fidelity_raw': ai_fidelity_raw,
            'timestamp': datetime.now().isoformat()
        }
        
        result = self.client.table('governance_trace_events').insert(event_data).execute()
        
        if result.data:
            print(f"✓ Response event recorded: Turn {turn_number}, Source {response_source}")
            return True
        else:
            print(f"❌ Failed to record response event")
            return False
    except Exception as e:
        print(f"❌ Error recording response event: {e}")
        return False


# Usage in beta_response_manager.py:
import time
start_time = time.time()
native_response = generate_response(user_input)
generation_time_ms = int((time.time() - start_time) * 1000)

if self.backend and self.backend.enabled:
    self.backend.record_response_event(
        session_id=session_id,
        turn_number=turn_number,
        response_source="native" if not should_intervene else "governed",
        response_length=len(native_response),
        generation_time_ms=generation_time_ms,
        tokens_generated=count_tokens(native_response),
        ai_response_fidelity=ai_fidelity,
        ai_fidelity_raw=raw_ai_fidelity
    )
```

---

## Pattern 4: Recording Adaptive Context Decisions

### Current (Missing)
```python
# In services/beta_response_manager.py line ~300
adaptive_result = self.adaptive_context_manager.process_message(
    text=user_input,
    embedding=embedding,
    fidelity_score=fidelity
)
# ❌ NO RECORD: Message type, phase, tier, adjustments all lost
```

### Solution
```python
# Add to BackendService class
def record_context_decision(
    self,
    session_id: str,
    turn_number: int,
    message_type: str,  # "DIRECT", "FOLLOW_UP", "CLARIFICATION", "ANAPHORA"
    conversation_phase: str,  # "EXPLORATION", "FOCUS", "DRIFT", "RECOVERY"
    tier: int,  # 1, 2, or 3
    threshold_adjustment: float,
    confidence: float
) -> bool:
    """Record adaptive context manager decision."""
    if not self.enabled:
        return False
    
    try:
        event_data = {
            'session_id': str(session_id),
            'turn_number': turn_number,
            'event_type': 'context_decision',
            'message_type': message_type,
            'conversation_phase': conversation_phase,
            'tier': tier,
            'threshold_adjustment': threshold_adjustment,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat()
        }
        
        result = self.client.table('governance_trace_events').insert(event_data).execute()
        
        if result.data:
            print(f"✓ Context decision recorded: Turn {turn_number}, Phase {conversation_phase}")
            return True
        else:
            print(f"❌ Failed to record context decision")
            return False
    except Exception as e:
        print(f"❌ Error recording context decision: {e}")
        return False


# Usage in beta_response_manager.py:
if ADAPTIVE_CONTEXT_AVAILABLE and ADAPTIVE_CONTEXT_ENABLED:
    adaptive_result = self.adaptive_context_manager.process_message(...)
    
    if self.backend and self.backend.enabled:
        self.backend.record_context_decision(
            session_id=session_id,
            turn_number=turn_number,
            message_type=adaptive_result.message_type.name,
            conversation_phase=adaptive_result.phase.name,
            tier=adaptive_result.tier,
            threshold_adjustment=adaptive_result.adjusted_threshold - INTERVENTION_THRESHOLD,
            confidence=0.85  # Could be computed from context buffer
        )
```

---

## Pattern 5: Batch Event Transmission

### Implementation
```python
# Add to BackendService class
def batch_transmit_events(
    self,
    session_id: str,
    events: List[Dict[str, Any]]
) -> bool:
    """Efficiently transmit multiple events in a single batch."""
    if not self.enabled or not events:
        return False
    
    try:
        # Add session_id to all events if not present
        for event in events:
            if 'session_id' not in event:
                event['session_id'] = str(session_id)
        
        result = self.client.table('governance_trace_events').insert(events).execute()
        
        if result.data:
            print(f"✓ Batch transmitted: {len(events)} events for session {session_id}")
            return True
        else:
            print(f"❌ Batch transmission failed")
            return False
    except Exception as e:
        print(f"❌ Error in batch transmission: {e}")
        return False


# Usage pattern: Collect events during turn, transmit at completion
events_to_transmit = []

# Record fidelity
events_to_transmit.append({
    'turn_number': turn_number,
    'event_type': 'fidelity_calculated',
    'raw_similarity': raw_similarity,
    'normalized_fidelity': normalized_fidelity,
    # ... other fields
})

# Record intervention
events_to_transmit.append({
    'turn_number': turn_number,
    'event_type': 'intervention_triggered',
    'intervention_level': intervention_level,
    # ... other fields
})

# Record response
events_to_transmit.append({
    'turn_number': turn_number,
    'event_type': 'response_generated',
    'response_source': response_source,
    # ... other fields
})

# Transmit all at once
if self.backend and self.backend.enabled:
    self.backend.batch_transmit_events(session_id, events_to_transmit)
```

---

## Pattern 6: Turn Snapshot Persistence

### Supabase Schema
```sql
CREATE TABLE turn_snapshots (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES beta_sessions(session_id),
    turn_number INT NOT NULL,
    user_fidelity_raw FLOAT,
    user_fidelity_normalized FLOAT,
    user_fidelity_zone TEXT,
    ai_fidelity_raw FLOAT,
    ai_fidelity_normalized FLOAT,
    intervention_applied BOOLEAN,
    intervention_semantic_band TEXT,
    response_source TEXT,
    response_length INT,
    adaptive_context JSONB,  -- Full adaptive context result
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(session_id, turn_number)
);
```

### Backend Method
```python
# Add to BackendService class
def save_turn_snapshot(
    self,
    session_id: str,
    turn_number: int,
    fidelity_data: Dict[str, Any],
    intervention_data: Dict[str, Any],
    response_data: Dict[str, Any],
    adaptive_context: Optional[Dict[str, Any]] = None
) -> bool:
    """Save comprehensive turn snapshot."""
    if not self.enabled:
        return False
    
    try:
        snapshot = {
            'session_id': str(session_id),
            'turn_number': turn_number,
            **fidelity_data,
            **intervention_data,
            **response_data,
            'adaptive_context': adaptive_context or {},
            'created_at': datetime.now().isoformat()
        }
        
        result = self.client.table('turn_snapshots').upsert(snapshot).execute()
        
        if result.data:
            print(f"✓ Turn snapshot saved: {session_id}, turn {turn_number}")
            return True
        else:
            print(f"❌ Turn snapshot save failed")
            return False
    except Exception as e:
        print(f"❌ Error saving turn snapshot: {e}")
        return False
```

---

## Pattern 7: Session Context Logging

### Supabase Schema
```sql
CREATE TABLE session_context (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL UNIQUE REFERENCES beta_sessions(session_id),
    embedding_model TEXT,  -- "sentence-transformers/all-MiniLM-L6-v2", "mistral-embed", etc.
    pa_establishment_method TEXT,  -- "template", "custom", "fresh_start"
    pa_template_name TEXT,
    feature_flags JSONB,  -- {"adaptive_context": true, "sci": true, ...}
    model_used TEXT,  -- "mistral-small-latest"
    browser_info TEXT,
    session_duration_seconds INT,
    created_at TIMESTAMP DEFAULT NOW()
);
```

### Backend Method
```python
# Add to BackendService class
def log_session_context(
    self,
    session_id: str,
    embedding_model: str,
    pa_establishment_method: str,
    pa_template_name: Optional[str],
    feature_flags: Dict[str, bool],
    model_used: str,
    browser_info: Optional[str] = None
) -> bool:
    """Log session configuration and feature flags."""
    if not self.enabled:
        return False
    
    try:
        context_data = {
            'session_id': str(session_id),
            'embedding_model': embedding_model,
            'pa_establishment_method': pa_establishment_method,
            'pa_template_name': pa_template_name,
            'feature_flags': feature_flags,
            'model_used': model_used,
            'browser_info': browser_info,
            'created_at': datetime.now().isoformat()
        }
        
        result = self.client.table('session_context').insert(context_data).execute()
        
        if result.data:
            print(f"✓ Session context logged: {session_id}")
            return True
        else:
            print(f"❌ Session context logging failed")
            return False
    except Exception as e:
        print(f"❌ Error logging session context: {e}")
        return False


# Usage in main.py during initialization:
backend.log_session_context(
    session_id=st.session_state.session_id,
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    pa_establishment_method="template",
    pa_template_name="collaborative_reasoning",
    feature_flags={
        "adaptive_context": True,
        "sci": True,
        "ai_response_fidelity": True
    },
    model_used="mistral-small-latest",
    browser_info=st.session_state.get('user_agent', None)
)
```

---

## Pattern 8: Integrating Governance Trace Collector

### Current Architecture
```
GovernanceTraceCollector → JSONL file (local only)
```

### Enhanced Architecture
```
GovernanceTraceCollector → JSONL file + Backend sync
    ├─→ Real-time streaming to Supabase (optional)
    └─→ Batch sync at session end (recommended)
```

### Implementation
```python
# In governance_trace_collector.py, add:

def sync_to_backend(self, backend_service: Optional['BackendService'] = None):
    """Sync JSONL events to Supabase backend."""
    if backend_service is None or not backend_service.enabled:
        return False
    
    try:
        events = []
        
        # Read JSONL file
        if self.trace_file.exists():
            with open(self.trace_file, 'r') as f:
                for line in f:
                    if line.strip():
                        event = json.loads(line)
                        events.append(event)
        
        # Transmit to backend
        if events:
            success = backend_service.batch_transmit_events(self.session_id, events)
            if success:
                logger.info(f"Synced {len(events)} events to Supabase")
                return True
        
        return False
    except Exception as e:
        logger.error(f"Error syncing to backend: {e}")
        return False


# Usage at session end:
collector.sync_to_backend(backend_service)
```

---

## Integration Checklist

- [ ] Add `record_fidelity_event()` to BackendService
- [ ] Add `record_intervention_event()` to BackendService
- [ ] Add `record_response_event()` to BackendService
- [ ] Add `record_context_decision()` to BackendService
- [ ] Add `batch_transmit_events()` to BackendService
- [ ] Create `governance_trace_events` table in Supabase
- [ ] Create `turn_snapshots` table in Supabase
- [ ] Create `session_context` table in Supabase
- [ ] Integrate event calls into `beta_response_manager.py`
- [ ] Add sync method to `governance_trace_collector.py`
- [ ] Add session context logging to `main.py`
- [ ] Test with sample beta session
- [ ] Validate data quality in Supabase

---

## Testing Queries

```sql
-- Verify events recorded
SELECT COUNT(*), event_type FROM governance_trace_events 
WHERE session_id = 'your-session-id'
GROUP BY event_type;

-- Check fidelity trajectory
SELECT turn_number, normalized_fidelity, fidelity_zone
FROM governance_trace_events
WHERE session_id = 'your-session-id' AND event_type = 'fidelity_calculated'
ORDER BY turn_number;

-- Analyze interventions
SELECT turn_number, intervention_level, semantic_band, controller_strength
FROM governance_trace_events
WHERE session_id = 'your-session-id' AND event_type = 'intervention_triggered'
ORDER BY turn_number;

-- Check adaptive context patterns
SELECT message_type, conversation_phase, COUNT(*) as count
FROM governance_trace_events
WHERE session_id = 'your-session-id' AND event_type = 'context_decision'
GROUP BY message_type, conversation_phase;
```

---

## Performance Notes

1. **Batch operations are faster** than individual inserts
2. **JSONB fields** allow flexible event payloads
3. **Indexes needed** on (session_id, turn_number) for query performance
4. **Consider partitioning** by session_id for large datasets
5. **Archive old sessions** to separate cold storage quarterly

---

## Privacy Safeguards

```python
# Ensure no content leakage
def record_intervention_event(..., trigger_reason: str, ...):
    # Validate reason doesn't contain user content
    if len(trigger_reason) > 200:
        logger.warning(f"Intervention reason too long: {len(trigger_reason)} chars")
        trigger_reason = trigger_reason[:200]  # Truncate
    
    # Only allow predefined reasons
    valid_reasons = [
        "Basin exit detected",
        "Drift below GREEN",
        "Hard block triggered",
        "Layer 1 baseline violation",
        "Layer 2 basin violation"
    ]
    
    if trigger_reason not in valid_reasons:
        logger.warning(f"Non-standard intervention reason: {trigger_reason}")
```

