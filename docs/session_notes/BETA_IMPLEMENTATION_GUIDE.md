# TELOS BETA Implementation Guide
**Using Existing Observatory Controls for Post-Session Review**

---

## Architecture Overview

**Key Simplification:** Post-session review uses EXISTING Observatory components (Observation Deck + Alignment Lens). No new review UI needed.

**Flow:**
1. User completes BETA session (PA + AB Testing + optional Full TELOS)
2. All data logged to Supabase during session
3. Post-session: Load BETA data into existing Observatory
4. User cycles through turns using existing controls
5. Metrics display via existing Observation Deck + Alignment Lens

---

## Implementation Components

### 1. Expedited PA Establishment (1-2 Turns)

**Location:** `components/beta_onboarding.py` (new)

**Flow:**
```python
# Turn 1: User states goal
user_input = "I want to debug my Python API authentication issue"

# Extract PA components using LLM
pa_components = extract_pa_from_statement(user_input)
# Returns: {
#   "purpose": ["Debug Python API authentication"],
#   "scope": ["API", "authentication", "Python", "debugging"],
#   "boundaries": ["Stay focused on auth issue", "Avoid unrelated topics"]
# }

# Turn 2: Confirm or refine
show_pa_confirmation(pa_components)
user_confirms = True  # or refines

# Derive AI PA (hidden from user during session)
ai_pa = derive_ai_pa_from_user_pa(pa_components)

# Save to session state and Supabase
save_beta_session(user_pa, ai_pa, basin_constant=1.0, constraint_tolerance=0.05)
```

**Files to Create:**
- `components/beta_pa_establishment.py` - PA extraction + confirmation UI
- `services/pa_extractor.py` - LLM-based PA extraction from user statement

---

### 2. AB Testing Phase (5-10 Turns)

**Location:** `components/beta_ab_testing.py` (new)

**Turn Flow:**
```python
# For each turn in AB phase:
def process_ab_turn(user_input, session_id, turn_number):
    # Random selection: TELOS or Native
    system_selected = random.choice(['telos', 'native'])

    # CRITICAL: Calculate drift for BOTH systems
    telos_metrics = calculate_telos_governance(user_input, user_pa, ai_pa)
    native_metrics = calculate_baseline_metrics(user_input, user_pa)

    # Generate responses from both systems
    telos_response = generate_telos_response(user_input, telos_metrics)
    native_response = generate_native_response(user_input)

    # Serve selected system's response
    response_delivered = telos_response if system_selected == 'telos' else native_response

    # Log to Supabase
    log_beta_turn({
        'session_id': session_id,
        'turn_number': turn_number,
        'phase': 'ab_testing',
        'user_message': user_input,
        'system_served': system_selected,
        'telos_response': telos_response,
        'native_response': native_response,
        'response_delivered': response_delivered,
        'user_fidelity': telos_metrics['user_fidelity'],
        'ai_fidelity': telos_metrics['ai_fidelity'],
        'primacy_state': telos_metrics['primacy_state'],
        'distance_from_pa': telos_metrics['distance'],
        'in_basin': telos_metrics['in_basin'],
        'intervention_calculated': telos_metrics['would_intervene'],
        'intervention_applied': system_selected == 'telos' and telos_metrics['would_intervene'],
        'intervention_type': telos_metrics.get('intervention_type'),
    })

    # Generate Steward interpretation ASYNC (don't block)
    async_generate_steward_interpretation(session_id, turn_number, telos_metrics, user_input)

    return response_delivered
```

**User Actions:**
```python
# Thumbs up/down
def handle_feedback(session_id, turn_number, feedback_type):
    update_turn_feedback(session_id, turn_number, user_action=feedback_type)

# Regenerate (switches systems)
def handle_regenerate(session_id, turn_number):
    # Get original turn data
    turn_data = get_turn_data(session_id, turn_number)

    # Switch systems
    new_system = 'native' if turn_data['system_served'] == 'telos' else 'telos'
    new_response = turn_data['native_response'] if new_system == 'native' else turn_data['telos_response']

    # Update turn with preference data
    update_turn(session_id, turn_number,
                system_served=new_system,
                response_delivered=new_response,
                user_preference=f'selected_{new_system}')

    return new_response
```

**Files to Create:**
- `components/beta_ab_testing.py` - AB phase UI + turn processing
- `services/beta_turn_processor.py` - Turn logic (random selection, dual calculation)
- `services/steward_interpreter.py` - Async Steward interpretation generation

---

### 3. Full TELOS Phase (Optional, 5-10 Turns)

**Location:** `components/beta_full_telos.py` (new)

**Simpler than AB - just run full governance:**
```python
def process_full_telos_turn(user_input, session_id, turn_number):
    # Calculate governance
    telos_metrics = calculate_telos_governance(user_input, user_pa, ai_pa)

    # Generate governed response
    telos_response = generate_telos_response(user_input, telos_metrics)

    # Apply intervention if needed (now visible to user)
    if telos_metrics['would_intervene']:
        show_intervention_warning(telos_metrics['intervention_type'])

    # Log to Supabase
    log_beta_turn({
        'session_id': session_id,
        'turn_number': turn_number,
        'phase': 'full_telos',
        'system_served': 'telos',
        'intervention_applied': telos_metrics['would_intervene'],
        # ... all metrics
    })

    # Steward interpretation async
    async_generate_steward_interpretation(session_id, turn_number, telos_metrics, user_input)

    return telos_response
```

**Phase Transition:**
```python
def offer_phase_2_continuation():
    st.info("Phase 1 (AB Testing) complete! You've completed 10 turns.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Continue to Full TELOS (Optional)", key="continue_phase2"):
            st.session_state.beta_phase = 'full_telos'
            st.rerun()

    with col2:
        if st.button("Finish & View Results", key="finish_beta"):
            st.session_state.beta_phase = 'review'
            st.rerun()
```

---

### 4. Post-Session Review (Use Existing Observatory)

**Location:** Modify `components/observation_deck.py` and `components/observatory_lens.py`

**Key Insight:** These components ALREADY display turn data, metrics, and visualizations. We just need to:
1. Load BETA session data from Supabase
2. Format it to match existing Observatory data structure
3. Let existing components render it

**Implementation:**
```python
def load_beta_session_for_review(session_id):
    """Load completed BETA session data and format for Observatory."""

    # Fetch from Supabase
    session_data = supabase.table('beta_sessions').select('*').eq('session_id', session_id).single()
    turns_data = supabase.table('beta_turns').select('*').eq('session_id', session_id).order('turn_number')

    # Format as Observatory-compatible structure
    observatory_data = {
        'session_id': session_id,
        'user_pa': session_data['user_pa_config'],
        'ai_pa': session_data['ai_pa_config'],  # Revealed post-session
        'turns': []
    }

    for turn in turns_data:
        observatory_data['turns'].append({
            'turn_number': turn['turn_number'],
            'user_message': turn['user_message'],
            'assistant_response': turn['response_delivered'],
            'system_served': turn['system_served'],  # Extra BETA info
            'user_fidelity': turn['user_fidelity'],
            'ai_fidelity': turn['ai_fidelity'],
            'primacy_state': turn['primacy_state'],
            'distance': turn['distance_from_pa'],
            'in_basin': turn['in_basin'],
            'intervention_calculated': turn['intervention_calculated'],
            'intervention_applied': turn['intervention_applied'],
            'steward_interpretation': turn['steward_interpretation'],  # New!
            'user_action': turn['user_action'],
            'user_preference': turn['user_preference']
        })

    return observatory_data

# In main review page:
def show_beta_review():
    """Show post-session review using existing Observatory."""

    session_id = st.session_state.beta_session_id
    observatory_data = load_beta_session_for_review(session_id)

    st.title("BETA Session Review")

    # Show User PA (was visible during session)
    with st.expander("Your Primacy Attractor", expanded=True):
        st.write("**Purpose:**", observatory_data['user_pa']['purpose'])
        st.write("**Scope:**", observatory_data['user_pa']['scope'])
        st.write("**Boundaries:**", observatory_data['user_pa']['boundaries'])

    # Reveal AI PA (was hidden during session)
    with st.expander("AI Primacy Attractor (Revealed)", expanded=False):
        st.info("This shows how the AI adapted its behavior to serve your purpose.")
        st.write("**Purpose:**", observatory_data['ai_pa']['purpose'])
        st.write("**Scope:**", observatory_data['ai_pa']['scope'])

    # EXISTING Observatory controls for turn navigation
    from components.observation_deck import ObservationDeck
    from components.observatory_lens import ObservatoryLens

    # Let user cycle through turns with existing controls
    observation_deck = ObservationDeck(observatory_data)
    observation_deck.render()  # Uses existing turn navigation

    # Show metrics with existing Alignment Lens
    observatory_lens = ObservatoryLens(observatory_data)
    observatory_lens.render()  # Uses existing visualizations

    # Add BETA-specific info (system served, preferences)
    show_beta_specific_metrics(observatory_data)
```

**BETA-Specific Additions:**
```python
def show_beta_specific_metrics(observatory_data):
    """Show AB testing results and preference data."""

    st.subheader("Your Preferences")

    turns = observatory_data['turns']
    telos_turns = [t for t in turns if t['system_served'] == 'telos']
    native_turns = [t for t in turns if t['system_served'] == 'native']

    thumbs_up_telos = len([t for t in telos_turns if t['user_action'] == 'thumbs_up'])
    thumbs_up_native = len([t for t in native_turns if t['user_action'] == 'thumbs_up'])

    col1, col2 = st.columns(2)
    with col1:
        st.metric("TELOS Thumbs Up", f"{thumbs_up_telos}/{len(telos_turns)}")
    with col2:
        st.metric("Native Thumbs Up", f"{thumbs_up_native}/{len(native_turns)}")

    # Show which system you preferred
    preferences = [t.get('user_preference') for t in turns if t.get('user_preference')]
    if preferences:
        telos_preferred = preferences.count('selected_telos')
        native_preferred = preferences.count('selected_native')
        st.write(f"When you regenerated: Chose TELOS {telos_preferred} times, Native {native_preferred} times")
```

**Files to Modify:**
- `components/observation_deck.py` - Add support for loading BETA session data
- `components/observatory_lens.py` - Add support for BETA metrics
- `components/beta_review.py` (NEW) - Post-session review orchestration

---

### 5. Steward Interpretation Logging

**Location:** `services/steward_interpreter.py` (new)

**Async Generation:**
```python
import asyncio
from telos_purpose.llm_clients.mistral_client import MistralClient

async def generate_steward_interpretation(session_id, turn_number, metrics, user_input, user_pa):
    """Generate Steward's interpretation of why drift was calculated."""

    client = MistralClient(model="mistral-small-latest")

    prompt = f"""You are Steward, explaining TELOS governance calculations.

User's Primacy Attractor:
Purpose: {user_pa['purpose']}
Scope: {user_pa['scope']}
Boundaries: {user_pa['boundaries']}

User asked: "{user_input}"

TELOS calculated:
- User fidelity: {metrics['user_fidelity']:.3f}
- Semantic distance: {metrics['distance']:.3f}
- Basin radius: {metrics['basin_radius']:.3f}
- In basin: {metrics['in_basin']}

Explain in 2-3 sentences WHY this fidelity score was calculated based on semantic analysis."""

    try:
        interpretation = await client.generate_async(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=200
        )

        # Save to Supabase
        supabase.table('beta_turns').update({
            'steward_interpretation': interpretation
        }).eq('session_id', session_id).eq('turn_number', turn_number).execute()

    except Exception as e:
        logger.error(f"Steward interpretation failed: {e}")
        # Store error marker
        supabase.table('beta_turns').update({
            'steward_interpretation': '[Interpretation unavailable]'
        }).eq('session_id', session_id).eq('turn_number', turn_number).execute()

# Call from turn processing (non-blocking)
def async_generate_steward_interpretation(session_id, turn_number, metrics, user_input, user_pa):
    """Non-blocking wrapper."""
    asyncio.create_task(
        generate_steward_interpretation(session_id, turn_number, metrics, user_input, user_pa)
    )
```

---

## Supabase Schema

### beta_sessions Table
```sql
CREATE TABLE beta_sessions (
    session_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_pa_config JSONB NOT NULL,
    ai_pa_config JSONB NOT NULL,
    basin_constant FLOAT DEFAULT 1.0,
    constraint_tolerance FLOAT DEFAULT 0.05,
    phase_1_complete BOOLEAN DEFAULT FALSE,
    phase_2_complete BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP,
    total_turns INTEGER DEFAULT 0
);
```

### beta_turns Table
```sql
CREATE TABLE beta_turns (
    turn_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID REFERENCES beta_sessions(session_id),
    turn_number INTEGER NOT NULL,
    phase TEXT CHECK (phase IN ('pa_establishment', 'ab_testing', 'full_telos')),
    user_message TEXT NOT NULL,
    system_served TEXT CHECK (system_served IN ('telos', 'native')),
    telos_response TEXT NOT NULL,
    native_response TEXT NOT NULL,
    response_delivered TEXT NOT NULL,
    user_fidelity FLOAT,
    ai_fidelity FLOAT,
    primacy_state FLOAT,
    distance_from_pa FLOAT,
    in_basin BOOLEAN,
    intervention_calculated BOOLEAN DEFAULT FALSE,
    intervention_applied BOOLEAN DEFAULT FALSE,
    intervention_type TEXT,
    steward_interpretation TEXT,
    user_action TEXT CHECK (user_action IN ('thumbs_up', 'thumbs_down', 'regenerate', 'none')),
    user_preference TEXT CHECK (user_preference IN ('selected_telos', 'selected_native', 'no_preference')),
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(session_id, turn_number)
);
```

---

## Implementation Checklist

### MVP (Phase 1)
- [ ] Create Supabase tables (beta_sessions, beta_turns)
- [ ] Build PA establishment UI (1-2 turn flow)
- [ ] Implement PA extraction from user statement (LLM)
- [ ] Build AB testing mechanics (random 50/50, dual calculation)
- [ ] Create turn processor (saves to Supabase)
- [ ] Add thumbs up/down tracking
- [ ] Add regenerate button (switches systems)
- [ ] Implement Steward interpretation (async)
- [ ] Modify Observatory to load BETA data
- [ ] Add BETA-specific metrics to review
- [ ] Test complete flow (PA → AB → Review)

### V1.1 (Phase 2)
- [ ] Build Full TELOS phase UI
- [ ] Add phase transition choice UI
- [ ] Implement visible interventions in Phase 2
- [ ] Add progress trackers (X/10 turns)
- [ ] Enhanced review visualizations

### V1.2 (Polish)
- [ ] Download options (JSON, PDF)
- [ ] Session resumption support
- [ ] Governance insights dashboard
- [ ] Comparison graphs (TELOS vs Native quality)

---

## Key Files to Create

**New Components:**
- `components/beta_pa_establishment.py` - PA extraction + confirmation
- `components/beta_ab_testing.py` - AB phase UI
- `components/beta_full_telos.py` - Full TELOS phase UI
- `components/beta_review.py` - Post-session review orchestrator

**New Services:**
- `services/pa_extractor.py` - Extract PA from user statement
- `services/beta_turn_processor.py` - Turn logic (random, dual calc)
- `services/steward_interpreter.py` - Async Steward interpretation

**Modified Components:**
- `components/observation_deck.py` - Support BETA data loading
- `components/observatory_lens.py` - Support BETA metrics
- `telos_purpose/core/primacy_math.py` - Already updated (basin constant = 1.0)

**SQL Scripts:**
- `sql/beta_schema.sql` - Create Supabase tables

---

## Success Criteria

1. ✅ User can establish PA in 1-2 turns (not 5-10)
2. ✅ AB testing randomly serves TELOS/Native (user unaware)
3. ✅ Both systems calculate drift every turn
4. ✅ Steward interpretation generated for all turns
5. ✅ Post-session: Existing Observatory shows all data
6. ✅ User can cycle through turns with existing controls
7. ✅ Metrics display via existing Observation Deck + Lens
8. ✅ Session completes in one sitting (ephemeral)
9. ✅ Users bring real work (not forced scenarios)
10. ✅ Data saved to Supabase for research

---

## Notes

- **Existing Components:** Observation Deck + Alignment Lens already built, just need BETA data
- **Granular Features:** Deferred to later (comparison graphs, detailed Steward UI, etc.)
- **Research Use:** Complete data in Supabase for future analysis
- **User Experience:** Clean live session, rich post-review insights
- **Basin Constant:** 1.0 proven to catch meaningful drift (quantum physics = 0.696)
- **Constraint Tolerance:** 0.05 for strict but reasonable governance
