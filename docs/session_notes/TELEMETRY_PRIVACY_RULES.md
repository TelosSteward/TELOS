# Telemetry Privacy Rules - CRITICAL

## When to Collect Semantic Telemetry

**ONLY collect when BOTH conditions are true:**

1. ✅ **Beta consent given** - User explicitly opted into research
2. ✅ **TELOS governance active** - PA evaluation running on live Mistral LLM

## When NOT to Collect

❌ **Demo Mode** - No telemetry, no Supabase transmission
❌ **Open Mode without governance** - Just chatting, no PA evaluation
❌ **Steward Panel** - Internal tool usage, not governed conversation
❌ **General Mistral usage** - LLM responses without TELOS governance

## Implementation Guard

```python
def should_transmit_telemetry(mode: str, governance_active: bool, consent_given: bool) -> bool:
    """
    Determine if telemetry should be transmitted.

    Args:
        mode: Current mode ('demo', 'beta', 'open')
        governance_active: Is TELOS governance evaluating this turn?
        consent_given: Has user given beta consent?

    Returns:
        True only if beta mode + governance active + consent given
    """
    # CRITICAL: All three must be true
    return (
        mode == 'beta' and
        governance_active and
        consent_given
    )
```

## Example Usage in Observatory

```python
# At the start of conversation turn processing
mode = st.session_state.get('mode', 'demo')
consent_given = st.session_state.get('beta_consent_given', False)
governance_active = (mode == 'beta' and consent_given)

# Check if TELOS governance should run
if governance_active:
    # User is in beta mode with consent - TELOS will govern

    # Initialize turn tracker with privacy parameters
    tracker = TurnTracker(
        session_id=st.session_state.session_id,
        turn_number=current_turn,
        mode='beta',
        governance_active=True,
        consent_given=consent_given
    )

    # Get user input
    user_message = st.session_state.user_input

    # Track lifecycle: calculating PA
    tracker.mark_calculating_pa()

    # Run TELOS governance evaluation
    pa_distance = calculate_pa_distance(user_message, pa_config)

    # Track lifecycle: evaluating
    tracker.mark_evaluating()

    # Evaluate governance metrics
    governance_metrics = evaluate_governance(user_message, pa_distance)

    # Get LLM response (governed by TELOS)
    assistant_response = get_mistral_response(user_message)

    # Extract semantic context
    semantic_context = SemanticAnalyzer.analyze_turn(
        user_message=user_message,
        governance_metrics=governance_metrics,
        pa_config=pa_config,
        assistant_response=assistant_response
    )

    # Complete turn with full telemetry
    tracker.complete_turn(
        governance_metrics=governance_metrics,
        semantic_context=semantic_context
    )
else:
    # NOT beta mode or no consent - just run LLM, no tracking
    governance_active = False

    # Get LLM response (NOT governed by TELOS)
    assistant_response = get_mistral_response(user_message)

    # NO telemetry transmission
```

## Consent Check

Before any telemetry collection:

```python
# Check for beta consent in session state
consent_given = st.session_state.get('beta_consent_given', False)

if not consent_given:
    # DO NOT collect telemetry
    # DO NOT transmit to Supabase
    # DO NOT track lifecycle
    pass
```

## Steward Panel - NO TRACKING

```python
# Steward panel interactions
if steward_panel.is_active():
    # This is internal tooling, NOT a governed conversation
    # NO telemetry collection
    # NO Supabase transmission
    steward_response = steward.process_query(user_query)
```

## Summary

**Telemetry Trigger:**
```
IF mode == 'beta'
AND beta_consent_given == True
AND TELOS_governance_running == True
THEN collect semantic telemetry
ELSE do not collect anything
```

**Privacy Guarantee:**
- Demo mode users: ZERO data collection
- Open mode users: ZERO data collection
- Beta users without consent: ZERO data collection
- Steward panel usage: ZERO data collection
- Only beta users with explicit consent who are actively using TELOS governance: data collected

This ensures telemetry ONLY captures what you need: how TELOS governance performs on real conversations with user consent.
