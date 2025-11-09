# Beta A/B Testing Architecture

## Overview

TELOS Observatory Beta implements a two-phase preference testing system:
- **Phase 1 (Turns 1-10)**: PA calibration with baseline LLM only
- **Phase 2 (Turns 11+)**: A/B testing with dual-response generation

## Key Architecture Decision: Single Conversation History

**Critical Requirement:** Both baseline and TELOS models must see the SAME conversation history - what the user actually saw.

### Why This Matters

If we track separate histories for baseline and TELOS paths:
- TELOS sees responses it generated but user never saw
- TELOS responds to a conversation that didn't happen from user's perspective
- Context contamination breaks the experiment validity

### The Solution: Stateless Dual Generation

Each turn:
1. **Generate baseline response** using current conversation history
2. **Generate TELOS response** using SAME conversation history
3. **Randomly choose** which response to show user
4. **Update THE SINGLE history** with what was actually shown
5. **Log both responses** for research (backend only)
6. **Collect feedback** on shown response only

## Implementation Flow

```python
def generate_beta_response(message, turn_idx):
    # Get current conversation history (what user saw)
    history = build_history_from_shown_responses()

    # Phase 1 (turns 1-10): PA calibration - baseline only
    if turn_idx < 10:
        baseline_response = llm.generate(history + [message])
        telos_result = telos_steward.process_turn(message, baseline_response)
        shown_response = baseline_response  # Always show baseline during calibration

        # Store metrics but don't show comparison
        store_turn_data(turn_idx, {
            'response': shown_response,
            'fidelity': telos_result['telic_fidelity'],
            'pa_status': 'calibrating'
        })
        return shown_response

    # Phase 2 (turns 11+): A/B testing
    else:
        # Generate both responses fresh using same context
        baseline_response = llm.generate(history + [message])
        telos_result = telos_steward.process_turn(message, baseline_response)
        telos_response = telos_result['final_response']

        # Randomly assign test condition (from session manager)
        test_condition = beta_session.test_condition

        if test_condition == "single_blind_baseline":
            shown_response = baseline_response
            response_source = "baseline"
        elif test_condition == "single_blind_telos":
            shown_response = telos_response
            response_source = "telos"
        else:  # head_to_head
            # Show both, let user choose
            shown_response = None  # Special handling
            response_source = "both"

        # Store turn data with BOTH responses (research only)
        store_turn_data(turn_idx, {
            'shown_response': shown_response,
            'baseline_response': baseline_response,  # Hidden
            'telos_response': telos_response,  # Hidden
            'response_source': response_source,  # Hidden
            'fidelity': telos_result['telic_fidelity'],
            'pa_status': 'established',
            'test_condition': test_condition
        })

        # After response shown, render feedback UI
        render_feedback_ui(turn_idx, test_condition)

        return shown_response
```

## Data Privacy

**Stored for research:**
- Both responses (baseline and TELOS)
- Which was shown to user
- User's feedback (thumbs up/down or preference)
- Fidelity metrics

**NOT stored:**
- User's conversation content (privacy-first)
- Personal identifying information
- API keys or credentials

## Beta Completion

After 2 weeks OR 50 turns with feedback:
- Set `st.session_state.beta_completed = True`
- Unlock DEMO and TELOS tabs
- Show transition message: "A/B testing complete! Full TELOS governance now active."
- Continue using the established PA from calibration phase

## Files Modified

1. `observatory/core/state_manager.py`: Add `generate_beta_response()` method
2. `observatory/components/conversation_display.py`: Integrate feedback UI rendering
3. `observatory/main.py`: Initialize beta session manager on consent

## Test Condition Assignment

From `beta_session_manager.py`:
- 40% single-blind baseline
- 40% single-blind TELOS
- 20% head-to-head comparison

Randomly assigned per conversation (not per turn).
