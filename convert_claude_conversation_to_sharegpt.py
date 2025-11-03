"""
Convert Claude Conversation to ShareGPT Format

Takes the phase2 analyzed Claude conversation and converts it to ShareGPT format
with proper PA and baseline fidelity measurements.
"""
import json
from pathlib import Path

# Load the phase2 study results to get the PA
print("Loading phase2 study results...")
with open('telos_observatory/phase2_validation_claude_test_1/study_results/phase2_study_summary.json', 'r') as f:
    phase2 = json.load(f)

attractor = phase2['completed_studies'][0]['attractor']
print(f"✓ PA extracted: {attractor['purpose'][0][:60]}...")

# Load the intervention file to get actual turn-by-turn data
print("Loading intervention file with turn data...")
with open('telos_observatory/phase2_validation_claude_test_1/study_results/claude_test_1/intervention_11_213733.json', 'r') as f:
    intervention_data = json.load(f)

# Load the parsed conversation
print("Loading parsed conversation...")
with open('test_sessions/claude_conversation_parsed.json', 'r') as f:
    parsed = json.load(f)

conversations = parsed['conversations']
print(f"✓ Found {len(conversations)} conversation entries")

# Convert to ShareGPT format with proper turn structure
turns = []
turn_num = 1
conv_idx = 0

while conv_idx < len(conversations):
    conv = conversations[conv_idx]

    if conv['from'] == 'human':
        user_input = conv['value']

        # Find the next GPT response
        next_idx = conv_idx + 1
        if next_idx < len(conversations) and conversations[next_idx]['from'] == 'gpt':
            assistant_response = conversations[next_idx]['value']

            # Get fidelity from intervention data if available
            # The intervention file has turn history with fidelity scores
            fidelity = 0.5  # Default placeholder
            if 'turn_history' in intervention_data:
                for turn_entry in intervention_data['turn_history']:
                    if turn_entry.get('turn_number') == turn_num:
                        fidelity = turn_entry.get('metrics', {}).get('fidelity', 0.5)
                        break

            turns.append({
                'turn': turn_num,
                'user_input': user_input,
                'assistant_response_telos': assistant_response,
                'fidelity_telos': fidelity
            })
            turn_num += 1
            conv_idx = next_idx + 1  # Skip the assistant response we just processed
        else:
            conv_idx += 1
    else:
        conv_idx += 1

print(f"✓ Converted {len(turns)} turns")

# Create ShareGPT format session
session = {
    'session_id': 'claude_conversation',
    'primacy_attractor': {
        'purpose': attractor['purpose'],
        'scope': attractor['scope'],
        'boundaries': attractor['boundaries'],
        'privacy_level': 0.8,
        'constraint_tolerance': 0.2,
        'task_priority': 0.7
    },
    'turns': turns
}

# Save to saved_sessions directory
output_file = Path('telos_observatory_v3/saved_sessions/sharegpt_claude_conversation.json')
with open(output_file, 'w') as f:
    json.dump(session, f, indent=2)

print(f"\n✓ Saved to: {output_file}")
print(f"  - Session ID: {session['session_id']}")
print(f"  - Total turns: {len(turns)}")
print(f"  - PA Purpose: {attractor['purpose'][0][:80]}...")
print(f"\nReady for dual PA regeneration!")
