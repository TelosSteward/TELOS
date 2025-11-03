"""
Convert Claude Conversation to Conversation Starters Only

Creates a proper session format with ONLY user inputs (conversation starters).
Dual PA will regenerate ALL responses from scratch - true A/B test.
"""
import json
from pathlib import Path

# Load the phase2 study results to get the PA
print("Loading phase2 study results...")
with open('telos_observatory/phase2_validation_claude_test_1/study_results/phase2_study_summary.json', 'r') as f:
    phase2 = json.load(f)

attractor = phase2['completed_studies'][0]['attractor']
print(f"✓ PA extracted: {attractor['purpose'][0][:60]}...")

# Load the parsed conversation
print("Loading parsed conversation...")
with open('test_sessions/claude_conversation_parsed.json', 'r') as f:
    parsed = json.load(f)

conversations = parsed['conversations']
print(f"✓ Found {len(conversations)} conversation entries")

# Extract ONLY user inputs (conversation starters)
# DO NOT include any existing assistant responses
conversation_starters = []

for conv in conversations:
    if conv['from'] == 'human':
        conversation_starters.append(conv['value'])

print(f"✓ Extracted {len(conversation_starters)} conversation starters (user inputs only)")

# Create session format with ONLY conversation starters
# The dual PA regeneration script will build context as it generates
session = {
    'conversation_id': 'claude_conversation',
    'primacy_attractor': {
        'purpose': attractor['purpose'],
        'scope': attractor['scope'],
        'boundaries': attractor['boundaries'],
        'privacy_level': 0.8,
        'constraint_tolerance': 0.2,
        'task_priority': 0.7
    },
    'conversation_starters': conversation_starters  # ONLY user inputs, no responses
}

# Save to a new location
output_file = Path('claude_conversation_starters_only.json')
with open(output_file, 'w') as f:
    json.dump(session, f, indent=2)

print(f"\n✓ Saved to: {output_file}")
print(f"  - Session ID: {session['conversation_id']}")
print(f"  - Conversation starters: {len(conversation_starters)}")
print(f"  - PA Purpose: {attractor['purpose'][0][:80]}...")
print(f"\nReady for FRESH dual PA regeneration (no existing responses)!")
