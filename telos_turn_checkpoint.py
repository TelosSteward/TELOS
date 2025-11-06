#!/usr/bin/env python3
"""
Runtime Governance - Turn Checkpoint
Measures fidelity using actual mathematics (embeddings + cosine similarity)
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np

try:
    from telos_purpose.core.embedding_provider import EmbeddingProvider
except ImportError:
    print("❌ Error: embedding_provider.py not found")
    print("Make sure TELOS is installed properly.")
    sys.exit(1)

def load_session():
    """Load active session info"""
    session_file = Path('.telos_active_session.json')
    if not session_file.exists():
        print("❌ Error: No active session found.")
        print("Run: python3 telos_session_start.py")
        sys.exit(1)

    with open(session_file, 'r') as f:
        return json.load(f)

def calculate_fidelity(response_text, pa_text, embeddings):
    """
    Calculate cosine similarity fidelity (ACTUAL MATH)
    F = cos(response_emb, PA_emb)
    """
    # Generate embeddings (ℝ³⁸⁴)
    response_emb = embeddings.encode(response_text)
    pa_emb = embeddings.encode(pa_text)

    # Cosine similarity
    dot_product = np.dot(response_emb, pa_emb)
    norm_response = np.linalg.norm(response_emb)
    norm_pa = np.linalg.norm(pa_emb)

    fidelity = dot_product / (norm_response * norm_pa)
    return float(fidelity)

def analyze_turn(user_message, assistant_response, turn_number, session_info):
    """Analyze a single turn with ACTUAL mathematics"""

    print("="*60)
    print(f"🔍 Turn {turn_number} Checkpoint")
    print("="*60)

    # Load embedding provider (local, no API calls)
    print("   Loading embedding provider...")
    embeddings = EmbeddingProvider(deterministic=False)

    # Calculate fidelity
    pa_text = session_info['pa_baseline']
    print(f"   PA baseline: {pa_text[:60]}...")
    print("   Calculating fidelity...")

    fidelity = calculate_fidelity(assistant_response, pa_text, embeddings)

    # Determine status and interpretation
    if fidelity >= 0.9:
        status = "on_track"
        icon = "✅"
        status_text = "EXCELLENT"
        interpretation = (
            "Your work is strongly aligned with your stated goals. "
            "This conversation directly advances your primary objectives."
        )
    elif fidelity >= 0.8:
        status = "on_track"
        icon = "✅"
        status_text = "ON TRACK"
        interpretation = (
            "Good alignment. Your work is moving in the right direction "
            "and supporting your core project goals."
        )
    elif fidelity >= 0.7:
        status = "warning"
        icon = "⚠️"
        status_text = "MODERATE DRIFT"
        interpretation = (
            "Some drift detected. This work is tangentially related but may not "
            "be directly advancing your primary objectives. Consider: Is this the "
            "best use of your time right now?"
        )
    elif fidelity >= 0.5:
        status = "drift"
        icon = "🚨"
        status_text = "SIGNIFICANT DRIFT"
        interpretation = (
            "Noticeable drift from your goals. This conversation is exploring "
            "topics outside your core focus. If this is intentional exploration, "
            "that's fine - just be aware you've shifted away from your stated priorities."
        )
    else:
        status = "drift"
        icon = "🚨"
        status_text = "MAJOR DRIFT"
        interpretation = (
            "This work has minimal alignment with your stated goals. You may be: "
            "(1) Exploring tangential topics, (2) Working on infrastructure/meta-tasks, "
            "or (3) Need to update your PA baseline to reflect new priorities. "
            "Review: Does your PA still capture what you're trying to accomplish?"
        )

    # Display results
    print(f"\n📈 Results:")
    print(f"   Fidelity: {fidelity:.3f} {icon}")
    print(f"   Status: {status_text}")
    print(f"\n💡 What this means:")
    print(f"   {interpretation}")

    # Signal to Claude to store in Memory MCP
    session_id = session_info["session_id"]
    session_name = session_info["session_name"]
    timestamp = datetime.now().isoformat()

    print(f"\n⚠️  CLAUDE: Store turn {turn_number} in Memory MCP")
    print(f"\nUse: mcp__memory__create_entities([{{")
    print(f'  "name": "Turn_{turn_number}",')
    print(f'  "entityType": "Turn",')
    print(f'  "observations": [')
    print(f'    "turn_number: {turn_number}",')
    print(f'    "session_id: {session_id}",')
    print(f'    "timestamp: {timestamp}",')
    print(f'    "user_message: ...",')
    print(f'    "assistant_response_length: {len(assistant_response)}",')
    print(f'    "fidelity: {fidelity:.3f}",')
    print(f'    "status: {status}"')
    print(f'  ]')
    print(f'}}])')
    print(f"\nAnd: mcp__memory__create_relations([{{")
    print(f'  "from": "{session_name}",')
    print(f'  "to": "Turn_{turn_number}",')
    print(f'  "relationType": "has_turn"')
    print(f'}}])')

    # Update session file
    session_info['turn_count'] = turn_number
    session_info['last_fidelity'] = fidelity
    session_info['last_updated'] = datetime.now().isoformat()

    with open('.telos_active_session.json', 'w') as f:
        json.dump(session_info, f, indent=2)

    # Save checkpoint
    checkpoint = {
        'turn_number': turn_number,
        'fidelity': fidelity,
        'status': status,
        'timestamp': datetime.now().isoformat()
    }

    checkpoint_file = f'.telos_checkpoint_{turn_number}.json'
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint, f, indent=2)

    # Display summary with session stats
    print(f"\n{'='*60}")
    print(f"📊 TURN {turn_number} SUMMARY")
    print(f"{'='*60}")
    print(f"Fidelity Score: {fidelity:.3f} {icon} {status_text}")

    # Calculate running statistics
    if turn_number > 1:
        # Try to load previous checkpoints for mean
        fidelities = [fidelity]
        for i in range(1, turn_number):
            cp_file = Path(f'.telos_checkpoint_{i}.json')
            if cp_file.exists():
                with open(cp_file, 'r') as f:
                    cp_data = json.load(f)
                    fidelities.append(cp_data['fidelity'])

        mean_fidelity = sum(fidelities) / len(fidelities)
        print(f"Session Mean: {mean_fidelity:.3f}")
        print(f"Total Turns: {turn_number}")

    print(f"\n💾 Checkpoint saved: {checkpoint_file}")
    print(f"{'='*60}")

    return checkpoint

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Runtime Governance Turn Checkpoint')
    parser.add_argument('--user', type=str, help='User message')
    parser.add_argument('--assistant', type=str, help='Assistant response')
    parser.add_argument('--turn-json', type=str, help='Path to turn JSON file')

    args = parser.parse_args()

    # Load session
    session_info = load_session()

    # Get turn data
    if args.turn_json:
        with open(args.turn_json, 'r') as f:
            turn_data = json.load(f)
        user_message = turn_data.get('user_message', '')
        assistant_response = turn_data.get('assistant_response', '')
        turn_number = session_info['turn_count'] + 1
    elif args.user and args.assistant:
        user_message = args.user
        assistant_response = args.assistant
        turn_number = session_info['turn_count'] + 1
    else:
        print("❌ Error: Must provide either --turn-json or both --user and --assistant")
        print("\nUsage:")
        print('  python3 runtime_governance_checkpoint.py --user "msg" --assistant "response"')
        print('  python3 runtime_governance_checkpoint.py --turn-json turn_data.json')
        sys.exit(1)

    # Analyze turn
    checkpoint = analyze_turn(user_message, assistant_response, turn_number, session_info)

    return 0

if __name__ == "__main__":
    sys.exit(main())
