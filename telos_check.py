#!/usr/bin/env python3
"""
TELOS Fidelity Check
====================

Check fidelity of a response against established PA.

Usage:
    python3 telos_check.py "response text here"

Requires:
    .telos_session_pa.json (created by telos_init.py)
"""

import sys
import json
from pathlib import Path
from datetime import datetime
import numpy as np

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from telos_purpose.core.embedding_provider import EmbeddingProvider


def check_fidelity(response_text: str, verbose: bool = True):
    """
    Check fidelity of response against PA

    Args:
        response_text: The response to check
        verbose: Print detailed output

    Returns:
        dict with fidelity, pass/fail, etc.
    """

    # Load PA
    pa_file = Path('.telos_session_pa.json')
    if not pa_file.exists():
        raise FileNotFoundError(
            'PA not established. Run: python3 telos_init.py'
        )

    with open(pa_file) as f:
        pa = json.load(f)

    pa_embedding = np.array(pa['pa_embedding'])
    threshold = pa['threshold']

    # Embed response
    embeddings = EmbeddingProvider(deterministic=False)
    response_embedding = embeddings.encode(response_text)

    # Calculate fidelity (cosine similarity)
    fidelity = np.dot(pa_embedding, response_embedding) / (
        np.linalg.norm(pa_embedding) * np.linalg.norm(response_embedding)
    )

    passed = fidelity >= threshold

    result = {
        'fidelity': float(fidelity),
        'threshold': threshold,
        'passed': passed,
        'response_length': len(response_text),
        'timestamp': datetime.now().isoformat()
    }

    # Log to session
    log_file = Path('.telos_session_log.json')
    if log_file.exists():
        with open(log_file) as f:
            log = json.load(f)
    else:
        log = {'session_start': datetime.now().isoformat(), 'turns': []}

    turn_number = len(log['turns']) + 1

    log['turns'].append({
        'turn': turn_number,
        'fidelity': float(fidelity),
        'passed': bool(passed),
        'timestamp': result['timestamp'],
        'response_preview': response_text[:100]
    })

    with open(log_file, 'w') as f:
        json.dump(log, f, indent=2)

    # Print result
    if verbose:
        print(f"\n📊 Fidelity Check - Turn {turn_number}")
        print("="*60)
        print(f"Response: {response_text[:80]}{'...' if len(response_text) > 80 else ''}")
        print(f"\nFidelity: {fidelity:.3f}")
        print(f"Threshold: {threshold}")
        print(f"Status: {'✅ PASS' if passed else '🚨 DRIFT'}")

        if not passed:
            print(f"\n⚠️  Fidelity below threshold!")
            print(f"   Drift magnitude: {threshold - fidelity:.3f}")
            print(f"   Recommendation: Review alignment with PA")

        print(f"\n💾 Logged to turn {turn_number}")

    return result


def show_session_summary():
    """Show summary of session so far"""

    log_file = Path('.telos_session_log.json')
    if not log_file.exists():
        print("No session log found")
        return

    with open(log_file) as f:
        log = json.load(f)

    turns = log.get('turns', [])
    if not turns:
        print("No turns logged yet")
        return

    fidelities = [t['fidelity'] for t in turns]

    print("\n📊 Session Summary")
    print("="*60)
    print(f"Session started: {log['session_start']}")
    print(f"Total turns: {len(turns)}")
    print(f"Mean fidelity: {sum(fidelities) / len(fidelities):.3f}")
    print(f"Min fidelity: {min(fidelities):.3f}")
    print(f"Max fidelity: {max(fidelities):.3f}")

    drift_turns = [t['turn'] for t in turns if not t['passed']]
    if drift_turns:
        print(f"\n🚨 Drift detected on turns: {drift_turns}")
    else:
        print(f"\n✅ No drift detected")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 telos_check.py \"response text\"")
        print("\nOr to show session summary:")
        print("       python3 telos_check.py --summary")
        sys.exit(1)

    if sys.argv[1] == '--summary':
        show_session_summary()
    else:
        response_text = ' '.join(sys.argv[1:])

        try:
            check_fidelity(response_text)
        except Exception as e:
            print(f"\n❌ Error: {e}")
            sys.exit(1)
