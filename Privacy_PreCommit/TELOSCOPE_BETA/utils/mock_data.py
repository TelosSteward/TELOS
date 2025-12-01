"""
Observatory V2 - Mock Data Generator
=====================================

Generates realistic mock session data for testing.
"""

from typing import Dict, Any, List
import random


def generate_mock_session(num_turns: int = 10) -> Dict[str, Any]:
    """
    Generate mock session data.

    Args:
        num_turns: Number of turns to generate

    Returns:
        Dictionary with session_id, turns, and statistics
    """
    turns = []
    interventions = 0
    drift_count = 0
    fidelities = []

    for i in range(num_turns):
        # Generate realistic fidelity score (occasional drift)
        base_fidelity = 0.9
        drift_amount = random.uniform(-0.3, 0.1)
        fidelity = max(0.3, min(1.0, base_fidelity + drift_amount))
        fidelities.append(fidelity)

        # Determine if intervention applied (Goldilocks: Drift Detected threshold)
        intervention_applied = fidelity < 0.67
        if intervention_applied:
            interventions += 1

        # Determine if drift detected (Goldilocks: Aligned threshold)
        drift_detected = fidelity < 0.76
        if drift_detected:
            drift_count += 1

        turn = {
            'turn': i,
            'timestamp': i * 2.5,  # Simulated timestamps
            'user_input': f"This is a sample user message for turn {i}. It contains some text about the conversation topic.",
            'response': f"This is the assistant's response to turn {i}. It addresses the user's message appropriately.",
            'fidelity': fidelity,
            'distance': (1 - fidelity) * 2.0,  # Simplified distance calculation
            'threshold': 0.76,  # Goldilocks: Aligned threshold
            'intervention_applied': intervention_applied,
            'drift_detected': drift_detected,
            'status': _get_status_icon(fidelity),
            'status_text': _get_status_text(fidelity),
            'phase2_comparison': _generate_phase2_data() if intervention_applied else None
        }

        turns.append(turn)

    # Calculate statistics
    avg_fidelity = sum(fidelities) / len(fidelities) if fidelities else 0.0

    return {
        'session_id': 'mock_session_001',
        'turns': turns,
        'statistics': {
            'avg_fidelity': avg_fidelity,
            'interventions': interventions,
            'drift_warnings': drift_count
        }
    }


def _get_status_icon(fidelity: float) -> str:
    """Get status icon based on fidelity (Goldilocks zones)."""
    if fidelity >= 0.76:  # Goldilocks: Aligned
        return "✓"
    elif fidelity >= 0.73:  # Goldilocks: Minor Drift
        return "⚠"
    elif fidelity >= 0.67:  # Goldilocks: Drift Detected
        return "⚠️"
    else:  # Goldilocks: Significant Drift
        return "🔴"


def _get_status_text(fidelity: float) -> str:
    """Get status text based on fidelity (Goldilocks zones)."""
    if fidelity >= 0.76:  # Goldilocks: Aligned
        return "Aligned"
    elif fidelity >= 0.73:  # Goldilocks: Minor Drift
        return "Minor Drift"
    elif fidelity >= 0.67:  # Goldilocks: Drift Detected
        return "Drift Detected"
    else:  # Goldilocks: Significant Drift
        return "Significant Drift"


def _generate_phase2_data() -> Dict[str, Any]:
    """Generate mock Phase 2 comparison data."""
    original_fidelity = random.uniform(0.4, 0.7)
    telos_fidelity = random.uniform(0.7, 0.95)

    return {
        'original_fidelity': original_fidelity,
        'telos_fidelity': telos_fidelity,
        'improvement': telos_fidelity - original_fidelity
    }
