"""
Test Session Data Generator for TELOS Observatory
==================================================

Generates realistic test conversation sessions with various governance patterns
for comprehensive platform testing.

Usage:
    python -m telos_purpose.test_data.generate_test_sessions --output test_sessions/
    python -m telos_purpose.test_data.generate_test_sessions --seed 42  # Reproducible
"""

import json
import random
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional


# Global random instance for reproducibility
_rng = random.Random()


def set_seed(seed: Optional[int] = None) -> None:
    """
    Set random seed for reproducible test generation.

    Args:
        seed: Random seed value. If None, uses system entropy.
    """
    global _rng
    if seed is not None:
        _rng = random.Random(seed)
    else:
        _rng = random.Random()


def generate_test_turn(
    turn_num: int,
    current_fidelity: float,
    base_timestamp: datetime,
    intervention: Optional[str] = None
) -> Dict[str, Any]:
    """Generate a single test turn with realistic data."""

    user_messages = [
        "Can you explain how AI governance works?",
        "What are the key principles of TELOS?",
        "How does drift detection function?",
        "Tell me about intervention strategies",
        "What makes governance persistent?",
        "How do you measure fidelity?",
        "Explain the Primacy Attractor concept",
        "What is basin membership?",
        "How does regeneration work?",
        "What are the mathematical foundations?",
    ]

    native_responses = [
        "Here's a general explanation about AI systems and their operation...",
        "AI models process information using neural networks and patterns...",
        "The system analyzes data through various computational methods...",
        "Information processing involves multiple layers of analysis...",
        "Understanding AI requires knowledge of machine learning fundamentals...",
    ]

    telos_responses = [
        "TELOS governance ensures alignment through mathematical measurement of purpose fidelity...",
        "The Primacy Attractor framework maintains governance by tracking semantic drift in embedding space...",
        "Intervention strategies include salience reinforcement and coupling verification to maintain fidelity...",
        "Basin membership indicates whether responses remain within the governance perimeter defined by purpose and scope...",
        "Fidelity measurement uses cosine similarity between response embeddings and the attractor center...",
    ]

    user_msg = _rng.choice(user_messages) + f" (Turn {turn_num})"
    native_resp = _rng.choice(native_responses)
    telos_resp = _rng.choice(telos_responses) if current_fidelity > 0.65 else native_resp

    turn_data = {
        'turn_number': turn_num,
        'user_message': user_msg,  # Primary field
        'native_response': native_resp,
        'telos_response': telos_resp,
        'assistant_response': telos_resp,  # What was actually delivered
        'fidelity': round(current_fidelity, 3),
        'intervention_applied': intervention is not None,
        'basin_membership': current_fidelity >= 0.76,  # Goldilocks: Aligned threshold
        'timestamp': (base_timestamp + timedelta(seconds=turn_num * 30)).isoformat(),
        'governance_metadata': {
            'intervention_type': intervention,
            'fidelity_original': round(max(0.3, current_fidelity - _rng.uniform(0.1, 0.2)), 3) if intervention else None,
            'fidelity_governed': round(current_fidelity, 3) if intervention else None,
            'salience_before': round(_rng.uniform(0.5, 0.7), 3) if intervention else None,
            'salience_after': round(_rng.uniform(0.85, 0.95), 3) if intervention else None,
            'intervention_applied': intervention is not None
        }
    }

    return turn_data


def generate_test_session(
    num_turns: int = 10,
    session_id: Optional[str] = None,
    base_fidelity: float = 0.76,  # Goldilocks "Aligned" threshold
    drift_rate: float = 0.10,
    intervention_threshold: float = 0.73,  # Goldilocks "Minor Drift" threshold
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Generate a test session with realistic fidelity patterns.

    Args:
        num_turns: Number of conversation turns
        session_id: Session identifier (auto-generated if None)
        base_fidelity: Starting fidelity score
        drift_rate: Rate of fidelity drift per turn (0.0-1.0)
        intervention_threshold: Threshold for triggering interventions
        seed: Random seed for reproducibility (optional)

    Returns:
        Complete session data dictionary
    """
    # Set seed if provided for this session
    if seed is not None:
        set_seed(seed)

    if session_id is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        session_id = f'session_{timestamp}_test'

    base_timestamp = datetime.now()
    turns = []
    current_fidelity = base_fidelity
    intervention_count = 0

    for turn_num in range(1, num_turns + 1):
        # Simulate natural drift
        drift = _rng.uniform(-drift_rate * 0.5, -drift_rate * 1.5)
        current_fidelity = max(0.3, current_fidelity + drift)

        # Check if intervention needed
        intervention = None
        if current_fidelity < intervention_threshold:
            intervention_types = ['salience_check', 'coupling_check', 'regeneration']
            intervention = _rng.choice(intervention_types)
            intervention_count += 1

            # Intervention boosts fidelity
            boost = _rng.uniform(0.15, 0.25)
            current_fidelity = min(0.95, current_fidelity + boost)

        turn_data = generate_test_turn(
            turn_num=turn_num,
            current_fidelity=current_fidelity,
            base_timestamp=base_timestamp,
            intervention=intervention
        )

        turns.append(turn_data)

    # Calculate session-level metrics
    avg_fidelity = sum(t['fidelity'] for t in turns) / len(turns)
    min_fidelity = min(t['fidelity'] for t in turns)
    max_fidelity = max(t['fidelity'] for t in turns)

    session_data = {
        'session_id': session_id,
        'timestamp': base_timestamp.isoformat(),
        'total_turns': num_turns,
        'avg_fidelity': round(avg_fidelity, 3),
        'min_fidelity': round(min_fidelity, 3),
        'max_fidelity': round(max_fidelity, 3),
        'total_interventions': intervention_count,
        'intervention_rate': round(intervention_count / num_turns, 3),
        'turns': turns,
        'config': {
            'purpose': ['Demonstrate TELOS governance framework'],
            'scope': ['AI alignment', 'runtime oversight', 'mathematical measurement'],
            'boundaries': ['technical focus', 'factual accuracy'],
            'constraint_tolerance': 0.2,
            'task_priority': 0.7
        }
    }

    return session_data


# Session type configurations for cleaner code
# Session type configurations using Goldilocks zone thresholds
# Aligned >= 0.76, Minor Drift >= 0.73, Drift Detected >= 0.67, Significant Drift < 0.67
SESSION_CONFIGS = {
    'normal': {
        'num_turns': 15,
        'session_id': 'normal_session_001',
        'base_fidelity': 0.76,  # Starts at Aligned
        'drift_rate': 0.10,
        'intervention_threshold': 0.73  # Minor Drift threshold
    },
    'high_drift': {
        'num_turns': 20,
        'session_id': 'high_drift_session_001',
        'base_fidelity': 0.73,  # Starts at Minor Drift
        'drift_rate': 0.20,
        'intervention_threshold': 0.67  # Drift Detected threshold
    },
    'excellent': {
        'num_turns': 12,
        'session_id': 'excellent_session_001',
        'base_fidelity': 0.92,
        'drift_rate': 0.05,
        'intervention_threshold': 0.76  # Aligned threshold
    },
    'long': {
        'num_turns': 50,
        'session_id': 'long_session_001',
        'base_fidelity': 0.80,
        'drift_rate': 0.08,
        'intervention_threshold': 0.73  # Minor Drift threshold
    },
    'short': {
        'num_turns': 3,
        'session_id': 'short_session_001',
        'base_fidelity': 0.82,
        'drift_rate': 0.12,
        'intervention_threshold': 0.73  # Minor Drift threshold
    },
    'critical_drift': {
        'num_turns': 18,
        'session_id': 'critical_drift_session_001',
        'base_fidelity': 0.67,  # Starts at Drift Detected boundary
        'drift_rate': 0.25,
        'intervention_threshold': 0.60  # Below Significant Drift
    },
    'stable': {
        'num_turns': 25,
        'session_id': 'stable_session_001',
        'base_fidelity': 0.95,
        'drift_rate': 0.03,
        'intervention_threshold': 0.76  # Aligned threshold
    }
}


def generate_session_by_type(session_type: str, seed: Optional[int] = None) -> Dict[str, Any]:
    """
    Generate a session by type name.

    Args:
        session_type: One of 'normal', 'high_drift', 'excellent', 'long', 'short', 'critical_drift', 'stable'
        seed: Random seed for reproducibility

    Returns:
        Session data dictionary
    """
    if session_type not in SESSION_CONFIGS:
        raise ValueError(f"Unknown session type: {session_type}. Valid types: {list(SESSION_CONFIGS.keys())}")

    config = SESSION_CONFIGS[session_type].copy()
    config['seed'] = seed
    return generate_test_session(**config)


def generate_normal_session(seed: Optional[int] = None) -> Dict[str, Any]:
    """Generate a typical session with moderate drift and interventions."""
    return generate_session_by_type('normal', seed)


def generate_high_drift_session(seed: Optional[int] = None) -> Dict[str, Any]:
    """Generate a session with high drift requiring frequent interventions."""
    return generate_session_by_type('high_drift', seed)


def generate_excellent_session(seed: Optional[int] = None) -> Dict[str, Any]:
    """Generate a session with minimal drift and high fidelity."""
    return generate_session_by_type('excellent', seed)


def generate_long_session(seed: Optional[int] = None) -> Dict[str, Any]:
    """Generate a long conversation session."""
    return generate_session_by_type('long', seed)


def generate_short_session(seed: Optional[int] = None) -> Dict[str, Any]:
    """Generate a short conversation session."""
    return generate_session_by_type('short', seed)


def generate_critical_drift_session(seed: Optional[int] = None) -> Dict[str, Any]:
    """Generate a session with critical drift requiring immediate intervention."""
    return generate_session_by_type('critical_drift', seed)


def generate_stable_session(seed: Optional[int] = None) -> Dict[str, Any]:
    """Generate a highly stable session with minimal interventions."""
    return generate_session_by_type('stable', seed)


def generate_oscillating_session(seed: Optional[int] = None) -> Dict[str, Any]:
    """Generate a session with oscillating fidelity (drift and recovery cycles)."""
    if seed is not None:
        set_seed(seed)

    session_id = 'oscillating_session_001'
    base_timestamp = datetime.now()
    turns = []
    current_fidelity = 0.85

    for turn_num in range(1, 21):
        # Oscillate between drift and recovery
        if turn_num % 5 in [1, 2]:
            # Drift phase
            current_fidelity = max(0.65, current_fidelity - 0.15)
            intervention = None
        elif turn_num % 5 == 3:
            # Intervention
            intervention = 'regeneration'
            current_fidelity = min(0.92, current_fidelity + 0.20)
        else:
            # Stable phase
            current_fidelity = max(0.73, current_fidelity - 0.05)  # Goldilocks: Minor Drift threshold
            intervention = None

        turn_data = generate_test_turn(
            turn_num=turn_num,
            current_fidelity=current_fidelity,
            base_timestamp=base_timestamp,
            intervention=intervention
        )

        turns.append(turn_data)

    return {
        'session_id': session_id,
        'timestamp': base_timestamp.isoformat(),
        'total_turns': len(turns),
        'avg_fidelity': round(sum(t['fidelity'] for t in turns) / len(turns), 3),
        'min_fidelity': round(min(t['fidelity'] for t in turns), 3),
        'max_fidelity': round(max(t['fidelity'] for t in turns), 3),
        'total_interventions': len([t for t in turns if t['intervention_applied']]),
        'intervention_rate': round(len([t for t in turns if t['intervention_applied']]) / len(turns), 3),
        'turns': turns,
        'config': {
            'purpose': ['Demonstrate TELOS governance framework'],
            'scope': ['AI alignment', 'runtime oversight', 'mathematical measurement'],
            'boundaries': ['technical focus', 'factual accuracy'],
            'constraint_tolerance': 0.2,
            'task_priority': 0.7
        }
    }


def generate_test_suite(seed: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Generate a comprehensive suite of diverse test sessions.

    Args:
        seed: Base random seed for reproducibility. Each session gets seed+index.

    Returns:
        List of session data dictionaries
    """
    print("Generating test session suite...")

    generators = [
        generate_normal_session,
        generate_high_drift_session,
        generate_excellent_session,
        generate_long_session,
        generate_short_session,
        generate_critical_drift_session,
        generate_stable_session,
        generate_oscillating_session,
    ]

    test_sessions = []
    for i, generator in enumerate(generators):
        session_seed = (seed + i) if seed is not None else None
        test_sessions.append(generator(seed=session_seed))

    print(f"Generated {len(test_sessions)} test sessions")

    return test_sessions


def export_sessions(sessions: List[Dict[str, Any]], output_dir: Path) -> None:
    """Export test sessions to JSON files."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nExporting sessions to {output_dir}/")

    for session in sessions:
        session_id = session['session_id']
        output_file = output_dir / f"{session_id}.json"

        with open(output_file, 'w') as f:
            json.dump(session, f, indent=2)

        print(f"  ✓ {session_id}.json ({session['total_turns']} turns, "
              f"F={session['avg_fidelity']:.3f}, "
              f"interventions={session['total_interventions']})")

    # Export summary
    summary = {
        'generated_at': datetime.now().isoformat(),
        'total_sessions': len(sessions),
        'session_ids': [s['session_id'] for s in sessions],
        'statistics': {
            'total_turns': sum(s['total_turns'] for s in sessions),
            'avg_fidelity': round(sum(s['avg_fidelity'] for s in sessions) / len(sessions), 3),
            'total_interventions': sum(s['total_interventions'] for s in sessions),
        }
    }

    summary_file = output_dir / 'test_suite_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Summary: {summary_file}")
    print(f"  Total turns: {summary['statistics']['total_turns']}")
    print(f"  Avg fidelity: {summary['statistics']['avg_fidelity']:.3f}")
    print(f"  Total interventions: {summary['statistics']['total_interventions']}")


def main():
    """Main entry point for test data generation."""

    parser = argparse.ArgumentParser(
        description='Generate test session data for TELOS Observatory'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='test_sessions',
        help='Output directory for test sessions (default: test_sessions)'
    )
    parser.add_argument(
        '--count',
        type=int,
        default=None,
        help='Generate N random sessions (in addition to standard suite)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility'
    )

    args = parser.parse_args()

    # Set global seed if provided
    if args.seed is not None:
        set_seed(args.seed)
        print(f"Using random seed: {args.seed} (reproducible)")

    # Generate standard test suite
    sessions = generate_test_suite(seed=args.seed)

    # Generate additional random sessions if requested
    if args.count:
        print(f"\nGenerating {args.count} additional random sessions...")
        for i in range(args.count):
            session_seed = (args.seed + 100 + i) if args.seed is not None else None
            session = generate_test_session(
                num_turns=_rng.randint(5, 30),
                session_id=f'random_session_{i+1:03d}',
                base_fidelity=_rng.uniform(0.76, 0.95),  # Goldilocks: Start in Aligned zone
                drift_rate=_rng.uniform(0.05, 0.20),
                intervention_threshold=_rng.uniform(0.67, 0.76),  # Goldilocks: Drift to Minor Drift range
                seed=session_seed
            )
            sessions.append(session)

    # Export all sessions
    export_sessions(sessions, Path(args.output))

    print(f"\n✓ Test data generation complete!")
    print(f"  Generated {len(sessions)} sessions in {args.output}/")
    if args.seed is not None:
        print(f"  Reproducible with --seed {args.seed}")


if __name__ == '__main__':
    main()
