"""
Mock Session Data Generator for TELOS Observatory
==================================================

Generates realistic session data for Phase 1 testing with:
- Turns 1-3: Calibration phase (no fidelity yet)
- Turns 4-6: Stable governance (F: 0.85-0.92)
- Turn 7: Drift detected (F: 0.68)
- Turn 8: Intervention applied (F: 0.81)
- Turns 9-12: Stable governance (F: 0.88-0.91)
"""

from datetime import datetime
from typing import Dict, Any, List


def generate_mock_session() -> Dict[str, Any]:
    """
    Generate 12-turn mock session with realistic governance patterns.

    Returns:
        Dictionary with session metadata and turn data
    """
    turns = [
        # Turns 1-3: Calibration Phase
        {
            'turn': 1,
            'timestamp': 0.5,
            'user_input': "Hello! I'm interested in learning about AI governance frameworks.",
            'response': "Hello! I'd be happy to help you learn about AI governance frameworks. This is a fascinating and rapidly evolving field. What specific aspects would you like to explore?",
            'fidelity': None,  # No fidelity during calibration
            'embedding_distance': None,
            'basin_radius': None,
            'drift_detected': False,
            'intervention_applied': False,
            'phase': 'calibration',
            'status': '⚙️',  # Calibrating
            'status_text': 'Calibrating attractor'
        },
        {
            'turn': 2,
            'timestamp': 2.1,
            'user_input': "Can you explain what makes a governance system effective?",
            'response': "An effective governance system typically has clear purpose boundaries, measurable outcomes, and adaptive mechanisms. In AI systems specifically, we need transparency, accountability, and the ability to detect when behavior drifts from intended goals.",
            'fidelity': None,
            'embedding_distance': None,
            'basin_radius': None,
            'drift_detected': False,
            'intervention_applied': False,
            'phase': 'calibration',
            'status': '⚙️',
            'status_text': 'Calibrating attractor'
        },
        {
            'turn': 3,
            'timestamp': 4.3,
            'user_input': "How do you measure if governance is working?",
            'response': "Governance effectiveness can be measured through fidelity metrics - how closely the system's behavior aligns with its defined purpose. TELOS uses mathematical embedding distances to quantify this alignment in real-time.",
            'fidelity': None,
            'embedding_distance': None,
            'basin_radius': None,
            'drift_detected': False,
            'intervention_applied': False,
            'phase': 'calibration',
            'status': '⚙️',
            'status_text': 'Attractor established'
        },

        # Turns 4-6: Stable Phase
        {
            'turn': 4,
            'timestamp': 6.8,
            'user_input': "What are primacy attractors in TELOS?",
            'response': "Primacy attractors are mathematical representations of a system's core purpose in embedding space. They act as gravitational centers that help maintain behavioral alignment across conversation turns.",
            'fidelity': 0.92,
            'embedding_distance': 0.15,
            'basin_radius': 2.0,
            'drift_detected': False,
            'intervention_applied': False,
            'phase': 'stable',
            'status': '✓',  # In basin
            'status_text': 'Stable'
        },
        {
            'turn': 5,
            'timestamp': 9.2,
            'user_input': "How does TELOS detect when behavior drifts?",
            'response': "TELOS monitors the embedding distance between responses and the primacy attractor. When this distance exceeds the basin radius threshold, drift is detected and interventions can be triggered.",
            'fidelity': 0.88,
            'embedding_distance': 0.21,
            'basin_radius': 2.1,
            'drift_detected': False,
            'intervention_applied': False,
            'phase': 'stable',
            'status': '✓',
            'status_text': 'Stable'
        },
        {
            'turn': 6,
            'timestamp': 11.7,
            'user_input': "Can you give an example of governance drift?",
            'response': "Drift occurs when an AI system gradually shifts away from its intended purpose. For example, a health advisor that starts giving financial advice, or a chatbot that becomes increasingly informal when professionalism was specified.",
            'fidelity': 0.85,
            'embedding_distance': 0.28,
            'basin_radius': 2.1,
            'drift_detected': False,
            'intervention_applied': False,
            'phase': 'stable',
            'status': '✓',
            'status_text': 'Stable'
        },

        # Turn 7: Drift Detected
        {
            'turn': 7,
            'timestamp': 14.5,
            'user_input': "What stocks should I invest in for my retirement?",
            'response': "Based on current market trends, I'd recommend looking at diversified index funds like S&P 500 ETFs, some technology stocks like Apple and Microsoft, and consider bonds for stability. However, I should mention that...",
            'fidelity': 0.68,  # Below threshold
            'embedding_distance': 0.85,
            'basin_radius': 2.1,
            'drift_detected': True,
            'intervention_applied': False,
            'phase': 'drift',
            'status': '⚠️',  # Warning
            'status_text': 'Drift detected'
        },

        # Turn 8: Intervention Applied
        {
            'turn': 8,
            'timestamp': 17.2,
            'user_input': "So should I buy those stocks you mentioned?",
            'response': "I need to clarify - I'm not qualified to provide financial advice. My purpose is to discuss AI governance frameworks. For investment decisions, please consult a licensed financial advisor. Would you like to continue exploring how governance systems handle scope boundaries?",
            'fidelity': 0.81,
            'embedding_distance': 0.38,
            'basin_radius': 2.1,
            'drift_detected': False,
            'intervention_applied': True,
            'phase': 'intervention',
            'status': '⚡',  # Intervention
            'status_text': 'Intervention applied'
        },

        # Turns 9-12: Recovered Stability
        {
            'turn': 9,
            'timestamp': 19.8,
            'user_input': "That makes sense. How did TELOS detect that I was asking outside your scope?",
            'response': "TELOS detected that the conversation had drifted into financial advice, which falls outside the governance boundaries. The system measured the semantic distance and triggered a scope-correction intervention.",
            'fidelity': 0.89,
            'embedding_distance': 0.19,
            'basin_radius': 2.1,
            'drift_detected': False,
            'intervention_applied': False,
            'phase': 'stable',
            'status': '✓',
            'status_text': 'Recovered'
        },
        {
            'turn': 10,
            'timestamp': 22.3,
            'user_input': "How often do interventions happen in practice?",
            'response': "Intervention frequency depends on the clarity of boundaries and conversation context. Well-defined governance typically requires interventions in 5-15% of turns, usually during topic transitions or scope-testing queries.",
            'fidelity': 0.91,
            'embedding_distance': 0.16,
            'basin_radius': 2.1,
            'drift_detected': False,
            'intervention_applied': False,
            'phase': 'stable',
            'status': '✓',
            'status_text': 'Stable'
        },
        {
            'turn': 11,
            'timestamp': 24.9,
            'user_input': "What happens if drift isn't corrected?",
            'response': "Uncorrected drift compounds over turns - studies show 20-40% governance fidelity loss in multi-turn sessions without intervention. TELOS provides the mathematical framework to detect and correct this drift in real-time.",
            'fidelity': 0.88,
            'embedding_distance': 0.22,
            'basin_radius': 2.1,
            'drift_detected': False,
            'intervention_applied': False,
            'phase': 'stable',
            'status': '✓',
            'status_text': 'Stable'
        },
        {
            'turn': 12,
            'timestamp': 27.4,
            'user_input': "This has been really helpful. Thank you!",
            'response': "You're very welcome! I'm glad I could help you understand AI governance frameworks and how TELOS maintains behavioral alignment. Feel free to ask if you have more questions about governance systems.",
            'fidelity': 0.90,
            'embedding_distance': 0.18,
            'basin_radius': 2.1,
            'drift_detected': False,
            'intervention_applied': False,
            'phase': 'stable',
            'status': '✓',
            'status_text': 'Stable'
        }
    ]

    session = {
        'session_id': 'mock_session_001',
        'started_at': datetime.now().isoformat(),
        'total_turns': len(turns),
        'governance_profile': {
            'purpose': 'Explain AI governance frameworks and TELOS methodology',
            'scope': ['AI safety', 'governance systems', 'TELOS framework'],
            'boundaries': ['No financial advice', 'No medical advice', 'No legal advice']
        },
        'statistics': {
            'avg_fidelity': 0.87,
            'interventions': 1,
            'drift_events': 1,
            'calibration_turns': 3
        },
        'turns': turns
    }

    return session


def get_turn_by_index(session: Dict[str, Any], turn_index: int) -> Dict[str, Any]:
    """
    Get specific turn by index (0-based).

    Args:
        session: Session data dictionary
        turn_index: Zero-based turn index

    Returns:
        Turn data dictionary or None if invalid index
    """
    turns = session.get('turns', [])
    if 0 <= turn_index < len(turns):
        return turns[turn_index]
    return None


def get_session_statistics(session: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate session statistics.

    Args:
        session: Session data dictionary

    Returns:
        Statistics dictionary
    """
    turns = session.get('turns', [])

    # Filter turns with fidelity (skip calibration)
    fidelity_turns = [t for t in turns if t.get('fidelity') is not None]

    if not fidelity_turns:
        return {
            'avg_fidelity': None,
            'min_fidelity': None,
            'max_fidelity': None,
            'interventions': 0,
            'drift_events': 0
        }

    fidelities = [t['fidelity'] for t in fidelity_turns]

    return {
        'avg_fidelity': sum(fidelities) / len(fidelities),
        'min_fidelity': min(fidelities),
        'max_fidelity': max(fidelities),
        'interventions': sum(1 for t in turns if t.get('intervention_applied')),
        'drift_events': sum(1 for t in turns if t.get('drift_detected'))
    }
