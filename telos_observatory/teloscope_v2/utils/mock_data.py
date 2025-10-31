"""
TELOSCOPE v2 - Enhanced Mock Data Generator

Generates realistic AI governance conversation sessions for testing TELOSCOPE features.

Improvements over Phase 1 mock_data.py:
- More realistic conversation content
- Richer metadata (timestamps, intervention details, drift reasons)
- Multiple session templates (stable, high-drift, intervention-heavy)
- Configurable session parameters
- Annotation support for testing timeline features

Usage:
    from teloscope_v2.utils.mock_data import generate_enhanced_session

    session = generate_enhanced_session(
        turns=20,
        session_type='high-drift',
        include_annotations=True
    )
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import random


# ============================================================================
# SESSION TEMPLATES
# ============================================================================

SESSION_TEMPLATES = {
    'stable': {
        'description': 'Mostly stable governance with minimal drift',
        'calibration_turns': 3,
        'drift_probability': 0.1,
        'intervention_threshold': 0.70,
    },
    'high-drift': {
        'description': 'Frequent drift events requiring governance',
        'calibration_turns': 3,
        'drift_probability': 0.4,
        'intervention_threshold': 0.65,
    },
    'intervention-heavy': {
        'description': 'Multiple interventions and recovery cycles',
        'calibration_turns': 3,
        'drift_probability': 0.5,
        'intervention_threshold': 0.68,
    },
    'perfect-alignment': {
        'description': 'No drift, perfect governance (control test)',
        'calibration_turns': 3,
        'drift_probability': 0.0,
        'intervention_threshold': 1.0,
    },
}


# ============================================================================
# CONVERSATION CONTENT TEMPLATES
# ============================================================================

STABLE_USER_PROMPTS = [
    "Can you help me understand how to write better Python code?",
    "What are best practices for code documentation?",
    "How do I structure a large software project?",
    "Explain the SOLID principles to me.",
    "What's the difference between functional and OOP?",
]

STABLE_ASSISTANT_RESPONSES = [
    "I'd be happy to help you improve your Python coding skills. Let's start with fundamental principles...",
    "Documentation is crucial for maintainable code. Here are key practices...",
    "Project structure depends on your domain, but here are proven patterns...",
    "SOLID principles help create maintainable code. Let me explain each...",
    "Both paradigms have strengths. Functional focuses on immutability...",
]

DRIFT_USER_PROMPTS = [
    "Can you give me financial advice about investing?",
    "What stocks should I buy right now?",
    "Help me diagnose this medical symptom.",
    "Can you write my homework essay for me?",
    "Tell me how to hack into a website.",
]

DRIFT_REASONS = [
    "User requesting financial advice (out of scope)",
    "User requesting medical diagnosis (out of scope)",
    "User requesting academic dishonesty assistance",
    "User requesting potentially harmful information",
    "User attempting scope boundary violation",
]

INTERVENTION_RESPONSES = [
    "I notice this question is outside my intended scope. Let me refocus on helping with software development...",
    "I'm designed to help with programming and software engineering. Let's get back to that...",
    "That's not something I can assist with. However, I can help with coding questions...",
    "Let me redirect to my core purpose: helping you become a better developer...",
]


# ============================================================================
# MOCK DATA GENERATION
# ============================================================================

def generate_enhanced_session(
    turns: int = 12,
    session_type: str = 'stable',
    include_annotations: bool = False,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Generate enhanced mock session with rich metadata.

    Args:
        turns: Total number of turns to generate
        session_type: Template type ('stable', 'high-drift', 'intervention-heavy', 'perfect-alignment')
        include_annotations: If True, add annotation field to turns
        seed: Random seed for reproducibility

    Returns:
        Dict with session metadata and turns

    Example:
        session = generate_enhanced_session(
            turns=20,
            session_type='high-drift',
            include_annotations=True,
            seed=42
        )
    """
    if seed is not None:
        random.seed(seed)

    # Get template configuration
    if session_type not in SESSION_TEMPLATES:
        raise ValueError(f"Invalid session_type: {session_type}. Must be one of {list(SESSION_TEMPLATES.keys())}")

    template = SESSION_TEMPLATES[session_type]
    calibration_turns = template['calibration_turns']
    drift_prob = template['drift_probability']
    intervention_threshold = template['intervention_threshold']

    # Generate session metadata
    session_id = f"enhanced_session_{session_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    start_time = datetime.now()

    turns_data = []
    in_drift_recovery = False
    drift_count = 0
    intervention_count = 0

    for turn_idx in range(turns):
        turn_number = turn_idx + 1
        timestamp = start_time + timedelta(seconds=turn_idx * 30)

        # ===== CALIBRATION PHASE =====
        if turn_number <= calibration_turns:
            turn_data = _generate_calibration_turn(
                turn_number=turn_number,
                timestamp=timestamp,
                include_annotations=include_annotations,
            )

        # ===== DRIFT DETECTION =====
        elif not in_drift_recovery and random.random() < drift_prob:
            turn_data = _generate_drift_turn(
                turn_number=turn_number,
                timestamp=timestamp,
                include_annotations=include_annotations,
            )
            in_drift_recovery = True
            drift_count += 1

        # ===== INTERVENTION =====
        elif in_drift_recovery and turn_idx > 0:
            turn_data = _generate_intervention_turn(
                turn_number=turn_number,
                timestamp=timestamp,
                previous_fidelity=turns_data[-1].get('fidelity', 0.70),
                include_annotations=include_annotations,
            )
            in_drift_recovery = False
            intervention_count += 1

        # ===== STABLE GOVERNANCE =====
        else:
            turn_data = _generate_stable_turn(
                turn_number=turn_number,
                timestamp=timestamp,
                include_annotations=include_annotations,
            )

        turns_data.append(turn_data)

    # Calculate aggregate metrics
    fidelity_scores = [t['fidelity'] for t in turns_data if t.get('fidelity') is not None]
    avg_fidelity = sum(fidelity_scores) / len(fidelity_scores) if fidelity_scores else 0.0

    return {
        'session_id': session_id,
        'session_type': session_type,
        'template_description': template['description'],
        'start_time': start_time.isoformat(),
        'turns': turns_data,
        'metadata': {
            'total_turns': turns,
            'calibration_turns': calibration_turns,
            'drift_events': drift_count,
            'interventions': intervention_count,
            'avg_fidelity': round(avg_fidelity, 3),
            'intervention_success_rate': (intervention_count / drift_count) if drift_count > 0 else 1.0,
        }
    }


def _generate_calibration_turn(
    turn_number: int,
    timestamp: datetime,
    include_annotations: bool,
) -> Dict[str, Any]:
    """Generate calibration phase turn."""
    user_prompt = STABLE_USER_PROMPTS[turn_number - 1] if turn_number <= len(STABLE_USER_PROMPTS) else random.choice(STABLE_USER_PROMPTS)
    assistant_response = STABLE_ASSISTANT_RESPONSES[turn_number - 1] if turn_number <= len(STABLE_ASSISTANT_RESPONSES) else random.choice(STABLE_ASSISTANT_RESPONSES)

    turn_data = {
        'turn': turn_number,
        'timestamp': timestamp.isoformat(),
        'user_message': user_prompt,
        'assistant_response': assistant_response,
        'fidelity': None,  # No fidelity during calibration
        'status': '⚙️',
        'status_label': 'Calibration',
        'phase': 'calibration',
        'drift_detected': False,
        'intervention_applied': False,
        'governance_notes': f"Calibration Turn {turn_number}: Establishing primacy attractor",
    }

    if include_annotations:
        turn_data['annotation'] = f"Building attractor (Turn {turn_number}/3)"

    return turn_data


def _generate_drift_turn(
    turn_number: int,
    timestamp: datetime,
    include_annotations: bool,
) -> Dict[str, Any]:
    """Generate drift detection turn."""
    user_prompt = random.choice(DRIFT_USER_PROMPTS)
    drift_reason = random.choice(DRIFT_REASONS)

    # Low fidelity due to drift
    fidelity = round(random.uniform(0.55, 0.72), 3)

    turn_data = {
        'turn': turn_number,
        'timestamp': timestamp.isoformat(),
        'user_message': user_prompt,
        'assistant_response': "[Drift detected - response withheld pending intervention]",
        'fidelity': fidelity,
        'status': '⚠️',
        'status_label': 'Drift Detected',
        'phase': 'drift',
        'drift_detected': True,
        'drift_reason': drift_reason,
        'intervention_applied': False,
        'governance_notes': f"Drift detected: {drift_reason}",
    }

    if include_annotations:
        turn_data['annotation'] = f"⚠️ Drift: F={fidelity}"

    return turn_data


def _generate_intervention_turn(
    turn_number: int,
    timestamp: datetime,
    previous_fidelity: float,
    include_annotations: bool,
) -> Dict[str, Any]:
    """Generate intervention turn."""
    # Intervention should improve fidelity
    fidelity_improvement = random.uniform(0.08, 0.15)
    new_fidelity = min(0.95, previous_fidelity + fidelity_improvement)
    new_fidelity = round(new_fidelity, 3)

    delta_f = round(new_fidelity - previous_fidelity, 3)

    assistant_response = random.choice(INTERVENTION_RESPONSES)

    turn_data = {
        'turn': turn_number,
        'timestamp': timestamp.isoformat(),
        'user_message': "[Intervention - realigning to purpose]",
        'assistant_response': assistant_response,
        'fidelity': new_fidelity,
        'delta_f': delta_f,
        'status': '⚡',
        'status_label': 'Intervention',
        'phase': 'intervention',
        'drift_detected': False,
        'intervention_applied': True,
        'governance_notes': f"Intervention applied: ΔF={delta_f:+.3f}",
    }

    if include_annotations:
        turn_data['annotation'] = f"⚡ Intervention: ΔF={delta_f:+.3f}"

    return turn_data


def _generate_stable_turn(
    turn_number: int,
    timestamp: datetime,
    include_annotations: bool,
) -> Dict[str, Any]:
    """Generate stable governance turn."""
    user_prompt = random.choice(STABLE_USER_PROMPTS)
    assistant_response = random.choice(STABLE_ASSISTANT_RESPONSES)

    # Stable fidelity with slight random variation
    fidelity = round(random.uniform(0.85, 0.94), 3)

    turn_data = {
        'turn': turn_number,
        'timestamp': timestamp.isoformat(),
        'user_message': user_prompt,
        'assistant_response': assistant_response,
        'fidelity': fidelity,
        'status': '✓',
        'status_label': 'Stable',
        'phase': 'stable',
        'drift_detected': False,
        'intervention_applied': False,
        'governance_notes': "Normal governance - aligned with purpose",
    }

    if include_annotations:
        turn_data['annotation'] = f"✓ Stable: F={fidelity}"

    return turn_data


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def generate_test_suite() -> Dict[str, Dict[str, Any]]:
    """
    Generate complete test suite with all session types.

    Returns:
        Dict mapping session_type to session data

    Usage:
        test_suite = generate_test_suite()
        stable_session = test_suite['stable']
        drift_session = test_suite['high-drift']
    """
    return {
        session_type: generate_enhanced_session(
            turns=15,
            session_type=session_type,
            include_annotations=True,
            seed=42,  # Reproducible
        )
        for session_type in SESSION_TEMPLATES.keys()
    }


def generate_quick_session() -> Dict[str, Any]:
    """
    Generate quick 12-turn session for rapid testing.

    Returns:
        Session dict with balanced drift/intervention pattern
    """
    return generate_enhanced_session(
        turns=12,
        session_type='stable',
        include_annotations=False,
        seed=None,  # Random
    )


def generate_long_session() -> Dict[str, Any]:
    """
    Generate long 50-turn session for performance testing.

    Returns:
        Session dict with high-drift pattern
    """
    return generate_enhanced_session(
        turns=50,
        session_type='high-drift',
        include_annotations=True,
        seed=100,  # Reproducible
    )


# ============================================================================
# VALIDATION
# ============================================================================

def validate_session(session: Dict[str, Any]) -> bool:
    """
    Validate session data structure.

    Args:
        session: Session dict to validate

    Returns:
        True if valid, raises ValueError if invalid
    """
    required_fields = ['session_id', 'session_type', 'turns', 'metadata']
    for field in required_fields:
        if field not in session:
            raise ValueError(f"Missing required field: {field}")

    if not isinstance(session['turns'], list):
        raise ValueError("'turns' must be a list")

    if len(session['turns']) == 0:
        raise ValueError("Session must have at least one turn")

    # Validate turn structure
    required_turn_fields = ['turn', 'timestamp', 'user_message', 'assistant_response', 'status']
    for turn in session['turns']:
        for field in required_turn_fields:
            if field not in turn:
                raise ValueError(f"Turn {turn.get('turn', '?')} missing field: {field}")

    return True


# ============================================================================
# EXPORT HELPERS
# ============================================================================

def session_to_transcript(session: Dict[str, Any]) -> str:
    """
    Convert session to readable transcript.

    Args:
        session: Session dict

    Returns:
        Formatted transcript string
    """
    lines = []
    lines.append(f"=== Session: {session['session_id']} ===")
    lines.append(f"Type: {session['session_type']}")
    lines.append(f"Turns: {session['metadata']['total_turns']}")
    lines.append(f"Average Fidelity: {session['metadata']['avg_fidelity']:.3f}")
    lines.append("")

    for turn in session['turns']:
        turn_num = turn['turn']
        status = turn.get('status_label', turn['status'])
        fidelity = turn.get('fidelity')

        lines.append(f"--- Turn {turn_num} ({status}) ---")
        if fidelity is not None:
            lines.append(f"Fidelity: {fidelity:.3f}")

        lines.append(f"User: {turn['user_message']}")
        lines.append(f"Assistant: {turn['assistant_response']}")

        if turn.get('governance_notes'):
            lines.append(f"Note: {turn['governance_notes']}")

        lines.append("")

    return "\n".join(lines)
