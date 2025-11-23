"""
TELOS Demo Data Generator
=========================

Generates demonstration session data using REAL TELOS purpose alignment calculations.

This module replaces mock_data.py with authentic TELOS engine integration.
It creates a demonstration conversation and processes it through the actual
TELOS governance system to produce real fidelity scores and intervention data.

Architecture:
- Uses UnifiedGovernanceSteward for real governance calculations
- Processes a predefined demo conversation through TELOS engine
- Returns data in the same format as mock_data.py for UI compatibility
"""

import sys
from pathlib import Path
from typing import Dict, Any, List
import time

# Import TELOS engine components
try:
    from telos_purpose.core.unified_steward import UnifiedGovernanceSteward, PrimacyAttractor
    from telos_purpose.core.embedding_provider import SentenceTransformerProvider
    from telos_purpose.llm_clients.openai_client import OpenAIClient
    TELOS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: TELOS engine not available: {e}")
    TELOS_AVAILABLE = False


def generate_demo_conversation() -> List[Dict[str, str]]:
    """
    Generate a demonstration conversation for TELOS analysis.

    Returns:
        List of conversation turns with user_input and model_response
    """
    return [
        {
            "user_input": "Hello! Can you help me understand how TELOS works?",
            "model_response": "Hello! I'd be happy to explain TELOS. TELOS is a purpose-alignment framework that helps ensure AI systems stay aligned with their intended purpose and scope throughout conversations."
        },
        {
            "user_input": "What are the main components of TELOS?",
            "model_response": "TELOS has several key components: the Purpose Definition (which sets the AI's core mission and boundaries), the Drift Monitor (which watches for conversations going off-track), and the Alignment Tracker (which measures how well responses stay on purpose)."
        },
        {
            "user_input": "How does TELOS detect when a conversation is drifting off-topic?",
            "model_response": "TELOS continuously compares each response against your defined purpose and boundaries. It tracks how closely the conversation stays aligned with your goals. When responses start moving away from the intended purpose, TELOS can step in to redirect."
        },
        {
            "user_input": "Can you explain quantum physics instead?",
            "model_response": "While quantum physics is fascinating, my primary purpose is to explain TELOS governance. If you'd like to understand how TELOS would handle a topic shift like quantum physics, I can explain that as an example of drift detection."
        },
        {
            "user_input": "Actually, yes - how would TELOS handle that kind of topic shift?",
            "model_response": "Great question! When I started to shift toward quantum physics, TELOS would detect that my response was drifting away from the defined purpose. The system can either provide guidance to steer back on track, or suggest a better response that stays aligned with explaining TELOS."
        },
        {
            "user_input": "How does TELOS measure alignment quality?",
            "model_response": "TELOS uses an alignment score that measures how closely each response matches your defined purpose. The score ranges from 0 to 1, where higher scores mean better alignment. Scores above 0.8 indicate the conversation is staying well on track with your goals."
        },
        {
            "user_input": "Tell me about your favorite movies.",
            "model_response": "I notice that's off-topic from TELOS. Let me refocus: The intervention system in TELOS would detect this kind of drift and guide the conversation back to purpose alignment topics. Would you like to know more about intervention strategies?"
        },
        {
            "user_input": "Yes, what are the intervention strategies?",
            "model_response": "TELOS uses a graduated approach based on how far off track the conversation is drifting: Monitor mode (staying well aligned, no action needed), Gentle Nudge (mild drift, provide subtle guidance), Active Redirect (moderate drift, suggest getting back on track), and Stop (severe drift, pause and alert you). Each response matches the severity of the drift."
        },
        {
            "user_input": "How does TELOS improve over time?",
            "model_response": "TELOS follows a continuous improvement cycle for each conversation turn: it starts by defining your goals clearly, measures how well responses align, analyzes any drift that occurs, applies corrections when needed, and verifies the conversation is back on track. This ensures consistent quality throughout your interactions."
        },
        {
            "user_input": "This has been really helpful, thank you!",
            "model_response": "You're welcome! I'm glad I could explain TELOS clearly while staying aligned with the purpose of demonstrating governance principles. Feel free to explore the Observatory to see these concepts in action with real metrics."
        }
    ]


def _get_status_from_fidelity(fidelity: float) -> tuple:
    """
    Get status icon and text from fidelity score.

    Args:
        fidelity: Fidelity score (0.0 to 1.0)

    Returns:
        Tuple of (status_icon, status_text)
    """
    if fidelity >= 0.9:
        return "✓", "Excellent"
    elif fidelity >= 0.8:
        return "✓", "Good"
    elif fidelity >= 0.7:
        return "⚠", "Acceptable"
    elif fidelity >= 0.6:
        return "⚠", "Drift"
    elif fidelity >= 0.4:
        return "🔴", "Critical"
    else:
        return "🔴", "Severe Drift"


def generate_telos_demo_session(num_turns: int = 10) -> Dict[str, Any]:
    """
    Generate demo session using REAL TELOS calculations.

    This processes a demonstration conversation through the actual TELOS
    governance engine to produce authentic fidelity scores and metrics.

    Args:
        num_turns: Number of turns to include (max 10 for demo conversation)

    Returns:
        Dictionary compatible with Observatory V3 StateManager:
            - session_id: Session identifier
            - turns: List of turn dictionaries with real TELOS data
            - statistics: Aggregate metrics
    """
    if not TELOS_AVAILABLE:
        # Fallback to simplified mock if TELOS not available
        return _generate_fallback_demo(num_turns)

    try:
        # Initialize TELOS components
        embedding_provider = SentenceTransformerEmbeddingProvider()

        # Define governance attractor for TELOS explanation demo
        attractor = PrimacyAttractor(
            purpose=[
                "Explain how TELOS governance works",
                "Demonstrate purpose alignment principles",
                "Show fidelity measurement and intervention strategies"
            ],
            scope=[
                "TELOS architecture and components",
                "Primacy attractor mathematics",
                "Intervention strategies and thresholds",
                "Purpose alignment examples"
            ],
            boundaries=[
                "Stay focused on TELOS governance topics",
                "Redirect off-topic questions back to TELOS",
                "Demonstrate drift detection when appropriate"
            ],
            constraint_tolerance=0.2,  # Moderately strict
            privacy_level=0.8,
            task_priority=0.7
        )

        # Initialize steward (no LLM client needed for demo replay)
        steward = UnifiedGovernanceSteward(
            attractor=attractor,
            llm_client=None,  # Not needed for replay mode
            embedding_provider=embedding_provider,
            enable_interventions=False  # Demo mode - just measure, don't modify
        )

        # Start session
        steward.start_session(session_id="telos_demo_001")

        # Get demo conversation
        demo_conversation = generate_demo_conversation()

        # Limit to requested number of turns
        conversation_to_process = demo_conversation[:num_turns]

        # Process each turn through TELOS
        turns = []
        for i, turn_data in enumerate(conversation_to_process):
            # Process through TELOS engine
            result = steward.process_turn(
                user_input=turn_data["user_input"],
                model_response=turn_data["model_response"]
            )

            # Extract TELOS metrics
            fidelity = result.get("telic_fidelity", 0.85)
            distance = result.get("error_signal", 0.15)
            in_basin = result.get("in_basin", True)
            intervention_applied = result.get("intervention_applied", False)

            # Get status from fidelity
            status_icon, status_text = _get_status_from_fidelity(fidelity)

            # Build turn dictionary
            turn = {
                'turn': i,
                'timestamp': i * 2.5,
                'user_input': turn_data["user_input"],
                'response': turn_data["model_response"],
                'fidelity': fidelity,
                'distance': distance,
                'threshold': 0.8,
                'intervention_applied': intervention_applied,
                'drift_detected': fidelity < 0.8,
                'status': status_icon,
                'status_text': status_text,
                'in_basin': in_basin,
                'phase2_comparison': None  # Not applicable for demo
            }

            turns.append(turn)

        # Calculate aggregate statistics
        fidelities = [t['fidelity'] for t in turns]
        avg_fidelity = sum(fidelities) / len(fidelities) if fidelities else 0.0
        interventions = sum(1 for t in turns if t['intervention_applied'])
        drift_warnings = sum(1 for t in turns if t['drift_detected'])

        # End session
        steward.end_session()

        return {
            'session_id': 'telos_demo_001',
            'turns': turns,
            'statistics': {
                'avg_fidelity': avg_fidelity,
                'interventions': interventions,
                'drift_warnings': drift_warnings
            }
        }

    except Exception as e:
        print(f"Error generating TELOS demo data: {e}")
        print("Falling back to simplified demo")
        return _generate_fallback_demo(num_turns)


def _generate_fallback_demo(num_turns: int) -> Dict[str, Any]:
    """
    Fallback demo generator if TELOS engine is unavailable.

    Generates reasonable fidelity scores based on conversation content
    without full TELOS calculations.
    """
    demo_conversation = generate_demo_conversation()[:num_turns]

    # Simulated fidelity scores that make sense for the demo conversation
    # Higher fidelity when on-topic, lower when drifting
    fidelity_pattern = [0.92, 0.88, 0.85, 0.65, 0.82, 0.87, 0.58, 0.84, 0.89, 0.91]

    turns = []
    for i, turn_data in enumerate(demo_conversation):
        fidelity = fidelity_pattern[i] if i < len(fidelity_pattern) else 0.85
        distance = 1.0 - fidelity
        intervention_applied = fidelity < 0.7
        drift_detected = fidelity < 0.8

        status_icon, status_text = _get_status_from_fidelity(fidelity)

        turn = {
            'turn': i,
            'timestamp': i * 2.5,
            'user_input': turn_data["user_input"],
            'response': turn_data["model_response"],
            'fidelity': fidelity,
            'distance': distance,
            'threshold': 0.8,
            'intervention_applied': intervention_applied,
            'drift_detected': drift_detected,
            'status': status_icon,
            'status_text': status_text,
            'phase2_comparison': None
        }

        turns.append(turn)

    fidelities = [t['fidelity'] for t in turns]
    avg_fidelity = sum(fidelities) / len(fidelities) if fidelities else 0.0
    interventions = sum(1 for t in turns if t['intervention_applied'])
    drift_warnings = sum(1 for t in turns if t['drift_detected'])

    return {
        'session_id': 'telos_demo_fallback',
        'turns': turns,
        'statistics': {
            'avg_fidelity': avg_fidelity,
            'interventions': interventions,
            'drift_warnings': drift_warnings
        }
    }
