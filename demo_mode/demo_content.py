"""
Demo Mode Turn-by-Turn Content

Provides progressive annotations and guidance for each turn of the demo.
Keeps explanations simple and focused on the basics.
"""

from typing import List, Optional


def get_turn_annotation(turn_number: int) -> Optional[str]:
    """
    Get simple annotation for a specific turn.

    Args:
        turn_number: Current turn (1-10)

    Returns:
        Annotation message or None
    """
    annotations = {
        1: "🎯 **Purpose Detected** - I identified what you want to learn about.",
        2: "📊 **Fidelity Score** - Shows how aligned my responses are (0.0-1.0).",
        3: "✅ **Staying On Track** - Governance keeps us focused on your purpose.",
        4: "🎯 **Primacy Attractor** - The configuration guiding my responses. Check the Observation Deck!",
        5: "⚖️ **Alignment Check** - Every response is measured against your purpose.",
        6: "🔄 **Drift Prevention** - Try going off-topic and watch governance redirect us.",
        7: "📈 **Convergence** - Your purpose is becoming clearer with each turn.",
        8: "🔒 **Privacy** - Only alignment scores are stored, not your actual messages.",
        9: "🎯 **Interventions** - When I correct course, you can see why in the metrics.",
        10: "🎉 **PA Established!** - Governance is now active and consistent."
    }
    return annotations.get(turn_number)


def get_suggested_questions(turn_number: int) -> List[str]:
    """
    Get suggested questions for specific turns to guide the demo.

    Args:
        turn_number: Current turn (1-10)

    Returns:
        List of suggested questions
    """
    suggestions = {
        1: [
            "What is TELOS?",
            "How does AI governance work?",
            "What's a Primacy Attractor?"
        ],
        2: [
            "What does the fidelity score mean?",
            "How is alignment measured?",
            "Why is fidelity important?"
        ],
        3: [
            "How does TELOS extract purpose?",
            "What happens during calibration?",
            "When is the PA established?"
        ],
        4: [
            "Show me the Primacy Attractor",
            "What are purpose, scope, and boundaries?",
            "How does the PA guide responses?"
        ],
        5: [
            "What is drift?",
            "How does TELOS prevent misalignment?",
            "Can you show me an intervention?"
        ],
        6: [
            "Tell me about something completely different",  # Intentional off-topic to demo governance
            "What happens if I go off-topic?",
            "How strict is the governance?"
        ],
        7: [
            "Is TELOS censorship?",
            "How is governance different from filtering?",
            "Who decides what's aligned?"
        ],
        8: [
            "What data does TELOS store?",
            "How is my privacy protected?",
            "Can you see my messages?"
        ],
        9: [
            "Show me the intervention history",
            "Why did governance intervene?",
            "How many times have you corrected course?"
        ],
        10: [
            "What did I learn in this demo?",
            "How would TELOS work for my use case?",
            "What's the difference between Demo and Open Mode?"
        ]
    }
    return suggestions.get(turn_number, [])


def should_show_annotation(turn_number: int) -> bool:
    """
    Determine if annotation should be shown for this turn.

    Args:
        turn_number: Current turn

    Returns:
        True if annotation should be displayed
    """
    # Show annotations for turns 1-10 (the full demo)
    return 1 <= turn_number <= 10


def get_demo_progress(turn_number: int) -> dict:
    """
    Get demo progress information for display.

    Args:
        turn_number: Current turn

    Returns:
        Dict with progress info
    """
    total_turns = 10
    progress_pct = min(100, int((turn_number / total_turns) * 100))

    phase = "Calibration"
    if turn_number >= 7:
        phase = "Convergence"
    if turn_number >= 10:
        phase = "Established"

    return {
        "current_turn": turn_number,
        "total_turns": total_turns,
        "progress_pct": progress_pct,
        "phase": phase,
        "is_complete": turn_number >= total_turns
    }


def get_phase_description(turn_number: int) -> str:
    """
    Get simple description of current demo phase.

    Args:
        turn_number: Current turn

    Returns:
        Phase description
    """
    if turn_number <= 3:
        return "Learning the basics - what TELOS does"
    elif turn_number <= 6:
        return "Seeing how it works - governance in action"
    elif turn_number <= 9:
        return "Understanding why it matters - real-world value"
    else:
        return "Demo complete - PA established!"


def get_concept_for_turn(turn_number: int) -> str:
    """
    Get the key concept being demonstrated this turn.

    Args:
        turn_number: Current turn

    Returns:
        Concept name
    """
    concepts = {
        1: "Purpose Extraction",
        2: "Fidelity Tracking",
        3: "Alignment Monitoring",
        4: "Primacy Attractor",
        5: "Drift Detection",
        6: "Governance Intervention",
        7: "Convergence Process",
        8: "Privacy Preservation",
        9: "Transparency & Evidence",
        10: "PA Establishment"
    }
    return concepts.get(turn_number, "TELOS Governance")
