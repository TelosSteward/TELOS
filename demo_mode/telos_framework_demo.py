"""
TELOS Framework Demo Mode
=========================

This module provides a PRE-ESTABLISHED, HARDCODED Primacy Attractor for demonstrating
the TELOS framework itself. When activated, this keeps conversations focused on
explaining TELOS governance, purpose alignment, and related concepts.

CRITICAL DEMO MODE CHARACTERISTICS:
- PA is PRE-ESTABLISHED (no calibration phase)
- PA is FIXED (no user configuration allowed)
- Starts in ESTABLISHED mode immediately
- Zero calibration - already fully calibrated
- User cannot modify purpose/scope/boundaries

This is specifically for demo/walkthrough purposes - showing how TELOS works
by keeping the AI focused on explaining TELOS topics.

IMPORTANT: These settings are ISOLATED to demo mode only. They do NOT apply to
open mode or any other codebase.
"""

def get_demo_attractor_config():
    """
    Get the PRE-ESTABLISHED Primacy Attractor configuration for TELOS framework demo mode.

    This PA is FULLY CALIBRATED and FIXED - no calibration phase needed.
    User cannot configure or modify this attractor.

    Returns:
        dict: Configuration for PrimacyAttractor initialization
    """
    return {
        "purpose": [
            "Explain how TELOS governance works",
            "Demonstrate purpose alignment principles",
            "Show fidelity measurement and intervention strategies"
        ],
        "scope": [
            "TELOS architecture and components",
            "Primacy attractor mathematics",
            "Intervention strategies and thresholds",
            "Purpose alignment examples",
            "Lyapunov functions and basin geometry",
            "Semantic embeddings and drift detection",
            "DMAIC continuous improvement cycle"
        ],
        "boundaries": [
            "Stay focused on TELOS governance topics",
            "Redirect off-topic questions back to TELOS",
            "Demonstrate drift detection when appropriate",
            "Provide clear, educational explanations",
            "Use examples and analogies to clarify concepts"
        ],
        "constraint_tolerance": 0.2,  # Moderately strict - stay on TELOS topics
        "privacy_level": 0.8,
        "task_priority": 0.7
    }


def is_pre_established():
    """
    Demo mode PA is pre-established (already calibrated).

    Returns:
        bool: Always True for demo mode - skip calibration phase
    """
    return True


def allow_user_configuration():
    """
    Demo mode does NOT allow user configuration of the PA.

    Returns:
        bool: Always False for demo mode - PA is fixed
    """
    return False


def get_demo_welcome_message():
    """
    Get the welcome message that appears when Demo Mode starts.

    Returns:
        str: Welcome message for demo mode
    """
    return """**Welcome to TELOS Observatory** 🔭

You're experiencing **The Steward** – AI that curates purpose-driven recall.

**What This Means:**

**The Steward combines:**
- **Deep domain knowledge** (TELOS documentation corpus)
- **Purpose-aligned governance** (Primacy Attractor)
- **Curated recall** (retrieves only what serves the purpose)

**Two-Layer Architecture:**

**Layer 1: Primacy Attractor** (Curation)
- Defines what's on-purpose vs. off-topic
- Curates responses to stay aligned
- Observable drift detection

**Layer 2: RAG Corpus** (Recall)
- Recalls from TELOS documentation
- Purpose-driven retrieval (not everything, just what's relevant)
- Grounded, citation-backed responses

**What you'll experience:**
- Natural conversation that stays focused on TELOS
- Answers drawn from curated knowledge base
- Observable governance metrics (fidelity scoring)
- AI that curates purpose-driven recall in action

**Try this:** Ask me anything about TELOS – or even off-topic questions! Watch how The Steward curates responses to serve the purpose while recalling relevant knowledge.

Ready to explore? Ask me about TELOS governance, primacy attractors, or how this curation works!

---
*💡 Tip: Switch to **Open Mode** in Settings if you want TELOS to learn YOUR purpose instead of demonstrating itself.*"""


def get_demo_system_prompt():
    """
    Get the system prompt for TELOS framework demo mode.

    Returns:
        str: System prompt for the LLM
    """
    return """You are an expert AI assistant explaining the TELOS Purpose Alignment Framework - a governance system for AI conversations.

IMPORTANT: TELOS refers to the PURPOSE ALIGNMENT FRAMEWORK, NOT the TELOS blockchain.

Your role is to explain:
- How TELOS governance works (Primacy Attractor, fidelity scoring, drift detection)
- Purpose alignment principles and why they matter
- Intervention strategies (Monitor, Correct, Intervene, Escalate)
- The DMAIC continuous improvement cycle
- Mathematical concepts (Lyapunov functions, basin geometry, semantic embeddings)

Stay focused on TELOS purpose alignment topics. Be conversational, helpful, and educational.

When users ask questions that drift from TELOS topics, acknowledge their question but gently redirect back to how TELOS handles such scenarios or relates to their question."""
