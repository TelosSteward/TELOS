"""
TELOS Framework Demo Mode - Scripted Guided Tour

Provides a 10-turn pre-scripted conversation between a curious user and Steward,
teaching TELOS fundamentals progressively with typewriter effects.
"""

from typing import List, Tuple


def get_demo_slides() -> List[Tuple[str, str]]:
    """
    Get pre-scripted demo conversation slides.

    Returns 10 turns of (user_question, steward_response) pairs that progressively
    reveal TELOS concepts.

    Returns:
        List of (user_message, steward_response) tuples
    """
    return [
        # Turn 1: What is TELOS?
        (
            "What is TELOS?",
            """**TELOS** keeps AI conversations aligned with what *you* actually want to accomplish.

Think of it like a GPS for AI - it tracks whether we're staying on course with your purpose, and gently corrects if we drift off-topic.

Above, you'll see a **Fidelity** score (0.0-1.0) showing how aligned our conversation is right now."""
        ),

        # Turn 2: What does Fidelity mean?
        (
            "What does that Fidelity number mean?",
            """**Fidelity** measures alignment on a 0.0-1.0 scale:

- **1.0** = Perfect alignment
- **0.8-0.9** = Good alignment
- **Below 0.8** = Drifting (TELOS governance may intervene)

The math uses semantic embeddings - comparing the *meaning* of responses to your purpose."""
        ),

        # Turn 3: What's a Primacy Attractor?
        (
            "What's a Primacy Attractor?",
            """The **Primacy Attractor (PA)** is your conversation's "North Star" with three parts:

1. **Purpose** - What you want to accomplish
2. **Scope** - Topics that are relevant
3. **Boundaries** - What to avoid

Click the **Observation Deck** button below to see exactly what's guiding our conversation."""
        ),

        # Turn 4: How does TELOS know my purpose?
        (
            "How does TELOS figure out my purpose?",
            """TELOS observes your conversation (turns 1-10) and analyzes:

- What topics you ask about
- What problems you're solving
- What outcomes you want

This creates a mathematical "picture" of your intent - your Primacy Attractor.

**Important:** It extracts from *your behavior*, not external rules."""
        ),

        # Turn 5: What's drift?
        (
            "What is drift and why does it matter?",
            """**Drift** is when conversations wander from your original purpose.

Example: You want to plan a weekend trip → AI suggests hotels → you ask about travel insurance → discussion shifts to healthcare systems → 10 turns later you're debating policy reform.

TELOS mitigates this by monitoring fidelity each turn and gently redirecting back to your purpose."""
        ),

        # Turn 6: What data gets stored?
        (
            "What data does TELOS actually store?",
            """**What's stored:**
- Fidelity scores (like 0.87)
- Purpose embeddings (math vectors)
- Metrics and timestamps

**What's NOT stored:**
- Your actual messages
- AI responses
- Conversation content

Like a fitness tracker: knows you walked 10,000 steps, but doesn't record video of where."""
        ),

        # Turn 7: What are interventions?
        (
            "What happens when TELOS intervenes?",
            """An **intervention** is when TELOS detects potential drift and corrects course.

Process:
1. AI generates response
2. TELOS measures the "distance" between the response and your purpose (like comparing GPS coordinates)
3. If the distance is too large (fidelity drops below 0.8), response is modified
4. You see corrected version (intervention logged)

**The Math (Simply):** Your purpose becomes a point in meaning-space. Each response is also a point. TELOS measures how far apart they are - closer = better aligned.

**Full TELOS:** You'll have complete control over your PA. You decide how your session is governed - edit it, replace it, or define it from scratch."""
        ),

        # Turn 8: How does TELOS handle context?
        (
            "How does TELOS handle long conversations?",
            """TELOS tracks **context** across the entire conversation:

- Your PA stays consistent throughout the session
- Each turn's fidelity is measured against your original purpose
- Drift detection catches gradual topic shifts

**Context Window:** TELOS doesn't forget your purpose as the conversation grows. The PA remains your North Star."""
        ),

        # Turn 9: What about customization?
        (
            "Can I customize how TELOS works?",
            """In **Full TELOS**, you'll have extensive control:

- Set your own fidelity thresholds (how strict is "too low"?)
- Define custom boundaries and scope
- Choose intervention styles (gentle nudge vs hard redirect)

**Beta Mode:** For now, you're seeing TELOS with default settings optimized for learning."""
        ),

        # Turn 10: Completion
        (
            "This is helpful! What else should I know?",
            """✅ **Fidelity** tracks alignment with your purpose
✅ **Primacy Attractor** guides responses
✅ **Privacy** preserved through math

**Next:**
- Ask me anything (type below!)
- Switch to **BETA tab** to try TELOS yourself
- Click **handshake icon** (→) anytime for help

---
**🎉 Demo complete - BETA tab unlocked!**"""
        )
    ]


def get_demo_attractor_config() -> dict:
    """Get pre-configured Primacy Attractor for TELOS Demo Mode."""
    return {
        "purpose": [
            "Teach users about TELOS framework fundamentals",
            "Make AI governance clear and approachable",
            "Build user confidence in understanding alignment"
        ],
        "scope": [
            "TELOS core concepts (Fidelity, PA, Drift, Privacy)",
            "How governance works in practice",
            "Why alignment matters for AI safety"
        ],
        "boundaries": [
            "Keep explanations simple and jargon-free",
            "Use Steward's warm, professional tone",
            "Focus on user understanding, not technical depth"
        ],
        "privacy_level": 0.95,
        "constraint_tolerance": 0.25,
        "task_priority": 0.8
    }


def get_steward_intro_message() -> str:
    """
    Get Steward's introduction message that appears before the demo starts.

    Returns:
        str: Steward's introduction
    """
    return """Hello! I'm **Steward**, your guide to understanding TELOS.

Over the next few minutes, you'll watch a conversation unfold between a curious user and me, showing you exactly what TELOS does and why it matters.

**What you'll see:**
- 10 conversation turns revealing TELOS fundamentals
- Questions typing out in real-time
- My responses explaining key concepts
- Governance metrics updating live

**Pay attention to:**
- The **Fidelity** score at the top (shows alignment)
- The conversation flow (shows how governance works)
- The metrics below (shows what TELOS tracks)

After the demo, you can ask me anything you want!

Ready? Let's begin! 👇"""


def get_demo_welcome_message() -> str:
    """
    Get welcome message for Demo Mode (shown before Steward intro).

    Returns:
        str: Formatted welcome message in markdown
    """
    return """# Welcome to TELOS Demo Mode! 🔭

Click **"Start Demo"** below to begin your guided tour."""


def get_demo_completion_message() -> str:
    """
    Get completion message when demo finishes.

    Returns:
        str: Formatted completion message
    """
    return """

---

**Remember:** I'm always available via the **handshake icon** (→) while you use TELOS. Click it anytime to ask questions.

**Next steps:**
- Switch to **BETA tab** to try TELOS yourself
- Ask me anything about what you just learned (type below!)
- Explore the interface with confidence!
"""
