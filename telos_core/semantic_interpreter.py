"""
Semantic Interpreter - Bridge between Mathematical Governance and Linguistic Output

This module translates fidelity metrics into concrete linguistic specifications.
It is the intermediary layer that uses two focal points:
  1. Fidelity Value - where we are on the alignment spectrum
  2. Purpose - what we're maintaining (the semantic anchor)

The interpreter maps mathematical drift measurements to specific linguistic features
that scale proportionally to produce the corrective force needed to restore primacy state.

NO abstract tone words ("be warm", "be direct").
ONLY concrete linguistic specifications that an LLM can execute deterministically.
"""

from dataclasses import dataclass
from typing import Optional
import logging

# Setup logger
logger = logging.getLogger(__name__)

# Proportional controller gain (from proportional_controller.py)
K_ATTRACTOR = 1.5


@dataclass
class SemanticSpec:
    """
    Concrete linguistic specification for redirect responses.

    These are measurable, executable features - not abstract tone guidance.
    Each feature is deterministic: the LLM knows exactly what to produce.
    """
    # Sentence structure
    sentence_form: str          # "questions", "soft statements", "direct statements", "directives"

    # Hedging level
    hedging: str                # "heavy" (might, perhaps, could), "light" (maybe), "none"
    hedging_words: list         # Specific words to use/avoid

    # Options to offer
    options_count: int          # How many paths to offer (1-3)
    options_style: str          # "invitations", "suggestions", "single path"

    # Drift acknowledgment
    drift_acknowledgment: str   # "implicit", "brief", "explicit", "named"

    # Shift mention (offer to change purpose)
    include_shift_mention: bool
    shift_urgency: str          # "casual", "available", "prominent"

    # Purpose reference
    purpose_reference: str      # How to reference the purpose

    # Calculated values (for transparency/logging)
    fidelity: float
    strength: float

    def to_prompt_block(self, purpose: str) -> str:
        """
        Convert this spec into a compact prompt block.
        No redundancy - each instruction appears once.
        """
        # Compact hedging
        hedge = f"Hedge: {', '.join(self.hedging_words)}" if self.hedging_words else "No hedging"

        # Compact options
        opts = f"{self.options_count} {self.options_style}" if self.options_count > 1 else self.options_style

        # Compact drift handling
        drift_map = {
            "implicit": "Acknowledge interest, don't name drift",
            "brief": "Brief tangent note",
            "explicit": "Name the drift",
            "named": "State clearly: far from purpose"
        }
        drift = drift_map.get(self.drift_acknowledgment, "")

        # Compact shift
        shift = "Mention shift option" if self.include_shift_mention else ""

        # Build minimal spec block
        instructions = [
            f"Form: {self.sentence_form}",
            hedge,
            f"Options: {opts}",
            f"Drift: {drift}",
        ]
        if shift:
            instructions.append(shift)

        return "\n".join(instructions)


def interpret(fidelity: float, purpose: str) -> SemanticSpec:
    """
    The Interpreter Algorithm.

    Takes fidelity value and purpose, returns concrete linguistic specifications
    scaled proportionally to restore primacy state.

    Args:
        fidelity: Current alignment score (0.0 to 1.0)
        purpose: User's stated purpose (semantic anchor)

    Returns:
        SemanticSpec with all linguistic features defined
    """
    # Calculate intervention strength using proportional controller formula
    error_signal = 1.0 - fidelity
    strength = min(K_ATTRACTOR * error_signal, 1.0)

    # Map strength to linguistic features
    # The boundaries are calibrated to the proportional controller output

    if strength < 0.45:
        # MINIMAL CORRECTION (fidelity ~0.70+, just at threshold)
        return SemanticSpec(
            sentence_form="questions and invitations",
            hedging="heavy",
            hedging_words=["might", "perhaps", "could", "wondering if"],
            options_count=2,
            options_style="invitations to explore",
            drift_acknowledgment="implicit",
            include_shift_mention=False,
            shift_urgency="",
            purpose_reference="Since you mentioned wanting to {purpose}...",
            fidelity=fidelity,
            strength=strength
        )

    elif strength < 0.60:
        # LIGHT CORRECTION (fidelity ~0.60-0.70, YELLOW zone)
        return SemanticSpec(
            sentence_form="soft statements with question endings",
            hedging="light",
            hedging_words=["maybe", "possibly"],
            options_count=2,
            options_style="suggestions",
            drift_acknowledgment="brief",
            include_shift_mention=True,
            shift_urgency="casual",
            purpose_reference="Your goal was to {purpose} - ",
            fidelity=fidelity,
            strength=strength
        )

    elif strength < 0.75:
        # MODERATE CORRECTION (fidelity ~0.50-0.60, ORANGE zone)
        return SemanticSpec(
            sentence_form="direct statements",
            hedging="none",
            hedging_words=[],
            options_count=1,
            options_style="clear suggestion",
            drift_acknowledgment="explicit",
            include_shift_mention=True,
            shift_urgency="available",
            purpose_reference="You came here to {purpose}. ",
            fidelity=fidelity,
            strength=strength
        )

    elif strength < 0.85:
        # FIRM CORRECTION (fidelity ~0.40-0.50, approaching RED)
        return SemanticSpec(
            sentence_form="directives",
            hedging="none",
            hedging_words=[],
            options_count=1,
            options_style="single path back",
            drift_acknowledgment="named",
            include_shift_mention=True,
            shift_urgency="available",
            purpose_reference="Your stated purpose: {purpose}. ",
            fidelity=fidelity,
            strength=strength
        )

    else:
        # STRONG CORRECTION (fidelity < 0.40, deep RED)
        return SemanticSpec(
            sentence_form="clear directives with anchoring statement",
            hedging="none",
            hedging_words=[],
            options_count=1,
            options_style="the path back",
            drift_acknowledgment="named",
            include_shift_mention=True,
            shift_urgency="prominent",
            purpose_reference="This is far from your purpose: {purpose}. ",
            fidelity=fidelity,
            strength=strength
        )


# =============================================================================
# EXEMPLAR CORPUS - Internal RAG for Organic Responses
# =============================================================================

EXEMPLAR_CORPUS = {
    "minimal": [
        "That's an interesting angle - I'm wondering if we might explore how that connects to {purpose}?",
        "Curious thought! Perhaps there's a way to bridge that with what you mentioned wanting to focus on?",
        "That could be worth exploring. I'm thinking it might connect back to {purpose} in an interesting way?",
    ],
    "light": [
        "That's a bit of a tangent. Your goal was {purpose} - maybe we could look at the connection there? Or if priorities shifted, we can adjust.",
        "I hear you. Since you mentioned wanting to {purpose}, perhaps we could explore how this relates? If your focus has changed, that's fine too.",
    ],
    "moderate": [
        "That's moved away from your focus. You came here to {purpose}. Let me suggest we look at the aspect closest to what you asked about.",
        "I notice that's off your stated topic. Your goal was {purpose} - here's a path back that might still address your interest.",
    ],
    "firm": [
        "That's quite far from your purpose: {purpose}. Here's a path back. If your priorities have changed, you can shift focus.",
        "We've moved significantly from what you stated: {purpose}. Let me redirect us. Update your focus if your goals have changed.",
    ],
    "strong": [
        "This is far from your purpose: {purpose}. I'll redirect us back. If your focus has genuinely shifted, you can update it and we'll go from there.",
        "That's well outside what you came here for: {purpose}. Let's get back on track. You can always update your focus if things have changed.",
    ],
}


def get_exemplar(strength: float, purpose: str = "{purpose}") -> Optional[str]:
    """
    Return a calibrated exemplar response for this strength band.

    Args:
        strength: Calculated intervention strength (0.0 to 1.0)
        purpose: User's purpose to interpolate into exemplar

    Returns:
        Exemplar string with purpose interpolated
    """
    import hashlib

    if strength < 0.45:
        band = "minimal"
    elif strength < 0.60:
        band = "light"
    elif strength < 0.75:
        band = "moderate"
    elif strength < 0.85:
        band = "firm"
    else:
        band = "strong"

    exemplars = EXEMPLAR_CORPUS[band]
    index = int((strength * 1000) % len(exemplars))
    return exemplars[index].format(purpose=purpose)


def get_behavioral_fidelity_band(user_fidelity: float) -> str:
    """
    Get the expected intervention band name for a given user fidelity.

    Args:
        user_fidelity: User fidelity value (0.0 to 1.0)

    Returns:
        Band name: "minimal", "light", "moderate", "firm", or "strong"
    """
    error_signal = 1.0 - user_fidelity
    strength = min(K_ATTRACTOR * error_signal, 1.0)

    if strength < 0.45:
        return "minimal"
    elif strength < 0.60:
        return "light"
    elif strength < 0.75:
        return "moderate"
    elif strength < 0.85:
        return "firm"
    else:
        return "strong"


def compute_behavioral_fidelity(
    ai_response: str, user_fidelity: float, embedding_provider=None
) -> float:
    """
    Compute AI Behavioral Fidelity during interventions.

    Measures how well the AI response matches expected intervention behavior
    by comparing against the EXEMPLAR_CORPUS.

    Args:
        ai_response: The AI's intervention response text
        user_fidelity: User fidelity that triggered intervention
        embedding_provider: Embedding provider. If None, uses cached MPNet provider.

    Returns:
        Behavioral fidelity score (0.0 to 1.0).
    """
    import numpy as np

    if embedding_provider is None:
        from telos_core.embedding_provider import get_cached_mpnet_provider
        embedding_provider = get_cached_mpnet_provider()

    # Determine which band based on user fidelity
    band = get_behavioral_fidelity_band(user_fidelity)

    # Get exemplars for this band
    exemplars = EXEMPLAR_CORPUS[band]

    # Embed the AI response
    ai_embedding = embedding_provider.encode(ai_response)

    # Compute max similarity across all exemplars in the band
    max_similarity = 0.0
    for exemplar in exemplars:
        generic_exemplar = exemplar.format(purpose="your stated purpose")
        exemplar_embedding = embedding_provider.encode(generic_exemplar)
        similarity = np.dot(ai_embedding, exemplar_embedding) / (
            np.linalg.norm(ai_embedding) * np.linalg.norm(exemplar_embedding) + 1e-10
        )
        max_similarity = max(max_similarity, float(similarity))

    # Normalize: map raw MPNet similarity to display range
    # MPNet raw 0.40 for AI text -> display 0.70 (GREEN threshold)
    normalized = min(1.0, max(0.0, max_similarity * 1.75))
    return normalized
