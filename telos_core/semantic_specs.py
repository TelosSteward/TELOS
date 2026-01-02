"""
TELOS Semantic Specifications
=============================

Maps fidelity scores to concrete linguistic specifications.
Provides guidance for how AI responses should be styled
based on intervention strength.

Extended for Agentic AI Governance:
- ActionDecision enum for TELOS Tools matching
- Intent-to-Tool fidelity interpretation
- INERT state when no tool matches user intent

Factored from TELOS Observatory V3 SemanticInterpreter.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple
import random

from .constants import (
    STRENGTH_BANDS,
    FIDELITY_GREEN,
    FIDELITY_YELLOW,
    FIDELITY_ORANGE,
    SIMILARITY_BASELINE,
)


# =============================================================================
# AGENTIC ACTION DECISIONS (TELOS Tools)
# =============================================================================
# CRITICAL DISTINCTION: Tool selection is BINARY, not a range.
#
# Semantic alignment (conversations): 70-100% fidelity = "aligned enough"
# because interpretation is involved - AI and user negotiate meaning.
#
# Tool selection (agentic): Either the tool matches or it doesn't.
# The 0-100 scale works for MEASUREMENT, but DECISION thresholds are tighter.
# You cannot "kind of" execute the right tool - it's correct or not.
# =============================================================================

# Tighter thresholds for tool selection (no interpretation wiggle room)
AGENTIC_EXECUTE_THRESHOLD = 0.85    # Must be highly confident to execute
AGENTIC_CLARIFY_THRESHOLD = 0.70    # Close match but verify first
AGENTIC_SUGGEST_THRESHOLD = 0.50    # Vaguely related, offer alternatives
# Below SUGGEST = INERT (no tool matches)


class ActionDecision(Enum):
    """
    Action decisions for TELOS Tools matching.

    Unlike semantic zones (which allow ranges), tool selection is discrete:
    - EXECUTE: High-fidelity match, proceed with execution
    - CLARIFY: Close match, but verify user intent before acting
    - SUGGEST: Vague match, suggest alternatives to user
    - INERT: No match, agent goes inert (no hallucinated capabilities)
    - ESCALATE: No match + high-risk context, require human expert review

    ESCALATE vs INERT:
    - INERT: Agent acknowledges it can't help, conversation continues
    - ESCALATE: Agent pauses, human expert must intervene to proceed

    Use ESCALATE when:
    - Tool selection involves irreversible actions (financial, medical, legal)
    - Confidence is low but stakes are high
    - Domain expertise required for decision validation
    - Regulatory or compliance requirements mandate human oversight
    """
    EXECUTE = "execute"     # fidelity >= 0.85: Confirmed match, proceed
    CLARIFY = "clarify"     # fidelity 0.70-0.84: Verify before acting
    SUGGEST = "suggest"     # fidelity 0.50-0.69: Offer alternatives
    INERT = "inert"         # fidelity < 0.50: No match, stay inert
    ESCALATE = "escalate"   # fidelity < 0.50 + high_risk: Human expert required


class StrengthBand(Enum):
    """Intervention strength bands."""
    MINIMAL = "minimal"     # strength < 0.45
    LIGHT = "light"         # strength 0.45-0.60
    MODERATE = "moderate"   # strength 0.60-0.75
    FIRM = "firm"           # strength 0.75-0.85
    STRONG = "strong"       # strength >= 0.85


@dataclass
class SemanticSpec:
    """
    Semantic specification for response generation.

    Provides concrete guidance on linguistic style based on
    intervention strength.
    """
    band: StrengthBand
    sentence_structure: str
    hedging_level: str
    assertiveness: str
    purpose_reference: str
    example_phrases: List[str]

    def to_dict(self) -> dict:
        return {
            "band": self.band.value,
            "sentence_structure": self.sentence_structure,
            "hedging_level": self.hedging_level,
            "assertiveness": self.assertiveness,
            "purpose_reference": self.purpose_reference,
            "example_phrases": self.example_phrases,
        }


# =============================================================================
# STRENGTH BAND SPECIFICATIONS
# =============================================================================

SEMANTIC_SPECS: Dict[StrengthBand, SemanticSpec] = {
    StrengthBand.MINIMAL: SemanticSpec(
        band=StrengthBand.MINIMAL,
        sentence_structure="Questions and soft suggestions",
        hedging_level="Heavy hedging (might, perhaps, could)",
        assertiveness="Very low - exploratory tone",
        purpose_reference="Implicit, through context",
        example_phrases=[
            "I wonder if we might explore...",
            "Perhaps we could consider...",
            "Would it be helpful to think about...",
            "It might be worth noting...",
        ],
    ),
    StrengthBand.LIGHT: SemanticSpec(
        band=StrengthBand.LIGHT,
        sentence_structure="Soft statements with qualifiers",
        hedging_level="Light hedging (may, generally, often)",
        assertiveness="Low - gentle guidance",
        purpose_reference="Mentioned naturally in flow",
        example_phrases=[
            "This may connect to what we discussed...",
            "Generally, it helps to focus on...",
            "Often, returning to our purpose helps...",
            "It seems like we could align this with...",
        ],
    ),
    StrengthBand.MODERATE: SemanticSpec(
        band=StrengthBand.MODERATE,
        sentence_structure="Direct statements",
        hedging_level="No hedging",
        assertiveness="Medium - clear guidance",
        purpose_reference="Explicitly stated",
        example_phrases=[
            "Let's return to our focus on...",
            "Our purpose here is...",
            "To stay aligned with our goal...",
            "This connects directly to...",
        ],
    ),
    StrengthBand.FIRM: SemanticSpec(
        band=StrengthBand.FIRM,
        sentence_structure="Directive statements",
        hedging_level="None - confident tone",
        assertiveness="High - clear direction",
        purpose_reference="Named and emphasized",
        example_phrases=[
            "I notice we've drifted from our purpose...",
            "Let's refocus on what we set out to do...",
            "Our agreed focus is... Let's return to that.",
            "This discussion is moving away from...",
        ],
    ),
    StrengthBand.STRONG: SemanticSpec(
        band=StrengthBand.STRONG,
        sentence_structure="Clear directives",
        hedging_level="None - authoritative",
        assertiveness="Very high - firm redirection",
        purpose_reference="Prominent and repeated",
        example_phrases=[
            "I need to redirect us to our stated purpose...",
            "This is outside our agreed scope. Our focus is...",
            "Let's pause and return to our core objective...",
            "To maintain alignment, we need to focus on...",
        ],
    ),
}


# =============================================================================
# EXEMPLAR CORPUS
# =============================================================================

EXEMPLAR_CORPUS = {
    StrengthBand.MINIMAL: [
        "I wonder if there might be a connection here...",
        "Perhaps we could explore this angle...",
        "Could this relate to what we're working on?",
        "It might be interesting to consider...",
        "Would you like to think about how this fits?",
    ],
    StrengthBand.LIGHT: [
        "This seems to connect to our discussion...",
        "We might find value in linking this to...",
        "Generally, this type of topic relates to...",
        "It often helps to frame this in terms of...",
        "There may be a path back to our focus here...",
    ],
    StrengthBand.MODERATE: [
        "Let's connect this to our purpose.",
        "Our focus is on... How does this relate?",
        "To stay aligned, let's consider...",
        "I want to make sure we're addressing...",
        "This is an opportunity to return to...",
    ],
    StrengthBand.FIRM: [
        "I notice we've moved away from our goal.",
        "Let's refocus on what we set out to accomplish.",
        "Our stated purpose is... Let's return to that.",
        "This seems off-topic. Our focus should be...",
        "To maintain alignment with our objectives...",
    ],
    StrengthBand.STRONG: [
        "I need to redirect this conversation.",
        "This is outside our agreed scope.",
        "Let's pause and return to our core purpose.",
        "Our commitment is to... Let's honor that.",
        "To stay true to our objectives, we must focus on...",
    ],
}


# =============================================================================
# INTERPRETATION FUNCTIONS
# =============================================================================

def get_strength_band(intervention_strength: float) -> StrengthBand:
    """
    Get strength band from intervention strength.

    Uses thresholds from STRENGTH_BANDS constant.

    Args:
        intervention_strength: Strength value in [0, 1]

    Returns:
        Corresponding StrengthBand
    """
    bands = STRENGTH_BANDS

    if intervention_strength < bands["MINIMAL"]["max_strength"]:
        return StrengthBand.MINIMAL
    elif intervention_strength < bands["LIGHT"]["max_strength"]:
        return StrengthBand.LIGHT
    elif intervention_strength < bands["MODERATE"]["max_strength"]:
        return StrengthBand.MODERATE
    elif intervention_strength < bands["FIRM"]["max_strength"]:
        return StrengthBand.FIRM
    else:
        return StrengthBand.STRONG


def interpret_fidelity(
    fidelity: float,
    intervention_strength: Optional[float] = None,
) -> SemanticSpec:
    """
    Get semantic specification for fidelity level.

    If intervention_strength provided, uses that directly.
    Otherwise, derives strength from fidelity.

    Args:
        fidelity: Fidelity score [0, 1]
        intervention_strength: Optional direct strength

    Returns:
        SemanticSpec for response generation
    """
    if intervention_strength is not None:
        strength = intervention_strength
    else:
        # Derive strength from fidelity
        # Lower fidelity = higher intervention strength
        strength = 1.0 - fidelity

    band = get_strength_band(strength)
    return SEMANTIC_SPECS[band]


def get_exemplar(
    band: StrengthBand,
    random_selection: bool = True,
) -> str:
    """
    Get an exemplar phrase for the strength band.

    Args:
        band: Strength band
        random_selection: If True, select randomly; otherwise first

    Returns:
        Example phrase for the band
    """
    exemplars = EXEMPLAR_CORPUS.get(band, EXEMPLAR_CORPUS[StrengthBand.MODERATE])

    if random_selection:
        return random.choice(exemplars)
    else:
        return exemplars[0]


def generate_intervention_prompt(
    purpose_text: str,
    fidelity: float,
    intervention_strength: float,
    context: Optional[str] = None,
) -> str:
    """
    Generate an intervention prompt for AI response.

    Creates a prompt that guides the AI to respond in a way
    consistent with the intervention strength.

    Args:
        purpose_text: The PA purpose statement
        fidelity: Current fidelity score
        intervention_strength: Intervention strength
        context: Optional additional context

    Returns:
        Prompt string for AI
    """
    spec = interpret_fidelity(fidelity, intervention_strength)
    exemplar = get_exemplar(spec.band)

    prompt = f"""
You are responding in a governed conversation with the following purpose:
"{purpose_text}"

Current alignment status:
- Fidelity: {fidelity:.2f}
- Intervention strength: {intervention_strength:.2f}
- Response style: {spec.band.value}

Linguistic guidelines for this response:
- Sentence structure: {spec.sentence_structure}
- Hedging level: {spec.hedging_level}
- Assertiveness: {spec.assertiveness}
- Purpose reference: {spec.purpose_reference}

Example phrase for this style: "{exemplar}"
"""

    if context:
        prompt += f"\nAdditional context: {context}"

    return prompt


# =============================================================================
# AGENTIC AI SPECIFICATIONS (TELOS Tools)
# =============================================================================

@dataclass
class AgenticSpec:
    """
    Specification for agentic action decisions.

    Unlike SemanticSpec (which provides gradual styling guidance),
    AgenticSpec provides binary execution guidance for tool selection.
    """
    decision: ActionDecision
    should_execute: bool
    requires_confirmation: bool
    user_communication: str
    agent_behavior: str
    example_responses: List[str]

    def to_dict(self) -> dict:
        return {
            "decision": self.decision.value,
            "should_execute": self.should_execute,
            "requires_confirmation": self.requires_confirmation,
            "user_communication": self.user_communication,
            "agent_behavior": self.agent_behavior,
            "example_responses": self.example_responses,
        }


AGENTIC_SPECS: Dict[ActionDecision, AgenticSpec] = {
    ActionDecision.EXECUTE: AgenticSpec(
        decision=ActionDecision.EXECUTE,
        should_execute=True,
        requires_confirmation=False,
        user_communication="Proceed with confirmed tool execution",
        agent_behavior="Execute tool directly, report results",
        example_responses=[
            "Executing [tool] for your request...",
            "Running [tool] now...",
            "Processing with [tool]...",
        ],
    ),
    ActionDecision.CLARIFY: AgenticSpec(
        decision=ActionDecision.CLARIFY,
        should_execute=False,
        requires_confirmation=True,
        user_communication="Confirm intent before tool execution",
        agent_behavior="Present matched tool, ask for confirmation",
        example_responses=[
            "I found [tool] that may help. Should I proceed?",
            "Did you want me to use [tool] for this?",
            "I can execute [tool] - is that what you intended?",
        ],
    ),
    ActionDecision.SUGGEST: AgenticSpec(
        decision=ActionDecision.SUGGEST,
        should_execute=False,
        requires_confirmation=True,
        user_communication="Suggest related tools, ask for guidance",
        agent_behavior="Present alternatives, do not assume intent",
        example_responses=[
            "I have some tools that might relate: [list]. Which would help?",
            "This could involve [tool A] or [tool B]. Can you clarify?",
            "I'm not certain which tool fits best. Can you specify?",
        ],
    ),
    ActionDecision.INERT: AgenticSpec(
        decision=ActionDecision.INERT,
        should_execute=False,
        requires_confirmation=False,
        user_communication="Acknowledge limitation honestly",
        agent_behavior="Go inert - do not hallucinate capabilities",
        example_responses=[
            "I don't have a tool that matches this request.",
            "This is outside my available capabilities.",
            "I cannot perform this action with my current tools.",
        ],
    ),
    ActionDecision.ESCALATE: AgenticSpec(
        decision=ActionDecision.ESCALATE,
        should_execute=False,
        requires_confirmation=True,  # Human MUST intervene
        user_communication="Request human expert review",
        agent_behavior="Pause execution, escalate to human expert via interrupt()",
        example_responses=[
            "This requires human expert review before proceeding.",
            "I'm pausing for a qualified person to evaluate this action.",
            "This decision needs human oversight - escalating for review.",
        ],
    ),
}


@dataclass
class ToolMatchResult:
    """
    Result of intent-to-tool fidelity matching.

    Contains the decision, fidelity score, and matched tool info.
    """
    decision: ActionDecision
    fidelity: float
    spec: AgenticSpec
    matched_tool: Optional[str] = None
    confidence: float = 0.0
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "decision": self.decision.value,
            "fidelity": self.fidelity,
            "spec": self.spec.to_dict(),
            "matched_tool": self.matched_tool,
            "confidence": self.confidence,
            "reason": self.reason,
        }


def get_action_decision(
    fidelity: float,
    high_risk: bool = False,
) -> ActionDecision:
    """
    Get action decision from intent-to-tool fidelity.

    Uses tighter thresholds than semantic alignment because
    tool selection is binary - you either execute the right
    tool or you don't. No interpretation wiggle room.

    Args:
        fidelity: Intent-to-tool fidelity score [0, 1]
        high_risk: If True, low-fidelity situations escalate to human
                   instead of going inert. Use for irreversible actions
                   (financial, medical, legal) or compliance requirements.

    Returns:
        ActionDecision (EXECUTE, CLARIFY, SUGGEST, INERT, or ESCALATE)
    """
    # Hard block for baseline violations (extreme mismatch)
    if fidelity < SIMILARITY_BASELINE:
        return ActionDecision.ESCALATE if high_risk else ActionDecision.INERT

    # Tighter thresholds for tool execution
    if fidelity >= AGENTIC_EXECUTE_THRESHOLD:
        return ActionDecision.EXECUTE
    elif fidelity >= AGENTIC_CLARIFY_THRESHOLD:
        # In high-risk mode, even CLARIFY requires human confirmation
        return ActionDecision.CLARIFY
    elif fidelity >= AGENTIC_SUGGEST_THRESHOLD:
        return ActionDecision.SUGGEST
    else:
        # Low fidelity: escalate if high-risk, otherwise go inert
        return ActionDecision.ESCALATE if high_risk else ActionDecision.INERT


def interpret_intent_to_tool_fidelity(
    fidelity: float,
    matched_tool: Optional[str] = None,
    raw_similarity: Optional[float] = None,
    high_risk: bool = False,
) -> ToolMatchResult:
    """
    Interpret intent-to-tool fidelity and return action specification.

    This is the core function for TELOS Tools governance. Unlike
    semantic interpretation (which allows ranges), this makes a
    binary decision about whether to execute a tool.

    Args:
        fidelity: Intent-to-tool fidelity score [0, 1]
        matched_tool: Name of the best-matching tool (if any)
        raw_similarity: Optional raw cosine similarity for logging
        high_risk: If True, low-fidelity triggers human escalation instead
                   of going inert. Use for irreversible actions (financial,
                   medical, legal) or when compliance requires human oversight.

    Returns:
        ToolMatchResult with decision and guidance
    """
    decision = get_action_decision(fidelity, high_risk=high_risk)
    spec = AGENTIC_SPECS[decision]

    # Calculate confidence as distance from threshold
    if decision == ActionDecision.EXECUTE:
        confidence = min(1.0, (fidelity - AGENTIC_EXECUTE_THRESHOLD) / 0.15 + 0.85)
        reason = f"High-fidelity match ({fidelity:.2f} >= {AGENTIC_EXECUTE_THRESHOLD})"
    elif decision == ActionDecision.CLARIFY:
        confidence = (fidelity - AGENTIC_CLARIFY_THRESHOLD) / (AGENTIC_EXECUTE_THRESHOLD - AGENTIC_CLARIFY_THRESHOLD)
        reason = f"Close match, needs confirmation ({fidelity:.2f})"
    elif decision == ActionDecision.SUGGEST:
        confidence = (fidelity - AGENTIC_SUGGEST_THRESHOLD) / (AGENTIC_CLARIFY_THRESHOLD - AGENTIC_SUGGEST_THRESHOLD)
        reason = f"Vague match, suggest alternatives ({fidelity:.2f})"
    elif decision == ActionDecision.ESCALATE:
        confidence = 0.0
        reason = f"High-risk context: escalating to human expert ({fidelity:.2f})"
    else:  # INERT
        confidence = 0.0
        reason = f"No suitable tool ({fidelity:.2f} < {AGENTIC_SUGGEST_THRESHOLD})"

    return ToolMatchResult(
        decision=decision,
        fidelity=fidelity,
        spec=spec,
        matched_tool=matched_tool if decision not in [ActionDecision.INERT, ActionDecision.ESCALATE] else None,
        confidence=max(0.0, min(1.0, confidence)),
        reason=reason,
    )


def should_execute_tool(fidelity: float) -> bool:
    """
    Quick check: should the agent execute the tool?

    Args:
        fidelity: Intent-to-tool fidelity score

    Returns:
        True only if fidelity >= EXECUTE threshold
    """
    return fidelity >= AGENTIC_EXECUTE_THRESHOLD


def should_go_inert(fidelity: float) -> bool:
    """
    Quick check: should the agent go inert?

    Args:
        fidelity: Intent-to-tool fidelity score

    Returns:
        True if no tool matches (below SUGGEST threshold)
    """
    return fidelity < AGENTIC_SUGGEST_THRESHOLD or fidelity < SIMILARITY_BASELINE
