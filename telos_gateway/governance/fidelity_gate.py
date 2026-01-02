"""
Fidelity Gate
=============

The core governance component - decides whether to EXECUTE, CLARIFY,
SUGGEST, INERT, or ESCALATE based on fidelity measurement.

This is the "Constitutional Filter" applied to agentic AI.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

from ..config import config
from ..models.openai_types import (
    ChatCompletionRequest,
    ChatMessage,
    Tool,
    extract_last_user_message,
    extract_tool_names,
)
from ..models.governance_types import (
    ActionDecision,
    GovernanceDecision,
    GovernanceResult,
)
from .pa_extractor import PrimacyAttractor

logger = logging.getLogger(__name__)


class FidelityGate:
    """
    The TELOS Governance Gate for agentic AI.

    Measures fidelity of requests against the Primacy Attractor
    and makes governance decisions.

    Agentic thresholds are TIGHTER than conversational because
    tool selection is binary - you either execute the right tool
    or you don't. No interpretation wiggle room.
    """

    def __init__(
        self,
        embed_fn: Callable[[str], np.ndarray],
        execute_threshold: float = None,
        clarify_threshold: float = None,
        suggest_threshold: float = None,
        baseline_threshold: float = None,
    ):
        """
        Initialize the fidelity gate.

        Args:
            embed_fn: Function to embed text strings
            execute_threshold: Fidelity threshold for EXECUTE (default 0.85)
            clarify_threshold: Fidelity threshold for CLARIFY (default 0.70)
            suggest_threshold: Fidelity threshold for SUGGEST (default 0.50)
            baseline_threshold: Hard block threshold (default 0.20)
        """
        self.embed_fn = embed_fn

        # Thresholds (use config defaults if not specified)
        self.execute_threshold = execute_threshold or config.agentic_execute_threshold
        self.clarify_threshold = clarify_threshold or config.agentic_clarify_threshold
        self.suggest_threshold = suggest_threshold or config.agentic_suggest_threshold
        self.baseline_threshold = baseline_threshold or config.similarity_baseline

    def check_request(
        self,
        request: ChatCompletionRequest,
        pa: PrimacyAttractor,
        high_risk: bool = False,
    ) -> GovernanceResult:
        """
        Check a request against the Primacy Attractor.

        This is the main entry point for governance.

        Args:
            request: The chat completion request
            pa: The Primacy Attractor to check against
            high_risk: If True, low fidelity triggers ESCALATE instead of INERT

        Returns:
            GovernanceResult with decision and metadata
        """
        import uuid
        from datetime import datetime

        request_id = str(uuid.uuid4())[:8]

        # Step 1: Check input fidelity (last user message)
        input_decision = self._check_input_fidelity(request, pa, high_risk)

        # Step 2: Check tool fidelity (if tools are defined)
        tool_decisions = None
        tools_blocked = 0
        if request.tools or request.functions:
            tool_decisions = self._check_tool_fidelity(request, pa)
            tools_blocked = sum(1 for d in tool_decisions.values() if not d.should_forward)

        # Step 3: Determine final decision
        # If input is blocked, don't forward
        # If any critical tools are blocked, may need to clarify
        if not input_decision.should_forward:
            final_decision = input_decision.decision
            forwarded = False
        elif tools_blocked > 0 and input_decision.decision == ActionDecision.EXECUTE:
            # Downgrade to CLARIFY if tools were blocked
            final_decision = ActionDecision.CLARIFY
            forwarded = True  # Still forward, but with modified tool set
        else:
            final_decision = input_decision.decision
            forwarded = input_decision.should_forward

        # Build result
        return GovernanceResult(
            request_id=request_id,
            timestamp=datetime.utcnow(),
            pa_text=pa.text,
            pa_source=pa.source,
            input_fidelity=input_decision.fidelity,
            input_decision=input_decision.decision,
            input_blocked=not input_decision.should_forward,
            tools_checked=len(tool_decisions) if tool_decisions else 0,
            tools_blocked=tools_blocked,
            tool_decisions=tool_decisions,
            final_decision=final_decision,
            forwarded_to_llm=forwarded,
            governance_response=self._generate_governance_response(input_decision) if not forwarded else None,
        )

    def _check_input_fidelity(
        self,
        request: ChatCompletionRequest,
        pa: PrimacyAttractor,
        high_risk: bool = False,
    ) -> GovernanceDecision:
        """Check fidelity of the input (last user message)."""
        # Extract last user message
        user_message = extract_last_user_message(request.messages)

        if not user_message:
            # No user message - this is unusual but allowed
            return GovernanceDecision(
                decision=ActionDecision.EXECUTE,
                fidelity=1.0,
                raw_similarity=1.0,
                reason="No user message to check",
                should_forward=True,
            )

        # Calculate fidelity
        raw_similarity, fidelity = self._calculate_fidelity(user_message, pa)

        # Make decision based on thresholds
        decision = self._make_decision(fidelity, raw_similarity, high_risk)

        return GovernanceDecision(
            decision=decision,
            fidelity=fidelity,
            raw_similarity=raw_similarity,
            reason=self._get_decision_reason(decision, fidelity),
            should_forward=decision in [ActionDecision.EXECUTE, ActionDecision.CLARIFY, ActionDecision.SUGGEST],
        )

    def _check_tool_fidelity(
        self,
        request: ChatCompletionRequest,
        pa: PrimacyAttractor,
    ) -> Dict[str, GovernanceDecision]:
        """
        Check fidelity of each tool against the PA.

        Tools with low fidelity should be blocked - they're not
        aligned with the agent's declared purpose.
        """
        tool_decisions = {}

        # Get tool definitions
        tools = []
        if request.tools:
            for tool in request.tools:
                tools.append({
                    "name": tool.function.name,
                    "description": tool.function.description or tool.function.name,
                })
        if request.functions:
            for func in request.functions:
                tools.append({
                    "name": func.name,
                    "description": func.description or func.name,
                })

        # Check each tool
        for tool in tools:
            tool_text = f"{tool['name']}: {tool['description']}"
            raw_similarity, fidelity = self._calculate_fidelity(tool_text, pa)
            decision = self._make_decision(fidelity, raw_similarity, high_risk=False)

            tool_decisions[tool["name"]] = GovernanceDecision(
                decision=decision,
                fidelity=fidelity,
                raw_similarity=raw_similarity,
                reason=f"Tool '{tool['name']}' fidelity: {fidelity:.2f}",
                should_forward=decision != ActionDecision.INERT and decision != ActionDecision.ESCALATE,
            )

            if not tool_decisions[tool["name"]].should_forward:
                logger.warning(
                    f"Tool '{tool['name']}' blocked: fidelity {fidelity:.2f} below threshold"
                )

        return tool_decisions

    def _calculate_fidelity(
        self,
        text: str,
        pa: PrimacyAttractor,
    ) -> Tuple[float, float]:
        """
        Calculate fidelity of text against PA.

        Returns:
            (raw_similarity, normalized_fidelity)
        """
        # Embed the text
        text_embedding = self.embed_fn(text)

        # Calculate cosine similarity
        raw_similarity = self._cosine_similarity(text_embedding, pa.embedding)

        # Normalize to fidelity score
        # Using the same normalization as telos_core
        # Maps raw similarity to [0, 1] range with baseline adjustment
        if raw_similarity < self.baseline_threshold:
            fidelity = raw_similarity / self.baseline_threshold * 0.3
        else:
            fidelity = 0.3 + (raw_similarity - self.baseline_threshold) / (1 - self.baseline_threshold) * 0.7

        return float(raw_similarity), float(min(1.0, max(0.0, fidelity)))

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(np.dot(a, b) / (norm_a * norm_b))

    def _make_decision(
        self,
        fidelity: float,
        raw_similarity: float,
        high_risk: bool = False,
    ) -> ActionDecision:
        """
        Make governance decision based on fidelity.

        Agentic thresholds:
        - EXECUTE: >= 0.85 (high confidence)
        - CLARIFY: 0.70-0.84 (verify first)
        - SUGGEST: 0.50-0.69 (offer alternatives)
        - INERT/ESCALATE: < 0.50 (no match)
        """
        # Hard block for baseline violations
        if raw_similarity < self.baseline_threshold:
            return ActionDecision.ESCALATE if high_risk else ActionDecision.INERT

        # Threshold-based decisions
        if fidelity >= self.execute_threshold:
            return ActionDecision.EXECUTE
        elif fidelity >= self.clarify_threshold:
            return ActionDecision.CLARIFY
        elif fidelity >= self.suggest_threshold:
            return ActionDecision.SUGGEST
        else:
            return ActionDecision.ESCALATE if high_risk else ActionDecision.INERT

    def _get_decision_reason(self, decision: ActionDecision, fidelity: float) -> str:
        """Generate human-readable reason for decision."""
        reasons = {
            ActionDecision.EXECUTE: f"High fidelity ({fidelity:.2f}) - proceeding with request",
            ActionDecision.CLARIFY: f"Moderate fidelity ({fidelity:.2f}) - forwarding with governance context",
            ActionDecision.SUGGEST: f"Low fidelity ({fidelity:.2f}) - suggesting alternatives",
            ActionDecision.INERT: f"Very low fidelity ({fidelity:.2f}) - request outside agent's purpose",
            ActionDecision.ESCALATE: f"Very low fidelity ({fidelity:.2f}) in high-risk context - requiring review",
        }
        return reasons.get(decision, f"Fidelity: {fidelity:.2f}")

    def _generate_governance_response(self, decision: GovernanceDecision) -> str:
        """Generate a governance response when request is blocked."""
        if decision.decision == ActionDecision.INERT:
            return (
                "I appreciate your request, but it appears to be outside the scope of "
                "my defined purpose. I'm designed to help with specific tasks as defined "
                "in my system configuration. Could you rephrase your request to align "
                "with my intended function?"
            )
        elif decision.decision == ActionDecision.ESCALATE:
            return (
                "This request requires human review before I can proceed. The request "
                "has been flagged for expert evaluation due to potential misalignment "
                "with my operational boundaries."
            )
        else:
            return (
                f"Request governance check: {decision.reason}. "
                "Please clarify your intent."
            )
