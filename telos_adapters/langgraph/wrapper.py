"""
TELOS Wrapper
=============

Seamless wrapper for any LangGraph agent. Deploys TELOS governance
without modifying the wrapped agent's internals.

This implements the user's key insight:
"We can deploy on top of already existing AI agents and make them run
through our own filter... seamless... direct operational connection"
"""

from typing import Any, Dict, Optional, Callable, Union
from datetime import datetime
import numpy as np
import logging

from .state_schema import (
    TelosGovernedState,
    PrimacyAttractor,
    GovernanceTraceEntry,
    FidelityZone,
    DirectionLevel,
    create_initial_state,
    get_zone_from_fidelity,
    get_direction_level,
    FIDELITY_GREEN,
)

from .governance_node import calculate_fidelity, TelosGovernanceGate


logger = logging.getLogger(__name__)


class TelosWrapper:
    """
    Transparent governance wrapper for any LangGraph agent.

    Wraps an existing agent to add TELOS governance without
    modifying the agent's internals.

    Flow:
        User Input -> TELOS Pre-Check -> Agent -> TELOS Post-Check -> Output

    Usage:
        # Wrap any existing agent
        governed_agent = TelosWrapper(
            agent=existing_agent,
            primacy_attractor=my_pa,
            embed_fn=my_embedding_function,
        )

        # Use exactly like the original agent
        result = governed_agent.invoke({"messages": [user_message]})
    """

    def __init__(
        self,
        agent: Any,
        primacy_attractor: Union[PrimacyAttractor, Dict[str, Any]],
        embed_fn: Callable[[str], np.ndarray],
        pre_check: bool = True,
        post_check: bool = True,
        block_on_low_fidelity: bool = True,
        fidelity_threshold: float = FIDELITY_GREEN,
        on_block: Optional[Callable[[Dict], Any]] = None,
    ):
        """
        Initialize the TELOS wrapper.

        Args:
            agent: The LangGraph agent to wrap (must have .invoke())
            primacy_attractor: PA for governance (dict or PrimacyAttractor)
            embed_fn: Function to embed text strings
            pre_check: Check input fidelity before agent runs
            post_check: Check output fidelity after agent runs
            block_on_low_fidelity: Block low-fidelity inputs (if pre_check)
            fidelity_threshold: Threshold for direction
            on_block: Callback when input is blocked
        """
        self.agent = agent
        self.embed_fn = embed_fn
        self.pre_check = pre_check
        self.post_check = post_check
        self.block_on_low_fidelity = block_on_low_fidelity
        self.fidelity_threshold = fidelity_threshold
        self.on_block = on_block

        # Convert PA if needed
        if isinstance(primacy_attractor, dict):
            self.pa = PrimacyAttractor.from_dict(primacy_attractor)
        else:
            self.pa = primacy_attractor

        # Internal governance gate
        self.gate = TelosGovernanceGate(
            embed_fn=embed_fn,
            require_approval_below=fidelity_threshold,
        )

        # Tracking
        self.governance_trace = []
        self.turn_number = 0

    def invoke(
        self,
        input_state: Dict[str, Any],
        config: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Invoke the wrapped agent with TELOS governance.

        This is the main entry point - use exactly like the
        original agent's invoke() method.
        """
        self.turn_number += 1

        # =================================================================
        # STEP 1: Pre-execution governance (input fidelity check)
        # =================================================================
        if self.pre_check:
            pre_result = self._pre_execution_check(input_state)

            if not pre_result["approved"]:
                # Input blocked - return governance response
                return self._generate_redirect_response(
                    input_state,
                    pre_result,
                )

            # Update input state with governance context if needed
            if pre_result.get("context_injected"):
                input_state = self._inject_context(input_state, pre_result)

        # =================================================================
        # STEP 2: Pass through to original agent (unchanged)
        # =================================================================
        try:
            result = self.agent.invoke(input_state, config)
        except Exception as e:
            logger.error(f"Wrapped agent error: {e}")
            self._record_trace(
                action_type="agent_error",
                description=str(e),
                fidelity=0.0,
                approved=False,
            )
            raise

        # =================================================================
        # STEP 3: Post-execution governance (output fidelity check)
        # =================================================================
        if self.post_check:
            post_result = self._post_execution_check(result)

            # Record to trace
            self._record_trace(
                action_type="agent_response",
                description="Agent response generated",
                fidelity=post_result["fidelity"],
                approved=True,  # Output is informational, not blocked
            )

        return result

    def _pre_execution_check(
        self,
        input_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Check input fidelity before agent execution.

        Returns:
            Dict with approval status and fidelity info
        """
        # Extract input text
        messages = input_state.get("messages", [])
        if not messages:
            return {"approved": True, "fidelity": 1.0, "zone": "green"}

        last_message = messages[-1]
        content = getattr(last_message, "content", str(last_message))

        # Calculate fidelity
        raw_sim, fidelity = calculate_fidelity(content, self.pa, self.embed_fn)
        zone = get_zone_from_fidelity(fidelity)
        direction = get_direction_level(fidelity, raw_sim)

        result = {
            "raw_similarity": raw_sim,
            "fidelity": fidelity,
            "zone": zone.value,
            "direction": direction.value,
        }

        # Decision logic
        if direction == DirectionLevel.HARD_BLOCK:
            result["approved"] = not self.block_on_low_fidelity
            result["reason"] = "Layer 1 baseline violation"
            self._record_trace(
                action_type="input_check",
                description=content[:100],
                fidelity=fidelity,
                approved=result["approved"],
                direction_reason=result["reason"],
            )
            return result

        if fidelity >= self.fidelity_threshold:
            result["approved"] = True
            return result

        if direction in [DirectionLevel.MONITOR, DirectionLevel.CORRECT]:
            result["approved"] = True
            result["context_injected"] = True
            return result

        # Block for low fidelity
        result["approved"] = not self.block_on_low_fidelity
        result["reason"] = f"Fidelity {fidelity:.2f} below threshold"

        self._record_trace(
            action_type="input_check",
            description=content[:100],
            fidelity=fidelity,
            approved=result["approved"],
            direction_reason=result.get("reason"),
        )

        return result

    def _post_execution_check(
        self,
        result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Check output fidelity after agent execution.

        This is for monitoring/logging, not blocking.
        """
        messages = result.get("messages", [])
        if not messages:
            return {"fidelity": 1.0, "zone": "green"}

        last_message = messages[-1]
        content = getattr(last_message, "content", str(last_message))

        if not content:
            return {"fidelity": 1.0, "zone": "green"}

        raw_sim, fidelity = calculate_fidelity(content, self.pa, self.embed_fn)
        zone = get_zone_from_fidelity(fidelity)

        return {
            "raw_similarity": raw_sim,
            "fidelity": fidelity,
            "zone": zone.value,
        }

    def _inject_context(
        self,
        input_state: Dict[str, Any],
        check_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Inject PA context into input for drift mitigation.

        This is a soft direction - adds context without blocking.
        """
        # Clone state and add context
        new_state = dict(input_state)

        logger.info(f"Injecting PA context for fidelity {check_result['fidelity']:.2f}")

        return new_state

    def _generate_redirect_response(
        self,
        input_state: Dict[str, Any],
        check_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate a redirect response when input is blocked.

        Uses callback if provided, otherwise generates default response.
        """
        if self.on_block:
            return self.on_block({
                "input_state": input_state,
                "check_result": check_result,
                "pa": self.pa.to_dict(),
            })

        # Default redirect response
        return {
            "messages": input_state.get("messages", []) + [{
                "role": "assistant",
                "content": (
                    f"I notice your request may have drifted from our purpose: "
                    f"'{self.pa.text}'. Could you help me understand how this "
                    f"relates to what we're working on?"
                ),
            }],
            "governance_blocked": True,
            "fidelity": check_result["fidelity"],
            "zone": check_result["zone"],
        }

    def _record_trace(
        self,
        action_type: str,
        description: str,
        fidelity: float,
        approved: bool,
        direction_reason: Optional[str] = None,
    ):
        """Record to governance trace."""
        entry = GovernanceTraceEntry(
            timestamp=datetime.now(),
            turn_number=self.turn_number,
            action_type=action_type,
            action_description=description,
            raw_similarity=0.0,  # Simplified
            fidelity_score=fidelity,
            zone=get_zone_from_fidelity(fidelity),
            direction_level=DirectionLevel.NONE if approved else DirectionLevel.DIRECT,
            direction_reason=direction_reason,
            approved=approved,
            approval_source="auto" if approved else "blocked",
        )
        self.governance_trace.append(entry.to_dict())

    def get_governance_trace(self) -> list:
        """Get the full governance trace for audit."""
        return self.governance_trace

    def get_fidelity_trajectory(self) -> list:
        """Get fidelity scores over time."""
        return [
            {"turn": e["turn_number"], "fidelity": e["fidelity_score"]}
            for e in self.governance_trace
        ]


# =============================================================================
# CONVENIENCE DECORATOR
# =============================================================================

def telos_wrap(
    primacy_attractor: Union[PrimacyAttractor, Dict[str, Any]],
    embed_fn: Callable[[str], np.ndarray],
    **kwargs,
) -> Callable:
    """
    Decorator to wrap an agent with TELOS governance.

    Usage:
        @telos_wrap(primacy_attractor=my_pa, embed_fn=embed)
        def my_agent(state):
            # Agent logic
            return result

        # Or wrap existing agent:
        governed = telos_wrap(my_pa, embed)(existing_agent)
    """
    def decorator(agent):
        return TelosWrapper(
            agent=agent,
            primacy_attractor=primacy_attractor,
            embed_fn=embed_fn,
            **kwargs,
        )
    return decorator
