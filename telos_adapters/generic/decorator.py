"""
Generic TELOS Governance Decorator

@telos_governed decorator for any Python function.
Works without LangGraph or any framework dependency.

Usage:
    from telos_adapters.generic import telos_governed

    @telos_governed(
        purpose="Help users with financial analysis",
        threshold=0.85,
    )
    def analyze_portfolio(user_request: str) -> str:
        # Your function logic here
        return result
"""
import functools
import logging
from typing import Any, Callable, Optional

import numpy as np

# Import from telos_core (single source of truth)
from telos_core.constants import (
    FIDELITY_GREEN,
    SIMILARITY_BASELINE,
    AGENTIC_EXECUTE_THRESHOLD,
)

# Import governance types from telos_governance (hard import — no fallback shadow types)
from telos_governance.types import ActionDecision, DirectionLevel

logger = logging.getLogger(__name__)


def telos_governed(
    purpose: str,
    threshold: float = FIDELITY_GREEN,
    embed_fn: Optional[Callable[[str], np.ndarray]] = None,
    on_block: Optional[Callable[[str, float], Any]] = None,
    high_risk: bool = False,
):
    """
    Decorator that applies TELOS governance to any Python function.

    Args:
        purpose: The purpose statement (Primacy Attractor text)
        threshold: Minimum fidelity to proceed (default: FIDELITY_GREEN = 0.70)
        embed_fn: Function to generate embeddings. If None, uses telos_core default.
        on_block: Callback when action is blocked. Receives (input_text, fidelity).
        high_risk: If True, low fidelity triggers ESCALATE instead of INERT.

    Returns:
        Decorated function with governance checks.
    """
    def decorator(func: Callable) -> Callable:
        # Cache the purpose embedding
        _purpose_embedding = None

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal _purpose_embedding

            # Get the input text (first string argument)
            input_text = None
            for arg in args:
                if isinstance(arg, str):
                    input_text = arg
                    break
            if input_text is None:
                input_text = kwargs.get("user_request", kwargs.get("query", kwargs.get("input", "")))

            if not input_text:
                # No text input to govern, pass through
                return func(*args, **kwargs)

            # Get embedding function
            _embed = embed_fn
            if _embed is None:
                try:
                    from telos_core.embedding_provider import get_cached_minilm_provider
                    provider = get_cached_minilm_provider()
                    _embed = provider.encode
                except Exception:
                    logger.warning("No embedding provider available, passing through")
                    return func(*args, **kwargs)

            # Compute purpose embedding (cached)
            if _purpose_embedding is None:
                _purpose_embedding = _embed(purpose)

            # Compute input embedding
            input_embedding = _embed(input_text)

            # Calculate cosine similarity
            similarity = float(np.dot(input_embedding, _purpose_embedding) / (
                np.linalg.norm(input_embedding) * np.linalg.norm(_purpose_embedding) + 1e-10
            ))

            # Layer 1: Baseline check
            if similarity < SIMILARITY_BASELINE:
                logger.warning(
                    f"TELOS HARD_BLOCK: similarity {similarity:.3f} < baseline {SIMILARITY_BASELINE}"
                )
                if on_block:
                    return on_block(input_text, similarity)
                raise ValueError(
                    f"Action blocked by TELOS governance: fidelity {similarity:.3f} "
                    f"below baseline {SIMILARITY_BASELINE}"
                )

            # Classify decision using agentic thresholds from telos_core.constants
            if similarity >= AGENTIC_EXECUTE_THRESHOLD:
                decision = ActionDecision.EXECUTE
            elif similarity >= threshold:
                decision = ActionDecision.CLARIFY
            elif similarity >= 0.50:
                decision = ActionDecision.SUGGEST
            elif high_risk:
                decision = ActionDecision.ESCALATE
            else:
                decision = ActionDecision.INERT

            # Act on decision
            if decision == ActionDecision.EXECUTE:
                return func(*args, **kwargs)
            elif decision == ActionDecision.CLARIFY:
                logger.info(f"TELOS CLARIFY: fidelity {similarity:.3f}, proceeding with caution")
                return func(*args, **kwargs)
            elif decision == ActionDecision.SUGGEST:
                logger.warning(f"TELOS SUGGEST: fidelity {similarity:.3f}, low confidence")
                return func(*args, **kwargs)
            else:
                # INERT or ESCALATE
                logger.warning(
                    f"TELOS {decision.value.upper()}: fidelity {similarity:.3f}, blocking"
                )
                if on_block:
                    return on_block(input_text, similarity)
                raise ValueError(
                    f"Action blocked by TELOS governance: {decision.value} "
                    f"(fidelity {similarity:.3f})"
                )

        # Store governance metadata on the wrapper
        wrapper._telos_purpose = purpose
        wrapper._telos_threshold = threshold
        wrapper._telos_high_risk = high_risk

        return wrapper
    return decorator
