"""
Dual Primacy Attractor Architecture

Implements dual PA governance:
- User PA: Governs conversation purpose (WHAT to discuss)
- AI PA: Governs AI behavior/role (HOW to help)

Key Innovation: Lock-on derivation - AI PA computed from User PA
to ensure automatic alignment.

Status: Experimental (v1.2-dual-attractor)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import numpy as np
import logging
import asyncio

from telos_purpose.core.primacy_math import PrimacyAttractorMath, MathematicalState

logger = logging.getLogger(__name__)


# Intent to Role Mapping for PA Derivation
INTENT_TO_ROLE_MAP = {
    'learn': 'teach',
    'understand': 'explain',
    'solve': 'help solve',
    'create': 'help create',
    'decide': 'help decide',
    'explore': 'guide exploration',
    'analyze': 'help analyze',
    'fix': 'help fix',
    'debug': 'help debug',
    'optimize': 'help optimize',
    'research': 'help research',
    'plan': 'help plan'
}


@dataclass
class DualPrimacyAttractor:
    """
    Dual attractor for governing both user purpose and AI behavior.

    Attributes:
        user_pa: User's purpose attractor (WHAT conversation should accomplish)
        ai_pa: AI's role attractor (HOW AI should behave)
        correlation: Alignment score between user_pa and ai_pa
        governance_mode: 'single' or 'dual'
    """
    user_pa: Dict[str, Any]  # Purpose, scope, boundaries, thresholds
    ai_pa: Optional[Dict[str, Any]] = None  # AI role configuration
    correlation: float = 0.0  # Correlation between PAs
    governance_mode: str = 'dual'  # 'single' or 'dual' - Default to 'dual' (canonical implementation)

    def __post_init__(self):
        """Validate configuration."""
        if self.governance_mode == 'dual' and self.ai_pa is None:
            raise ValueError("Dual mode requires ai_pa to be set")

        if self.governance_mode == 'dual' and self.correlation < 0.2:
            logger.warning(
                f"Low PA correlation ({self.correlation:.2f}). "
                f"Consider fallback to single PA mode."
            )

    def is_dual_mode(self) -> bool:
        """Check if running in dual PA mode."""
        return self.governance_mode == 'dual'

    def get_user_threshold(self) -> float:
        """Get fidelity threshold for user PA."""
        return self.user_pa.get('fidelity_threshold', 0.65)

    def get_ai_threshold(self) -> float:
        """Get fidelity threshold for AI PA."""
        if not self.is_dual_mode():
            return 0.0
        return self.ai_pa.get('fidelity_threshold', 0.70)


async def detect_user_intent(
    user_pa: Dict[str, Any],
    client: Any
) -> str:
    """
    Detect primary user intent from PA purpose statements.

    Maps purpose to action verb for role derivation.

    Args:
        user_pa: User's primacy attractor configuration
        client: Any client for LLM calls

    Returns:
        Primary intent verb (e.g., 'learn', 'solve', 'create')
    """
    purpose_statements = user_pa.get('purpose', [])
    purpose_text = ' '.join(purpose_statements) if isinstance(purpose_statements, list) else str(purpose_statements)

    # Use LLM to extract intent
    prompt = f"""Analyze this conversation purpose and identify the primary user intent.

Purpose: {purpose_text}

What is the primary action the user wants to accomplish? Choose ONE from:
{', '.join(INTENT_TO_ROLE_MAP.keys())}

Respond with ONLY the single word intent verb, nothing else."""

    try:
        response = await asyncio.to_thread(
            client.messages.create,
            model="claude-3-haiku-20240307",
            max_tokens=10,
            messages=[{"role": "user", "content": prompt}]
        )

        intent = response.content[0].text.strip().lower()

        # Validate intent is in our map
        if intent in INTENT_TO_ROLE_MAP:
            return intent
        else:
            # Fallback to 'understand' if unknown
            logger.warning(f"Unknown intent '{intent}', defaulting to 'understand'")
            return 'understand'

    except Exception as e:
        logger.error(f"Error detecting intent: {e}")
        return 'understand'  # Safe default


async def derive_ai_pa_from_user_pa(
    user_pa: Dict[str, Any],
    client: Any,
    template: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Derive AI's role PA from user's purpose PA using lock-on derivation.

    Core innovation: AI PA is COMPUTED from user PA, not independent.
    This ensures automatic alignment.

    Args:
        user_pa: User's primacy attractor configuration
        client: Any client for LLM calls
        template: Optional role customization (e.g., professional vs friendly)

    Returns:
        AI PA configuration dict
    """
    # Detect user's primary intent
    intent = await detect_user_intent(user_pa, client)
    role_action = INTENT_TO_ROLE_MAP.get(intent, 'help')

    logger.info(f"Detected intent: '{intent}' → AI role: '{role_action}'")

    # Extract user purpose for context
    user_purpose = user_pa.get('purpose', [])
    if isinstance(user_purpose, list):
        user_purpose_text = ' '.join(user_purpose)
    else:
        user_purpose_text = str(user_purpose)

    # Generate AI role purpose statement
    ai_purpose = f"{role_action.capitalize()} the user as they work to: {user_purpose_text}"

    # Default boundaries (can be customized via template)
    default_boundaries = [
        "Stay focused on serving the user's purpose",
        "Maintain professional and helpful demeanor",
        "Provide clear, actionable responses",
        "Ask for clarification when needed"
    ]

    # Apply template customization if provided
    if template and 'boundaries' in template:
        boundaries = template['boundaries']
    else:
        boundaries = default_boundaries

    # Generate AI scope based on user scope
    user_scope = user_pa.get('scope', [])
    ai_scope = [f"Support user in: {item}" for item in user_scope] if isinstance(user_scope, list) else [f"Support user in: {user_scope}"]

    # Construct AI PA
    ai_pa = {
        'purpose': [ai_purpose],
        'scope': ai_scope,
        'boundaries': boundaries,
        'constraint_tolerance': user_pa.get('constraint_tolerance', 0.2),
        'privacy_level': user_pa.get('privacy_level', 0.8),
        'task_priority': user_pa.get('task_priority', 0.7),
        'fidelity_threshold': 0.70,  # Slightly higher than user PA
        'derived_from_intent': intent,
        'derived_role_action': role_action
    }

    return ai_pa


async def check_pa_correlation(
    user_pa: Dict[str, Any],
    ai_pa: Dict[str, Any],
    client: Any
) -> float:
    """
    Check alignment between user PA and AI PA.

    Uses embedding similarity to measure how well AI role serves user purpose.

    Args:
        user_pa: User's primacy attractor
        ai_pa: AI's role attractor
        client: Any client for embeddings

    Returns:
        Correlation score in [0, 1] where higher = better alignment
    """
    # Extract purpose statements
    user_purpose = user_pa.get('purpose', [])
    ai_purpose = ai_pa.get('purpose', [])

    # Convert to text
    user_text = ' '.join(user_purpose) if isinstance(user_purpose, list) else str(user_purpose)
    ai_text = ' '.join(ai_purpose) if isinstance(ai_purpose, list) else str(ai_purpose)

    # Get embeddings (placeholder - would use actual embedding service)
    # For now, use simple heuristic based on shared terms
    user_terms = set(user_text.lower().split())
    ai_terms = set(ai_text.lower().split())

    if not user_terms or not ai_terms:
        return 0.0

    # Jaccard similarity as proxy for correlation
    intersection = user_terms & ai_terms
    union = user_terms | ai_terms

    correlation = len(intersection) / len(union) if union else 0.0

    # Scale to be more permissive (since AI PA includes user purpose text)
    # Real implementation would use cosine similarity of embeddings
    correlation = min(correlation * 2.0, 1.0)

    logger.info(f"PA correlation: {correlation:.2f}")

    return correlation


async def create_dual_pa(
    user_pa: Dict[str, Any],
    client: Any,
    enable_dual_mode: bool = True,
    template: Optional[Dict[str, Any]] = None
) -> DualPrimacyAttractor:
    """
    Create a dual primacy attractor with user PA and derived AI PA.

    Main entry point for dual PA initialization.

    Args:
        user_pa: User's primacy attractor configuration
        client: Any client for LLM calls
        enable_dual_mode: If True, derive AI PA. If False, single PA mode.
        template: Optional AI role customization

    Returns:
        DualPrimacyAttractor instance
    """
    if not enable_dual_mode:
        # Single PA mode
        logger.info("Single PA mode enabled")
        return DualPrimacyAttractor(
            user_pa=user_pa,
            ai_pa=None,
            correlation=0.0,
            governance_mode='single'
        )

    # Dual PA mode - derive AI PA
    logger.info("Dual PA mode enabled - deriving AI PA from user PA")

    try:
        # Derive AI PA (async)
        ai_pa = await derive_ai_pa_from_user_pa(user_pa, client, template)

        # Check correlation
        correlation = await check_pa_correlation(user_pa, ai_pa, client)

        # Fallback to single PA if correlation too low
        if correlation < 0.2:
            logger.warning(
                f"PA correlation ({correlation:.2f}) below minimum (0.2). "
                f"Falling back to single PA mode."
            )
            return DualPrimacyAttractor(
                user_pa=user_pa,
                ai_pa=None,
                correlation=correlation,
                governance_mode='single'
            )

        # Create dual PA
        return DualPrimacyAttractor(
            user_pa=user_pa,
            ai_pa=ai_pa,
            correlation=correlation,
            governance_mode='dual'
        )

    except Exception as e:
        logger.error(f"Error creating dual PA: {e}. Falling back to single PA mode.")
        return DualPrimacyAttractor(
            user_pa=user_pa,
            ai_pa=None,
            correlation=0.0,
            governance_mode='single'
        )


@dataclass
class DualFidelityResult:
    """
    Result of dual PA fidelity check.

    Attributes:
        user_fidelity: Distance from user PA
        ai_fidelity: Distance from AI PA (0.0 in single PA mode)
        user_pass: Whether user PA check passed
        ai_pass: Whether AI PA check passed (always True in single PA mode)
        overall_pass: Whether both PAs passed
        dominant_failure: Which PA failed more ('user', 'ai', 'both', or None)
        governance_mode: Which mode was used
    """
    user_fidelity: float
    ai_fidelity: float
    user_pass: bool
    ai_pass: bool
    overall_pass: bool
    dominant_failure: Optional[str]
    governance_mode: str

    def __str__(self) -> str:
        """Human-readable summary."""
        if self.governance_mode == 'single':
            return f"Single PA: fidelity={self.user_fidelity:.2f}, pass={self.overall_pass}"
        else:
            return (
                f"Dual PA: user_fidelity={self.user_fidelity:.2f}, "
                f"ai_fidelity={self.ai_fidelity:.2f}, pass={self.overall_pass}, "
                f"failure={self.dominant_failure}"
            )


def check_dual_pa_fidelity(
    response_embedding: np.ndarray,
    dual_pa: DualPrimacyAttractor,
    embedding_provider: Any
) -> DualFidelityResult:
    """
    Check fidelity against both user PA and AI PA using proper attractor math.

    This uses the same mathematical formula as single PA:
    â = (τ·p + (1-τ)·s) / ||τ·p + (1-τ)·s||

    Args:
        response_embedding: Embedding of LLM response
        dual_pa: Dual PA configuration
        embedding_provider: Embedding provider for encoding purpose/scope

    Returns:
        DualFidelityResult with pass/fail for both PAs
    """
    # Build user PA attractor using proper mathematical formula
    user_purpose_text = ' '.join(dual_pa.user_pa['purpose']) if isinstance(dual_pa.user_pa['purpose'], list) else str(dual_pa.user_pa['purpose'])
    user_scope_text = ' '.join(dual_pa.user_pa['scope']) if isinstance(dual_pa.user_pa['scope'], list) else str(dual_pa.user_pa['scope'])

    user_p_vec = embedding_provider.encode(user_purpose_text)
    user_s_vec = embedding_provider.encode(user_scope_text)

    user_attractor = PrimacyAttractorMath(
        purpose_vector=user_p_vec,
        scope_vector=user_s_vec,
        privacy_level=dual_pa.user_pa.get('privacy_level', 0.8),
        constraint_tolerance=dual_pa.user_pa.get('constraint_tolerance', 0.2),
        task_priority=dual_pa.user_pa.get('task_priority', 0.7)
    )

    # Calculate user PA fidelity (distance to attractor center)
    user_distance = np.linalg.norm(response_embedding - user_attractor.attractor_center)
    user_in_basin = user_distance <= user_attractor.basin_radius
    user_fidelity = 1.0 if user_in_basin else (1.0 - (user_distance / (2 * user_attractor.basin_radius)))
    user_fidelity = float(max(0.0, min(1.0, user_fidelity)))  # Clamp to [0, 1]
    user_pass = user_fidelity >= dual_pa.get_user_threshold()

    # Single PA mode
    if not dual_pa.is_dual_mode():
        return DualFidelityResult(
            user_fidelity=user_fidelity,
            ai_fidelity=0.0,
            user_pass=user_pass,
            ai_pass=True,  # N/A in single mode
            overall_pass=user_pass,
            dominant_failure=None if user_pass else 'user',
            governance_mode='single'
        )

    # Dual PA mode - build AI PA attractor
    ai_purpose_text = ' '.join(dual_pa.ai_pa['purpose']) if isinstance(dual_pa.ai_pa['purpose'], list) else str(dual_pa.ai_pa['purpose'])
    ai_scope_text = ' '.join(dual_pa.ai_pa['scope']) if isinstance(dual_pa.ai_pa['scope'], list) else str(dual_pa.ai_pa['scope'])

    ai_p_vec = embedding_provider.encode(ai_purpose_text)
    ai_s_vec = embedding_provider.encode(ai_scope_text)

    ai_attractor = PrimacyAttractorMath(
        purpose_vector=ai_p_vec,
        scope_vector=ai_s_vec,
        privacy_level=dual_pa.ai_pa.get('privacy_level', 0.8),
        constraint_tolerance=dual_pa.ai_pa.get('constraint_tolerance', 0.2),
        task_priority=dual_pa.ai_pa.get('task_priority', 0.7)
    )

    # Calculate AI PA fidelity (distance to attractor center)
    ai_distance = np.linalg.norm(response_embedding - ai_attractor.attractor_center)
    ai_in_basin = ai_distance <= ai_attractor.basin_radius
    ai_fidelity = 1.0 if ai_in_basin else (1.0 - (ai_distance / (2 * ai_attractor.basin_radius)))
    ai_fidelity = float(max(0.0, min(1.0, ai_fidelity)))  # Clamp to [0, 1]
    ai_pass = ai_fidelity >= dual_pa.get_ai_threshold()

    # Determine overall pass and dominant failure
    overall_pass = user_pass and ai_pass

    if overall_pass:
        dominant_failure = None
    elif not user_pass and not ai_pass:
        dominant_failure = 'both'
    elif not user_pass:
        dominant_failure = 'user'
    else:
        dominant_failure = 'ai'

    return DualFidelityResult(
        user_fidelity=user_fidelity,
        ai_fidelity=ai_fidelity,
        user_pass=user_pass,
        ai_pass=ai_pass,
        overall_pass=overall_pass,
        dominant_failure=dominant_failure,
        governance_mode='dual'
    )
