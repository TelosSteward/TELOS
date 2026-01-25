"""
TELOS Configurator Engine Package
==================================

Core governance engine for the TELOS Corpus Configurator MVP.

Author: TELOS AI Labs Inc.
Contact: contact@telos-labs.ai
"""

from .governance_engine import (
    # Core Classes
    GovernanceEngine,
    PrimacyAttractor,
    ThresholdConfig,
    GovernanceResult,

    # PA Functions
    create_pa,
    embed_pa,
    save_pa,
    load_pa,

    # Fidelity Functions
    compute_fidelity,
    classify_tier,

    # RAG Functions
    retrieve_relevant_policies,
    process_query,

    # Utilities
    get_embedding,
    cosine_similarity,

    # Constants
    TIER_NAMES,
    TIER_ACTIONS,
    DEFAULT_TIER_1_THRESHOLD,
    DEFAULT_TIER_2_LOWER,
    DEFAULT_TIER_2_UPPER,
    DEFAULT_RAG_RELEVANCE,
)

__all__ = [
    # Core Classes
    "GovernanceEngine",
    "PrimacyAttractor",
    "ThresholdConfig",
    "GovernanceResult",

    # PA Functions
    "create_pa",
    "embed_pa",
    "save_pa",
    "load_pa",

    # Fidelity Functions
    "compute_fidelity",
    "classify_tier",

    # RAG Functions
    "retrieve_relevant_policies",
    "process_query",

    # Utilities
    "get_embedding",
    "cosine_similarity",

    # Constants
    "TIER_NAMES",
    "TIER_ACTIONS",
    "DEFAULT_TIER_1_THRESHOLD",
    "DEFAULT_TIER_2_LOWER",
    "DEFAULT_TIER_2_UPPER",
    "DEFAULT_RAG_RELEVANCE",
]
