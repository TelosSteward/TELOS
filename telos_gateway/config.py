"""
TELOS Gateway Configuration
===========================

Configuration settings for the governance gateway.
"""

import os
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


@dataclass
class GatewayConfig:
    """Gateway configuration settings."""

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000

    # LLM Provider settings
    openai_api_key: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    openai_base_url: str = "https://api.openai.com/v1"

    anthropic_api_key: Optional[str] = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY"))
    mistral_api_key: Optional[str] = field(default_factory=lambda: os.getenv("MISTRAL_API_KEY"))

    # Embedding settings (for fidelity calculation)
    # Use local SentenceTransformer for speed, or Mistral API for quality
    embedding_provider: str = "sentence_transformer"  # or "mistral"
    embedding_model: str = "all-MiniLM-L6-v2"

    # Governance thresholds (calibrated for embedding space)
    # SentenceTransformer (all-MiniLM-L6-v2) range: ~0.15-0.55
    # Mistral embeddings range: ~0.40-0.75
    # These thresholds are for SentenceTransformer - adjust if using Mistral
    agentic_execute_threshold: float = 0.45   # High fidelity - forward to LLM
    agentic_clarify_threshold: float = 0.35   # Medium fidelity - ask clarification
    agentic_suggest_threshold: float = 0.25   # Low fidelity - suggest alternatives
    similarity_baseline: float = 0.15          # Hard block below this (off-topic noise)

    # Governance behavior
    block_on_low_fidelity: bool = True
    log_governance_traces: bool = True
    trace_storage_dir: Path = field(default_factory=lambda: Path("./telos_gateway_traces"))

    # Streaming behavior
    allow_streaming: bool = False  # MVP: disable streaming, buffer responses

    def __post_init__(self):
        """Ensure trace directory exists."""
        if self.log_governance_traces:
            self.trace_storage_dir.mkdir(parents=True, exist_ok=True)


# Global config instance
config = GatewayConfig()
