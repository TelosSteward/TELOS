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
    # Use Mistral API for quality embeddings (required for accurate governance)
    # SentenceTransformer is NOT suitable for comparing long PAs to short queries
    embedding_provider: str = "mistral"  # Changed from sentence_transformer - see bug fix note
    embedding_model: str = "mistral-embed"  # 1024-dim embeddings

    # Governance thresholds (calibrated for NORMALIZED fidelity 0-1)
    # The fidelity normalization maps raw_similarity < baseline to fidelity < 0.30
    # These thresholds work for both SentenceTransformer and Mistral embeddings
    # because they operate on normalized fidelity, not raw similarity
    agentic_execute_threshold: float = 0.45   # High fidelity - forward to LLM (display: 90%+)
    agentic_clarify_threshold: float = 0.35   # Medium fidelity - ask clarification (display: 80-89%)
    agentic_suggest_threshold: float = 0.30   # Low fidelity - suggest alternatives (display: 70-79%)
    similarity_baseline: float = 0.15          # Raw similarity baseline for normalization

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
