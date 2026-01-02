"""
Embedding Provider for TELOS Gateway
=====================================

Provides semantic embeddings for governance operations using Mistral API.

This is the Gateway's own embedding provider, independent of the Observatory
codebase, allowing the Gateway to be self-contained.
"""

import os
import logging
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)


class MistralEmbeddingProvider:
    """
    Mistral embedding provider using mistral-embed API.

    Provides high-quality 1024-dimensional embeddings for:
    - Primacy Attractor encoding
    - User request encoding
    - Tool description encoding
    - Semantic fidelity calculations
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Mistral embedding provider.

        Args:
            api_key: Mistral API key. If not provided, uses MISTRAL_API_KEY env var.
        """
        from mistralai import Mistral

        self.api_key = api_key or os.environ.get('MISTRAL_API_KEY')
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY not found in environment or parameters")

        self.client = Mistral(api_key=self.api_key)
        self.dimension = 1024  # mistral-embed produces 1024-dimensional vectors
        self.model_name = "mistral-embed"

        logger.info(f"MistralEmbeddingProvider initialized with {self.dimension}D embeddings")

    def encode(self, text: str) -> np.ndarray:
        """
        Generate semantic embedding for text.

        Args:
            text: Input text to embed

        Returns:
            1024-dimensional embedding vector
        """
        try:
            response = self.client.embeddings.create(
                model=self.model_name,
                inputs=[text]
            )
            embedding = response.data[0].embedding
            return np.array(embedding, dtype=np.float32)
        except Exception as e:
            logger.error(f"Mistral embedding API error: {e}")
            raise RuntimeError(f"Mistral embedding API error: {e}")

    def embed(self, text: str) -> np.ndarray:
        """
        Alias for encode() to match common interface patterns.

        Args:
            text: Input text to embed

        Returns:
            1024-dimensional embedding vector
        """
        return self.encode(text)

    def batch_encode(self, texts: list) -> list:
        """
        Generate embeddings for multiple texts in a single API call.

        More efficient than calling encode() multiple times.

        Args:
            texts: List of input texts

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        try:
            response = self.client.embeddings.create(
                model=self.model_name,
                inputs=texts
            )
            embeddings = [
                np.array(item.embedding, dtype=np.float32)
                for item in response.data
            ]
            return embeddings
        except Exception as e:
            logger.error(f"Mistral batch embedding API error: {e}")
            raise RuntimeError(f"Mistral batch embedding API error: {e}")


# =============================================================================
# Factory Function
# =============================================================================

_cached_provider: Optional[MistralEmbeddingProvider] = None


def get_embedding_provider(api_key: Optional[str] = None) -> MistralEmbeddingProvider:
    """
    Get a cached embedding provider instance.

    This function caches the provider to avoid repeated initialization
    and API client creation.

    Args:
        api_key: Optional Mistral API key. Uses env var if not provided.

    Returns:
        MistralEmbeddingProvider instance
    """
    global _cached_provider

    if _cached_provider is None:
        _cached_provider = MistralEmbeddingProvider(api_key=api_key)
        logger.info("Created cached embedding provider")

    return _cached_provider


def clear_provider_cache():
    """Clear the cached provider (useful for testing)."""
    global _cached_provider
    _cached_provider = None
