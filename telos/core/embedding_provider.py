"""
Embedding providers for TELOS.

Provides three implementations:
1. DeterministicEmbeddingProvider - Fast, deterministic (for testing)
2. SentenceTransformerProvider - Real semantic embeddings (for production)
3. MistralEmbeddingProvider - Mistral API embeddings with rate limiting
"""

from __future__ import annotations
import hashlib
import numpy as np
from typing import Optional, List
import logging
import time
import os

logger = logging.getLogger(__name__)


class DeterministicEmbeddingProvider:
    """
    Deterministic embedding provider for testing.

    Uses hash-based embeddings that are:
    - Deterministic (same text → same embedding)
    - Fast (no model loading)
    - Suitable for unit tests and rapid iteration

    NOT suitable for production (no semantic meaning).
    """

    def __init__(self, dimension: int = 384):
        """
        Args:
            dimension: Embedding dimension
        """
        self.dimension = dimension
        self.expected_dim = dimension

    def _is_valid_embedding(self, embedding: np.ndarray) -> bool:
        """Validate embedding contains no NaN/Inf and has correct shape.

        Args:
            embedding: Vector to validate

        Returns:
            bool: True if valid, False otherwise
        """
        if embedding is None or embedding.size == 0:
            return False

        # Check for NaN or Inf values
        if not np.all(np.isfinite(embedding)):
            return False

        # Check dimension matches expected
        if embedding.shape[0] != self.expected_dim:
            return False

        return True

    def encode(self, text: str) -> np.ndarray:
        """
        Generate deterministic embedding from text hash with validation.

        Args:
            text: Input text

        Returns:
            Deterministic embedding vector

        Raises:
            ValueError: If generated embedding is invalid
        """
        # Create hash of text
        hash_obj = hashlib.sha256(text.encode('utf-8'))
        hash_bytes = hash_obj.digest()

        # Expand hash to desired dimension
        embedding = []
        for i in range(self.dimension):
            byte_index = i % len(hash_bytes)
            value = hash_bytes[byte_index] / 255.0  # Normalize to [0, 1]
            embedding.append(value)

        embedding = np.array(embedding, dtype=np.float32)

        # Validate embedding before returning
        if not self._is_valid_embedding(embedding):
            raise ValueError(
                f"Invalid embedding generated: shape={embedding.shape}, "
                f"has_nan={np.any(np.isnan(embedding))}, "
                f"has_inf={np.any(np.isinf(embedding))}"
            )

        return embedding

    def encode_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate batch embeddings with validation.

        Args:
            texts: List of input texts

        Returns:
            List of validated embedding vectors

        Raises:
            ValueError: If any embedding is invalid
        """
        embeddings = []
        for i, text in enumerate(texts):
            try:
                emb = self.encode(text)
                embeddings.append(emb)
            except ValueError as e:
                raise ValueError(f"Invalid embedding at index {i}: {e}")

        return embeddings


class SentenceTransformerProvider:
    """
    Production embedding provider using sentence-transformers.

    Provides semantic embeddings suitable for:
    - Real governance applications
    - Semantic similarity comparisons
    - Production deployments
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Args:
            model_name: HuggingFace model identifier
        """
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.expected_dim = self.dimension

    def _is_valid_embedding(self, embedding: np.ndarray) -> bool:
        """Validate embedding contains no NaN/Inf and has correct shape.

        Args:
            embedding: Vector to validate

        Returns:
            bool: True if valid, False otherwise
        """
        if embedding is None or embedding.size == 0:
            return False

        # Check for NaN or Inf values
        if not np.all(np.isfinite(embedding)):
            return False

        # Check dimension matches expected
        if embedding.shape[0] != self.expected_dim:
            return False

        return True

    def encode(self, text: str) -> np.ndarray:
        """
        Generate semantic embedding for text with comprehensive validation.

        Args:
            text: Input text

        Returns:
            Semantic embedding vector

        Raises:
            ValueError: If embedding is invalid (NaN/Inf/wrong dimension)
        """
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            embedding = embedding.astype(np.float32)

            # Validate embedding before returning
            if not self._is_valid_embedding(embedding):
                raise ValueError(
                    f"Invalid embedding received: shape={embedding.shape}, "
                    f"has_nan={np.any(np.isnan(embedding))}, "
                    f"has_inf={np.any(np.isinf(embedding))}"
                )

            return embedding

        except Exception as e:
            logger.error(f"Embedding API failed for text: '{text[:50]}...': {e}")
            raise  # Re-raise for upstream handling

    def encode_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate batch embeddings with validation.

        Args:
            texts: List of input texts

        Returns:
            List of validated embedding vectors

        Raises:
            ValueError: If any embedding is invalid
        """
        try:
            # Use batch encoding for efficiency
            embeddings = self.model.encode(texts, convert_to_numpy=True)

            # Convert to list of arrays with proper dtype
            embeddings = [emb.astype(np.float32) for emb in embeddings]

            # Validate all embeddings
            for i, emb in enumerate(embeddings):
                if not self._is_valid_embedding(emb):
                    raise ValueError(
                        f"Invalid embedding at index {i}: shape={emb.shape}, "
                        f"has_nan={np.any(np.isnan(emb))}, "
                        f"has_inf={np.any(np.isinf(emb))}"
                    )

            return embeddings

        except Exception as e:
            logger.error(f"Batch embedding API failed: {e}")
            raise  # Re-raise for upstream handling


class MistralEmbeddingProvider:
    """
    Mistral API embedding provider with rate limiting.

    Features:
    - 10-second delay between embedding calls
    - Retry-until-success with exponential backoff
    - Proper rate limit handling
    """

    def __init__(self, model_name: str = "mistral-embed", api_delay: float = 10.0):
        """
        Args:
            model_name: Mistral embedding model name
            api_delay: Delay in seconds between API calls (default: 10.0)
        """
        from mistralai import Mistral

        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY environment variable not set")

        self.client = Mistral(api_key=api_key)
        self.model_name = model_name
        self.api_delay = api_delay
        self.dimension = 1024  # Mistral embeddings are 1024-dimensional
        self.expected_dim = 1024
        self.last_call_time = 0

        logger.info(f"Initialized MistralEmbeddingProvider with {api_delay}s delay")

    def _is_valid_embedding(self, embedding: np.ndarray) -> bool:
        """Validate embedding contains no NaN/Inf and has correct shape."""
        if embedding is None or embedding.size == 0:
            return False

        if not np.all(np.isfinite(embedding)):
            return False

        if embedding.shape[0] != self.expected_dim:
            return False

        return True

    def _wait_for_rate_limit(self):
        """Enforce minimum delay between API calls."""
        current_time = time.time()
        time_since_last_call = current_time - self.last_call_time

        if time_since_last_call < self.api_delay:
            wait_time = self.api_delay - time_since_last_call
            logger.info(f"⏳ Waiting {wait_time:.1f}s before next API call...")
            time.sleep(wait_time)

        self.last_call_time = time.time()

    def encode(self, text: str) -> np.ndarray:
        """
        Generate Mistral embedding with retry-until-success logic.

        Args:
            text: Input text

        Returns:
            Embedding vector

        Raises:
            ValueError: If embedding is invalid after successful API call
        """
        max_retries = 50  # Retry until successful
        retry_delay = 10
        retry_count = 0

        while retry_count < max_retries:
            try:
                # Enforce rate limiting
                self._wait_for_rate_limit()

                logger.info(f"📡 Calling Mistral API for embedding (attempt {retry_count + 1})...")

                # Make API call
                response = self.client.embeddings.create(
                    model=self.model_name,
                    inputs=[text]
                )

                # Extract embedding
                embedding_data = response.data[0].embedding
                embedding = np.array(embedding_data, dtype=np.float32)

                # Validate
                if not self._is_valid_embedding(embedding):
                    raise ValueError(
                        f"Invalid embedding received: shape={embedding.shape}, "
                        f"has_nan={np.any(np.isnan(embedding))}, "
                        f"has_inf={np.any(np.isinf(embedding))}"
                    )

                logger.info(f"✅ Successfully got embedding (dim={embedding.shape[0]})")
                return embedding

            except Exception as e:
                retry_count += 1

                if "429" in str(e) or "rate" in str(e).lower():
                    logger.warning(f"⚠️  Rate limit hit (attempt {retry_count}/{max_retries})")
                    logger.info(f"    Waiting {retry_delay}s before retry...")
                    time.sleep(retry_delay)
                    # Exponential backoff, cap at 60s
                    retry_delay = min(retry_delay * 1.5, 60)
                elif retry_count < max_retries:
                    logger.warning(f"⚠️  Error getting embedding: {e}")
                    logger.info(f"    Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"❌ Failed after {max_retries} attempts")
                    raise

        raise RuntimeError(f"Failed to get embedding after {max_retries} retries")

    def encode_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate batch embeddings (calls encode sequentially with delays).

        Args:
            texts: List of input texts

        Returns:
            List of embedding vectors
        """
        embeddings = []
        for i, text in enumerate(texts):
            logger.info(f"Processing text {i+1}/{len(texts)}")
            emb = self.encode(text)
            embeddings.append(emb)

        return embeddings


def EmbeddingProvider(deterministic: bool = False, use_mistral: bool = False, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Factory function to create appropriate embedding provider.

    Args:
        deterministic: If True, use fast deterministic embeddings (testing).
        use_mistral: If True, use Mistral API embeddings with rate limiting.
        model_name: Model name for SentenceTransformer (ignored if deterministic=True or use_mistral=True)

    Returns:
        Embedding provider instance
    """
    if deterministic:
        return DeterministicEmbeddingProvider()
    elif use_mistral:
        return MistralEmbeddingProvider()
    else:
        return SentenceTransformerProvider(model_name=model_name)
