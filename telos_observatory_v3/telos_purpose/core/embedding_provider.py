"""
Embedding providers for TELOS.

Provides two implementations:
1. DeterministicEmbeddingProvider - Fast, deterministic (for testing)
2. SentenceTransformerProvider - Real semantic embeddings (for production)
"""

from __future__ import annotations
import hashlib
import numpy as np
from typing import Optional


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

    def encode(self, text: str) -> np.ndarray:
        """
        Generate deterministic embedding from text hash.

        Args:
            text: Input text

        Returns:
            Deterministic embedding vector
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

        return np.array(embedding, dtype=np.float32)


class MistralEmbeddingProvider:
    """
    Mistral embedding provider using mistral-embed API.

    Provides high-quality 1024-dimensional embeddings via Mistral's API.
    Requires MISTRAL_API_KEY environment variable.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Args:
            api_key: Mistral API key (optional, will use env var if not provided)
        """
        import os
        from mistralai import Mistral

        self.api_key = api_key or os.environ.get('MISTRAL_API_KEY')
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY not found in environment or parameters")

        self.client = Mistral(api_key=self.api_key)
        self.dimension = 1024  # mistral-embed produces 1024-dimensional vectors
        self.model_name = "mistral-embed"

    def encode(self, text: str) -> np.ndarray:
        """
        Generate semantic embedding using Mistral API.

        Args:
            text: Input text

        Returns:
            Semantic embedding vector (1024 dimensions)
        """
        try:
            response = self.client.embeddings.create(
                model=self.model_name,
                inputs=[text]
            )
            # Extract embedding from response
            embedding = response.data[0].embedding
            return np.array(embedding, dtype=np.float32)
        except Exception as e:
            raise RuntimeError(f"Mistral embedding API error: {e}")

    def batch_encode(self, texts: list) -> list:
        """
        Generate semantic embeddings for multiple texts in a single API call.

        This is much faster than calling encode() multiple times because
        it batches all texts into one API request.

        Args:
            texts: List of input texts

        Returns:
            List of semantic embedding vectors (1024 dimensions each)
        """
        if not texts:
            return []

        try:
            response = self.client.embeddings.create(
                model=self.model_name,
                inputs=texts
            )
            # Extract embeddings from response, preserving order
            embeddings = [np.array(item.embedding, dtype=np.float32) for item in response.data]
            return embeddings
        except Exception as e:
            raise RuntimeError(f"Mistral batch embedding API error: {e}")


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
        import os
        import logging

        # Force offline mode to use cached model and avoid network hangs
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'

        from sentence_transformers import SentenceTransformer
        import torch

        self.model_name = model_name

        # Fix for PyTorch 2.x "Cannot copy out of meta tensor" error
        # Do NOT specify device during init - let library handle it
        # Specifying device='cpu' causes meta tensor copy failures in PyTorch 2.x

        try:
            # Load without device specification - avoids meta tensor issues
            self.model = SentenceTransformer(model_name)
            logging.info(f"SentenceTransformer loaded successfully: {model_name}")
        except Exception as e:
            logging.warning(f"Primary load failed: {e}, trying alternative approach")
            try:
                # Alternative: explicitly disable lazy loading features
                # This forces immediate tensor materialization
                import torch
                with torch.inference_mode():
                    self.model = SentenceTransformer(model_name, device='cpu')
            except Exception as e2:
                logging.error(f"All SentenceTransformer loading approaches failed: {e2}")
                raise RuntimeError(f"Failed to load SentenceTransformer model: {e2}")

        self.dimension = self.model.get_sentence_embedding_dimension()

    def encode(self, text: str) -> np.ndarray:
        """
        Generate semantic embedding for text.

        Args:
            text: Input text

        Returns:
            Semantic embedding vector
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.astype(np.float32)


def EmbeddingProvider(deterministic: bool = False, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Factory function to create appropriate embedding provider.

    Args:
        deterministic: If True, use fast deterministic embeddings (testing).
                      If False, use real semantic embeddings (production).
        model_name: Model name for SentenceTransformer (ignored if deterministic=True)

    Returns:
        Embedding provider instance
    """
    if deterministic:
        return DeterministicEmbeddingProvider()
    else:
        return SentenceTransformerProvider(model_name=model_name)


# =============================================================================
# FIDELITY RESCALING FOR SENTENCETRANSFORMER
# =============================================================================
# SentenceTransformer produces raw cosine similarity scores in a narrow range:
#   - Off-topic content: typically -0.1 to 0.15
#   - On-topic content: typically 0.25 to 0.50
#
# TELOS fidelity requires:
#   - Off-topic: should be < 0.50 (RED zone)
#   - On-topic: should be >= 0.70 (GREEN zone)
#
# This rescaling maps SentenceTransformer scores to TELOS-compatible range.
# =============================================================================

def rescale_sentence_transformer_fidelity(raw_score: float) -> float:
    """
    Rescale SentenceTransformer cosine similarity to TELOS fidelity range.

    This function maps the narrow discriminative range of SentenceTransformer
    (where on-topic is ~0.25-0.50) to the TELOS fidelity range (where on-topic
    must be >= 0.70 for GREEN zone).

    Formula: fidelity = 0.25 + raw_score * 1.8, clamped to [0.0, 1.0]

    This maps:
        raw 0.00 → 0.25 (RED - clearly off-topic)
        raw 0.15 → 0.52 (ORANGE - borderline)
        raw 0.25 → 0.70 (GREEN - on-topic threshold)
        raw 0.35 → 0.88 (GREEN - strongly aligned)
        raw 0.50 → 1.00 (GREEN - perfect alignment)

    For negative scores (very off-topic):
        raw -0.10 → 0.07 (very RED)

    Args:
        raw_score: Raw cosine similarity from SentenceTransformer (-1.0 to 1.0)

    Returns:
        Rescaled fidelity score for TELOS (0.0 to 1.0)
    """
    # Apply linear rescaling
    # 0.25 baseline ensures raw 0.0 maps to RED zone
    # 1.8 scale factor ensures raw 0.25 maps to GREEN zone (0.70)
    rescaled = 0.25 + raw_score * 1.8

    # Clamp to valid fidelity range
    return max(0.0, min(1.0, rescaled))


# =============================================================================
# CACHED EMBEDDING PROVIDER FACTORIES
# =============================================================================
# These functions use @st.cache_resource to cache expensive model loading.
# This prevents reloading neural network weights on every Streamlit rerun,
# which was causing 60-90 second delays when clicking Alignment Lens button.
#
# Usage: Call get_cached_minilm_provider() instead of creating new instances.
# =============================================================================

def get_cached_minilm_provider():
    """
    Get a cached MiniLM embedding provider.

    Uses Streamlit's @st.cache_resource to ensure the model is only loaded
    once per server session, not on every rerun.

    Returns:
        SentenceTransformerProvider instance with all-MiniLM-L6-v2 model
    """
    try:
        import streamlit as st

        @st.cache_resource(show_spinner=False)
        def _load_minilm():
            return SentenceTransformerProvider(model_name="sentence-transformers/all-MiniLM-L6-v2")

        return _load_minilm()
    except ImportError:
        # Fallback for non-Streamlit contexts (testing, CLI)
        return SentenceTransformerProvider(model_name="sentence-transformers/all-MiniLM-L6-v2")


def get_cached_mpnet_provider():
    """
    Get a cached MPNet embedding provider.

    Uses Streamlit's @st.cache_resource to ensure the model is only loaded
    once per server session, not on every rerun.

    MPNet (all-mpnet-base-v2) produces 768-dimensional embeddings and is
    used for AI fidelity calculations. It's larger than MiniLM but more
    accurate for semantic similarity.

    Returns:
        SentenceTransformerProvider instance with all-mpnet-base-v2 model
    """
    try:
        import streamlit as st

        @st.cache_resource(show_spinner=False)
        def _load_mpnet():
            return SentenceTransformerProvider(model_name="sentence-transformers/all-mpnet-base-v2")

        return _load_mpnet()
    except ImportError:
        # Fallback for non-Streamlit contexts (testing, CLI)
        return SentenceTransformerProvider(model_name="sentence-transformers/all-mpnet-base-v2")


def get_cached_universal_lane_centroid():
    """
    Get cached centroid of universal lane expansion patterns.

    PERFORMANCE: These embeddings are computed from fixed strings in pa_templates.py.
    Caching the centroid avoids ~15 embedding calls every time PA is established.

    Returns:
        np.ndarray: Normalized centroid of universal lane expansion embeddings
    """
    try:
        import streamlit as st

        @st.cache_resource(show_spinner=False)
        def _compute_universal_centroid():
            from config.pa_templates import UNIVERSAL_LANE_EXPANSION
            provider = get_cached_minilm_provider()

            # Embed all universal lane expansion patterns
            embeddings = [np.array(provider.encode(text)) for text in UNIVERSAL_LANE_EXPANSION]
            centroid = np.mean(embeddings, axis=0)

            # Normalize the centroid
            centroid = centroid / np.linalg.norm(centroid)
            return centroid

        return _compute_universal_centroid()
    except ImportError:
        # Fallback for non-Streamlit contexts
        from config.pa_templates import UNIVERSAL_LANE_EXPANSION
        provider = SentenceTransformerProvider(model_name="sentence-transformers/all-MiniLM-L6-v2")
        embeddings = [np.array(provider.encode(text)) for text in UNIVERSAL_LANE_EXPANSION]
        centroid = np.mean(embeddings, axis=0)
        return centroid / np.linalg.norm(centroid)


def get_cached_template_domain_centroid(template_id: str):
    """
    Get cached domain centroid for a specific template's example queries.

    PERFORMANCE: Each template has fixed example queries that get embedded
    every time the template is selected. Caching by template_id avoids
    ~8-15 embedding calls per template selection.

    Args:
        template_id: The template ID from PA_TEMPLATES (e.g., 'creative_writing')

    Returns:
        np.ndarray: Normalized centroid of example query embeddings, or None if no queries
    """
    try:
        import streamlit as st

        @st.cache_resource(show_spinner=False)
        def _compute_template_centroid(tid: str):
            from config.pa_templates import get_template_by_id
            provider = get_cached_minilm_provider()

            template = get_template_by_id(tid)
            if not template:
                return None

            example_queries = template.get('example_queries', [])
            if not example_queries:
                return None

            # Embed all example queries for this template
            embeddings = [np.array(provider.encode(text)) for text in example_queries]
            centroid = np.mean(embeddings, axis=0)

            # Normalize the centroid
            return centroid / np.linalg.norm(centroid)

        return _compute_template_centroid(template_id)
    except ImportError:
        # Fallback for non-Streamlit contexts
        from config.pa_templates import get_template_by_id
        provider = SentenceTransformerProvider(model_name="sentence-transformers/all-MiniLM-L6-v2")

        template = get_template_by_id(template_id)
        if not template:
            return None

        example_queries = template.get('example_queries', [])
        if not example_queries:
            return None

        embeddings = [np.array(provider.encode(text)) for text in example_queries]
        centroid = np.mean(embeddings, axis=0)
        return centroid / np.linalg.norm(centroid)
