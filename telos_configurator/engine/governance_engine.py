#!/usr/bin/env python3
"""
TELOS Governance Engine
=======================

A comprehensive governance module implementing the three-tier TELOS framework:
- Tier 1: Primacy Attractor (PA) Mathematical Block
- Tier 2: RAG Policy Retrieval
- Tier 3: Expert Escalation

This module provides the core governance logic for the TELOS Corpus Configurator MVP.

Author: TELOS AI Labs Inc.
Contact: contact@telos-labs.ai
Date: 2026-01-23
"""

import json
import os
import threading
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
import numpy as np
import requests


# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

# Ollama Configuration
OLLAMA_URL = "http://localhost:11434/api/embeddings"
EMBEDDING_MODEL = "nomic-embed-text"
EMBEDDING_DIM = 768
EMBEDDING_TIMEOUT = 30  # seconds

# Default Threshold Configuration
DEFAULT_TIER_1_THRESHOLD = 0.65  # PA mathematical block (fidelity >= 0.65)
DEFAULT_TIER_2_LOWER = 0.35      # RAG zone lower bound
DEFAULT_TIER_2_UPPER = 0.65      # RAG zone upper bound (same as tier_1)
DEFAULT_RAG_RELEVANCE = 0.50     # Minimum relevance for RAG retrieval

# Tier Names
TIER_NAMES = {
    1: "PA_Block",
    2: "RAG_Policy",
    3: "Expert_Escalation"
}

# Tier Actions
TIER_ACTIONS = {
    1: "BLOCKED",
    2: "POLICY_RETRIEVED",
    3: "ESCALATED"
}


# ============================================================================
# EMBEDDING UTILITIES
# ============================================================================

def get_embedding(text: str) -> Optional[np.ndarray]:
    """
    Generate embedding using Ollama's nomic-embed-text model.

    Args:
        text: Text to embed

    Returns:
        numpy array of embedding or None if failed
    """
    if not text or not text.strip():
        return None

    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": EMBEDDING_MODEL, "prompt": text},
            timeout=EMBEDDING_TIMEOUT
        )

        if response.status_code == 200:
            embedding = response.json().get("embedding", [])
            if embedding:
                return np.array(embedding, dtype=np.float32)
        else:
            print(f"Embedding API error: {response.status_code} - {response.text}")

    except requests.exceptions.Timeout:
        print(f"Embedding request timeout after {EMBEDDING_TIMEOUT}s")
    except Exception as e:
        print(f"Embedding error: {e}")

    return None


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        a: First vector
        b: Second vector

    Returns:
        Cosine similarity in range [-1, 1], or 0.0 if vectors are invalid
    """
    if a is None or b is None:
        return 0.0

    if a.size == 0 or b.size == 0:
        return 0.0

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class PrimacyAttractor:
    """
    Primacy Attractor (PA) configuration.

    The PA defines the purpose, scope, exclusions, and prohibitions that
    govern the conversational space. It is embedded as a single vector
    for fidelity computation.
    """
    name: str
    purpose_statement: str
    scope: List[str]
    exclusions: List[str]
    prohibitions: List[str]
    embedding: Optional[np.ndarray] = None
    created_at: Optional[str] = None

    def __post_init__(self):
        """Set creation timestamp if not provided."""
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()

    def get_combined_text(self) -> str:
        """
        Combine all PA fields into a single text for embedding.

        Returns:
            Concatenated text of all PA components
        """
        parts = [
            f"Purpose: {self.purpose_statement}",
            f"Scope: {', '.join(self.scope)}",
            f"Exclusions: {', '.join(self.exclusions)}",
            f"Prohibitions: {', '.join(self.prohibitions)}"
        ]
        return " ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert PA to dictionary (for JSON serialization).

        Note: Embedding is excluded as it's not JSON-serializable.
        """
        return {
            "name": self.name,
            "purpose_statement": self.purpose_statement,
            "scope": self.scope,
            "exclusions": self.exclusions,
            "prohibitions": self.prohibitions,
            "created_at": self.created_at
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PrimacyAttractor':
        """Create PA from dictionary."""
        return cls(
            name=data.get("name", ""),
            purpose_statement=data.get("purpose_statement", ""),
            scope=data.get("scope", []),
            exclusions=data.get("exclusions", []),
            prohibitions=data.get("prohibitions", []),
            created_at=data.get("created_at")
        )


@dataclass
class ThresholdConfig:
    """
    Threshold configuration for three-tier governance.

    Tier classification:
    - Tier 1: fidelity >= tier_1_threshold (PA blocks)
    - Tier 2: tier_2_lower <= fidelity < tier_2_upper (RAG retrieves)
    - Tier 3: fidelity < tier_2_lower (Expert escalates)
    """
    tier_1_threshold: float = DEFAULT_TIER_1_THRESHOLD
    tier_2_lower: float = DEFAULT_TIER_2_LOWER
    tier_2_upper: float = DEFAULT_TIER_2_UPPER
    rag_relevance: float = DEFAULT_RAG_RELEVANCE

    def validate(self) -> Tuple[bool, Optional[str]]:
        """
        Validate threshold configuration.

        Returns:
            (is_valid, error_message)
        """
        if not (0.0 <= self.tier_1_threshold <= 1.0):
            return False, "tier_1_threshold must be in [0, 1]"
        if not (0.0 <= self.tier_2_lower <= 1.0):
            return False, "tier_2_lower must be in [0, 1]"
        if not (0.0 <= self.tier_2_upper <= 1.0):
            return False, "tier_2_upper must be in [0, 1]"
        if self.tier_2_lower > self.tier_2_upper:
            return False, "tier_2_lower must be <= tier_2_upper"
        if self.tier_2_upper != self.tier_1_threshold:
            return False, "tier_2_upper should equal tier_1_threshold"
        if not (0.0 <= self.rag_relevance <= 1.0):
            return False, "rag_relevance must be in [0, 1]"

        return True, None

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'ThresholdConfig':
        """Create from dictionary."""
        return cls(
            tier_1_threshold=data.get("tier_1_threshold", DEFAULT_TIER_1_THRESHOLD),
            tier_2_lower=data.get("tier_2_lower", DEFAULT_TIER_2_LOWER),
            tier_2_upper=data.get("tier_2_upper", DEFAULT_TIER_2_UPPER),
            rag_relevance=data.get("rag_relevance", DEFAULT_RAG_RELEVANCE)
        )


@dataclass
class GovernanceResult:
    """
    Result of a governance decision for a single query.

    Contains tier classification, fidelity score, action taken,
    and any retrieved policies (for Tier 2).
    """
    query: str
    fidelity: float
    tier: int
    tier_name: str
    action: str
    retrieved_policies: List[Dict[str, Any]] = field(default_factory=list)
    blocking_reason: Optional[str] = None
    timestamp: Optional[str] = None

    def __post_init__(self):
        """Set timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "query": self.query,
            "fidelity": round(self.fidelity, 4),
            "tier": self.tier,
            "tier_name": self.tier_name,
            "action": self.action,
            "retrieved_policies": self.retrieved_policies,
            "blocking_reason": self.blocking_reason,
            "timestamp": self.timestamp
        }


# ============================================================================
# PRIMACY ATTRACTOR FUNCTIONS
# ============================================================================

def create_pa(
    name: str,
    purpose: str,
    scope: List[str],
    exclusions: List[str],
    prohibitions: List[str]
) -> PrimacyAttractor:
    """
    Create a new Primacy Attractor configuration.

    Args:
        name: PA name/identifier
        purpose: Purpose statement
        scope: List of in-scope topics
        exclusions: List of out-of-scope topics
        prohibitions: List of prohibited actions/topics

    Returns:
        PrimacyAttractor instance (without embedding)
    """
    return PrimacyAttractor(
        name=name,
        purpose_statement=purpose,
        scope=scope,
        exclusions=exclusions,
        prohibitions=prohibitions
    )


def embed_pa(pa: PrimacyAttractor) -> bool:
    """
    Generate embedding for a Primacy Attractor.

    Args:
        pa: PrimacyAttractor to embed

    Returns:
        True if successful, False otherwise
    """
    combined_text = pa.get_combined_text()
    embedding = get_embedding(combined_text)

    if embedding is not None:
        pa.embedding = embedding
        return True

    return False


def save_pa(pa: PrimacyAttractor, filepath: str) -> bool:
    """
    Save Primacy Attractor to JSON file.

    Args:
        pa: PrimacyAttractor to save
        filepath: Path to JSON file

    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(pa.to_dict(), f, indent=2)

        return True

    except Exception as e:
        print(f"Error saving PA to {filepath}: {e}")
        return False


def load_pa(filepath: str) -> Optional[PrimacyAttractor]:
    """
    Load Primacy Attractor from JSON file.

    Args:
        filepath: Path to JSON file

    Returns:
        PrimacyAttractor instance or None if failed
    """
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)

        pa = PrimacyAttractor.from_dict(data)

        # Re-embed the PA
        if not embed_pa(pa):
            print(f"Warning: Failed to embed PA loaded from {filepath}")

        return pa

    except Exception as e:
        print(f"Error loading PA from {filepath}: {e}")
        return None


# ============================================================================
# FIDELITY COMPUTATION
# ============================================================================

def compute_fidelity(query_embedding: np.ndarray, pa_embedding: np.ndarray) -> float:
    """
    Compute fidelity score between query and PA.

    Fidelity is the cosine similarity between embeddings, normalized to [0, 1].

    Args:
        query_embedding: Query embedding vector
        pa_embedding: PA embedding vector

    Returns:
        Fidelity score in [0, 1] (or -1 if invalid)
    """
    if query_embedding is None or pa_embedding is None:
        return -1.0

    # Compute cosine similarity (range: [-1, 1])
    similarity = cosine_similarity(query_embedding, pa_embedding)

    # Normalize to [0, 1] range
    # similarity = (similarity + 1) / 2  # Optional: normalize to [0, 1]

    # TELOS uses raw cosine similarity which is already suitable
    return max(0.0, similarity)  # Clamp negative values to 0


def classify_tier(fidelity: float, thresholds: ThresholdConfig) -> int:
    """
    Classify query into Tier 1, 2, or 3 based on fidelity.

    Args:
        fidelity: Fidelity score
        thresholds: Threshold configuration

    Returns:
        Tier number (1, 2, or 3)
    """
    if fidelity >= thresholds.tier_1_threshold:
        return 1  # PA Block
    elif fidelity >= thresholds.tier_2_lower:
        return 2  # RAG Policy
    else:
        return 3  # Expert Escalation


# ============================================================================
# RAG RETRIEVAL
# ============================================================================

def retrieve_relevant_policies(
    query_embedding: np.ndarray,
    corpus_embeddings: List[np.ndarray],
    corpus_docs: List[Dict[str, Any]],
    top_k: int = 3,
    relevance_threshold: float = DEFAULT_RAG_RELEVANCE
) -> List[Dict[str, Any]]:
    """
    Retrieve top-k most relevant policies from corpus.

    Args:
        query_embedding: Query embedding vector
        corpus_embeddings: List of corpus document embeddings
        corpus_docs: List of corpus document metadata
        top_k: Number of documents to retrieve
        relevance_threshold: Minimum similarity to include

    Returns:
        List of retrieved policy documents with similarity scores
    """
    if query_embedding is None or not corpus_embeddings:
        return []

    similarities = []

    for i, doc_embedding in enumerate(corpus_embeddings):
        if i >= len(corpus_docs):
            break

        sim = cosine_similarity(query_embedding, doc_embedding)

        if sim >= relevance_threshold:
            similarities.append((i, sim))

    # Sort by similarity descending
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Return top-k with metadata
    results = []
    for i, sim in similarities[:top_k]:
        doc = corpus_docs[i].copy()
        doc["similarity"] = round(sim, 4)
        results.append(doc)

    return results


# ============================================================================
# THREE-TIER GOVERNANCE PROCESSING
# ============================================================================

def process_query(
    query: str,
    pa: PrimacyAttractor,
    corpus_embeddings: List[np.ndarray],
    corpus_docs: List[Dict[str, Any]],
    thresholds: ThresholdConfig,
    top_k: int = 3
) -> GovernanceResult:
    """
    Process a query through the three-tier governance pipeline.

    Pipeline:
    1. Embed query
    2. Compute fidelity against PA
    3. Classify tier
    4. If Tier 2: retrieve relevant policies
    5. Return GovernanceResult

    Args:
        query: User query text
        pa: Primacy Attractor
        corpus_embeddings: List of corpus embeddings
        corpus_docs: List of corpus document metadata
        thresholds: Threshold configuration
        top_k: Number of policies to retrieve for Tier 2

    Returns:
        GovernanceResult with decision
    """
    # Step 1: Embed query
    query_embedding = get_embedding(query)

    if query_embedding is None:
        return GovernanceResult(
            query=query,
            fidelity=-1.0,
            tier=3,
            tier_name=TIER_NAMES[3],
            action="ESCALATED",
            blocking_reason="Failed to embed query"
        )

    # Step 2: Compute fidelity
    if pa.embedding is None:
        return GovernanceResult(
            query=query,
            fidelity=-1.0,
            tier=3,
            tier_name=TIER_NAMES[3],
            action="ESCALATED",
            blocking_reason="PA not embedded"
        )

    fidelity = compute_fidelity(query_embedding, pa.embedding)

    # Step 3: Classify tier
    tier = classify_tier(fidelity, thresholds)
    tier_name = TIER_NAMES[tier]
    action = TIER_ACTIONS[tier]

    # Step 4: Tier-specific processing
    retrieved_policies = []
    blocking_reason = None

    if tier == 1:
        # Tier 1: PA Mathematical Block
        blocking_reason = f"Query fidelity ({fidelity:.4f}) >= tier 1 threshold ({thresholds.tier_1_threshold})"

    elif tier == 2:
        # Tier 2: RAG Policy Retrieval
        retrieved_policies = retrieve_relevant_policies(
            query_embedding=query_embedding,
            corpus_embeddings=corpus_embeddings,
            corpus_docs=corpus_docs,
            top_k=top_k,
            relevance_threshold=thresholds.rag_relevance
        )

        if not retrieved_policies:
            # No relevant policies found - escalate to Tier 3
            tier = 3
            tier_name = TIER_NAMES[3]
            action = "ESCALATED"
            blocking_reason = "No relevant policies found in corpus"

    elif tier == 3:
        # Tier 3: Expert Escalation
        blocking_reason = f"Query fidelity ({fidelity:.4f}) < tier 2 threshold ({thresholds.tier_2_lower})"

    # Step 5: Return result
    return GovernanceResult(
        query=query,
        fidelity=fidelity,
        tier=tier,
        tier_name=tier_name,
        action=action,
        retrieved_policies=retrieved_policies,
        blocking_reason=blocking_reason
    )


# ============================================================================
# GOVERNANCE ENGINE CLASS
# ============================================================================

class GovernanceEngine:
    """
    Main governance engine for three-tier TELOS framework.

    Manages PA configuration, corpus, thresholds, and query processing.
    Thread-safe for use in Streamlit applications.
    """

    def __init__(self):
        """Initialize governance engine."""
        self.pa: Optional[PrimacyAttractor] = None
        self.thresholds: ThresholdConfig = ThresholdConfig()
        self.corpus_embeddings: List[np.ndarray] = []
        self.corpus_docs: List[Dict[str, Any]] = []
        self.query_log: List[GovernanceResult] = []

        # Thread lock for thread-safe operations
        self._lock = threading.Lock()

    def configure(
        self,
        pa: PrimacyAttractor,
        thresholds: ThresholdConfig,
        corpus_docs: List[Dict[str, Any]],
        corpus_embeddings: List[np.ndarray]
    ) -> Tuple[bool, Optional[str]]:
        """
        Configure the governance engine.

        Args:
            pa: Primacy Attractor (must be embedded)
            thresholds: Threshold configuration
            corpus_docs: List of corpus document metadata
            corpus_embeddings: List of corpus embeddings

        Returns:
            (success, error_message)
        """
        with self._lock:
            # Validate PA
            if pa.embedding is None:
                return False, "PA must be embedded before configuration"

            # Validate thresholds
            is_valid, error = thresholds.validate()
            if not is_valid:
                return False, f"Invalid thresholds: {error}"

            # Validate corpus
            if len(corpus_docs) != len(corpus_embeddings):
                return False, "Corpus docs and embeddings must have same length"

            # Set configuration
            self.pa = pa
            self.thresholds = thresholds
            self.corpus_docs = corpus_docs
            self.corpus_embeddings = corpus_embeddings
            self.query_log = []  # Reset log on reconfiguration

            return True, None

    def is_active(self) -> bool:
        """
        Check if engine is configured and ready.

        Returns:
            True if configured, False otherwise
        """
        with self._lock:
            return (
                self.pa is not None and
                self.pa.embedding is not None and
                len(self.corpus_docs) == len(self.corpus_embeddings)
            )

    def process(self, query: str, top_k: int = 3) -> GovernanceResult:
        """
        Process a query through the governance pipeline.

        Args:
            query: User query text
            top_k: Number of policies to retrieve for Tier 2

        Returns:
            GovernanceResult
        """
        if not self.is_active():
            return GovernanceResult(
                query=query,
                fidelity=-1.0,
                tier=3,
                tier_name=TIER_NAMES[3],
                action="ESCALATED",
                blocking_reason="Governance engine not configured"
            )

        # Process query
        result = process_query(
            query=query,
            pa=self.pa,
            corpus_embeddings=self.corpus_embeddings,
            corpus_docs=self.corpus_docs,
            thresholds=self.thresholds,
            top_k=top_k
        )

        # Log result
        with self._lock:
            self.query_log.append(result)

        return result

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get governance statistics from query log.

        Returns:
            Dictionary with tier distribution, avg fidelity, etc.
        """
        with self._lock:
            if not self.query_log:
                return {
                    "total_queries": 0,
                    "avg_fidelity": 0.0,
                    "tier_distribution": {1: 0, 2: 0, 3: 0},
                    "tier_percentages": {1: "0.0%", 2: "0.0%", 3: "0.0%"}
                }

            total = len(self.query_log)
            valid_fidelities = [r.fidelity for r in self.query_log if r.fidelity >= 0]

            tier_counts = {1: 0, 2: 0, 3: 0}
            for result in self.query_log:
                tier_counts[result.tier] += 1

            return {
                "total_queries": total,
                "avg_fidelity": np.mean(valid_fidelities) if valid_fidelities else 0.0,
                "min_fidelity": min(valid_fidelities) if valid_fidelities else 0.0,
                "max_fidelity": max(valid_fidelities) if valid_fidelities else 0.0,
                "tier_distribution": tier_counts,
                "tier_percentages": {
                    tier: f"{(count / total * 100):.1f}%"
                    for tier, count in tier_counts.items()
                }
            }

    def export_audit_log(self) -> List[Dict[str, Any]]:
        """
        Export complete audit log of all governance decisions.

        Returns:
            List of governance results as dictionaries
        """
        with self._lock:
            return [result.to_dict() for result in self.query_log]

    def clear_log(self):
        """Clear the query log."""
        with self._lock:
            self.query_log = []

    def get_pa_info(self) -> Optional[Dict[str, Any]]:
        """
        Get Primacy Attractor information.

        Returns:
            PA dictionary or None if not configured
        """
        with self._lock:
            if self.pa is None:
                return None
            return self.pa.to_dict()

    def get_threshold_info(self) -> Dict[str, float]:
        """
        Get threshold configuration.

        Returns:
            Threshold dictionary
        """
        with self._lock:
            return self.thresholds.to_dict()

    def get_corpus_info(self) -> Dict[str, Any]:
        """
        Get corpus information.

        Returns:
            Dictionary with corpus metadata
        """
        with self._lock:
            return {
                "document_count": len(self.corpus_docs),
                "embedded_count": len(self.corpus_embeddings),
                "documents": self.corpus_docs
            }


# ============================================================================
# MAIN (FOR TESTING)
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("TELOS Governance Engine - Test")
    print("=" * 70)

    # Create a test PA
    print("\n[1] Creating test Primacy Attractor...")
    pa = create_pa(
        name="Healthcare HIPAA",
        purpose="Ensure all healthcare conversations comply with HIPAA regulations",
        scope=["patient privacy", "medical data protection", "healthcare compliance"],
        exclusions=["general medical information", "public health data"],
        prohibitions=["disclosing PHI", "unauthorized data sharing", "HIPAA violations"]
    )

    print(f"  PA Name: {pa.name}")
    print(f"  Purpose: {pa.purpose_statement}")

    # Embed PA
    print("\n[2] Embedding PA...")
    if embed_pa(pa):
        print(f"  PA embedded successfully (dim: {pa.embedding.shape})")
    else:
        print("  ERROR: Failed to embed PA")
        exit(1)

    # Create mock corpus
    print("\n[3] Creating mock corpus...")
    corpus_docs = [
        {
            "document_id": "HIPAA-001",
            "title": "HIPAA Privacy Rule",
            "category": "Privacy",
            "summary": "Protected Health Information privacy requirements"
        },
        {
            "document_id": "HIPAA-002",
            "title": "HIPAA Security Rule",
            "category": "Security",
            "summary": "Electronic PHI security safeguards"
        }
    ]

    print("  Embedding corpus documents...")
    corpus_embeddings = []
    for doc in corpus_docs:
        doc_text = f"{doc['title']} {doc['summary']}"
        embedding = get_embedding(doc_text)
        if embedding is not None:
            corpus_embeddings.append(embedding)
            print(f"    Embedded: {doc['document_id']}")

    # Initialize engine
    print("\n[4] Initializing Governance Engine...")
    engine = GovernanceEngine()

    success, error = engine.configure(
        pa=pa,
        thresholds=ThresholdConfig(),
        corpus_docs=corpus_docs,
        corpus_embeddings=corpus_embeddings
    )

    if success:
        print("  Engine configured successfully")
    else:
        print(f"  ERROR: {error}")
        exit(1)

    # Test queries
    print("\n[5] Testing queries...")
    test_queries = [
        "What are the requirements for protecting patient data?",  # Tier 1 (high fidelity)
        "Can I share medical records with family members?",        # Tier 2 (medium fidelity)
        "What is the weather today?"                               # Tier 3 (low fidelity)
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n  Query {i}: {query}")
        result = engine.process(query)
        print(f"    Fidelity: {result.fidelity:.4f}")
        print(f"    Tier: {result.tier} ({result.tier_name})")
        print(f"    Action: {result.action}")
        if result.retrieved_policies:
            print(f"    Retrieved Policies: {len(result.retrieved_policies)}")
        if result.blocking_reason:
            print(f"    Reason: {result.blocking_reason}")

    # Show statistics
    print("\n[6] Governance Statistics:")
    stats = engine.get_statistics()
    print(f"  Total Queries: {stats['total_queries']}")
    print(f"  Avg Fidelity: {stats['avg_fidelity']:.4f}")
    print(f"  Tier Distribution:")
    for tier in [1, 2, 3]:
        count = stats['tier_distribution'][tier]
        pct = stats['tier_percentages'][tier]
        print(f"    Tier {tier}: {count} ({pct})")

    print("\n" + "=" * 70)
    print("Test complete!")
    print("=" * 70)
