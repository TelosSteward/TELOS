#!/usr/bin/env python3
"""
Primacy State Performance Benchmark
Tests computational overhead of PS calculation
Target: <20ms per calculation (p95)
"""

import time
import numpy as np
from typing import List, Tuple
import statistics

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)

def compute_primacy_state(
    response_embedding: np.ndarray,
    user_pa_embedding: np.ndarray,
    ai_pa_embedding: np.ndarray
) -> Tuple[float, float, float, float]:
    """
    Compute Primacy State from dual PA dynamics.

    Returns:
        Tuple of (PS, F_user, F_AI, rho_PA)
    """
    # Component fidelities
    F_user = cosine_similarity(response_embedding, user_pa_embedding)
    F_AI = cosine_similarity(response_embedding, ai_pa_embedding)

    # PA correlation (could be cached per session)
    rho_PA = cosine_similarity(user_pa_embedding, ai_pa_embedding)

    # Harmonic mean with epsilon for stability
    epsilon = 1e-10
    if F_user + F_AI > epsilon:
        harmonic_mean = (2 * F_user * F_AI) / (F_user + F_AI + epsilon)
    else:
        harmonic_mean = 0.0

    # Primacy State Score
    PS = rho_PA * harmonic_mean

    return PS, F_user, F_AI, rho_PA

def benchmark_ps_calculation(
    iterations: int = 1000,
    embedding_dim: int = 1536  # OpenAI ada-002 dimension
) -> dict:
    """
    Benchmark PS calculation performance.

    Args:
        iterations: Number of calculations to perform
        embedding_dim: Dimension of embeddings (1536 for OpenAI)

    Returns:
        Performance statistics
    """
    print(f"🔬 Benchmarking Primacy State calculation")
    print(f"   Iterations: {iterations}")
    print(f"   Embedding dimension: {embedding_dim}")
    print()

    # Generate random embeddings for testing
    np.random.seed(42)
    timings = []

    for i in range(iterations):
        # Generate test embeddings (normalized)
        response = np.random.randn(embedding_dim)
        response = response / np.linalg.norm(response)

        user_pa = np.random.randn(embedding_dim)
        user_pa = user_pa / np.linalg.norm(user_pa)

        ai_pa = np.random.randn(embedding_dim)
        ai_pa = ai_pa / np.linalg.norm(ai_pa)

        # Time the PS calculation
        start_time = time.perf_counter()
        PS, F_user, F_AI, rho_PA = compute_primacy_state(response, user_pa, ai_pa)
        end_time = time.perf_counter()

        timing_ms = (end_time - start_time) * 1000
        timings.append(timing_ms)

        # Show progress
        if (i + 1) % 100 == 0:
            print(f"   Completed {i + 1}/{iterations} iterations...")

    # Calculate statistics
    timings.sort()

    stats = {
        'count': len(timings),
        'mean_ms': statistics.mean(timings),
        'median_ms': statistics.median(timings),
        'stdev_ms': statistics.stdev(timings) if len(timings) > 1 else 0,
        'min_ms': min(timings),
        'max_ms': max(timings),
        'p50_ms': timings[int(len(timings) * 0.50)],
        'p90_ms': timings[int(len(timings) * 0.90)],
        'p95_ms': timings[int(len(timings) * 0.95)],
        'p99_ms': timings[int(len(timings) * 0.99)],
    }

    return stats

def test_ps_correctness():
    """Test PS calculation correctness with known values."""
    print("🧪 Testing PS calculation correctness")
    print()

    # Test 1: Perfect alignment
    embedding_dim = 100
    response = np.ones(embedding_dim) / np.sqrt(embedding_dim)
    user_pa = np.ones(embedding_dim) / np.sqrt(embedding_dim)
    ai_pa = np.ones(embedding_dim) / np.sqrt(embedding_dim)

    PS, F_user, F_AI, rho_PA = compute_primacy_state(response, user_pa, ai_pa)

    print(f"Test 1 - Perfect Alignment:")
    print(f"  F_user = {F_user:.4f} (expected: 1.0000)")
    print(f"  F_AI = {F_AI:.4f} (expected: 1.0000)")
    print(f"  ρ_PA = {rho_PA:.4f} (expected: 1.0000)")
    print(f"  PS = {PS:.4f} (expected: 1.0000)")
    print()

    # Test 2: User PA drift
    response = np.zeros(embedding_dim)
    response[0] = 1.0  # Orthogonal to user_pa

    PS, F_user, F_AI, rho_PA = compute_primacy_state(response, user_pa, ai_pa)

    print(f"Test 2 - User PA Drift:")
    print(f"  F_user = {F_user:.4f} (expected: ~0.1)")
    print(f"  F_AI = {F_AI:.4f} (expected: ~0.1)")
    print(f"  ρ_PA = {rho_PA:.4f} (expected: 1.0000)")
    print(f"  PS = {PS:.4f} (expected: low)")
    print()

    # Test 3: PA misalignment
    user_pa = np.ones(embedding_dim) / np.sqrt(embedding_dim)
    ai_pa = np.zeros(embedding_dim)
    ai_pa[0] = 1.0  # Orthogonal to user_pa
    response = user_pa.copy()

    PS, F_user, F_AI, rho_PA = compute_primacy_state(response, user_pa, ai_pa)

    print(f"Test 3 - PA Misalignment:")
    print(f"  F_user = {F_user:.4f} (expected: 1.0000)")
    print(f"  F_AI = {F_AI:.4f} (expected: ~0.1)")
    print(f"  ρ_PA = {rho_PA:.4f} (expected: ~0.1)")
    print(f"  PS = {PS:.4f} (expected: very low)")
    print()

def main():
    """Run performance benchmarks and correctness tests."""
    print("=" * 60)
    print("PRIMACY STATE PERFORMANCE BENCHMARK")
    print("=" * 60)
    print()

    # Test correctness first
    test_ps_correctness()

    print("-" * 60)

    # Run performance benchmark
    stats = benchmark_ps_calculation(iterations=1000)

    print()
    print("📊 Performance Results:")
    print(f"  Mean:   {stats['mean_ms']:.3f} ms")
    print(f"  Median: {stats['median_ms']:.3f} ms")
    print(f"  StdDev: {stats['stdev_ms']:.3f} ms")
    print(f"  Min:    {stats['min_ms']:.3f} ms")
    print(f"  Max:    {stats['max_ms']:.3f} ms")
    print()
    print(f"📈 Percentiles:")
    print(f"  p50:    {stats['p50_ms']:.3f} ms")
    print(f"  p90:    {stats['p90_ms']:.3f} ms")
    print(f"  p95:    {stats['p95_ms']:.3f} ms ← Target: <20ms")
    print(f"  p99:    {stats['p99_ms']:.3f} ms")
    print()

    # Check if we meet target
    target_ms = 20.0
    if stats['p95_ms'] < target_ms:
        print(f"✅ SUCCESS: p95 latency ({stats['p95_ms']:.3f} ms) < {target_ms} ms target")
        print("   PS calculation is fast enough for production!")
    else:
        print(f"⚠️  WARNING: p95 latency ({stats['p95_ms']:.3f} ms) > {target_ms} ms target")
        print("   May need optimization for production use")

    print()
    print("=" * 60)

    # Return verdict
    return stats['p95_ms'] < target_ms

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)