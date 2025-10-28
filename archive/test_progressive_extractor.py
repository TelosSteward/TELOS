"""
Test script for ProgressivePrimacyExtractor
"""

from telos_purpose.profiling.progressive_primacy_extractor import ProgressivePrimacyExtractor
from telos_purpose.core.unified_steward import PrimacyAttractor
from telos_purpose.llm_clients.mistral_client import TelosMistralClient
from telos_purpose.core.embedding_provider import EmbeddingProvider

# Initialize clients
llm = TelosMistralClient()
embeddings = EmbeddingProvider(deterministic=False)

print("=" * 60)
print("TESTING PROGRESSIVE MODE")
print("=" * 60)

# Test Progressive Mode with statistical convergence
progressive = ProgressivePrimacyExtractor(
    llm_client=llm,
    embedding_provider=embeddings,
    mode='progressive',
    window_size=3,  # Small window for quick test
    centroid_stability_threshold=0.95,
    variance_stability_threshold=0.15,
    confidence_threshold=0.75,
    consecutive_stable_turns=2
)

test_turns = [
    ("What is Python?", "Python is a high-level programming language."),
    ("How do I define a function?", "You can define a function using the 'def' keyword."),
    ("What are lists?", "Lists are mutable sequences in Python."),
    ("How do I use loops?", "You can use for loops and while loops in Python."),
]

print("\nProcessing turns (Progressive Mode):\n")
for i, (user_msg, assistant_msg) in enumerate(test_turns, 1):
    result = progressive.add_turn(user_msg, assistant_msg)

    print(f"Turn {i}:")
    print(f"  User: {user_msg}")
    print(f"  Status: {result['status_message']}")
    if result['fidelity'] is not None:
        print(f"  Fidelity: {result['fidelity']:.3f}")
    print()

print(f"\n✅ Baseline established: {progressive.is_ready()}")

if progressive.is_ready():
    attractor = progressive.get_attractor()
    print(f"\nLearned Attractor:")
    print(f"  Purpose: {attractor.purpose}")
    print(f"  Scope: {attractor.scope}")
    print(f"  Boundaries: {attractor.boundaries}")

print("\n" + "=" * 60)
print("TESTING HYBRID MODE")
print("=" * 60)

# Test Hybrid Mode
seed_attractor = PrimacyAttractor(
    purpose=["Provide programming guidance"],
    scope=["Software development", "Coding best practices"],
    boundaries=["No medical advice", "No financial advice"],
    constraint_tolerance=0.2,
    privacy_level=0.8,
    task_priority=0.7
)

hybrid = ProgressivePrimacyExtractor(
    llm_client=llm,
    embedding_provider=embeddings,
    mode='hybrid',
    seed_attractor=seed_attractor,
    window_size=3,  # Small window for quick test
    centroid_stability_threshold=0.95,
    variance_stability_threshold=0.15,
    confidence_threshold=0.75,
    consecutive_stable_turns=2
)

print("\nSeed Attractor:")
print(f"  Boundaries: {seed_attractor.boundaries}")

print("\nProcessing turns (Hybrid Mode):\n")
for i, (user_msg, assistant_msg) in enumerate(test_turns, 1):
    result = hybrid.add_turn(user_msg, assistant_msg)

    print(f"Turn {i}:")
    print(f"  User: {user_msg}")
    print(f"  Status: {result['status_message']}")
    if result['fidelity'] is not None:
        print(f"  Fidelity: {result['fidelity']:.3f}")
    print()

print(f"\n✅ Baseline established: {hybrid.is_ready()}")

if hybrid.is_ready():
    attractor = hybrid.get_attractor()
    print(f"\nHybrid Attractor:")
    print(f"  Purpose (LEARNED): {attractor.purpose}")
    print(f"  Scope (LEARNED): {attractor.scope}")
    print(f"  Boundaries (PRE-DEFINED): {attractor.boundaries}")
    print(f"  Privacy (PRE-DEFINED): {attractor.privacy_level}")
    print(f"  Constraint Tolerance (PRE-DEFINED): {attractor.constraint_tolerance}")

print("\n" + "=" * 60)
print("✅ ALL TESTS COMPLETE")
print("=" * 60)
