"""
Test script to measure actual embedding distances for calibration.
This will help us set the correct distance_scale for fidelity calculation.
"""

import numpy as np
from telos_purpose.core.embedding_provider import MistralEmbeddingProvider

# Initialize embedding provider
embedder = MistralEmbeddingProvider()

# Test cases: PA vs various user inputs
test_cases = [
    {
        "pa": "I want to continue developing my AI governance project TELOS",
        "tests": [
            ("Tell me what you know about AI governance", "HIGHLY RELATED"),
            ("How does TELOS work?", "HIGHLY RELATED"),
            ("What are the technical challenges in AI alignment?", "RELATED"),
            ("Can you help me write Python code for embeddings?", "SOMEWHAT RELATED"),
            ("What's the weather like today?", "UNRELATED"),
            ("How do I make a PB&J sandwich?", "UNRELATED"),
        ]
    }
]

print("=" * 80)
print("EMBEDDING DISTANCE CALIBRATION TEST")
print("=" * 80)
print(f"Embedding dimension: {embedder.dimension}")
print()

for case in test_cases:
    pa_text = case["pa"]
    pa_embedding = embedder.encode(pa_text)

    print(f"PA: {pa_text}")
    print(f"PA embedding shape: {pa_embedding.shape}")
    print(f"PA embedding norm: {np.linalg.norm(pa_embedding):.3f}")
    print()

    print(f"{'User Input':<60} {'Category':<20} {'Distance':<10} {'Fidelity@2.0':<12} {'Fidelity@8.0'}")
    print("-" * 120)

    for user_input, category in case["tests"]:
        user_embedding = embedder.encode(user_input)

        # Calculate Euclidean distance
        distance = float(np.linalg.norm(user_embedding - pa_embedding))

        # Calculate fidelity with different distance scales
        fidelity_2 = max(0.0, min(1.0, 1.0 - (distance / 2.0)))
        fidelity_8 = max(0.0, min(1.0, 1.0 - (distance / 8.0)))

        print(f"{user_input:<60} {category:<20} {distance:<10.3f} {fidelity_2:<12.3f} {fidelity_8:.3f}")

    print()

print()
print("RECOMMENDATION:")
print("- If HIGHLY RELATED should get fidelity >= 0.85, we need distance_scale where:")
print("  fidelity = 1.0 - (distance / scale) >= 0.85")
print("  scale >= distance / 0.15")
print()
print("- If UNRELATED should get fidelity <= 0.30, we need:")
print("  fidelity = 1.0 - (distance / scale) <= 0.30")
print("  scale <= distance / 0.70")
print()
