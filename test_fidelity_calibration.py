#!/usr/bin/env python3
"""
Test fidelity calculations with the new basin/tolerance values.
Uses Mistral embeddings to calculate cosine similarity.
"""
import os
import sys
import numpy as np

# Add project to path
sys.path.insert(0, '/Users/brunnerjf/Desktop/telos_privacy')
sys.path.insert(0, '/Users/brunnerjf/Desktop/telos_privacy/telos_observatory_v3')
os.chdir('/Users/brunnerjf/Desktop/telos_privacy')

import toml

# Load from secrets.toml
secrets_path = '/Users/brunnerjf/Desktop/telos_privacy/.streamlit/secrets.toml'
secrets = toml.load(secrets_path)

from mistralai import Mistral

class MistralEmbeddingProvider:
    """Simple Mistral embedding provider for testing."""
    def __init__(self):
        api_key = secrets.get("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY not found in secrets.toml")
        self.client = Mistral(api_key=api_key)
        self.model = "mistral-embed"
        self.dimension = 1024

    def encode(self, text: str) -> list:
        """Get embedding for text."""
        response = self.client.embeddings.create(
            model=self.model,
            inputs=[text]
        )
        return response.data[0].embedding

# Your PA components
PA_PURPOSE = "Explore and test TELOS Observatory capabilities, understand AI governance through purposeful conversation"
PA_SCOPE = "Focus on: AI governance, TELOS features, meaningful dialogue. Avoid: completely unrelated tangents"
PA_SUCCESS = "Productive conversation that demonstrates TELOS principles and stays purposefully aligned"
PA_STYLE = "Natural, conversational, thoughtful"

# Construct PA embedding text (matching how beta_response_manager does it)
PA_TEXT = f"Purpose: {PA_PURPOSE}. Scope: {PA_SCOPE}."

# New thresholds
BASIN = 0.40
TOLERANCE = 0.04
INTERVENTION_THRESHOLD = BASIN - TOLERANCE  # 0.36
SIMILARITY_BASELINE = 0.35

FIDELITY_GREEN = 0.85
FIDELITY_YELLOW = 0.70
FIDELITY_ORANGE = 0.50

# Test messages - mix of on-topic, related, and off-topic
TEST_MESSAGES = [
    # Should be HIGH fidelity (green)
    "How does TELOS detect drift in conversations?",
    "Tell me about the TELOS Observatory features",
    "What is AI governance and how does TELOS implement it?",
    "How does the fidelity calculation work in TELOS?",

    # Should be MODERATE fidelity (yellow)
    "I want to tell you all about my AI Alignment project TELOS",
    "Can we have a meaningful dialogue about AI safety?",
    "What are the key principles of purposeful conversation?",

    # Should be LOW fidelity (orange)
    "Let's discuss machine learning models in general",
    "How do neural networks work?",

    # Should trigger INTERVENTION (red/outside basin)
    "What's a good recipe for chocolate cake?",
    "Tell me about the history of ancient Rome",
    "How do I fix my car's transmission?",
]

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_fidelity_zone(score):
    """Determine which zone a fidelity score falls into."""
    if score >= FIDELITY_GREEN:
        return "GREEN (aligned)"
    elif score >= FIDELITY_YELLOW:
        return "YELLOW (minor drift)"
    elif score >= FIDELITY_ORANGE:
        return "ORANGE (moderate drift)"
    elif score >= INTERVENTION_THRESHOLD:
        return "ORANGE-LOW (near intervention)"
    else:
        return "RED (INTERVENTION)"

def main():
    print("=" * 70)
    print("TELOS Fidelity Calibration Test")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  BASIN = {BASIN}")
    print(f"  TOLERANCE = {TOLERANCE}")
    print(f"  INTERVENTION_THRESHOLD = {INTERVENTION_THRESHOLD}")
    print(f"  SIMILARITY_BASELINE = {SIMILARITY_BASELINE}")
    print(f"\nUI Thresholds:")
    print(f"  GREEN >= {FIDELITY_GREEN}")
    print(f"  YELLOW >= {FIDELITY_YELLOW}")
    print(f"  ORANGE >= {FIDELITY_ORANGE}")
    print(f"  RED < {INTERVENTION_THRESHOLD}")

    print(f"\nPA Text for embedding:")
    print(f"  '{PA_TEXT[:80]}...'")

    print("\n" + "=" * 70)
    print("Initializing Mistral embedding provider...")

    try:
        provider = MistralEmbeddingProvider()
        print(f"  Dimension: {provider.dimension}")
    except Exception as e:
        print(f"ERROR: Could not initialize embedding provider: {e}")
        return

    # Get PA embedding
    print("\nGenerating PA embedding...")
    pa_embedding = np.array(provider.encode(PA_TEXT))
    print(f"  PA embedding shape: {pa_embedding.shape}")

    # Test each message
    print("\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)

    results = []
    for msg in TEST_MESSAGES:
        msg_embedding = np.array(provider.encode(msg))
        similarity = cosine_similarity(msg_embedding, pa_embedding)
        zone = get_fidelity_zone(similarity)
        results.append((msg, similarity, zone))

        # Check baseline
        baseline_block = similarity < SIMILARITY_BASELINE
        intervention = similarity < INTERVENTION_THRESHOLD

        print(f"\n{similarity:.3f} | {zone}")
        print(f"  '{msg[:60]}{'...' if len(msg) > 60 else ''}'")
        if baseline_block:
            print(f"  >>> BASELINE HARD BLOCK (raw_sim < {SIMILARITY_BASELINE})")
        elif intervention:
            print(f"  >>> INTERVENTION TRIGGERED (fidelity < {INTERVENTION_THRESHOLD})")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    scores = [r[1] for r in results]
    print(f"  Min: {min(scores):.3f}")
    print(f"  Max: {max(scores):.3f}")
    print(f"  Mean: {np.mean(scores):.3f}")
    print(f"  Std: {np.std(scores):.3f}")

    # Zone distribution
    zones = {}
    for _, _, zone in results:
        zones[zone] = zones.get(zone, 0) + 1
    print(f"\nZone distribution:")
    for zone, count in sorted(zones.items()):
        print(f"  {zone}: {count}")

if __name__ == "__main__":
    main()
