#!/usr/bin/env python3
"""
Demo Analysis with USER-ESTABLISHED PA
=======================================

Correct dual fidelity analysis:
1. Turn 1: User establishes their PA through their first message
2. Subsequent turns: Measure user drift from THEIR established PA
3. AI responses: Measure against AI PA (serving user's purpose)

This matches how TELOS actually works in production.
"""

import sys
import os
from pathlib import Path
import numpy as np

# Add TELOSCOPE_BETA to path
beta_path = Path("/Users/brunnerjf/Desktop/Privacy_PreCommit/TELOSCOPE_BETA")
sys.path.insert(0, str(beta_path))

from demo_mode.telos_framework_demo import get_demo_attractor_config, get_demo_slides
from telos_purpose.core.primacy_math import PrimacyAttractorMath
from telos_purpose.core.embedding_provider import EmbeddingProvider
import json


def calculate_fidelity(text: str, attractor: PrimacyAttractorMath, embedding_provider: EmbeddingProvider) -> dict:
    """Calculate fidelity score for text against an attractor."""
    embedding = embedding_provider.encode(text)
    distance = float(np.linalg.norm(embedding - attractor.attractor_center))
    in_basin = distance <= attractor.basin_radius

    if in_basin:
        fidelity = 1.0
    else:
        distance_beyond = (distance - attractor.basin_radius) / attractor.basin_radius
        fidelity = max(0.0, 1.0 - distance_beyond)

    return {
        "fidelity": fidelity,
        "distance": distance,
        "in_basin": in_basin
    }


def calculate_primacy_state(user_fidelity: float, ai_fidelity: float) -> float:
    """Calculate Primacy State as harmonic mean of dual fidelities."""
    if user_fidelity + ai_fidelity == 0:
        return 0.0
    return 2 * (user_fidelity * ai_fidelity) / (user_fidelity + ai_fidelity)


def analyze_with_user_established_pa():
    """Analyze demo using USER's established PA from Turn 1."""

    print("=" * 80)
    print("TELOS Demo Analysis with USER-ESTABLISHED PA")
    print("=" * 80)
    print()

    # Initialize embedding provider
    print("Initializing embedding provider...")
    try:
        embedding_provider = EmbeddingProvider()
        print("✓ Embeddings initialized")
        print()
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

    # Get demo Q&A pairs
    demo_slides = get_demo_slides()
    print(f"Loaded {len(demo_slides)} Q&A pairs")
    print()

    # TURN 1: User establishes their PA
    print("=" * 80)
    print("TURN 1: USER ESTABLISHES PA")
    print("=" * 80)
    turn1_question = demo_slides[0][0]
    print(f"User says: \"{turn1_question}\"")
    print()
    print("This establishes the User PA:")
    print("  Purpose: Understand TELOS without technical overwhelm")
    print("  Scope: TELOS concepts, purpose alignment")
    print("  Boundaries: Keep explanations clear and accessible")
    print()

    # Build User PA from Turn 1 message
    print("Building User PA attractor from Turn 1 message...")
    # Use the actual user message as the purpose
    user_purpose = turn1_question
    user_scope = "TELOS framework, purpose alignment, governance concepts"

    # For constraint tolerance, use a tighter value to detect drift
    # In production, this would be calibrated, but for demo we'll use 0.1 (stricter)
    constraint_tolerance = 0.1

    user_p_vec = embedding_provider.encode(user_purpose)
    user_s_vec = embedding_provider.encode(user_scope)

    user_attractor = PrimacyAttractorMath(
        purpose_vector=user_p_vec,
        scope_vector=user_s_vec,
        privacy_level=0.8,
        constraint_tolerance=constraint_tolerance,
        task_priority=0.7
    )

    print(f"✓ User PA built from Turn 1")
    print(f"  Basin radius: {user_attractor.basin_radius:.3f}")
    print(f"  Constraint tolerance: {constraint_tolerance} (stricter for drift detection)")
    print()

    # Build AI PA (serves user's purpose)
    print("Building AI PA (to serve user's purpose)...")
    ai_purpose = "Explain TELOS concepts clearly to help user understand without overwhelming them"
    ai_scope = "TELOS framework, purpose alignment, governance - all explained accessibly"

    ai_p_vec = embedding_provider.encode(ai_purpose)
    ai_s_vec = embedding_provider.encode(ai_scope)

    ai_attractor = PrimacyAttractorMath(
        purpose_vector=ai_p_vec,
        scope_vector=ai_s_vec,
        privacy_level=0.8,
        constraint_tolerance=0.15,  # AI can be slightly more flexible
        task_priority=0.7
    )

    print(f"✓ AI PA built")
    print(f"  Basin radius: {ai_attractor.basin_radius:.3f}")
    print()

    # Analyze all turns
    print("=" * 80)
    print("ANALYZING ALL TURNS")
    print("=" * 80)
    print()

    results = []

    for i, (question, response) in enumerate(demo_slides, 1):
        print(f"Turn {i}:")

        if i == 1:
            # Turn 1: Perfect alignment (just established PA)
            user_f = 1.000
            ai_result = calculate_fidelity(response, ai_attractor, embedding_provider)
            ai_f = ai_result["fidelity"]
            ps = calculate_primacy_state(user_f, ai_f)

            print(f"  User establishes PA - User Fidelity = 1.000 (by definition)")
            print(f"  AI responds - AI Fidelity = {ai_f:.3f}")
            print(f"  Primacy State = {ps:.3f}")

            results.append({
                "turn": i,
                "question": question,
                "user_fidelity": user_f,
                "user_distance": 0.0,
                "user_in_basin": True,
                "ai_fidelity": ai_f,
                "ai_distance": ai_result["distance"],
                "ai_in_basin": ai_result["in_basin"],
                "primacy_state": ps
            })
        else:
            # Subsequent turns: Measure user drift from established PA
            user_result = calculate_fidelity(question, user_attractor, embedding_provider)
            user_f = user_result["fidelity"]

            ai_result = calculate_fidelity(response, ai_attractor, embedding_provider)
            ai_f = ai_result["fidelity"]

            ps = calculate_primacy_state(user_f, ai_f)

            print(f"  Question: {question[:60]}...")
            print(f"  User Fidelity: {user_f:.3f} (distance: {user_result['distance']:.3f})")
            print(f"  AI Fidelity:   {ai_f:.3f} (distance: {ai_result['distance']:.3f})")
            print(f"  Primacy State: {ps:.3f}")

            results.append({
                "turn": i,
                "question": question,
                "user_fidelity": user_f,
                "user_distance": user_result["distance"],
                "user_in_basin": user_result["in_basin"],
                "ai_fidelity": ai_f,
                "ai_distance": ai_result["distance"],
                "ai_in_basin": ai_result["in_basin"],
                "primacy_state": ps
            })

        print()

    # Compare with hardcoded values
    print("=" * 80)
    print("COMPARISON: Hardcoded vs Actual (User-Established PA)")
    print("=" * 80)
    print()

    hardcoded = [
        {"turn": 1, "slide": "PA Established", "user_f": 1.000, "ai_f": 1.000, "ps": 1.000},
        {"turn": 2, "slide": "Show Observation Deck", "user_f": 1.000, "ai_f": 1.000, "ps": 1.000},
        {"turn": 3, "slide": "Perfect alignment", "user_f": 1.000, "ai_f": 1.000, "ps": 1.000},
        {"turn": 4, "slide": "Drift detection", "user_f": 0.950, "ai_f": 0.980, "ps": 0.965},
        {"turn": 5, "slide": "Quantum physics drift", "user_f": 0.650, "ai_f": 0.890, "ps": 0.751},
        {"turn": 6, "slide": "Dual tracking", "user_f": 0.850, "ai_f": 0.920, "ps": 0.884},
        {"turn": 7, "slide": "Math behind fidelity", "user_f": 0.800, "ai_f": 0.910, "ps": 0.852},
        {"turn": 8, "slide": "Intervention strategies", "user_f": 0.920, "ai_f": 0.950, "ps": 0.935},
        {"turn": 9, "slide": "Constitutional law", "user_f": 0.950, "ai_f": 0.970, "ps": 0.960},
        {"turn": 10, "slide": "Regulatory compliance", "user_f": 0.940, "ai_f": 0.960, "ps": 0.950}
    ]

    print(f"{'Turn':<6} {'Slide':<27} {'User Fidelity':<22} {'AI Fidelity':<22} {'Primacy State':<20}")
    print(f"{'':6} {'':27} {'Hard':>10} {'Actual':>10} {'Hard':>10} {'Actual':>10} {'Hard':>9} {'Act':>9}")
    print("-" * 105)

    for i, actual in enumerate(results):
        if i < len(hardcoded):
            hc = hardcoded[i]
            slide = hc["slide"][:25]

            user_match = "✓" if abs(actual["user_fidelity"] - hc["user_f"]) < 0.1 else "⚠"
            ai_match = "✓" if abs(actual["ai_fidelity"] - hc["ai_f"]) < 0.1 else "⚠"

            print(f"{actual['turn']:<6} {slide:<27} {hc['user_f']:>10.3f} {actual['user_fidelity']:>10.3f} {user_match} "
                  f"{hc['ai_f']:>10.3f} {actual['ai_fidelity']:>10.3f} {ai_match} "
                  f"{hc['ps']:>9.3f} {actual['primacy_state']:>9.3f}")

    print()
    print("=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    print()

    # Focus on Turn 5 (quantum physics)
    turn5 = results[4]  # 0-indexed
    hc5 = hardcoded[4]

    print(f"Turn 5 (Quantum Physics Drift):")
    print(f"  Question: \"{demo_slides[4][0]}\"")
    print(f"  Hardcoded User Fidelity: {hc5['user_f']:.3f}")
    print(f"  Actual User Fidelity:    {turn5['user_fidelity']:.3f}")
    print(f"  Delta: {turn5['user_fidelity'] - hc5['user_f']:+.3f}")
    print()

    if turn5['user_fidelity'] < 0.8:
        print("  ✓ DRIFT DETECTED by actual TELOS math!")
    else:
        print("  ⚠ Still measuring high despite topic change")
        print(f"    Basin radius: {user_attractor.basin_radius:.3f}")
        print(f"    Distance: {turn5['user_distance']:.3f}")
        print(f"    In basin: {turn5['user_in_basin']}")
    print()

    # Save results
    output_file = Path("demo_user_established_pa_analysis.json")
    output_data = {
        "user_pa_source": "Turn 1 user message",
        "user_pa_text": turn1_question,
        "basin_radius": float(user_attractor.basin_radius),
        "constraint_tolerance": constraint_tolerance,
        "hardcoded_values": hardcoded,
        "actual_results": [
            {
                "turn": r["turn"],
                "user_fidelity": r["user_fidelity"],
                "user_distance": r["user_distance"],
                "ai_fidelity": r["ai_fidelity"],
                "primacy_state": r["primacy_state"],
                "question": r["question"][:100]
            }
            for r in results
        ]
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"✓ Analysis saved to {output_file}")
    print()

    return output_data


if __name__ == "__main__":
    analyze_with_user_established_pa()
