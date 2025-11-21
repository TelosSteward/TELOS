#!/usr/bin/env python3
"""
Demo Analysis with constraint_tolerance = 0.05 (Recommended BETA Setting)
===========================================================================

Tests whether stricter basin tolerance catches drift in demo conversation.
"""

import sys
import os
from pathlib import Path
import numpy as np

# Add TELOSCOPE_BETA to path
beta_path = Path("/Users/brunnerjf/Desktop/Privacy_PreCommit/TELOSCOPE_BETA")
sys.path.insert(0, str(beta_path))

from demo_mode.telos_framework_demo import get_demo_slides
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


def analyze_with_tolerance_05():
    """Analyze demo with constraint_tolerance = 0.05 (recommended BETA setting)."""

    print("=" * 80)
    print("TELOS Demo Analysis with constraint_tolerance = 0.05")
    print("(Recommended BETA Setting)")
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

    # TURN 1: User establishes PA
    print("=" * 80)
    print("BUILDING ATTRACTORS WITH TOLERANCE = 0.05")
    print("=" * 80)
    turn1_question = demo_slides[0][0]
    print(f"User PA from Turn 1: \"{turn1_question}\"")
    print()

    # CRITICAL: Use constraint_tolerance = 0.05
    constraint_tolerance = 0.05

    # Build User PA
    user_purpose = turn1_question
    user_scope = "TELOS framework, purpose alignment, governance concepts"

    user_p_vec = embedding_provider.encode(user_purpose)
    user_s_vec = embedding_provider.encode(user_scope)

    user_attractor = PrimacyAttractorMath(
        purpose_vector=user_p_vec,
        scope_vector=user_s_vec,
        privacy_level=0.8,
        constraint_tolerance=constraint_tolerance,  # ← 0.05
        task_priority=0.7
    )

    print(f"✓ User PA built with tolerance = {constraint_tolerance}")
    print(f"  Basin radius: {user_attractor.basin_radius:.3f}")
    print(f"  Rigidity: {1.0 - constraint_tolerance:.2f}")
    print()

    # Build AI PA
    ai_purpose = "Explain TELOS concepts clearly to help user understand without overwhelming them"
    ai_scope = "TELOS framework, purpose alignment, governance - all explained accessibly"

    ai_p_vec = embedding_provider.encode(ai_purpose)
    ai_s_vec = embedding_provider.encode(ai_scope)

    ai_attractor = PrimacyAttractorMath(
        purpose_vector=ai_p_vec,
        scope_vector=ai_s_vec,
        privacy_level=0.8,
        constraint_tolerance=0.08,  # AI slightly more flexible
        task_priority=0.7
    )

    print(f"✓ AI PA built with tolerance = 0.08")
    print(f"  Basin radius: {ai_attractor.basin_radius:.3f}")
    print()

    # Analyze all turns
    print("=" * 80)
    print("ANALYZING ALL TURNS")
    print("=" * 80)
    print()

    results = []

    for i, (question, response) in enumerate(demo_slides, 1):
        if i == 1:
            # Turn 1: Perfect alignment (just established PA)
            user_f = 1.000
            ai_result = calculate_fidelity(response, ai_attractor, embedding_provider)
            ai_f = ai_result["fidelity"]
            ps = calculate_primacy_state(user_f, ai_f)

            print(f"Turn {i}: PA Established")
            print(f"  User Fidelity = 1.000 (by definition)")
            print(f"  AI Fidelity = {ai_f:.3f}")
            print(f"  Primacy State = {ps:.3f}")
            print()

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
            # Measure user drift
            user_result = calculate_fidelity(question, user_attractor, embedding_provider)
            user_f = user_result["fidelity"]

            ai_result = calculate_fidelity(response, ai_attractor, embedding_provider)
            ai_f = ai_result["fidelity"]

            ps = calculate_primacy_state(user_f, ai_f)

            # Highlight drift detection
            drift_indicator = ""
            if not user_result["in_basin"]:
                drift_indicator = " ⚠️ DRIFT DETECTED"
            elif user_f < 0.9:
                drift_indicator = " ⚠️ APPROACHING DRIFT"

            print(f"Turn {i}:{drift_indicator}")
            print(f"  Question: {question[:60]}...")
            print(f"  User Fidelity: {user_f:.3f} (distance: {user_result['distance']:.3f}, in_basin: {user_result['in_basin']})")
            print(f"  AI Fidelity:   {ai_f:.3f} (distance: {ai_result['distance']:.3f}, in_basin: {ai_result['in_basin']})")
            print(f"  Primacy State: {ps:.3f}")
            print()

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

    # Detailed comparison
    print("=" * 80)
    print("COMPARISON: Hardcoded vs Actual (Tolerance = 0.05)")
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

    print(f"{'Turn':<6} {'Slide':<27} {'User F (Hard)':<14} {'User F (Act)':<14} {'Delta':<10} {'Status':<20}")
    print("-" * 100)

    for i, actual in enumerate(results):
        if i < len(hardcoded):
            hc = hardcoded[i]
            slide = hc["slide"][:25]
            delta = actual["user_fidelity"] - hc["user_f"]

            # Status indicator
            if abs(delta) < 0.05:
                status = "✓ Close match"
            elif abs(delta) < 0.15:
                status = "⚠ Moderate diff"
            elif not actual["user_in_basin"]:
                status = "⚠⚠ OUT OF BASIN"
            else:
                status = "⚠⚠ Large diff"

            print(f"{actual['turn']:<6} {slide:<27} {hc['user_f']:<14.3f} {actual['user_fidelity']:<14.3f} {delta:+10.3f} {status:<20}")

    print()
    print("=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    print()

    # Turn 5 analysis
    turn5 = results[4]
    hc5 = hardcoded[4]

    print(f"Turn 5 (Quantum Physics) - The Critical Test:")
    print(f"  Question: \"{demo_slides[4][0]}\"")
    print(f"  Basin radius: {user_attractor.basin_radius:.3f}")
    print(f"  User distance: {turn5['user_distance']:.3f}")
    print(f"  In basin: {turn5['user_in_basin']}")
    print(f"  Hardcoded fidelity: {hc5['user_f']:.3f}")
    print(f"  Actual fidelity: {turn5['user_fidelity']:.3f}")
    print(f"  Delta: {turn5['user_fidelity'] - hc5['user_f']:+.3f}")
    print()

    if not turn5['user_in_basin']:
        print("  ✅ SUCCESS: Drift detected with tolerance = 0.05!")
        print(f"     User fell outside basin ({turn5['user_distance']:.3f} > {user_attractor.basin_radius:.3f})")
    elif turn5['user_fidelity'] < 0.85:
        print("  ⚠️ PARTIAL: Still in basin but fidelity degraded")
    else:
        print("  ❌ MISS: No drift detected even at 0.05 tolerance")
        print(f"     Distance ({turn5['user_distance']:.3f}) still within basin ({user_attractor.basin_radius:.3f})")
    print()

    # Statistics
    print("Overall Statistics:")
    drift_detected = sum(1 for r in results if not r["user_in_basin"])
    low_fidelity = sum(1 for r in results if r["user_fidelity"] < 0.85)

    print(f"  Turns with drift detected (out of basin): {drift_detected}/{len(results)}")
    print(f"  Turns with fidelity < 0.85: {low_fidelity}/{len(results)}")
    print(f"  Basin adherence rate: {sum(1 for r in results if r['user_in_basin'])}/{len(results)} ({sum(1 for r in results if r['user_in_basin'])/len(results)*100:.1f}%)")
    print()

    # Save results
    output_file = Path("demo_analysis_tolerance_05.json")
    output_data = {
        "constraint_tolerance": constraint_tolerance,
        "user_basin_radius": float(user_attractor.basin_radius),
        "ai_basin_radius": float(ai_attractor.basin_radius),
        "hardcoded_values": hardcoded,
        "actual_results": [
            {
                "turn": r["turn"],
                "user_fidelity": r["user_fidelity"],
                "user_distance": r["user_distance"],
                "user_in_basin": r["user_in_basin"],
                "ai_fidelity": r["ai_fidelity"],
                "primacy_state": r["primacy_state"],
                "question": r["question"][:100]
            }
            for r in results
        ],
        "statistics": {
            "drift_detected_count": drift_detected,
            "low_fidelity_count": low_fidelity,
            "basin_adherence_rate": sum(1 for r in results if r['user_in_basin']) / len(results)
        }
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"✓ Analysis saved to {output_file}")
    print()

    return output_data


if __name__ == "__main__":
    analyze_with_tolerance_05()
