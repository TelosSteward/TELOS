#!/usr/bin/env python3
"""
Direct Demo Response Fidelity Analysis
=======================================

Calculates actual TELOS fidelity scores for the EXISTING hardcoded
demo responses (not generating new ones).

This validates whether the hardcoded fidelity values in the demo are realistic.
"""

import sys
import os
from pathlib import Path
import numpy as np

# Add TELOSCOPE_BETA to path
beta_path = Path("/Users/brunnerjf/Desktop/Privacy_PreCommit/TELOSCOPE_BETA")
sys.path.insert(0, str(beta_path))

from demo_mode.telos_framework_demo import get_demo_attractor_config, get_demo_slides
from telos_purpose.core.primacy_math import PrimacyAttractorMath, MathematicalState
from telos_purpose.core.embedding_provider import EmbeddingProvider
import json
import time


def calculate_fidelity_for_response(response_text: str, attractor: PrimacyAttractorMath, embedding_provider: EmbeddingProvider) -> dict:
    """
    Calculate fidelity score for a single response.

    Args:
        response_text: The assistant's response
        attractor: The primacy attractor
        embedding_provider: For encoding text to embeddings

    Returns:
        Dict with fidelity, distance, and basin membership
    """
    # Encode response
    response_embedding = embedding_provider.encode(response_text)

    # Calculate distance to attractor center
    distance = float(np.linalg.norm(response_embedding - attractor.attractor_center))

    # Check basin membership
    in_basin = distance <= attractor.basin_radius

    # Calculate fidelity
    # Per TELOS math: if in basin, fidelity = 1.0
    # If outside basin, fidelity decreases with distance
    if in_basin:
        fidelity = 1.0
    else:
        # Normalized distance beyond basin (0 = at edge, 1 = far beyond)
        distance_beyond = (distance - attractor.basin_radius) / attractor.basin_radius
        # Fidelity decreases as distance increases
        fidelity = max(0.0, 1.0 - distance_beyond)

    return {
        "fidelity": fidelity,
        "distance": distance,
        "in_basin": in_basin,
        "basin_radius": attractor.basin_radius
    }


def analyze_demo_responses():
    """Calculate actual fidelity scores for hardcoded demo responses."""

    print("=" * 80)
    print("TELOS Demo Response Fidelity Analysis (Direct Calculation)")
    print("=" * 80)
    print()

    # 1. Load demo PA configuration
    print("Loading demo PA configuration...")
    demo_config = get_demo_attractor_config()
    print(f"✓ Demo PA loaded")
    print(f"  Purpose: {demo_config['purpose'][0][:60]}...")
    print(f"  Constraint Tolerance: {demo_config['constraint_tolerance']}")
    print()

    # 2. Initialize embedding provider
    print("Initializing embedding provider...")
    try:
        embedding_provider = EmbeddingProvider()
        print("✓ Embeddings initialized")
        print()
    except Exception as e:
        print(f"❌ Error initializing embeddings: {e}")
        return None

    # 3. Build primacy attractor
    print("Building primacy attractor...")
    try:
        # Encode purpose and scope
        purpose_text = " ".join(demo_config["purpose"])
        scope_text = " ".join(demo_config["scope"])

        p_vec = embedding_provider.encode(purpose_text)
        s_vec = embedding_provider.encode(scope_text)

        attractor = PrimacyAttractorMath(
            purpose_vector=p_vec,
            scope_vector=s_vec,
            privacy_level=demo_config["privacy_level"],
            constraint_tolerance=demo_config["constraint_tolerance"],
            task_priority=demo_config["task_priority"]
        )

        print("✓ Attractor built")
        print(f"  Basin radius: {attractor.basin_radius:.3f}")
        print(f"  Constraint tolerance: {demo_config['constraint_tolerance']}")
        print()
    except Exception as e:
        print(f"❌ Error building attractor: {e}")
        import traceback
        traceback.print_exc()
        return None

    # 4. Get demo Q&A pairs
    print("Loading demo Q&A pairs...")
    demo_slides = get_demo_slides()
    print(f"✓ Loaded {len(demo_slides)} Q&A pairs")
    print()

    # 5. Calculate fidelity for each response
    print("Calculating actual fidelity scores for demo responses...")
    print()

    actual_results = []

    for i, (question, response) in enumerate(demo_slides, 1):
        print(f"Turn {i}: Calculating fidelity...")
        print(f"  Question: {question[:60]}...")

        try:
            result = calculate_fidelity_for_response(response, attractor, embedding_provider)
            result["turn"] = i
            result["question"] = question
            result["response_preview"] = response[:100] + "..." if len(response) > 100 else response

            actual_results.append(result)

            print(f"  ✓ Fidelity: {result['fidelity']:.3f}")
            print(f"    Distance: {result['distance']:.3f}")
            print(f"    In Basin: {result['in_basin']}")
            print()

        except Exception as e:
            print(f"  ❌ Error calculating fidelity: {e}")
            print()

    # 6. Compare with hardcoded values
    print("=" * 80)
    print("COMPARISON: Hardcoded vs Actual TELOS Fidelity Scores")
    print("=" * 80)
    print()

    # Hardcoded values from demo slides (USER fidelity only, since that's what we're measuring)
    hardcoded_fidelities = [
        {"turn": 1, "slide": "PA Established", "user_f": 1.000},
        {"turn": 2, "slide": "Show Observation Deck", "user_f": 1.000},
        {"turn": 3, "slide": "Perfect alignment", "user_f": 1.000},
        {"turn": 4, "slide": "Drift detection", "user_f": 0.950},
        {"turn": 5, "slide": "Quantum physics drift", "user_f": 0.650},
        {"turn": 6, "slide": "Dual tracking", "user_f": 0.850},
        {"turn": 7, "slide": "Math behind fidelity", "user_f": 0.800},
        {"turn": 8, "slide": "Intervention strategies", "user_f": 0.920},
        {"turn": 9, "slide": "Constitutional law", "user_f": 0.950},
        {"turn": 10, "slide": "Regulatory compliance", "user_f": 0.940}
    ]

    print(f"{'Turn':<6} {'Slide Description':<30} {'Hardcoded':<12} {'Actual':<12} {'Delta':<10} {'Status':<15}")
    print("-" * 95)

    for i, actual in enumerate(actual_results):
        turn = actual["turn"]
        hardcoded = hardcoded_fidelities[i] if i < len(hardcoded_fidelities) else None

        if hardcoded:
            hardcoded_f = hardcoded["user_f"]
            actual_f = actual["fidelity"]
            delta = actual_f - hardcoded_f

            # Determine status
            if abs(delta) < 0.05:
                status = "✓ Accurate"
            elif abs(delta) < 0.15:
                status = "⚠ Moderate diff"
            else:
                status = "⚠⚠ Large diff"

            slide_desc = hardcoded["slide"][:28]
            print(f"{turn:<6} {slide_desc:<30} {hardcoded_f:<12.3f} {actual_f:<12.3f} {delta:+10.3f} {status:<15}")
        else:
            print(f"{turn:<6} {'N/A':<30} {'N/A':<12} {actual['fidelity']:<12.3f} {'N/A':<10} {'N/A':<15}")

    print()
    print("=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)
    print()

    # Calculate statistics
    deltas = []
    for i, actual in enumerate(actual_results):
        if i < len(hardcoded_fidelities):
            delta = actual["fidelity"] - hardcoded_fidelities[i]["user_f"]
            deltas.append(delta)

    if deltas:
        mean_delta = np.mean(deltas)
        max_delta = max(deltas, key=abs)
        accurate_count = sum(1 for d in deltas if abs(d) < 0.05)

        print(f"Mean Delta: {mean_delta:+.3f}")
        print(f"Max Delta: {max_delta:+.3f}")
        print(f"Accurate (within ±0.05): {accurate_count}/{len(deltas)} ({accurate_count/len(deltas)*100:.1f}%)")
        print()

        # Identify problematic slides
        print("Slides needing review (|delta| > 0.15):")
        for i, delta in enumerate(deltas):
            if abs(delta) > 0.15:
                actual_f = actual_results[i]["fidelity"]
                hardcoded_f = hardcoded_fidelities[i]["user_f"]
                slide = hardcoded_fidelities[i]["slide"]
                print(f"  Turn {i+1} ({slide}): Hardcoded={hardcoded_f:.3f}, Actual={actual_f:.3f}, Delta={delta:+.3f}")
        print()

    # Save results
    output_file = Path("demo_fidelity_comparison.json")
    output_data = {
        "hardcoded_fidelities": hardcoded_fidelities,
        "actual_results": [
            {
                "turn": r["turn"],
                "fidelity": r["fidelity"],
                "distance": r["distance"],
                "in_basin": r["in_basin"],
                "question_preview": r["question"][:100],
                "response_preview": r["response_preview"]
            }
            for r in actual_results
        ],
        "statistics": {
            "mean_delta": float(mean_delta) if deltas else None,
            "max_delta": float(max_delta) if deltas else None,
            "accurate_count": accurate_count if deltas else 0,
            "total_count": len(deltas) if deltas else 0
        },
        "attractor_config": {
            "basin_radius": float(attractor.basin_radius),
            "constraint_tolerance": demo_config["constraint_tolerance"],
            "privacy_level": demo_config["privacy_level"],
            "task_priority": demo_config["task_priority"]
        }
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"✓ Full analysis saved to {output_file}")
    print()

    return output_data


if __name__ == "__main__":
    analyze_demo_responses()
