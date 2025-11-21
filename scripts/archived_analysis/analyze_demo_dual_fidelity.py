#!/usr/bin/env python3
"""
Dual Fidelity Analysis for Demo Conversation
=============================================

Calculates BOTH user and AI fidelity scores for the demo Q&A pairs.

USER FIDELITY: How aligned is the user's question with their stated purpose?
AI FIDELITY: How aligned is the AI's response with serving that purpose?

This is the correct dual-attractor analysis.
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
    """
    Calculate fidelity score for any text (user input or AI response).

    Args:
        text: The text to measure
        attractor: The primacy attractor
        embedding_provider: For encoding text to embeddings

    Returns:
        Dict with fidelity, distance, and basin membership
    """
    # Encode text
    embedding = embedding_provider.encode(text)

    # Calculate distance to attractor center
    distance = float(np.linalg.norm(embedding - attractor.attractor_center))

    # Check basin membership
    in_basin = distance <= attractor.basin_radius

    # Calculate fidelity
    if in_basin:
        fidelity = 1.0
    else:
        # Distance beyond basin
        distance_beyond = (distance - attractor.basin_radius) / attractor.basin_radius
        fidelity = max(0.0, 1.0 - distance_beyond)

    return {
        "fidelity": fidelity,
        "distance": distance,
        "in_basin": in_basin
    }


def calculate_primacy_state(user_fidelity: float, ai_fidelity: float) -> float:
    """
    Calculate Primacy State from dual fidelities.

    Per TELOS: Primacy State is the harmonic mean of user and AI fidelities.
    """
    if user_fidelity + ai_fidelity == 0:
        return 0.0

    # Harmonic mean
    return 2 * (user_fidelity * ai_fidelity) / (user_fidelity + ai_fidelity)


def analyze_dual_fidelity():
    """Calculate both user and AI fidelity scores for demo conversation."""

    print("=" * 80)
    print("TELOS Dual Fidelity Analysis - Demo Conversation")
    print("=" * 80)
    print()

    # 1. Load demo PA configuration
    print("Loading demo PA configuration (User PA)...")
    demo_config = get_demo_attractor_config()
    print(f"✓ User PA loaded")
    print(f"  Purpose: {demo_config['purpose'][0]}")
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

    # 3. Build User PA attractor (what user wants to accomplish)
    print("Building User PA attractor...")
    try:
        purpose_text = " ".join(demo_config["purpose"])
        scope_text = " ".join(demo_config["scope"])

        p_vec = embedding_provider.encode(purpose_text)
        s_vec = embedding_provider.encode(scope_text)

        user_attractor = PrimacyAttractorMath(
            purpose_vector=p_vec,
            scope_vector=s_vec,
            privacy_level=demo_config["privacy_level"],
            constraint_tolerance=demo_config["constraint_tolerance"],
            task_priority=demo_config["task_priority"]
        )

        print("✓ User PA attractor built")
        print(f"  Basin radius: {user_attractor.basin_radius:.3f}")
        print()
    except Exception as e:
        print(f"❌ Error building user attractor: {e}")
        import traceback
        traceback.print_exc()
        return None

    # 4. Build AI PA attractor (how AI should serve user's purpose)
    print("Building AI PA attractor...")
    # AI PA is derived from user PA - stays focused on explaining TELOS
    ai_purpose = ["Explain TELOS concepts clearly to help user understand purpose alignment"]
    ai_scope = ["TELOS framework concepts", "Purpose alignment", "Governance mechanisms"]

    ai_p_vec = embedding_provider.encode(" ".join(ai_purpose))
    ai_s_vec = embedding_provider.encode(" ".join(ai_scope))

    ai_attractor = PrimacyAttractorMath(
        purpose_vector=ai_p_vec,
        scope_vector=ai_s_vec,
        privacy_level=demo_config["privacy_level"],
        constraint_tolerance=demo_config["constraint_tolerance"],
        task_priority=demo_config["task_priority"]
    )

    print("✓ AI PA attractor built")
    print(f"  Basin radius: {ai_attractor.basin_radius:.3f}")
    print()

    # 5. Get demo Q&A pairs
    print("Loading demo Q&A pairs...")
    demo_slides = get_demo_slides()
    print(f"✓ Loaded {len(demo_slides)} Q&A pairs")
    print()

    # 6. Calculate dual fidelity for each turn
    print("Calculating dual fidelity scores...")
    print()

    results = []

    for i, (question, response) in enumerate(demo_slides, 1):
        print(f"Turn {i}:")
        print(f"  Question: {question[:70]}...")

        try:
            # USER FIDELITY: Measure user's question against User PA
            user_result = calculate_fidelity(question, user_attractor, embedding_provider)
            user_f = user_result["fidelity"]

            # AI FIDELITY: Measure AI's response against AI PA
            ai_result = calculate_fidelity(response, ai_attractor, embedding_provider)
            ai_f = ai_result["fidelity"]

            # PRIMACY STATE: Combine both fidelities
            ps = calculate_primacy_state(user_f, ai_f)

            results.append({
                "turn": i,
                "question": question,
                "response": response[:100] + "..." if len(response) > 100 else response,
                "user_fidelity": user_f,
                "user_distance": user_result["distance"],
                "user_in_basin": user_result["in_basin"],
                "ai_fidelity": ai_f,
                "ai_distance": ai_result["distance"],
                "ai_in_basin": ai_result["in_basin"],
                "primacy_state": ps
            })

            print(f"  ✓ User Fidelity: {user_f:.3f} (distance: {user_result['distance']:.3f})")
            print(f"    AI Fidelity:   {ai_f:.3f} (distance: {ai_result['distance']:.3f})")
            print(f"    Primacy State: {ps:.3f}")
            print()

        except Exception as e:
            print(f"  ❌ Error: {e}")
            print()

    # 7. Compare with hardcoded values
    print("=" * 80)
    print("COMPARISON: Hardcoded vs Actual Dual Fidelity Scores")
    print("=" * 80)
    print()

    # Hardcoded values from demo
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

    print(f"{'Turn':<6} {'Slide':<25} {'User F':<20} {'AI F':<20} {'PS':<15}")
    print(f"{'':6} {'':25} {'Hard':>9} {'Actual':>9} {'Hard':>9} {'Actual':>9} {'Hard':>7} {'Act':>7}")
    print("-" * 95)

    for i, actual in enumerate(results):
        if i < len(hardcoded):
            hc = hardcoded[i]
            slide = hc["slide"][:23]

            user_delta = actual["user_fidelity"] - hc["user_f"]
            ai_delta = actual["ai_fidelity"] - hc["ai_f"]
            ps_delta = actual["primacy_state"] - hc["ps"]

            print(f"{actual['turn']:<6} {slide:<25} {hc['user_f']:>9.3f} {actual['user_fidelity']:>9.3f} "
                  f"{hc['ai_f']:>9.3f} {actual['ai_fidelity']:>9.3f} "
                  f"{hc['ps']:>7.3f} {actual['primacy_state']:>7.3f}")

    print()
    print("=" * 80)
    print("DELTA ANALYSIS")
    print("=" * 80)
    print()

    user_deltas = []
    ai_deltas = []

    for i, actual in enumerate(results):
        if i < len(hardcoded):
            hc = hardcoded[i]
            user_delta = actual["user_fidelity"] - hc["user_f"]
            ai_delta = actual["ai_fidelity"] - hc["ai_f"]
            user_deltas.append(user_delta)
            ai_deltas.append(ai_delta)

            if abs(user_delta) > 0.15 or abs(ai_delta) > 0.15:
                print(f"Turn {actual['turn']} ({hc['slide']}):")
                print(f"  User: Hardcoded={hc['user_f']:.3f}, Actual={actual['user_fidelity']:.3f}, Delta={user_delta:+.3f}")
                print(f"  AI:   Hardcoded={hc['ai_f']:.3f}, Actual={actual['ai_fidelity']:.3f}, Delta={ai_delta:+.3f}")
                print()

    if user_deltas and ai_deltas:
        print(f"User Fidelity - Mean Delta: {np.mean(user_deltas):+.3f}, Max Delta: {max(user_deltas, key=abs):+.3f}")
        print(f"AI Fidelity   - Mean Delta: {np.mean(ai_deltas):+.3f}, Max Delta: {max(ai_deltas, key=abs):+.3f}")
        print()

    # Save results
    output_file = Path("demo_dual_fidelity_analysis.json")
    output_data = {
        "hardcoded_values": hardcoded,
        "actual_results": [
            {
                "turn": r["turn"],
                "user_fidelity": r["user_fidelity"],
                "ai_fidelity": r["ai_fidelity"],
                "primacy_state": r["primacy_state"],
                "question_preview": r["question"][:100]
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
    analyze_dual_fidelity()
