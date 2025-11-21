#!/usr/bin/env python3
"""
Demo Conversation Fidelity Analysis
====================================

Runs the hardcoded demo Q&A pairs through actual TELOS calculations
using ObservationRunner to compare hardcoded vs actual fidelity scores.

This validates whether the demo's fidelity values are realistic.
"""

import sys
import os
from pathlib import Path

# Add TELOSCOPE_BETA to path
beta_path = Path("/Users/brunnerjf/Desktop/Privacy_PreCommit/TELOSCOPE_BETA")
sys.path.insert(0, str(beta_path))

from demo_mode.telos_framework_demo import get_demo_attractor_config, get_demo_slides
from telos_purpose.validation.baseline_runners import ObservationRunner
from telos_purpose.core.unified_steward import PrimacyAttractor
from telos_purpose.core.embedding_provider import EmbeddingProvider
from telos_purpose.llm_clients.mistral_client import MistralClient
import json


def analyze_demo_conversation():
    """Run demo Q&A pairs through TELOS observation mode."""

    print("=" * 80)
    print("TELOS Demo Conversation Fidelity Analysis")
    print("=" * 80)
    print()

    # 1. Load demo PA configuration
    print("Loading demo PA configuration...")
    demo_config = get_demo_attractor_config()
    attractor = PrimacyAttractor(
        purpose=demo_config["purpose"],
        scope=demo_config["scope"],
        boundaries=demo_config["boundaries"],
        constraint_tolerance=demo_config["constraint_tolerance"],
        privacy_level=demo_config["privacy_level"],
        task_priority=demo_config["task_priority"]
    )
    print(f"✓ Demo PA loaded with {len(demo_config['purpose'])} purpose statements")
    print()

    # 2. Get demo Q&A pairs
    print("Loading demo Q&A pairs...")
    demo_slides = get_demo_slides()
    conversation = [(q, a) for q, a in demo_slides]
    print(f"✓ Loaded {len(conversation)} Q&A pairs from demo")
    print()

    # 3. Initialize TELOS components
    print("Initializing TELOS components...")

    # Check for Mistral API key
    if not os.getenv('MISTRAL_API_KEY'):
        print("❌ Error: MISTRAL_API_KEY environment variable not set")
        print("   Set it using: export MISTRAL_API_KEY='your-key-here'")
        return None

    try:
        llm_client = MistralClient(model="mistral-small-latest")
        embedding_provider = EmbeddingProvider()
        print("✓ LLM client and embeddings initialized")
        print()
    except Exception as e:
        print(f"❌ Error initializing TELOS components: {e}")
        return None

    # 4. Run observation mode analysis
    print("Running ObservationRunner on demo conversation...")
    print("(This calculates actual TELOS fidelity scores without interventions)")
    print()

    try:
        runner = ObservationRunner(
            llm_client=llm_client,
            embedding_provider=embedding_provider,
            attractor_config=attractor
        )

        result = runner.run_conversation(conversation)
        print("✓ Observation run completed")
        print()

    except Exception as e:
        print(f"❌ Error running observation: {e}")
        import traceback
        traceback.print_exc()
        return None

    # 5. Extract actual fidelity scores and compare with hardcoded values
    print("=" * 80)
    print("COMPARISON: Hardcoded vs Actual TELOS Fidelity Scores")
    print("=" * 80)
    print()

    # Hardcoded values from demo slides
    hardcoded_fidelities = [
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

    # Print comparison table
    print(f"{'Turn':<6} {'Slide Description':<25} {'Hardcoded F':<14} {'Actual F':<14} {'Delta':<10}")
    print("-" * 80)

    actual_fidelities = []
    for i, turn_result in enumerate(result.turn_results):
        turn_num = turn_result["turn"]

        # Calculate fidelity from distance
        # Note: This is a simplified calculation. The actual fidelity calculation
        # in ObservationRunner uses basin membership and normalized distance.
        distance = turn_result["distance_to_attractor"]
        in_basin = turn_result["in_basin"]

        # Approximate fidelity (matches ObservationRunner's logic)
        # For observation mode, we're measuring against user PA only
        fidelity = 1.0 if in_basin else max(0.0, 1.0 - (distance / 2.0))

        actual_fidelities.append({
            "turn": turn_num,
            "fidelity": fidelity,
            "distance": distance,
            "in_basin": in_basin
        })

        # Get hardcoded value for comparison
        hardcoded = hardcoded_fidelities[i] if i < len(hardcoded_fidelities) else None
        hardcoded_f = hardcoded["user_f"] if hardcoded else None

        if hardcoded_f is not None:
            delta = fidelity - hardcoded_f
            delta_str = f"{delta:+.3f}"

            # Color code delta
            if abs(delta) < 0.05:
                status = "✓ Close"
            elif abs(delta) < 0.15:
                status = "⚠ Moderate"
            else:
                status = "⚠ Large"

            slide_desc = hardcoded["slide"][:23]
            print(f"{turn_num:<6} {slide_desc:<25} {hardcoded_f:<14.3f} {fidelity:<14.3f} {delta_str:<10} {status}")
        else:
            print(f"{turn_num:<6} {'N/A':<25} {'N/A':<14} {fidelity:<14.3f} {'N/A':<10}")

    print()
    print("=" * 80)
    print("FINAL METRICS")
    print("=" * 80)
    print(f"Final Fidelity (Actual): {result.final_metrics['fidelity']:.3f}")
    print(f"Average Distance: {result.final_metrics['avg_distance']:.3f}")
    print(f"Basin Adherence: {result.final_metrics['basin_adherence']:.1%}")
    print(f"Would-Intervene Count: {result.metadata.get('would_intervene_count', 0)}")
    print()

    # Save results
    output_file = Path("demo_fidelity_analysis.json")
    output_data = {
        "hardcoded_fidelities": hardcoded_fidelities,
        "actual_fidelities": actual_fidelities,
        "final_metrics": result.final_metrics,
        "metadata": result.metadata
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"✓ Full analysis saved to {output_file}")
    print()

    return output_data


if __name__ == "__main__":
    analyze_demo_conversation()
