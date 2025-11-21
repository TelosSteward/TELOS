#!/usr/bin/env python3
"""
Steward Interprets Demo Drift Detections
=========================================

Uses Steward's LLM reasoning to explain WHY drift was detected mathematically.
"""

import sys
from pathlib import Path

# Add TELOSCOPE_BETA to path
beta_path = Path("/Users/brunnerjf/Desktop/Privacy_PreCommit/TELOSCOPE_BETA")
sys.path.insert(0, str(beta_path))

from telos_purpose.llm_clients.mistral_client import MistralClient
import json


def load_analysis_results():
    """Load the mathematical analysis results."""
    results_file = Path("demo_analysis_tolerance_05.json")
    if not results_file.exists():
        print("❌ Error: demo_analysis_tolerance_05.json not found")
        print("   Run analyze_demo_tolerance_05.py first")
        return None

    with open(results_file, 'r') as f:
        return json.load(f)


def ask_steward_to_interpret(client, turn_data, pa_config):
    """Ask Steward to interpret why drift was detected for a specific turn."""

    question = turn_data['question']
    user_f = turn_data['user_fidelity']
    distance = turn_data['user_distance']
    in_basin = turn_data['user_in_basin']
    basin_radius = turn_data.get('basin_radius', 1.053)

    # User's established PA from Turn 1
    user_pa = "I want to understand TELOS but without overwhelming technical details. Can you help?"

    prompt = f"""You are Steward, an expert in TELOS governance mathematics and semantic interpretation.

CONTEXT:
A user established their Primacy Attractor (PA) with this statement:
"{user_pa}"

This creates a governance boundary with:
- Basin radius: {basin_radius:.3f}
- Anything within this distance is "aligned"
- Anything beyond is "drift"

MATHEMATICAL RESULT:
The user then asked: "{question}"

TELOS calculated:
- Semantic distance from PA: {distance:.3f}
- Basin radius: {basin_radius:.3f}
- In basin: {in_basin}
- Fidelity score: {user_f:.3f}

YOUR TASK:
Explain in 2-3 sentences WHY this question {"stayed aligned" if in_basin else "drifted"} based on semantic analysis.

Consider:
1. Does the question serve the stated purpose ("understand TELOS")?
2. Does it respect the boundary ("without overwhelming technical details")?
3. What semantic elements caused the distance measurement?

Be specific about the semantic reasoning, not just the numbers."""

    try:
        response = client.generate(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=300
        )
        return response
    except Exception as e:
        return f"[Steward unavailable: {e}]"


def main():
    """Run Steward's interpretation of all drift detections."""

    print("=" * 80)
    print("STEWARD'S SEMANTIC INTERPRETATION OF DRIFT DETECTIONS")
    print("=" * 80)
    print()

    # Load analysis results
    print("Loading mathematical analysis results...")
    data = load_analysis_results()
    if not data:
        return

    print("✓ Loaded results")
    print()

    # Initialize Mistral client
    print("Initializing Steward (Mistral)...")
    try:
        client = MistralClient(model="mistral-small-latest")
        print("✓ Steward online")
        print()
    except Exception as e:
        print(f"❌ Error: {e}")
        return

    # Get results
    actual_results = data['actual_results']
    basin_radius = data['user_basin_radius']

    # Add basin radius to each result
    for result in actual_results:
        result['basin_radius'] = basin_radius

    # Focus on key turns
    key_turns = [
        (2, "Show Observation Deck - Flagged as drift"),
        (3, "Perfect alignment - Flagged as drift"),
        (5, "Quantum physics - Correctly flagged"),
        (7, "Math details - Flagged as drift"),
    ]

    print("=" * 80)
    print("STEWARD'S ANALYSIS")
    print("=" * 80)
    print()

    for turn_num, description in key_turns:
        turn_data = actual_results[turn_num - 1]  # 0-indexed

        print(f"Turn {turn_num}: {description}")
        print(f"Question: \"{turn_data['question'][:70]}...\"")
        print(f"Mathematical Result: Fidelity = {turn_data['user_fidelity']:.3f}, Distance = {turn_data['user_distance']:.3f}, In Basin = {turn_data['user_in_basin']}")
        print()
        print("Steward's Interpretation:")

        interpretation = ask_steward_to_interpret(client, turn_data, data)
        print(f"  {interpretation}")
        print()
        print("-" * 80)
        print()

    print("=" * 80)
    print("SUMMARY QUESTION FOR STEWARD")
    print("=" * 80)
    print()

    # Ask Steward to summarize the overall pattern
    summary_prompt = f"""You are Steward, analyzing TELOS governance results.

The user established: "I want to understand TELOS without overwhelming technical details"
Basin radius: {basin_radius:.3f}

Results:
- Turn 2: "How can I see my PA?" → Drift (distance 1.427)
- Turn 3: "Why are fidelities 1.000?" → Drift (distance 1.304)
- Turn 4: "How does TELOS detect drift?" → Aligned (distance 1.009)
- Turn 5: "Can you explain quantum physics?" → Drift (distance 1.373)
- Turn 6: "How does TELOS track fidelities?" → Aligned (distance 0.998)
- Turn 7: "What's the math behind fidelity?" → Drift (distance 1.361)

In 2-3 sentences, explain the PATTERN: Why do some TELOS questions drift while others stay aligned?
What semantic principle explains this?"""

    print("Question: What's the semantic pattern here?")
    print()
    summary = ask_steward_to_interpret(client, {"question": "pattern analysis"}, None) if False else client.generate(
        messages=[{"role": "user", "content": summary_prompt}],
        temperature=0.7,
        max_tokens=300
    )
    print(f"Steward: {summary}")
    print()


if __name__ == "__main__":
    main()
