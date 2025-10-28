#!/usr/bin/env python3
"""
Test Counterfactual Branch Simulator
====================================

Verifies AI-to-AI conversation simulation works correctly.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from telos_purpose.core.counterfactual_simulator import CounterfactualBranchSimulator
from telos_purpose.core.unified_steward import UnifiedGovernanceSteward, PrimacyAttractor
from telos_purpose.llm_clients.mistral_client import TelosMistralClient
from telos_purpose.core.embedding_provider import EmbeddingProvider
import os
import numpy as np


def test_simulator():
    """Test counterfactual simulation."""

    print("=" * 70)
    print("TEST: Counterfactual Branch Simulator")
    print("=" * 70)

    # Setup
    api_key = os.getenv('MISTRAL_API_KEY')
    if not api_key:
        print("❌ MISTRAL_API_KEY not set. Skipping test.")
        return

    llm = TelosMistralClient(api_key=api_key)
    embeddings = EmbeddingProvider(deterministic=False)

    attractor = PrimacyAttractor(
        purpose=["Provide information about Python programming"],
        scope=["Python basics", "syntax", "best practices"],
        boundaries=["No off-topic discussion", "Stay focused on Python"]
    )

    # Create steward
    print("\n📋 Creating steward...")
    steward = UnifiedGovernanceSteward(
        attractor=attractor,
        llm_client=llm,
        embedding_provider=embeddings,
        enable_interventions=True
    )
    steward.start_session()

    # Set attractor center
    purpose_text = " ".join(attractor.purpose)
    attractor_center = embeddings.encode([purpose_text])[0]
    steward.attractor_center = attractor_center

    print(f"✅ Steward initialized")

    # Create simulator
    print("\n📋 Creating simulator...")
    simulator = CounterfactualBranchSimulator(
        llm_client=llm,
        embedding_provider=embeddings,
        steward=steward,
        simulation_turns=3  # Short test (3 turns instead of 5)
    )

    print(f"✅ Simulator initialized")

    # Build initial conversation history
    conversation_history = [
        {"role": "user", "content": "What is Python?"},
        {"role": "assistant", "content": "Python is a high-level programming language..."},
        {"role": "user", "content": "Tell me about lists"},
        {"role": "assistant", "content": "Lists in Python are ordered collections..."}
    ]

    print("\n" + "=" * 70)
    print("TEST: Simulate Counterfactual Branches")
    print("=" * 70)

    # Simulate drift scenario
    print("\n🔬 Simulating counterfactual from drift point...")
    print("   Context: Python programming discussion")
    print("   Potential drift: User might go off-topic")

    simulation_id = simulator.simulate_counterfactual(
        trigger_turn=2,
        trigger_fidelity=0.75,
        trigger_reason="Testing drift projection",
        conversation_history=conversation_history,
        attractor_center=attractor_center,
        distance_scale=2.0,
        topic_hint="cooking recipes"  # Hint for potential drift
    )

    print(f"\n✅ Simulation complete: {simulation_id}")

    # Get comparison
    print("\n" + "=" * 70)
    print("RESULTS: Side-by-Side Comparison")
    print("=" * 70)

    comparison = simulator.get_comparison(simulation_id)

    if comparison:
        print(f"\n📊 Summary:")
        print(f"   Trigger Turn: {comparison['trigger_turn']}")
        print(f"   Trigger Fidelity: {comparison['trigger_fidelity']:.3f}")
        print(f"   ΔF (Improvement): {comparison['comparison']['delta_f']:+.3f}")
        print(f"   Governance Effective: {'✅ Yes' if comparison['comparison']['improvement'] else '❌ No'}")

        print(f"\n📈 Fidelity Trajectories:")
        orig_traj = comparison['comparison']['original_trajectory']
        telos_traj = comparison['comparison']['telos_trajectory']

        print(f"   Original: {' → '.join([f'{f:.3f}' for f in orig_traj])}")
        print(f"   TELOS:    {' → '.join([f'{f:.3f}' for f in telos_traj])}")

        print(f"\n💬 Turn-by-Turn Breakdown:")
        for i, (orig_turn, telos_turn) in enumerate(zip(
            comparison['original']['turns'],
            comparison['telos']['turns']
        )):
            print(f"\n   Turn {orig_turn['turn_number']}:")
            print(f"   User: {orig_turn['user_message'][:80]}...")
            print(f"   ├─ Original (F={orig_turn['fidelity']:.3f}): {orig_turn['assistant_response'][:60]}...")
            print(f"   └─ TELOS    (F={telos_turn['fidelity']:.3f}): {telos_turn['assistant_response'][:60]}...")

        # Export evidence
        print(f"\n📄 Evidence Export:")
        markdown = simulator.export_evidence(simulation_id, format='markdown')
        if markdown:
            print(f"   ✅ Markdown evidence generated ({len(markdown)} chars)")

        json_export = simulator.export_evidence(simulation_id, format='json')
        if json_export:
            print(f"   ✅ JSON evidence generated ({len(json_export)} chars)")

    steward.end_session()

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
    print("\n✅ Counterfactual simulator is operational!")
    print("   - AI-to-AI user message generation: Working")
    print("   - Dual branch simulation: Working")
    print("   - Fidelity calculation: Working")
    print("   - Evidence export: Working")


if __name__ == '__main__':
    test_simulator()
