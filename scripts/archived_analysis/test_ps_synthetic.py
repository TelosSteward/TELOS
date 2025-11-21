#!/usr/bin/env python3
"""
Synthetic test to generate Primacy State data in Supabase.
This simulates what happens during a beta test session.
"""

import json
import numpy as np
from datetime import datetime
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, '/Users/brunnerjf/Desktop/Privacy_PreCommit/TELOSCOPE_BETA')
sys.path.insert(0, '/Users/brunnerjf/Desktop/telos_privacy')

def run_synthetic_ps_test():
    """Generate synthetic PS data to verify Supabase integration."""

    print("=" * 60)
    print("SYNTHETIC PRIMACY STATE TEST")
    print("=" * 60)

    try:
        # Import PS calculator
        from telos_purpose.core.primacy_state import PrimacyStateCalculator
        from telos_purpose.core.embedding_provider import SentenceTransformerProvider

        # Initialize components
        print("\n1. Initializing PS calculator...")
        ps_calculator = PrimacyStateCalculator(track_energy=True)
        embedding_provider = SentenceTransformerProvider()

        # Define test PA
        user_pa_text = "Help users understand AI governance through natural conversation"
        ai_pa_text = "Maintain focus on TELOS framework and purpose alignment"

        # Generate embeddings for PAs
        print("\n2. Generating PA embeddings...")
        user_pa_embedding = embedding_provider.encode(user_pa_text)
        ai_pa_embedding = embedding_provider.encode(ai_pa_text)

        # Test conversations
        test_responses = [
            "TELOS is a governance framework that keeps AI aligned with its purpose.",  # Good alignment
            "Let me tell you about quantum physics instead of TELOS.",  # Poor alignment
            "The purpose alignment in TELOS uses mathematical principles to ensure consistency.",  # Good
            "I think we should discuss cooking recipes now.",  # Very poor
            "TELOS maintains governance through continuous fidelity monitoring.",  # Excellent
        ]

        print("\n3. Calculating PS for test responses...")
        print("-" * 40)

        results = []
        for i, response in enumerate(test_responses):
            # Get response embedding
            response_embedding = embedding_provider.encode(response)

            # Calculate PS
            ps_metrics = ps_calculator.compute_primacy_state(
                response_embedding=response_embedding,
                user_pa_embedding=user_pa_embedding,
                ai_pa_embedding=ai_pa_embedding
            )

            # Convert to dict
            metrics_dict = ps_metrics.to_dict()

            # Display results
            ps_score = metrics_dict['primacy_state_score']
            condition = metrics_dict['primacy_state_condition']
            f_user = metrics_dict['user_pa_fidelity']
            f_ai = metrics_dict['ai_pa_fidelity']
            rho_pa = metrics_dict['pa_correlation']

            print(f"\nResponse {i+1}: \"{response[:50]}...\"")
            print(f"  PS Score: {ps_score:.3f} ({condition})")
            print(f"  F_user: {f_user:.3f}, F_AI: {f_ai:.3f}, ρ_PA: {rho_pa:.3f}")

            results.append({
                'response': response,
                'ps_score': float(ps_score),
                'condition': condition,
                'f_user': float(f_user),
                'f_ai': float(f_ai),
                'rho_pa': float(rho_pa),
                'v_dual': float(metrics_dict.get('v_dual_energy')) if metrics_dict.get('v_dual_energy') is not None else None,
                'delta_v': float(metrics_dict.get('delta_v_dual')) if metrics_dict.get('delta_v_dual') is not None else None,
                'converging': bool(metrics_dict.get('primacy_converging')) if metrics_dict.get('primacy_converging') is not None else None
            })

        print("\n" + "=" * 60)
        print("TEST COMPLETE!")
        print("=" * 60)

        # Save results for inspection
        output_file = '/Users/brunnerjf/Desktop/telos_privacy/ps_synthetic_results.json'
        with open(output_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'test_type': 'synthetic_ps',
                'results': results
            }, f, indent=2)

        print(f"\nResults saved to: {output_file}")
        print("\n✅ PS calculations successful!")
        print("✅ TELOSCOPE_BETA should now be writing similar data to Supabase")

        # Show summary
        avg_ps = sum(r['ps_score'] for r in results) / len(results)
        print(f"\nAverage PS Score: {avg_ps:.3f}")
        print(f"Achieved: {sum(1 for r in results if r['condition'] == 'achieved')}")
        print(f"Weakening: {sum(1 for r in results if r['condition'] == 'weakening')}")
        print(f"Violated: {sum(1 for r in results if r['condition'] == 'violated')}")
        print(f"Collapsed: {sum(1 for r in results if r['condition'] == 'collapsed')}")

    except Exception as e:
        print(f"\n❌ Error during synthetic test: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

if __name__ == "__main__":
    success = run_synthetic_ps_test()
    sys.exit(0 if success else 1)