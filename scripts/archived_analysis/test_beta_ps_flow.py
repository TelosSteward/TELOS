#!/usr/bin/env python3
"""
Test script to verify PS data flows to Supabase in beta mode.
This simulates a beta test session with multiple turns.
"""

import sys
import os
import time
from datetime import datetime

# Add path for imports
sys.path.insert(0, '/Users/brunnerjf/Desktop/Privacy_PreCommit/TELOSCOPE_BETA')

def test_ps_flow():
    """Test that PS metrics are being sent to Supabase."""

    print("=" * 60)
    print("PS FLOW TO SUPABASE TEST")
    print("=" * 60)

    try:
        # Import the Supabase client
        from services.supabase_client import get_supabase_service

        # Get Supabase service
        supabase = get_supabase_service()

        if not supabase.enabled:
            print("❌ Supabase is not enabled!")
            return False

        # Create a test session
        import uuid
        test_session_id = str(uuid.uuid4())
        print(f"\n1. Created test session: {test_session_id}")

        # Log consent (required for beta)
        if supabase.log_consent(
            session_id=test_session_id,
            consent_statement="Test consent for PS flow testing",
            consent_version="ps_1.0"
        ):
            print("✓ Consent logged successfully")
        else:
            print("❌ Failed to log consent")
            return False

        # Create test delta with PS metrics
        print("\n2. Sending test delta with PS metrics...")

        test_delta = {
            'session_id': test_session_id,
            'turn_number': 1,
            'fidelity_score': 0.85,
            'distance_from_pa': 0.15,
            'mode': 'beta',
            'test_condition': 'single_blind_telos',
            # PS metrics (matching our SQL schema)
            'primacy_state_score': 0.73,
            'primacy_state_condition': 'achieved',
            'user_pa_fidelity': 0.82,
            'ai_pa_fidelity': 0.79,
            'pa_correlation': 0.65,
            'v_dual_energy': 0.42,
            'delta_v_dual': -0.03,
            'primacy_converging': True
        }

        if supabase.transmit_delta(test_delta):
            print("✅ PS delta transmitted successfully!")
        else:
            print("❌ Failed to transmit PS delta")
            return False

        # Send a few more turns with varying PS scores
        ps_test_cases = [
            {'ps_score': 0.65, 'condition': 'achieved', 'f_user': 0.75, 'f_ai': 0.71},
            {'ps_score': 0.45, 'condition': 'weakening', 'f_user': 0.55, 'f_ai': 0.62},
            {'ps_score': 0.28, 'condition': 'violated', 'f_user': 0.42, 'f_ai': 0.38},
            {'ps_score': 0.81, 'condition': 'achieved', 'f_user': 0.88, 'f_ai': 0.85}
        ]

        print("\n3. Sending multiple turns with varying PS conditions...")
        for i, ps_case in enumerate(ps_test_cases, start=2):
            test_delta = {
                'session_id': test_session_id,
                'turn_number': i,
                'fidelity_score': ps_case['f_user'],  # Use f_user as proxy
                'distance_from_pa': 1.0 - ps_case['f_user'],
                'mode': 'beta',
                'primacy_state_score': ps_case['ps_score'],
                'primacy_state_condition': ps_case['condition'],
                'user_pa_fidelity': ps_case['f_user'],
                'ai_pa_fidelity': ps_case['f_ai'],
                'pa_correlation': 0.65,
                'v_dual_energy': 0.5 + (i * 0.1),
                'delta_v_dual': 0.02 if i % 2 == 0 else -0.03,
                'primacy_converging': i % 2 == 1
            }

            if supabase.transmit_delta(test_delta):
                print(f"  ✓ Turn {i}: PS={ps_case['ps_score']:.2f} ({ps_case['condition']})")
            else:
                print(f"  ❌ Turn {i} failed")

            time.sleep(0.5)  # Small delay between turns

        print("\n" + "=" * 60)
        print("TEST COMPLETE!")
        print("=" * 60)
        print(f"\nSession ID: {test_session_id}")
        print("Check Supabase governance_deltas table for:")
        print("  - 5 new rows with this session_id")
        print("  - All 8 PS columns should have data")
        print("  - Conditions: 2 achieved, 1 weakening, 1 violated")

        return True

    except Exception as e:
        print(f"\n❌ Error during PS flow test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_ps_flow()
    if success:
        print("\n✅ PS metrics should now be visible in Supabase!")
    else:
        print("\n❌ Test failed - check error messages above")

    sys.exit(0 if success else 1)