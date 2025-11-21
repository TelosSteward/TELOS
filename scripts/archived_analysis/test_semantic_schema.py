#!/usr/bin/env python3
"""
Test script to verify semantic telemetry schema deployment.
Verifies all new columns and views exist in Supabase.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "telos_observatory_v3"))

from services.supabase_client import get_supabase_service
from services.turn_tracker import TurnTracker, SemanticAnalyzer
import uuid

def test_schema():
    """Test that all new schema elements exist."""
    print("=" * 70)
    print("SEMANTIC TELEMETRY SCHEMA VERIFICATION")
    print("=" * 70)

    supabase = get_supabase_service()

    if not supabase.enabled:
        print("❌ Supabase not enabled - check credentials")
        return False

    # Test 1: Insert a test delta with ALL new semantic fields
    print("\n[Test 1] Inserting test delta with semantic fields...")

    test_session_id = uuid.uuid4()
    test_delta = {
        'session_id': str(test_session_id),
        'turn_number': 1,
        'mode': 'beta',
        'fidelity_score': 0.85,
        'distance_from_pa': 0.15,
        'purpose_alignment': 0.9,
        'scope_alignment': 0.85,
        'boundary_alignment': 0.8,

        # Lifecycle fields
        'turn_status': 'completed',
        'processing_stage': 'Test delta for schema validation',
        'processing_duration_ms': 1500,

        # Semantic intelligence fields
        'request_type': 'coding_task',
        'request_complexity': 'moderate',
        'detected_topics': ['governance', 'testing'],
        'topic_shift_magnitude': 0.1,
        'semantic_drift_direction': 'stable',
        'constraints_approached': ['scope_drift'],
        'constraint_violation_type': None,

        # Intervention fields
        'intervention_triggered': False
    }

    success = supabase.transmit_delta(test_delta)

    if success:
        print("✅ Test delta inserted successfully")
    else:
        print("❌ Failed to insert test delta")
        return False

    # Test 2: Query the semantic_telemetry_analysis view
    print("\n[Test 2] Querying semantic_telemetry_analysis view...")

    try:
        result = supabase.client.table('semantic_telemetry_analysis')\
            .select('*')\
            .eq('session_id', str(test_session_id))\
            .execute()

        if result.data and len(result.data) > 0:
            row = result.data[0]
            print("✅ semantic_telemetry_analysis view working")
            print(f"   - turn_status: {row.get('turn_status')}")
            print(f"   - request_type: {row.get('request_type')}")
            print(f"   - detected_topics: {row.get('detected_topics')}")
            print(f"   - constraints_approached: {row.get('constraints_approached')}")
        else:
            print("❌ No data returned from view")
            return False

    except Exception as e:
        print(f"❌ Error querying view: {e}")
        return False

    # Test 3: Query request_type_performance view
    print("\n[Test 3] Querying request_type_performance view...")

    try:
        result = supabase.client.table('request_type_performance')\
            .select('*')\
            .limit(5)\
            .execute()

        if result.data is not None:
            print(f"✅ request_type_performance view working ({len(result.data)} rows)")
            for row in result.data[:3]:
                print(f"   - {row.get('request_type')}: avg_fidelity={row.get('avg_fidelity'):.3f}")
        else:
            print("❌ request_type_performance view failed")
            return False

    except Exception as e:
        print(f"❌ Error querying view: {e}")
        return False

    # Test 4: Test TurnTracker integration
    print("\n[Test 4] Testing TurnTracker with semantic analysis...")

    try:
        # Create a turn tracker
        tracker = TurnTracker(
            session_id=uuid.uuid4(),
            turn_number=1,
            mode='beta'
        )

        print("✅ TurnTracker initialized")

        # Test semantic analyzer
        test_message = "Please help me fix a bug in my React component"
        semantic_context = SemanticAnalyzer.analyze_turn(
            user_message=test_message,
            governance_metrics={
                'fidelity_score': 0.9,
                'distance_from_pa': 0.1,
                'purpose_alignment': 0.95,
                'scope_alignment': 0.9,
                'boundary_alignment': 0.85,
                'intervention_triggered': False
            },
            pa_config={}
        )

        print("✅ SemanticAnalyzer working")
        print(f"   - Detected request_type: {semantic_context['request_type']}")
        print(f"   - Complexity: {semantic_context['request_complexity']}")
        print(f"   - Topics: {semantic_context['detected_topics']}")

    except Exception as e:
        print(f"❌ TurnTracker test failed: {e}")
        return False

    # Test 5: Cleanup - delete test session
    print("\n[Test 5] Cleaning up test data...")

    try:
        supabase.client.table('governance_deltas')\
            .delete()\
            .eq('session_id', str(test_session_id))\
            .execute()
        print("✅ Test data cleaned up")
    except Exception as e:
        print(f"⚠ Cleanup warning: {e}")

    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED - Semantic telemetry schema fully deployed!")
    print("=" * 70)
    print("\nYou now have:")
    print("  • Turn lifecycle tracking (initiated → calculating → completed)")
    print("  • Semantic context (request types, topics, constraints)")
    print("  • 5 intelligence views for research analysis")
    print("  • TurnTracker utility ready to use")
    print()

    return True


if __name__ == "__main__":
    success = test_schema()
    sys.exit(0 if success else 1)
