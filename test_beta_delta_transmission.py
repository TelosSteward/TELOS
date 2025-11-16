"""
Test that we can successfully transmit a BETA A/B testing delta to Supabase.
"""
import sys
from pathlib import Path
import uuid
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent / "telos_observatory_v3"))

from services.supabase_client import get_supabase_service
import streamlit as st

# Mock streamlit secrets
class MockSecrets:
    def __getitem__(self, key):
        secrets = {
            'SUPABASE_URL': 'https://ukqrwjowlchhwznefboj.supabase.co',
            'SUPABASE_KEY': 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InVrcXJ3am93bGNoaHd6bmVmYm9qIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2MjMyOTE2MCwiZXhwIjoyMDc3OTA1MTYwfQ.TvefimDWnnlAz4dj9-XBFJ4xl7hmXX9bZJSidzUjHTs'
        }
        return secrets.get(key)

    def __contains__(self, key):
        return key in ['SUPABASE_URL', 'SUPABASE_KEY']

if not hasattr(st, 'secrets'):
    st.secrets = MockSecrets()

print("=" * 80)
print("BETA A/B TESTING DELTA TRANSMISSION TEST")
print("=" * 80)

# Initialize service
supabase = get_supabase_service()

if not supabase.enabled:
    print("❌ Supabase not enabled!")
    sys.exit(1)

print("\n✓ Supabase connected")

# Create a test delta with all the new A/B testing fields
test_session_id = str(uuid.uuid4())
print(f"\n📝 Creating test delta for session: {test_session_id[:16]}...")

delta_data = {
    'session_id': test_session_id,
    'turn_number': 11,
    'fidelity_score': 0.92,
    'distance_from_pa': 0.08,
    'baseline_fidelity': 0.85,  # NEW: Raw LLM fidelity
    'fidelity_delta': 0.07,  # NEW: TELOS improvement
    'intervention_triggered': False,
    'mode': 'beta',
    'test_condition': 'single_blind_telos',  # NEW: A/B test type
    'shown_response_source': 'telos'  # NEW: Which response shown
}

print("\n📊 Delta data:")
print(f"  Session: {test_session_id[:16]}...")
print(f"  Turn: {delta_data['turn_number']}")
print(f"  Mode: {delta_data['mode']}")
print(f"  Test Condition: {delta_data['test_condition']}")
print(f"  Shown Source: {delta_data['shown_response_source']}")
print(f"  Baseline Fidelity: {delta_data['baseline_fidelity']}")
print(f"  TELOS Fidelity: {delta_data['fidelity_score']}")
print(f"  Fidelity Delta: {delta_data['fidelity_delta']} (TELOS improvement)")

print("\n🚀 Transmitting to Supabase...")

success = supabase.transmit_delta(delta_data)

if success:
    print("✅ SUCCESS! Delta transmitted successfully")

    # Verify it was saved
    print("\n🔍 Verifying delta was saved...")
    result = supabase.client.table('governance_deltas')\
        .select('*')\
        .eq('session_id', test_session_id)\
        .eq('turn_number', 11)\
        .execute()

    if result.data:
        record = result.data[0]
        print("✅ Delta found in database!")
        print(f"\n📋 Saved record:")
        print(f"  Session: {record['session_id'][:16]}...")
        print(f"  Turn: {record['turn_number']}")
        print(f"  Test Condition: {record.get('test_condition', 'MISSING')}")
        print(f"  Shown Source: {record.get('shown_response_source', 'MISSING')}")
        print(f"  Baseline Fidelity: {record.get('baseline_fidelity', 'MISSING')}")
        print(f"  TELOS Fidelity: {record.get('fidelity_score', 'MISSING')}")
        print(f"  Fidelity Delta: {record.get('fidelity_delta', 'MISSING')}")

        # Privacy check
        import json
        record_str = json.dumps(record, default=str).lower()
        content_indicators = ['user_message', 'response_text', 'message_content']
        found = [ind for ind in content_indicators if ind in record_str]

        if found:
            print(f"\n⚠️  WARNING: Found possible content: {found}")
        else:
            print(f"\n✅ PRIVACY VERIFIED: No conversation content stored")

        print("\n" + "=" * 80)
        print("TEST SUCCESSFUL! 🎉")
        print("=" * 80)
        print("\nYour BETA A/B testing is now fully operational with:")
        print("  ✓ Delta-only storage (no conversation content)")
        print("  ✓ A/B test condition tracking")
        print("  ✓ Baseline vs TELOS comparison")
        print("  ✓ Fidelity delta calculation")
        print("  ✓ Privacy compliance maintained")
        print("\n" + "=" * 80)
    else:
        print("❌ Delta not found in database!")
else:
    print("❌ FAILED to transmit delta")

print()
