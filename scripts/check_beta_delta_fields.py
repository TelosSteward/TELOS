"""
Check if BETA mode delta transmission includes all required fields.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "telos_observatory_v3"))

from services.supabase_client import get_supabase_service
import streamlit as st
import json

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

supabase = get_supabase_service()

print("=" * 80)
print("CHECKING BETA MODE DELTAS")
print("=" * 80)

# Get BETA mode records
result = supabase.client.table('governance_deltas')\
    .select('*')\
    .eq('mode', 'beta')\
    .order('created_at', desc=True)\
    .limit(5)\
    .execute()

if result.data:
    print(f"\nFound {len(result.data)} BETA mode records\n")

    for record in result.data:
        print("=" * 80)
        print(f"Session: {record['session_id']}")
        print(f"Turn: {record['turn_number']} | Created: {record.get('created_at')}")
        print("-" * 80)

        # New delta-only fields we just added
        delta_fields = {
            'fidelity_score': record.get('fidelity_score'),
            'distance_from_pa': record.get('distance_from_pa'),
            'baseline_fidelity': record.get('baseline_fidelity'),  # NEW
            'fidelity_delta': record.get('delta_from_previous'),  # Should be baseline vs telos delta
            'intervention_triggered': record.get('intervention_triggered'),
            'mode': record.get('mode'),
        }

        # Check for A/B testing specific fields (not in current schema)
        ab_fields = ['test_condition', 'shown_response_source', 'baseline_fidelity']

        print("\n✓ DELTA FIELDS:")
        for field, value in delta_fields.items():
            status = "✓" if value is not None else "✗"
            print(f"  {status} {field}: {value}")

        print("\n⚠ A/B TESTING FIELDS (may need schema update):")
        for field in ab_fields:
            value = record.get(field)
            status = "✓" if value is not None else "✗"
            print(f"  {status} {field}: {value}")

        # CRITICAL: Check for any conversation content
        record_str = json.dumps(record, default=str).lower()
        content_indicators = ['user_message', 'response_text', 'response_a', 'response_b', 'message_content']
        found_content = [ind for ind in content_indicators if ind in record_str]

        if found_content:
            print(f"\n❌ WARNING: Possible content fields: {found_content}")
        else:
            print(f"\n✅ NO CONVERSATION CONTENT")

        print()
else:
    print("\n⚠ No BETA mode records found yet")
    print("\nThis is expected if:")
    print("  1. Beta testing hasn't started yet")
    print("  2. Code changes haven't been deployed")
    print("  3. No user has completed turn 11+ in BETA mode")

print("\n" + "=" * 80)
print("CHECKING governance_deltas SCHEMA")
print("=" * 80)

# Get one record to see all available columns
result = supabase.client.table('governance_deltas').select('*').limit(1).execute()

if result.data:
    columns = list(result.data[0].keys())
    print(f"\nAvailable columns ({len(columns)}):")

    # Group by category
    required_for_beta = ['session_id', 'turn_number', 'fidelity_score', 'distance_from_pa']
    ab_testing_fields = ['test_condition', 'shown_response_source', 'baseline_fidelity', 'fidelity_delta']

    print("\n✓ REQUIRED FIELDS:")
    for col in required_for_beta:
        status = "✓" if col in columns else "✗ MISSING"
        print(f"  {status} {col}")

    print("\n⚠ A/B TESTING FIELDS (for new beta implementation):")
    for col in ab_testing_fields:
        status = "✓" if col in columns else "✗ MISSING (needs schema update)"
        print(f"  {status} {col}")

    print(f"\nAll columns:\n  {', '.join(sorted(columns))}")

print("\n" + "=" * 80)
print("RECOMMENDATION")
print("=" * 80)

print("""
The governance_deltas table may need additional columns for full A/B testing:

If missing, run this SQL in Supabase:

  ALTER TABLE governance_deltas
  ADD COLUMN IF NOT EXISTS test_condition TEXT,
  ADD COLUMN IF NOT EXISTS shown_response_source TEXT,
  ADD COLUMN IF NOT EXISTS baseline_fidelity FLOAT8,
  ADD COLUMN IF NOT EXISTS fidelity_delta FLOAT8;

  -- Add index for filtering beta test data
  CREATE INDEX IF NOT EXISTS idx_governance_deltas_test_condition
  ON governance_deltas(test_condition);

This allows storing:
  - test_condition: "single_blind_baseline" | "single_blind_telos" | "head_to_head"
  - shown_response_source: "baseline" | "telos"
  - baseline_fidelity: Raw LLM fidelity score
  - fidelity_delta: telos_fidelity - baseline_fidelity

All without storing any conversation content!
""")

print("=" * 80)
