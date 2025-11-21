"""
Inspect what data is currently in Supabase.
"""
import sys
from pathlib import Path
import json

# Add project to path
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

supabase = get_supabase_service()

print("=" * 80)
print("GOVERNANCE DELTAS")
print("=" * 80)

# Get all governance deltas
result = supabase.client.table('governance_deltas').select('*').order('created_at', desc=True).limit(10).execute()

if result.data:
    for record in result.data:
        print(f"\nSession: {record['session_id'][:16]}... | Turn: {record['turn_number']} | Mode: {record.get('mode', 'N/A')}")
        print(f"  Fidelity: {record.get('fidelity_score', 'N/A')}")
        print(f"  Distance: {record.get('distance_from_pa', 'N/A')}")
        print(f"  Intervention: {record.get('intervention_triggered', False)}")
        print(f"  Status: {record.get('turn_status', 'N/A')} | Stage: {record.get('processing_stage', 'N/A')}")
        print(f"  Created: {record.get('created_at', 'N/A')}")

        # Check for ANY conversation content (shouldn't be any)
        if any(key in str(record).lower() for key in ['user_message', 'response_text', 'message_content']):
            print(f"  ⚠️  WARNING: May contain conversation content!")
else:
    print("No governance deltas found")

print("\n" + "=" * 80)
print("SESSION SUMMARIES")
print("=" * 80)

result = supabase.client.table('session_summaries').select('*').order('created_at', desc=True).limit(5).execute()

if result.data:
    for record in result.data:
        print(f"\nSession: {record['session_id'][:16]}... | Mode: {record.get('mode', 'N/A')}")
        print(f"  Total Turns: {record.get('total_turns', 'N/A')}")
        print(f"  Avg Fidelity: {record.get('avg_fidelity_score', 'N/A')}")
        intervention_rate = record.get('intervention_rate') or 0
        print(f"  Interventions: {record.get('total_interventions', 'N/A')} ({intervention_rate*100:.1f}%)")
        print(f"  Beta Consent: {record.get('beta_consent_given', False)}")
else:
    print("No session summaries found")

print("\n" + "=" * 80)
print("BETA CONSENT LOG")
print("=" * 80)

result = supabase.client.table('beta_consent_log').select('session_id,consent_timestamp,consent_version').order('consent_timestamp', desc=True).limit(5).execute()

if result.data:
    for record in result.data:
        print(f"  {record['consent_timestamp']} | Session: {record['session_id'][:16]}... | Version: {record.get('consent_version', 'N/A')}")
else:
    print("No consent logs found")

print("\n" + "=" * 80)
print("PRIVACY CHECK: Looking for conversation content...")
print("=" * 80)

# Check ALL tables for any conversation content
tables = ['governance_deltas', 'session_summaries', 'beta_consent_log', 'primacy_attractor_configs']
content_found = False

for table in tables:
    result = supabase.client.table(table).select('*').limit(100).execute()

    # Convert all records to string and check for suspicious patterns
    all_data = json.dumps(result.data, default=str)

    # Patterns that would indicate conversation content
    suspicious_patterns = [
        '"user_input":', '"response":', '"message":',
        '"user_message":', '"assistant_message":',
        '"response_text":', '"response_a_text":', '"response_b_text":'
    ]

    found_patterns = [p for p in suspicious_patterns if p in all_data]

    if found_patterns:
        print(f"\n⚠️  {table}: Found suspicious fields: {found_patterns}")
        content_found = True

if not content_found:
    print("\n✅ NO CONVERSATION CONTENT FOUND - Privacy claim validated!")
else:
    print("\n⚠️  WARNING: Possible conversation content detected!")

print("\n" + "=" * 80)
