"""
Test Supabase connection and check database status.
"""
import sys
from pathlib import Path

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

# Mock streamlit
if not hasattr(st, 'secrets'):
    st.secrets = MockSecrets()

print("=" * 60)
print("SUPABASE CONNECTION TEST")
print("=" * 60)

# Initialize service
supabase = get_supabase_service()

print(f"\n✓ Service initialized")
print(f"  Enabled: {supabase.enabled}")

if not supabase.enabled:
    print("\n❌ Supabase is not enabled!")
    sys.exit(1)

# Test connection
print(f"\nTesting connection...")
connection_ok = supabase.test_connection()

if not connection_ok:
    print("❌ Connection test failed")
    sys.exit(1)

# Check tables
print("\n" + "=" * 60)
print("CHECKING TABLES")
print("=" * 60)

tables = ['governance_deltas', 'session_summaries', 'beta_consent_log', 'primacy_attractor_configs']

for table in tables:
    try:
        result = supabase.client.table(table).select('*').limit(1).execute()
        print(f"\n✓ Table '{table}' exists")
        print(f"  Columns: {list(result.data[0].keys()) if result.data else 'No data yet'}")
    except Exception as e:
        print(f"\n❌ Table '{table}' error: {e}")

# Check for existing data
print("\n" + "=" * 60)
print("CHECKING EXISTING DATA")
print("=" * 60)

for table in tables:
    try:
        result = supabase.client.table(table).select('*', count='exact').limit(0).execute()
        count = result.count if hasattr(result, 'count') else 0
        print(f"\n{table}: {count} records")
    except Exception as e:
        print(f"\n{table}: Error - {e}")

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)
