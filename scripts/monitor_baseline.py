#!/usr/bin/env python3
"""
Monitor baseline comparison progress in real-time.
"""

import os
import time
os.environ['SUPABASE_URL'] = 'https://ukqrwjowlchhwznefboj.supabase.co'
os.environ['SUPABASE_KEY'] = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InVrcXJ3am93bGNoaHd6bmVmYm9qIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2MjMyOTE2MCwiZXhwIjoyMDc3OTA1MTYwfQ.TvefimDWnnlAz4dj9-XBFJ4xl7hmXX9bZJSidzUjHTs'

from telos_purpose.storage.validation_storage import ValidationStorage

storage = ValidationStorage()

print("=" * 80)
print("BASELINE COMPARISON PROGRESS MONITOR")
print("=" * 80)
print("\nExpected: 3 governance modes × 5 turns each = 15 total turns\n")

# Query for baseline sessions
sessions = storage.client.table("validation_telemetric_sessions")\
    .select("*")\
    .like("validation_study_name", "baseline_comparison%")\
    .order("created_at", desc=False)\
    .execute()

if not sessions.data:
    print("No baseline sessions found yet.")
else:
    total_turns = 0
    for session in sessions.data:
        study_name = session['validation_study_name']
        mode = study_name.replace('baseline_comparison_', '')
        turns = session['total_turns']
        avg_fidelity = session.get('avg_fidelity', 'N/A')
        completed = 'Yes' if session.get('completed_at') else 'No'

        print(f"{mode.upper()}:")
        print(f"  Turns: {turns}/5")
        print(f"  Avg Fidelity: {avg_fidelity if avg_fidelity != 'N/A' else 'N/A':.3f}")
        print(f"  Completed: {completed}")
        print()

        total_turns += turns

    print(f"Total Progress: {total_turns}/15 turns ({total_turns/15*100:.1f}%)")
    print("=" * 80)
