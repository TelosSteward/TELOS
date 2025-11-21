#!/usr/bin/env python3
"""
Quick status check for validation pipeline.
Shows recent sessions and their completion status.
"""

import os
os.environ['SUPABASE_URL'] = 'https://ukqrwjowlchhwznefboj.supabase.co'
os.environ['SUPABASE_KEY'] = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InVrcXJ3am93bGNoaHd6bmVmYm9qIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2MjMyOTE2MCwiZXhwIjoyMDc3OTA1MTYwfQ.TvefimDWnnlAz4dj9-XBFJ4xl7hmXX9bZJSidzUjHTs'

from telos_purpose.storage.validation_storage import ValidationStorage
from datetime import datetime

storage = ValidationStorage()

print("=" * 80)
print("VALIDATION PIPELINE STATUS")
print("=" * 80)
print()

# Get recent sessions
sessions = storage.query_sessions(limit=10)

if not sessions:
    print("No validation sessions found yet.")
else:
    print(f"Total sessions: {len(sessions)}")
    print()

    for session in sessions:
        print(f"Study: {session['validation_study_name']}")
        print(f"  Session ID: {session['session_id']}")
        print(f"  Created: {session['created_at']}")
        print(f"  Model: {session['model_used']}")
        print(f"  Turns: {session['total_turns']}")
        print(f"  Avg Fidelity: {session.get('avg_fidelity', 'N/A')}")
        print(f"  Completed: {'Yes' if session.get('completed_at') else 'No'}")

        # Get IP proof
        try:
            ip_proof = storage.get_ip_proof(session['session_id'])
            if ip_proof:
                print(f"  Signed turns: {ip_proof['signed_turns']}/{ip_proof['total_turns']}")
                print(f"  Signature chain: {len(ip_proof['signature_chain'])} signatures")
        except:
            pass

        print()

print("=" * 80)
