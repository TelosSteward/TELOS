#!/usr/bin/env python3
"""
Test script to verify Supabase connection and delta transmission.
"""

import sys
import os
import uuid
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load secrets
import toml
secrets_path = os.path.join(os.path.dirname(__file__), '.streamlit', 'secrets.toml')
if os.path.exists(secrets_path):
    secrets = toml.load(secrets_path)
    os.environ['SUPABASE_URL'] = secrets['SUPABASE_URL']
    os.environ['SUPABASE_KEY'] = secrets['SUPABASE_KEY']

import streamlit as st
from services.supabase_client import get_supabase_service

def test_supabase():
    """Test Supabase connection and operations."""
    print("=" * 60)
    print("TELOSCOPE_BETA - Supabase Integration Test")
    print("=" * 60)

    # Mock streamlit secrets
    if not hasattr(st, 'secrets'):
        st.secrets = secrets

    # Get Supabase service
    print("\n1. Initializing Supabase service...")
    supabase = get_supabase_service()

    if not supabase.enabled:
        print("❌ Supabase service not enabled")
        return False

    # Test connection
    print("\n2. Testing connection...")
    if supabase.test_connection():
        print("✅ Connection test successful")
    else:
        print("❌ Connection test failed")
        return False

    # Test delta transmission
    print("\n3. Testing delta transmission...")
    test_session_id = uuid.uuid4()
    test_delta = {
        'session_id': str(test_session_id),
        'turn_number': 1,
        'fidelity_score': 0.95,
        'distance_from_pa': 0.05,
        'mode': 'demo',  # Changed from 'test' to 'demo'
        'delta_from_previous': 0.02,
        'intervention_triggered': False
    }

    if supabase.transmit_delta(test_delta):
        print("✅ Delta transmission successful")
    else:
        print("❌ Delta transmission failed")
        return False

    # Test consent logging
    print("\n4. Testing consent logging...")
    if supabase.log_consent(
        session_id=test_session_id,
        consent_statement="Test consent for integration testing",
        consent_version="test_1.0"
    ):
        print("✅ Consent logging successful")
    else:
        print("❌ Consent logging failed")
        return False

    # Test session summary update
    print("\n5. Testing session summary update...")
    summary_data = {
        'mode': 'demo',  # Changed from 'test' to 'demo'
        'total_turns': 1,
        'avg_fidelity_score': 0.95,
        'min_fidelity_score': 0.95,
        'max_fidelity_score': 0.95,
        'total_interventions': 0,
        'beta_consent_given': True,
        'beta_consent_timestamp': datetime.now().isoformat()
    }

    if supabase.update_session_summary(test_session_id, summary_data):
        print("✅ Session summary update successful")
    else:
        print("❌ Session summary update failed")
        return False

    # Test turn lifecycle
    print("\n6. Testing turn lifecycle tracking...")

    # Initiate turn
    if supabase.initiate_turn(test_session_id, 2, 'demo'):
        print("✅ Turn initiation successful")
    else:
        print("❌ Turn initiation failed")
        return False

    # Mark calculating PA
    if supabase.mark_calculating_pa(test_session_id, 2):
        print("✅ PA calculation marking successful")
    else:
        print("❌ PA calculation marking failed")
        return False

    # Complete turn
    final_delta = {
        'fidelity_score': 0.93,
        'distance_from_pa': 0.07,
        'delta_from_previous': -0.02
    }

    if supabase.complete_turn(test_session_id, 2, final_delta):
        print("✅ Turn completion successful")
    else:
        print("❌ Turn completion failed")
        return False

    print("\n" + "=" * 60)
    return True

if __name__ == "__main__":
    success = test_supabase()

    if success:
        print("\n✅ ALL SUPABASE TESTS PASSED")
        print("The following operations are working:")
        print("  • Database connection")
        print("  • Delta transmission")
        print("  • Consent logging")
        print("  • Session summary updates")
        print("  • Turn lifecycle tracking")
    else:
        print("\n❌ SUPABASE TESTS FAILED")

    sys.exit(0 if success else 1)