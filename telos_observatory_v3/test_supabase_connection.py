"""
Test script to verify Supabase connection and delta transmission.
Run this after setting up Supabase credentials in .streamlit/secrets.toml
"""

import sys
from pathlib import Path
import uuid
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

def test_supabase_connection():
    """Test Supabase connection and basic operations."""

    print("=" * 60)
    print("SUPABASE CONNECTION TEST")
    print("=" * 60)
    print()

    # Import after path is set
    try:
        from services.supabase_client import get_supabase_service
        print("✓ Successfully imported Supabase service")
    except ImportError as e:
        print(f"❌ Failed to import Supabase service: {e}")
        print("\nMake sure you have installed supabase-py:")
        print("  pip install supabase")
        return False

    # Initialize service
    print("\n1. Initializing Supabase service...")
    service = get_supabase_service()

    if not service.enabled:
        print("❌ Supabase service not enabled")
        print("\nPlease check:")
        print("1. .streamlit/secrets.toml exists")
        print("2. SUPABASE_URL is set")
        print("3. SUPABASE_KEY is set (use service_role key)")
        return False

    print("✓ Supabase service initialized")

    # Test connection
    print("\n2. Testing connection...")
    if not service.test_connection():
        print("❌ Connection test failed")
        return False

    print("✓ Connection successful")

    # Generate test data
    test_session_id = uuid.uuid4()
    print(f"\n3. Generated test session ID: {test_session_id}")

    # Test delta transmission
    print("\n4. Testing delta transmission...")
    test_delta = {
        'session_id': str(test_session_id),
        'turn_number': 1,
        'fidelity_score': 0.87,
        'distance_from_pa': 0.23,
        'delta_from_previous': None,
        'intervention_triggered': False,
        'intervention_type': None,
        'intervention_reason': None,
        'mode': 'beta',
        'model_used': 'mistral-large-latest'
    }

    if service.transmit_delta(test_delta):
        print("✓ Delta transmission successful")
    else:
        print("❌ Delta transmission failed")
        return False

    # Test consent logging
    print("\n5. Testing consent logging...")
    consent_statement = "I consent to participate in TELOS Beta testing (TEST)"
    if service.log_consent(test_session_id, consent_statement, "3.0"):
        print("✓ Consent logging successful")
    else:
        print("❌ Consent logging failed")
        return False

    # Test session summary update
    print("\n6. Testing session summary update...")
    summary_data = {
        'mode': 'beta',
        'total_turns': 1,
        'avg_fidelity_score': 0.87,
        'min_fidelity_score': 0.87,
        'max_fidelity_score': 0.87,
        'total_interventions': 0,
        'beta_consent_given': True,
        'beta_consent_timestamp': datetime.now().isoformat()
    }

    if service.update_session_summary(test_session_id, summary_data):
        print("✓ Session summary update successful")
    else:
        print("❌ Session summary update failed")
        return False

    # Test PA config logging
    print("\n7. Testing PA config logging...")
    pa_config = {
        'purpose_elements': 3,
        'scope_elements': 5,
        'boundary_elements': 4,
        'constraint_tolerance': 0.15,
        'mode': 'beta'
    }

    if service.log_pa_config(test_session_id, pa_config):
        print("✓ PA config logging successful")
    else:
        print("❌ PA config logging failed")
        return False

    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. Check your Supabase dashboard to see the test data")
    print("2. Verify the data looks correct in each table:")
    print("   - governance_deltas (1 row)")
    print("   - session_summaries (1 row)")
    print("   - beta_consent_log (1 row)")
    print("   - primacy_attractor_configs (1 row)")
    print()
    print(f"Test session ID: {test_session_id}")
    print("You can query this session_id in Supabase to verify.")
    print()

    return True


if __name__ == "__main__":
    # Simulate Streamlit secrets for testing
    # In production, this comes from .streamlit/secrets.toml
    import streamlit as st

    # Check if secrets file exists
    secrets_file = Path(__file__).parent / ".streamlit" / "secrets.toml"
    if not secrets_file.exists():
        print("❌ No secrets.toml file found")
        print(f"\nExpected location: {secrets_file}")
        print("\nCreate .streamlit/secrets.toml with:")
        print("""
# Supabase Configuration
SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_KEY = "your-service-role-key-here"

# Mistral API (if not already present)
MISTRAL_API_KEY = "your-mistral-key"
        """)
        sys.exit(1)

    # Run tests
    success = test_supabase_connection()

    sys.exit(0 if success else 1)
