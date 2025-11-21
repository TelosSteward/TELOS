#!/usr/bin/env python3
"""
End-to-End Test: Validation Pipeline with Telemetric Signatures.

Tests the complete flow:
1. Ollama client generates response
2. Telemetric signatures are created
3. Data is stored in Supabase with signatures
4. IP proof can be retrieved

This validates that all components work together.
"""

import os
import sys
import time
import uuid
from datetime import datetime

# Set environment
os.environ['SUPABASE_URL'] = 'https://ukqrwjowlchhwznefboj.supabase.co'
os.environ['SUPABASE_KEY'] = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InVrcXJ3am93bGNoaHd6bmVmYm9qIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2MjMyOTE2MCwiZXhwIjoyMDc3OTA1MTYwfQ.TvefimDWnnlAz4dj9-XBFJ4xl7hmXX9bZJSidzUjHTs'

# Import components
from telos_purpose.llm_clients.ollama_client import OllamaClient
from telos_privacy.cryptography.telemetric_keys_quantum import (
    QuantumTelemetricKeyGenerator,
    TelemetricSignatureGenerator
)
from telos_purpose.storage.validation_storage import ValidationStorage


def test_e2e_validation_pipeline():
    """Test complete validation pipeline end-to-end."""
    print("=" * 80)
    print("END-TO-END VALIDATION PIPELINE TEST")
    print("=" * 80)
    print()

    # Step 1: Initialize Ollama
    print("1. Initializing Ollama client...")
    try:
        ollama = OllamaClient(model="mistral:latest")
        print("   ✓ Ollama client initialized")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False

    # Step 2: Initialize Telemetric Signatures
    print()
    print("2. Initializing telemetric signature system...")
    try:
        session_id = str(uuid.uuid4())
        tkey_gen = QuantumTelemetricKeyGenerator(session_id=session_id)
        signature_gen = TelemetricSignatureGenerator(tkey_gen)
        print(f"   ✓ Session ID: {session_id}")
        print(f"   ✓ Telemetric keys initialized")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False

    # Step 3: Initialize Supabase storage
    print()
    print("3. Initializing Supabase storage...")
    try:
        storage = ValidationStorage()
        print("   ✓ Storage initialized")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False

    # Step 4: Create validation session in Supabase
    print()
    print("4. Creating validation session...")
    try:
        session_fingerprint = tkey_gen.get_session_fingerprint()

        session_record = storage.create_validation_session({
            "session_id": session_id,
            "validation_study_name": "e2e_test",
            "session_signature": session_fingerprint["key_history_hash"],  # Use key history as signature
            "key_history_hash": session_fingerprint["key_history_hash"],
            "model": "mistral:latest",
            "total_turns": 3,
            "dataset_source": "manual_test",
            "pa_configuration": {
                "purpose": "Test purpose",
                "scope": "Test scope"
            }
        })
        print(f"   ✓ Session created in Supabase")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False

    # Step 5: Run test turns with signatures
    print()
    print("5. Running test turns with telemetric signatures...")

    test_messages = [
        "Hello, how are you?",
        "What is 2 + 2?",
        "Explain AI in one sentence."
    ]

    for turn_num, user_msg in enumerate(test_messages, 1):
        print(f"\n   Turn {turn_num}/{len(test_messages)}")
        print(f"   User: {user_msg}")

        try:
            # Generate response with Ollama
            start_time = time.time()
            response = ollama.generate([{"role": "user", "content": user_msg}])
            delta_t_ms = int((time.time() - start_time) * 1000)

            print(f"   Assistant: {response[:80]}...")
            print(f"   Time: {delta_t_ms}ms")

            # Create delta data
            delta_data = {
                "session_id": session_id,
                "turn_number": turn_num,
                "timestamp": datetime.now().isoformat(),
                "delta_t_ms": delta_t_ms,
                "user_message_length": len(user_msg),
                "response_length": len(response),
                "fidelity_score": 0.85  # Mock fidelity for test
            }

            # Sign the delta
            signed_delta = signature_gen.sign_delta(delta_data)
            print(f"   Signature: {signed_delta['signature'][:32]}...")

            # Store to Supabase
            turn_record = storage.store_signed_turn({
                "session_id": session_id,
                "turn_number": turn_num,
                "user_message": user_msg,
                "assistant_response": response,
                "fidelity_score": 0.85,
                "turn_telemetric_signature": signed_delta["signature"],
                "key_rotation_number": signed_delta["key_rotation_number"],
                "delta_t_ms": delta_t_ms,
                "governance_mode": "telos"
            })
            print(f"   ✓ Turn stored to Supabase (ID: {turn_record['id']})")

        except Exception as e:
            print(f"   ✗ Failed: {e}")
            return False

    # Step 6: Mark session complete
    print()
    print("6. Marking session complete...")
    try:
        storage.mark_session_complete(session_id)
        print("   ✓ Session marked complete")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False

    # Step 7: Retrieve IP proof
    print()
    print("7. Retrieving IP proof...")
    try:
        ip_proof = storage.get_ip_proof(session_id)

        print("   " + "=" * 70)
        print("   IP PROOF VERIFICATION")
        print("   " + "=" * 70)
        print(f"   Session ID: {ip_proof['session_id']}")
        print(f"   Study: {ip_proof['validation_study_name']}")
        print(f"   Created: {ip_proof['created_at']}")
        print(f"   Model: {ip_proof['model_used']}")
        print(f"   Total turns: {ip_proof['total_turns']}")
        print(f"   Signed turns: {ip_proof['signed_turns']}")
        print(f"   Session signature: {ip_proof['session_signature'][:32]}...")
        print(f"   Key history hash: {ip_proof['key_history_hash'][:32]}...")
        print(f"   Signature algorithm: {ip_proof['signature_algorithm']}")
        print(f"   Signature chain: {len(ip_proof['signature_chain'])} signatures")
        print("   " + "=" * 70)

        if ip_proof['signed_turns'] == ip_proof['total_turns']:
            print("   ✓ All turns signed successfully")
        else:
            print(f"   ⚠️  Only {ip_proof['signed_turns']}/{ip_proof['total_turns']} turns signed")

    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False

    # Success!
    print()
    print("=" * 80)
    print("✓ END-TO-END TEST PASSED")
    print("=" * 80)
    print()
    print("All components working:")
    print("  ✓ Ollama generates responses")
    print("  ✓ Telemetric signatures created")
    print("  ✓ Data stored in Supabase")
    print("  ✓ IP proof retrievable")
    print()
    print(f"Test session ID: {session_id}")
    print("You can now run full validation studies!")
    print()

    return True


if __name__ == "__main__":
    success = test_e2e_validation_pipeline()
    sys.exit(0 if success else 1)
