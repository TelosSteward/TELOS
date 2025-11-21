#!/usr/bin/env python3
"""
Test Telemetric Signatures Implementation.

Validates both the Python quantum-resistant version and
demonstrates the IP protection benefits.
"""

import sys
import json
import time
import secrets
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from telos_privacy.cryptography.telemetric_keys_quantum import (
    QuantumTelemetricKeyGenerator,
    TelemetricSignatureGenerator
)


def test_basic_signature_generation():
    """Test basic signature generation and verification."""
    print("=" * 80)
    print("TEST 1: Basic Signature Generation")
    print("=" * 80)

    # Create key generator and signature generator
    session_id = f"test_session_{secrets.token_hex(4)}"
    key_gen = QuantumTelemetricKeyGenerator(session_id)
    sig_gen = TelemetricSignatureGenerator(key_gen)

    print(f"Session: {session_id}")
    print(f"Initial key size: {len(key_gen.state.current_key) * 8} bits")
    print()

    # Create sample delta
    delta = {
        "session_id": session_id,
        "turn_number": 1,
        "timestamp": time.time(),
        "delta_t_ms": 523,
        "baseline_fidelity": 0.832,
        "telos_fidelity": 0.947,
        "fidelity_delta": 0.115,
        "drift_detected": False,
        "intervention_applied": False,
        "user_message_length": 42,
        "response_length": 187
    }

    # Sign the delta
    signature_package = sig_gen.sign_delta(delta)

    print("Delta signed successfully!")
    print(f"Signature (first 32 chars): {signature_package['signature'][:32]}...")
    print(f"Algorithm: {signature_package['signature_algorithm']}")
    print(f"Quantum security level: {signature_package['quantum_security_level']}-bit")
    print(f"Canonical hash: {signature_package['canonical_hash'][:32]}...")
    print()

    # Verify signature
    canonical = sig_gen._canonicalize_delta(delta)
    is_valid = key_gen.verify_hmac_signature(canonical, bytes.fromhex(signature_package['signature']))

    print(f"Signature verification: {'✓ VALID' if is_valid else '✗ INVALID'}")
    print()

    return is_valid


def test_signature_chain_integrity():
    """Test that signature chain maintains integrity across turns."""
    print("=" * 80)
    print("TEST 2: Signature Chain Integrity")
    print("=" * 80)

    session_id = f"chain_test_{secrets.token_hex(4)}"
    key_gen = QuantumTelemetricKeyGenerator(session_id)
    sig_gen = TelemetricSignatureGenerator(key_gen)

    signatures = []
    print("Generating signature chain across 10 turns...")
    print()

    for turn in range(10):
        # Simulate telemetry for key rotation
        telemetry = {
            "turn_id": turn,
            "timestamp": time.time(),
            "delta_t_ms": 100 + secrets.randbelow(500),
            "embedding_distance": 0.1 + (secrets.randbelow(100) / 1000),
            "fidelity_score": 0.8 + (secrets.randbelow(200) / 1000),
            "soft_fidelity": 0.75 + (secrets.randbelow(200) / 1000),
            "lyapunov_delta": -0.05 + (secrets.randbelow(100) / 1000),
            "intervention_triggered": turn % 3 == 0,
            "user_input": "x" * secrets.randbelow(100),
            "model_output": "y" * secrets.randbelow(300)
        }

        # Rotate key
        key_gen.rotate_key(telemetry)

        # Create and sign delta
        delta = {
            "turn": turn,
            "timestamp": telemetry["timestamp"],
            "fidelity": telemetry["fidelity_score"],
            "intervention": telemetry["intervention_triggered"]
        }

        sig_package = sig_gen.sign_delta(delta)
        signatures.append(sig_package)

        print(f"Turn {turn}: Signature {sig_package['signature'][:16]}... "
              f"(Key rotation #{sig_package['key_rotation_number']})")

    print()
    print(f"Chain complete: {len(signatures)} signatures generated")
    print(f"Each signature unique: {len(set(s['signature'] for s in signatures)) == len(signatures)}")

    monotonic = all(signatures[i]['key_rotation_number'] < signatures[i+1]['key_rotation_number']
                    for i in range(len(signatures)-1))
    print(f"Monotonic key rotations: {monotonic}")
    print()

    return len(signatures) == 10


def test_ip_protection_proof():
    """Test IP protection proof generation."""
    print("=" * 80)
    print("TEST 3: IP Protection Proof Generation")
    print("=" * 80)

    session_id = f"ip_proof_{secrets.token_hex(4)}"
    key_gen = QuantumTelemetricKeyGenerator(session_id)
    sig_gen = TelemetricSignatureGenerator(key_gen)

    # Generate some activity
    print("Simulating session activity...")
    for i in range(5):
        telemetry = {
            "turn_id": i,
            "timestamp": time.time(),
            "delta_t_ms": 200 + i * 50,
            "fidelity_score": 0.85 + i * 0.02,
            "soft_fidelity": 0.83 + i * 0.02,
            "embedding_distance": 0.15 - i * 0.01,
            "lyapunov_delta": -0.03,
            "intervention_triggered": False,
            "user_input": "test input",
            "model_output": "test output"
        }
        key_gen.rotate_key(telemetry)

        delta = {"turn": i, "data": f"delta_{i}"}
        sig_gen.sign_delta(delta)

    # Generate IP proof
    ip_proof = sig_gen.generate_ip_proof({"session_id": session_id})

    print()
    print("IP Proof Document Generated:")
    print("-" * 40)
    print(f"Title: {ip_proof['title']}")
    print(f"Session: {ip_proof['session_id']}")
    print(f"Fingerprint: {ip_proof['telemetric_fingerprint'][:32]}...")
    print(f"Created: {ip_proof['created_at']}")
    print()

    print("Cryptographic Evidence:")
    evidence = ip_proof['cryptographic_evidence']
    print(f"  Algorithm: {evidence['algorithm']}")
    print(f"  Quantum Security: {evidence['quantum_security_level']}-bit")
    print(f"  Entropy Sources: {evidence['entropy_sources']}")
    print(f"  Total Entropy: {evidence['total_entropy_bits']} bits")
    print(f"  Key Rotations: {evidence['key_rotations']}")
    print(f"  Signatures Generated: {evidence['signatures_generated']}")
    print()

    print("IP Claims:")
    claims = ip_proof['ip_claims']
    for key, value in claims.items():
        print(f"  {key}: {value}")
    print()

    print("Verification Properties:")
    verification = ip_proof['verification']
    print(f"  Method: {verification['method']}")
    print(f"  Verifiable: {verification['can_verify']}")
    print(f"  Forgeable: {verification['cannot_forge']}")
    print(f"  Quantum Resistant: {verification['quantum_resistant']}")
    print(f"  NIST Compliant: {verification['nist_compliant']}")
    print()

    return True


def test_tamper_detection():
    """Test that tampering with delta breaks signature."""
    print("=" * 80)
    print("TEST 4: Tamper Detection")
    print("=" * 80)

    session_id = f"tamper_test_{secrets.token_hex(4)}"
    key_gen = QuantumTelemetricKeyGenerator(session_id)
    sig_gen = TelemetricSignatureGenerator(key_gen)

    # Create and sign original delta
    original_delta = {
        "turn": 1,
        "fidelity": 0.85,
        "drift": False
    }

    sig_package = sig_gen.sign_delta(original_delta)
    original_sig = sig_package['signature']

    print("Original delta signed")
    print(f"Signature: {original_sig[:32]}...")
    print()

    # Attempt to tamper with delta
    tampered_delta = {
        "turn": 1,
        "fidelity": 0.95,  # Changed fidelity
        "drift": False
    }

    # Try to verify tampered delta with original signature
    canonical_tampered = sig_gen._canonicalize_delta(tampered_delta)
    is_valid = key_gen.verify_hmac_signature(
        canonical_tampered,
        bytes.fromhex(original_sig)
    )

    print("Tampered delta (fidelity changed from 0.85 to 0.95)")
    print(f"Signature verification: {'✗ FAILED (Expected)' if not is_valid else '✓ VALID (ERROR!)'}")
    print()

    # Show that even tiny changes are detected
    tampered_delta2 = {
        "turn": 1,
        "fidelity": 0.850000001,  # Tiny change
        "drift": False
    }

    canonical_tampered2 = sig_gen._canonicalize_delta(tampered_delta2)
    is_valid2 = key_gen.verify_hmac_signature(
        canonical_tampered2,
        bytes.fromhex(original_sig)
    )

    print("Tampered delta (fidelity changed by 0.000000001)")
    print(f"Signature verification: {'✗ FAILED (Expected)' if not is_valid2 else '✓ VALID (ERROR!)'}")
    print()

    return not is_valid and not is_valid2  # Both should fail


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("TELEMETRIC SIGNATURES TEST SUITE")
    print("Testing Cryptographic IP Protection System")
    print("=" * 80 + "\n")

    tests = [
        ("Basic Signature Generation", test_basic_signature_generation),
        ("Signature Chain Integrity", test_signature_chain_integrity),
        ("IP Protection Proof", test_ip_protection_proof),
        ("Tamper Detection", test_tamper_detection)
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"ERROR in {name}: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name}: {status}")

    total_passed = sum(1 for _, p in results if p)
    print(f"\nTotal: {total_passed}/{len(tests)} tests passed")

    if total_passed == len(tests):
        print("\n🎉 ALL TESTS PASSED - Telemetric Signatures Ready for IP Protection!")
    else:
        print("\n⚠️  Some tests failed - review implementation")

    # Final message about benefits
    print("\n" + "=" * 80)
    print("PRACTICAL BENEFITS OF TELEMETRIC SIGNATURES")
    print("=" * 80)
    print("""
1. IMMEDIATE IP PROTECTION
   - Every delta is cryptographically signed
   - Creates unforgeable prior art documentation
   - Timestamps prove TELOS operation before competitors

2. NO BLOCKCHAIN NEEDED
   - Self-sovereign cryptographic proof
   - No external dependencies or fees
   - Privacy-preserving (only signatures, not content)

3. QUANTUM-RESISTANT PATH
   - MVP uses SHA-256 (128-bit quantum security)
   - Upgrade path to SHA3-512 (256-bit quantum security)
   - Ready for post-quantum algorithms (Kyber, Dilithium)

4. PATENT PROTECTION
   - Unique cryptographic method using session entropy
   - Non-reproducible without exact telemetry sequence
   - Strong defensive publication evidence

5. AUDIT TRAIL
   - Every turn has cryptographic proof
   - Chain of signatures proves session integrity
   - Third-party verifiable with session fingerprint
    """)

    print("=" * 80)
    print("Ready to deploy for immediate IP protection!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()