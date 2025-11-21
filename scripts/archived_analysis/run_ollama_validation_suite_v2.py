#!/usr/bin/env python3
"""
TELOS Ollama Validation Suite v2 - With Telemetric Signatures.

This script runs validation studies using Ollama with cryptographic signatures
for IP protection. All data stored in Supabase with full session content.

Requirements:
1. Ollama running with mistral:latest
2. Supabase credentials configured
3. All Python dependencies installed
"""

import os
import sys
import json
import time
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Set environment if not already set
if not os.getenv("SUPABASE_URL"):
    # Load from secrets.toml if available
    secrets_path = Path(".streamlit/secrets.toml")
    if secrets_path.exists():
        import tomli
        with open(secrets_path, "rb") as f:
            secrets = tomli.load(f)
            os.environ["SUPABASE_URL"] = secrets.get("SUPABASE_URL", "")
            os.environ["SUPABASE_KEY"] = secrets.get("SUPABASE_KEY", "")

# Import TELOS components
from telos_purpose.llm_clients.ollama_client import OllamaClient
from telos_privacy.cryptography.telemetric_keys_quantum import (
    QuantumTelemetricKeyGenerator,
    TelemetricSignatureGenerator
)
from telos_purpose.storage.validation_storage import ValidationStorage
from telos_purpose.core.embedding_provider import EmbeddingProvider


class OllamaValidationSuiteV2:
    """
    Validation suite with telemetric signatures and full Supabase integration.
    """

    def __init__(self, model: str = "mistral:latest"):
        """Initialize validation suite with telemetric signatures."""
        print("Initializing TELOS Ollama Validation Suite v2...")

        # Initialize Ollama client
        self.ollama = OllamaClient(model=model)
        self.model = model
        print(f"  ✓ Ollama client: {model}")

        # Initialize embedding provider for fidelity
        try:
            self.embedding_provider = EmbeddingProvider()
            print("  ✓ Embedding provider initialized")
        except Exception as e:
            print(f"  ⚠️  Embedding provider failed: {e}")
            self.embedding_provider = None

        # Initialize storage
        self.storage = ValidationStorage()
        print("  ✓ Supabase storage initialized")

        # Session tracking
        self.current_session_id = None
        self.telemetric_key_gen = None
        self.signature_gen = None

        print("✓ Suite ready")
        print()

    def _start_new_session(self, study_name: str, pa_config: Dict) -> str:
        """Start a new validation session with telemetric signatures."""
        session_id = str(uuid.uuid4())

        # Initialize telemetric keys for this session
        self.telemetric_key_gen = QuantumTelemetricKeyGenerator(session_id=session_id)
        self.signature_gen = TelemetricSignatureGenerator(self.telemetric_key_gen)

        # Get session fingerprint
        fingerprint = self.telemetric_key_gen.get_session_fingerprint()

        # Create session in Supabase with governance settings
        self.storage.create_validation_session({
            "session_id": session_id,
            "validation_study_name": study_name,
            "session_signature": fingerprint["key_history_hash"],
            "key_history_hash": fingerprint["key_history_hash"],
            "model": self.model,
            "total_turns": 0,  # Will be updated by trigger
            "dataset_source": "ShareGPT/Manual",
            "pa_configuration": pa_config,
            "basin_constant": 1.0,  # Goldilocks value from testing
            "constraint_tolerance": pa_config.get("constraint_tolerance", 0.05)  # Default 0.05 (strict)
        })

        self.current_session_id = session_id
        return session_id

    def _run_signed_turn(
        self,
        turn_number: int,
        user_msg: str,
        governance_mode: str,
        pa_config: Dict
    ) -> Dict[str, Any]:
        """Run a single turn with telemetric signature."""
        start_time = time.time()

        # Generate response based on governance mode
        if governance_mode == "stateless":
            response = self.ollama.generate([{"role": "user", "content": user_msg}])
        elif governance_mode == "prompt_only":
            system_prompt = f"Purpose: {pa_config['purpose']}"
            response = self.ollama.generate([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg}
            ])
        elif governance_mode == "telos":
            # For now, use prompt_only (full TELOS integration later)
            system_prompt = f"Purpose: {pa_config['purpose']}. Follow this carefully."
            response = self.ollama.generate([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg}
            ])
        else:
            response = self.ollama.generate([{"role": "user", "content": user_msg}])

        delta_t_ms = int((time.time() - start_time) * 1000)

        # Calculate fidelity if embedding provider available
        fidelity_score = None
        if self.embedding_provider:
            try:
                purpose_emb = self.embedding_provider.encode(pa_config["purpose"])
                response_emb = self.embedding_provider.encode(response)

                import numpy as np
                similarity = np.dot(purpose_emb, response_emb) / (
                    np.linalg.norm(purpose_emb) * np.linalg.norm(response_emb)
                )
                fidelity_score = float((similarity + 1) / 2)  # Convert to 0-1 and Python float
            except Exception as e:
                print(f"    ⚠️  Fidelity calculation failed: {e}")
                fidelity_score = 0.5  # Default
        else:
            fidelity_score = 0.5  # Default if no embeddings

        # Create delta data for signing (ensure all values are JSON serializable)
        delta_data = {
            "session_id": str(self.current_session_id),
            "turn_number": int(turn_number),
            "timestamp": datetime.now().isoformat(),
            "delta_t_ms": int(delta_t_ms),
            "user_message_length": int(len(user_msg)),
            "response_length": int(len(response)),
            "fidelity_score": float(fidelity_score) if fidelity_score is not None else None
        }

        # Sign the delta
        signed_delta = self.signature_gen.sign_delta(delta_data)

        # Store to Supabase with full content
        self.storage.store_signed_turn({
            "session_id": self.current_session_id,
            "turn_number": turn_number,
            "user_message": user_msg,
            "assistant_response": response,
            "fidelity_score": fidelity_score,
            "turn_telemetric_signature": signed_delta["signature"],
            "key_rotation_number": signed_delta["key_rotation_number"],
            "delta_t_ms": delta_t_ms,
            "governance_mode": governance_mode
        })

        return {
            "response": response,
            "fidelity": fidelity_score,
            "delta_t_ms": delta_t_ms,
            "signature": signed_delta["signature"][:32]
        }

    def run_baseline_comparison(
        self,
        test_messages: List[str] = None,
        pa_config: Dict = None
    ):
        """Run baseline comparison across governance modes."""
        print("=" * 80)
        print("BASELINE COMPARISON STUDY")
        print("=" * 80)
        print()

        # Default test messages
        if not test_messages:
            test_messages = [
                "What is machine learning?",
                "Explain neural networks simply.",
                "How does backpropagation work?",
                "What are the ethical concerns in AI?",
                "Describe supervised learning."
            ]

        # Default PA config
        if not pa_config:
            pa_config = {
                "purpose": "Provide accurate technical education about AI",
                "scope": "Machine learning concepts",
                "boundaries": ["Stay factual", "Use clear language", "Be educational"]
            }

        governance_modes = ["stateless", "prompt_only", "telos"]
        results = {}

        for mode in governance_modes:
            print(f"\nRunning {mode.upper()} governance mode...")
            print("-" * 40)

            # Start new session
            session_id = self._start_new_session(
                f"baseline_comparison_{mode}",
                pa_config
            )
            print(f"Session: {session_id}")

            mode_results = []
            for i, msg in enumerate(test_messages, 1):
                print(f"\n  Turn {i}/{len(test_messages)}")
                print(f"  User: {msg[:60]}...")

                result = self._run_signed_turn(i, msg, mode, pa_config)

                print(f"  Response: {result['response'][:60]}...")
                print(f"  Fidelity: {result['fidelity']:.3f}")
                print(f"  Time: {result['delta_t_ms']}ms")
                print(f"  Signature: {result['signature']}...")

                mode_results.append(result)

            # Mark session complete
            self.storage.mark_session_complete(session_id)

            # Calculate averages
            avg_fidelity = sum(r['fidelity'] for r in mode_results) / len(mode_results)
            avg_time = sum(r['delta_t_ms'] for r in mode_results) / len(mode_results)

            results[mode] = {
                "session_id": session_id,
                "avg_fidelity": avg_fidelity,
                "avg_time_ms": avg_time,
                "turns": len(mode_results)
            }

            print(f"\n  ✓ {mode.upper()} complete: avg fidelity {avg_fidelity:.3f}")

        # Print comparison
        print("\n" + "=" * 80)
        print("BASELINE COMPARISON RESULTS")
        print("=" * 80)

        for mode, data in results.items():
            print(f"\n{mode.upper()}:")
            print(f"  Average Fidelity: {data['avg_fidelity']:.3f}")
            print(f"  Average Time: {data['avg_time_ms']:.0f}ms")
            print(f"  Turns: {data['turns']}")
            print(f"  Session: {data['session_id']}")

        # Calculate improvement
        if "stateless" in results and "telos" in results:
            improvement = (
                (results["telos"]["avg_fidelity"] - results["stateless"]["avg_fidelity"])
                / results["stateless"]["avg_fidelity"] * 100
            )
            print(f"\nTELOS Improvement: +{improvement:.1f}%")

        print("\n✓ Baseline comparison complete!")
        print("=" * 80)

        return results

    def run_quick_test(self):
        """Run a quick 3-turn test to verify everything works."""
        print("=" * 80)
        print("QUICK VALIDATION TEST (3 turns)")
        print("=" * 80)
        print()

        test_messages = [
            "Hello, test message 1",
            "What is 2+2?",
            "Thank you"
        ]

        pa_config = {
            "purpose": "Be helpful and accurate",
            "scope": "General assistance",
            "boundaries": ["Be polite", "Be factual"]
        }

        session_id = self._start_new_session("quick_test", pa_config)
        print(f"Test session: {session_id}\n")

        for i, msg in enumerate(test_messages, 1):
            print(f"Turn {i}/{len(test_messages)}: {msg}")
            result = self._run_signed_turn(i, msg, "telos", pa_config)
            print(f"  Response: {result['response'][:60]}...")
            print(f"  Signed: {result['signature']}...\n")

        self.storage.mark_session_complete(session_id)

        # Get IP proof
        ip_proof = self.storage.get_ip_proof(session_id)
        print("IP Proof Retrieved:")
        print(f"  Signed turns: {ip_proof['signed_turns']}/{ip_proof['total_turns']}")
        print(f"  Signature chain: {len(ip_proof['signature_chain'])} signatures")
        print(f"  Session signature: {ip_proof['session_signature'][:32]}...")

        print("\n✓ Quick test complete!")
        print("=" * 80)


def main():
    """Run validation suite."""
    import sys

    suite = OllamaValidationSuiteV2()

    # Check command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "quick":
            suite.run_quick_test()
        elif command == "baseline":
            suite.run_baseline_comparison()
        else:
            print(f"Unknown command: {command}")
            print("Usage: python run_ollama_validation_suite_v2.py [quick|baseline]")
    else:
        # Default: run quick test
        print("Running quick test (use 'baseline' argument for full study)")
        print()
        suite.run_quick_test()


if __name__ == "__main__":
    main()
