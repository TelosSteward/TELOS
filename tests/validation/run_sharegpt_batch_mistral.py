#!/usr/bin/env python3
"""
Batch validation of 45 ShareGPT sessions using Mistral embeddings.
Runs full forensic validation protocol with 10-second delays and retry-until-success.
"""
import json
import sys
import os
import glob
from datetime import datetime
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from tests.validation.run_forensic_validation import ForensicValidator
from telos.core.unified_steward import PrimacyAttractor
from telos.core.embedding_provider import EmbeddingProvider

def load_sharegpt_sessions():
    """Load all 45 ShareGPT sessions."""
    base_path = "tests/validation_data/baseline_conversations"
    pattern = os.path.join(base_path, "sharegpt_filtered_*.json")
    files = sorted(glob.glob(pattern), key=lambda x: int(x.split('_')[-1].split('.')[0]))

    print(f"📦 Found {len(files)} ShareGPT session files")
    return files

def extract_conversation_from_session(session_data):
    """
    Extract conversation turns from processed ShareGPT session.
    Reconstructs original conversation format from stored turns.
    """
    conversations = []

    # Get PA from session
    pa_data = session_data.get('primacy_attractor', {})

    # Add initial turns to establish PA (we'll use stored PA)
    # Reconstruct conversation from turns (using ShareGPT format)
    for turn_data in session_data.get('turns', []):
        # User message
        conversations.append({
            "from": "human",
            "value": turn_data.get('user_message', turn_data.get('user_input', ''))
        })

        # Use native response (original response without TELOS)
        conversations.append({
            "from": "gpt",
            "value": turn_data.get('assistant_response_native', turn_data.get('response', ''))
        })

    return conversations, pa_data

def main():
    """Run batch validation on all ShareGPT sessions with Mistral embeddings."""

    print("="*80)
    print("🔬 SHAREGPT BATCH VALIDATION WITH MISTRAL EMBEDDINGS")
    print("="*80)
    print("⚠️  Using MISTRAL API embeddings (10-second delay, retry-until-success)")
    print()

    # Load all ShareGPT sessions
    session_files = load_sharegpt_sessions()

    if len(session_files) != 45:
        print(f"⚠️  WARNING: Expected 45 files, found {len(session_files)}")

    # Initialize Mistral embeddings
    print("📦 Loading Mistral Embeddings (10s delay between calls)...")
    embedder = EmbeddingProvider(use_mistral=True)
    print(f"✅ Mistral embeddings loaded")
    print(f"   Dimension: {embedder.dimension}")
    print(f"   Rate limit: 10 seconds between calls")
    print()

    # Aggregate results
    all_results = []
    successful = 0
    failed = 0

    for idx, session_file in enumerate(session_files, 1):
        session_name = os.path.basename(session_file)

        print("\n" + "="*80)
        print(f"SESSION {idx}/{len(session_files)}: {session_name}")
        print("="*80)

        try:
            # Load session data
            with open(session_file, 'r') as f:
                session_data = json.load(f)

            # Extract conversation and PA
            conversations, pa_data = extract_conversation_from_session(session_data)

            print(f"📊 Session info:")
            print(f"   Turns: {len(conversations) // 2}")
            print(f"   Original PA: {pa_data.get('purpose', ['Unknown'])[0][:60]}...")
            print()

            # Create PA from session data
            pa = PrimacyAttractor(
                purpose=pa_data.get('purpose', ["Unknown purpose"]),
                scope=pa_data.get('scope', ["Unknown scope"]),
                boundaries=pa_data.get('boundaries', ["Unknown boundaries"])
            )

            # Initialize validator for this session
            validator = ForensicValidator()
            validator.embedder = embedder

            # These sessions contain ONLY post-convergence turns, so set convergence to 0
            # (PA was already established in original run, these are the counterfactual validation turns)
            convergence_turn = 0
            original_convergence = session_data.get('metadata', {}).get('convergence_turn', 'N/A')

            print(f"📍 Using PA from original session")
            print(f"   Original convergence turn: {original_convergence}")
            print(f"   Validating {len(conversations) // 2} post-convergence turns")

            # Create mock extractor with PA
            class MockExtractor:
                def __init__(self, pa, embedder, conv_turn):
                    self.attractor = pa
                    self.converged = True
                    self.convergence_turn = conv_turn
                    pa_text = " ".join(pa.purpose + pa.scope + pa.boundaries)
                    self.attractor_centroid = embedder.encode(pa_text)

                def get_attractor(self):
                    return self.attractor

                def get_convergence_turn(self):
                    return self.convergence_turn

            pa_extractor = MockExtractor(pa, embedder, convergence_turn)

            # Initialize report structure
            validator.report = {
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "model": validator.model,
                    "embedding_provider": "Mistral API (mistral-embed, 10s delay)",
                    "session_file": session_name
                },
                "pa_establishment": {
                    "turn_by_turn_convergence": [],
                    "convergence_summary": {
                        "converged": "yes",
                        "convergence_turn": convergence_turn,
                        "method": "pre-established-from-session",
                        "attractor": {
                            "purpose": pa.purpose,
                            "scope": pa.scope,
                            "boundaries": pa.boundaries
                        },
                        "confidence": 1.0,
                        "centroid_stability": 1.0,
                        "variance_stability": 1.0
                    }
                },
                "counterfactual_branches": [],
                "intervention_analysis": {
                    "decision_points": [],
                    "api_call_log": []
                },
                "conversational_dna": {
                    "baseline_branches": [],
                    "telos_branches": []
                },
                "comparative_metrics": {
                    "per_turn_fidelity": [],
                    "branch_comparisons": []
                }
            }

            # Run counterfactual validation
            num_turns = len(conversations) // 2
            print(f"\n🔬 Starting counterfactual validation ({num_turns} turns)...")
            validator._run_counterfactual_validation(conversations, pa_extractor)

            # Generate report for this session
            validator._generate_final_report()

            # Extract key metrics
            avg_baseline = validator.report.get('comparative_metrics', {}).get('average_baseline_fidelity', 0)
            avg_telos = validator.report.get('comparative_metrics', {}).get('average_telos_fidelity', 0)
            delta_f = validator.report.get('comparative_metrics', {}).get('average_delta_f', 0)

            all_results.append({
                "session": session_name,
                "turns_validated": len(conversations) // 2 - convergence_turn,
                "avg_baseline_fidelity": avg_baseline,
                "avg_telos_fidelity": avg_telos,
                "delta_f": delta_f,
                "improvement_pct": (delta_f / avg_baseline * 100) if avg_baseline > 0 else 0
            })

            successful += 1

            print(f"\n✅ Session {session_name} complete!")
            print(f"   Avg Baseline Fidelity: {avg_baseline:.4f}")
            print(f"   Avg TELOS Fidelity: {avg_telos:.4f}")
            print(f"   ΔF: {delta_f:+.4f} ({(delta_f / avg_baseline * 100):+.1f}%)")

        except Exception as e:
            print(f"\n❌ Failed to process {session_name}: {e}")
            failed += 1
            import traceback
            traceback.print_exc()
            continue

    # Print aggregate summary
    print("\n" + "="*80)
    print("📊 BATCH VALIDATION SUMMARY")
    print("="*80)
    print(f"Total sessions: {len(session_files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print()

    if all_results:
        avg_delta_f = sum(r['delta_f'] for r in all_results) / len(all_results)
        avg_improvement = sum(r['improvement_pct'] for r in all_results) / len(all_results)

        print(f"Average ΔF across all sessions: {avg_delta_f:+.4f}")
        print(f"Average improvement: {avg_improvement:+.1f}%")
        print()

        # Show top 5 best improvements
        print("🏆 Top 5 sessions with highest TELOS improvement:")
        sorted_results = sorted(all_results, key=lambda x: x['delta_f'], reverse=True)[:5]
        for i, result in enumerate(sorted_results, 1):
            print(f"   {i}. {result['session']}: ΔF={result['delta_f']:+.4f} ({result['improvement_pct']:+.1f}%)")

        print()
        print("📉 Bottom 5 sessions (lowest improvement):")
        sorted_results_bottom = sorted(all_results, key=lambda x: x['delta_f'])[:5]
        for i, result in enumerate(sorted_results_bottom, 1):
            print(f"   {i}. {result['session']}: ΔF={result['delta_f']:+.4f} ({result['improvement_pct']:+.1f}%)")

    print("\n✅ BATCH VALIDATION COMPLETE!")

if __name__ == "__main__":
    main()
