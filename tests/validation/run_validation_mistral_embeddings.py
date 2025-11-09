#!/usr/bin/env python3
"""
Forensic validation using PRE-ESTABLISHED PA and MISTRAL EMBEDDINGS.
Runs counterfactual validation with 10-second delays and retry-until-success.
"""
import json
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from tests.validation.run_forensic_validation import ForensicValidator
from telos.core.unified_steward import PrimacyAttractor
from telos.core.embedding_provider import EmbeddingProvider

def main():
    """Run validation using pre-established PA with Mistral embeddings."""

    print("🔬 TELOS VALIDATION WITH MISTRAL EMBEDDINGS")
    print("="*80)
    print("⚠️  Using MISTRAL API embeddings (10-second delay, retry-until-success)")
    print()

    # Initialize validator
    validator = ForensicValidator()

    # Override embedder with Mistral version
    print("📦 Loading Mistral Embeddings (10s delay between calls)...")
    validator.embedder = EmbeddingProvider(use_mistral=True)
    print(f"✅ Mistral embeddings loaded")
    print(f"   Dimension: {validator.embedder.dimension}")
    print(f"   Rate limit: 10 seconds between calls")
    print()

    # Load conversation
    conversation_file = "tests/validation_data/baseline_conversations/real_claude_conversation.json"
    with open(conversation_file, 'r') as f:
        data = json.load(f)
    conversations = data['conversations']

    print(f"📊 Total conversation turns: {len(conversations) // 2}")
    print(f"⏩ Starting validation from turn 9")
    print(f"   Using PRE-ESTABLISHED PA (no need to re-discover it!)\n")

    # Create PRE-ESTABLISHED PA (from successful run)
    pa = PrimacyAttractor(
        purpose=[
            "To validate the TELOS framework for governance persistence",
            "To address and resolve critical observations and concerns about the TELOS framework"
        ],
        scope=[
            "Technical implementation and migration issues",
            "Validation framework and methodologies",
            "Federated protocols and privacy concerns",
            "Regulatory positioning and compliance",
            "Complexity and efficiency of the TELOS framework",
            "Grant application assessment and funding"
        ],
        boundaries=[
            "Specific implementation details of the TELOS framework",
            "Detailed regulatory compliance standards",
            "Pilot data showing TELOS outperforming baselines",
            "Fully pre-registered validation protocols",
            "Calibration of privacy budgets for federated protocols"
        ]
    )

    print("📍 PRE-ESTABLISHED PRIMACY ATTRACTOR:")
    print(f"   Purpose: {', '.join(pa.purpose)}")
    print(f"   Scope: {len(pa.scope)} areas")
    print(f"   Boundaries: {len(pa.boundaries)} constraints")

    # Create a mock extractor with the PA
    class MockExtractor:
        def __init__(self, pa, embedder):
            self.attractor = pa
            self.converged = True
            self.convergence_turn = 18  # From previous run
            # Create embedding for the PA (needed for fidelity calculations)
            pa_text = " ".join(pa.purpose + pa.scope + pa.boundaries)
            self.attractor_centroid = embedder.encode(pa_text)

        def get_attractor(self):
            """Return the primacy attractor."""
            return self.attractor

        def get_convergence_turn(self):
            """Return the turn where PA converged."""
            return self.convergence_turn

    pa_extractor = MockExtractor(pa, validator.embedder)

    # Initialize report with complete structure
    from datetime import datetime
    validator.report = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "model": validator.model,
            "embedding_provider": "Mistral API (mistral-embed, 10s delay)"
        },
        "pa_establishment": {
            "turn_by_turn_convergence": [],
            "convergence_summary": {
                "converged": "yes",
                "convergence_turn": 18,
                "method": "pre-established",
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

    # Run Phase 2 on ALL turns (function will start after convergence_turn)
    print("\n" + "="*80)
    print("COUNTERFACTUAL VALIDATION - ALL TURNS")
    print("="*80)

    validator._run_counterfactual_validation(conversations, pa_extractor)

    # Generate report
    print("\n" + "="*80)
    print("GENERATING FORENSIC REPORT")
    print("="*80)

    validator._generate_final_report()

    print("\n✅ VALIDATION COMPLETE!")

if __name__ == "__main__":
    main()
