#!/usr/bin/env python3
"""
Phase 2 TELOS Study Runner
===========================

Complete end-to-end study pipeline:
1. Load ShareGPT Top 25 conversations
2. Establish primacy attractor with LLM-at-every-turn
3. Monitor for drift after PA establishment
4. Trigger counterfactual branching on drift
5. Generate comparison evidence

This is the production validation pipeline for TELOS governance.
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from telos_purpose.profiling.progressive_primacy_extractor import ProgressivePrimacyExtractor
from telos_purpose.llm_clients.mistral_client import MistralClient
from telos_purpose.core.embedding_provider import SentenceTransformerProvider
from telos_observatory.phase2_branch_manager import Phase2BranchManager


class Phase2StudyRunner:
    """
    Complete Phase 2 study runner.

    Implements the full production workflow:
    - LLM semantic analysis at every turn
    - Statistical convergence detection
    - Drift monitoring post-PA
    - Counterfactual branching
    - Evidence generation
    """

    def __init__(
        self,
        conversations_file: Path,
        output_dir: Path,
        mistral_api_key: str,
        drift_threshold: float = 0.8,
        branch_length: int = 5,
        distance_scale: float = 2.0
    ):
        """
        Initialize Phase 2 study runner.

        Args:
            conversations_file: Path to sharegpt_top25_conversations.json
            output_dir: Where to save study results
            mistral_api_key: Mistral API key
            drift_threshold: Fidelity threshold for drift detection
            branch_length: Number of turns in counterfactual branches
            distance_scale: Distance-to-fidelity scaling factor
        """
        self.conversations_file = conversations_file
        self.output_dir = output_dir
        self.drift_threshold = drift_threshold
        self.branch_length = branch_length
        self.distance_scale = distance_scale

        # Initialize LLM and embeddings
        self.llm_client = MistralClient(api_key=mistral_api_key)
        self.embedding_provider = SentenceTransformerProvider()

        # Initialize counterfactual branch manager (Phase 2 standalone version)
        self.branch_manager = Phase2BranchManager(
            llm_client=self.llm_client,
            embedding_provider=self.embedding_provider,
            distance_scale=self.distance_scale
        )

        # Results tracking
        self.completed_studies: List[Dict[str, Any]] = []
        self.failed_studies: List[Dict[str, Any]] = []

        # Create output directory
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def load_conversations(self) -> List[Dict[str, Any]]:
        """
        Load ShareGPT top 25 conversations.

        Returns:
            List of conversation dicts
        """
        with open(self.conversations_file, 'r') as f:
            return json.load(f)

    def run_single_study(
        self,
        conversation: Dict[str, Any],
        study_index: int
    ) -> Optional[Dict[str, Any]]:
        """
        Run complete study on single conversation.

        Workflow:
        1. Establish PA with LLM-at-every-turn (max 10 turns)
        2. Continue processing remaining turns
        3. Detect drift
        4. Trigger counterfactual branching
        5. Generate evidence

        Args:
            conversation: ShareGPT conversation dict
            study_index: Index in dataset (for naming)

        Returns:
            Study result dict or None if failed
        """
        conv_id = conversation['id']
        turns = conversation['turns']

        print(f"\n{'=' * 70}")
        print(f"STUDY #{study_index + 1}: {conv_id}")
        print(f"{'=' * 70}")
        print(f"Total turns: {len(turns)}")

        # ============================================================
        # PHASE 1: PRIMACY ATTRACTOR ESTABLISHMENT
        # ============================================================

        print("\n📊 Phase 1: Establishing Primacy Attractor (LLM-at-every-turn)...")

        extractor = ProgressivePrimacyExtractor(
            llm_client=self.llm_client,
            embedding_provider=self.embedding_provider,
            mode='progressive',
            llm_per_turn=True,  # PHASE 2 MODE: LLM at every turn
            window_size=3,
            centroid_stability_threshold=0.90,
            variance_stability_threshold=0.20,
            confidence_threshold=0.70,
            consecutive_stable_turns=2,
            max_turns_safety=10,  # Must converge within 10 turns
            distance_scale=self.distance_scale
        )

        pa_established = False
        convergence_turn = None
        attractor = None
        attractor_centroid = None
        llm_analyses = []

        # Process turns until PA converges
        for turn_idx, (user_msg, assistant_msg) in enumerate(turns, start=1):
            result = extractor.add_turn(user_msg, assistant_msg)

            print(f"  Turn {turn_idx}: {result['status']}")

            if result['status'] == 'converged':
                pa_established = True
                convergence_turn = result['convergence_turn']
                attractor = result['attractor']
                attractor_centroid = extractor.attractor_centroid
                llm_analyses = result.get('llm_analyses', [])

                print(f"\n✅ PA ESTABLISHED at turn {convergence_turn}")
                print(f"   Purpose: {', '.join(attractor.purpose)}")
                print(f"   Scope: {', '.join(attractor.scope[:3])}...")
                print(f"   LLM analyses collected: {len(llm_analyses)}")

                break

            elif result['status'] == 'error':
                print(f"\n❌ ERROR: {result['message']}")
                return None

            # Safety: Should never exceed 10 turns
            if turn_idx >= 10:
                print(f"\n❌ FAILED: PA did not converge within 10 turns")
                return None

        if not pa_established:
            print(f"\n❌ FAILED: PA not established (only {len(turns)} turns)")
            return None

        # Check if enough turns remain for drift analysis
        remaining_turns = turns[convergence_turn:]

        if len(remaining_turns) < 3:
            print(f"\n⚠️  SKIPPED: Only {len(remaining_turns)} turns after PA (need 3+ for analysis)")
            return None

        print(f"\n📈 Continuing with {len(remaining_turns)} remaining turns...")

        # ============================================================
        # PHASE 2: DRIFT MONITORING
        # ============================================================

        print("\n🔍 Phase 2: Monitoring for drift...")

        drift_detected = False
        drift_turn = None
        drift_fidelity = None

        for turn_idx, (user_msg, assistant_msg) in enumerate(remaining_turns, start=convergence_turn + 1):
            # Calculate fidelity
            response_emb = self.embedding_provider.encode(assistant_msg)
            distance = np.linalg.norm(response_emb - attractor_centroid)
            fidelity = max(0.0, 1.0 - (distance / self.distance_scale))

            print(f"  Turn {turn_idx}: F = {fidelity:.3f}", end="")

            if fidelity < self.drift_threshold:
                drift_detected = True
                drift_turn = turn_idx
                drift_fidelity = fidelity
                print(f" ⚠️  DRIFT DETECTED!")
                break
            else:
                print(f" ✓")

        if not drift_detected:
            print(f"\n✅ NO DRIFT: Conversation maintained alignment")
            # Still a valid study - no intervention needed
            return {
                'conversation_id': conv_id,
                'study_index': int(study_index),
                'pa_established': True,
                'convergence_turn': int(convergence_turn),
                'drift_detected': False,
                'attractor': {
                    'purpose': attractor.purpose,
                    'scope': attractor.scope,
                    'boundaries': attractor.boundaries
                },
                'llm_analyses': llm_analyses,
                'total_turns': int(len(turns)),
                'timestamp': datetime.now().isoformat()
            }

        # ============================================================
        # PHASE 3: COUNTERFACTUAL BRANCHING
        # ============================================================

        print(f"\n🌿 Phase 3: Generating counterfactual branches...")

        # Get remaining turns after drift for branching
        turns_after_drift = remaining_turns[drift_turn - convergence_turn:]

        if len(turns_after_drift) < self.branch_length:
            print(f"⚠️  Only {len(turns_after_drift)} turns available (need {self.branch_length})")
            branch_length = len(turns_after_drift)
        else:
            branch_length = self.branch_length

        branch_turns = turns_after_drift[:branch_length]

        # Trigger counterfactual branching
        try:
            branch_id = self.branch_manager.trigger_counterfactual(
                trigger_turn=drift_turn,
                trigger_fidelity=drift_fidelity,
                remaining_turns=branch_turns,
                attractor_center=attractor_centroid,
                conversation_history=turns[:drift_turn],  # History up to drift
                attractor_purpose=attractor.purpose,
                attractor_scope=attractor.scope,
                attractor_boundaries=attractor.boundaries,
                branch_length=branch_length
            )

            print(f"✅ Branches generated: {branch_id}")

            # Get branch data
            branch_data = self.branch_manager.get_branch(branch_id)

            if branch_data:
                orig_final_f = branch_data['original']['final_fidelity']
                telos_final_f = branch_data['telos']['final_fidelity']
                delta_f = branch_data['comparison']['delta_f']

                print(f"\n📊 Results:")
                print(f"   Original final F: {orig_final_f:.3f}")
                print(f"   TELOS final F: {telos_final_f:.3f}")
                print(f"   ΔF: {delta_f:+.3f}")

                # Export evidence
                evidence_dir = self.output_dir / conv_id
                evidence_dir.mkdir(exist_ok=True, parents=True)

                self.branch_manager.export_evidence(
                    branch_id=branch_id,
                    output_dir=str(evidence_dir)
                )

                print(f"   Evidence exported to: {evidence_dir}")

        except Exception as e:
            print(f"❌ Branching failed: {e}")
            return None

        # ============================================================
        # RETURN COMPLETE STUDY RESULT
        # ============================================================

        return {
            'conversation_id': conv_id,
            'study_index': int(study_index),
            'pa_established': True,
            'convergence_turn': int(convergence_turn),
            'drift_detected': True,
            'drift_turn': int(drift_turn),
            'drift_fidelity': float(drift_fidelity),
            'branch_id': branch_id,
            'attractor': {
                'purpose': attractor.purpose,
                'scope': attractor.scope,
                'boundaries': attractor.boundaries
            },
            'llm_analyses': llm_analyses,
            'counterfactual_results': {
                'original_final_f': float(orig_final_f),
                'telos_final_f': float(telos_final_f),
                'delta_f': float(delta_f),
                'governance_effective': bool(delta_f > 0)
            },
            'total_turns': int(len(turns)),
            'timestamp': datetime.now().isoformat()
        }

    def run_all_studies(self, max_studies: Optional[int] = None):
        """
        Run studies on all (or subset of) top 25 conversations.

        Args:
            max_studies: Optional limit on number of studies to run
        """
        print("=" * 70)
        print("PHASE 2 TELOS STUDY - PRODUCTION VALIDATION")
        print("=" * 70)

        # Load conversations
        conversations = self.load_conversations()
        print(f"\nLoaded {len(conversations)} conversations")

        if max_studies:
            conversations = conversations[:max_studies]
            print(f"Limiting to first {max_studies} studies")

        # Run each study
        for idx, conversation in enumerate(conversations):
            result = self.run_single_study(conversation, idx)

            if result:
                self.completed_studies.append(result)
                print(f"\n✅ Study #{idx + 1} COMPLETED")
            else:
                self.failed_studies.append({
                    'conversation_id': conversation['id'],
                    'study_index': idx,
                    'reason': 'See logs above'
                })
                print(f"\n❌ Study #{idx + 1} FAILED")

        # ============================================================
        # FINAL SUMMARY
        # ============================================================

        print(f"\n{'=' * 70}")
        print("STUDY COMPLETE")
        print(f"{'=' * 70}")
        print(f"\nCompleted: {len(self.completed_studies)}")
        print(f"Failed: {len(self.failed_studies)}")

        if self.completed_studies:
            with_drift = [s for s in self.completed_studies if s['drift_detected']]
            without_drift = [s for s in self.completed_studies if not s['drift_detected']]

            print(f"\nDrift detected: {len(with_drift)}")
            print(f"No drift: {len(without_drift)}")

            if with_drift:
                avg_delta_f = np.mean([s['counterfactual_results']['delta_f'] for s in with_drift])
                effective_count = sum(1 for s in with_drift if s['counterfactual_results']['governance_effective'])

                print(f"\nGovernance Effectiveness:")
                print(f"  Average ΔF: {avg_delta_f:+.3f}")
                print(f"  Effective interventions: {effective_count}/{len(with_drift)} ({100*effective_count/len(with_drift):.1f}%)")

        # Save summary
        summary_path = self.output_dir / 'phase2_study_summary.json'
        with open(summary_path, 'w') as f:
            json.dump({
                'completed_studies': self.completed_studies,
                'failed_studies': self.failed_studies,
                'summary': {
                    'total': len(conversations),
                    'completed': len(self.completed_studies),
                    'failed': len(self.failed_studies),
                    'with_drift': len([s for s in self.completed_studies if s['drift_detected']]),
                    'without_drift': len([s for s in self.completed_studies if not s['drift_detected']])
                },
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)

        print(f"\n💾 Summary saved: {summary_path}")
        print()


def main():
    """Run Phase 2 study."""
    import os

    # Paths
    conversations_file = Path(__file__).parent / 'sharegpt_data' / 'sharegpt_filtered_conversations.json'
    output_dir = Path(__file__).parent / 'phase2_study_results'

    # Get API key
    api_key = os.environ.get('MISTRAL_API_KEY')
    if not api_key:
        print("❌ ERROR: MISTRAL_API_KEY environment variable not set")
        print("\nSet it with:")
        print("  export MISTRAL_API_KEY='your-key-here'")
        return  # Exit gracefully instead of sys.exit

    # Create runner
    runner = Phase2StudyRunner(
        conversations_file=conversations_file,
        output_dir=output_dir,
        mistral_api_key=api_key,
        drift_threshold=0.8,
        branch_length=5,
        distance_scale=2.0
    )

    # Run studies
    if len(sys.argv) > 1:
        max_studies = int(sys.argv[1])
    else:
        max_studies = None  # Default: run all conversations

    runner.run_all_studies(max_studies=max_studies)


if __name__ == '__main__':
    main()
