#!/usr/bin/env python3
"""
Phase 2B Continuous Monitoring Study Runner
============================================

CRITICAL DIFFERENCE FROM PHASE 2:
- Phase 2: Detects FIRST drift, creates ONE branch pair, stops
- Phase 2B: Detects ALL drifts, creates MULTIPLE independent branch pairs, continues until conversation ends

This tests TRUE continuous governance as TELOS operates in production:
- Monitors EVERY turn
- Intervenes EVERY time F < threshold
- Each intervention is an independent experiment
- Returns aggregate metrics (mean ΔF, total interventions, effectiveness rate)

Methodology:
- Original Branch: Always uses historical responses (baseline)
- TELOS Branch: Creates independent branch pairs at each drift point
- Returns to baseline after each branch pair for fair comparison
- Aggregate results show continuous governance capability

Usage:
    python run_phase2b_continuous.py <input_file> <output_name>
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from telos_observatory.phase2_branch_manager import Phase2BranchManager
from telos_purpose.llm_clients.mistral_client import MistralClient
from telos_purpose.core.embedding_provider import SentenceTransformerProvider


@dataclass
class PrimacyAttractor:
    """Primacy attractor representation."""
    purpose: List[str]
    scope: List[str]
    boundaries: List[str]


class Phase2BContinuousStudyRunner:
    """
    Phase 2B study runner with continuous monitoring.

    Key difference: Monitors ALL turns and creates independent branch pairs
    at EVERY drift point, not just the first one.
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
        """Initialize Phase 2B continuous study runner."""
        self.conversations_file = conversations_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        self.drift_threshold = drift_threshold
        self.branch_length = branch_length
        self.distance_scale = distance_scale

        # Initialize providers
        self.llm_client = MistralClient(api_key=mistral_api_key)
        self.embedding_provider = SentenceTransformerProvider()

        # Initialize branch manager
        self.branch_manager = Phase2BranchManager(
            llm_client=self.llm_client,
            embedding_provider=self.embedding_provider,
            distance_scale=distance_scale
        )

    def load_conversations(self) -> List[Dict[str, Any]]:
        """Load conversations from ShareGPT format JSON."""
        with open(self.conversations_file) as f:
            data = json.load(f)

        # Handle both list format and single conversation format
        if isinstance(data, list):
            return data
        else:
            return [data]

    def establish_primacy_attractor(
        self,
        turns: List[Tuple[str, str]],
        convergence_window: int = 10
    ) -> Tuple[Optional[PrimacyAttractor], int, np.ndarray]:
        """
        Establish primacy attractor using first N turns.

        Returns:
            (attractor, convergence_turn, centroid)
        """
        if len(turns) < convergence_window:
            return None, 0, None

        # Use first N turns to establish PA
        baseline_turns = turns[:convergence_window]

        # Calculate centroid from assistant responses
        embeddings = []
        for user_msg, assistant_msg in baseline_turns:
            emb = self.embedding_provider.encode(assistant_msg)
            embeddings.append(emb)

        centroid = np.mean(embeddings, axis=0)

        # Create simple PA (in real implementation, would use LLM analysis)
        attractor = PrimacyAttractor(
            purpose=["Establish conversation purpose from baseline turns"],
            scope=["Topics covered in baseline"],
            boundaries=["Off-topic discussions"]
        )

        return attractor, convergence_window, centroid

    def run_single_study(
        self,
        conversation: Dict[str, Any],
        study_index: int
    ) -> Optional[Dict[str, Any]]:
        """
        Run Phase 2B continuous monitoring study on single conversation.

        This is the key method that implements CONTINUOUS monitoring.
        """
        conv_id = conversation.get('id', f'conversation_{study_index}')

        print("\n" + "=" * 70)
        print(f"STUDY #{study_index + 1}: {conv_id}")
        print("=" * 70)

        # Extract turns
        turns = [(t[0], t[1]) for t in conversation['turns']]

        print(f"Total turns: {len(turns)}")

        # ============================================================
        # PHASE 1: ESTABLISH PRIMACY ATTRACTOR
        # ============================================================

        print(f"\n📊 Phase 1: Establishing Primacy Attractor (first 10 turns)...")

        attractor, convergence_turn, attractor_centroid = self.establish_primacy_attractor(turns)

        if not attractor:
            print("❌ Failed to establish PA")
            return None

        print(f"✅ PA ESTABLISHED at turn {convergence_turn}")

        # ============================================================
        # PHASE 2: CONTINUOUS DRIFT MONITORING
        # ============================================================

        print(f"\n🔍 Phase 2: Continuous drift monitoring...")

        remaining_turns = turns[convergence_turn:]
        drift_triggers = []  # Store all drift points

        # CRITICAL: Monitor ALL remaining turns, don't break after first drift
        for turn_idx, (user_msg, assistant_msg) in enumerate(remaining_turns, start=convergence_turn + 1):
            # Calculate fidelity
            response_emb = self.embedding_provider.encode(assistant_msg)
            distance = np.linalg.norm(response_emb - attractor_centroid)
            fidelity = max(0.0, 1.0 - (distance / self.distance_scale))

            print(f"  Turn {turn_idx}: F = {fidelity:.3f}", end="")

            if fidelity < self.drift_threshold:
                print(f" ⚠️  DRIFT DETECTED!")
                drift_triggers.append({
                    'turn': turn_idx,
                    'fidelity': fidelity,
                    'relative_index': turn_idx - convergence_turn - 1
                })
            else:
                print(f" ✓")

        if not drift_triggers:
            print(f"\n✅ NO DRIFT: Conversation maintained alignment")
            return {
                'conversation_id': conv_id,
                'study_index': int(study_index),
                'pa_established': True,
                'convergence_turn': int(convergence_turn),
                'drift_detected': False,
                'total_interventions': 0,
                'total_turns': int(len(turns)),
                'timestamp': datetime.now().isoformat()
            }

        print(f"\n📊 Found {len(drift_triggers)} drift point(s): {[t['turn'] for t in drift_triggers]}")

        # ============================================================
        # PHASE 3: INDEPENDENT COUNTERFACTUAL BRANCHES
        # ============================================================

        print(f"\n🌿 Phase 3: Generating independent counterfactual branches...")

        branch_results = []

        for trigger in drift_triggers:
            trigger_turn = trigger['turn']
            trigger_fidelity = trigger['fidelity']
            relative_idx = trigger['relative_index']

            print(f"\n  📍 Trigger #{len(branch_results) + 1} at turn {trigger_turn} (F={trigger_fidelity:.3f})")

            # Get turns after this trigger for branching
            turns_after_trigger = remaining_turns[relative_idx:]

            if len(turns_after_trigger) < self.branch_length:
                branch_length = len(turns_after_trigger)
                if branch_length == 0:
                    print(f"     ⚠️  No turns available after trigger, skipping")
                    continue
                print(f"     ⚠️  Only {branch_length} turns available")
            else:
                branch_length = self.branch_length

            branch_turns = turns_after_trigger[:branch_length]

            # Generate independent branch pair
            try:
                branch_id = self.branch_manager.trigger_counterfactual(
                    trigger_turn=trigger_turn,
                    trigger_fidelity=trigger_fidelity,
                    remaining_turns=branch_turns,
                    attractor_center=attractor_centroid,
                    conversation_history=turns[:trigger_turn],
                    attractor_purpose=attractor.purpose,
                    attractor_scope=attractor.scope,
                    attractor_boundaries=attractor.boundaries,
                    branch_length=branch_length
                )

                # Get results
                branch_data = self.branch_manager.get_branch(branch_id)

                if branch_data:
                    delta_f = branch_data['comparison']['delta_f']

                    print(f"     ✅ Branch generated: ΔF = {delta_f:+.3f}")

                    branch_results.append({
                        'trigger_turn': trigger_turn,
                        'trigger_fidelity': trigger_fidelity,
                        'branch_id': branch_id,
                        'delta_f': delta_f,
                        'original_final_f': branch_data['original']['final_fidelity'],
                        'telos_final_f': branch_data['telos']['final_fidelity'],
                        'governance_effective': delta_f > 0
                    })

                    # Export evidence
                    evidence_dir = self.output_dir / conv_id / branch_id
                    evidence_dir.mkdir(exist_ok=True, parents=True)

                    self.branch_manager.export_evidence(
                        branch_id=branch_id,
                        output_dir=str(evidence_dir)
                    )

            except Exception as e:
                print(f"     ❌ Branch generation failed: {e}")
                continue

        # ============================================================
        # PHASE 4: AGGREGATE RESULTS
        # ============================================================

        if not branch_results:
            print("\n❌ No branches generated")
            return None

        print(f"\n" + "=" * 70)
        print(f"CONTINUOUS GOVERNANCE RESULTS")
        print("=" * 70)

        # Calculate aggregate metrics
        delta_f_values = [b['delta_f'] for b in branch_results]
        mean_delta_f = np.mean(delta_f_values)
        effective_count = sum(1 for b in branch_results if b['governance_effective'])
        effectiveness_rate = effective_count / len(branch_results)

        print(f"\n📊 Aggregate Metrics:")
        print(f"   Total interventions: {len(branch_results)}")
        print(f"   Mean ΔF: {mean_delta_f:+.3f}")
        print(f"   Effective interventions: {effective_count}/{len(branch_results)} ({effectiveness_rate*100:.1f}%)")
        print(f"   ΔF range: [{min(delta_f_values):+.3f}, {max(delta_f_values):+.3f}]")

        # Return complete study result
        return {
            'conversation_id': conv_id,
            'study_index': int(study_index),
            'pa_established': True,
            'convergence_turn': int(convergence_turn),
            'drift_detected': True,
            'total_interventions': len(branch_results),
            'interventions': branch_results,
            'aggregate_metrics': {
                'mean_delta_f': float(mean_delta_f),
                'median_delta_f': float(np.median(delta_f_values)),
                'std_delta_f': float(np.std(delta_f_values)),
                'min_delta_f': float(min(delta_f_values)),
                'max_delta_f': float(max(delta_f_values)),
                'effectiveness_rate': float(effectiveness_rate),
                'effective_count': int(effective_count),
                'ineffective_count': int(len(branch_results) - effective_count)
            },
            'attractor': {
                'purpose': attractor.purpose,
                'scope': attractor.scope,
                'boundaries': attractor.boundaries
            },
            'total_turns': int(len(turns)),
            'timestamp': datetime.now().isoformat()
        }

    def run_all_studies(self, max_studies: Optional[int] = None):
        """Run Phase 2B continuous studies on all conversations."""
        print("=" * 70)
        print("PHASE 2B CONTINUOUS GOVERNANCE VALIDATION")
        print("=" * 70)

        conversations = self.load_conversations()
        print(f"\nLoaded {len(conversations)} conversations")

        if max_studies:
            conversations = conversations[:max_studies]
            print(f"Limiting to first {max_studies} studies")

        completed_studies = []
        failed_studies = []

        for idx, conversation in enumerate(conversations):
            try:
                result = self.run_single_study(conversation, idx)

                if result:
                    completed_studies.append(result)
                    print(f"\n✅ Study #{idx + 1} COMPLETED")
                else:
                    failed_studies.append({'index': idx, 'reason': 'No result returned'})
                    print(f"\n❌ Study #{idx + 1} FAILED")

            except Exception as e:
                failed_studies.append({'index': idx, 'reason': str(e)})
                print(f"\n❌ Study #{idx + 1} FAILED: {e}")

        # Save summary
        summary_path = self.output_dir / 'phase2b_continuous_summary.json'

        # Calculate aggregate statistics
        all_interventions = []
        for study in completed_studies:
            if study.get('drift_detected'):
                all_interventions.extend(study['interventions'])

        if all_interventions:
            all_delta_f = [i['delta_f'] for i in all_interventions]
            overall_stats = {
                'total_studies': len(conversations),
                'completed': len(completed_studies),
                'failed': len(failed_studies),
                'total_interventions': len(all_interventions),
                'mean_delta_f': float(np.mean(all_delta_f)),
                'median_delta_f': float(np.median(all_delta_f)),
                'std_delta_f': float(np.std(all_delta_f)),
                'effectiveness_rate': float(sum(1 for i in all_interventions if i['governance_effective']) / len(all_interventions))
            }
        else:
            overall_stats = {
                'total_studies': len(conversations),
                'completed': len(completed_studies),
                'failed': len(failed_studies),
                'total_interventions': 0
            }

        # Clean data for JSON serialization
        def make_json_safe(obj):
            """Recursively convert numpy types and other non-serializable objects."""
            if isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: make_json_safe(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [make_json_safe(item) for item in obj]
            elif isinstance(obj, (str, int, float, bool)) or obj is None:
                return obj
            else:
                # For other objects, try to convert to string
                return str(obj)

        # Apply to all data before serializing
        summary = {
            'phase': '2B_continuous',
            'timestamp': datetime.now().isoformat(),
            'overall_statistics': make_json_safe(overall_stats),
            'completed_studies': make_json_safe(completed_studies),
            'failed_studies': make_json_safe(failed_studies)
        }

        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n" + "=" * 70)
        print("PHASE 2B CONTINUOUS VALIDATION COMPLETE")
        print("=" * 70)
        print(f"\nCompleted: {len(completed_studies)}/{len(conversations)}")
        print(f"Failed: {len(failed_studies)}/{len(conversations)}")

        if all_interventions:
            print(f"\n📊 Overall Statistics:")
            print(f"   Total interventions: {len(all_interventions)}")
            print(f"   Mean ΔF: {overall_stats['mean_delta_f']:+.3f}")
            print(f"   Effectiveness rate: {overall_stats['effectiveness_rate']*100:.1f}%")

        print(f"\n💾 Summary saved: {summary_path}")


def main():
    """Main entry point."""

    # Check API key
    api_key = os.environ.get('MISTRAL_API_KEY')
    if not api_key:
        print("❌ ERROR: MISTRAL_API_KEY environment variable not set")
        sys.exit(1)

    # Parse arguments
    if len(sys.argv) < 3:
        print("Usage: python run_phase2b_continuous.py <input_file> <output_name>")
        sys.exit(1)

    input_file = Path(sys.argv[1])
    output_name = sys.argv[2]

    if not input_file.exists():
        print(f"❌ ERROR: File not found: {input_file}")
        sys.exit(1)

    # Create output directory
    output_dir = Path(__file__).parent / f'phase2b_continuous_{output_name}'

    # Run Phase 2B continuous validation
    runner = Phase2BContinuousStudyRunner(
        conversations_file=input_file,
        output_dir=output_dir,
        mistral_api_key=api_key,
        drift_threshold=0.8,
        branch_length=5,
        distance_scale=2.0
    )

    runner.run_all_studies()


if __name__ == '__main__':
    main()
