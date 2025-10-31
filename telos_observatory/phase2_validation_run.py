#!/usr/bin/env python3
"""
Phase 2 Validation Run - Unified Study Pipeline
================================================

Runs complete Phase 2 TELOS validation studies on any dataset:
1. LLM-at-every-turn primacy attractor establishment
2. Statistical convergence detection
3. Drift monitoring
4. Counterfactual branching
5. Evidence generation
6. Research brief creation

This is the unified, reusable pipeline for Phase 2 validation.

Usage:
    python phase2_validation_run.py <input_file> <output_name>
    python phase2_validation_run.py --sharegpt     # Run on ShareGPT data
    python phase2_validation_run.py --test         # Run on internal test data
    python phase2_validation_run.py --all          # Run on all datasets

Examples:
    python phase2_validation_run.py --sharegpt
    python phase2_validation_run.py --test
    python phase2_validation_run.py data.json my_study
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from telos_observatory.run_phase2_study import Phase2StudyRunner
from telos_observatory.generate_research_briefs import ResearchBriefGenerator


class Phase2ValidationPipeline:
    """
    Complete Phase 2 validation pipeline.

    Wraps Phase2StudyRunner and ResearchBriefGenerator into a single
    reusable command.
    """

    def __init__(self, api_key: str):
        """Initialize pipeline with API key."""
        self.api_key = api_key
        self.base_dir = Path(__file__).parent

    def run_validation(
        self,
        input_file: Path,
        study_name: str,
        max_studies: Optional[int] = None
    ):
        """
        Run complete Phase 2 validation pipeline.

        Args:
            input_file: Path to ShareGPT-formatted conversations JSON
            study_name: Name for this study run (used in output paths)
            max_studies: Optional limit on number of conversations to process
        """
        print("=" * 70)
        print(f"PHASE 2 VALIDATION RUN: {study_name.upper()}")
        print("=" * 70)
        print(f"Input: {input_file}")
        print(f"Timestamp: {datetime.now().isoformat()}")
        print("=" * 70)

        # Create output directory
        output_dir = self.base_dir / f'phase2_validation_{study_name}'
        output_dir.mkdir(exist_ok=True, parents=True)

        # ============================================================
        # STEP 1: RUN PHASE 2 STUDIES
        # ============================================================

        print("\n" + "=" * 70)
        print("STEP 1: RUNNING PHASE 2 STUDIES")
        print("=" * 70)

        runner = Phase2StudyRunner(
            conversations_file=input_file,
            output_dir=output_dir / 'study_results',
            mistral_api_key=self.api_key,
            drift_threshold=0.8,
            branch_length=5,
            distance_scale=2.0
        )

        runner.run_all_studies(max_studies=max_studies)

        summary_path = output_dir / 'study_results' / 'phase2_study_summary.json'

        if not summary_path.exists():
            print("\n❌ ERROR: Study summary not generated")
            return False

        # ============================================================
        # STEP 2: GENERATE RESEARCH BRIEFS
        # ============================================================

        print("\n" + "=" * 70)
        print("STEP 2: GENERATING RESEARCH BRIEFS")
        print("=" * 70)

        briefs_dir = output_dir / 'research_briefs'
        briefs_dir.mkdir(exist_ok=True, parents=True)

        generator = ResearchBriefGenerator(
            summary_path=summary_path,
            output_dir=briefs_dir
        )

        generator.generate_all_briefs()

        # ============================================================
        # STEP 3: FINAL SUMMARY
        # ============================================================

        print("\n" + "=" * 70)
        print("VALIDATION RUN COMPLETE")
        print("=" * 70)

        with open(summary_path) as f:
            summary = json.load(f)

        completed = len(summary.get('completed_studies', []))
        failed = len(summary.get('failed_studies', []))
        total = completed + failed

        print(f"\nResults:")
        print(f"  Total conversations: {total}")
        print(f"  Completed: {completed}")
        print(f"  Failed: {failed}")

        if completed > 0:
            with_drift = summary['summary'].get('with_drift', 0)
            without_drift = summary['summary'].get('without_drift', 0)

            print(f"\nDrift Analysis:")
            print(f"  Drift detected: {with_drift}")
            print(f"  No drift: {without_drift}")

            # Calculate effectiveness if we have drift cases
            completed_studies = summary.get('completed_studies', [])
            drift_studies = [s for s in completed_studies if s.get('drift_detected')]

            if drift_studies:
                delta_f_values = [
                    s['counterfactual_results']['delta_f']
                    for s in drift_studies
                ]
                avg_delta_f = sum(delta_f_values) / len(delta_f_values)
                effective_count = sum(1 for df in delta_f_values if df > 0)

                print(f"\nGovernance Effectiveness:")
                print(f"  Average ΔF: {avg_delta_f:+.3f}")
                print(f"  Effective: {effective_count}/{len(drift_studies)} ({100*effective_count/len(drift_studies):.1f}%)")

        print(f"\nOutput Directory: {output_dir}")
        print(f"  Study results: {output_dir / 'study_results'}")
        print(f"  Research briefs: {briefs_dir}")
        print()

        return True


def main():
    """Main entry point."""

    # Check API key
    api_key = os.environ.get('MISTRAL_API_KEY')
    if not api_key:
        print("❌ ERROR: MISTRAL_API_KEY environment variable not set")
        print("\nSet it with:")
        print("  export MISTRAL_API_KEY='your-key-here'")
        sys.exit(1)

    # Parse arguments
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python phase2_validation_run.py <input_file> <output_name>")
        print("  python phase2_validation_run.py --sharegpt")
        print("  python phase2_validation_run.py --test")
        print("  python phase2_validation_run.py --all")
        sys.exit(1)

    pipeline = Phase2ValidationPipeline(api_key=api_key)
    base_dir = Path(__file__).parent

    arg = sys.argv[1]

    # Preset datasets
    if arg == '--sharegpt':
        input_file = base_dir / 'sharegpt_data' / 'sharegpt_filtered_conversations.json'
        study_name = 'sharegpt'
        pipeline.run_validation(input_file, study_name)

    elif arg == '--test':
        # Run both test_sessions and edge_cases
        print("\n🧪 Running internal test data validation...\n")

        # Test sessions
        test_sessions_file = base_dir / 'test_data_converted' / 'test_sessions_sharegpt.json'
        if test_sessions_file.exists():
            print("\n" + "=" * 70)
            print("RUNNING: TEST_SESSIONS")
            print("=" * 70)
            pipeline.run_validation(test_sessions_file, 'test_sessions')

        # Edge cases
        edge_cases_file = base_dir / 'test_data_converted' / 'edge_cases_sharegpt.json'
        if edge_cases_file.exists():
            print("\n" + "=" * 70)
            print("RUNNING: EDGE_CASES")
            print("=" * 70)
            pipeline.run_validation(edge_cases_file, 'edge_cases')

    elif arg == '--all':
        print("\n🌐 Running all datasets...\n")

        # ShareGPT
        sharegpt_file = base_dir / 'sharegpt_data' / 'sharegpt_filtered_conversations.json'
        if sharegpt_file.exists():
            print("\n" + "=" * 70)
            print("RUNNING: SHAREGPT")
            print("=" * 70)
            pipeline.run_validation(sharegpt_file, 'sharegpt')

        # Test sessions
        test_sessions_file = base_dir / 'test_data_converted' / 'test_sessions_sharegpt.json'
        if test_sessions_file.exists():
            print("\n" + "=" * 70)
            print("RUNNING: TEST_SESSIONS")
            print("=" * 70)
            pipeline.run_validation(test_sessions_file, 'test_sessions')

        # Edge cases
        edge_cases_file = base_dir / 'test_data_converted' / 'edge_cases_sharegpt.json'
        if edge_cases_file.exists():
            print("\n" + "=" * 70)
            print("RUNNING: EDGE_CASES")
            print("=" * 70)
            pipeline.run_validation(edge_cases_file, 'edge_cases')

    else:
        # Custom file
        input_file = Path(arg)
        if not input_file.exists():
            print(f"❌ ERROR: File not found: {input_file}")
            sys.exit(1)

        if len(sys.argv) < 3:
            study_name = input_file.stem
        else:
            study_name = sys.argv[2]

        pipeline.run_validation(input_file, study_name)


if __name__ == '__main__':
    main()
