#!/usr/bin/env python3
"""
TELOS Validation Study Runner
Runs counterfactual validation tests across all baseline conversations.
Compares 5 baseline conditions to demonstrate TELOS efficacy.
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.validation.baseline_runners import (
    StatelessRunner,
    PromptOnlyRunner,
    CadenceReminderRunner,
    ObservationRunner,
    TELOSRunner,
    BaselineResult
)
from tests.validation.comparative_test import ComparativeValidator
from telos.core.unified_steward import PrimacyAttractor


class ValidationStudyRunner:
    """Orchestrates comprehensive validation study across all baseline conversations."""

    def __init__(self, data_dir: str = "tests/validation_data/baseline_conversations"):
        self.data_dir = Path(data_dir)
        self.results_dir = Path("tests/validation_results")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Track overall statistics
        self.overall_stats = {
            "total_conversations": 0,
            "total_turns": 0,
            "baseline_avg_fidelity": {},
            "improvement_over_baseline": {},
            "conversations_processed": []
        }

    def load_conversations(self) -> List[Dict[str, Any]]:
        """Load all baseline conversations from data directory."""
        conversations = []

        for json_file in sorted(self.data_dir.glob("*.json")):
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                conversations.append(data)
                print(f"Loaded: {json_file.name}")

        return conversations

    def extract_conversation_turns(self, session_data: Dict) -> List[tuple]:
        """Extract user-assistant pairs from session data."""
        turns = []

        for turn in session_data.get('turns', []):
            user_input = turn.get('user_input', turn.get('user_message', ''))
            assistant_response = turn.get('response', turn.get('assistant_response_native', ''))

            if user_input and assistant_response:
                turns.append((user_input, assistant_response))

        return turns

    def extract_primacy_attractor(self, session_data: Dict) -> PrimacyAttractor:
        """Extract primacy attractor configuration from session data."""
        pa_data = session_data.get('primacy_attractor', {})

        return PrimacyAttractor(
            purpose=pa_data.get('purpose', ['General conversation']),
            scope=pa_data.get('scope', ['Open-ended dialogue']),
            boundaries=pa_data.get('boundaries', ['Maintain respectful discourse'])
        )

    def run_single_validation(self, session_data: Dict) -> Dict[str, Any]:
        """Run validation on a single conversation."""
        session_id = session_data.get('session_id', 'unknown')
        print(f"\n{'='*60}")
        print(f"Running validation for: {session_id}")
        print(f"{'='*60}")

        # Extract conversation and PA
        conversation = self.extract_conversation_turns(session_data)
        attractor_config = self.extract_primacy_attractor(session_data)

        if not conversation:
            print(f"Warning: No valid turns found in {session_id}")
            return None

        print(f"Conversation length: {len(conversation)} turns")
        print(f"Primacy Attractor: {attractor_config.purpose[0][:50]}...")

        # Initialize with REAL API clients
        from mistralai import Mistral
        from sentence_transformers import SentenceTransformer

        # Load API key
        with open('.streamlit/secrets.toml', 'r') as f:
            content = f.read()
            for line in content.split('\n'):
                if 'MISTRAL_API_KEY' in line:
                    api_key = line.split('=')[1].strip().strip('"')
                    break

        # Create real clients
        mistral_client = Mistral(api_key=api_key)
        embedder = SentenceTransformer('all-MiniLM-L6-v2')

        # Wrapper to match expected interface
        class LLMClient:
            def generate(self, prompt, max_tokens=500):
                response = mistral_client.chat.complete(
                    model='mistral-small-2501',
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content

        class EmbeddingProvider:
            def embed(self, text):
                return embedder.encode(text)

        validator = ComparativeValidator(
            llm_client=LLMClient(),
            embedding_provider=EmbeddingProvider(),
            output_dir=str(self.results_dir / session_id)
        )

        # Run comparative study
        try:
            results = validator.run_comparative_study(
                conversation=conversation,
                attractor_config=attractor_config,
                study_id=session_id
            )

            # Save individual results
            output_file = self.results_dir / f"{session_id}_results.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)

            print(f"Results saved to: {output_file}")
            return results

        except Exception as e:
            print(f"Error processing {session_id}: {e}")
            return None

    def run_full_study(self):
        """Run validation study across all baseline conversations."""
        print("\n" + "="*70)
        print("TELOS VALIDATION STUDY")
        print("="*70)

        # Load conversations
        conversations = self.load_conversations()
        self.overall_stats["total_conversations"] = len(conversations)

        print(f"\nLoaded {len(conversations)} baseline conversations")
        print("Starting validation tests...\n")

        # Process each conversation
        all_results = []
        for i, session_data in enumerate(conversations, 1):
            print(f"\nProcessing {i}/{len(conversations)}...")

            result = self.run_single_validation(session_data)
            if result:
                all_results.append(result)
                self.overall_stats["conversations_processed"].append(
                    session_data.get('session_id')
                )

        # Calculate aggregate statistics
        self.calculate_aggregate_stats(all_results)

        # Generate summary report
        self.generate_summary_report(all_results)

        print("\n" + "="*70)
        print("VALIDATION STUDY COMPLETE")
        print("="*70)

    def calculate_aggregate_stats(self, results: List[Dict]):
        """Calculate aggregate statistics across all validation results."""
        if not results:
            return

        # Aggregate by baseline type
        baseline_types = ['stateless', 'prompt_only', 'cadence', 'observation', 'telos']

        for baseline_type in baseline_types:
            fidelities = []

            for result in results:
                if baseline_type in result:
                    fidelity = result[baseline_type].get('final_metrics', {}).get('fidelity', 0)
                    fidelities.append(fidelity)

            if fidelities:
                self.overall_stats["baseline_avg_fidelity"][baseline_type] = np.mean(fidelities)

        # Calculate improvement percentages
        if 'stateless' in self.overall_stats["baseline_avg_fidelity"]:
            baseline_fidelity = self.overall_stats["baseline_avg_fidelity"]['stateless']

            for baseline_type in baseline_types:
                if baseline_type in self.overall_stats["baseline_avg_fidelity"]:
                    current_fidelity = self.overall_stats["baseline_avg_fidelity"][baseline_type]
                    improvement = ((current_fidelity - baseline_fidelity) / baseline_fidelity) * 100
                    self.overall_stats["improvement_over_baseline"][baseline_type] = improvement

    def generate_summary_report(self, results: List[Dict]):
        """Generate comprehensive summary report."""
        report_file = self.results_dir / f"validation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        summary = {
            "study_metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_conversations": self.overall_stats["total_conversations"],
                "conversations_processed": len(self.overall_stats["conversations_processed"]),
                "success_rate": len(self.overall_stats["conversations_processed"]) / self.overall_stats["total_conversations"] * 100
            },
            "aggregate_results": {
                "average_fidelity_by_baseline": self.overall_stats["baseline_avg_fidelity"],
                "improvement_over_stateless": self.overall_stats["improvement_over_baseline"]
            },
            "key_findings": {
                "telos_improvement": self.overall_stats["improvement_over_baseline"].get('telos', 0),
                "best_performer": max(self.overall_stats["baseline_avg_fidelity"],
                                     key=self.overall_stats["baseline_avg_fidelity"].get),
                "conversations_analyzed": self.overall_stats["conversations_processed"]
            }
        }

        with open(report_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n📊 Summary Report Generated: {report_file}")
        print("\n" + "="*50)
        print("KEY RESULTS:")
        print("="*50)

        for baseline, fidelity in self.overall_stats["baseline_avg_fidelity"].items():
            improvement = self.overall_stats["improvement_over_baseline"].get(baseline, 0)
            print(f"{baseline.upper():15} Avg Fidelity: {fidelity:.3f} | Improvement: {improvement:+.1f}%")

        print("="*50)


def main():
    """Main entry point for validation study."""
    print("🚀 Starting TELOS Validation Study...")

    runner = ValidationStudyRunner()
    runner.run_full_study()

    print("\n✅ Validation Study Complete!")


if __name__ == "__main__":
    main()