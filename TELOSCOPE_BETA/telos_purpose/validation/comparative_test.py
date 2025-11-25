"""
Comparative validation framework.
Runs identical conversations through all four baselines and compares outcomes.
"""

from __future__ import annotations
import json
import time
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from telos_purpose.validation.baseline_runners import (
    StatelessRunner,
    PromptOnlyRunner,
    CadenceReminderRunner,
    TELOSRunner,
    BaselineResult
)
from telos_purpose.core.unified_steward import PrimacyAttractor


class ComparativeValidator:
    """
    Run comparative validation across all baselines.
    Tests hypothesis: TELOS interventions reduce drift more than simpler alternatives.
    """

    def __init__(
        self,
        llm_client,
        embedding_provider,
        output_dir: str = "./validation_results"
    ):
        self.llm = llm_client
        self.embedding_provider = embedding_provider
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_comparative_study(
        self,
        conversation: List[tuple],
        attractor_config: PrimacyAttractor,
        study_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run identical conversation through all four baselines.

        Returns:
            Comparative results with statistical analysis
        """
        study_id = study_id or f"study_{int(time.time())}"

        print(f"\n{'='*60}")
        print(f"COMPARATIVE VALIDATION STUDY: {study_id}")
        print(f"{'='*60}\n")
        print(f"Conversation: {len(conversation)} turns")
        print(f"Testing 4 conditions...\n")

        results = {}

        print("Running Stateless baseline...")
        stateless = StatelessRunner(
            self.llm, self.embedding_provider, attractor_config
        )
        results['stateless'] = stateless.run_conversation(conversation)
        print(f"  Final fidelity: {results['stateless'].final_metrics['fidelity']:.3f}\n")

        print("Running Prompt-Only baseline...")
        prompt_only = PromptOnlyRunner(
            self.llm, self.embedding_provider, attractor_config
        )
        results['prompt_only'] = prompt_only.run_conversation(conversation)
        print(f"  Final fidelity: {results['prompt_only'].final_metrics['fidelity']:.3f}\n")

        print("Running Cadence-Reminder baseline...")
        cadence = CadenceReminderRunner(
            self.llm, self.embedding_provider, attractor_config, reminder_cadence=3
        )
        results['cadence'] = cadence.run_conversation(conversation)
        print(f"  Final fidelity: {results['cadence'].final_metrics['fidelity']:.3f}\n")

        print("Running TELOS (full interventions)...")
        telos = TELOSRunner(
            self.llm, self.embedding_provider, attractor_config
        )
        results['telos'] = telos.run_conversation(conversation)
        print(f"  Final fidelity: {results['telos'].final_metrics['fidelity']:.3f}\n")

        analysis = self._analyze_results(results)

        output_path = self.output_dir / f"{study_id}.json"
        self._save_results(study_id, results, analysis, output_path)

        print(f"\n{'='*60}")
        print("COMPARATIVE ANALYSIS")
        print(f"{'='*60}\n")
        self._print_analysis(analysis)
        print(f"\nResults saved to: {output_path}")

        return {
            "study_id": study_id,
            "results": {k: self._result_to_dict(v) for k, v in results.items()},
            "analysis": analysis,
            "output_path": str(output_path)
        }

    def _analyze_results(
        self,
        results: Dict[str, BaselineResult]
    ) -> Dict[str, Any]:
        """Perform statistical comparison of baseline results."""
        analysis = {
            "fidelity_comparison": {},
            "drift_comparison": {},
            "deltas_vs_telos": {},
            "hypothesis_tests": {}
        }

        fidelities = {k: v.final_metrics['fidelity'] for k, v in results.items()}
        avg_distances = {k: v.final_metrics['avg_distance'] for k, v in results.items()}
        basin_adherence = {k: v.final_metrics['basin_adherence'] for k, v in results.items()}

        analysis["fidelity_comparison"] = fidelities
        analysis["drift_comparison"] = avg_distances
        analysis["basin_adherence"] = basin_adherence

        telos_fidelity = fidelities['telos']
        for condition in ['stateless', 'prompt_only', 'cadence']:
            delta = telos_fidelity - fidelities[condition]
            analysis["deltas_vs_telos"][condition] = {
                "fidelity_improvement": delta,
                "percentage_improvement": (delta / fidelities[condition] * 100) if fidelities[condition] > 0 else 0
            }

        best_delta = max(
            analysis["deltas_vs_telos"][c]["fidelity_improvement"]
            for c in ['stateless', 'prompt_only', 'cadence']
        )

        analysis["hypothesis_tests"]["H1_minimum_improvement"] = {
            "threshold": 0.15,
            "best_delta": best_delta,
            "passes_threshold": best_delta >= 0.15
        }

        best_condition = max(fidelities.items(), key=lambda x: x[1])
        analysis["hypothesis_tests"]["H2_telos_best"] = {
            "best_condition": best_condition[0],
            "telos_is_best": best_condition[0] == 'telos'
        }

        analysis["effect_sizes"] = {}
        for condition in ['stateless', 'prompt_only', 'cadence']:
            mean_diff = telos_fidelity - fidelities[condition]
            pooled_std = 0.1
            cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0

            analysis["effect_sizes"][condition] = {
                "cohens_d": cohens_d,
                "interpretation": self._interpret_effect_size(cohens_d)
            }

        return analysis

    def _interpret_effect_size(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"

    def _print_analysis(self, analysis: Dict[str, Any]) -> None:
        """Print human-readable analysis."""
        print("Fidelity Scores:")
        for condition, fidelity in analysis["fidelity_comparison"].items():
            print(f"  {condition:20s}: {fidelity:.3f}")

        print("\nImprovement vs Baselines:")
        for condition, delta_info in analysis["deltas_vs_telos"].items():
            improvement = delta_info["fidelity_improvement"]
            pct = delta_info["percentage_improvement"]
            print(f"  TELOS vs {condition:15s}: +{improvement:.3f} ({pct:+.1f}%)")

        print("\nHypothesis Testing:")
        h1 = analysis["hypothesis_tests"]["H1_minimum_improvement"]
        print(f"  H1 (ΔF > 0.15): {'PASS' if h1['passes_threshold'] else 'FAIL'}")
        print(f"     Best delta: {h1['best_delta']:.3f}")

        h2 = analysis["hypothesis_tests"]["H2_telos_best"]
        print(f"  H2 (TELOS best): {'PASS' if h2['telos_is_best'] else 'FAIL'}")
        print(f"     Winner: {h2['best_condition']}")

        print("\nEffect Sizes (Cohen's d):")
        for condition, effect in analysis["effect_sizes"].items():
            d = effect["cohens_d"]
            interp = effect["interpretation"]
            print(f"  vs {condition:15s}: d={d:.2f} ({interp})")

    def _result_to_dict(self, result: BaselineResult) -> Dict[str, Any]:
        """Convert BaselineResult to dict."""
        return {
            "runner_type": result.runner_type,
            "session_id": result.session_id,
            "turn_results": result.turn_results,
            "final_metrics": result.final_metrics,
            "metadata": result.metadata
        }

    def _save_results(
        self,
        study_id: str,
        results: Dict[str, BaselineResult],
        analysis: Dict[str, Any],
        output_path: Path
    ) -> None:
        """Save results to JSON."""
        output_data = {
            "study_id": study_id,
            "timestamp": time.time(),
            "conditions": {
                condition: self._result_to_dict(result)
                for condition, result in results.items()
            },
            "comparative_analysis": analysis
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
