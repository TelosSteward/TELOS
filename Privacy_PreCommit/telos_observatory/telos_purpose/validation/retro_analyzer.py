"""
retrospective_analyzer.py

Analyze existing conversation transcripts with TELOS governance applied retroactively.
Uses foundations-aligned PrimacyAttractorMath with correct basin geometry and center weighting.
"""

from __future__ import annotations
import time
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from telos_purpose.core.primacy_math import (
    MathematicalState,
    PrimacyAttractorMath,
    TelicFidelityCalculator
)
from telos_purpose.core.intervention_controller import MathematicalInterventionController
from telos_purpose.core.unified_steward import PrimacyAttractor
from telos_purpose.core.embedding_provider import EmbeddingProvider


class RetrospectiveAnalyzer:
    """
    Apply TELOS governance retroactively using foundations-aligned mathematical components.

    Uses actual InterventionController logic and correct PrimacyAttractorMath formulas.
    """

    def __init__(
        self,
        governance_profile: PrimacyAttractor,
        embedding_provider: EmbeddingProvider
    ):
        """
        Initialize with foundations-aligned TELOS components.

        Args:
            governance_profile: PAP defining purpose/scope/boundaries
            embedding_provider: For computing semantic embeddings
        """
        self.governance_profile = governance_profile
        self.embedding_provider = embedding_provider
        self.fidelity_calc = TelicFidelityCalculator()

        # Build attractor math from governance profile
        purpose_text = " ".join(governance_profile.purpose)
        scope_text = " ".join(governance_profile.scope)

        purpose_vec = embedding_provider.encode(purpose_text)
        scope_vec = embedding_provider.encode(scope_text)

        # Uses foundations-aligned formulas: τ-weighted center, r=2/ρ radius
        self.attractor_math = PrimacyAttractorMath(
            purpose_vector=purpose_vec,
            scope_vector=scope_vec,
            privacy_level=governance_profile.privacy_level,
            constraint_tolerance=governance_profile.constraint_tolerance,
            task_priority=governance_profile.task_priority
        )

        # Use actual intervention controller (no LLM needed for analysis)
        self.intervention_controller = MathematicalInterventionController(
            attractor=self.attractor_math,
            llm_client=None,  # Not needed for retroactive analysis
            embedding_provider=embedding_provider,
            enable_interventions=False  # Evaluate only, don't execute
        )

    def analyze_conversation(
        self,
        transcript: List[Dict[str, str]],
        extract_timestamps: bool = False
    ) -> Dict[str, Any]:
        """
        Analyze conversation using foundations-aligned TELOS mathematical logic.

        Args:
            transcript: List of message dicts with 'role' and 'content'
            extract_timestamps: Try to extract timing from transcript metadata

        Returns:
            Dict with turn-by-turn telemetry and session metrics
        """
        states = []
        turn_results = []
        turn_num = 0

        # Build conversation history for intervention evaluation
        conversation_history = []

        for i, message in enumerate(transcript):
            conversation_history.append(message)

            # Only analyze assistant responses
            if message.get("role") != "assistant":
                continue

            turn_num += 1
            content = message["content"]
            timestamp = message.get("timestamp", time.time())

            # Compute embedding and create mathematical state
            embedding = self.embedding_provider.encode(content)
            state = MathematicalState(
                embedding=embedding,
                turn_number=turn_num,
                timestamp=timestamp,
                text_content=content
            )
            states.append(state)

            # Use actual intervention controller to evaluate
            intervention_eval = self.intervention_controller.process_turn(
                state=state,
                response_text=content,
                conversation_history=conversation_history,
                turn_number=turn_num
            )

            # Compute metrics using foundations-aligned attractor math
            error_signal = self.attractor_math.compute_error_signal(state)
            in_basin = self.attractor_math.compute_basin_membership(state)
            lyapunov = self.attractor_math.compute_lyapunov_function(state)

            # Calculate Lyapunov delta
            lyapunov_delta = 0.0
            if len(states) > 1:
                prev_lyapunov = self.attractor_math.compute_lyapunov_function(states[-2])
                lyapunov_delta = lyapunov - prev_lyapunov

            # Compute cumulative fidelity
            current_fidelity = self.fidelity_calc.compute_hard_fidelity(states, self.attractor_math)
            soft_fidelity = self.fidelity_calc.compute_soft_fidelity(states, self.attractor_math)

            # Extract intervention decision
            would_intervene = intervention_eval["intervention_applied"]
            intervention_result = intervention_eval["intervention_result"]

            intervention_type = None
            intervention_strength = 0.0
            if intervention_result:
                intervention_type = intervention_result.type
                intervention_strength = intervention_result.strength

            # Raw distance from attractor center
            distance = float(np.linalg.norm(embedding - self.attractor_math.attractor_center))

            turn_results.append({
                "turn_id": turn_num,
                "timestamp": timestamp,
                "content_preview": content[:200] + "..." if len(content) > 200 else content,
                "embedding_distance": round(distance, 4),
                "fidelity_score": round(current_fidelity, 4),
                "soft_fidelity": round(soft_fidelity, 4),
                "error_signal": round(error_signal, 4),
                "lyapunov_value": round(lyapunov, 4),
                "lyapunov_delta": round(lyapunov_delta, 4),
                "in_basin": in_basin,
                "would_intervene": would_intervene,
                "intervention_type": intervention_type,
                "intervention_strength": round(intervention_strength, 4) if intervention_strength else 0.0,
                "governance_drift_flag": not in_basin or error_signal > self.intervention_controller.epsilon_min,
                "is_meta": intervention_eval.get("is_meta", False)
            })

        # Compute final session metrics
        final_hard_fidelity = self.fidelity_calc.compute_hard_fidelity(states, self.attractor_math)
        final_soft_fidelity = self.fidelity_calc.compute_soft_fidelity(states, self.attractor_math)
        trajectory_stability = self.fidelity_calc.compute_trajectory_stability(states, self.attractor_math)

        intervention_stats = self.intervention_controller.get_intervention_statistics()

        distances = [r["embedding_distance"] for r in turn_results]
        error_signals = [r["error_signal"] for r in turn_results]

        return {
            "analysis_type": "retrospective",
            "session_id": f"retro_{int(time.time())}",
            "math_version": "foundations_aligned_v2",
            "governance_profile": {
                "purpose": self.governance_profile.purpose,
                "scope": self.governance_profile.scope,
                "boundaries": self.governance_profile.boundaries,
                "constraint_tolerance": self.governance_profile.constraint_tolerance,
                "privacy_level": self.governance_profile.privacy_level,
                "task_priority": self.governance_profile.task_priority
            },
            "intervention_thresholds": {
                "epsilon_min": self.intervention_controller.epsilon_min,
                "epsilon_max": self.intervention_controller.epsilon_max,
                "basin_radius": self.attractor_math.basin_radius,
                "constraint_rigidity": self.attractor_math.constraint_rigidity
            },
            "turn_results": turn_results,
            "session_metrics": {
                "total_turns": len(turn_results),
                "final_hard_fidelity": round(final_hard_fidelity, 4),
                "final_soft_fidelity": round(final_soft_fidelity, 4),
                "trajectory_stability": round(trajectory_stability, 4),
                "avg_hard_fidelity": round(np.mean([r["fidelity_score"] for r in turn_results]), 4),
                "min_fidelity": round(min(r["fidelity_score"] for r in turn_results), 4),
                "max_fidelity": round(max(r["fidelity_score"] for r in turn_results), 4),
                "avg_distance": round(np.mean(distances), 4),
                "avg_error_signal": round(np.mean(error_signals), 4),
                "basin_adherence": sum(r["in_basin"] for r in turn_results) / len(turn_results),
                "would_intervene_count": sum(r["would_intervene"] for r in turn_results),
                "intervention_rate": sum(r["would_intervene"] for r in turn_results) / len(turn_results),
                "drift_events": sum(r["governance_drift_flag"] for r in turn_results),
                "drift_rate": sum(r["governance_drift_flag"] for r in turn_results) / len(turn_results),
                "meta_events": sum(r["is_meta"] for r in turn_results),
                "lyapunov_convergent_turns": sum(1 for r in turn_results if r["lyapunov_delta"] < 0),
                "lyapunov_divergent_turns": sum(1 for r in turn_results if r["lyapunov_delta"] > 0)
            },
            "intervention_breakdown": intervention_stats
        }

    def export_as_baseline_csv(self, results: Dict[str, Any], output_path: Path):
        """Export results in validation template CSV format."""
        import csv

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "session_id", "condition", "turn_id", "timestamp",
                "delta_t_ms", "user_input", "model_output",
                "embedding_distance", "fidelity_score", "lyapunov_delta",
                "intervention_triggered", "intervention_type",
                "governance_drift_flag", "governance_correction_applied", "notes"
            ])

            session_id = results["session_id"]
            for turn in results["turn_results"]:
                writer.writerow([
                    session_id, "stateless_retro", turn["turn_id"], turn["timestamp"],
                    0, "", turn["content_preview"],
                    turn["embedding_distance"], turn["fidelity_score"], turn["lyapunov_delta"],
                    turn["would_intervene"], turn["intervention_type"] or "none",
                    turn["governance_drift_flag"], False,
                    f"Error={turn['error_signal']:.3f}, Basin={turn['in_basin']}, Meta={turn['is_meta']}, MathV2"
                ])

    def export_summary_json(self, results: Dict[str, Any], output_path: Path):
        """Export session summary as JSON matching template."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        summary = {
            "session_metadata": {
                "session_id": results["session_id"],
                "condition": "stateless_retrospective",
                "analysis_type": "retrospective",
                "math_version": "foundations_aligned_v2",
                "governance_profile_id": "retro_pap",
                "constraint_tolerance": results["governance_profile"]["constraint_tolerance"],
                "purpose_prompt": " | ".join(results["governance_profile"]["purpose"]),
                "observer_mode": False,
                "intervention_mode": "none",
                "runtime_version": "retro_v2"
            },
            "intervention_thresholds": results["intervention_thresholds"],
            "session_metrics": results["session_metrics"],
            "intervention_breakdown": results["intervention_breakdown"],
            "turn_summary": [
                {
                    "turn": t["turn_id"],
                    "fidelity": t["fidelity_score"],
                    "soft_fidelity": t["soft_fidelity"],
                    "drift": t["governance_drift_flag"],
                    "intervention": t["would_intervene"]
                }
                for t in results["turn_results"]
            ]
        }

        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)


def analyze_transcript_file(
    transcript_path: str,
    governance_profile: PrimacyAttractor,
    embedding_provider: EmbeddingProvider,
    output_dir: str = "./validation_results/stateless"
) -> Dict[str, Any]:
    """
    Convenience function: analyze transcript and export results.
    Uses foundations-aligned TELOS mathematical components.
    """
    with open(transcript_path) as f:
        transcript = json.load(f)

    analyzer = RetrospectiveAnalyzer(governance_profile, embedding_provider)
    results = analyzer.analyze_conversation(transcript)

    output_path = Path(output_dir)
    session_id = results["session_id"]

    analyzer.export_as_baseline_csv(
        results,
        output_path / f"{session_id}_turns.csv"
    )

    analyzer.export_summary_json(
        results,
        output_path / f"{session_id}_summary.json"
    )

    print(f"\nRetrospective Analysis Complete: {session_id}")
    print(f"  Math Version: {results['math_version']}")
    print(f"  Final Hard Fidelity: {results['session_metrics']['final_hard_fidelity']:.3f}")
    print(f"  Final Soft Fidelity: {results['session_metrics']['final_soft_fidelity']:.3f}")
    print(f"  Trajectory Stability: {results['session_metrics']['trajectory_stability']:.3f}")
    print(f"  Would Intervene: {results['session_metrics']['would_intervene_count']} times")
    print(f"  Intervention Rate: {results['session_metrics']['intervention_rate']:.2%}")
    print(f"  Basin Radius: {results['intervention_thresholds']['basin_radius']:.3f}")
    print(f"  Thresholds: ε_min={results['intervention_thresholds']['epsilon_min']:.3f}, ε_max={results['intervention_thresholds']['epsilon_max']:.3f}")
    print(f"  Results saved to: {output_path}\n")

    return results


if __name__ == "__main__":
    from telos_purpose.core.embedding_provider import DeterministicEmbeddingProvider

    profile = PrimacyAttractor(
        purpose=["Execute TELOS validation roadmap", "Provide technical implementation guidance"],
        scope=["Embedding validity testing", "Pilot study design", "Documentation improvement"],
        boundaries=["Do not re-litigate strategic framing", "Stay focused on execution"],
        constraint_tolerance=0.2,
        privacy_level=0.8,
        task_priority=0.9
    )

    results = analyze_transcript_file(
        "path/to/conversation_transcript.json",
        profile,
        DeterministicEmbeddingProvider(),
        "./validation_results/stateless"
    )
