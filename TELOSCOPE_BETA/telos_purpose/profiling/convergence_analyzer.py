"""
Convergence Analyzer for Progressive Primacy Attractors
========================================================

Analyzes convergence patterns across multiple conversation sessions
to derive optimal parameters and provide statistical evidence.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
from scipy import stats
import json


@dataclass
class ConvergenceRecord:
    """Record of a single conversation's convergence."""
    session_id: str
    convergence_turn: int
    total_turns: int
    confidence_score: float
    centroid_stability: float
    variance_stability: float
    convergence_time: float  # Time to converge in seconds
    attractor_quality: Dict[str, Any]  # Purpose/scope/boundaries metadata


class ConvergenceAnalyzer:
    """
    Analyze convergence patterns across multiple sessions to derive
    optimal parameters and generate statistical evidence.
    """

    def __init__(self):
        self.records: List[ConvergenceRecord] = []

    def add_record(self, record: ConvergenceRecord):
        """Add a convergence record from a session."""
        self.records.append(record)

    def compute_statistics(self) -> Dict[str, Any]:
        """
        Compute comprehensive statistics across all recorded sessions.

        Returns:
            Dict with:
            - convergence_turns: {mean, std, median, min, max, percentiles}
            - confidence_scores: {mean, std, distribution}
            - stability_metrics: {centroid_mean, variance_mean}
            - convergence_rate: percentage that converged
            - quality_metrics: attractor quality statistics
        """
        if not self.records:
            return {'error': 'No records available'}

        turns = [r.convergence_turn for r in self.records if r.convergence_turn > 0]
        confidences = [r.confidence_score for r in self.records]
        centroid_stabilities = [r.centroid_stability for r in self.records]
        variance_stabilities = [r.variance_stability for r in self.records]

        if not turns:
            return {'error': 'No successful convergences'}

        # Convergence turn statistics
        turn_stats = {
            'mean': float(np.mean(turns)),
            'std': float(np.std(turns)),
            'median': float(np.median(turns)),
            'min': int(np.min(turns)),
            'max': int(np.max(turns)),
            'percentile_25': float(np.percentile(turns, 25)),
            'percentile_75': float(np.percentile(turns, 75)),
            'percentile_90': float(np.percentile(turns, 90)),
            'percentile_95': float(np.percentile(turns, 95)),
        }

        # Confidence interval (95%)
        if len(turns) > 1:
            ci = stats.t.interval(
                0.95,
                len(turns) - 1,
                loc=np.mean(turns),
                scale=stats.sem(turns)
            )
            turn_stats['ci_95_lower'] = float(ci[0])
            turn_stats['ci_95_upper'] = float(ci[1])

        # Convergence rate
        convergence_rate = len(turns) / len(self.records) if self.records else 0

        # Stability metrics
        stability = {
            'centroid_mean': float(np.mean(centroid_stabilities)),
            'centroid_std': float(np.std(centroid_stabilities)),
            'variance_mean': float(np.mean(variance_stabilities)),
            'variance_std': float(np.std(variance_stabilities)),
        }

        # Confidence scores
        confidence = {
            'mean': float(np.mean(confidences)),
            'std': float(np.std(confidences)),
            'min': float(np.min(confidences)),
            'max': float(np.max(confidences)),
        }

        return {
            'n_sessions': len(self.records),
            'n_converged': len(turns),
            'convergence_rate': convergence_rate,
            'convergence_turns': turn_stats,
            'confidence_scores': confidence,
            'stability_metrics': stability,
        }

    def recommend_parameters(self) -> Dict[str, Any]:
        """
        Generate data-driven parameter recommendations.

        Returns:
            Dict with:
            - recommended_window_size: Based on typical convergence
            - recommended_confidence_threshold: Based on distribution
            - recommended_stability_threshold: Based on observed stability
            - evidence: Supporting statistics
        """
        stats = self.compute_statistics()

        if 'error' in stats:
            return {'error': stats['error']}

        # Recommended window size: Use median + 1 std as safe default
        # This ensures we don't cut off too early for most conversations
        median_turns = stats['convergence_turns']['median']
        std_turns = stats['convergence_turns']['std']
        recommended_window = int(np.ceil(median_turns + std_turns))

        # Recommended confidence threshold: Use mean - 0.5*std
        # This ensures we're confident but not overly strict
        mean_confidence = stats['confidence_scores']['mean']
        std_confidence = stats['confidence_scores']['std']
        recommended_confidence = max(0.7, mean_confidence - 0.5 * std_confidence)

        # Recommended stability thresholds based on observed data
        centroid_stability = stats['stability_metrics']['centroid_mean']
        variance_stability = stats['stability_metrics']['variance_mean']

        return {
            'recommended_window_size': recommended_window,
            'recommended_confidence_threshold': round(recommended_confidence, 2),
            'recommended_centroid_threshold': round(centroid_stability * 1.1, 4),
            'recommended_variance_threshold': round(variance_stability * 1.1, 4),
            'evidence': {
                'median_convergence': median_turns,
                'std_convergence': std_turns,
                'convergence_rate': stats['convergence_rate'],
                'n_sessions': stats['n_sessions'],
            },
            'interpretation': f"""
Based on {stats['n_sessions']} sessions ({stats['convergence_rate']:.1%} convergence rate):
- Typical convergence: {median_turns:.0f} turns (median)
- Recommended window: {recommended_window} turns (median + 1 std)
- Confidence threshold: {recommended_confidence:.2f} (mean - 0.5 std)

This provides a data-driven parameter set that adapts to your conversation patterns.
            """.strip()
        }

    def generate_distribution_data(self) -> Dict[str, List[int]]:
        """
        Generate distribution data for visualization.

        Returns:
            Dict with:
            - turns: List of convergence turns
            - bins: Histogram bins
            - counts: Histogram counts
        """
        turns = [r.convergence_turn for r in self.records if r.convergence_turn > 0]

        if not turns:
            return {'turns': [], 'bins': [], 'counts': []}

        # Create histogram
        counts, bins = np.histogram(turns, bins='auto')

        return {
            'turns': turns,
            'bins': bins.tolist(),
            'counts': counts.tolist(),
        }

    def export_report(self, output_path: str):
        """
        Export comprehensive analysis report to JSON.

        Args:
            output_path: Path to output JSON file
        """
        report = {
            'statistics': self.compute_statistics(),
            'recommendations': self.recommend_parameters(),
            'distribution': self.generate_distribution_data(),
            'records': [
                {
                    'session_id': r.session_id,
                    'convergence_turn': r.convergence_turn,
                    'total_turns': r.total_turns,
                    'confidence': r.confidence_score,
                    'centroid_stability': r.centroid_stability,
                    'variance_stability': r.variance_stability,
                }
                for r in self.records
            ]
        }

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

    def print_summary(self):
        """Print human-readable summary of convergence analysis."""
        stats = self.compute_statistics()

        if 'error' in stats:
            print(f"❌ {stats['error']}")
            return

        print("=" * 70)
        print("CONVERGENCE ANALYSIS SUMMARY")
        print("=" * 70)
        print()
        print(f"Sessions analyzed: {stats['n_sessions']}")
        print(f"Converged sessions: {stats['n_converged']} ({stats['convergence_rate']:.1%})")
        print()
        print("Convergence Turn Statistics:")
        print(f"  Mean:   {stats['convergence_turns']['mean']:.1f} turns")
        print(f"  Median: {stats['convergence_turns']['median']:.1f} turns")
        print(f"  Std:    {stats['convergence_turns']['std']:.1f} turns")
        print(f"  Range:  [{stats['convergence_turns']['min']}, {stats['convergence_turns']['max']}]")

        if 'ci_95_lower' in stats['convergence_turns']:
            print(f"  95% CI: [{stats['convergence_turns']['ci_95_lower']:.1f}, "
                  f"{stats['convergence_turns']['ci_95_upper']:.1f}]")

        print()
        print("Percentiles:")
        print(f"  25th: {stats['convergence_turns']['percentile_25']:.0f} turns")
        print(f"  75th: {stats['convergence_turns']['percentile_75']:.0f} turns")
        print(f"  90th: {stats['convergence_turns']['percentile_90']:.0f} turns")
        print(f"  95th: {stats['convergence_turns']['percentile_95']:.0f} turns")
        print()

        recommendations = self.recommend_parameters()
        if 'error' not in recommendations:
            print("=" * 70)
            print("DATA-DRIVEN RECOMMENDATIONS")
            print("=" * 70)
            print()
            print(f"Window size: {recommendations['recommended_window_size']} turns")
            print(f"Confidence threshold: {recommendations['recommended_confidence_threshold']}")
            print(f"Centroid stability threshold: {recommendations['recommended_centroid_threshold']}")
            print(f"Variance stability threshold: {recommendations['recommended_variance_threshold']}")
            print()
            print(recommendations['interpretation'])
            print()

        print("=" * 70)
