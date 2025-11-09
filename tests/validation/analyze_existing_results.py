#!/usr/bin/env python3
"""
TELOS Validation Results Analyzer
Analyzes existing baseline conversation data that already contains
native vs TELOS responses and fidelity measurements.
"""

import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import numpy as np


class ValidationAnalyzer:
    """Analyze pre-computed validation results from baseline conversations."""

    def __init__(self, data_dir: str = "tests/validation_data/baseline_conversations"):
        self.data_dir = Path(data_dir)
        self.results_dir = Path("tests/validation_results")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Track statistics
        self.stats = {
            "total_sessions": 0,
            "total_turns": 0,
            "turns_with_drift": 0,
            "interventions_applied": 0,
            "native_fidelities": [],
            "telos_fidelities": [],
            "improvements": [],
            "session_results": []
        }

    def analyze_session(self, session_data: Dict) -> Dict[str, Any]:
        """Analyze a single session's performance."""
        session_id = session_data.get('session_id', 'unknown')

        # Extract metadata
        metadata = session_data.get('metadata', {})
        turns = session_data.get('turns', [])

        # Session-level metrics
        session_metrics = {
            'session_id': session_id,
            'total_turns': len(turns),
            'pa_established': metadata.get('pa_established', False),
            'convergence_turn': metadata.get('convergence_turn', 0),
            'total_interventions': metadata.get('total_interventions', 0),
            'trigger_turn': metadata.get('trigger_turn', 0),
            'trigger_fidelity': metadata.get('trigger_fidelity', 0),
            'governance_effective': metadata.get('governance_effective', False)
        }

        # Turn-level analysis
        turn_metrics = []
        native_fidelities = []
        telos_fidelities = []

        for turn in turns:
            turn_num = turn.get('turn_number', turn.get('turn', 0))
            fidelity_native = turn.get('fidelity_native', 0)
            fidelity_telos = turn.get('fidelity_telos', turn.get('fidelity', 0))
            drift_detected = turn.get('drift_detected', False)
            intervention_applied = turn.get('intervention_applied', False)

            # Calculate improvement
            improvement = fidelity_telos - fidelity_native if fidelity_native else 0

            turn_metric = {
                'turn': turn_num,
                'fidelity_native': fidelity_native,
                'fidelity_telos': fidelity_telos,
                'improvement': improvement,
                'improvement_pct': (improvement / fidelity_native * 100) if fidelity_native > 0 else 0,
                'drift_detected': drift_detected,
                'intervention_applied': intervention_applied
            }

            turn_metrics.append(turn_metric)

            if fidelity_native:
                native_fidelities.append(fidelity_native)
            if fidelity_telos:
                telos_fidelities.append(fidelity_telos)

        # Calculate session aggregates
        session_metrics['avg_native_fidelity'] = np.mean(native_fidelities) if native_fidelities else 0
        session_metrics['avg_telos_fidelity'] = np.mean(telos_fidelities) if telos_fidelities else 0
        session_metrics['avg_improvement'] = session_metrics['avg_telos_fidelity'] - session_metrics['avg_native_fidelity']
        session_metrics['improvement_pct'] = (session_metrics['avg_improvement'] / session_metrics['avg_native_fidelity'] * 100) if session_metrics['avg_native_fidelity'] > 0 else 0
        session_metrics['turn_metrics'] = turn_metrics

        return session_metrics

    def run_analysis(self):
        """Run analysis on all baseline conversations."""
        print("\n" + "="*70)
        print("TELOS VALIDATION ANALYSIS")
        print("Analyzing pre-computed results from baseline conversations")
        print("="*70)

        # Load all sessions
        sessions = []
        for json_file in sorted(self.data_dir.glob("*.json")):
            with open(json_file, 'r', encoding='utf-8') as f:
                sessions.append(json.load(f))

        self.stats['total_sessions'] = len(sessions)
        print(f"\nAnalyzing {len(sessions)} sessions...")

        # Analyze each session
        for session_data in sessions:
            session_id = session_data.get('session_id', 'unknown')
            print(f"\nProcessing: {session_id}")

            try:
                metrics = self.analyze_session(session_data)
                self.stats['session_results'].append(metrics)

                # Update global statistics
                self.stats['total_turns'] += metrics['total_turns']
                self.stats['interventions_applied'] += metrics['total_interventions']

                # Collect fidelities
                for turn_metric in metrics['turn_metrics']:
                    if turn_metric['fidelity_native']:
                        self.stats['native_fidelities'].append(turn_metric['fidelity_native'])
                    if turn_metric['fidelity_telos']:
                        self.stats['telos_fidelities'].append(turn_metric['fidelity_telos'])
                    if turn_metric['improvement']:
                        self.stats['improvements'].append(turn_metric['improvement'])
                    if turn_metric['drift_detected']:
                        self.stats['turns_with_drift'] += 1

                # Print session summary
                print(f"  • Turns: {metrics['total_turns']}")
                print(f"  • Avg Native Fidelity: {metrics['avg_native_fidelity']:.3f}")
                print(f"  • Avg TELOS Fidelity: {metrics['avg_telos_fidelity']:.3f}")
                print(f"  • Improvement: {metrics['improvement_pct']:+.1f}%")

            except Exception as e:
                print(f"  ⚠ Error processing {session_id}: {e}")

        # Generate final report
        self.generate_report()

    def generate_report(self):
        """Generate comprehensive validation report."""
        print("\n" + "="*70)
        print("VALIDATION RESULTS SUMMARY")
        print("="*70)

        # Calculate aggregate metrics
        avg_native = np.mean(self.stats['native_fidelities']) if self.stats['native_fidelities'] else 0
        avg_telos = np.mean(self.stats['telos_fidelities']) if self.stats['telos_fidelities'] else 0
        avg_improvement = avg_telos - avg_native
        improvement_pct = (avg_improvement / avg_native * 100) if avg_native > 0 else 0

        # Variance and standard deviation
        std_native = np.std(self.stats['native_fidelities']) if self.stats['native_fidelities'] else 0
        std_telos = np.std(self.stats['telos_fidelities']) if self.stats['telos_fidelities'] else 0

        # Statistical significance (simple t-test approximation)
        try:
            if len(self.stats['native_fidelities']) > 1 and len(self.stats['telos_fidelities']) > 1:
                from scipy import stats
                t_stat, p_value = stats.ttest_ind(self.stats['telos_fidelities'], self.stats['native_fidelities'])
                significant = p_value < 0.05
            else:
                t_stat, p_value, significant = 0, 1, False
        except ImportError:
            # scipy not available, skip statistical test
            t_stat, p_value, significant = 0, 1, False

        # Print summary
        print(f"\n📊 AGGREGATE METRICS")
        print(f"  Sessions Analyzed: {self.stats['total_sessions']}")
        print(f"  Total Turns: {self.stats['total_turns']}")
        print(f"  Turns with Drift: {self.stats['turns_with_drift']}")
        print(f"  Interventions Applied: {self.stats['interventions_applied']}")

        print(f"\n📈 FIDELITY COMPARISON")
        print(f"  Native Baseline:")
        print(f"    • Mean: {avg_native:.4f}")
        print(f"    • Std Dev: {std_native:.4f}")
        print(f"  TELOS Governed:")
        print(f"    • Mean: {avg_telos:.4f}")
        print(f"    • Std Dev: {std_telos:.4f}")

        print(f"\n🎯 IMPROVEMENT METRICS")
        print(f"  Absolute Improvement: {avg_improvement:+.4f}")
        print(f"  Percentage Improvement: {improvement_pct:+.2f}%")
        print(f"  Statistical Significance: {'✓ YES' if significant else '✗ NO'} (p={p_value:.4f})")

        # Best and worst performers
        session_results = sorted(self.stats['session_results'],
                               key=lambda x: x['improvement_pct'],
                               reverse=True)

        print(f"\n🏆 TOP PERFORMERS")
        for session in session_results[:3]:
            print(f"  • {session['session_id']}: {session['improvement_pct']:+.1f}%")

        print(f"\n⚠️  SESSIONS NEEDING ATTENTION")
        for session in session_results[-3:]:
            if session['improvement_pct'] < 0:
                print(f"  • {session['session_id']}: {session['improvement_pct']:+.1f}%")

        # Save detailed report
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'sessions_analyzed': self.stats['total_sessions'],
                'total_turns': self.stats['total_turns'],
                'turns_with_drift': self.stats['turns_with_drift'],
                'interventions_applied': self.stats['interventions_applied']
            },
            'fidelity_metrics': {
                'native': {
                    'mean': avg_native,
                    'std': std_native,
                    'n': len(self.stats['native_fidelities'])
                },
                'telos': {
                    'mean': avg_telos,
                    'std': std_telos,
                    'n': len(self.stats['telos_fidelities'])
                },
                'improvement': {
                    'absolute': avg_improvement,
                    'percentage': improvement_pct
                }
            },
            'statistical_test': {
                't_statistic': float(t_stat) if t_stat else 0,
                'p_value': float(p_value) if p_value else 1,
                'significant': bool(significant)
            },
            'session_results': self.stats['session_results']
        }

        report_file = self.results_dir / f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n💾 Detailed report saved to: {report_file}")

        # Final verdict
        print("\n" + "="*70)
        print("VALIDATION VERDICT")
        print("="*70)

        if improvement_pct > 50:
            print(f"✨ EXCEPTIONAL: TELOS shows {improvement_pct:.1f}% improvement over baseline!")
        elif improvement_pct > 20:
            print(f"✅ STRONG: TELOS demonstrates {improvement_pct:.1f}% improvement over baseline!")
        elif improvement_pct > 10:
            print(f"👍 POSITIVE: TELOS achieves {improvement_pct:.1f}% improvement over baseline!")
        elif improvement_pct > 0:
            print(f"📊 MODEST: TELOS shows {improvement_pct:.1f}% improvement over baseline")
        else:
            print(f"⚠️  REVIEW NEEDED: TELOS shows {improvement_pct:.1f}% change vs baseline")

        print("="*70)


def main():
    """Run validation analysis."""
    analyzer = ValidationAnalyzer()
    analyzer.run_analysis()


if __name__ == "__main__":
    main()