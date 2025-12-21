"""
Branch Comparator for TELOSCOPE
===============================

Compares counterfactual branches and generates visualizations for Streamlit UI.
Provides statistical analysis of governance efficacy.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy import stats

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    print("Warning: plotly not installed. Install with: pip install plotly")


class BranchComparator:
    """
    Compares counterfactual branches and generates visualizations.

    Provides statistical evidence of TELOS governance efficacy through
    ΔF improvement metrics and visual comparisons.
    """

    def __init__(self):
        """Initialize branch comparator."""
        if not HAS_PLOTLY:
            print("Warning: Plotly visualizations will not be available")

    def compare_branches(
        self,
        baseline_branch: Dict[str, Any],
        telos_branch: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compare baseline and TELOS branches.

        Args:
            baseline_branch: Baseline branch data (no intervention)
            telos_branch: TELOS branch data (with intervention)

        Returns:
            Comprehensive comparison dict with all metrics
        """
        # Extract turn-by-turn metrics
        baseline_turns = baseline_branch.get('turns', [])
        telos_turns = telos_branch.get('turns', [])

        baseline_fidelities = [t['metrics']['telic_fidelity'] for t in baseline_turns]
        telos_fidelities = [t['metrics']['telic_fidelity'] for t in telos_turns]

        baseline_distances = [t['metrics']['drift_distance'] for t in baseline_turns]
        telos_distances = [t['metrics']['drift_distance'] for t in telos_turns]

        # Calculate aggregate metrics
        comparison = {
            'baseline': {
                'final_fidelity': baseline_fidelities[-1] if baseline_fidelities else 0.0,
                'avg_fidelity': np.mean(baseline_fidelities) if baseline_fidelities else 0.0,
                'min_fidelity': np.min(baseline_fidelities) if baseline_fidelities else 0.0,
                'max_fidelity': np.max(baseline_fidelities) if baseline_fidelities else 0.0,
                'fidelity_trend': baseline_fidelities,
                'distance_trend': baseline_distances
            },
            'telos': {
                'final_fidelity': telos_fidelities[-1] if telos_fidelities else 0.0,
                'avg_fidelity': np.mean(telos_fidelities) if telos_fidelities else 0.0,
                'min_fidelity': np.min(telos_fidelities) if telos_fidelities else 0.0,
                'max_fidelity': np.max(telos_fidelities) if telos_fidelities else 0.0,
                'fidelity_trend': telos_fidelities,
                'distance_trend': telos_distances
            },
            'delta': {
                'delta_f': (telos_fidelities[-1] - baseline_fidelities[-1]) if telos_fidelities and baseline_fidelities else 0.0,
                'avg_improvement': (np.mean(telos_fidelities) - np.mean(baseline_fidelities)) if telos_fidelities and baseline_fidelities else 0.0,
                'max_improvement': max(t - b for t, b in zip(telos_fidelities, baseline_fidelities)) if telos_fidelities and baseline_fidelities else 0.0
            },
            'metadata': {
                'branch_length': len(baseline_turns),
                'trigger_turn': baseline_branch.get('trigger_turn', 0)
            }
        }

        # Add statistical significance
        if len(baseline_fidelities) > 1 and len(telos_fidelities) > 1:
            comparison['statistics'] = self.get_statistical_significance(
                baseline_fidelities,
                telos_fidelities
            )

        return comparison

    def calculate_delta_f(
        self,
        baseline_branch: Dict[str, Any],
        telos_branch: Dict[str, Any]
    ) -> float:
        """
        Calculate ΔF (fidelity improvement).

        Args:
            baseline_branch: Baseline branch data
            telos_branch: TELOS branch data

        Returns:
            ΔF value (TELOS final - baseline final)
        """
        baseline_turns = baseline_branch.get('turns', [])
        telos_turns = telos_branch.get('turns', [])

        if not baseline_turns or not telos_turns:
            return 0.0

        baseline_final = baseline_turns[-1]['metrics']['telic_fidelity']
        telos_final = telos_turns[-1]['metrics']['telic_fidelity']

        return telos_final - baseline_final

    def generate_divergence_chart(
        self,
        comparison: Dict[str, Any],
        title: str = "Fidelity Divergence: Baseline vs TELOS"
    ) -> Optional[Any]:
        """
        Generate Plotly divergence chart showing fidelity over turns.

        Args:
            comparison: Comparison dict from compare_branches()
            title: Chart title

        Returns:
            plotly.graph_objects.Figure or None if plotly not available
        """
        if not HAS_PLOTLY:
            return None

        baseline_fidelities = comparison['baseline']['fidelity_trend']
        telos_fidelities = comparison['telos']['fidelity_trend']

        # Create figure
        fig = go.Figure()

        # Baseline line
        fig.add_trace(go.Scatter(
            x=list(range(1, len(baseline_fidelities) + 1)),
            y=baseline_fidelities,
            mode='lines+markers',
            name='Baseline (No Intervention)',
            line=dict(color='#ff6b6b', width=3),
            marker=dict(size=8, symbol='x')
        ))

        # TELOS line
        fig.add_trace(go.Scatter(
            x=list(range(1, len(telos_fidelities) + 1)),
            y=telos_fidelities,
            mode='lines+markers',
            name='TELOS (With Intervention)',
            line=dict(color='#51cf66', width=3),
            marker=dict(size=8, symbol='circle')
        ))

        # Add ΔF annotation
        delta_f = comparison['delta']['delta_f']
        fig.add_annotation(
            x=len(telos_fidelities),
            y=(baseline_fidelities[-1] + telos_fidelities[-1]) / 2,
            text=f"ΔF = {delta_f:+.3f}",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="#339af0",
            font=dict(size=14, color="#339af0", family="monospace"),
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="#339af0",
            borderwidth=2
        )

        # Layout
        fig.update_layout(
            title=dict(text=title, font=dict(size=18, family="Arial")),
            xaxis_title="Turn Number",
            yaxis_title="Telic Fidelity",
            hovermode='x unified',
            template='plotly_white',
            height=400,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="gray",
                borderwidth=1
            )
        )

        # Add Goldilocks zone threshold lines
        fig.add_hline(
            y=0.76,
            line_dash="dash",
            line_color="green",
            annotation_text="Aligned Threshold (F=0.76)",
            annotation_position="right"
        )

        fig.add_hline(
            y=0.73,
            line_dash="dash",
            line_color="gold",
            annotation_text="Minor Drift (F=0.73)",
            annotation_position="right"
        )

        fig.add_hline(
            y=0.67,
            line_dash="dash",
            line_color="red",
            annotation_text="Drift Threshold (F=0.67)",
            annotation_position="right"
        )

        return fig

    def generate_metrics_table(
        self,
        comparison: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Generate metrics comparison table for Streamlit.

        Args:
            comparison: Comparison dict from compare_branches()

        Returns:
            pandas DataFrame ready for st.dataframe()
        """
        data = {
            'Metric': [
                'Final Fidelity',
                'Average Fidelity',
                'Min Fidelity',
                'Max Fidelity',
                'Improvement (ΔF)'
            ],
            'Baseline': [
                f"{comparison['baseline']['final_fidelity']:.3f}",
                f"{comparison['baseline']['avg_fidelity']:.3f}",
                f"{comparison['baseline']['min_fidelity']:.3f}",
                f"{comparison['baseline']['max_fidelity']:.3f}",
                "-"
            ],
            'TELOS': [
                f"{comparison['telos']['final_fidelity']:.3f}",
                f"{comparison['telos']['avg_fidelity']:.3f}",
                f"{comparison['telos']['min_fidelity']:.3f}",
                f"{comparison['telos']['max_fidelity']:.3f}",
                f"+{comparison['delta']['delta_f']:.3f}" if comparison['delta']['delta_f'] > 0 else f"{comparison['delta']['delta_f']:.3f}"
            ]
        }

        df = pd.DataFrame(data)
        return df

    def generate_distance_chart(
        self,
        comparison: Dict[str, Any],
        title: str = "Drift Distance Over Time"
    ) -> Optional[Any]:
        """
        Generate chart showing drift distance evolution.

        Args:
            comparison: Comparison dict from compare_branches()
            title: Chart title

        Returns:
            plotly.graph_objects.Figure or None
        """
        if not HAS_PLOTLY:
            return None

        baseline_distances = comparison['baseline']['distance_trend']
        telos_distances = comparison['telos']['distance_trend']

        fig = go.Figure()

        # Baseline distances
        fig.add_trace(go.Scatter(
            x=list(range(1, len(baseline_distances) + 1)),
            y=baseline_distances,
            mode='lines+markers',
            name='Baseline',
            line=dict(color='#ff6b6b', width=2),
            fill='tozeroy',
            fillcolor='rgba(255, 107, 107, 0.2)'
        ))

        # TELOS distances
        fig.add_trace(go.Scatter(
            x=list(range(1, len(telos_distances) + 1)),
            y=telos_distances,
            mode='lines+markers',
            name='TELOS',
            line=dict(color='#51cf66', width=2),
            fill='tozeroy',
            fillcolor='rgba(81, 207, 102, 0.2)'
        ))

        fig.update_layout(
            title=title,
            xaxis_title="Turn Number",
            yaxis_title="Distance from Attractor",
            hovermode='x unified',
            template='plotly_white',
            height=300
        )

        return fig

    def generate_comparison_dashboard(
        self,
        comparison: Dict[str, Any]
    ) -> Optional[Any]:
        """
        Generate comprehensive 2x2 dashboard with multiple metrics.

        Args:
            comparison: Comparison dict from compare_branches()

        Returns:
            plotly.graph_objects.Figure with subplots
        """
        if not HAS_PLOTLY:
            return None

        # Create 2x2 subplot figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Fidelity Divergence',
                'Drift Distance',
                'Turn-by-Turn Comparison',
                'Final Metrics'
            ),
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "bar"}]
            ]
        )

        baseline_fidelities = comparison['baseline']['fidelity_trend']
        telos_fidelities = comparison['telos']['fidelity_trend']
        baseline_distances = comparison['baseline']['distance_trend']
        telos_distances = comparison['telos']['distance_trend']

        # Subplot 1: Fidelity
        fig.add_trace(go.Scatter(
            x=list(range(1, len(baseline_fidelities) + 1)),
            y=baseline_fidelities,
            name='Baseline',
            line=dict(color='#ff6b6b')
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=list(range(1, len(telos_fidelities) + 1)),
            y=telos_fidelities,
            name='TELOS',
            line=dict(color='#51cf66')
        ), row=1, col=1)

        # Subplot 2: Distance
        fig.add_trace(go.Scatter(
            x=list(range(1, len(baseline_distances) + 1)),
            y=baseline_distances,
            name='Baseline Dist',
            line=dict(color='#ff6b6b', dash='dot'),
            showlegend=False
        ), row=1, col=2)

        fig.add_trace(go.Scatter(
            x=list(range(1, len(telos_distances) + 1)),
            y=telos_distances,
            name='TELOS Dist',
            line=dict(color='#51cf66', dash='dot'),
            showlegend=False
        ), row=1, col=2)

        # Subplot 3: Turn-by-turn bars
        turns = list(range(1, len(baseline_fidelities) + 1))
        fig.add_trace(go.Bar(
            x=turns,
            y=baseline_fidelities,
            name='Baseline',
            marker_color='#ff6b6b',
            showlegend=False
        ), row=2, col=1)

        fig.add_trace(go.Bar(
            x=turns,
            y=telos_fidelities,
            name='TELOS',
            marker_color='#51cf66',
            showlegend=False
        ), row=2, col=1)

        # Subplot 4: Final comparison
        fig.add_trace(go.Bar(
            x=['Final Fidelity', 'Avg Fidelity'],
            y=[comparison['baseline']['final_fidelity'], comparison['baseline']['avg_fidelity']],
            name='Baseline',
            marker_color='#ff6b6b',
            showlegend=False
        ), row=2, col=2)

        fig.add_trace(go.Bar(
            x=['Final Fidelity', 'Avg Fidelity'],
            y=[comparison['telos']['final_fidelity'], comparison['telos']['avg_fidelity']],
            name='TELOS',
            marker_color='#51cf66',
            showlegend=False
        ), row=2, col=2)

        fig.update_layout(
            height=700,
            showlegend=True,
            template='plotly_white',
            title_text="TELOSCOPE Branch Comparison Dashboard"
        )

        return fig

    def get_statistical_significance(
        self,
        baseline_values: List[float],
        telos_values: List[float]
    ) -> Dict[str, float]:
        """
        Calculate statistical significance of improvement.

        Args:
            baseline_values: Baseline fidelity values
            telos_values: TELOS fidelity values

        Returns:
            Dict with p_value, effect_size, confidence_interval
        """
        # Perform paired t-test
        t_statistic, p_value = stats.ttest_rel(telos_values, baseline_values)

        # Calculate Cohen's d (effect size)
        mean_diff = np.mean(telos_values) - np.mean(baseline_values)
        pooled_std = np.sqrt((np.std(baseline_values)**2 + np.std(telos_values)**2) / 2)
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0.0

        # Calculate 95% confidence interval
        std_error = pooled_std / np.sqrt(len(baseline_values))
        confidence_interval = (
            mean_diff - 1.96 * std_error,
            mean_diff + 1.96 * std_error
        )

        return {
            'p_value': p_value,
            't_statistic': t_statistic,
            'effect_size_cohens_d': cohens_d,
            'mean_difference': mean_diff,
            'confidence_interval_95': confidence_interval,
            'significant': p_value < 0.05
        }

    def format_statistics_text(self, stats: Dict[str, float]) -> str:
        """
        Format statistical results as readable text.

        Args:
            stats: Statistics dict from get_statistical_significance()

        Returns:
            Formatted text string
        """
        sig_text = "**Statistically significant**" if stats['significant'] else "Not statistically significant"

        text = f"""
**Statistical Analysis:**

- {sig_text} (p={stats['p_value']:.4f})
- Effect size (Cohen's d): {stats['effect_size_cohens_d']:.3f}
- Mean improvement: {stats['mean_difference']:.3f}
- 95% CI: [{stats['confidence_interval_95'][0]:.3f}, {stats['confidence_interval_95'][1]:.3f}]
        """

        return text.strip()
