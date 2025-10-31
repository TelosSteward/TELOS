"""
Comparison Adapter for Observatory v2

Wraps telos_purpose.validation.branch_comparator for Observatory integration.

Provides simplified interface for comparing baseline and TELOS branches,
generating visualizations, and computing statistical significance.

Usage:
    from teloscope_v2.utils.comparison_adapter import ComparisonAdapter

    adapter = ComparisonAdapter()

    # Compare branches
    comparison = adapter.compare_results(baseline_branch, telos_branch)

    # Get ΔF
    delta_f = comparison['delta']['delta_f']

    # Generate chart
    fig = adapter.generate_chart(comparison, chart_type='divergence')
    st.plotly_chart(fig)
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List
import sys
from pathlib import Path
import pandas as pd

# Add telos_purpose to path for imports
telos_root = Path(__file__).parent.parent.parent.parent / 'telos_purpose'
if str(telos_root) not in sys.path:
    sys.path.insert(0, str(telos_root))

try:
    from telos_purpose.validation.branch_comparator import BranchComparator
    BRANCH_COMPARATOR_AVAILABLE = True
except ImportError as e:
    BRANCH_COMPARATOR_AVAILABLE = False
    print(f"Warning: Could not import branch_comparator: {e}")


class ComparisonAdapter:
    """
    Adapter for branch comparison in Observatory v2.

    Simplifies comparison analysis for Observatory UI by providing:
    - Branch-to-branch comparison
    - ΔF (fidelity improvement) calculation
    - Statistical significance testing
    - Plotly visualizations
    - Metrics tables for Streamlit
    """

    def __init__(self):
        """Initialize comparison adapter."""
        if not BRANCH_COMPARATOR_AVAILABLE:
            raise ImportError(
                "branch_comparator not available. Ensure telos_purpose is installed."
            )

        self.comparator = BranchComparator()

    def compare_results(
        self,
        baseline_branch: Dict[str, Any],
        telos_branch: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compare baseline and TELOS branches.

        Computes comprehensive comparison including:
        - Final fidelity for both branches
        - Average fidelity over all turns
        - ΔF (improvement)
        - Fidelity trajectories
        - Statistical significance (if sufficient data)

        Args:
            baseline_branch: Baseline branch data (must have 'turns' key)
            telos_branch: TELOS branch data (must have 'turns' key)

        Returns:
            Comprehensive comparison dict with structure:
            {
                'baseline': {
                    'final_fidelity': float,
                    'avg_fidelity': float,
                    'fidelity_trend': List[float],
                    ...
                },
                'telos': {...},
                'delta': {
                    'delta_f': float,
                    'avg_improvement': float,
                    ...
                },
                'statistics': {  # If sufficient data
                    'p_value': float,
                    'significant': bool,
                    ...
                }
            }

        Example:
            comparison = adapter.compare_results(baseline, telos)
            print(f"ΔF: {comparison['delta']['delta_f']:+.3f}")
        """
        return self.comparator.compare_branches(baseline_branch, telos_branch)

    def get_delta_f(
        self,
        baseline_branch: Dict[str, Any],
        telos_branch: Dict[str, Any]
    ) -> float:
        """
        Get ΔF (fidelity improvement).

        Quick method to get just the improvement value.

        Args:
            baseline_branch: Baseline branch data
            telos_branch: TELOS branch data

        Returns:
            ΔF = TELOS final fidelity - Baseline final fidelity
        """
        return self.comparator.calculate_delta_f(baseline_branch, telos_branch)

    def generate_chart(
        self,
        comparison: Dict[str, Any],
        chart_type: str = 'divergence',
        title: Optional[str] = None
    ) -> Optional[Any]:
        """
        Generate visualization.

        Args:
            comparison: Comparison dict from compare_results()
            chart_type: One of:
                - 'divergence': Fidelity over time (both branches)
                - 'distance': Drift distance over time
                - 'dashboard': 2x2 comprehensive dashboard
            title: Optional custom title (uses default if None)

        Returns:
            plotly.graph_objects.Figure or None if plotly not available

        Example:
            fig = adapter.generate_chart(comparison, chart_type='divergence')
            st.plotly_chart(fig, use_container_width=True)
        """
        if chart_type == 'divergence':
            if title:
                return self.comparator.generate_divergence_chart(comparison, title)
            else:
                return self.comparator.generate_divergence_chart(comparison)

        elif chart_type == 'distance':
            if title:
                return self.comparator.generate_distance_chart(comparison, title)
            else:
                return self.comparator.generate_distance_chart(comparison)

        elif chart_type == 'dashboard':
            return self.comparator.generate_comparison_dashboard(comparison)

        else:
            print(f"Warning: Unknown chart type '{chart_type}'")
            return None

    def generate_metrics_table(
        self,
        comparison: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Generate metrics comparison table for Streamlit.

        Creates a formatted DataFrame showing key metrics side-by-side.

        Args:
            comparison: Comparison dict from compare_results()

        Returns:
            pandas DataFrame ready for st.dataframe() or st.table()

        Example:
            df = adapter.generate_metrics_table(comparison)
            st.dataframe(df, use_container_width=True)
        """
        return self.comparator.generate_metrics_table(comparison)

    def format_statistics(
        self,
        comparison: Dict[str, Any]
    ) -> str:
        """
        Format statistical results as readable text.

        Args:
            comparison: Comparison dict from compare_results()

        Returns:
            Formatted Markdown string with statistical analysis

        Example:
            text = adapter.format_statistics(comparison)
            st.markdown(text)
        """
        if 'statistics' not in comparison:
            return "**Statistical Analysis:** Insufficient data for statistical analysis (need 2+ turns)"

        return self.comparator.format_statistics_text(comparison['statistics'])

    def is_governance_effective(
        self,
        comparison: Dict[str, Any],
        threshold: float = 0.0
    ) -> bool:
        """
        Determine if TELOS governance was effective.

        Args:
            comparison: Comparison dict from compare_results()
            threshold: Minimum ΔF to be considered effective (default: 0.0)

        Returns:
            True if ΔF > threshold, False otherwise
        """
        delta_f = comparison.get('delta', {}).get('delta_f', 0.0)
        return delta_f > threshold

    def get_improvement_summary(
        self,
        comparison: Dict[str, Any]
    ) -> str:
        """
        Get human-readable improvement summary.

        Args:
            comparison: Comparison dict from compare_results()

        Returns:
            Formatted summary string

        Example:
            summary = adapter.get_improvement_summary(comparison)
            st.info(summary)
        """
        baseline_f = comparison['baseline']['final_fidelity']
        telos_f = comparison['telos']['final_fidelity']
        delta_f = comparison['delta']['delta_f']
        avg_improvement = comparison['delta']['avg_improvement']

        if delta_f > 0:
            effectiveness = "✅ EFFECTIVE"
            direction = "improved"
        elif delta_f < 0:
            effectiveness = "❌ INEFFECTIVE"
            direction = "degraded"
        else:
            effectiveness = "➖ NEUTRAL"
            direction = "unchanged"

        summary = f"""
**Governance Summary:**

- **Status**: {effectiveness}
- **Baseline Final Fidelity**: {baseline_f:.3f}
- **TELOS Final Fidelity**: {telos_f:.3f}
- **ΔF (Improvement)**: {delta_f:+.3f}
- **Average Improvement**: {avg_improvement:+.3f}
- **Result**: TELOS governance {direction} fidelity by {abs(delta_f):.3f} points.
        """

        # Add statistical significance if available
        if 'statistics' in comparison:
            stats = comparison['statistics']
            if stats['significant']:
                summary += f"\n- **Statistical Significance**: ✅ p={stats['p_value']:.4f} (significant)"
            else:
                summary += f"\n- **Statistical Significance**: ℹ️ p={stats['p_value']:.4f} (not significant)"

        return summary.strip()

    def convert_baseline_result_to_branch(
        self,
        baseline_result
    ) -> Dict[str, Any]:
        """
        Convert BaselineResult to branch format for comparison.

        Converts from baseline_runner output to branch_comparator input format.

        Args:
            baseline_result: BaselineResult from BaselineAdapter

        Returns:
            Branch dict compatible with compare_results()
        """
        # Convert turn_results to branch format
        turns = []
        for turn_data in baseline_result.turn_results:
            turns.append({
                'turn_number': turn_data.get('turn', 0),
                'user_input': turn_data.get('user_input', ''),
                'assistant_response': turn_data.get('response', ''),
                'metrics': {
                    'telic_fidelity': turn_data.get('distance_to_attractor', 0.0),
                    'drift_distance': turn_data.get('distance_to_attractor', 0.0),
                    'error_signal': 1.0 - turn_data.get('distance_to_attractor', 0.0),
                    'primacy_basin_membership': turn_data.get('in_basin', False)
                },
                'intervention_applied': turn_data.get('intervention_applied', False),
                'intervention_type': turn_data.get('intervention_type', None)
            })

        branch = {
            'branch_id': baseline_result.session_id,
            'branch_type': baseline_result.runner_type,
            'turns': turns,
            'final_fidelity': baseline_result.final_metrics.get('fidelity', 0.0),
            'metadata': baseline_result.metadata
        }

        # Include turn_results for runtime validation (if available)
        if hasattr(baseline_result, 'turn_results') and baseline_result.turn_results:
            branch['turn_results'] = baseline_result.turn_results

        return branch


def check_comparator_availability() -> bool:
    """
    Check if branch comparator is available.

    Returns:
        True if available, False otherwise
    """
    return BRANCH_COMPARATOR_AVAILABLE
