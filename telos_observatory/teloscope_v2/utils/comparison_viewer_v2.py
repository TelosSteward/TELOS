"""
Comparison Viewer v2 for Observatory

Complete implementation of side-by-side TELOS vs Baseline comparison UI.

Displays:
- Split-view comparison (Baseline | TELOS)
- Turn-by-turn responses
- Fidelity metrics
- Intervention highlighting
- Summary statistics

Usage:
    from teloscope_v2.utils.comparison_viewer_v2 import ComparisonViewerV2

    viewer = ComparisonViewerV2()

    # Render specific turn
    viewer.render_turn(baseline_data, telos_data, turn_index=0)

    # Render all turns
    viewer.render_all_turns(baseline_data, telos_data)

    # Render summary
    viewer.render_summary(comparison)
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List
import streamlit as st

try:
    from teloscope_v2.utils.runtime_validator import RuntimeValidator
    RUNTIME_VALIDATOR_AVAILABLE = True
except ImportError:
    RUNTIME_VALIDATOR_AVAILABLE = False


class ComparisonViewerV2:
    """
    Side-by-side comparison viewer for Observatory v2.

    Displays TELOS vs Baseline responses with intervention highlighting,
    fidelity metrics, and turn-by-turn analysis.
    """

    def __init__(self, show_validation: bool = True):
        """
        Initialize comparison viewer.

        Args:
            show_validation: If True, show runtime validation status (default: True)
        """
        self.show_validation = show_validation and RUNTIME_VALIDATOR_AVAILABLE
        if self.show_validation:
            self.validator = RuntimeValidator()
        else:
            self.validator = None

    def render_turn(
        self,
        baseline_turn: Dict[str, Any],
        telos_turn: Dict[str, Any],
        turn_number: Optional[int] = None
    ):
        """
        Render single turn comparison.

        Args:
            baseline_turn: Baseline turn data
            telos_turn: TELOS turn data
            turn_number: Turn number (optional, extracted from data if None)

        Example:
            viewer.render_turn(
                baseline_turn={'user_input': '...', 'assistant_response': '...', 'metrics': {...}},
                telos_turn={'user_input': '...', 'assistant_response': '...', 'metrics': {...}},
                turn_number=1
            )
        """
        # Get turn number
        if turn_number is None:
            turn_number = baseline_turn.get('turn_number', baseline_turn.get('turn', 1))

        # Show user input (same for both)
        user_input = baseline_turn.get('user_input', '')
        st.markdown(f"**👤 User (Turn {turn_number}):**")
        st.markdown(f"> {user_input}")

        st.markdown("---")

        # Split view for responses
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🔴 Baseline (Ungoverned)")
            self._render_response_box(
                baseline_turn.get('assistant_response', 'N/A'),
                baseline_turn.get('metrics', {}),
                is_telos=False
            )

        with col2:
            st.markdown("### 🟢 TELOS (Governed)")
            intervention = telos_turn.get('intervention_applied', False)
            self._render_response_box(
                telos_turn.get('assistant_response', 'N/A'),
                telos_turn.get('metrics', {}),
                is_telos=True,
                intervention_applied=intervention,
                intervention_type=telos_turn.get('intervention_type')
            )

    def render_all_turns(
        self,
        baseline_data: Dict[str, Any],
        telos_data: Dict[str, Any],
        expanded_first: bool = True
    ):
        """
        Render all turns in expandable sections.

        Args:
            baseline_data: Baseline branch data (with 'turns' key)
            telos_data: TELOS branch data (with 'turns' key)
            expanded_first: If True, first turn is expanded by default

        Example:
            viewer.render_all_turns(baseline_branch, telos_branch)
        """
        baseline_turns = baseline_data.get('turns', [])
        telos_turns = telos_data.get('turns', [])

        if not baseline_turns or not telos_turns:
            st.warning("No turn data available for comparison")
            return

        st.markdown("### 🔀 Turn-by-Turn Comparison")
        st.caption(f"Comparing {len(baseline_turns)} turns")

        for i, (baseline_turn, telos_turn) in enumerate(zip(baseline_turns, telos_turns)):
            turn_num = baseline_turn.get('turn_number', baseline_turn.get('turn', i + 1))

            # Get fidelity for header
            baseline_f = baseline_turn.get('metrics', {}).get('telic_fidelity', 0.0)
            telos_f = telos_turn.get('metrics', {}).get('telic_fidelity', 0.0)
            delta_f = telos_f - baseline_f

            # Color-code delta
            if delta_f > 0:
                delta_icon = "✅"
                delta_color = "#90EE90"
            elif delta_f < 0:
                delta_icon = "❌"
                delta_color = "#FF6B6B"
            else:
                delta_icon = "➖"
                delta_color = "#CCCCCC"

            expander_title = f"Turn {turn_num} | Baseline: {baseline_f:.3f} | TELOS: {telos_f:.3f} | Δ: {delta_f:+.3f} {delta_icon}"

            with st.expander(expander_title, expanded=(i == 0 and expanded_first)):
                self.render_turn(baseline_turn, telos_turn, turn_num)

    def _render_response_box(
        self,
        response: str,
        metrics: Dict[str, float],
        is_telos: bool,
        intervention_applied: bool = False,
        intervention_type: Optional[str] = None
    ):
        """
        Render response with metrics box.

        Args:
            response: Response text
            metrics: Metrics dict
            is_telos: If True, TELOS response
            intervention_applied: If True, intervention was applied
            intervention_type: Type of intervention
        """
        # Get fidelity for color-coding
        fidelity = metrics.get('telic_fidelity', 0.0)

        # Determine border color based on fidelity
        if fidelity >= 0.8:
            border_color = '#90EE90'  # Green - Good
            fidelity_status = "✅ Good"
        elif fidelity >= 0.5:
            border_color = '#FFA500'  # Orange - Warning
            fidelity_status = "⚠️ Warning"
        else:
            border_color = '#FF6B6B'  # Red - Critical
            fidelity_status = "❌ Critical"

        # Response box with color-coded border
        response_html = f"""
        <div style="
            border-left: 4px solid {border_color};
            padding: 16px;
            border-radius: 4px;
            background: rgba(255, 255, 255, 0.02);
            margin-bottom: 16px;
            min-height: 100px;
        ">
            <div style="font-size: 14px; line-height: 1.6;">
                {response}
            </div>
        </div>
        """
        st.markdown(response_html, unsafe_allow_html=True)

        # Metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Fidelity", f"{fidelity:.3f}", help=fidelity_status)

        with col2:
            distance = metrics.get('drift_distance', 0.0)
            st.metric("Distance", f"{distance:.3f}", help="Distance from attractor")

        with col3:
            in_basin = metrics.get('primacy_basin_membership', False)
            basin_status = "✓ Yes" if in_basin else "✗ No"
            st.metric("In Basin", basin_status, help="Within primacy basin")

        # Intervention badge
        if is_telos and intervention_applied:
            intervention_label = intervention_type or "correction"
            st.markdown(f"**🛡️ Intervention Applied:** `{intervention_label}`")

    def render_summary(
        self,
        comparison: Dict[str, Any],
        show_chart: bool = True
    ):
        """
        Render comparison summary with key metrics.

        Args:
            comparison: Comparison dict from ComparisonAdapter
            show_chart: If True, show fidelity trajectory chart

        Example:
            viewer.render_summary(comparison, show_chart=True)
        """
        st.markdown("### 📊 Comparison Summary")

        # Key metrics in columns
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            baseline_f = comparison['baseline']['final_fidelity']
            st.metric(
                "Baseline Final F",
                f"{baseline_f:.3f}",
                help="Final fidelity without governance"
            )

        with col2:
            telos_f = comparison['telos']['final_fidelity']
            st.metric(
                "TELOS Final F",
                f"{telos_f:.3f}",
                help="Final fidelity with governance"
            )

        with col3:
            delta_f = comparison['delta']['delta_f']
            st.metric(
                "ΔF (Improvement)",
                f"{delta_f:+.3f}",
                delta=f"{delta_f:+.3f}",
                help="TELOS - Baseline"
            )

        with col4:
            avg_improvement = comparison['delta']['avg_improvement']
            st.metric(
                "Avg Improvement",
                f"{avg_improvement:+.3f}",
                help="Average improvement across all turns"
            )

        # Effectiveness badge
        if delta_f > 0:
            st.success(f"✅ **TELOS governance was EFFECTIVE** (improved fidelity by {delta_f:.3f})")
        elif delta_f < 0:
            st.error(f"❌ **TELOS governance was INEFFECTIVE** (degraded fidelity by {abs(delta_f):.3f})")
        else:
            st.info("➖ **TELOS governance had NEUTRAL impact** (no change in fidelity)")

        # Statistical significance
        if 'statistics' in comparison:
            stats = comparison['statistics']

            st.markdown("---")
            st.markdown("#### 📈 Statistical Analysis")

            col1, col2, col3 = st.columns(3)

            with col1:
                p_value = stats['p_value']
                if stats['significant']:
                    st.metric("P-value", f"{p_value:.4f}", help="✅ Statistically significant (p < 0.05)")
                else:
                    st.metric("P-value", f"{p_value:.4f}", help="ℹ️ Not statistically significant")

            with col2:
                cohens_d = stats['effect_size_cohens_d']
                st.metric("Effect Size (Cohen's d)", f"{cohens_d:.3f}", help="Effect size magnitude")

            with col3:
                ci_low, ci_high = stats['confidence_interval_95']
                st.metric("95% CI", f"[{ci_low:.3f}, {ci_high:.3f}]", help="95% Confidence Interval")

            # Interpretation
            if stats['significant']:
                st.success(f"✅ Result is **statistically significant** (p={p_value:.4f})")
            else:
                st.info(f"ℹ️ Result is **not statistically significant** (p={p_value:.4f})")

        # Runtime validation section
        if self.show_validation and self.validator:
            st.markdown("---")
            st.markdown("#### 🔍 Runtime Simulation Verification")

            # Validate TELOS branch
            telos_branch = comparison.get('telos', {})
            if 'turn_results' in telos_branch:
                telos_results = {'turn_results': telos_branch['turn_results']}

                # Run validation
                test_results = []
                all_passed = True

                for test_func in self.validator.validation_tests:
                    result = test_func(telos_results)
                    test_results.append(result)
                    if not result.passed:
                        all_passed = False

                # Show overall status
                if all_passed:
                    st.success(f"✅ **Runtime Simulation VERIFIED** - All {len(test_results)} tests passed")
                else:
                    failed_count = sum(1 for r in test_results if not r.passed)
                    st.warning(f"⚠️ **Validation Issues** - {failed_count}/{len(test_results)} tests failed")

                # Show test details in expander
                with st.expander("📋 View Validation Details", expanded=False):
                    for result in test_results:
                        status_icon = "✅" if result.passed else "❌"
                        status_color = "green" if result.passed else "red"

                        st.markdown(f"""
                        <div style="padding: 8px; border-left: 3px solid {status_color}; margin-bottom: 8px; background: rgba(255,255,255,0.02);">
                            <strong>{status_icon} {result.test_name}</strong><br/>
                            <span style="font-size: 0.9em;">{result.message}</span>
                        </div>
                        """, unsafe_allow_html=True)

                    # Show timing summary if available
                    timing = self.validator.get_timing_summary(telos_results)
                    if timing['turn_count'] > 0:
                        st.markdown("**Timing Summary:**")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Time", f"{timing['total_ms']:.1f} ms")
                        with col2:
                            st.metric("Avg per Turn", f"{timing['avg_ms']:.1f} ms")
                        with col3:
                            st.metric("Min/Max", f"{timing['min_ms']:.1f} / {timing['max_ms']:.1f} ms")

                    # Add methodology statement
                    st.info("""
                    **Runtime Simulation Methodology**: This counterfactual analysis uses
                    pure runtime simulation architecture. Each conversation turn is processed
                    sequentially with access to historical context only (Turns 0 to N-1).
                    No future knowledge or batch analysis artifacts are used.
                    """)

        # Show chart if requested (requires plotly)
        if show_chart:
            try:
                from ..utils.comparison_adapter import ComparisonAdapter
                adapter = ComparisonAdapter()

                st.markdown("---")
                st.markdown("#### 📉 Fidelity Divergence")

                fig = adapter.generate_chart(comparison, chart_type='divergence')
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                st.info("Install plotly for visualization: `pip install plotly`")
            except Exception as e:
                st.warning(f"Could not generate chart: {e}")

    def render_metrics_table(
        self,
        comparison: Dict[str, Any]
    ):
        """
        Render metrics comparison table.

        Args:
            comparison: Comparison dict from ComparisonAdapter

        Example:
            viewer.render_metrics_table(comparison)
        """
        from ..utils.comparison_adapter import ComparisonAdapter

        adapter = ComparisonAdapter()
        df = adapter.generate_metrics_table(comparison)

        st.markdown("### 📋 Detailed Metrics")
        st.dataframe(df, use_container_width=True)

    def render_compact_comparison(
        self,
        baseline_turn: Dict[str, Any],
        telos_turn: Dict[str, Any]
    ):
        """
        Render compact single-line comparison (for dashboards).

        Args:
            baseline_turn: Baseline turn data
            telos_turn: TELOS turn data

        Example:
            viewer.render_compact_comparison(baseline_turn, telos_turn)
        """
        baseline_f = baseline_turn.get('metrics', {}).get('telic_fidelity', 0.0)
        telos_f = telos_turn.get('metrics', {}).get('telic_fidelity', 0.0)
        delta_f = telos_f - baseline_f

        col1, col2, col3 = st.columns([2, 2, 1])

        with col1:
            st.metric("Baseline", f"{baseline_f:.3f}")

        with col2:
            st.metric("TELOS", f"{telos_f:.3f}")

        with col3:
            st.metric("Δ", f"{delta_f:+.3f}")


def render_quick_comparison(
    baseline_data: Dict[str, Any],
    telos_data: Dict[str, Any],
    comparison: Optional[Dict[str, Any]] = None
):
    """
    Convenience function for quick comparison rendering.

    Args:
        baseline_data: Baseline branch data
        telos_data: TELOS branch data
        comparison: Optional pre-computed comparison dict

    Example:
        render_quick_comparison(baseline_branch, telos_branch)
    """
    viewer = ComparisonViewerV2()

    # Render summary if comparison provided
    if comparison:
        viewer.render_summary(comparison)
        st.markdown("---")

    # Render turns
    viewer.render_all_turns(baseline_data, telos_data)
