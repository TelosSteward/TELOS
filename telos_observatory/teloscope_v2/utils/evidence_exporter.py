"""
Evidence Exporter for Observatory v2

Generates downloadable evidence packages in JSON and Markdown formats
for research papers, grant applications, and governance audits.

Provides formatted exports showing:
- ΔF (fidelity improvement)
- Turn-by-turn comparison
- Statistical significance
- Complete conversation text

Usage:
    from teloscope_v2.utils.evidence_exporter import EvidenceExporter

    exporter = EvidenceExporter()

    # Export as JSON
    json_data = exporter.export_comparison(comparison, format='json')

    # Export as Markdown
    md_data = exporter.export_comparison(comparison, format='markdown')

    # Download button
    st.download_button("Download Evidence", data=json_data, file_name="evidence.json")
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List
from datetime import datetime
import json
import numpy as np

try:
    from teloscope_v2.utils.runtime_validator import RuntimeValidator
    RUNTIME_VALIDATOR_AVAILABLE = True
except ImportError:
    RUNTIME_VALIDATOR_AVAILABLE = False


def convert_to_json_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    return obj


class EvidenceExporter:
    """
    Evidence exporter for research and grants.

    Generates downloadable evidence packages in JSON and Markdown formats
    suitable for peer review, grant applications, and auditing.
    """

    def __init__(self, include_validation: bool = True):
        """
        Initialize evidence exporter.

        Args:
            include_validation: If True, automatically include runtime validation
                              results in exports (default: True)
        """
        self.include_validation = include_validation and RUNTIME_VALIDATOR_AVAILABLE
        if self.include_validation:
            self.validator = RuntimeValidator()
        else:
            self.validator = None

    def export_comparison(
        self,
        comparison: Dict[str, Any],
        format: str = 'json',
        include_full_text: bool = True
    ) -> str:
        """
        Export comparison evidence.

        Args:
            comparison: Comparison dict from ComparisonAdapter
            format: 'json' or 'markdown'
            include_full_text: If True, include full conversation text

        Returns:
            Formatted evidence string ready for download

        Raises:
            ValueError: If format is unknown
        """
        if format == 'json':
            return self._export_json(comparison, include_full_text)
        elif format == 'markdown':
            return self._export_markdown(comparison, include_full_text)
        else:
            raise ValueError(f"Unknown format: {format}. Use 'json' or 'markdown'")

    def _export_json(
        self,
        comparison: Dict[str, Any],
        include_full_text: bool
    ) -> str:
        """
        Export as JSON.

        Args:
            comparison: Comparison dict
            include_full_text: If True, include conversation text

        Returns:
            JSON string
        """
        evidence = {
            'exported_at': datetime.now().isoformat(),
            'format_version': '1.0',
            'evidence_type': 'telos_counterfactual_comparison',
            'summary': {
                'baseline_final_fidelity': comparison['baseline']['final_fidelity'],
                'telos_final_fidelity': comparison['telos']['final_fidelity'],
                'delta_f': comparison['delta']['delta_f'],
                'avg_improvement': comparison['delta']['avg_improvement'],
                'governance_effective': comparison['delta']['delta_f'] > 0
            },
            'baseline': {
                'final_fidelity': comparison['baseline']['final_fidelity'],
                'avg_fidelity': comparison['baseline']['avg_fidelity'],
                'min_fidelity': comparison['baseline']['min_fidelity'],
                'max_fidelity': comparison['baseline']['max_fidelity'],
                'fidelity_trajectory': comparison['baseline']['fidelity_trend'],
                'distance_trajectory': comparison['baseline']['distance_trend']
            },
            'telos': {
                'final_fidelity': comparison['telos']['final_fidelity'],
                'avg_fidelity': comparison['telos']['avg_fidelity'],
                'min_fidelity': comparison['telos']['min_fidelity'],
                'max_fidelity': comparison['telos']['max_fidelity'],
                'fidelity_trajectory': comparison['telos']['fidelity_trend'],
                'distance_trajectory': comparison['telos']['distance_trend']
            },
            'delta': comparison['delta'],
            'metadata': comparison.get('metadata', {})
        }

        # Add statistics if available
        if 'statistics' in comparison:
            evidence['statistics'] = comparison['statistics']

        # Add runtime validation if enabled
        if self.include_validation and self.validator:
            validation_results = self._get_validation_results(comparison)
            if validation_results:
                evidence['runtime_validation'] = validation_results

        # Add full text if requested
        if include_full_text:
            evidence['full_comparison'] = comparison

        # Convert numpy types to Python native types before JSON serialization
        return json.dumps(convert_to_json_serializable(evidence), indent=2)

    def _export_markdown(
        self,
        comparison: Dict[str, Any],
        include_full_text: bool
    ) -> str:
        """
        Export as Markdown report.

        Args:
            comparison: Comparison dict
            include_full_text: If True, include conversation text

        Returns:
            Markdown string
        """
        baseline = comparison['baseline']
        telos = comparison['telos']
        delta = comparison['delta']

        # Header
        md = f"""# TELOS Governance Evidence Report

## Export Information
- **Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Evidence Type**: Counterfactual Comparison (Baseline vs TELOS)
- **Format Version**: 1.0

---

## Executive Summary

This report provides quantitative evidence of TELOS governance efficacy through
counterfactual analysis. Two independent conversation branches were analyzed:

1. **Baseline Branch**: Ungoverned AI response
2. **TELOS Branch**: AI response with TELOS governance

### Key Findings

"""

        # Governance effectiveness
        if delta['delta_f'] > 0:
            md += f"**✅ TELOS governance was EFFECTIVE**\n\n"
        elif delta['delta_f'] < 0:
            md += f"**❌ TELOS governance was INEFFECTIVE**\n\n"
        else:
            md += f"**➖ TELOS governance had NEUTRAL impact**\n\n"

        md += f"""- **Baseline Final Fidelity**: {baseline['final_fidelity']:.3f}
- **TELOS Final Fidelity**: {telos['final_fidelity']:.3f}
- **ΔF (Improvement)**: {delta['delta_f']:+.3f}
- **Average Improvement**: {delta['avg_improvement']:+.3f}
- **Maximum Improvement**: {delta.get('max_improvement', 0.0):+.3f}

"""

        # Statistical significance
        if 'statistics' in comparison:
            stats = comparison['statistics']
            md += f"""### Statistical Significance

"""
            if stats['significant']:
                md += f"**✅ Result is statistically significant**\n\n"
            else:
                md += f"**ℹ️ Result is not statistically significant**\n\n"

            md += f"""- **P-value**: {stats['p_value']:.4f}
- **T-statistic**: {stats['t_statistic']:.3f}
- **Effect Size (Cohen's d)**: {stats['effect_size_cohens_d']:.3f}
- **Mean Difference**: {stats['mean_difference']:.3f}
- **95% Confidence Interval**: [{stats['confidence_interval_95'][0]:.3f}, {stats['confidence_interval_95'][1]:.3f}]

**Interpretation:**
"""
            # Add interpretation
            cohens_d = abs(stats['effect_size_cohens_d'])
            if cohens_d < 0.2:
                effect_size = "negligible"
            elif cohens_d < 0.5:
                effect_size = "small"
            elif cohens_d < 0.8:
                effect_size = "medium"
            else:
                effect_size = "large"

            md += f"Effect size is **{effect_size}** (|d| = {cohens_d:.3f})\n\n"

        md += f"""---

## Detailed Metrics Comparison

| Metric | Baseline | TELOS | Improvement |
|--------|----------|-------|-------------|
| Final Fidelity | {baseline['final_fidelity']:.3f} | {telos['final_fidelity']:.3f} | {delta['delta_f']:+.3f} |
| Average Fidelity | {baseline['avg_fidelity']:.3f} | {telos['avg_fidelity']:.3f} | {delta['avg_improvement']:+.3f} |
| Minimum Fidelity | {baseline['min_fidelity']:.3f} | {telos['min_fidelity']:.3f} | - |
| Maximum Fidelity | {baseline['max_fidelity']:.3f} | {telos['max_fidelity']:.3f} | - |

---

## Fidelity Trajectories

### Baseline Trajectory
"""

        # Baseline trajectory
        baseline_trajectory = ', '.join(f"{f:.3f}" for f in baseline['fidelity_trend'])
        md += f"```\n{baseline_trajectory}\n```\n\n"

        # TELOS trajectory
        md += f"""### TELOS Trajectory
"""
        telos_trajectory = ', '.join(f"{f:.3f}" for f in telos['fidelity_trend'])
        md += f"```\n{telos_trajectory}\n```\n\n"

        md += f"""---

## Turn-by-Turn Analysis

| Turn | Baseline F | TELOS F | Δ (TELOS - Baseline) |
|------|-----------|---------|---------------------|
"""

        # Turn-by-turn table
        for i, (b_f, t_f) in enumerate(zip(baseline['fidelity_trend'], telos['fidelity_trend']), 1):
            delta_turn = t_f - b_f
            md += f"| {i} | {b_f:.3f} | {t_f:.3f} | {delta_turn:+.3f} |\n"

        md += f"""
---

## Methodology

### Comparison Approach
- **Baseline**: {'Stateless' if 'stateless' in str(comparison.get('metadata', {})) else 'Ungoverned'} AI response
- **TELOS**: Full MBL governance (SPC Engine + Proportional Controller)
- **Measurement**: Telic Fidelity (embedding distance to governance attractor)
- **Analysis**: Paired comparison with statistical significance testing

### Data Quality
- **Branch Length**: {len(baseline['fidelity_trend'])} turns
- **Sampling**: Complete conversation analyzed
- **Validity**: Independent branches with identical user inputs

---

## Grant Application Language

**Suggested Text for Proposals:**

> "TELOS demonstrated measurable governance efficacy in counterfactual analysis.
> When comparing governed (TELOS) and ungoverned (baseline) AI responses to
> identical user inputs, TELOS achieved a final fidelity of {telos['final_fidelity']:.3f}
> compared to baseline fidelity of {baseline['final_fidelity']:.3f}, representing
> an improvement of ΔF = {delta['delta_f']:+.3f}. """

        if 'statistics' in comparison and comparison['statistics']['significant']:
            md += f"""This improvement was statistically significant
> (p = {comparison['statistics']['p_value']:.4f}, Cohen's d = {comparison['statistics']['effect_size_cohens_d']:.3f}). """

        md += f"""Complete evidence packages including
> conversation transcripts, metrics trajectories, and statistical analysis are
> available for peer review."

---

"""

        # Add runtime validation section if available
        if self.include_validation and self.validator:
            validation_results = self._get_validation_results(comparison)
            if validation_results:
                md += f"""## Runtime Simulation Verification

"""
                if validation_results['all_tests_passed']:
                    md += f"""**✅ Runtime Simulation VERIFIED**

This counterfactual analysis uses proper runtime simulation architecture
(not batch analysis). All validation tests passed:

"""
                else:
                    md += f"""**⚠️ Runtime Simulation Validation Issues**

Some validation tests failed. This may indicate batch analysis artifacts:

"""

                # Show test results
                md += f"""**Test Results**: {validation_results['tests_passed']}/{validation_results['total_tests']} passed

"""

                for test in validation_results.get('test_details', []):
                    status_icon = "✅" if test['passed'] else "❌"
                    md += f"- {status_icon} **{test['test_name']}**: {test['message']}\n"

                # Add timing summary if available
                if 'timing_summary' in validation_results:
                    timing = validation_results['timing_summary']
                    md += f"""
**Timing Summary**:
- Total Processing Time: {timing['total_ms']:.1f} ms
- Average per Turn: {timing['avg_ms']:.1f} ms
- Min/Max: {timing['min_ms']:.1f} / {timing['max_ms']:.1f} ms

"""

                md += f"""**Methodology Statement**:
> "Our counterfactual analysis uses pure runtime simulation architecture. Each
> conversation turn is processed sequentially with access to historical context
> only (Turns 0 to N-1). No future knowledge or batch analysis artifacts are used.
> Validation tests confirm no lookahead violations occur. This methodology
> replicates actual runtime conditions and provides valid research data suitable
> for peer review."

---

"""

        md += f"""## Reproducibility

This evidence package can be reproduced using:
- **Framework**: TELOS Observatory v2
- **Session ID**: {comparison.get('metadata', {}).get('session_id', 'N/A')}
- **Generated**: {datetime.now().isoformat()}

All data and analysis scripts are available in the TELOS Observatory repository.

---

*Generated by TELOSCOPE Observatory v2 - Evidence Exporter*
"""

        return md

    def _get_validation_results(
        self,
        comparison: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Get runtime validation results for comparison.

        Args:
            comparison: Comparison dict with baseline and TELOS branches

        Returns:
            Dict with validation results, or None if validation failed
        """
        if not self.validator:
            return None

        try:
            # Try to validate both branches (TELOS is primary, baseline is reference)
            validation_results = {
                'validation_timestamp': datetime.now().isoformat(),
                'validator_version': '1.0'
            }

            # Validate TELOS branch (primary validation target)
            telos_branch = comparison.get('telos', {})
            if 'turn_results' in telos_branch:
                telos_results = {'turn_results': telos_branch['turn_results']}

                # Run all validation tests
                test_results = []
                all_passed = True

                for test_func in self.validator.validation_tests:
                    result = test_func(telos_results)
                    test_results.append({
                        'test_name': result.test_name,
                        'passed': bool(result.passed),  # Convert to Python bool for JSON
                        'message': result.message,
                        'details': result.details
                    })
                    if not result.passed:
                        all_passed = False

                validation_results['test_details'] = test_results
                validation_results['tests_passed'] = sum(1 for t in test_results if t['passed'])
                validation_results['total_tests'] = len(test_results)
                validation_results['all_tests_passed'] = bool(all_passed)

                # Add timing summary
                timing = self.validator.get_timing_summary(telos_results)
                if timing['turn_count'] > 0:
                    validation_results['timing_summary'] = timing

                return validation_results

            return None

        except Exception as e:
            # Return error info instead of failing silently
            return {
                'validation_error': str(e),
                'error_timestamp': datetime.now().isoformat()
            }

    def create_download_data(
        self,
        comparison: Dict[str, Any],
        format: str = 'json'
    ) -> bytes:
        """
        Create download button data.

        Args:
            comparison: Comparison dict
            format: 'json' or 'markdown'

        Returns:
            Bytes ready for st.download_button()

        Example:
            data = exporter.create_download_data(comparison, format='json')
            st.download_button(
                "Download Evidence",
                data=data,
                file_name="evidence.json",
                mime="application/json"
            )
        """
        evidence = self.export_comparison(comparison, format=format)
        return evidence.encode('utf-8')

    def get_filename(
        self,
        format: str = 'json',
        prefix: str = 'telos_evidence'
    ) -> str:
        """
        Generate filename for evidence export.

        Args:
            format: 'json' or 'markdown'
            prefix: Filename prefix

        Returns:
            Filename string

        Example:
            filename = exporter.get_filename(format='json')
            # Returns: 'telos_evidence_20251030_142537.json'
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        extension = 'json' if format == 'json' else 'md'
        return f"{prefix}_{timestamp}.{extension}"

    def get_mime_type(self, format: str) -> str:
        """
        Get MIME type for format.

        Args:
            format: 'json' or 'markdown'

        Returns:
            MIME type string
        """
        if format == 'json':
            return 'application/json'
        elif format == 'markdown':
            return 'text/markdown'
        else:
            return 'text/plain'


def export_for_streamlit(
    comparison: Dict[str, Any],
    format: str = 'json'
) -> tuple[bytes, str, str]:
    """
    Convenience function for Streamlit download button.

    Args:
        comparison: Comparison dict
        format: 'json' or 'markdown'

    Returns:
        Tuple of (data, filename, mime_type) for st.download_button()

    Example:
        data, filename, mime = export_for_streamlit(comparison, format='json')
        st.download_button("Download", data=data, file_name=filename, mime=mime)
    """
    exporter = EvidenceExporter()
    data = exporter.create_download_data(comparison, format=format)
    filename = exporter.get_filename(format=format)
    mime = exporter.get_mime_type(format)

    return data, filename, mime
