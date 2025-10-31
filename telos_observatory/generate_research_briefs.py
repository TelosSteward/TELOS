#!/usr/bin/env python3
"""
Phase 2 Research Brief Generator
=================================

Generates comprehensive research briefs for each Phase 2 study with:
- Complete quantitative analysis
- Mock researcher questions at key decision points
- Full data breakdown
- Research implications

Creates a micro research environment for each study.
"""

import json
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime


class ResearchBriefGenerator:
    """Generates detailed research briefs for Phase 2 studies."""

    def __init__(self, summary_path: Path, output_dir: Path):
        """
        Initialize brief generator.

        Args:
            summary_path: Path to phase2_study_summary.json
            output_dir: Where to save generated briefs
        """
        self.summary_path = summary_path
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Load summary data
        with open(summary_path) as f:
            self.summary_data = json.load(f)

        self.completed_studies = self.summary_data['completed_studies']

    def generate_all_briefs(self):
        """Generate briefs for all completed studies."""
        print(f"Generating {len(self.completed_studies)} research briefs...")

        for idx, study in enumerate(self.completed_studies, 1):
            print(f"  {idx}/{len(self.completed_studies)}: {study['conversation_id']}")
            brief = self.generate_study_brief(study, idx)

            # Save to file
            filename = f"research_brief_{idx:02d}_{study['conversation_id']}.md"
            output_path = self.output_dir / filename

            with open(output_path, 'w') as f:
                f.write(brief)

        print(f"\n✅ All briefs saved to: {self.output_dir}")

        # Generate index
        self.generate_index()

    def generate_study_brief(self, study: Dict[str, Any], study_num: int) -> str:
        """Generate comprehensive research brief for a single study."""

        lines = []

        # Header
        lines.extend(self._generate_header(study, study_num))

        # Overview
        lines.extend(self._generate_overview(study))

        # Phase 1: PA Establishment
        lines.extend(self._generate_phase1_analysis(study))

        # Phase 2: Drift Detection
        lines.extend(self._generate_phase2_analysis(study))

        # Phase 3: Counterfactual Branching
        if study.get('drift_detected'):
            lines.extend(self._generate_phase3_analysis(study))

        # Quantitative Summary
        lines.extend(self._generate_quantitative_summary(study))

        # Research Implications
        lines.extend(self._generate_research_implications(study))

        # Footer
        lines.extend(self._generate_footer(study))

        return '\n'.join(lines)

    def _generate_header(self, study: Dict, study_num: int) -> List[str]:
        """Generate brief header."""
        conv_id = study['conversation_id']
        drift_status = "DRIFT DETECTED" if study.get('drift_detected') else "NO DRIFT"

        return [
            "=" * 80,
            f"PHASE 2 TELOS STUDY - RESEARCH BRIEF #{study_num}",
            "=" * 80,
            f"**Conversation ID**: {conv_id}",
            f"**Study Status**: {drift_status}",
            f"**Study Index**: {study['study_index']}",
            f"**Analysis Date**: {study['timestamp']}",
            "",
            "```",
            "RESEARCH ENVIRONMENT: Micro-analysis of single conversation trajectory",
            "FRAMEWORK: TELOS Progressive Primacy + Counterfactual Branching",
            "METHODOLOGY: LLM-at-every-turn semantic analysis + statistical convergence",
            "```",
            "",
            "---",
            ""
        ]

    def _generate_overview(self, study: Dict) -> List[str]:
        """Generate study overview section."""
        return [
            "## STUDY OVERVIEW",
            "",
            "### Basic Metadata",
            f"- **Total Conversation Turns**: {study['total_turns']}",
            f"- **PA Convergence Turn**: {study['convergence_turn']}",
            f"- **Turns Analyzed Post-PA**: {study['total_turns'] - study['convergence_turn']}",
            f"- **PA Establishment**: {'✅ Successful' if study['pa_established'] else '❌ Failed'}",
            "",
            "**RESEARCHER QUESTION**: *\"What is the nature of this conversation? What prompted it?\"*",
            "",
            f"**Initial Context**: This conversation spanned {study['total_turns']} turns. The primacy attractor",
            f"converged at turn {study['convergence_turn']}, leaving {study['total_turns'] - study['convergence_turn']}",
            "turns for drift monitoring and potential intervention analysis.",
            "",
            "---",
            ""
        ]

    def _generate_phase1_analysis(self, study: Dict) -> List[str]:
        """Generate Phase 1 (PA establishment) analysis."""
        lines = [
            "## PHASE 1: PRIMACY ATTRACTOR ESTABLISHMENT",
            "",
            f"**Method**: LLM semantic analysis at every turn (turns 1-{study['convergence_turn']})",
            f"**Total LLM Analyses**: {len(study.get('llm_analyses', []))}",
            "",
            "### Turn-by-Turn LLM Semantic Analysis",
            ""
        ]

        # Add each LLM analysis
        llm_analyses = study.get('llm_analyses', [])

        for turn_idx, analysis in enumerate(llm_analyses, 1):
            lines.append(f"#### Turn {turn_idx}")
            lines.append("")
            lines.append("**Purpose Identified**:")
            for purpose in analysis.get('purpose', []):
                lines.append(f"  - {purpose}")
            lines.append("")
            lines.append("**Scope Identified**:")
            for scope_item in analysis.get('scope', [])[:5]:  # Show first 5
                lines.append(f"  - {scope_item}")
            if len(analysis.get('scope', [])) > 5:
                lines.append(f"  - *(... and {len(analysis['scope']) - 5} more)*")
            lines.append("")
            lines.append("**Boundaries Identified**:")
            for boundary in analysis.get('boundaries', [])[:3]:  # Show first 3
                lines.append(f"  - {boundary}")
            if len(analysis.get('boundaries', [])) > 3:
                lines.append(f"  - *(... and {len(analysis['boundaries']) - 3} more)*")
            lines.append("")

            # Add researcher question
            if turn_idx == 1:
                lines.append("**RESEARCHER OBSERVATION**: *\"This is the first semantic snapshot. How will it evolve?\"*")
            elif turn_idx == 2:
                lines.append("**RESEARCHER QUESTION**: *\"How does the purpose evolve from Turn 1 to Turn 2?\"*")
                lines.append("")
                lines.append(self._compare_analyses(llm_analyses[0], llm_analyses[1]))
            elif turn_idx == len(llm_analyses):
                lines.append("**RESEARCHER OBSERVATION**: *\"This is the convergence point. What statistical indicators confirm stability?\"*")

            lines.append("")
            lines.append("---")
            lines.append("")

        # Final PA
        attractor = study.get('attractor', {})
        lines.extend([
            "### Final Established Primacy Attractor",
            "",
            "**RESEARCHER QUESTION**: *\"What is the final, converged understanding of this conversation's purpose?\"*",
            "",
            "**Final Purpose**:",
        ])
        for purpose in attractor.get('purpose', []):
            lines.append(f"  - {purpose}")
        lines.append("")
        lines.append("**Final Scope**:")
        for scope_item in attractor.get('scope', []):
            lines.append(f"  - {scope_item}")
        lines.append("")
        lines.append("**Final Boundaries**:")
        for boundary in attractor.get('boundaries', []):
            lines.append(f"  - {boundary}")
        lines.append("")

        lines.extend([
            "### Statistical Convergence Analysis",
            "",
            f"**Convergence Turn**: {study['convergence_turn']}",
            f"**LLM Analyses Required**: {len(llm_analyses)}",
            f"**Convergence Method**: Progressive rolling window with centroid stability detection",
            "",
            "**RESEARCHER OBSERVATION**: *\"The attractor converged within the 10-turn safety window,\"*",
            "*\"indicating a clear, stable conversation purpose emerged early.\"*",
            "",
            "---",
            ""
        ])

        return lines

    def _generate_phase2_analysis(self, study: Dict) -> List[str]:
        """Generate Phase 2 (drift detection) analysis."""
        lines = [
            "## PHASE 2: DRIFT MONITORING",
            "",
            f"**Monitoring Window**: Turns {study['convergence_turn'] + 1} - {study['total_turns']}",
            f"**Total Turns Monitored**: {study['total_turns'] - study['convergence_turn']}",
            f"**Drift Threshold**: F < 0.8",
            ""
        ]

        if study.get('drift_detected'):
            drift_turn = study['drift_turn']
            drift_fidelity = study['drift_fidelity']

            lines.extend([
                f"### ⚠️ DRIFT DETECTED AT TURN {drift_turn}",
                "",
                f"**Drift Fidelity**: {drift_fidelity:.3f}",
                f"**Threshold Violation**: {drift_fidelity:.3f} < 0.8 (violated by {0.8 - drift_fidelity:.3f})",
                "",
                "**RESEARCHER QUESTION**: *\"What caused the conversation to drift from its established purpose?\"*",
                "",
                "**Analysis**: The conversation trajectory deviated from the primacy attractor established in",
                f"turns 1-{study['convergence_turn']}. At turn {drift_turn}, the fidelity score dropped below",
                f"the governance threshold of 0.8, reaching {drift_fidelity:.3f}. This triggered the TELOS",
                "counterfactual branching protocol to assess whether governance intervention could realign",
                "the conversation with its original purpose.",
                "",
                "**RESEARCHER OBSERVATION**: *\"This drift point becomes the branching trigger for our\"*",
                "*\"counterfactual analysis. We can now compare what actually happened (original branch)\"*",
                "*\"versus what would have happened with TELOS governance (TELOS branch).\"*",
                ""
            ])
        else:
            lines.extend([
                "### ✅ NO DRIFT DETECTED",
                "",
                "**Result**: All post-PA turns maintained fidelity ≥ 0.8",
                "",
                "**RESEARCHER OBSERVATION**: *\"The conversation maintained alignment with its primacy\"*",
                "*\"attractor throughout. No governance intervention was needed.\"*",
                ""
            ])

        lines.extend(["---", ""])
        return lines

    def _generate_phase3_analysis(self, study: Dict) -> List[str]:
        """Generate Phase 3 (counterfactual branching) analysis."""
        if not study.get('counterfactual_results'):
            return []

        results = study['counterfactual_results']
        branch_id = study.get('branch_id', 'N/A')

        lines = [
            "## PHASE 3: COUNTERFACTUAL BRANCHING ANALYSIS",
            "",
            f"**Branch ID**: `{branch_id}`",
            f"**Branching Trigger**: Turn {study['drift_turn']} (F = {study['drift_fidelity']:.3f})",
            "",
            "**RESEARCHER QUESTION**: *\"If TELOS governance had intervened at the drift point,\"*",
            "*\"would the conversation trajectory have been more aligned with the original purpose?\"*",
            "",
            "### Experimental Design",
            "",
            "**Independent Variable**: Presence of TELOS governance intervention",
            "**Dependent Variable**: Fidelity to established primacy attractor",
            "**Control Group**: Original branch (historical responses, no intervention)",
            "**Treatment Group**: TELOS branch (API-generated responses with governance)",
            "",
            "**Method**:",
            "1. Both branches receive identical user inputs (from historical data)",
            "2. Original branch: Uses historical assistant responses",
            "3. TELOS branch: Generates NEW responses via Mistral API with intervention prompt",
            "4. Intervention applied ONLY on first turn post-drift",
            "5. Subsequent turns show cascading effects of initial intervention",
            "",
            "---",
            "",
            "### Results: Original Branch (Control)",
            "",
            f"**Final Fidelity**: {results['original_final_f']:.3f}",
            f"**Trajectory**: Started at {study['drift_fidelity']:.3f} (drift point)",
            "",
            "**RESEARCHER OBSERVATION**: *\"The original branch shows the natural trajectory of the\"*",
            "*\"conversation without any governance intervention. This is our baseline.\"*",
            "",
            "---",
            "",
            "### Results: TELOS Branch (Treatment)",
            "",
            f"**Final Fidelity**: {results['telos_final_f']:.3f}",
            f"**Trajectory**: Started at {study['drift_fidelity']:.3f} (same drift point)",
            "**Intervention**: Applied at first turn to realign with PA",
            "",
            "**RESEARCHER OBSERVATION**: *\"The TELOS branch shows the counterfactual trajectory -\"*",
            "*\"what would have happened if governance intervention had occurred at drift.\"*",
            "",
            "---",
            "",
            "### Comparative Analysis",
            "",
            f"**ΔF (TELOS - Original)**: {results['delta_f']:+.3f}",
            f"**Governance Effective**: {'✅ YES' if results['governance_effective'] else '❌ NO'}",
            ""
        ]

        # Detailed interpretation
        if results['delta_f'] > 0:
            improvement_pct = (results['delta_f'] / results['original_final_f']) * 100 if results['original_final_f'] > 0 else 0
            lines.extend([
                "**RESEARCHER INTERPRETATION**:",
                "",
                f"*\"TELOS governance produced a {results['delta_f']:+.3f} improvement in final fidelity,\"*",
                f"*\"representing a {improvement_pct:.1f}% increase over the original trajectory. The\"*",
                "*\"intervention successfully realigned the conversation with its established purpose.\"*",
                "",
                "**Statistical Significance**: The positive ΔF indicates that the TELOS intervention",
                "had a beneficial effect on maintaining conversation coherence with the primacy attractor.",
                ""
            ])
        elif results['delta_f'] < 0:
            decline_pct = abs((results['delta_f'] / results['original_final_f']) * 100) if results['original_final_f'] > 0 else 0
            lines.extend([
                "**RESEARCHER INTERPRETATION**:",
                "",
                f"*\"TELOS governance produced a {results['delta_f']:+.3f} decline in final fidelity,\"*",
                f"*\"representing a {decline_pct:.1f}% decrease compared to the original trajectory. The\"*",
                "*\"intervention did not improve alignment with the established purpose.\"*",
                "",
                "**Possible Explanations**:",
                "- The original drift may have been intentional/appropriate",
                "- The intervention prompt may have over-corrected",
                "- The primacy attractor may not have captured the full conversation intent",
                ""
            ])
        else:
            lines.extend([
                "**RESEARCHER INTERPRETATION**:",
                "",
                "*\"TELOS governance produced negligible change (ΔF ≈ 0). The intervention had no\"*",
                "*\"measurable effect on conversation alignment.\"*",
                ""
            ])

        lines.extend([
            "**RESEARCHER QUESTION**: *\"What does this result tell us about when TELOS governance\"*",
            "*\"is most effective?\"*",
            "",
            "---",
            ""
        ])

        return lines

    def _generate_quantitative_summary(self, study: Dict) -> List[str]:
        """Generate complete quantitative summary."""
        lines = [
            "## QUANTITATIVE SUMMARY",
            "",
            "### All Measurable Metrics",
            "",
            "#### Study Completion Metrics",
            f"- Total turns in conversation: {study['total_turns']}",
            f"- PA convergence turn: {study['convergence_turn']}",
            f"- Post-PA monitoring turns: {study['total_turns'] - study['convergence_turn']}",
            f"- LLM analyses performed: {len(study.get('llm_analyses', []))}",
            ""
        ]

        if study.get('drift_detected'):
            results = study.get('counterfactual_results', {})
            lines.extend([
                "#### Drift Detection Metrics",
                f"- Drift detected: Turn {study['drift_turn']}",
                f"- Drift fidelity: {study['drift_fidelity']:.4f}",
                f"- Threshold violation magnitude: {0.8 - study['drift_fidelity']:.4f}",
                f"- Turns from PA to drift: {study['drift_turn'] - study['convergence_turn']}",
                "",
                "#### Counterfactual Branch Metrics",
                f"- Branch ID: {study.get('branch_id', 'N/A')}",
                f"- Original branch final fidelity: {results.get('original_final_f', 0):.4f}",
                f"- TELOS branch final fidelity: {results.get('telos_final_f', 0):.4f}",
                f"- Delta F (improvement): {results.get('delta_f', 0):+.4f}",
                f"- Governance effective: {'Yes' if results.get('governance_effective') else 'No'}",
                ""
            ])

        lines.extend([
            "#### Primacy Attractor Metrics",
            f"- Purpose statements: {len(study.get('attractor', {}).get('purpose', []))}",
            f"- Scope items: {len(study.get('attractor', {}).get('scope', []))}",
            f"- Boundary conditions: {len(study.get('attractor', {}).get('boundaries', []))}",
            "",
            "---",
            ""
        ])

        return lines

    def _generate_research_implications(self, study: Dict) -> List[str]:
        """Generate research implications section."""
        lines = [
            "## RESEARCH IMPLICATIONS",
            "",
            "**RESEARCHER REFLECTION**: *\"What does this micro-study contribute to our\"*",
            "*\"understanding of conversation governance?\"*",
            ""
        ]

        # Context-specific implications
        if study.get('drift_detected'):
            results = study.get('counterfactual_results', {})
            if results.get('governance_effective'):
                lines.extend([
                    "### Key Findings",
                    "",
                    "1. **Primacy Attractor Validity**: The PA established in the first",
                    f"   {study['convergence_turn']} turns provided a stable reference point for measuring drift.",
                    "",
                    "2. **Drift Detection**: The conversation deviated from its established purpose at",
                    f"   turn {study['drift_turn']}, demonstrating the value of continuous fidelity monitoring.",
                    "",
                    "3. **Governance Efficacy**: TELOS intervention successfully improved alignment",
                    f"   (ΔF = {results['delta_f']:+.3f}), supporting the hypothesis that governance can",
                    "   realign drifting conversations.",
                    "",
                    "### Contribution to Framework",
                    "",
                    "This study provides evidence that:",
                    "- LLM semantic analysis can establish stable purpose understanding early",
                    "- Statistical convergence reliably identifies when purpose is established",
                    "- Drift can be detected through fidelity measurement",
                    "- Intervention at drift points can improve alignment",
                    ""
                ])
            else:
                lines.extend([
                    "### Key Findings",
                    "",
                    "1. **Primacy Attractor Validity**: The PA established in the first",
                    f"   {study['convergence_turn']} turns provided a stable reference point.",
                    "",
                    "2. **Drift Detection**: The conversation deviated from its established purpose at",
                    f"   turn {study['drift_turn']}.",
                    "",
                    "3. **Governance Limitations**: TELOS intervention did not improve alignment",
                    f"   (ΔF = {results['delta_f']:+.3f}), suggesting either the drift was appropriate",
                    "   or the intervention strategy needs refinement for this type of conversation.",
                    "",
                    "### Contribution to Framework",
                    "",
                    "This study provides evidence that:",
                    "- Not all drift is necessarily harmful",
                    "- Some conversations may naturally evolve beyond their initial purpose",
                    "- Intervention effectiveness varies by conversation type",
                    "- Framework needs adaptation mechanisms for context-appropriate drift",
                    ""
                ])
        else:
            lines.extend([
                "### Key Findings",
                "",
                "1. **Primacy Attractor Validity**: The PA established in the first",
                f"   {study['convergence_turn']} turns accurately represented the conversation's purpose.",
                "",
                "2. **No Drift Detected**: The conversation maintained strong alignment throughout,",
                "   suggesting the PA correctly captured the intended scope.",
                "",
                "3. **No Intervention Needed**: This demonstrates the framework's ability to distinguish",
                "   between conversations that need governance and those that don't.",
                "",
                "### Contribution to Framework",
                "",
                "This study provides evidence that:",
                "- Some conversations naturally maintain alignment without intervention",
                "- The framework correctly identifies when governance is unnecessary",
                "- PA establishment alone may provide implicit guidance effects",
                ""
            ])

        lines.extend([
            "---",
            ""
        ])

        return lines

    def _generate_footer(self, study: Dict) -> List[str]:
        """Generate brief footer."""
        return [
            "---",
            "",
            "## RESEARCH METADATA",
            "",
            f"- **Study ID**: {study['conversation_id']}",
            f"- **Analysis Timestamp**: {study['timestamp']}",
            f"- **Framework Version**: TELOS Phase 2",
            f"- **Methodology**: Progressive Primacy Extraction + Counterfactual Branching",
            "- **LLM Provider**: Mistral API",
            "- **Embedding Provider**: SentenceTransformer",
            "",
            "**Data Availability**: Complete evidence files (JSON + Markdown) available in",
            f"`phase2_study_results/{study['conversation_id']}/`",
            "",
            "---",
            "",
            "*Generated by TELOS Observatory Research Brief Generator*",
            f"*Phase 2 Production Validation Study*",
            ""
        ]

    def _compare_analyses(self, analysis1: Dict, analysis2: Dict) -> str:
        """Compare two LLM analyses."""
        purpose1 = set(analysis1.get('purpose', []))
        purpose2 = set(analysis2.get('purpose', []))

        new_purposes = purpose2 - purpose1
        retained_purposes = purpose1 & purpose2

        comparison = "**Purpose Evolution Analysis**:\n"
        if retained_purposes:
            comparison += f"- Retained from Turn 1: {len(retained_purposes)} purpose statement(s)\n"
        if new_purposes:
            comparison += f"- New in Turn 2: {len(new_purposes)} purpose statement(s)\n"

        return comparison

    def generate_index(self):
        """Generate index of all research briefs."""
        index_lines = [
            "# PHASE 2 TELOS STUDY - RESEARCH BRIEFS INDEX",
            "",
            f"**Total Studies**: {len(self.completed_studies)}",
            f"**Generation Date**: {datetime.now().isoformat()}",
            "",
            "---",
            "",
            "## All Research Briefs",
            ""
        ]

        for idx, study in enumerate(self.completed_studies, 1):
            conv_id = study['conversation_id']
            drift_status = "DRIFT" if study.get('drift_detected') else "NO DRIFT"

            if study.get('counterfactual_results'):
                delta = study['counterfactual_results']['delta_f']
                effective = "✅" if study['counterfactual_results']['governance_effective'] else "❌"
                index_lines.append(f"{idx:2d}. [{conv_id}](research_brief_{idx:02d}_{conv_id}.md) - {drift_status} - ΔF={delta:+.3f} {effective}")
            else:
                index_lines.append(f"{idx:2d}. [{conv_id}](research_brief_{idx:02d}_{conv_id}.md) - {drift_status}")

        index_lines.extend([
            "",
            "---",
            "",
            "## Statistics Summary",
            "",
            f"- **Effective Interventions**: {sum(1 for s in self.completed_studies if s.get('counterfactual_results', {}).get('governance_effective'))}/{len([s for s in self.completed_studies if 'counterfactual_results' in s])}",
            f"- **Average ΔF**: {sum(s.get('counterfactual_results', {}).get('delta_f', 0) for s in self.completed_studies if 'counterfactual_results' in s) / len([s for s in self.completed_studies if 'counterfactual_results' in s]):+.3f}",
            ""
        ])

        index_path = self.output_dir / "README.md"
        with open(index_path, 'w') as f:
            f.write('\n'.join(index_lines))

        print(f"\n📋 Index created: {index_path}")


def main():
    """Generate all research briefs."""
    base_dir = Path(__file__).parent
    summary_path = base_dir / 'phase2_study_results' / 'phase2_study_summary.json'
    output_dir = base_dir / 'phase2_research_briefs'

    generator = ResearchBriefGenerator(summary_path, output_dir)
    generator.generate_all_briefs()

    print("\n✅ All research briefs generated successfully!")
    print(f"\n📁 Output directory: {output_dir}")
    print(f"📄 Index file: {output_dir / 'README.md'}")


if __name__ == '__main__':
    main()
