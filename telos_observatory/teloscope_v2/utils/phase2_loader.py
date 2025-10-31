"""
Phase 2 Results Loader
======================

Loads Phase 2 validation results into Observatory UI.

Functions:
- load_study_summary: Load aggregate summary JSON
- load_study_evidence: Load individual study intervention data
- load_research_brief: Load research brief markdown
- get_available_studies: List all completed studies
- get_study_statistics: Calculate aggregate statistics
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class StudyMetadata:
    """Metadata for a single Phase 2 or Phase 2B study."""
    conversation_id: str
    study_index: int
    dataset: str  # 'sharegpt', 'test_sessions', 'edge_cases'
    pa_established: bool
    convergence_turn: Optional[int]
    drift_detected: bool

    # Phase 2 fields (single intervention)
    drift_turn: Optional[int]
    drift_fidelity: Optional[float]
    delta_f: Optional[float]
    governance_effective: Optional[bool]

    # Phase 2B fields (multiple interventions)
    phase: str = '2'  # '2' or '2B'
    total_interventions: Optional[int] = None
    interventions: Optional[List[Dict]] = None
    aggregate_metrics: Optional[Dict] = None

    total_turns: int = 0
    timestamp: str = ''


class Phase2Loader:
    """Load Phase 2 validation results for Observatory UI."""

    def __init__(self, base_dir: Optional[Path] = None):
        """
        Initialize loader.

        Args:
            base_dir: Base directory containing phase2_study_results
                     Defaults to telos_observatory/
        """
        if base_dir is None:
            base_dir = Path(__file__).parent.parent.parent

        self.base_dir = Path(base_dir)
        self.results_dir = self.base_dir / 'phase2_study_results'
        self.briefs_dir = self.base_dir / 'phase2_research_briefs'

        # Validation result directories (Phase 2 and Phase 2B)
        self.validation_dirs = [
            self.base_dir / 'phase2_study_results',
            self.base_dir / 'phase2_validation_test_sessions' / 'study_results',
            self.base_dir / 'phase2_validation_edge_cases' / 'study_results',
        ]

        # Research brief directories
        self.brief_dirs = [
            self.base_dir / 'phase2_research_briefs',
            self.base_dir / 'phase2_validation_test_sessions' / 'research_briefs',
            self.base_dir / 'phase2_validation_edge_cases' / 'research_briefs',
        ]

        # Phase 2B continuous monitoring directories (will be auto-discovered)
        self.phase2b_dirs = []
        self._discover_phase2b_dirs()

    def _discover_phase2b_dirs(self):
        """Discover Phase 2B continuous monitoring result directories."""
        # Look for directories matching phase2b_continuous_* pattern
        for path in self.base_dir.glob('phase2b_continuous_*'):
            if path.is_dir():
                summary_file = path / 'phase2b_continuous_summary.json'
                if summary_file.exists():
                    self.phase2b_dirs.append(path)

    def load_study_summary(self, summary_path: Optional[Path] = None) -> Dict:
        """
        Load phase2_study_summary.json

        Args:
            summary_path: Path to summary JSON (optional)

        Returns:
            Dict with completed_studies, failed_studies, summary, timestamp
        """
        if summary_path is None:
            summary_path = self.results_dir / 'phase2_study_summary.json'

        if not summary_path.exists():
            return {
                'completed_studies': [],
                'failed_studies': [],
                'summary': {},
                'timestamp': None
            }

        with open(summary_path) as f:
            return json.load(f)

    def load_all_summaries(self) -> List[Dict]:
        """Load all study summaries from all validation directories (Phase 2 and Phase 2B)."""
        all_summaries = []

        # Load Phase 2 summaries
        for results_dir in self.validation_dirs:
            summary_path = results_dir / 'phase2_study_summary.json'
            if summary_path.exists():
                summary = self.load_study_summary(summary_path)
                all_summaries.append({
                    'source': results_dir.parent.name,
                    'data': summary,
                    'phase': '2'
                })

        # Load Phase 2B summaries
        for phase2b_dir in self.phase2b_dirs:
            summary_path = phase2b_dir / 'phase2b_continuous_summary.json'
            if summary_path.exists():
                with open(summary_path) as f:
                    summary = json.load(f)
                all_summaries.append({
                    'source': phase2b_dir.name,
                    'data': summary,
                    'phase': '2B'
                })

        return all_summaries

    def get_available_studies(self, dataset_filter: Optional[str] = None) -> List[StudyMetadata]:
        """
        Get list of all available studies with metadata (Phase 2 and Phase 2B).

        Args:
            dataset_filter: Filter by dataset ('sharegpt', 'test_sessions', 'edge_cases', 'phase2b')

        Returns:
            List of StudyMetadata objects
        """
        studies = []

        summaries = self.load_all_summaries()

        for summary_info in summaries:
            source = summary_info['source']
            summary = summary_info['data']
            phase = summary_info.get('phase', '2')

            # Determine dataset from source
            if phase == '2B':
                # Phase 2B studies get special dataset tag
                dataset = 'phase2b'
            elif 'test_sessions' in source:
                dataset = 'test_sessions'
            elif 'edge_cases' in source:
                dataset = 'edge_cases'
            else:
                dataset = 'sharegpt'

            # Apply filter if specified
            if dataset_filter and dataset != dataset_filter:
                continue

            for study in summary.get('completed_studies', []):
                # Check if this is a Phase 2B study (has total_interventions)
                is_phase2b = 'total_interventions' in study

                if is_phase2b:
                    # Phase 2B: Multiple interventions
                    metadata = StudyMetadata(
                        conversation_id=study['conversation_id'],
                        study_index=study['study_index'],
                        dataset=dataset,
                        pa_established=study.get('pa_established', False),
                        convergence_turn=study.get('convergence_turn'),
                        drift_detected=study.get('drift_detected', False),
                        # Phase 2 fields (None for 2B)
                        drift_turn=None,
                        drift_fidelity=None,
                        delta_f=study.get('aggregate_metrics', {}).get('mean_delta_f'),  # Use mean for Phase 2B
                        governance_effective=None,  # No single value for 2B
                        # Phase 2B fields
                        phase='2B',
                        total_interventions=study.get('total_interventions'),
                        interventions=study.get('interventions', []),
                        aggregate_metrics=study.get('aggregate_metrics', {}),
                        total_turns=study.get('total_turns', 0),
                        timestamp=study.get('timestamp', '')
                    )
                else:
                    # Phase 2: Single intervention
                    metadata = StudyMetadata(
                        conversation_id=study['conversation_id'],
                        study_index=study['study_index'],
                        dataset=dataset,
                        pa_established=study.get('pa_established', False),
                        convergence_turn=study.get('convergence_turn'),
                        drift_detected=study.get('drift_detected', False),
                        # Phase 2 fields
                        drift_turn=study.get('drift_turn'),
                        drift_fidelity=study.get('drift_fidelity'),
                        delta_f=study.get('counterfactual_results', {}).get('delta_f') if study.get('drift_detected') else None,
                        governance_effective=study.get('counterfactual_results', {}).get('governance_effective') if study.get('drift_detected') else None,
                        # Phase 2B fields (None for Phase 2)
                        phase='2',
                        total_interventions=None,
                        interventions=None,
                        aggregate_metrics=None,
                        total_turns=study.get('total_turns', 0),
                        timestamp=study.get('timestamp', '')
                    )
                studies.append(metadata)

        # Sort by phase (2 before 2B), then dataset, then study index
        studies.sort(key=lambda s: (s.phase, s.dataset, s.study_index))

        return studies

    def load_study_evidence(self, conversation_id: str, dataset: str = 'sharegpt') -> Optional[Dict]:
        """
        Load intervention evidence JSON for a specific study.

        Args:
            conversation_id: ID of conversation
            dataset: Which dataset ('sharegpt', 'test_sessions', 'edge_cases')

        Returns:
            Dict with intervention data or None if not found
        """
        # Determine results directory
        if dataset == 'test_sessions':
            results_dir = self.base_dir / 'phase2_validation_test_sessions' / 'study_results'
        elif dataset == 'edge_cases':
            results_dir = self.base_dir / 'phase2_validation_edge_cases' / 'study_results'
        else:
            results_dir = self.results_dir

        study_dir = results_dir / conversation_id

        if not study_dir.exists():
            return None

        # Find intervention JSON file
        intervention_files = list(study_dir.glob('intervention_*.json'))

        if not intervention_files:
            return None

        # Load first (should only be one)
        with open(intervention_files[0]) as f:
            return json.load(f)

    def load_research_brief(self, conversation_id: str, dataset: str = 'sharegpt') -> Optional[str]:
        """
        Load research brief markdown for a specific study.

        Args:
            conversation_id: ID of conversation
            dataset: Which dataset

        Returns:
            Markdown string or None if not found
        """
        # Determine briefs directory
        if dataset == 'test_sessions':
            briefs_dir = self.base_dir / 'phase2_validation_test_sessions' / 'research_briefs'
        elif dataset == 'edge_cases':
            briefs_dir = self.base_dir / 'phase2_validation_edge_cases' / 'research_briefs'
        else:
            briefs_dir = self.briefs_dir

        # Find brief file (may have different naming patterns)
        brief_files = list(briefs_dir.glob(f'*{conversation_id}*.md'))

        if not brief_files:
            return None

        # Load first match
        with open(brief_files[0]) as f:
            return f.read()

    def get_study_statistics(self, dataset_filter: Optional[str] = None) -> Dict:
        """
        Calculate aggregate statistics across studies.

        Args:
            dataset_filter: Filter by dataset

        Returns:
            Dict with statistics
        """
        studies = self.get_available_studies(dataset_filter)

        if not studies:
            return {
                'total_studies': 0,
                'with_drift': 0,
                'without_drift': 0,
                'effective_interventions': 0,
                'ineffective_interventions': 0,
                'average_delta_f': 0,
                'delta_f_range': (0, 0),
                'effectiveness_rate': 0
            }

        # Filter to studies with drift (only they have counterfactuals)
        drift_studies = [s for s in studies if s.drift_detected]

        # Calculate metrics
        effective = [s for s in drift_studies if s.governance_effective]
        ineffective = [s for s in drift_studies if not s.governance_effective]

        delta_f_values = [s.delta_f for s in drift_studies if s.delta_f is not None]

        avg_delta_f = sum(delta_f_values) / len(delta_f_values) if delta_f_values else 0
        delta_f_range = (min(delta_f_values), max(delta_f_values)) if delta_f_values else (0, 0)

        effectiveness_rate = len(effective) / len(drift_studies) if drift_studies else 0

        return {
            'total_studies': len(studies),
            'with_drift': len(drift_studies),
            'without_drift': len(studies) - len(drift_studies),
            'effective_interventions': len(effective),
            'ineffective_interventions': len(ineffective),
            'average_delta_f': avg_delta_f,
            'delta_f_range': delta_f_range,
            'effectiveness_rate': effectiveness_rate
        }

    def get_dataset_comparison(self) -> Dict[str, Dict]:
        """
        Get statistics broken down by dataset.

        Returns:
            Dict mapping dataset name to statistics
        """
        datasets = ['sharegpt', 'test_sessions', 'edge_cases']

        comparison = {}
        for dataset in datasets:
            stats = self.get_study_statistics(dataset_filter=dataset)
            if stats['total_studies'] > 0:
                comparison[dataset] = stats

        return comparison

    def get_delta_f_distribution(self, dataset_filter: Optional[str] = None) -> List[Tuple[str, float]]:
        """
        Get list of (conversation_id, delta_f) tuples for plotting.

        Args:
            dataset_filter: Filter by dataset

        Returns:
            List of (conversation_id, delta_f) tuples sorted by delta_f
        """
        studies = self.get_available_studies(dataset_filter)

        # Filter to studies with drift and delta_f
        distribution = [
            (s.conversation_id, s.delta_f)
            for s in studies
            if s.drift_detected and s.delta_f is not None
        ]

        # Sort by delta_f (descending)
        distribution.sort(key=lambda x: x[1], reverse=True)

        return distribution

    def search_studies(self, query: str) -> List[StudyMetadata]:
        """
        Search studies by conversation ID or dataset.

        Args:
            query: Search query string

        Returns:
            List of matching StudyMetadata objects
        """
        all_studies = self.get_available_studies()

        query_lower = query.lower()

        matches = [
            s for s in all_studies
            if query_lower in s.conversation_id.lower() or query_lower in s.dataset.lower()
        ]

        return matches


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def load_phase2_data(base_dir: Optional[Path] = None) -> Phase2Loader:
    """
    Convenience function to create loader instance.

    Args:
        base_dir: Base directory (defaults to telos_observatory/)

    Returns:
        Phase2Loader instance
    """
    return Phase2Loader(base_dir)


def get_study_list(dataset: Optional[str] = None) -> List[Dict]:
    """
    Get list of studies as dicts (for UI display).

    Args:
        dataset: Filter by dataset

    Returns:
        List of study dicts with display info
    """
    loader = Phase2Loader()
    studies = loader.get_available_studies(dataset)

    return [
        {
            'id': s.conversation_id,
            'index': s.study_index,
            'dataset': s.dataset,
            'turns': s.total_turns,
            'drift': s.drift_detected,
            'delta_f': s.delta_f,
            'effective': s.governance_effective,
            'timestamp': s.timestamp
        }
        for s in studies
    ]


def get_quick_stats() -> Dict:
    """Get quick statistics for dashboard display."""
    loader = Phase2Loader()
    return loader.get_study_statistics()


if __name__ == '__main__':
    """Test the loader."""
    print("Testing Phase 2 Loader...")
    print("=" * 70)

    loader = Phase2Loader()

    # Test 1: Load studies
    print("\n1. Loading available studies...")
    studies = loader.get_available_studies()
    print(f"   Found {len(studies)} studies")

    # Test 2: Get statistics
    print("\n2. Calculating statistics...")
    stats = loader.get_study_statistics()
    print(f"   Total: {stats['total_studies']}")
    print(f"   With drift: {stats['with_drift']}")
    print(f"   Effective: {stats['effective_interventions']}/{stats['with_drift']}")
    print(f"   Average ΔF: {stats['average_delta_f']:+.3f}")

    # Test 3: Dataset comparison
    print("\n3. Dataset comparison...")
    comparison = loader.get_dataset_comparison()
    for dataset, data in comparison.items():
        print(f"   {dataset}: {data['effectiveness_rate']*100:.1f}% effective")

    # Test 4: Load specific study
    print("\n4. Loading specific study evidence...")
    if studies:
        first_study = studies[0]
        evidence = loader.load_study_evidence(
            first_study.conversation_id,
            first_study.dataset
        )
        if evidence:
            print(f"   Loaded evidence for {first_study.conversation_id}")
            print(f"   Branch ID: {evidence.get('branch_id')}")
        else:
            print(f"   No evidence found for {first_study.conversation_id}")

    print("\n" + "=" * 70)
    print("Loader test complete!")
