"""
Steward Planning Mode Logger
=============================
Logs planning decisions and data extraction progress for Technical Deep Dive Compendium.

This module tracks:
- Planning decisions and rationale
- Data source inventory
- Section dependencies
- Quality gate progress
- Integration decisions

Created: 2025-01-12 (Compendium Phase 1)
"""

import json
import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional


class StewardPlanningLogger:
    """Logger for planning mode decisions during Compendium authoring."""

    def __init__(self, log_dir: str = "healthcare_validation"):
        """Initialize planning logger.

        Args:
            log_dir: Directory for planning logs
        """
        self.log_dir = Path(log_dir)
        self.log_file = self.log_dir / "planning_log.jsonl"
        self.session_start = datetime.datetime.now().isoformat()

        # Initialize log file
        self._init_log()

    def _init_log(self):
        """Initialize planning log with session header."""
        # Ensure log directory exists
        self.log_dir.mkdir(parents=True, exist_ok=True)

        session_header = {
            "event_type": "session_start",
            "timestamp": self.session_start,
            "document": "Technical Deep Dive Compendium",
            "phase": "Phase 1 - Data Extraction & Inventory",
            "purpose": "Track planning decisions for rigorous technical documentation"
        }
        self._append_log(session_header)

    def _append_log(self, entry: Dict[str, Any]):
        """Append entry to planning log.

        Args:
            entry: Log entry dictionary
        """
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')

    def log_data_source(
        self,
        source_path: str,
        source_type: str,
        purpose: str,
        key_findings: List[str]
    ):
        """Log data source extraction.

        Args:
            source_path: Path to source file
            source_type: Type of source (e.g., "attack_library", "validation_results")
            purpose: Why this source is needed
            key_findings: List of key findings from source
        """
        entry = {
            "event_type": "data_source_extracted",
            "timestamp": datetime.datetime.now().isoformat(),
            "source_path": source_path,
            "source_type": source_type,
            "purpose": purpose,
            "key_findings": key_findings
        }
        self._append_log(entry)

    def log_planning_decision(
        self,
        decision_type: str,
        rationale: str,
        alternatives_considered: Optional[List[str]] = None,
        impact: Optional[str] = None
    ):
        """Log a planning decision.

        Args:
            decision_type: Type of decision (e.g., "section_structure", "data_source_selection")
            rationale: Why this decision was made
            alternatives_considered: Other options that were considered
            impact: Expected impact on document quality
        """
        entry = {
            "event_type": "planning_decision",
            "timestamp": datetime.datetime.now().isoformat(),
            "decision_type": decision_type,
            "rationale": rationale,
            "alternatives_considered": alternatives_considered or [],
            "impact": impact
        }
        self._append_log(entry)

    def log_section_dependency(
        self,
        section_number: int,
        section_title: str,
        depends_on_sections: List[int],
        depends_on_data_sources: List[str],
        rationale: str
    ):
        """Log section dependencies for integration planning.

        Args:
            section_number: Section number (1-10)
            section_title: Section title
            depends_on_sections: List of section numbers this depends on
            depends_on_data_sources: List of data source types needed
            rationale: Why these dependencies exist
        """
        entry = {
            "event_type": "section_dependency",
            "timestamp": datetime.datetime.now().isoformat(),
            "section_number": section_number,
            "section_title": section_title,
            "depends_on_sections": depends_on_sections,
            "depends_on_data_sources": depends_on_data_sources,
            "rationale": rationale
        }
        self._append_log(entry)

    def log_quality_gate_progress(
        self,
        section_number: int,
        quality_gate: str,
        status: str,
        evidence: str,
        notes: Optional[str] = None
    ):
        """Log quality gate verification progress.

        Args:
            section_number: Section number (1-10)
            quality_gate: Which gate (reproducibility, completeness, mathematical_accuracy,
                          code_data_alignment, regulatory_precision, peer_review_readiness)
            status: Status (not_started, in_progress, passed, failed)
            evidence: Evidence for gate status
            notes: Additional notes
        """
        entry = {
            "event_type": "quality_gate_progress",
            "timestamp": datetime.datetime.now().isoformat(),
            "section_number": section_number,
            "quality_gate": quality_gate,
            "status": status,
            "evidence": evidence,
            "notes": notes
        }
        self._append_log(entry)

    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of current planning session.

        Returns:
            Dictionary with session summary statistics
        """
        # Read log file
        entries = []
        with open(self.log_file, 'r') as f:
            for line in f:
                entries.append(json.loads(line))

        # Count event types
        event_counts = {}
        for entry in entries:
            event_type = entry.get("event_type", "unknown")
            event_counts[event_type] = event_counts.get(event_type, 0) + 1

        return {
            "session_start": self.session_start,
            "total_events": len(entries),
            "event_counts": event_counts,
            "data_sources_extracted": event_counts.get("data_source_extracted", 0),
            "planning_decisions": event_counts.get("planning_decision", 0),
            "section_dependencies_mapped": event_counts.get("section_dependency", 0),
            "quality_gates_logged": event_counts.get("quality_gate_progress", 0)
        }


if __name__ == '__main__':
    # Test logger
    logger = StewardPlanningLogger()

    # Log a test data source
    logger.log_data_source(
        source_path="healthcare_validation/attacks/healthcare_attack_library.py",
        source_type="attack_library",
        purpose="Extract 30 HIPAA-specific attacks for Section 9 (Healthcare Validation Deep Dive)",
        key_findings=[
            "30 attacks across 5 categories",
            "Sophistication levels 1-4",
            "Each attack maps to specific HIPAA CFR provisions"
        ]
    )

    # Log a test planning decision
    logger.log_planning_decision(
        decision_type="section_structure",
        rationale="Section 9 (Healthcare) before Section 10 (Future Research) because healthcare validation is complete and provides foundation for roadmap",
        alternatives_considered=[
            "Healthcare as Appendix",
            "Healthcare integrated into Section 4 (Results)"
        ],
        impact="Allows Section 9 to serve as template for future domain-specific validations"
    )

    # Print summary
    summary = logger.get_session_summary()
    print("\n" + "="*60)
    print("STEWARD PLANNING LOG SUMMARY")
    print("="*60)
    for key, value in summary.items():
        print(f"{key}: {value}")
