"""
The Constitutional Filter: Governance Integration Module

Orchestration-layer governance enforcement for session-level constitutional law.

This module provides the integration layer between Steward PM and TELOS core
components, enabling constitutional actors to author and enforce governance
requirements through the Primacy Attractor (instantiated constitutional law).
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from telos.core.unified_steward import UnifiedGovernanceSteward, PrimacyAttractor
from steward.steward_governance_orchestrator import StewardGovernanceOrchestrator


class TelosGovernance:
    """
    The Constitutional Filter governance interface for Steward PM integration.

    Enables constitutional actors to author session-level governance requirements
    and enforces those requirements as constitutional law through the Primacy
    Attractor. Provides high-level governance summary and compliance metrics.
    """

    def __init__(self):
        self.orchestrator = StewardGovernanceOrchestrator()
        self.session_active = False
        self.current_steward: Optional[UnifiedGovernanceSteward] = None

    def display_governance_summary(self):
        """Display comprehensive governance status"""
        print("\n" + "="*70)
        print("🎯 TELOS GOVERNANCE STATUS")
        print("="*70 + "\n")

        if self.session_active and self.current_steward:
            print("✅ Governance Session: ACTIVE")
            print(f"   Mode: {'Monitor-Only' if not self.current_steward.enable_interventions else 'Full Governance'}")
            print(f"   Session Started: {getattr(self.current_steward, 'session_start_time', 'Unknown')}")

            # Get session metrics if available
            if hasattr(self.current_steward, 'get_session_summary'):
                summary = self.current_steward.get_session_summary()
                print("\n📊 Session Metrics:")
                print(f"   Turns Processed: {summary.get('total_turns', 0)}")
                print(f"   Average Fidelity: {summary.get('avg_fidelity', 0):.3f}")
                print(f"   Drift Incidents: {summary.get('drift_count', 0)}")
        else:
            print("⚠️  Governance Session: INACTIVE")
            print("   Run 'python steward_pm.py dashboard' to view full status")

        print("\n💡 Available Commands:")
        print("   python steward_pm.py dashboard    - Launch TELOS HUD")
        print("   python steward_pm.py metrics      - Export metrics")
        print("   python steward_pm.py health       - System health check")
        print("\n" + "="*70 + "\n")

    def register_session(self, steward: UnifiedGovernanceSteward):
        """Register active governance session"""
        self.current_steward = steward
        self.session_active = True

    def end_session(self):
        """End current governance session"""
        self.session_active = False
        self.current_steward = None


class DashboardIntegration:
    """
    Integration with TELOS Development Dashboard.
    Provides metrics export and dashboard launching.
    """

    @staticmethod
    def launch_dashboard():
        """Launch the TELOS Development Dashboard"""
        print("🎯 Launching TELOS Development Dashboard...")
        print("\n💡 Dashboard would display:")
        print("   • Real-time fidelity metrics")
        print("   • Drift pattern visualization")
        print("   • Session history and trends")
        print("   • System health monitoring")
        print("   • Code metrics and TODO tracking")
        print("\n⚠️  Full Streamlit dashboard available in observatory/")
        print("   Run: streamlit run observatory/main.py")

    @staticmethod
    def export_dashboard_metrics() -> Dict[str, Any]:
        """Export current project metrics"""
        from pathlib import Path
        import json

        project_root = Path(__file__).parent.parent

        metrics = {
            'timestamp': None,
            'files': {
                'total': 0,
                'total_size_mb': 0
            },
            'code_analysis': {
                'code_lines': 0,
                'python_files': 0
            },
            'todos': {
                'total': 0
            },
            'git_stats': {
                'initialized': False,
                'commits': 0
            }
        }

        # Count Python files
        python_files = list(project_root.rglob('*.py'))
        metrics['code_analysis']['python_files'] = len(python_files)

        # Count total lines
        total_lines = 0
        for py_file in python_files:
            try:
                with open(py_file, 'r') as f:
                    total_lines += len(f.readlines())
            except:
                pass

        metrics['code_analysis']['code_lines'] = total_lines
        metrics['files']['total'] = len(python_files)

        # Check git status
        git_dir = project_root / '.git'
        if git_dir.exists():
            metrics['git_stats']['initialized'] = True
            try:
                import subprocess
                result = subprocess.run(
                    ['git', 'rev-list', '--count', 'HEAD'],
                    capture_output=True,
                    text=True,
                    cwd=project_root
                )
                if result.returncode == 0:
                    metrics['git_stats']['commits'] = int(result.stdout.strip())
            except:
                pass

        # Add timestamp
        from datetime import datetime
        metrics['timestamp'] = datetime.now().isoformat()

        return metrics
