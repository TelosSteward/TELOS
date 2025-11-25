#!/usr/bin/env python3
"""
STEWARD UNIFIED - Complete TELOS Project Manager with MCP Orchestration

Combines functionality from steward_pm.py and steward_v2.py:

1. PROJECT MANAGEMENT (from steward_pm.py):
   - Full TELOS project oversight (grants, partnerships, validation)
   - Memory MCP integration for dynamic state tracking
   - Health monitoring and system diagnostics
   - Dashboard/governance system integration
   - Repository strategy and sanitization workflows

2. ACTIVE ORCHESTRATION (from steward_v2.py):
   - Automatic MCP orchestration (decides when to invoke MCPs)
   - Git Guardian with pre-commit security audits
   - Continuous daemon monitoring mode
   - Task synchronization (TodoWrite <-> TASKS.md)
   - Smart commit timing recommendations

Commands:
    # Project Management
    python steward_unified.py status           - Full project status with MCP insights
    python steward_unified.py next             - Priority recommendations
    python steward_unified.py risks            - Risk analysis
    python steward_unified.py grants           - Grant application status
    python steward_unified.py partnerships     - Partnership progress

    # System Monitoring & Governance
    python steward_unified.py health           - System health check
    python steward_unified.py monitor          - Continuous health monitoring
    python steward_unified.py dashboard        - Launch TELOS HUD
    python steward_unified.py governance       - Unified governance summary
    python steward_unified.py metrics          - Export project metrics

    # Development & Git
    python steward_unified.py repos            - Repository strategy
    python steward_unified.py sanitize         - Sanitization workflow
    python steward_unified.py git              - Git MCP info
    python steward_unified.py git-audit        - Security audit on staged files
    python steward_unified.py should-commit    - Check if it's time to commit

    # Active Monitoring
    python steward_unified.py daemon           - Start continuous monitoring
    python steward_unified.py orchestrate      - Show MCP recommendations
    python steward_unified.py sync-tasks       - Sync TodoWrite with TASKS.md
"""

import sys
import os
import re
import subprocess
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import json

# Import optional dependencies
try:
    from health_monitor import HealthMonitor
    HAS_HEALTH_MONITOR = True
except ImportError:
    HAS_HEALTH_MONITOR = False

try:
    from telos_governance import TelosGovernance, DashboardIntegration
    HAS_GOVERNANCE = True
except ImportError:
    HAS_GOVERNANCE = False


class MCPOrchestrator:
    """Decides when to automatically invoke MCPs"""

    MCP_RULES = {
        'git': {
            'triggers': [
                'staged_files_exist',
                'commits_since_last_push > 3',
                'uncommitted_changes_age > 2h'
            ],
            'actions': ['suggest_commit', 'run_security_audit', 'check_branch_strategy']
        },
        'postgresql': {
            'triggers': [
                'new_validation_data',
                'grant_application_metrics_needed'
            ],
            'actions': ['query_study_results', 'generate_metrics']
        },
        'memory': {
            'triggers': [
                'project_state_changed',
                'grant_deadline_approaching'
            ],
            'actions': ['update_project_state', 'track_progress']
        },
        'playwright': {
            'triggers': [
                'observatory_ui_changed',
                'grant_screenshots_needed'
            ],
            'actions': ['capture_dashboard', 'generate_visual_assets']
        }
    }

    def should_use_git_mcp(self) -> Tuple[bool, str]:
        """Check if Git MCP should be invoked"""
        try:
            # Check for staged files
            result = subprocess.run(
                ['git', 'diff', '--cached', '--name-only'],
                capture_output=True,
                text=True,
                check=True
            )

            if result.stdout.strip():
                return True, "Staged files exist - Git MCP should manage commit"

            # Check for uncommitted changes
            result = subprocess.run(
                ['git', 'status', '--porcelain'],
                capture_output=True,
                text=True,
                check=True
            )

            if result.stdout.strip():
                return True, "Uncommitted changes - Git MCP should suggest commit"

            return False, "No git activity requiring MCP"

        except subprocess.CalledProcessError:
            return False, "Git not available"

    def should_use_memory_mcp(self) -> Tuple[bool, str]:
        """Check if Memory MCP should be invoked for project state"""
        # Memory MCP should be used for tracking grants, partnerships, validation
        return True, "Memory MCP recommended for tracking project state"

    def should_use_postgres_mcp(self) -> Tuple[bool, str]:
        """Check if PostgreSQL MCP should be invoked"""
        # Check if validation data needs to be queried
        validation_dir = Path('tests/validation_data')
        if validation_dir.exists():
            recent_files = list(validation_dir.glob('**/*.json'))
            if recent_files:
                return True, "Validation data available - PostgreSQL MCP can query results"

        return False, "No database queries needed at this time"

    def get_recommendations(self) -> Dict[str, List[str]]:
        """Get all MCP recommendations"""
        recommendations = {}

        should_git, git_reason = self.should_use_git_mcp()
        if should_git:
            recommendations['git'] = [git_reason]

        should_memory, memory_reason = self.should_use_memory_mcp()
        if should_memory:
            recommendations['memory'] = [memory_reason]

        should_postgres, pg_reason = self.should_use_postgres_mcp()
        if should_postgres:
            recommendations['postgresql'] = [pg_reason]

        return recommendations


class GitGuardian:
    """Monitors git state and runs security audits"""

    PROPRIETARY_TERMS = [
        r'dual.*pa\b',
        r'dual.*primacy.*attractor',
        r'DMAIC',
        r'SPC',
        r'statistical.*process.*control',
        r'originmind',
        r'telemetric.*key',
        r'progressive.*pa.*extractor',
        r'lock-on.*derivation',
        r'85\.32',  # The specific improvement metric
        r'\+85\.32%'
    ]

    def __init__(self):
        self.repo_root = Path.cwd()

    def audit_staged_files(self) -> Tuple[bool, List[str]]:
        """Run security audit on staged files"""

        # Whitelist: Files that are ALLOWED to contain proprietary terms
        WHITELIST = [
            'steward_v2.py',
            'steward_unified.py',  # This file contains patterns as search definitions
            'steward.py',
            'steward_pm.py',
            'docs/STEWARD_PM_2.0.md',
            '.claude_project.md',
            'docs/REPOSITORY_SEPARATION_STRATEGY.md',
        ]

        try:
            result = subprocess.run(
                ['git', 'diff', '--cached', '--name-only'],
                capture_output=True,
                text=True,
                check=True
            )

            staged_files = result.stdout.strip().split('\n')
            if not staged_files or staged_files == ['']:
                return True, []

            violations = []

            for file in staged_files:
                if not file:
                    continue

                # Check whitelist
                if any(file.endswith(whitelisted) for whitelisted in WHITELIST):
                    continue

                file_path = self.repo_root / file
                if not file_path.exists():
                    continue

                # Skip binary files
                if file_path.suffix in ['.pyc', '.png', '.jpg', '.gif']:
                    continue

                try:
                    content = file_path.read_text()

                    # Check for proprietary terms
                    for pattern in self.PROPRIETARY_TERMS:
                        matches = re.finditer(pattern, content, re.IGNORECASE)
                        for match in matches:
                            violations.append(
                                f"🚨 {file}:{self._get_line_number(content, match.start())} "
                                f"- Found proprietary term: '{match.group()}'"
                            )
                except:
                    continue

            return len(violations) == 0, violations

        except subprocess.CalledProcessError:
            return True, []

    def _get_line_number(self, content: str, position: int) -> int:
        """Get line number for a position in text"""
        return content[:position].count('\n') + 1

    def should_commit_now(self) -> Tuple[bool, str]:
        """Decide if it's time to create a commit"""
        try:
            # Check for uncommitted changes
            result = subprocess.run(
                ['git', 'status', '--porcelain'],
                capture_output=True,
                text=True,
                check=True
            )

            if not result.stdout.strip():
                return False, "No uncommitted changes"

            # Check number of changed files
            changed_files = len(result.stdout.strip().split('\n'))

            if changed_files >= 5:
                return True, f"{changed_files} files changed - consider committing"

            # Check time since last commit
            result = subprocess.run(
                ['git', 'log', '-1', '--format=%ct'],
                capture_output=True,
                text=True,
                check=True
            )

            if result.stdout.strip():
                last_commit_time = int(result.stdout.strip())
                hours_since = (time.time() - last_commit_time) / 3600

                if hours_since > 2 and changed_files > 0:
                    return True, f"{hours_since:.1f}h since last commit - consider committing"

            return False, "No urgent need to commit"

        except subprocess.CalledProcessError:
            return False, "Git not available"

    def get_current_branch(self) -> str:
        """Get current git branch"""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except:
            return "unknown"

    def check_branch_strategy(self) -> List[str]:
        """Check if user is on correct branch for current work"""
        current_branch = self.get_current_branch()

        warnings = []

        # Check if working on grant applications but not on grant branch
        if current_branch not in ['ltff-application', 'ev-application', 'eu-application', 'main']:
            warnings.append(
                f"⚠️  Currently on '{current_branch}' - "
                "Consider switching to grant application branch if working on grants"
            )

        return warnings


class TaskSynchronizer:
    """Syncs between Claude Code TodoWrite and TASKS.md"""

    def __init__(self):
        self.tasks_file = Path("docs/prd/TASKS.md")

    def sync_to_tasks_md(self, todos: List[Dict]) -> None:
        """Sync TodoWrite todos to TASKS.md"""
        # This would parse TASKS.md and update based on todos
        pass

    def extract_from_tasks_md(self) -> List[Dict]:
        """Extract todos from TASKS.md"""
        if not self.tasks_file.exists():
            return []
        # Parse TASKS.md and extract pending items
        return []


class StewardUnified:
    """Unified TELOS Project Manager - Complete orchestration and oversight"""

    def __init__(self):
        self.orchestrator = MCPOrchestrator()
        self.git_guardian = GitGuardian()
        self.task_sync = TaskSynchronizer()
        self.project_context = self._load_project_context()
        self.repo_type = self._detect_repo_type()

    def _detect_repo_type(self) -> str:
        """Detect if we're in public (purpose) or private repo"""
        cwd = Path.cwd()

        if 'telos' in str(cwd).lower():
            return 'public'
        elif 'telos_privacy' in str(cwd).lower() or 'telos_observatory' in str(cwd).lower():
            return 'private'
        else:
            return 'unknown'

    def _load_project_context(self) -> Dict:
        """Load .claude_project.md and other context"""
        context = {}

        claude_project = Path(".claude_project.md")
        if claude_project.exists():
            context['project'] = claude_project.read_text()

        steward_md = Path("STEWARD.md")
        if steward_md.exists():
            context['steward'] = steward_md.read_text()

        return context

    # ============================================================================
    # PROJECT MANAGEMENT (from steward_pm.py)
    # ============================================================================

    def display_steward_prompt(self, show_health=True):
        """Signal to Claude to act as Steward PM using Memory MCP"""
        print("\n" + "="*70)
        print("🤖 STEWARD UNIFIED - TELOS Project Manager")
        print("="*70 + "\n")

        # Show health summary if available
        if show_health and HAS_HEALTH_MONITOR:
            try:
                monitor = HealthMonitor()
                summary = monitor.get_health_summary()
                print(f"{summary['status_emoji']} System Health: {summary['status'].upper()} ({summary['health_score']}%)")

                if summary['alerts']:
                    print(f"⚠️  {len(summary['alerts'])} active alert(s)")

                print("")
            except Exception as e:
                print(f"⚠️  Health check failed: {e}\n")

        print("⚠️  CLAUDE: You are now acting as Steward PM")
        print("")
        print("📋 Your Role:")
        print("  • Project Manager for ENTIRE TELOS project")
        print("  • Use Memory MCP for dynamic state tracking")
        print("  • Orchestrate Git, PostgreSQL, Playwright MCPs as needed")
        print("  • Provide strategic recommendations and priority guidance")
        print("  • Track grants, partnerships, validation, and technical work")
        print("")
        print("🔍 Query Memory MCP:")
        print("  1. mcp__memory__search_nodes() - Find relevant entities")
        print("  2. mcp__memory__open_nodes() - Get detailed status")
        print("  3. Analyze dependencies and blockers")
        print("  4. Generate prioritized recommendations")
        print("")
        print("🎯 Focus Areas:")
        print("  • Grant Applications (LTFF, EV, EU)")
        print("  • Institutional Partnerships (GMU is CRITICAL)")
        print("  • Validation Studies (need 60+ for LTFF)")
        print("  • Technical Components (Observatory, Dual PA, etc.)")
        print("  • Strategic Timeline (February 2026 deadline)")
        print("")
        print("📊 Provide:")
        print("  • Current status summary")
        print("  • Critical path analysis")
        print("  • Top 3 priority recommendations with rationale")
        print("  • Blockers and risks")
        print("  • Next actions for each priority")
        print("")
        print("💾 After Analysis:")
        print("  • Update Memory MCP with any decisions or progress")
        print("  • Use mcp__memory__add_observations() to track changes")
        print("")

        # Show MCP orchestration recommendations
        mcp_recs = self.orchestrator.get_recommendations()
        if mcp_recs:
            print("🤖 Active MCP Recommendations:")
            for mcp, reasons in mcp_recs.items():
                print(f"  • {mcp.upper()}: {reasons[0]}")
            print("")

        print("="*70 + "\n")

    def status(self):
        """Enhanced status with MCP insights"""
        print("\n" + "="*70)
        print("🔭 TELOS PROJECT STATUS (Enhanced)")
        print("="*70 + "\n")

        # Current branch
        branch = self.git_guardian.get_current_branch()
        print(f"📍 Current Branch: {branch}")
        print(f"📁 Repository Type: {self.repo_type}")

        # Git state
        should_commit, reason = self.git_guardian.should_commit_now()
        print(f"📝 Git State: {reason}")

        # MCP recommendations
        recs = self.orchestrator.get_recommendations()
        if recs:
            print(f"\n🤖 Active MCP Recommendations:")
            for mcp, reasons in recs.items():
                print(f"  • {mcp.upper()}: {reasons[0]}")
        else:
            print("\n🤖 Active MCP Recommendations: None")

        # System health
        if HAS_HEALTH_MONITOR:
            try:
                monitor = HealthMonitor()
                summary = monitor.get_health_summary()
                print(f"\n{summary['status_emoji']} System Health: {summary['status'].upper()} ({summary['health_score']}%)")
            except:
                pass

        print("\n" + "="*70 + "\n")

    def display_repo_strategy(self):
        """Display repository classification strategy"""
        print("\n" + "="*70)
        print("📦 REPOSITORY STRATEGY")
        print("="*70 + "\n")

        print("🌐 PUBLIC: TELOS")
        print("   Purpose: Research community, developer adoption")
        print("   Contains:")
        print("   • Single PA mathematics (9/30/2024 baseline)")
        print("   • Runtime Governance (generic)")
        print("   • Basic fidelity measurement")
        print("   • Sanitized documentation")
        print("")
        print("   EXCLUDES (proprietary):")
        print("   ❌ Dual PA / AI PA / Lock-on derivation")
        print("   ❌ SPC/DMAIC frameworks")
        print("   ❌ Telemetric Keys")
        print("   ❌ OriginMind")
        print("   ❌ Progressive PA Extractor")
        print("   ❌ +85.32% metrics")
        print("   ❌ Observatory platform")
        print("")
        print("🔒 PRIVATE: telos_observatory")
        print("   Purpose: Institutional deployment, proprietary IP")
        print("   Contains:")
        print("   • Complete Observatory V3")
        print("   • Full Dual PA implementation")
        print("   • SPC/DMAIC integration")
        print("   • All proprietary innovations")
        print("   • Deployment configs")
        print("   • Grant materials")
        print("")
        print("   Deploys to: Streamlit Cloud (app public, code private)")
        print("")
        print("="*70 + "\n")

    def sanitization_workflow(self):
        """Display sanitization workflow with Git MCP"""
        print("\n" + "="*70)
        print("🔒 SANITIZATION WORKFLOW (Pre-Commit)")
        print("="*70 + "\n")

        print("BEFORE committing to PUBLIC repo:")
        print("")
        print("1. Run sanitization check:")
        print("   python3 steward_unified.py git-audit")
        print("")
        print("2. If BLOCKED:")
        print("   • Review HIGH-severity findings")
        print("   • Remove or sanitize proprietary terms")
        print("   • Re-run check until clean")
        print("")
        print("3. Use Git MCP for operations:")
        print("   ⚠️  CLAUDE: Use mcp__git__* tools")
        print("")
        print("   Example workflow:")
        print("   • mcp__git__status() - Check repo state")
        print("   • mcp__git__diff_unstaged() - Review changes")
        print("   • [Run sanitization check]")
        print("   • mcp__git__add() - Stage clean files")
        print("   • mcp__git__commit() - Commit with message")
        print("")
        print("4. Protected terms (NEVER commit to public):")
        print("   • 'Dual Attractor' / 'Dual PA' / 'AI PA'")
        print("   • 'Lock-on derivation'")
        print("   • 'DMAIC for AI' / 'SPC for AI'")
        print("   • 'Telemetric Keys' / 'OriginMind'")
        print("   • '+85.32%' / '+85%' (specific metrics)")
        print("")
        print("="*70 + "\n")

    def run_health_check(self, interactive=True):
        """Run comprehensive health check"""
        if not HAS_HEALTH_MONITOR:
            print("❌ Health monitor not available.")
            print("   Install required dependencies: pip install psutil requests")
            return 1

        monitor = HealthMonitor()
        monitor.display_health_dashboard()

        if interactive and sys.stdin.isatty():
            try:
                export = input("\n💾 Export health report? (y/n): ")
                if export.lower() == 'y':
                    filepath = monitor.export_health_report()
                    print(f"✅ Report exported to: {filepath}")
            except (EOFError, KeyboardInterrupt):
                print("\n✅ Health check complete.")

        return 0

    def launch_dashboard(self):
        """Launch the TELOS Development Dashboard (HUD)"""
        if not HAS_GOVERNANCE:
            print("❌ Dashboard integration not available.")
            print("   Make sure dev_dashboard directory exists")
            return 1

        print("\n" + "="*70)
        print("🎯 TELOS GOVERNANCE HUD - Development Dashboard")
        print("="*70 + "\n")

        print("This dashboard provides:")
        print("  • Real-time project analysis")
        print("  • System health monitoring")
        print("  • Code metrics and TODO tracking")
        print("  • Git statistics and insights")
        print("  • Claude AI integration interface")
        print("  • Unified governance view of TELOS")
        print("")

        DashboardIntegration.launch_dashboard()
        return 0

    def show_governance(self):
        """Show TELOS governance summary with all metrics"""
        if not HAS_GOVERNANCE:
            print("❌ Governance system not available.")
            return 1

        governance = TelosGovernance()
        governance.display_governance_summary()

        if HAS_HEALTH_MONITOR:
            print("💡 Run 'python3 steward_unified.py health' for detailed health report")

        print("💡 Run 'python3 steward_unified.py dashboard' to launch the full HUD")
        return 0

    def export_metrics(self):
        """Export current project metrics from dashboard analysis"""
        if not HAS_GOVERNANCE:
            print("❌ Governance system not available.")
            return 1

        print("\n🔍 Analyzing project and exporting metrics...")
        metrics = DashboardIntegration.export_dashboard_metrics()

        if metrics:
            print("\n📊 EXPORTED METRICS:")
            if 'files' in metrics:
                print(f"   Files: {metrics['files']['total']}")
                print(f"   Size: {metrics['files']['total_size_mb']} MB")
            if 'code_analysis' in metrics:
                print(f"   Code Lines: {metrics['code_analysis']['code_lines']:,}")
            if 'todos' in metrics:
                print(f"   TODOs: {metrics['todos']['total']}")
            if 'git_stats' in metrics and metrics['git_stats']['initialized']:
                print(f"   Git Commits: {metrics['git_stats']['commits']}")

            print("\n✅ Metrics saved to governance system")
            print("   Access via: python3 steward_unified.py governance")

        return 0

    # ============================================================================
    # ACTIVE ORCHESTRATION (from steward_v2.py)
    # ============================================================================

    def daemon_mode(self, interval: int = 300):
        """Run continuous monitoring at intervals (default 5 min)"""
        print(f"\n🤖 Steward Unified - Daemon Mode")
        print(f"   Monitoring every {interval}s (Ctrl+C to stop)\n")

        try:
            while True:
                self._monitoring_cycle()
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\n\n👋 Daemon stopped\n")

    def _monitoring_cycle(self):
        """Single monitoring cycle"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"\n{'='*70}")
        print(f"🔍 Monitoring Cycle: {timestamp}")
        print(f"{'='*70}\n")

        # Check MCP orchestration
        mcp_recs = self.orchestrator.get_recommendations()
        if mcp_recs:
            print("📋 MCP Recommendations:")
            for mcp, reasons in mcp_recs.items():
                print(f"\n  {mcp.upper()} MCP:")
                for reason in reasons:
                    print(f"    • {reason}")

        # Check git state
        should_commit, reason = self.git_guardian.should_commit_now()
        if should_commit:
            print(f"\n⏰ Commit Suggestion:")
            print(f"    • {reason}")

        # Check branch strategy
        branch_warnings = self.git_guardian.check_branch_strategy()
        if branch_warnings:
            print(f"\n⚠️  Branch Warnings:")
            for warning in branch_warnings:
                print(f"    • {warning}")

        # Check system health
        if HAS_HEALTH_MONITOR:
            try:
                monitor = HealthMonitor()
                summary = monitor.get_health_summary()
                print(f"\n{summary['status_emoji']} System Health: {summary['status'].upper()} ({summary['health_score']}%)")
            except:
                pass

        print(f"\n{'='*70}\n")

    def git_audit(self):
        """Run security audit on staged files"""
        # Only audit public repo - private repo is SUPPOSED to have proprietary terms
        if self.repo_type == 'private':
            print("\n⏭️  Skipping security audit (private repo - proprietary terms expected)\n")
            return True

        if self.repo_type == 'unknown':
            print("\n⚠️  Warning: Cannot determine repo type - skipping audit\n")
            return True

        print("\n🔍 Running security audit on staged files (PUBLIC REPO)...\n")

        clean, violations = self.git_guardian.audit_staged_files()

        if clean:
            print("✅ Security audit PASSED - No proprietary terms detected\n")
            return True
        else:
            print("🚨 Security audit FAILED - Proprietary terms detected:\n")
            for violation in violations:
                print(f"  {violation}")
            print("\n❌ DO NOT COMMIT - Remove proprietary terms first\n")
            return False

    def should_commit(self):
        """Check if it's time to commit"""
        should, reason = self.git_guardian.should_commit_now()

        print(f"\n{'='*70}")
        if should:
            print(f"✅ Commit recommended: {reason}")
        else:
            print(f"⏸️  No urgent need to commit: {reason}")
        print(f"{'='*70}\n")

    def orchestrate(self):
        """Show MCP orchestration recommendations"""
        print(f"\n{'='*70}")
        print("🎯 MCP Orchestration Recommendations")
        print(f"{'='*70}\n")

        recs = self.orchestrator.get_recommendations()

        if not recs:
            print("✅ No MCPs need to be invoked at this time\n")
            return

        for mcp, reasons in recs.items():
            print(f"{mcp.upper()} MCP:")
            for reason in reasons:
                print(f"  • {reason}")
            print()

    def sync_tasks(self):
        """Sync TodoWrite with TASKS.md"""
        print("\n🔄 Task Synchronization")
        print("   (Feature in development - will sync TodoWrite <-> TASKS.md)")
        print("")

    def run_health_monitor_continuous(self):
        """Run continuous health monitoring with auto-refresh"""
        if not HAS_HEALTH_MONITOR:
            print("❌ Health monitor not available.")
            return 1

        monitor = HealthMonitor()

        print("🔄 Starting continuous health monitoring (Ctrl+C to stop)...")
        print("   Refresh interval: 30 seconds\n")

        try:
            while True:
                os.system('clear' if os.name == 'posix' else 'cls')
                monitor.display_health_dashboard()
                print("\n⏰ Next refresh in 30 seconds... (Ctrl+C to stop)")
                time.sleep(30)
        except KeyboardInterrupt:
            print("\n\n✅ Health monitoring stopped.")
            return 0


def main():
    """Main entry point"""
    steward = StewardUnified()

    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        print(f"\n🤖 STEWARD UNIFIED MODE: {mode}")
        print("")

        # Project Management modes
        if mode == "status":
            steward.status()
            steward.display_steward_prompt(show_health=False)
        elif mode == "next":
            print("Requesting: Top 3 priority recommendations")
            steward.display_steward_prompt()
        elif mode == "risks":
            print("Requesting: Risk analysis and blockers")
            steward.display_steward_prompt()
        elif mode == "grants":
            print("Requesting: Grant application status and next actions")
            steward.display_steward_prompt()
        elif mode == "partnerships":
            print("Requesting: Institutional partnership progress")
            steward.display_steward_prompt()

        # System Monitoring & Governance
        elif mode == "health":
            return steward.run_health_check()
        elif mode == "monitor":
            return steward.run_health_monitor_continuous()
        elif mode == "dashboard":
            return steward.launch_dashboard()
        elif mode == "governance":
            return steward.show_governance()
        elif mode == "metrics":
            return steward.export_metrics()

        # Development & Git
        elif mode == "repos":
            steward.display_repo_strategy()
            return 0
        elif mode == "sanitize":
            steward.sanitization_workflow()
            return 0
        elif mode == "git":
            print("✅ Git MCP available (accessed via Claude Code)")
            print("\n⚠️  CLAUDE: Use Git MCP tools for repository operations")
            print("   Available: mcp__git__status, mcp__git__add, mcp__git__commit, etc.")
            return 0
        elif mode == "git-audit":
            passed = steward.git_audit()
            return 0 if passed else 1
        elif mode == "should-commit":
            steward.should_commit()

        # Active Monitoring
        elif mode == "daemon":
            steward.daemon_mode()
        elif mode == "orchestrate":
            steward.orchestrate()
        elif mode == "sync-tasks":
            steward.sync_tasks()

        else:
            print(f"Unknown mode: {mode}")
            print("\nValid modes:")
            print("  Project: status, next, risks, grants, partnerships")
            print("  System: health, monitor, dashboard, governance, metrics")
            print("  Dev/Git: repos, sanitize, git, git-audit, should-commit")
            print("  Active: daemon, orchestrate, sync-tasks")

        return 0

    # Show help if no arguments
    print("\n📋 STEWARD UNIFIED - Available Commands:\n")
    print("🎯 TELOS Governance:")
    print("  python steward_unified.py dashboard    - Launch TELOS HUD")
    print("  python steward_unified.py governance   - Unified governance summary")
    print("  python steward_unified.py metrics      - Export project metrics")
    print("")
    print("📊 Project Management:")
    print("  python steward_unified.py status       - Full project status")
    print("  python steward_unified.py next         - Priority recommendations")
    print("  python steward_unified.py risks        - Risk analysis")
    print("  python steward_unified.py grants       - Grant application status")
    print("  python steward_unified.py partnerships - Partnership progress")
    print("")
    print("🛠️ Development & Git:")
    print("  python steward_unified.py repos        - Repository strategy")
    print("  python steward_unified.py sanitize     - Sanitization workflow")
    print("  python steward_unified.py git          - Git MCP info")
    print("  python steward_unified.py git-audit    - Security audit on staged files")
    print("  python steward_unified.py should-commit - Check if it's time to commit")
    print("")
    print("🏥 System Monitoring:")
    print("  python steward_unified.py health       - System health check")
    print("  python steward_unified.py monitor      - Continuous monitoring")
    print("")
    print("🤖 Active Orchestration:")
    print("  python steward_unified.py daemon       - Start continuous monitoring")
    print("  python steward_unified.py orchestrate  - Show MCP recommendations")
    print("  python steward_unified.py sync-tasks   - Sync TodoWrite with TASKS.md")
    print("")
    print("💡 TIP: Use 'dashboard' for the complete TELOS governance HUD!")
    print("")

    steward.display_steward_prompt()
    return 0


if __name__ == "__main__":
    sys.exit(main())
