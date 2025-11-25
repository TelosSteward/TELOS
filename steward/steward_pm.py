#!/usr/bin/env python3
"""
STEWARD PM - TELOS Project Manager (Memory + Git MCP Integration)

Unlike steward.py (Observatory V1.00 PM), this manages the FULL TELOS project:
- Grant applications (LTFF, EV, EU)
- Institutional partnerships (GMU, Oxford, Berkeley)
- Validation studies and technical components
- Strategic positioning and timeline management
- Repository management (public vs private, sanitization)
- Health monitoring and system diagnostics

Uses Memory MCP for dynamic state tracking and Git MCP for repo operations.
Enhanced with comprehensive health monitoring capabilities.
"""

import sys
import subprocess
from pathlib import Path

# Import health monitor if available
try:
    from health_monitor import HealthMonitor
    HAS_HEALTH_MONITOR = True
except ImportError:
    HAS_HEALTH_MONITOR = False
    print("⚠️  Health monitor not available. Install psutil: pip install psutil")

# Import governance system if available
try:
    from telos_governance import TelosGovernance, DashboardIntegration
    HAS_GOVERNANCE = True
except ImportError:
    HAS_GOVERNANCE = False

def display_steward_pm_prompt(show_health=True):
    """
    Signal to Claude to act as Steward PM using Memory MCP.
    Optionally shows health status on startup.
    """

    print("\n" + "="*70)
    print("🤖 STEWARD PM - TELOS Project Manager")
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
    print("  • Project Manager for ENTIRE TELOS project (not just Observatory)")
    print("  • Use Memory MCP for dynamic state (not static files)")
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
    print("="*70 + "\n")


def check_git_mcp():
    """Check if Git MCP is available."""
    # Git MCP is accessed through Claude via mcp__git__* functions
    print("✅ Git MCP available (accessed via Claude Code)")
    return True

def display_repo_strategy():
    """Display repository classification strategy."""
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

def sanitization_workflow():
    """Display sanitization workflow with Git MCP."""
    print("\n" + "="*70)
    print("🔒 SANITIZATION WORKFLOW (Pre-Commit)")
    print("="*70 + "\n")

    print("BEFORE committing to PUBLIC repo:")
    print("")
    print("1. Run sanitization check:")
    print("   python3 steward_sanitization_check.py <path>")
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

def run_health_check(interactive=True):
    """Run comprehensive health check."""
    if not HAS_HEALTH_MONITOR:
        print("❌ Health monitor not available.")
        print("   Install required dependencies: pip install psutil requests")
        return 1

    monitor = HealthMonitor()
    monitor.display_health_dashboard()

    # Offer to export report only in interactive mode
    if interactive and sys.stdin.isatty():
        try:
            export = input("\n💾 Export health report? (y/n): ")
            if export.lower() == 'y':
                filepath = monitor.export_health_report()
                print(f"✅ Report exported to: {filepath}")
        except (EOFError, KeyboardInterrupt):
            print("\n✅ Health check complete.")

    return 0


def launch_dashboard():
    """Launch the TELOS Development Dashboard (HUD)."""
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


def show_governance():
    """Show TELOS governance summary with all metrics."""
    if not HAS_GOVERNANCE:
        print("❌ Governance system not available.")
        return 1

    governance = TelosGovernance()
    governance.display_governance_summary()

    # Also show health if available
    if HAS_HEALTH_MONITOR:
        print("💡 Run 'python3 steward_pm.py health' for detailed health report")

    print("💡 Run 'python3 steward_pm.py dashboard' to launch the full HUD")
    return 0


def export_metrics():
    """Export current project metrics from dashboard analysis."""
    if not HAS_GOVERNANCE:
        print("❌ Governance system not available.")
        return 1

    print("\n🔍 Analyzing project and exporting metrics...")
    metrics = DashboardIntegration.export_dashboard_metrics()

    if metrics:
        # Display summary
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
        print("   Access via: python3 steward_pm.py governance")

    return 0


def run_health_monitor_continuous():
    """Run continuous health monitoring with auto-refresh."""
    if not HAS_HEALTH_MONITOR:
        print("❌ Health monitor not available.")
        return 1

    import time
    monitor = HealthMonitor()

    print("🔄 Starting continuous health monitoring (Ctrl+C to stop)...")
    print("   Refresh interval: 30 seconds\n")

    try:
        while True:
            # Clear screen (platform-specific)
            import os
            os.system('clear' if os.name == 'posix' else 'cls')

            monitor.display_health_dashboard()
            print("\n⏰ Next refresh in 30 seconds... (Ctrl+C to stop)")

            time.sleep(30)

    except KeyboardInterrupt:
        print("\n\n✅ Health monitoring stopped.")
        return 0


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        print(f"\n🤖 STEWARD PM MODE: {mode}")
        print("")

        if mode == "status":
            print("Requesting: Full project status across all areas")
        elif mode == "next":
            print("Requesting: Top 3 priority recommendations")
        elif mode == "risks":
            print("Requesting: Risk analysis and blockers")
        elif mode == "grants":
            print("Requesting: Grant application status and next actions")
        elif mode == "partnerships":
            print("Requesting: Institutional partnership progress")
        elif mode == "repos":
            display_repo_strategy()
            return 0
        elif mode == "sanitize":
            sanitization_workflow()
            return 0
        elif mode == "git":
            check_git_mcp()
            print("\n⚠️  CLAUDE: Use Git MCP tools for repository operations")
            print("   Available: mcp__git__status, mcp__git__add, mcp__git__commit, etc.")
            return 0
        elif mode == "health":
            return run_health_check()
        elif mode == "monitor":
            return run_health_monitor_continuous()
        elif mode == "dashboard":
            return launch_dashboard()
        elif mode == "governance":
            return show_governance()
        elif mode == "metrics":
            return export_metrics()
        else:
            print(f"Unknown mode: {mode}")
            print("Valid modes: status, next, risks, grants, partnerships, repos, sanitize, git")
            print("           health, monitor, dashboard, governance, metrics")

        print("")

    # Show help if no arguments
    if len(sys.argv) == 1:
        print("\n📋 STEWARD PM - Available Commands:\n")
        print("🎯 TELOS Governance (NEW!):")
        print("  python steward_pm.py dashboard    - Launch TELOS HUD (Dev Dashboard)")
        print("  python steward_pm.py governance   - Unified governance summary")
        print("  python steward_pm.py metrics      - Export project metrics")
        print("")
        print("📊 Project Management:")
        print("  python steward_pm.py status       - Full project status")
        print("  python steward_pm.py next         - Priority recommendations")
        print("  python steward_pm.py risks        - Risk analysis")
        print("  python steward_pm.py grants       - Grant application status")
        print("  python steward_pm.py partnerships - Partnership progress")
        print("")
        print("🛠️ Development:")
        print("  python steward_pm.py repos        - Repository strategy")
        print("  python steward_pm.py sanitize     - Sanitization workflow")
        print("  python steward_pm.py git          - Git MCP info")
        print("")
        print("🏥 System Monitoring:")
        print("  python steward_pm.py health       - System health check")
        print("  python steward_pm.py monitor      - Continuous monitoring")
        print("")
        print("💡 TIP: Use 'dashboard' for the complete TELOS governance HUD!")
        print("")

    display_steward_pm_prompt()
    return 0


if __name__ == "__main__":
    sys.exit(main())
