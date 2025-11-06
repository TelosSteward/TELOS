#!/usr/bin/env python3
"""
STEWARD PM - TELOS Project Manager (Memory + Git MCP Integration)

Unlike steward.py (Observatory V1.00 PM), this manages the FULL TELOS project:
- Grant applications (LTFF, EV, EU)
- Institutional partnerships (GMU, Oxford, Berkeley)
- Validation studies and technical components
- Strategic positioning and timeline management
- Repository management (public vs private, sanitization)

Uses Memory MCP for dynamic state tracking and Git MCP for repo operations.
"""

import sys
import subprocess
from pathlib import Path

def display_steward_pm_prompt():
    """
    Signal to Claude to act as Steward PM using Memory MCP.
    """

    print("\n" + "="*70)
    print("🤖 STEWARD PM - TELOS Project Manager")
    print("="*70 + "\n")

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

    print("🌐 PUBLIC: telos_purpose")
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
        else:
            print(f"Unknown mode: {mode}")
            print("Valid modes: status, next, risks, grants, partnerships, repos, sanitize, git")

        print("")

    display_steward_pm_prompt()
    return 0


if __name__ == "__main__":
    sys.exit(main())
