#!/usr/bin/env python3
"""
TELOS Memory Query Script
Queries Memory MCP for dynamic project state across sessions.
"""

import json
import os
import sys

def format_entity_display(entity):
    """Format entity for display."""
    name = entity.get('name', 'Unknown')
    entity_type = entity.get('entityType', 'Unknown')
    observations = entity.get('observations', [])

    return {
        'name': name,
        'type': entity_type,
        'observations': observations
    }

def display_memory_state():
    """
    Display reminder for Claude to query Memory MCP.
    This script signals to Claude to query the memory graph and report status.
    """

    print("📚 MEMORY MCP STATE")
    print("")

    # Check for active runtime governance session
    from pathlib import Path
    session_file = Path('.telos_active_session.json')
    if session_file.exists():
        with open(session_file, 'r') as f:
            session_info = json.load(f)
        print("🔭 RUNTIME GOVERNANCE ACTIVE")
        print(f"   Session: {session_info.get('session_name', 'Unknown')}")
        print(f"   Turn count: {session_info.get('turn_count', 0)}")
        if 'last_fidelity' in session_info:
            fidelity = session_info['last_fidelity']
            icon = "✅" if fidelity >= 0.8 else "⚠️" if fidelity >= 0.7 else "🚨"
            print(f"   Last fidelity: {fidelity:.3f} {icon}")
        print("")

    print("⚠️  CLAUDE: Query Memory MCP now using mcp__memory__read_graph()")
    print("")
    print("Expected entities in memory:")
    print("  • Grant Applications (LTFF, EV, EU)")
    print("  • Institutional Partners (GMU, Oxford, Berkeley)")
    print("  • Technical Components (Dual PA, Observatory, DMAIC/SPC)")
    print("  • Validation Data (45+ studies)")
    print("  • Milestones (Trail of Bits audit, Black Belt)")
    if session_file.exists():
        print(f"  • Active Session ({session_info.get('session_name', 'Unknown')})")
        print(f"  • Turn entities (Turn_1 through Turn_{session_info.get('turn_count', 0)})")
    print("")
    print("Display summary of:")
    print("  🎯 Grant status and next actions")
    print("  🤝 Partnership progress and contacts")
    print("  🔬 Validation study completion")
    print("  ⚙️  Technical component status")
    if session_file.exists():
        print("  🔭 Active session statistics (turns, mean fidelity)")
    print("  📅 Recent session work (if tracked)")
    print("")

def main():
    """Main entry point."""
    try:
        display_memory_state()
        return 0
    except Exception as e:
        print(f"❌ Error querying memory: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())
