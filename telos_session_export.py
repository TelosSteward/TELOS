#!/usr/bin/env python3
"""
TELOS Session Export - Runtime Governance
=========================================

Exports complete session data from Memory MCP for dashboard viewing and grant reports.

Usage:
    python3 telos_session_export.py
    python3 telos_session_export.py --session ActiveSession_2025-11-05
    python3 telos_session_export.py --format dashboard  # For TELOSCOPE
    python3 telos_session_export.py --format grant      # For grant reports

Outputs:
    - JSON file with complete session data
    - Dashboard-compatible format
    - Grant validation report
"""

import sys
import json
from pathlib import Path
from datetime import datetime


def load_session_info():
    """Load active session info"""
    session_file = Path('.telos_active_session.json')

    if not session_file.exists():
        raise FileNotFoundError(
            "No active session found. Run: python3 telos_session_start.py"
        )

    with open(session_file, 'r') as f:
        return json.load(f)


def export_session(session_name=None, format_type='standard'):
    """
    Export session data from Memory MCP

    Note: This script signals Claude to query Memory MCP and format the data.
    Claude will read all turns and generate the export.
    """

    print("\n🔭 TELOS Session Export")
    print("="*60)

    # Load session info
    try:
        session_info = load_session_info()
        if session_name is None:
            session_name = session_info['session_name']
    except FileNotFoundError:
        if session_name is None:
            print("❌ No active session. Specify --session SESSION_NAME")
            sys.exit(1)

    print(f"\n📋 Session: {session_name}")
    print(f"   Format: {format_type}")

    # Signal to Claude to query Memory MCP
    print("\n⚠️  CLAUDE: Export session from Memory MCP")
    print(f"\n1. Query session entity:")
    print(f"   mcp__memory__open_nodes(['{session_name}'])")
    print()
    print(f"2. Get all turns:")
    print(f"   mcp__memory__search_nodes(query='Turn session_id:{session_info.get(\"session_id\", \"unknown\")}')")
    print()
    print(f"3. Format and export based on format_type:")

    if format_type == 'dashboard':
        print("\n   Dashboard Format (TELOSCOPE compatible):")
        print("   {")
        print('     "session_metadata": {')
        print(f'       "session_id": "{session_info.get("session_id", "unknown")}",')
        print(f'       "started_at": "...",')
        print('       "total_turns": N,')
        print('       "mean_fidelity": X.XXX')
        print('     },')
        print('     "snapshots": [')
        print('       {')
        print('         "turn_number": 1,')
        print('         "user_input": "...",')
        print('         "telos_response": "...",')
        print('         "telic_fidelity": 0.XXX,')
        print('         "basin_membership": true/false')
        print('       },')
        print('       ...')
        print('     ]')
        print('   }')

    elif format_type == 'grant':
        print("\n   Grant Report Format:")
        print("   {")
        print('     "title": "TELOS Runtime Governance Session Report",')
        print('     "session_id": "...",')
        print('     "date_range": "...",')
        print('     "statistics": {')
        print('       "total_turns": N,')
        print('       "mean_fidelity": X.XXX,')
        print('       "min_fidelity": X.XXX,')
        print('       "max_fidelity": X.XXX,')
        print('       "drift_events": N,')
        print('       "pa_stable": true/false')
        print('     },')
        print('     "narrative": "..."')
        print('   }')

    else:
        print("\n   Standard Format:")
        print("   Complete session data with all turns and metadata")

    # Output file naming
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"session_export_{format_type}_{timestamp}.json"

    print(f"\n💾 Output file: {output_file}")
    print(f"\n✅ After export:")
    print(f"   - Load in TELOSCOPE dashboard")
    print(f"   - Include in grant applications")
    print(f"   - Analyze longitudinal trends")
    print()

    return {
        'session_name': session_name,
        'format': format_type,
        'output_file': output_file,
        'instruction': 'Claude should query Memory MCP and generate export'
    }


def main():
    """Main entry point"""

    session_name = None
    format_type = 'standard'

    # Parse arguments
    if '--session' in sys.argv:
        idx = sys.argv.index('--session')
        session_name = sys.argv[idx + 1]

    if '--format' in sys.argv:
        idx = sys.argv.index('--format')
        format_type = sys.argv[idx + 1]

        if format_type not in ['standard', 'dashboard', 'grant']:
            print(f"❌ Invalid format: {format_type}")
            print("Valid formats: standard, dashboard, grant")
            sys.exit(1)

    # Export session
    result = export_session(session_name, format_type)

    # Save instruction file for Claude
    instruction_file = '.telos_export_instruction.json'
    with open(instruction_file, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"💾 Export instruction saved: {instruction_file}")
    print("   Claude will read this and generate the export\n")


if __name__ == "__main__":
    main()
