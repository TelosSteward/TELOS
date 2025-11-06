#!/usr/bin/env python3
"""
TELOS Session Initialization - Runtime Governance
=================================================

Creates ActiveSession entity in Memory MCP with PA baseline from .claude_project.md

Usage:
    python3 telos_session_start.py

Creates Memory MCP entities:
    - ActiveSession_[date] with PA baseline and metadata
    - Initializes turn counter at 0

Called automatically by Telos_recall.sh at session start.
"""

import sys
import json
from pathlib import Path
from datetime import datetime


def extract_pa_from_claude_project():
    """Extract Session Purpose from .claude_project.md"""
    project_file = Path('.claude_project.md')

    if not project_file.exists():
        raise FileNotFoundError('.claude_project.md not found')

    content = project_file.read_text()

    # Extract Session Purpose
    if '### Session Purpose (User PA)' not in content:
        raise ValueError('Session Purpose not found in .claude_project.md')

    lines = content.split('\n')

    # Find Session Purpose section
    purpose_start = None
    purpose_end = None
    for i, line in enumerate(lines):
        if '### Session Purpose (User PA)' in line:
            purpose_start = i + 2  # Skip header and blank line
        elif purpose_start and line.startswith('###'):
            purpose_end = i
            break

    if purpose_start is None:
        raise ValueError('Could not parse Session Purpose')

    if purpose_end is None:
        purpose_end = len(lines)

    purpose_text = '\n'.join(lines[purpose_start:purpose_end]).strip()
    return purpose_text


def main():
    """Initialize session in Memory MCP"""

    print("\n🔭 TELOS Runtime Governance - Session Initialization")
    print("="*60)

    # Extract PA baseline
    print("\n📋 Extracting PA baseline from .claude_project.md...")
    try:
        pa_text = extract_pa_from_claude_project()
        print(f"✅ PA extracted ({len(pa_text)} chars)")
        print(f"   Purpose: {pa_text[:80]}...")
    except Exception as e:
        print(f"❌ Error extracting PA: {e}")
        sys.exit(1)

    # Generate session ID
    session_date = datetime.now().strftime('%Y-%m-%d')
    session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    session_name = f"ActiveSession_{session_date}"

    print(f"\n🆔 Session ID: {session_id}")
    print(f"   Entity Name: {session_name}")

    # Signal to Claude to create Memory MCP entity
    print("\n⚠️  CLAUDE: Create Memory MCP entity now")
    print(f"\nUse: mcp__memory__create_entities([{{")
    print(f'  "name": "{session_name}",')
    print(f'  "entityType": "Session",')
    print(f'  "observations": [')
    print(f'    "session_id: {session_id}",')
    print(f'    "started_at: {datetime.now().isoformat()}",')
    print(f'    "pa_baseline: {pa_text[:100]}...",')
    print(f'    "turn_count: 0",')
    print(f'    "status: active"')
    print(f'  ]')
    print(f'}}])')

    # Save session info locally for scripts
    session_info = {
        'session_name': session_name,
        'session_id': session_id,
        'started_at': datetime.now().isoformat(),
        'pa_baseline': pa_text,
        'turn_count': 0
    }

    with open('.telos_active_session.json', 'w') as f:
        json.dump(session_info, f, indent=2)

    print(f"\n💾 Session info saved to .telos_active_session.json")
    print("\n" + "="*60)
    print("✅ TELOS Runtime Governance INITIALIZED")
    print("="*60)
    print(f"\n📊 Turn-by-turn tracking active")
    print(f"   Every turn will be measured and logged")
    print(f"   Fidelity reports after each response")
    print(f"   Session data stored in Memory MCP")
    print()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
