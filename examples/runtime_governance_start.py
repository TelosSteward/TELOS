#!/usr/bin/env python3
"""
Runtime Governance - Session Initialization
Creates ActiveSession entity in Memory MCP with PA baseline from .claude_project.md
"""

import sys
import json
from pathlib import Path
from datetime import datetime

def extract_pa_from_claude_project():
    """Extract PA baseline from .claude_project.md"""
    project_file = Path('.claude_project.md')
    if not project_file.exists():
        raise FileNotFoundError('.claude_project.md not found')

    content = project_file.read_text()

    # Look for PA baseline section
    markers = [
        '### Session Purpose (User PA)',
        '**PA Baseline:**',
        '## PA Baseline',
        '### PA Baseline'
    ]

    pa_start = None
    for marker in markers:
        if marker in content:
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if marker in line:
                    pa_start = i + 1
                    # Skip empty lines
                    while pa_start < len(lines) and not lines[pa_start].strip():
                        pa_start += 1
                    break
            if pa_start:
                break

    if pa_start is None:
        raise ValueError('PA baseline not found in .claude_project.md. Add a section with "PA Baseline:" or "Session Purpose"')

    # Extract PA text until next section or end
    pa_lines = []
    for i in range(pa_start, len(lines)):
        line = lines[i]
        # Stop at next markdown section
        if line.startswith('#') or line.startswith('**Fidelity Thresholds:**') or line.startswith('---'):
            break
        pa_lines.append(line)

    pa_text = '\n'.join(pa_lines).strip()

    # Clean up if it's quoted
    if pa_text.startswith('"') and pa_text.endswith('"'):
        pa_text = pa_text[1:-1]

    return pa_text

def main():
    """Initialize session in Memory MCP"""
    print("\n🔭 Runtime Governance - Session Initialization")
    print("="*60)

    # Extract PA baseline
    print("\n📋 Extracting PA baseline from .claude_project.md...")
    try:
        pa_text = extract_pa_from_claude_project()
        print(f"✅ PA extracted ({len(pa_text)} chars)")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nMake sure your .claude_project.md contains a PA baseline section.")
        print("See documentation for template.")
        return 1

    # Generate session ID
    session_date = datetime.now().strftime('%Y-%m-%d')
    session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    session_name = f"ActiveSession_{session_date}"

    # Signal to Claude to create Memory MCP entity
    print(f"\n⚠️  CLAUDE: Create Memory MCP entity now")
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

    # Save session info locally
    session_info = {
        'session_name': session_name,
        'session_id': session_id,
        'started_at': datetime.now().isoformat(),
        'pa_baseline': pa_text,
        'turn_count': 0
    }

    with open('.runtime_governance_session.json', 'w') as f:
        json.dump(session_info, f, indent=2)

    print(f"\n💾 Session info saved to .runtime_governance_session.json")
    print("\n✅ Runtime Governance INITIALIZED")
    print("\n📊 Checkpoints will run automatically after each Claude response.")

if __name__ == "__main__":
    sys.exit(main())
