#!/usr/bin/env python3
"""
Quick Conversation Export Helper
=================================

Helper script for /monitor export command.
Formats conversation data for analysis by claude_code_governance_monitor.py

Usage from /monitor export:
    1. Write conversation data to temp file in simple format
    2. Call this script to process and analyze
    3. Clean up temp file
"""

import sys
import os
from pathlib import Path
import asyncio

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from claude_code_governance_monitor import ClaudeCodeGovernanceMonitor


async def analyze_conversation_file(file_path: str):
    """
    Analyze conversation from file in simple format

    Format:
        USER: message
        ASSISTANT: response
        USER: message
        ASSISTANT: response
    """

    monitor = ClaudeCodeGovernanceMonitor()

    # Establish PA
    await monitor.establish_session_pa()

    print("\n" + "="*60)
    print("📝 Analyzing conversation from file")
    print("="*60)

    # Read and parse file
    with open(file_path, 'r') as f:
        content = f.read()

    turns = []
    lines = content.split('\n')
    current_user = None
    current_assistant = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith('USER: '):
            if current_user and current_assistant:
                turns.append((current_user, current_assistant))
            current_user = line[6:].strip()
            current_assistant = None
        elif line.startswith('ASSISTANT: '):
            current_assistant = line[11:].strip()

    # Don't forget the last turn
    if current_user and current_assistant:
        turns.append((current_user, current_assistant))

    if not turns:
        print("❌ No conversation turns found in file")
        print("\nExpected format:")
        print("USER: message")
        print("ASSISTANT: response")
        return

    print(f"\n📊 Found {len(turns)} conversation turns")
    print("Analyzing with ACTUAL TELOS...\n")

    # Analyze each turn
    for user_msg, assistant_msg in turns:
        await monitor.analyze_turn(user_msg, assistant_msg)
        await asyncio.sleep(0.2)  # Brief pause

    # Export session
    output_path = monitor.export_session()

    # Summary
    print("\n" + "="*60)
    print("📊 Session Summary")
    print("="*60)

    fidelities = monitor.session_manager.get_fidelity_history()
    if fidelities:
        print(f"   Total Turns: {len(fidelities)}")
        print(f"   Mean Fidelity: {sum(fidelities) / len(fidelities):.3f}")
        print(f"   Min Fidelity: {min(fidelities):.3f}")
        print(f"   Max Fidelity: {max(fidelities):.3f}")

        drift_turns = [i+1 for i, f in enumerate(fidelities) if f < 0.7]
        if drift_turns:
            print(f"\n   🚨 Drift detected on turns: {drift_turns}")
        else:
            print(f"\n   ✅ No significant drift detected")

        print(f"\n📊 Grant Application Value:")
        print(f"   ✅ Real fidelity measurements from actual development")
        print(f"   ✅ Demonstrates TELOS governing conversation building TELOS")
        print(f"   ✅ Validation data for institutional deployment claims")

        print(f"\nTo view in dashboard:")
        print(f"   1. ./launch_dashboard.sh")
        print(f"   2. Load session file: {output_path.name}")
        print(f"   3. View turn-by-turn metrics in TELOSCOPE tab")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 export_conversation.py <conversation_file>")
        print("\nFile format:")
        print("USER: message")
        print("ASSISTANT: response")
        sys.exit(1)

    file_path = sys.argv[1]

    if not Path(file_path).exists():
        print(f"❌ File not found: {file_path}")
        sys.exit(1)

    asyncio.run(analyze_conversation_file(file_path))
