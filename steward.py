#!/usr/bin/env python3
"""
STEWARD - TELOS Project Manager Agent

Simple PM assistant that tracks progress, suggests priorities, and updates status.

Commands:
    python steward.py status    - Show current state across all trackers
    python steward.py next      - Suggest what to work on next
    python steward.py complete "task name" - Mark task as complete
"""

import sys
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple


class Steward:
    """TELOS Project Manager Agent"""

    def __init__(self):
        self.prd_dir = Path("docs/prd")
        self.steward_file = Path("STEWARD.md")

        # Validate files exist
        required_files = [
            self.prd_dir / "PRD.md",
            self.prd_dir / "PLATFORM_STATUS.md",
            self.prd_dir / "UI_PHASES.md",
            self.prd_dir / "TASKS.md",
            self.steward_file
        ]

        for file in required_files:
            if not file.exists():
                print(f"❌ Error: Required file not found: {file}")
                sys.exit(1)

    def status(self):
        """Show current project status across all trackers"""
        print("\n" + "="*60)
        print("🔭 TELOS Project Status")
        print("="*60 + "\n")

        # Parse PRD deliverables
        prd_complete, prd_total = self._parse_checkboxes(self.prd_dir / "PRD.md")
        print(f"📄 V1.00 Deliverables:  {prd_complete}/{prd_total} complete ({self._percent(prd_complete, prd_total)}%)")

        # Parse TASKS
        tasks_line = self._find_line(self.prd_dir / "TASKS.md", r"\*\*Status\*\*:.*complete")
        if tasks_line:
            match = re.search(r'(\d+)/(\d+)', tasks_line)
            if match:
                done, total = match.groups()
                print(f"📋 Task Backlog:        {done}/{total} complete ({self._percent(int(done), int(total))}%)")

        # Parse UI Phases
        ui_line = self._find_line(self.prd_dir / "UI_PHASES.md", r"\*\*Overall Progress\*\*:")
        if ui_line:
            match = re.search(r'(\d+\.\d+)%.*\((\d+)/(\d+)', ui_line)
            if match:
                pct, done, total = match.groups()
                print(f"🎨 UI Overhaul:         {done}/{total} complete ({pct}%)")

        # Parse Platform Status
        platform_line = self._find_line(self.prd_dir / "PLATFORM_STATUS.md", r"\*\*Status\*\*:.*ready")
        if platform_line:
            print(f"⚙️  Platform:            ~85% complete (Infrastructure ready)")

        # Show current focus
        print("\n" + "-"*60)
        print("🎯 Current Focus:\n")
        focus_section = self._extract_section(self.steward_file, "## Current Focus")
        if focus_section:
            for line in focus_section.split('\n')[1:6]:  # First 5 lines
                if line.strip():
                    print(f"   {line.strip()}")

        # Show blockers
        print("\n" + "-"*60)
        print("🚧 Blockers:\n")
        blocker_section = self._extract_section(self.steward_file, "## Blockers")
        if blocker_section:
            for line in blocker_section.split('\n')[1:5]:  # First 4 lines
                if line.strip().startswith('-'):
                    print(f"   {line.strip()}")

        print("\n" + "="*60 + "\n")

    def next(self):
        """Suggest what to work on next based on dependencies"""
        print("\n" + "="*60)
        print("🎯 STEWARD Suggests:")
        print("="*60 + "\n")

        # Read critical path from PRD
        prd_content = (self.prd_dir / "PRD.md").read_text()

        # Check V1.00 deliverables
        deliverables = [
            ("Pilot Brief", "Not Started", "Write pilot methodology doc"),
            ("Test Conversations", "Not Started", "Run 3-5 pilot conversations"),
            ("Comparative Summary JSON", "Not Started", "Generate statistical results"),
            ("Grant Package", "Not Started", "Compile all evidence"),
            ("Testing Suite", "Partially Complete", "Expand test coverage")
        ]

        # Find first incomplete deliverable
        incomplete = []
        for name, status, action in deliverables:
            if "complete" not in status.lower() or status == "Partially Complete":
                incomplete.append((name, action))

        if incomplete:
            print("📋 Priority Tasks (V1.00 Critical Path):\n")
            for i, (name, action) in enumerate(incomplete[:3], 1):
                print(f"   {i}. {action}")
                print(f"      → Deliverable: {name}\n")

        # Show next steps from TASKS.md
        tasks_content = (self.prd_dir / "TASKS.md").read_text()
        if "## 🔥 SECTION 1: IMMEDIATE" in tasks_content:
            print("-"*60)
            print("🔥 Immediate Tasks (This Week):\n")

            # Extract immediate tasks
            immediate_section = tasks_content.split("## 🔥 SECTION 1: IMMEDIATE")[1]
            immediate_section = immediate_section.split("##")[0]  # Until next section

            # Find unchecked tasks
            unchecked = re.findall(r'- \[ \] \*\*(.*?)\*\*', immediate_section)
            for i, task in enumerate(unchecked[:3], 1):
                print(f"   {i}. {task}")

        print("\n" + "="*60)
        print("\n💡 Recommendation: Start with Pilot Conversations")
        print("   This unlocks Pilot Brief and Comparative Summary")
        print("\n" + "="*60 + "\n")

    def complete(self, task_name: str):
        """Mark a task as complete"""
        print(f"\n✅ Marking complete: {task_name}\n")

        # Update STEWARD.md Recent Completions
        steward_content = self.steward_file.read_text()
        timestamp = datetime.now().strftime('%Y-%m-%d')

        # Add to recent completions
        if "## Recent Completions" in steward_content:
            # Insert at position 1
            lines = steward_content.split('\n')
            recent_idx = next(i for i, line in enumerate(lines) if "## Recent Completions" in line)

            # Shift existing items down
            new_completion = f"1. ✅ {task_name} ({timestamp})"
            lines.insert(recent_idx + 2, new_completion)

            # Renumber existing items (keep only top 5)
            completion_count = 1
            for i in range(recent_idx + 3, len(lines)):
                if lines[i].strip().startswith(('1.', '2.', '3.', '4.', '5.', '6.')):
                    completion_count += 1
                    if completion_count <= 5:
                        lines[i] = re.sub(r'^\d+\.', f'{completion_count}.', lines[i])
                    else:
                        lines.pop(i)
                        break
                elif lines[i].strip().startswith('#'):
                    break

            steward_content = '\n'.join(lines)

            # Update timestamp
            steward_content = re.sub(
                r'\*Last Updated:.*\*',
                f'*Last Updated: {timestamp}*',
                steward_content
            )

            self.steward_file.write_text(steward_content)
            print(f"✅ Updated STEWARD.md")

        print(f"\n💡 Run 'python steward.py status' to see updated progress\n")

    # Helper methods

    def _parse_checkboxes(self, file: Path) -> Tuple[int, int]:
        """Count checked vs total checkboxes in file"""
        content = file.read_text()
        checked = len(re.findall(r'- \[x\]', content, re.IGNORECASE))
        total = len(re.findall(r'- \[ ?\]', content)) + checked
        return checked, total

    def _find_line(self, file: Path, pattern: str) -> str:
        """Find first line matching pattern"""
        content = file.read_text()
        match = re.search(pattern, content, re.IGNORECASE)
        return match.group(0) if match else ""

    def _extract_section(self, file: Path, header: str) -> str:
        """Extract content of markdown section"""
        content = file.read_text()
        if header not in content:
            return ""

        section = content.split(header)[1]
        # Get until next ## header
        next_header = re.search(r'\n##[^#]', section)
        if next_header:
            section = section[:next_header.start()]

        return section.strip()

    def _percent(self, done: int, total: int) -> int:
        """Calculate percentage"""
        return int((done / total * 100)) if total > 0 else 0


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("\n🔭 STEWARD - TELOS Project Manager")
        print("\nUsage:")
        print("  python steward.py status              - Show current state")
        print("  python steward.py next                - Suggest what to work on")
        print('  python steward.py complete "task"     - Mark task complete')
        print()
        sys.exit(1)

    command = sys.argv[1].lower()
    steward = Steward()

    if command == "status":
        steward.status()
    elif command == "next":
        steward.next()
    elif command == "complete":
        if len(sys.argv) < 3:
            print("❌ Error: Please provide task name")
            print('Usage: python steward.py complete "task name"')
            sys.exit(1)
        task_name = sys.argv[2]
        steward.complete(task_name)
    else:
        print(f"❌ Unknown command: {command}")
        print("Valid commands: status, next, complete")
        sys.exit(1)


if __name__ == "__main__":
    main()
