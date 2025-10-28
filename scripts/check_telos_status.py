#!/usr/bin/env python3
"""
TELOS Status Checker

Analyzes TASKS.md and provides status summary.
"""

import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple


class TelosStatusChecker:
    """Check TELOS project status from TASKS.md."""

    def __init__(self, tasks_file: Path):
        self.tasks_file = tasks_file
        self.content = self._read_tasks()

    def _read_tasks(self) -> str:
        """Read TASKS.md file."""
        if not self.tasks_file.exists():
            raise FileNotFoundError(f"TASKS.md not found at {self.tasks_file}")
        return self.tasks_file.read_text()

    def count_by_status(self) -> Dict[str, int]:
        """Count tasks by status indicator."""
        statuses = {
            '✅': 0,  # Completed
            '🔨': 0,  # In progress
            '⏳': 0,  # Pending
            '❌': 0,  # Failed/blocked
        }

        for line in self.content.split('\n'):
            for status in statuses.keys():
                if status in line and ('- [' in line or '- [ ]' in line or '- [x]' in line):
                    statuses[status] += 1

        return statuses

    def count_checkboxes(self) -> Tuple[int, int]:
        """Count checked vs unchecked tasks."""
        checked = len(re.findall(r'- \[x\]', self.content))
        unchecked = len(re.findall(r'- \[ \]', self.content))
        return checked, unchecked

    def extract_sections(self) -> Dict[str, List[str]]:
        """Extract tasks by section."""
        sections = {
            'Immediate': [],
            'Short-Term': [],
            'Future': [],
            'Completed': []
        }

        current_section = None
        for line in self.content.split('\n'):
            if '## 🔥 SECTION 1: IMMEDIATE' in line:
                current_section = 'Immediate'
            elif '## 🔶 SECTION 2: SHORT-TERM' in line:
                current_section = 'Short-Term'
            elif '## 🔷 SECTION 3: FUTURE' in line:
                current_section = 'Future'
            elif '## ✅ SECTION 4: COMPLETED' in line:
                current_section = 'Completed'

            if current_section and line.strip().startswith('- ['):
                sections[current_section].append(line.strip())

        return sections

    def get_completion_percentage(self) -> float:
        """Calculate completion percentage."""
        checked, unchecked = self.count_checkboxes()
        total = checked + unchecked
        if total == 0:
            return 0.0
        return (checked / total) * 100

    def get_next_tasks(self, limit: int = 5) -> List[str]:
        """Get next pending tasks."""
        next_tasks = []
        in_immediate = False

        for line in self.content.split('\n'):
            if '## 🔥 SECTION 1: IMMEDIATE' in line:
                in_immediate = True
            elif line.startswith('## ') and in_immediate:
                break

            if in_immediate and '- [ ]' in line and '⏳' in line:
                # Clean up the task description
                task = line.strip()
                task = task.replace('- [ ]', '').strip()
                next_tasks.append(task)

                if len(next_tasks) >= limit:
                    break

        return next_tasks

    def get_last_updated(self) -> str:
        """Extract last updated date from TASKS.md."""
        match = re.search(r'\*\*Last Updated\*\*:\s*(\d{4}-\d{2}-\d{2})', self.content)
        if match:
            return match.group(1)
        return "Unknown"

    def generate_report(self) -> str:
        """Generate comprehensive status report."""
        report = []

        # Header
        report.append("=" * 60)
        report.append("TELOS STATUS REPORT")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Last Updated: {self.get_last_updated()}")
        report.append("=" * 60)
        report.append("")

        # Progress summary
        checked, unchecked = self.count_checkboxes()
        total = checked + unchecked
        percentage = self.get_completion_percentage()

        report.append("📊 PROGRESS SUMMARY")
        report.append("-" * 60)
        report.append(f"Total Tasks:       {total}")
        report.append(f"Completed:         {checked} ({percentage:.1f}%)")
        report.append(f"Remaining:         {unchecked} ({100-percentage:.1f}%)")
        report.append("")

        # Status breakdown
        statuses = self.count_by_status()
        report.append("📈 STATUS BREAKDOWN")
        report.append("-" * 60)
        report.append(f"✅ Completed:      {statuses['✅']}")
        report.append(f"🔨 In Progress:    {statuses['🔨']}")
        report.append(f"⏳ Pending:        {statuses['⏳']}")
        report.append(f"❌ Blocked:        {statuses['❌']}")
        report.append("")

        # Section summary
        sections = self.extract_sections()
        report.append("📂 SECTION BREAKDOWN")
        report.append("-" * 60)
        for section, tasks in sections.items():
            report.append(f"{section:15} {len(tasks)} tasks")
        report.append("")

        # Next tasks
        next_tasks = self.get_next_tasks(5)
        if next_tasks:
            report.append("🎯 NEXT TASKS (Top 5 Immediate)")
            report.append("-" * 60)
            for i, task in enumerate(next_tasks, 1):
                # Truncate long tasks
                if len(task) > 55:
                    task = task[:52] + "..."
                report.append(f"{i}. {task}")
            report.append("")

        # Progress bar
        report.append("📊 COMPLETION PROGRESS")
        report.append("-" * 60)
        bar_length = 40
        filled = int(bar_length * percentage / 100)
        bar = "█" * filled + "░" * (bar_length - filled)
        report.append(f"[{bar}] {percentage:.1f}%")
        report.append("")

        # Footer
        report.append("=" * 60)
        report.append("💡 TIP: Run './scripts/update_task.py <task_name> ✅' to mark as complete")
        report.append("🔭 Making AI Governance Observable")
        report.append("=" * 60)

        return "\n".join(report)


def main():
    """Main entry point."""
    import sys

    # Find TASKS.md
    tasks_file = Path(__file__).parent.parent / "TASKS.md"

    if not tasks_file.exists():
        print(f"❌ Error: TASKS.md not found at {tasks_file}")
        sys.exit(1)

    # Create checker
    checker = TelosStatusChecker(tasks_file)

    # Generate and print report
    report = checker.generate_report()
    print(report)


if __name__ == "__main__":
    main()
