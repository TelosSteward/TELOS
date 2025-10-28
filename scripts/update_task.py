#!/usr/bin/env python3
"""
TELOS Task Updater

Update task status in TASKS.md.
"""

import re
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


class TaskUpdater:
    """Update tasks in TASKS.md."""

    def __init__(self, tasks_file: Path):
        self.tasks_file = tasks_file
        self.content = self._read_tasks()

    def _read_tasks(self) -> str:
        """Read TASKS.md file."""
        if not self.tasks_file.exists():
            raise FileNotFoundError(f"TASKS.md not found at {self.tasks_file}")
        return self.tasks_file.read_text()

    def _write_tasks(self, content: str) -> None:
        """Write updated content to TASKS.md."""
        self.tasks_file.write_text(content)
        print(f"✅ Updated {self.tasks_file}")

    def update_last_updated(self, content: str) -> str:
        """Update the 'Last Updated' timestamp."""
        today = datetime.now().strftime('%Y-%m-%d')
        pattern = r'(\*\*Last Updated\*\*:\s*)(\d{4}-\d{2}-\d{2})'
        replacement = f'\\1{today}'
        return re.sub(pattern, replacement, content)

    def find_task(self, task_name: str) -> Optional[str]:
        """Find a task line by name (fuzzy match)."""
        task_name_lower = task_name.lower()

        for line in self.content.split('\n'):
            if '- [' in line:
                # Extract task description
                task_desc = line.split(']', 1)[1] if ']' in line else line
                if task_name_lower in task_desc.lower():
                    return line

        return None

    def update_task_status(self, task_name: str, new_status: str) -> bool:
        """
        Update task status.

        Args:
            task_name: Name or partial name of task
            new_status: One of ✅, 🔨, ⏳, ❌
        """
        valid_statuses = ['✅', '🔨', '⏳', '❌']
        if new_status not in valid_statuses:
            print(f"❌ Error: Invalid status '{new_status}'")
            print(f"   Valid options: {', '.join(valid_statuses)}")
            return False

        # Find the task
        task_line = self.find_task(task_name)
        if not task_line:
            print(f"❌ Error: Task not found matching '{task_name}'")
            return False

        print(f"Found task: {task_line.strip()}")

        # Update checkbox if marking as complete
        updated_line = task_line
        if new_status == '✅':
            updated_line = updated_line.replace('- [ ]', '- [x]')

        # Update status indicator
        for old_status in valid_statuses:
            if old_status in updated_line and old_status != new_status:
                updated_line = updated_line.replace(old_status, new_status)
                break

        # If no status indicator found, add it
        if not any(status in updated_line for status in valid_statuses):
            # Add status after checkbox
            updated_line = updated_line.replace('- [x]', f'- [x] {new_status}')
            updated_line = updated_line.replace('- [ ]', f'- [ ] {new_status}')

        # Replace in content
        new_content = self.content.replace(task_line, updated_line)

        # Update timestamp
        new_content = self.update_last_updated(new_content)

        # Write back
        self._write_tasks(new_content)
        self.content = new_content

        print(f"✅ Updated task status to {new_status}")
        return True

    def mark_complete(self, task_name: str, completion_date: Optional[str] = None) -> bool:
        """
        Mark task as complete and optionally add completion date.

        Args:
            task_name: Name or partial name of task
            completion_date: Date string (YYYY-MM-DD) or None for today
        """
        if completion_date is None:
            completion_date = datetime.now().strftime('%Y-%m-%d')

        # Find the task
        task_line = self.find_task(task_name)
        if not task_line:
            print(f"❌ Error: Task not found matching '{task_name}'")
            return False

        print(f"Found task: {task_line.strip()}")

        # Update checkbox and status
        updated_line = task_line.replace('- [ ]', '- [x]')

        # Update or add status indicator
        if '⏳' in updated_line:
            updated_line = updated_line.replace('⏳', '✅')
        elif '🔨' in updated_line:
            updated_line = updated_line.replace('🔨', '✅')
        elif '✅' not in updated_line:
            updated_line = updated_line.replace('- [x]', '- [x] ✅')

        # Add completion date if not present
        if 'Completed' not in updated_line and completion_date:
            updated_line = updated_line.rstrip()
            updated_line += f' - Completed {completion_date}'

        # Replace in content
        new_content = self.content.replace(task_line, updated_line)

        # Update timestamp
        new_content = self.update_last_updated(new_content)

        # Write back
        self._write_tasks(new_content)
        self.content = new_content

        print(f"✅ Marked task as complete (date: {completion_date})")
        return True

    def add_task(self, section: str, task_description: str, status: str = '⏳') -> bool:
        """
        Add a new task to a section.

        Args:
            section: Section name (Immediate, Short-Term, Future)
            task_description: Task description
            status: Initial status (default: ⏳)
        """
        section_markers = {
            'immediate': '## 🔥 SECTION 1: IMMEDIATE',
            'short-term': '## 🔶 SECTION 2: SHORT-TERM',
            'future': '## 🔷 SECTION 3: FUTURE',
        }

        section_key = section.lower().replace(' ', '-')
        if section_key not in section_markers:
            print(f"❌ Error: Unknown section '{section}'")
            print(f"   Valid options: {', '.join(section_markers.keys())}")
            return False

        marker = section_markers[section_key]

        # Find section
        lines = self.content.split('\n')
        section_index = None
        for i, line in enumerate(lines):
            if marker in line:
                section_index = i
                break

        if section_index is None:
            print(f"❌ Error: Section '{section}' not found")
            return False

        # Find next subsection or section
        insert_index = section_index + 1
        for i in range(section_index + 1, len(lines)):
            if lines[i].startswith('##'):
                insert_index = i
                break
            if lines[i].strip() and not lines[i].startswith('-'):
                insert_index = i

        # Create task line
        task_line = f"- [ ] {status} **{task_description}**"

        # Insert
        lines.insert(insert_index, task_line)
        lines.insert(insert_index + 1, "")  # Add blank line

        new_content = '\n'.join(lines)

        # Update timestamp
        new_content = self.update_last_updated(new_content)

        # Write back
        self._write_tasks(new_content)
        self.content = new_content

        print(f"✅ Added task to {section} section")
        return True


def print_usage():
    """Print usage instructions."""
    print("""
TELOS Task Updater
==================

Usage:
    ./scripts/update_task.py <command> <args>

Commands:
    status <task_name> <status>
        Update task status
        Statuses: ✅ (complete), 🔨 (in progress), ⏳ (pending), ❌ (blocked)

    complete <task_name> [date]
        Mark task as complete
        Date format: YYYY-MM-DD (optional, defaults to today)

    add <section> <task_description>
        Add new task to section
        Sections: immediate, short-term, future

Examples:
    # Mark task as in progress
    ./scripts/update_task.py status "Build SharedSalienceExtractor" 🔨

    # Mark task as complete
    ./scripts/update_task.py complete "Test TELOSCOPE Dashboard"

    # Mark task as complete with specific date
    ./scripts/update_task.py complete "Build PrimacyAttractor" 2025-10-22

    # Add new task
    ./scripts/update_task.py add immediate "Write integration tests"
""")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)

    # Find TASKS.md
    tasks_file = Path(__file__).parent.parent / "TASKS.md"

    if not tasks_file.exists():
        print(f"❌ Error: TASKS.md not found at {tasks_file}")
        sys.exit(1)

    # Create updater
    updater = TaskUpdater(tasks_file)

    command = sys.argv[1].lower()

    if command == "status":
        if len(sys.argv) < 4:
            print("❌ Error: Missing arguments")
            print("Usage: ./scripts/update_task.py status <task_name> <status>")
            sys.exit(1)

        task_name = sys.argv[2]
        status = sys.argv[3]
        success = updater.update_task_status(task_name, status)
        sys.exit(0 if success else 1)

    elif command == "complete":
        if len(sys.argv) < 3:
            print("❌ Error: Missing task name")
            print("Usage: ./scripts/update_task.py complete <task_name> [date]")
            sys.exit(1)

        task_name = sys.argv[2]
        date = sys.argv[3] if len(sys.argv) > 3 else None
        success = updater.mark_complete(task_name, date)
        sys.exit(0 if success else 1)

    elif command == "add":
        if len(sys.argv) < 4:
            print("❌ Error: Missing arguments")
            print("Usage: ./scripts/update_task.py add <section> <task_description>")
            sys.exit(1)

        section = sys.argv[2]
        task_description = ' '.join(sys.argv[3:])
        success = updater.add_task(section, task_description)
        sys.exit(0 if success else 1)

    else:
        print(f"❌ Error: Unknown command '{command}'")
        print_usage()
        sys.exit(1)


if __name__ == "__main__":
    main()
