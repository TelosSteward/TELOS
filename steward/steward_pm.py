"""
TELOS Project Steward - Project Management Module
==================================================
Created: 2025-11-13
Purpose: Persistent project management and task tracking for TELOS development
         Integrates with Memory MCP for cross-session persistence

This module manages:
- Technical Paper restructuring tasks
- Grant application deadlines
- EU regulatory submission timeline
- Publication targets
- Development milestones
"""

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path to import task manager
sys.path.append(str(Path(__file__).parent.parent))
from TELOS_PAPER_RESTRUCTURE_TASKS import TELOSPaperTaskManager, TaskStatus, Priority

class TELOSSteward:
    """
    Central project management system for TELOS.
    Coordinates all development, publication, and regulatory activities.
    """

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.task_manager = TELOSPaperTaskManager()
        self.state_file = self.project_root / "steward_state.json"
        self.load_state()

    def load_state(self):
        """Load persisted state from previous sessions"""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                self.state = json.load(f)
        else:
            self.state = self.initialize_state()

    def initialize_state(self) -> dict:
        """Initialize fresh project state"""
        return {
            "project_name": "TELOS",
            "current_phase": "Technical Paper Restructuring",
            "session_count": 0,
            "last_updated": datetime.now().isoformat(),
            "milestones": {
                "paper_separation": {
                    "status": "NOT_STARTED",
                    "target_date": "2025-11-20",
                    "description": "Separate 50K word compendium into 3 documents"
                },
                "eu_consultation": {
                    "status": "NOT_STARTED",
                    "target_date": "2025-11-15",
                    "description": "Schedule EU Commission preliminary consultation"
                },
                "neurips_submission": {
                    "status": "NOT_STARTED",
                    "target_date": "2025-05-15",
                    "description": "Submit to NeurIPS 2025"
                },
                "eu_article_72": {
                    "status": "NOT_STARTED",
                    "target_date": "2026-02-01",
                    "description": "EU AI Act Article 72 submission"
                }
            },
            "critical_reminders": [],
            "session_history": []
        }

    def save_state(self):
        """Persist current state to disk"""
        self.state["last_updated"] = datetime.now().isoformat()
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)

    def start_session(self):
        """Initialize a new work session"""
        self.state["session_count"] += 1
        session_info = {
            "session_id": f"session_{self.state['session_count']}",
            "started": datetime.now().isoformat(),
            "tasks_completed": [],
            "notes": []
        }
        self.state["session_history"].append(session_info)
        self.save_state()
        return session_info["session_id"]

    def get_current_priorities(self) -> str:
        """Get current priority tasks and deadlines"""
        report = """
TELOS PROJECT STATUS - CRITICAL PRIORITIES
==========================================

IMMEDIATE ACTIONS REQUIRED:
---------------------------"""

        # Check for immediate tasks
        immediate_tasks = self.task_manager.get_tasks_by_priority(Priority.IMMEDIATE)
        for task in immediate_tasks:
            if task.status != TaskStatus.COMPLETED:
                deadline_str = ""
                if task.deadline:
                    days_until = (task.deadline - datetime.now()).days
                    if days_until < 0:
                        deadline_str = f" [OVERDUE by {-days_until} days]"
                    else:
                        deadline_str = f" [Due in {days_until} days]"
                report += f"\n• {task.title}{deadline_str}"
                report += f"\n  ID: {task.task_id} | Est: {task.estimated_hours}h"

        # Check critical milestones
        report += "\n\nCRITICAL MILESTONES:\n--------------------"
        for milestone_key, milestone in self.state["milestones"].items():
            if milestone["status"] != "COMPLETED":
                target = datetime.fromisoformat(milestone["target_date"])
                days_until = (target - datetime.now()).days
                status_indicator = "⚠️" if days_until < 7 else "📅"
                report += f"\n{status_indicator} {milestone['description']}"
                report += f"\n  Target: {milestone['target_date']} ({days_until} days)"

        # Add ready tasks
        ready_tasks = self.task_manager.get_ready_tasks()
        if ready_tasks:
            report += "\n\nREADY TO START:\n---------------"
            for task in ready_tasks[:3]:
                report += f"\n• [{task.task_id}] {task.title}"

        return report

    def update_task(self, task_id: str, status: str, notes: str = ""):
        """Update a task and persist the change"""
        task_status = TaskStatus[status.upper()]
        self.task_manager.update_task_status(task_id, task_status, notes)

        # Log to session history
        if self.state["session_history"]:
            current_session = self.state["session_history"][-1]
            if task_status == TaskStatus.COMPLETED:
                current_session["tasks_completed"].append(task_id)
            if notes:
                current_session["notes"].append(f"[{task_id}] {notes}")

        self.save_state()

    def generate_weekly_report(self) -> str:
        """Generate weekly progress report"""
        report = f"""
TELOS WEEKLY PROGRESS REPORT
============================
Week of: {datetime.now().strftime('%Y-%m-%d')}

{self.task_manager.generate_status_report()}

UPCOMING DEADLINES:
-------------------"""

        # Find tasks with deadlines in next 30 days
        for task in self.task_manager.tasks.values():
            if task.deadline and task.status != TaskStatus.COMPLETED:
                days_until = (task.deadline - datetime.now()).days
                if 0 <= days_until <= 30:
                    report += f"\n• {task.title}: {task.deadline.strftime('%Y-%m-%d')} ({days_until} days)"

        return report

    def export_for_memory_mcp(self) -> dict:
        """Export current state for Memory MCP persistence"""
        return {
            "project_state": self.state,
            "task_summary": {
                "total_tasks": len(self.task_manager.tasks),
                "completed": len(self.task_manager.get_tasks_by_status(TaskStatus.COMPLETED)),
                "in_progress": len(self.task_manager.get_tasks_by_status(TaskStatus.IN_PROGRESS)),
                "blocked": len(self.task_manager.get_blocked_tasks())
            },
            "critical_path": self.task_manager.get_critical_path(),
            "immediate_priorities": [
                task.task_id for task in
                self.task_manager.get_tasks_by_priority(Priority.IMMEDIATE)
                if task.status != TaskStatus.COMPLETED
            ]
        }

# Initialize steward on module load
steward = TELOSSteward()

if __name__ == "__main__":
    # Start a new session
    session_id = steward.start_session()
    print(f"Started session: {session_id}")

    # Show current priorities
    print(steward.get_current_priorities())

    # Export for Memory MCP
    mcp_data = steward.export_for_memory_mcp()
    with open(steward.project_root / "steward_mcp_export.json", 'w') as f:
        json.dump(mcp_data, f, indent=2)
    print("\nExported state for Memory MCP to steward_mcp_export.json")