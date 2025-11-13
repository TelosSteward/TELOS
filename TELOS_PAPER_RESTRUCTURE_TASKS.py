"""
TELOS Technical Paper Restructuring Task Manager
=================================================
Created: 2025-11-13
Purpose: Systematic task management for restructuring the TELOS Technical Paper Compendium
         into publication-ready documents for academic venues, regulatory submission,
         and implementation guidance.

This task manager will be integrated with Steward_pm.py for session persistence.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
from datetime import datetime, timedelta
import json

class Priority(Enum):
    IMMEDIATE = "IMMEDIATE"  # Blocking other tasks
    HIGH = "HIGH"            # Time-sensitive
    MEDIUM = "MEDIUM"        # Important but not urgent
    LOW = "LOW"              # Nice to have

class TaskStatus(Enum):
    NOT_STARTED = "NOT_STARTED"
    IN_PROGRESS = "IN_PROGRESS"
    BLOCKED = "BLOCKED"
    COMPLETED = "COMPLETED"
    DEFERRED = "DEFERRED"

class DocumentType(Enum):
    ACADEMIC_PAPER = "ACADEMIC_PAPER"
    EU_REGULATORY = "EU_REGULATORY"
    IMPLEMENTATION_GUIDE = "IMPLEMENTATION_GUIDE"
    GRANT_APPLICATION = "GRANT_APPLICATION"
    SUPPLEMENTARY = "SUPPLEMENTARY"

@dataclass
class TELOSTask:
    """Individual task for TELOS paper restructuring"""
    task_id: str
    title: str
    description: str
    priority: Priority
    status: TaskStatus = TaskStatus.NOT_STARTED
    document_type: Optional[DocumentType] = None
    dependencies: List[str] = field(default_factory=list)
    deadline: Optional[datetime] = None
    estimated_hours: float = 0.0
    actual_hours: float = 0.0
    assignee: str = "Owner"
    notes: str = ""
    completion_criteria: List[str] = field(default_factory=list)
    artifacts: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "title": self.title,
            "description": self.description,
            "priority": self.priority.value,
            "status": self.status.value,
            "document_type": self.document_type.value if self.document_type else None,
            "dependencies": self.dependencies,
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "estimated_hours": self.estimated_hours,
            "actual_hours": self.actual_hours,
            "assignee": self.assignee,
            "notes": self.notes,
            "completion_criteria": self.completion_criteria,
            "artifacts": self.artifacts
        }

class TELOSPaperTaskManager:
    """
    Main task manager for TELOS Technical Paper restructuring project.
    Tracks all tasks, dependencies, and progress toward publication goals.
    """

    def __init__(self):
        self.tasks: Dict[str, TELOSTask] = {}
        self.initialize_tasks()

    def initialize_tasks(self):
        """Initialize all tasks for the TELOS paper restructuring project"""

        # ========== PRIMARY TASK: DOCUMENT SEPARATION ==========
        self.add_task(TELOSTask(
            task_id="DOC_SEP_001",
            title="Separate Compendium into Three Root Documents",
            description="Split the 50,800-word TELOS_TECHNICAL_PAPER.md into three specialized documents for different audiences",
            priority=Priority.IMMEDIATE,
            document_type=DocumentType.ACADEMIC_PAPER,
            estimated_hours=8.0,
            completion_criteria=[
                "TELOS_Academic_Paper.md created (8-12K words)",
                "TELOS_EU_Article72_Submission.md created (15-20K words)",
                "TELOS_Implementation_Guide.md created (20-30K words)",
                "Cross-references between documents established",
                "Each document has appropriate front matter"
            ]
        ))

        # ========== ACADEMIC PAPER TASKS ==========
        self.add_task(TELOSTask(
            task_id="ACAD_001",
            title="Add Statistical Validity Subsection",
            description="Add section proving why 84 attacks establishes 0% ASR with 99% confidence using Wilson score intervals",
            priority=Priority.HIGH,
            document_type=DocumentType.ACADEMIC_PAPER,
            dependencies=["DOC_SEP_001"],
            estimated_hours=4.0,
            completion_criteria=[
                "Wilson score confidence interval calculation for 0/84",
                "Power analysis for sample size justification",
                "Comparison to adversarial testing literature baselines",
                "Discussion of type II error rates"
            ]
        ))

        self.add_task(TELOSTask(
            task_id="ACAD_002",
            title="Create Essential Figures and Diagrams",
            description="Design and create 5-7 key figures for the academic paper",
            priority=Priority.HIGH,
            document_type=DocumentType.ACADEMIC_PAPER,
            dependencies=["DOC_SEP_001"],
            estimated_hours=12.0,
            completion_criteria=[
                "Three-tier architecture diagram",
                "PA basin visualization in 2D/3D",
                "Attack sophistication vs ASR comparison chart",
                "Forensic decision trace flowchart",
                "TELOSCOPE counterfactual branching diagram",
                "Optional: Lyapunov stability phase portrait",
                "Optional: RoPE attention decay visualization"
            ]
        ))

        self.add_task(TELOSTask(
            task_id="ACAD_003",
            title="Write Limitations Section",
            description="Add 1-2 page limitations section acknowledging system boundaries",
            priority=Priority.HIGH,
            document_type=DocumentType.ACADEMIC_PAPER,
            dependencies=["DOC_SEP_001"],
            estimated_hours=3.0,
            completion_criteria=[
                "Embedding model dependency discussion",
                "Domain specificity limitations",
                "Computational overhead analysis",
                "Human expert scalability constraints",
                "Out-of-distribution attack discussion"
            ]
        ))

        self.add_task(TELOSTask(
            task_id="ACAD_004",
            title="Add Related Work Comparison Table",
            description="Create quantitative comparison with competing approaches",
            priority=Priority.MEDIUM,
            document_type=DocumentType.ACADEMIC_PAPER,
            dependencies=["DOC_SEP_001"],
            estimated_hours=6.0,
            completion_criteria=[
                "TELOS vs Anthropic Constitutional AI comparison",
                "TELOS vs OpenAI Moderation API comparison",
                "TELOS vs LLaMA Guard comparison",
                "Metrics: ASR, latency, cost, regulatory compliance",
                "Architectural differences summary"
            ]
        ))

        self.add_task(TELOSTask(
            task_id="ACAD_005",
            title="Create Minimal Reproducible Example",
            description="Write self-contained Python script (100-150 lines) demonstrating core TELOS functionality",
            priority=Priority.MEDIUM,
            document_type=DocumentType.ACADEMIC_PAPER,
            dependencies=["DOC_SEP_001"],
            estimated_hours=4.0,
            completion_criteria=[
                "PA instantiation code",
                "Fidelity calculation demo",
                "Simple attack blocking example",
                "Forensic trace output",
                "Clear comments and documentation"
            ]
        ))

        self.add_task(TELOSTask(
            task_id="ACAD_006",
            title="Reframe Paper Around TELOSCOPE",
            description="Position TELOSCOPE as primary methodological contribution with TELOS as use case",
            priority=Priority.HIGH,
            document_type=DocumentType.ACADEMIC_PAPER,
            dependencies=["DOC_SEP_001", "ACAD_002"],
            estimated_hours=6.0,
            completion_criteria=[
                "Abstract rewritten to emphasize TELOSCOPE",
                "Introduction positions observability problem",
                "TELOSCOPE section moved earlier",
                "Comparison with existing observability tools",
                "Broader applicability emphasized"
            ]
        ))

        # ========== EU REGULATORY TASKS ==========
        self.add_task(TELOSTask(
            task_id="EU_001",
            title="Create Article 72 Post-Market Monitoring Plan",
            description="Draft comprehensive monitoring plan per EU AI Act Article 72 requirements",
            priority=Priority.IMMEDIATE,
            document_type=DocumentType.EU_REGULATORY,
            dependencies=["DOC_SEP_001"],
            deadline=datetime(2025, 12, 1),  # Must be ready before consultation
            estimated_hours=8.0,
            completion_criteria=[
                "Map telemetry to Article 72(1)(a-f)",
                "Define detection thresholds for drift",
                "Document incident response workflows",
                "Specify update/patching procedures",
                "Create periodic re-validation schedule",
                "Include corrective action procedures"
            ]
        ))

        self.add_task(TELOSTask(
            task_id="EU_002",
            title="Schedule EU Commission Consultation",
            description="Initiate preliminary consultation with EU Commission (12-month lead time typical)",
            priority=Priority.IMMEDIATE,
            document_type=DocumentType.EU_REGULATORY,
            deadline=datetime(2025, 11, 15),  # ASAP
            estimated_hours=2.0,
            completion_criteria=[
                "Identify correct EU contact point",
                "Prepare preliminary documentation package",
                "Submit consultation request",
                "Schedule initial meeting"
            ]
        ))

        self.add_task(TELOSTask(
            task_id="EU_003",
            title="Prepare Incident Response Workflow",
            description="Document complete incident response procedures for AI governance violations",
            priority=Priority.HIGH,
            document_type=DocumentType.EU_REGULATORY,
            dependencies=["DOC_SEP_001", "EU_001"],
            estimated_hours=6.0,
            completion_criteria=[
                "Detection mechanisms documented",
                "Escalation procedures defined",
                "Response team roles specified",
                "Communication protocols established",
                "Remediation procedures outlined",
                "Post-incident review process"
            ]
        ))

        # ========== IMPLEMENTATION GUIDE TASKS ==========
        self.add_task(TELOSTask(
            task_id="IMPL_001",
            title="Add Deployment Cost Analysis",
            description="Provide concrete cost estimates for TELOS deployment at various scales",
            priority=Priority.MEDIUM,
            document_type=DocumentType.IMPLEMENTATION_GUIDE,
            dependencies=["DOC_SEP_001"],
            estimated_hours=4.0,
            completion_criteria=[
                "API cost estimates ($/1000 queries)",
                "Infrastructure requirements by scale",
                "Human expert cost modeling",
                "ROI calculation framework",
                "Comparison with manual review costs"
            ]
        ))

        self.add_task(TELOSTask(
            task_id="IMPL_002",
            title="Create Integration Examples",
            description="Provide code examples for integrating TELOS with popular frameworks",
            priority=Priority.MEDIUM,
            document_type=DocumentType.IMPLEMENTATION_GUIDE,
            dependencies=["DOC_SEP_001"],
            estimated_hours=8.0,
            completion_criteria=[
                "LangChain integration example",
                "Microsoft Semantic Kernel example",
                "NVIDIA NeMo Guardrails example",
                "Direct API integration example",
                "Testing harness examples"
            ]
        ))

        # ========== GRANT APPLICATION TASKS ==========
        self.add_task(TELOSTask(
            task_id="GRANT_001",
            title="Write Plain English Executive Summary",
            description="Create 1-page non-technical summary for Emergent Ventures reviewers",
            priority=Priority.HIGH,
            document_type=DocumentType.GRANT_APPLICATION,
            dependencies=["DOC_SEP_001"],
            estimated_hours=3.0,
            completion_criteria=[
                "Problem statement in plain language",
                "Solution description without jargon",
                "Impact statement",
                "Team qualifications",
                "Funding use description"
            ]
        ))

        self.add_task(TELOSTask(
            task_id="GRANT_002",
            title="Healthcare AI Market Sizing Analysis",
            description="Quantify market opportunity for NSF SBIR application",
            priority=Priority.MEDIUM,
            document_type=DocumentType.GRANT_APPLICATION,
            dependencies=["DOC_SEP_001"],
            estimated_hours=6.0,
            completion_criteria=[
                "TAM/SAM/SOM analysis",
                "Healthcare AI governance market size",
                "Customer segments identified",
                "Pricing model analysis",
                "5-year revenue projections"
            ]
        ))

        self.add_task(TELOSTask(
            task_id="GRANT_003",
            title="Identify EU Consortium Partners",
            description="Find 2-3 EU academic/industry partners for Horizon Europe application",
            priority=Priority.HIGH,
            document_type=DocumentType.GRANT_APPLICATION,
            deadline=datetime(2025, 12, 15),
            estimated_hours=10.0,
            completion_criteria=[
                "List of potential partners compiled",
                "Initial outreach emails sent",
                "Letters of intent secured",
                "Consortium agreement drafted",
                "Work package allocation defined"
            ]
        ))

        # ========== PUBLICATION SUBMISSION TASKS ==========
        self.add_task(TELOSTask(
            task_id="PUB_001",
            title="NeurIPS 2025 Submission Preparation",
            description="Prepare submission for NeurIPS 2025 (deadline ~May 2025)",
            priority=Priority.MEDIUM,
            document_type=DocumentType.ACADEMIC_PAPER,
            dependencies=["ACAD_001", "ACAD_002", "ACAD_003", "ACAD_006"],
            deadline=datetime(2025, 5, 15),  # Approximate
            estimated_hours=20.0,
            completion_criteria=[
                "Paper formatted to NeurIPS template",
                "Supplementary materials prepared",
                "Code repository public",
                "Anonymous submission prepared",
                "Internal review completed"
            ]
        ))

        self.add_task(TELOSTask(
            task_id="PUB_002",
            title="USENIX Security 2025 Submission",
            description="Alternative venue focusing on adversarial validation",
            priority=Priority.LOW,
            document_type=DocumentType.ACADEMIC_PAPER,
            dependencies=["ACAD_001", "ACAD_002", "ACAD_003"],
            deadline=datetime(2025, 2, 28),  # Typical deadline
            estimated_hours=15.0,
            completion_criteria=[
                "Security angle emphasized",
                "Attack taxonomy highlighted",
                "Forensic analysis featured",
                "Format per USENIX requirements"
            ]
        ))

    def add_task(self, task: TELOSTask):
        """Add a task to the manager"""
        self.tasks[task.task_id] = task

    def get_task(self, task_id: str) -> Optional[TELOSTask]:
        """Retrieve a task by ID"""
        return self.tasks.get(task_id)

    def update_task_status(self, task_id: str, new_status: TaskStatus, notes: str = "", hours_spent: float = 0.0):
        """Update the status of a task"""
        if task_id in self.tasks:
            self.tasks[task_id].status = new_status
            if hours_spent > 0:
                self.tasks[task_id].actual_hours = hours_spent
            if notes:
                self.tasks[task_id].notes += f"\n[{datetime.now().isoformat()}] {notes}"

    def get_tasks_by_priority(self, priority: Priority) -> List[TELOSTask]:
        """Get all tasks with a specific priority"""
        return [t for t in self.tasks.values() if t.priority == priority]

    def get_tasks_by_status(self, status: TaskStatus) -> List[TELOSTask]:
        """Get all tasks with a specific status"""
        return [t for t in self.tasks.values() if t.status == status]

    def get_ready_tasks(self) -> List[TELOSTask]:
        """Get tasks that are ready to start (dependencies met)"""
        ready = []
        for task in self.tasks.values():
            if task.status == TaskStatus.NOT_STARTED:
                deps_met = all(
                    self.tasks[dep].status == TaskStatus.COMPLETED
                    for dep in task.dependencies
                )
                if deps_met:
                    ready.append(task)
        return ready

    def get_blocked_tasks(self) -> List[TELOSTask]:
        """Get tasks blocked by incomplete dependencies"""
        blocked = []
        for task in self.tasks.values():
            if task.status == TaskStatus.NOT_STARTED:
                deps_incomplete = any(
                    self.tasks[dep].status != TaskStatus.COMPLETED
                    for dep in task.dependencies
                )
                if deps_incomplete and task.dependencies:
                    blocked.append(task)
        return blocked

    def export_to_json(self, filepath: str):
        """Export all tasks to JSON for persistence"""
        data = {
            "project": "TELOS Technical Paper Restructuring",
            "created": datetime.now().isoformat(),
            "tasks": [task.to_dict() for task in self.tasks.values()]
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def generate_status_report(self) -> str:
        """Generate a human-readable status report"""
        total = len(self.tasks)
        completed = len(self.get_tasks_by_status(TaskStatus.COMPLETED))
        in_progress = len(self.get_tasks_by_status(TaskStatus.IN_PROGRESS))
        not_started = len(self.get_tasks_by_status(TaskStatus.NOT_STARTED))
        blocked_tasks = self.get_blocked_tasks()
        blocked = len(blocked_tasks)

        immediate = len(self.get_tasks_by_priority(Priority.IMMEDIATE))
        high = len(self.get_tasks_by_priority(Priority.HIGH))

        ready = self.get_ready_tasks()

        report = f"""
TELOS TECHNICAL PAPER RESTRUCTURING - STATUS REPORT
====================================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

OVERALL PROGRESS:
-----------------
Total Tasks: {total}
Completed: {completed} ({100*completed/total:.1f}%)
In Progress: {in_progress}
Not Started: {not_started}
Blocked: {blocked}

PRIORITY BREAKDOWN:
-------------------
IMMEDIATE Priority: {immediate} tasks
HIGH Priority: {high} tasks

READY TO START:
---------------"""

        for task in ready[:5]:  # Show top 5 ready tasks
            report += f"\n• [{task.task_id}] {task.title}"
            if task.priority == Priority.IMMEDIATE:
                report += " [IMMEDIATE]"
            elif task.priority == Priority.HIGH:
                report += " [HIGH]"

        if blocked_tasks:
            report += "\n\nBLOCKED TASKS:\n--------------"
            for task in blocked_tasks[:5]:
                deps = ", ".join(task.dependencies)
                report += f"\n• [{task.task_id}] {task.title}\n  Waiting on: {deps}"

        return report

    def get_critical_path(self) -> List[str]:
        """Identify the critical path of tasks for EU submission deadline"""
        # Simplified critical path for EU February 2026 deadline
        return [
            "DOC_SEP_001",  # Must separate documents first
            "EU_002",       # Schedule consultation ASAP
            "EU_001",       # Create monitoring plan
            "EU_003",       # Prepare incident response
        ]

# Initialize the task manager
task_manager = TELOSPaperTaskManager()

if __name__ == "__main__":
    # Generate initial status report
    print(task_manager.generate_status_report())

    # Export tasks for persistence
    task_manager.export_to_json("/Users/brunnerjf/Desktop/telos_privacy/telos_paper_tasks.json")
    print("\nTasks exported to telos_paper_tasks.json")

    # Show critical path
    print("\nCRITICAL PATH (for EU deadline):")
    for task_id in task_manager.get_critical_path():
        task = task_manager.get_task(task_id)
        print(f"  → {task.title}")