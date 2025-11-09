#!/usr/bin/env python3
"""
Structured Planning Tool for TELOS Adversarial Validation
Uses sequential thinking methodology without requiring MCP server connection.
"""

import json
from datetime import datetime
from pathlib import Path


class StructuredPlanner:
    """
    Implements structured, sequential thinking for complex planning tasks.
    Based on @modelcontextprotocol/server-sequential-thinking methodology.
    """

    def __init__(self, project_name: str, output_dir: str = "./planning_output"):
        self.project_name = project_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.plan = {
            "project": project_name,
            "created": datetime.now().isoformat(),
            "phases": [],
            "dependencies": [],
            "risks": [],
            "success_criteria": []
        }

    def add_phase(self, name: str, description: str, steps: list,
                  duration_estimate: str = None, dependencies: list = None):
        """Add a planning phase with sequential steps."""
        phase = {
            "phase_number": len(self.plan["phases"]) + 1,
            "name": name,
            "description": description,
            "steps": steps,
            "duration_estimate": duration_estimate,
            "dependencies": dependencies or [],
            "status": "planned"
        }
        self.plan["phases"].append(phase)

    def add_risk(self, risk: str, mitigation: str, severity: str = "medium"):
        """Document potential risks and mitigation strategies."""
        self.plan["risks"].append({
            "risk": risk,
            "severity": severity,
            "mitigation": mitigation
        })

    def add_success_criterion(self, criterion: str, measurement: str):
        """Define success criteria and how to measure them."""
        self.plan["success_criteria"].append({
            "criterion": criterion,
            "measurement": measurement
        })

    def add_dependency(self, from_phase: str, to_phase: str, reason: str):
        """Document dependencies between phases."""
        self.plan["dependencies"].append({
            "from": from_phase,
            "to": to_phase,
            "reason": reason
        })

    def save_plan(self, filename: str = None):
        """Save plan to JSON file."""
        if filename is None:
            filename = f"{self.project_name.lower().replace(' ', '_')}_plan.json"

        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            json.dump(self.plan, f, indent=2)

        print(f"✅ Plan saved to: {output_path}")
        return output_path

    def generate_markdown_report(self, filename: str = None):
        """Generate human-readable markdown report."""
        if filename is None:
            filename = f"{self.project_name.lower().replace(' ', '_')}_plan.md"

        output_path = self.output_dir / filename

        md_content = f"""# {self.plan['project']} - Structured Planning Document

**Created**: {self.plan['created']}

---

## Overview

This document outlines the comprehensive plan for {self.plan['project']}.

---

## Phases

"""

        for phase in self.plan["phases"]:
            md_content += f"""### Phase {phase['phase_number']}: {phase['name']}

**Description**: {phase['description']}

**Duration Estimate**: {phase.get('duration_estimate', 'TBD')}

**Dependencies**: {', '.join(phase.get('dependencies', [])) or 'None'}

**Steps**:

"""
            for i, step in enumerate(phase['steps'], 1):
                md_content += f"{i}. {step}\n"

            md_content += "\n---\n\n"

        # Add dependencies section
        if self.plan['dependencies']:
            md_content += "## Phase Dependencies\n\n"
            for dep in self.plan['dependencies']:
                md_content += f"- **{dep['from']}** → **{dep['to']}**: {dep['reason']}\n"
            md_content += "\n---\n\n"

        # Add risks section
        if self.plan['risks']:
            md_content += "## Risk Analysis\n\n"
            for risk in self.plan['risks']:
                md_content += f"""### {risk['severity'].upper()}: {risk['risk']}

**Mitigation**: {risk['mitigation']}

"""
            md_content += "---\n\n"

        # Add success criteria
        if self.plan['success_criteria']:
            md_content += "## Success Criteria\n\n"
            for sc in self.plan['success_criteria']:
                md_content += f"- **{sc['criterion']}**\n  - Measurement: {sc['measurement']}\n"
            md_content += "\n---\n\n"

        md_content += f"""## Next Steps

1. Review this plan with stakeholders
2. Validate dependencies and timeline
3. Begin Phase 1 implementation
4. Track progress against success criteria

---

*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

        with open(output_path, 'w') as f:
            f.write(md_content)

        print(f"✅ Markdown report saved to: {output_path}")
        return output_path

    def print_summary(self):
        """Print a summary of the plan."""
        print(f"\n{'='*80}")
        print(f"📋 PLANNING SUMMARY: {self.project_name}")
        print(f"{'='*80}\n")

        print(f"Total Phases: {len(self.plan['phases'])}")
        print(f"Total Risks Identified: {len(self.plan['risks'])}")
        print(f"Success Criteria: {len(self.plan['success_criteria'])}\n")

        print("PHASES:")
        for phase in self.plan['phases']:
            print(f"  Phase {phase['phase_number']}: {phase['name']}")
            print(f"    - Steps: {len(phase['steps'])}")
            print(f"    - Duration: {phase.get('duration_estimate', 'TBD')}")
            print()


def main():
    """Example usage of StructuredPlanner."""
    planner = StructuredPlanner("Example Project")

    planner.add_phase(
        name="Planning",
        description="Initial planning and requirements gathering",
        steps=[
            "Define objectives",
            "Identify stakeholders",
            "Gather requirements"
        ],
        duration_estimate="1 week"
    )

    planner.add_risk(
        risk="Timeline may slip due to unforeseen complexity",
        mitigation="Build buffer time into estimates",
        severity="medium"
    )

    planner.add_success_criterion(
        criterion="Project completed on time",
        measurement="Final delivery date vs planned date"
    )

    planner.print_summary()
    planner.save_plan()
    planner.generate_markdown_report()


if __name__ == "__main__":
    main()
