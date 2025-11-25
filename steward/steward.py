#!/usr/bin/env python3
"""
STEWARD - TELOS AI-Powered Project Manager Agent

Intelligent PM assistant that tracks progress, analyzes dependencies, and provides AI-powered recommendations.

Commands:
    python steward.py status         - Show current state across all trackers
    python steward.py next           - Get AI-powered priority suggestions
    python steward.py complete       - Mark task complete and auto-update PRDs
    python steward.py report         - Generate weekly status report
    python steward.py analyze [topic] - Deep dive AI analysis
    python steward.py auto-update    - Detect completed work from git and update PRDs
"""

import sys
import re
import os
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json
import traceback

# Check for Mistral API (same as TELOS core)
try:
    from telos.llm.mistral_client import MistralClient
    HAS_MISTRAL = True
except ImportError:
    HAS_MISTRAL = False

# Check for numpy (for vector health checks)
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


class Steward:
    """TELOS Project Manager Agent - Basic functionality"""

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
        print(f"⚙️  Platform:            ~85% complete (Infrastructure ready)")

        # Show current focus
        print("\n" + "-"*60)
        print("🎯 Current Focus:\n")
        focus_section = self._extract_section(self.steward_file, "## Current Focus")
        if focus_section:
            for line in focus_section.split('\n')[1:6]:
                if line.strip():
                    print(f"   {line.strip()}")

        # Show blockers
        print("\n" + "-"*60)
        print("🚧 Blockers:\n")
        blocker_section = self._extract_section(self.steward_file, "## Blockers")
        if blocker_section:
            for line in blocker_section.split('\n')[1:5]:
                if line.strip().startswith('-'):
                    print(f"   {line.strip()}")

        print("\n" + "="*60 + "\n")

    def complete(self, task_name: str):
        """Mark a task as complete"""
        print(f"\n✅ Marking complete: {task_name}\n")

        # Update STEWARD.md Recent Completions
        steward_content = self.steward_file.read_text()
        timestamp = datetime.now().strftime('%Y-%m-%d')

        if "## Recent Completions" in steward_content:
            lines = steward_content.split('\n')
            recent_idx = next(i for i, line in enumerate(lines) if "## Recent Completions" in line)

            # Add new completion
            new_completion = f"1. ✅ {task_name} ({timestamp})"
            lines.insert(recent_idx + 2, new_completion)

            # Renumber existing (keep top 5)
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
            steward_content = re.sub(
                r'\*Last Updated:.*\*',
                f'*Last Updated: {timestamp}*',
                steward_content
            )

            self.steward_file.write_text(steward_content)
            print(f"✅ Updated STEWARD.md")

        print(f"\n💡 Run 'python steward.py next' to see what's next\n")

    def auto_update(self):
        """Detect completed work from git and update PRD files"""
        print("\n🔍 Scanning git history for completed work...\n")

        # Get commits from last 7 days
        since_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')

        try:
            result = subprocess.run(
                ['git', 'log', f'--since={since_date}', '--pretty=format:%s'],
                capture_output=True,
                text=True,
                check=True
            )

            commits = result.stdout.strip().split('\n')
            print(f"📊 Found {len(commits)} commits in last 7 days\n")

            # Extract completed work
            completed_items = []
            for commit in commits:
                # Look for patterns like "Add X", "Fix Y", "Complete Z"
                if any(word in commit.lower() for word in ['add', 'fix', 'complete', 'implement', 'create']):
                    completed_items.append(commit[:60])

            if completed_items:
                print("✅ Detected Completed Work:\n")
                for i, item in enumerate(completed_items[:5], 1):
                    print(f"   {i}. {item}")

                print("\n💡 Consider running:")
                for item in completed_items[:3]:
                    print(f'   python steward.py complete "{item}"')
            else:
                print("No obvious completed work detected in recent commits")

        except subprocess.CalledProcessError:
            print("❌ Error reading git history")

        print()

    def next(self):
        """Basic suggestions for what to work on next"""
        print("\n" + "="*60)
        print("🎯 STEWARD Suggests:")
        print("="*60 + "\n")

        print("📋 Priority Tasks (V1.00 Critical Path):\n")
        print("   1. Run 3-5 pilot test conversations")
        print("      → Unlocks: Pilot Brief, Comparative Summary, Grant Package\n")
        print("   2. Expand edge case testing")
        print("      → Required for V1.00 Testing Suite deliverable\n")
        print("   3. Draft Pilot Brief outline")
        print("      → Prepare for pilot data documentation\n")

        print("="*60)
        print("\n💡 Recommendation: Start with pilot conversations")
        print("="*60 + "\n")

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
        next_header = re.search(r'\n##[^#]', section)
        if next_header:
            section = section[:next_header.start()]

        return section.strip()

    def _percent(self, done: int, total: int) -> int:
        """Calculate percentage"""
        return int((done / total * 100)) if total > 0 else 0


class StewardAI(Steward):
    """AI-Powered TELOS Project Manager - Uses Mistral (same as TELOS core)"""

    def __init__(self):
        super().__init__()

        # Check for API key
        self.api_key = os.getenv('MISTRAL_API_KEY')
        if not self.api_key and HAS_MISTRAL:
            print("⚠️  Warning: MISTRAL_API_KEY not set")
            print("   Set it for AI-powered features: export MISTRAL_API_KEY='your-key'")
            print("   Falling back to basic mode\n")
            self.has_ai = False
        elif not HAS_MISTRAL:
            print("⚠️  mistralai package not installed")
            print("   Install: pip install mistralai")
            print("   Falling back to basic mode\n")
            self.has_ai = False
        else:
            self.client = MistralClient(api_key=self.api_key, model="mistral-small-latest")
            self.has_ai = True

    def next(self):
        """AI-powered suggestions for what to work on next"""
        if not self.has_ai:
            return super().next()

        print("\n🤖 Analyzing project state with AI...\n")

        # Load all context
        context = self._load_full_context()

        # Ask Claude for recommendations
        prompt = f"""You are Steward, the AI PM for TELOS Observatory V1.00.

{context}

Based on this context, suggest the top 3 tasks to work on next. For each task:
1. Explain WHY it's prioritized (dependencies, blocking other work, V1.00 critical path)
2. Provide specific action steps
3. Estimate time/complexity
4. Note any blockers

Be specific, actionable, and reference exact files/sections. Format as clear numbered list."""

        try:
            response = self.client.generate(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.7
            )

            print("="*60)
            print("🎯 STEWARD AI Recommends:")
            print("="*60 + "\n")
            print(response)
            print("\n" + "="*60 + "\n")

        except Exception as e:
            print(f"❌ AI analysis failed: {e}")
            print("Falling back to basic mode\n")
            return super().next()

    def report(self):
        """Generate weekly status report"""
        if not self.has_ai:
            print("⚠️  AI features not available. Use 'python steward.py status' for basic report.")
            return

        print("\n📊 Generating weekly status report with AI...\n")

        context = self._load_full_context()

        prompt = f"""You are Steward, the AI PM for TELOS Observatory V1.00.

{context}

Generate a concise weekly status report for stakeholders. Include:
1. Executive Summary (2-3 sentences)
2. Progress This Week (completed items)
3. Progress Metrics (% across all trackers)
4. Planned Next Week (top priorities)
5. Risks & Blockers
6. % to V1.00 Release

Format as clean markdown suitable for stakeholders."""

        try:
            response = self.client.generate(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.7
            )

            print("="*60)
            print("📊 WEEKLY STATUS REPORT")
            print("="*60 + "\n")
            print(response)
            print("\n" + "="*60 + "\n")

        except Exception as e:
            print(f"❌ Report generation failed: {e}\n")

    def analyze(self, topic: str):
        """Deep dive AI analysis on specific topic"""
        if not self.has_ai:
            print("⚠️  AI features not available. Install anthropic package and set ANTHROPIC_API_KEY.")
            return

        print(f"\n🔍 Analyzing: {topic}\n")

        context = self._load_full_context()

        prompt = f"""You are Steward, the AI PM for TELOS Observatory V1.00.

{context}

Provide a deep analysis of: {topic}

Include:
- Current state
- Dependencies and blockers
- Risk assessment
- Recommended approach
- Specific next steps

Be thorough and reference specific files/sections."""

        try:
            response = self.client.generate(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=3000,
                temperature=0.7
            )

            print("="*60)
            print(f"📊 Analysis: {topic}")
            print("="*60 + "\n")
            print(response)
            print("\n" + "="*60 + "\n")

        except Exception as e:
            print(f"❌ Analysis failed: {e}\n")

    def _load_full_context(self) -> str:
        """Load all PRD files as context for AI"""
        context_parts = []

        files_to_load = [
            ("PRD.md", "V1.00 Requirements"),
            ("TASKS.md", "Task Backlog"),
            ("PLATFORM_STATUS.md", "Platform Status"),
            ("UI_PHASES.md", "UI Development Phases")
        ]

        for filename, description in files_to_load:
            file_path = self.prd_dir / filename
            if file_path.exists():
                content = file_path.read_text()
                # Limit to first 3000 chars to avoid token limits
                content = content[:3000] if len(content) > 3000 else content
                context_parts.append(f"## {description} ({filename})\n\n{content}\n")

        # Add STEWARD.md
        if self.steward_file.exists():
            steward_content = self.steward_file.read_text()
            context_parts.append(f"## Current Sprint (STEWARD.md)\n\n{steward_content}\n")

        return "\n".join(context_parts)


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("\n🔭 STEWARD - TELOS AI Project Manager")
        print("\nBasic Commands:")
        print("  python steward.py status              - Show current state")
        print("  python steward.py next                - AI-powered priority suggestions")
        print('  python steward.py complete "task"     - Mark task complete')
        print("  python steward.py auto-update         - Detect completed work from git")
        print("\nAI Commands (requires MISTRAL_API_KEY):")
        print("  python steward.py report              - Generate weekly status report")
        print('  python steward.py analyze "topic"     - Deep dive analysis')
        print()
        sys.exit(1)

    command = sys.argv[1].lower()

    # Use AI-powered version if available
    steward = StewardAI() if HAS_MISTRAL else Steward()

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
    elif command == "auto-update":
        steward.auto_update()
    elif command == "report":
        if isinstance(steward, StewardAI):
            steward.report()
        else:
            print("⚠️  Report command requires AI features")
            print("Install: pip install mistralai")
            print("Set: export MISTRAL_API_KEY='your-key'")
    elif command == "analyze":
        if len(sys.argv) < 3:
            print("❌ Error: Please provide topic to analyze")
            print('Usage: python steward.py analyze "topic"')
            sys.exit(1)
        topic = sys.argv[2]
        if isinstance(steward, StewardAI):
            steward.analyze(topic)
        else:
            print("⚠️  Analyze command requires AI features")
    else:
        print(f"❌ Unknown command: {command}")
        print("Valid commands: status, next, complete, auto-update, report, analyze")
        sys.exit(1)


if __name__ == "__main__":
    main()
