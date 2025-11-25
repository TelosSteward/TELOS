#!/usr/bin/env python3
"""
TELOS Documentation Reorganization Script

This script reorganizes the TELOS repository documentation structure
from a cluttered root directory to a professional, hierarchical organization.

IMPORTANT: This script preserves all files - nothing is deleted, only moved.
A complete migration log is generated for audit purposes.

Usage:
    python scripts/reorganize_docs.py --dry-run  # Preview changes
    python scripts/reorganize_docs.py            # Execute reorganization
"""

import os
import shutil
from pathlib import Path
from datetime import datetime
import json

# Base repository path
REPO_ROOT = Path(__file__).parent.parent

# Migration plan: (source_file, destination_path, category)
MIGRATION_PLAN = [
    # === ARCHIVED SESSION NOTES ===
    ("NEXT_SESSION_HANDOFF.md", "docs/archived/session-notes/", "session"),
    ("NEXT_SESSION_START_HERE.md", "docs/archived/session-notes/", "session"),
    ("SESSION_SUMMARY_2025-11-08.md", "docs/archived/session-notes/", "session"),
    ("QUICK_START_NEXT_SESSION.md", "docs/archived/session-notes/", "session"),

    # === ARCHIVED PLANNING ===
    ("BETA_TESTING_DECISION_LOG.md", "docs/archived/planning/", "planning"),
    ("BUTTON_HOVER_EXPANSION_NOTE.md", "docs/archived/planning/", "planning"),
    ("SEQUENTIAL_ANALYSIS_ENHANCED_OPTION_B+.md", "docs/archived/planning/", "planning"),
    ("PLAYWRIGHT_MCP_SETUP_AND_TESTING.md", "docs/archived/planning/", "planning"),
    ("SETUP_NOTES.md", "docs/archived/planning/", "planning"),
    ("BUILD_TAG_v0.1.0-beta-testing.md", "docs/archived/planning/", "planning"),
    ("STEWARD_UNIFIED_README.md", "docs/archived/planning/", "planning"),

    # === IMPLEMENTATION DOCUMENTATION ===
    ("BETA_IMPLEMENTATION_PLAN.md", "docs/implementation/beta-testing/", "implementation"),
    ("BETA_TESTING_AUTOMATION_REPORT.md", "docs/implementation/beta-testing/", "implementation"),
    ("BETA_TESTING_IMPLEMENTATION_COMPLETE.md", "docs/implementation/beta-testing/", "implementation"),
    ("DEMO_MODE_DESIGN_PLAN.md", "docs/implementation/features/", "implementation"),
    ("DEMO_MODE_STATUS.md", "docs/implementation/features/", "implementation"),

    # === VALIDATION REPORTS ===
    ("VALIDATION_AUDIT_REPORT.md", "docs/validation/", "validation"),
    ("VALIDATION_STATUS_REPORT.md", "docs/validation/", "validation"),

    # === GETTING STARTED GUIDES ===
    ("QUICK_START_GUIDE.md", "docs/getting-started/", "user-docs"),
    ("DEPLOYMENT_GUIDE.md", "docs/getting-started/", "user-docs"),
    ("EXECUTIVE_SUMMARY.md", "docs/", "user-docs"),

    # === NEXT VERSION PLAN (special case - large file) ===
    ("NEXT_VERSION_PLAN.md", "docs/archived/planning/", "planning"),
]

# Files to keep at root (these are professional/essential)
KEEP_AT_ROOT = [
    "README.md",
    "COPYRIGHT.md",
]

# Directories to create
DIRECTORIES_TO_CREATE = [
    "docs/getting-started",
    "docs/implementation/beta-testing",
    "docs/implementation/features",
    "docs/validation",
    "docs/architecture",
    "docs/archived/session-notes",
    "docs/archived/planning",
    "docs/archived/test-results",
]

# README content for new directories
DIRECTORY_READMES = {
    "docs/": """# TELOS Documentation

Complete documentation for the TELOS (Telically Entrained Linguistic Operational Substrate) project.

## Quick Links

- **[Getting Started](getting-started/)** - Installation, quick start, deployment
- **[Implementation](implementation/)** - Implementation details and feature plans
- **[Validation](validation/)** - Validation reports and test results
- **[Archived](archived/)** - Historical planning documents and session notes

## Main Documentation

- **[Executive Summary](EXECUTIVE_SUMMARY.md)** - Project overview and beta testing summary

## Navigation

- Start with the [Quick Start Guide](getting-started/QUICK_START_GUIDE.md) for immediate setup
- See [Deployment Guide](getting-started/DEPLOYMENT_GUIDE.md) for production deployment
- Review [Validation Reports](validation/) for empirical results
""",

    "docs/getting-started/": """# Getting Started with TELOS

This directory contains guides for getting up and running with TELOS.

## Available Guides

- **[Quick Start Guide](QUICK_START_GUIDE.md)** - Beta testing integration quick start
- **[Deployment Guide](DEPLOYMENT_GUIDE.md)** - Deploy to Streamlit Cloud

## From the Main Docs

See also the main [README](../../README.md) for mathematical foundations and project overview.
""",

    "docs/implementation/": """# Implementation Documentation

Detailed documentation about TELOS implementation, features, and beta testing.

## Contents

- **[Beta Testing](beta-testing/)** - Beta testing implementation plans and reports
- **[Features](features/)** - Feature design and status documentation
""",

    "docs/implementation/beta-testing/": """# Beta Testing Implementation

Complete documentation of the TELOS beta testing implementation.

## Files

- **BETA_IMPLEMENTATION_PLAN.md** - Complete implementation plan
- **BETA_TESTING_AUTOMATION_REPORT.md** - Automated testing results
- **BETA_TESTING_IMPLEMENTATION_COMPLETE.md** - Implementation completion status
""",

    "docs/implementation/features/": """# Feature Documentation

Design documents and status reports for TELOS features.

## Files

- **DEMO_MODE_DESIGN_PLAN.md** - Demo mode design and architecture
- **DEMO_MODE_STATUS.md** - Current demo mode implementation status
""",

    "docs/validation/": """# Validation Reports

Empirical validation reports and test results for TELOS governance.

## Reports

- **VALIDATION_AUDIT_REPORT.md** - Comprehensive validation audit
- **VALIDATION_STATUS_REPORT.md** - Current validation status

## Additional Validation Data

See also:
- `planning_output/TELOS_UNIFIED_VALIDATION_REPORT.md` - Unified validation report
- `planning_output/SCALABILITY_VALIDATION_RESULTS.md` - Scalability results
- `planning_output/defense_validation_report.md` - Defense validation
""",

    "docs/archived/": """# Archived Documentation

Historical planning documents, session notes, and completed implementation logs.

**Note:** These files are preserved for historical reference but are no longer actively maintained.

## Contents

- **[Session Notes](session-notes/)** - Development session summaries and handoffs
- **[Planning](planning/)** - Historical planning documents and decision logs
- **[Test Results](test-results/)** - Archived test outputs (to be populated during cleanup)
""",

    "docs/archived/session-notes/": """# Archived Session Notes

Development session summaries, handoffs, and progress notes from active development.

**Status:** Historical archive - preserved for reference

These documents capture the development process but are no longer actively maintained.
""",

    "docs/archived/planning/": """# Archived Planning Documents

Historical planning documents, decision logs, and implementation notes.

**Status:** Historical archive - preserved for reference

These documents show the evolution of TELOS but decisions have been implemented
and these files are no longer actively updated.
""",
}


class DocumentationReorganizer:
    """Handles the reorganization of TELOS documentation."""

    def __init__(self, dry_run=False):
        self.dry_run = dry_run
        self.migration_log = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def log_action(self, action, source, destination=None, status="success"):
        """Log a migration action."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "source": str(source),
            "destination": str(destination) if destination else None,
            "status": status,
        }
        self.migration_log.append(entry)

        # Print to console
        if self.dry_run:
            prefix = "[DRY RUN]"
        else:
            prefix = "✓" if status == "success" else "✗"

        if destination:
            print(f"{prefix} {action}: {source} -> {destination}")
        else:
            print(f"{prefix} {action}: {source}")

    def create_directory(self, path):
        """Create a directory if it doesn't exist."""
        full_path = REPO_ROOT / path

        if self.dry_run:
            self.log_action("CREATE_DIR", path)
            return

        if not full_path.exists():
            full_path.mkdir(parents=True, exist_ok=True)
            self.log_action("CREATE_DIR", path)
        else:
            self.log_action("SKIP_DIR", path, status="already_exists")

    def create_readme(self, directory, content):
        """Create a README.md in the specified directory."""
        readme_path = REPO_ROOT / directory / "README.md"

        if self.dry_run:
            self.log_action("CREATE_README", f"{directory}/README.md")
            return

        if not readme_path.exists():
            readme_path.write_text(content)
            self.log_action("CREATE_README", f"{directory}/README.md")
        else:
            self.log_action("SKIP_README", f"{directory}/README.md", status="already_exists")

    def move_file(self, source, destination_dir):
        """Move a file from root to destination directory."""
        source_path = REPO_ROOT / source
        dest_dir_path = REPO_ROOT / destination_dir
        dest_path = dest_dir_path / source

        if not source_path.exists():
            self.log_action("SKIP_FILE", source, destination_dir, status="not_found")
            return False

        if self.dry_run:
            self.log_action("MOVE_FILE", source, destination_dir)
            return True

        # Ensure destination directory exists
        dest_dir_path.mkdir(parents=True, exist_ok=True)

        # Move file
        shutil.move(str(source_path), str(dest_path))
        self.log_action("MOVE_FILE", source, destination_dir)
        return True

    def save_migration_log(self):
        """Save the migration log to a file."""
        log_filename = f"docs/MIGRATION_LOG_{self.timestamp}.md"
        log_path = REPO_ROOT / log_filename

        if self.dry_run:
            print(f"\n[DRY RUN] Would save migration log to: {log_filename}")
            return

        # Generate markdown report
        report = f"""# TELOS Documentation Reorganization Log

**Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Total Actions:** {len(self.migration_log)}

## Summary

This log documents the reorganization of TELOS documentation from a cluttered
root directory structure to a professional, hierarchical organization.

### Actions Performed

"""

        # Count actions by type
        action_counts = {}
        for entry in self.migration_log:
            action = entry['action']
            action_counts[action] = action_counts.get(action, 0) + 1

        for action, count in sorted(action_counts.items()):
            report += f"- **{action}**: {count}\n"

        report += "\n## Detailed Log\n\n"
        report += "| Timestamp | Action | Source | Destination | Status |\n"
        report += "|-----------|--------|--------|-------------|--------|\n"

        for entry in self.migration_log:
            timestamp = entry['timestamp'].split('T')[1][:8]  # Just time
            action = entry['action']
            source = entry['source']
            dest = entry['destination'] or '-'
            status = entry['status']
            report += f"| {timestamp} | {action} | `{source}` | `{dest}` | {status} |\n"

        report += "\n## File Locations After Reorganization\n\n"
        report += "### Root Directory (Clean)\n"
        report += "- README.md\n"
        report += "- COPYRIGHT.md\n\n"

        report += "### Documentation Structure\n"
        report += "```\n"
        report += "docs/\n"
        report += "├── EXECUTIVE_SUMMARY.md\n"
        report += "├── getting-started/\n"
        report += "│   ├── QUICK_START_GUIDE.md\n"
        report += "│   └── DEPLOYMENT_GUIDE.md\n"
        report += "├── implementation/\n"
        report += "│   ├── beta-testing/\n"
        report += "│   └── features/\n"
        report += "├── validation/\n"
        report += "└── archived/\n"
        report += "    ├── session-notes/\n"
        report += "    └── planning/\n"
        report += "```\n\n"

        report += "## Notes\n\n"
        report += "- All files were preserved - nothing was deleted\n"
        report += "- All content remains accessible in new locations\n"
        report += "- Internal documentation links may need updating\n"
        report += "- This reorganization improves professional appearance for grants/collaboration\n"

        # Save report
        log_path.write_text(report)
        print(f"\n✓ Migration log saved to: {log_filename}")

        # Also save JSON version
        json_log_path = REPO_ROOT / f"docs/migration_log_{self.timestamp}.json"
        with open(json_log_path, 'w') as f:
            json.dump(self.migration_log, f, indent=2)
        print(f"✓ JSON log saved to: docs/migration_log_{self.timestamp}.json")

    def reorganize(self):
        """Execute the full reorganization."""
        print("=" * 70)
        if self.dry_run:
            print("DRY RUN MODE - No files will be modified")
        else:
            print("EXECUTING TELOS Documentation Reorganization")
        print("=" * 70)
        print()

        # Step 1: Create directory structure
        print("Step 1: Creating directory structure...")
        for directory in DIRECTORIES_TO_CREATE:
            self.create_directory(directory)
        print()

        # Step 2: Create README files
        print("Step 2: Creating directory README files...")
        for directory, content in DIRECTORY_READMES.items():
            self.create_readme(directory, content)
        print()

        # Step 3: Move files according to migration plan
        print("Step 3: Moving files to new locations...")
        moved_count = 0
        for source, destination, category in MIGRATION_PLAN:
            if self.move_file(source, destination):
                moved_count += 1
        print(f"\nMoved {moved_count} files")
        print()

        # Step 4: Save migration log
        print("Step 4: Saving migration log...")
        self.save_migration_log()
        print()

        # Summary
        print("=" * 70)
        print("Reorganization Complete!")
        print("=" * 70)
        print()
        print("Summary:")
        print(f"  - Directories created: {len(DIRECTORIES_TO_CREATE)}")
        print(f"  - README files added: {len(DIRECTORY_READMES)}")
        print(f"  - Files moved: {moved_count}")
        print(f"  - Files kept at root: {len(KEEP_AT_ROOT)}")
        print()

        if self.dry_run:
            print("This was a DRY RUN. To execute, run without --dry-run flag.")
        else:
            print("Next steps:")
            print("  1. Review the migration log in docs/")
            print("  2. Test that documentation is accessible")
            print("  3. Update any internal links if needed")
            print("  4. Commit changes: git add . && git commit -m 'Reorganize documentation structure'")
        print()


def main():
    """Main entry point."""
    import sys

    dry_run = "--dry-run" in sys.argv or "-n" in sys.argv

    reorganizer = DocumentationReorganizer(dry_run=dry_run)
    reorganizer.reorganize()


if __name__ == "__main__":
    main()
