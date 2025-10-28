#!/usr/bin/env python3
"""
TELOS Directory Organization Script
Organizes 174 files into proper directory structure
"""

import os
import shutil
from pathlib import Path
from datetime import datetime

# Base directory
BASE_DIR = Path("/Users/brunnerjf/Desktop/telos")

# Define organization rules
ORGANIZATION_RULES = {
    # Guides
    'docs/guides': [
        'COUNTERFACTUAL_BRANCHING_GUIDE.md',
        'PROGRESSIVE_ATTRACTOR_GUIDE.md',
        'QUICK_START_GUIDE.md',
        'SCREENCASTING_GUIDE.md',
        'INSTALLATION_SUCCESS.md',
        'SESSIONLOADER_FLEXIBLE_FORMATS.md',
        'PHASE1_INTEGRATION_GUIDE.md',
        'PREFLIGHT_CHECK_REPORT.md',
        'WIRING_VERIFICATION.md',
    ],

    # Architecture
    'docs/architecture': [
        'ARCHITECTURE_DIAGNOSIS.md',
    ],

    # Implementation summaries
    'docs/implementation': [
        'PHASE1_IMPLEMENTATION_SUMMARY.md',
        'COUNTERFACTUAL_SIMULATOR_SUMMARY.md',
        'DASHBOARD_ACTIVE_MITIGATION_ASSESSMENT.md',
        'DASHBOARD_COMPLETE.md',
        'DASHBOARD_SIMULATOR_INTEGRATION.md',
        'REORGANIZATION_SUMMARY.md',
        'TELOSCOPE_BUILD_SUMMARY.md',
        'TELOSCOPE_COMPLETE.md',
        'TELOSCOPE_DEPLOYMENT_READY.md',
        'TELOSCOPE_IMPLEMENTATION_STATUS.md',
        'TELOSCOPE_STREAMLIT_COMPLETE.md',
    ],

    # Keep in root
    'keep_in_root': [
        'README.md',
        'README_TELOSCOPE.md',
        'TASKS.md',
        'launch_dashboard.sh',
        'requirements.txt',
        '.gitignore',
    ],
}

def get_all_files_in_root():
    """Get all files in root directory (not subdirectories)."""
    return [f for f in BASE_DIR.iterdir() if f.is_file() and not f.name.startswith('.')]

def categorize_file(filename):
    """Determine where a file should go."""
    name = filename.name

    # Check explicit rules
    for dest, files in ORGANIZATION_RULES.items():
        if name in files:
            return dest

    # Pattern-based categorization
    if 'FIX' in name.upper() or 'FIXED' in name.upper():
        return 'docs/fixes'

    if 'GUIDE' in name.upper():
        return 'docs/guides'

    if 'SUMMARY' in name.upper() or 'COMPLETE' in name.upper() or 'ASSESSMENT' in name.upper():
        return 'docs/implementation'

    if 'ARCHITECTURE' in name.upper():
        return 'docs/architecture'

    # Duplicates (files with numbers or .txt versions)
    if '(' in name and ')' in name:  # e.g., file(1).txt
        return 'archive/duplicates'

    if name.endswith('.py.txt') or name.endswith('.md.txt') or name.endswith('.json.txt'):
        return 'archive/txt-versions'

    # Config files
    if name.startswith('config') and name.endswith('.json'):
        return 'config'

    # Scripts
    if name.endswith('.sh') and name != 'launch_dashboard.sh':
        return 'scripts'

    # Unknown - needs manual review
    return 'archive'

def organize_files(dry_run=False):
    """Organize all files according to rules."""
    moves = []

    files = get_all_files_in_root()

    for file in files:
        # Skip if in telos_purpose, venv, etc
        if file.is_dir():
            continue

        # Skip if already in docs/, config/, etc
        if str(file).startswith(str(BASE_DIR / 'docs')) or \
           str(file).startswith(str(BASE_DIR / 'config')) or \
           str(file).startswith(str(BASE_DIR / 'archive')):
            continue

        dest_cat = categorize_file(file)

        if dest_cat == 'keep_in_root':
            continue

        dest_dir = BASE_DIR / dest_cat
        dest_path = dest_dir / file.name

        moves.append({
            'source': file,
            'dest': dest_path,
            'category': dest_cat
        })

    # Execute moves
    if not dry_run:
        for move in moves:
            move['dest'].parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.move(str(move['source']), str(move['dest']))
                print(f"✅ {move['source'].name} → {move['category']}")
            except Exception as e:
                print(f"❌ Error moving {move['source'].name}: {e}")

    return moves

def generate_report(moves):
    """Generate organization report."""
    report = []
    report.append("# TELOS Directory Organization Report")
    report.append(f"\n**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"\n**Total Files Organized**: {len(moves)}")

    # Group by category
    by_category = {}
    for move in moves:
        cat = move['category']
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(move)

    report.append("\n\n## Files Moved by Category\n")

    for cat in sorted(by_category.keys()):
        report.append(f"\n### {cat} ({len(by_category[cat])} files)\n")
        for move in by_category[cat]:
            report.append(f"- `{move['source'].name}`")

    report.append("\n\n## Directory Structure")
    report.append("\n```")
    report.append("telos/")
    report.append("├── telos_purpose/          # Core code")
    report.append("├── docs/")
    report.append("│   ├── architecture/       # Architecture documentation")
    report.append("│   ├── fixes/             # Bug fix reports")
    report.append("│   ├── guides/            # User & deployment guides")
    report.append("│   ├── implementation/    # Implementation summaries")
    report.append("│   └── archive/           # Archived documentation")
    report.append("├── config/                 # Configuration files")
    report.append("├── scripts/                # Utility scripts")
    report.append("├── tests/                  # Test files")
    report.append("├── archive/")
    report.append("│   ├── duplicates/        # Duplicate files")
    report.append("│   └── txt-versions/      # .txt versions of code files")
    report.append("├── venv/                   # Virtual environment")
    report.append("├── requirements.txt")
    report.append("├── README.md")
    report.append("├── launch_dashboard.sh")
    report.append("└── .gitignore")
    report.append("```")

    return "\n".join(report)

if __name__ == "__main__":
    print("=" * 60)
    print("TELOS Directory Organization")
    print("=" * 60)
    print()

    # Organize files
    moves = organize_files(dry_run=False)

    print()
    print("=" * 60)
    print(f"✅ Organized {len(moves)} files")
    print("=" * 60)

    # Generate report
    report = generate_report(moves)

    # Save report
    report_path = BASE_DIR / "ORGANIZATION_COMPLETE.md"
    report_path.write_text(report)

    print(f"\n📄 Report saved to: {report_path}")
