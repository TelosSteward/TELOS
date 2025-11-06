#!/usr/bin/env python3
"""
Concatenate TELOS Essential Files for Opus Audit

Combines all 75 essential .py files into a single TELOS_COMPLETE.py file
for efficient one-shot Opus audit.

Usage:
    python3 concatenate_for_opus.py
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# Project root
PROJECT_ROOT = Path(__file__).parent.resolve()

# Check both locations for opus_review_package
OPUS_PACKAGE_LOCAL = PROJECT_ROOT / "opus_review_package"
OPUS_PACKAGE_DESKTOP = Path.home() / "Desktop" / "opus_review_package"

# Use whichever exists
if OPUS_PACKAGE_LOCAL.exists():
    OPUS_PACKAGE = OPUS_PACKAGE_LOCAL
elif OPUS_PACKAGE_DESKTOP.exists():
    OPUS_PACKAGE = OPUS_PACKAGE_DESKTOP
else:
    OPUS_PACKAGE = None

OUTPUT_FILE = PROJECT_ROOT / "TELOS_COMPLETE.py"

# Header template for each file
FILE_HEADER_TEMPLATE = """
{'='*80}
FILE: {file_path}
{'='*80}

"""


def collect_python_files(package_dir: Path) -> list[Path]:
    """Collect all .py files from opus_review_package in sorted order."""
    py_files = []

    # Walk directory structure
    for root, dirs, files in os.walk(package_dir):
        # Skip __pycache__ and .git
        dirs[:] = [d for d in dirs if d not in ['__pycache__', '.git', '.pytest_cache']]

        for file in sorted(files):
            if file.endswith('.py'):
                file_path = Path(root) / file
                py_files.append(file_path)

    return sorted(py_files)


def create_mega_file(py_files: list[Path], output_path: Path):
    """Concatenate all Python files into single mega-file."""

    with open(output_path, 'w', encoding='utf-8') as out:
        # Write header
        out.write(f'''"""
TELOS COMPLETE CODEBASE - OPUS AUDIT VERSION

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total Files: {len(py_files)}
Purpose: Single-file comprehensive audit for Claude Opus

This file concatenates all 75 essential TELOS Python files for efficient review.
Each file is clearly separated with headers showing the original file path.

AUDIT FOCUS AREAS:
1. Mathematical Correctness - Dual PA, fidelity calculations, intervention logic
2. Security & Privacy - Telemetric Keys cryptographic implementation
3. Edge Cases & Error Handling - Robustness improvements
4. Architecture & Design - Separation of concerns, maintainability
5. Production Readiness - Performance bottlenecks, hardening needs

WORKFLOW:
1. Opus reviews this complete file and generates findings report
2. Sonnet (Claude Code) implements fixes based on Opus recommendations
3. Each fix = separate commit for easy rollback if needed

ROLLBACK SAFETY:
- Git branch 'pre-opus-audit' = snapshot before ANY changes
- Git branch 'post-opus-audit' = all Opus-recommended fixes
- Individual commits per fix = granular rollback capability

"""

{"="*80}
TELOS COMPLETE CODEBASE - {len(py_files)} FILES
{"="*80}

''')

        # Process each file
        for idx, py_file in enumerate(py_files, 1):
            rel_path = py_file.relative_to(OPUS_PACKAGE)

            print(f"  [{idx}/{len(py_files)}] {rel_path}")

            # Write file header
            out.write(f"\n{'='*80}\n")
            out.write(f"FILE {idx}/{len(py_files)}: {rel_path}\n")
            out.write(f"{'='*80}\n\n")

            # Write file contents
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    out.write(content)

                    # Ensure file ends with newline
                    if content and not content.endswith('\n'):
                        out.write('\n')

            except Exception as e:
                out.write(f"# ERROR READING FILE: {e}\n")
                print(f"    ⚠️  Error reading {rel_path}: {e}")

            # Add spacing between files
            out.write("\n\n")


def main():
    """Main execution."""
    print("\n" + "="*80)
    print("🔗 TELOS Concatenation for Opus Audit")
    print("="*80 + "\n")

    # Check opus_review_package exists
    if OPUS_PACKAGE is None or not OPUS_PACKAGE.exists():
        print(f"❌ Error: opus_review_package not found")
        print("   Checked locations:")
        print(f"     - {OPUS_PACKAGE_LOCAL}")
        print(f"     - {OPUS_PACKAGE_DESKTOP}")
        print("   Run extract_essential_telos.py first or move package to one of these locations")
        return 1

    # Collect Python files
    print(f"📂 Scanning {OPUS_PACKAGE}...")
    py_files = collect_python_files(OPUS_PACKAGE)

    if not py_files:
        print("❌ No Python files found")
        return 1

    print(f"✅ Found {len(py_files)} Python files\n")

    # Create mega-file
    print(f"📝 Creating {OUTPUT_FILE.name}...\n")
    create_mega_file(py_files, OUTPUT_FILE)

    # Calculate size
    size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)

    print("\n" + "="*80)
    print(f"✅ Successfully created {OUTPUT_FILE.name}")
    print(f"📊 File size: {size_mb:.2f} MB")
    print(f"📁 Location: {OUTPUT_FILE}")
    print(f"📋 Contains: {len(py_files)} Python files")
    print("="*80 + "\n")

    print("🎯 Next steps:")
    print("  1. Review TELOS_COMPLETE.py")
    print("  2. Send to Opus with OPUS_AUDIT_BRIEF.md")
    print("  3. Receive Opus findings report")
    print("  4. Sonnet implements fixes with individual commits")
    print("  5. Test and verify or rollback to pre-opus-audit branch\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
