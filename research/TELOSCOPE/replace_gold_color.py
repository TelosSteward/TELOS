#!/usr/bin/env python3
"""
Replace #FFD700 (old bright gold) with #F4D03F (refined gold) throughout codebase.
ONLY affects .py files, not documentation.
"""

import re
from pathlib import Path

OLD_GOLD = '#FFD700'
NEW_GOLD = '#F4D03F'

# Files to process
ROOT = Path(__file__).parent
PYTHON_FILES = list(ROOT.glob('**/*.py'))

# Exclude this script and virtual env
PYTHON_FILES = [f for f in PYTHON_FILES if f.name != 'replace_gold_color.py' and 'venv' not in str(f)]

def replace_in_file(file_path):
    """Replace color in a single file."""
    with open(file_path, 'r') as f:
        content = f.read()

    # Count occurrences
    count = content.count(OLD_GOLD)

    if count == 0:
        return 0

    # Replace (case-insensitive)
    new_content = re.sub(
        r'#FFD700',
        NEW_GOLD,
        content,
        flags=re.IGNORECASE
    )

    # Write back
    with open(file_path, 'w') as f:
        f.write(new_content)

    return count

# Process all files
total_replacements = 0
files_modified = []

for py_file in PYTHON_FILES:
    count = replace_in_file(py_file)
    if count > 0:
        total_replacements += count
        files_modified.append((py_file.relative_to(ROOT), count))
        print(f"✓ {py_file.relative_to(ROOT)}: {count} replacements")

print(f"\n{'='*60}")
print(f"Total: {total_replacements} replacements in {len(files_modified)} files")
print(f"{'='*60}")

print("\nModified files:")
for file, count in files_modified:
    print(f"  {file}: {count}")
