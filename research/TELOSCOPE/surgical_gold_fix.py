#!/usr/bin/env python3
"""
Surgical Gold Color Replacement
================================
ONLY replaces #F4D03F (old bright gold) with #F4D03F (refined gold)
Does NOT touch any other colors to preserve:
- Green borders on observation deck
- Color-coded fidelity values
- Font sizes
- Any other styling
"""

import re
from pathlib import Path

# Define the ONLY change we're making
OLD_GOLD = '#F4D03F'
NEW_GOLD = '#F4D03F'

# Files to update
FILES_TO_UPDATE = [
    'main.py',
    'components/conversation_display.py',
    'components/teloscope_controls.py',
    'components/observatory_lens.py',
    'components/observation_deck.py',
    'components/beta_review.py',
    'components/beta_pa_establishment.py',
    'components/beta_onboarding.py',
    'components/control_strip.py',
    'components/steward_panel.py',
    'components/beta_completion.py',
    'components/sidebar_actions_beta.py',
    'components/sidebar_actions.py',
]

def replace_gold_in_file(filepath: Path):
    """Replace ONLY the gold color, nothing else."""
    if not filepath.exists():
        print(f"⚠️  Skipping {filepath} - file not found")
        return

    # Read the file
    content = filepath.read_text()

    # Count occurrences
    count = content.count(OLD_GOLD)

    if count == 0:
        print(f"✓ {filepath.name} - no changes needed")
        return

    # Replace ONLY #F4D03F with #F4D03F (case-insensitive)
    new_content = re.sub(
        r'#F4D03F',
        NEW_GOLD,
        content,
        flags=re.IGNORECASE
    )

    # Write back
    filepath.write_text(new_content)
    print(f"✅ {filepath.name} - replaced {count} occurrence(s)")

def main():
    base_dir = Path('/Users/brunnerjf/Desktop/Privacy_PreCommit/TELOSCOPE_BETA')

    print("🔧 Surgical Gold Color Replacement")
    print(f"   Old: {OLD_GOLD} → New: {NEW_GOLD}")
    print("=" * 60)

    for file_path in FILES_TO_UPDATE:
        full_path = base_dir / file_path
        replace_gold_in_file(full_path)

    print("=" * 60)
    print("✅ Surgical replacement complete!")
    print("\nThis script ONLY changed the gold color.")
    print("All other colors, borders, fonts remain untouched.")

if __name__ == '__main__':
    main()
