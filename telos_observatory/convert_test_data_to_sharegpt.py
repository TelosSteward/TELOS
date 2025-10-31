#!/usr/bin/env python3
"""
Convert TELOS test_data formats to ShareGPT format for Phase 2 studies.

Converts test_sessions and edge_cases JSON files to ShareGPT format:
{
  "id": "session_id",
  "turns": [
    ["user message", "assistant message"],
    ...
  ]
}
"""

import json
from pathlib import Path
from typing import List, Dict, Any


def convert_test_session_to_sharegpt(session_file: Path) -> Dict[str, Any]:
    """
    Convert test_session format to ShareGPT format.

    Args:
        session_file: Path to test session JSON file

    Returns:
        ShareGPT-formatted conversation dict
    """
    with open(session_file) as f:
        session_data = json.load(f)

    # Extract conversation ID
    conv_id = session_data.get('session_id', session_file.stem)

    # Extract turns (user_input, assistant_response)
    turns = []
    for turn in session_data.get('turns', []):
        user_msg = turn.get('user_input') or turn.get('user_message', '')
        assistant_msg = turn.get('assistant_response', '')

        # Only include turns with both user and assistant content
        if user_msg and assistant_msg:
            turns.append([user_msg, assistant_msg])

    return {
        'id': conv_id,
        'turns': turns
    }


def convert_directory_to_sharegpt(
    input_dir: Path,
    output_file: Path,
    file_pattern: str = '*.json'
) -> List[Dict[str, Any]]:
    """
    Convert all JSON files in directory to ShareGPT format.

    Args:
        input_dir: Directory containing test session files
        output_file: Where to save converted conversations
        file_pattern: Glob pattern for files to convert

    Returns:
        List of converted conversation dicts
    """
    conversations = []

    # Find all matching JSON files
    json_files = sorted(input_dir.glob(file_pattern))

    print(f"Found {len(json_files)} files in {input_dir}")

    for json_file in json_files:
        # Skip summary files
        if 'summary' in json_file.name.lower():
            print(f"  Skipping summary file: {json_file.name}")
            continue

        try:
            conv = convert_test_session_to_sharegpt(json_file)

            # Only include if it has actual turns
            if conv['turns']:
                conversations.append(conv)
                print(f"  ✓ Converted {json_file.name}: {len(conv['turns'])} turns")
            else:
                print(f"  ⚠ Skipped {json_file.name}: no valid turns")

        except Exception as e:
            print(f"  ✗ Error converting {json_file.name}: {e}")

    # Save to output file
    with open(output_file, 'w') as f:
        json.dump(conversations, f, indent=2)

    print(f"\n✓ Saved {len(conversations)} conversations to {output_file}")

    return conversations


def main():
    """Convert all test data to ShareGPT format."""

    # Paths
    test_data_dir = Path(__file__).parent.parent / 'telos_purpose' / 'test_data'
    output_dir = Path(__file__).parent / 'test_data_converted'
    output_dir.mkdir(exist_ok=True, parents=True)

    print("=" * 70)
    print("TELOS TEST DATA → SHAREGPT FORMAT CONVERTER")
    print("=" * 70)

    # Convert test_sessions
    if (test_data_dir / 'test_sessions').exists():
        print("\n📁 Converting test_sessions...")
        convert_directory_to_sharegpt(
            input_dir=test_data_dir / 'test_sessions',
            output_file=output_dir / 'test_sessions_sharegpt.json'
        )

    # Convert edge_cases
    if (test_data_dir / 'edge_cases').exists():
        print("\n📁 Converting edge_cases...")
        convert_directory_to_sharegpt(
            input_dir=test_data_dir / 'edge_cases',
            output_file=output_dir / 'edge_cases_sharegpt.json'
        )

    print("\n" + "=" * 70)
    print("CONVERSION COMPLETE")
    print("=" * 70)
    print(f"\nConverted files saved to: {output_dir}")
    print()


if __name__ == '__main__':
    main()
