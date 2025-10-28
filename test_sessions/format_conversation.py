#!/usr/bin/env python3
"""
Auto-formatter for Claude Conversation exports
Adds turn markers assuming alternating Human/Claude turns
"""

import re
from pathlib import Path

def clean_line(line):
    """Remove file upload noise and UI elements."""
    # Remove file metadata patterns (be more specific)
    line = re.sub(r'[A-Za-z_]+\.(?:txt|py|md|json)\d+:\d+\.txt\d+\s+lines?\s*', '', line)

    # Remove standalone format markers
    if line.strip() in ['TXT', 'PDF', 'Edit', 'Retry']:
        return ''

    # Remove excessive whitespace
    line = re.sub(r'\s+', ' ', line)

    return line.strip()

def format_conversation(input_path, output_path):
    """Format Claude conversation with turn markers."""

    # Read file with proper encoding handling
    with open(input_path, 'rb') as f:
        content = f.read()

    # Try to decode - handle both UTF-8 and ISO-8859
    try:
        text = content.decode('utf-8')
    except UnicodeDecodeError:
        text = content.decode('iso-8859-1')

    # Convert CR to LF
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # Split into lines
    lines = text.split('\n')

    # First pass: detect paragraph boundaries BEFORE cleaning
    # Empty lines indicate paragraph breaks
    paragraphs = []
    current_para = []

    for line in lines:
        stripped = line.strip()

        # Empty line = paragraph boundary
        if not stripped:
            if current_para:
                # Join and clean the paragraph
                para_text = ' '.join(current_para)
                cleaned = clean_line(para_text)
                if cleaned and len(cleaned) > 30:  # Minimum substantive length
                    paragraphs.append(cleaned)
                current_para = []
        else:
            current_para.append(stripped)

    # Add last paragraph
    if current_para:
        para_text = ' '.join(current_para)
        cleaned = clean_line(para_text)
        if cleaned and len(cleaned) > 30:
            paragraphs.append(cleaned)

    print(f"[DEBUG] Raw paragraphs found: {len(paragraphs)}")
    if paragraphs:
        print(f"[DEBUG] First 3 paragraphs:")
        for i, p in enumerate(paragraphs[:3]):
            print(f"  [{i}] {p[:80]}...")

    # Format with alternating turn markers
    formatted_turns = []
    human_count = 0
    claude_count = 0

    # Skip title if first paragraph looks like a title
    start_idx = 0
    if paragraphs and 'Testing Data' in paragraphs[0]:
        start_idx = 1

    for i, para in enumerate(paragraphs[start_idx:]):
        if i % 2 == 0:
            # Human turn
            formatted_turns.append(f"Human: {para}")
            human_count += 1
        else:
            # Claude turn
            formatted_turns.append(f"Claude: {para}")
            claude_count += 1

    # Write formatted output
    output_text = '\n\n'.join(formatted_turns)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(output_text)

    return len(formatted_turns), human_count, claude_count

def main():
    input_file = Path(__file__).parent / 'Claude Conversation.txt'
    output_file = Path(__file__).parent / 'Claude_Conversation_formatted.txt'

    print("=" * 60)
    print("CLAUDE CONVERSATION AUTO-FORMATTER")
    print("=" * 60)
    print(f"Input: {input_file.name}")
    print(f"Output: {output_file.name}")
    print()

    if not input_file.exists():
        print(f"❌ Error: {input_file} not found!")
        return 1

    try:
        total, human, claude = format_conversation(input_file, output_file)

        print("✅ Formatting complete!")
        print(f"   Total turns: {total}")
        print(f"   Human turns: {human}")
        print(f"   Claude turns: {claude}")
        print()
        print(f"📝 Formatted file: {output_file}")
        print()
        print("You can now load this file in TELOSCOPE!")
        print("=" * 60)

        return 0

    except Exception as e:
        print(f"❌ Error during formatting: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    exit(main())
