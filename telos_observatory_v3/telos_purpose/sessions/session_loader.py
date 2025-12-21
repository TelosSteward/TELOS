"""
Session Loader for TELOSCOPE

Load conversation transcripts from files and convert to TELOS session format
for drift analysis and counterfactual evidence generation.

Supports:
- Plain text format (Human:/Assistant: markers)
- JSON format (ShareGPT-style)
- Markdown format (headers and bold text)
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import time


@dataclass
class LoadedTurn:
    """Represents one conversation turn from loaded session."""
    turn_id: int
    user_message: str
    assistant_response: str
    timestamp: float  # Synthetic timestamp for loaded sessions


@dataclass
class LoadedSession:
    """Complete loaded conversation session."""
    session_id: str
    source_file: str
    format: str
    turns: List[LoadedTurn]
    metadata: Dict


class SessionLoader:
    """
    Load conversation transcripts from files and convert to
    TELOS session format for analysis.

    Supported formats:
    - Plain text (Human:/Assistant:)
    - JSON (ShareGPT-style)
    - Markdown (## Turn headers)
    """

    def __init__(self, debug: bool = False):
        self.supported_formats = ['txt', 'json', 'md', 'markdown']
        self.debug = debug

    def load_from_text(self, content: str) -> LoadedSession:
        """
        Parse plain text format with turn markers.

        Supports multiple formats:
            Human: / Assistant:
            User: / Assistant:
            Human: / Claude:
            You: / Claude:

        All patterns are case-insensitive.
        """
        # Define flexible patterns for user and assistant turns
        user_patterns = [
            r'^Human:\s*',
            r'^User:\s*',
            r'^You:\s*',
        ]

        assistant_patterns = [
            r'^Assistant:\s*',
            r'^Claude:\s*',
        ]

        turns = []
        current_user = None
        current_assistant = None
        turn_id = 0

        # For debug: track detected markers
        detected_markers = {'user': set(), 'assistant': set()}

        lines = content.split('\n')

        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                continue

            # Check for user message with flexible patterns
            user_match = None
            for pattern in user_patterns:
                if re.match(pattern, line_stripped, re.IGNORECASE):
                    user_match = pattern
                    break

            if user_match:
                # Save previous turn if complete
                if current_user and current_assistant:
                    turns.append(LoadedTurn(
                        turn_id=turn_id,
                        user_message=current_user,
                        assistant_response=current_assistant,
                        timestamp=time.time() + turn_id
                    ))
                    turn_id += 1
                    current_assistant = None

                # Extract message (remove marker prefix)
                current_user = re.sub(user_match, '', line_stripped, flags=re.IGNORECASE).strip()

                # Track detected marker for debug
                marker = re.match(r'^(\w+):', line_stripped, re.IGNORECASE).group(1)
                detected_markers['user'].add(marker.lower())

                continue

            # Check for assistant message with flexible patterns
            assistant_match = None
            for pattern in assistant_patterns:
                if re.match(pattern, line_stripped, re.IGNORECASE):
                    assistant_match = pattern
                    break

            if assistant_match:
                # Extract message (remove marker prefix)
                current_assistant = re.sub(assistant_match, '', line_stripped, flags=re.IGNORECASE).strip()

                # Track detected marker for debug
                marker = re.match(r'^(\w+):', line_stripped, re.IGNORECASE).group(1)
                detected_markers['assistant'].add(marker.lower())

                continue

            # Continuation of previous message
            if current_user and not current_assistant:
                current_user += ' ' + line_stripped
            elif current_assistant:
                current_assistant += ' ' + line_stripped

        # Add final turn if complete
        if current_user and current_assistant:
            turns.append(LoadedTurn(
                turn_id=turn_id,
                user_message=current_user,
                assistant_response=current_assistant,
                timestamp=time.time() + turn_id
            ))

        # Debug output
        if self.debug:
            print(f"[DEBUG] Detected user markers: {detected_markers['user']}")
            print(f"[DEBUG] Detected assistant markers: {detected_markers['assistant']}")
            print(f"[DEBUG] Total turns parsed: {len(turns)}")
            if not turns:
                print(f"[DEBUG] No turns found. First 500 chars of content:")
                print(f"[DEBUG] {content[:500]}")

        return LoadedSession(
            session_id=f"loaded_{int(time.time())}",
            source_file="text_format",
            format="text",
            turns=turns,
            metadata={
                'original_format': 'text',
                'detected_user_markers': list(detected_markers['user']),
                'detected_assistant_markers': list(detected_markers['assistant'])
            }
        )

    def load_from_json(self, content: str) -> LoadedSession:
        """
        Parse JSON format (ShareGPT-style).

        Format:
        {
          "conversation_id": "test_01",
          "turns": [
            {"from": "human", "value": "Message"},
            {"from": "assistant", "value": "Response"}
          ]
        }
        """
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")

        turns = []
        turn_id = 0
        current_user = None

        # Handle different JSON structures
        if 'turns' in data:
            turn_list = data['turns']
        elif 'conversation' in data:
            turn_list = data['conversation']
        elif isinstance(data, list):
            turn_list = data
        else:
            raise ValueError("JSON must contain 'turns' or 'conversation' key, or be a list")

        for item in turn_list:
            from_field = item.get('from', item.get('role', '')).lower()
            value = item.get('value', item.get('content', ''))

            if from_field in ['human', 'user']:
                current_user = value
            elif from_field in ['assistant', 'claude', 'gpt']:
                if current_user:
                    turns.append(LoadedTurn(
                        turn_id=turn_id,
                        user_message=current_user,
                        assistant_response=value,
                        timestamp=time.time() + turn_id
                    ))
                    turn_id += 1
                    current_user = None

        return LoadedSession(
            session_id=data.get('conversation_id', data.get('id', f"loaded_{int(time.time())}")),
            source_file="json_format",
            format="json",
            turns=turns,
            metadata=data.get('metadata', {})
        )

    def load_from_markdown(self, content: str) -> LoadedSession:
        """
        Parse markdown format with headers.

        Supports multiple formats:
        ## Turn 1
        **Human:** / **Assistant:**
        **User:** / **Assistant:**
        **Human:** / **Claude:**
        **You:** / **Claude:**

        Also plain text markers (Human:, User:, You:, Assistant:, Claude:)
        All patterns are case-insensitive.
        """
        # Define flexible patterns
        user_bold_pattern = r'\*\*\s*(human|user|you)\s*:\*\*'
        assistant_bold_pattern = r'\*\*\s*(assistant|claude)\s*:\*\*'

        user_plain_patterns = [r'^human:\s*', r'^user:\s*', r'^you:\s*']
        assistant_plain_patterns = [r'^assistant:\s*', r'^claude:\s*']

        turns = []
        turn_id = 0
        current_user = None
        current_assistant = None

        # For debug: track detected markers
        detected_markers = {'user': set(), 'assistant': set()}

        lines = content.split('\n')

        for line in lines:
            line_stripped = line.strip()
            if not line_stripped or line_stripped.startswith('#'):
                # Save previous turn if complete
                if current_user and current_assistant:
                    turns.append(LoadedTurn(
                        turn_id=turn_id,
                        user_message=current_user,
                        assistant_response=current_assistant,
                        timestamp=time.time() + turn_id
                    ))
                    turn_id += 1
                    current_user = None
                    current_assistant = None
                continue

            # Check for user message with bold marker
            if re.match(user_bold_pattern, line_stripped, re.IGNORECASE):
                current_user = re.sub(user_bold_pattern, '', line_stripped, flags=re.IGNORECASE).strip()
                marker_match = re.match(r'\*\*\s*(\w+)\s*:\*\*', line_stripped, re.IGNORECASE)
                if marker_match:
                    detected_markers['user'].add(marker_match.group(1).lower())
                continue

            # Check for assistant message with bold marker
            if re.match(assistant_bold_pattern, line_stripped, re.IGNORECASE):
                current_assistant = re.sub(assistant_bold_pattern, '', line_stripped, flags=re.IGNORECASE).strip()
                marker_match = re.match(r'\*\*\s*(\w+)\s*:\*\*', line_stripped, re.IGNORECASE)
                if marker_match:
                    detected_markers['assistant'].add(marker_match.group(1).lower())
                continue

            # Check for plain user markers
            user_plain_match = None
            for pattern in user_plain_patterns:
                if re.match(pattern, line_stripped, re.IGNORECASE):
                    user_plain_match = pattern
                    break

            if user_plain_match:
                current_user = re.sub(user_plain_match, '', line_stripped, flags=re.IGNORECASE).strip()
                marker = re.match(r'^(\w+):', line_stripped, re.IGNORECASE).group(1)
                detected_markers['user'].add(marker.lower())
                continue

            # Check for plain assistant markers
            assistant_plain_match = None
            for pattern in assistant_plain_patterns:
                if re.match(pattern, line_stripped, re.IGNORECASE):
                    assistant_plain_match = pattern
                    break

            if assistant_plain_match:
                current_assistant = re.sub(assistant_plain_match, '', line_stripped, flags=re.IGNORECASE).strip()
                marker = re.match(r'^(\w+):', line_stripped, re.IGNORECASE).group(1)
                detected_markers['assistant'].add(marker.lower())
                continue

            # Continuation of previous message
            if current_user and not current_assistant:
                current_user += ' ' + line_stripped
            elif current_assistant:
                current_assistant += ' ' + line_stripped

        # Add final turn if complete
        if current_user and current_assistant:
            turns.append(LoadedTurn(
                turn_id=turn_id,
                user_message=current_user,
                assistant_response=current_assistant,
                timestamp=time.time() + turn_id
            ))

        # Debug output
        if self.debug:
            print(f"[DEBUG] Markdown - Detected user markers: {detected_markers['user']}")
            print(f"[DEBUG] Markdown - Detected assistant markers: {detected_markers['assistant']}")
            print(f"[DEBUG] Markdown - Total turns parsed: {len(turns)}")

        return LoadedSession(
            session_id=f"loaded_{int(time.time())}",
            source_file="markdown_format",
            format="markdown",
            turns=turns,
            metadata={
                'original_format': 'markdown',
                'detected_user_markers': list(detected_markers['user']),
                'detected_assistant_markers': list(detected_markers['assistant'])
            }
        )

    def auto_detect_format(self, filepath: str, content: str) -> str:
        """
        Detect format from file extension and content.

        Returns: 'text', 'json', or 'markdown'
        """
        path = Path(filepath) if isinstance(filepath, str) else filepath
        ext = path.suffix.lower()

        # Try extension first
        if ext in ['.json']:
            return 'json'
        elif ext in ['.md', '.markdown']:
            return 'markdown'
        elif ext in ['.txt']:
            # Check content for format hints
            if content.strip().startswith('{') or content.strip().startswith('['):
                return 'json'
            elif '**Human:**' in content or '**Assistant:**' in content or '## Turn' in content:
                return 'markdown'
            else:
                return 'text'

        # Fallback: analyze content
        if content.strip().startswith('{') or content.strip().startswith('['):
            return 'json'
        elif '**Human:**' in content or '## Turn' in content:
            return 'markdown'
        else:
            return 'text'

    def load_session(self, filepath: str) -> LoadedSession:
        """
        Main entry point - auto-detect format and load.

        Args:
            filepath: Path to conversation file

        Returns:
            LoadedSession object

        Raises:
            ValueError: If file format invalid or parsing fails
            FileNotFoundError: If file doesn't exist
        """
        path = Path(filepath)

        if not path.exists():
            raise FileNotFoundError(f"Session file not found: {filepath}")

        # Read file content
        try:
            content = path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            # Try with different encoding
            content = path.read_text(encoding='latin-1')

        if not content.strip():
            raise ValueError("Session file is empty")

        # Auto-detect format
        format_type = self.auto_detect_format(filepath, content)

        # Load based on format
        if format_type == 'json':
            session = self.load_from_json(content)
        elif format_type == 'markdown':
            session = self.load_from_markdown(content)
        else:  # text
            session = self.load_from_text(content)

        # Update source file
        session.source_file = str(path.name)

        # Validate session
        if not session.turns:
            raise ValueError("No valid conversation turns found in file")

        return session

    def validate_session(self, session: LoadedSession) -> Tuple[bool, List[str]]:
        """
        Validate loaded session.

        Returns:
            (is_valid, list_of_warnings)
        """
        warnings = []

        # Check turn count
        if len(session.turns) == 0:
            return False, ["No turns found"]

        if len(session.turns) > 100:
            warnings.append(f"Large session ({len(session.turns)} turns) may take time to process")

        # Check for empty messages
        for turn in session.turns:
            if not turn.user_message.strip():
                warnings.append(f"Turn {turn.turn_id}: Empty user message")
            if not turn.assistant_response.strip():
                warnings.append(f"Turn {turn.turn_id}: Empty assistant response")

        return True, warnings

    def export_session(self, session: LoadedSession, output_path: str,
                      format: str = 'json') -> None:
        """
        Export session to file.

        Args:
            session: LoadedSession to export
            output_path: Destination file path
            format: 'json', 'text', or 'markdown'
        """
        path = Path(output_path)

        if format == 'json':
            data = {
                'conversation_id': session.session_id,
                'source_file': session.source_file,
                'metadata': session.metadata,
                'turns': [
                    {
                        'turn_id': turn.turn_id,
                        'from': 'human',
                        'value': turn.user_message,
                    } if i % 2 == 0 else {
                        'turn_id': turn.turn_id,
                        'from': 'assistant',
                        'value': turn.assistant_response,
                    }
                    for i, turn in enumerate(session.turns)
                    for _ in range(2)  # Each turn has 2 messages
                ]
            }
            path.write_text(json.dumps(data, indent=2))

        elif format == 'text':
            lines = []
            for turn in session.turns:
                lines.append(f"Human: {turn.user_message}")
                lines.append(f"Assistant: {turn.assistant_response}")
                lines.append("")  # Blank line
            path.write_text('\n'.join(lines))

        elif format == 'markdown':
            lines = [f"# Conversation: {session.session_id}", ""]
            for turn in session.turns:
                lines.append(f"## Turn {turn.turn_id + 1}")
                lines.append(f"**Human:** {turn.user_message}")
                lines.append(f"**Assistant:** {turn.assistant_response}")
                lines.append("")
            path.write_text('\n'.join(lines))


def create_example_session(output_dir: str = "test_sessions") -> None:
    """
    Create example conversation files for testing.

    Args:
        output_dir: Directory to create example files
    """
    import os

    os.makedirs(output_dir, exist_ok=True)

    # Example 1: Text format
    text_content = """Human: What is the TELOS framework?
Assistant: TELOS is a mathematical framework for AI governance that uses embedding-based primacy attractors to maintain telic fidelity. It ensures AI systems stay aligned with their specified purpose through continuous drift detection and intervention."""

    # Write text file
    with open(f"{output_dir}/example_conversation.txt", "w") as f:
        f.write(text_content)

    print(f"✅ Example session created in {output_dir}/")
