"""
ShareGPT Importer for Observatory v2

Parses ShareGPT conversation format and converts to TELOS session format
for batch testing and analysis.

Supports:
- ShareGPT v1 format (standard)
- OpenAI chat format
- Custom conversation formats

Usage:
    from teloscope_v2.utils.sharegpt_importer import ShareGPTImporter

    importer = ShareGPTImporter()

    # Import file
    sessions = importer.import_file('conversations.json')

    # Validate
    for session in sessions:
        if importer.validate_session(session):
            # Process with TELOS
            ...
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
from pathlib import Path


class ShareGPTImporter:
    """
    Imports ShareGPT conversations for TELOS analysis.

    Converts ShareGPT JSON format to TELOS session format, enabling batch
    testing of real-world conversations with TELOS governance.
    """

    def __init__(self):
        """Initialize importer."""
        self.supported_formats = [
            'sharegpt_v1',
            'openai_chat',
            'generic_conversation'
        ]

    def import_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Import ShareGPT file and convert to TELOS sessions.

        Args:
            file_path: Path to ShareGPT JSON file

        Returns:
            List of TELOS session dicts

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
            json.JSONDecodeError: If JSON is malformed

        Example:
            sessions = importer.import_file('sharegpt_conversations.json')
            print(f"Imported {len(sessions)} conversations")
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return self.parse_sharegpt(data)

    def parse_sharegpt(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Parse ShareGPT JSON data.

        Handles multiple format variations:
        - {'conversations': [...]}
        - Direct list: [...]
        - Single conversation: {...}

        Args:
            data: ShareGPT JSON data

        Returns:
            List of TELOS session dicts

        Raises:
            ValueError: If format is unrecognized
        """
        sessions = []

        # Detect format and extract conversations
        if isinstance(data, dict):
            if 'conversations' in data:
                # Format: {'conversations': [...]}
                conversations = data['conversations']
            elif 'messages' in data:
                # Single conversation format
                conversations = [data]
            elif 'turns' in data:
                # Alternative single conversation format
                conversations = [data]
            else:
                # Try to detect if it's a single conversation object
                if self._looks_like_conversation(data):
                    conversations = [data]
                else:
                    raise ValueError(
                        "Unknown ShareGPT format. Expected 'conversations', 'messages', or 'turns' key"
                    )

        elif isinstance(data, list):
            # Direct list of conversations
            conversations = data

        else:
            raise ValueError(f"Unknown data type: {type(data)}")

        # Convert each conversation
        for conv in conversations:
            try:
                session = self._convert_conversation(conv)
                if session:
                    sessions.append(session)
            except Exception as e:
                print(f"Warning: Failed to convert conversation: {e}")
                continue

        return sessions

    def _looks_like_conversation(self, data: Dict[str, Any]) -> bool:
        """
        Check if dict looks like a conversation object.

        Args:
            data: Dict to check

        Returns:
            True if it appears to be a conversation
        """
        # Check for common conversation keys
        conversation_keys = ['id', 'turns', 'messages', 'history', 'conversation']
        return any(key in data for key in conversation_keys)

    def _convert_conversation(self, conv: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Convert single ShareGPT conversation to TELOS session.

        Args:
            conv: ShareGPT conversation dict

        Returns:
            TELOS session dict or None if invalid

        Example Input (ShareGPT):
            {
                'id': 'conv_123',
                'turns': [
                    {'from': 'human', 'value': 'What is TELOS?'},
                    {'from': 'gpt', 'value': 'TELOS is...'}
                ]
            }

        Example Output (TELOS):
            {
                'session_id': 'sharegpt_conv_123',
                'turns': [
                    {
                        'turn': 1,
                        'user_input': 'What is TELOS?',
                        'assistant_response': 'TELOS is...',
                        'status': '✓',
                        'fidelity': None
                    }
                ],
                'metadata': {...}
            }
        """
        # Extract turns/messages
        turns_raw = self._extract_turns(conv)

        if not turns_raw:
            return None

        # Convert to TELOS format
        telos_turns = []
        turn_num = 1

        # Process pairs of messages (user + assistant)
        i = 0
        while i < len(turns_raw):
            user_turn = turns_raw[i]

            # Check if this is a user message
            if not self._is_user_message(user_turn):
                i += 1
                continue

            # Get next message (should be assistant)
            if i + 1 >= len(turns_raw):
                # No assistant response, skip
                break

            assistant_turn = turns_raw[i + 1]

            # Check if assistant message
            if not self._is_assistant_message(assistant_turn):
                # Not an assistant message, skip
                i += 1
                continue

            # Extract content
            user_content = self._extract_content(user_turn)
            assistant_content = self._extract_content(assistant_turn)

            if not user_content or not assistant_content:
                i += 2
                continue

            telos_turns.append({
                'turn': turn_num,
                'user_input': user_content,
                'assistant_response': assistant_content,
                'status': '✓',  # Will be updated after analysis
                'fidelity': None,  # Will be calculated
                'timestamp': None  # Unknown from ShareGPT
            })

            turn_num += 1
            i += 2  # Move to next pair

        if not telos_turns:
            return None

        # Build session
        session_id = self._generate_session_id(conv)

        return {
            'session_id': session_id,
            'session_type': 'imported',
            'turns': telos_turns,
            'metadata': {
                'source': 'sharegpt',
                'original_id': conv.get('id', 'unknown'),
                'imported_at': datetime.now().isoformat(),
                'original_turn_count': len(turns_raw) // 2,
                'original_data': self._extract_metadata(conv)
            }
        }

    def _extract_turns(self, conv: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract turns/messages from conversation.

        Args:
            conv: Conversation dict

        Returns:
            List of turn dicts
        """
        # Try different keys
        if 'turns' in conv:
            return conv['turns']
        elif 'messages' in conv:
            return conv['messages']
        elif 'conversation' in conv:
            return conv['conversation']
        elif 'history' in conv:
            return conv['history']
        else:
            return []

    def _is_user_message(self, turn: Dict[str, Any]) -> bool:
        """
        Check if turn is a user message.

        Args:
            turn: Turn dict

        Returns:
            True if user message, False otherwise
        """
        role_field = turn.get('from', turn.get('role', ''))
        return role_field.lower() in ['human', 'user', 'person']

    def _is_assistant_message(self, turn: Dict[str, Any]) -> bool:
        """
        Check if turn is an assistant message.

        Args:
            turn: Turn dict

        Returns:
            True if assistant message, False otherwise
        """
        role_field = turn.get('from', turn.get('role', ''))
        return role_field.lower() in ['gpt', 'assistant', 'bot', 'ai']

    def _extract_content(self, turn: Dict[str, Any]) -> Optional[str]:
        """
        Extract content from turn (handles different field names).

        Args:
            turn: Turn dict from ShareGPT

        Returns:
            Content string or None
        """
        # Try different field names
        if 'value' in turn:
            return turn['value']
        elif 'content' in turn:
            return turn['content']
        elif 'text' in turn:
            return turn['text']
        elif 'message' in turn:
            return turn['message']
        else:
            return None

    def _generate_session_id(self, conv: Dict[str, Any]) -> str:
        """
        Generate session ID from conversation.

        Args:
            conv: Conversation dict

        Returns:
            Session ID string
        """
        if 'id' in conv:
            return f"sharegpt_{conv['id']}"
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            return f"sharegpt_{timestamp}"

    def _extract_metadata(self, conv: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract metadata from conversation.

        Args:
            conv: Conversation dict

        Returns:
            Metadata dict
        """
        metadata = {}

        # Copy common metadata fields
        metadata_fields = ['title', 'tags', 'language', 'model', 'created_at']

        for field in metadata_fields:
            if field in conv:
                metadata[field] = conv[field]

        return metadata

    def validate_session(self, session: Dict[str, Any]) -> bool:
        """
        Validate imported session.

        Checks:
        - Required fields present
        - Turns list not empty
        - Each turn has user_input and assistant_response

        Args:
            session: TELOS session dict

        Returns:
            True if valid, False otherwise
        """
        # Check required fields
        required_fields = ['session_id', 'turns', 'metadata']

        if not all(field in session for field in required_fields):
            return False

        # Check turns not empty
        if not session['turns']:
            return False

        # Check each turn
        for turn in session['turns']:
            if 'user_input' not in turn or 'assistant_response' not in turn:
                return False

            # Check content not empty
            if not turn['user_input'] or not turn['assistant_response']:
                return False

        return True

    def get_session_summary(self, session: Dict[str, Any]) -> str:
        """
        Get human-readable summary of imported session.

        Args:
            session: TELOS session dict

        Returns:
            Summary string
        """
        turn_count = len(session.get('turns', []))
        session_id = session.get('session_id', 'unknown')
        source = session.get('metadata', {}).get('source', 'unknown')
        original_id = session.get('metadata', {}).get('original_id', 'unknown')

        summary = f"""
Session Summary:
- ID: {session_id}
- Source: {source}
- Original ID: {original_id}
- Turns: {turn_count}
- Valid: {'✅' if self.validate_session(session) else '❌'}
        """

        return summary.strip()

    def import_multiple_files(
        self,
        file_paths: List[str]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Import multiple ShareGPT files.

        Args:
            file_paths: List of file paths

        Returns:
            Dict mapping file path to list of sessions

        Example:
            files = ['file1.json', 'file2.json']
            results = importer.import_multiple_files(files)

            for file_path, sessions in results.items():
                print(f"{file_path}: {len(sessions)} conversations")
        """
        results = {}

        for file_path in file_paths:
            try:
                sessions = self.import_file(file_path)
                results[file_path] = sessions
            except Exception as e:
                print(f"Warning: Failed to import {file_path}: {e}")
                results[file_path] = []

        return results

    def export_sample_format(self, output_path: str):
        """
        Export sample ShareGPT format for reference.

        Args:
            output_path: Path to save sample file

        Example:
            importer.export_sample_format('sample_sharegpt.json')
        """
        sample = {
            'conversations': [
                {
                    'id': 'sample_conv_001',
                    'title': 'Sample Conversation',
                    'turns': [
                        {
                            'from': 'human',
                            'value': 'What is TELOS governance?'
                        },
                        {
                            'from': 'gpt',
                            'value': 'TELOS is a Model-Based Lens (MBL) governance framework...'
                        },
                        {
                            'from': 'human',
                            'value': 'How does it work?'
                        },
                        {
                            'from': 'gpt',
                            'value': 'TELOS uses a Primacy Attractor to define governance boundaries...'
                        }
                    ]
                }
            ]
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dumps(sample, f, indent=2)


def quick_import(file_path: str) -> List[Dict[str, Any]]:
    """
    Convenience function for quick import.

    Args:
        file_path: Path to ShareGPT JSON file

    Returns:
        List of TELOS session dicts

    Example:
        sessions = quick_import('conversations.json')
    """
    importer = ShareGPTImporter()
    return importer.import_file(file_path)
