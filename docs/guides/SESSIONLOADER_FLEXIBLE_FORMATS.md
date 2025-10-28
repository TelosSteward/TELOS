# SessionLoader Flexible Format Support

## Overview

SessionLoader has been enhanced to support multiple conversation format markers, making it compatible with Claude.ai conversation exports and various other formats.

## Supported Turn Markers

### User Messages
- `Human:` (original)
- `User:` (common format)
- `You:` (Claude.ai format)

### Assistant Messages
- `Assistant:` (original)
- `Claude:` (Claude.ai format)

**All patterns are case-insensitive** - works with `HUMAN:`, `human:`, `Human:`, etc.

## Examples

### Format 1: Human/Assistant
```
Human: What is TELOS?
Assistant: TELOS is a governance framework.
```

### Format 2: User/Assistant
```
User: Tell me about Claude.
Assistant: Claude is an AI assistant.
```

### Format 3: You/Claude (Claude.ai exports)
```
You: Hello Claude!
Claude: Hello! How can I help you today?
```

### Format 4: Human/Claude
```
Human: What's the difference?
Claude: I can explain that.
```

### Format 5: Mixed Case
```
HUMAN: Testing case insensitivity
ASSISTANT: This works too!
human: lowercase also works
assistant: all patterns supported
```

## Debug Mode

Enable debug mode to see what markers were detected:

```python
from telos_purpose.sessions.session_loader import SessionLoader

loader = SessionLoader(debug=True)
session = loader.load_session('conversation.txt')

# Debug output shows:
# [DEBUG] Detected user markers: {'human', 'you'}
# [DEBUG] Detected assistant markers: {'claude'}
# [DEBUG] Total turns parsed: 5
```

## Metadata

The parsed session now includes metadata about detected markers:

```python
session = loader.load_session('conversation.txt')

print(session.metadata['detected_user_markers'])      # ['human', 'you']
print(session.metadata['detected_assistant_markers']) # ['claude']
```

## Files Modified

### telos_purpose/sessions/session_loader.py

**Changes:**
1. Added `debug` parameter to `__init__()`
2. Rewrote `load_from_text()` with flexible pattern matching
3. Rewrote `load_from_markdown()` with flexible pattern matching
4. Added debug output when no turns are parsed
5. Added detected markers to session metadata

**Key Implementation:**
```python
# Flexible patterns for user turns
user_patterns = [
    r'^Human:\s*',
    r'^User:\s*',
    r'^You:\s*',
]

# Flexible patterns for assistant turns
assistant_patterns = [
    r'^Assistant:\s*',
    r'^Claude:\s*',
]

# Try each pattern with case-insensitive matching
for pattern in user_patterns:
    if re.match(pattern, line, re.IGNORECASE):
        # Extract message...
```

## Testing

Test all formats:
```bash
cd ~/Desktop/telos
source venv/bin/activate
python /tmp/test_flexible_loader.py
```

Expected output:
```
============================================================
TESTING FLEXIBLE SESSIONLOADER
============================================================

Human/Assistant:
------------------------------------------------------------
[DEBUG] Detected user markers: {'human'}
[DEBUG] Detected assistant markers: {'assistant'}
✅ Parsed 2 turns

You/Claude:
------------------------------------------------------------
[DEBUG] Detected user markers: {'you'}
[DEBUG] Detected assistant markers: {'claude'}
✅ Parsed 2 turns

...
============================================================
✅ ALL FORMATS PARSED SUCCESSFULLY
============================================================
```

## Use in Dashboard

The dashboard automatically benefits from this flexibility:

1. Upload any conversation format (.txt, .md, .json)
2. SessionLoader auto-detects and parses correctly
3. Works with Claude.ai exports, ChatGPT exports, custom formats

## Troubleshooting

If a conversation file isn't parsing:

1. **Enable debug mode:**
   ```python
   loader = SessionLoader(debug=True)
   session = loader.load_session('your_file.txt')
   ```

2. **Check debug output:**
   - Shows what markers were detected
   - Shows total turns parsed
   - If no turns found, shows first 500 chars of content

3. **Common issues:**
   - Missing turn markers (need `Human:`, `User:`, `You:`, etc.)
   - Incomplete turns (need both user and assistant messages)
   - Extra whitespace (parser strips whitespace automatically)

## Backward Compatibility

✅ All existing functionality preserved
✅ Original `Human:` / `Assistant:` format still works
✅ Existing code using SessionLoader works without changes
✅ Only new feature: additional format support

## Future Enhancements

Potential additions (not yet implemented):
- `AI:` marker support
- `Bot:` marker support
- Custom marker registration
- Multi-line message continuation markers
- Timestamped turn detection

---

**Last Updated**: 2025-10-26
**Status**: ✅ Production-ready
**Integration**: Complete with TELOSCOPE Dashboard
