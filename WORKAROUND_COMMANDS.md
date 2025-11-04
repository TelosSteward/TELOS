# Slash Commands Workaround

## Problem

Custom slash commands (`/telos`, `/monitor-export`, etc.) are not appearing in Claude Code even after:
- ✅ Proper YAML frontmatter
- ✅ Correct file structure in `.claude/commands/`
- ✅ Valid Markdown format
- ✅ Claude Code restart

## Root Cause (Unknown)

Possible reasons:
1. SlashCommand tool might be disabled in Claude Code settings
2. Version of Claude Code might not fully support custom slash commands
3. Unknown configuration issue
4. Bug in command discovery

## Workaround: Use Bash Scripts Instead

Since slash commands aren't working, use these alternative approaches:

### Option 1: Run Scripts Directly

**Instead of `/telos`, run:**
```bash
./telos_recall.sh
```

**Instead of `/monitor-export`, ask me to:**
- Extract conversation history
- Write to temporary file
- Run `python3 export_conversation.py`

**Instead of `/monitor-status`, run:**
```bash
python3 -c "
import json
from pathlib import Path

if Path('.telos_session_pa.json').exists():
    with open('.telos_session_pa.json') as f:
        pa = json.load(f)
    print('✅ PA Established')
    print(f'   Threshold: {pa[\"threshold\"]}')

    if Path('.telos_session_log.json').exists():
        with open('.telos_session_log.json') as f:
            log = json.load(f)
        print(f'   Turns: {len(log.get(\"turns\", []))}')
else:
    print('❌ PA not established')
"
```

### Option 2: Just Ask Me

Instead of slash commands, just ask me directly:

**For full recall:**
> "Run the telos recall protocol - show me .claude_project.md, git status, and establish PA"

**For monitoring:**
> "Check the TELOS monitoring status"

**For export:**
> "Export this conversation for grant validation"

I'll execute the same steps that the slash commands would have run.

### Option 3: Create Bash Aliases

Add to your `~/.bashrc` or `~/.zshrc`:

```bash
alias telos-recall='cd /Users/brunnerjf/Desktop/telos_privacy && ./telos_recall.sh'
alias telos-status='cd /Users/brunnerjf/Desktop/telos_privacy && python3 -c "import json; from pathlib import Path; ..."'
```

## What We've Tried

1. ✅ Restructured commands from subcommands to flat commands
2. ✅ Added proper YAML frontmatter to all commands
3. ✅ Restarted Claude Code
4. ✅ Validated YAML syntax
5. ✅ Checked file permissions
6. ✅ Verified file encoding (UTF-8)
7. ✅ Committed to git
8. ❌ Commands still don't appear

## Next Steps to Debug (If You Want)

1. Check Claude Code version: Maybe slash commands require a specific version
2. Check for errors: Look in Claude Code logs or developer console
3. Try minimal command: Create simplest possible command to test
4. Contact support: This might be a bug or version issue

## Conclusion

For now, **use `./telos_recall.sh` instead of `/telos`** and **just ask me directly** for other operations.

The functionality works - it's just the slash command discovery that's broken.
