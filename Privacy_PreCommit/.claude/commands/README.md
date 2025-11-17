# Claude Code Custom Commands

This directory contains custom slash commands for TELOS development.

## Available Commands

### `/telos`
**Full project recall protocol**
- Display complete `.claude_project.md` (PA, values, scope, boundaries)
- Show git status and recent commits
- Establish mathematical PA (ℝ³⁸⁴ embedding)
- Activate real-time governance
- Provide immediate action clarity

**Use at start of EVERY new session.**

### `/monitor-export`
**Generate grant validation data**
- Analyze conversation with TELOS governance
- Generate fidelity measurements for each turn
- Create session file for dashboard
- Show validation summary for grants

**Use at end of sessions (optional) to generate evidence.**

### `/monitor-status`
**Check monitoring state**
- Show if PA established
- Display current session metrics
- Quick status check

### `/steward`
**Run Steward PM checks**
- Git status and commit recommendations
- Branch strategy suggestions
- Security audit status
- MCP orchestration guidance

## Setup Note

**If commands don't appear after creating/updating them:**

1. Restart Claude Code completely (quit and relaunch)
2. Commands should then appear in autocomplete
3. Check with `/help` to see available commands

## Files

- `telos.md` - Full recall protocol
- `monitor.md` - Export and validation
- `steward.md` - Steward PM integration

## Version Control

These commands are tracked in git (unlike `.claude/settings.local.json` which is ignored).

This means:
- ✅ Commands are version controlled
- ✅ Available in all clones of the repo
- ✅ Can be updated and committed
- ❌ May require Claude Code restart to pick up changes
