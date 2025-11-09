# Steward Unified - Integration Summary

## What Was Done

### 1. MCP Configuration ✅
Updated your Claude Desktop configuration at:
`~/Library/Application Support/Claude/claude_desktop_config.json`

**Added MCPs:**
- **memory** - Memory server for project state tracking
- **git** - Git operations (already configured for TELOS_CLEAN)
- **context7** - Context management via Upstash
- **supabase** - PostgreSQL database (your TELOS Supabase instance)
- **github** - GitHub integration
- **playwright** - Browser automation

**Action Required:** Restart Claude Desktop for MCP changes to take effect.

---

### 2. Steward Unified ✅
Created `steward/steward_unified.py` - a single unified steward combining:

**From steward_pm.py (Project Management):**
- Full TELOS project oversight (grants, partnerships, validation)
- Memory MCP integration for dynamic state tracking
- Health monitoring and system diagnostics
- Dashboard/governance system integration
- Repository strategy and sanitization workflows

**From steward_v2.py (Active Orchestration):**
- Automatic MCP orchestration (decides when to invoke MCPs)
- Git Guardian with pre-commit security audits
- Continuous daemon monitoring mode
- Task synchronization (TodoWrite <-> TASKS.md)
- Smart commit timing recommendations

---

### 3. Updated telos_recall.sh ✅
Modified the recall script to use `steward_unified.py` instead of separate stewards.

Now when you run `./telos_recall.sh`, it:
1. Shows full project context
2. Displays git status
3. Shows validation data status
4. Checks testing framework
5. Verifies Observatory status
6. Shows environment info
7. Displays key findings
8. **Initializes Steward Unified** (instead of separate stewards)

---

## How to Use

### Basic Commands

```bash
# Get full project status with MCP insights
python3 steward/steward_unified.py status

# Show what MCPs should be invoked
python3 steward/steward_unified.py orchestrate

# Run security audit before committing
python3 steward/steward_unified.py git-audit

# Check if it's time to commit
python3 steward/steward_unified.py should-commit
```

### Project Management

```bash
# Priority recommendations
python3 steward/steward_unified.py next

# Grant application status
python3 steward/steward_unified.py grants

# Risk analysis
python3 steward/steward_unified.py risks

# Partnership progress
python3 steward/steward_unified.py partnerships
```

### System Monitoring

```bash
# System health check
python3 steward/steward_unified.py health

# Launch TELOS HUD dashboard
python3 steward/steward_unified.py dashboard

# Continuous monitoring (daemon mode)
python3 steward/steward_unified.py daemon

# Show governance summary
python3 steward/steward_unified.py governance
```

### Development & Git

```bash
# Show repository strategy
python3 steward/steward_unified.py repos

# Show sanitization workflow
python3 steward/steward_unified.py sanitize

# Git MCP info
python3 steward/steward_unified.py git
```

---

## MCP Integration

The unified steward automatically detects when MCPs should be invoked:

### Git MCP
- **Triggered when:** Staged files exist, uncommitted changes, time to commit
- **Actions:** Suggest commits, run security audits, check branch strategy

### Memory MCP
- **Triggered when:** Project state changes, grant deadlines approaching
- **Actions:** Update project state, track progress

### PostgreSQL MCP (Supabase)
- **Triggered when:** New validation data, grant metrics needed
- **Actions:** Query study results, generate metrics

### Playwright MCP
- **Triggered when:** Observatory UI changes, screenshots needed
- **Actions:** Capture dashboard, generate visual assets

---

## Running the Recall Script

```bash
# From TELOS_CLEAN directory
./telos_recall.sh
```

This will:
1. Load full TELOS project context
2. Show git status and recent commits
3. Display validation data availability
4. Check testing framework components
5. Verify Observatory is running
6. Show environment and dependencies
7. Display key research findings
8. **Initialize Steward Unified with MCP recommendations**

---

## Next Steps

1. **Restart Claude Desktop** to activate the new MCP configuration
2. Run `./telos_recall.sh` to test the full integration
3. Use `python3 steward/steward_unified.py status` to see MCP recommendations
4. Start using MCPs through Claude Code as suggested by Steward

---

## Benefits

✅ **Single unified steward** - No more separate steward processes
✅ **Automatic MCP orchestration** - Steward tells you when to use MCPs
✅ **Security auditing** - Pre-commit checks for IP leaks
✅ **Smart commit timing** - Recommendations based on git state
✅ **Full project oversight** - Grants, partnerships, validation tracking
✅ **Continuous monitoring** - Daemon mode for active oversight

---

## File Locations

- **Unified Steward:** `steward/steward_unified.py`
- **Recall Script:** `telos_recall.sh`
- **MCP Config:** `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Project MCP Config:** `.mcp.json`

---

## Old Stewards (Deprecated)

These files are now superseded by `steward_unified.py`:
- ~~`steward/steward_pm.py`~~ (merged into unified)
- ~~`steward/steward_v2.py`~~ (merged into unified)

You can keep them for reference or delete them.
