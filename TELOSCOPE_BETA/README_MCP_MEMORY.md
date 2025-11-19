# MCP Memory Configuration for TELOS

This document explains how to configure Claude Code's MCP memory server to automatically recall TELOS context.

## Quick Start

### Option 1: Use the Recall Script (Immediate)

```bash
# Run this at the start of any Claude Code session
./Telos_recall.sh
```

This will display all critical context documents in your terminal.

### Option 2: Use Steward_PM.py (Programmatic)

```python
from Steward_PM import StewardPM

# Initialize
pm = StewardPM()

# Get specific document
beta_flow = pm.get_beta_experience_flow()

# Get all documents
all_docs = pm.get_all_documents()

# Print summary
pm.print_context_summary()
```

### Option 3: Configure MCP Memory Server

If you have the MCP memory server installed, you can configure it to automatically recall TELOS context.

## MCP Memory Server Setup

### 1. Install MCP Memory Server

```bash
npm install -g @modelcontextprotocol/server-memory
```

### 2. Configure Claude Code

Create or edit `~/.config/claude-code/mcp.json`:

```json
{
  "mcpServers": {
    "memory": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-memory"]
    }
  }
}
```

### 3. Store TELOS Context in MCP Memory

Once MCP memory is configured, you can store key context:

```bash
# Store BETA Experience Flow
mcp-memory store telos_beta_flow "$(cat BETA_EXPERIENCE_MASTER_FLOW.md)"

# Store BETA Status
mcp-memory store telos_beta_status "$(cat BETA_STATUS_SUMMARY.md)"

# Store Market Context
mcp-memory store telos_market "$(cat business/TELOS_Market_Position_Reality_Check.md)"

# Store Regulatory Context
mcp-memory store telos_regulatory "$(cat business/Regulatory_Forcing_Function.md)"
```

### 4. Recall Context in Claude Code

In your Claude Code conversation, you can reference:

```
Please recall telos_beta_flow
```

Or Claude Code will automatically access MCP memory when needed.

## Critical Documents to Always Recall

1. **BETA_EXPERIENCE_MASTER_FLOW.md**
   - When: Discussing BETA user experience, A/B testing, PA establishment
   - Contains: Complete 15-turn flow, Observatory access, fidelity viz

2. **BETA_STATUS_SUMMARY.md**
   - When: Checking integration status, debugging, reviewing components
   - Contains: DEMO/BETA status, component files, known issues

3. **business/TELOS_Market_Position_Reality_Check.md**
   - When: Discussing strategy, partnerships, market opportunity
   - Contains: TAM ($3.8B-$7.6B by 2027), valuation trajectory, moat analysis

4. **business/Regulatory_Forcing_Function.md**
   - When: Discussing compliance, enterprise urgency, timing
   - Contains: SB 53 deadline (Jan 1, 2026), penalty structure, forcing function

## File Reference Quick Map

| Need | File Path |
|------|-----------|
| PA Onboarding | `components/pa_onboarding.py` |
| A/B Testing Logic | `services/beta_sequence_generator.py` (lines 89-90) |
| BETA Integration | `main.py` (lines 1034-1100+) |
| Fidelity Viz | `components/fidelity_visualization.py` |
| Observatory | `components/observatory_review.py` |
| Steward Explanations | `services/beta_sequence_generator.py` (lines 158-214) |

## Context Recall on Session Restart

When restarting a Claude Code session:

1. **Immediate:** Run `./Telos_recall.sh` in terminal
2. **In conversation:** "Please read BETA_EXPERIENCE_MASTER_FLOW.md and BETA_STATUS_SUMMARY.md"
3. **Programmatic:** Import and use `Steward_PM.py`
4. **MCP:** Say "Recall TELOS context" (if MCP memory configured)

## Recent Changes to Recall

**November 18, 2025:**
- ✅ Removed 3 `st.balloons()` calls (unprofessional)
- ✅ Updated PA onboarding: yellow border, larger fonts, generic placeholder
- ✅ Fixed `pa_onboarding` parameter passing in `main.py`
- ✅ Created BETA_EXPERIENCE_MASTER_FLOW.md (complete 15-turn flow)
- ✅ Created Telos_recall.sh and Steward_PM.py (context mechanisms)

## User Preferences to Remember

**Styling:**
- Yellow borders (#FFD700, 3px)
- No emoji (unprofessional)
- Large fonts (20px+ for user input)

**Documentation:**
- Comprehensive, technical, with examples
- Markdown with tables, clear headings
- File references with line numbers

**Development:**
- No balloons
- Generic placeholder text (not TELOS-specific when used with native LLM)

---

**For Claude Code:** When this session is restarted, read this file first, then load the critical documents listed above.
