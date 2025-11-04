# How to Export Claude Code Sessions for TELOS Analysis

**Purpose:** Generate grant validation data from your Claude Code development sessions.

## Quick Start

### Option 1: Use `/monitor export` Command (Easiest)

In Claude Code session:
```
/monitor export
```

Claude will:
1. Extract conversation history from current session
2. Write to temp file in simple format
3. Run `export_conversation.py` automatically
4. Show fidelity analysis results
5. Generate session file for dashboard

### Option 2: Manual Export (If `/monitor export` Not Available)

**Step 1: Copy conversation to file**

Create a text file with this format:
```
USER: first message
ASSISTANT: first response
USER: second message
ASSISTANT: second response
```

Example (`my_session.txt`):
```
USER: Is there an mcp that would help with this internal verse external chain of command?
ASSISTANT: Great question! Let me think through what we have and what an MCP could add...
USER: Okay so I want a /monitor export function
ASSISTANT: Perfect! Let me create a streamlined /monitor export function...
```

**Step 2: Run analysis**

```bash
python3 export_conversation.py my_session.txt
```

**Output:**
```
🔭 TELOS Claude Code Governance Monitor
============================================================
Initializing ACTUAL TELOS infrastructure...

📊 Establishing Session Primacy Attractor...
✅ Dual PA Established

📊 Found 2 conversation turns
Analyzing with ACTUAL TELOS...

============================================================
🔍 Analyzing Turn 1
============================================================
   User Fidelity:  1.000 ✅
   AI Fidelity:    1.000 ✅
   Overall:        ✅ PASS

[...continues for all turns...]

============================================================
📊 Session Summary
============================================================
   Total Turns: 2
   Mean Fidelity: 1.000
   Min Fidelity: 1.000
   Max Fidelity: 1.000
   ✅ No significant drift detected

📊 Grant Application Value:
   ✅ Real fidelity measurements from actual development
   ✅ Demonstrates TELOS governing conversation building TELOS
   ✅ Validation data for institutional deployment claims

💾 Session exported to: sessions/claude_code_session_20251103_211947.json
```

**Step 3: View in dashboard**

```bash
./launch_dashboard.sh
# Opens at http://localhost:8501
# Load session file in TELOSCOPE tab
```

## What Gets Measured

**Real TELOS Infrastructure:**
- ✅ Dual PA architecture (`dual_attractor.py`)
- ✅ Mistral API for intent detection and PA derivation
- ✅ SentenceTransformer embeddings (local, no API calls)
- ✅ Actual fidelity calculations via `check_dual_pa_fidelity()`
- ✅ Steward PM orchestration (intelligent intervention decisions)
- ✅ Session state management (immutable snapshots)

**Metrics Generated:**
- User PA fidelity (alignment with session purpose)
- AI PA fidelity (adherence to role constraints)
- Turn-by-turn drift detection
- Mean/min/max fidelity scores
- Intervention recommendations

**Session Files:**
- Location: `sessions/claude_code_session_YYYYMMDD_HHMMSS.json`
- Format: Compatible with TELOSCOPE dashboard
- Size: ~50-200KB depending on turns
- Contains: Embeddings, metrics, full conversation history

## File Format Details

### Simple Format (Recommended)

```
USER: message text here
ASSISTANT: response text here
USER: next message
ASSISTANT: next response
```

**Rules:**
- Each line starts with `USER: ` or `ASSISTANT: `
- Blank lines are ignored
- Multi-line messages should be on one line
- Keep substantive turns (skip meta-commands like `/help`)

### What to Include

**✅ Include:**
- Technical discussions about TELOS
- Problem-solving conversations
- Design decisions
- Code reviews
- Planning sessions

**❌ Skip:**
- Simple commands (`/help`, `/clear`)
- File reads without discussion
- Trivial acknowledgments

## Grant Application Usage

**What this proves:**
> "TELOS doesn't just govern conversations - it governed the 1,000+ hour conversation that built TELOS itself."

**Evidence generated:**
1. Real session telemetry from actual development
2. Fidelity measurements on every conversation turn
3. Drift detection demonstrating governance effectiveness
4. Meta-demonstration (framework monitoring its own creation)
5. Validation data from real-world development work

**How to cite in grants:**
```
"We validated TELOS through self-application, monitoring the
development conversation itself with actual Dual PA governance.
Across X sessions totaling Y turns, mean fidelity was Z.XXX,
demonstrating effective governance even during complex technical
discussions about building the framework itself."
```

## Troubleshooting

**Error: "No conversation turns found"**
- Check file format (must start with `USER: ` or `ASSISTANT: `)
- Ensure alternating user/assistant pattern
- Verify file is not empty

**Error: "MistralClient object has no attribute..."**
- Already fixed with `mistral_adapter.py`
- Should not occur with current setup

**Error: "MISTRAL_API_KEY not found"**
```bash
# Check .env file
cat .env | grep MISTRAL_API_KEY

# Or set directly:
export MISTRAL_API_KEY="your-key-here"
```

**Low fidelity scores (< 0.7)**
- Expected! Means conversation drifted from session purpose
- NOT a bug - this is what TELOS detects
- Use for grant evidence: "detected drift on turns X, Y, Z"

## Advanced Usage

### Batch Analysis

Analyze multiple sessions at once:
```bash
for file in sessions/conversation_*.txt; do
    python3 export_conversation.py "$file"
done
```

### Custom PA for Analysis

Edit `claude_code_governance_monitor.py` line 67-90 to change the session PA:

```python
self.user_pa = {
    'purpose': ["Your custom purpose"],
    'scope': ["Item 1", "Item 2"],
    'boundaries': ["Boundary 1", "Boundary 2"],
    'fidelity_threshold': 0.65
}
```

### Generate Summary Report

```bash
# Analyze all sessions and generate summary
ls sessions/*.json | wc -l  # Count sessions
# Use dashboard to aggregate statistics
```

## Files Involved

**Scripts:**
- `export_conversation.py` - Main export script (use this!)
- `claude_code_governance_monitor.py` - Core monitoring engine
- `mistral_adapter.py` - Makes Mistral compatible with dual_attractor.py
- `steward_governance_orchestrator.py` - Intelligent interventions
- `claude_project_pa_controller.py` - Updates .claude_project.md

**Session Data:**
- `sessions/` - Directory for exported session files
- `.claude_project.md` - Session PA definition
- `example_conversation.txt` - Example format

**Dashboard:**
- `./launch_dashboard.sh` - Starts Streamlit dashboard
- TELOSCOPE tab - Load and visualize sessions

## Summary

**Easiest way:**
```
/monitor export
```

**Manual way:**
```bash
# 1. Copy conversation to text file
# 2. Run export
python3 export_conversation.py my_session.txt

# 3. View results
./launch_dashboard.sh
```

**Result:** Grant-ready validation data from your actual development work.

**This is "working smarter not harder" - your development time generates validation data.**
