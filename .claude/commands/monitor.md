---
description: Control TELOS governance monitoring for this Claude Code session
---

# Monitor Command - TELOS Governance Control

**Purpose:** Toggle and control real-time TELOS governance monitoring of this Claude Code session.

## What This Does

When monitoring is enabled (default for TELOS work), this Claude Code session operates under ACTUAL TELOS runtime governance:

- External measurement via `claude_code_governance_monitor.py`
- Real Dual PA architecture from `telos_purpose/core/dual_attractor.py`
- Turn-by-turn fidelity calculations (actual embeddings + math)
- Steward PM intelligent intervention decisions
- Updates to `.claude_project.md` if drift detected
- Session telemetry exported for grant validation data

This is the meta-demonstration: **TELOS governing the conversation building TELOS.**

## Usage

The command was invoked with: `{{command}}`

Parse the command to determine action:

### `/monitor` or `/monitor status`
**Check current monitoring status**

Output:
```
🔭 TELOS Governance Monitoring Status

Session PA:
  Purpose: [from .claude_project.md]
  Fidelity Threshold: 0.65
  Status: ✅ Active

Monitoring:
  Status: ✅ Enabled (default for TELOS work)
  Script: claude_code_governance_monitor.py
  Sessions Dir: sessions/
  Latest Session: claude_code_session_YYYYMMDD_HHMMSS.json

Dashboard:
  Command: ./launch_dashboard.sh
  URL: http://localhost:8501

Current Turn: [estimate from session]
Mean Fidelity: [from latest session if available]
```

### `/monitor on`
**Enable monitoring (if disabled)**

1. Verify `claude_code_governance_monitor.py` exists
2. Check that Session PA is established in `.claude_project.md`
3. Confirm monitoring dependencies (Mistral API key, embeddings)
4. Enable background monitoring
5. Confirm to user:
```
✅ TELOS governance monitoring ENABLED

- Turn-by-turn fidelity measurement active
- Steward PM orchestrating interventions
- Session telemetry being recorded
- Drift detection active

Session PA:
  Purpose: [from .claude_project.md]

This session is now generating validation data for grants.
```

### `/monitor off`
**Disable monitoring (for non-TELOS work)**

1. Stop background monitoring
2. Preserve any existing session data
3. Confirm to user:
```
⏸️  TELOS governance monitoring DISABLED

- No fidelity measurement
- No interventions
- No telemetry recording

Use `/monitor on` to re-enable.

Note: Monitoring is recommended for TELOS development to:
- Generate validation data
- Demonstrate self-application
- Ensure governance effectiveness
```

### `/monitor export`
**Analyze this conversation with TELOS governance and generate validation data**

This is the PRIMARY way to generate grant validation data from Claude Code sessions.

**What it does:**
1. Takes conversation history from this session
2. Runs through ACTUAL TELOS governance (dual_attractor.py)
3. Generates fidelity measurements for each turn
4. Creates session file for dashboard viewing
5. Shows validation summary

**Implementation:**

When user invokes `/monitor export`:

1. **Extract conversation turns** from current session context
   - Get as many user/assistant turns as available
   - Focus on substantive exchanges (skip meta-commands)

2. **Write to temporary file** in simple format:
   ```
   USER: first user message
   ASSISTANT: first assistant response
   USER: second user message
   ASSISTANT: second assistant response
   ```

3. **Run analysis script:**
   ```bash
   python3 export_conversation.py /tmp/conversation_export_TIMESTAMP.txt
   ```

4. **Display results** and clean up temp file

**Output format:**

```
🔭 TELOS Governance Analysis

Running ACTUAL TELOS on this session...

============================================================
📊 Establishing Session Primacy Attractor...
   Purpose: [from .claude_project.md]

[Run through turns...]

============================================================
📊 Session Summary
============================================================
   Total Turns Analyzed: X
   Mean Fidelity: X.XXX
   Min Fidelity: X.XXX
   Max Fidelity: X.XXX

   [✅ No significant drift detected | 🚨 Drift detected on turns: X, Y, Z]

💾 Session exported to: sessions/claude_code_session_YYYYMMDD_HHMMSS.json

📊 Grant Application Value:
   ✅ Real fidelity measurements from actual development
   ✅ Demonstrates TELOS governing conversation building TELOS
   ✅ Validation data for institutional deployment claims

To view in dashboard:
   1. ./launch_dashboard.sh
   2. Load session file in TELOSCOPE tab
   3. View turn-by-turn metrics and drift timeline
```

**Use this command at end of TELOS development sessions to generate validation data.**

### `/monitor validate`
**Run validation analysis on current session**

1. Analyze latest session data
2. Check for:
   - PA adherence throughout session
   - Drift patterns (if any)
   - Intervention effectiveness
   - Grant-ready metrics
3. Output validation report:
```
✅ Session Validation Report

Governance Effectiveness:
  Mean Fidelity: X.XXX (target ≥0.70)
  Basin Adherence: XX% of turns
  Drift Events: X
  Interventions: X

Grant Application Value:
  ✅ Demonstrates self-application
  ✅ Real telemetry from development
  ✅ Evidence of drift prevention

Quality Metrics:
  - Session remained on track: [Yes/No]
  - Purpose fulfilled: [Analysis]
  - Boundaries respected: [Analysis]

This session data is ready for grant applications.
```

## Implementation Notes

**For Claude Code:**

1. **Monitoring is ON by default** for TELOS development sessions
   - Session PA is already established in `.claude_project.md`
   - External monitoring via `claude_code_governance_monitor.py`
   - No action needed from user unless toggling

2. **Session Files Location:** `sessions/claude_code_session_YYYYMMDD_HHMMSS.json`
   - Compatible with TELOSCOPE dashboard
   - Contains full turn-by-turn telemetry
   - Embeddings, fidelity scores, interventions

3. **Dashboard Access:**
   ```bash
   ./launch_dashboard.sh
   # Opens at http://localhost:8501
   # Load session in TELOSCOPE tab
   ```

4. **Technical Components:**
   - `claude_code_governance_monitor.py` - Main monitoring
   - `mistral_adapter.py` - Client compatibility layer
   - `steward_governance_orchestrator.py` - Intervention intelligence
   - `claude_project_pa_controller.py` - Updates .claude_project.md

5. **Grant Application Value:**
   Every monitored session generates validation data proving:
   - TELOS governs the conversation building TELOS (meta-demonstration)
   - Real embeddings, real fidelity, real drift detection
   - Self-validating framework
   - Evidence no other AI governance project has

## Security

All monitoring infrastructure is in **private repo** (telos_privacy):
- ✅ Proprietary Dual PA architecture
- ✅ Complete DMAIC/SPC implementation
- ✅ Steward PM orchestration
- ❌ Never expose publicly

## Error Handling

If monitoring fails:
- Check MISTRAL_API_KEY in `.env`
- Verify `telos_purpose/` is accessible
- Check `sessions/` directory exists
- Ensure `claude_code_governance_monitor.py` is executable

If uncertain, default to showing current status and offering to enable monitoring.
