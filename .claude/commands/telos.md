---
description: Full project recall protocol - Refresh context and establish TELOS governance
---

# `/telos init` - Full Project Recall & Governance Protocol

**Purpose:** Complete project context refresh for new Claude Code sessions + mathematical PA establishment + real-time governance activation.

## When to Use

**Use `/telos init` at start of EVERY new Claude Code session to:**
1. Refresh full project context (.claude_project.md, git status, priorities)
2. Establish mathematical Primacy Attractor governance
3. Activate real-time fidelity monitoring
4. Get ready to continue development with full context

**This is the "catch me up on everything" command.**

## Implementation

When user invokes `/telos init` or `/telos`, I should:

### PART 1: PROJECT CONTEXT REFRESH

**1. Display Full .claude_project.md**
- Read entire `.claude_project.md` file
- Show: Project overview, core innovations, grant strategy, session PA, core values, scope, boundaries, PBC charter
- This refreshes me on what TELOS is, what we're building, and how to govern

**2. Git Status & Recent Work**
```bash
git status
git log --oneline -5
git branch -a
```
Show: Current branch, uncommitted files, recent commits, active branches (ltff, ev, eu)

**3. Steward PM Status & Recommendations**
```bash
python3 steward_v2.py status
python3 steward_v2.py should-commit
python3 steward_v2.py orchestrate
```
Show:
- Current repo (private/public)
- Git state (uncommitted changes, time since last commit)
- Branch strategy recommendations
- MCP orchestration suggestions
- Security audit status
- **Steward PM snapped to attention, providing clarity on what's needed**

**4. Key Architecture Files**
List and briefly describe:
- `telos_purpose/core/dual_attractor.py` - Dual PA engine
- `telos_purpose/core/primacy_math.py` - Mathematical foundations
- `steward_governance_orchestrator.py` - Intelligent interventions
- `claude_project_pa_controller.py` - PA updates
- `steward_v2.py` - Steward PM (security, git, orchestration)
- `telos_init.py`, `telos_check.py` - Runtime governance scripts

**5. Current Priorities (from .claude_project.md)**
Highlight:
- Grant applications (LTFF $150K, EV $400K, EU €350K)
- Validation studies and metrics
- GMU partnership (institutional anchor for EV)
- Observatory/TELOSCOPE demos
- Trail of Bits audit prep

**6. Recent Session Metrics (if available)**
Check for:
- Previous session logs (`.telos_session_log.json`)
- Recent fidelity trends
- Any drift patterns to be aware of

### PART 2: ESTABLISH MATHEMATICAL GOVERNANCE

**7. Run PA Establishment**
```bash
python3 telos_init.py
```

This:
- Extracts Session Purpose from `.claude_project.md`
- Generates embedding (ℝ³⁸⁴ mathematical PA center)
- Stores in `.telos_session_pa.json`
- Initializes `.telos_session_log.json`
- Confirms governance active

**8. Confirm Readiness with Steward PM Clarity**
Show:
```
✅ TELOS SESSION READY

📊 Project Context Refreshed:
   - PA, values, scope, boundaries loaded
   - Git status checked
   - Priorities identified
   - Architecture files indexed

🔭 Governance Active:
   - Mathematical PA established (ℝ³⁸⁴)
   - Real-time fidelity monitoring enabled
   - Threshold: F ≥ 0.65

🤖 Steward PM Status:
   - Repo: [private/public]
   - Git: [X files changed, Y hours since commit]
   - Recommendation: [commit/continue/branch]
   - Security: [audit status]
   - MCP: [orchestration suggestions]

🚀 Current Focus (Top 3):
   1. [Priority 1 from .claude_project.md]
   2. [Priority 2]
   3. [Priority 3]

📋 Steward PM Clarity:
   [Specific recommendation: "Commit recent changes before proceeding"
    or "Continue on grants work" or "Switch to ltff-application branch" etc.]

Ready to execute under full TELOS governance.
```

### PART 3: PROACTIVE SELF-MONITORING

**After `/telos init`, I should:**

1. **After each substantive response:**
   - Run: `python3 telos_check.py "[my response text]"`
   - Log fidelity score
   - Show brief status: `📊 [F=0.847 ✅] Turn 3`

2. **If drift detected (F < 0.65):**
   - Show warning: `🚨 [F=0.612 DRIFT] Turn 5`
   - Update `.claude_project.md` with intervention guidance
   - Course correct next response

3. **Periodically check alignment:**
   - Every 5-10 turns, remind myself of PA
   - Check if conversation drifting from priorities
   - Proactively refocus if needed

## Command Variants

**`/telos init`** - Full recall protocol (use at session start)

**`/telos status`** - Check if PA established and show current metrics
```bash
python3 -c "
import json
from pathlib import Path

if Path('.telos_session_pa.json').exists():
    with open('.telos_session_pa.json') as f:
        pa = json.load(f)
    print('✅ Governance active')
    print(f'Established: {pa[\"established_at\"]}')

    if Path('.telos_session_log.json').exists():
        with open('.telos_session_log.json') as f:
            log = json.load(f)
        turns = len(log.get('turns', []))
        if turns > 0:
            print(f'Turns logged: {turns}')
            recent = log['turns'][-1]
            print(f'Latest fidelity: {recent[\"fidelity\"]:.3f}')
else:
    print('❌ Governance not active. Run /telos init')
"
```

**`/telos check`** - Manually check last response fidelity (for testing)

## What Gets Created

**Session files (temporary, in .gitignore):**
- `.telos_session_pa.json` - PA embedding and metadata
- `.telos_session_log.json` - Turn-by-turn fidelity log

**These are local session files, auto-deleted on cleanup**

## Integration with Other Commands

**Workflow:**
1. New session starts
2. User: `/telos init` (or I proactively do it)
3. Development work (governed in real-time)
4. Session end: Optional `/monitor export` for grant validation data

**The full stack:**
- `/telos init` - Start of session (full context + governance)
- Real-time self-monitoring - During session (automatic)
- `/monitor export` - End of session (grant validation)

## Steward PM Integration

**When `/telos init` runs, Steward PM provides immediate clarity:**

**Repo Context:**
```bash
python3 steward_v2.py status
```
→ Private repo vs public, security audit status

**Git Recommendations:**
```bash
python3 steward_v2.py should-commit
```
→ "5 files changed, 2.3h since last commit → ✅ Commit recommended"
→ "Continue working, no commit needed yet"

**Branch Strategy:**
```bash
python3 steward_v2.py orchestrate
```
→ "Working on grants → Consider switching to ltff-application branch"
→ "Git MCP suggests: commit current work before switching branches"

**Specific Clarity Examples:**

**Scenario 1: Uncommitted work**
```
🤖 Steward PM: ⚠️ 8 files uncommitted, 3.5 hours since last commit
📋 Recommendation: Commit current work before proceeding
   Suggested: git add . && git commit -m "..."
```

**Scenario 2: Clean state**
```
🤖 Steward PM: ✅ Clean working tree, last commit 20 minutes ago
📋 Recommendation: Continue on current priorities (grants)
   Focus: LTFF application refinement
```

**Scenario 3: Branch suggestion**
```
🤖 Steward PM: 💡 Working on EU grant content
📋 Recommendation: Switch to eu-application branch
   Command: git checkout eu-application
   Rationale: Keep grant work isolated for clean PR later
```

**Steward PM is snapped to attention, providing:**
- Clear status (what's the repo state?)
- Specific recommendation (what should I do?)
- Rationale (why does it matter?)
- Action (exact command if needed)

## Why This Matters

**Without `/telos init`:**
- New Claude Code session starts "cold"
- No project context loaded
- No governance active
- Have to re-explain everything
- No clarity on what to do first

**With `/telos init`:**
- Full project context instantly refreshed
- Mathematical governance established
- Priorities clear
- **Steward PM provides immediate action clarity**
- Ready to execute immediately
- Self-monitoring active from start

**This is the "full recall protocol" that makes each new session productive from turn 1.**

## Example Output

```
📋 TELOS FULL PROJECT RECALL
============================================================

## 1. PROJECT CONTEXT (.claude_project.md)

[Full .claude_project.md displayed - all sections]

## 2. GIT STATUS

On branch: DualAttractorCanonicalImplementation
Uncommitted changes:
  M .claude_project.md
  M telos_init.py

Recent commits:
  a1b2c3d Establish mathematical PA and real-time monitoring
  d4e5f6g Update PA with comprehensive values and PBC charter
  ...

Active branches:
  * DualAttractorCanonicalImplementation
    ltff-application
    ev-application
    eu-application

## 3. KEY ARCHITECTURE

- telos_purpose/core/dual_attractor.py (Dual PA engine)
- telos_purpose/core/primacy_math.py (Math foundations)
- steward_governance_orchestrator.py (Interventions)
- telos_init.py (PA establishment)
- telos_check.py (Fidelity checking)

## 4. CURRENT PRIORITIES

High Priority Execution:
✅ Grant applications (LTFF, EV, EU)
✅ Validation studies and metrics
✅ GMU partnership (EV anchor)
✅ Observatory/TELOSCOPE demos
✅ Trail of Bits audit prep

## 5. ESTABLISHING GOVERNANCE

🔭 TELOS Session Initialization
============================================================
Reading PA from .claude_project.md...
✅ PA extracted (1634 chars)

Generating embedding (ℝ³⁸⁴ space)...
✅ Embedding generated

💾 Session PA saved
💾 Session log initialized

============================================================
✅ TELOS Session Governance ACTIVE
============================================================

📊 Primacy Attractor Established:
   Purpose: Ship TELOS to institutional deployment by Feb 2026...
   Embedding: ℝ³⁸⁴ (384-dimensional space)
   Threshold: F ≥ 0.65

🔭 Active Monitoring:
   ✅ Real-time fidelity checking enabled
   ✅ Turn-by-turn self-evaluation active
   ✅ Automatic intervention on drift

🚀 Ready for governed session.
============================================================

✅ TELOS SESSION READY - Full context loaded, governance active.
```
