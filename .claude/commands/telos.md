---
description: Full project recall - Refresh context and establish TELOS governance
---

# TELOS Full Project Recall

Complete project context refresh for new Claude Code sessions + mathematical PA establishment + real-time governance activation.

## PART 1: PROJECT CONTEXT REFRESH

**1. Display Full .claude_project.md**

Read the entire `.claude_project.md` file and display it to refresh on:
- Project overview, core innovations, grant strategy
- Session PA, core values, scope, boundaries, PBC charter
- What TELOS is, what we're building, and how to govern

**2. Git Status & Recent Work**

Run:
```bash
git status
git log --oneline -5
git branch -a
```

Show: Current branch, uncommitted files, recent commits, active branches

**3. Current Priorities**

Highlight from .claude_project.md:
- Grant applications (LTFF $150K, EV $400K, EU €350K)
- Validation studies and metrics
- GMU partnership (institutional anchor for EV)
- Observatory/TELOSCOPE demos
- Trail of Bits audit prep

## PART 2: ESTABLISH MATHEMATICAL GOVERNANCE

**4. Run PA Establishment**

Execute:
```bash
python3 telos_init.py
```

This:
- Extracts Session Purpose from `.claude_project.md`
- Generates embedding (ℝ³⁸⁴ mathematical PA center)
- Stores in `.telos_session_pa.json`
- Initializes `.telos_session_log.json`
- Confirms governance active

**5. Confirm Readiness**

Show summary:
```
✅ TELOS SESSION READY

📊 Project Context Refreshed:
   - PA, values, scope, boundaries loaded
   - Git status checked
   - Priorities identified

🔭 Governance Active:
   - Mathematical PA established (ℝ³⁸⁴)
   - Real-time fidelity monitoring enabled
   - Threshold: F ≥ 0.65

🚀 Current Focus:
   [Top 3 priorities from .claude_project.md]

Ready to execute under full TELOS governance.
```

## PART 3: PROACTIVE SELF-MONITORING

After `/telos`, I should:

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

---

**Use `/telos` at start of EVERY new Claude Code session for full recall protocol.**
