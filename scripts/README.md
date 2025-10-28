# TELOS Automation Scripts

**Purpose**: Automation tools for TELOS project management and status tracking.

---

## Scripts

### 1. check_telos_status.py

**Purpose**: Generate comprehensive status report from TASKS.md

**Usage**:
```bash
cd ~/Desktop/telos
python3 scripts/check_telos_status.py
```

**Output**:
- Total tasks and completion percentage
- Status breakdown (✅ 🔨 ⏳ ❌)
- Section breakdown (Immediate, Short-term, Future, Completed)
- Next 5 immediate tasks
- Progress bar visualization

**Example Output**:
```
============================================================
TELOS STATUS REPORT
Generated: 2025-10-25 23:02:51
Last Updated: 2025-10-25
============================================================

📊 PROGRESS SUMMARY
------------------------------------------------------------
Total Tasks:       56
Completed:         14 (25.0%)
Remaining:         42 (75.0%)

📈 STATUS BREAKDOWN
------------------------------------------------------------
✅ Completed:      14
🔨 In Progress:    2
⏳ Pending:        40
❌ Blocked:        0

📊 COMPLETION PROGRESS
------------------------------------------------------------
[██████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 25.0%
```

---

### 2. update_task.py

**Purpose**: Update task status in TASKS.md programmatically

**Commands**:

#### Update Status
```bash
python3 scripts/update_task.py status "<task_name>" <status>
```

Statuses:
- `✅` - Complete
- `🔨` - In Progress
- `⏳` - Pending
- `❌` - Blocked

**Examples**:
```bash
# Mark as in progress
python3 scripts/update_task.py status "Build SharedSalienceExtractor" 🔨

# Mark as complete
python3 scripts/update_task.py status "Test TELOSCOPE Dashboard" ✅

# Mark as blocked
python3 scripts/update_task.py status "Deploy to production" ❌
```

#### Mark Complete (with date)
```bash
python3 scripts/update_task.py complete "<task_name>" [YYYY-MM-DD]
```

**Examples**:
```bash
# Mark complete (today's date)
python3 scripts/update_task.py complete "Test TELOSCOPE Dashboard"

# Mark complete with specific date
python3 scripts/update_task.py complete "Build PrimacyAttractor" 2025-10-22
```

#### Add New Task
```bash
python3 scripts/update_task.py add <section> "<task_description>"
```

Sections:
- `immediate` - Section 1: Immediate tasks
- `short-term` - Section 2: Short-term tasks
- `future` - Section 3: Future tasks

**Examples**:
```bash
# Add to immediate section
python3 scripts/update_task.py add immediate "Write integration tests"

# Add to short-term section
python3 scripts/update_task.py add short-term "Implement caching layer"
```

---

## Workflow Examples

### Daily Standup

```bash
# Check status
python3 scripts/check_telos_status.py

# Start working on task
python3 scripts/update_task.py status "Build SharedSalienceExtractor" 🔨

# Complete task at end of day
python3 scripts/update_task.py complete "Build SharedSalienceExtractor"

# Check status again
python3 scripts/check_telos_status.py
```

### Weekly Review

```bash
# Generate report
python3 scripts/check_telos_status.py > reports/status_$(date +%Y-%m-%d).txt

# Review completed tasks
grep "✅" TASKS.md | grep -c "Completed 2025-10"

# Plan next week
python3 scripts/update_task.py add immediate "Task for next week"
```

---

## Integration with TASKS.md

The scripts read and write TASKS.md directly. Changes are:
- Atomic (file rewritten completely)
- Timestamped (updates "Last Updated" field)
- Validated (fuzzy matching for task names)

**Important**: Always use these scripts to update TASKS.md to ensure consistency.

---

## Future Enhancements

Potential additions:
- [ ] Export to CSV/JSON
- [ ] Gantt chart generation
- [ ] Time tracking integration
- [ ] Slack/Discord notifications
- [ ] GitHub issue sync
- [ ] Burndown chart visualization

---

## Troubleshooting

### Script Not Found
```bash
# Make sure you're in the telos directory
cd ~/Desktop/telos

# Make scripts executable
chmod +x scripts/*.py
```

### Task Not Found
The script uses fuzzy matching. Try:
- More specific task name
- Partial match (e.g., "SharedSalience" instead of full name)
- Check TASKS.md manually for exact wording

### Permission Denied
```bash
chmod +x scripts/check_telos_status.py
chmod +x scripts/update_task.py
```

---

**Last Updated**: 2025-10-25
**Maintained By**: TELOS Development Team

🔭 **Making AI Governance Observable**
