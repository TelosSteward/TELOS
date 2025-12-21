---
description: Check TELOS governance monitoring status
---

# Monitor Status

Check current TELOS governance monitoring status for this Claude Code session.

## Implementation

**1. Check if PA is established:**

```bash
python3 -c "
import json
from pathlib import Path

if Path('.telos_session_pa.json').exists():
    with open('.telos_session_pa.json') as f:
        pa = json.load(f)
    print('✅ PA Established')
    print(f'   Established: {pa[\"established_at\"]}')
    print(f'   Threshold: F ≥ {pa[\"threshold\"]}')

    if Path('.telos_session_log.json').exists():
        with open('.telos_session_log.json') as f:
            log = json.load(f)
        turns = len(log.get('turns', []))
        if turns > 0:
            print(f'\n📊 Session Metrics:')
            print(f'   Turns logged: {turns}')
            recent = log['turns'][-1]
            print(f'   Latest fidelity: {recent[\"fidelity\"]:.3f}')
            status = '✅' if recent['passed'] else '🚨'
            print(f'   Status: {status}')
else:
    print('❌ PA not established')
    print('\nRun: /telos to establish governance')
"
```

**2. Display status summary:**

```
🔭 TELOS Governance Monitoring Status

Session PA:
  Purpose: [first 100 chars from .claude_project.md]
  Fidelity Threshold: 0.65
  Status: ✅ Active | ❌ Not Active

Monitoring:
  Real-time fidelity: [Enabled/Disabled]
  Turn-by-turn logging: [Yes/No]
  Current turn: X
  Mean fidelity: X.XXX

Dashboard:
  Command: ./launch_dashboard.sh
  URL: http://localhost:8501

Sessions Directory: sessions/
Latest Session: [filename if available]
```

---

**Use `/monitor-status` to quickly check if governance is active.**
