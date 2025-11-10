# Dev Dashboard Integration Plan

**Date**: 2025-11-09
**Goal**: Wire dev_dashboard to TELOS runtime + Steward PM for live governance monitoring

---

## Current State Analysis

### ✅ What's Already Wired
- **Project file analysis**: Reads real LOC, file counts, dependencies from filesystem
- **Git stats**: Reads actual commits, branches, status from `.git/`
- **Dark theme**: Consistent with Observatory styling

### ❌ What's NOT Wired (Critical Gaps)
1. **TELOS Runtime Data**: No connection to Observatory StateManager
2. **Steward PM Integration**: Shows placeholder message, no actual PM data
3. **Health Monitor**: Requires `psutil` installation
4. **PA/Fidelity Metrics**: Not displayed at all
5. **Intervention Tracking**: No access to governance logs

---

## Integration Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    Dev Dashboard HUD                          │
│                    (http://localhost:8502)                    │
└─────────────┬────────────────────────────────┬───────────────┘
              │                                │
              │ Read Runtime State             │ Read PM Data
              │                                │
    ┌─────────▼─────────┐          ┌──────────▼───────────┐
    │  Observatory      │          │   Steward PM         │
    │  StateManager     │          │   (steward_pm.py)    │
    │                   │          │                      │
    │  - PA status      │          │  - Partnerships      │
    │  - Fidelity       │          │  - Grants            │
    │  - Interventions  │          │  - Priorities        │
    │  - Turn data      │          │  - Task tracking     │
    └───────────────────┘          └──────────────────────┘
```

---

## Phase 1: Wire TELOS Runtime Metrics

### **Step 1: Create TELOS Data Bridge**

**New File**: `dev_dashboard/components/telos_bridge.py`

```python
"""
TELOS Bridge - Connects dev_dashboard to Observatory StateManager.
Provides real-time access to PA status, fidelity, interventions.
"""

import sys
from pathlib import Path

# Add Observatory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'observatory'))

from observatory.core.state_manager import StateManager
import streamlit as st


class TelosBridge:
    """Bridge between dev_dashboard and Observatory runtime state."""

    def __init__(self):
        """Initialize bridge to Observatory state."""
        self.state_manager = None
        self.session_loaded = False

    def load_active_session(self, session_path=None):
        """Load active Observatory session or latest saved session."""
        if session_path is None:
            # Auto-detect latest session
            session_path = self._find_latest_session()

        if session_path and session_path.exists():
            self.state_manager = StateManager()
            # Load session data
            import json
            with open(session_path, 'r') as f:
                session_data = json.load(f)
            self.state_manager.load_from_session(session_data)
            self.session_loaded = True
            return True

        return False

    def _find_latest_session(self):
        """Find most recent Observatory session file."""
        sessions_dir = Path(__file__).parent.parent.parent / 'observatory' / 'saved_sessions'
        if not sessions_dir.exists():
            return None

        session_files = list(sessions_dir.glob('session_*.json'))
        if not session_files:
            return None

        # Return most recent
        return max(session_files, key=lambda p: p.stat().st_mtime)

    def get_pa_status(self):
        """Get current Primacy Attractor status."""
        if not self.session_loaded or not self.state_manager:
            return {
                'state': 'not_loaded',
                'converged': False,
                'turns_to_convergence': None,
                'drift': None
            }

        state = self.state_manager.state
        pa_converged = getattr(state, 'pa_converged', False)
        current_turn = state.current_turn

        return {
            'state': 'converged' if pa_converged else 'calibrating',
            'converged': pa_converged,
            'turns_to_convergence': max(0, 10 - current_turn) if not pa_converged else 0,
            'current_turn': current_turn,
            'total_turns': state.total_turns,
            'drift': None  # TODO: Calculate from embedding distance
        }

    def get_fidelity_metrics(self):
        """Get fidelity tracking metrics."""
        if not self.session_loaded or not self.state_manager:
            return {
                'current': None,
                'average': None,
                'trend': [],
                'violations': 0
            }

        state = self.state_manager.state

        # Extract fidelity from turns
        fidelity_scores = [
            turn.get('fidelity', 0.0)
            for turn in state.turns
            if 'fidelity' in turn
        ]

        if not fidelity_scores:
            return {
                'current': None,
                'average': None,
                'trend': [],
                'violations': 0
            }

        current_fidelity = fidelity_scores[-1] if fidelity_scores else None
        avg_fidelity = sum(fidelity_scores) / len(fidelity_scores)
        violations = sum(1 for f in fidelity_scores if f < 0.7)

        return {
            'current': round(current_fidelity, 3) if current_fidelity else None,
            'average': round(avg_fidelity, 3),
            'trend': fidelity_scores[-10:],  # Last 10 turns
            'violations': violations
        }

    def get_intervention_log(self):
        """Get intervention history."""
        if not self.session_loaded or not self.state_manager:
            return []

        state = self.state_manager.state

        interventions = []
        for turn in state.turns:
            if turn.get('intervention_applied', False):
                interventions.append({
                    'turn': turn.get('turn_number', 0),
                    'type': turn.get('intervention_type', 'unknown'),
                    'reason': turn.get('intervention_reason', 'N/A'),
                    'distance': turn.get('distance_to_pa', None)
                })

        return interventions

    def get_session_stats(self):
        """Get overall session statistics."""
        if not self.session_loaded or not self.state_manager:
            return {
                'session_id': 'No session loaded',
                'total_turns': 0,
                'total_interventions': 0,
                'avg_fidelity': None
            }

        state = self.state_manager.state

        return {
            'session_id': state.session_id,
            'total_turns': state.total_turns,
            'total_interventions': state.total_interventions,
            'avg_fidelity': round(state.avg_fidelity, 3) if state.avg_fidelity else None,
            'drift_warnings': getattr(state, 'drift_warnings', 0)
        }
```

### **Step 2: Create TELOS Governance Panel**

**New File**: `dev_dashboard/components/telos_governance.py`

```python
"""TELOS Governance Panel - Shows runtime PA, fidelity, interventions."""

import streamlit as st
from .telos_bridge import TelosBridge


class TelosGovernance:
    """Minimalist TELOS governance dashboard."""

    def __init__(self):
        """Initialize with TELOS bridge."""
        if 'telos_bridge' not in st.session_state:
            st.session_state.telos_bridge = TelosBridge()
            # Auto-load latest session on init
            st.session_state.telos_bridge.load_active_session()

        self.bridge = st.session_state.telos_bridge

    def render(self):
        """Render TELOS governance view."""
        st.markdown("## 🎯 TELOS Governance Monitor")
        st.markdown("*Real-time PA tracking, fidelity metrics, and intervention logs*")

        # Session selector
        col1, col2 = st.columns([3, 1])
        with col1:
            if self.bridge.session_loaded:
                stats = self.bridge.get_session_stats()
                st.success(f"✅ Loaded: {stats['session_id']}")
            else:
                st.warning("⚠️ No active session - Start Observatory to generate data")

        with col2:
            if st.button("🔄 Reload", use_container_width=True):
                self.bridge.load_active_session()
                st.rerun()

        st.markdown("---")

        # Three-column minimalist layout
        col1, col2, col3 = st.columns(3)

        with col1:
            self.render_pa_status()

        with col2:
            self.render_fidelity_metrics()

        with col3:
            self.render_intervention_summary()

        st.markdown("---")

        # Detailed intervention log
        self.render_intervention_log()

    def render_pa_status(self):
        """Render PA status panel."""
        st.markdown("### 🎯 Primacy Attractor")

        pa_status = self.bridge.get_pa_status()

        if pa_status['state'] == 'not_loaded':
            st.info("No PA data available")
            return

        # Status indicator
        if pa_status['converged']:
            st.success("✅ Converged")
        else:
            st.warning(f"⏳ Calibrating ({pa_status['current_turn']}/~10)")

        # Metrics
        st.metric("Current Turn", pa_status['current_turn'])
        st.metric("Total Turns", pa_status['total_turns'])

        if not pa_status['converged']:
            progress = min(pa_status['current_turn'] / 10.0, 1.0)
            st.progress(progress, text=f"{int(progress*100)}% to convergence")

    def render_fidelity_metrics(self):
        """Render fidelity metrics panel."""
        st.markdown("### 📊 Fidelity Tracking")

        metrics = self.bridge.get_fidelity_metrics()

        if metrics['current'] is None:
            st.info("No fidelity data available")
            return

        # Current fidelity with color coding
        if metrics['current'] >= 0.8:
            st.success(f"Current: {metrics['current']:.3f}")
        elif metrics['current'] >= 0.6:
            st.warning(f"Current: {metrics['current']:.3f}")
        else:
            st.error(f"Current: {metrics['current']:.3f}")

        st.metric("Average", metrics['average'])
        st.metric("Violations", metrics['violations'],
                 delta="Low" if metrics['violations'] == 0 else "Check logs")

        # Mini trend chart
        if metrics['trend']:
            st.line_chart(metrics['trend'])

    def render_intervention_summary(self):
        """Render intervention summary panel."""
        st.markdown("### ⚠️ Interventions")

        interventions = self.bridge.get_intervention_log()
        stats = self.bridge.get_session_stats()

        st.metric("Total", stats['total_interventions'])

        if interventions:
            latest = interventions[-1]
            st.info(f"""
            **Latest**: Turn {latest['turn']}
            Type: {latest['type']}
            """)
        else:
            st.success("No interventions needed")

    def render_intervention_log(self):
        """Render detailed intervention log."""
        st.markdown("### 📋 Intervention Log")

        interventions = self.bridge.get_intervention_log()

        if not interventions:
            st.info("No interventions recorded in this session")
            return

        # Display as table
        for intervention in reversed(interventions):  # Most recent first
            with st.expander(f"Turn {intervention['turn']} - {intervention['type']}"):
                st.markdown(f"""
                **Reason**: {intervention['reason']}
                **Distance to PA**: {intervention['distance']:.3f if intervention['distance'] else 'N/A'}
                """)
```

### **Step 3: Update main.py to Use TELOS Governance**

**File**: `dev_dashboard/main.py`

```python
# Add import
from components.telos_governance import TelosGovernance

# Update navigation
with st.sidebar:
    st.markdown("### 🧭 Navigation")

    # TELOS-focused navigation
    if st.button("🎯 TELOS Governance", use_container_width=True):
        st.session_state.active_tab = 'telos_governance'

    if st.button("🔬 Project Analysis", use_container_width=True):
        st.session_state.active_tab = 'real_analysis'

    if st.button("🎯 Strategic Overview", use_container_width=True):
        st.session_state.active_tab = 'strategic'

# Update main content rendering
if st.session_state.active_tab == 'telos_governance':
    show_telos_governance()

# Add function
def show_telos_governance():
    """Display TELOS governance monitor."""
    gov = TelosGovernance()
    gov.render()
```

---

## Phase 2: Wire Steward PM Integration

### **Step 1: Create Steward PM Bridge**

**New File**: `dev_dashboard/components/steward_bridge.py`

```python
"""Steward PM Bridge - Connects to Steward Project Manager data."""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add steward to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'steward'))


class StewardBridge:
    """Bridge to Steward PM data."""

    def __init__(self):
        """Initialize Steward PM bridge."""
        self.steward_data_dir = Path(__file__).parent.parent.parent / 'steward' / 'data'
        self.steward_data_dir.mkdir(parents=True, exist_ok=True)

    def get_partnerships(self):
        """Get partnership tracking data."""
        partnerships_file = self.steward_data_dir / 'partnerships.json'

        if not partnerships_file.exists():
            return {
                'active': [],
                'pending': [],
                'total': 0
            }

        with open(partnerships_file, 'r') as f:
            data = json.load(f)

        return data

    def get_grants(self):
        """Get grant application status."""
        grants_file = self.steward_data_dir / 'grants.json'

        if not grants_file.exists():
            return {
                'active': [],
                'submitted': [],
                'awarded': [],
                'total_value': 0
            }

        with open(grants_file, 'r') as f:
            data = json.load(f)

        return data

    def get_priorities(self):
        """Get current priorities from Steward PM."""
        priorities_file = self.steward_data_dir / 'priorities.json'

        if not priorities_file.exists():
            return {
                'high': [],
                'medium': [],
                'low': []
            }

        with open(priorities_file, 'r') as f:
            data = json.load(f)

        return data

    def sync_from_steward_pm(self):
        """Run Steward PM to sync latest data."""
        # This would call steward_pm.py commands
        # For now, just read existing data
        return {
            'partnerships': self.get_partnerships(),
            'grants': self.get_grants(),
            'priorities': self.get_priorities(),
            'last_sync': datetime.now().isoformat()
        }
```

### **Step 2: Update Strategic View**

**File**: `dev_dashboard/components/strategic_view.py`

Add Steward PM integration:

```python
from .steward_bridge import StewardBridge

class StrategicView:
    def __init__(self):
        self.bridge = StewardBridge()

    def render(self):
        st.markdown("## 🎯 TELOS Strategic Overview")

        # Sync button
        if st.button("🔄 Sync with Steward PM"):
            data = self.bridge.sync_from_steward_pm()
            st.success(f"✅ Synced at {data['last_sync']}")

        # Display partnerships
        partnerships = self.bridge.get_partnerships()
        if partnerships['total'] > 0:
            st.markdown("### 🤝 Partnerships")
            st.metric("Active", len(partnerships['active']))
            st.metric("Pending", len(partnerships['pending']))

        # Display grants
        grants = self.bridge.get_grants()
        if grants['total_value'] > 0:
            st.markdown("### 💰 Grants")
            st.metric("Total Value", f"${grants['total_value']:,}")

        # Display priorities
        priorities = self.bridge.get_priorities()
        if priorities['high']:
            st.markdown("### 🎯 Current Priorities")
            for priority in priorities['high']:
                st.warning(f"🔴 {priority}")
```

---

## Phase 3: Install Health Monitor (Optional)

```bash
pip install psutil
```

Then health monitor will auto-activate showing CPU, memory, disk, network stats.

---

## Implementation Priority

### **Critical (Must Have)**
1. ✅ TELOS Bridge (`telos_bridge.py`) - **PRIORITY 1**
2. ✅ TELOS Governance Panel (`telos_governance.py`) - **PRIORITY 1**
3. ✅ Update main.py navigation - **PRIORITY 1**

### **High (Should Have)**
4. ✅ Steward PM Bridge (`steward_bridge.py`) - **PRIORITY 2**
5. ✅ Update Strategic View with PM data - **PRIORITY 2**

### **Optional (Nice to Have)**
6. ⏭️ Install psutil for Health Monitor - **PRIORITY 3**
7. ⏭️ Add DMAIC cycle visualization - **PRIORITY 4**
8. ⏭️ Add embedding space 2D/3D plots - **PRIORITY 4**

---

## Data Flow Summary

```
Observatory Session (running)
    ↓ Saves to
saved_sessions/session_*.json
    ↓ Read by
dev_dashboard/TelosBridge
    ↓ Displays
PA Status, Fidelity, Interventions

Steward PM (steward_pm.py)
    ↓ Saves to
steward/data/*.json
    ↓ Read by
dev_dashboard/StewardBridge
    ↓ Displays
Partnerships, Grants, Priorities
```

---

## Next Steps

1. Create `dev_dashboard/components/telos_bridge.py`
2. Create `dev_dashboard/components/telos_governance.py`
3. Update `dev_dashboard/main.py` with new navigation
4. Test with active Observatory session
5. Create `dev_dashboard/components/steward_bridge.py`
6. Update `dev_dashboard/components/strategic_view.py`
7. Test Steward PM sync

---

**Document Created**: 2025-11-09
**Ready to Implement**: YES
