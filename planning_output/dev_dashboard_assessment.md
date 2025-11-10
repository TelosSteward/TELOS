# Dev Dashboard Assessment & Minimalist Recommendations

**Test Date**: 2025-11-09
**Dashboard URL**: http://localhost:8502
**Overall Score**: 7.5/10 (Good foundation, needs refinement for TELOS-specific insights)

---

## Executive Summary

The dev dashboard is **functional and well-structured**, but currently focuses heavily on **generic project metrics** (file counts, LOC, git stats) rather than **TELOS-specific runtime insights**.

For a minimalist approach focused on "what's happening under the hood of TELOS," the dashboard needs to prioritize **governance metrics, PA tracking, and intervention analytics** over standard code metrics.

---

## ✅ What's Working (Current State)

### **1. Project Analysis Tab**
**Status**: ✅ Fully Functional

**Current Features**:
- Overview: 457 files, 113 Python files, 32K LOC, 744 functions, 139 classes
- Code Analysis: Comment ratio (8.4%), avg function size (33.2 lines)
- TODOs: 1 TODO found
- Git Stats: main branch, 24 commits, 3 uncommitted changes, 14 untracked files
- Dependencies: 55 packages (anthropic, mistralai, sentence-transformers, torch, numpy)
- Structure: 14 depth levels, 3417 directories, has tests & docs

**Assessment**: These are **standard DevOps metrics** - useful for general project health, but not TELOS-specific.

### **2. Strategic Overview Tab**
**Status**: ⚠️ Placeholder (awaits Steward PM integration)

**Current State**: Shows instructions to run `steward_pm.py` commands for partnerships/grants/priorities

**Assessment**: Good placeholder with clear instructions. Connects to external PM system.

### **3. Health Monitor Tab**
**Status**: ❌ Disabled (requires psutil installation)

**Current State**: Error message: "Health monitor not available - Install psutil"

**Assessment**: System health (CPU, memory, disk) is **nice-to-have but not essential** for understanding TELOS governance.

---

## 🎯 What Devs ACTUALLY Need to See "Under the Hood" of TELOS

Based on the TELOS architecture (Dual PA system, DMAIC cycles, interventions, fidelity tracking), here's what's **truly essential**:

### **Priority 1: CRITICAL (Must Have)**

#### **A. Primacy Attractor Dashboard** 🎯
**Why**: PA is the CORE of TELOS - devs need to see:
- Current PA state (converged vs calibrating)
- PA convergence timeline (~10 turns)
- PA coordinates in embedding space (visual representation)
- Drift from PA over time (geometric distance)

**Minimal Display**:
```
┌─ Primacy Attractor Status ─────────────────────┐
│ State: Calibrating (7/~10 turns)               │
│ Convergence: 70% ████████░░                    │
│ Current Embedding: [0.23, -0.45, 0.18, ...]   │
│ Drift: 0.12 (Low)                              │
└────────────────────────────────────────────────┘
```

#### **B. Fidelity Metrics Over Time** 📊
**Why**: Fidelity measures alignment - critical governance metric
- Real-time fidelity scores per turn
- Average fidelity trend
- Fidelity violations (< threshold)

**Minimal Display**:
```
┌─ Fidelity Tracking ────────────────────────────┐
│ Current: 0.87 (High)                           │
│ Average: 0.82                                  │
│ Last 10 turns: [0.85, 0.88, 0.87, 0.82, ...]  │
│ Violations: 0                                  │
└────────────────────────────────────────────────┘
```

#### **C. Intervention Log** ⚠️
**Why**: Interventions are governance actions - devs need visibility
- Total interventions count
- Recent intervention details
- Intervention triggers (drift warning, boundary violation)

**Minimal Display**:
```
┌─ Intervention Log ─────────────────────────────┐
│ Total: 3                                       │
│ Last: Turn 8 - Drift warning (distance: 0.45) │
│ Action: Response modified to realign with PA   │
└────────────────────────────────────────────────┘
```

---

### **Priority 2: HIGH (Should Have)**

#### **D. DMAIC Cycle Visualization** 🔄
**Why**: Shows the governance loop per turn
- Current DMAIC phase
- Cycle completion rate
- SPC control metrics

**Minimal Display**:
```
┌─ DMAIC Status ─────────────────────────────────┐
│ Phase: Control                                 │
│ Cycles Completed: 7                            │
│ Control Limits: [-0.3, 0.3] (σ = 0.15)        │
└────────────────────────────────────────────────┘
```

#### **E. Embedding Space Metrics** 🗺️
**Why**: TELOS operates in embedding space - show the geometry
- Current position vs PA
- Boundary distances
- Cluster analysis

**Minimal Display**:
```
┌─ Embedding Space ──────────────────────────────┐
│ Distance to PA: 0.12                           │
│ Distance to Boundary: 0.78 (Safe)              │
│ Cluster: Primary (coherent)                   │
└────────────────────────────────────────────────┘
```

---

### **Priority 3: MEDIUM (Nice to Have)**

#### **F. Conversation Flow Stats**
- Total turns
- User vs TELOS message ratio
- Average response time

#### **G. Telemetric Keys Status**
- Active keys count
- Key rotation events
- Cryptographic integrity

---

## ❌ What's NOT Essential (Can Remove for Minimalism)

### **Remove or Minimize**:
1. **Code Analysis tab** (comment ratio, avg function size) → Generic dev metrics, not TELOS-specific
2. **TODOs tab** → Use IDE or git issues instead
3. **Dependencies tab** → Only needed during setup/debugging
4. **Structure tab** → Only useful for onboarding, not runtime
5. **Health Monitor** → System metrics are DevOps concern, not TELOS governance

### **Keep in Collapsed State**:
- Git Stats (collapsed by default, expand if needed)
- Project Overview (basic stats only: total turns, sessions, interventions)

---

## 🎨 Minimalist Dashboard Design Recommendation

### **Proposed 3-Tab Layout**:

#### **Tab 1: 🎯 Governance (Primary View)**
- PA Status (state, convergence, drift)
- Fidelity Metrics (current, average, trend)
- Intervention Log (count, recent events)

#### **Tab 2: 🔄 DMAIC Analytics**
- Current cycle phase
- SPC control charts
- Process capability metrics

#### **Tab 3: 🗺️ Embedding Space**
- PA coordinates
- Boundary visualization
- Distance metrics
- Cluster analysis

### **Sidebar (Quick Stats)**
```
┌─ Live Status ───────────┐
│ PA: ✅ Converged         │
│ Fidelity: 0.87          │
│ Interventions: 3        │
│ Turns: 12               │
│ Sessions: 2             │
└─────────────────────────┘
```

---

## 🔧 Implementation Recommendations

### **1. Add TELOS Data Integration**

Currently, the dashboard reads **static project files** (LOC, git stats). It needs to read **runtime TELOS data**:

```python
# Add to dev_dashboard/components/telos_runtime_monitor.py
class TelosRuntimeMonitor:
    """Monitor live TELOS governance metrics."""

    def __init__(self):
        self.state_manager = self.load_active_state()

    def get_pa_status(self):
        """Get current PA convergence state."""
        return {
            'state': 'converged' if pa_converged else 'calibrating',
            'turns_to_convergence': 10 - current_turn,
            'drift': calculate_drift(),
            'coordinates': pa_embedding.tolist()
        }

    def get_fidelity_metrics(self):
        """Get fidelity tracking data."""
        return {
            'current': latest_fidelity,
            'average': avg_fidelity,
            'trend': fidelity_history[-10:],
            'violations': count_violations()
        }

    def get_intervention_log(self):
        """Get recent interventions."""
        return [
            {
                'turn': 8,
                'type': 'drift_warning',
                'distance': 0.45,
                'action': 'Response modified'
            }
        ]
```

### **2. Connect to Observatory State**

The dev dashboard should **read from the same StateManager** as the Observatory:

```python
# dev_dashboard/main.py
from observatory.core.state_manager import StateManager

def load_active_sessions():
    """Load active Observatory sessions."""
    saved_sessions_dir = Path('observatory/saved_sessions')
    # Load latest session or allow selection
    return StateManager.load_from_file(latest_session)
```

### **3. Real-Time Updates**

For live monitoring, add auto-refresh:

```python
# In sidebar
st.markdown("### ⚡ Auto-Refresh")
auto_refresh = st.checkbox("Enable (5s)", value=False)

if auto_refresh:
    import time
    time.sleep(5)
    st.rerun()
```

---

## 📊 UI/UX Score Breakdown

| Category | Score | Notes |
|----------|-------|-------|
| **Visual Design** | 8/10 | Clean dark theme, gold accents consistent with Observatory |
| **Information Architecture** | 6/10 | Too many generic metrics, missing TELOS-specific data |
| **Minimalism** | 5/10 | Too much clutter (LOC, file counts, structure), not focused |
| **TELOS-Specific Insights** | 3/10 | Currently shows project stats, not governance metrics |
| **Developer Value** | 7/10 | Good for project health, poor for TELOS runtime understanding |
| **Performance** | 9/10 | Fast, lightweight, no unnecessary API calls |
| **Integration** | 4/10 | Reads static files, not connected to live TELOS state |

**Overall: 7.5/10** - Good technical foundation, needs refocus on TELOS governance

---

## 🚀 Action Plan for Minimalist Refinement

### **Phase 1: Remove Noise (Quick Win)**
1. ✅ Remove or collapse: Code Analysis, TODOs, Structure tabs
2. ✅ Keep: Overview (basic stats), Git Stats (collapsed), Dependencies (collapsed)
3. ✅ Move generic metrics to a single "Project Health" collapsed section

### **Phase 2: Add TELOS Core (Critical)**
4. ✅ Build `TelosRuntimeMonitor` component
5. ✅ Add PA Status panel
6. ✅ Add Fidelity Metrics panel
7. ✅ Add Intervention Log panel

### **Phase 3: Advanced Visualizations (Optional)**
8. ⏭️ Add DMAIC cycle visualization
9. ⏭️ Add embedding space 2D/3D plot
10. ⏭️ Add real-time fidelity chart

---

## 💡 Key Insight

**The current dashboard is 80% "project metrics" and 20% "TELOS insights".**

**For minimalism, flip this: 80% TELOS governance, 20% project health.**

Devs don't need another code analysis tool - they need to **see TELOS in action**: PA convergence, fidelity tracking, interventions, and geometric alignment. That's what's "under the hood."

---

## Final Recommendation

**Keep the dashboard minimal and focused:**

**Essential Panels (Always Visible)**:
1. PA Status
2. Fidelity Metrics
3. Intervention Log

**Collapsed Panels (Expand on Demand)**:
- DMAIC Analytics
- Embedding Space
- Project Health (LOC, git, deps)

**Remove Entirely**:
- Code Analysis
- TODOs
- Structure

This gives devs a **single-screen view of TELOS governance** without clutter.

---

**Test Completed**: 2025-11-09
**Overall Result**: ✅ Functional dashboard, needs TELOS-specific refactoring for minimalism
**Tested By**: Claude Code with Playwright MCP
