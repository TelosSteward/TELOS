# TELOS Observatory - Phase 1

**Frame-by-Frame AI Governance Analysis Platform**

Version: 1.0.0-phase1
Status: ✅ Ready for Testing
Date: 2025-10-30

---

## What is This?

The **TELOS Observatory** is a standalone research platform for analyzing AI governance conversations frame-by-frame. Think of it like a video player for AI conversations, where you can:

- **Scrub through conversation history** like a timeline
- **See governance metrics for each turn** (fidelity, drift, interventions)
- **Navigate with TELOSCOPE controls** (Prev/Play/Pause/Next buttons)
- **Visualize conversation flow** with distance-based dimming

Phase 1 focuses on the core navigation and visualization infrastructure.

---

## Quick Start

### Run the Observatory

```bash
cd ~/Desktop/TELOS
./venv/bin/streamlit run telos_observatory/main_observatory.py
```

The app will open at `http://localhost:8501`

### What You'll See

1. **Observation Deck** (main viewport)
   - Session info: Session ID, total turns, avg fidelity, interventions
   - Scrollable conversation with 12 mock turns
   - Turns dim based on distance from active turn (focus effect)
   - Active turn has gold border and full brightness

2. **TELOSCOPE Controls** (bottom, fixed position)
   - Timeline scrubber with color-coded markers
   - Navigation buttons: ⏮️ First | ◀️ Prev | ▶️ Play | Next ▶️ | Last ⏭️
   - Turn counter showing X / Y
   - Legend explaining marker colors

3. **Control Strip** (top-right, fixed position)
   - Current turn number
   - Fidelity score for active turn
   - Governance status

---

## File Structure

```
telos_observatory/
├── main_observatory.py              # Entry point
├── mock_data.py                     # 12-turn mock session
├── observation_deck/
│   ├── deck_interface.py            # Main deck orchestrator
│   └── turn_renderer.py             # Turn display with dimming
├── teloscope/
│   ├── teloscope_controller.py      # Main TELOSCOPE interface
│   ├── navigation_controls.py       # Prev/Play/Next buttons
│   └── timeline_scrubber.py         # Interactive timeline
└── docs/
    └── streamlit_patterns.md        # Implementation patterns
```

---

## Mock Data

The Phase 1 app uses 12 mock turns demonstrating typical governance patterns:

| Turns | Phase | Fidelity | Description |
|-------|-------|----------|-------------|
| 1-3 | Calibration | None | Establishing primacy attractor |
| 4-6 | Stable | 0.85-0.92 | Normal governance |
| 7 | Drift | 0.68 | User asks financial advice (out of scope) |
| 8 | Intervention | 0.81 | Governance corrects scope |
| 9-12 | Recovered | 0.88-0.91 | Back to stable governance |

---

## Features

### ✅ Implemented (Phase 1)

1. **Turn-by-Turn Navigation**
   - Scrub timeline to any turn
   - Navigate with buttons (First/Prev/Next/Last)
   - Auto-play mode (2 seconds per turn)

2. **Distance-Based Dimming**
   - Active turn: opacity 1.0, gold border
   - Adjacent (±1): opacity 0.7
   - Two away (±2): opacity 0.4
   - Far (±3+): opacity 0.2
   - Smooth CSS transitions (0.3s ease)

3. **Timeline Visualization**
   - Color-coded markers:
     - ✓ Green: Stable turns
     - ⚠️ Orange: Drift detected
     - ⚡ Yellow: Intervention applied
     - ⚙️ Blue: Calibration phase
   - Interactive slider
   - Legend

4. **Status Display**
   - Real-time fidelity for active turn
   - Governance status indicators
   - Session statistics

### 🔨 Planned (Phase 2)

- Comparison Viewer (TELOS vs Baseline side-by-side)
- Calculation Window (mathematical transparency)
- Steward Chat (AI research assistant)
- Integration with live TELOS sessions

---

## Testing Checklist

### Manual Testing

Run the app and verify:

- [ ] **Timeline scrubbing works**
  - Drag slider, viewport scrolls to turn
  - Active turn gets gold border
  - Dimming updates correctly

- [ ] **Navigation buttons work**
  - First/Prev/Next/Last navigate correctly
  - Buttons disable at boundaries (First at turn 1, Last at turn 12)

- [ ] **Play mode works**
  - Click Play, turns advance automatically
  - Button changes to Pause
  - Stops at last turn

- [ ] **Dimming algorithm correct**
  - Active turn: full brightness, gold border
  - Adjacent: slightly dimmed
  - Far turns: very dim but visible

- [ ] **Control strip updates**
  - Shows correct turn number (1-12)
  - Fidelity updates per turn
  - Status matches turn data

- [ ] **Performance**
  - Smooth scrolling
  - No lag when scrubbing
  - Transitions are smooth (0.3s)

---

## Architecture

### Streamlit Patterns

The implementation follows strict Streamlit patterns (see `docs/streamlit_patterns.md`):

1. **Session State First**
   - All UI state in `st.session_state`
   - Initialized in `init_session_state()`
   - Never use local variables for state

2. **Callback Pattern**
   - Callbacks ONLY update state
   - NO rendering in callbacks
   - Always call `st.rerun()` after state change

3. **Widget Keys**
   - Every widget has unique key
   - Required for loops, multiple buttons

4. **Layout Management**
   - Complete all columns before next section
   - Max 3 levels of nesting

### Data Flow

```
User Action (Button/Slider)
    ↓
Callback updates st.session_state.current_turn
    ↓
st.rerun() called
    ↓
Streamlit re-executes main()
    ↓
render_chat_viewport() reads current_turn
    ↓
turn_renderer.py applies dimming + scrolls
    ↓
User sees updated UI
```

---

## Integration with Steward PM

Phase 1 tasks are tracked in:

- **`STEWARD.md`**: Current focus section updated
- **`docs/prd/TASKS.md`**: Section 1B with 10/11 tasks completed (91%)

Run Steward status:

```bash
cd ~/Desktop/TELOS
python3 steward.py status
```

---

## Troubleshooting

### App won't start

```bash
# Check Python environment
cd ~/Desktop/TELOS
./venv/bin/python3 --version

# Reinstall streamlit if needed
./venv/bin/pip install streamlit
```

### Import errors

```bash
# Ensure you're running from TELOS directory
cd ~/Desktop/TELOS
./venv/bin/streamlit run telos_observatory/main_observatory.py
```

### State not persisting

- Make sure you're using `st.session_state.variable_name`
- NOT local variables like `variable_name = value`

---

## Next Steps

### For Development

1. **Test Phase 1**
   ```bash
   ./venv/bin/streamlit run telos_observatory/main_observatory.py
   ```

2. **Review Patterns**
   - Read `docs/streamlit_patterns.md`
   - Understand session state management
   - Follow callback pattern strictly

3. **Plan Phase 2**
   - Comparison Viewer wireframe
   - Calculation Window data sources
   - Integration strategy with main dashboard

### For Integration

Phase 2 will integrate Observatory with existing `streamlit_live_comparison.py`:

- Replace tab-based UI with Observatory layout
- Connect to real LiveInterceptor data
- Add Comparison Viewer for counterfactual analysis
- Add Mathematical Detail window

---

## Documentation

- **Streamlit Patterns**: `docs/streamlit_patterns.md` (complete guide)
- **Project Tracking**: `docs/prd/TASKS.md` (Section 1B, Phase 1)
- **Steward Status**: `STEWARD.md` (Current Focus section)

---

## Credits

Built with:
- **Streamlit**: Web framework
- **Python 3.9+**: Backend
- **TELOS Framework**: Governance system

**Status**: Phase 1 Complete ✅
**Next**: Live testing and Phase 2 planning

---

**Questions?** Check `docs/streamlit_patterns.md` or run `python3 steward.py status`
