# TELOSCOPE Control System - Architecture Documentation

**Version**: 1.0.0-phase1
**Last Updated**: 2025-10-30
**Status**: Phase 1 Complete

---

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Component Hierarchy](#component-hierarchy)
4. [Data Flow](#data-flow)
5. [State Management](#state-management)
6. [Navigation System](#navigation-system)
7. [Timeline System](#timeline-system)
8. [Integration Points](#integration-points)
9. [Extension Architecture](#extension-architecture)

---

## Overview

**TELOSCOPE** (Telically Entrained Linguistic Operational Substrate Counterfactual Observation via Purpose-scoped Experimentation) is the control system for frame-by-frame analysis of AI governance conversations.

### Purpose

TELOSCOPE provides researchers with:
- **Temporal Navigation**: Move through conversation history like video playback
- **Visual Timeline**: See governance state across all turns at a glance
- **Playback Control**: Auto-play mode for presentation and analysis
- **Turn Synchronization**: Coordinate all Observatory components to same turn

### Metaphor

Think of TELOSCOPE as a **video player** for AI conversations:
- Timeline scrubber = seek bar
- Prev/Play/Next = standard playback controls
- Turn markers = chapter markers
- Observation Deck = video viewport

---

## System Architecture

### Three-Layer Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    TELOSCOPE Controller                  │
│              (teloscope_controller.py)                   │
│                                                          │
│  Orchestrates: Navigation + Timeline + Autoplay         │
└─────────────────────────────────────────────────────────┘
                           │
        ┌──────────────────┴──────────────────┐
        │                                     │
┌───────▼───────────┐              ┌─────────▼──────────┐
│  Navigation Layer │              │   Timeline Layer   │
│  ─────────────── │              │  ──────────────── │
│ - First/Prev    │              │ - Scrubber Slider │
│ - Play/Pause    │              │ - Turn Markers    │
│ - Next/Last     │              │ - Legend Display  │
│ - Autoplay      │              │ - Position Sync   │
└───────┬───────────┘              └─────────┬──────────┘
        │                                     │
        └──────────────────┬──────────────────┘
                           │
                ┌──────────▼──────────┐
                │   State Manager     │
                │  ───────────────   │
                │ st.session_state   │
                │  - current_turn    │
                │  - playing         │
                │  - session_data    │
                └────────────────────┘
```

### Component Files

```
teloscope/
├── __init__.py
├── teloscope_controller.py       # Main orchestrator
├── navigation_controls.py        # Playback buttons
└── timeline_scrubber.py          # Interactive timeline
```

---

## Component Hierarchy

### 1. TELOSCOPE Controller (Top Level)

**File**: `teloscope/teloscope_controller.py`

**Responsibilities**:
- Orchestrate navigation + timeline rendering
- Handle autoplay timing logic
- Provide fixed-position glassmorphism container
- Expose status API for debugging

**API**:
```python
def render_teloscope():
    """
    Main entry point - renders complete TELOSCOPE interface.
    Fixed position at bottom of viewport.
    """

def get_teloscope_status() -> dict:
    """
    Get current TELOSCOPE status.
    Returns: {current_turn, total_turns, playing, playback_speed, session_id}
    """

def handle_autoplay():
    """
    Check if turn should advance during play mode.
    Called every render cycle.
    """
```

**Dependencies**:
- `navigation_controls.render_navigation_controls()`
- `navigation_controls.handle_autoplay()`
- `timeline_scrubber.render_timeline_scrubber()`
- `timeline_scrubber.render_timeline_legend()`

---

### 2. Navigation Controls (Mid Level)

**File**: `teloscope/navigation_controls.py`

**Responsibilities**:
- Render playback buttons (First/Prev/Play/Next/Last)
- Handle button click callbacks (update state only)
- Manage autoplay timing and advancement
- Disable buttons at boundaries

**Callbacks** (State Updates Only):
```python
def on_first_turn():
    """Jump to turn 0"""
    st.session_state.current_turn = 0
    st.rerun()

def on_previous_turn():
    """Go back one turn"""
    if st.session_state.current_turn > 0:
        st.session_state.current_turn -= 1
    st.rerun()

def on_next_turn():
    """Advance one turn"""
    max_turns = len(st.session_state.session_data['turns']) - 1
    if st.session_state.current_turn < max_turns:
        st.session_state.current_turn += 1
    st.rerun()

def on_last_turn():
    """Jump to last turn"""
    st.session_state.current_turn = max_turns
    st.rerun()

def on_toggle_play():
    """Toggle play/pause state"""
    st.session_state.playing = not st.session_state.playing
    if st.session_state.playing:
        st.session_state.last_play_time = time.time()
    st.rerun()
```

**Render Function**:
```python
def render_navigation_controls():
    """
    Render 5-button navigation row.

    Layout: [First] [Prev] [Play] [Next] [Last]

    Each button:
    - Unique key (nav_first, nav_prev, etc.)
    - on_click callback (state update only)
    - disabled at boundaries
    """
```

**Autoplay Logic**:
```python
def handle_autoplay():
    """
    Check timing and advance turn if needed.

    Logic:
    1. Only run if st.session_state.playing is True
    2. Check time since last_play_time
    3. If interval elapsed, advance turn
    4. Stop at last turn
    5. st.rerun() to update UI
    """
```

---

### 3. Timeline Scrubber (Mid Level)

**File**: `teloscope/timeline_scrubber.py`

**Responsibilities**:
- Render interactive slider
- Generate color-coded turn markers
- Display turn counter (X / Y)
- Show legend for marker colors

**Callback**:
```python
def on_timeline_change():
    """
    Handle timeline slider drag.

    Streamlit auto-syncs slider to st.session_state.timeline_slider
    because we use key="timeline_slider"
    """
    new_turn = st.session_state.timeline_slider
    st.session_state.current_turn = new_turn
    st.rerun()
```

**Render Functions**:
```python
def render_timeline_scrubber():
    """
    Main timeline rendering:
    1. Generate turn markers (HTML)
    2. Render slider widget (key="timeline_slider")
    3. Display turn counter
    """

def get_turn_marker_html(turns: list, current_turn: int) -> str:
    """
    Generate HTML for visual markers above slider.

    Returns: HTML string with colored dots

    Marker sizes:
    - Active turn: 12px
    - Other turns: 8px

    Marker colors:
    - ✓ #90EE90 (green) - Stable
    - ⚠️ #FFA500 (orange) - Drift
    - ⚡ #FFD700 (gold) - Intervention
    - ⚙️ #6495ED (blue) - Calibration
    """

def render_timeline_legend():
    """
    Display legend explaining marker colors.
    Compact, bottom of TELOSCOPE.
    """
```

---

## Data Flow

### User Interaction Flow

```
┌──────────────────────────────────────────────────────────┐
│ User Action: Click "Next" button                         │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│ Callback: on_next_turn()                                 │
│   - Update: st.session_state.current_turn += 1          │
│   - Trigger: st.rerun()                                  │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│ Streamlit Re-executes main()                             │
│   - init_session_state() (skipped, already initialized)  │
│   - render_observation_deck()                            │
│   - render_teloscope()                                   │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│ Observation Deck Reads State                             │
│   - active_turn = st.session_state.current_turn          │
│   - Apply dimming based on distance                      │
│   - Scroll to active turn                                │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│ TELOSCOPE Reads State                                    │
│   - Update timeline slider position                      │
│   - Highlight active turn marker                         │
│   - Update button disabled states                        │
└──────────────────────────────────────────────────────────┘
                       │
                       ▼
                  [User Sees Updated UI]
```

### Timeline Scrubber Flow

```
┌──────────────────────────────────────────────────────────┐
│ User Action: Drag timeline slider                        │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│ Streamlit Auto-Updates                                   │
│   - st.session_state.timeline_slider = new_value        │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│ Callback: on_timeline_change()                           │
│   - Read: st.session_state.timeline_slider              │
│   - Update: st.session_state.current_turn               │
│   - Trigger: st.rerun()                                  │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
                  [Same rerun flow as above]
```

### Autoplay Flow

```
┌──────────────────────────────────────────────────────────┐
│ User Clicks: Play button                                 │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│ Callback: on_toggle_play()                               │
│   - Toggle: st.session_state.playing = True             │
│   - Record: st.session_state.last_play_time = now       │
│   - Trigger: st.rerun()                                  │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│ Every Rerun: handle_autoplay() called                    │
│   - Check: if playing == True                            │
│   - Check: if time_elapsed >= interval                   │
│   - If yes: advance turn, st.rerun()                     │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
           [Loop continues until last turn]
```

---

## State Management

### Session State Variables

TELOSCOPE relies on these `st.session_state` variables:

```python
# Navigation State
st.session_state.current_turn: int          # Active turn index (0-based)
st.session_state.playing: bool              # Play/pause state
st.session_state.playback_speed: float      # Speed multiplier (default 1.0)
st.session_state.last_play_time: float      # Timestamp for autoplay timing

# Data State
st.session_state.session_data: dict         # Full session with turns
st.session_state.initialized: bool          # Init flag

# UI State (future)
st.session_state.telescope_open: bool       # TELOSCOPE visibility toggle
```

### State Initialization

**Location**: `main_observatory.py:init_session_state()`

```python
def init_session_state():
    """
    Initialize ALL session state - called FIRST in main().

    CRITICAL: Uses if 'initialized' not in st.session_state guard
    to prevent re-initialization on reruns.
    """
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.current_turn = 0
        st.session_state.playing = False
        st.session_state.playback_speed = 1.0
        st.session_state.last_play_time = 0
        st.session_state.telescope_open = True
        st.session_state.session_data = generate_mock_session()
```

### State Update Rules

**Rule 1**: Callbacks ONLY update state
```python
# ✅ CORRECT
def on_next_turn():
    st.session_state.current_turn += 1
    st.rerun()

# ❌ WRONG
def on_next_turn():
    st.session_state.current_turn += 1
    st.write("Updated!")  # NO RENDERING IN CALLBACKS!
```

**Rule 2**: Always call `st.rerun()` after state change
```python
# ✅ CORRECT
st.session_state.current_turn += 1
st.rerun()  # Triggers re-render

# ❌ WRONG
st.session_state.current_turn += 1
# Missing st.rerun() - UI won't update!
```

**Rule 3**: Read state in render functions, not callbacks
```python
# ✅ CORRECT PATTERN
def on_button_click():
    st.session_state.value += 1  # Update only
    st.rerun()

def render_ui():
    current = st.session_state.value  # Read during render
    st.write(f"Value: {current}")

# ❌ WRONG
def on_button_click():
    st.session_state.value += 1
    current = st.session_state.value  # Don't read in callback
    st.write(current)  # Don't render in callback
```

---

## Navigation System

### Button Layout

```
┌───────┬───────┬───────┬───────┬───────┐
│ First │ Prev  │ Play  │ Next  │ Last  │
│  ⏮️   │  ◀️   │  ▶️   │  ▶️   │  ⏭️   │
└───────┴───────┴───────┴───────┴───────┘
```

### Button Behavior

| Button | Action | Disabled When |
|--------|--------|---------------|
| First | Jump to turn 0 | Already at turn 0 |
| Prev | Go back one turn | At turn 0 |
| Play/Pause | Toggle autoplay | Never (always enabled) |
| Next | Advance one turn | At last turn |
| Last | Jump to last turn | Already at last turn |

### Play Button States

```python
# Playing
[⏸️ Pause]  # Primary button, autoplay active

# Paused
[▶️ Play]   # Secondary button, autoplay stopped
```

### Autoplay Algorithm

```python
def handle_autoplay():
    # Only run if playing
    if not st.session_state.playing:
        return

    # Get timing parameters
    current_turn = st.session_state.current_turn
    max_turns = len(st.session_state.session_data['turns']) - 1
    last_play_time = st.session_state.last_play_time
    playback_speed = st.session_state.playback_speed

    # Stop at end
    if current_turn >= max_turns:
        st.session_state.playing = False
        st.rerun()
        return

    # Check timing
    advance_interval = 2.0 / playback_speed  # Default: 2 sec per turn
    time_elapsed = time.time() - last_play_time

    # Advance if ready
    if time_elapsed >= advance_interval:
        st.session_state.current_turn += 1
        st.session_state.last_play_time = time.time()
        st.rerun()
```

---

## Timeline System

### Timeline Components

1. **Visual Markers** (above slider)
2. **Slider Widget** (interactive scrubber)
3. **Turn Counter** (X / Y display)
4. **Legend** (color key)

### Marker Rendering

```python
def get_turn_marker_html(turns: list, current_turn: int) -> str:
    markers = []

    for idx, turn in enumerate(turns):
        # Get status from turn data
        status = turn.get('status', '✓')
        is_active = (idx == current_turn)

        # Determine color
        color = {
            '✓': '#90EE90',  # Green - stable
            '⚠️': '#FFA500',  # Orange - drift
            '⚡': '#FFD700',  # Gold - intervention
            '⚙️': '#6495ED'   # Blue - calibration
        }.get(status, '#90EE90')

        # Size based on active state
        size = "12px" if is_active else "8px"
        opacity = "1.0" if is_active else "0.7"

        # Generate HTML
        style = f"""
            width: {size};
            height: {size};
            background: {color};
            border-radius: 50%;
            opacity: {opacity};
            transition: all 0.3s ease;
        """

        if is_active:
            style += f"box-shadow: 0 0 8px {color};"

        markers.append(f'<span style="{style}"></span>')

    return '<div style="text-align: center;">' + ''.join(markers) + '</div>'
```

### Slider Integration

```python
st.slider(
    label="Timeline",
    min_value=0,
    max_value=len(turns) - 1,
    value=st.session_state.current_turn,
    key="timeline_slider",  # ✅ Unique key
    on_change=on_timeline_change,  # Callback
    label_visibility="collapsed"  # Hide label
)
```

### Legend Display

```
┌────────────────────────────────────────┐
│ Legend:                                │
│ ● Stable  ● Drift  ● Intervention  ● Calibration │
└────────────────────────────────────────┘
```

---

## Integration Points

### With Observation Deck

TELOSCOPE drives Observation Deck display:

```python
# TELOSCOPE updates current_turn
st.session_state.current_turn = new_turn
st.rerun()

# Observation Deck reads current_turn
active_turn = st.session_state.current_turn

# Applies dimming to all turns
for idx, turn in enumerate(turns):
    opacity = calculate_turn_opacity(idx, active_turn)
    render_turn(turn, opacity)

# Scrolls to active turn
scroll_to_turn(active_turn)
```

### With Main Observatory

TELOSCOPE is a module called by main app:

```python
# main_observatory.py
from teloscope.teloscope_controller import render_teloscope

def main():
    init_session_state()
    render_observation_deck()
    render_teloscope()  # ✅ TELOSCOPE renders here
```

### With Future Components (Phase 2)

```python
# TELOSCOPE will drive:
- Comparison Viewer: Show baseline vs TELOS for current_turn
- Calculation Window: Display math for current_turn
- Steward Chat: Context-aware based on current_turn
```

---

## Extension Architecture

### Adding New Navigation Buttons

1. **Add callback** in `navigation_controls.py`:
```python
def on_jump_to_drift():
    """Jump to first drift event"""
    turns = st.session_state.session_data['turns']
    for idx, turn in enumerate(turns):
        if turn.get('drift_detected'):
            st.session_state.current_turn = idx
            break
    st.rerun()
```

2. **Add button** in `render_navigation_controls()`:
```python
st.button(
    "⚠️ Jump to Drift",
    on_click=on_jump_to_drift,
    key="nav_drift",
    use_container_width=True
)
```

### Adding Playback Speed Control

1. **Add state** in `init_session_state()`:
```python
st.session_state.playback_speed = 1.0
```

2. **Add slider** in `teloscope_controller.py`:
```python
st.slider(
    "Speed",
    min_value=0.5,
    max_value=2.0,
    value=1.0,
    step=0.5,
    key="speed_slider",
    on_change=on_speed_change
)
```

3. **Use in autoplay**:
```python
advance_interval = 2.0 / st.session_state.playback_speed
```

### Adding Timeline Annotations

1. **Extend turn data** with annotation field
2. **Update marker HTML** to show annotation icon
3. **Add tooltip** with annotation text

---

## Performance Considerations

### Autoplay Efficiency

- Autoplay checks timing every render cycle
- Uses `time.time()` for precision
- Only triggers rerun when turn should advance
- No polling loops (relies on Streamlit rerun cycle)

### Timeline Rendering

- Markers generated as static HTML (cached)
- Only regenerates when turns change
- CSS transitions handled by browser
- No JavaScript for basic functionality

### State Management

- Minimal state (5 variables)
- No complex data structures
- All state in st.session_state (persists across reruns)
- No database queries during navigation

---

## Testing TELOSCOPE

### Manual Test Checklist

```markdown
## Navigation Tests
- [ ] First button jumps to turn 0
- [ ] Prev button decrements turn
- [ ] Next button increments turn
- [ ] Last button jumps to final turn
- [ ] Buttons disable at boundaries

## Timeline Tests
- [ ] Slider scrubs to any turn
- [ ] Markers update to show active turn
- [ ] Turn counter displays X / Y correctly
- [ ] Legend shows correct colors

## Autoplay Tests
- [ ] Play starts advancement
- [ ] Button changes to Pause
- [ ] Turns advance every 2 seconds
- [ ] Stops at last turn
- [ ] Pause stops advancement

## Integration Tests
- [ ] Observation Deck scrolls to active turn
- [ ] Dimming updates on navigation
- [ ] Control strip updates with correct data
- [ ] No lag or jank during scrubbing
```

### Automated Tests

```python
def test_navigation_bounds():
    """Test buttons respect boundaries"""
    st.session_state.current_turn = 0
    on_previous_turn()
    assert st.session_state.current_turn == 0  # Can't go below 0

def test_timeline_sync():
    """Test timeline syncs with current_turn"""
    st.session_state.current_turn = 5
    assert st.session_state.timeline_slider == 5
```

---

## Future Enhancements

### Phase 2 Features

- **Keyboard shortcuts**: Arrow keys, spacebar for play
- **Speed control**: 0.5x, 1x, 2x playback speed slider
- **Jump to event**: Buttons for drift, intervention, calibration
- **Bookmarks**: User can mark important turns
- **Loop mode**: Replay turns in a range

### Phase 3 Features

- **Timeline zoom**: Focus on specific turn range
- **Multi-track**: Show multiple sessions stacked
- **Export timeline**: Save as image or video
- **Collaborative**: Shared timeline state across users

---

## Troubleshooting

### Timeline not updating

**Symptom**: Slider moves but viewport doesn't scroll

**Cause**: Missing `st.rerun()` in callback

**Fix**:
```python
def on_timeline_change():
    st.session_state.current_turn = st.session_state.timeline_slider
    st.rerun()  # ✅ Add this
```

### Play button doesn't work

**Symptom**: Click Play, nothing happens

**Cause**: `handle_autoplay()` not called

**Fix**: Ensure `render_teloscope()` calls `handle_autoplay()` before rendering

### Buttons lag

**Symptom**: Click button, 1-2 second delay

**Cause**: Heavy computation in callback

**Fix**: Move computation out of callback, only update state

---

## References

- **Streamlit Patterns**: `docs/streamlit_patterns.md`
- **Phase 1 Implementation**: `docs/PHASE1_IMPLEMENTATION_GUIDE.md`
- **API Reference**: `docs/TELOSCOPE_API.md`
- **Main Observatory**: `main_observatory.py`

---

**End of TELOSCOPE Architecture Documentation**
