# Streamlit Implementation Patterns for TELOS Observatory

**Version**: 1.0
**Last Updated**: 2025-10-30
**Purpose**: Critical patterns for building reactive, bug-free Streamlit applications

---

## Table of Contents

1. [Session State Pattern](#session-state-pattern)
2. [Callback Pattern](#callback-pattern)
3. [Widget Key Management](#widget-key-management)
4. [Layout Management](#layout-management)
5. [Dimming Algorithm](#dimming-algorithm)
6. [Navigation Synchronization](#navigation-synchronization)
7. [Common Anti-Patterns to Avoid](#common-anti-patterns-to-avoid)

---

## Session State Pattern

### **Rule**: All UI state lives in `st.session_state`

**Why**: Streamlit reruns the entire script on every interaction. Local variables are reset. Only `st.session_state` persists across reruns.

### ✅ Correct Pattern

```python
def init_session_state():
    """Initialize ALL session state variables - call this FIRST in main()"""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.current_turn = 0
        st.session_state.playing = False
        st.session_state.telescope_open = False
        st.session_state.session_data = load_mock_session()

def main():
    st.set_page_config(layout="wide")

    # Initialize state FIRST, before any rendering
    init_session_state()

    # Now render UI - reads from st.session_state
    render_ui()

if __name__ == "__main__":
    main()
```

### ❌ Wrong - Local Variable

```python
# DON'T DO THIS
def main():
    current_turn = 0  # ❌ Reset to 0 on every rerun!

    if st.button("Next"):
        current_turn += 1  # ❌ This won't persist

    st.write(f"Turn: {current_turn}")  # Always shows 0
```

### ✅ Correct - Session State

```python
# DO THIS
def main():
    if 'current_turn' not in st.session_state:
        st.session_state.current_turn = 0

    if st.button("Next"):
        st.session_state.current_turn += 1
        st.rerun()

    st.write(f"Turn: {st.session_state.current_turn}")  # Persists!
```

---

## Callback Pattern

### **Rule**: Callbacks ONLY update `st.session_state`, never render UI

**Why**: Streamlit processes callbacks before rendering. UI updates must happen in the main execution flow, not in callbacks.

### Data Flow

```
User clicks button
    ↓
Callback executes (updates state only)
    ↓
st.rerun() called
    ↓
Script re-executes from top
    ↓
UI renders with new state
```

### ✅ Correct Pattern

```python
def on_next_turn():
    """Callback: ONLY update state, NO rendering"""
    max_turns = len(st.session_state.session_data['turns'])
    if st.session_state.current_turn < max_turns - 1:
        st.session_state.current_turn += 1
    st.rerun()

def render_controls():
    """Rendering function: reads state, shows UI"""
    st.button("Next Turn", on_click=on_next_turn, key="next_btn")
    st.write(f"Current: {st.session_state.current_turn}")

def main():
    init_session_state()
    render_controls()  # UI renders here, after callbacks processed
```

### ❌ Wrong - Rendering in Callback

```python
def on_next_turn_wrong():
    """❌ DON'T DO THIS"""
    st.session_state.current_turn += 1
    st.write("Updated!")  # ❌ UI rendering in callback!
    st.success("Next turn!")  # ❌ This won't display correctly

def render_controls():
    st.button("Next", on_click=on_next_turn_wrong)  # ❌ Broken
```

### Callback Rules Summary

| ✅ DO | ❌ DON'T |
|-------|----------|
| Update `st.session_state` | Call `st.write()` |
| Call `st.rerun()` | Call `st.success()` / `st.error()` |
| Perform calculations | Render UI components |
| Validate input | Create widgets |

---

## Widget Key Management

### **Rule**: Every widget needs a unique key, especially in loops

**Why**: Streamlit uses keys to track widget state. Duplicate keys cause crashes. Missing keys in loops cause unpredictable behavior.

### ✅ Correct - Unique Keys

```python
def render_turns():
    turns = st.session_state.session_data['turns']

    for idx, turn in enumerate(turns):
        # Unique key using index
        st.button(
            f"Turn {turn['turn']}",
            key=f"turn_btn_{idx}",  # ✅ Unique key
            on_click=on_turn_selected,
            args=(idx,)
        )
```

### ❌ Wrong - No Keys in Loop

```python
def render_turns_wrong():
    turns = st.session_state.session_data['turns']

    for turn in turns:
        # ❌ No key - Streamlit can't track state
        st.button(f"Turn {turn['turn']}")  # ❌ Breaks!
```

### ❌ Wrong - Duplicate Keys

```python
def render_controls_wrong():
    col1, col2 = st.columns(2)

    with col1:
        st.button("Action", key="btn")  # Key: "btn"

    with col2:
        st.button("Action", key="btn")  # ❌ Duplicate key! Crashes!
```

---

## Layout Management

### **Rule**: Complete all columns before moving to next section, max 3 levels deep

**Why**: Streamlit processes layout sequentially. Incomplete columns cause layout breaks. Deep nesting causes rendering issues.

### ✅ Correct - Complete Columns

```python
def render_control_strip():
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        st.write("Title")
        st.write("Subtitle")  # Complete col1

    with col2:
        st.metric("Fidelity", "0.89")  # Complete col2

    with col3:
        st.button("Tools")  # Complete col3

    # ✅ All columns complete, now move to next section
    st.markdown("---")
    render_next_section()
```

### ❌ Wrong - Incomplete Columns

```python
def render_control_strip_wrong():
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        st.write("Title")

    # ❌ col2 and col3 never used - layout breaks!

    st.markdown("---")  # ❌ Rendered while columns incomplete
```

### Nesting Limit

```python
# ✅ OK - 3 levels deep
def render_nested():
    with st.container():  # Level 1
        col1, col2 = st.columns(2)  # Level 2
        with col1:
            with st.expander("Details"):  # Level 3 - still OK
                st.write("Content")

# ❌ Avoid - 4+ levels deep
def render_nested_wrong():
    with st.container():  # Level 1
        col1, col2 = st.columns(2)  # Level 2
        with col1:
            with st.expander("Details"):  # Level 3
                with st.container():  # Level 4 - too deep!
                    st.write("Content")  # ❌ May break
```

---

## Dimming Algorithm

### **Purpose**: Dim turns based on distance from active turn for visual focus

### Algorithm

```python
def calculate_turn_opacity(turn_index: int, active_turn_index: int) -> float:
    """
    Calculate opacity based on distance from active turn.

    Args:
        turn_index: Index of turn being rendered
        active_turn_index: Index of currently active turn

    Returns:
        Opacity value between 0.2 and 1.0
    """
    distance = abs(turn_index - active_turn_index)

    if distance == 0:
        return 1.0  # Active turn: full brightness
    elif distance == 1:
        return 0.7  # Adjacent: slight dim
    elif distance == 2:
        return 0.4  # Two away: more dim
    else:
        return 0.2  # Far: very dim but visible
```

### Visual Representation

```
Turn Index:     0    1    2    3    4    5    6    7    8
Active Turn:                   ^
                               3

Opacity:      0.2  0.4  0.7  1.0  0.7  0.4  0.2  0.2  0.2
Visual:        ▁    ▃    ▅    █    ▅    ▃    ▁    ▁    ▁
Border:                       ═══ (gold border on active)
```

### CSS Styling

```python
def get_turn_style(turn_index: int, active_turn_index: int) -> str:
    """Generate CSS style for turn based on distance from active"""
    opacity = calculate_turn_opacity(turn_index, active_turn_index)
    is_active = (turn_index == active_turn_index)

    border = "3px solid #FFD700" if is_active else "1px solid rgba(255,255,255,0.1)"
    box_shadow = "0 0 10px rgba(255,215,0,0.3)" if is_active else "none"

    return f"""
        opacity: {opacity};
        border: {border};
        box-shadow: {box_shadow};
        transition: all 0.3s ease;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        background: rgba(255,255,255,0.05);
    """
```

---

## Navigation Synchronization

### **Pattern**: TELOSCOPE controls → State → Viewport update

### Data Flow

```
User Action
    ↓
[Timeline Slider] → on_timeline_change()
[Next Button]     → on_next_turn()
[Play Timer]      → on_play_tick()
    ↓
Update st.session_state.current_turn
    ↓
st.rerun()
    ↓
Viewport reads current_turn
    ↓
Apply dimming to all turns
Scroll to active turn
Highlight with gold border
```

### Implementation

```python
# Step 1: Callbacks update state
def on_timeline_change():
    """User dragged timeline slider"""
    new_turn = st.session_state.timeline_slider  # Widget auto-syncs to key
    st.session_state.current_turn = new_turn
    st.rerun()

def on_next_turn():
    """User clicked Next button"""
    max_turns = len(st.session_state.session_data['turns']) - 1
    if st.session_state.current_turn < max_turns:
        st.session_state.current_turn += 1
    st.rerun()

# Step 2: Render timeline
def render_timeline():
    """Timeline scrubber"""
    max_turns = len(st.session_state.session_data['turns']) - 1

    st.slider(
        "Turn",
        min_value=0,
        max_value=max_turns,
        value=st.session_state.current_turn,
        key="timeline_slider",  # ✅ Unique key
        on_change=on_timeline_change
    )

# Step 3: Render viewport (reads state)
def render_viewport():
    """Chat viewport with dimming"""
    active_turn = st.session_state.current_turn
    turns = st.session_state.session_data['turns']

    for idx, turn in enumerate(turns):
        opacity = calculate_turn_opacity(idx, active_turn)
        style = get_turn_style(idx, active_turn)

        # Render turn with styling
        st.markdown(
            f'<div style="{style}">{turn["user_input"]}</div>',
            unsafe_allow_html=True
        )

    # Auto-scroll to active turn (JavaScript)
    scroll_to_turn(active_turn)
```

---

## Common Anti-Patterns to Avoid

### ❌ Anti-Pattern 1: Rendering in Callbacks

```python
# WRONG
def on_button_click():
    st.session_state.value += 1
    st.write(f"Value: {st.session_state.value}")  # ❌ Won't display

# RIGHT
def on_button_click():
    st.session_state.value += 1
    st.rerun()

def render_ui():
    st.write(f"Value: {st.session_state.value}")  # ✅ Displays correctly
```

### ❌ Anti-Pattern 2: Forgetting st.rerun()

```python
# WRONG - State changes but UI doesn't update
def on_button_click():
    st.session_state.current_turn += 1
    # ❌ No st.rerun() - UI frozen!

# RIGHT
def on_button_click():
    st.session_state.current_turn += 1
    st.rerun()  # ✅ UI updates immediately
```

### ❌ Anti-Pattern 3: Using Local Variables for State

```python
# WRONG
def main():
    playing = False  # ❌ Reset every rerun

    if st.button("Play"):
        playing = True  # ❌ Lost on next rerun

    if playing:
        st.write("Playing...")  # ❌ Never executes

# RIGHT
def main():
    if 'playing' not in st.session_state:
        st.session_state.playing = False  # ✅ Persists

    if st.button("Play"):
        st.session_state.playing = True
        st.rerun()

    if st.session_state.playing:
        st.write("Playing...")  # ✅ Works!
```

### ❌ Anti-Pattern 4: Missing Widget Keys in Loops

```python
# WRONG
for item in items:
    st.button(item)  # ❌ No key - breaks!

# RIGHT
for idx, item in enumerate(items):
    st.button(item, key=f"btn_{idx}")  # ✅ Unique key
```

### ❌ Anti-Pattern 5: Incomplete Column Layouts

```python
# WRONG
col1, col2, col3 = st.columns(3)
with col1:
    st.write("Content")
# ❌ col2 and col3 unused - layout breaks

# RIGHT
col1, col2, col3 = st.columns(3)
with col1:
    st.write("Content")
with col2:
    st.write("More")
with col3:
    st.write("Even more")
# ✅ All columns complete
```

---

## Quick Reference Checklist

Before committing code, verify:

- [ ] All UI state is in `st.session_state`, not local variables
- [ ] State is initialized with `if 'key' not in st.session_state` checks
- [ ] Callbacks only update state and call `st.rerun()`
- [ ] No `st.write()` or UI rendering inside callbacks
- [ ] Every widget in a loop has a unique key
- [ ] All columns are completed before moving to next section
- [ ] Layout nesting is ≤ 3 levels deep
- [ ] After state changes, `st.rerun()` is called
- [ ] Opacity transitions use CSS `transition: all 0.3s ease`

---

## Resources

- **Streamlit Docs**: https://docs.streamlit.io/library/advanced-features/session-state
- **Callback Pattern**: https://docs.streamlit.io/library/advanced-features/button-behavior-and-examples
- **TELOS Observatory**: Phase 1 implementation guide

---

**End of Document**
