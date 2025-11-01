# TELOS Observatory - Governance Modes

## Overview

TELOS Observatory V3 supports **two distinct governance modes** that manage how the Primacy Attractor (PA) is established and how conversations are governed.

---

## Demo Mode (DEFAULT)

**Purpose:** Demonstrate how TELOS framework governs conversations by keeping focus on explaining TELOS itself.

### Characteristics:

🔒 **Pre-Established Primacy Attractor**
- PA is FULLY CALIBRATED from the start
- NO calibration phase required
- Already converged and ready to govern
- Skips statistical convergence process

🚫 **No User Configuration**
- Users CANNOT modify the PA
- Purpose, scope, and boundaries are LOCKED
- Intentional design to show fixed governance

📍 **Fixed Purpose**
- Purpose: Explain TELOS governance, demonstrate purpose alignment, show intervention strategies
- Scope: TELOS architecture, mathematics, interventions, examples
- Boundaries: Stay focused on TELOS topics, redirect drift back to TELOS

⚙️ **System Behavior**
- Conversations stay focused on TELOS framework topics
- Off-topic questions are redirected back to TELOS
- Demonstrates drift detection and intervention in action
- Perfect for demos, walkthroughs, and showcasing TELOS

### When to Use:
- First-time visitors (default experience)
- Demonstrations and presentations
- Educational walkthroughs
- Showing how TELOS governance works

### Code Location:
- Configuration: `/demo_mode/telos_framework_demo.py`
- Documentation: `/demo_mode/README.md`
- Completely isolated from open mode code

---

## Open Mode

**Purpose:** Real-world application where TELOS extracts and learns the user's actual purpose dynamically.

### Characteristics:

⚡ **Dynamic Purpose Extraction**
- NO pre-established PA
- TELOS extracts purpose from user's conversation
- Uses LLM-based analysis and statistical convergence
- PA emerges naturally from dialogue

📊 **Calibration Phase**
- Requires calibration period to learn user's goals
- Monitors conversation for statistical convergence
- Refines PA as understanding improves
- Multi-component attractors can emerge

🔧 **User Configuration** (Future Feature)
- Users CAN configure their purpose/scope/boundaries
- PA adapts to user's needs
- Flexible and application-specific

🌐 **Open-Ended Governance**
- No hardcoded topic constraints
- Adapts to any domain or purpose
- Governance based on LEARNED purpose, not imposed purpose

### When to Use:
- Real applications and production use
- Users with specific purposes (not TELOS education)
- When PA should adapt to user's needs
- Research and custom implementations

### Code Behavior:
- Sets `attractor = None` in state_manager
- Uses neutral system prompt ("helpful AI assistant")
- Allows TELOS to build PA dynamically
- Standard TELOS calibration and convergence

---

## Switching Between Modes

### In Settings Panel:

1. Open **Settings** in sidebar
2. Under **Governance Mode**, select:
   - **Demo Mode** (pre-established PA, TELOS-focused)
   - **Open Mode** (dynamic extraction, user-focused)
3. Switching modes triggers:
   - Session reset (clears all conversation data)
   - PA re-initialization
   - Conversation flag reset

### Technical Details:

**State Flag:** `st.session_state.telos_demo_mode`
- `True` = Demo Mode
- `False` = Open Mode
- **Default:** `True` (Demo Mode)

**Mode Switching Logic:**
```python
if mode_options[selected_mode] != st.session_state.telos_demo_mode:
    st.session_state.telos_demo_mode = mode_options[selected_mode]
    self.state_manager.clear_demo_data()  # Clear session
    st.rerun()  # Reload with new mode
```

---

## Architecture Notes

### Code Isolation

✅ **Demo Mode Code:**
- Lives in `/demo_mode/` folder
- Completely isolated from other codebases
- Changes to demo mode do NOT affect open mode

✅ **Open Mode Code:**
- Standard TELOS governance flow
- No special restrictions or hardcoded PAs
- Future-ready for user configuration features

### State Management

**Demo Mode:**
- Calls `get_demo_attractor_config()` to get pre-established PA
- Uses `get_demo_system_prompt()` for TELOS-focused prompt
- PA passed directly to `UnifiedGovernanceSteward`

**Open Mode:**
- Sets `attractor = None`
- Uses neutral system prompt
- `UnifiedGovernanceSteward` initializes PA dynamically
- Runs calibration phase

---

## For Steward PM: Key Points

### Two Management Systems:

1. **Demo Mode Management:**
   - Pre-established, locked PA
   - No calibration needed
   - Topic-constrained (TELOS only)
   - Great for showcasing

2. **Open Mode Management:**
   - Dynamic PA extraction
   - Calibration required
   - Topic-agnostic
   - Production-ready

### Critical Understanding:

- These are **fundamentally different governance approaches**
- Demo mode demonstrates TELOS BY constraining to TELOS topics
- Open mode demonstrates TELOS BY adapting to user's topics
- Both use the same TELOS engine, different initialization

### User Experience:

- **First visit:** Demo Mode (shows what TELOS does)
- **After understanding:** Switch to Open Mode (use TELOS for their needs)
- **Settings toggle:** Easy switching between modes

---

## Future Enhancements

### Demo Mode:
- Add more demo scenarios (customer service PA, coding assistant PA, etc.)
- Multiple demo PAs to showcase different use cases
- Demo mode gallery

### Open Mode:
- User configuration UI for custom PAs
- PA save/load functionality
- Template PAs for common use cases
- Real-time calibration visualization

---

**Last Updated:** 2025-11-01
**Version:** TELOS Observatory V3
