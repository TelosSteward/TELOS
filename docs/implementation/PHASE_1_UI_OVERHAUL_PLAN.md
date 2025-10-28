# TELOS UI Overhaul - Phase 1 Implementation Plan

**Date**: 2025-10-27
**Status**: Planning Complete - Ready for Implementation
**Estimated Effort**: 30-40 hours across 7 phases

---

## Executive Summary

Complete architectural redesign of TELOS Observatory dashboard from multi-tab research interface to ChatGPT-style conversation interface with floating diagnostic windows. This plan preserves all core governance logic while rebuilding the presentation layer.

**Key Changes**:
- ChatGPT-style clean message interface
- Floating draggable overlay windows (Steward Lens, TELOSCOPE)
- Governance toggle (Native Mistral vs TELOS Steward)
- Turn-based navigation with play/pause controls
- Keyboard-driven interaction
- Complete rebranding (TELOS, Steward, TELOSCOPIC Tools)

**What's Preserved**:
- All core governance logic (UnifiedGovernanceSteward, ProgressivePrimacyExtractor)
- Drift detection and intervention generation
- Fidelity measurement
- Counterfactual analysis
- Session management

---

## Current Architecture Analysis

### File: streamlit_live_comparison.py
- **Size**: 2,865 lines
- **Dependency Status**: ✅ Isolated (no other files import from it)
- **Safety Level**: High - Can be refactored without breaking other modules
- **Functions**: 14 total (1 helper, 11 rendering, 2 core)

### Function Map

| Line | Function | Purpose | Category | Action |
|------|----------|---------|----------|--------|
| 104 | `get_mode()` | Returns current UI mode | UI Helper | **REMOVE** |
| 110 | `get_terminology()` | Mode-specific language | UI Helper | **MODIFY** |
| 152 | `render_research_mode_observatory()` | Research mode math | UI Rendering | **REPLACE** |
| 358 | `initialize_teloscope()` | System initialization | Core Logic | **MODIFY** |
| 550 | `render_sidebar()` | Configuration sidebar | UI Rendering | **REPLACE** |
| 749 | `render_live_session()` | Live chat interface | UI Rendering | **REPLACE** |
| 1573 | `render_session_replay()` | Timeline replay | UI Rendering | **MERGE** |
| 1710 | `render_teloscope_view()` | Counterfactual viewer | UI Rendering | **REPLACE** |
| 2000 | `render_analytics_dashboard()` | Analytics charts | UI Rendering | **KEEP** |
| 2238 | `render_intervention_timeline()` | Timeline viz | UI Rendering | **KEEP** |
| 2346 | `render_simulation_ui()` | Simulation controls | UI Rendering | **KEEP** |
| 2429 | `render_simulation_results()` | Results display | UI Rendering | **KEEP** |
| 2578 | `show_onboarding_screen()` | Attractor mode selection | UI Rendering | **SIMPLIFY** |
| 2772 | `main()` | Application entry | Core Logic | **MODIFY** |

---

## Core Governance Logic - DO NOT TOUCH

These modules contain the actual governance intelligence and are imported from external files:

### ✅ Preserved Modules

**1. telos_purpose.core.unified_steward.UnifiedGovernanceSteward**
- Fidelity measurement
- Drift detection
- Intervention generation
- Basin verification

**2. telos_purpose.profiling.progressive_primacy_extractor.ProgressivePrimacyExtractor**
- Purpose extraction from conversation
- Statistical convergence detection
- Attractor refinement

**3. telos_purpose.sessions.live_interceptor.LiveInterceptor**
- Response interception
- Governance application
- Turn-by-turn processing

**4. telos_purpose.core.session_state.SessionStateManager**
- Turn storage
- Session persistence
- State management

**5. telos_purpose.core.counterfactual_branch_manager.CounterfactualBranchManager**
- Counterfactual generation
- Branch tracking

**6. telos_purpose.llm_clients.mistral_client.TelosMistralClient**
- LLM communication
- API handling

**7. telos_purpose.core.embedding_provider.EmbeddingProvider**
- Embedding generation
- Semantic distance calculation

---

## Session State Variables

### Existing (Preserved)

```python
# Core Components
st.session_state.llm                       # TelosMistralClient instance
st.session_state.embedding_provider        # EmbeddingProvider instance
st.session_state.attractor                 # PrimacyAttractor instance
st.session_state.steward                   # UnifiedGovernanceSteward instance
st.session_state.progressive_extractor     # ProgressivePrimacyExtractor (if applicable)
st.session_state.interceptor               # LiveInterceptor instance

# Session Management
st.session_state.web_session               # WebSessionManager instance
st.session_state.session_manager           # SessionStateManager instance
st.session_state.current_session           # Current session data
st.session_state.config                    # Configuration dict

# Counterfactual/Simulation
st.session_state.branch_manager            # CounterfactualBranchManager instance
st.session_state.simulator                 # CounterfactualBranchSimulator instance
st.session_state.comparator                # BranchComparator instance
st.session_state.counterfactual_branches   # List of branches
st.session_state.simulation_results        # Dict of simulation results

# UI State (Old - to be migrated/removed)
st.session_state.mode                      # 'Basic'/'Advanced'/'Research Mode' - DEPRECATED
st.session_state.research_lens_active      # Research Lens toggle - MIGRATE
st.session_state.selected_trigger          # Selected trigger for TELOSCOPE - KEEP
st.session_state.active_tab                # Current tab index - REMOVE
st.session_state.replay_turn               # Replay timeline position - MIGRATE

# Onboarding
st.session_state.attractor_mode            # 'progressive'/'predefined'/'hybrid'/'pristine'
st.session_state.onboarding_complete       # Onboarding done flag
st.session_state.predefined_purpose        # Pre-defined mode purpose
st.session_state.predefined_boundaries     # Pre-defined mode boundaries
st.session_state.hybrid_seed_purpose       # Hybrid mode seed
st.session_state.hybrid_seed_boundaries    # Hybrid mode seed boundaries

# System
st.session_state.teloscope_initialized     # Init flag
```

### New (Phase 1 Additions)

```python
# New UI Configuration
st.session_state.ui_config = {
    'governance_enabled': True,           # Toggle Native vs TELOS
    'interface_mode': 'telos',            # 'telos' or 'teloscope'
    'current_turn': 0,                    # Current turn index for navigation
    'steward_lens_visible': False,        # Steward Lens overlay visibility
    'teloscope_visible': False,           # TELOSCOPE window visibility
    'turn_controls_visible': True,        # Show/hide turn controls
    'is_playing': False,                  # Playback state
    'playback_speed': 1.0,                # Playback speed multiplier
    'window_positions': {                 # Floating window positions
        'steward_lens': {'x': 100, 'y': 100},
        'teloscope': {'x': 200, 'y': 100},
    }
}

# Keyboard Shortcuts State
st.session_state.keyboard_shortcuts = {
    'cmd_g': False,  # Toggle governance
    'cmd_l': False,  # Toggle Steward Lens
    'cmd_t': False,  # Toggle TELOSCOPE
    'space': False,  # Play/Pause
}
```

---

## Implementation Phases

### PHASE 1: Foundation (Low Risk)
**Estimated Time**: 2-3 hours
**Lines Changed**: ~100
**Risk Level**: Low

#### Objectives:
1. Add new session state variables without breaking existing UI
2. Create utility functions for ChatGPT-style rendering
3. Establish foundation for subsequent phases

#### Tasks:
1. Update `initialize_teloscope()` to add `ui_config` to session state
2. Create `get_ui_config()` helper function
3. Create `render_chat_bubble()` utility (user/assistant styling)
4. Create `format_timestamp()` utility
5. Test that existing UI still works with new session state

#### Acceptance Criteria:
- [ ] New session state variables initialize correctly
- [ ] Existing UI continues to function
- [ ] Chat bubble rendering works in isolation
- [ ] No regressions in governance logic

#### Files Modified:
- `streamlit_live_comparison.py` (Lines 358-548: `initialize_teloscope()`)

---

### PHASE 2: Message Display (Medium Risk)
**Estimated Time**: 4-6 hours
**Lines Changed**: ~300
**Risk Level**: Medium

#### Objectives:
1. Replace tabbed chat display with ChatGPT-style continuous feed
2. Implement message bubbles (user = right, assistant = left)
3. Add metadata toggle (timestamps, fidelity scores)

#### Tasks:
1. Create `render_chat_message(turn_data, show_metadata=False)` function
2. Create `render_chat_history(turns)` function
3. Replace message rendering in `render_live_session()` (Lines 1339-1400)
4. Add metadata toggle button
5. Style message bubbles (CSS styling)
6. Test with existing turn data
7. Verify intervention display still works

#### Acceptance Criteria:
- [ ] Messages display in ChatGPT-style bubbles
- [ ] User messages align right, assistant left
- [ ] Metadata toggle shows/hides timestamps and scores
- [ ] Intervention indicators still visible
- [ ] No loss of turn data
- [ ] Governance continues to work

#### Files Modified:
- `streamlit_live_comparison.py` (Lines 749-1570: `render_live_session()`)

---

### PHASE 3: Floating Windows (Medium Risk)
**Estimated Time**: 6-8 hours
**Lines Changed**: ~400
**Risk Level**: Medium

#### Objectives:
1. Create floating window infrastructure
2. Implement Steward Lens (replaces Research Mode observatory)
3. Add window dragging and positioning

#### Tasks:
1. Research Streamlit draggable window options (streamlit-extras, custom HTML)
2. Create `render_floating_window(title, content, position, key)` base component
3. Create `render_steward_lens(turn_data)` using floating window
4. Add toggle button for Steward Lens
5. Implement window positioning (save to `ui_config.window_positions`)
6. Migrate Research Mode observatory content to Steward Lens
7. Test Steward Lens shows correct governance data
8. Add close button to window

#### Acceptance Criteria:
- [ ] Floating window renders above main content
- [ ] Steward Lens shows all governance metrics
- [ ] Window is draggable (if technically feasible)
- [ ] Position persists across reruns
- [ ] Toggle works (show/hide)
- [ ] Governance data displays correctly

#### Files Modified:
- `streamlit_live_comparison.py` (New functions + integration)

#### Technical Challenges:
- Streamlit has limited dragging support (may need JavaScript injection)
- Window positioning may require `st.components.v1.html()`

---

### PHASE 4: Governance Toggle (High Risk - Core Functionality)
**Estimated Time**: 8-10 hours
**Lines Changed**: ~200
**Risk Level**: **High**

#### Objectives:
1. Add toggle to enable/disable TELOS governance
2. Store both Native and TELOS responses in turn data
3. Switch between responses without regeneration

#### Critical Implementation Details:

**Turn Data Structure Enhancement**:
```python
turn_data = {
    'user_input': "...",
    'native_response': "...",      # NEW: Direct Mistral response
    'telos_response': "...",        # NEW: Governed response
    'active_response': "telos",     # NEW: Which one is displayed
    'governance_applied': True,     # Existing
    'intervention_type': "...",     # Existing
    'metrics': {...},               # Existing
    # ... rest of existing fields
}
```

#### Tasks:
1. **CRITICAL**: Modify LiveInterceptor to generate BOTH responses
   - Add `generate_dual_responses(messages)` method
   - Call Native LLM first
   - Apply governance to get TELOS response
   - Return both
2. Update turn storage to include both responses
3. Create `render_governance_toggle()` UI component
4. Implement toggle logic to switch `active_response`
5. Update message rendering to use `active_response`
6. Add visual indicator (badge) showing which response is active
7. Test switching between Native and TELOS
8. Verify governance metrics only shown for TELOS response

#### Acceptance Criteria:
- [ ] Toggle switches between Native and TELOS responses
- [ ] Both responses stored in turn data
- [ ] No regeneration required when toggling
- [ ] Governance metrics only visible in TELOS mode
- [ ] Steward Lens only available in TELOS mode
- [ ] Visual indicator shows active mode

#### Files Modified:
- `telos_purpose/sessions/live_interceptor.py` (**External module - requires coordination**)
- `streamlit_live_comparison.py` (Turn storage and rendering)

#### Risks:
- LiveInterceptor modification may break existing functionality
- Dual response generation increases API calls/cost
- Session state size increases with dual storage

---

### PHASE 5: Turn Navigation (Medium Risk)
**Estimated Time**: 5-7 hours
**Lines Changed**: ~250
**Risk Level**: Medium

#### Objectives:
1. Merge Live Session and Session Replay into single interface
2. Add turn navigation controls (play/pause, forward/back, scrubber)
3. Implement keyboard shortcuts

#### Tasks:
1. Create `render_turn_controls()` component
   - Play/Pause button
   - Forward/Back buttons
   - Turn scrubber (slider)
   - Speed control (1x, 2x, 4x)
2. Implement turn navigation logic
   - `navigate_to_turn(turn_index)`
   - `play_turns()` with auto-advance
   - `pause_playback()`
3. Add keyboard event handling
   - Left/Right arrows: Navigate turns
   - Space: Play/Pause
   - CMD+G: Toggle governance
   - CMD+L: Toggle Steward Lens
4. Merge `render_live_session()` and `render_session_replay()`
5. Update UI to show current turn indicator
6. Test navigation with existing session data

#### Acceptance Criteria:
- [ ] Turn controls display correctly
- [ ] Play/Pause works
- [ ] Navigation updates displayed turn
- [ ] Keyboard shortcuts work
- [ ] Turn scrubber syncs with current turn
- [ ] Speed control adjusts playback rate

#### Files Modified:
- `streamlit_live_comparison.py` (Lines 749-1708: Merge functions)

#### Technical Challenges:
- Streamlit keyboard events require JavaScript injection
- Auto-advance during playback requires careful rerun management

---

### PHASE 6: TELOSCOPE Window (Low Risk)
**Estimated Time**: 3-4 hours
**Lines Changed**: ~200
**Risk Level**: Low

#### Objectives:
1. Convert TELOSCOPE tab to floating window overlay
2. Migrate counterfactual display logic
3. Add trigger selection in main interface

#### Tasks:
1. Create `render_teloscope_window(trigger_data)` using floating window base
2. Migrate counterfactual comparison display from `render_teloscope_view()`
3. Add "View in TELOSCOPE" button to intervention indicators
4. Implement trigger selection flow
5. Add keyboard shortcut (CMD+T) to toggle TELOSCOPE
6. Test counterfactual display in window
7. Verify branch comparison works

#### Acceptance Criteria:
- [ ] TELOSCOPE displays as floating window
- [ ] Counterfactual data shows correctly
- [ ] Trigger selection opens TELOSCOPE
- [ ] Window toggle works
- [ ] Branch comparison displays properly

#### Files Modified:
- `streamlit_live_comparison.py` (Lines 1710-1998: Refactor to window)

---

### PHASE 7: Cleanup (Low Risk)
**Estimated Time**: 2-3 hours
**Lines Changed**: ~150
**Risk Level**: Low

#### Objectives:
1. Remove deprecated code
2. Update documentation
3. Clean up unused session state variables

#### Tasks:
1. Remove `get_mode()` function (Line 104)
2. Remove old tab system from `main()` (Line 2772)
3. Remove `st.session_state.mode` references
4. Remove `st.session_state.active_tab` references
5. Update `get_terminology()` to work without modes
6. Remove deprecated Research Mode code
7. Clean up comments and debug prints
8. Update docstrings
9. Run full test suite
10. Create migration guide for users

#### Acceptance Criteria:
- [ ] No references to old mode system
- [ ] No tab-related code
- [ ] All deprecated session state removed
- [ ] Documentation updated
- [ ] Tests pass
- [ ] User migration guide created

#### Files Modified:
- `streamlit_live_comparison.py` (Throughout)
- `docs/` (New migration guide)

---

## Risk Assessment

### High-Risk Areas

**1. Turn Data Structure Enhancement (Phase 4)**
- **Risk**: Breaking existing turn storage/retrieval
- **Impact**: Loss of session data, broken replay
- **Mitigation**:
  - Add new fields alongside existing ones (backwards compatible)
  - Test extensively with old session data
  - Implement graceful fallback for missing fields

**2. LiveInterceptor Modification (Phase 4)**
- **Risk**: Breaking governance application
- **Impact**: No interventions, drift undetected
- **Mitigation**:
  - Add `enabled` flag to LiveInterceptor
  - Preserve original `generate()` method
  - New `generate_dual()` method as separate code path
  - Feature flag to switch between old/new behavior

**3. Streamlit Limitations**
- **Risk**: Technical limitations (dragging, keyboard events)
- **Impact**: Reduced UX quality
- **Mitigation**:
  - Research streamlit-extras library
  - Prepare JavaScript injection fallback
  - Accept reduced dragging if necessary (fixed positioning)

### Medium-Risk Areas

**4. Session State Bloat**
- **Risk**: Too many state variables, complexity
- **Impact**: Bugs, hard to maintain
- **Mitigation**: Namespace under `ui_config` dict

**5. Performance**
- **Risk**: Floating windows + frequent reruns = lag
- **Impact**: Poor user experience
- **Mitigation**:
  - Use `@st.cache_data` aggressively
  - Minimize rerun triggers
  - Profile with Streamlit profiler

---

## Testing Strategy

### Unit Tests (Per Phase)
- Test new functions in isolation
- Mock session state
- Verify outputs match expected format

### Integration Tests (Per Phase)
- Test with real turn data
- Verify no regressions in existing functionality
- Check governance logic still works

### End-to-End Tests (After Phase 7)
- Complete user workflow: onboarding → chat → intervention → toggle → navigate
- Verify all windows render
- Test all keyboard shortcuts
- Load test with 100+ turn session

---

## Rollback Strategy

### Phase-Level Rollback
- Each phase commits separately
- Git branch per phase
- Can revert to previous phase if issues

### Feature Flags
```python
ENABLE_NEW_UI = os.getenv('TELOS_NEW_UI', 'false').lower() == 'true'

if ENABLE_NEW_UI:
    render_new_interface()
else:
    render_old_interface()  # Keep old code during transition
```

---

## Dependencies & Blockers

### External Dependencies
- `streamlit` (core)
- `streamlit-extras` (for draggable components - optional)
- No new Python packages required

### Blockers
1. **LiveInterceptor bypass capability** (Phase 4)
   - Requires coordination with governance module
   - May need API changes

2. **Turn data structure agreement** (Phase 4)
   - Needs validation that dual storage won't break SessionStateManager
   - Requires testing with existing session files

---

## Success Criteria

### Phase 1-3 (Foundation + UI)
- [ ] New interface renders without errors
- [ ] ChatGPT-style messages display correctly
- [ ] Steward Lens shows governance data
- [ ] No regression in governance functionality

### Phase 4-5 (Core Features)
- [ ] Governance toggle works
- [ ] Native and TELOS responses both available
- [ ] Turn navigation functional
- [ ] Keyboard shortcuts work

### Phase 6-7 (Polish + Cleanup)
- [ ] TELOSCOPE window integrated
- [ ] All deprecated code removed
- [ ] Documentation complete
- [ ] User migration path clear

### Overall Success
- [ ] All 7 phases complete
- [ ] Core governance preserved
- [ ] New UX implemented
- [ ] Performance acceptable
- [ ] Tests passing
- [ ] User feedback positive

---

## Timeline

| Phase | Duration | Cumulative |
|-------|----------|------------|
| Phase 1 | 2-3 hours | 2-3 hours |
| Phase 2 | 4-6 hours | 6-9 hours |
| Phase 3 | 6-8 hours | 12-17 hours |
| Phase 4 | 8-10 hours | 20-27 hours |
| Phase 5 | 5-7 hours | 25-34 hours |
| Phase 6 | 3-4 hours | 28-38 hours |
| Phase 7 | 2-3 hours | 30-41 hours |

**Total Estimated Effort**: 30-41 hours

---

## Next Steps

1. ✅ **Reconnaissance Complete** - This document
2. **Approve Plan** - Review and sign-off
3. **Begin Phase 1** - Foundation implementation
4. **Incremental Progress** - Complete phases sequentially
5. **Testing at Each Phase** - Validate before moving forward
6. **Final Integration** - Phase 7 cleanup and documentation

---

## Notes

- This is a **major architectural change** - treat with appropriate caution
- **Core governance logic is NOT touched** - only presentation layer
- **Incremental approach** minimizes risk
- **Rollback possible** at each phase
- **Session-independent documentation** enables handoff

**Status**: ✅ Ready for Phase 1 Implementation
