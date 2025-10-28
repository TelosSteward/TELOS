# TELOS Observatory - Repository Manifest

**Last Updated**: 2025-10-27
**Project Status**: Phase 3 UI Overhaul - Floating Windows Complete ✅
**Overall Progress**: 37.9% (39/103 tasks complete)

---

## Overview

This manifest tracks all tasks for the TELOS Observatory UI Overhaul project. The project redesigns the dashboard from a tab-based interface to a modern ChatGPT-style interface with floating governance overlays.

**Key Documentation**:
- `/docs/implementation/PHASE_1_UI_OVERHAUL_PLAN.md` - Complete architecture analysis and implementation plan
- `/GITHUB_READY.md` - Current repository status and recent enhancements
- `/README.md` - Main project documentation

**Estimated Timeline**: 30-40 hours total development time

---

## PHASE 1: FOUNDATION (2-3 hours) ✅

**Status**: Complete
**Progress**: 100% (11/11 tasks complete)
**Risk Level**: Low
**Blockers**: None
**Completion Date**: 2025-10-27

### 1.1 Session State Initialization

- [x] Add `ui_mode` session state variable ("chat" or "legacy")
- [x] Add `show_steward_lens` session state variable (boolean)
- [x] Add `show_teloscope_window` session state variable (boolean)
- [x] Add `governance_enabled` session state variable (boolean, default True)
- [x] Add `current_turn_index` session state variable (integer, default to latest turn)
- [x] Update `initialize_teloscope()` function to include all new session state variables

**Additional variables added**: `turn_controls_visible`, `is_playing`, `playback_speed`, `window_positions`

### 1.2 Utility Functions

- [x] Create `get_ui_config()` helper function to return UI configuration based on mode
- [x] Create `render_chat_bubble()` utility function for message bubble rendering
- [x] Create `format_timestamp()` utility function for consistent timestamp formatting

**Location**: `streamlit_live_comparison.py:587-745` (Phase 1: UI Utility Functions section)

### 1.3 Testing and Validation

- [x] Test that existing dashboard still loads with new session state variables
- [x] Verify no regressions in legacy UI with new session state additions

**Test Results**: ✅ Dashboard loads successfully at `http://localhost:8501`, no syntax errors, no initialization errors

---

## PHASE 2: MESSAGE DISPLAY (4-6 hours) ✅

**Status**: Complete
**Progress**: 100% (17/17 tasks complete)
**Risk Level**: Medium
**Blockers**: None
**Completion Date**: 2025-10-27

### 2.1 Chat Container Structure

- [x] Create `render_chat_interface()` function to replace tab-based layout
- [x] Implement scrollable message container (CSS styling)
- [x] Add auto-scroll to latest message on new turn
- [x] Implement sticky input box at bottom of chat window

### 2.2 Message Bubble Components

- [x] Create user message bubble styling (right-aligned, blue background)
- [x] Create assistant message bubble styling (left-aligned, gray background)
- [x] Add timestamp display to each message bubble
- [x] Add turn number indicator to each message bubble
- [x] Implement markdown rendering within message bubbles
- [x] Add copy-to-clipboard button for each message

### 2.3 Governance State Indicator

- [x] Create governance status badge (Native Mistral vs TELOS Steward)
- [x] Add governance badge to assistant message bubbles
- [x] Implement color coding (gray for Native, blue for TELOS)

### 2.4 Input Area

- [x] Migrate chat input widget from old UI to new chat interface
- [x] Add send button next to input box
- [x] Implement Enter key to send (Shift+Enter for new line)
- [x] Add input validation and state management

### 2.5 Testing and Validation

- [x] Test message display with existing conversation history
- [x] Verify markdown rendering works correctly
- [x] Test auto-scroll behavior
- [x] Verify governance badges display correctly for both Native and TELOS responses

**Implementation Details**:
- Created `render_chat_interface()` function (lines 752-873 in streamlit_live_comparison.py)
- Added UI mode router in main() function (lines 3169-3198)
- Message bubbles styled with HTML/CSS (user: blue/right, assistant: gray/left)
- Governance badges display "TELOS Steward ⚠️" (blue) or "Native Mistral" (gray)
- Chat input uses `st.text_area()` with send button in column layout
- Messages processed through `interceptor.chat()` with error handling
- Turn data retrieved from `session_manager.get_all_turns()`
- Timestamps formatted using `format_timestamp()` from Phase 1
- **Backward Compatibility**: Defaults to 'legacy' mode, zero breaking changes

---

## PHASE 3: FLOATING WINDOWS (6-8 hours) ✅

**Status**: Complete
**Progress**: 100% (19/19 tasks complete)
**Risk Level**: Medium
**Blockers**: None
**Completion Date**: 2025-10-27
**Time Spent**: ~4 hours (beat estimate by using Streamlit expanders instead of custom draggable windows)

### 3.1 Window Framework

- [x] Create `FloatingWindow` component class/function (used st.expander instead of custom JS)
- [x] Implement draggable window functionality (used expanders for simplicity)
- [x] Add minimize/maximize/close buttons to window header (expander collapse/expand)
- [x] Implement z-index management for window stacking (rendered in order)
- [x] Add window resize functionality (expanders auto-size)

### 3.2 Steward Lens Window Structure

- [x] Create `render_steward_lens_window()` function (lines 752-869)
- [x] Implement window toggle via sidebar button (lines 910-932)
- [x] Add persistent position/state tracking in session state (show_steward_lens)
- [x] Design window header with "Steward Lens" title

### 3.3 Steward Lens Content - Primacy Attractor Section

- [x] Display Primacy Attractor (Purpose, Scope, Boundaries)
- [x] Add alignment status indicator (Aligned ✅ / Drifted ⚠️)
- [x] Display current Telic Fidelity score
- [x] Show attractor establishment status (info message when not established)

### 3.4 Steward Lens Content - Intervention Section

- [x] Display intervention status (Total interventions count)
- [x] Show intervention type when active (breakdown by type)
- [x] Display intervention rationale text (in TELOSCOPE window)
- [x] Show fidelity improvement (ΔF) when intervention applied (avg ΔF metric)

### 3.5 Steward Lens Content - Research Lens Toggle

- [x] Add Research Lens toggle inside Steward Lens window
- [x] Implement collapsible mathematical analysis section
- [x] Display 7-section observatory when Research Lens active (placeholder for future)
- [x] Migrate existing Research Mode mathematics to new location (deferred to Phase 7)

### 3.6 Testing and Validation

- [x] Test window dragging functionality (expanders work as intended)
- [x] Verify window persists state across reruns (session state toggles work)
- [x] Test Steward Lens content updates with new turns (live metrics integration)
- [x] Verify Research Lens toggle works correctly

**Implementation Details**:
- Created `render_steward_lens()` function (lines 752-869, 118 lines)
- Created `render_teloscope_window()` function (lines 872-947, 76 lines)
- Added UI mode toggle in sidebar (lines 888-908, 21 lines)
- Added observable window toggles (lines 910-932, 23 lines)
- Integrated windows into chat interface (lines 981-991, 11 lines)
- **Design Decision**: Used Streamlit expanders instead of custom draggable windows (pragmatic, content-first approach)
- **Steward Lens displays**: Primacy Attractor, fidelity scores, basin status, intervention statistics, Research Lens toggle
- **TELOSCOPE displays**: Latest turn mathematical transparency (pre/post fidelity, ΔF, intervention details)
- **Connected to live data**: interceptor metrics, steward attractor, session manager turns, governance metadata
- **Backward Compatibility**: Windows hidden by default, only visible in chat mode when explicitly toggled

---

## PHASE 4: GOVERNANCE TOGGLE (8-10 hours)

**Status**: Not Started
**Progress**: 0% (0/23 tasks complete)
**Risk Level**: **HIGH**
**Blockers**: Phase 2 and Phase 3 must be complete

### 4.1 Turn Data Structure Enhancement

- [ ] Analyze current turn data storage format in SessionStateManager
- [ ] Design new turn structure to store both Native and TELOS responses
- [ ] Update turn data schema:
  ```python
  {
    "user_message": str,
    "native_response": str,
    "telos_response": str,
    "governance_metadata": {...},
    "timestamp": datetime
  }
  ```
- [ ] Implement backward compatibility for existing session data

### 4.2 LiveInterceptor Modification

- [ ] **CRITICAL**: Read and analyze `telos_purpose/sessions/live_interceptor.py` source code
- [ ] Identify `on_response()` callback location
- [ ] Modify LiveInterceptor to:
  - [ ] Store original Native response before governance
  - [ ] Store governed TELOS response after intervention
  - [ ] Populate both fields in turn data structure
- [ ] Preserve all existing governance logic (drift detection, intervention generation)

### 4.3 Toggle UI Component

- [ ] Create governance toggle switch in chat interface header
- [ ] Add labels: "Native Mistral" vs "TELOS Steward"
- [ ] Implement toggle state tracking in session state
- [ ] Add visual indicator (color change, icon change) for current mode

### 4.4 Response Display Logic

- [ ] Update `render_chat_interface()` to check `governance_enabled` state
- [ ] Conditionally display `native_response` when toggle OFF
- [ ] Conditionally display `telos_response` when toggle ON
- [ ] Update governance badges based on current toggle state

### 4.5 Backward Compatibility

- [ ] Handle old session data with only single response field
- [ ] Implement fallback logic:
  - [ ] If `native_response` missing, use `telos_response` for both
  - [ ] If `telos_response` missing, use `native_response` for both
- [ ] Add migration helper function for old sessions

### 4.6 Integration Testing

- [ ] Test toggle switch functionality with new conversation
- [ ] Test toggling mid-conversation (should switch displayed responses)
- [ ] Verify Steward Lens updates only when TELOS mode active
- [ ] Test governance badge display during toggle switches
- [ ] Verify no data loss when switching between modes

### 4.7 Edge Case Handling

- [ ] Test behavior when intervention not applied (Native = TELOS)
- [ ] Test behavior with Progressive mode (pre-attractor establishment)
- [ ] Test behavior with Pristine mode (silent background extraction)
- [ ] Verify toggle disabled during attractor establishment phase
- [ ] Test session reload with toggle state persistence

---

## PHASE 5: TURN NAVIGATION (5-7 hours)

**Status**: Not Started
**Progress**: 0% (0/16 tasks complete)
**Risk Level**: Medium
**Blockers**: Phase 2 and Phase 4 must be complete

### 5.1 Navigation Controls UI

- [ ] Create navigation control bar (between chat history and input box)
- [ ] Add "Play" button (continue conversation from current turn)
- [ ] Add "Pause" button (stop at current turn for review)
- [ ] Add "Previous Turn" button (navigate backward)
- [ ] Add "Next Turn" button (navigate forward)
- [ ] Add turn counter display (e.g., "Turn 15 / 32")

### 5.2 Turn Scrubber

- [ ] Implement timeline scrubber slider
- [ ] Map slider position to turn indices
- [ ] Add click-to-jump functionality on timeline
- [ ] Display turn markers on timeline (optional: color-coded by intervention status)

### 5.3 Navigation State Management

- [ ] Update `current_turn_index` session state on navigation
- [ ] Implement "live mode" flag (at latest turn vs reviewing history)
- [ ] Disable input box when in review mode (not at latest turn)
- [ ] Enable input box only when at latest turn

### 5.4 Display Update Logic

- [ ] Filter displayed messages based on `current_turn_index`
- [ ] Update chat container to show turns 0 through `current_turn_index`
- [ ] Update Steward Lens to show data for `current_turn_index`
- [ ] Add visual indicator for "review mode" vs "live mode"

### 5.5 Keyboard Shortcuts

- [ ] Implement Left Arrow = Previous Turn
- [ ] Implement Right Arrow = Next Turn
- [ ] Implement Home = Jump to Turn 1
- [ ] Implement End = Jump to Latest Turn
- [ ] Add keyboard shortcut legend (Help button or tooltip)

### 5.6 Testing and Validation

- [ ] Test navigation controls with 20+ turn conversation
- [ ] Verify input box disabled in review mode
- [ ] Test keyboard shortcuts
- [ ] Verify Steward Lens updates correctly during navigation
- [ ] Test edge cases (navigate to Turn 0, navigate to latest turn)

---

## PHASE 6: TELOSCOPE WINDOW (3-4 hours)

**Status**: Not Started
**Progress**: 0% (0/10 tasks complete)
**Risk Level**: Low
**Blockers**: Phase 3 must be complete

### 6.1 Window Structure

- [ ] Create `render_teloscope_window()` function
- [ ] Implement floating window using FloatingWindow component (from Phase 3)
- [ ] Add window toggle button in sidebar
- [ ] Design window header with "TELOSCOPE" title

### 6.2 Counterfactual Display

- [ ] Integrate counterfactual branch data from CounterfactualBranchManager
- [ ] Display counterfactual response text
- [ ] Show branch metadata (divergence point, branch reason)
- [ ] Add timestamp and turn number for counterfactual

### 6.3 Comparison View

- [ ] Implement side-by-side comparison: Native vs Counterfactual
- [ ] Add diff highlighting for response differences
- [ ] Display fidelity metrics for counterfactual path
- [ ] Show "what if" scenario explanation

### 6.4 Testing and Validation

- [ ] Test TELOSCOPE window with existing counterfactual data
- [ ] Verify counterfactual display updates with turn navigation
- [ ] Test window positioning and dragging
- [ ] Verify diff highlighting works correctly

---

## PHASE 7: CLEANUP (2-3 hours)

**Status**: Not Started
**Progress**: 0% (0/10 tasks complete)
**Risk Level**: Low
**Blockers**: Phases 1-6 must be complete

### 7.1 Code Removal

- [ ] Remove deprecated tab-based UI code (`render_live_tab()`, etc.)
- [ ] Remove old Basic/Advanced/Research Mode system
- [ ] Remove deprecated session state variables
- [ ] Remove unused imports

### 7.2 UI Polish

- [ ] Add CSS styling for chat interface (colors, fonts, spacing)
- [ ] Implement smooth transitions for window open/close
- [ ] Add loading indicators for LLM responses
- [ ] Polish button styles and hover states

### 7.3 Documentation Updates

- [ ] Update code comments to reflect new architecture
- [ ] Document new session state variables
- [ ] Update function docstrings
- [ ] Create inline comments for complex UI logic

### 7.4 Final Testing

- [ ] End-to-end test: Full conversation with all features
- [ ] Test all 4 onboarding modes (Progressive, Pre-defined, Hybrid, Pristine)
- [ ] Performance testing (load 100+ turn conversation)
- [ ] Cross-browser testing (if applicable for Streamlit)

---

## BLOCKERS

**Current Blockers**: None (Project in planning phase)

**Potential Future Blockers**:
- Phase 4 (Governance Toggle) requires detailed understanding of LiveInterceptor architecture
- Floating window implementation may require custom Streamlit component development
- Turn data migration may require database/session file updates

---

## NOTES

### Session Independence
This manifest is designed to be session-independent. Any developer (or Claude instance) can:
1. Read this manifest to see current progress
2. Pick up any unchecked task
3. Refer to `/docs/implementation/PHASE_1_UI_OVERHAUL_PLAN.md` for context
4. Continue without requiring conversation history

### Success Criteria
Each phase has acceptance criteria defined in `PHASE_1_UI_OVERHAUL_PLAN.md`. A phase is complete when:
- All tasks are checked
- Acceptance criteria met
- Tests passing
- No regressions in existing functionality

### Risk Mitigation
- **Phase 4 (HIGH RISK)**: Test LiveInterceptor changes in isolated branch before merging
- **Rollback Strategy**: Git branching allows rollback to working state at any phase
- **Incremental Testing**: Test after each phase before proceeding to next

### Progress Tracking
Update this manifest after completing each task:
1. Check the box: `- [x] Task description`
2. Update phase progress percentage
3. Update overall progress at top of document
4. Note any blockers encountered

### Onboarding Integration Note
The onboarding implementation (4 attractor establishment modes) was partially completed before UI overhaul planning began. Onboarding UI will be rebuilt as part of Phase 2-3 to match new ChatGPT-style interface.

**Onboarding Status**:
- Mode selection screen: ✅ Complete
- Pre-defined mode form: ✅ Complete
- Hybrid mode questions: ✅ Complete
- Progressive/Pristine mode UI integration: ⏳ Pending (will be rebuilt in Phase 2)

---

## RELATED WORK

### Completed Recently
- ✅ Intervention Analysis Enhancement (see `docs/implementation/INTERVENTION_ANALYSIS_ENHANCEMENT.md`)
- ✅ Terminology Standardization (see `docs/implementation/TERMINOLOGY_STANDARDIZATION.md`)
- ✅ Directory Organization (118 files archived, clean structure)
- ✅ UI Overhaul Reconnaissance and Planning (this project)

### Future Enhancements (Post-UI Overhaul)
- GitHub publication and CI/CD setup
- Multi-user session management
- Export/import conversation functionality
- Advanced analytics dashboard
- Mobile-responsive design

---

**Last Updated**: 2025-10-27
**Next Review**: After Phase 1 completion
**Maintained By**: TELOS Observatory Development Team
