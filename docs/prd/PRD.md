# TELOS Observatory - Product Requirements Document (PRD)

## Version 1.00 Release Requirements

**Last Updated**: 2025-10-30
**Target Release**: Q4 2025
**Status**: 87% Complete

**Recent Progress**:
- ✅ Observatory Phase 1 Complete (standalone app with TELOSCOPE)
- ✅ 10/11 Phase 1 tasks complete (91%)
- ⏳ Live testing pending

---

## Product Vision

**TELOS is mathematical infrastructure for observable, quantifiable AI governance.**

We provide counterfactual evidence proving governance effectiveness through the TELOSCOPE Observatory - a research instrument that generates statistical evidence of AI alignment.

---

## V1.00 Success Criteria

Version 1.00 is achieved when:

1. 🔨 **Interface Complete** - Minimalistic ChatGPT-style research instrument with Observation Deck
2. ✅ **Core Governance Functional** - UnifiedGovernanceSteward with Primacy Attractor
3. ✅ **Counterfactual System Working** - TELOSCOPE generates comparative evidence
4. ⏳ **Pilot Data Generated** - 3-5 documented test conversations
5. ⏳ **Documentation Package** - Pilot Brief + Grant Package ready
6. ⏳ **Validation Suite** - Comprehensive tests prove robustness

---

## Required Deliverables

### 1. Pilot Brief 📄
**Status**: Not Started
**Owner**: TBD
**Dependencies**: Test conversations must be completed first

**Requirements**:
- Methodology description (how TELOS governance works)
- Test design (conversation scenarios, evaluation criteria)
- Hypothesis statement (what we expect TELOS to achieve)
- Data collection protocol (what metrics we track)
- Analysis framework (how we measure success)

**Acceptance Criteria**:
- [ ] Clear explanation of TELOS governance mechanism
- [ ] Defined test scenarios with rationale
- [ ] Measurable success metrics identified
- [ ] Reproducible methodology documented

---

### 2. Test Conversations 💬
**Status**: Not Started
**Owner**: TBD
**Dependencies**: Interface complete (✅), session saving implemented (✅)

**Requirements**:
- Run 3-5 pilot conversations demonstrating governance
- Save full session data (saved_sessions/ directory)
- Export evidence packages (JSON, CSV, Transcript, Report)
- Document scenarios and outcomes

**Acceptance Criteria**:
- [ ] At least 3 conversations saved with full metadata
- [ ] Mix of intervention scenarios (drift detected, aligned paths)
- [ ] Quantifiable ΔF (fidelity improvement) metrics captured
- [ ] Exportable evidence for each conversation

---

### 3. Comparative Summary JSON 📊
**Status**: Not Started
**Owner**: TBD
**Dependencies**: Test conversations completed

**Requirements**:
- `comparative_summary.json` file with aggregated pilot results
- Statistical analysis of governance effectiveness
- Native vs TELOS comparison metrics
- Evidence of ΔF improvements across conversations

**Structure**:
```json
{
  "pilot_metadata": {...},
  "conversations": [...],
  "aggregate_metrics": {
    "avg_fidelity_native": 0.XX,
    "avg_fidelity_telos": 0.XX,
    "avg_delta_f": 0.XX,
    "interventions_total": XX,
    "interventions_successful": XX
  },
  "statistical_significance": {...}
}
```

**Acceptance Criteria**:
- [ ] Valid JSON file generated
- [ ] All pilot conversations included
- [ ] Statistical metrics calculated
- [ ] Evidence of governance effectiveness demonstrated

---

### 4. Grant Package 📦
**Status**: Not Started
**Owner**: TBD
**Dependencies**: All other deliverables complete

**Requirements**:
- Compilation of all evidence and documentation
- Executive summary of TELOS capabilities
- Technical specifications
- Pilot results and statistical analysis
- Future roadmap and compliance goals

**Contents**:
1. Executive Summary (1-2 pages)
2. Pilot Brief (methodology)
3. Comparative Summary JSON (results)
4. Technical Documentation (architecture)
5. Evidence Exports (transcripts, visualizations)
6. Roadmap (V2.0 and beyond)

**Acceptance Criteria**:
- [ ] Professional formatting
- [ ] Complete evidence package
- [ ] Clear value proposition
- [ ] Regulatory compliance positioning

---

### 5. Comprehensive Testing Suite 🧪
**Status**: Partially Complete
**Owner**: TBD
**Dependencies**: Core components complete (✅)

**Requirements**:
- Edge case testing (boundary conditions, error handling)
- Integration tests (end-to-end conversation flows)
- Performance validation (100+ turn conversations)
- Data validation scripts (session integrity, export accuracy)

**Test Categories**:
- ✅ Unit tests for core components (existing)
- ⏳ Edge case testing suite
- ⏳ Integration testing suite
- ⏳ Performance benchmarks
- ⏳ Data validation scripts

**Acceptance Criteria**:
- [ ] All critical paths have test coverage
- [ ] Edge cases documented and tested
- [ ] Performance baseline established
- [ ] No critical bugs in production paths

---

### 6. TELOS Observatory with TELOSCOPE 🔭
**Status**: Phase 1 Complete (91%), Phase 1.5 Foundation Complete (100%)
**Owner**: Observatory Team
**Dependencies**: Interface complete (✅), backend telemetry complete (✅)

**Phase 1 Deliverables** (✅ Complete):
- **Standalone Observatory App**: `telos_observatory/` with complete architecture
- **TELOSCOPE Navigation**: Frame-by-frame control system with timeline scrubber
- **Distance-Based Dimming**: Visual focus system (opacity 1.0→0.7→0.4→0.2)
- **Mock Data System**: 12-turn session demonstrating calibration, drift, intervention, recovery
- **Complete Documentation**: Streamlit patterns, architecture guide, user guide
- **Turn Synchronization**: All components read from unified `st.session_state.current_turn`

**Phase 1.5 Deliverables** (✅ Foundation Complete, Core Components Pending):
- **TELOSCOPE v2 Spec Implementation**: Production-grade control system in `teloscope_v2/`
- **Centralized State Management**: Nested `st.session_state.teloscope` namespace
- **Enhanced Mock Data**: Multiple session templates with rich metadata
- **Advanced Components**: Turn indicator, marker generator, scroll controller
- **Integration Reconciliation**: Coexistence strategy for Phase 1 and v2
- **Documentation**: Complete reconciliation guide and architecture specs

**Phase 1.5 Architecture**:
```
telos_observatory/
├── teloscope/              # Phase 1 (working prototype - frozen)
│   ├── teloscope_controller.py
│   ├── navigation_controls.py
│   └── timeline_scrubber.py
└── teloscope_v2/           # Spec v2 (production build - active)
    ├── components/
    │   ├── navigation_controls.py    # Enhanced
    │   ├── timeline_scrubber.py      # Enhanced
    │   ├── tool_buttons.py           # NEW
    │   ├── position_manager.py       # NEW
    │   └── turn_indicator.py         # NEW (✅ Complete)
    ├── state/
    │   └── teloscope_state.py        # NEW (✅ Complete)
    └── utils/
        ├── mock_data.py              # Enhanced (✅ Complete)
        ├── marker_generator.py       # NEW (✅ Complete)
        └── scroll_controller.py      # NEW (✅ Complete)
```

**Phase 1.5 Progress**:
- Week 1-2 Foundation: ✅ Complete (6/6 tasks, 100%)
- Week 3-4 Core Components: ⏳ Pending (0/6 tasks, 0%)

**Phase 2 Requirements** (⏳ Pending):
- Transform standalone app into integrated research platform
- Collapsible research panel with TELOSCOPIC Tools and Steward integration
- Two control strip systems (Observatory + Observation Deck)
- Dynamic layout system (4 column width states)
- Integration with live TELOS sessions

**Components**:
- **Observatory Control Strip** (top-right thermometer): Turn counter, fidelity gauge, calibration progress
- **Observation Deck Control Strip** (sidebar header): Telescope toggle, symbolic flow, stats
- **TELOSCOPIC Tools** (FREE): Comparison Viewer, Calculation Window, Turn Navigator
- **Steward Integration** (PAID): Conversational Q&A about sessions (~$0.002/query)
- **Calibration Logger**: Mistral reasoning visualization (Turns 1-3)
- **Symbolic Flow**: Governance pipeline animator (👤→⚡→🔄→🤖→✓)

**Architecture**:
```
telos_purpose/dev_dashboard/observation_deck/
  ├── observatory_control_strip.py
  ├── deck_control_strip.py
  ├── teloscopic_tools/
  │   ├── comparison_viewer.py
  │   ├── calculation_window.py
  │   └── turn_navigator.py
  ├── steward_integration/
  │   └── steward_chat.py
  ├── calibration_logger.py
  └── symbolic_flow.py
```

**Phase 1 Acceptance Criteria** (✅ Complete):
- [x] Standalone Observatory app created (`telos_observatory/`)
- [x] TELOSCOPE navigation controls implemented (First/Prev/Play/Next/Last)
- [x] Timeline scrubber with color-coded markers (✓⚠️⚡⚙️)
- [x] Distance-based dimming algorithm (opacity based on distance from active turn)
- [x] Turn synchronization via `st.session_state.current_turn`
- [x] Auto-play mode with configurable speed
- [x] Control strip showing turn/fidelity/status
- [x] Mock data system with 12 realistic turns
- [x] Complete documentation (patterns, architecture, user guide)
- [x] All imports validated, no syntax errors

**Phase 2 Acceptance Criteria** (⏳ Pending):
- [ ] Observatory Control Strip functional (top-right position)
- [ ] Observation Deck Control Strip functional (sidebar header)
- [ ] Deck opens/closes with dynamic width adjustment
- [ ] At least 2 of 3 TELOSCOPIC Tools working (Comparison + Calculation minimum)
- [ ] Calibration Logger shows Turns 1-3 attractor formation
- [ ] Basic Steward integration (text Q&A working)
- [ ] All 4 layout states tested and functional
- [ ] Turn marker synchronization with existing dashboard

**Timeline**:
- Phase 1: ✅ Complete (10/11 tasks, 91%)
- Phase 1.5 Foundation: ✅ Complete (6/6 tasks, 100%)
- Phase 1.5 Core: ⏳ Pending (0/6 tasks, Week 3-4)
- Phase 2: 3-week implementation (Integration with existing dashboard)

---

## Technical Requirements

### Interface (🔨 In Progress - Observation Deck)
- [x] Minimalistic ChatGPT-style UI
- [x] Dark/Light mode toggle
- [x] Native Mistral ↔ TELOS toggle
- [x] Session saving/loading (research instrument)
- [x] Floating draggable windows (STEWARD, TELOSCOPE)
- [x] Keyboard shortcuts for rapid access
- [x] Export capabilities (JSON, CSV, Transcript, Report)
- [ ] Observation Deck (collapsible research panel) - **In Progress**
- [ ] Observatory Control Strip (top-right thermometer) - **In Progress**
- [ ] TELOSCOPIC Tools (Comparison, Calculation, Turn Navigator) - **In Progress**
- [ ] Steward Integration (conversational Q&A) - **In Progress**

### Core Governance (✅ Complete)
- [x] UnifiedGovernanceSteward
- [x] PrimacyAttractor (Purpose, Scope, Boundaries)
- [x] Drift detection and intervention
- [x] Telic Fidelity (F) calculation
- [x] ΔF (fidelity improvement) tracking

### Counterfactual System (✅ Complete)
- [x] CounterfactualBranchManager
- [x] BranchComparator
- [x] Native vs TELOS response storage
- [x] Mathematical transparency (7-step observatory)
- [x] Evidence generation and export

### Research Instrument (✅ Complete)
- [x] WebSessionManager
- [x] LiveInterceptor
- [x] Session persistence (saved_sessions/)
- [x] Turn-by-turn metadata tracking
- [x] Export functionality

---

## User Acceptance Criteria

A researcher using TELOS Observatory should be able to:

1. ✅ **Start a conversation** with governance enabled or disabled
2. ✅ **Toggle between Native Mistral and TELOS** mid-conversation
3. ✅ **View live governance** through STEWARD window
4. ✅ **Inspect mathematics** through TELOSCOPE window
5. ✅ **Save sessions** for later review
6. ✅ **Export evidence** in multiple formats
7. ⏳ **Reproduce results** from saved sessions
8. ⏳ **Understand methodology** from documentation
9. ⏳ **Validate effectiveness** from pilot data

---

## Open Questions for V1.00

1. **Pilot Scenarios** - What conversation types best demonstrate TELOS?
2. **Success Metrics** - What ΔF threshold indicates "effective" governance?
3. **Documentation Format** - What level of technical detail for grant package?
4. **Validation Approach** - Statistical tests for significance?

---

## V1.00 Critical Path

```
1. Run Pilot Conversations (Week 1)
   ↓
2. Write Pilot Brief (Week 1-2)
   ↓
3. Generate comparative_summary.json (Week 2)
   ↓
4. Complete Testing Suite (Week 2-3)
   ↓
5. Assemble Grant Package (Week 3)
   ↓
6. V1.00 RELEASE
```

**Estimated Timeline**: 3-4 weeks from start of pilot conversations

---

## Post-V1.00 Roadmap

### V1.1 - Enhanced Testing
- Expanded test suite
- Performance optimization
- Edge case hardening

### V1.5 - Governance Toggle Complete
- Phase 4 of UI overhaul (REPO_MANIFEST.md)
- Native vs TELOS comparison in UI
- Turn navigation with keyboard shortcuts

### V2.0 - Production Ready
- Database persistence
- Multi-user support
- Advanced analytics dashboard

---

## References

- **Build Manifest**: `archive/TELOS_BUILD_MANIFEST.md` - Overall platform status
- **Repository Manifest**: `REPO_MANIFEST.md` - UI overhaul phases
- **STEWARD**: `STEWARD.md` - Project manager tracking
- **Architecture**: `docs/TELOS_Architecture_and_Development_Roadmap.md` (if exists)

---

**Maintained By**: TELOS Development Team
**Review Cadence**: Weekly during V1.00 sprint
**Update Protocol**: Update after completing each deliverable
