# TELOS Observatory - Complete Wiring Plan
## From Pre-Wired to Demo-Ready

**Status**: v0.9-prewired → v1.0-demo-ready
**Timeline**: 3-5 days intensive work
**Goal**: Fully integrated Observatory UI showcasing Phase 2 validation

---

## Current State (v0.9-prewired)

### What Works (Command Line) ✅
- Phase 2 validation pipeline (`phase2_validation_run.py`)
- Research brief generation (67 briefs created)
- Evidence file exports (JSON + Markdown)
- Statistical analysis and reporting
- Data format converters
- Reusable skill (`phase2-validation-run`)

### What Exists (Observatory UI) ⏸️
- Teloscope v2 foundation components
- Mock data generation
- Turn indicator and timeline markers
- Scroll controller with dimming
- State management system
- Counterfactual comparison viewer (Phase 1.5B)

### What's Missing (Integration) ❌
- Phase 2 results loading into UI
- Live research brief viewer
- Interactive study browser
- Demo mode with real validation data
- Counterfactual visualization for Phase 2 studies

---

## Integration Architecture

### Data Flow

```
Phase 2 Validation Results
    ↓
telos_observatory/phase2_study_results/
    ├── phase2_study_summary.json           → Study selector
    ├── {conversation_id}/
    │   ├── intervention_{id}.json          → Counterfactual viewer
    │   └── intervention_{id}.md            → Evidence display
    ↓
Observatory UI (main_observatory_v2.py)
    ├── Study Browser Tab
    ├── Research Brief Viewer Tab
    ├── Counterfactual Comparison Tab
    └── Statistics Dashboard Tab
```

---

## Wiring Tasks (Priority Order)

### Task 1: Phase 2 Results Loader
**File**: `telos_observatory/teloscope_v2/utils/phase2_loader.py`

**Purpose**: Load Phase 2 validation results into Observatory

**Functions needed**:
```python
def load_study_summary(summary_path: str) -> Dict
def load_study_evidence(study_id: str, results_dir: str) -> Dict
def load_research_brief(brief_path: str) -> str
def get_available_studies(results_dir: str) -> List[Dict]
def get_study_statistics(summary_data: Dict) -> Dict
```

**Status**: TO CREATE

---

### Task 2: Study Browser Component
**File**: `telos_observatory/teloscope_v2/components/study_browser.py`

**Purpose**: Browse and select Phase 2 validation studies

**Features**:
- List all 56 completed studies
- Filter by dataset (ShareGPT, Test Sessions, Edge Cases)
- Show key metrics (ΔF, effectiveness, turns)
- Select study for detailed view

**UI Layout**:
```
┌─────────────────────────────────────────┐
│ TELOS Phase 2 Validation Studies        │
├─────────────────────────────────────────┤
│ Filter: [All] [ShareGPT] [Tests] [Edge] │
│                                          │
│ Study                    ΔF    Status    │
│ ─────────────────────────────────────   │
│ sharegpt_filtered_3    +0.153  ✅       │
│ excellent_session_001  +0.149  ✅       │
│ high_drift_session_001 +0.109  ✅       │
│ ...                                      │
└─────────────────────────────────────────┘
```

**Status**: TO CREATE

---

### Task 3: Research Brief Viewer
**File**: `telos_observatory/teloscope_v2/components/brief_viewer.py`

**Purpose**: Display research briefs with interactive navigation

**Features**:
- Render markdown research briefs
- Highlight mock researcher questions
- Navigate sections (PA establishment, drift, counterfactual)
- Export as PDF for grant materials

**UI Layout**:
```
┌─────────────────────────────────────────┐
│ Research Brief: sharegpt_filtered_3     │
├─────────────────────────────────────────┤
│ [Overview] [PA Analysis] [Drift] [CF]   │
│                                          │
│ PHASE 1: PRIMACY ATTRACTOR              │
│ ═══════════════════════════════════════ │
│                                          │
│ Turn-by-turn LLM semantic analysis...   │
│                                          │
│ RESEARCHER QUESTION: "What is the       │
│ nature of this conversation?"           │
│                                          │
│ [Export PDF] [Share]                    │
└─────────────────────────────────────────┘
```

**Status**: TO CREATE

---

### Task 4: Phase 2 Counterfactual Viewer
**File**: Update existing `comparison_viewer_v2.py`

**Purpose**: Visualize Phase 2 counterfactual branches

**Features**:
- Load Phase 2 intervention JSON
- Display original vs TELOS trajectories
- Show ΔF calculation
- Highlight intervention points

**Integration**:
```python
# Adapt Phase 2 intervention data to viewer format
from telos_observatory.teloscope_v2.utils.phase2_adapter import Phase2Adapter

adapter = Phase2Adapter(intervention_json_path)
comparison_data = adapter.to_comparison_format()

viewer = ComparisonViewerV2()
viewer.render(comparison_data)
```

**Status**: TO UPDATE

---

### Task 5: Statistics Dashboard
**File**: `telos_observatory/teloscope_v2/components/stats_dashboard.py`

**Purpose**: Aggregate statistics across all studies

**Features**:
- Overall effectiveness (66.7% ShareGPT, 81.8% internal tests)
- ΔF distribution histogram
- Dataset comparison table
- Success/failure breakdown

**UI Layout**:
```
┌─────────────────────────────────────────┐
│ Phase 2 Validation Statistics           │
├─────────────────────────────────────────┤
│ Total Studies: 56                        │
│ Completed: 56 | Failed: 5               │
│                                          │
│ Effectiveness by Dataset:                │
│ ├─ ShareGPT:      66.7% (30/45)         │
│ ├─ Test Sessions: 71.4% (5/7)           │
│ └─ Edge Cases:    100%  (4/4)           │
│                                          │
│ Average ΔF: +0.010 (ShareGPT)           │
│            +0.073 (Internal Tests)       │
│                                          │
│ [ΔF Distribution Chart]                 │
└─────────────────────────────────────────┘
```

**Status**: TO CREATE

---

### Task 6: Demo Mode
**File**: `telos_observatory/teloscope_v2/utils/demo_mode.py`

**Purpose**: Curated demo experience for grant presentations

**Features**:
- Showcase 3-5 best studies
- Guided tour through features
- Auto-play mode for presentations
- Highlight key findings

**Demo Flow**:
1. Start → Statistics overview
2. Select excellent_session_001 (ΔF = +0.149)
3. Show research brief with researcher questions
4. Display counterfactual comparison
5. Show ΔF calculation
6. End → Call to action (infrastructure vision)

**Status**: TO CREATE

---

## Implementation Order

### Phase 1: Core Integration (Days 1-2)
1. ✅ Create `phase2_loader.py` - Load validation results
2. ✅ Create `study_browser.py` - Browse studies UI
3. ✅ Update `main_observatory_v2.py` - Add Phase 2 tab

### Phase 2: Visualization (Days 2-3)
4. ✅ Create `brief_viewer.py` - Research brief display
5. ✅ Update `comparison_viewer_v2.py` - Phase 2 adapter
6. ✅ Create `phase2_adapter.py` - Data format adapter

### Phase 3: Polish & Demo (Days 3-4)
7. ✅ Create `stats_dashboard.py` - Aggregate statistics
8. ✅ Create `demo_mode.py` - Curated demo experience
9. ✅ Test full system end-to-end

### Phase 4: Documentation (Day 5)
10. ✅ Update README with demo instructions
11. ✅ Create demo script for presentations
12. ✅ Record demo video (optional)

---

## Observatory UI Structure (Post-Wiring)

```
main_observatory_v2.py
├── Tab 1: Live Session (existing mock data)
├── Tab 2: Phase 2 Studies ← NEW
│   ├── Study Browser
│   ├── Research Brief Viewer
│   └── Counterfactual Comparison
├── Tab 3: Statistics Dashboard ← NEW
│   ├── Aggregate metrics
│   ├── Dataset comparisons
│   └── ΔF distributions
└── Tab 4: Demo Mode ← NEW
    └── Guided tour
```

---

## Testing Checklist

### Data Loading
- [ ] Load phase2_study_summary.json
- [ ] Load individual intervention JSONs
- [ ] Load research briefs (markdown)
- [ ] Handle missing/corrupted data gracefully

### UI Components
- [ ] Study browser renders all 56 studies
- [ ] Filtering by dataset works
- [ ] Study selection updates detail view
- [ ] Research brief renders markdown correctly
- [ ] Counterfactual comparison shows trajectories
- [ ] Statistics dashboard calculates correctly

### Demo Mode
- [ ] Auto-advance through studies works
- [ ] Highlighting draws attention to key points
- [ ] Export functionality generates clean outputs
- [ ] Performance acceptable on laptop

### Cross-Browser
- [ ] Works in Chrome
- [ ] Works in Firefox
- [ ] Works in Safari
- [ ] Mobile-responsive (bonus)

---

## Success Criteria (v1.0-demo-ready)

### Functional
✅ All 56 Phase 2 studies loadable
✅ Research briefs viewable in UI
✅ Counterfactual comparisons visualized
✅ Statistics dashboard shows aggregate metrics
✅ Demo mode runs smoothly

### User Experience
✅ Intuitive navigation between studies
✅ Fast loading (<2 seconds per study)
✅ Clear visual hierarchy
✅ Professional appearance for demos

### Grant Application Ready
✅ Screenshots for application materials
✅ Demo video recordable
✅ Impressive first impression
✅ "Hands-on research lab" feel achieved

---

## Post-Wiring: Grant Application

Once wiring complete, create:
1. **LTFF Application** (based on infrastructure framing)
2. **Screenshots** (UI showcasing key features)
3. **Demo Script** (for video/presentations)
4. **Technical Documentation** (for reviewers)

---

## Notes

- **Priority**: Get something demo-ready FAST, then polish
- **Philosophy**: Working > perfect
- **Timeline**: 3-5 days if focused
- **Blocker**: None - all data exists, just needs UI wiring

---

**Next Action**: Start with `phase2_loader.py` - the foundation for everything else.
