# TELOSCOPE v2 - Production-Grade Control System

**Version**: 2.0.0-spec
**Status**: Foundation Complete (6/6), Phase 1.5B Complete (5/5), Core Components Pending (0/6)
**Phase**: Week 1-2 Complete ✅, Phase 1.5B Complete ✅, Week 3-4 Pending ⏳
**Last Updated**: 2025-10-30

---

## Purpose

TELOSCOPE v2 is the **production-grade implementation** of the TELOSCOPE specification.

It implements the full spec with:
- Centralized state management
- Advanced navigation controls
- Tool toggle system
- Position management (drag-and-drop)
- Enhanced timeline with marker generation
- Scroll synchronization utilities

**Coexistence**: v2 is built alongside Phase 1 prototype during development. Both implementations coexist safely.

---

## Folder Structure

```
teloscope_v2/
├── components/               # UI components
│   ├── turn_indicator.py        # ✅ Turn X/Y display, jump-to
│   ├── navigation_controls.py   # ⏳ Enhanced Prev/Play/Next
│   ├── timeline_scrubber.py     # ⏳ Advanced timeline with markers
│   ├── tool_buttons.py          # ⏳ Tool toggle controls
│   └── position_manager.py      # ⏳ Drag-and-drop positioning
├── state/                    # State management
│   └── teloscope_state.py       # ✅ Centralized nested state
└── utils/                    # Utilities
    ├── mock_data.py             # ✅ Enhanced mock data generator
    ├── marker_generator.py      # ✅ Timeline marker generation
    ├── scroll_controller.py     # ✅ Scroll & dimming controller
    ├── baseline_adapter.py      # ✅ Phase 1.5B: Baseline runner wrapper
    ├── comparison_adapter.py    # ✅ Phase 1.5B: Branch comparator wrapper
    ├── evidence_exporter.py     # ✅ Phase 1.5B: Evidence generator
    ├── sharegpt_importer.py     # ✅ Phase 1.5B: ShareGPT parser
    └── comparison_viewer_v2.py  # ✅ Phase 1.5B: UI comparison component
```

**Legend**:
- ✅ Complete (Week 1-2 Foundation + Phase 1.5B Counterfactual)
- ⏳ Pending (Week 3-4 Core Components)

---

## Foundation Components (✅ Complete)

### 1. **Centralized State Management**
**File**: `state/teloscope_state.py`

**Features**:
- Nested `st.session_state.teloscope` namespace
- Getter/setter helpers for all state
- Navigation helpers (current_turn, playing, speed)
- Tool management (toggle, active state)
- Position tracking (fixed/floating/hidden)
- Migration helpers (Phase 1 → v2)

**Usage**:
```python
from teloscope_v2.state.teloscope_state import (
    init_teloscope_state,
    get_current_turn,
    set_current_turn,
    is_playing,
    set_playing,
)

# Initialize
init_teloscope_state(session_data)

# Use
current = get_current_turn()
set_current_turn(5, rerun=True)
```

---

### 2. **Enhanced Mock Data Generator**
**File**: `utils/mock_data.py`

**Features**:
- 4 session templates (stable, high-drift, intervention-heavy, perfect-alignment)
- Rich metadata (timestamps, drift reasons, intervention details)
- Annotation support for timeline features
- Validation and export helpers

**Usage**:
```python
from teloscope_v2.utils.mock_data import generate_enhanced_session

session = generate_enhanced_session(
    turns=20,
    session_type='high-drift',
    include_annotations=True,
    seed=42,  # Reproducible
)
```

---

### 3. **Turn Indicator Component**
**File**: `components/turn_indicator.py`

**Features**:
- Turn X/Y display with current position
- Jump-to input with validation
- First/Last navigation buttons
- Multiple layouts (compact, full, inline, progress bar)

**Usage**:
```python
from teloscope_v2.components.turn_indicator import render_turn_indicator

# Compact layout
render_turn_indicator(total_turns=50, compact=True)

# Progress bar mode
render_turn_progress_bar(total_turns=50)
```

---

### 4. **Timeline Marker Generator**
**File**: `utils/marker_generator.py`

**Features**:
- 3 configurable styles (minimal, standard, enhanced)
- Custom color mapping
- Tooltip and click-to-jump support
- Annotation rendering above markers
- Legend generation
- Performance caching

**Usage**:
```python
from teloscope_v2.utils.marker_generator import (
    generate_timeline_markers,
    generate_timeline_legend,
)

# Generate markers
html = generate_timeline_markers(
    turns=session['turns'],
    current_turn=5,
    style='enhanced',
    show_tooltips=True,
)
st.markdown(html, unsafe_allow_html=True)

# Generate legend
legend = generate_timeline_legend(compact=True)
st.markdown(legend, unsafe_allow_html=True)
```

---

### 5. **Scroll Controller**
**File**: `utils/scroll_controller.py`

**Features**:
- Multiple scroll strategies (instant, smooth, delayed)
- Anchor system for Streamlit scroll control
- Viewport visibility detection
- Auto-scroll during playback
- Dimming algorithm (extracted from Phase 1)

**Usage**:
```python
from teloscope_v2.utils.scroll_controller import (
    scroll_to_turn,
    calculate_turn_opacity,
)

# Scroll to turn
scroll_to_turn(turn_index=5, behavior='smooth')

# Calculate opacity for dimming
opacity = calculate_turn_opacity(turn_index=3, active_turn_index=5)
# Returns 0.4 (distance of 2)
```

---

## Phase 1.5B: Counterfactual Analysis (✅ Complete)

**Purpose**: THE core feature for V1.00 validation - proving TELOS governance works through quantitative evidence.

Phase 1.5B adds counterfactual analysis capabilities to Observatory v2, enabling:
- Baseline vs TELOS comparison
- ΔF (fidelity improvement) calculation
- Statistical significance testing
- Evidence export for research/grants
- ShareGPT conversation import

### 1. **Baseline Adapter**
**File**: `utils/baseline_adapter.py`

**Purpose**: Wrapper around `telos_purpose.validation.baseline_runners`

**Features**:
- 5 baseline types (stateless, prompt-only, cadence, observation, TELOS)
- Simplified comparison interface
- Observatory integration helpers

**Usage**:
```python
from teloscope_v2.utils.baseline_adapter import BaselineAdapter

adapter = BaselineAdapter(llm_client, embedding_provider, attractor_config)

# Run comparison
results = adapter.run_comparison(conversation, baseline_type='stateless')

# Display
baseline_f = results['baseline'].final_metrics['fidelity']
telos_f = results['telos'].final_metrics['fidelity']
delta_f = telos_f - baseline_f
```

---

### 2. **Comparison Adapter**
**File**: `utils/comparison_adapter.py`

**Purpose**: Wrapper around `telos_purpose.validation.branch_comparator`

**Features**:
- Branch-to-branch comparison
- ΔF calculation
- Statistical significance testing (t-test, Cohen's d, confidence intervals)
- Plotly visualizations (divergence charts, dashboards)
- Metrics tables for Streamlit

**Usage**:
```python
from teloscope_v2.utils.comparison_adapter import ComparisonAdapter

adapter = ComparisonAdapter()

# Compare
comparison = adapter.compare_results(baseline_branch, telos_branch)

# Get ΔF
delta_f = comparison['delta']['delta_f']  # e.g., +0.47

# Generate chart
fig = adapter.generate_chart(comparison, chart_type='divergence')
st.plotly_chart(fig)

# Generate metrics table
df = adapter.generate_metrics_table(comparison)
st.dataframe(df)
```

---

### 3. **Evidence Exporter**
**File**: `utils/evidence_exporter.py`

**Purpose**: Generate downloadable evidence packages for research/grants

**Features**:
- JSON export (machine-readable)
- Markdown export (human-readable report)
- Grant application language
- Statistical analysis formatting

**Usage**:
```python
from teloscope_v2.utils.evidence_exporter import EvidenceExporter

exporter = EvidenceExporter()

# Export JSON
json_data = exporter.create_download_data(comparison, format='json')
st.download_button("Download Evidence", data=json_data, file_name="evidence.json")

# Export Markdown
md_data = exporter.create_download_data(comparison, format='markdown')
st.download_button("Download Report", data=md_data, file_name="report.md")
```

**Markdown Report Includes**:
- Executive summary
- Statistical significance analysis
- Turn-by-turn comparison
- Grant application language
- Reproducibility information

---

### 4. **ShareGPT Importer**
**File**: `utils/sharegpt_importer.py`

**Purpose**: Import ShareGPT conversations for batch testing

**Features**:
- Multiple format support (ShareGPT v1, OpenAI chat, generic)
- Format validation
- Metadata preservation
- Batch import

**Usage**:
```python
from teloscope_v2.utils.sharegpt_importer import ShareGPTImporter

importer = ShareGPTImporter()

# Import file
sessions = importer.import_file('conversations.json')

# Validate
for session in sessions:
    if importer.validate_session(session):
        # Run TELOS analysis
        conversation = adapter.convert_session_to_conversation(session)
        result = baseline_adapter.run_baseline('telos', conversation)
```

**ShareGPT Format**:
```json
{
  "conversations": [
    {
      "id": "conv_123",
      "turns": [
        {"from": "human", "value": "What is TELOS?"},
        {"from": "gpt", "value": "TELOS is..."}
      ]
    }
  ]
}
```

---

### 5. **Comparison Viewer v2**
**File**: `utils/comparison_viewer_v2.py`

**Purpose**: Side-by-side UI comparison component

**Features**:
- Split-view rendering (Baseline | TELOS)
- Turn-by-turn comparison
- Intervention highlighting
- Fidelity color-coding
- Summary statistics
- Compact mode for dashboards

**Usage**:
```python
from teloscope_v2.utils.comparison_viewer_v2 import ComparisonViewerV2

viewer = ComparisonViewerV2()

# Render specific turn
viewer.render_turn(baseline_turn, telos_turn, turn_number=1)

# Render all turns
viewer.render_all_turns(baseline_branch, telos_branch)

# Render summary
viewer.render_summary(comparison, show_chart=True)
```

**UI Features**:
- Color-coded borders (green = good, orange = warning, red = critical)
- Intervention badges (🛡️ Intervention Applied)
- Fidelity metrics (F, distance, basin membership)
- Expandable turn sections
- ΔF calculation per turn

---

## Counterfactual Workflow

**Complete workflow for generating evidence**:

```python
# 1. Generate or import session
session = generate_enhanced_session(turns=12, session_type='high-drift')

# 2. Convert to conversation format
conversation = BaselineAdapter().convert_session_to_conversation(session)

# 3. Run baseline comparison
adapter = BaselineAdapter(llm_client, embedding_provider, attractor_config)
results = adapter.run_comparison(conversation, baseline_type='stateless')

# 4. Convert results to branch format
comparison_adapter = ComparisonAdapter()
baseline_branch = comparison_adapter.convert_baseline_result_to_branch(results['baseline'])
telos_branch = comparison_adapter.convert_baseline_result_to_branch(results['telos'])

# 5. Compare branches
comparison = comparison_adapter.compare_results(baseline_branch, telos_branch)

# 6. Display comparison
viewer = ComparisonViewerV2()
viewer.render_summary(comparison, show_chart=True)
viewer.render_all_turns(baseline_branch, telos_branch)

# 7. Export evidence
exporter = EvidenceExporter()
json_data = exporter.create_download_data(comparison, format='json')
md_data = exporter.create_download_data(comparison, format='markdown')
```

---

## Testing

### Test Foundation Components

Run the test harness to validate all foundation components:

```bash
cd ~/Desktop/TELOS
./venv/bin/streamlit run telos_observatory/main_observatory_v2.py
```

**What You'll See**:
- Turn indicators (compact, inline, progress bar modes)
- Timeline markers (standard, enhanced, annotated)
- Timeline legend
- Dimming algorithm visualization
- **Counterfactual analysis demo** (Phase 1.5B)
  - Mock baseline vs TELOS comparison
  - ΔF calculation
  - Side-by-side turn comparison
  - Evidence export (JSON + Markdown)

---

### Validate Imports

Ensure all v2 imports work:

```bash
cd ~/Desktop/TELOS/telos_observatory
python3 test_imports_v2.py
```

**Expected Output**:
```
Testing TELOSCOPE v2 imports...
✅ State management imports
✅ Mock data imports
✅ Marker generator imports
✅ Scroll controller imports
✅ Component imports

✅ All v2 imports successful!
```

---

## Coexistence Strategy

### Phase 1 vs v2

**Phase 1** (`teloscope/`):
- Working prototype with basic features
- Frozen for pilot testing
- Simple flat state management
- Complete for V1.00 pilot testing

**v2 Spec** (`teloscope_v2/`):
- Production-grade implementation
- Active development
- Centralized nested state
- Advanced features for Phase 2+

**Both coexist safely** - no conflicts, separate folders, separate imports.

### Migration Path

1. **Phase 1A** (Current): Phase 1 working, v2 foundation complete
2. **Phase 1B** (Week 3-4): v2 core components built, test side-by-side
3. **Phase 2** (Future): Migrate to v2 when it reaches parity

**See**: `docs/TELOSCOPE_INTEGRATION_RECONCILIATION.md` for complete strategy

---

## State Management Design

### Flat State (Phase 1)
```python
st.session_state.current_turn = 5
st.session_state.playing = True
```

### Nested State (v2)
```python
st.session_state.teloscope = {
    'current_turn': 5,
    'playing': True,
    'playback_speed': 1.0,
    'position': 'fixed-bottom',
    'active_tools': {...},
}
```

**Shared State**:
- `session_data` is intentionally shared between Phase 1 and v2
- It's the actual session data, not TELOSCOPE-specific state

---

## Documentation

- **Reconciliation Guide**: `docs/TELOSCOPE_INTEGRATION_RECONCILIATION.md`
- **Architecture**: `docs/TELOSCOPE_ARCHITECTURE.md`
- **Streamlit Patterns**: `docs/streamlit_patterns.md`
- **Phase 1 README**: `../README.md`

---

## Next Steps (Week 3-4)

Build core components to complete the TELOSCOPE v2 spec:

1. **Enhanced Navigation Controls** (`components/navigation_controls.py`)
   - Speed control, improved autoplay
   - First/Last buttons
   - Better error handling

2. **Enhanced Timeline Scrubber** (`components/timeline_scrubber.py`)
   - Marker generator integration
   - Hover tooltips
   - Click-to-jump markers

3. **Tool Buttons Component** (`components/tool_buttons.py`)
   - Toggle comparison viewer
   - Toggle calculation window
   - Toggle turn navigator
   - Toggle steward chat

4. **Position Manager** (`components/position_manager.py`)
   - Fixed/floating/hidden modes
   - Drag-and-drop positioning

5. **TELOSCOPE v2 Controller** (`teloscope_controller.py`)
   - Orchestrate all v2 components
   - Position management
   - Tool integration

6. **Test Entry Point v2** (`main_observatory_v2.py` enhancement)
   - Full feature testing
   - Side-by-side comparison with Phase 1

---

## API Reference

### State Management

```python
# Initialization
init_teloscope_state(session_data: Optional[Dict] = None)
reset_teloscope_state()

# Navigation
get_current_turn() -> int
set_current_turn(turn: int, rerun: bool = True)
is_playing() -> bool
set_playing(playing: bool, rerun: bool = True)
get_playback_speed() -> float
set_playback_speed(speed: float, rerun: bool = False)

# Tools
is_tool_active(tool_name: str) -> bool
toggle_tool(tool_name: str, rerun: bool = True)
set_tool_active(tool_name: str, active: bool, rerun: bool = True)

# Position
get_teloscope_position() -> str  # 'fixed-bottom' | 'floating' | 'hidden'
set_teloscope_position(position: str, rerun: bool = True)

# Debug
get_state_summary() -> Dict[str, Any]
```

### Mock Data

```python
generate_enhanced_session(
    turns: int = 12,
    session_type: str = 'stable',  # 'high-drift', 'intervention-heavy', 'perfect-alignment'
    include_annotations: bool = False,
    seed: Optional[int] = None,
) -> Dict[str, Any]

generate_test_suite() -> Dict[str, Dict[str, Any]]
generate_quick_session() -> Dict[str, Any]
generate_long_session() -> Dict[str, Any]
```

---

## Troubleshooting

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'teloscope_v2'`

**Solution**: Ensure you're running from TELOS directory:
```bash
cd ~/Desktop/TELOS
./venv/bin/streamlit run telos_observatory/main_observatory_v2.py
```

---

### State Not Updating

**Problem**: State changes don't reflect in UI

**Solution**: Ensure `rerun=True` in setter calls:
```python
set_current_turn(5, rerun=True)  # ✅ UI updates
set_current_turn(5)  # ❌ No rerun, UI won't update
```

---

### Controller Import Error

**Problem**: `from .teloscope_controller import render_teloscope_v2` fails

**Expected**: This is normal! Controller doesn't exist yet (Week 3-4).

The import is commented out in `__init__.py` until Week 3-4.

---

## Credits

Built with:
- **Streamlit**: Web framework
- **Python 3.9+**: Backend
- **TELOS Framework**: Governance system

**Status**: Foundation Complete ✅
**Next**: Week 3-4 Core Components ⏳

---

**Questions?** Check `docs/streamlit_patterns.md` or run `python3 steward.py status`
