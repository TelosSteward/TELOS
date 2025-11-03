# TELOS File Organization Plan
**Date**: November 3, 2024
**Purpose**: Prepare files for dual GitHub repository structure (telos-purpose & telos-privacy)
**Status**: Pre-deployment cleanup

---

## Executive Summary

You've added **TELOS_Whitepaper_v2.2.md** (49KB) and **TELOS_Whitepaper_v2.2.docx** (35KB) which integrate all dual PA validation results. This organization plan will:

1. Move V2.2 whitepaper to proper location
2. Archive old whitepaper versions
3. Organize all documentation for dual-repo split
4. Create clean structure for production deployment

---

## 1. Whitepaper V2.2 - Immediate Actions

### 1.1 Current Location
```
/Users/brunnerjf/Desktop/telos/
├── TELOS_Whitepaper_v2.2.md (NEW - 49KB)
├── TELOS_Whitepaper_v2.2.docx (NEW - 35KB)
└── docs/
    └── TELOS_Whitepaper.md (OLD - 96KB)
```

### 1.2 Recommended Actions

**Move V2.2 to canonical location**:
```bash
# Move markdown to docs (canonical location)
mv TELOS_Whitepaper_v2.2.md docs/TELOS_Whitepaper_v2.2.md

# Keep docx in docs for distribution
mv TELOS_Whitepaper_v2.2.docx docs/TELOS_Whitepaper_v2.2.docx

# Archive old whitepaper
mv docs/TELOS_Whitepaper.md docs/archive/TELOS_Whitepaper_v2.1.md
```

**Update symlink/primary reference**:
```bash
# Create canonical reference (no version number)
cd docs
cp TELOS_Whitepaper_v2.2.md TELOS_Whitepaper.md
```

### 1.3 What's New in V2.2

**Major Updates**:
- ✅ Dual PA architecture documented (Section 2.3)
- ✅ +85.32% validation results integrated (Abstract, Appendix D)
- ✅ Counterfactual vs Runtime validation distinction clarified
- ✅ Status changed from "unvalidated" to "partially validated"
- ✅ Statistical analysis included (p < 0.001, Cohen's d = 0.87)

**Appendices Added**:
- Appendix D: Dual PA Validation Results
- Appendix E: Sample Telemetry
- Appendix F: Key Terms

---

## 2. Dual GitHub Repository Structure

### 2.1 Target Architecture

**telos-purpose** (Purpose Drop - Public First):
```
telos-purpose/
├── README.md
├── docs/
│   ├── TELOS_Whitepaper_v2.2.md (CANONICAL)
│   ├── TELOS_Whitepaper_v2.2.docx (DISTRIBUTION)
│   ├── DUAL_PA_VALIDATION_SUMMARY.md
│   ├── DEPLOYMENT_ROADMAP.md
│   ├── DUAL_DROP_STRATEGY.md
│   └── archive/
│       └── TELOS_Whitepaper_v2.1.md
├── telos_purpose/
│   ├── core/
│   │   ├── dual_pa.py
│   │   ├── unified_orchestrator_steward.py
│   │   ├── intervention_engine.py
│   │   └── governance_config.py
│   ├── llm_clients/
│   ├── embedding/
│   └── utils/
├── examples/
│   ├── streamlit/  # Observatory v3
│   ├── telegram/   # Future
│   └── discord/    # Future
├── validation/
│   ├── datasets/
│   │   └── sharegpt_cleaned/
│   └── briefs/
│       └── dual_pa_research_briefs/ (46 files)
├── tests/
└── requirements.txt
```

**telos-privacy** (Privacy Drop - Future):
```
telos-privacy/
├── README.md
├── docs/
│   └── PRIVACY_ARCHITECTURE.md
├── telos_privacy/
│   ├── tap/           # TELOS Adaptive Periphery
│   ├── containers/     # Federated containers
│   └── irb/           # IRB compliance
├── examples/
└── requirements.txt
```

### 2.2 Current Repo Organization for Purpose Drop

**Keep in Root** (Project Management):
```
/
├── README.md (UPDATE for dual PA)
├── requirements.txt
├── .gitignore
├── setup.py (when ready)
└── LICENSE
```

**Move to docs/** (Documentation):
```
docs/
├── TELOS_Whitepaper_v2.2.md ← MOVE HERE (canonical)
├── TELOS_Whitepaper_v2.2.docx ← MOVE HERE
├── DUAL_PA_VALIDATION_SUMMARY.md ← ALREADY EXISTS
├── DEPLOYMENT_ROADMAP.md ← ALREADY EXISTS
├── DUAL_DROP_STRATEGY.md ← ALREADY EXISTS
├── WHITEPAPER_UPDATE_NOTES.md ← MOVE HERE
├── STREAMLIT_CLOUD_DEPLOYMENT.md ← MOVE HERE
├── REPO_MIGRATION_PLAN.md ← MOVE HERE
├── PERSISTENT_PRIMACY_ATTRACTOR.md ← MOVE HERE
└── archive/
    ├── TELOS_Whitepaper_v2.1.md
    ├── BUILD_NOTES_v1.1.md
    ├── DEPLOYMENT.md (old)
    ├── GITHUB_CLEANUP_PLAN.md
    ├── GITHUB_READY.md
    ├── ORGANIZATION_COMPLETE.md
    └── TASKS.md
```

**Archive** (Research/Internal):
```
docs/archive/research/
├── DUAL_ATTRACTOR_ARCHITECTURE.md
├── DEMO_MODE_TWO_LAYER_IMPLEMENTATION.md
├── GOVERNANCE_MODES.md
├── OBSERVATORY_ADVANCED_FEATURES.md
├── PRODUCT_DISCOVERY_TRANSCRIPT.md
├── REPO_MANIFEST.md
└── WIRING_PLAN.md
```

**Keep Separate** (Internal/Personal):
```
/
├── JB_PROTOCOLS.md (KEEP IN ROOT - personal workflow)
├── STEWARD.md (INTERNAL - privacy drop)
└── STEWARD_PRODUCT_VISION.md (INTERNAL - privacy drop)
```

---

## 3. Detailed File Mapping

### 3.1 Whitepapers (Critical)

| Current Location | Target Location | Action | Notes |
|-----------------|-----------------|--------|-------|
| `TELOS_Whitepaper_v2.2.md` | `docs/TELOS_Whitepaper_v2.2.md` | **MOVE** | Canonical v2.2 |
| `TELOS_Whitepaper_v2.2.docx` | `docs/TELOS_Whitepaper_v2.2.docx` | **MOVE** | Distribution |
| `docs/TELOS_Whitepaper.md` | `docs/archive/TELOS_Whitepaper_v2.1.md` | **ARCHIVE** | Old version |

**Create Reference**:
```bash
cd docs
cp TELOS_Whitepaper_v2.2.md TELOS_Whitepaper.md
```

### 3.2 Validation & Deployment Docs (Keep in docs/)

| File | Status | Action |
|------|--------|--------|
| `DUAL_PA_VALIDATION_SUMMARY.md` | ✅ Already in root | Keep (or move to docs/) |
| `DEPLOYMENT_ROADMAP.md` | ✅ Already in root | Keep (or move to docs/) |
| `DUAL_DROP_STRATEGY.md` | ✅ Already in root | Keep (or move to docs/) |
| `WHITEPAPER_UPDATE_NOTES.md` | Currently in root | Move to `docs/` |
| `STREAMLIT_CLOUD_DEPLOYMENT.md` | Currently in root | Move to `docs/` |
| `REPO_MIGRATION_PLAN.md` | Currently in root | Move to `docs/` |

### 3.3 Architecture Docs (Move to docs/)

| File | Target | Notes |
|------|--------|-------|
| `PERSISTENT_PRIMACY_ATTRACTOR.md` | `docs/` | Core architecture |
| `DUAL_ATTRACTOR_ARCHITECTURE.md` | `docs/archive/research/` | Research notes |

### 3.4 Internal/Archive (Move to docs/archive/)

| File | Target Location | Reason |
|------|----------------|---------|
| `BUILD_NOTES_v1.1.md` | `docs/archive/` | Historical |
| `DEPLOYMENT.md` | `docs/archive/` | Superseded by DEPLOYMENT_ROADMAP |
| `GITHUB_CLEANUP_PLAN.md` | `docs/archive/` | Completed |
| `GITHUB_READY.md` | `docs/archive/` | Completed |
| `ORGANIZATION_COMPLETE.md` | `docs/archive/` | Completed |
| `TASKS.md` | `docs/archive/` | Superseded |
| `DEMO_MODE_TWO_LAYER_IMPLEMENTATION.md` | `docs/archive/research/` | Experimental |
| `GOVERNANCE_MODES.md` | `docs/archive/research/` | Research |
| `OBSERVATORY_ADVANCED_FEATURES.md` | `docs/archive/research/` | Feature ideas |
| `PRODUCT_DISCOVERY_TRANSCRIPT.md` | `docs/archive/research/` | Session notes |
| `REPO_MANIFEST.md` | `docs/archive/` | Historical |
| `WIRING_PLAN.md` | `docs/archive/research/` | Implementation notes |

### 3.5 Special Cases

| File | Action | Notes |
|------|--------|-------|
| `JB_PROTOCOLS.md` | **KEEP IN ROOT** | Personal workflow, not for GitHub |
| `STEWARD.md` | **KEEP SEPARATE** | Privacy Drop content |
| `STEWARD_PRODUCT_VISION.md` | **KEEP SEPARATE** | Privacy Drop content |
| `README_TELOSCOPE.md` | **KEEP** | Valid module README |

---

## 4. Validation Data Files

### 4.1 Dual PA Research Briefs

**Current**: `dual_pa_research_briefs/` (46 files)
**Target**: `validation/briefs/dual_pa_research_briefs/`

```bash
mkdir -p validation/briefs
mv dual_pa_research_briefs validation/briefs/
```

### 4.2 Results JSON Files

**Current Root Files**:
```
dual_pa_proper_comparison_results.json (772KB)
claude_conversation_dual_pa_fresh_results.json (290KB)
claude_conversation_starters_only.json (45KB)
dual_pa_counterfactual_results.json (7.7KB)
dual_pa_counterfactual_results_BROKEN.json (8.0KB - DELETE)
```

**Target**: `validation/results/dual_pa/`

```bash
mkdir -p validation/results/dual_pa
mv dual_pa_proper_comparison_results.json validation/results/dual_pa/
mv claude_conversation_dual_pa_fresh_results.json validation/results/dual_pa/
mv claude_conversation_starters_only.json validation/results/dual_pa/
mv dual_pa_counterfactual_results.json validation/results/dual_pa/
rm dual_pa_counterfactual_results_BROKEN.json # Delete broken file
```

### 4.3 Validation Scripts

**Current Root Scripts**:
```
run_proper_dual_pa_comparison.py
summarize_dual_pa_results.py
generate_dual_pa_research_briefs.py
regenerate_claude_starters_dual_pa.py
run_claude_conversation_dual_pa.py
convert_claude_to_starters_only.py
convert_claude_conversation_to_sharegpt.py
check_multiple_sessions.py
check_single_session_distances.py
analyze_basin_calibration.py
run_dual_pa_counterfactual.py
```

**Target**: `validation/scripts/`

```bash
mkdir -p validation/scripts
mv run_proper_dual_pa_comparison.py validation/scripts/
mv summarize_dual_pa_results.py validation/scripts/
mv generate_dual_pa_research_briefs.py validation/scripts/
mv regenerate_claude_starters_dual_pa.py validation/scripts/
mv run_claude_conversation_dual_pa.py validation/scripts/
mv convert_claude_to_starters_only.py validation/scripts/
mv convert_claude_conversation_to_sharegpt.py validation/scripts/
mv check_multiple_sessions.py validation/scripts/
mv check_single_session_distances.py validation/scripts/
mv analyze_basin_calibration.py validation/scripts/
mv run_dual_pa_counterfactual.py validation/scripts/
```

---

## 5. Execution Plan

### Phase 1: Whitepaper V2.2 (IMMEDIATE)

```bash
# 1. Create necessary directories
mkdir -p docs/archive
mkdir -p docs/archive/research

# 2. Move V2.2 whitepapers to docs
mv TELOS_Whitepaper_v2.2.md docs/TELOS_Whitepaper_v2.2.md
mv TELOS_Whitepaper_v2.2.docx docs/TELOS_Whitepaper_v2.2.docx

# 3. Archive old whitepaper
mv docs/TELOS_Whitepaper.md docs/archive/TELOS_Whitepaper_v2.1.md

# 4. Create canonical reference
cd docs
cp TELOS_Whitepaper_v2.2.md TELOS_Whitepaper.md
cd ..
```

### Phase 2: Documentation Organization

```bash
# Move current docs to proper locations
mv WHITEPAPER_UPDATE_NOTES.md docs/
mv STREAMLIT_CLOUD_DEPLOYMENT.md docs/
mv REPO_MIGRATION_PLAN.md docs/
mv PERSISTENT_PRIMACY_ATTRACTOR.md docs/

# Archive completed/superseded docs
mv BUILD_NOTES_v1.1.md docs/archive/
mv DEPLOYMENT.md docs/archive/
mv GITHUB_CLEANUP_PLAN.md docs/archive/
mv GITHUB_READY.md docs/archive/
mv ORGANIZATION_COMPLETE.md docs/archive/
mv TASKS.md docs/archive/
mv REPO_MANIFEST.md docs/archive/

# Archive research docs
mv DUAL_ATTRACTOR_ARCHITECTURE.md docs/archive/research/
mv DEMO_MODE_TWO_LAYER_IMPLEMENTATION.md docs/archive/research/
mv GOVERNANCE_MODES.md docs/archive/research/
mv OBSERVATORY_ADVANCED_FEATURES.md docs/archive/research/
mv PRODUCT_DISCOVERY_TRANSCRIPT.md docs/archive/research/
mv WIRING_PLAN.md docs/archive/research/
```

### Phase 3: Validation Data Organization

```bash
# Create validation structure
mkdir -p validation/briefs
mkdir -p validation/results/dual_pa
mkdir -p validation/scripts

# Move research briefs
mv dual_pa_research_briefs validation/briefs/

# Move results
mv dual_pa_proper_comparison_results.json validation/results/dual_pa/
mv claude_conversation_dual_pa_fresh_results.json validation/results/dual_pa/
mv claude_conversation_starters_only.json validation/results/dual_pa/
mv dual_pa_counterfactual_results.json validation/results/dual_pa/
rm dual_pa_counterfactual_results_BROKEN.json

# Move validation scripts
mv run_proper_dual_pa_comparison.py validation/scripts/
mv summarize_dual_pa_results.py validation/scripts/
mv generate_dual_pa_research_briefs.py validation/scripts/
mv regenerate_claude_starters_dual_pa.py validation/scripts/
mv run_claude_conversation_dual_pa.py validation/scripts/
mv convert_claude_to_starters_only.py validation/scripts/
mv convert_claude_conversation_to_sharegpt.py validation/scripts/
mv check_multiple_sessions.py validation/scripts/
mv check_single_session_distances.py validation/scripts/
mv analyze_basin_calibration.py validation/scripts/
mv run_dual_pa_counterfactual.py validation/scripts/
```

### Phase 4: Update References

```bash
# Update README to reference new whitepaper location
# Update .gitignore if needed
# Create validation/README.md explaining structure
# Create docs/README.md as documentation index
```

---

## 6. Final Structure (Post-Organization)

```
telos/ (current repo - will become telos-purpose)
├── README.md
├── requirements.txt
├── .gitignore
├── JB_PROTOCOLS.md (personal, not for GitHub)
│
├── docs/
│   ├── README.md (documentation index)
│   ├── TELOS_Whitepaper.md (canonical - copy of v2.2)
│   ├── TELOS_Whitepaper_v2.2.md
│   ├── TELOS_Whitepaper_v2.2.docx
│   ├── DUAL_PA_VALIDATION_SUMMARY.md
│   ├── DEPLOYMENT_ROADMAP.md
│   ├── DUAL_DROP_STRATEGY.md
│   ├── WHITEPAPER_UPDATE_NOTES.md
│   ├── STREAMLIT_CLOUD_DEPLOYMENT.md
│   ├── REPO_MIGRATION_PLAN.md
│   ├── PERSISTENT_PRIMACY_ATTRACTOR.md
│   ├── archive/
│   │   ├── TELOS_Whitepaper_v2.1.md
│   │   ├── BUILD_NOTES_v1.1.md
│   │   ├── DEPLOYMENT.md
│   │   ├── GITHUB_CLEANUP_PLAN.md
│   │   ├── GITHUB_READY.md
│   │   ├── ORGANIZATION_COMPLETE.md
│   │   ├── TASKS.md
│   │   ├── REPO_MANIFEST.md
│   │   └── research/
│   │       ├── DUAL_ATTRACTOR_ARCHITECTURE.md
│   │       ├── DEMO_MODE_TWO_LAYER_IMPLEMENTATION.md
│   │       ├── GOVERNANCE_MODES.md
│   │       ├── OBSERVATORY_ADVANCED_FEATURES.md
│   │       ├── PRODUCT_DISCOVERY_TRANSCRIPT.md
│   │       └── WIRING_PLAN.md
│   └── [existing docs/ structure...]
│
├── telos_purpose/ (core code)
├── telos_observatory_v3/ (streamlit app)
│
├── validation/
│   ├── README.md (explains validation structure)
│   ├── briefs/
│   │   └── dual_pa_research_briefs/ (46 files)
│   ├── results/
│   │   └── dual_pa/
│   │       ├── dual_pa_proper_comparison_results.json
│   │       ├── claude_conversation_dual_pa_fresh_results.json
│   │       ├── claude_conversation_starters_only.json
│   │       └── dual_pa_counterfactual_results.json
│   └── scripts/
│       ├── run_proper_dual_pa_comparison.py
│       ├── summarize_dual_pa_results.py
│       ├── generate_dual_pa_research_briefs.py
│       ├── regenerate_claude_starters_dual_pa.py
│       ├── run_claude_conversation_dual_pa.py
│       ├── convert_claude_to_starters_only.py
│       ├── convert_claude_conversation_to_sharegpt.py
│       ├── check_multiple_sessions.py
│       ├── check_single_session_distances.py
│       ├── analyze_basin_calibration.py
│       └── run_dual_pa_counterfactual.py
│
└── [other existing directories...]
```

---

## 7. Git Operations

### 7.1 Before Moving Files

```bash
# Ensure all current work is committed
git status
git add -A
git commit -m "Pre-organization checkpoint: V2.2 whitepaper added"
```

### 7.2 Moving Files (Preserves Git History)

```bash
# Use git mv instead of mv to preserve history
git mv TELOS_Whitepaper_v2.2.md docs/TELOS_Whitepaper_v2.2.md
git mv TELOS_Whitepaper_v2.2.docx docs/TELOS_Whitepaper_v2.2.docx
# ... etc for all moves
```

### 7.3 After Organization

```bash
git add -A
git commit -m "docs: Organize files for dual-repo deployment

- Move V2.2 whitepaper to docs/
- Archive old whitepaper versions
- Organize validation data and scripts
- Create clean structure for telos-purpose repo

Preparation for telos-purpose/telos-privacy split"
```

---

## 8. Updates Needed After Organization

### 8.1 Update README.md

```markdown
# TELOS Framework

**Status**: Dual PA Architecture Validated (+85.32% improvement)
**Version**: v1.0.0-dual-pa-canonical

## Documentation

- 📄 [Whitepaper (V2.2)](docs/TELOS_Whitepaper_v2.2.md) - Complete framework documentation
- 📊 [Validation Summary](DUAL_PA_VALIDATION_SUMMARY.md) - +85.32% improvement results
- 🚀 [Deployment Roadmap](DEPLOYMENT_ROADMAP.md) - Multi-platform deployment plan
- 🔬 [Dual Drop Strategy](DUAL_DROP_STRATEGY.md) - Purpose + Privacy development timeline

## Validation Results

See [validation/](validation/) for complete validation data:
- 46 research briefs documenting dual PA performance
- Statistical analysis (p < 0.001, Cohen's d = 0.87)
- Full conversation regeneration results

[Rest of README...]
```

### 8.2 Create validation/README.md

```markdown
# TELOS Validation Data

This directory contains all validation studies for the dual PA architecture.

## Structure

- `briefs/` - Research briefs analyzing individual sessions
- `results/` - Raw validation results (JSON)
- `scripts/` - Validation and analysis scripts

## Dual PA Validation (November 2024)

**Results**: +85.32% improvement over single PA baseline
**Status**: v1.0.0-dual-pa-canonical

See [../docs/DUAL_PA_VALIDATION_SUMMARY.md](../docs/DUAL_PA_VALIDATION_SUMMARY.md) for complete analysis.
```

### 8.3 Create docs/README.md

```markdown
# TELOS Documentation

## Core Documents

- [TELOS Whitepaper V2.2](TELOS_Whitepaper_v2.2.md) - Official framework documentation
- [Dual PA Validation Summary](../DUAL_PA_VALIDATION_SUMMARY.md) - Validation results
- [Deployment Roadmap](../DEPLOYMENT_ROADMAP.md) - Multi-platform deployment plan

## Architecture

- [Persistent Primacy Attractor](PERSISTENT_PRIMACY_ATTRACTOR.md) - Core governance mechanism
- [Dual Drop Strategy](../DUAL_DROP_STRATEGY.md) - Purpose + Privacy development

## Development

- [Streamlit Cloud Deployment](STREAMLIT_CLOUD_DEPLOYMENT.md) - Observatory deployment
- [Repository Migration Plan](REPO_MIGRATION_PLAN.md) - Dual-repo strategy

## Archive

Historical documents and research notes are in [archive/](archive/).
```

---

## 9. Checklist

### Immediate (Whitepaper V2.2)
- [ ] Create `docs/archive/` and `docs/archive/research/` directories
- [ ] Move `TELOS_Whitepaper_v2.2.md` to `docs/`
- [ ] Move `TELOS_Whitepaper_v2.2.docx` to `docs/`
- [ ] Archive old `docs/TELOS_Whitepaper.md` as `v2.1`
- [ ] Create canonical `docs/TELOS_Whitepaper.md` (copy of v2.2)
- [ ] Commit: "docs: Add TELOS Whitepaper V2.2 with dual PA validation"

### Phase 2 (Documentation)
- [ ] Move current docs to `docs/` as mapped above
- [ ] Archive completed/superseded docs
- [ ] Archive research docs to `docs/archive/research/`
- [ ] Commit: "docs: Reorganize documentation structure"

### Phase 3 (Validation Data)
- [ ] Create `validation/` directory structure
- [ ] Move research briefs to `validation/briefs/`
- [ ] Move results JSON to `validation/results/dual_pa/`
- [ ] Move validation scripts to `validation/scripts/`
- [ ] Delete broken results file
- [ ] Commit: "refactor: Organize validation data and scripts"

### Phase 4 (References & READMEs)
- [ ] Update root `README.md` with new structure
- [ ] Create `validation/README.md`
- [ ] Create `docs/README.md`
- [ ] Update any hardcoded paths in scripts
- [ ] Commit: "docs: Update READMEs for new structure"

### Final
- [ ] Review all file locations
- [ ] Test that scripts still run with new paths
- [ ] Create git tag: `v1.0.0-organized`
- [ ] Verify .gitignore covers all necessary exclusions

---

## 10. Benefits of This Organization

✅ **Clean Repository**: Professional structure ready for public deployment
✅ **Clear Documentation**: All docs in `docs/`, validation in `validation/`
✅ **Version Control**: V2.2 clearly designated, old versions archived
✅ **Dual-Repo Ready**: Easy to split into telos-purpose/telos-privacy
✅ **Audit Trail**: Git history preserved through `git mv`
✅ **Discoverable**: Clear README files guide users to content

---

**Next Step**: Review this plan, then I'll execute the file moves using `git mv` to preserve history.
