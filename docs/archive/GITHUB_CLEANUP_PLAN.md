# GitHub Repository Cleanup Plan
**TELOS Observatory - Repository Organization & Cleanup**

Generated: 2025-10-31
Repository: https://github.com/TelosSteward/TELOS-Observatory

---

## Executive Summary

This document outlines a comprehensive plan to clean up and organize the TELOS Observatory repository. The repository has accumulated legacy code, obsolete versions, test data, and duplicate implementations that need to be archived or removed to improve maintainability and clarity.

**Current State**: Multiple versions (V1, V2, V3), scattered test data, numerous validation runs, and background process artifacts.

**Target State**: Clean repository with V3 as primary, Observatory Advanced (dish icon) as secondary, legacy code archived, and clear documentation.

---

## Phase 1: Repository Assessment (COMPLETED)

### Current Directory Structure
```
/telos_observatory/              # Original observatory (legacy)
/telos_observatory_v2/           # Second iteration (legacy)
/telos_observatory_v3/           # ✅ ACTIVE - V3.0.0 (telescope icon)
/telos_purpose/                  # Core TELOS purpose alignment system
/telos_outreach/                 # Grant applications and outreach materials
/grant_applications/             # Duplicate grant materials?
/test_sessions/                  # Various test data files
/.streamlit/                     # Streamlit configuration
/docs/                          # Documentation (PRD, tasks)
/tests/                         # Test suite (untracked)
```

### Files to Review
- Modified but uncommitted:
  - `.gitignore`
  - `STEWARD.md`
  - `docs/prd/PRD.md`, `docs/prd/TASKS.md`
  - Legacy observatory files (v1, v2)
  - Purpose dashboard files

- Untracked files/directories:
  - `JB_PROTOCOLS.md`
  - `WIRING_PLAN.md`
  - `grant_applications/`
  - Multiple phase2 validation directories
  - `tests/`

---

## Phase 2: Cleanup Strategy

### 2.1 Archive Legacy Versions

**Action**: Move old observatory versions to `/archive/` directory

**Directories to Archive**:
```
archive/
├── telos_observatory/           # V1 - Original implementation
├── telos_observatory_v2/        # V2 - Second iteration
└── README_ARCHIVE.md            # Documentation of archived versions
```

**Rationale**: Keep for reference but remove from main directory to reduce clutter.

### 2.2 Consolidate Test Data

**Action**: Organize all test/validation data into single directory structure

**Target Structure**:
```
test_data/
├── phase2_validation/
│   ├── claude_test_1/
│   ├── edge_cases/
│   └── study_results/
├── sharegpt_data/
│   └── quality_analysis_report.md
└── test_sessions/
    ├── claude_conversation_parsed.json
    └── phase2_format/
```

**Directories to Consolidate**:
- `telos_observatory/phase2_study_results/`
- `telos_observatory/phase2_validation_*` (multiple)
- `telos_observatory/phase2b_continuous_*`
- `telos_observatory/sharegpt_data/`
- `telos_purpose/test_data/`
- `test_sessions/`

### 2.3 Organize Grant/Outreach Materials

**Action**: Merge duplicate grant materials

**Target Structure**:
```
telos_outreach/
├── grant_applications/
│   ├── LTFF_APPLICATION_DRAFT.md
│   └── [other applications]
├── domain_specs/
│   └── DOMAIN_SPECIFIC_PA_SPECS.md
├── testing/
│   ├── USER_TESTING_PLAN.md
│   └── TESTING_PARTICIPANT_GUIDE.md
└── status/
    ├── GRANT_PREP_STATUS.md
    └── project_status.md
```

**Note**: Check for duplicates between `grant_applications/` and `telos_outreach/`

### 2.4 Clean Up Root Directory

**Files to Keep**:
- `README.md` (update to reflect V3)
- `STEWARD.md` (commit changes)
- `.gitignore` (commit changes)
- `requirements.txt`
- Configuration files

**Files to Review/Move**:
- `JB_PROTOCOLS.md` → `docs/protocols/`
- `WIRING_PLAN.md` → `docs/technical/`
- `GITHUB_CLEANUP_PLAN.md` (this file) → `docs/maintenance/`

### 2.5 Organize Documentation

**Target Structure**:
```
docs/
├── prd/
│   ├── PRD.md
│   └── TASKS.md
├── technical/
│   ├── WIRING_PLAN.md
│   └── architecture.md
├── protocols/
│   └── JB_PROTOCOLS.md
├── maintenance/
│   └── GITHUB_CLEANUP_PLAN.md
└── README.md (index of all docs)
```

---

## Phase 3: Prepare for Observatory Advanced

### 3.1 Clone V3 to Create Observatory Advanced

**Command**:
```bash
cp -r telos_observatory_v3 telos_observatory_advanced
```

**Changes Needed**:
1. Replace telescope icon (🔭) with dish icon (🛸)
2. Update branding: "TELOS OBSERVATORY" instead of "TELOS"
3. Keep all other functionality identical
4. Update configuration to run on different port (8502?)

**Files to Modify**:
- `telos_observatory_advanced/components/sidebar_actions.py` (branding)
- `telos_observatory_advanced/main.py` (title, icon)
- `telos_observatory_advanced/README.md` (create new)

### 3.2 Update Repository README

**Sections to Add**:
```markdown
# TELOS Observatory

## Active Versions

### TELOS V3 (Telescope Icon)
Clean UI with telescope branding for standard use.
- Location: `/telos_observatory_v3/`
- Port: 8501
- Tag: v3.0.0

### TELOS Observatory Advanced (Dish Icon)
Advanced observatory interface with dish branding.
- Location: `/telos_observatory_advanced/`
- Port: 8502
- Tag: tbd

## Archived Versions
Legacy versions preserved in `/archive/` for reference.

## Running the Applications
[Instructions here]
```

---

## Phase 4: .gitignore Updates

### Patterns to Add:
```gitignore
# Python cache
__pycache__/
*.pyc
*.pyo
*.pyd
.Python

# Virtual environments
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Test data (selectively)
test_data/*/results/
*.log

# Temporary files
*.tmp
*.temp

# API keys (if any)
*.key
.env
```

---

## Phase 5: Execution Plan

### Step-by-Step Execution

**Pre-Cleanup Checklist**:
- [ ] Create backup branch: `git checkout -b pre-cleanup-backup`
- [ ] Push backup: `git push origin pre-cleanup-backup`
- [ ] Document current state
- [ ] Create this cleanup plan document

**Cleanup Execution** (To be run when ready):
1. Create `archive/` directory
2. Move legacy observatories to archive
3. Create `test_data/` structure and consolidate
4. Organize grant materials
5. Clean root directory
6. Update documentation structure
7. Update `.gitignore`
8. Commit all cleanup changes
9. Create v3.1.0 tag (post-cleanup)

**Post-Cleanup**:
1. Clone V3 to Observatory Advanced
2. Update branding (telescope → dish)
3. Update README for both versions
4. Test both applications
5. Commit Observatory Advanced
6. Create tags and push to GitHub

---

## Phase 6: Ongoing Maintenance

### Monthly Tasks:
- Review and archive old test data
- Update documentation
- Clean up temporary files
- Review .gitignore effectiveness

### Before Each Release:
- Review directory structure
- Update version tags
- Clean up any accumulated test data
- Update README with new features

---

## Estimated Impact

### Repository Size:
- **Current**: ~100+ directories/files in root and subdirs
- **Target**: ~20-30 main directories, organized structure
- **Reduction**: ~40-50% reduction in visible clutter

### Benefits:
1. **Clarity**: Clear distinction between active and legacy code
2. **Maintainability**: Easier to find and modify code
3. **Onboarding**: New developers can understand structure quickly
4. **Performance**: Fewer files for Git to track actively
5. **Professional**: Clean, organized repository for grants/demos

---

## Risk Assessment

### Low Risk:
- Moving to archive (can easily restore)
- Organizing test data
- Documentation reorganization

### Medium Risk:
- Consolidating grant materials (check for duplicates first)
- .gitignore updates (might accidentally ignore needed files)

### Mitigation:
- Create backup branch before starting
- Test after each major change
- Keep archive accessible
- Document all moves in commit messages

---

## Next Steps

1. **Review this plan** with stakeholders
2. **Create backup branch**
3. **Execute Phase 5** cleanup steps
4. **Create Observatory Advanced** (dish icon version)
5. **Update documentation**
6. **Push to GitHub**

---

## Notes

- This cleanup should be done during a non-critical period
- Estimate: 2-4 hours for full cleanup
- Test both V3 and Observatory Advanced after cleanup
- Update grant applications with new GitHub structure

---

Generated with Claude Code
https://claude.com/claude-code
