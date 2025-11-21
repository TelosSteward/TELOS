# GitHub Repository Cleanup Plan
## TELOS-Observatory Main Branch Cleanup

## Current State
- **Total files on GitHub**: 521 files
- **Target files**: ~56 files (matching Privacy_PreCommit)
- **Reduction**: ~90% cleanup needed

## Problem Analysis

### Categories of Messy Files on GitHub:

1. **Session/Planning Documents** (Should NOT be on GitHub)
   - NEXT_SESSION_*.md
   - SESSION_SUMMARY_*.md
   - BETA_IMPLEMENTATION_PLAN.md
   - DEMO_MODE_DESIGN_PLAN.md
   - PLAYWRIGHT_MCP_SETUP_AND_TESTING.md
   - planning_output/* (entire directory)

2. **Screenshots/Images** (Development artifacts)
   - .playwright-mcp/*.png (25+ screenshots)
   - All test result screenshots

3. **Test Results** (Should be in .gitignore)
   - tests/validation_results/* (100+ files)
   - tests/test_results/*
   - observatory/beta_consents/consent_log.json

4. **Beta Testing Files** (Internal only)
   - BETA_*.md files
   - docs/BETA_*.md
   - observatory/beta_testing/*
   - tests/beta_validation/*

5. **Reference Materials** (Not needed publicly)
   - reference_materials/*
   - VALIDATION_AUDIT_REPORT.md
   - VALIDATION_STATUS_REPORT.md

6. **Development Tools** (Not for production)
   - dev_dashboard/* (entire directory)
   - scripts/plan_*.py
   - scripts/structured_planning.py

7. **Duplicate/Old Config**
   - config/governance_config.json (should be .example only)
   - .streamlit/secrets.toml.example

## Cleanup Strategy

### Phase 1: Create Clean Branch from Privacy_PreCommit
1. Create new branch `cleanup-main-2025-11`
2. Copy Privacy_PreCommit structure to this branch
3. Ensure all essential files are included
4. Test the clean version

### Phase 2: Remove Deprecated Files from Main
Execute git commands to remove unwanted files

### Phase 3: Push Clean Version
Force push clean structure to main (with backup)

---

## Files to KEEP (56 Essential Files)

### Root Level
- ✅ README.md (update with clean version)
- ✅ LICENSE
- ✅ .gitignore (comprehensive version)
- ✅ requirements.txt
- ✅ setup.py

### Documentation
- ✅ docs/whitepapers/TELOS_Whitepaper.md (v2.3)
- ✅ docs/whitepapers/TELOS_Academic_Paper.md
- ✅ docs/whitepapers/TELOS_Technical_Paper.md
- ✅ docs/whitepapers/Statistical_Validity.md
- ✅ docs/guides/Implementation_Guide.md
- ✅ docs/guides/Architecture_Diagrams.md
- ✅ docs/guides/Quick_Start_Guide.md
- ✅ docs/regulatory/EU_Article72_Submission.md
- ✅ docs/QUICK_START.md

### Core TELOS Implementation
- ✅ telos/__init__.py
- ✅ telos/core/__init__.py
- ✅ telos/core/dual_attractor.py
- ✅ telos/core/unified_steward.py
- ✅ telos/core/primacy_math.py
- ✅ telos/core/intervention_controller.py
- ✅ telos/core/proportional_controller.py
- ✅ telos/core/governance_config.py
- ✅ telos/core/embedding_provider.py
- ✅ telos/core/intercepting_llm_wrapper.py
- ✅ telos/utils/__init__.py
- ✅ telos/utils/conversation_manager.py
- ✅ telos/utils/embedding_provider.py
- ✅ telos/utils/mistral_client.py

### Observatory (Production UI)
- ✅ telos_observatory/__init__.py
- ✅ telos_observatory/main.py
- ✅ telos_observatory/requirements.txt
- ✅ telos_observatory/components/* (all components)
- ✅ telos_observatory/core/* (state_manager, async_processor)
- ✅ telos_observatory/utils/* (intro_messages, mock_data)

### Examples & Config
- ✅ examples/configs/governance_config.json
- ✅ examples/configs/config_example.json
- ✅ examples/runtime_governance_start.py
- ✅ examples/runtime_governance_checkpoint.py

### Claude Code Configuration
- ✅ .claude/commands/*.md (slash commands)

---

## Files to REMOVE (465+ Files)

### Remove Entire Directories:
```bash
# Planning and session documents
planning_output/
reference_materials/

# Development dashboard (not production ready)
dev_dashboard/

# Test results (should never be committed)
tests/validation_results/
tests/test_results/
.playwright-mcp/

# Beta testing internals
observatory/beta_testing/
observatory/beta_consents/
tests/beta_validation/

# Demo mode (if not production ready)
demo_mode/

# Old observatory structure
observatory/  # Will be replaced with telos_observatory/
```

### Remove Individual Files:
```bash
# Session and planning docs
NEXT_SESSION_*.md
SESSION_SUMMARY_*.md
QUICK_START_NEXT_SESSION.md
BETA_*.md
BUILD_TAG_*.md
BUTTON_HOVER_EXPANSION_NOTE.md
DEMO_MODE_*.md
DEPLOYMENT_GUIDE.md  # If outdated
EXECUTIVE_SUMMARY.md
NEXT_VERSION_PLAN.md
PLAYWRIGHT_MCP_*.md
SEQUENTIAL_ANALYSIS_*.md
SETUP_NOTES.md
VALIDATION_*.md
.claude_project.md

# Config that should be examples only
config/governance_config.json  # Keep only .example
.streamlit/secrets.toml.example  # Remove if not needed

# Test files that shouldn't be in main
test_*.py (root level)
tests/test_*.py (direct test scripts)
tests/playwright_*.py
```

---

## Git Commands for Cleanup

### Step 1: Create Backup Branch
```bash
git checkout main
git pull origin main
git checkout -b backup-main-before-cleanup-2025-11-13
git push origin backup-main-before-cleanup-2025-11-13
```

### Step 2: Create Clean Branch
```bash
git checkout main
git checkout -b cleanup-main-2025-11
```

### Step 3: Remove Unwanted Directories
```bash
git rm -r planning_output/
git rm -r reference_materials/
git rm -r dev_dashboard/
git rm -r tests/validation_results/
git rm -r tests/test_results/
git rm -r .playwright-mcp/
git rm -r observatory/beta_testing/
git rm -r observatory/beta_consents/
git rm -r tests/beta_validation/
git rm -r demo_mode/ 2>/dev/null || true
```

### Step 4: Remove Individual Files
```bash
# Session documents
git rm NEXT_SESSION_*.md
git rm SESSION_SUMMARY_*.md
git rm QUICK_START_NEXT_SESSION.md

# Beta/testing docs
git rm BETA_*.md
git rm BUILD_TAG_*.md
git rm docs/BETA_*.md

# Planning docs
git rm BUTTON_HOVER_EXPANSION_NOTE.md
git rm DEMO_MODE_*.md
git rm EXECUTIVE_SUMMARY.md
git rm NEXT_VERSION_PLAN.md
git rm PLAYWRIGHT_MCP_*.md
git rm SEQUENTIAL_ANALYSIS_*.md
git rm SETUP_NOTES.md
git rm VALIDATION_*.md
git rm .claude_project.md

# Test files in root
git rm test_*.py 2>/dev/null || true
git rm tests/playwright_*.py
git rm tests/test_direct_*.py
git rm tests/test_mistral_*.py
git rm tests/test_rate_limit.py
git rm tests/test_single_*.py
git rm tests/test_steward_*.py

# Streamlit secrets example (if not needed)
git rm .streamlit/secrets.toml.example 2>/dev/null || true

# Config that shouldn't be committed
git rm config/governance_config.json
```

### Step 5: Reorganize from Privacy_PreCommit
```bash
# Remove old observatory structure
git rm -r observatory/

# Copy clean structure from Privacy_PreCommit
# (We'll do this programmatically)
```

### Step 6: Commit Cleanup
```bash
git commit -m "Major cleanup: Remove 465+ internal/temp files, restructure to production-ready state

REMOVED:
- planning_output/ (internal planning documents)
- reference_materials/ (research materials)
- dev_dashboard/ (development tool, not production)
- tests/validation_results/ (100+ test result files)
- tests/test_results/ (test artifacts)
- .playwright-mcp/ (25+ screenshots)
- observatory/beta_testing/ (internal beta infrastructure)
- Session summary and handoff documents
- Beta testing plans and reports
- Validation audit documents

RESTRUCTURED:
- observatory/ → telos_observatory/ (clean production UI)
- Added comprehensive .gitignore
- Updated README with professional structure

RESULT:
- Reduced from 521 files to 56 essential files
- Production-ready, professional repository
- All development artifacts removed
- Clean structure for institutional deployment

Co-Authored-By: Claude <noreply@anthropic.com>"
```

### Step 7: Review Before Push
```bash
# Count files in clean branch
git ls-files | wc -l
# Should be ~56 files

# Review what's included
git ls-files

# Check diff from main
git diff main --stat
```

### Step 8: Push Clean Version (WITH APPROVAL)
```bash
# Push cleanup branch first for review
git push origin cleanup-main-2025-11

# After approval, merge to main
git checkout main
git merge cleanup-main-2025-11
git push origin main
```

---

## Safety Measures

1. ✅ **Backup branch created first** (`backup-main-before-cleanup-2025-11-13`)
2. ✅ **All local files safe** in telos_privacy/
3. ✅ **Work in cleanup branch** before touching main
4. ✅ **Review before push** - verify file count and structure
5. ✅ **Can revert if needed** - backup branch preserved

---

## Expected Outcome

### Before Cleanup:
```
TELOS-Observatory (main branch)
├── 521 files total
├── Messy structure
├── Internal documents exposed
├── Test results committed
└── Development artifacts everywhere
```

### After Cleanup:
```
TELOS-Observatory (main branch)
├── 56 essential files
├── Clean professional structure
├── Production-ready code only
├── Proper documentation
└── Ready for institutional deployment
```

---

## Verification Checklist

After cleanup, verify:

- [ ] File count is ~56 files
- [ ] No test results in repository
- [ ] No session summaries or planning docs
- [ ] No screenshots or images
- [ ] Observatory is telos_observatory/
- [ ] All core TELOS files present
- [ ] Documentation complete and organized
- [ ] Examples and configs included
- [ ] .gitignore is comprehensive
- [ ] README is professional
- [ ] Can install and run: `pip install -r requirements.txt`
- [ ] Observatory launches: `streamlit run telos_observatory/main.py`

---

## NEXT STEPS

1. **Review this plan** - Ensure all critical files preserved
2. **Approve execution** - Give explicit go-ahead
3. **Execute cleanup** - Run git commands in order
4. **Verify results** - Check file count and structure
5. **Push to GitHub** - After final approval

**Status**: ⏸️ AWAITING YOUR APPROVAL TO EXECUTE
