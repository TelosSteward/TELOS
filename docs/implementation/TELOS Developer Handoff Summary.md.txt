# TELOS Developer Handoff Summary – Canonical Repository v2.0

**Origin Industries PBC / TELOS Labs LLC — October 2025**
**File:** setup/TELOS_Integrations_Handoff.txt  
**Purpose:** Defines canonical procedures for post-validation developer handoff and operational verification.  
**Status:** Canonical – Current  
**Maintainer:** @TelosSteward
-----

## Repository Structure Overview

```
telos/
├── telos_purpose/              # Main package directory
│   ├── __init__.py
│   │
│   ├── core/                   # Mathematical & runtime orchestration core
│   │   ├── __init__.py
│   │   ├── primacy_math.py
│   │   ├── unified_steward.py  # includes TeleologicalOperator alias
│   │   ├── intervention_controller.py
│   │   ├── embedding_provider.py
│   │   └── conversation_manager.py
│   │
│   ├── validation/             # Validation and benchmarking framework
│   │   ├── __init__.py
│   │   ├── run_internal_test0.py
│   │   ├── summarize_internal_test0.py
│   │   ├── comparative_test.py
│   │   ├── retro_analyzer.py
│   │   ├── system_health_monitor.py
│   │   └── telemetry_utils.py
│   │
│   ├── sessions/               # Execution-level runtime tools
│   │   ├── __init__.py
│   │   ├── run_with_dashboard.py
│   │   ├── observation_validation_run.py
│   │   └── profile_extractor_cli.py
│   │
│   ├── llm_clients/            # External LLM adapters
│   │   ├── __init__.py
│   │   └── mistral_client.py
│   │
│   ├── test_conversations/     # Validation datasets
│   │   ├── test_convo_001.json
│   │   ├── test_convo_002.json
│   │   └── test_convo_003.json
│   │
│   └── dev_dashboard/          # Optional visualization layer
│       ├── __init__.py
│       └── streamlit_live_comparison.py
│
├── docs/                       # Canonical documentation corpus
│   ├── README.md
│   ├── TELOS_Whitepaper.md
│   ├── TELOS_Architecture_and_Development_Roadmap.md
│   ├── TELOS_Developer_and_Research_Operations_Guide.md
│   ├── TELOS_Repository_Structure_v2.0.md
│   └── TELOS_Documentation_Index_v2.0.md
│
├── public/                     # External & regulatory materials
│   ├── TELOS_Executive_Summary.md
│   ├── TELOS_Grant_Application.txt
│   └── Why_TELOS_Had_to_Be_an_OS.md
│
├── validation_results/         # Output artifacts (gitignored)
│   └── internal_test0/
│
├── config.json
├── Makefile
├── requirements.txt
├── setup.py
├── .gitignore
└── README.md
```

-----

## Verification and Validation Workflow

### 1. Verify imports and baseline functionality

```bash
# Test core imports
python -c "from telos_purpose.core import primacy_math, unified_steward; print('✅ Core imports OK')"

# Test TeleologicalOperator alias
python -c "from telos_purpose.core.unified_steward import TeleologicalOperator; print('✅ Alias OK')"
```

### 2. Run Internal Test 0

```bash
# Execute validation across all 5 conditions
python -m telos_purpose.validation.run_internal_test0

# Generate comparative summary
python -m telos_purpose.validation.summarize_internal_test0
```

**Expected Output:**

- `validation_results/internal_test0/` directory created
- 5 CSV files (one per condition: stateless, prompt_only, cadence, observation, telos)
- 5 JSON summary files
- Console output showing hypothesis test results (H1, H2)

### 3. Verify output artifacts

```bash
# Check that all files were generated
ls validation_results/internal_test0/*.json
ls validation_results/internal_test0/*.csv

# Should show 10 files total (5 JSON + 5 CSV)
```

-----

## Commit and Push Process

### 1. Stage all changes

```bash
git add .
```

### 2. Commit with descriptive message

```bash
git commit -m "TELOS v2.0: Canonical structure + TeleologicalOperator alias + Internal Test 0 ready"
```

### 3. Tag the release

```bash
git tag -a v2.0 -m "Runtime Steward (Teleological Operator pattern) + validation infrastructure"
git push origin main --tags
```

-----

## Post-Commit Verification Checklist

- [ ] README.md renders correctly on GitHub
- [ ] Documentation index links resolve (`docs/TELOS_Documentation_Index_v2.0.md`)
- [ ] Core imports work without errors
- [ ] TeleologicalOperator alias imports successfully
- [ ] Internal Test 0 runs and generates all output files
- [ ] Summary script produces hypothesis test results
- [ ] `.gitignore` excludes `validation_results/*`
- [ ] All documentation references use correct paths

-----

## Critical Path Notes

**Before running Internal Test 0:**

1. Ensure `MISTRAL_API_KEY` is set in environment
1. Verify `config.json` uses `constraint_tolerance` parameter (not old names)
1. Confirm test conversation files exist in `test_conversations/`

**After Internal Test 0 completes:**

1. Update README.md Section 5 with actual validation results
1. Replace placeholder metrics table with real data
1. Commit updated README with validation evidence

**If validation fails:**

1. Check `validation_results/internal_test0/*.json` for error messages
1. Verify embedding provider is accessible
1. Review parameter values in `config.json`
1. Check LLM client authentication

-----

## Key File Locations

|Component         |Path                                                  |
|------------------|------------------------------------------------------|
|Core math         |`telos_purpose/core/primacy_math.py`                  |
|Runtime steward   |`telos_purpose/core/unified_steward.py`               |
|Validation runner |`telos_purpose/validation/run_internal_test0.py`      |
|Summary generator |`telos_purpose/validation/summarize_internal_test0.py`|
|Config file       |`config.json` (root)                                  |
|Test conversations|`telos_purpose/test_conversations/*.json`             |
|Output directory  |`validation_results/internal_test0/`                  |

-----

## Developer Environment Setup

```bash
# 1. Clone repository
git clone https://github.com/your-org/telos.git
cd telos

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set environment variables
export MISTRAL_API_KEY="your_key_here"

# 5. Verify installation
python -c "from telos_purpose.core import unified_steward; print('✅ Installation complete')"
```

-----

## Next Steps After v2.0 Commit

1. **Run validation:** Execute Internal Test 0 and generate results
1. **Update documentation:** Replace placeholder metrics with actual data
1. **Grant preparation:** Use validation results in funding applications
1. **Community release:** Prepare for public repository announcement
1. **Federation planning:** Begin institutional partnership outreach

-----

**Status:** Canonical handoff documentation  
**Version:** 2.0  
**Maintained by:** @TelosSteward  
**Last Updated:** October 2025
