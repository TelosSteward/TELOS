# TELOS Deployment Checklist for Dev Team

**Target**: Get Test 0 validation running and generate grant-worthy data

-----

## Pre-Flight Checklist

### ☐ Environment Setup

- [ ] Python 3.8+ installed (`python --version`)
- [ ] pip updated (`pip install --upgrade pip`)
- [ ] Git configured (if using version control)
- [ ] Terminal/command line ready

### ☐ API Access

- [ ] Mistral API account created (https://console.mistral.ai/)
- [ ] API key obtained
- [ ] API key set as environment variable
  
  ```bash
  export MISTRAL_API_KEY="your_actual_key_here"
  # Verify: echo $MISTRAL_API_KEY
  ```

### ☐ Repository Files

- [ ] All files from package extracted
- [ ] Directory structure verified
- [ ] config.json present
- [ ] test_conversations/ directory exists

-----

## Installation Steps

### ☐ Step 1: Install Dependencies (5 min)

```bash
cd /path/to/telos
pip install -r requirements.txt
```

**Expected packages:**

- mistralai
- sentence-transformers
- torch
- numpy
- pandas

**Verify installation:**

```bash
python -c "from telos_purpose.core import primacy_math; print('✓ Core OK')"
python -c "import mistralai; print('✓ Mistral OK')"
python -c "import sentence_transformers; print('✓ Embeddings OK')"
```

### ☐ Step 2: Quick Sanity Check (2 min)

```bash
# Verify imports work
python -c "
from telos_purpose.core.primacy_math import PrimacyAttractorMath
from telos_purpose.core.unified_steward import UnifiedGovernanceSteward
from telos_purpose.llm_clients.mistral_client import TelosMistralClient
print('✓ All critical imports successful')
"
```

-----

## Running Test 0

### ☐ Step 3: Execute Validation (10-15 min)

```bash
# Run 5-condition test
python -m telos_purpose.validation.run_internal_test0
```

**What happens:**

1. Loads test conversation (5 turns with deliberate drift)
1. Runs 5 conditions sequentially:
- Stateless (no governance)
- Prompt-Only (constraints stated once)
- Cadence (fixed-interval reminders)
- Observation (math active, no interventions)
- TELOS (full adaptive governance)
1. Makes real Mistral API calls (~15-25 calls total)
1. Computes real embeddings using sentence transformers
1. Exports 10 files: 5 CSVs + 5 JSONs

**Expected console output:**

```
============================================================
INTERNAL TEST 0: 5-CONDITION VALIDATION
============================================================

▶ Running: STATELESS
  ✓ Final fidelity: 0.XXXX

▶ Running: PROMPT_ONLY
  ✓ Final fidelity: 0.XXXX

▶ Running: CADENCE
  ✓ Final fidelity: 0.XXXX

▶ Running: OBSERVATION
  ✓ Final fidelity: 0.XXXX

▶ Running: TELOS
  ✓ Final fidelity: 0.XXXX

============================================================
✓ INTERNAL TEST 0 COMPLETE
============================================================
```

### ☐ Step 4: Analyze Results (instant)

```bash
# Generate comparative report
python -m telos_purpose.validation.summarize_internal_test0
```

**Expected output:**

```
======================================================================
TELOS Internal Test 0 - Comparative Results
======================================================================

Condition      | Avg Fidelity | Basin Adherence | Intervention Rate
---------------|--------------|-----------------|-------------------
stateless      |       0.XXXX |          0.XXXX |            0.XXXX
prompt_only    |       0.XXXX |          0.XXXX |            0.XXXX
cadence        |       0.XXXX |          0.XXXX |            0.XXXX
observation    |       0.XXXX |          0.XXXX |            0.XXXX
telos          |       0.XXXX |          0.XXXX |            0.XXXX

======================================================================
Hypothesis Testing
======================================================================

H1 (Minimum Improvement): Delta F > 0.15
  TELOS fidelity:        0.XXXX
  Best baseline (XXX):   0.XXXX
  Delta F:               +0.XXXX
  Result:                PASS or FAIL

H2 (TELOS is Best): TELOS achieves highest fidelity
  Best condition:        telos (0.XXXX)
  Result:                PASS or FAIL
```

-----

## Validation Checklist

### ☐ Step 5: Verify Outputs

**Check file generation:**

```bash
ls -lh validation_results/internal_test0/
```

**Expected files (10 total):**

- [ ] `internal_test0_stateless_turns.csv`
- [ ] `internal_test0_stateless_summary.json`
- [ ] `internal_test0_prompt_only_turns.csv`
- [ ] `internal_test0_prompt_only_summary.json`
- [ ] `internal_test0_cadence_turns.csv`
- [ ] `internal_test0_cadence_summary.json`
- [ ] `internal_test0_observation_turns.csv`
- [ ] `internal_test0_observation_summary.json`
- [ ] `internal_test0_telos_turns.csv`
- [ ] `internal_test0_telos_summary.json`

**Spot check CSV:**

```bash
head -5 validation_results/internal_test0/internal_test0_telos_turns.csv
```

**Spot check JSON:**

```bash
cat validation_results/internal_test0/internal_test0_telos_summary.json | head -20
```

### ☐ Step 6: Validate Results

**Check hypothesis outcomes:**

- [ ] H1 result noted (PASS/FAIL)
- [ ] H2 result noted (PASS/FAIL)
- [ ] Delta F value recorded
- [ ] TELOS fidelity vs baseline fidelity compared

**Quality checks:**

- [ ] All 5 conditions ran successfully
- [ ] Fidelity scores are reasonable (0.0-1.0 range)
- [ ] TELOS shows improvement over baselines (expected)
- [ ] Observation mode detected interventions (proves math works)

-----

## Troubleshooting

### Common Issues

**Problem**: `ModuleNotFoundError: No module named 'mistralai'`

```bash
pip install mistralai
```

**Problem**: `MISTRAL_API_KEY not set`

```bash
export MISTRAL_API_KEY="your_key"
# Add to ~/.bashrc or ~/.zshrc for persistence
```

**Problem**: `FileNotFoundError: test_convo_002.json`

```bash
# Verify test files exist
ls telos_purpose/test_conversations/

# If missing, file structure issue - check extraction
```

**Problem**: Slow execution (>30 minutes)

- Check internet connection (API calls)
- Verify embedding model downloaded (first run downloads ~80MB)
- Check API rate limits

**Problem**: Unexpected fidelity scores

- Verify API key is working (test with simple Mistral call)
- Check embedding model loaded correctly
- Ensure config.json values are sensible

-----

## Success Criteria

### Test 0 is successful if:

✅ All 5 conditions execute without errors  
✅ 10 output files generated  
✅ Hypothesis test results display  
✅ TELOS shows measurable improvement (H1 or H2 passes)  
✅ Observation mode detects interventions (proves detection logic works)

### Grant-ready if:

✅ ΔF ≥ 0.15 (40% improvement in synthetic test)  
✅ TELOS achieves highest fidelity  
✅ Clean telemetry exports (CSV/JSON well-formatted)  
✅ Reproducible (can rerun and get consistent results)

-----

## Deliverables Package

### ☐ Step 7: Package Results for Grant

**Create results archive:**

```bash
# Create summary directory
mkdir test0_results_YYYY-MM-DD

# Copy key files
cp validation_results/internal_test0/*.json test0_results_YYYY-MM-DD/
cp validation_results/internal_test0/*_telos*.csv test0_results_YYYY-MM-DD/

# Capture console output
python -m telos_purpose.validation.summarize_internal_test0 > test0_results_YYYY-MM-DD/summary.txt

# Archive
tar -czf test0_results_YYYY-MM-DD.tar.gz test0_results_YYYY-MM-DD/
```

**What to include in grant application:**

- Summary statistics (from summarize script)
- H1/H2 test results
- Sample CSV showing turn-by-turn telemetry
- Sample JSON showing session metrics
- Fidelity improvement graph (if time permits)

-----

## Timeline

- **Setup**: 5-10 minutes
- **Test 0 execution**: 10-15 minutes
- **Results validation**: 5 minutes
- **Troubleshooting buffer**: 10 minutes

**Total expected time**: 30-40 minutes

-----

## Next Steps After Test 0

Once Test 0 validates successfully:

1. **Tune parameters** (if H1/H2 failed)
- Adjust constraint_tolerance in config.json
- Modify intervention thresholds
- Rerun test
1. **Longer conversations** (if successful)
- Create 10-20 turn test conversations
- Test with more complex drift patterns
1. **Real conversation test**
- Extract actual session transcript
- Run governed vs ungoverned comparison
- Measure user burden reduction
1. **Documentation**
- Package results for LTFF grant
- Create figures/visualizations
- Write methods section

-----

## Support Contacts

**Technical issues**: [Your contact info]  
**Grant questions**: [Your contact info]  
**Repository issues**: Open GitHub issue

-----

## Final Checklist

Before declaring Test 0 complete:

- [ ] All installation steps completed
- [ ] Test 0 ran without errors
- [ ] All 10 output files generated
- [ ] Hypothesis results reviewed
- [ ] Results look reasonable (no obvious bugs)
- [ ] Deliverables packaged
- [ ] Next steps identified

**When all checked: Test 0 is COMPLETE ✅**

-----

**Document Version**: 1.0  
**Created**: October 2025  
**Status**: Ready for Deployment