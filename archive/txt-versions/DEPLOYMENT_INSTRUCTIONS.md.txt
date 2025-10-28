# TELOS Deployment Instructions

**Complete package for your dev team**

-----

## Package Contents

This deployment package contains everything needed to run TELOS Internal Test 0.

### 📁 File Inventory

**Setup Files** (8 new - created today):

- ✅ README.md - Repository overview
- ✅ QUICKSTART.md - 30-minute quick start guide
- ✅ DEVELOPER_CHECKLIST.md - Step-by-step deployment checklist
- ✅ RUNNING_TEST_0.md - Detailed validation guide
- ✅ requirements.txt - Python dependencies
- ✅ setup.py - Package installation
- ✅ Makefile - Command shortcuts
- ✅ .gitignore - Version control exclusions
- ✅ config.json - Governance configuration

**Python Code** (13 existing - from your uploads):

- ✅ `telos_purpose/core/primacy_math.py`
- ✅ `telos_purpose/core/unified_steward.py`
- ✅ `telos_purpose/core/intervention_controller.py`
- ✅ `telos_purpose/core/embedding_provider.py`
- ✅ `telos_purpose/core/conversation_manager.py`
- ✅ `telos_purpose/validation/baseline_runners.py`
- ✅ `telos_purpose/validation/run_internal_test0.py`
- ✅ `telos_purpose/validation/summarize_internal_test0.py`
- ✅ `telos_purpose/validation/telemetry_utils.py`
- ✅ `telos_purpose/llm_clients/mistral_client.py`
- ✅ `telos_purpose/test_conversations/test_convo_001.json`
- ✅ `telos_purpose/test_conversations/test_convo_002.json`
- ✅ `telos_purpose/test_conversations/test_convo_003.json`

**Package Initialization** (4 new - created today):

- ✅ `telos_purpose/__init__.py`
- ✅ `telos_purpose/core/__init__.py`
- ✅ `telos_purpose/validation/__init__.py`
- ✅ `telos_purpose/llm_clients/__init__.py`

**Documentation** (existing):

- ✅ `docs/TELOS_Whitepaper.md` (your original)
- ✅ `docs/TELOS_Architecture_and_Development_Roadmap.md` (FIXED formula)

-----

## Quick Deployment (For Your Dev)

### Step 1: Extract Package

```bash
# Unzip to local directory
cd ~/projects
unzip telos_deployment_package.zip
cd telos
```

### Step 2: Install

```bash
pip install -r requirements.txt
```

### Step 3: Configure API

```bash
export MISTRAL_API_KEY="your_mistral_key_here"
```

### Step 4: Run Test 0

```bash
python -m telos_purpose.validation.run_internal_test0
python -m telos_purpose.validation.summarize_internal_test0
```

### Step 5: Review Results

```bash
ls -lh validation_results/internal_test0/
```

**Expected**: 10 files (5 CSV + 5 JSON) with validation data

-----

## Repository Structure

```
telos/
│
├── README.md                         # Start here
├── QUICKSTART.md                     # 30-min guide
├── DEVELOPER_CHECKLIST.md            # Deployment steps
├── requirements.txt                  # Dependencies
├── setup.py                          # Installation
├── Makefile                          # Commands
├── .gitignore                        # Git exclusions
├── config.json                       # Configuration
│
├── telos_purpose/                    # Main package
│   ├── __init__.py
│   │
│   ├── core/                         # Math & orchestration
│   │   ├── __init__.py
│   │   ├── primacy_math.py          # Attractor dynamics
│   │   ├── unified_steward.py       # Orchestrator
│   │   ├── intervention_controller.py # Proportional control
│   │   ├── embedding_provider.py    # Text → vectors
│   │   └── conversation_manager.py  # History
│   │
│   ├── validation/                   # Testing framework
│   │   ├── __init__.py
│   │   ├── baseline_runners.py      # 5-way comparison
│   │   ├── run_internal_test0.py    # Test executor
│   │   ├── summarize_internal_test0.py # Results analysis
│   │   └── telemetry_utils.py       # Data export
│   │
│   ├── llm_clients/                  # API adapters
│   │   ├── __init__.py
│   │   └── mistral_client.py        # Mistral wrapper
│   │
│   └── test_conversations/           # Test data
│       ├── test_convo_001.json
│       ├── test_convo_002.json
│       └── test_convo_003.json
│
├── docs/                             # Documentation
│   ├── TELOS_Whitepaper.md
│   ├── TELOS_Architecture_and_Development_Roadmap.md
│   └── RUNNING_TEST_0.md
│
└── validation_results/               # Generated (gitignored)
    └── internal_test0/
        └── (10 files generated here)
```

-----

## Critical Files Explained

### For Your Dev

**Start with these in order**:

1. `README.md` - Overview
1. `QUICKSTART.md` - Fast path to validation
1. `DEVELOPER_CHECKLIST.md` - Step-by-step with troubleshooting

**If issues arise**:
4. `RUNNING_TEST_0.md` - Detailed validation guide

### Configuration

**`config.json`** - Governance parameters:

- `constraint_tolerance`: 0.0 = strict, 1.0 = permissive (default: 0.2)
- `epsilon_min`: Reminder threshold (default: 0.5)
- `epsilon_max`: Regeneration threshold (default: 0.8)

**Adjust if needed** based on Test 0 results.

### Test Data

**`test_conversations/*.json`** - Three test cases:

- `test_convo_001.json`: 2 turns, simple
- `test_convo_002.json`: 3 turns, moderate drift
- `test_convo_003.json`: 4 turns, clear drift patterns

**Default**: Test 0 uses `test_convo_002.json`

-----

## What Happens When Test 0 Runs

### Execution Flow

1. **Initialize** (5 sec)
- Load config
- Create attractor from purpose/scope
- Initialize LLM client
- Load test conversation
1. **Run 5 Conditions** (3-5 min)
- Each makes 5-7 Mistral API calls
- Each computes real semantic embeddings
- Each measures fidelity and drift
- Each exports telemetry
1. **Export Results** (instant)
- 5 CSV files (turn-by-turn data)
- 5 JSON files (session summaries)
1. **Analyze** (instant)
- Comparative table
- Hypothesis testing (H1, H2)
- Pass/fail determination

### Expected Console Output

```
============================================================
INTERNAL TEST 0: 5-CONDITION VALIDATION
============================================================

Loaded conversation: 3 turns

✓ All runners initialized

--------------------------------------------------------------

▶ Running: STATELESS
  ✓ Final fidelity: 0.6234

▶ Running: PROMPT_ONLY
  ✓ Final fidelity: 0.6789

▶ Running: CADENCE
  ✓ Final fidelity: 0.7123

▶ Running: OBSERVATION
  ✓ Final fidelity: 0.6234

▶ Running: TELOS
  ✓ Final fidelity: 0.8734

============================================================
✓ INTERNAL TEST 0 COMPLETE
============================================================

Results saved to: validation_results/internal_test0/

Generated files:
  - internal_test0_cadence_summary.json
  - internal_test0_cadence_turns.csv
  [... 8 more files ...]
```

-----

## Success Indicators

### ✅ Test 0 Passed If:

1. **No errors during execution**
1. **All 10 files generated**
1. **H1 passes** (ΔF ≥ 0.15)
1. **H2 passes** (TELOS achieves highest fidelity)
1. **Observation mode detects interventions** (proves math works)

### ⚠️ Needs Tuning If:

- H1 fails (ΔF < 0.15) → Adjust thresholds in config.json
- H2 fails (baseline beats TELOS) → Review intervention logic
- All fidelities >0.9 → Test too easy, try test_convo_003.json

-----

## For GitHub Repository

### Initialize Git

```bash
cd telos
git init
git add .
git commit -m "Initial commit: TELOS v1.0 - Internal Test 0 ready"
```

### Create Repository

1. Go to GitHub → New Repository
1. Name: `telos` (or `telos-purpose`)
1. Description: “Mathematical runtime governance framework for AI systems”
1. Public or Private (your choice)
1. Don’t initialize with README (you have one)

### Push Code

```bash
git remote add origin https://github.com/your-org/telos.git
git branch -M main
git push -u origin main
```

### Add Tags

```bash
git tag -a v1.0.0 -m "TELOS v1.0 - Test 0 validation ready"
git push origin v1.0.0
```

-----

## Timeline Estimate

### Initial Setup (Your Dev)

- Extract package: 2 min
- Install dependencies: 5 min
- Configure API key: 1 min
- **Total: ~10 minutes**

### First Run

- Execute Test 0: 5 min
- Review results: 5 min
- **Total: ~10 minutes**

### If Issues

- Troubleshooting: 10-20 min
- Rerun: 5 min
- **Total: ~15-25 minutes**

### Complete Deployment

**Best case**: 20 minutes  
**Expected**: 30-40 minutes  
**With issues**: 60 minutes

-----

## Support Checklist

If your dev hits issues:

### Common Problems

1. **Import errors** → Check `__init__.py` files present
1. **API key not working** → Verify export command ran
1. **No output files** → Check write permissions
1. **Slow execution** → First run downloads embedding model (~80MB)
1. **Hypothesis tests fail** → Expected - may need parameter tuning

### Debug Commands

```bash
# Verify installation
python -c "from telos_purpose.core import primacy_math; print('OK')"

# Check API key
echo $MISTRAL_API_KEY

# Test Mistral connection
python -c "from telos_purpose.llm_clients.mistral_client import TelosMistralClient; client = TelosMistralClient(); print('API OK')"

# Verify test files
ls telos_purpose/test_conversations/
```

-----

## Next Actions

### After Successful Test 0

**For Grant Application**:

1. Package results (CSV + JSON)
1. Create comparative visualization
1. Document hypothesis test outcomes
1. Include in LTFF application

**For Further Development**:

1. Run with longer conversations
1. Test with real session transcripts
1. Tune parameters based on results
1. Scale to multiple test cases

-----

## Contacts

**Technical Issues**: [Your contact]  
**Grant Questions**: [Your contact]  
**Repository Access**: [GitHub URL when ready]

-----

## File Checklist

Before sending to dev, verify package contains:

**Setup** (9 files):

- [ ] README.md
- [ ] QUICKSTART.md
- [ ] DEVELOPER_CHECKLIST.md
- [ ] RUNNING_TEST_0.md (in docs/)
- [ ] requirements.txt
- [ ] setup.py
- [ ] Makefile
- [ ] .gitignore
- [ ] config.json

**Code - Core** (5 files):

- [ ] primacy_math.py
- [ ] unified_steward.py
- [ ] intervention_controller.py
- [ ] embedding_provider.py
- [ ] conversation_manager.py

**Code - Validation** (4 files):

- [ ] baseline_runners.py
- [ ] run_internal_test0.py
- [ ] summarize_internal_test0.py
- [ ] telemetry_utils.py

**Code - Clients** (1 file):

- [ ] mistral_client.py

**Test Data** (3 files):

- [ ] test_convo_001.json
- [ ] test_convo_002.json
- [ ] test_convo_003.json

**Init Files** (4 files):

- [ ] telos_purpose/**init**.py
- [ ] telos_purpose/core/**init**.py
- [ ] telos_purpose/validation/**init**.py
- [ ] telos_purpose/llm_clients/**init**.py

**Documentation** (2 existing):

- [ ] TELOS_Whitepaper.md
- [ ] TELOS_Architecture_and_Development_Roadmap.md (FIXED)

**Total**: 28 files minimum for complete deployment

-----

**Package Status**: ✅ COMPLETE AND READY FOR DEPLOYMENT

**Created**: October 2025  
**Version**: 1.0.0  
**Validation Phase**: Internal Test 0