# TELOS Quick Start Guide
## After Reorganization

**Date**: 2025-10-25
**Status**: Repository reorganized and ready for development

---

## ✅ What's Been Done

The TELOS repository has been completely reorganized from a flat file structure into a proper Python package. See `REORGANIZATION_SUMMARY.md` for complete details.

---

## 🚀 Getting Started (5 Minutes)

### Step 1: Install the Package

```bash
# Navigate to the telos directory
cd ~/Desktop/telos

# Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install TELOS in development mode
pip install -e .
```

### Step 2: Set Up Environment

```bash
# Set your Mistral API key
export MISTRAL_API_KEY="your_api_key_here"

# Or create a .env file
echo "MISTRAL_API_KEY=your_api_key_here" > .env
```

### Step 3: Test the Installation

```bash
# Quick import test
python -c "from telos_purpose import UnifiedGovernanceSteward; print('✅ TELOS installed successfully')"

# Check package structure
python -c "import telos_purpose.core, telos_purpose.validation; print('✅ All modules accessible')"
```

---

## 🧪 Running Validation Tests

### Internal Test 0

```bash
# Run the main validation test
python -m telos_purpose.validation.run_internal_test0

# Summarize results
python -m telos_purpose.validation.summarize_internal_test0
```

### Expected Output Location
Results will be saved to: `validation_results/internal_test0/`

---

## 📦 Package Structure

```python
# Core imports
from telos_purpose import (
    UnifiedGovernanceSteward,     # Main runtime steward
    TeleologicalOperator,         # Alias
    PrimacyAttractorMath,         # Attractor dynamics
    MathematicalState,            # State representation
    TelicFidelityCalculator,      # Fidelity computation
)

# LLM client
from telos_purpose.llm_clients import MistralClient

# Constants
from telos_purpose.core.constants import (
    FIDELITY_MONITOR,    # 0.85
    FIDELITY_CORRECT,    # 0.70
    FIDELITY_INTERVENE,  # 0.50
    FIDELITY_ESCALATE,   # 0.50
)
```

---

## 🔧 Common Tasks

### Create a New Session

```python
from telos_purpose import UnifiedGovernanceSteward
from telos_purpose.llm_clients import MistralClient
from telos_purpose.core.embedding_provider import EmbeddingProvider

# Initialize components
llm_client = MistralClient(api_key="your_key")
embedder = EmbeddingProvider()

# Configure governance
attractor = {
    "purpose": ["Provide helpful information"],
    "scope": ["General knowledge questions"],
    "boundaries": ["Do not provide financial advice"],
    "constraint_tolerance": 0.2
}

# Create steward
steward = UnifiedGovernanceSteward(
    attractor=attractor,
    llm_client=llm_client,
    embedding_provider=embedder
)

# Run session
steward.start_session()
result = steward.process_turn(user_input, model_response)
summary = steward.end_session()
```

### Run with Dashboard

```python
from telos_purpose.sessions import run_with_dashboard

# Launch interactive dashboard
run_with_dashboard.main()
```

---

## 📚 Documentation

- **Complete Overview**: `docs/README.md`
- **Whitepaper**: `docs/TELOS_Whitepaper.md`
- **Architecture**: `docs/TELOS_Architecture_and_Development_Roadmap.md`
- **Developer Guide**: `docs/TELOS_Developer_and_Research_Operations_Guide.md`
- **Executive Summary**: `public/TELOS_Executive_Summary.md`
- **Setup Guide**: `setup/TELOS_Dev_Environment_Setup.md`

---

## ⚠️ Known Issues & Fixes Needed

### 1. Import Path Updates Required

Some Python files may still reference old import paths. If you encounter import errors:

**Old style**:
```python
from primacy_math import PrimacyAttractorMath
```

**New style**:
```python
from telos_purpose.core.primacy_math import PrimacyAttractorMath
```

### 2. File Syntax Issues

Some files copied from `.txt` may have formatting issues. Check:
- `telos_purpose/core/primacy_math.py` - Lines 14-36 have syntax errors
- Look for improperly closed docstrings or code blocks

### 3. Missing Dependencies

If imports fail, ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

---

## 🧹 Cleanup Recommendations

The root directory still contains old `.txt` files. Archive them:

```bash
# Create archive directory
mkdir archive

# Move all remaining .txt files
mv *.txt archive/

# Move PDF files
mv *.pdf archive/

# Keep only the organized structure
```

---

## 🎯 Next Development Steps

### Immediate (Today)
1. ✅ Repository reorganized
2. ⏳ Fix import paths in Python files
3. ⏳ Test all modules can import without errors
4. ⏳ Run Internal Test 0 successfully

### Short-term (This Week)
1. Update any hardcoded file paths
2. Create test data in `telos_purpose/test_conversations/`
3. Verify all validation scripts work
4. Document any breaking changes

### Medium-term (This Month)
1. Complete Internal Test 0 validation
2. Generate validation reports
3. Update README with actual test results
4. Prepare for Pilot 0 deployment

---

## 📞 Support

- **Documentation**: See `docs/` directory
- **Developer Handoff**: `setup/TELOS_Developer_Handoff_Summary.md`
- **Reorganization Details**: `REORGANIZATION_SUMMARY.md`
- **Integration Guide**: `setup/TELOS_Integrations_Handoff.md`

---

## ✨ Quick Reference

### Validate Installation
```bash
python -c "from telos_purpose import UnifiedGovernanceSteward; print('✅ OK')"
```

### Run Tests
```bash
python -m telos_purpose.validation.run_internal_test0
```

### View Structure
```bash
tree -L 3 -I '*.txt|*.pdf' telos_purpose docs public setup
```

### Install with All Features
```bash
pip install -e ".[dev,viz,docs]"
```

---

**Ready to develop! 🚀**

See `REORGANIZATION_SUMMARY.md` for complete details on what changed.
