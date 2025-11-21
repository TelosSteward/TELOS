# Governance Settings Update: Basin Constant & Tolerance

**Date**: November 20, 2025
**Status**: ✅ COMPLETE
**Purpose**: Implement tested "Goldilocks" governance settings across TELOS validation pipeline

---

## 🎯 Summary

Updated TELOS to use empirically-tested optimal governance settings:
- **Basin Constant**: 2.0 → **1.0** (Goldilocks value)
- **Constraint Tolerance**: 0.2 → **0.05** (strict drift detection)

These settings are now:
1. **Coded as defaults** in `primacy_math.py`
2. **Stored in Supabase** with every validation session
3. **Documented** for audit trail and reproducibility

---

## 📊 Why These Settings Matter

### Basin Constant = 1.0
**Formula**: `basin_radius = basin_constant / rigidity`

**Testing Results:**
- **2.0** = Too loose (caught nothing - missed actual drift)
- **0.5** = Too strict (flagged everything - false positives)
- **1.0** = **Goldilocks** (just right for drift detection)

**At tolerance 0.05:**
- Rigidity ρ = 0.95
- Basin radius r = 1.0 / 0.95 ≈ **1.053**

### Constraint Tolerance = 0.05
**Controls**: How strict the governance system is

**Testing Results (from BETA_DEPLOYMENT_GUIDE.md):**
- Quantum physics drift: **0.696 fidelity** (correctly flagged as drift)
- On-topic questions: **0.95+ fidelity** (correctly allowed)
- Demo conversation: **60% drift detection** (reasonable governance)

**Why 0.05:**
- Provides strict boundary enforcement
- Catches meaningful drift without over-triggering
- Balances user freedom with purpose alignment

---

## 🔧 Files Modified

### 1. Core Mathematics
**File**: `telos_purpose/core/primacy_math.py`

**Changes:**
```python
# Line 52: Updated default
constraint_tolerance: float = 0.05  # Was 0.2

# Line 92: Updated basin constant
self.basin_radius = 1.0 / rigidity_floored  # Was 2.0
```

**Documentation Added:**
- Comments explaining Goldilocks value
- Reference to tested optimal settings
- Formula documentation

### 2. Validation Storage
**File**: `telos_purpose/storage/validation_storage.py`

**Changes:**
```python
# Added to create_validation_session() parameters
"basin_constant": session_data.get("basin_constant", 1.0),
"constraint_tolerance": session_data.get("constraint_tolerance", 0.05),
```

**Why**: Stores governance settings with every validation session for audit trail

### 3. Validation Suite
**File**: `run_ollama_validation_suite_v2.py`

**Changes:**
```python
# In _start_new_session() - added to session creation
"basin_constant": 1.0,  # Goldilocks value from testing
"constraint_tolerance": pa_config.get("constraint_tolerance", 0.05)
```

**Why**: Passes settings to Supabase for every validation run

### 4. Supabase Schema
**Files**:
- `supabase_validation_schema_CLEAN.sql` (updated)
- `supabase_add_basin_tolerance_fields.sql` (new ALTER script)

**Changes:**
```sql
-- Added to validation_telemetric_sessions table
basin_constant REAL DEFAULT 1.0,
constraint_tolerance REAL DEFAULT 0.05,
```

**Why**: Permanently records governance settings with telemetry data

---

## 🎓 Impact on Validation Data

### Before This Update
- Sessions stored fidelity scores without context
- No way to know which governance calibration was used
- Couldn't reproduce or verify results
- Comparing across sessions was unreliable

### After This Update
Every validation session now includes:
```json
{
  "session_id": "abc-123",
  "basin_constant": 1.0,
  "constraint_tolerance": 0.05,
  "avg_fidelity": 0.723,
  "telemetric_signature": "d18a53b6...",
  "created_at": "2025-11-20T19:45:00Z"
}
```

**Benefits:**
1. ✅ **Audit Trail**: Know exactly what settings produced each result
2. ✅ **Reproducibility**: Can verify results using same calibration
3. ✅ **Comparability**: Ensure all tests use same baseline
4. ✅ **IP Protection**: Settings are cryptographically signed with session
5. ✅ **Scientific Validity**: Methodology is transparent and documented

---

## 🔬 Validation Impact

### Mock Tests (Currently Running)
The baseline comparison study will use:
- Basin constant: **1.0**
- Tolerance: **0.05** (via pa_config default)
- All 15 turns will have consistent governance settings

### Future Tests
**Test 0 (Claude Conversation):**
- Will use basin_constant = 1.0, tolerance = 0.05
- Results comparable to mock tests
- Governance evaluation will be consistent

**Counterfactual Analysis:**
- Both branches use same settings
- Differences = governance impact (not calibration variance)

**LangChain Partnership Demo:**
- Standard settings ensure reproducible governance
- Third parties can verify with documented calibration

---

## 📋 Next Steps

### 1. Apply Schema to Supabase ⏳
```bash
# Option A: Run ALTER script (for existing table)
# Copy contents of supabase_add_basin_tolerance_fields.sql
# Paste into Supabase SQL Editor

# Option B: Recreate table (if starting fresh)
# Use updated supabase_validation_schema_CLEAN.sql
```

### 2. Verify New Sessions
Once schema is applied, new sessions will automatically store:
- `basin_constant: 1.0`
- `constraint_tolerance: 0.05`

Check with:
```bash
python3 check_validation_status.py
```

### 3. Update Documentation
Mark in session handoff docs that governance settings are now:
- Standardized at tested optimal values
- Stored with all telemetry for reproducibility
- Part of cryptographic audit trail

---

## 🎉 Testing Validation

### Expected Behavior
With basin_constant=1.0 and tolerance=0.05:

**Drift Detection:**
- Responses with fidelity < 0.70 should be flagged
- Distance > basin_radius (≈1.053) triggers drift

**Baseline Comparison:**
- STATELESS: Lower avg fidelity (no governance)
- PROMPT_ONLY: Mid-range fidelity (static guidance)
- TELOS: Highest avg fidelity (dynamic governance)

**IP Protection:**
- Every session records exact governance calibration
- Telemetric signatures include setting state
- Third-party verification possible

---

## 📚 References

**Source Documents:**
- `BETA_DEPLOYMENT_GUIDE.md` (lines 440-456) - Basin calibration testing
- `analyze_demo_tolerance_05.py` - Empirical testing with 0.05 tolerance
- `FINAL_STATUS_ALL_WORKING.md` - Validation pipeline verification

**Testing History:**
1. Original: basin=2.0, tolerance=0.2 → Too permissive
2. Tested: basin=0.5, tolerance=0.5 → Too strict
3. **Optimal: basin=1.0, tolerance=0.05** → Goldilocks ✅

**Key Findings:**
- 60% drift detection rate in demo (reasonable)
- Quantum physics question: 0.696 fidelity (correctly flagged)
- On-topic questions: 0.95+ fidelity (correctly allowed)

---

## ✅ Completion Checklist

- [x] Update basin_constant default in primacy_math.py (2.0 → 1.0)
- [x] Update constraint_tolerance default in primacy_math.py (0.2 → 0.05)
- [x] Add basin_constant to ValidationStorage
- [x] Add constraint_tolerance to ValidationStorage
- [x] Update validation suite to pass settings
- [x] Create SQL ALTER script for Supabase
- [x] Update schema file for new installations
- [x] Document changes and rationale
- [ ] Apply schema updates to Supabase (user action required)
- [ ] Verify new sessions store settings correctly

---

**Status**: Ready for Supabase schema application
**Confidence**: High - based on empirical testing
**Next**: Apply `supabase_add_basin_tolerance_fields.sql` to database

---

*Generated: November 20, 2025*
*Validation Pipeline: TELOS v1.0*
*Settings: basin_constant=1.0, constraint_tolerance=0.05*
