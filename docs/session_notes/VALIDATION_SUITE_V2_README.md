# TELOS Ollama Validation Suite V2

## Overview

Complete validation pipeline with telemetric signatures for IP protection. All validation data cryptographically signed and stored in Supabase with full session content.

## What's New in V2

### ✅ Telemetric Signatures
- Every turn cryptographically signed using session entropy
- Non-reproducible signatures for IP protection
- Full signature chain stored for verification

### ✅ Complete Session Data
- Full user messages and assistant responses stored
- Allowed for public ShareGPT/research data
- Enables deep analysis and counterfactual comparisons

### ✅ Supabase Integration
- Automatic storage to validation tables
- IP proof retrieval via views
- Statistics calculations via database functions

### ✅ Real Ollama Data
- No more API calls - all local execution
- Actual response generation with governance
- Real timing and fidelity measurements

## Quick Start

### 1. Run Quick Test (3 turns)
```bash
python3 run_ollama_validation_suite_v2.py quick
```

This will:
- Create a test session with 3 turns
- Generate responses with Ollama
- Sign each turn with telemetric signatures
- Store full data in Supabase
- Retrieve and display IP proof

### 2. Run Baseline Comparison (5 governance modes × 5 turns)
```bash
python3 run_ollama_validation_suite_v2.py baseline
```

This compares:
- **Stateless**: No governance
- **Prompt-only**: Purpose stated once
- **TELOS**: Full governance (simplified for now)

Results show fidelity improvement with governance.

## Architecture

```
User Request
    ↓
OllamaClient generates response
    ↓
Calculate fidelity (embeddings)
    ↓
Create delta data (timing, lengths, fidelity)
    ↓
TelemetricSignatureGenerator signs delta
    ↓
ValidationStorage stores to Supabase:
  - Full user message
  - Full assistant response
  - Fidelity score
  - Telemetric signature
  - Timing data
    ↓
Session marked complete
    ↓
IP proof retrievable
```

## Data Flow

### Session Creation
```python
session_id = suite._start_new_session(
    study_name="baseline_comparison_telos",
    pa_config={
        "purpose": "Provide accurate AI education",
        "scope": "Machine learning",
        "boundaries": ["Stay factual", "Be clear"]
    }
)
```

Creates:
- UUID session identifier
- Telemetric key generator
- Signature generator
- Supabase session record

### Turn Processing
```python
result = suite._run_signed_turn(
    turn_number=1,
    user_msg="What is machine learning?",
    governance_mode="telos",
    pa_config=pa_config
)
```

Generates:
- Response via Ollama
- Fidelity measurement
- Telemetric signature
- Supabase turn record

Returns:
- `response`: Generated text
- `fidelity`: 0-1 score
- `delta_t_ms`: Response time
- `signature`: First 32 chars of signature

### IP Proof Retrieval
```python
ip_proof = storage.get_ip_proof(session_id)
```

Returns:
- Session signature
- Key history hash
- Signature chain (all turn signatures)
- Turn count
- Timestamps

## File Structure

### Created Files
```
telos_privacy/
├── run_ollama_validation_suite_v2.py    # NEW - Updated validation suite
├── test_validation_pipeline_e2e.py      # NEW - End-to-end test
├── telos_purpose/
│   └── storage/
│       ├── __init__.py                  # NEW - Package init
│       └── validation_storage.py        # NEW - Storage module
├── supabase_validation_schema_CLEAN.sql # NEW - Schema migration
└── VALIDATION_SUITE_V2_README.md        # NEW - This file
```

### Modified Files
None - V2 is additive, doesn't modify existing validation code.

## Supabase Schema

### Tables
- `validation_telemetric_sessions` - Session-level data
- `validation_sessions` - Individual turns with full content
- `validation_counterfactual_comparisons` - Branch analysis

### Views
- `validation_ip_proofs` - IP verification data
- `validation_baseline_comparison` - Governance mode comparison
- `validation_counterfactual_summary` - Branch comparison results

### Functions
- `calculate_validation_statistics(study_name)` - Compare all modes

## Usage Examples

### Quick Test
```bash
# Run 3-turn test
python3 run_ollama_validation_suite_v2.py quick
```

Expected output:
```
Quick session: <uuid>
Turn 1/3: Hello, test message 1
  Response: Hello! I'm here to help...
  Signed: 3ecedc91db1ae1b5...

IP Proof Retrieved:
  Signed turns: 3/3
  Signature chain: 3 signatures
  Session signature: c4a31fdf90e9e936...
```

### Baseline Comparison
```bash
# Run full baseline study
python3 run_ollama_validation_suite_v2.py baseline
```

Expected output:
```
BASELINE COMPARISON RESULTS

STATELESS:
  Average Fidelity: 0.723
  Average Time: 45000ms
  Turns: 5

TELOS:
  Average Fidelity: 0.891
  Average Time: 47000ms
  Turns: 5

TELOS Improvement: +23.2%
```

## Performance

### Timing
- Session creation: <100ms
- Turn processing: 20-60 seconds (Ollama generation)
- Signature generation: <5ms
- Supabase storage: <100ms

### Scale
Current implementation can handle:
- Multiple concurrent sessions
- Hundreds of turns per session
- Full conversation storage
- Real-time signature generation

## IP Protection Benefits

### What This Provides

1. **Cryptographic Timestamps**
   - Each turn timestamped via signature
   - Non-repudiable proof of operation
   - Session-entropy based keys

2. **Signature Chains**
   - Complete chain from session start to end
   - Tamper-evident
   - Third-party verifiable

3. **Prior Art Documentation**
   - Supabase data proves methodology timestamp
   - Signature chain proves authenticity
   - Full session data proves approach

4. **Patent Support**
   - IP proof documents generated automatically
   - Signature chains establish priority date
   - Verification tools for auditors

## Next Steps

### Immediate
1. ✅ Quick test validation
2. ✅ Baseline comparison
3. ⏳ Full validation studies (100+ sessions)

### Short-term
1. Counterfactual analysis integration
2. Dual PA validation
3. ShareGPT dataset processing

### Medium-term
1. LangChain integration demo
2. Audit trail generation
3. Patent filing documentation

## Troubleshooting

### Ollama Connection Failed
```bash
# Check Ollama is running
ollama list

# Start if not running
ollama serve
```

### Supabase Connection Failed
```bash
# Check credentials
echo $SUPABASE_URL
echo $SUPABASE_KEY

# Or check .streamlit/secrets.toml
cat .streamlit/secrets.toml
```

### Import Errors
```bash
# Install dependencies
pip install supabase-py
pip install python-dotenv

# For tomli (secrets loading)
pip install tomli
```

## Files Reference

### Main Script
`run_ollama_validation_suite_v2.py` - Main validation suite

Key classes:
- `OllamaValidationSuiteV2` - Main suite class
- Methods:
  - `_start_new_session()` - Create signed session
  - `_run_signed_turn()` - Process turn with signature
  - `run_baseline_comparison()` - Compare governance modes
  - `run_quick_test()` - 3-turn verification

### Storage Module
`telos_purpose/storage/validation_storage.py`

Key methods:
- `create_validation_session()` - Create session with signature
- `store_signed_turn()` - Store turn with full content
- `mark_session_complete()` - Close session
- `get_ip_proof()` - Retrieve verification data
- `get_baseline_comparison()` - Query comparison results

### Test Script
`test_validation_pipeline_e2e.py` - End-to-end verification

Tests:
- Ollama connection
- Signature generation
- Supabase storage
- IP proof retrieval

## Success Criteria

### ✅ MVP Complete
- [x] Ollama generates responses locally
- [x] Telemetric signatures created
- [x] Full session data stored
- [x] IP proofs retrievable
- [x] End-to-end test passes

### 🎯 Production Ready
- [ ] 100+ validation sessions completed
- [ ] Baseline comparison data collected
- [ ] Counterfactual analysis functional
- [ ] LangChain demo working

## Support

For questions or issues:
1. Check this README
2. Review test output from `test_validation_pipeline_e2e.py`
3. Verify Supabase schema is applied
4. Check Ollama is running and responsive

---

**Status**: MVP Complete ✅
**Version**: 2.0
**Last Updated**: 2025-11-20
