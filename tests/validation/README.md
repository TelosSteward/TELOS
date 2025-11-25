# Validation Scripts

This directory contains validation and testing scripts for the TELOS framework.

## Current Status

**Note:** Many scripts in this directory reference the old package name (`telos_purpose`) and use legacy test frameworks. These scripts are preserved for historical reference but may need updates to work with the current codebase.

## Active Validation

For current validation, use:
- `/scripts/validate_repository.py` - Comprehensive repository validation
- `/scripts/health_check.py` - Full health check including functional tests
- `/tests/unit/` - Updated unit tests

## Legacy Scripts (May Need Updates)

The following scripts use old import paths and may need updating:

### Test Data Generation
- `generate_test_sessions.py` - Test session data generation
- `edge_case_tests.py` - Edge case test generation

### Validation Runners
- `run_internal_test0.py` - Internal test runner (legacy)
- `validate_platform.py` - Platform validation
- `integration_tests.py` - Integration test suite
- `performance_check.py` - Performance benchmarking

### Analysis Tools
- `retro_analyzer.py` - Retrospective analysis tool
- `system_health_monitor.py` - System health monitoring
- `summarize_internal_test0.py` - Test summarization

## Updating Legacy Scripts

To update these scripts for current use:

1. Replace `telos_purpose` imports with `telos`:
   ```python
   # Old
   from telos_purpose.core.dual_attractor import DualPrimacyAttractor

   # New
   from telos.core.dual_attractor import DualPrimacyAttractor
   ```

2. Update module paths in documentation strings

3. Test the script works with current code structure

## Current Working Validation

See `VALIDATION_REPORT.md` in the repository root for the latest validation results using current tools.
