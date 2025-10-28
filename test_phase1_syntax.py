#!/usr/bin/env python3
"""
Phase 1 Syntax Test
Tests that streamlit_live_comparison.py has no syntax errors after Phase 1 changes.
"""

import py_compile
import sys

try:
    py_compile.compile(
        'telos_purpose/dev_dashboard/streamlit_live_comparison.py',
        doraise=True
    )
    print("✅ Syntax check PASSED - No Python syntax errors")
    sys.exit(0)
except py_compile.PyCompileError as e:
    print("❌ Syntax check FAILED")
    print(e)
    sys.exit(1)
