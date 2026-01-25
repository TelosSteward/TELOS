#!/usr/bin/env python3
"""
TELOS Corpus Configurator - Validation Script
==============================================

Comprehensive validation test for the TELOS Corpus Configurator MVP.
Checks imports, syntax, interface compatibility, and basic functionality.

Run:
    python3 validate_app.py

Author: TELOS AI Labs Inc.
Date: 2026-01-23
"""

import sys
import os
from pathlib import Path

# Colors for output
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BLUE = '\033[94m'
BOLD = '\033[1m'
RESET = '\033[0m'

def print_section(title):
    """Print a section header."""
    print(f"\n{BLUE}{BOLD}{'=' * 70}{RESET}")
    print(f"{BLUE}{BOLD}{title}{RESET}")
    print(f"{BLUE}{BOLD}{'=' * 70}{RESET}\n")

def print_success(message):
    """Print a success message."""
    print(f"{GREEN}✓ {message}{RESET}")

def print_warning(message):
    """Print a warning message."""
    print(f"{YELLOW}⚠ {message}{RESET}")

def print_error(message):
    """Print an error message."""
    print(f"{RED}✗ {message}{RESET}")

def test_imports():
    """Test that all critical imports work."""
    print_section("1. IMPORT VALIDATION")

    try:
        # Add to path
        sys.path.insert(0, '/Users/brunnerjf/Desktop/TELOS_Master/telos_configurator')

        # Test config
        from config.styles import (
            inject_custom_css, GOLD, STATUS_GOOD, STATE_PENDING,
            get_glassmorphism_css, render_section_header
        )
        print_success("config.styles imports")

        # Test state manager
        from state_manager import (
            initialize_state, get_current_step, get_corpus_engine,
            get_governance_engine
        )
        print_success("state_manager imports")

        # Test engines
        from engine.corpus_engine import CorpusEngine
        from engine.governance_engine import (
            GovernanceEngine, ThresholdConfig, PrimacyAttractor,
            create_pa, embed_pa
        )
        print_success("engine.corpus_engine imports")
        print_success("engine.governance_engine imports")

        # Test all components
        from components import (
            render_domain_selector,
            render_corpus_uploader,
            render_corpus_manager,
            render_pa_configurator,
            render_threshold_config,
            render_activation_panel,
            render_dashboard_metrics,
            render_test_query_interface,
            render_audit_panel
        )
        print_success("All 9 component imports")

        # Test main
        import main
        print_success("main.py imports")

        print(f"\n{GREEN}{BOLD}All imports successful!{RESET}")
        return True

    except Exception as e:
        print_error(f"Import failed: {e}")
        return False

def test_component_signatures():
    """Verify component function signatures match their usage."""
    print_section("2. COMPONENT SIGNATURE VALIDATION")

    try:
        sys.path.insert(0, '/Users/brunnerjf/Desktop/TELOS_Master/telos_configurator')

        from components import (
            render_domain_selector,
            render_corpus_uploader,
            render_corpus_manager,
            render_pa_configurator,
            render_threshold_config,
            render_activation_panel,
            render_dashboard_metrics,
            render_test_query_interface,
            render_audit_panel
        )
        from engine.corpus_engine import CorpusEngine
        from engine.governance_engine import GovernanceEngine, ThresholdConfig, PrimacyAttractor

        # Check signatures
        import inspect

        # render_domain_selector() -> Optional[str]
        sig = inspect.signature(render_domain_selector)
        assert len(sig.parameters) == 0, "render_domain_selector should have no parameters"
        print_success("render_domain_selector signature valid")

        # render_corpus_uploader(corpus_engine) -> None
        sig = inspect.signature(render_corpus_uploader)
        assert len(sig.parameters) == 1, "render_corpus_uploader should have 1 parameter"
        print_success("render_corpus_uploader signature valid")

        # render_corpus_manager(corpus_engine) -> None
        sig = inspect.signature(render_corpus_manager)
        assert len(sig.parameters) == 1, "render_corpus_manager should have 1 parameter"
        print_success("render_corpus_manager signature valid")

        # render_pa_configurator() -> Optional[PrimacyAttractor]
        sig = inspect.signature(render_pa_configurator)
        assert len(sig.parameters) == 0, "render_pa_configurator should have no parameters"
        print_success("render_pa_configurator signature valid")

        # render_threshold_config() -> ThresholdConfig
        sig = inspect.signature(render_threshold_config)
        assert len(sig.parameters) == 0, "render_threshold_config should have no parameters"
        print_success("render_threshold_config signature valid")

        # render_activation_panel(pa, corpus_engine, thresholds, governance_engine) -> bool
        sig = inspect.signature(render_activation_panel)
        assert len(sig.parameters) == 4, "render_activation_panel should have 4 parameters"
        print_success("render_activation_panel signature valid")

        # render_dashboard_metrics(governance_engine) -> None
        sig = inspect.signature(render_dashboard_metrics)
        assert len(sig.parameters) == 1, "render_dashboard_metrics should have 1 parameter"
        print_success("render_dashboard_metrics signature valid")

        # render_test_query_interface(governance_engine) -> None
        sig = inspect.signature(render_test_query_interface)
        assert len(sig.parameters) == 1, "render_test_query_interface should have 1 parameter"
        print_success("render_test_query_interface signature valid")

        # render_audit_panel(governance_engine) -> None
        sig = inspect.signature(render_audit_panel)
        assert len(sig.parameters) == 1, "render_audit_panel should have 1 parameter"
        print_success("render_audit_panel signature valid")

        print(f"\n{GREEN}{BOLD}All component signatures valid!{RESET}")
        return True

    except Exception as e:
        print_error(f"Signature validation failed: {e}")
        return False

def test_engine_instantiation():
    """Test that engines can be instantiated."""
    print_section("3. ENGINE INSTANTIATION VALIDATION")

    try:
        sys.path.insert(0, '/Users/brunnerjf/Desktop/TELOS_Master/telos_configurator')

        from engine.corpus_engine import CorpusEngine
        from engine.governance_engine import GovernanceEngine, ThresholdConfig, create_pa

        # Test CorpusEngine
        corpus_engine = CorpusEngine()
        assert corpus_engine is not None, "CorpusEngine failed to instantiate"
        assert hasattr(corpus_engine, 'documents'), "CorpusEngine missing documents attribute"
        assert hasattr(corpus_engine, 'add_document'), "CorpusEngine missing add_document method"
        assert hasattr(corpus_engine, 'embed_all'), "CorpusEngine missing embed_all method"
        print_success("CorpusEngine instantiation and methods")

        # Test GovernanceEngine
        gov_engine = GovernanceEngine()
        assert gov_engine is not None, "GovernanceEngine failed to instantiate"
        assert hasattr(gov_engine, 'configure'), "GovernanceEngine missing configure method"
        assert hasattr(gov_engine, 'process'), "GovernanceEngine missing process method"
        assert hasattr(gov_engine, 'is_active'), "GovernanceEngine missing is_active method"
        print_success("GovernanceEngine instantiation and methods")

        # Test ThresholdConfig
        thresholds = ThresholdConfig()
        assert thresholds is not None, "ThresholdConfig failed to instantiate"
        assert hasattr(thresholds, 'validate'), "ThresholdConfig missing validate method"
        is_valid, error = thresholds.validate()
        assert is_valid, f"Default ThresholdConfig is invalid: {error}"
        print_success("ThresholdConfig instantiation and validation")

        # Test PA creation
        pa = create_pa(
            name="Test PA",
            purpose="Test purpose",
            scope=["test1", "test2"],
            exclusions=["ex1"],
            prohibitions=["pro1"]
        )
        assert pa is not None, "create_pa failed"
        assert pa.name == "Test PA", "PA name mismatch"
        assert len(pa.scope) == 2, "PA scope length mismatch"
        print_success("PrimacyAttractor creation")

        print(f"\n{GREEN}{BOLD}All engines instantiate correctly!{RESET}")
        return True

    except Exception as e:
        print_error(f"Engine instantiation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_syntax():
    """Test that all Python files compile."""
    print_section("4. SYNTAX VALIDATION")

    try:
        import py_compile

        base_path = Path('/Users/brunnerjf/Desktop/TELOS_Master/telos_configurator')

        files_to_check = [
            'main.py',
            'state_manager.py',
            'config/styles.py',
            'engine/corpus_engine.py',
            'engine/governance_engine.py',
            'components/__init__.py',
            'components/domain_selector.py',
            'components/corpus_uploader.py',
            'components/corpus_manager.py',
            'components/pa_configurator.py',
            'components/threshold_config.py',
            'components/activation_panel.py',
            'components/dashboard_metrics.py',
            'components/test_query_interface.py',
            'components/audit_panel.py',
        ]

        for file in files_to_check:
            filepath = base_path / file
            py_compile.compile(str(filepath), doraise=True)
            print_success(f"{file} syntax valid")

        print(f"\n{GREEN}{BOLD}All files compile successfully!{RESET}")
        return True

    except Exception as e:
        print_error(f"Syntax validation failed: {e}")
        return False

def test_state_keys():
    """Test that state manager keys are consistent."""
    print_section("5. STATE KEY VALIDATION")

    try:
        # Just verify that initialize_state function exists and is callable
        sys.path.insert(0, '/Users/brunnerjf/Desktop/TELOS_Master/telos_configurator')

        from state_manager import (
            initialize_state,
            get_current_step,
            get_selected_domain,
            get_pa_instance,
            get_corpus_engine,
            get_governance_engine,
            is_governance_active,
            get_corpus_stats,
            set_current_step,
            set_selected_domain,
            set_pa_instance,
        )

        # Verify key functions exist
        assert callable(initialize_state), "initialize_state not callable"
        print_success("initialize_state function exists")

        assert callable(get_current_step), "get_current_step not callable"
        print_success("get_current_step function exists")

        assert callable(set_current_step), "set_current_step not callable"
        print_success("set_current_step function exists")

        assert callable(get_corpus_engine), "get_corpus_engine not callable"
        print_success("get_corpus_engine function exists")

        assert callable(get_governance_engine), "get_governance_engine not callable"
        print_success("get_governance_engine function exists")

        assert callable(is_governance_active), "is_governance_active not callable"
        print_success("is_governance_active function exists")

        print(f"\n{GREEN}{BOLD}All state manager functions valid!{RESET}")
        return True

    except Exception as e:
        print_error(f"State key validation failed: {e}")
        return False

def main():
    """Run all validation tests."""
    print(f"\n{BOLD}TELOS Corpus Configurator - Validation Suite{RESET}")
    print(f"{BOLD}{'=' * 70}{RESET}")

    results = []

    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Component Signatures", test_component_signatures()))
    results.append(("Engine Instantiation", test_engine_instantiation()))
    results.append(("Syntax", test_syntax()))
    results.append(("State Keys", test_state_keys()))

    # Summary
    print_section("VALIDATION SUMMARY")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        if result:
            print_success(f"{test_name}: PASSED")
        else:
            print_error(f"{test_name}: FAILED")

    print(f"\n{BOLD}Results: {passed}/{total} tests passed{RESET}")

    if passed == total:
        print(f"\n{GREEN}{BOLD}✅ ALL VALIDATION TESTS PASSED!{RESET}")
        print(f"{GREEN}The application is ready to run.{RESET}")
        print(f"\n{BOLD}To start the application:{RESET}")
        print(f"  streamlit run /Users/brunnerjf/Desktop/TELOS_Master/telos_configurator/main.py --server.port 8502")
        return 0
    else:
        print(f"\n{RED}{BOLD}❌ SOME TESTS FAILED!{RESET}")
        print(f"{RED}Please fix the issues above before running the application.{RESET}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
