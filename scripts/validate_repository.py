#!/usr/bin/env python3
"""
TELOS Repository Validation Script

Validates repository integrity after cleanup/reorganization.
Runs tiered validation from quick sanity checks to full integration tests.

Usage:
    python scripts/validate_repository.py [--level 1|2|3]
"""

import sys
import os
import importlib
import json
from pathlib import Path
from typing import List, Tuple, Dict

# Add repo root to path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

class ValidationResult:
    def __init__(self):
        self.passed = []
        self.failed = []
        self.warnings = []

    def add_pass(self, test_name: str, detail: str = ""):
        self.passed.append((test_name, detail))
        print(f"{Colors.GREEN}✓{Colors.END} {test_name} {detail}")

    def add_fail(self, test_name: str, error: str):
        self.failed.append((test_name, error))
        print(f"{Colors.RED}✗{Colors.END} {test_name}")
        print(f"  {Colors.RED}Error: {error}{Colors.END}")

    def add_warning(self, test_name: str, warning: str):
        self.warnings.append((test_name, warning))
        print(f"{Colors.YELLOW}⚠{Colors.END} {test_name}")
        print(f"  {Colors.YELLOW}Warning: {warning}{Colors.END}")

    def summary(self) -> bool:
        print("\n" + "="*70)
        print(f"{Colors.BOLD}VALIDATION SUMMARY{Colors.END}")
        print("="*70)
        print(f"{Colors.GREEN}Passed:{Colors.END}   {len(self.passed)}")
        print(f"{Colors.RED}Failed:{Colors.END}   {len(self.failed)}")
        print(f"{Colors.YELLOW}Warnings:{Colors.END} {len(self.warnings)}")

        if self.failed:
            print(f"\n{Colors.RED}{Colors.BOLD}FAILED TESTS:{Colors.END}")
            for test, error in self.failed:
                print(f"  • {test}: {error}")

        if self.warnings:
            print(f"\n{Colors.YELLOW}{Colors.BOLD}WARNINGS:{Colors.END}")
            for test, warning in self.warnings:
                print(f"  • {test}: {warning}")

        print("="*70)

        return len(self.failed) == 0


class RepositoryValidator:
    def __init__(self):
        self.result = ValidationResult()
        self.repo_root = REPO_ROOT

    def run_level_1(self):
        """Level 1: Quick Sanity Checks"""
        print("\n" + "="*70)
        print(f"{Colors.BOLD}{Colors.BLUE}LEVEL 1: QUICK SANITY CHECKS{Colors.END}")
        print("="*70 + "\n")

        self._check_core_imports()
        self._check_dependencies()
        self._check_config_files()
        self._check_critical_directories()

    def run_level_2(self):
        """Level 2: Unit Tests"""
        print("\n" + "="*70)
        print(f"{Colors.BOLD}{Colors.BLUE}LEVEL 2: UNIT TESTS{Colors.END}")
        print("="*70 + "\n")

        self._run_pytest_unit_tests()

    def run_level_3(self):
        """Level 3: Integration Tests"""
        print("\n" + "="*70)
        print(f"{Colors.BOLD}{Colors.BLUE}LEVEL 3: INTEGRATION VALIDATION{Colors.END}")
        print("="*70 + "\n")

        self._test_governance_engine()
        self._test_embedding_generation()
        self._test_steward_defense()

    def _check_core_imports(self):
        """Check that core modules can be imported"""
        modules_to_test = [
            "telos.core.governance_engine",
            "telos.core.dual_primacy_attractor",
            "telos.profiling.progressive_primacy_extractor",
            "telos.llm.claude_adapter",
            "telos.llm.mistral_adapter",
            "steward.steward_unified",
            "observatory.core.state_manager",
            "observatory.services.steward_defense",
        ]

        for module_name in modules_to_test:
            try:
                importlib.import_module(module_name)
                self.result.add_pass(f"Import {module_name}")
            except ImportError as e:
                self.result.add_fail(f"Import {module_name}", str(e))
            except Exception as e:
                self.result.add_warning(f"Import {module_name}", str(e))

    def _check_dependencies(self):
        """Check that key dependencies are available"""
        dependencies = [
            "streamlit",
            "mistralai",
            "anthropic",
            "sentence_transformers",
            "torch",
            "numpy",
            "pandas",
        ]

        for dep in dependencies:
            try:
                importlib.import_module(dep)
                self.result.add_pass(f"Dependency {dep}")
            except ImportError:
                self.result.add_warning(f"Dependency {dep}", "Not installed (may be optional)")

    def _check_config_files(self):
        """Check that configuration files exist"""
        config_files = [
            "requirements.txt",
            "setup.py",
            "config/governance_config.example.json",
        ]

        for config_file in config_files:
            path = self.repo_root / config_file
            if path.exists():
                self.result.add_pass(f"Config file {config_file}")
            else:
                self.result.add_fail(f"Config file {config_file}", "File not found")

        # Check if actual config exists (optional)
        actual_config = self.repo_root / "config/governance_config.json"
        if actual_config.exists():
            self.result.add_pass("Config file config/governance_config.json")
        else:
            self.result.add_warning("Config file config/governance_config.json",
                                   "Not found (create from example)")

    def _check_critical_directories(self):
        """Check that critical directories exist"""
        critical_dirs = [
            "telos/core",
            "telos/profiling",
            "telos/llm",
            "steward",
            "observatory/core",
            "observatory/components",
            "tests/unit",
            "tests/validation",
            "docs",
            "config",
        ]

        for dir_path in critical_dirs:
            path = self.repo_root / dir_path
            if path.exists() and path.is_dir():
                self.result.add_pass(f"Directory {dir_path}")
            else:
                self.result.add_fail(f"Directory {dir_path}", "Directory not found")

    def _run_pytest_unit_tests(self):
        """Run pytest on unit tests"""
        try:
            import pytest

            # Run pytest on unit tests
            test_dir = self.repo_root / "tests" / "unit"
            if not test_dir.exists():
                self.result.add_fail("Unit tests directory", "tests/unit not found")
                return

            # Count test files
            test_files = list(test_dir.glob("test_*.py"))
            if not test_files:
                self.result.add_warning("Unit tests", "No test files found in tests/unit/")
                return

            self.result.add_pass(f"Found {len(test_files)} unit test file(s)")

            # Run tests
            print(f"\n{Colors.BLUE}Running pytest on unit tests...{Colors.END}\n")
            exit_code = pytest.main([
                str(test_dir),
                "-v",
                "--tb=short",
                "-x"  # Stop on first failure
            ])

            if exit_code == 0:
                self.result.add_pass("Unit tests execution", "All tests passed")
            elif exit_code == 5:
                self.result.add_warning("Unit tests execution", "No tests collected")
            else:
                self.result.add_fail("Unit tests execution", f"Tests failed with exit code {exit_code}")

        except ImportError:
            self.result.add_warning("pytest", "Not installed - skipping unit tests")
        except Exception as e:
            self.result.add_fail("Unit tests execution", str(e))

    def _test_governance_engine(self):
        """Test basic governance engine functionality"""
        try:
            from telos.core.dual_primacy_attractor import DualPrimacyAttractor

            # Create a simple DPA instance
            dpa = DualPrimacyAttractor()
            self.result.add_pass("DualPrimacyAttractor instantiation")

            # Test basic operations
            if hasattr(dpa, 'user_pa') and hasattr(dpa, 'ai_pa'):
                self.result.add_pass("DualPrimacyAttractor attributes")
            else:
                self.result.add_warning("DualPrimacyAttractor attributes", "Missing expected attributes")

        except Exception as e:
            self.result.add_fail("Governance engine test", str(e))

    def _test_embedding_generation(self):
        """Test embedding generation"""
        try:
            from telos.core.embedding_provider import EmbeddingProvider

            # Try to create provider (may fail if no API key)
            try:
                provider = EmbeddingProvider()
                self.result.add_pass("EmbeddingProvider instantiation")
            except Exception as e:
                self.result.add_warning("EmbeddingProvider instantiation",
                                       f"Failed (API key may be missing): {str(e)}")

        except Exception as e:
            self.result.add_fail("Embedding provider test", str(e))

    def _test_steward_defense(self):
        """Test steward defense layers"""
        try:
            from observatory.services.steward_defense import StewardDefenseLayers

            # Create instance
            defense = StewardDefenseLayers()
            self.result.add_pass("StewardDefenseLayers instantiation")

            # Check if it has expected methods
            if hasattr(defense, 'apply_governance'):
                self.result.add_pass("StewardDefenseLayers has apply_governance method")
            else:
                self.result.add_warning("StewardDefenseLayers", "Missing apply_governance method")

        except Exception as e:
            self.result.add_fail("Steward defense test", str(e))


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Validate TELOS repository integrity")
    parser.add_argument("--level", type=int, default=2, choices=[1, 2, 3],
                       help="Validation level (1=sanity, 2=unit tests, 3=integration)")
    args = parser.parse_args()

    print(f"\n{Colors.BOLD}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}TELOS REPOSITORY VALIDATION{Colors.END}")
    print(f"{Colors.BOLD}{'='*70}{Colors.END}")
    print(f"Level: {args.level}")
    print(f"Repository: {REPO_ROOT}")
    print()

    validator = RepositoryValidator()

    # Run requested validation levels
    if args.level >= 1:
        validator.run_level_1()

    if args.level >= 2:
        validator.run_level_2()

    if args.level >= 3:
        validator.run_level_3()

    # Print summary
    success = validator.result.summary()

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
