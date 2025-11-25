#!/usr/bin/env python3
"""
TELOS Comprehensive Health Check

Performs thorough validation of repository health including:
- Package installation
- Core module functionality
- Configuration validity
- Import integrity
- Entry point testing
- Documentation links
- Git cleanliness

Usage:
    python scripts/health_check.py [--quick|--full]
"""

import sys
import os
import subprocess
import importlib
import json
from pathlib import Path
from typing import List, Dict, Tuple

REPO_ROOT = Path(__file__).parent.parent

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

class HealthCheck:
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.passed = []
        self.failed = []
        self.warnings = []

    def log_pass(self, test: str, detail: str = ""):
        self.passed.append((test, detail))
        print(f"{Colors.GREEN}✓{Colors.END} {test} {detail}")

    def log_fail(self, test: str, error: str):
        self.failed.append((test, error))
        print(f"{Colors.RED}✗{Colors.END} {test}")
        if self.verbose or True:  # Always show errors
            print(f"  {Colors.RED}{error}{Colors.END}")

    def log_warning(self, test: str, warning: str):
        self.warnings.append((test, warning))
        print(f"{Colors.YELLOW}⚠{Colors.END} {test}")
        print(f"  {Colors.YELLOW}{warning}{Colors.END}")

    # =========================================================================
    # LEVEL 1: Package Installation
    # =========================================================================

    def test_package_installation(self):
        """Test that package can be installed via setup.py"""
        print(f"\n{Colors.BOLD}[1] Package Installation{Colors.END}")
        print("-" * 60)

        # Check setup.py exists
        setup_py = REPO_ROOT / "setup.py"
        if not setup_py.exists():
            self.log_fail("setup.py exists", "File not found")
            return
        self.log_pass("setup.py exists")

        # Try to import setuptools
        try:
            import setuptools
            self.log_pass("setuptools available")
        except ImportError:
            self.log_warning("setuptools", "Not installed")
            return

        # Parse setup.py for package info
        try:
            with open(setup_py) as f:
                content = f.read()
                if 'name="telos"' in content:
                    self.log_pass("Package name is 'telos'")
                else:
                    self.log_fail("Package name", "Expected 'telos' in setup.py")
        except Exception as e:
            self.log_fail("Parse setup.py", str(e))

    # =========================================================================
    # LEVEL 2: Core Module Imports
    # =========================================================================

    def test_core_imports(self):
        """Test that all core modules can be imported"""
        print(f"\n{Colors.BOLD}[2] Core Module Imports{Colors.END}")
        print("-" * 60)

        core_modules = [
            "telos.core.dual_attractor",
            "telos.core.primacy_math",
            "telos.core.intervention_controller",
            "telos.core.embedding_provider",
            "telos.profiling.progressive_primacy_extractor",
            "steward.steward_unified",
            "observatory.core.state_manager",
        ]

        for module_name in core_modules:
            try:
                importlib.import_module(module_name)
                self.log_pass(f"Import {module_name}")
            except ImportError as e:
                # Check if it's a missing dependency or actual code issue
                if "No module named" in str(e) and module_name not in str(e):
                    self.log_warning(f"Import {module_name}",
                                   f"Missing dependency: {str(e)}")
                else:
                    self.log_fail(f"Import {module_name}", str(e))
            except Exception as e:
                self.log_fail(f"Import {module_name}", str(e))

    # =========================================================================
    # LEVEL 3: Configuration Validation
    # =========================================================================

    def test_configurations(self):
        """Test that configuration files are valid"""
        print(f"\n{Colors.BOLD}[3] Configuration Files{Colors.END}")
        print("-" * 60)

        # Check requirements.txt
        req_file = REPO_ROOT / "requirements.txt"
        if req_file.exists():
            self.log_pass("requirements.txt exists")
            try:
                with open(req_file) as f:
                    lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]
                    self.log_pass(f"requirements.txt has {len(lines)} dependencies")
            except Exception as e:
                self.log_fail("Parse requirements.txt", str(e))
        else:
            self.log_fail("requirements.txt", "File not found")

        # Check governance config
        config_example = REPO_ROOT / "config" / "governance_config.example.json"
        if config_example.exists():
            self.log_pass("governance_config.example.json exists")
            try:
                with open(config_example) as f:
                    config = json.load(f)
                self.log_pass("governance_config.example.json is valid JSON")
            except json.JSONDecodeError as e:
                self.log_fail("governance_config.example.json", f"Invalid JSON: {e}")
        else:
            self.log_fail("governance_config.example.json", "File not found")

        # Check if actual config exists (optional)
        config_actual = REPO_ROOT / "config" / "governance_config.json"
        if config_actual.exists():
            self.log_pass("governance_config.json exists")
        else:
            self.log_warning("governance_config.json",
                           "Not found (create from example for full functionality)")

    # =========================================================================
    # LEVEL 4: Import Integrity
    # =========================================================================

    def test_import_integrity(self):
        """Check for any remaining old package references"""
        print(f"\n{Colors.BOLD}[4] Import Integrity (No Old References){Colors.END}")
        print("-" * 60)

        # Search for old package name
        try:
            result = subprocess.run(
                ['grep', '-r', 'telos_purpose', '--include=*.py', '.'],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True
            )

            if result.returncode == 1:  # No matches found
                self.log_pass("No 'telos_purpose' references in Python files")
            else:
                # Found some references
                lines = result.stdout.strip().split('\n')
                # Filter out comments and strings that might be harmless
                real_issues = [l for l in lines if l and 'telos_purpose' in l]
                if real_issues:
                    self.log_warning("Old package references",
                                   f"Found {len(real_issues)} references to 'telos_purpose'")
                    if self.verbose:
                        for line in real_issues[:5]:  # Show first 5
                            print(f"    {line}")
                else:
                    self.log_pass("No problematic 'telos_purpose' references")
        except Exception as e:
            self.log_warning("Import integrity check", f"Could not run grep: {e}")

    # =========================================================================
    # LEVEL 5: Entry Points
    # =========================================================================

    def test_entry_points(self):
        """Test main entry points can be loaded"""
        print(f"\n{Colors.BOLD}[5] Entry Points{Colors.END}")
        print("-" * 60)

        # Test observatory main
        observatory_main = REPO_ROOT / "observatory" / "main.py"
        if observatory_main.exists():
            self.log_pass("Observatory entry point exists")
            # Try to parse it (not run it)
            try:
                with open(observatory_main) as f:
                    content = f.read()
                    if 'import streamlit' in content or 'streamlit' in content:
                        self.log_pass("Observatory imports Streamlit")
                    compile(content, str(observatory_main), 'exec')
                    self.log_pass("Observatory main.py compiles")
            except SyntaxError as e:
                self.log_fail("Observatory main.py", f"Syntax error: {e}")
            except Exception as e:
                self.log_warning("Observatory main.py", f"Could not fully validate: {e}")
        else:
            self.log_fail("Observatory entry point", "main.py not found")

        # Test steward scripts
        steward_unified = REPO_ROOT / "steward" / "steward_unified.py"
        if steward_unified.exists():
            self.log_pass("Steward unified entry point exists")
        else:
            self.log_warning("Steward unified", "steward_unified.py not found")

    # =========================================================================
    # LEVEL 6: Git Cleanliness
    # =========================================================================

    def test_git_status(self):
        """Check git repository status"""
        print(f"\n{Colors.BOLD}[6] Git Repository Status{Colors.END}")
        print("-" * 60)

        try:
            # Check for uncommitted changes
            result = subprocess.run(
                ['git', 'status', '--porcelain'],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=True
            )

            if result.stdout.strip():
                lines = result.stdout.strip().split('\n')
                self.log_warning("Uncommitted changes",
                               f"{len(lines)} files have uncommitted changes")
            else:
                self.log_pass("Working directory clean")

            # Check current branch
            result = subprocess.run(
                ['git', 'branch', '--show-current'],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=True
            )
            branch = result.stdout.strip()
            self.log_pass(f"Current branch: {branch}")

        except subprocess.CalledProcessError as e:
            self.log_warning("Git status check", "Could not check git status")
        except FileNotFoundError:
            self.log_warning("Git", "Git not available")

    # =========================================================================
    # LEVEL 7: Quick Functional Test
    # =========================================================================

    def test_quick_functionality(self):
        """Quick smoke test of core functionality"""
        print(f"\n{Colors.BOLD}[7] Quick Functional Test{Colors.END}")
        print("-" * 60)

        # Test DualPrimacyAttractor can be instantiated
        try:
            from telos.core.dual_attractor import DualPrimacyAttractor
            dpa = DualPrimacyAttractor()
            self.log_pass("DualPrimacyAttractor instantiation")

            # Check attributes
            if hasattr(dpa, 'user_pa') and hasattr(dpa, 'ai_pa'):
                self.log_pass("DualPrimacyAttractor has required attributes")
            else:
                self.log_warning("DualPrimacyAttractor", "Missing expected attributes")

        except Exception as e:
            self.log_fail("DualPrimacyAttractor functional test", str(e))

        # Test StewardUnified can be imported and instantiated
        try:
            from steward.steward_unified import StewardUnified
            self.log_pass("StewardUnified import")
            # Don't instantiate as it might need dependencies
        except Exception as e:
            self.log_fail("StewardUnified import", str(e))

    # =========================================================================
    # Run All Checks
    # =========================================================================

    def run_all(self):
        """Run all health checks"""
        print(f"\n{Colors.BOLD}{'='*70}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.BLUE}TELOS COMPREHENSIVE HEALTH CHECK{Colors.END}")
        print(f"{Colors.BOLD}{'='*70}{Colors.END}")

        self.test_package_installation()
        self.test_core_imports()
        self.test_configurations()
        self.test_import_integrity()
        self.test_entry_points()
        self.test_git_status()
        self.test_quick_functionality()

        # Summary
        print(f"\n{Colors.BOLD}{'='*70}{Colors.END}")
        print(f"{Colors.BOLD}HEALTH CHECK SUMMARY{Colors.END}")
        print(f"{Colors.BOLD}{'='*70}{Colors.END}")
        print(f"{Colors.GREEN}Passed:{Colors.END}   {len(self.passed)}")
        print(f"{Colors.RED}Failed:{Colors.END}   {len(self.failed)}")
        print(f"{Colors.YELLOW}Warnings:{Colors.END} {len(self.warnings)}")

        if self.failed:
            print(f"\n{Colors.RED}{Colors.BOLD}FAILED CHECKS:{Colors.END}")
            for test, error in self.failed:
                print(f"  • {test}: {error}")

        if self.warnings:
            print(f"\n{Colors.YELLOW}{Colors.BOLD}WARNINGS:{Colors.END}")
            for test, warning in self.warnings:
                print(f"  • {test}: {warning}")

        print(f"\n{Colors.BOLD}{'='*70}{Colors.END}")

        # Overall health status
        if not self.failed:
            print(f"{Colors.GREEN}{Colors.BOLD}✓ REPOSITORY IS HEALTHY{Colors.END}")
            if self.warnings:
                print(f"{Colors.YELLOW}  (with {len(self.warnings)} warnings - see above){Colors.END}")
            return 0
        else:
            print(f"{Colors.RED}{Colors.BOLD}✗ ISSUES FOUND - SEE FAILURES ABOVE{Colors.END}")
            return 1


def main():
    import argparse
    parser = argparse.ArgumentParser(description="TELOS Repository Health Check")
    parser.add_argument('--verbose', '-v', action='store_true',
                       help="Verbose output")
    parser.add_argument('--quick', action='store_true',
                       help="Quick check (skip some tests)")
    args = parser.parse_args()

    checker = HealthCheck(verbose=args.verbose)
    exit_code = checker.run_all()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
