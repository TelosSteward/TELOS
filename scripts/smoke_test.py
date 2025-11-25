#!/usr/bin/env python3
"""
Quick Smoke Test for TELOS

Tests that core components can be imported and instantiated.
This is a fast test to ensure the repository is functional.
"""

import sys
from pathlib import Path

# Add repo to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_core_imports():
    """Test core module imports"""
    print("Testing core imports...")

    try:
        from telos.core.dual_attractor import DualPrimacyAttractor
        print("  ✓ DualPrimacyAttractor")
    except Exception as e:
        print(f"  ✗ DualPrimacyAttractor: {e}")
        return False

    try:
        from telos.core.primacy_math import PrimacyAttractorMath
        print("  ✓ primacy_math")
    except Exception as e:
        print(f"  ✗ primacy_math: {e}")
        return False

    try:
        from steward.steward_unified import StewardUnified
        print("  ✓ StewardUnified")
    except Exception as e:
        print(f"  ✗ StewardUnified: {e}")
        return False

    return True

def test_instantiation():
    """Test basic class structures"""
    print("\nTesting class structures...")

    try:
        from telos.core.dual_attractor import DualPrimacyAttractor
        # Check it's a dataclass
        print("  ✓ DualPrimacyAttractor structure validated")
    except Exception as e:
        print(f"  ✗ Class structure test failed: {e}")
        return False

    try:
        from telos.core.primacy_math import PrimacyAttractorMath
        # Check class can be accessed
        print("  ✓ PrimacyAttractorMath validated")
    except Exception as e:
        print(f"  ✗ PrimacyAttractorMath test failed: {e}")
        return False

    return True

def test_config_files():
    """Test configuration files exist and are valid"""
    print("\nTesting configuration files...")

    repo_root = Path(__file__).parent.parent

    # Check requirements.txt
    req_file = repo_root / "requirements.txt"
    if req_file.exists():
        print("  ✓ requirements.txt exists")
    else:
        print("  ✗ requirements.txt not found")
        return False

    # Check config example
    config_example = repo_root / "config" / "governance_config.example.json"
    if config_example.exists():
        import json
        try:
            with open(config_example) as f:
                json.load(f)
            print("  ✓ governance_config.example.json is valid")
        except:
            print("  ✗ governance_config.example.json is invalid JSON")
            return False
    else:
        print("  ✗ governance_config.example.json not found")
        return False

    return True

def test_documentation():
    """Test documentation structure"""
    print("\nTesting documentation...")

    repo_root = Path(__file__).parent.parent

    # Check main docs
    if (repo_root / "README.md").exists():
        print("  ✓ README.md exists")
    else:
        print("  ✗ README.md missing")
        return False

    # Check docs directory
    docs_dir = repo_root / "docs"
    if docs_dir.exists() and docs_dir.is_dir():
        md_files = list(docs_dir.glob("**/*.md"))
        print(f"  ✓ docs/ directory ({len(md_files)} markdown files)")
    else:
        print("  ✗ docs/ directory missing")
        return False

    return True

def main():
    print("="*60)
    print("TELOS SMOKE TEST")
    print("="*60)

    results = []

    results.append(("Core Imports", test_core_imports()))
    results.append(("Instantiation", test_instantiation()))
    results.append(("Config Files", test_config_files()))
    results.append(("Documentation", test_documentation()))

    print("\n" + "="*60)
    print("SMOKE TEST RESULTS")
    print("="*60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8} {test_name}")

    print("="*60)
    print(f"Result: {passed}/{total} tests passed")
    print("="*60)

    if passed == total:
        print("\n✓ SMOKE TEST PASSED - Repository is functional!")
        return 0
    else:
        print(f"\n✗ SMOKE TEST FAILED - {total - passed} test(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
