#!/usr/bin/env python3
"""
Extract Essential TELOS Files for Opus Review

Traces import dependencies from main entry points to identify
only the files actually used in canonical TELOS execution.

Usage:
    python3 extract_essential_telos.py [--yes]

Options:
    --yes    Skip confirmation prompt and proceed with extraction
"""

import os
import sys
import ast
import shutil
from pathlib import Path
from typing import Set, List, Dict

# Project root
PROJECT_ROOT = Path(__file__).parent.resolve()

# Entry points for dependency tracing
ENTRY_POINTS = [
    # Observatory V3 (main UI)
    "telos_observatory_v3/main.py",

    # Core TELOS Runtime
    "telos_purpose/core/dual_attractor.py",
    "telos_purpose/core/proportional_controller.py",
    "telos_purpose/core/intervention_controller.py",
    "telos_purpose/core/unified_steward.py",
    "telos_purpose/core/session_state.py",

    # Steward Orchestration
    "steward.py",
    "steward_governance_orchestrator.py",

    # Privacy Infrastructure
    "telos_privacy/cryptography/telemetric_keys.py",
]

# Exclude patterns
EXCLUDE_PATTERNS = [
    "test_",           # Test files
    "_test.py",        # More test files
    "conftest.py",     # Pytest config
    "setup.py",        # Setup files
    "demo_mode/",      # Demo data
    "validation_results/",  # Results data
    "sessions/",       # Session data
    ".pytest_cache",   # Cache
    "__pycache__",     # Python cache
    ".git",            # Git
    ".claude",         # Claude config
    "scripts/",        # Utility scripts
    "docs/",           # Documentation
    "beta_data/",      # Beta data
    "grant_applications/",  # Grant materials
]

# Files to always include (even if not imported)
FORCE_INCLUDE = [
    "telos_purpose/core/primacy_math.py",  # Core math
    "telos_purpose/core/embedding_provider.py",  # Essential provider
    "telos_purpose/core/governance_config.py",  # Config
    "mistral_adapter.py",  # LLM adapter
]


class ImportTracer(ast.NodeVisitor):
    """AST visitor to extract imports from Python files."""

    def __init__(self):
        self.imports: Set[str] = set()

    def visit_Import(self, node):
        """Handle 'import x' statements."""
        for alias in node.names:
            self.imports.add(alias.name.split('.')[0])
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        """Handle 'from x import y' statements."""
        if node.module:
            self.imports.add(node.module.split('.')[0])
        self.generic_visit(node)


def should_exclude(file_path: str) -> bool:
    """Check if file should be excluded based on patterns."""
    for pattern in EXCLUDE_PATTERNS:
        if pattern in file_path:
            return True
    return False


def extract_imports(file_path: Path) -> Set[str]:
    """Extract import statements from a Python file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read(), filename=str(file_path))

        tracer = ImportTracer()
        tracer.visit(tree)
        return tracer.imports
    except Exception as e:
        print(f"⚠️  Error parsing {file_path}: {e}")
        return set()


def resolve_import_to_file(import_name: str, base_dir: Path) -> List[Path]:
    """Resolve an import name to potential file paths."""
    candidates = []

    # Try as direct module
    module_file = base_dir / f"{import_name}.py"
    if module_file.exists() and not should_exclude(str(module_file)):
        candidates.append(module_file)

    # Try as package
    package_dir = base_dir / import_name
    if package_dir.is_dir():
        # Find all .py files in package
        for py_file in package_dir.rglob("*.py"):
            if not should_exclude(str(py_file)):
                candidates.append(py_file)

    return candidates


def trace_dependencies(entry_point: Path, base_dir: Path) -> Set[Path]:
    """Recursively trace dependencies from an entry point."""
    visited = set()
    to_visit = {entry_point}
    dependencies = set()

    while to_visit:
        current = to_visit.pop()

        if current in visited or not current.exists():
            continue

        visited.add(current)
        dependencies.add(current)

        # Extract imports from current file
        imports = extract_imports(current)

        # Resolve imports to files
        for import_name in imports:
            # Only trace TELOS-specific imports
            if import_name in ["telos_purpose", "telos_observatory_v3", "telos_privacy"]:
                resolved_files = resolve_import_to_file(import_name, base_dir)
                for resolved in resolved_files:
                    if resolved not in visited:
                        to_visit.add(resolved)

    return dependencies


def main():
    """Main extraction logic."""
    print("\n" + "="*70)
    print("🔍 TELOS Essential File Extractor")
    print("="*70 + "\n")

    # Collect all essential files
    essential_files: Set[Path] = set()

    print("📍 Tracing dependencies from entry points...")
    for entry_point_str in ENTRY_POINTS:
        entry_path = PROJECT_ROOT / entry_point_str

        if not entry_path.exists():
            print(f"⚠️  Entry point not found: {entry_point_str}")
            continue

        print(f"  → {entry_point_str}")
        deps = trace_dependencies(entry_path, PROJECT_ROOT)
        essential_files.update(deps)

    print(f"\n✅ Found {len(essential_files)} files from dependency tracing\n")

    # Add force-include files
    print("📌 Adding force-include files...")
    for force_file_str in FORCE_INCLUDE:
        force_path = PROJECT_ROOT / force_file_str
        if force_path.exists():
            essential_files.add(force_path)
            print(f"  → {force_file_str}")

    # Filter out excluded files
    essential_files = {f for f in essential_files if not should_exclude(str(f))}

    print(f"\n✅ Total essential files: {len(essential_files)}\n")

    # Group by directory for display
    by_dir: Dict[str, List[Path]] = {}
    for file_path in sorted(essential_files):
        rel_path = file_path.relative_to(PROJECT_ROOT)
        dir_name = str(rel_path.parent) if rel_path.parent != Path('.') else 'root'

        if dir_name not in by_dir:
            by_dir[dir_name] = []
        by_dir[dir_name].append(rel_path)

    print("📂 Essential files by directory:\n")
    for dir_name, files in sorted(by_dir.items()):
        print(f"  {dir_name}/ ({len(files)} files)")
        for file_path in sorted(files):
            print(f"    - {file_path.name}")

    # Ask for confirmation before copying (unless --yes flag)
    auto_confirm = '--yes' in sys.argv

    if not auto_confirm:
        print("\n" + "="*70)
        response = input("\n✋ Copy these files to opus_review_package/? (y/n): ")

        if response.lower() != 'y':
            print("❌ Cancelled.")
            return 1
    else:
        print("\n" + "="*70)
        print("✅ Auto-confirming (--yes flag provided)")

    # Clean and recreate opus_review_package
    output_dir = PROJECT_ROOT / "opus_review_package"
    if output_dir.exists():
        print(f"\n🗑️  Cleaning existing {output_dir}...")
        shutil.rmtree(output_dir)

    output_dir.mkdir()
    print(f"✅ Created clean {output_dir}\n")

    # Copy files maintaining structure
    print("📋 Copying files...\n")
    for file_path in sorted(essential_files):
        rel_path = file_path.relative_to(PROJECT_ROOT)
        dest_path = output_dir / rel_path

        # Create parent directories
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Copy file
        shutil.copy2(file_path, dest_path)
        print(f"  ✓ {rel_path}")

    # Create README
    readme_path = output_dir / "README_FOR_OPUS.md"
    with open(readme_path, 'w') as f:
        f.write(f"""# TELOS Codebase - Opus Review Package

**Date:** November 2025
**Total Files:** {len(essential_files)} Python files
**Purpose:** Essential codebase audit (dependencies traced from entry points)

---

## Extraction Method

This package contains ONLY the files used in canonical TELOS execution:
1. Started from main entry points (Observatory, Core Runtime, Steward)
2. Traced import dependencies recursively
3. Excluded test files, demos, validation results, and utilities
4. Force-included critical files (math, config, adapters)

**Result:** Lean, auditable codebase with no cruft.

---

## Directory Structure

""")

        for dir_name, files in sorted(by_dir.items()):
            f.write(f"\n### {dir_name}/ ({len(files)} files)\n\n")
            for file_path in sorted(files):
                f.write(f"- `{file_path}`\n")

        f.write("""
---

## Review Objectives

1. **Code Quality & Architecture**
   - Clean abstractions and maintainability
   - Separation of concerns
   - Architectural improvements

2. **Security & Privacy**
   - Cryptographic implementation soundness
   - Privacy boundary enforcement
   - Vulnerability identification

3. **Mathematical Correctness**
   - Dual attractor implementation
   - Proportional control functionality
   - Fidelity calculation accuracy

4. **Observatory Functionality**
   - UI wiring to runtime
   - Telemetry capture and display
   - UX improvements

5. **Production Readiness**
   - Hardening requirements
   - Performance bottlenecks
   - Missing documentation

---

**For questions or clarifications:**
TELOS Labs
telos.steward@gmail.com
""")

    print(f"\n✅ Created {readme_path}\n")
    print("="*70)
    print(f"✅ Successfully extracted {len(essential_files)} essential files")
    print(f"📁 Output: {output_dir}")
    print("="*70 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
