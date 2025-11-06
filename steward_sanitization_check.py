#!/usr/bin/env python3
"""
Steward PM - Sanitization Check
Scans files/text for proprietary terms before public repo commits
"""

import sys
import re
from pathlib import Path
from typing import List, Tuple, Dict

# PROPRIETARY TERMS - DO NOT PUBLISH
PROPRIETARY_TERMS = {
    # Core innovations
    "dual attractor": "HIGH",
    "dual pa": "HIGH",
    "ai pa": "HIGH",
    "lock-on derivation": "HIGH",
    "lock-on formula": "HIGH",

    # Methodological innovations
    "dmaic for ai": "HIGH",
    "dmaic cycle": "MEDIUM",
    "spc for ai": "HIGH",
    "statistical process control for ai": "HIGH",
    "process capability analysis": "MEDIUM",
    "cpk for governance": "HIGH",

    # Proprietary systems
    "telemetric keys": "HIGH",
    "originmind": "HIGH",
    "federated governance delta": "HIGH",
    "progressive pa extractor": "HIGH",
    "progressive pa": "MEDIUM",

    # Implementation details
    "adaptive weighting": "MEDIUM",
    "proportional intervention": "MEDIUM",
    "variance tracking": "LOW",
    "centroid tracking": "LOW",

    # Business/strategy
    "gmu partnership": "MEDIUM",
    "trail of bits": "MEDIUM",
    "ltff": "LOW",
    "emergent ventures": "LOW",

    # Research specifics
    "+85.32%": "HIGH",  # Specific performance numbers
    "+85%": "HIGH",
    "45+ studies": "MEDIUM",
    "phase 2b validation": "MEDIUM",
}

# ALLOWED GENERIC TERMS
ALLOWED_TERMS = {
    "fidelity measurement": "Generic concept",
    "primacy attractor": "Published term (can mention generically)",
    "cosine similarity": "Standard mathematical operation",
    "embeddings": "Standard NLP technique",
    "governance": "Generic term",
    "alignment": "Generic term",
    "drift detection": "Generic concept",
}

def scan_text(text: str, filename: str = "unknown") -> List[Tuple[str, str, int]]:
    """
    Scan text for proprietary terms.
    Returns: List of (term, severity, line_number)
    """
    findings = []
    lines = text.split('\n')

    for line_num, line in enumerate(lines, 1):
        line_lower = line.lower()

        for term, severity in PROPRIETARY_TERMS.items():
            if term.lower() in line_lower:
                findings.append((term, severity, line_num))

    return findings

def scan_file(filepath: Path) -> Dict:
    """Scan a single file for proprietary terms."""
    try:
        text = filepath.read_text()
        findings = scan_text(text, filepath.name)

        return {
            'filepath': str(filepath),
            'findings': findings,
            'status': 'BLOCK' if any(sev == 'HIGH' for _, sev, _ in findings) else 'WARN' if findings else 'CLEAN'
        }
    except Exception as e:
        return {
            'filepath': str(filepath),
            'error': str(e),
            'status': 'ERROR'
        }

def scan_directory(directory: Path, extensions: List[str] = ['.py', '.md', '.txt', '.json']) -> Dict:
    """Scan directory for files containing proprietary terms."""
    results = {
        'blocked': [],
        'warnings': [],
        'clean': [],
        'errors': []
    }

    for ext in extensions:
        for filepath in directory.rglob(f'*{ext}'):
            # Skip certain directories
            if any(skip in str(filepath) for skip in ['.git', '__pycache__', 'venv', 'node_modules', '.telos', 'beta_consents']):
                continue

            result = scan_file(filepath)

            if result['status'] == 'BLOCK':
                results['blocked'].append(result)
            elif result['status'] == 'WARN':
                results['warnings'].append(result)
            elif result['status'] == 'CLEAN':
                results['clean'].append(result)
            else:
                results['errors'].append(result)

    return results

def display_report(results: Dict):
    """Display sanitization scan report."""
    print("\n" + "="*70)
    print("🔒 STEWARD PM - SANITIZATION SCAN")
    print("="*70)

    # Blocked files
    if results['blocked']:
        print(f"\n🚨 BLOCKED - {len(results['blocked'])} files contain HIGH-severity proprietary terms:")
        print("-"*70)
        for item in results['blocked']:
            print(f"\n❌ {item['filepath']}")
            for term, severity, line_num in item['findings']:
                if severity == 'HIGH':
                    print(f"   Line {line_num}: '{term}' [{severity}]")

    # Warnings
    if results['warnings']:
        print(f"\n⚠️  WARNING - {len(results['warnings'])} files contain MEDIUM/LOW proprietary terms:")
        print("-"*70)
        for item in results['warnings']:
            print(f"\n⚠️  {item['filepath']}")
            for term, severity, line_num in item['findings']:
                print(f"   Line {line_num}: '{term}' [{severity}]")

    # Clean files
    if results['clean']:
        print(f"\n✅ CLEAN - {len(results['clean'])} files scanned, no proprietary terms found")

    # Errors
    if results['errors']:
        print(f"\n❌ ERRORS - {len(results['errors'])} files could not be scanned:")
        for item in results['errors']:
            print(f"   {item['filepath']}: {item.get('error', 'Unknown error')}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY:")
    print(f"  Blocked: {len(results['blocked'])}")
    print(f"  Warnings: {len(results['warnings'])}")
    print(f"  Clean: {len(results['clean'])}")
    print(f"  Errors: {len(results['errors'])}")

    if results['blocked']:
        print("\n🚨 COMMIT BLOCKED - Remove or sanitize HIGH-severity terms before publishing")
        print("="*70)
        return False
    elif results['warnings']:
        print("\n⚠️  PROCEED WITH CAUTION - Review warnings manually")
        print("="*70)
        return True
    else:
        print("\n✅ SAFE TO PUBLISH")
        print("="*70)
        return True

def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Steward PM Sanitization Check')
    parser.add_argument('path', type=str, help='File or directory to scan')
    parser.add_argument('--strict', action='store_true', help='Block on warnings too')
    parser.add_argument('--quiet', action='store_true', help='Only show summary')

    args = parser.parse_args()

    path = Path(args.path)

    if not path.exists():
        print(f"❌ Error: Path does not exist: {path}")
        return 1

    # Scan
    if path.is_file():
        result = scan_file(path)
        results = {
            'blocked': [result] if result['status'] == 'BLOCK' else [],
            'warnings': [result] if result['status'] == 'WARN' else [],
            'clean': [result] if result['status'] == 'CLEAN' else [],
            'errors': [result] if result['status'] == 'ERROR' else []
        }
    else:
        results = scan_directory(path)

    # Display report
    if not args.quiet:
        safe = display_report(results)
    else:
        safe = len(results['blocked']) == 0 and (not args.strict or len(results['warnings']) == 0)
        if safe:
            print("✅ SAFE TO PUBLISH")
        else:
            print("🚨 BLOCKED")

    # Exit code
    if not safe or (args.strict and results['warnings']):
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
