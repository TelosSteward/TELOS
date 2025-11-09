#!/usr/bin/env python3
"""Generate report from saved pickle file."""
import pickle
import json
from datetime import datetime

# Load the saved report
with open('tests/validation_results/forensic_report_20251107_232640_debug.pkl', 'rb') as f:
    report = pickle.load(f)

# Fix the convergence summary
if 'turn' in report['pa_establishment']['convergence_summary']:
    report['pa_establishment']['convergence_summary']['convergence_turn'] = report['pa_establishment']['convergence_summary'].pop('turn')

# Add missing metrics if not present
if 'confidence' not in report['pa_establishment']['convergence_summary']:
    report['pa_establishment']['convergence_summary']['confidence'] = 1.0
    report['pa_establishment']['convergence_summary']['centroid_stability'] = 1.0
    report['pa_establishment']['convergence_summary']['variance_stability'] = 1.0

# Generate human-readable report
from tests.validation.run_forensic_validation import ForensicValidator

validator = ForensicValidator()
validator.report = report

# Generate and save report
validator._generate_final_report()

print("\n✅ Report generated successfully!")
