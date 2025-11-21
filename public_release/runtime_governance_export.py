#!/usr/bin/env python3
"""
Runtime Governance - Session Export
Exports session data for analysis, dashboards, or reports
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

def load_session():
    """Load active session info"""
    session_file = Path('.runtime_governance_session.json')
    if not session_file.exists():
        print("❌ Error: No active session found.")
        sys.exit(1)

    with open(session_file, 'r') as f:
        return json.load(f)

def load_checkpoints(session_info):
    """Load all checkpoint files for this session"""
    turn_count = session_info.get('turn_count', 0)
    checkpoints = []

    for i in range(1, turn_count + 1):
        checkpoint_file = Path(f'.runtime_governance_checkpoint_{i}.json')
        if checkpoint_file.exists():
            with open(checkpoint_file, 'r') as f:
                checkpoints.append(json.load(f))

    return checkpoints

def export_standard(session_info, checkpoints, output_file):
    """Export in standard JSON format"""
    export_data = {
        'session': session_info,
        'checkpoints': checkpoints,
        'summary': {
            'total_turns': len(checkpoints),
            'mean_fidelity': sum(c['fidelity'] for c in checkpoints) / len(checkpoints) if checkpoints else 0,
            'on_track_count': sum(1 for c in checkpoints if c['status'] == 'on_track'),
            'warning_count': sum(1 for c in checkpoints if c['status'] == 'warning'),
            'drift_count': sum(1 for c in checkpoints if c['status'] == 'drift')
        },
        'exported_at': datetime.now().isoformat()
    }

    with open(output_file, 'w') as f:
        json.dump(export_data, f, indent=2)

    print(f"✅ Exported to: {output_file}")
    return export_data

def export_dashboard(session_info, checkpoints, output_dir):
    """Export in dashboard-ready CSV + summary format"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # CSV for plotting
    csv_file = output_dir / 'fidelity_curve.csv'
    with open(csv_file, 'w') as f:
        f.write('turn,fidelity,status,timestamp\n')
        for cp in checkpoints:
            f.write(f"{cp['turn_number']},{cp['fidelity']},{cp['status']},{cp['timestamp']}\n")

    # Summary stats
    summary_file = output_dir / 'summary.json'
    summary = {
        'session_id': session_info['session_id'],
        'session_name': session_info['session_name'],
        'started_at': session_info['started_at'],
        'total_turns': len(checkpoints),
        'mean_fidelity': sum(c['fidelity'] for c in checkpoints) / len(checkpoints) if checkpoints else 0,
        'min_fidelity': min(c['fidelity'] for c in checkpoints) if checkpoints else 0,
        'max_fidelity': max(c['fidelity'] for c in checkpoints) if checkpoints else 0,
        'on_track_percentage': (sum(1 for c in checkpoints if c['status'] == 'on_track') / len(checkpoints) * 100) if checkpoints else 0,
        'drift_events': [c['turn_number'] for c in checkpoints if c['status'] == 'drift']
    }

    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"✅ Dashboard export complete:")
    print(f"   CSV: {csv_file}")
    print(f"   Summary: {summary_file}")

    return summary

def export_grant_report(session_info, checkpoints, output_file):
    """Export formatted report for grant applications"""
    if not checkpoints:
        print("⚠️  No turns to report")
        return

    mean_fidelity = sum(c['fidelity'] for c in checkpoints) / len(checkpoints)
    on_track_pct = (sum(1 for c in checkpoints if c['status'] == 'on_track') / len(checkpoints) * 100)

    report = f"""# Runtime Governance Report

**Session:** {session_info['session_name']}
**Period:** {session_info['started_at']} to {datetime.now().isoformat()}
**Total Turns:** {len(checkpoints)}

## Governance Metrics

- **Mean Fidelity:** {mean_fidelity:.3f}
- **On-Track Rate:** {on_track_pct:.1f}%
- **Total Drift Events:** {sum(1 for c in checkpoints if c['status'] == 'drift')}

## Fidelity Distribution

- ✅ On Track (F ≥ 0.8): {sum(1 for c in checkpoints if c['status'] == 'on_track')} turns ({sum(1 for c in checkpoints if c['status'] == 'on_track')/len(checkpoints)*100:.1f}%)
- ⚠️ Warning (0.7 ≤ F < 0.8): {sum(1 for c in checkpoints if c['status'] == 'warning')} turns ({sum(1 for c in checkpoints if c['status'] == 'warning')/len(checkpoints)*100:.1f}%)
- 🚨 Drift (F < 0.7): {sum(1 for c in checkpoints if c['status'] == 'drift')} turns ({sum(1 for c in checkpoints if c['status'] == 'drift')/len(checkpoints)*100:.1f}%)

## PA Baseline

{session_info['pa_baseline']}

## Summary

This session demonstrates systematic governance of AI-assisted development using Runtime Governance. Every conversation turn was measured against the established Primacy Attractor using mathematical fidelity analysis (embeddings in ℝ³⁸⁴, cosine similarity).

The {mean_fidelity:.3f} mean fidelity and {on_track_pct:.1f}% on-track rate indicate strong alignment between AI assistance and project objectives.

---

*Generated by Runtime Governance*
*Export date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""

    with open(output_file, 'w') as f:
        f.write(report)

    print(f"✅ Grant report exported to: {output_file}")
    return report

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Runtime Governance Session Export')
    parser.add_argument('--format', choices=['standard', 'dashboard', 'grant'], default='standard',
                        help='Export format')
    parser.add_argument('--output', type=str, help='Output file/directory path')

    args = parser.parse_args()

    # Load session data
    session_info = load_session()
    checkpoints = load_checkpoints(session_info)

    print(f"\n📊 Runtime Governance - Session Export")
    print("="*60)
    print(f"Session: {session_info['session_name']}")
    print(f"Turns: {len(checkpoints)}")

    if not checkpoints:
        print("\n⚠️  No checkpoints found. Run some turns first.")
        return 1

    # Export based on format
    if args.format == 'standard':
        output_file = args.output or f"session_export_{session_info['session_id']}.json"
        export_standard(session_info, checkpoints, output_file)

    elif args.format == 'dashboard':
        output_dir = args.output or 'dashboard_export'
        export_dashboard(session_info, checkpoints, output_dir)

    elif args.format == 'grant':
        output_file = args.output or f"governance_report_{session_info['session_id']}.md"
        export_grant_report(session_info, checkpoints, output_file)

    print("\n✅ Export complete")

    return 0

if __name__ == "__main__":
    sys.exit(main())
