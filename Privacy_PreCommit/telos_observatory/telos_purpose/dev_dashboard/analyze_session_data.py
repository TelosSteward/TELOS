"""
Session Data Analysis Tool
==========================

Analyzes exported TELOS session data and creates publication-quality plots.

Usage:
    python analyze_session_data.py session_data.json
    python analyze_session_data.py --compare session1.json session2.json session3.json
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any
import argparse

try:
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Rectangle
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("⚠️  matplotlib not installed. Install with: pip install matplotlib")


def load_session_data(filepath: str) -> Dict[str, Any]:
    """Load session data from JSON file."""
    with open(filepath) as f:
        return json.load(f)


def plot_single_session(data: Dict[str, Any], output_dir: str = "plots"):
    """Create comprehensive plots for a single session."""
    if not HAS_MATPLOTLIB:
        print("Cannot create plots without matplotlib")
        return

    Path(output_dir).mkdir(exist_ok=True)

    session_name = data.get('session_id', 'session')
    metrics = data['metrics']

    # Extract turn numbers
    turns = list(range(1, len(metrics['fidelity_history']) + 1))

    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'TELOS Session Analysis: {session_name}', fontsize=16, fontweight='bold')

    # ===== Plot 1: Telic Fidelity =====
    ax = axes[0, 0]
    ax.plot(turns, metrics['fidelity_history'], 'b-o', linewidth=2, markersize=8, label='Fidelity')

    # Mark interventions
    if 'intervention_log' in data:
        for intervention in data['intervention_log']:
            turn = intervention['turn']
            fidelity = metrics['fidelity_history'][turn - 1]
            ax.plot(turn, fidelity, 'rx', markersize=15, markeredgewidth=3, label='Intervention')

    # Goldilocks threshold zones
    ax.axhspan(0.76, 1.0, alpha=0.1, color='green', label='Aligned (F≥0.76)')
    ax.axhspan(0.73, 0.76, alpha=0.1, color='gold', label='Minor Drift (0.73-0.76)')
    ax.axhspan(0.67, 0.73, alpha=0.1, color='orange', label='Drift Detected (0.67-0.73)')
    ax.axhspan(0.0, 0.67, alpha=0.1, color='red', label='Significant Drift (F<0.67)')

    ax.axhline(y=0.76, color='green', linestyle='--', linewidth=1, alpha=0.7)
    ax.axhline(y=0.73, color='gold', linestyle='--', linewidth=1, alpha=0.7)
    ax.axhline(y=0.67, color='red', linestyle='--', linewidth=1, alpha=0.7)

    ax.set_xlabel('Turn', fontsize=12, fontweight='bold')
    ax.set_ylabel('Telic Fidelity (F)', fontsize=12, fontweight='bold')
    ax.set_title('Telic Fidelity Over Time', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

    # ===== Plot 2: Error Signal =====
    ax = axes[0, 1]
    ax.plot(turns, metrics['error_signal_history'], 'r-o', linewidth=2, markersize=8, label='Error Signal (ε)')

    # Get thresholds from config if available (Goldilocks defaults)
    epsilon_min = 0.24  # Goldilocks: 1 - 0.76 (Aligned threshold)
    epsilon_max = 0.33  # Goldilocks: 1 - 0.67 (Significant Drift threshold)
    if 'config' in data and 'intervention_thresholds' in data['config']:
        epsilon_min = data['config']['intervention_thresholds'].get('epsilon_min', 0.24)
        epsilon_max = data['config']['intervention_thresholds'].get('epsilon_max', 0.33)

    ax.axhspan(epsilon_min, epsilon_max, alpha=0.15, color='orange', label=f'Warning (ε > {epsilon_min})')
    ax.axhspan(epsilon_max, 1.0, alpha=0.15, color='red', label=f'Critical (ε > {epsilon_max})')

    ax.axhline(y=epsilon_min, color='orange', linestyle='--', linewidth=2, label=f'ε_min = {epsilon_min}')
    ax.axhline(y=epsilon_max, color='red', linestyle='--', linewidth=2, label=f'ε_max = {epsilon_max}')

    ax.set_xlabel('Turn', fontsize=12, fontweight='bold')
    ax.set_ylabel('Error Signal (ε)', fontsize=12, fontweight='bold')
    ax.set_title('Error Signal & Intervention Thresholds', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    ax.legend(fontsize=9, loc='upper right')

    # ===== Plot 3: Lyapunov Function =====
    ax = axes[1, 0]
    ax.fill_between(turns, 0, metrics['lyapunov_history'], alpha=0.3, color='purple', label='V(x)')
    ax.plot(turns, metrics['lyapunov_history'], 'purple', linewidth=2, marker='o', markersize=8)

    # Basin boundary (approximate as r² where r ≈ 1.2)
    basin_threshold = 1.44  # r² for typical basin radius
    ax.axhline(y=basin_threshold, color='darkviolet', linestyle='--', linewidth=2,
               label=f'Basin Boundary (r² ≈ {basin_threshold:.2f})')

    ax.set_xlabel('Turn', fontsize=12, fontweight='bold')
    ax.set_ylabel('Lyapunov Value V(x)', fontsize=12, fontweight='bold')
    ax.set_title('Lyapunov Function (System Energy)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    # ===== Plot 4: Basin Membership =====
    ax = axes[1, 1]

    # Convert boolean to numeric
    basin_numeric = [1 if b else 0 for b in metrics['basin_membership_history']]

    # Create bar chart
    colors = ['green' if b else 'red' for b in metrics['basin_membership_history']]
    ax.bar(turns, basin_numeric, color=colors, alpha=0.6, edgecolor='black')

    # Add text labels
    for i, (turn, val) in enumerate(zip(turns, basin_numeric)):
        label = '✓ Inside' if val == 1 else '✗ Outside'
        ax.text(turn, val + 0.05, label, ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xlabel('Turn', fontsize=12, fontweight='bold')
    ax.set_ylabel('Basin Membership', fontsize=12, fontweight='bold')
    ax.set_title('Primacy Basin Membership', fontsize=13, fontweight='bold')
    ax.set_ylim([0, 1.2])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Outside', 'Inside'])
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Save
    output_path = Path(output_dir) / f'{session_name}_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved plot: {output_path}")

    plt.close()


def plot_comparison(sessions: List[Dict[str, Any]], output_dir: str = "plots"):
    """Create comparison plot for multiple sessions."""
    if not HAS_MATPLOTLIB:
        print("Cannot create plots without matplotlib")
        return

    Path(output_dir).mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('TELOS Multi-Session Comparison', fontsize=16, fontweight='bold')

    # ===== Plot 1: Fidelity Comparison =====
    ax = axes[0]

    for i, session in enumerate(sessions):
        session_name = session.get('session_id', f'Session {i+1}')
        metrics = session['metrics']
        turns = list(range(1, len(metrics['fidelity_history']) + 1))

        ax.plot(turns, metrics['fidelity_history'], '-o', linewidth=2,
                markersize=6, label=session_name, alpha=0.8)

    ax.axhline(y=0.76, color='green', linestyle='--', linewidth=1, alpha=0.5)
    ax.axhline(y=0.73, color='gold', linestyle='--', linewidth=1, alpha=0.5)
    ax.axhline(y=0.67, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.axhspan(0.76, 1.0, alpha=0.05, color='green')
    ax.axhspan(0.73, 0.76, alpha=0.05, color='gold')
    ax.axhspan(0.67, 0.73, alpha=0.05, color='orange')

    ax.set_xlabel('Turn', fontsize=12, fontweight='bold')
    ax.set_ylabel('Telic Fidelity (F)', fontsize=12, fontweight='bold')
    ax.set_title('Fidelity Comparison', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

    # ===== Plot 2: Summary Statistics =====
    ax = axes[1]

    session_names = []
    avg_fidelities = []
    intervention_rates = []
    basin_times = []

    for i, session in enumerate(sessions):
        session_name = session.get('session_id', f'Session {i+1}')
        summary = session.get('summary', {})

        session_names.append(session_name[:20])  # Truncate long names
        avg_fidelities.append(summary.get('avg_fidelity', 0))
        intervention_rates.append(summary.get('intervention_rate', 0) * 100)
        basin_times.append(summary.get('basin_time', 0) * 100)

    x = np.arange(len(session_names))
    width = 0.25

    ax.bar(x - width, avg_fidelities, width, label='Avg Fidelity', alpha=0.8, color='blue')
    ax.bar(x, intervention_rates, width, label='Intervention Rate (%)', alpha=0.8, color='red')
    ax.bar(x + width, basin_times, width, label='Basin Time (%)', alpha=0.8, color='green')

    ax.set_xlabel('Session', fontsize=12, fontweight='bold')
    ax.set_ylabel('Value', fontsize=12, fontweight='bold')
    ax.set_title('Summary Statistics', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(session_names, rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 105])

    plt.tight_layout()

    # Save
    output_path = Path(output_dir) / 'session_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved comparison plot: {output_path}")

    plt.close()


def print_summary(data: Dict[str, Any]):
    """Print text summary of session."""
    print(f"\n{'='*70}")
    print("📊 Session Summary")
    print(f"{'='*70}")

    print(f"Session ID: {data.get('session_id', 'N/A')}")
    print(f"Timestamp: {data.get('timestamp', 'N/A')}")
    print(f"Total Turns: {data.get('total_turns', 0)}")
    print()

    if 'summary' in data:
        s = data['summary']
        print(f"Average Fidelity: {s.get('avg_fidelity', 0):.3f}")
        print(f"Final Fidelity: {s.get('final_fidelity', 0):.3f}")
        print(f"Fidelity Range: {s.get('min_fidelity', 0):.3f} - {s.get('max_fidelity', 0):.3f}")
        print()
        print(f"Interventions: {s.get('intervention_count', 0)} ({s.get('intervention_rate', 0)*100:.1f}%)")
        print(f"Time in Basin: {s.get('basin_time', 0)*100:.1f}%")
        print(f"Total Time: {s.get('total_time_seconds', 0):.1f}s")

    if 'intervention_log' in data and data['intervention_log']:
        print(f"\n{'='*70}")
        print("⚡ Interventions")
        print(f"{'='*70}")

        for intervention in data['intervention_log']:
            print(f"\nTurn {intervention['turn']}:")
            print(f"  Type: {intervention['type']}")
            print(f"  Error Signal: {intervention['error_signal']:.3f}")
            print(f"  Fidelity: {intervention['fidelity_before']:.3f} → {intervention['fidelity_after']:.3f}")
            print(f"  Reason: {intervention['reason'][:100]}...")

    print(f"\n{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description='Analyze TELOS session data')
    parser.add_argument('files', nargs='+', help='Session data JSON file(s)')
    parser.add_argument('--output-dir', default='plots', help='Output directory for plots')
    parser.add_argument('--compare', action='store_true',
                       help='Create comparison plot for multiple sessions')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip creating plots (text summary only)')

    args = parser.parse_args()

    # Load all sessions
    sessions = []
    for filepath in args.files:
        try:
            data = load_session_data(filepath)
            sessions.append(data)
            print(f"✅ Loaded: {filepath}")
        except Exception as e:
            print(f"❌ Error loading {filepath}: {e}")
            continue

    if not sessions:
        print("No valid session files loaded")
        sys.exit(1)

    # Print summaries
    for data in sessions:
        print_summary(data)

    # Create plots
    if not args.no_plots and HAS_MATPLOTLIB:
        for data in sessions:
            plot_single_session(data, args.output_dir)

        if args.compare and len(sessions) > 1:
            plot_comparison(sessions, args.output_dir)

    print(f"\n✅ Analysis complete! Plots saved to: {args.output_dir}/")


if __name__ == '__main__':
    main()
