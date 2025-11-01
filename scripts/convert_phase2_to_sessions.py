#!/usr/bin/env python3
"""
Convert Phase 2 validation studies to Observatory V3 saved session format.

This script transforms Phase 2 intervention data into the format expected by
Observatory V3's saved session system, making all 56 validation studies
loadable as interactive sessions.
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_phase2_summary(summary_path: Path) -> dict:
    """Load Phase 2 study summary."""
    with open(summary_path, 'r') as f:
        return json.load(f)


def load_intervention_data(study_dir: Path, conversation_id: str) -> dict:
    """Load intervention JSON for a study."""
    intervention_files = list(study_dir.glob("intervention_*.json"))

    if not intervention_files:
        return None

    # Load the first (usually only) intervention file
    with open(intervention_files[0], 'r') as f:
        return json.load(f)


def convert_study_to_session(study_data: dict, intervention_data: dict) -> dict:
    """Convert Phase 2 study to Observatory V3 session format.

    Args:
        study_data: Study metadata from phase2_study_summary.json
        intervention_data: Intervention data from intervention_*.json

    Returns:
        Session data compatible with Observatory V3
    """
    session_id = study_data['conversation_id']

    # Build turns array combining original and TELOS responses
    turns = []

    if intervention_data:
        # Get trigger turn
        trigger_turn = intervention_data['trigger_turn']

        # The intervention data contains post-intervention turns only
        # Both original and telos arrays should have same length
        orig_turns = intervention_data['original']['turns']
        telos_turns = intervention_data['telos']['turns']

        # Create turns showing both native and TELOS responses
        for i, (orig_turn, telos_turn) in enumerate(zip(orig_turns, telos_turns)):
            turn_data = {
                'turn': telos_turn['turn_number'] - 1,  # 0-indexed for state manager
                'turn_number': telos_turn['turn_number'],
                'timestamp': i * 2.5,  # Simple timestamp
                'user_input': telos_turn['user_input'],  # ConversationDisplay expects 'user_input'
                'response': telos_turn['assistant_response'],  # ConversationDisplay expects 'response'
                'user_message': telos_turn['user_input'],  # Keep for compatibility
                'assistant_response_native': orig_turn['assistant_response'],
                'assistant_response_telos': telos_turn['assistant_response'],
                'fidelity': telos_turn['fidelity'],  # Primary fidelity
                'fidelity_native': orig_turn['fidelity'],
                'fidelity_telos': telos_turn['fidelity'],
                'distance': 1.0 - telos_turn['fidelity'],  # Calculate distance
                'threshold': 0.8,  # Standard threshold
                'intervention_applied': telos_turn.get('intervention_applied', True),
                'drift_detected': telos_turn['fidelity'] < 0.8,
                'status': '✓' if telos_turn['fidelity'] >= 0.8 else '⚠',
                'status_text': 'Good' if telos_turn['fidelity'] >= 0.8 else 'Drift',
                'in_basin': telos_turn['fidelity'] >= 0.8,
                'delta_f': telos_turn['fidelity'] - orig_turn['fidelity'],
                'phase2_comparison': {
                    'native_response': orig_turn['assistant_response'],
                    'telos_response': telos_turn['assistant_response'],
                    'native_fidelity': orig_turn['fidelity'],
                    'telos_fidelity': telos_turn['fidelity'],
                    'improvement': telos_turn['fidelity'] - orig_turn['fidelity']
                }
            }
            turns.append(turn_data)

    # Build session object
    session = {
        'session_id': session_id,
        'name': f"Phase 2: {session_id}",
        'type': 'phase2_validation',
        'created_at': study_data.get('timestamp', datetime.now().isoformat()),
        'metadata': {
            'pa_established': study_data.get('pa_established', False),
            'convergence_turn': study_data.get('convergence_turn', 0),
            'drift_detected': study_data.get('drift_detected', False),
            'total_interventions': 1,  # Phase 2 has single interventions
            'trigger_turn': intervention_data['trigger_turn'] if intervention_data else 0,
            'trigger_fidelity': intervention_data['trigger_fidelity'] if intervention_data else 0,
            'delta_f': study_data.get('delta_f', 0),
            'governance_effective': study_data.get('governance_effective', False),
            'dataset': study_data.get('dataset', 'unknown')
        },
        'primacy_attractor': study_data.get('attractor', {
            'purpose': ['Establish conversation purpose from baseline turns'],
            'scope': ['Topics covered in baseline'],
            'boundaries': ['Off-topic discussions']
        }),
        'turns': turns,
        'total_turns': len(turns),
        'current_turn': len(turns) - 1,  # Start at end
        'statistics': {
            'avg_fidelity_native': intervention_data['original']['avg_fidelity'] if intervention_data else 0,
            'avg_fidelity_telos': intervention_data['telos']['avg_fidelity'] if intervention_data else 0,
            'final_fidelity_native': intervention_data['original']['final_fidelity'] if intervention_data else 0,
            'final_fidelity_telos': intervention_data['telos']['final_fidelity'] if intervention_data else 0,
            'delta_f': study_data.get('delta_f', 0)
        }
    }

    return session


def convert_all_studies(phase2_dir: Path, output_dir: Path):
    """Convert all Phase 2 studies to session format.

    Args:
        phase2_dir: Path to telos_observatory/phase2_study_results/
        output_dir: Path to save converted sessions
    """
    # Load summary
    summary_path = phase2_dir / 'phase2_study_summary.json'
    summary = load_phase2_summary(summary_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    converted_sessions = []
    failed_conversions = []

    print(f"Converting {len(summary['completed_studies'])} Phase 2 studies...")

    for study in summary['completed_studies']:
        conversation_id = study['conversation_id']
        study_dir = phase2_dir / conversation_id

        try:
            # Load intervention data
            intervention_data = load_intervention_data(study_dir, conversation_id)

            if not intervention_data:
                print(f"⚠️  No intervention data for {conversation_id}, skipping")
                failed_conversions.append(conversation_id)
                continue

            # Convert to session format
            session = convert_study_to_session(study, intervention_data)

            # Save to file
            output_file = output_dir / f"{conversation_id}.json"
            with open(output_file, 'w') as f:
                json.dump(session, f, indent=2, ensure_ascii=False)

            converted_sessions.append({
                'id': conversation_id,
                'name': session['name'],
                'date': session['created_at'],
                'type': 'phase2_validation',
                'file': str(output_file)
            })

            print(f"✅ {conversation_id} -> {output_file.name}")

        except Exception as e:
            print(f"❌ Failed to convert {conversation_id}: {e}")
            failed_conversions.append(conversation_id)

    # Save index file
    index_file = output_dir / 'session_index.json'
    with open(index_file, 'w') as f:
        json.dump({
            'total_sessions': len(converted_sessions),
            'sessions': converted_sessions,
            'failed': failed_conversions,
            'created_at': datetime.now().isoformat()
        }, f, indent=2)

    print(f"\n📊 Conversion Summary:")
    print(f"   ✅ Converted: {len(converted_sessions)}")
    print(f"   ❌ Failed: {len(failed_conversions)}")
    print(f"   📁 Output: {output_dir}")
    print(f"   📋 Index: {index_file}")

    return converted_sessions, failed_conversions


def main():
    """Main conversion script."""
    # Paths
    repo_root = Path(__file__).parent.parent
    phase2_dir = repo_root / 'telos_observatory' / 'phase2_study_results'
    output_dir = repo_root / 'telos_observatory_v3' / 'saved_sessions'

    if not phase2_dir.exists():
        print(f"❌ Phase 2 results directory not found: {phase2_dir}")
        return 1

    print(f"🔭 TELOS Phase 2 to Observatory V3 Session Converter")
    print(f"=" * 60)
    print(f"Source: {phase2_dir}")
    print(f"Target: {output_dir}")
    print(f"=" * 60)
    print()

    # Convert
    converted, failed = convert_all_studies(phase2_dir, output_dir)

    if failed:
        print(f"\n⚠️  Failed conversions: {', '.join(failed)}")
        return 1

    print(f"\n🎉 All {len(converted)} studies converted successfully!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
