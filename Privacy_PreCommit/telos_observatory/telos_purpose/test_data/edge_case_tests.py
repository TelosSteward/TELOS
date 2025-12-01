"""
Edge Case Testing Suite for TELOS Observatory
=============================================

Generates comprehensive edge case test sessions to verify platform robustness:
- Empty sessions (0 turns)
- Single turn sessions
- Very long sessions (100+ turns)
- Missing data fields
- Extreme fidelity values
- All/no interventions
- Malformed data structures

Usage:
    python -m telos_purpose.test_data.edge_case_tests
"""

import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any


def generate_empty_session() -> Dict[str, Any]:
    """
    EDGE CASE 1: Empty session (0 turns)

    Tests:
    - Dashboard handles empty turn list
    - Analytics don't crash on division by zero
    - Export functions handle empty data
    """
    return {
        'session_id': 'edge_empty_session_001',
        'timestamp': datetime.now().isoformat(),
        'total_turns': 0,
        'avg_fidelity': None,
        'turns': [],
        'config': {
            'purpose': ['Test empty session handling'],
            'scope': ['Edge case testing'],
            'boundaries': ['No actual conversation']
        }
    }


def generate_single_turn_session() -> Dict[str, Any]:
    """
    EDGE CASE 2: Single turn session

    Tests:
    - Minimum viable session
    - Analytics with n=1
    - Trend analysis with single data point
    """
    return {
        'session_id': 'edge_single_turn_001',
        'timestamp': datetime.now().isoformat(),
        'total_turns': 1,
        'avg_fidelity': 0.85,
        'turns': [{
            'turn_number': 1,
            'user_message': 'Single message test',
            'native_response': 'Single native response',
            'telos_response': 'Single TELOS response',
            'assistant_response': 'Single TELOS response',
            'fidelity': 0.85,
            'intervention_applied': False,
            'basin_membership': True,
            'timestamp': datetime.now().isoformat()
        }],
        'config': {
            'purpose': ['Test single turn handling'],
            'scope': ['Minimum session'],
            'boundaries': []
        }
    }


def generate_very_long_session() -> Dict[str, Any]:
    """
    EDGE CASE 3: Very long session (100 turns)

    Tests:
    - Performance with large datasets
    - Memory handling
    - UI responsiveness with many turns
    - Visualization scaling
    """
    turns = []
    base_fidelity = 0.85

    for i in range(1, 101):
        # Simulate gradual drift with occasional recoveries
        drift = -0.002 * i + (0.15 if i % 20 == 0 else 0)
        fidelity = max(0.3, min(0.95, base_fidelity + drift))

        turns.append({
            'turn_number': i,
            'user_message': f'Long session message {i}',
            'native_response': f'Native response {i}',
            'telos_response': f'TELOS response {i}',
            'assistant_response': f'TELOS response {i}',
            'fidelity': round(fidelity, 3),
            'intervention_applied': i % 10 == 0 and fidelity < 0.67,  # Goldilocks: Drift threshold
            'basin_membership': fidelity >= 0.76,  # Goldilocks: Aligned threshold
            'timestamp': datetime.now().isoformat()
        })

    return {
        'session_id': 'edge_very_long_session_001',
        'timestamp': datetime.now().isoformat(),
        'total_turns': 100,
        'avg_fidelity': round(sum(t['fidelity'] for t in turns) / 100, 3),
        'turns': turns,
        'config': {
            'purpose': ['Test performance with long sessions'],
            'scope': ['Scalability testing'],
            'boundaries': []
        }
    }


def generate_missing_fidelity_session() -> Dict[str, Any]:
    """
    EDGE CASE 4: Missing fidelity scores

    Tests:
    - Graceful handling of None values
    - Analytics with missing data
    - Display of incomplete metrics
    """
    return {
        'session_id': 'edge_missing_fidelity_001',
        'timestamp': datetime.now().isoformat(),
        'total_turns': 5,
        'avg_fidelity': None,
        'turns': [{
            'turn_number': i,
            'user_message': f'Message {i}',
            'native_response': f'Native response {i}',
            'telos_response': f'TELOS response {i}',
            'assistant_response': f'TELOS response {i}',
            'fidelity': None,  # Missing fidelity!
            'intervention_applied': False,
            'basin_membership': None,
            'timestamp': datetime.now().isoformat()
        } for i in range(1, 6)],
        'config': {
            'purpose': ['Test missing data handling'],
            'scope': ['Edge case testing'],
            'boundaries': []
        }
    }


def generate_extreme_fidelity_session() -> Dict[str, Any]:
    """
    EDGE CASE 5: Extreme fidelity values (boundary testing)

    Tests:
    - Min/max fidelity values (0.0, 1.0)
    - Values at thresholds (0.76 Goldilocks aligned boundary)
    - Very precise values (0.999, 0.001)
    """
    turns = [
        {
            'turn_number': 1,
            'fidelity': 0.0,
            'intervention_applied': True,
            'basin_membership': False,
            'user_message': 'Complete failure case',
            'native_response': 'Completely off-topic response',
            'telos_response': 'Regenerated on-topic response',
            'timestamp': datetime.now().isoformat()
        },
        {
            'turn_number': 2,
            'fidelity': 1.0,
            'intervention_applied': False,
            'basin_membership': True,
            'user_message': 'Perfect alignment case',
            'native_response': 'Perfectly aligned response',
            'telos_response': 'Perfectly aligned response',
            'timestamp': datetime.now().isoformat()
        },
        {
            'turn_number': 3,
            'fidelity': 0.5,
            'intervention_applied': True,
            'basin_membership': False,
            'user_message': 'Middle case',
            'native_response': 'Mediocre response',
            'telos_response': 'Improved response',
            'timestamp': datetime.now().isoformat()
        },
        {
            'turn_number': 4,
            'fidelity': 0.999,
            'intervention_applied': False,
            'basin_membership': True,
            'user_message': 'Near-perfect case',
            'native_response': 'Almost perfect response',
            'telos_response': 'Almost perfect response',
            'timestamp': datetime.now().isoformat()
        },
        {
            'turn_number': 5,
            'fidelity': 0.001,
            'intervention_applied': True,
            'basin_membership': False,
            'user_message': 'Near-zero case',
            'native_response': 'Terrible response',
            'telos_response': 'Heavily corrected response',
            'timestamp': datetime.now().isoformat()
        }
    ]

    # Add required fields to all turns
    for turn in turns:
        turn['assistant_response'] = turn['telos_response']

    return {
        'session_id': 'edge_extreme_fidelity_001',
        'timestamp': datetime.now().isoformat(),
        'total_turns': 5,
        'avg_fidelity': 0.5,
        'turns': turns,
        'config': {
            'purpose': ['Test extreme value handling'],
            'scope': ['Boundary testing'],
            'boundaries': []
        }
    }


def generate_missing_fields_session() -> Dict[str, Any]:
    """
    EDGE CASE 6: Missing required fields

    Tests:
    - Partial data structures
    - Missing critical fields
    - Fallback behavior
    """
    return {
        'session_id': 'edge_missing_fields_001',
        # Missing timestamp!
        'total_turns': 3,
        'turns': [
            {
                'turn_number': 1,
                # Missing almost all fields!
            },
            {
                'turn_number': 2,
                'fidelity': 0.8,
                # Missing user_input, responses, etc.
            },
            {
                'turn_number': 3,
                'user_input': 'Partial message',
                'fidelity': 0.75,
                # Missing responses
            }
        ]
    }


def generate_all_interventions_session() -> Dict[str, Any]:
    """
    EDGE CASE 7: Maximum governance (all interventions)

    Tests:
    - 100% intervention rate
    - Continuous governance activity
    - Performance under heavy governance
    """
    intervention_types = ['salience_check', 'coupling_check', 'regeneration']

    turns = []
    for i in range(1, 11):
        turns.append({
            'turn_number': i,
            'user_message': f'Message requiring intervention {i}',
            'native_response': f'Problematic native response {i}',
            'telos_response': f'Corrected TELOS response {i}',
            'assistant_response': f'Corrected TELOS response {i}',
            'fidelity': 0.65,
            'intervention_applied': True,
            'basin_membership': False,
            'timestamp': datetime.now().isoformat(),
            'governance_metadata': {
                'intervention_type': intervention_types[i % 3],
                'fidelity_original': 0.55,
                'fidelity_governed': 0.65,
                'intervention_applied': True
            }
        })

    return {
        'session_id': 'edge_all_interventions_001',
        'timestamp': datetime.now().isoformat(),
        'total_turns': 10,
        'avg_fidelity': 0.65,
        'total_interventions': 10,
        'intervention_rate': 1.0,
        'turns': turns,
        'config': {
            'purpose': ['Test maximum governance'],
            'scope': ['High intervention scenario'],
            'boundaries': []
        }
    }


def generate_no_interventions_session() -> Dict[str, Any]:
    """
    EDGE CASE 8: Zero governance (no interventions)

    Tests:
    - 0% intervention rate
    - Pure observation mode
    - Baseline behavior
    """
    turns = []
    for i in range(1, 16):
        turns.append({
            'turn_number': i,
            'user_message': f'Well-aligned message {i}',
            'native_response': f'Already good response {i}',
            'telos_response': f'Already good response {i}',
            'assistant_response': f'Already good response {i}',
            'fidelity': 0.92,
            'intervention_applied': False,
            'basin_membership': True,
            'timestamp': datetime.now().isoformat(),
            'governance_metadata': {
                'intervention_type': None,
                'intervention_applied': False
            }
        })

    return {
        'session_id': 'edge_no_interventions_001',
        'timestamp': datetime.now().isoformat(),
        'total_turns': 15,
        'avg_fidelity': 0.92,
        'total_interventions': 0,
        'intervention_rate': 0.0,
        'turns': turns,
        'config': {
            'purpose': ['Test zero governance'],
            'scope': ['Baseline scenario'],
            'boundaries': []
        }
    }


def generate_alternating_fidelity_session() -> Dict[str, Any]:
    """
    EDGE CASE 9: Extreme oscillation

    Tests:
    - Rapid fidelity changes
    - Frequent basin transitions
    - Visualization of volatile data
    """
    turns = []
    for i in range(1, 21):
        # Alternate between high and low fidelity
        fidelity = 0.95 if i % 2 == 0 else 0.55

        turns.append({
            'turn_number': i,
            'user_message': f'Oscillating message {i}',
            'native_response': f'Response {i}',
            'telos_response': f'Response {i}',
            'assistant_response': f'Response {i}',
            'fidelity': fidelity,
            'intervention_applied': fidelity < 0.67,  # Goldilocks: Drift threshold
            'basin_membership': fidelity >= 0.76,  # Goldilocks: Aligned threshold
            'timestamp': datetime.now().isoformat()
        })

    return {
        'session_id': 'edge_alternating_fidelity_001',
        'timestamp': datetime.now().isoformat(),
        'total_turns': 20,
        'avg_fidelity': 0.75,
        'turns': turns,
        'config': {
            'purpose': ['Test oscillation handling'],
            'scope': ['Volatile behavior'],
            'boundaries': []
        }
    }


def generate_unicode_special_chars_session() -> Dict[str, Any]:
    """
    EDGE CASE 10: Unicode and special characters

    Tests:
    - Unicode handling
    - Special character rendering
    - Export format compatibility
    """
    turns = [{
        'turn_number': 1,
        'user_message': 'Test with émojis 🎉 and spëcial çhars: <>&"\'',
        'native_response': 'Response with 中文字符 and مرحبا',
        'telos_response': 'Response with 中文字符 and مرحبا',
        'assistant_response': 'Response with 中文字符 and مرحبا',
        'fidelity': 0.85,
        'intervention_applied': False,
        'basin_membership': True,
        'timestamp': datetime.now().isoformat()
    }]

    return {
        'session_id': 'edge_unicode_special_chars_001',
        'timestamp': datetime.now().isoformat(),
        'total_turns': 1,
        'avg_fidelity': 0.85,
        'turns': turns,
        'config': {
            'purpose': ['Test unicode handling 🔬'],
            'scope': ['Special characters'],
            'boundaries': ['<HTML> & "quotes"']
        }
    }


def generate_all_edge_cases() -> List[Dict[str, Any]]:
    """Generate complete suite of edge case test sessions."""

    print('Generating edge case test suite...')

    edge_cases = [
        generate_empty_session(),
        generate_single_turn_session(),
        generate_very_long_session(),
        generate_missing_fidelity_session(),
        generate_extreme_fidelity_session(),
        generate_missing_fields_session(),
        generate_all_interventions_session(),
        generate_no_interventions_session(),
        generate_alternating_fidelity_session(),
        generate_unicode_special_chars_session(),
    ]

    print(f'Generated {len(edge_cases)} edge case sessions')

    return edge_cases


def export_edge_cases(edge_cases: List[Dict[str, Any]], output_dir: Path) -> None:
    """Export edge case sessions to JSON files."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f'\nExporting edge cases to {output_dir}/...')

    for case in edge_cases:
        session_id = case['session_id']
        output_file = output_dir / f'{session_id}.json'

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(case, f, indent=2, ensure_ascii=False)

        turn_count = case.get('total_turns', 0)
        print(f'  ✓ {session_id}.json ({turn_count} turns)')

    # Export summary
    summary = {
        'generated_at': datetime.now().isoformat(),
        'total_edge_cases': len(edge_cases),
        'edge_case_types': [
            'Empty session (0 turns)',
            'Single turn',
            'Very long (100 turns)',
            'Missing fidelity scores',
            'Extreme fidelity values',
            'Missing required fields',
            'All interventions',
            'No interventions',
            'Alternating fidelity',
            'Unicode & special chars'
        ],
        'session_ids': [c['session_id'] for c in edge_cases]
    }

    summary_file = output_dir / 'edge_cases_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'\n✓ Summary: {summary_file}')
    print(f'  Total edge cases: {len(edge_cases)}')


def main():
    """Main entry point for edge case test generation."""

    print('\n🔬 TELOS Observatory - Edge Case Test Suite')
    print('=' * 60)

    # Generate all edge cases
    edge_cases = generate_all_edge_cases()

    # Export to test directory
    output_dir = Path('telos_purpose/test_data/edge_cases')
    export_edge_cases(edge_cases, output_dir)

    print('\n' + '=' * 60)
    print('✓ Edge case test suite generated successfully!')
    print(f'✓ Test files available in: {output_dir}/')
    print('\nEdge Cases Generated:')
    for i, case in enumerate(edge_cases, 1):
        turn_count = case.get('total_turns', 0)
        print(f'  {i}. {case["session_id"]}: {turn_count} turns')

    print('\nUse these files to test dashboard robustness:')
    print('  - Load into dashboard via session selector')
    print('  - Verify graceful handling of edge cases')
    print('  - Check analytics don\'t crash on extreme values')
    print('  - Test export functions with unusual data')


if __name__ == '__main__':
    main()
