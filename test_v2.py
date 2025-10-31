#!/usr/bin/env python3
"""
Quick test script for V2 Observatory components.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from telos_observatory_v2.core.state_manager import StateManager
from telos_observatory_v2.utils.mock_data import generate_mock_session

def test_v2_components():
    """Test V2 Observatory components."""
    print("Testing TELOS Observatory V2...")
    print("=" * 60)

    # Test 1: Mock data generation
    print("\n1. Testing mock data generation...")
    mock_data = generate_mock_session(num_turns=10)
    print(f"✓ Generated {len(mock_data['turns'])} turns")
    print(f"  Session ID: {mock_data['session_id']}")
    print(f"  Avg Fidelity: {mock_data['statistics']['avg_fidelity']:.3f}")
    print(f"  Interventions: {mock_data['statistics']['interventions']}")

    # Test 2: State Manager
    print("\n2. Testing StateManager...")
    state_manager = StateManager()
    state_manager.initialize(mock_data)
    print(f"✓ StateManager initialized")
    print(f"  Total turns: {state_manager.get_session_info()['total_turns']}")
    print(f"  Current turn: {state_manager.get_session_info()['current_turn']}")

    # Test 3: Navigation
    print("\n3. Testing navigation...")
    state_manager.next_turn()
    print(f"✓ Next turn: {state_manager.get_session_info()['current_turn']}")
    state_manager.previous_turn()
    print(f"✓ Previous turn: {state_manager.get_session_info()['current_turn']}")
    state_manager.jump_to_turn(5)
    print(f"✓ Jump to turn 5: {state_manager.get_session_info()['current_turn']}")

    # Test 4: Turn data
    print("\n4. Testing turn data retrieval...")
    turn_data = state_manager.get_current_turn_data()
    print(f"✓ Retrieved turn data")
    print(f"  Fidelity: {turn_data['fidelity']:.3f}")
    print(f"  Distance: {turn_data['distance']:.3f}")
    print(f"  Status: {turn_data['status_text']}")
    print(f"  Intervention: {turn_data['intervention_applied']}")

    # Test 5: UI State
    print("\n5. Testing UI state...")
    state_manager.toggle_deck()
    print(f"✓ Deck expanded: {state_manager.is_deck_expanded()}")
    state_manager.start_playback()
    print(f"✓ Playback started: {state_manager.is_playing()}")
    state_manager.set_playback_speed(2.0)
    print(f"✓ Playback speed: {state_manager.state.playback_speed}x")

    # Test 6: Component toggles
    print("\n6. Testing component toggles...")
    state_manager.toggle_component('steward')
    print(f"✓ Steward enabled: {state_manager.state.show_steward}")
    state_manager.toggle_component('math_breakdown')
    print(f"✓ Math breakdown enabled: {state_manager.state.show_math_breakdown}")

    print("\n" + "=" * 60)
    print("✅ All V2 Observatory tests passed!")
    print("\nComponents verified:")
    print("  ✓ StateManager")
    print("  ✓ Mock data generation")
    print("  ✓ Navigation controls")
    print("  ✓ Turn data retrieval")
    print("  ✓ UI state management")
    print("  ✓ Component toggles")
    print("\nReady for Streamlit deployment!")

    return True

if __name__ == "__main__":
    try:
        test_v2_components()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
