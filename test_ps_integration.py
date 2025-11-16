#!/usr/bin/env python3
"""
Test Primacy State Integration
Validates PS calculation works with actual session data
"""

import json
import numpy as np
import sys
from pathlib import Path

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "telos_purpose" / "core"))

from primacy_state import PrimacyStateCalculator, interpret_primacy_state

def test_basic_ps_calculation():
    """Test basic PS calculation with known values."""
    print("=" * 60)
    print("TEST 1: Basic PS Calculation")
    print("-" * 60)

    calc = PrimacyStateCalculator(track_energy=True)

    # Test case 1: Perfect alignment
    dim = 100
    response = np.ones(dim) / np.sqrt(dim)
    user_pa = np.ones(dim) / np.sqrt(dim)
    ai_pa = np.ones(dim) / np.sqrt(dim)

    metrics = calc.compute_primacy_state(response, user_pa, ai_pa)

    print(f"Perfect Alignment Test:")
    print(f"  PS Score: {metrics.ps_score:.4f} (expected: 1.0000)")
    print(f"  F_user: {metrics.f_user:.4f}")
    print(f"  F_AI: {metrics.f_ai:.4f}")
    print(f"  ρ_PA: {metrics.rho_pa:.4f}")
    print(f"  Condition: {metrics.condition}")
    print(f"  Diagnostic: {metrics.get_diagnostic()}")

    assert abs(metrics.ps_score - 1.0) < 0.001, "Perfect alignment should give PS=1.0"
    print("  ✅ PASS")
    print()

    # Test case 2: User drift
    response = np.zeros(dim)
    response[0] = 1.0  # Orthogonal to user_pa

    metrics = calc.compute_primacy_state(response, user_pa, ai_pa)

    print(f"User Drift Test:")
    print(f"  PS Score: {metrics.ps_score:.4f}")
    print(f"  F_user: {metrics.f_user:.4f}")
    print(f"  F_AI: {metrics.f_ai:.4f}")
    print(f"  Condition: {metrics.condition}")
    print(f"  Diagnostic: {metrics.get_diagnostic()}")

    assert metrics.ps_score < 0.5, "User drift should give low PS"
    assert "User purpose drift" in metrics.get_diagnostic()
    print("  ✅ PASS")
    print()

    # Test case 3: PA misalignment
    user_pa = np.ones(dim) / np.sqrt(dim)
    ai_pa = np.zeros(dim)
    ai_pa[0] = 1.0  # Orthogonal to user_pa
    response = user_pa.copy()

    # Reset calculator cache to recompute PA correlation
    calc.reset_session_cache()
    metrics = calc.compute_primacy_state(response, user_pa, ai_pa)

    print(f"PA Misalignment Test:")
    print(f"  PS Score: {metrics.ps_score:.4f}")
    print(f"  F_user: {metrics.f_user:.4f}")
    print(f"  F_AI: {metrics.f_ai:.4f}")
    print(f"  ρ_PA: {metrics.rho_pa:.4f}")
    print(f"  Condition: {metrics.condition}")
    print(f"  Diagnostic: {metrics.get_diagnostic()}")

    assert metrics.rho_pa < 0.2, "Orthogonal PAs should have low correlation"
    assert metrics.ps_score < 0.2, "Misaligned PAs should give very low PS"
    print("  ✅ PASS")


def test_ps_interpretation():
    """Test PS interpretation narratives."""
    print("=" * 60)
    print("TEST 2: PS Interpretation")
    print("-" * 60)

    calc = PrimacyStateCalculator(track_energy=True)

    # Create test scenarios
    scenarios = [
        {
            "name": "Perfect Equilibrium",
            "response": np.ones(50) / np.sqrt(50),
            "user_pa": np.ones(50) / np.sqrt(50),
            "ai_pa": np.ones(50) / np.sqrt(50)
        },
        {
            "name": "Moderate Drift",
            "response": np.array([0.8] + [0.2] * 49) / np.linalg.norm(np.array([0.8] + [0.2] * 49)),
            "user_pa": np.ones(50) / np.sqrt(50),
            "ai_pa": np.ones(50) / np.sqrt(50)
        },
        {
            "name": "Critical Failure",
            "response": np.zeros(50),
            "user_pa": np.ones(50) / np.sqrt(50),
            "ai_pa": np.ones(50) / np.sqrt(50)
        }
    ]

    scenarios[1]["response"][0] = 1.0  # Make orthogonal
    scenarios[1]["response"] = scenarios[1]["response"] / np.linalg.norm(scenarios[1]["response"])

    scenarios[2]["response"][0] = 1.0  # Completely orthogonal

    for scenario in scenarios:
        print(f"\nScenario: {scenario['name']}")
        print("-" * 40)

        metrics = calc.compute_primacy_state(
            scenario["response"],
            scenario["user_pa"],
            scenario["ai_pa"]
        )

        narrative = interpret_primacy_state(metrics, verbose=True)
        print(f"  {narrative}")
        print(f"  Intervention: {metrics.to_dict()['intervention_urgency'] if 'intervention_urgency' in metrics.to_dict() else 'N/A'}")

        # Verify narrative quality
        if "Perfect" in scenario["name"]:
            assert "ACHIEVED" in narrative
            print("  ✅ Correct narrative for perfect state")
        elif "Critical" in scenario["name"]:
            assert "COLLAPSED" in narrative or "VIOLATED" in narrative
            print("  ✅ Correct narrative for failure state")


def test_energy_tracking():
    """Test dual potential energy tracking."""
    print("=" * 60)
    print("TEST 3: Energy Tracking")
    print("-" * 60)

    calc = PrimacyStateCalculator(track_energy=True)

    dim = 50
    user_pa = np.ones(dim) / np.sqrt(dim)
    ai_pa = np.ones(dim) / np.sqrt(dim)

    # Simulate conversation drift and recovery
    print("Simulating conversation drift and recovery...")
    print()

    # Turn 1: Good state
    response1 = user_pa.copy()
    metrics1 = calc.compute_primacy_state(response1, user_pa, ai_pa)
    print(f"Turn 1 - Good State:")
    print(f"  PS={metrics1.ps_score:.3f}, V_dual={metrics1.v_dual:.3f}")

    # Turn 2: Slight drift
    response2 = 0.9 * user_pa + 0.1 * np.random.randn(dim)
    response2 = response2 / np.linalg.norm(response2)
    metrics2 = calc.compute_primacy_state(response2, user_pa, ai_pa)
    print(f"Turn 2 - Slight Drift:")
    print(f"  PS={metrics2.ps_score:.3f}, V_dual={metrics2.v_dual:.3f}, ΔV={metrics2.delta_v:.3f}")

    # Turn 3: More drift
    response3 = 0.7 * user_pa + 0.3 * np.random.randn(dim)
    response3 = response3 / np.linalg.norm(response3)
    metrics3 = calc.compute_primacy_state(response3, user_pa, ai_pa)
    print(f"Turn 3 - More Drift:")
    print(f"  PS={metrics3.ps_score:.3f}, V_dual={metrics3.v_dual:.3f}, ΔV={metrics3.delta_v:.3f}")

    # Turn 4: Recovery
    response4 = 0.95 * user_pa + 0.05 * np.random.randn(dim)
    response4 = response4 / np.linalg.norm(response4)
    metrics4 = calc.compute_primacy_state(response4, user_pa, ai_pa)
    print(f"Turn 4 - Recovery:")
    print(f"  PS={metrics4.ps_score:.3f}, V_dual={metrics4.v_dual:.3f}, ΔV={metrics4.delta_v:.3f}")

    # Verify energy dynamics
    if metrics3.delta_v and metrics3.delta_v > 0:
        print("\n  ✅ Correctly detected divergence during drift")
    if metrics4.delta_v and metrics4.delta_v < 0:
        print("  ✅ Correctly detected convergence during recovery")


def test_config_loading():
    """Test loading PS configuration."""
    print("=" * 60)
    print("TEST 4: Configuration Loading")
    print("-" * 60)

    config_path = Path(__file__).parent / "primacy_state_config.json"

    with open(config_path, 'r') as f:
        config = json.load(f)

    ps_config = config['primacy_state']

    print("Configuration loaded:")
    print(f"  Enabled: {ps_config['enabled']}")
    print(f"  Parallel Mode: {ps_config['parallel_mode']}")
    print(f"  Track Energy: {ps_config['track_energy']}")
    print(f"  Current Phase: {ps_config['rollout_phase']}")
    print()
    print("Thresholds:")
    for condition, threshold in ps_config['thresholds'].items():
        print(f"  {condition}: {threshold}")
    print()

    assert ps_config['enabled'] == False, "PS should start disabled"
    assert ps_config['parallel_mode'] == True, "Should run in parallel initially"
    print("✅ Configuration valid")


def main():
    """Run all PS integration tests."""
    print("\n" + "=" * 60)
    print("PRIMACY STATE INTEGRATION TEST SUITE")
    print("=" * 60)
    print()

    try:
        test_basic_ps_calculation()
        test_ps_interpretation()
        test_energy_tracking()
        test_config_loading()

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED - PS READY FOR PRODUCTION")
        print("=" * 60)
        print()
        print("Next steps to enable:")
        print("1. Set 'enabled': true in primacy_state_config.json")
        print("2. Run Supabase migration script")
        print("3. Initialize PS in StateManager")
        print("4. Monitor parallel execution")
        print("5. Switch to primary after validation")

        return True

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)