#!/usr/bin/env python3
"""
Primacy State Activation Script
Enables PS in production when ready
"""

import json
from pathlib import Path
import sys


def activate_primacy_state(mode='shadow_mode'):
    """
    Activate Primacy State in specified mode.

    Args:
        mode: One of 'shadow_mode', 'parallel_validation', 'primary_with_fallback', 'primary_only'
    """
    config_path = Path(__file__).parent / "primacy_state_config.json"

    # Load current config
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Update settings
    config['primacy_state']['enabled'] = True
    config['primacy_state']['rollout_phase'] = mode

    # Set mode-specific settings
    if mode == 'shadow_mode':
        config['primacy_state']['parallel_mode'] = True
        print("🔄 PS activated in SHADOW MODE")
        print("   - Calculating PS alongside fidelity")
        print("   - No governance changes")
        print("   - Logging for comparison")

    elif mode == 'parallel_validation':
        config['primacy_state']['parallel_mode'] = True
        print("🔄 PS activated in PARALLEL VALIDATION")
        print("   - Both metrics calculated")
        print("   - Comparing decisions")
        print("   - Validating agreement")

    elif mode == 'primary_with_fallback':
        config['primacy_state']['parallel_mode'] = True
        print("⚡ PS activated as PRIMARY (with fallback)")
        print("   - PS used for governance")
        print("   - Fidelity as backup")
        print("   - Full diagnostic mode")

    elif mode == 'primary_only':
        config['primacy_state']['parallel_mode'] = False
        print("🚀 PS activated as PRIMARY ONLY")
        print("   - PS is sole governance metric")
        print("   - Fidelity deprecated")
        print("   - Production mode")

    # Save updated config
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n✅ Configuration saved to {config_path}")
    return True


def deactivate_primacy_state():
    """Deactivate Primacy State (rollback)."""
    config_path = Path(__file__).parent / "primacy_state_config.json"

    with open(config_path, 'r') as f:
        config = json.load(f)

    config['primacy_state']['enabled'] = False
    config['primacy_state']['rollout_phase'] = 'disabled'

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print("🔴 PS deactivated")
    print(f"✅ Configuration saved to {config_path}")
    return True


def check_status():
    """Check current PS status."""
    config_path = Path(__file__).parent / "primacy_state_config.json"

    with open(config_path, 'r') as f:
        config = json.load(f)

    ps = config['primacy_state']

    print("\n" + "=" * 60)
    print("PRIMACY STATE STATUS")
    print("=" * 60)
    print(f"Enabled: {'✅ YES' if ps['enabled'] else '❌ NO'}")
    print(f"Phase: {ps['rollout_phase'].upper()}")
    print(f"Parallel Mode: {'ON' if ps['parallel_mode'] else 'OFF'}")
    print(f"Energy Tracking: {'ON' if ps['track_energy'] else 'OFF'}")
    print("\nThresholds:")
    for level, threshold in ps['thresholds'].items():
        print(f"  {level}: ≥{threshold}")
    print("=" * 60)


def main():
    """CLI for PS activation."""
    if len(sys.argv) < 2:
        print("Usage: python activate_primacy_state.py [command]")
        print("\nCommands:")
        print("  status      - Check current PS status")
        print("  shadow      - Enable shadow mode (logging only)")
        print("  parallel    - Enable parallel validation")
        print("  primary     - Enable as primary with fallback")
        print("  full        - Enable as sole metric (production)")
        print("  disable     - Disable PS (rollback)")
        return

    command = sys.argv[1].lower()

    if command == 'status':
        check_status()
    elif command == 'shadow':
        activate_primacy_state('shadow_mode')
        check_status()
    elif command == 'parallel':
        activate_primacy_state('parallel_validation')
        check_status()
    elif command == 'primary':
        activate_primacy_state('primary_with_fallback')
        check_status()
    elif command == 'full':
        response = input("⚠️  This makes PS the ONLY governance metric. Are you sure? (yes/no): ")
        if response.lower() == 'yes':
            activate_primacy_state('primary_only')
            check_status()
        else:
            print("Aborted.")
    elif command == 'disable':
        deactivate_primacy_state()
        check_status()
    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    main()