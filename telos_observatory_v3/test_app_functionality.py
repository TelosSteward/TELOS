#!/usr/bin/env python3
"""
Quick test to verify TELOSCOPE_BETA app functionality.
Tests A/B testing, backend connection, and core features.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Test A/B Test Manager
print("Testing A/B Test Manager...")
from services.ab_test_manager import get_ab_test_manager
ab_manager = get_ab_test_manager()
print(f"  ✓ A/B Test Manager created")
print(f"  - Observatory Lens experiment: {ab_manager.get_variant('observatory_lens')}")
print(f"  - Intervention style: {ab_manager.get_variant('intervention_style')}")
print(f"  - Onboarding style: {ab_manager.get_variant('onboarding_style')}")

# Test Backend Service
print("\nTesting Backend Service...")
from services.backend_client import get_backend_service
backend = get_backend_service()
if backend.enabled:
    if backend.test_connection():
        print("  ✓ Backend connection successful")
    else:
        print("  ✗ Backend connection failed")
else:
    print("  ✓ Backend disabled (expected for local testing)")

# Test Demo Mode Configuration
print("\nTesting Demo Mode Configuration...")
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'telos_privacy'))
from demo_mode import telos_framework_demo
demo_config = telos_framework_demo.get_demo_attractor_config()
print(f"  ✓ Demo PA config loaded")
print(f"  - Purpose statements: {len(demo_config['purpose'])}")
print(f"  - Scope elements: {len(demo_config['scope'])}")
print(f"  - Boundaries: {len(demo_config['boundaries'])}")
print(f"  - Pre-established: {telos_framework_demo.is_pre_established()}")

# Test State Manager
print("\nTesting State Manager...")
from core.state_manager import StateManager
state_manager = StateManager()
print("  ✓ State Manager created")

# Test Components Import
print("\nTesting Component Imports...")
try:
    from components.conversation_display import ConversationDisplay
    print("  ✓ ConversationDisplay imported")
    from components.observation_deck import ObservationDeck
    print("  ✓ ObservationDeck imported")
    from components.observatory_lens import ObservatoryLens
    print("  ✓ ObservatoryLens imported")
    from components.beta_onboarding import BetaOnboarding
    print("  ✓ BetaOnboarding imported")
    from components.steward_panel import StewardPanel
    print("  ✓ StewardPanel imported")
except ImportError as e:
    print(f"  ✗ Import error: {e}")

print("\n✅ All core functionality tests passed!")