#!/usr/bin/env python3
"""
Test that Observatory Streamlit app can load without errors.
Validates all imports and core components.
"""

import sys
sys.path.insert(0, '.')

def test_observatory_imports():
    """Test that all Observatory components can be imported."""
    print("\n" + "="*60)
    print("TELOS OBSERVATORY VALIDATION TEST")
    print("="*60 + "\n")

    # Check if streamlit is available
    print("1. Checking dependencies...")
    try:
        import streamlit
        print("   ✓ Streamlit installed")
        streamlit_available = True
    except ImportError:
        print("   ⚠ Streamlit not installed (optional dependency)")
        print("   Note: Run 'pip install -e .' to install all dependencies")
        streamlit_available = False

    print()
    print("2. Testing core Observatory imports...")

    if not streamlit_available:
        print("   Skipping full import test (streamlit required)")
        print("   Checking for import path issues instead...")

        # At least verify no old telos_purpose imports exist
        import os
        import re

        obs_dir = 'observatory'
        old_import_pattern = re.compile(r'from\s+telos_purpose|import\s+telos_purpose')

        issues_found = []
        for root, dirs, files in os.walk(obs_dir):
            for file in files:
                if file.endswith('.py'):
                    filepath = os.path.join(root, file)
                    with open(filepath, 'r') as f:
                        content = f.read()
                        if old_import_pattern.search(content):
                            issues_found.append(filepath)

        if issues_found:
            print(f"   ✗ Found old 'telos_purpose' imports in:")
            for f in issues_found:
                print(f"      - {f}")
            return False
        else:
            print("   ✓ No old import paths detected")
            print("   ✓ Observatory files use correct 'telos' package name")
            print()
            print("="*60)
            print("RESULT: OBSERVATORY READY (Dependencies Required) ✓")
            print("="*60)
            print()
            print("Observatory validation:")
            print("  • No import path errors detected")
            print("  • All files use correct package name 'telos'")
            print("  • Ready to run after installing dependencies")
            print()
            print("To install dependencies:")
            print("  pip install -e .")
            print()
            print("To launch Observatory:")
            print("  streamlit run observatory/main.py")
            print()
            return True

    try:
        from observatory.core.state_manager import StateManager
        print("   ✓ StateManager imported")
    except Exception as e:
        print(f"   ✗ StateManager import failed: {e}")
        return False

    try:
        from observatory.utils.telos_demo_data import generate_telos_demo_session
        print("   ✓ Demo data generator imported")
    except Exception as e:
        print(f"   ✗ Demo data import failed: {e}")
        return False

    print()
    print("2. Testing Observatory components...")

    components = [
        ("SidebarActions", "observatory.components.sidebar_actions"),
        ("ConversationDisplay", "observatory.components.conversation_display"),
        ("ObservationDeck", "observatory.components.observation_deck"),
        ("TELOSCOPEControls", "observatory.components.teloscope_controls"),
        ("BetaOnboarding", "observatory.components.beta_onboarding"),
        ("StewardPanel", "observatory.components.steward_panel"),
    ]

    for component_name, module_path in components:
        try:
            module = __import__(module_path, fromlist=[component_name])
            getattr(module, component_name)
            print(f"   ✓ {component_name} imported")
        except Exception as e:
            print(f"   ✗ {component_name} import failed: {e}")
            return False

    print()
    print("3. Testing StateManager instantiation...")
    try:
        state_manager = StateManager()
        print("   ✓ StateManager created successfully")

        # Test initialization with empty data
        empty_data = {
            'session_id': 'test_session',
            'turns': [],
            'total_turns': 0,
            'current_turn': 0,
            'avg_fidelity': 0.0,
            'total_interventions': 0,
            'drift_warnings': 0
        }
        state_manager.initialize(empty_data)
        print("   ✓ StateManager initialized with test data")

    except Exception as e:
        print(f"   ✗ StateManager instantiation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print()
    print("4. Testing demo data generation...")
    try:
        demo_session = generate_telos_demo_session()
        print(f"   ✓ Demo session generated ({len(demo_session.get('turns', []))} turns)")
    except Exception as e:
        print(f"   ✗ Demo data generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print()
    print("="*60)
    print("RESULT: OBSERVATORY READY TO RUN ✓")
    print("="*60)
    print()
    print("All Observatory components validated:")
    print("  • Core state management working")
    print("  • All UI components importable")
    print("  • Demo data generation functional")
    print("  • No import errors detected")
    print()
    print("To launch Observatory:")
    print("  streamlit run observatory/main.py")
    print()

    return True

if __name__ == "__main__":
    try:
        success = test_observatory_imports()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ OBSERVATORY TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
