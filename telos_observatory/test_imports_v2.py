#!/usr/bin/env python3
"""
TELOSCOPE v2 Import Validation

Tests that all v2 foundation components can be imported successfully.

Usage:
    cd ~/Desktop/TELOS/telos_observatory
    python3 test_imports_v2.py

Expected Output:
    Testing TELOSCOPE v2 imports...
    ✅ State management imports
    ✅ Utility imports
    ✅ Component imports

    ✅ All v2 imports successful!

    Components validated:
    - teloscope_state (state management)
    - mock_data (session generation)
    - marker_generator (timeline markers)
    - scroll_controller (scrolling & dimming)
    - turn_indicator (turn display)
"""

import sys
from pathlib import Path


def test_imports():
    """Test all v2 foundation component imports."""
    print("\n" + "=" * 60)
    print("TELOSCOPE v2 Import Validation")
    print("=" * 60 + "\n")

    print("Testing TELOSCOPE v2 imports...\n")

    errors = []

    # ===== STATE MANAGEMENT =====
    try:
        from teloscope_v2.state.teloscope_state import (
            init_teloscope_state,
            get_teloscope_state,
            update_teloscope_state,
            reset_teloscope_state,
            get_current_turn,
            set_current_turn,
            is_playing,
            set_playing,
            get_playback_speed,
            set_playback_speed,
            is_tool_active,
            toggle_tool,
            get_teloscope_position,
            set_teloscope_position,
            get_state_summary,
        )
        print("✅ State management imports")
        print("   - init_teloscope_state")
        print("   - get/update/reset state")
        print("   - navigation helpers (get/set current_turn, playing, speed)")
        print("   - tool management (is_tool_active, toggle_tool)")
        print("   - position helpers (get/set position)")
        print("   - debug (get_state_summary)")
    except ImportError as e:
        errors.append(f"State management: {e}")
        print("❌ State management imports FAILED")
        print(f"   Error: {e}")

    print()

    # ===== UTILS =====
    try:
        from teloscope_v2.utils.mock_data import (
            generate_enhanced_session,
            generate_test_suite,
            generate_quick_session,
            generate_long_session,
            validate_session,
            session_to_transcript,
        )
        print("✅ Mock data imports")
        print("   - generate_enhanced_session (main generator)")
        print("   - generate_test_suite (all session types)")
        print("   - generate_quick_session (12 turns)")
        print("   - generate_long_session (50 turns)")
        print("   - validate_session")
        print("   - session_to_transcript")
    except ImportError as e:
        errors.append(f"Mock data: {e}")
        print("❌ Mock data imports FAILED")
        print(f"   Error: {e}")

    print()

    try:
        from teloscope_v2.utils.marker_generator import (
            generate_timeline_markers,
            generate_timeline_legend,
            generate_annotated_markers,
            get_marker_color,
            calculate_marker_width,
            get_marker_css,
        )
        print("✅ Marker generator imports")
        print("   - generate_timeline_markers (main generator)")
        print("   - generate_timeline_legend")
        print("   - generate_annotated_markers")
        print("   - get_marker_color")
        print("   - calculate_marker_width")
        print("   - get_marker_css")
    except ImportError as e:
        errors.append(f"Marker generator: {e}")
        print("❌ Marker generator imports FAILED")
        print(f"   Error: {e}")

    print()

    try:
        from teloscope_v2.utils.scroll_controller import (
            scroll_to_turn,
            render_scroll_anchor,
            scroll_to_turn_anchor,
            is_turn_in_viewport,
            calculate_scroll_offset,
            enable_auto_scroll,
            is_auto_scroll_enabled,
            calculate_turn_opacity,
            get_turn_border_style,
            get_turn_css_classes,
            get_scroll_container_css,
        )
        print("✅ Scroll controller imports")
        print("   - scroll_to_turn (main scroll function)")
        print("   - render_scroll_anchor")
        print("   - viewport helpers (is_turn_in_viewport, calculate_offset)")
        print("   - auto-scroll (enable, is_enabled)")
        print("   - dimming (calculate_turn_opacity)")
        print("   - styling (get_turn_border_style, get_css)")
    except ImportError as e:
        errors.append(f"Scroll controller: {e}")
        print("❌ Scroll controller imports FAILED")
        print(f"   Error: {e}")

    print()

    # ===== COMPONENTS =====
    try:
        from teloscope_v2.components.turn_indicator import (
            render_turn_indicator,
            render_turn_indicator_inline,
            render_turn_progress_bar,
            get_turn_info,
            is_first_turn,
            is_last_turn,
            get_relative_position,
            get_turn_indicator_css,
        )
        print("✅ Component imports")
        print("   - render_turn_indicator (main render)")
        print("   - render_turn_indicator_inline (compact)")
        print("   - render_turn_progress_bar (progress mode)")
        print("   - get_turn_info")
        print("   - is_first_turn, is_last_turn")
        print("   - get_relative_position")
        print("   - get_turn_indicator_css")
    except ImportError as e:
        errors.append(f"Components: {e}")
        print("❌ Component imports FAILED")
        print(f"   Error: {e}")

    print()

    # ===== SUMMARY =====
    print("=" * 60)
    if errors:
        print("❌ IMPORT VALIDATION FAILED\n")
        print("Errors:")
        for error in errors:
            print(f"  - {error}")
        print()
        return False
    else:
        print("✅ ALL v2 IMPORTS SUCCESSFUL!\n")
        print("Components validated:")
        print("  - teloscope_state (state management)")
        print("  - mock_data (session generation)")
        print("  - marker_generator (timeline markers)")
        print("  - scroll_controller (scrolling & dimming)")
        print("  - turn_indicator (turn display)")
        print()
        print("Foundation Status: ✅ 6/6 components importable")
        print()
        return True


def test_version_info():
    """Test version information."""
    try:
        from teloscope_v2 import __version__
        print(f"TELOSCOPE v2 Version: {__version__}")
        return True
    except ImportError as e:
        print(f"⚠️  Could not import package version: {e}")
        return False


def main():
    """Main entry point."""
    # Test version
    test_version_info()
    print()

    # Test imports
    success = test_imports()

    # Exit code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
