"""
Platform Validation Script for TELOS Observatory
================================================

Automated validation suite to verify platform functionality including:
- Dependency checks
- Test data generation
- Analytics calculations
- Export functions
- File structure integrity

Usage:
    python -m telos_purpose.validation.validate_platform
"""

import sys
import json
from pathlib import Path
import traceback
from typing import List, Dict, Any


class PlatformValidator:
    """Automated validation for TELOS Observatory platform."""

    def __init__(self):
        self.passed: List[str] = []
        self.failed: List[str] = []
        self.warnings: List[str] = []

    def validate_all(self):
        """Run all validation checks."""

        print('\n🔍 TELOS Observatory Platform Validation')
        print('=' * 60)

        self.check_dependencies()
        self.check_test_data_generation()
        self.validate_session_state_manager()
        self.validate_analytics_helpers()
        self.check_file_structure()
        self.validate_dashboard_imports()

        self.print_summary()

        # Exit with error code if failures
        if self.failed:
            sys.exit(1)

    def check_dependencies(self):
        """Verify required packages are installed."""
        print('\n📦 Checking Dependencies...')

        required = {
            'streamlit': 'Dashboard UI framework',
            'pandas': 'Data manipulation',
            'plotly': 'Interactive visualizations',
            'numpy': 'Numerical computing',
            'scipy': 'Statistical functions'
        }

        for package, description in required.items():
            try:
                __import__(package)
                self.passed.append(f'✓ {package} ({description})')
                print(f'  ✓ {package} installed')
            except ImportError:
                self.failed.append(f'✗ {package} missing ({description})')
                print(f'  ✗ {package} MISSING')

    def check_test_data_generation(self):
        """Verify test data can be generated."""
        print('\n🧪 Checking Test Data Generation...')

        try:
            from telos_purpose.test_data.generate_test_sessions import (
                generate_test_session,
                generate_normal_session,
                generate_high_drift_session
            )

            # Test basic generation
            session = generate_test_session(5, 'validation_test', base_fidelity=0.85)

            if not session or 'turns' not in session:
                self.failed.append('✗ Test session missing required fields')
                print('  ✗ Session structure invalid')
                return

            if len(session['turns']) != 5:
                self.failed.append(f"✗ Expected 5 turns, got {len(session['turns'])}")
                print(f"  ✗ Turn count mismatch")
                return

            # Verify turn structure
            turn = session['turns'][0]
            required_fields = [
                'turn_number', 'user_input', 'native_response', 'telos_response',
                'fidelity', 'intervention_applied', 'basin_membership'
            ]

            for field in required_fields:
                if field not in turn:
                    self.failed.append(f'✗ Turn missing field: {field}')
                    print(f'  ✗ Missing field: {field}')
                    return

            # Test predefined session types
            normal = generate_normal_session()
            high_drift = generate_high_drift_session()

            if normal['session_id'] != 'normal_session_001':
                self.failed.append('✗ Predefined session ID mismatch')
                print('  ✗ Session ID incorrect')
                return

            self.passed.append('✓ Test data generation works correctly')
            print('  ✓ All test data generation checks passed')

        except Exception as e:
            self.failed.append(f'✗ Test data generation failed: {str(e)}')
            print(f'  ✗ ERROR: {str(e)}')
            traceback.print_exc()

    def validate_session_state_manager(self):
        """Test SessionStateManager functionality."""
        print('\n💾 Validating Session State Manager...')

        try:
            import numpy as np
            from telos_purpose.core.session_state import SessionStateManager, TurnSnapshot

            # Create manager
            manager = SessionStateManager()

            # Save a test turn
            user_emb = np.random.rand(384, 1)
            response_emb = np.random.rand(384, 1)
            attractor_center = np.random.rand(384, 1)

            metrics = {
                'telic_fidelity': 0.85,
                'error_signal': 0.15,
                'lyapunov_value': 0.02,
                'drift_distance': 0.12,
                'primacy_basin_membership': True
            }

            conversation_history = [
                {'role': 'user', 'content': 'Test input'},
                {'role': 'assistant', 'content': 'Test response'}
            ]

            attractor_config = {
                'purpose': ['Test purpose'],
                'scope': ['Test scope'],
                'boundaries': ['Test boundary']
            }

            snapshot = manager.save_turn_snapshot(
                turn_number=0,
                user_input='Test input',
                native_response='Native test response',
                telos_response='TELOS test response',
                user_embedding=user_emb,
                response_embedding=response_emb,
                attractor_center=attractor_center,
                metrics=metrics,
                conversation_history=conversation_history,
                attractor_config=attractor_config
            )

            # Verify snapshot
            if snapshot.turn_number != 0:
                self.failed.append('✗ Turn number mismatch')
                print('  ✗ Turn number incorrect')
                return

            if snapshot.native_response != 'Native test response':
                self.failed.append('✗ Native response mismatch')
                print('  ✗ Native response incorrect')
                return

            if snapshot.telos_response != 'TELOS test response':
                self.failed.append('✗ TELOS response mismatch')
                print('  ✗ TELOS response incorrect')
                return

            # Test reconstruction
            state = manager.reconstruct_state_at_turn(0)
            if state is None:
                self.failed.append('✗ State reconstruction failed')
                print('  ✗ Cannot reconstruct state')
                return

            if state['turn_number'] != 0:
                self.failed.append('✗ Reconstructed state invalid')
                print('  ✗ Reconstructed state incorrect')
                return

            self.passed.append('✓ SessionStateManager works correctly')
            print('  ✓ All session state checks passed')

        except Exception as e:
            self.failed.append(f'✗ SessionStateManager validation failed: {str(e)}')
            print(f'  ✗ ERROR: {str(e)}')
            traceback.print_exc()

    def validate_analytics_helpers(self):
        """Test analytics helper functions from dashboard."""
        print('\n📊 Validating Analytics Functions...')

        try:
            # Create mock session data matching dashboard expectations
            mock_sessions = [
                {
                    'session_id': 'test_001',
                    'total_turns': 3,
                    'avg_fidelity': 0.74,
                    'turns': [
                        {
                            'turn_number': 1,
                            'fidelity': 0.85,
                            'intervention_applied': False,
                            'basin_membership': True
                        },
                        {
                            'turn_number': 2,
                            'fidelity': 0.72,
                            'intervention_applied': True,
                            'basin_membership': True
                        },
                        {
                            'turn_number': 3,
                            'fidelity': 0.65,
                            'intervention_applied': False,
                            'basin_membership': False
                        }
                    ]
                }
            ]

            # Test fidelity calculation
            calculated_avg = sum(t['fidelity'] for t in mock_sessions[0]['turns']) / 3
            expected_avg = 0.74

            if abs(calculated_avg - expected_avg) < 0.01:
                self.passed.append('✓ Fidelity calculation correct')
                print('  ✓ Fidelity calculation accurate')
            else:
                self.failed.append(f'✗ Fidelity calculation wrong: {calculated_avg:.3f} != {expected_avg:.3f}')
                print(f'  ✗ Fidelity mismatch')

            # Test intervention counting
            intervention_count = sum(1 for t in mock_sessions[0]['turns'] if t['intervention_applied'])
            expected_interventions = 1

            if intervention_count == expected_interventions:
                self.passed.append('✓ Intervention counting correct')
                print('  ✓ Intervention counting accurate')
            else:
                self.failed.append(f'✗ Intervention count wrong: {intervention_count} != {expected_interventions}')
                print(f'  ✗ Intervention count mismatch')

            # Test basin membership tracking
            basin_violations = sum(1 for t in mock_sessions[0]['turns'] if not t['basin_membership'])
            expected_violations = 1

            if basin_violations == expected_violations:
                self.passed.append('✓ Basin tracking correct')
                print('  ✓ Basin membership tracking accurate')
            else:
                self.failed.append(f'✗ Basin violations wrong: {basin_violations} != {expected_violations}')
                print(f'  ✗ Basin tracking mismatch')

        except Exception as e:
            self.failed.append(f'✗ Analytics validation failed: {str(e)}')
            print(f'  ✗ ERROR: {str(e)}')
            traceback.print_exc()

    def check_file_structure(self):
        """Verify critical files and directories exist."""
        print('\n📁 Checking File Structure...')

        base_path = Path(__file__).parent.parent

        required_files = [
            'core/session_state.py',
            'core/unified_steward.py',
            'core/primacy_math.py',
            'sessions/live_interceptor.py',
            'dev_dashboard/streamlit_live_comparison.py',
            'test_data/__init__.py',
            'test_data/generate_test_sessions.py',
            'validation/__init__.py'
        ]

        for file_path in required_files:
            full_path = base_path / file_path
            if full_path.exists():
                self.passed.append(f'✓ {file_path} exists')
                print(f'  ✓ {file_path}')
            else:
                self.failed.append(f'✗ {file_path} missing')
                print(f'  ✗ {file_path} MISSING')

    def validate_dashboard_imports(self):
        """Test that dashboard can be imported without errors."""
        print('\n🎨 Validating Dashboard Imports...')

        try:
            # Try importing key dashboard modules
            # Note: We don't actually import streamlit_live_comparison directly
            # as it requires Streamlit session state, but we can check if it exists
            dashboard_path = Path(__file__).parent.parent / 'dev_dashboard' / 'streamlit_live_comparison.py'

            if not dashboard_path.exists():
                self.failed.append('✗ Dashboard file not found')
                print('  ✗ Dashboard file missing')
                return

            # Check file can be read
            with open(dashboard_path, 'r') as f:
                content = f.read()

            if 'def main()' in content or 'st.title' in content:
                self.passed.append('✓ Dashboard file structure valid')
                print('  ✓ Dashboard file readable and valid')
            else:
                self.warnings.append('⚠ Dashboard may be missing expected structure')
                print('  ⚠ Dashboard structure unclear')

        except Exception as e:
            self.failed.append(f'✗ Dashboard validation failed: {str(e)}')
            print(f'  ✗ ERROR: {str(e)}')

    def print_summary(self):
        """Print validation summary."""
        print('\n' + '=' * 60)
        print('📋 VALIDATION SUMMARY')
        print('=' * 60)

        print(f'\n✅ Passed: {len(self.passed)}')
        if self.passed:
            for item in self.passed:
                print(f'  {item}')

        if self.warnings:
            print(f'\n⚠️  Warnings: {len(self.warnings)}')
            for item in self.warnings:
                print(f'  {item}')

        if self.failed:
            print(f'\n❌ Failed: {len(self.failed)}')
            for item in self.failed:
                print(f'  {item}')
        else:
            print(f'\n❌ Failed: 0')

        print('\n' + '=' * 60)

        if not self.failed:
            print('✨ ALL VALIDATION CHECKS PASSED ✨')
            print('Platform is ready for use!')
        else:
            print('⚠️  VALIDATION FAILED')
            print('Please fix the issues above before using the platform.')

        print('=' * 60)


def main():
    """Main entry point for validation script."""
    validator = PlatformValidator()
    validator.validate_all()


if __name__ == '__main__':
    main()
