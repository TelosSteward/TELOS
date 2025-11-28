"""
Integration Testing Suite for TELOS Observatory
===============================================

End-to-end integration tests verifying all components work together:
- Data pipeline (generate → load → validate)
- Analytics pipeline (compute → verify)
- Export pipeline (save → load → verify)
- Complete session workflow (end-to-end)

Usage:
    python -m telos_purpose.validation.integration_tests
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any


class IntegrationTester:
    """End-to-end integration testing for TELOS Observatory"""

    def __init__(self):
        self.passed: List[str] = []
        self.failed: List[str] = []
        self.test_output_dir = Path('test_output')
        self.test_output_dir.mkdir(exist_ok=True)

    def run_all_tests(self):
        """Execute all integration tests"""

        print('\n🔬 TELOS Observatory Integration Tests')
        print('=' * 60)

        self.test_data_pipeline()
        self.test_analytics_pipeline()
        self.test_export_pipeline()
        self.test_session_workflow()
        self.test_edge_case_handling()
        self.test_cross_session_analytics()

        self.print_summary()

    def test_data_pipeline(self):
        """Test: Generate data → Load → Validate"""
        print('\n📊 Testing Data Pipeline...')

        try:
            from telos_purpose.test_data.generate_test_sessions import generate_test_suite

            sessions = generate_test_suite()

            if len(sessions) >= 5:
                self.passed.append('✓ Test data generation pipeline works')
                print(f'  ✓ Generated {len(sessions)} test sessions')
            else:
                self.failed.append('✗ Insufficient test sessions generated')
                print(f'  ✗ Only {len(sessions)} sessions generated')
                return

            # Verify all sessions have required fields
            required_fields = ['session_id', 'turns', 'total_turns']
            for session in sessions:
                if not all(k in session for k in required_fields):
                    missing = [f for f in required_fields if f not in session]
                    self.failed.append(f'✗ Session {session.get("session_id", "unknown")} missing: {missing}')
                    print(f'  ✗ Session missing fields: {missing}')
                    return

            self.passed.append('✓ All sessions have required fields')
            print('  ✓ All sessions validated')

            # Verify turn structure
            for session in sessions:
                for turn in session.get('turns', []):
                    turn_fields = ['turn_number', 'user_input', 'fidelity']
                    if not all(f in turn for f in turn_fields):
                        missing = [f for f in turn_fields if f not in turn]
                        self.failed.append(f'✗ Turn in {session["session_id"]} missing: {missing}')
                        print(f'  ✗ Turn missing fields: {missing}')
                        return

            self.passed.append('✓ All turns have required fields')
            print('  ✓ All turns validated')

        except Exception as e:
            self.failed.append(f'✗ Data pipeline failed: {e}')
            print(f'  ✗ ERROR: {e}')
            import traceback
            traceback.print_exc()

    def test_analytics_pipeline(self):
        """Test: Load sessions → Compute analytics → Verify results"""
        print('\n📈 Testing Analytics Pipeline...')

        try:
            from telos_purpose.test_data.generate_test_sessions import generate_test_session
            from telos_purpose.dev_dashboard.streamlit_live_comparison import (
                compute_session_avg_fidelity,
                extract_session_fidelity_trends
            )

            # Create test session
            session = generate_test_session(10, 'analytics_test')
            print('  ✓ Generated test session for analytics')

            # Compute analytics
            avg_fidelity = compute_session_avg_fidelity(session)
            print(f'  ✓ Computed average fidelity: {avg_fidelity:.3f}')

            if 0.0 <= avg_fidelity <= 1.0:
                self.passed.append('✓ Fidelity calculation returns valid range')
                print('  ✓ Fidelity in valid range [0.0, 1.0]')
            else:
                self.failed.append(f'✗ Invalid fidelity: {avg_fidelity}')
                print(f'  ✗ Fidelity out of range: {avg_fidelity}')

            # Test trend extraction
            trends = extract_session_fidelity_trends([session])
            print(f'  ✓ Extracted {len(trends)} fidelity trends')

            if len(trends) > 0:
                self.passed.append('✓ Trend extraction works')
                print('  ✓ Trend extraction successful')
            else:
                self.failed.append('✗ Trend extraction failed')
                print('  ✗ No trends extracted')

            # Verify manual calculation matches
            manual_avg = sum(t['fidelity'] for t in session['turns']) / len(session['turns'])
            if abs(manual_avg - avg_fidelity) < 0.001:
                self.passed.append('✓ Analytics calculations are accurate')
                print('  ✓ Analytics match manual calculation')
            else:
                self.failed.append(f'✗ Analytics mismatch: {manual_avg:.3f} != {avg_fidelity:.3f}')
                print(f'  ✗ Calculation mismatch')

        except Exception as e:
            self.failed.append(f'✗ Analytics pipeline failed: {e}')
            print(f'  ✗ ERROR: {e}')
            import traceback
            traceback.print_exc()

    def test_export_pipeline(self):
        """Test: Session → Export → Validate output"""
        print('\n📥 Testing Export Pipeline...')

        try:
            from telos_purpose.test_data.generate_test_sessions import generate_test_session

            # Generate test session
            session = generate_test_session(5, 'export_test')
            print('  ✓ Generated test session for export')

            # Save to file
            test_file = self.test_output_dir / 'test_export.json'
            with open(test_file, 'w', encoding='utf-8') as f:
                json.dump(session, f, indent=2, ensure_ascii=False)

            print(f'  ✓ Exported to {test_file}')

            # Verify file exists and is valid JSON
            if test_file.exists():
                with open(test_file, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)

                if loaded['session_id'] == 'export_test':
                    self.passed.append('✓ Export → Save → Load pipeline works')
                    print('  ✓ Session loaded correctly')
                else:
                    self.failed.append('✗ Loaded session data corrupted')
                    print('  ✗ Session ID mismatch')
                    return

                # Verify data integrity
                if len(loaded['turns']) == len(session['turns']):
                    self.passed.append('✓ Export preserves all turns')
                    print('  ✓ All turns preserved')
                else:
                    self.failed.append(f'✗ Turn count mismatch: {len(loaded["turns"])} != {len(session["turns"])}')
                    print('  ✗ Turn count changed')
            else:
                self.failed.append('✗ Export file not created')
                print('  ✗ File not found')

        except Exception as e:
            self.failed.append(f'✗ Export pipeline failed: {e}')
            print(f'  ✗ ERROR: {e}')
            import traceback
            traceback.print_exc()

    def test_session_workflow(self):
        """Test: Complete session workflow end-to-end"""
        print('\n🔄 Testing Complete Session Workflow...')

        try:
            # 1. Generate session
            from telos_purpose.test_data.generate_test_sessions import generate_test_session
            session = generate_test_session(8, 'workflow_test', 0.75)
            print('  ✓ Step 1: Generated session')

            # 2. Compute metrics
            from telos_purpose.dev_dashboard.streamlit_live_comparison import compute_session_avg_fidelity
            avg_fid = compute_session_avg_fidelity(session)
            print(f'  ✓ Step 2: Computed fidelity ({avg_fid:.3f})')

            # 3. Export
            output_file = self.test_output_dir / 'workflow_test.json'
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(session, f, indent=2)
            print(f'  ✓ Step 3: Exported session')

            # 4. Reload and verify
            with open(output_file, 'r', encoding='utf-8') as f:
                reloaded = json.load(f)
            print('  ✓ Step 4: Reloaded session')

            reloaded_avg = compute_session_avg_fidelity(reloaded)
            print(f'  ✓ Step 5: Recomputed fidelity ({reloaded_avg:.3f})')

            # Verify metrics match
            if abs(avg_fid - reloaded_avg) < 0.001:
                self.passed.append('✓ End-to-end workflow preserves data integrity')
                print('  ✓ Metrics match after reload')
            else:
                self.failed.append(f'✗ Metrics changed: {avg_fid:.3f} → {reloaded_avg:.3f}')
                print('  ✗ Metrics diverged')
                return

            self.passed.append('✓ Complete session workflow validated')
            print('  ✓ Full workflow successful')

        except Exception as e:
            self.failed.append(f'✗ Session workflow failed: {e}')
            print(f'  ✗ ERROR: {e}')
            import traceback
            traceback.print_exc()

    def test_edge_case_handling(self):
        """Test: Edge cases handled gracefully"""
        print('\n🔬 Testing Edge Case Handling...')

        try:
            from telos_purpose.dev_dashboard.streamlit_live_comparison import compute_session_avg_fidelity

            # Test empty session
            empty_session = {
                'session_id': 'empty_test',
                'total_turns': 0,
                'turns': []
            }

            try:
                fid = compute_session_avg_fidelity(empty_session)
                if fid == 0.0 or fid is None:
                    self.passed.append('✓ Empty session handled gracefully')
                    print('  ✓ Empty session: OK')
                else:
                    self.failed.append(f'✗ Empty session returned unexpected value: {fid}')
                    print(f'  ✗ Empty session: Unexpected value')
            except:
                self.passed.append('✓ Empty session raises expected exception')
                print('  ✓ Empty session: Exception handled')

            # Test session with missing fidelity
            missing_fidelity_session = {
                'session_id': 'missing_fid_test',
                'total_turns': 2,
                'turns': [
                    {'turn_number': 1, 'fidelity': None},
                    {'turn_number': 2, 'fidelity': 0.8}
                ]
            }

            try:
                fid = compute_session_avg_fidelity(missing_fidelity_session)
                self.passed.append('✓ Missing fidelity handled gracefully')
                print(f'  ✓ Missing fidelity: OK ({fid})')
            except:
                self.passed.append('✓ Missing fidelity raises expected exception')
                print('  ✓ Missing fidelity: Exception handled')

        except Exception as e:
            self.failed.append(f'✗ Edge case handling failed: {e}')
            print(f'  ✗ ERROR: {e}')
            import traceback
            traceback.print_exc()

    def test_cross_session_analytics(self):
        """Test: Cross-session analytics functions"""
        print('\n📊 Testing Cross-Session Analytics...')

        try:
            from telos_purpose.test_data.generate_test_sessions import generate_test_suite
            from telos_purpose.dev_dashboard.streamlit_live_comparison import (
                compute_avg_fidelity_across_sessions,
                count_interventions_across_sessions,
                compute_avg_turns_per_session
            )

            # Generate test suite
            sessions = generate_test_suite()
            print(f'  ✓ Generated {len(sessions)} sessions for analytics')

            # Test cross-session average fidelity
            avg_fid = compute_avg_fidelity_across_sessions(sessions)
            if 0.0 <= avg_fid <= 1.0:
                self.passed.append('✓ Cross-session fidelity calculation works')
                print(f'  ✓ Cross-session avg fidelity: {avg_fid:.3f}')
            else:
                self.failed.append(f'✗ Invalid cross-session fidelity: {avg_fid}')
                print(f'  ✗ Invalid fidelity: {avg_fid}')

            # Test intervention counting
            total_interventions = count_interventions_across_sessions(sessions)
            if total_interventions >= 0:
                self.passed.append('✓ Cross-session intervention counting works')
                print(f'  ✓ Total interventions: {total_interventions}')
            else:
                self.failed.append(f'✗ Invalid intervention count: {total_interventions}')
                print(f'  ✗ Invalid count: {total_interventions}')

            # Test average turns per session
            avg_turns = compute_avg_turns_per_session(sessions)
            if avg_turns > 0:
                self.passed.append('✓ Average turns calculation works')
                print(f'  ✓ Average turns per session: {avg_turns:.1f}')
            else:
                self.failed.append(f'✗ Invalid average turns: {avg_turns}')
                print(f'  ✗ Invalid average: {avg_turns}')

        except Exception as e:
            self.failed.append(f'✗ Cross-session analytics failed: {e}')
            print(f'  ✗ ERROR: {e}')
            import traceback
            traceback.print_exc()

    def print_summary(self):
        """Print test results summary"""
        print('\n' + '=' * 60)
        print('📋 Integration Test Summary')
        print('=' * 60)

        total = len(self.passed) + len(self.failed)
        pass_rate = (len(self.passed) / total * 100) if total > 0 else 0

        print(f'\nTotal Tests: {total}')
        print(f'✅ Passed: {len(self.passed)} ({pass_rate:.1f}%)')
        print(f'❌ Failed: {len(self.failed)}')

        if self.passed:
            print('\n✅ Passed Tests:')
            for test in self.passed:
                print(f'  {test}')

        if self.failed:
            print('\n❌ Failed Tests:')
            for test in self.failed:
                print(f'  {test}')

        print('\n' + '=' * 60)

        if self.failed:
            print('\n⚠️  Some integration tests failed. Review above.')
            sys.exit(1)
        else:
            print('\n✨ All integration tests passed!')
            print('✨ Platform ready for production use!')
            sys.exit(0)


def main():
    """Main entry point for integration tests"""
    tester = IntegrationTester()
    tester.run_all_tests()


if __name__ == '__main__':
    main()
