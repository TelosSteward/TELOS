"""
Performance Analysis Tool for TELOS Observatory
===============================================

Analyzes platform performance for production readiness:
- Import time
- Data loading speed
- Analytics computation performance
- Memory usage

Usage:
    python -m telos_purpose.validation.performance_check
"""

import time
import tracemalloc
from pathlib import Path
import sys
from typing import List, Tuple


class PerformanceAnalyzer:
    """Analyze platform performance for optimization"""

    def __init__(self):
        self.results: List[Tuple[str, str, str]] = []

    def run_analysis(self):
        """Execute all performance checks"""

        print('\n⚡ TELOS Observatory Performance Analysis')
        print('=' * 60)

        self.check_import_time()
        self.check_data_loading_speed()
        self.check_analytics_performance()
        self.check_memory_usage()
        self.check_export_performance()

        self.print_summary()

    def check_import_time(self):
        """Measure module import time"""
        print('\n📦 Checking Import Performance...')

        start = time.time()
        try:
            from telos_purpose.dev_dashboard import streamlit_live_comparison
            elapsed = time.time() - start

            print(f'  ⏱  Dashboard import took {elapsed:.2f}s')

            if elapsed < 2.0:
                self.results.append(('✓ Dashboard import time', f'{elapsed:.2f}s', 'Good'))
                print(f'  ✓ Good performance')
            elif elapsed < 5.0:
                self.results.append(('⚠ Dashboard import time', f'{elapsed:.2f}s', 'Acceptable'))
                print(f'  ⚠ Acceptable performance')
            else:
                self.results.append(('✗ Dashboard import time', f'{elapsed:.2f}s', 'Slow'))
                print(f'  ✗ Slow - needs optimization')
        except Exception as e:
            self.results.append(('✗ Dashboard import', 'Failed', str(e)))
            print(f'  ✗ ERROR: {e}')

    def check_data_loading_speed(self):
        """Measure session data loading performance"""
        print('\n💾 Checking Data Loading Speed...')

        try:
            from telos_purpose.test_data.generate_test_sessions import generate_test_session

            # Generate test session
            start = time.time()
            session = generate_test_session(50, 'perf_test')  # 50 turn session
            elapsed = time.time() - start

            print(f'  ⏱  Generated 50-turn session in {elapsed:.3f}s')

            if elapsed < 0.5:
                self.results.append(('✓ Session generation (50 turns)', f'{elapsed:.3f}s', 'Fast'))
                print(f'  ✓ Fast generation')
            elif elapsed < 1.0:
                self.results.append(('⚠ Session generation (50 turns)', f'{elapsed:.3f}s', 'Acceptable'))
                print(f'  ⚠ Acceptable speed')
            else:
                self.results.append(('✗ Session generation (50 turns)', f'{elapsed:.3f}s', 'Slow'))
                print(f'  ✗ Slow - needs optimization')

            # Test turn generation rate
            turns_per_sec = 50 / elapsed
            print(f'  📊 Generation rate: {turns_per_sec:.1f} turns/second')

        except Exception as e:
            self.results.append(('✗ Data loading', 'Failed', str(e)))
            print(f'  ✗ ERROR: {e}')
            import traceback
            traceback.print_exc()

    def check_analytics_performance(self):
        """Measure analytics computation speed"""
        print('\n📊 Checking Analytics Performance...')

        try:
            from telos_purpose.test_data.generate_test_sessions import generate_test_suite
            from telos_purpose.dev_dashboard.streamlit_live_comparison import (
                compute_avg_fidelity_across_sessions,
                count_interventions_across_sessions
            )

            # Generate test sessions
            print('  📦 Generating test suite...')
            sessions = generate_test_suite()
            print(f'  ✓ Generated {len(sessions)} sessions')

            # Measure analytics computation
            start = time.time()
            avg_fid = compute_avg_fidelity_across_sessions(sessions)
            interventions = count_interventions_across_sessions(sessions)
            elapsed = time.time() - start

            print(f'  ⏱  Analytics computed in {elapsed:.3f}s')
            print(f'  📈 Avg fidelity: {avg_fid:.3f}, Interventions: {interventions}')

            if elapsed < 0.1:
                self.results.append(('✓ Analytics computation', f'{elapsed:.3f}s', 'Fast'))
                print(f'  ✓ Fast computation')
            elif elapsed < 0.5:
                self.results.append(('⚠ Analytics computation', f'{elapsed:.3f}s', 'Acceptable'))
                print(f'  ⚠ Acceptable speed')
            else:
                self.results.append(('✗ Analytics computation', f'{elapsed:.3f}s', 'Slow'))
                print(f'  ✗ Slow - needs optimization')
        except Exception as e:
            self.results.append(('✗ Analytics performance', 'Failed', str(e)))
            print(f'  ✗ ERROR: {e}')
            import traceback
            traceback.print_exc()

    def check_memory_usage(self):
        """Measure memory footprint"""
        print('\n🧠 Checking Memory Usage...')

        try:
            tracemalloc.start()

            # Load test data
            print('  📦 Loading test data...')
            from telos_purpose.test_data.generate_test_sessions import generate_test_suite
            sessions = generate_test_suite()

            # Compute analytics multiple times
            print('  🔄 Running analytics 10 times...')
            from telos_purpose.dev_dashboard.streamlit_live_comparison import (
                compute_avg_fidelity_across_sessions
            )
            for i in range(10):
                compute_avg_fidelity_across_sessions(sessions)

            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            current_mb = current / 1024 / 1024
            peak_mb = peak / 1024 / 1024

            print(f'  📊 Current memory: {current_mb:.1f} MB')
            print(f'  📊 Peak memory: {peak_mb:.1f} MB')

            if peak_mb < 50:
                self.results.append(('✓ Peak memory usage', f'{peak_mb:.1f} MB', 'Good'))
                print(f'  ✓ Low memory usage')
            elif peak_mb < 100:
                self.results.append(('⚠ Peak memory usage', f'{peak_mb:.1f} MB', 'Acceptable'))
                print(f'  ⚠ Moderate memory usage')
            else:
                self.results.append(('✗ Peak memory usage', f'{peak_mb:.1f} MB', 'High'))
                print(f'  ✗ High memory usage - needs optimization')
        except Exception as e:
            self.results.append(('✗ Memory check', 'Failed', str(e)))
            print(f'  ✗ ERROR: {e}')
            import traceback
            traceback.print_exc()

    def check_export_performance(self):
        """Measure export operation speed"""
        print('\n📥 Checking Export Performance...')

        try:
            import json
            import tempfile
            from telos_purpose.test_data.generate_test_sessions import generate_test_session

            # Generate test session
            session = generate_test_session(20, 'export_perf_test')

            # Measure JSON export
            start = time.time()
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=True) as f:
                json.dump(session, f, indent=2)
                f.flush()
            elapsed = time.time() - start

            print(f'  ⏱  JSON export (20 turns) took {elapsed:.3f}s')

            if elapsed < 0.1:
                self.results.append(('✓ JSON export (20 turns)', f'{elapsed:.3f}s', 'Fast'))
                print(f'  ✓ Fast export')
            elif elapsed < 0.5:
                self.results.append(('⚠ JSON export (20 turns)', f'{elapsed:.3f}s', 'Acceptable'))
                print(f'  ⚠ Acceptable speed')
            else:
                self.results.append(('✗ JSON export (20 turns)', f'{elapsed:.3f}s', 'Slow'))
                print(f'  ✗ Slow - needs optimization')

        except Exception as e:
            self.results.append(('✗ Export performance', 'Failed', str(e)))
            print(f'  ✗ ERROR: {e}')

    def print_summary(self):
        """Print performance analysis summary"""
        print('\n' + '=' * 60)
        print('📋 Performance Analysis Summary')
        print('=' * 60)

        for test, value, status in self.results:
            status_icon = '✓' if status in ['Good', 'Fast'] else ('⚠' if status == 'Acceptable' else '✗')
            print(f'{status_icon} {test}: {value} ({status})')

        print('\n' + '=' * 60)

        # Overall assessment
        failures = sum(1 for _, _, s in self.results if '✗' in str(s) or s == 'Failed')
        warnings = sum(1 for _, _, s in self.results if '⚠' in str(s) or s == 'Acceptable')
        successes = sum(1 for _, _, s in self.results if '✓' in str(s) or s in ['Good', 'Fast'])

        total = len(self.results)
        print(f'\n📊 Results: {successes} Good, {warnings} Acceptable, {failures} Need Optimization')
        print(f'   Total Checks: {total}')

        if failures == 0 and warnings == 0:
            print('\n✨ Excellent performance! All checks passed.')
            sys.exit(0)
        elif failures == 0:
            print(f'\n⚠️  Performance acceptable with {warnings} warnings.')
            print('   Consider optimization for best user experience.')
            sys.exit(0)
        else:
            print(f'\n⚠️  Performance needs optimization: {failures} issues found.')
            print('   Review slow operations and apply optimizations.')
            sys.exit(1)


def main():
    """Main entry point for performance analysis"""
    analyzer = PerformanceAnalyzer()
    analyzer.run_analysis()


if __name__ == '__main__':
    main()
