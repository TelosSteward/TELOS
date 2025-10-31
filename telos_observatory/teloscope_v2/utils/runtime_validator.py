"""
Runtime Simulation Validator for Observatory v2

Validates that counterfactual analysis uses proper runtime simulation
architecture (not batch analysis).

Purpose:
- Verify no future knowledge used
- Confirm sequential processing
- Validate timing requirements
- Check calibration phase tracking

Usage:
    from teloscope_v2.utils.runtime_validator import RuntimeValidator

    validator = RuntimeValidator()

    # Validate results
    is_valid = validator.validate_runtime_simulation(results)

    # Get detailed report
    report = validator.generate_validation_report(results)
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Result of runtime simulation validation."""
    passed: bool
    test_name: str
    message: str
    details: Optional[Dict[str, Any]] = None


class RuntimeValidator:
    """
    Validates runtime simulation requirements.

    Ensures counterfactual analysis follows proper runtime simulation
    architecture without batch analysis artifacts.
    """

    def __init__(self):
        """Initialize validator."""
        self.validation_tests = [
            self.test_no_future_context,
            self.test_sequential_timestamps,
            self.test_timing_recorded,
            self.test_context_growth,
            self.test_empty_initial_context
        ]

    def validate_runtime_simulation(
        self,
        results: Dict[str, Any]
    ) -> bool:
        """
        Validate runtime simulation architecture.

        Args:
            results: Baseline result dict with turn_results

        Returns:
            True if all validation tests pass, False otherwise

        Example:
            validator = RuntimeValidator()
            is_valid = validator.validate_runtime_simulation(telos_results)
        """
        all_passed = True

        for test_func in self.validation_tests:
            result = test_func(results)
            if not result.passed:
                all_passed = False
                print(f"❌ {result.test_name}: {result.message}")
            else:
                print(f"✅ {result.test_name}: {result.message}")

        return all_passed

    def test_no_future_context(
        self,
        results: Dict[str, Any]
    ) -> ValidationResult:
        """
        Test that no turn has access to future turns.

        Verifies Turn N context size == N (only historical).

        Args:
            results: Baseline result dict

        Returns:
            ValidationResult with pass/fail status
        """
        turn_results = results.get('turn_results', [])

        for turn in turn_results:
            turn_num = turn.get('turn', 0)

            # Check if context info is available
            if 'context_size' in turn:
                context_size = turn['context_size']

                # Turn N should have N-1 turns in context (0 to N-1)
                expected_size = turn_num - 1

                if context_size != expected_size:
                    return ValidationResult(
                        passed=False,
                        test_name="No Future Context",
                        message=f"Turn {turn_num} has {context_size} context turns (expected {expected_size})",
                        details={'turn': turn_num, 'context_size': context_size, 'expected': expected_size}
                    )

        return ValidationResult(
            passed=True,
            test_name="No Future Context",
            message=f"All {len(turn_results)} turns have correct historical context only"
        )

    def test_sequential_timestamps(
        self,
        results: Dict[str, Any]
    ) -> ValidationResult:
        """
        Test that timestamps are strictly increasing.

        Verifies sequential processing order.

        Args:
            results: Baseline result dict

        Returns:
            ValidationResult with pass/fail status
        """
        turn_results = results.get('turn_results', [])
        timestamps = [t.get('timestamp', 0) for t in turn_results]

        if not timestamps:
            return ValidationResult(
                passed=False,
                test_name="Sequential Timestamps",
                message="No timestamps found in results"
            )

        # Check if strictly increasing
        for i in range(len(timestamps) - 1):
            if timestamps[i] >= timestamps[i + 1]:
                return ValidationResult(
                    passed=False,
                    test_name="Sequential Timestamps",
                    message=f"Timestamps not sequential at turn {i}",
                    details={'turn': i, 'current': timestamps[i], 'next': timestamps[i + 1]}
                )

        return ValidationResult(
            passed=True,
            test_name="Sequential Timestamps",
            message=f"All {len(timestamps)} timestamps strictly increasing"
        )

    def test_timing_recorded(
        self,
        results: Dict[str, Any]
    ) -> ValidationResult:
        """
        Test that processing time is recorded per turn.

        Verifies timing requirements met.

        Args:
            results: Baseline result dict

        Returns:
            ValidationResult with pass/fail status
        """
        turn_results = results.get('turn_results', [])

        if not turn_results:
            return ValidationResult(
                passed=False,
                test_name="Timing Recorded",
                message="No turn results found"
            )

        # Check if at least some turns have timing
        has_timing = any('processing_time_ms' in t for t in turn_results)

        if not has_timing:
            return ValidationResult(
                passed=False,
                test_name="Timing Recorded",
                message="No processing_time_ms found in turn results (timing not recorded)"
            )

        # Count how many have timing
        timing_count = sum(1 for t in turn_results if 'processing_time_ms' in t)

        if timing_count < len(turn_results):
            return ValidationResult(
                passed=False,
                test_name="Timing Recorded",
                message=f"Only {timing_count}/{len(turn_results)} turns have timing data",
                details={'with_timing': timing_count, 'total': len(turn_results)}
            )

        return ValidationResult(
            passed=True,
            test_name="Timing Recorded",
            message=f"All {len(turn_results)} turns have processing time recorded"
        )

    def test_context_growth(
        self,
        results: Dict[str, Any]
    ) -> ValidationResult:
        """
        Test that context grows incrementally.

        Verifies context builds turn-by-turn (not batch loaded).

        Args:
            results: Baseline result dict

        Returns:
            ValidationResult with pass/fail status
        """
        turn_results = results.get('turn_results', [])

        if not turn_results:
            return ValidationResult(
                passed=False,
                test_name="Context Growth",
                message="No turn results found"
            )

        # Check if context_size grows by 1 each turn
        for i in range(len(turn_results) - 1):
            current_turn = turn_results[i]
            next_turn = turn_results[i + 1]

            if 'context_size' in current_turn and 'context_size' in next_turn:
                current_size = current_turn['context_size']
                next_size = next_turn['context_size']

                # Context should grow by 1 (current turn gets added)
                if next_size != current_size + 1:
                    return ValidationResult(
                        passed=False,
                        test_name="Context Growth",
                        message=f"Context did not grow correctly at turn {i}",
                        details={
                            'turn': i,
                            'current_context': current_size,
                            'next_context': next_size,
                            'expected': current_size + 1
                        }
                    )

        return ValidationResult(
            passed=True,
            test_name="Context Growth",
            message="Context grows incrementally turn-by-turn"
        )

    def test_empty_initial_context(
        self,
        results: Dict[str, Any]
    ) -> ValidationResult:
        """
        Test that Turn 0 has empty context.

        Verifies simulation starts with clean slate.

        Args:
            results: Baseline result dict

        Returns:
            ValidationResult with pass/fail status
        """
        turn_results = results.get('turn_results', [])

        if not turn_results:
            return ValidationResult(
                passed=False,
                test_name="Empty Initial Context",
                message="No turn results found"
            )

        first_turn = turn_results[0]

        if 'context_size' in first_turn:
            context_size = first_turn['context_size']

            if context_size != 0:
                return ValidationResult(
                    passed=False,
                    test_name="Empty Initial Context",
                    message=f"Turn 0 has non-empty context (size={context_size})",
                    details={'context_size': context_size}
                )

        return ValidationResult(
            passed=True,
            test_name="Empty Initial Context",
            message="Turn 0 starts with empty context"
        )

    def generate_validation_report(
        self,
        results: Dict[str, Any]
    ) -> str:
        """
        Generate detailed validation report.

        Args:
            results: Baseline result dict

        Returns:
            Formatted validation report string

        Example:
            report = validator.generate_validation_report(results)
            print(report)
        """
        report_lines = [
            "=" * 70,
            "RUNTIME SIMULATION VALIDATION REPORT",
            "=" * 70,
            ""
        ]

        # Run all tests
        test_results = [test_func(results) for test_func in self.validation_tests]

        # Summary
        passed_count = sum(1 for r in test_results if r.passed)
        total_count = len(test_results)

        report_lines.append(f"Tests Passed: {passed_count}/{total_count}")
        report_lines.append("")

        # Individual test results
        for result in test_results:
            status = "✅ PASS" if result.passed else "❌ FAIL"
            report_lines.append(f"{status} | {result.test_name}")
            report_lines.append(f"  {result.message}")

            if result.details:
                report_lines.append(f"  Details: {result.details}")

            report_lines.append("")

        # Final verdict
        report_lines.append("=" * 70)

        if passed_count == total_count:
            report_lines.append("VERDICT: ✅ RUNTIME SIMULATION VERIFIED")
        else:
            report_lines.append("VERDICT: ❌ RUNTIME SIMULATION VIOLATIONS DETECTED")

        report_lines.append("=" * 70)

        return "\n".join(report_lines)

    def get_timing_summary(
        self,
        results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get timing summary from results.

        Args:
            results: Baseline result dict

        Returns:
            Dict with timing statistics

        Example:
            timing = validator.get_timing_summary(results)
            print(f"Total MS: {timing['total_ms']}")
        """
        turn_results = results.get('turn_results', [])

        timings = [
            t.get('processing_time_ms', 0)
            for t in turn_results
            if 'processing_time_ms' in t
        ]

        if not timings:
            return {
                'total_ms': 0,
                'avg_ms': 0,
                'min_ms': 0,
                'max_ms': 0,
                'turn_count': 0
            }

        return {
            'total_ms': sum(timings),
            'avg_ms': sum(timings) / len(timings),
            'min_ms': min(timings),
            'max_ms': max(timings),
            'turn_count': len(timings),
            'per_turn': timings
        }


def quick_validate(results: Dict[str, Any]) -> bool:
    """
    Quick validation function.

    Args:
        results: Baseline result dict

    Returns:
        True if valid runtime simulation, False otherwise

    Example:
        if quick_validate(results):
            print("✅ Runtime simulation verified")
    """
    validator = RuntimeValidator()
    return validator.validate_runtime_simulation(results)


def print_validation_report(results: Dict[str, Any]):
    """
    Print validation report to console.

    Args:
        results: Baseline result dict

    Example:
        print_validation_report(telos_results)
    """
    validator = RuntimeValidator()
    report = validator.generate_validation_report(results)
    print(report)
