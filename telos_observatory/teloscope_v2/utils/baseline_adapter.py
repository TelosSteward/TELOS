"""
Baseline Adapter for Observatory v2

Wraps telos_purpose.validation.baseline_runners for Observatory integration.

Provides simplified interface for running baseline comparisons (TELOS vs ungoverned)
in the Observatory testing environment.

Usage:
    from teloscope_v2.utils.baseline_adapter import BaselineAdapter

    adapter = BaselineAdapter(llm_client, embedding_provider, attractor_config)

    # Run comparison
    results = adapter.run_comparison(conversation)

    # Display
    st.metric("Baseline F", results['baseline'].final_metrics['fidelity'])
    st.metric("TELOS F", results['telos'].final_metrics['fidelity'])
"""

from __future__ import annotations
from typing import List, Tuple, Dict, Any, Optional
import sys
from pathlib import Path

# Add telos_purpose to path for imports
telos_root = Path(__file__).parent.parent.parent.parent / 'telos_purpose'
if str(telos_root) not in sys.path:
    sys.path.insert(0, str(telos_root))

try:
    from telos_purpose.validation.baseline_runners import (
        BaselineRunner,
        StatelessRunner,
        PromptOnlyRunner,
        CadenceReminderRunner,
        ObservationRunner,
        TELOSRunner,
        BaselineResult
    )
    BASELINE_RUNNERS_AVAILABLE = True
except ImportError as e:
    BASELINE_RUNNERS_AVAILABLE = False
    print(f"Warning: Could not import baseline_runners: {e}")


class BaselineAdapter:
    """
    Adapter for running baselines in Observatory v2.

    Simplifies baseline runner usage for Observatory UI by providing:
    - Pre-instantiated runners for all baseline types
    - Simplified comparison interface (baseline vs TELOS)
    - Consistent error handling
    - Integration with Observatory mock data format
    """

    def __init__(
        self,
        llm_client,
        embedding_provider,
        attractor_config
    ):
        """
        Initialize adapter with dependencies.

        Args:
            llm_client: LLM client for generation
            embedding_provider: Text-to-vector encoder
            attractor_config: PrimacyAttractor configuration
        """
        if not BASELINE_RUNNERS_AVAILABLE:
            raise ImportError(
                "baseline_runners not available. Ensure telos_purpose is installed."
            )

        self.llm = llm_client
        self.embeddings = embedding_provider
        self.attractor = attractor_config

        # Pre-instantiate all runners
        self.runners = self._create_runners()

    def _create_runners(self) -> Dict[str, BaselineRunner]:
        """
        Create all baseline runners.

        Returns:
            Dict mapping baseline type to runner instance
        """
        return {
            'stateless': StatelessRunner(
                self.llm,
                self.embeddings,
                self.attractor
            ),
            'prompt_only': PromptOnlyRunner(
                self.llm,
                self.embeddings,
                self.attractor
            ),
            'cadence': CadenceReminderRunner(
                self.llm,
                self.embeddings,
                self.attractor,
                reminder_cadence=3
            ),
            'observation': ObservationRunner(
                self.llm,
                self.embeddings,
                self.attractor
            ),
            'telos': TELOSRunner(
                self.llm,
                self.embeddings,
                self.attractor
            )
        }

    def run_baseline(
        self,
        baseline_type: str,
        conversation: List[Tuple[str, str]],
        track_timing: bool = True,
        track_calibration: bool = True
    ) -> BaselineResult:
        """
        Run a single baseline with enhanced runtime tracking.

        Args:
            baseline_type: One of 'stateless', 'prompt_only', 'cadence',
                          'observation', 'telos'
            conversation: List of (user_input, expected_response) tuples
                         Note: expected_response can be empty string
            track_timing: If True, add processing_time_ms per turn
            track_calibration: If True, add calibration_phase tracking

        Returns:
            BaselineResult with complete telemetry + runtime tracking

        Raises:
            ValueError: If baseline_type is unknown

        Note: Runtime simulation verified - sequential processing only.
        """
        if baseline_type not in self.runners:
            raise ValueError(
                f"Unknown baseline type: {baseline_type}. "
                f"Available: {list(self.runners.keys())}"
            )

        import time

        runner = self.runners[baseline_type]

        # Track total execution time
        start_time = time.time()

        # Run baseline (sequential processing)
        result = runner.run_conversation(conversation)

        # Calculate total elapsed time
        total_elapsed_ms = (time.time() - start_time) * 1000

        # Enhance results with timing and calibration tracking
        if track_timing or track_calibration:
            result = self._enhance_result_with_tracking(
                result,
                baseline_type,
                track_timing,
                track_calibration,
                total_elapsed_ms,
                session_start_time=start_time  # Pass session start for accurate first turn timing
            )

        return result

    def _enhance_result_with_tracking(
        self,
        result: BaselineResult,
        baseline_type: str,
        track_timing: bool,
        track_calibration: bool,
        total_elapsed_ms: float,
        session_start_time: Optional[float] = None
    ) -> BaselineResult:
        """
        Enhance baseline result with runtime tracking metadata.

        Args:
            result: BaselineResult from runner
            baseline_type: Type of baseline
            track_timing: Add timing data
            track_calibration: Add calibration phase tracking
            total_elapsed_ms: Total execution time
            session_start_time: Session start timestamp (for accurate first turn timing)

        Returns:
            Enhanced BaselineResult
        """
        # Calculate timing per turn (actual from timestamps if available, else estimate)
        if track_timing and result.turn_results:
            # Try to calculate actual per-turn timing from timestamps
            timestamps = [turn.get('timestamp') for turn in result.turn_results]

            # If all timestamps present, calculate actual timing
            if all(ts is not None for ts in timestamps) and len(timestamps) > 0:
                cumulative_ms = 0.0

                for i, turn in enumerate(result.turn_results):
                    if i == 0:
                        # First turn: calculate from session start to first timestamp
                        if session_start_time is not None:
                            # Actual: time from session start to first turn completion
                            turn_time_ms = (timestamps[0] - session_start_time) * 1000
                        elif len(timestamps) > 1:
                            # Fallback: estimate as same duration as second turn
                            turn_time_ms = (timestamps[1] - timestamps[0]) * 1000
                        else:
                            # Single turn: use total time
                            turn_time_ms = total_elapsed_ms
                    else:
                        # Subsequent turns: delta from previous timestamp (ACTUAL)
                        turn_time_ms = (timestamps[i] - timestamps[i-1]) * 1000

                    turn['processing_time_ms'] = turn_time_ms
                    cumulative_ms += turn_time_ms
                    turn['cumulative_time_ms'] = cumulative_ms
            else:
                # Fallback: Estimate MS per turn (total / turns) if timestamps unavailable
                avg_ms_per_turn = total_elapsed_ms / len(result.turn_results)

                for i, turn in enumerate(result.turn_results):
                    if 'processing_time_ms' not in turn:
                        turn['processing_time_ms'] = avg_ms_per_turn
                        turn['cumulative_time_ms'] = avg_ms_per_turn * (i + 1)

        # Add calibration phase tracking (first 3 turns typically)
        if track_calibration:
            CALIBRATION_TURNS = 3  # Standard calibration window

            for i, turn in enumerate(result.turn_results):
                turn_num = turn.get('turn', i + 1)

                # Mark calibration phase
                turn['calibration_phase'] = turn_num <= CALIBRATION_TURNS
                turn['calibration_turns_remaining'] = max(0, CALIBRATION_TURNS - turn_num + 1)

                # Attractor established after calibration
                turn['primacy_attractor_established'] = turn_num > CALIBRATION_TURNS

        # Add context size tracking for runtime verification
        for i, turn in enumerate(result.turn_results):
            # Turn N has N turns in history (0 to N-1)
            turn['context_size'] = i

        # Add metadata
        result.metadata['total_processing_time_ms'] = total_elapsed_ms
        result.metadata['runtime_simulation_verified'] = True
        result.metadata['processing_pattern'] = 'sequential'

        return result

    def run_comparison(
        self,
        conversation: List[Tuple[str, str]],
        baseline_type: str = 'stateless',
        track_timing: bool = True,
        track_calibration: bool = True
    ) -> Dict[str, BaselineResult]:
        """
        Run baseline vs TELOS comparison with runtime tracking.

        This is the primary method for Observatory counterfactual analysis.
        Generates two independent branches using RUNTIME SIMULATION:
        - Baseline: Specified baseline type (default: stateless)
        - TELOS: Full governance with MBL

        Both paths use sequential processing (no batch analysis).

        Args:
            conversation: Conversation to test
            baseline_type: Type of baseline to compare against
                          (default: 'stateless' = no governance)
            track_timing: Add timing data per turn
            track_calibration: Add calibration phase tracking

        Returns:
            Dict with 'baseline' and 'telos' results, including:
            - processing_time_ms per turn
            - calibration_phase flags
            - context_size for runtime verification
            - total_processing_time_ms

        Example:
            results = adapter.run_comparison(conversation)
            baseline_f = results['baseline'].final_metrics['fidelity']
            telos_f = results['telos'].final_metrics['fidelity']
            delta_f = telos_f - baseline_f

            # Get timing comparison
            baseline_ms = results['baseline'].metadata['total_processing_time_ms']
            telos_ms = results['telos'].metadata['total_processing_time_ms']
        """
        return {
            'baseline': self.run_baseline(
                baseline_type,
                conversation,
                track_timing=track_timing,
                track_calibration=track_calibration
            ),
            'telos': self.run_baseline(
                'telos',
                conversation,
                track_timing=track_timing,
                track_calibration=track_calibration
            )
        }

    def run_all_baselines(
        self,
        conversation: List[Tuple[str, str]]
    ) -> Dict[str, BaselineResult]:
        """
        Run all baseline types for comprehensive comparison.

        Useful for research papers showing TELOS vs multiple baselines.

        Args:
            conversation: Conversation to test

        Returns:
            Dict mapping baseline type to result
        """
        results = {}
        for baseline_type in self.runners.keys():
            try:
                results[baseline_type] = self.run_baseline(
                    baseline_type,
                    conversation
                )
            except Exception as e:
                print(f"Warning: {baseline_type} failed: {e}")
                results[baseline_type] = None

        return results

    def get_available_baselines(self) -> List[str]:
        """
        Get list of available baseline types.

        Returns:
            List of baseline type strings
        """
        return list(self.runners.keys())

    def convert_session_to_conversation(
        self,
        session: Dict[str, Any]
    ) -> List[Tuple[str, str]]:
        """
        Convert Observatory session format to conversation format.

        Converts from:
            {'turns': [{'user_input': ..., 'assistant_response': ...}, ...]}

        To:
            [(user_input, assistant_response), ...]

        Args:
            session: Observatory session dict

        Returns:
            Conversation list of tuples
        """
        conversation = []

        for turn in session.get('turns', []):
            user_input = turn.get('user_input', '')
            assistant_response = turn.get('assistant_response', '')

            if user_input:  # Only include if user input exists
                conversation.append((user_input, assistant_response))

        return conversation

    def format_result_for_display(
        self,
        result: BaselineResult
    ) -> Dict[str, Any]:
        """
        Format BaselineResult for Observatory display.

        Extracts key metrics for UI rendering.

        Args:
            result: BaselineResult from run_baseline()

        Returns:
            Dict with formatted metrics
        """
        return {
            'runner_type': result.runner_type,
            'session_id': result.session_id,
            'final_fidelity': result.final_metrics.get('fidelity', 0.0),
            'avg_distance': result.final_metrics.get('avg_distance', 0.0),
            'basin_adherence': result.final_metrics.get('basin_adherence', 0.0),
            'total_turns': result.metadata.get('total_turns', 0),
            'total_interventions': result.metadata.get('total_interventions', 0),
            'intervention_rate': result.metadata.get('intervention_rate', 0.0),
            'turn_results': result.turn_results
        }


def check_baseline_availability() -> bool:
    """
    Check if baseline runners are available.

    Returns:
        True if available, False otherwise
    """
    return BASELINE_RUNNERS_AVAILABLE


def get_baseline_info() -> Dict[str, str]:
    """
    Get information about available baselines.

    Returns:
        Dict mapping baseline type to description
    """
    return {
        'stateless': 'No governance memory (null hypothesis)',
        'prompt_only': 'Constraints stated once at start',
        'cadence': 'Fixed-interval reminders (every 3 turns)',
        'observation': 'Full math active, no interventions',
        'telos': 'Full MBL (SPC Engine + Proportional Controller)'
    }
