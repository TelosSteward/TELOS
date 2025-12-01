"""
PA Comparison Runner

Runs dual PA governance over existing single PA datasets to compare fidelity scores.
Provides empirical evidence for dual PA adoption decisions.

Usage:
    python -m telos_purpose.analysis.pa_comparison_runner \
        --dataset saved_sessions/session_*.json \
        --output comparison_results.json

Status: Experimental (v1.2-dual-attractor)
"""

from __future__ import annotations
import asyncio
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import logging
import numpy as np
from anthropic import Anthropic

from telos_purpose.core.governance_config import (
    GovernanceConfig,
    GovernanceMode,
    ComparisonMetrics
)
from telos_purpose.core.dual_attractor import (
    create_dual_pa,
    check_dual_pa_fidelity,
    DualPrimacyAttractor
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ComparisonResult:
    """Result from comparing single PA vs dual PA on a dataset."""

    session_id: str
    total_turns: int

    # Single PA metrics (from original dataset)
    single_pa_avg_fidelity: float
    single_pa_pass_rate: float
    single_pa_interventions: int

    # Dual PA metrics (from replay)
    dual_pa_avg_user_fidelity: float
    dual_pa_avg_ai_fidelity: Optional[float]
    dual_pa_pass_rate: float
    dual_pa_interventions: int
    dual_pa_correlation: Optional[float]
    dual_pa_mode_used: str  # 'dual' or 'single' (fallback)

    # Performance
    replay_time_seconds: float
    avg_turn_time_ms: float

    # Errors
    errors_encountered: int
    fallbacks_used: int

    # Detailed metrics per turn
    turn_metrics: List[Dict[str, Any]]

    def improvement_summary(self) -> Dict[str, Any]:
        """Calculate improvement metrics."""
        fidelity_improvement = self.dual_pa_avg_user_fidelity - self.single_pa_avg_fidelity
        pass_rate_improvement = self.dual_pa_pass_rate - self.single_pa_pass_rate
        intervention_reduction = self.single_pa_interventions - self.dual_pa_interventions

        return {
            'session_id': self.session_id,
            'fidelity_improvement': fidelity_improvement,
            'pass_rate_improvement': pass_rate_improvement,
            'intervention_reduction': intervention_reduction,
            'dual_pa_effective': self.dual_pa_mode_used == 'dual',
            'correlation': self.dual_pa_correlation,
            'summary': (
                f"Fidelity: {fidelity_improvement:+.3f}, "
                f"Pass rate: {pass_rate_improvement:+.1%}, "
                f"Interventions: {intervention_reduction:+d}"
            )
        }


class PAComparisonRunner:
    """
    Runs PA mode comparison on datasets.

    Loads saved sessions with single PA results, replays with dual PA,
    and generates comparison metrics.
    """

    def __init__(
        self,
        client: Anthropic,
        config: Optional[GovernanceConfig] = None
    ):
        """
        Initialize comparison runner.

        Args:
            client: Anthropic client for LLM calls
            config: Governance config (defaults to dual PA mode)
        """
        self.client = client
        self.config = config or GovernanceConfig.dual_pa_config()

    async def replay_session_with_dual_pa(
        self,
        session_data: Dict[str, Any]
    ) -> ComparisonResult:
        """
        Replay a single session with dual PA governance.

        Args:
            session_data: Loaded session JSON from saved_sessions/

        Returns:
            ComparisonResult with metrics from both PA modes
        """
        start_time = time.time()

        session_id = session_data.get('session_id', 'unknown')
        turns = session_data.get('turns', [])
        pa_config = session_data.get('pa_config', {})

        logger.info(f"Replaying session {session_id} with {len(turns)} turns")

        # Extract single PA metrics from original data
        single_pa_avg_fidelity = session_data.get('avg_fidelity', 0.0)
        single_pa_interventions = session_data.get('total_interventions', 0)

        # Calculate single PA pass rate
        single_pa_passes = sum(1 for turn in turns if turn.get('fidelity', 0.0) >= 0.65)
        single_pa_pass_rate = single_pa_passes / len(turns) if turns else 0.0

        # Create user PA from session config
        user_pa = {
            'purpose': pa_config.get('purpose', []),
            'scope': pa_config.get('scope', []),
            'boundaries': pa_config.get('boundaries', []),
            'constraint_tolerance': pa_config.get('constraint_tolerance', 0.2),
            'privacy_level': pa_config.get('privacy_level', 0.8),
            'task_priority': pa_config.get('task_priority', 0.7),
            'fidelity_threshold': pa_config.get('fidelity_threshold', 0.65)
        }

        # Create dual PA (with iron-clad error handling)
        dual_pa = None
        correlation = None
        dual_pa_mode_used = 'single'  # Default fallback

        try:
            dual_pa = await asyncio.wait_for(
                create_dual_pa(user_pa, self.client, enable_dual_mode=True),
                timeout=self.config.derivation_timeout_seconds
            )
            correlation = dual_pa.correlation
            dual_pa_mode_used = dual_pa.governance_mode

            logger.info(
                f"Dual PA created: mode={dual_pa_mode_used}, "
                f"correlation={correlation:.2f if correlation else 'N/A'}"
            )

        except asyncio.TimeoutError:
            logger.error(f"Dual PA derivation timeout for session {session_id}")
            if self.config.strict_mode:
                raise
            # Fallback to single PA mode
            dual_pa = DualPrimacyAttractor(user_pa=user_pa, governance_mode='single')
            dual_pa_mode_used = 'single'

        except Exception as e:
            logger.error(f"Error creating dual PA for session {session_id}: {e}")
            if self.config.strict_mode:
                raise
            # Fallback to single PA mode
            dual_pa = DualPrimacyAttractor(user_pa=user_pa, governance_mode='single')
            dual_pa_mode_used = 'single'

        # Get embeddings for PA purposes (mock for now - would use actual embedding service)
        user_pa_embedding = np.random.rand(384)  # Placeholder
        ai_pa_embedding = np.random.rand(384) if dual_pa.is_dual_mode() else None

        # Replay each turn with dual PA
        turn_metrics = []
        dual_pa_fidelities = []
        dual_pa_ai_fidelities = []
        dual_pa_passes = 0
        dual_pa_interventions = 0
        errors_encountered = 0
        fallbacks_used = 0 if dual_pa_mode_used == 'dual' else 1

        for turn in turns:
            turn_start = time.time()

            response_text = turn.get('response', '')

            # Get response embedding (mock for now)
            response_embedding = np.random.rand(384)  # Placeholder

            try:
                # Check fidelity with dual PA
                fidelity_result = check_dual_pa_fidelity(
                    response_embedding,
                    user_pa_embedding,
                    ai_pa_embedding,
                    dual_pa
                )

                # Collect metrics
                dual_pa_fidelities.append(fidelity_result.user_fidelity)
                if fidelity_result.ai_fidelity > 0:
                    dual_pa_ai_fidelities.append(fidelity_result.ai_fidelity)

                if fidelity_result.overall_pass:
                    dual_pa_passes += 1
                else:
                    dual_pa_interventions += 1

                # Record turn metric
                turn_time_ms = (time.time() - turn_start) * 1000
                turn_metrics.append({
                    'turn_number': turn.get('turn_number', 0),
                    'user_fidelity': fidelity_result.user_fidelity,
                    'ai_fidelity': fidelity_result.ai_fidelity,
                    'overall_pass': fidelity_result.overall_pass,
                    'dominant_failure': fidelity_result.dominant_failure,
                    'governance_mode': fidelity_result.governance_mode,
                    'processing_time_ms': turn_time_ms
                })

            except Exception as e:
                logger.error(f"Error processing turn {turn.get('turn_number')}: {e}")
                errors_encountered += 1

                if self.config.strict_mode:
                    raise

                # Record error in metrics
                turn_metrics.append({
                    'turn_number': turn.get('turn_number', 0),
                    'error': str(e),
                    'processing_time_ms': 0.0
                })

        # Calculate aggregate metrics
        dual_pa_avg_user_fidelity = np.mean(dual_pa_fidelities) if dual_pa_fidelities else 0.0
        dual_pa_avg_ai_fidelity = np.mean(dual_pa_ai_fidelities) if dual_pa_ai_fidelities else None
        dual_pa_pass_rate = dual_pa_passes / len(turns) if turns else 0.0

        replay_time = time.time() - start_time
        avg_turn_time_ms = (replay_time * 1000) / len(turns) if turns else 0.0

        logger.info(
            f"Session {session_id} replay complete: "
            f"user_fidelity={dual_pa_avg_user_fidelity:.3f}, "
            f"pass_rate={dual_pa_pass_rate:.1%}, "
            f"time={replay_time:.2f}s"
        )

        return ComparisonResult(
            session_id=session_id,
            total_turns=len(turns),
            single_pa_avg_fidelity=single_pa_avg_fidelity,
            single_pa_pass_rate=single_pa_pass_rate,
            single_pa_interventions=single_pa_interventions,
            dual_pa_avg_user_fidelity=dual_pa_avg_user_fidelity,
            dual_pa_avg_ai_fidelity=dual_pa_avg_ai_fidelity,
            dual_pa_pass_rate=dual_pa_pass_rate,
            dual_pa_interventions=dual_pa_interventions,
            dual_pa_correlation=correlation,
            dual_pa_mode_used=dual_pa_mode_used,
            replay_time_seconds=replay_time,
            avg_turn_time_ms=avg_turn_time_ms,
            errors_encountered=errors_encountered,
            fallbacks_used=fallbacks_used,
            turn_metrics=turn_metrics
        )

    async def run_comparison_batch(
        self,
        session_files: List[Path],
        max_concurrent: int = 3
    ) -> List[ComparisonResult]:
        """
        Run comparison on multiple sessions with concurrency control.

        Args:
            session_files: List of session JSON file paths
            max_concurrent: Maximum concurrent replays

        Returns:
            List of ComparisonResults
        """
        logger.info(f"Running comparison on {len(session_files)} sessions (max_concurrent={max_concurrent})")

        semaphore = asyncio.Semaphore(max_concurrent)

        async def replay_with_semaphore(session_file: Path) -> Optional[ComparisonResult]:
            async with semaphore:
                try:
                    with open(session_file, 'r') as f:
                        session_data = json.load(f)
                    return await self.replay_session_with_dual_pa(session_data)
                except Exception as e:
                    logger.error(f"Error loading/replaying {session_file}: {e}")
                    if self.config.strict_mode:
                        raise
                    return None

        # Run all replays concurrently (with semaphore limiting)
        tasks = [replay_with_semaphore(f) for f in session_files]
        results = await asyncio.gather(*tasks, return_exceptions=not self.config.strict_mode)

        # Filter out None results
        valid_results = [r for r in results if isinstance(r, ComparisonResult)]

        logger.info(f"Completed {len(valid_results)}/{len(session_files)} session replays")

        return valid_results

    def aggregate_results(self, results: List[ComparisonResult]) -> Dict[str, Any]:
        """
        Aggregate comparison results across all sessions.

        Args:
            results: List of comparison results

        Returns:
            Aggregated metrics and summary
        """
        if not results:
            return {'error': 'No results to aggregate'}

        # Calculate overall metrics
        total_turns = sum(r.total_turns for r in results)

        avg_single_fidelity = np.mean([r.single_pa_avg_fidelity for r in results])
        avg_dual_user_fidelity = np.mean([r.dual_pa_avg_user_fidelity for r in results])

        avg_single_pass_rate = np.mean([r.single_pa_pass_rate for r in results])
        avg_dual_pass_rate = np.mean([r.dual_pa_pass_rate for r in results])

        total_single_interventions = sum(r.single_pa_interventions for r in results)
        total_dual_interventions = sum(r.dual_pa_interventions for r in results)

        # Count dual PA usage
        dual_pa_used_count = sum(1 for r in results if r.dual_pa_mode_used == 'dual')
        dual_pa_usage_rate = dual_pa_used_count / len(results)

        # Average correlation (for sessions that used dual PA)
        correlations = [r.dual_pa_correlation for r in results if r.dual_pa_correlation is not None]
        avg_correlation = np.mean(correlations) if correlations else None

        # Performance
        avg_replay_time = np.mean([r.replay_time_seconds for r in results])
        avg_turn_time = np.mean([r.avg_turn_time_ms for r in results])

        # Errors
        total_errors = sum(r.errors_encountered for r in results)
        total_fallbacks = sum(r.fallbacks_used for r in results)

        # Improvement metrics
        fidelity_improvement = avg_dual_user_fidelity - avg_single_fidelity
        pass_rate_improvement = avg_dual_pass_rate - avg_single_pass_rate
        intervention_reduction = total_single_interventions - total_dual_interventions

        return {
            'summary': {
                'total_sessions': len(results),
                'total_turns': total_turns,
                'dual_pa_usage_rate': dual_pa_usage_rate,
                'avg_correlation': avg_correlation
            },
            'single_pa_metrics': {
                'avg_fidelity': avg_single_fidelity,
                'avg_pass_rate': avg_single_pass_rate,
                'total_interventions': total_single_interventions
            },
            'dual_pa_metrics': {
                'avg_user_fidelity': avg_dual_user_fidelity,
                'avg_pass_rate': avg_dual_pass_rate,
                'total_interventions': total_dual_interventions
            },
            'improvements': {
                'fidelity_improvement': fidelity_improvement,
                'fidelity_improvement_pct': (fidelity_improvement / avg_single_fidelity * 100) if avg_single_fidelity > 0 else 0,
                'pass_rate_improvement': pass_rate_improvement,
                'intervention_reduction': intervention_reduction,
                'intervention_reduction_pct': (intervention_reduction / total_single_interventions * 100) if total_single_interventions > 0 else 0
            },
            'performance': {
                'avg_replay_time_seconds': avg_replay_time,
                'avg_turn_time_ms': avg_turn_time
            },
            'reliability': {
                'total_errors': total_errors,
                'total_fallbacks': total_fallbacks,
                'error_rate': total_errors / total_turns if total_turns > 0 else 0
            },
            'per_session_improvements': [r.improvement_summary() for r in results]
        }


async def main():
    """CLI entry point for comparison runner."""
    import argparse

    parser = argparse.ArgumentParser(description='Compare single PA vs dual PA on saved sessions')
    parser.add_argument('--sessions-dir', type=str, default='telos_observatory_v3/saved_sessions',
                        help='Directory containing session JSON files')
    parser.add_argument('--output', type=str, default='comparison_results.json',
                        help='Output file for comparison results')
    parser.add_argument('--max-concurrent', type=int, default=3,
                        help='Maximum concurrent session replays')
    parser.add_argument('--strict', action='store_true',
                        help='Strict mode: fail on any error')

    args = parser.parse_args()

    # Load session files
    sessions_dir = Path(args.sessions_dir)
    session_files = list(sessions_dir.glob('*.json'))

    if not session_files:
        logger.error(f"No session files found in {sessions_dir}")
        return

    logger.info(f"Found {len(session_files)} session files")

    # Create client and runner
    client = Anthropic()  # Uses ANTHROPIC_API_KEY env var
    config = GovernanceConfig.dual_pa_config(strict_mode=args.strict)
    runner = PAComparisonRunner(client, config)

    # Run comparison
    results = await runner.run_comparison_batch(session_files, max_concurrent=args.max_concurrent)

    # Aggregate and save
    aggregated = runner.aggregate_results(results)

    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(aggregated, f, indent=2)

    logger.info(f"Results saved to {output_path}")
    logger.info(f"\nSummary:\n{json.dumps(aggregated['improvements'], indent=2)}")


if __name__ == '__main__':
    asyncio.run(main())
