"""
TELOS Internal Test 0 - Improved Version
-----------------------------------------
Enhanced with comprehensive error handling, graceful degradation,
and detailed logging.

Improvements over original:
- Comprehensive error handling for all failure modes
- Graceful degradation (continue if one baseline fails)
- Detailed progress reporting
- Automatic recovery from transient errors
- Clear error messages with recovery suggestions
- Complete telemetry even on partial failure
"""

import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

# Import TELOS exceptions
from telos_purpose.exceptions import (
    TELOSError,
    MissingAPIKeyError,
    TestConversationError,
    OutputDirectoryError,
    APIConnectionError,
    validate_api_key,
    ensure_output_directory,
    error_context,
    setup_error_logging
)

# Import core components
from telos_purpose.core.primacy_math import PrimacyAttractorMath
from telos_purpose.core.embedding_provider import EmbeddingProvider
from telos_purpose.core.unified_steward import PrimacyAttractor
from telos_purpose.llm_clients.mistral_client import TelosMistralClient
from telos_purpose.validation.baseline_runners import (
    StatelessRunner,
    PromptOnlyRunner,
    CadenceReminderRunner,
    ObservationRunner,
    TELOSRunner
)

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Result from running a single baseline condition."""
    condition: str
    success: bool
    error: Optional[str]
    data: Optional[Dict[str, Any]]
    runtime_seconds: float


class InternalTest0Runner:
    """
    Orchestrates Internal Test 0 with robust error handling.
    
    Improvements:
    - Validates environment before starting
    - Continues execution even if one baseline fails
    - Provides detailed progress reporting
    - Exports results even on partial failure
    - Logs all errors for debugging
    """
    
    def __init__(
        self,
        config_path: str = "config.json",
        output_dir: str = "validation_results/internal_test0",
        test_conversation_path: str = "telos_purpose/test_conversations/test_convo_001.json"
    ):
        self.config_path = config_path
        self.output_dir = Path(output_dir)
        self.test_conversation_path = test_conversation_path
        
        self.results: List[TestResult] = []
        self.llm_client: Optional[TelosMistralClient] = None
        self.embedding_provider: Optional[EmbeddingProvider] = None
        self.attractor_config: Optional[PrimacyAttractor] = None
        self.test_conversation: Optional[List[tuple]] = None
    
    def run(self) -> Dict[str, Any]:
        """
        Execute complete Test 0 with error handling.
        
        Returns:
            Summary dict with results and any errors
        """
        start_time = time.time()
        
        print("\n" + "="*70)
        print("TELOS INTERNAL TEST 0 - ENHANCED VERSION")
        print("="*70 + "\n")
        
        try:
            # Phase 1: Validation
            print("📋 Phase 1: Pre-flight Validation")
            self._validate_environment()
            print("✅ Environment validated\n")
            
            # Phase 2: Initialization
            print("🔧 Phase 2: Initialization")
            self._initialize_components()
            print("✅ Components initialized\n")
            
            # Phase 3: Load test data
            print("📚 Phase 3: Loading Test Data")
            self._load_test_conversation()
            print(f"✅ Loaded conversation ({len(self.test_conversation)} turns)\n")
            
            # Phase 4: Run baselines
            print("🚀 Phase 4: Running Baseline Conditions")
            self._run_all_baselines()
            print("✅ All baselines completed\n")
            
            # Phase 5: Export results
            print("💾 Phase 5: Exporting Results")
            self._export_results()
            print("✅ Results exported\n")
            
            # Generate summary
            summary = self._generate_summary(time.time() - start_time)
            
            print("="*70)
            print("✅ TEST 0 COMPLETE")
            print("="*70 + "\n")
            
            return summary
            
        except TELOSError as e:
            logger.error(f"Test 0 failed: {e}")
            print(f"\n❌ TEST 0 FAILED\n{e}\n")
            return {
                'success': False,
                'error': str(e),
                'runtime_seconds': time.time() - start_time,
                'results': [r.__dict__ for r in self.results]
            }
        except Exception as e:
            logger.exception("Unexpected error in Test 0")
            print(f"\n❌ UNEXPECTED ERROR: {e}\n")
            return {
                'success': False,
                'error': f"Unexpected error: {e}",
                'runtime_seconds': time.time() - start_time,
                'results': [r.__dict__ for r in self.results]
            }
    
    def _validate_environment(self) -> None:
        """Validate environment before starting."""
        print("  • Checking API key...")
        validate_api_key("Mistral", "MISTRAL_API_KEY")
        
        print("  • Checking output directory...")
        ensure_output_directory(str(self.output_dir))
        
        print("  • Checking config file...")
        if not Path(self.config_path).exists():
            from telos_purpose.exceptions import FileNotFoundError
            raise FileNotFoundError(self.config_path, "configuration file")
        
        print("  • Checking test conversation...")
        if not Path(self.test_conversation_path).exists():
            from telos_purpose.exceptions import FileNotFoundError
            raise FileNotFoundError(self.test_conversation_path, "test conversation")
    
    def _initialize_components(self) -> None:
        """Initialize LLM client, embeddings, and config."""
        with error_context("initializing components"):
            # Load config
            print("  • Loading configuration...")
            with open(self.config_path) as f:
                config = json.load(f)
            
            gov_profile = config.get('governance_profile', {})
            self.attractor_config = PrimacyAttractor(
                purpose=gov_profile.get('purpose', []),
                scope=gov_profile.get('scope', []),
                boundaries=gov_profile.get('boundaries', []),
                constraint_tolerance=config.get('attractor_parameters', {}).get('constraint_tolerance', 0.2),
                privacy_level=config.get('attractor_parameters', {}).get('privacy_level', 0.8),
                task_priority=config.get('attractor_parameters', {}).get('task_priority', 0.9)
            )
            
            # Initialize LLM client
            print("  • Initializing LLM client...")
            try:
                self.llm_client = TelosMistralClient()
            except Exception as e:
                raise APIConnectionError("Mistral", e)
            
            # Initialize embedding provider
            print("  • Loading embedding model (first run downloads ~80MB)...")
            try:
                self.embedding_provider = EmbeddingProvider(deterministic=False)
            except Exception as e:
                from telos_purpose.exceptions import ModelLoadError
                raise ModelLoadError("sentence-transformers/all-MiniLM-L6-v2", e)
    
    def _load_test_conversation(self) -> None:
        """Load and validate test conversation."""
        with error_context("loading test conversation", file=self.test_conversation_path):
            try:
                with open(self.test_conversation_path) as f:
                    data = json.load(f)
                
                # Validate structure
                if not isinstance(data, list):
                    raise TestConversationError(
                        self.test_conversation_path,
                        "Expected list of conversation turns"
                    )
                
                # Convert to (user, assistant) tuples
                self.test_conversation = []
                for turn in data:
                    if 'user' not in turn:
                        raise TestConversationError(
                            self.test_conversation_path,
                            f"Turn missing 'user' field: {turn}"
                        )
                    user_msg = turn['user']
                    asst_msg = turn.get('assistant', '')  # Assistant responses generated during test
                    self.test_conversation.append((user_msg, asst_msg))
                
            except json.JSONDecodeError as e:
                raise TestConversationError(
                    self.test_conversation_path,
                    f"Invalid JSON: {e}"
                )
    
    def _run_all_baselines(self) -> None:
        """Run all 5 baseline conditions with error handling."""
        baselines = [
            ("Stateless", StatelessRunner),
            ("Prompt-Only", PromptOnlyRunner),
            ("Cadence", CadenceReminderRunner),
            ("Observation", ObservationRunner),
            ("TELOS", TELOSRunner)
        ]
        
        for i, (name, runner_class) in enumerate(baselines, 1):
            print(f"\n  [{i}/5] Running {name} baseline...")
            result = self._run_single_baseline(name, runner_class)
            self.results.append(result)
            
            if result.success:
                print(f"      ✅ {name} completed ({result.runtime_seconds:.1f}s)")
            else:
                print(f"      ⚠️  {name} failed: {result.error}")
                print(f"      ℹ️  Continuing with remaining baselines...")
    
    def _run_single_baseline(self, name: str, runner_class) -> TestResult:
        """Run a single baseline with error handling and retries."""
        start_time = time.time()
        
        max_retries = 2
        for attempt in range(max_retries):
            try:
                # Initialize runner
                runner = runner_class(
                    llm_client=self.llm_client,
                    embedding_provider=self.embedding_provider,
                    attractor_config=self.attractor_config
                )
                
                # Run conversation
                result_data = runner.run_conversation(self.test_conversation)
                
                return TestResult(
                    condition=name.lower().replace('-', '_').replace(' ', '_'),
                    success=True,
                    error=None,
                    data=result_data,
                    runtime_seconds=time.time() - start_time
                )
                
            except APIConnectionError as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"      ⏳ Connection error, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    return TestResult(
                        condition=name.lower().replace('-', '_').replace(' ', '_'),
                        success=False,
                        error=f"API connection failed after {max_retries} attempts: {e}",
                        data=None,
                        runtime_seconds=time.time() - start_time
                    )
            
            except Exception as e:
                logger.exception(f"Error in {name} baseline")
                return TestResult(
                    condition=name.lower().replace('-', '_').replace(' ', '_'),
                    success=False,
                    error=f"{type(e).__name__}: {str(e)[:200]}",
                    data=None,
                    runtime_seconds=time.time() - start_time
                )
    
    def _export_results(self) -> None:
        """Export results with error handling."""
        successful_results = [r for r in self.results if r.success]

        if not successful_results:
            print("  ⚠️  No successful results to export")
            return

        print(f"  • Exporting {len(successful_results)} successful results...")

        for result in successful_results:
            try:
                self._export_single_result(result)
                print(f"    ✓ {result.condition}")
            except Exception as e:
                # Log full traceback for debugging
                logger.exception(f"Failed to export {result.condition}")
                print(f"    ✗ {result.condition}: {type(e).__name__}: {e}")
    
    def _export_single_result(self, result: TestResult) -> None:
        """Export a single result to CSV and JSON."""
        from telos_purpose.validation.telemetry_utils import export_turn_csv, export_session_json

        with error_context(f"exporting {result.condition} results"):
            # Convert BaselineResult to dict format expected by export functions
            result_dict = {
                'runner_type': result.data.runner_type,
                'session_id': result.data.session_id,
                'turn_results': result.data.turn_results,
                'final_metrics': result.data.final_metrics,
                'metadata': result.data.metadata
            }

            session_id = f"internal_test0_{result.condition}"

            # Export turn-level CSV
            export_turn_csv(result_dict, self.output_dir, session_id, result.condition)

            # Export session summary JSON
            export_session_json(result_dict, self.output_dir, session_id, result.condition)
    
    def _generate_summary(self, runtime_seconds: float) -> Dict[str, Any]:
        """Generate test summary."""
        successful = [r for r in self.results if r.success]
        failed = [r for r in self.results if not r.success]
        
        summary = {
            'success': len(failed) == 0,
            'total_conditions': len(self.results),
            'successful_conditions': len(successful),
            'failed_conditions': len(failed),
            'runtime_seconds': runtime_seconds,
            'results': {
                'successful': [r.condition for r in successful],
                'failed': [(r.condition, r.error) for r in failed]
            }
        }
        
        # Print summary
        print(f"\n📊 Summary:")
        print(f"  • Total runtime: {runtime_seconds:.1f}s")
        print(f"  • Successful: {len(successful)}/5")
        print(f"  • Failed: {len(failed)}/5")
        
        if failed:
            print(f"\n⚠️  Failed conditions:")
            for r in failed:
                print(f"  • {r.condition}: {r.error}")
        
        return summary


def main():
    """Main entry point."""
    # Setup logging
    setup_error_logging(
        log_file="validation_results/internal_test0.log",
        level=logging.INFO
    )

    # Run test
    runner = InternalTest0Runner()
    summary = runner.run()

    # Print API usage statistics
    if runner.llm_client:
        print("\n" + "="*70)
        print("📊 API USAGE STATISTICS")
        print("="*70)
        try:
            usage = runner.llm_client.get_usage_stats()
            print(f"Total Requests: {usage.get('total_requests', 0)}")
            print(f"Total Tokens: {usage.get('total_tokens', 0)}")
            print(f"Estimated Cost: ${usage.get('estimated_cost_usd', 0):.4f}")
        except Exception as e:
            print(f"Unable to retrieve usage stats: {e}")
        print("="*70 + "\n")

    # Exit with appropriate code
    exit(0 if summary['success'] else 1)


if __name__ == "__main__":
    main()
