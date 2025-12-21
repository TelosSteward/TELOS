"""
Conversation Replayer for TELOS Dashboard
==========================================

Automates conversation replay through TELOS for validation and screencasting.

Features:
- Load conversations from JSON files
- Replay through TELOS with configurable delays
- Generate comprehensive metrics
- Export results for analysis
- Perfect for demo and validation workflows
"""

import json
import time
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from telos_purpose.core.unified_steward import UnifiedGovernanceSteward, PrimacyAttractor
from telos_purpose.llm_clients.mistral_client import TelosMistralClient
from telos_purpose.core.embedding_provider import EmbeddingProvider
import os


class ConversationReplayer:
    """Replay conversations through TELOS for validation and demonstration."""

    def __init__(
        self,
        config_path: str = "config.json",
        delay_between_turns: float = 2.0,
        verbose: bool = True
    ):
        """
        Initialize replayer.

        Args:
            config_path: Path to TELOS configuration
            delay_between_turns: Seconds to wait between turns (for demos)
            verbose: Print progress updates
        """
        self.config_path = config_path
        self.delay_between_turns = delay_between_turns
        self.verbose = verbose

        # Components
        self.llm = None
        self.embedding_provider = None
        self.telos_steward = None
        self.config = None

        # Results
        self.session_results = []

    def initialize(self) -> bool:
        """Initialize TELOS components."""
        try:
            if self.verbose:
                print("🔧 Initializing TELOS components...")

            # Load configuration
            with open(self.config_path) as f:
                self.config = json.load(f)

            # Initialize LLM
            api_key = os.getenv('MISTRAL_API_KEY')
            if not api_key:
                print("❌ MISTRAL_API_KEY not set")
                return False

            self.llm = TelosMistralClient(api_key=api_key)

            # Initialize embeddings
            use_real = self.config.get('validation_settings', {}).get('use_real_embeddings', True)
            self.embedding_provider = EmbeddingProvider(deterministic=not use_real)

            # Create attractor
            gov_profile = self.config.get('governance_profile', {})
            attractor_params = self.config.get('attractor_parameters', {})

            attractor = PrimacyAttractor(
                purpose=gov_profile.get('purpose', []),
                scope=gov_profile.get('scope', []),
                boundaries=gov_profile.get('boundaries', []),
                constraint_tolerance=attractor_params.get('constraint_tolerance', 0.2),
                privacy_level=attractor_params.get('privacy_level', 0.8),
                task_priority=attractor_params.get('task_priority', 0.9)
            )

            # Create steward
            self.telos_steward = UnifiedGovernanceSteward(
                attractor=attractor,
                llm_client=self.llm,
                embedding_provider=self.embedding_provider,
                enable_interventions=True,
                dev_commentary_mode="silent"
            )

            if self.verbose:
                print("✅ TELOS initialized successfully")

            return True

        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            return False

    def load_conversation(self, conversation_file: str) -> List[str]:
        """
        Load conversation from JSON file.

        Args:
            conversation_file: Path to JSON file

        Returns:
            List of user messages
        """
        with open(conversation_file) as f:
            data = json.load(f)

        # Extract user messages
        if isinstance(data, list):
            messages = [turn['user'] for turn in data if 'user' in turn]
        else:
            raise ValueError("Conversation file must be a JSON array of turn objects")

        return messages

    def replay_conversation(
        self,
        conversation_file: str,
        session_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Replay a conversation through TELOS.

        Args:
            conversation_file: Path to conversation JSON
            session_name: Optional name for this session

        Returns:
            Session results dictionary
        """
        if session_name is None:
            session_name = Path(conversation_file).stem

        if self.verbose:
            print(f"\n{'='*70}")
            print(f"📝 Replaying: {session_name}")
            print(f"{'='*70}\n")

        # Load conversation
        messages = self.load_conversation(conversation_file)

        if self.verbose:
            print(f"📚 Loaded {len(messages)} turns\n")

        # Start TELOS session
        self.telos_steward.start_session()
        session_id = self.telos_steward.session_id

        # Replay turns
        turn_results = []
        start_time = time.time()

        for i, user_message in enumerate(messages, 1):
            if self.verbose:
                print(f"[Turn {i}/{len(messages)}] User: {user_message[:60]}...")

            # Get messages for API
            api_messages = self.telos_steward.conversation.get_messages_for_api()
            api_messages.append({"role": "user", "content": user_message})

            # Generate response
            initial_response = self.llm.generate(messages=api_messages, max_tokens=500)

            # Process through TELOS
            result = self.telos_steward.process_turn(user_message, initial_response)

            # Extract metrics
            turn_data = {
                'turn': i,
                'user_message': user_message,
                'initial_response': initial_response,
                'final_response': result['final_response'],
                'metrics': result['metrics'],
                'intervention_applied': result['intervention_applied'],
                'intervention_type': result['intervention_result']['type'] if result['intervention_result'] else None
            }

            turn_results.append(turn_data)

            if self.verbose:
                metrics = result['metrics']
                print(f"    F={metrics['telic_fidelity']:.3f} | "
                      f"ε={metrics['error_signal']:.3f} | "
                      f"V(x)={metrics['lyapunov_value']:.3f} | "
                      f"Basin={'✅' if metrics['primacy_basin_membership'] else '❌'}")

                if result['intervention_applied']:
                    print(f"    ⚡ Intervention: {result['intervention_result']['type']}")

                print()

            # Delay between turns (for demos)
            if i < len(messages) and self.delay_between_turns > 0:
                time.sleep(self.delay_between_turns)

        total_time = time.time() - start_time

        # Calculate session metrics
        fidelities = [t['metrics']['telic_fidelity'] for t in turn_results]
        interventions = [t for t in turn_results if t['intervention_applied']]

        session_results = {
            'session_name': session_name,
            'session_id': session_id,
            'conversation_file': conversation_file,
            'timestamp': datetime.now().isoformat(),
            'total_turns': len(messages),
            'total_time_seconds': total_time,
            'turn_results': turn_results,
            'summary': {
                'avg_fidelity': sum(fidelities) / len(fidelities),
                'final_fidelity': fidelities[-1],
                'min_fidelity': min(fidelities),
                'max_fidelity': max(fidelities),
                'intervention_count': len(interventions),
                'intervention_rate': len(interventions) / len(messages),
                'basin_time': sum(1 for t in turn_results if t['metrics']['primacy_basin_membership']) / len(turn_results)
            }
        }

        self.session_results.append(session_results)

        # End session
        self.telos_steward.end_session()

        if self.verbose:
            self._print_session_summary(session_results)

        return session_results

    def replay_multiple(
        self,
        conversation_files: List[str],
        export_path: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Replay multiple conversations.

        Args:
            conversation_files: List of paths to conversation JSON files
            export_path: Optional path to export results

        Returns:
            List of session results
        """
        results = []

        for conv_file in conversation_files:
            result = self.replay_conversation(conv_file)
            results.append(result)

        # Export if requested
        if export_path:
            self.export_results(export_path)

        # Print comparative summary
        if self.verbose and len(results) > 1:
            self._print_comparative_summary(results)

        return results

    def export_results(self, export_path: str):
        """Export all session results to JSON."""
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'total_sessions': len(self.session_results),
            'config': self.config,
            'sessions': self.session_results
        }

        with open(export_path, 'w') as f:
            json.dump(export_data, f, indent=2)

        if self.verbose:
            print(f"\n💾 Results exported to: {export_path}")

    def _print_session_summary(self, results: Dict[str, Any]):
        """Print summary for a single session."""
        summary = results['summary']

        print(f"\n{'='*70}")
        print("📊 Session Summary")
        print(f"{'='*70}")
        print(f"Session: {results['session_name']}")
        print(f"Turns: {results['total_turns']}")
        print(f"Time: {results['total_time_seconds']:.1f}s")
        print()
        print(f"Avg Fidelity: {summary['avg_fidelity']:.3f}")
        print(f"Final Fidelity: {summary['final_fidelity']:.3f}")
        print(f"Fidelity Range: {summary['min_fidelity']:.3f} - {summary['max_fidelity']:.3f}")
        print()
        print(f"Interventions: {summary['intervention_count']} ({summary['intervention_rate']*100:.0f}%)")
        print(f"Time in Basin: {summary['basin_time']*100:.0f}%")
        print(f"{'='*70}\n")

    def _print_comparative_summary(self, results: List[Dict[str, Any]]):
        """Print comparative summary for multiple sessions."""
        print(f"\n{'='*70}")
        print("📊 Comparative Summary")
        print(f"{'='*70}\n")

        print(f"{'Session':<30} {'Turns':<8} {'Avg F':<10} {'Interventions':<15} {'Basin %':<10}")
        print("-" * 70)

        for r in results:
            s = r['summary']
            print(f"{r['session_name']:<30} "
                  f"{r['total_turns']:<8} "
                  f"{s['avg_fidelity']:<10.3f} "
                  f"{s['intervention_count']:>3} ({s['intervention_rate']*100:>3.0f}%)    "
                  f"{s['basin_time']*100:>5.0f}%")

        print(f"{'='*70}\n")


def main():
    """Main CLI interface."""
    import argparse

    parser = argparse.ArgumentParser(description="Replay conversations through TELOS")
    parser.add_argument('conversations', nargs='+', help='Conversation JSON files to replay')
    parser.add_argument('--config', default='config.json', help='Config file path')
    parser.add_argument('--delay', type=float, default=0.0, help='Delay between turns (seconds)')
    parser.add_argument('--export', help='Export results to JSON file')
    parser.add_argument('--quiet', action='store_true', help='Minimal output')

    args = parser.parse_args()

    # Create replayer
    replayer = ConversationReplayer(
        config_path=args.config,
        delay_between_turns=args.delay,
        verbose=not args.quiet
    )

    # Initialize
    if not replayer.initialize():
        sys.exit(1)

    # Replay conversations
    replayer.replay_multiple(args.conversations, export_path=args.export)

    print("\n✅ All replays complete!")


if __name__ == "__main__":
    main()
