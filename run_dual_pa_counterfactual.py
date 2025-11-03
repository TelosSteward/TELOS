"""
Counterfactual Analysis: Single PA vs Dual PA on ShareGPT Sessions

Re-runs the 45 ShareGPT test sessions through dual PA architecture
to compare governance effectiveness against single PA baseline.
"""

import json
import os
import asyncio
from pathlib import Path
from typing import Dict, List, Any
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from telos_purpose.core.unified_orchestrator_steward import UnifiedOrchestratorSteward
from telos_purpose.core.governance_config import GovernanceConfig
from telos_purpose.llm_clients.mistral_client import MistralClient

# Load environment variables
load_dotenv()

# Configuration
SESSIONS_DIR = Path("telos_observatory_v3/saved_sessions")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

if not MISTRAL_API_KEY:
    raise ValueError("MISTRAL_API_KEY environment variable not set")


def load_session_file(file_path: Path) -> Dict[str, Any]:
    """Load ShareGPT session file."""
    with open(file_path, 'r') as f:
        return json.load(f)


async def analyze_session_dual_pa(
    session_data: Dict[str, Any],
    llm_client,
    embedding_provider
) -> Dict[str, Any]:
    """
    Analyze a single session using dual PA architecture.

    Returns:
        Analysis results with dual PA metrics for each turn
    """
    session_id = session_data['session_id']
    user_pa_config = session_data['primacy_attractor']
    turns = session_data['turns']

    print(f"\n  Processing {session_id}...")
    print(f"    User PA: {user_pa_config['purpose'][0][:60]}...")

    # Create dual PA config
    config = GovernanceConfig.dual_pa_config(strict_mode=False)

    # Initialize orchestrator
    orchestrator = UnifiedOrchestratorSteward(
        governance_config=config,
        user_pa_config=user_pa_config,
        llm_client=llm_client,
        embedding_provider=embedding_provider,
        enable_interventions=True,
        dev_commentary_mode="silent"
    )

    # Initialize governance (this will attempt dual PA derivation)
    try:
        await orchestrator.initialize_governance()
    except Exception as e:
        print(f"    WARNING: Governance initialization failed: {e}")
        return None

    # Start session
    orchestrator.start_session(session_id=session_id)

    # Get governance mode actually used
    actual_mode = orchestrator.actual_governance_mode
    is_dual_mode = orchestrator.dual_pa.is_dual_mode() if orchestrator.dual_pa else False
    correlation = orchestrator.dual_pa.correlation if is_dual_mode else None

    print(f"    Governance mode: {actual_mode} (dual PA: {is_dual_mode})")
    if correlation:
        print(f"    PA correlation: {correlation:.3f}")

    # Process each turn
    turn_results = []
    for turn_data in turns:
        user_input = turn_data['user_input']
        # Use the TELOS response from single PA as the model response to analyze
        model_response = turn_data['assistant_response_telos']

        # Process turn through dual PA
        result = orchestrator.process_turn(
            user_input=user_input,
            model_response=model_response
        )

        # Extract metrics
        turn_result = {
            'turn': turn_data['turn'],
            'single_pa_fidelity': turn_data.get('fidelity_telos', turn_data.get('fidelity_native', 0)),
            'dual_pa_mode_used': is_dual_mode,
            'fidelity': result.get('metrics', {}).get('fidelity', 0),
        }

        # Add dual PA specific metrics if available
        if 'dual_pa_metrics' in result:
            dual_metrics = result['dual_pa_metrics']
            turn_result.update({
                'user_pa_fidelity': dual_metrics['user_fidelity'],
                'ai_pa_fidelity': dual_metrics['ai_fidelity'],
                'user_pa_pass': dual_metrics['user_pass'],
                'ai_pa_pass': dual_metrics['ai_pass'],
                'overall_pass': dual_metrics['overall_pass'],
                'dominant_failure': dual_metrics['dominant_failure']
            })

        turn_results.append(turn_result)

    # End session
    summary = orchestrator.end_session()

    # Compile session analysis
    analysis = {
        'session_id': session_id,
        'governance_mode_actual': actual_mode,
        'dual_pa_used': is_dual_mode,
        'dual_pa_correlation': correlation,
        'user_pa': {
            'purpose': user_pa_config['purpose'],
            'scope': user_pa_config['scope'],
            'boundaries': user_pa_config['boundaries']
        },
        'ai_pa': None,
        'turn_count': len(turn_results),
        'turns': turn_results,
        'session_summary': summary
    }

    # Add AI PA if dual mode was used
    if is_dual_mode and orchestrator.dual_pa:
        analysis['ai_pa'] = {
            'purpose': orchestrator.dual_pa.ai_pa.get('purpose', []),
            'scope': orchestrator.dual_pa.ai_pa.get('scope', []),
            'boundaries': orchestrator.dual_pa.ai_pa.get('boundaries', [])
        }

    return analysis


def calculate_statistics(all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate comparative statistics across all sessions."""
    import numpy as np

    # Separate by governance mode
    single_pa_sessions = [s for s in all_results if not s['dual_pa_used']]
    dual_pa_sessions = [s for s in all_results if s['dual_pa_used']]

    # Collect all turn fidelities
    single_pa_fidelities = []
    dual_pa_user_fidelities = []
    dual_pa_ai_fidelities = []

    for session in all_results:
        for turn in session['turns']:
            single_pa_fidelities.append(turn['single_pa_fidelity'])

            if turn['dual_pa_mode_used']:
                dual_pa_user_fidelities.append(turn.get('user_pa_fidelity', 0))
                dual_pa_ai_fidelities.append(turn.get('ai_pa_fidelity', 0))

    # Calculate statistics
    stats = {
        'total_sessions': len(all_results),
        'sessions_analyzed': {
            'single_pa': len(single_pa_sessions),
            'dual_pa': len(dual_pa_sessions)
        },
        'total_turns': len(single_pa_fidelities),

        'single_pa': {
            'mean_fidelity': float(np.mean(single_pa_fidelities)) if single_pa_fidelities else 0,
            'median_fidelity': float(np.median(single_pa_fidelities)) if single_pa_fidelities else 0,
            'std_fidelity': float(np.std(single_pa_fidelities)) if single_pa_fidelities else 0,
            'min_fidelity': float(np.min(single_pa_fidelities)) if single_pa_fidelities else 0,
            'max_fidelity': float(np.max(single_pa_fidelities)) if single_pa_fidelities else 0,
        },

        'dual_pa_user': {
            'mean_fidelity': float(np.mean(dual_pa_user_fidelities)) if dual_pa_user_fidelities else 0,
            'median_fidelity': float(np.median(dual_pa_user_fidelities)) if dual_pa_user_fidelities else 0,
            'std_fidelity': float(np.std(dual_pa_user_fidelities)) if dual_pa_user_fidelities else 0,
            'min_fidelity': float(np.min(dual_pa_user_fidelities)) if dual_pa_user_fidelities else 0,
            'max_fidelity': float(np.max(dual_pa_user_fidelities)) if dual_pa_user_fidelities else 0,
        },

        'dual_pa_ai': {
            'mean_fidelity': float(np.mean(dual_pa_ai_fidelities)) if dual_pa_ai_fidelities else 0,
            'median_fidelity': float(np.median(dual_pa_ai_fidelities)) if dual_pa_ai_fidelities else 0,
            'std_fidelity': float(np.std(dual_pa_ai_fidelities)) if dual_pa_ai_fidelities else 0,
            'min_fidelity': float(np.min(dual_pa_ai_fidelities)) if dual_pa_ai_fidelities else 0,
            'max_fidelity': float(np.max(dual_pa_ai_fidelities)) if dual_pa_ai_fidelities else 0,
        },

        'correlations': {
            'dual_pa_correlations': [s['dual_pa_correlation'] for s in dual_pa_sessions if s['dual_pa_correlation']],
            'mean_correlation': float(np.mean([s['dual_pa_correlation'] for s in dual_pa_sessions if s['dual_pa_correlation']])) if dual_pa_sessions else 0
        }
    }

    # Calculate improvement metrics
    if dual_pa_user_fidelities:
        stats['improvement'] = {
            'user_pa_vs_single_pa': {
                'mean_diff': stats['dual_pa_user']['mean_fidelity'] - stats['single_pa']['mean_fidelity'],
                'percent_improvement': ((stats['dual_pa_user']['mean_fidelity'] - stats['single_pa']['mean_fidelity']) / stats['single_pa']['mean_fidelity'] * 100) if stats['single_pa']['mean_fidelity'] > 0 else 0
            }
        }

    return stats


async def main():
    """Run counterfactual analysis on all ShareGPT sessions."""
    print("=" * 80)
    print("DUAL PA COUNTERFACTUAL ANALYSIS")
    print("=" * 80)
    print(f"Analyzing ShareGPT sessions from: {SESSIONS_DIR}")

    # Find all session files
    session_files = sorted(SESSIONS_DIR.glob("sharegpt_filtered_*.json"))
    print(f"Found {len(session_files)} session files")

    if not session_files:
        print("ERROR: No session files found!")
        return

    # Initialize LLM and embedding provider (shared across all sessions)
    print("\nInitializing LLM client and embedding provider...")
    llm_client = MistralClient(api_key=MISTRAL_API_KEY, model="mistral-large-latest")
    embedding_provider = SentenceTransformer('all-MiniLM-L6-v2')
    print("  ✓ Ready")

    # Analyze each session
    print(f"\nAnalyzing {len(session_files)} sessions...")
    all_results = []

    for i, session_file in enumerate(session_files, 1):
        print(f"\n[{i}/{len(session_files)}] {session_file.name}")

        try:
            # Load session
            session_data = load_session_file(session_file)

            # Analyze with dual PA
            result = await analyze_session_dual_pa(
                session_data=session_data,
                llm_client=llm_client,
                embedding_provider=embedding_provider
            )

            if result:
                all_results.append(result)
                print(f"    ✓ Completed ({len(result['turns'])} turns)")
            else:
                print(f"    ✗ Failed to analyze session")

        except Exception as e:
            print(f"    ✗ Error: {e}")
            import traceback
            traceback.print_exc()

    # Calculate statistics
    print("\n" + "=" * 80)
    print("CALCULATING STATISTICS")
    print("=" * 80)

    stats = calculate_statistics(all_results)

    # Display results
    print(f"\nTotal sessions analyzed: {stats['total_sessions']}")
    print(f"  - Single PA fallback: {stats['sessions_analyzed']['single_pa']}")
    print(f"  - Dual PA success: {stats['sessions_analyzed']['dual_pa']}")
    print(f"Total turns: {stats['total_turns']}")

    print("\n--- SINGLE PA BASELINE ---")
    print(f"Mean fidelity:   {stats['single_pa']['mean_fidelity']:.4f}")
    print(f"Median fidelity: {stats['single_pa']['median_fidelity']:.4f}")
    print(f"Std dev:         {stats['single_pa']['std_fidelity']:.4f}")
    print(f"Range:           [{stats['single_pa']['min_fidelity']:.4f}, {stats['single_pa']['max_fidelity']:.4f}]")

    if stats['sessions_analyzed']['dual_pa'] > 0:
        print("\n--- DUAL PA USER FIDELITY ---")
        print(f"Mean fidelity:   {stats['dual_pa_user']['mean_fidelity']:.4f}")
        print(f"Median fidelity: {stats['dual_pa_user']['median_fidelity']:.4f}")
        print(f"Std dev:         {stats['dual_pa_user']['std_fidelity']:.4f}")
        print(f"Range:           [{stats['dual_pa_user']['min_fidelity']:.4f}, {stats['dual_pa_user']['max_fidelity']:.4f}]")

        print("\n--- DUAL PA AI FIDELITY ---")
        print(f"Mean fidelity:   {stats['dual_pa_ai']['mean_fidelity']:.4f}")
        print(f"Median fidelity: {stats['dual_pa_ai']['median_fidelity']:.4f}")
        print(f"Std dev:         {stats['dual_pa_ai']['std_fidelity']:.4f}")
        print(f"Range:           [{stats['dual_pa_ai']['min_fidelity']:.4f}, {stats['dual_pa_ai']['max_fidelity']:.4f}]")

        print("\n--- PA CORRELATIONS ---")
        print(f"Mean correlation: {stats['correlations']['mean_correlation']:.4f}")
        print(f"Correlations: {[f'{c:.3f}' for c in stats['correlations']['dual_pa_correlations'][:10]]}")

        if 'improvement' in stats:
            print("\n--- IMPROVEMENT METRICS ---")
            print(f"Dual PA User vs Single PA:")
            print(f"  Mean diff:    {stats['improvement']['user_pa_vs_single_pa']['mean_diff']:+.4f}")
            print(f"  % change:     {stats['improvement']['user_pa_vs_single_pa']['percent_improvement']:+.2f}%")

    # Save results
    output_file = Path("dual_pa_counterfactual_results.json")
    output_data = {
        'statistics': stats,
        'session_results': all_results
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✓ Full results saved to: {output_file}")
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
