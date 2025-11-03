"""
Dual PA Regeneration on Claude Conversation

Regenerates the Claude conversation (where you struggled with drift)
using dual PA governance to demonstrate effectiveness on real-world drift scenario.
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
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

if not MISTRAL_API_KEY:
    raise ValueError("MISTRAL_API_KEY environment variable not set")


async def regenerate_claude_conversation(
    session_data: Dict[str, Any],
    llm_client,
    embedding_provider
) -> Dict[str, Any]:
    """
    Regenerate Claude conversation with DUAL PA governance.

    This is YOUR actual conversation where you struggled with Claude understanding
    your purpose - a perfect stress test for dual PA governance.
    """
    session_id = session_data['session_id']
    user_pa_config = session_data['primacy_attractor']
    turns = session_data['turns']

    print(f"\n  Regenerating {session_id} with DUAL PA governance...")
    print(f"    Your Purpose: {user_pa_config['purpose'][0][:70]}...")
    print(f"    Total turns to regenerate: {len(turns)}")

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
    orchestrator.start_session(session_id=f"{session_id}_dual_pa_regen")

    # Get governance mode actually used
    actual_mode = orchestrator.actual_governance_mode
    is_dual_mode = orchestrator.dual_pa.is_dual_mode() if orchestrator.dual_pa else False
    correlation = orchestrator.dual_pa.correlation if is_dual_mode else None

    print(f"    Governance mode: {actual_mode} (dual PA: {is_dual_mode})")
    if correlation:
        print(f"    PA correlation: {correlation:.3f}")

    # Build conversation context
    conversation_context = []
    turn_results = []

    for turn_data in turns:
        user_input = turn_data['user_input']

        # Add user message to context
        conversation_context.append({
            "role": "user",
            "content": user_input
        })

        # CRITICAL: Use generate_governed_response() to REGENERATE with dual PA
        result = orchestrator.generate_governed_response(
            user_input=user_input,
            conversation_context=conversation_context
        )

        # Add assistant response to context
        conversation_context.append({
            "role": "assistant",
            "content": result['governed_response']
        })

        # Extract metrics
        turn_result = {
            'turn': turn_data['turn'],
            'user_input': user_input,
            'original_response': turn_data['assistant_response_telos'],
            'original_fidelity': turn_data.get('fidelity_telos', 0.5),
            'dual_pa_response': result['governed_response'],
            'dual_pa_mode_used': is_dual_mode,
            'intervention_applied': result.get('intervention_applied', False),
            'intervention_type': result.get('intervention_type', 'none')
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
        else:
            turn_result['fidelity'] = result.get('metrics', {}).get('fidelity', 0)

        turn_results.append(turn_result)

        # Progress indicator
        if turn_result['turn'] % 10 == 0:
            print(f"    Progress: {turn_result['turn']}/{len(turns)} turns completed")

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


async def main():
    """Run dual PA regeneration on Claude conversation."""
    print("=" * 80)
    print("CLAUDE CONVERSATION DUAL PA REGENERATION")
    print("=" * 80)
    print("Testing dual PA governance on YOUR actual conversation")
    print("where you struggled with drift/misalignment.")
    print("=" * 80)

    # Load the converted Claude conversation
    session_file = Path("telos_observatory_v3/saved_sessions/sharegpt_claude_conversation.json")

    if not session_file.exists():
        print(f"ERROR: Session file not found: {session_file}")
        return

    print(f"\nLoading session: {session_file.name}")
    with open(session_file, 'r') as f:
        session_data = json.load(f)

    print(f"  Session ID: {session_data['session_id']}")
    print(f"  Total turns: {len(session_data['turns'])}")

    # Initialize LLM and embedding provider
    print("\nInitializing LLM client and embedding provider...")
    llm_client = MistralClient(api_key=MISTRAL_API_KEY, model="mistral-large-latest")
    embedding_provider = SentenceTransformer('all-MiniLM-L6-v2')
    print("  ✓ Ready")

    # Regenerate with dual PA
    print(f"\nRegenerating Claude conversation with DUAL PA governance...")

    try:
        result = await regenerate_claude_conversation(
            session_data=session_data,
            llm_client=llm_client,
            embedding_provider=embedding_provider
        )

        if result:
            print(f"\n    ✓ Regeneration completed ({len(result['turns'])} turns)")

            # Calculate summary statistics
            original_fidelities = [t['original_fidelity'] for t in result['turns']]
            user_fidelities = [t.get('user_pa_fidelity', 0) for t in result['turns'] if 'user_pa_fidelity' in t]
            ai_fidelities = [t.get('ai_pa_fidelity', 0) for t in result['turns'] if 'ai_pa_fidelity' in t]
            interventions = sum(1 for t in result['turns'] if t.get('intervention_applied', False))

            print("\n" + "=" * 80)
            print("RESULTS SUMMARY")
            print("=" * 80)
            print(f"Original (Single PA) mean fidelity: {sum(original_fidelities)/len(original_fidelities):.4f}")
            if user_fidelities:
                print(f"Dual PA User fidelity:              {sum(user_fidelities)/len(user_fidelities):.4f}")
                print(f"Dual PA AI fidelity:                {sum(ai_fidelities)/len(ai_fidelities):.4f}")
            print(f"Interventions applied:              {interventions}")
            print(f"Intervention rate:                  {interventions/len(result['turns']):.2%}")

            # Save results
            output_file = Path("claude_conversation_dual_pa_results.json")
            with open(output_file, 'w') as f:
                json.dump({
                    'session_result': result,
                    'methodology': {
                        'type': 'regeneration',
                        'description': 'Dual PA governance on real conversation with documented drift',
                        'baseline': 'Original Claude conversation responses',
                        'treatment': 'Dual PA governed responses (newly generated)'
                    }
                }, f, indent=2)

            print(f"\n✓ Full results saved to: {output_file}")
            print("\n" + "=" * 80)
            print("REGENERATION COMPLETE")
            print("=" * 80)
        else:
            print(f"\n    ✗ Failed to regenerate session")

    except Exception as e:
        print(f"\n    ✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
