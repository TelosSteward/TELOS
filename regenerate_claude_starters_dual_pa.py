"""
Regenerate Claude Conversation with Dual PA (Starters Only)

Takes ONLY the conversation starters (user inputs) and regenerates ALL responses
with dual PA governance from scratch. True isolated A/B test.
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


async def regenerate_from_starters(
    session_data: Dict[str, Any],
    llm_client,
    embedding_provider
) -> Dict[str, Any]:
    """
    Regenerate conversation from ONLY conversation starters.

    This creates a true A/B test:
    - No existing responses used
    - Dual PA establishes governance fresh
    - All responses generated with dual PA active
    """
    conversation_id = session_data['conversation_id']
    user_pa_config = session_data['primacy_attractor']
    conversation_starters = session_data['conversation_starters']

    print(f"\n  Regenerating {conversation_id} with DUAL PA governance...")
    print(f"    User PA Purpose: {user_pa_config['purpose'][0][:70]}...")
    print(f"    Total conversation starters: {len(conversation_starters)}")

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

    # Initialize governance (dual PA derivation)
    try:
        await orchestrator.initialize_governance()
    except Exception as e:
        print(f"    WARNING: Governance initialization failed: {e}")
        return None

    # Start session
    orchestrator.start_session(session_id=f"{conversation_id}_dual_pa_fresh")

    # Get governance mode
    actual_mode = orchestrator.actual_governance_mode
    is_dual_mode = orchestrator.dual_pa.is_dual_mode() if orchestrator.dual_pa else False
    correlation = orchestrator.dual_pa.correlation if is_dual_mode else None

    print(f"    Governance mode: {actual_mode} (dual PA: {is_dual_mode})")
    if correlation:
        print(f"    PA correlation: {correlation:.3f}")

    # Build conversation context incrementally
    conversation_context = []
    turn_results = []

    for turn_num, user_input in enumerate(conversation_starters, 1):
        # Add user message to context
        conversation_context.append({
            "role": "user",
            "content": user_input
        })

        # Generate response with dual PA governance
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
            'turn': turn_num,
            'user_input': user_input,
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
        if turn_num % 10 == 0:
            print(f"    Progress: {turn_num}/{len(conversation_starters)} turns completed")

    # End session
    summary = orchestrator.end_session()

    # Compile analysis
    analysis = {
        'conversation_id': conversation_id,
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
    """Regenerate Claude conversation from starters only."""
    print("=" * 80)
    print("CLAUDE CONVERSATION DUAL PA REGENERATION (STARTERS ONLY)")
    print("=" * 80)
    print("Testing dual PA governance on YOUR actual conversation")
    print("where you struggled with drift/misalignment.")
    print("\nMethodology: ISOLATED SESSION")
    print("  - Uses ONLY conversation starters (no existing responses)")
    print("  - Dual PA establishes governance fresh from scratch")
    print("  - ALL responses generated with dual PA active")
    print("  - True A/B test like ShareGPT comparison")
    print("=" * 80)

    # Load the conversation starters file
    starters_file = Path("claude_conversation_starters_only.json")

    if not starters_file.exists():
        print(f"ERROR: Starters file not found: {starters_file}")
        return

    print(f"\nLoading conversation starters: {starters_file.name}")
    with open(starters_file, 'r') as f:
        session_data = json.load(f)

    print(f"  Conversation ID: {session_data['conversation_id']}")
    print(f"  Total conversation starters: {len(session_data['conversation_starters'])}")
    print(f"  User PA Purpose: {session_data['primacy_attractor']['purpose'][0][:80]}...")

    # Initialize LLM and embedding provider
    print("\nInitializing LLM client and embedding provider...")
    llm_client = MistralClient(api_key=MISTRAL_API_KEY, model="mistral-large-latest")
    embedding_provider = SentenceTransformer('all-MiniLM-L6-v2')
    print("  ✓ Ready")

    # Regenerate with dual PA
    print(f"\nRegenerating conversation with DUAL PA governance...")

    try:
        result = await regenerate_from_starters(
            session_data=session_data,
            llm_client=llm_client,
            embedding_provider=embedding_provider
        )

        if result:
            print(f"\n    ✓ Regeneration completed ({len(result['turns'])} turns)")

            # Calculate summary statistics
            if result['dual_pa_used']:
                user_fidelities = [t.get('user_pa_fidelity', 0) for t in result['turns'] if 'user_pa_fidelity' in t]
                ai_fidelities = [t.get('ai_pa_fidelity', 0) for t in result['turns'] if 'ai_pa_fidelity' in t]
                interventions = sum(1 for t in result['turns'] if t.get('intervention_applied', False))

                print("\n" + "=" * 80)
                print("RESULTS SUMMARY")
                print("=" * 80)
                print(f"Governance mode:        {result['governance_mode_actual']}")
                print(f"Dual PA correlation:    {result['dual_pa_correlation']:.4f}")
                print(f"Total turns:            {len(result['turns'])}")

                if user_fidelities:
                    print(f"\nDual PA User fidelity:  {sum(user_fidelities)/len(user_fidelities):.4f}")
                    print(f"Dual PA AI fidelity:    {sum(ai_fidelities)/len(ai_fidelities):.4f}")

                print(f"\nInterventions applied:  {interventions}")
                print(f"Intervention rate:      {interventions/len(result['turns']):.2%}")

                # Show User PA vs AI PA
                print("\n" + "-" * 80)
                print("USER PA (Your Purpose):")
                for p in result['user_pa']['purpose'][:3]:
                    print(f"  • {p}")

                if result['ai_pa']:
                    print("\nAI PA (Derived Purpose):")
                    for p in result['ai_pa']['purpose'][:3]:
                        print(f"  • {p}")

            # Save results
            output_file = Path("claude_conversation_dual_pa_fresh_results.json")
            with open(output_file, 'w') as f:
                json.dump({
                    'result': result,
                    'methodology': {
                        'type': 'isolated_regeneration',
                        'description': 'Dual PA governance on conversation starters only',
                        'approach': 'True A/B test - no existing responses used',
                        'comparison_baseline': 'Original Claude conversation (from phase2 analysis)',
                        'treatment': 'Fresh dual PA governed responses (all generated new)'
                    }
                }, f, indent=2)

            print(f"\n✓ Full results saved to: {output_file}")
            print("\n" + "=" * 80)
            print("REGENERATION COMPLETE")
            print("=" * 80)
            print("\nThis demonstrates dual PA effectiveness on YOUR actual")
            print("conversation where you experienced drift - a real-world")
            print("stress test of the governance system.")
        else:
            print(f"\n    ✗ Failed to regenerate conversation")

    except Exception as e:
        print(f"\n    ✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
