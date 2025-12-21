"""
TELOS Runner Script

Run a single conversation with governance enabled.
Useful for testing constraint tolerance settings and observation mode.
"""

import argparse
import json
from telos_purpose.core.unified_steward import UnifiedGovernanceSteward, PrimacyAttractor
from telos_purpose.core.embedding_provider import DeterministicEmbeddingProvider
from telos_purpose.llm_clients.mistral_client import TelosMistralClient


def run_conversation(config_path: str, conversation_path: str, observation_mode: bool = False):
    # Load configuration
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Load test conversation
    with open(conversation_path, "r", encoding="utf-8") as f:
        conversation = json.load(f)

    # Initialize attractor
    attractor = PrimacyAttractor(
        purpose=config["purpose"],
        scope=config["scope"],
        boundaries=config["boundaries"],
        privacy_level=config.get("privacy_level", 0.8),
        constraint_tolerance=config.get("constraint_tolerance", 0.2),
        task_priority=config.get("task_priority", 0.7),
    )

    # Initialize clients
    llm = TelosMistralClient()
    embeddings = DeterministicEmbeddingProvider()

    # Initialize steward
    steward = UnifiedGovernanceSteward(
        attractor=attractor,
        llm_client=llm,
        embedding_provider=embeddings,
        enable_interventions=not observation_mode,
        dev_commentary_mode="verbose"
    )

    print("\n" + "=" * 60)
    if observation_mode:
        print("TELOS OBSERVATION MODE (Metrics Only, No Interventions)")
    else:
        print("TELOS GOVERNED SESSION")
    print("=" * 60)
    print(f"Config: {config_path}")
    print(f"Conversation: {conversation_path}")
    print(f"Constraint Tolerance: {config.get('constraint_tolerance', 0.2):.2f}")
    if observation_mode:
        print("Warning: Interventions DISABLED - logging metrics only")
    print("=" * 60 + "\n")

    # Run session
    steward.start_session("runner_test")
    for turn in conversation:
        user_input = turn["user"]
        model_response = turn["assistant"]

        result = steward.process_turn(user_input, model_response)

        print(f"\nUser: {user_input}")
        print(f"Assistant: {result['final_response']}")

        if observation_mode and result['intervention_applied']:
            print(f"[Would have intervened: {result['intervention_result'].type}]")
        elif not observation_mode and result['response_was_modified']:
            print(f"[Modified via: {result['governance_action']}]")

        print(f"Metrics: F={result['metrics']['telic_fidelity']:.3f}, "
              f"Basin={result['metrics']['primacy_basin_membership']}, "
              f"Error={result['metrics']['error_signal']:.3f}")

    summary = steward.end_session()

    print("\n" + "=" * 60)
    print("SESSION SUMMARY")
    print("=" * 60)
    print(f"Total Turns: {summary['total_turns']}")
    print(f"Final Fidelity: {summary['session_metadata']['final_telic_fidelity']:.3f}")
    print(f"Trajectory Stability: {summary['session_metadata']['trajectory_stability']:.3f}")
    print(f"Basin Adherence: {summary['session_metadata']['basin_adherence']:.3f}")
    print(f"Total Interventions: {summary['intervention_statistics']['total_interventions']}")
    if observation_mode:
        print("\nNote: Intervention count shows what WOULD have triggered in governed mode")
    print("=" * 60 + "\n")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TELOS on a test conversation")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config file")
    parser.add_argument("--conversation", type=str, required=True, help="Path to test conversation JSON")
    parser.add_argument("--observation-mode", action="store_true",
                        help="Run in observation mode (compute metrics but don't intervene)")
    args = parser.parse_args()

    run_conversation(args.config, args.conversation, args.observation_mode)
