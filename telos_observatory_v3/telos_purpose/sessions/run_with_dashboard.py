"""Interactive session with live dashboard."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from telos_purpose.llm_clients.mistral_client import TelosMistralClient
from telos_purpose.core.embedding_provider import SentenceTransformerProvider, DeterministicEmbeddingProvider
from telos_purpose.core.unified_steward import UnifiedGovernanceSteward, PrimacyAttractor
from telos_purpose.dev_dashboard.dashboard import DevDashboard


def main():
    parser = argparse.ArgumentParser(description="Interactive session with dashboard")
    parser.add_argument("--config", required=True, help="Config file")
    parser.add_argument("--model", default="mistral-small-latest", help="Mistral model")
    parser.add_argument("--use-real-embeddings", action="store_true")
    parser.add_argument("--output", help="Save session to file")

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    try:
        llm = TelosMistralClient(model=args.model)
    except ValueError as e:
        print(f"Error: {e}")
        print("Set MISTRAL_API_KEY environment variable")
        sys.exit(1)

    if args.use_real_embeddings:
        try:
            embedding_provider = SentenceTransformerProvider()
        except ImportError:
            print("Error: sentence-transformers not installed")
            print("Install: pip install sentence-transformers")
            sys.exit(1)
    else:
        embedding_provider = DeterministicEmbeddingProvider()

    attractor = PrimacyAttractor(
        purpose=config.get("purpose", []),
        scope=config.get("scope", []),
        boundaries=config.get("boundaries", []),
        privacy_level=float(config.get("privacy_level", 0.8)),
        constraint_tolerance=float(config.get("constraint_tolerance", 0.2)),
        task_priority=float(config.get("task_priority", 0.7))
    )

    steward = UnifiedGovernanceSteward(
        attractor=attractor,
        llm_client=llm,
        embedding_provider=embedding_provider,
        enable_interventions=True,
        dev_commentary_mode="silent"
    )

    dashboard = DevDashboard(steward)

    print("\n" + "=" * 60)
    print("TELOS INTERACTIVE SESSION WITH DASHBOARD")
    print("=" * 60)
    print("\nDashboard commands: status, explain, diagnose, history, intervention, watch, help")
    print("Or just chat normally. Type 'quit' to exit.\n")

    steward.start_session()

    while True:
        user_input = input("> ").strip()

        if not user_input:
            continue

        if user_input == 'quit':
            break

        if user_input in ['status', 'explain', 'diagnose', 'history', 'intervention', 'watch', 'help']:
            dashboard.cmd(user_input)
            continue

        messages = steward.conversation.get_messages_for_api()
        messages.append({"role": "user", "content": user_input})

        try:
            model_response = llm.generate(messages=messages, max_tokens=500)
        except Exception as e:
            print(f"\nError: {e}\n")
            continue

        result = steward.process_turn(user_input, model_response)

        print(f"\n{result['final_response']}\n")

        if result['response_was_modified']:
            mod_type = result['intervention_result'].type
            print(f"[Modified via {mod_type}]\n")

        if dashboard.watch_mode:
            dashboard.cmd('status')

    print("\nEnding session...")
    summary = steward.end_session()

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Session saved to: {output_path}")

    print("\nSession complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
