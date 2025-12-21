"""CLI tool for extracting governance profiles from conversation history."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from telos_purpose.llm_clients.mistral_client import TelosMistralClient
from telos_purpose.profiling.profile_extractor import ProfileExtractor


def main():
    parser = argparse.ArgumentParser(
        description="Extract governance profile from conversation history"
    )
    parser.add_argument(
        "input",
        help="Conversation file (txt, json, csv, etc.)"
    )
    parser.add_argument(
        "--output",
        default="extracted_profile.json",
        help="Output profile file (default: extracted_profile.json)"
    )
    parser.add_argument(
        "--model",
        default="mistral-small-latest",
        help="Mistral model to use"
    )
    parser.add_argument(
        "--show-reasoning",
        action="store_true",
        help="Display extraction reasoning"
    )

    args = parser.parse_args()

    try:
        llm = TelosMistralClient(model=args.model)
    except ValueError as e:
        print(f"Error: {e}")
        print("Set MISTRAL_API_KEY environment variable")
        sys.exit(1)

    print(f"Analyzing conversation: {args.input}")
    print(f"Using model: {args.model}\n")

    extractor = ProfileExtractor(llm)

    try:
        profile, confidence = extractor.extract_from_file(args.input)
    except FileNotFoundError:
        print(f"Error: File not found: {args.input}")
        sys.exit(1)
    except Exception as e:
        print(f"Error extracting profile: {e}")
        sys.exit(1)

    print("=" * 60)
    print("EXTRACTED GOVERNANCE PERIMETERS")
    print("=" * 60)
    print(f"\nConstraint Rigidity Perimeter: {profile['constraint_rigidity']:.2f}")
    print(f"Privacy Perimeter: {profile['privacy_level']:.2f}")
    print(f"Task Priority Perimeter: {profile['task_priority']:.2f}")
    print(f"Confidence: {confidence:.2f}\n")

    print("Purpose Perimeter:")
    for p in profile['purpose']:
        print(f"  - {p}")

    print("\nScope Perimeter:")
    for s in profile['scope']:
        print(f"  - {s}")

    print("\nBoundary Perimeter:")
    for b in profile['boundaries']:
        print(f"  - {b}")

    if args.show_reasoning and 'reasoning' in profile:
        print(f"\nReasoning:\n{profile['reasoning']}")

    output_data = {
        "export_dir": "./purpose_protocol_exports",
        "purpose": profile["purpose"],
        "scope": profile["scope"],
        "boundaries": profile["boundaries"],
        "privacy_level": profile["privacy_level"],
        "constraint_rigidity": profile["constraint_rigidity"],
        "task_priority": profile["task_priority"],
        "extraction_confidence": confidence,
        "extraction_metadata": {
            "source_file": args.input,
            "model_used": args.model,
            "reasoning": profile.get("reasoning", "")
        }
    }

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"Profile saved to: {args.output}")
    print(f"{'=' * 60}\n")
    print("Use this profile with:")
    print(f"  python -m telos_purpose.sessions.run_with_dashboard \\")
    print(f"    --config {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
