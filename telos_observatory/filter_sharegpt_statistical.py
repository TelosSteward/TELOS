#!/usr/bin/env python3
"""
ShareGPT Statistical Convergence Filter
========================================

Filters ShareGPT conversations using ONLY statistical convergence detection.
NO LLM API calls during filtering - only embeddings.

This uses the same turn-by-turn historical processing as runtime,
ensuring NO future knowledge leakage.

Cost: ~$0.01-0.05 for 500 conversations (embeddings only)
Time: ~12-15 minutes
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any
from datasets import load_dataset
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from telos_purpose.profiling.progressive_primacy_extractor import ProgressivePrimacyExtractor
from telos_purpose.core.embedding_provider import EmbeddingProvider


class ShareGPTStatisticalFilter:
    """
    Filter ShareGPT conversations using statistical convergence.

    Architecture:
    - Turn-by-turn processing (historical data only)
    - Statistical convergence detection (no LLM)
    - Embeddings for distance measurement
    - Filter for conversations converging in target range
    """

    def __init__(
        self,
        min_turns: int = 10,
        max_turns: int = 25,
        target_convergence_turns: int = 10,
        convergence_tolerance: int = 3,
        num_conversations: int = 500,
        window_size: int = 3,
        centroid_stability_threshold: float = 0.95,
        variance_stability_threshold: float = 0.15,
        confidence_threshold: float = 0.75,
        consecutive_stable_turns: int = 2
    ):
        """
        Initialize statistical filter.

        Args:
            min_turns: Minimum conversation length
            max_turns: Maximum conversation length
            target_convergence_turns: Target convergence turn
            convergence_tolerance: +/- tolerance for target
            num_conversations: How many to filter
            window_size: Statistical window size
            centroid_stability_threshold: Convergence threshold
            variance_stability_threshold: Variance threshold
            confidence_threshold: Confidence threshold
            consecutive_stable_turns: Stability duration
        """
        self.min_turns = min_turns
        self.max_turns = max_turns
        self.target_convergence_turns = target_convergence_turns
        self.convergence_tolerance = convergence_tolerance
        self.num_conversations = num_conversations

        # Statistical convergence parameters
        self.window_size = window_size
        self.centroid_stability_threshold = centroid_stability_threshold
        self.variance_stability_threshold = variance_stability_threshold
        self.confidence_threshold = confidence_threshold
        self.consecutive_stable_turns = consecutive_stable_turns

        # Initialize embedding provider
        print("🔧 Initializing embedding provider...")
        self.embedding_provider = EmbeddingProvider(deterministic=False)

        print(f"✅ Filter initialized")
        print(f"   Target: {num_conversations} conversations")
        print(f"   Length: {min_turns}-{max_turns} turns")
        print(f"   Convergence target: {target_convergence_turns} ± {convergence_tolerance} turns")

    def load_sharegpt_dataset(self):
        """Load ShareGPT dataset from HuggingFace."""
        print("\n🔄 Loading ShareGPT dataset...")

        try:
            dataset = load_dataset(
                "anon8231489123/ShareGPT_Vicuna_unfiltered",
                split="train",
                streaming=True
            )
            print("✅ Dataset loaded: ShareGPT_Vicuna_unfiltered")
            return dataset
        except Exception as e:
            print(f"⚠️  Primary dataset failed: {e}")
            print("🔄 Trying fallback dataset...")
            dataset = load_dataset(
                "philschmid/sharegpt-raw",
                split="train",
                streaming=True
            )
            print("✅ Dataset loaded: sharegpt-raw")
            return dataset

    def parse_conversation(self, item: Dict) -> List[tuple]:
        """
        Parse conversation from ShareGPT format.

        Returns:
            List of (user_message, assistant_response) tuples
        """
        # Get conversations field
        if 'conversations' in item:
            conversations = item['conversations']
        elif 'messages' in item:
            conversations = item['messages']
        else:
            return []

        # Parse into turn pairs
        turns = []
        for i in range(0, len(conversations) - 1, 2):
            if i + 1 >= len(conversations):
                break

            human_msg = conversations[i]
            assistant_msg = conversations[i + 1]

            # Validate format
            if (human_msg.get('from') in ['human', 'user'] and
                assistant_msg.get('from') in ['gpt', 'assistant']):

                user_text = human_msg.get('value', '').strip()
                assistant_text = assistant_msg.get('value', '').strip()

                # Basic quality filters
                if len(user_text) < 10 or len(assistant_text) < 20:
                    return []  # Too short

                # Skip problematic content
                problematic = ['[INST]', '<<SYS>>', 'sorry, I cannot', 'I apologize']
                if any(marker.lower() in assistant_text.lower() for marker in problematic):
                    return []

                turns.append((user_text, assistant_text))
            else:
                return []  # Invalid format

        return turns

    def test_convergence(self, turns: List[tuple]) -> Dict[str, Any]:
        """
        Test conversation for statistical convergence.

        Uses historical-only processing - each turn sees only previous turns.

        Args:
            turns: List of (user, assistant) tuples

        Returns:
            Dict with convergence metrics or None if failed
        """
        # Initialize extractor with NO LLM (statistical only)
        extractor = ProgressivePrimacyExtractor(
            llm_client=None,  # NO LLM - statistical convergence only
            embedding_provider=self.embedding_provider,
            mode='progressive',
            window_size=self.window_size,
            centroid_stability_threshold=self.centroid_stability_threshold,
            variance_stability_threshold=self.variance_stability_threshold,
            confidence_threshold=self.confidence_threshold,
            consecutive_stable_turns=self.consecutive_stable_turns,
            max_turns_safety=self.max_turns
        )

        # Process turn-by-turn (historical only)
        for turn_idx, (user_msg, assistant_msg) in enumerate(turns, start=1):
            result = extractor.add_turn(user_msg, assistant_msg)

            # Check if converged
            if result['status'] == 'converged_statistical_only':
                return {
                    'converged': True,
                    'convergence_turn': result['convergence_turn'],
                    'metrics': result['convergence_metrics'],
                    'total_turns': len(turns)
                }

        # Did not converge
        return {
            'converged': False,
            'convergence_turn': None,
            'metrics': None,
            'total_turns': len(turns)
        }

    def filter_conversations(self, output_dir: Path):
        """
        Filter ShareGPT conversations using statistical convergence.

        Args:
            output_dir: Where to save filtered conversations
        """
        output_dir.mkdir(exist_ok=True, parents=True)

        # Load dataset
        dataset = self.load_sharegpt_dataset()

        # Results tracking
        filtered_conversations = []
        convergence_stats = []
        processed = 0
        skipped_length = 0
        skipped_parse = 0
        skipped_no_convergence = 0
        skipped_convergence_range = 0

        print(f"\n📊 Processing conversations...")
        print(f"   Progress will be shown every 50 items\n")

        # Process conversations
        pbar = tqdm(total=self.num_conversations, desc="Filtering", unit="conv")

        for item in dataset:
            processed += 1

            # Parse conversation
            turns = self.parse_conversation(item)

            if not turns:
                skipped_parse += 1
                continue

            # Filter by length
            if not (self.min_turns <= len(turns) <= self.max_turns):
                skipped_length += 1
                continue

            # Test for statistical convergence (historical processing)
            convergence_result = self.test_convergence(turns)

            # Check if converged
            if not convergence_result['converged']:
                skipped_no_convergence += 1
                continue

            # Check if converged within target range
            conv_turn = convergence_result['convergence_turn']
            target_min = self.target_convergence_turns - self.convergence_tolerance
            target_max = self.target_convergence_turns + self.convergence_tolerance

            if not (target_min <= conv_turn <= target_max):
                skipped_convergence_range += 1
                continue

            # PASSED ALL FILTERS - save it
            conversation_record = {
                'id': f"sharegpt_filtered_{len(filtered_conversations) + 1}",
                'turns': turns,
                'turn_count': len(turns),
                'convergence_turn': conv_turn,
                'convergence_metrics': convergence_result['metrics'],
                'source': 'ShareGPT'
            }

            filtered_conversations.append(conversation_record)
            convergence_stats.append(conv_turn)

            pbar.update(1)
            pbar.set_postfix({
                'converged': len(filtered_conversations),
                'processed': processed
            })

            # Check if we have enough
            if len(filtered_conversations) >= self.num_conversations:
                break

        pbar.close()

        # Print summary
        print(f"\n" + "=" * 70)
        print("FILTERING SUMMARY")
        print("=" * 70)
        print(f"Processed: {processed} items")
        print(f"Filtered: {len(filtered_conversations)} conversations")
        print(f"\nSkipped:")
        print(f"  - Parse failures: {skipped_parse}")
        print(f"  - Length filter: {skipped_length}")
        print(f"  - No convergence: {skipped_no_convergence}")
        print(f"  - Convergence out of range: {skipped_convergence_range}")
        print()

        if filtered_conversations:
            import numpy as np
            print("Convergence Statistics:")
            print(f"  Mean: {np.mean(convergence_stats):.1f} turns")
            print(f"  Median: {np.median(convergence_stats):.1f} turns")
            print(f"  Min: {np.min(convergence_stats)} turns")
            print(f"  Max: {np.max(convergence_stats)} turns")
            print()

        # Save filtered conversations
        output_file = output_dir / "sharegpt_filtered_conversations.json"
        with open(output_file, 'w') as f:
            json.dump(filtered_conversations, f, indent=2)

        print(f"💾 Saved to: {output_file}")

        # Save convergence stats
        stats_file = output_dir / "convergence_statistics.json"
        stats_data = {
            'num_conversations': len(filtered_conversations),
            'convergence_turns': convergence_stats,
            'mean_convergence': float(np.mean(convergence_stats)) if convergence_stats else 0,
            'median_convergence': float(np.median(convergence_stats)) if convergence_stats else 0,
            'filter_config': {
                'min_turns': self.min_turns,
                'max_turns': self.max_turns,
                'target_convergence': self.target_convergence_turns,
                'tolerance': self.convergence_tolerance,
                'window_size': self.window_size,
                'centroid_stability_threshold': self.centroid_stability_threshold,
                'variance_stability_threshold': self.variance_stability_threshold,
                'confidence_threshold': self.confidence_threshold
            }
        }

        with open(stats_file, 'w') as f:
            json.dump(stats_data, f, indent=2)

        print(f"📊 Statistics saved to: {stats_file}")

        # Print sample
        if filtered_conversations:
            print(f"\n📋 Sample conversation:")
            sample = filtered_conversations[0]
            print(f"   ID: {sample['id']}")
            print(f"   Turns: {sample['turn_count']}")
            print(f"   Converged at turn: {sample['convergence_turn']}")
            print(f"   First exchange:")
            print(f"      User: {sample['turns'][0][0][:100]}...")
            print(f"      Assistant: {sample['turns'][0][1][:100]}...")

        print("\n" + "=" * 70)
        print("✅ FILTERING COMPLETE")
        print("=" * 70)


def main():
    """Run ShareGPT statistical filtering."""

    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 10 + "ShareGPT Statistical Convergence Filter" + " " * 19 + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    print("Filters ShareGPT conversations using ONLY statistical convergence.")
    print("NO LLM API calls - just embeddings (very low cost).")
    print()
    print("Architecture:")
    print("  ✓ Turn-by-turn historical processing")
    print("  ✓ Statistical convergence detection")
    print("  ✓ No future knowledge leakage")
    print("  ✓ Same runtime architecture as counterfactual testing")
    print()

    # Parse command line arguments
    num_conversations = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    target_convergence = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    # Configuration
    output_dir = Path(__file__).parent / "sharegpt_data"

    print("Configuration:")
    print(f"  - Target conversations: {num_conversations}")
    print(f"  - Conversation length: 10-25 turns")
    print(f"  - Target convergence: {target_convergence} ± 3 turns")
    print(f"  - Output directory: {output_dir}")
    print(f"  - Relaxed convergence criteria for better yield")
    print()

    # Create filter with RELAXED parameters for better yield
    filter_obj = ShareGPTStatisticalFilter(
        min_turns=10,
        max_turns=25,
        target_convergence_turns=target_convergence,
        convergence_tolerance=3,
        num_conversations=num_conversations,
        window_size=3,
        centroid_stability_threshold=0.90,  # Relaxed from 0.95
        variance_stability_threshold=0.20,  # Relaxed from 0.15
        confidence_threshold=0.70,          # Relaxed from 0.75
        consecutive_stable_turns=2
    )

    # Run filtering
    filter_obj.filter_conversations(output_dir)

    print()
    print("💡 Next steps:")
    print("   1. Review filtered conversations in sharegpt_data/")
    print("   2. Use these for Phase 2 counterfactual testing")
    print("   3. Optionally run LLM analysis on filtered set later")
    print()


if __name__ == "__main__":
    main()
