#!/usr/bin/env python3
"""
Conversation Quality Analyzer
==============================

Analyzes filtered ShareGPT conversations and selects the top candidates
for TELOS Phase 2 counterfactual testing.

Selection criteria:
- English language only
- Fastest convergence (7-8 turns preferred)
- Highest convergence quality (centroid stability + variance)

NO LLM calls - pure statistical analysis.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any
from collections import Counter


class ConversationQualityAnalyzer:
    """
    Analyze and rank conversations by quality metrics.
    """

    # English stopwords for language detection
    ENGLISH_STOPWORDS = {
        'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
        'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
        'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
        'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their',
        'what', 'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go',
        'me', 'when', 'make', 'can', 'like', 'time', 'no', 'just', 'him',
        'know', 'take', 'people', 'into', 'year', 'your', 'good', 'some',
        'could', 'them', 'see', 'other', 'than', 'then', 'now', 'look', 'only',
        'come', 'its', 'over', 'think', 'also', 'back', 'after', 'use', 'two',
        'how', 'our', 'work', 'first', 'well', 'way', 'even', 'new', 'want',
        'because', 'any', 'these', 'give', 'day', 'most', 'us'
    }

    def __init__(
        self,
        input_file: Path,
        output_dir: Path,
        target_count: int = 25,
        speed_weight: float = 0.40,
        centroid_weight: float = 0.30,
        variance_weight: float = 0.30
    ):
        """
        Initialize analyzer.

        Args:
            input_file: Path to filtered conversations JSON
            output_dir: Where to save analysis results
            target_count: How many top conversations to select
            speed_weight: Weight for convergence speed (0-1)
            centroid_weight: Weight for centroid stability (0-1)
            variance_weight: Weight for variance stability (0-1)
        """
        self.input_file = input_file
        self.output_dir = output_dir
        self.target_count = target_count
        self.speed_weight = speed_weight
        self.centroid_weight = centroid_weight
        self.variance_weight = variance_weight

        # Validate weights
        total = speed_weight + centroid_weight + variance_weight
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total}")

    def detect_language(self, text: str) -> str:
        """
        Detect if text is English, Chinese, or other.

        Args:
            text: Text to analyze

        Returns:
            'english', 'chinese', or 'other'
        """
        if not text or len(text) < 10:
            return 'other'

        # Calculate ASCII ratio
        ascii_chars = sum(1 for c in text if ord(c) < 128)
        ascii_ratio = ascii_chars / len(text)

        # Chinese characters typically have high Unicode values
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        chinese_ratio = chinese_chars / len(text)

        if chinese_ratio > 0.3:
            return 'chinese'

        if ascii_ratio < 0.5:
            return 'other'

        # Check for English stopwords
        words = re.findall(r'\b\w+\b', text.lower())
        if not words:
            return 'other'

        stopword_count = sum(1 for w in words if w in self.ENGLISH_STOPWORDS)
        stopword_ratio = stopword_count / len(words)

        if stopword_ratio > 0.15:  # At least 15% stopwords
            return 'english'

        return 'other'

    def detect_conversation_language(self, conversation: Dict) -> str:
        """
        Detect overall language of a conversation.

        Args:
            conversation: Conversation dict with turns

        Returns:
            'english', 'chinese', or 'other'
        """
        # Sample first 3 turn pairs
        sample_turns = conversation['turns'][:3]
        texts = []
        for user_msg, assistant_msg in sample_turns:
            texts.append(user_msg)
            texts.append(assistant_msg)

        # Combine and detect
        combined_text = ' '.join(texts)
        return self.detect_language(combined_text)

    def calculate_quality_score(self, conversation: Dict, language: str) -> Dict[str, float]:
        """
        Calculate quality score for a conversation.

        Scoring (0-100 scale):
        - Convergence speed: 40 points (faster = better)
        - Centroid stability: 30 points (higher = better)
        - Variance stability: 30 points (higher = better)

        Args:
            conversation: Conversation dict
            language: Detected language

        Returns:
            Dict with score breakdown
        """
        metrics = conversation['convergence_metrics']
        conv_turn = metrics['turns_to_convergence']
        centroid_stab = metrics['centroid_stability']
        variance_stab = metrics['variance_stability']

        # Speed score (40 points max)
        # 7 turns = 40 pts, 12 turns = ~6.7 pts
        speed_score = self.speed_weight * 100 * (13 - conv_turn) / 6
        speed_score = max(0, min(self.speed_weight * 100, speed_score))

        # Centroid stability score (30 points max)
        centroid_score = self.centroid_weight * 100 * centroid_stab

        # Variance stability score (30 points max)
        variance_score = self.variance_weight * 100 * variance_stab

        # Total quality score
        total_score = speed_score + centroid_score + variance_score

        return {
            'total_score': round(total_score, 2),
            'speed_score': round(speed_score, 2),
            'centroid_score': round(centroid_score, 2),
            'variance_score': round(variance_score, 2),
            'convergence_turn': conv_turn,
            'centroid_stability': round(centroid_stab, 4),
            'variance_stability': round(variance_stab, 4),
            'language': language
        }

    def analyze_conversations(self) -> List[Dict]:
        """
        Analyze all conversations and return ranked list.

        Returns:
            List of conversation dicts with scores, sorted by quality
        """
        print("📊 Analyzing conversation quality...\n")

        # Load conversations
        with open(self.input_file, 'r') as f:
            conversations = json.load(f)

        print(f"Loaded {len(conversations)} conversations")

        # Analyze each
        analyzed = []
        language_counts = Counter()

        for conv in conversations:
            # Detect language
            language = self.detect_conversation_language(conv)
            language_counts[language] += 1

            # Calculate scores
            scores = self.calculate_quality_score(conv, language)

            # Add to analysis
            analyzed.append({
                'id': conv['id'],
                'turn_count': conv['turn_count'],
                'language': language,
                'scores': scores,
                'conversation': conv  # Keep full conversation for export
            })

        # Sort by total score (descending)
        analyzed.sort(key=lambda x: x['scores']['total_score'], reverse=True)

        # Add rankings
        for rank, item in enumerate(analyzed, start=1):
            item['rank'] = rank

        print(f"\nLanguage distribution:")
        for lang, count in language_counts.most_common():
            print(f"  - {lang}: {count}")

        return analyzed

    def select_top_conversations(self, analyzed: List[Dict]) -> List[Dict]:
        """
        Select top N English conversations.

        Args:
            analyzed: List of analyzed conversations

        Returns:
            Top N conversations (English only)
        """
        # Filter for English
        english_convs = [c for c in analyzed if c['language'] == 'english']

        print(f"\nEnglish conversations: {len(english_convs)}")

        if len(english_convs) < self.target_count:
            print(f"⚠️  Warning: Only found {len(english_convs)} English conversations")
            print(f"   Requested: {self.target_count}")
            return english_convs

        # Select top N
        selected = english_convs[:self.target_count]

        print(f"✅ Selected top {len(selected)} English conversations")

        return selected

    def generate_reports(self, analyzed: List[Dict], selected: List[Dict]):
        """
        Generate JSON and Markdown quality reports.

        Args:
            analyzed: All analyzed conversations
            selected: Top selected conversations
        """
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # JSON report
        json_report = {
            'analysis_date': '2025-10-30',
            'total_conversations': len(analyzed),
            'target_count': self.target_count,
            'selected_count': len(selected),
            'selection_criteria': {
                'language': 'english',
                'scoring_weights': {
                    'convergence_speed': self.speed_weight,
                    'centroid_stability': self.centroid_weight,
                    'variance_stability': self.variance_weight
                }
            },
            'language_distribution': dict(Counter(c['language'] for c in analyzed)),
            'score_statistics': {
                'min': min(c['scores']['total_score'] for c in analyzed),
                'max': max(c['scores']['total_score'] for c in analyzed),
                'mean': sum(c['scores']['total_score'] for c in analyzed) / len(analyzed),
                'median': sorted([c['scores']['total_score'] for c in analyzed])[len(analyzed) // 2]
            },
            'top_selected': [
                {
                    'rank': c['rank'],
                    'id': c['id'],
                    'language': c['language'],
                    'turn_count': c['turn_count'],
                    'scores': c['scores']
                }
                for c in selected
            ],
            'all_conversations': [
                {
                    'rank': c['rank'],
                    'id': c['id'],
                    'language': c['language'],
                    'turn_count': c['turn_count'],
                    'scores': c['scores']
                }
                for c in analyzed
            ]
        }

        json_path = self.output_dir / 'quality_analysis_report.json'
        with open(json_path, 'w') as f:
            json.dump(json_report, f, indent=2)

        print(f"\n💾 JSON report saved: {json_path}")

        # Markdown report
        md_lines = [
            "# Conversation Quality Analysis Report",
            "",
            f"**Analysis Date**: 2025-10-30",
            f"**Total Conversations**: {len(analyzed)}",
            f"**English Conversations**: {len([c for c in analyzed if c['language'] == 'english'])}",
            f"**Selected for Testing**: {len(selected)}",
            "",
            "---",
            "",
            "## Selection Criteria",
            "",
            "**Language**: English only",
            "",
            "**Scoring Weights**:",
            f"- Convergence Speed: {self.speed_weight*100:.0f}%",
            f"- Centroid Stability: {self.centroid_weight*100:.0f}%",
            f"- Variance Stability: {self.variance_weight*100:.0f}%",
            "",
            "---",
            "",
            "## Top 25 Selected Conversations",
            "",
            "| Rank | ID | Turn Count | Total Score | Speed | Centroid | Variance | Conv Turn |",
            "|------|----|-----------:|------------:|------:|---------:|---------:|----------:|"
        ]

        for c in selected[:25]:
            s = c['scores']
            md_lines.append(
                f"| {c['rank']} | {c['id']} | {c['turn_count']} | "
                f"{s['total_score']:.1f} | {s['speed_score']:.1f} | "
                f"{s['centroid_score']:.1f} | {s['variance_score']:.1f} | "
                f"{s['convergence_turn']} |"
            )

        md_lines.extend([
            "",
            "---",
            "",
            "## Score Distribution",
            "",
            f"- **Min Score**: {json_report['score_statistics']['min']:.2f}",
            f"- **Max Score**: {json_report['score_statistics']['max']:.2f}",
            f"- **Mean Score**: {json_report['score_statistics']['mean']:.2f}",
            f"- **Median Score**: {json_report['score_statistics']['median']:.2f}",
            "",
            "---",
            "",
            "## Language Distribution",
            ""
        ])

        for lang, count in sorted(json_report['language_distribution'].items()):
            pct = (count / len(analyzed)) * 100
            md_lines.append(f"- **{lang.title()}**: {count} ({pct:.1f}%)")

        md_lines.extend([
            "",
            "---",
            "",
            "## Sample Conversation",
            "",
            f"**ID**: {selected[0]['id']}",
            f"**Rank**: #{selected[0]['rank']}",
            f"**Score**: {selected[0]['scores']['total_score']:.1f}",
            f"**Convergence**: Turn {selected[0]['scores']['convergence_turn']}",
            "",
            "**First exchange**:",
            f"- User: {selected[0]['conversation']['turns'][0][0][:100]}...",
            f"- Assistant: {selected[0]['conversation']['turns'][0][1][:100]}...",
            "",
            "---",
            "",
            "**Generated by**: TELOSCOPE Observatory",
            "**Purpose**: Phase 2 TELOS counterfactual testing preparation"
        ])

        md_path = self.output_dir / 'quality_analysis_report.md'
        with open(md_path, 'w') as f:
            f.write('\n'.join(md_lines))

        print(f"📄 Markdown report saved: {md_path}")

    def export_top_conversations(self, selected: List[Dict]):
        """
        Export top conversations to new JSON file.

        Args:
            selected: Top selected conversations
        """
        # Extract just the conversation data (no analysis metadata)
        top_conversations = [c['conversation'] for c in selected]

        output_path = self.output_dir / 'sharegpt_top25_conversations.json'
        with open(output_path, 'w') as f:
            json.dump(top_conversations, f, indent=2)

        print(f"💾 Top {len(selected)} conversations saved: {output_path}")

    def run(self):
        """Run full analysis pipeline."""
        print("=" * 70)
        print("CONVERSATION QUALITY ANALYSIS")
        print("=" * 70)
        print()

        # Step 1: Analyze
        analyzed = self.analyze_conversations()

        # Step 2: Select top
        selected = self.select_top_conversations(analyzed)

        # Step 3: Generate reports
        self.generate_reports(analyzed, selected)

        # Step 4: Export selected
        self.export_top_conversations(selected)

        print()
        print("=" * 70)
        print("✅ ANALYSIS COMPLETE")
        print("=" * 70)
        print()
        print("Results:")
        print(f"  - Analyzed: {len(analyzed)} conversations")
        print(f"  - Selected: {len(selected)} English conversations")
        print(f"  - Avg Score: {sum(c['scores']['total_score'] for c in selected) / len(selected):.1f}")
        print(f"  - Score Range: {min(c['scores']['total_score'] for c in selected):.1f} - {max(c['scores']['total_score'] for c in selected):.1f}")
        print()
        print("Next steps:")
        print("  1. Review quality reports in sharegpt_data/")
        print("  2. Proceed to Phase 2 counterfactual testing")
        print("  3. Run LLM semantic analysis when ready")
        print()


def main():
    """Run conversation quality analysis."""
    # Paths
    input_file = Path(__file__).parent / 'sharegpt_data' / 'sharegpt_filtered_conversations.json'
    output_dir = Path(__file__).parent / 'sharegpt_data'

    # Create analyzer
    analyzer = ConversationQualityAnalyzer(
        input_file=input_file,
        output_dir=output_dir,
        target_count=25,
        speed_weight=0.40,
        centroid_weight=0.30,
        variance_weight=0.30
    )

    # Run analysis
    analyzer.run()


if __name__ == '__main__':
    main()
