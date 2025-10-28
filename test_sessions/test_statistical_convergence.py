#!/usr/bin/env python3
"""
Test Statistical Convergence System
====================================

Demonstrates the new statistical convergence approach:
1. NO arbitrary turn limits
2. Statistical convergence detection
3. Multi-session analysis with ConvergenceAnalyzer
4. Data-driven parameter recommendations
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from telos_purpose.sessions.session_loader import SessionLoader
from telos_purpose.profiling.progressive_primacy_extractor import ProgressivePrimacyExtractor
from telos_purpose.profiling.convergence_analyzer import ConvergenceAnalyzer, ConvergenceRecord
from telos_purpose.core.embedding_providers import get_embedding_function


def test_single_session_convergence():
    """Test statistical convergence on a single conversation."""

    print("=" * 70)
    print("TEST 1: Single Session Statistical Convergence")
    print("=" * 70)
    print()

    # Load the formatted Claude conversation
    loader = SessionLoader(debug=True)
    conversation_path = Path(__file__).parent / 'Claude_Conversation_formatted.txt'

    if not conversation_path.exists():
        print(f"❌ Error: {conversation_path} not found!")
        print("   Run format_conversation.py first to create this file.")
        return None

    print(f"Loading conversation from: {conversation_path.name}")
    session = loader.load(conversation_path)
    print(f"✅ Loaded {len(session.turns)} turns")
    print()

    # Initialize extractor with statistical convergence
    print("Initializing ProgressivePrimacyExtractor with statistical convergence...")
    print("Parameters:")
    print("  - window_size: 8 turns")
    print("  - centroid_stability_threshold: 0.95 (95% similarity required)")
    print("  - variance_stability_threshold: 0.15 (low variance required)")
    print("  - confidence_threshold: 0.75")
    print("  - consecutive_stable_turns: 3")
    print()

    embedding_fn = get_embedding_function(provider='mistral', model='mistral-embed')

    extractor = ProgressivePrimacyExtractor(
        embedding_function=embedding_fn,
        llm_analyzer=None,  # Skip LLM for now
        window_size=8,
        centroid_stability_threshold=0.95,
        variance_stability_threshold=0.15,
        confidence_threshold=0.75,
        consecutive_stable_turns=3,
    )

    # Process turns progressively
    print("Processing turns progressively...")
    print("-" * 70)

    for i, turn in enumerate(session.turns, start=1):
        extractor.add_turn(
            speaker=turn.speaker,
            message=turn.content
        )

        # Show progress every 5 turns
        if i % 5 == 0:
            status = extractor.get_status()
            print(f"Turn {i:3d}: stable_count={status['stable_turn_count']}, "
                  f"confidence={status.get('latest_confidence', 0):.3f}")

        # Check if converged
        if extractor.converged:
            print()
            print(f"🎯 CONVERGENCE DETECTED at turn {extractor.convergence_turn}!")
            break

    print("-" * 70)
    print()

    # Show final status
    status = extractor.get_status()
    print("Final Status:")
    print(f"  Converged: {status['converged']}")
    print(f"  Convergence turn: {status.get('convergence_turn', 'N/A')}")
    print(f"  Total turns processed: {status['turn_count']}")
    print(f"  Stable turn count: {status['stable_turn_count']}")
    print()

    if status['converged']:
        metrics = status.get('convergence_metrics', {})
        print("Convergence Metrics:")
        print(f"  Confidence: {metrics.get('confidence', 0):.3f}")
        print(f"  Centroid stability: {metrics.get('centroid_stability', 0):.3f}")
        print(f"  Variance stability: {metrics.get('variance_stability', 0):.3f}")
        print()

        # Get convergence record
        record = extractor.get_convergence_record(session_id='claude_conversation_1')
        return record

    return None


def test_multi_session_analysis(records: list):
    """Test ConvergenceAnalyzer with multiple session records."""

    print()
    print("=" * 70)
    print("TEST 2: Multi-Session Convergence Analysis")
    print("=" * 70)
    print()

    if not records:
        print("❌ No convergence records available for analysis")
        return

    # Create analyzer
    analyzer = ConvergenceAnalyzer()

    # Add records
    for record in records:
        analyzer.add_record(record)

    print(f"Added {len(records)} convergence record(s) to analyzer")
    print()

    # Print summary
    analyzer.print_summary()

    # Generate recommendations
    print()
    print("=" * 70)
    print("PARAMETER RECOMMENDATIONS")
    print("=" * 70)
    print()

    recommendations = analyzer.recommend_parameters()

    if 'error' not in recommendations:
        print("Recommended Parameters (data-driven):")
        print(f"  window_size: {recommendations['recommended_window_size']}")
        print(f"  confidence_threshold: {recommendations['recommended_confidence_threshold']}")
        print(f"  centroid_stability_threshold: {recommendations['recommended_centroid_threshold']}")
        print(f"  variance_stability_threshold: {recommendations['recommended_variance_threshold']}")
        print()
        print("Evidence:")
        for key, value in recommendations['evidence'].items():
            print(f"  {key}: {value}")
        print()

    # Export report
    output_path = Path(__file__).parent / 'convergence_analysis_report.json'
    analyzer.export_report(str(output_path))
    print(f"📊 Full report exported to: {output_path.name}")
    print()


def test_example_conversation():
    """Test with the simple example conversation."""

    print("=" * 70)
    print("TEST 3: Example Conversation")
    print("=" * 70)
    print()

    example_path = Path('/tmp/claude_conversation_example.txt')

    if not example_path.exists():
        print(f"❌ Error: {example_path} not found!")
        return None

    print(f"Loading: {example_path.name}")

    loader = SessionLoader(debug=False)
    session = loader.load(example_path)

    print(f"✅ Loaded {len(session.turns)} turns")
    print()

    embedding_fn = get_embedding_function(provider='mistral', model='mistral-embed')

    extractor = ProgressivePrimacyExtractor(
        embedding_function=embedding_fn,
        llm_analyzer=None,
        window_size=3,  # Smaller window for small conversation
        consecutive_stable_turns=2,
    )

    print("Processing turns...")
    for i, turn in enumerate(session.turns, start=1):
        extractor.add_turn(speaker=turn.speaker, message=turn.content)

        if extractor.converged:
            print(f"✅ Converged at turn {extractor.convergence_turn}")
            break

    status = extractor.get_status()
    print(f"Final turn count: {status['turn_count']}")
    print(f"Converged: {status['converged']}")
    print()

    if status['converged']:
        record = extractor.get_convergence_record(session_id='example_1')
        return record

    return None


def main():
    """Run all tests."""

    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "STATISTICAL CONVERGENCE TEST SUITE" + " " * 19 + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    print("Testing the new statistical convergence system:")
    print("  ✓ No arbitrary turn limits")
    print("  ✓ Rolling window centroid comparison")
    print("  ✓ Variance stability tracking")
    print("  ✓ Confidence scoring")
    print("  ✓ Multi-session analysis")
    print("  ✓ Data-driven parameter recommendations")
    print()

    # Collect convergence records
    records = []

    # Test 1: Main Claude conversation
    try:
        record1 = test_single_session_convergence()
        if record1:
            records.append(record1)
    except Exception as e:
        print(f"❌ Test 1 failed: {e}")
        import traceback
        traceback.print_exc()

    # Test 3: Example conversation
    try:
        record2 = test_example_conversation()
        if record2:
            records.append(record2)
    except Exception as e:
        print(f"❌ Test 3 failed: {e}")
        import traceback
        traceback.print_exc()

    # Test 2: Multi-session analysis
    if records:
        try:
            test_multi_session_analysis(records)
        except Exception as e:
            print(f"❌ Test 2 failed: {e}")
            import traceback
            traceback.print_exc()

    print()
    print("=" * 70)
    print("✅ TEST SUITE COMPLETE")
    print("=" * 70)
    print()


if __name__ == '__main__':
    main()
