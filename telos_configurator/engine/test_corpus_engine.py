"""
Test script for CorpusEngine.

Quick validation of core functionality.
"""

import json
import tempfile
from pathlib import Path
from corpus_engine import CorpusEngine, get_corpus_engine


def test_basic_functionality():
    """Test basic corpus engine operations."""
    print("Testing CorpusEngine...")

    # Create engine instance
    engine = CorpusEngine()
    print("✓ Engine initialized")

    # Test singleton pattern
    engine2 = get_corpus_engine()
    print("✓ Singleton pattern works")

    # Test stats on empty corpus
    stats = engine.get_stats()
    assert stats["total_documents"] == 0
    print("✓ Stats on empty corpus")

    # Test embedding status
    status = engine.get_embedding_status()
    assert status["total_documents"] == 0
    print("✓ Embedding status")

    # Test list documents
    docs = engine.list_documents()
    assert len(docs) == 0
    print("✓ List documents (empty)")

    print("\nAll basic tests passed!")


def test_document_parsing():
    """Test document structure parsing."""
    print("\nTesting document parsing...")

    engine = CorpusEngine()

    # Test JSON parsing
    json_content = json.dumps({
        "title": "Test Policy",
        "text_content": "This is a test policy document.",
        "key_provisions": ["Provision 1", "Provision 2"],
        "escalation_triggers": ["Trigger 1"]
    })

    parsed = engine._parse_document_structure(json_content, "test.json")
    assert parsed["title"] == "Test Policy"
    assert len(parsed["key_provisions"]) == 2
    assert len(parsed["escalation_triggers"]) == 1
    print("✓ JSON parsing")

    # Test plain text parsing
    text_content = "This is plain text content."
    parsed = engine._parse_document_structure(text_content, "test.txt")
    assert parsed["title"] == "test.txt"
    assert parsed["text_content"] == text_content
    assert len(parsed["key_provisions"]) == 0
    print("✓ Plain text parsing")

    print("Document parsing tests passed!")


def test_persistence():
    """Test save/load functionality."""
    print("\nTesting corpus persistence...")

    engine = CorpusEngine()

    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        temp_path = f.name

    try:
        # Test save empty corpus
        success, message = engine.save_corpus(temp_path)
        assert success
        print("✓ Save empty corpus")

        # Test load corpus
        success, message = engine.load_corpus(temp_path)
        assert success
        print("✓ Load corpus")

        # Verify stats after load
        stats = engine.get_stats()
        assert stats["total_documents"] == 0
        print("✓ Verify stats after load")

    finally:
        # Cleanup
        Path(temp_path).unlink(missing_ok=True)

    print("Persistence tests passed!")


def test_cosine_similarity():
    """Test cosine similarity calculation."""
    print("\nTesting cosine similarity...")

    import numpy as np
    engine = CorpusEngine()

    # Test identical vectors
    vec1 = np.array([1.0, 0.0, 0.0])
    vec2 = np.array([1.0, 0.0, 0.0])
    sim = engine._cosine_similarity(vec1, vec2)
    assert abs(sim - 1.0) < 0.001
    print("✓ Identical vectors (similarity = 1.0)")

    # Test orthogonal vectors
    vec1 = np.array([1.0, 0.0, 0.0])
    vec2 = np.array([0.0, 1.0, 0.0])
    sim = engine._cosine_similarity(vec1, vec2)
    assert abs(sim) < 0.001
    print("✓ Orthogonal vectors (similarity = 0.0)")

    # Test opposite vectors
    vec1 = np.array([1.0, 0.0, 0.0])
    vec2 = np.array([-1.0, 0.0, 0.0])
    sim = engine._cosine_similarity(vec1, vec2)
    assert abs(sim - (-1.0)) < 0.001
    print("✓ Opposite vectors (similarity = -1.0)")

    # Test zero vector handling
    vec1 = np.array([1.0, 0.0, 0.0])
    vec2 = np.array([0.0, 0.0, 0.0])
    sim = engine._cosine_similarity(vec1, vec2)
    assert sim == 0.0
    print("✓ Zero vector handling")

    print("Cosine similarity tests passed!")


if __name__ == "__main__":
    print("=" * 60)
    print("TELOS Corpus Engine Test Suite")
    print("=" * 60)

    try:
        test_basic_functionality()
        test_document_parsing()
        test_persistence()
        test_cosine_similarity()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
