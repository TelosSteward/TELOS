"""
Tests for telos_adapters.generic.decorator

Tests the @telos_governed decorator:
- Pass-through when no text input
- Blocking on low fidelity
- Decision classification (EXECUTE, CLARIFY, SUGGEST, INERT, ESCALATE)
- Custom on_block callback
- Metadata on wrapper
"""

import pytest
import numpy as np


def _make_embed_fn():
    """Create a mock embedding function that uses text content deterministically."""
    _cache = {}

    def embed(text: str) -> np.ndarray:
        if text not in _cache:
            h = hash(text) % 10000
            rng = np.random.RandomState(h)
            vec = rng.randn(32)
            _cache[text] = vec / np.linalg.norm(vec)
        return _cache[text]

    return embed


def _make_aligned_embed_fn(purpose_text: str):
    """
    Create an embedding function where similar text to purpose
    returns high cosine similarity and dissimilar returns low.
    """
    purpose_vec = np.ones(32) / np.sqrt(32)

    def embed(text: str) -> np.ndarray:
        if text == purpose_text:
            return purpose_vec.copy()
        # Simulate similarity based on shared words
        shared = len(set(purpose_text.lower().split()) & set(text.lower().split()))
        total = max(len(set(purpose_text.lower().split()) | set(text.lower().split())), 1)
        overlap = shared / total
        # Mix purpose vector with random noise based on overlap
        rng = np.random.RandomState(hash(text) % 10000)
        noise = rng.randn(32)
        noise = noise / np.linalg.norm(noise)
        vec = overlap * purpose_vec + (1 - overlap) * noise * 0.5
        return vec / np.linalg.norm(vec)

    return embed


class TestTelosGovernedImport:
    def test_import_from_generic(self):
        from telos_adapters.generic import telos_governed
        assert callable(telos_governed)

    def test_import_from_decorator_module(self):
        from telos_adapters.generic.decorator import telos_governed
        assert callable(telos_governed)


class TestPassThrough:
    def test_no_string_args_passes_through(self):
        from telos_adapters.generic.decorator import telos_governed
        embed = _make_embed_fn()

        @telos_governed(purpose="financial analysis", embed_fn=embed)
        def my_func(x: int) -> int:
            return x * 2

        assert my_func(5) == 10

    def test_empty_string_passes_through(self):
        from telos_adapters.generic.decorator import telos_governed
        embed = _make_embed_fn()

        @telos_governed(purpose="financial analysis", embed_fn=embed)
        def my_func(text: str) -> str:
            return f"processed: {text}"

        assert my_func("") == "processed: "


class TestGovernanceMetadata:
    def test_metadata_on_wrapper(self):
        from telos_adapters.generic.decorator import telos_governed
        embed = _make_embed_fn()

        @telos_governed(
            purpose="test purpose",
            threshold=0.80,
            high_risk=True,
            embed_fn=embed,
        )
        def my_func(text: str) -> str:
            return text

        assert my_func._telos_purpose == "test purpose"
        assert my_func._telos_threshold == 0.80
        assert my_func._telos_high_risk is True

    def test_functools_wraps_preserves_name(self):
        from telos_adapters.generic.decorator import telos_governed
        embed = _make_embed_fn()

        @telos_governed(purpose="test", embed_fn=embed)
        def my_special_function(text: str) -> str:
            """My docstring."""
            return text

        assert my_special_function.__name__ == "my_special_function"
        assert my_special_function.__doc__ == "My docstring."


class TestHighSimilarity:
    def test_identical_text_executes(self):
        """When input is identical to purpose, should always execute."""
        from telos_adapters.generic.decorator import telos_governed
        embed = _make_embed_fn()

        called = []

        @telos_governed(purpose="financial analysis", embed_fn=embed)
        def my_func(text: str) -> str:
            called.append(text)
            return f"result: {text}"

        # Same text as purpose -> cosine similarity 1.0 -> EXECUTE
        result = my_func("financial analysis")
        assert result == "result: financial analysis"
        assert len(called) == 1


class TestOnBlockCallback:
    def test_on_block_called_for_baseline_violation(self):
        """Test that on_block is called when similarity < SIMILARITY_BASELINE."""
        from telos_adapters.generic.decorator import telos_governed

        # Create embedding function where target text is orthogonal to purpose
        purpose_vec = np.array([1.0] + [0.0] * 31)
        purpose_vec = purpose_vec / np.linalg.norm(purpose_vec)

        blocked_calls = []

        def embed(text: str) -> np.ndarray:
            if text == "my purpose":
                return purpose_vec.copy()
            # Return near-zero similarity vector
            vec = np.array([0.0, 1.0] + [0.0] * 30)
            return vec / np.linalg.norm(vec)

        def on_block(text, fidelity):
            blocked_calls.append((text, fidelity))
            return "BLOCKED"

        @telos_governed(
            purpose="my purpose",
            embed_fn=embed,
            on_block=on_block,
        )
        def my_func(text: str) -> str:
            return text

        result = my_func("completely unrelated")
        assert result == "BLOCKED"
        assert len(blocked_calls) == 1
        assert blocked_calls[0][0] == "completely unrelated"

    def test_raises_value_error_without_on_block(self):
        """Test that ValueError is raised when blocked and no on_block callback."""
        from telos_adapters.generic.decorator import telos_governed

        purpose_vec = np.array([1.0] + [0.0] * 31)
        purpose_vec = purpose_vec / np.linalg.norm(purpose_vec)

        def embed(text: str) -> np.ndarray:
            if text == "my purpose":
                return purpose_vec.copy()
            # Return near-zero similarity
            vec = np.array([0.0, 1.0] + [0.0] * 30)
            return vec / np.linalg.norm(vec)

        @telos_governed(purpose="my purpose", embed_fn=embed)
        def my_func(text: str) -> str:
            return text

        with pytest.raises(ValueError, match="blocked by TELOS governance"):
            my_func("completely unrelated")


class TestKwargsExtraction:
    def test_user_request_kwarg(self):
        from telos_adapters.generic.decorator import telos_governed
        embed = _make_embed_fn()

        @telos_governed(purpose="financial analysis", embed_fn=embed)
        def my_func(user_request: str = "") -> str:
            return f"processed: {user_request}"

        # Call with keyword argument
        result = my_func(user_request="financial analysis")
        assert result == "processed: financial analysis"

    def test_query_kwarg(self):
        from telos_adapters.generic.decorator import telos_governed
        embed = _make_embed_fn()

        @telos_governed(purpose="financial analysis", embed_fn=embed)
        def my_func(query: str = "") -> str:
            return f"processed: {query}"

        result = my_func(query="financial analysis")
        assert result == "processed: financial analysis"


class TestDecisionClassification:
    def _make_controlled_decorator(self, similarity: float):
        """Create a decorator where we control the exact cosine similarity."""
        purpose_vec = np.array([1.0] + [0.0] * 31)
        purpose_vec = purpose_vec / np.linalg.norm(purpose_vec)

        def embed(text: str) -> np.ndarray:
            if text == "purpose":
                return purpose_vec.copy()
            # Create a vector with the desired cosine similarity
            # cos(theta) = similarity, sin(theta) = sqrt(1 - similarity^2)
            s = max(-1.0, min(1.0, similarity))
            orthogonal = np.array([0.0, 1.0] + [0.0] * 30)
            orthogonal = orthogonal / np.linalg.norm(orthogonal)
            vec = s * purpose_vec + np.sqrt(max(0, 1 - s * s)) * orthogonal
            return vec / np.linalg.norm(vec)

        return embed

    def test_execute_above_085(self):
        from telos_adapters.generic.decorator import telos_governed
        embed = self._make_controlled_decorator(0.90)

        @telos_governed(purpose="purpose", embed_fn=embed)
        def func(text: str) -> str:
            return "executed"

        assert func("test input") == "executed"

    def test_clarify_between_070_085(self):
        from telos_adapters.generic.decorator import telos_governed
        embed = self._make_controlled_decorator(0.75)

        @telos_governed(purpose="purpose", threshold=0.70, embed_fn=embed)
        def func(text: str) -> str:
            return "executed_with_clarify"

        # CLARIFY still proceeds
        assert func("test input") == "executed_with_clarify"

    def test_suggest_between_050_070(self):
        from telos_adapters.generic.decorator import telos_governed
        embed = self._make_controlled_decorator(0.55)

        @telos_governed(purpose="purpose", threshold=0.70, embed_fn=embed)
        def func(text: str) -> str:
            return "executed_with_suggest"

        # SUGGEST still proceeds
        assert func("test input") == "executed_with_suggest"

    def test_inert_below_050(self):
        from telos_adapters.generic.decorator import telos_governed
        embed = self._make_controlled_decorator(0.35)

        @telos_governed(purpose="purpose", threshold=0.70, embed_fn=embed)
        def func(text: str) -> str:
            return "should_not_reach"

        with pytest.raises(ValueError, match="blocked by TELOS governance"):
            func("test input")

    def test_escalate_below_050_high_risk(self):
        from telos_adapters.generic.decorator import telos_governed
        embed = self._make_controlled_decorator(0.35)

        blocked_calls = []

        def on_block(text, fidelity):
            blocked_calls.append(("escalate", fidelity))
            return "ESCALATED"

        @telos_governed(
            purpose="purpose",
            threshold=0.70,
            embed_fn=embed,
            high_risk=True,
            on_block=on_block,
        )
        def func(text: str) -> str:
            return "should_not_reach"

        result = func("test input")
        assert result == "ESCALATED"
        assert len(blocked_calls) == 1


class TestGenericInit:
    def test_all_export(self):
        from telos_adapters.generic import __all__
        assert "telos_governed" in __all__
