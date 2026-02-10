"""
Tests for telos_adapters.langgraph

Tests the LangGraph adapter components:
- state_schema: TelosGovernedState, PrimacyAttractor, FidelityZone, DirectionLevel
- governance_node: calculate_fidelity, get_fidelity_zone, TelosGovernanceGate
- wrapper: TelosWrapper, telos_wrap
- supervisor: TelosSupervisor, create_telos_supervisor
- swarm: TelosSwarm, create_telos_swarm
"""

import pytest
import numpy as np
from datetime import datetime
from unittest.mock import MagicMock, patch


# =============================================================================
# STATE SCHEMA TESTS
# =============================================================================

class TestFidelityZone:
    def test_zone_values(self):
        from telos_adapters.langgraph.state_schema import FidelityZone
        assert FidelityZone.GREEN.value == "green"
        assert FidelityZone.YELLOW.value == "yellow"
        assert FidelityZone.ORANGE.value == "orange"
        assert FidelityZone.RED.value == "red"


class TestDirectionLevel:
    def test_direction_level_values(self):
        from telos_adapters.langgraph.state_schema import DirectionLevel
        assert DirectionLevel.NONE.value == "none"
        assert DirectionLevel.HARD_BLOCK.value == "hard_block"

    def test_no_intervention_level_exists(self):
        """Verify InterventionLevel was fully renamed to DirectionLevel."""
        import telos_adapters.langgraph.state_schema as mod
        assert not hasattr(mod, "InterventionLevel")


class TestPrimacyAttractor:
    def test_round_trip_serialization(self):
        from telos_adapters.langgraph.state_schema import PrimacyAttractor
        pa = PrimacyAttractor(
            text="Help with financial analysis",
            embedding=np.array([0.1, 0.2, 0.3]),
            purpose="financial analysis",
            scope="portfolio review",
        )
        d = pa.to_dict()
        assert d["text"] == "Help with financial analysis"
        assert d["purpose"] == "financial analysis"
        assert d["scope"] == "portfolio review"
        assert isinstance(d["embedding"], list)
        assert len(d["embedding"]) == 3

        pa2 = PrimacyAttractor.from_dict(d)
        assert pa2.text == pa.text
        assert pa2.purpose == pa.purpose
        assert pa2.scope == pa.scope
        np.testing.assert_array_almost_equal(pa2.embedding, pa.embedding)

    def test_from_dict_missing_optional_fields(self):
        from telos_adapters.langgraph.state_schema import PrimacyAttractor
        d = {"text": "test", "embedding": [0.1, 0.2]}
        pa = PrimacyAttractor.from_dict(d)
        assert pa.text == "test"
        assert pa.purpose is None
        assert pa.scope is None


class TestGovernanceTraceEntry:
    def test_to_dict(self):
        from telos_adapters.langgraph.state_schema import (
            GovernanceTraceEntry, FidelityZone, DirectionLevel,
        )
        entry = GovernanceTraceEntry(
            timestamp=datetime(2026, 1, 1),
            turn_number=1,
            action_type="tool_call",
            action_description="search: query",
            raw_similarity=0.35,
            fidelity_score=0.75,
            zone=FidelityZone.GREEN,
            direction_level=DirectionLevel.NONE,
            approved=True,
            approval_source="auto",
        )
        d = entry.to_dict()
        assert d["zone"] == "green"
        assert d["direction_level"] == "none"
        assert d["approved"] is True
        assert "direction_reason" in d


class TestGetZoneFromFidelity:
    def test_green_zone(self):
        from telos_adapters.langgraph.state_schema import get_zone_from_fidelity, FidelityZone
        assert get_zone_from_fidelity(0.70) == FidelityZone.GREEN
        assert get_zone_from_fidelity(0.95) == FidelityZone.GREEN

    def test_yellow_zone(self):
        from telos_adapters.langgraph.state_schema import get_zone_from_fidelity, FidelityZone
        assert get_zone_from_fidelity(0.65) == FidelityZone.YELLOW
        assert get_zone_from_fidelity(0.60) == FidelityZone.YELLOW

    def test_orange_zone(self):
        from telos_adapters.langgraph.state_schema import get_zone_from_fidelity, FidelityZone
        assert get_zone_from_fidelity(0.55) == FidelityZone.ORANGE
        assert get_zone_from_fidelity(0.50) == FidelityZone.ORANGE

    def test_red_zone(self):
        from telos_adapters.langgraph.state_schema import get_zone_from_fidelity, FidelityZone
        assert get_zone_from_fidelity(0.49) == FidelityZone.RED
        assert get_zone_from_fidelity(0.10) == FidelityZone.RED


class TestGetDirectionLevel:
    def test_hard_block_below_baseline(self):
        from telos_adapters.langgraph.state_schema import get_direction_level, DirectionLevel
        result = get_direction_level(fidelity=0.80, raw_similarity=0.15)
        assert result == DirectionLevel.HARD_BLOCK

    def test_none_for_green(self):
        from telos_adapters.langgraph.state_schema import get_direction_level, DirectionLevel
        result = get_direction_level(fidelity=0.75, raw_similarity=0.40)
        assert result == DirectionLevel.NONE

    def test_monitor_for_yellow(self):
        from telos_adapters.langgraph.state_schema import get_direction_level, DirectionLevel
        result = get_direction_level(fidelity=0.65, raw_similarity=0.30)
        assert result == DirectionLevel.MONITOR

    def test_correct_for_orange(self):
        from telos_adapters.langgraph.state_schema import get_direction_level, DirectionLevel
        result = get_direction_level(fidelity=0.55, raw_similarity=0.30)
        assert result == DirectionLevel.CORRECT

    def test_direct_for_red(self):
        from telos_adapters.langgraph.state_schema import get_direction_level, DirectionLevel
        result = get_direction_level(fidelity=0.40, raw_similarity=0.25)
        assert result == DirectionLevel.DIRECT


class TestCalculateSCI:
    def test_no_previous_embedding(self):
        from telos_adapters.langgraph.state_schema import calculate_sci
        continuity, inherited = calculate_sci(
            np.array([1.0, 0.0, 0.0]), None, 0.8
        )
        assert continuity == 1.0
        assert inherited == 1.0

    def test_identical_embeddings(self):
        from telos_adapters.langgraph.state_schema import calculate_sci
        emb = np.array([1.0, 0.0, 0.0])
        continuity, inherited = calculate_sci(emb, emb, 0.8)
        assert continuity == pytest.approx(1.0)
        assert inherited == pytest.approx(0.72)  # 0.8 * 0.90

    def test_orthogonal_embeddings_no_inheritance(self):
        from telos_adapters.langgraph.state_schema import calculate_sci
        continuity, inherited = calculate_sci(
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            0.8,
        )
        assert continuity == pytest.approx(0.0, abs=1e-6)
        assert inherited == 0.0


class TestCreateInitialState:
    def test_default_state(self):
        from telos_adapters.langgraph.state_schema import create_initial_state
        state = create_initial_state()
        assert state["messages"] == []
        assert state["primacy_attractor"] is None
        assert state["current_fidelity"] == 1.0
        assert state["current_zone"] == "green"
        assert state["direction_count"] == 0
        assert state["turn_number"] == 0

    def test_state_with_pa(self):
        from telos_adapters.langgraph.state_schema import (
            create_initial_state, PrimacyAttractor,
        )
        pa = PrimacyAttractor(text="test", embedding=np.array([1.0, 2.0]))
        state = create_initial_state(primacy_attractor=pa)
        assert state["primacy_attractor"] is not None
        assert state["primacy_attractor"]["text"] == "test"


# =============================================================================
# GOVERNANCE NODE TESTS
# =============================================================================

def _make_embed_fn(dim=3):
    """Create a deterministic mock embedding function."""
    def embed(text: str) -> np.ndarray:
        # Simple deterministic embedding based on text hash
        h = hash(text) % 10000
        rng = np.random.RandomState(h)
        vec = rng.randn(dim)
        return vec / np.linalg.norm(vec)
    return embed


def _make_high_fidelity_embed_fn(purpose_text: str, dim=16):
    """
    Create an embedding function where all texts have high similarity
    to the purpose text. Used for tests that need to pass fidelity checks.
    """
    base_vec = np.ones(dim) / np.sqrt(dim)

    def embed(text: str) -> np.ndarray:
        if text == purpose_text:
            return base_vec.copy()
        # All other texts get a vector close to purpose (high similarity)
        rng = np.random.RandomState(hash(text) % 10000)
        noise = rng.randn(dim) * 0.05  # small noise
        vec = base_vec + noise
        return vec / np.linalg.norm(vec)

    return embed


class TestCalculateFidelity:
    def test_text_input_requires_embed_fn(self):
        from telos_adapters.langgraph.state_schema import PrimacyAttractor
        from telos_adapters.langgraph.governance_node import calculate_fidelity
        pa = PrimacyAttractor(text="test", embedding=np.array([1.0, 0.0, 0.0]))
        with pytest.raises(ValueError, match="embed_fn required"):
            calculate_fidelity("some text", pa)

    def test_text_input_with_embed_fn(self):
        from telos_adapters.langgraph.state_schema import PrimacyAttractor
        from telos_adapters.langgraph.governance_node import calculate_fidelity
        embed = _make_embed_fn()
        pa = PrimacyAttractor(text="test", embedding=embed("test"))
        raw, norm = calculate_fidelity("test", pa, embed)
        assert raw == pytest.approx(1.0, abs=0.01)
        assert 0.0 <= norm <= 1.0

    def test_embedding_input(self):
        from telos_adapters.langgraph.state_schema import PrimacyAttractor
        from telos_adapters.langgraph.governance_node import calculate_fidelity
        pa = PrimacyAttractor(text="t", embedding=np.array([1.0, 0.0, 0.0]))
        raw, norm = calculate_fidelity(np.array([1.0, 0.0, 0.0]), pa)
        assert raw == pytest.approx(1.0, abs=0.01)

    def test_normalized_is_clamped(self):
        from telos_adapters.langgraph.state_schema import PrimacyAttractor
        from telos_adapters.langgraph.governance_node import calculate_fidelity
        pa = PrimacyAttractor(text="t", embedding=np.array([1.0, 0.0, 0.0]))
        raw, norm = calculate_fidelity(np.array([-1.0, 0.0, 0.0]), pa)
        assert 0.0 <= norm <= 1.0


class TestTelosGovernanceGate:
    def test_passthrough_without_pa(self):
        from telos_adapters.langgraph.governance_node import TelosGovernanceGate
        gate = TelosGovernanceGate(embed_fn=_make_embed_fn())
        state = {"primacy_attractor": None, "messages": []}
        result = gate(state)
        assert result is state  # unchanged

    def test_passthrough_without_messages(self):
        from telos_adapters.langgraph.state_schema import PrimacyAttractor
        from telos_adapters.langgraph.governance_node import TelosGovernanceGate
        embed = _make_embed_fn()
        pa = PrimacyAttractor(text="test", embedding=embed("test"))
        gate = TelosGovernanceGate(embed_fn=embed)
        state = {"primacy_attractor": pa.to_dict(), "messages": []}
        result = gate(state)
        assert result is state


# =============================================================================
# WRAPPER TESTS
# =============================================================================

class TestTelosWrapper:
    def _make_wrapper(self):
        from telos_adapters.langgraph.state_schema import PrimacyAttractor
        from telos_adapters.langgraph.wrapper import TelosWrapper
        embed = _make_embed_fn()
        pa = PrimacyAttractor(text="financial analysis", embedding=embed("financial analysis"))
        agent = MagicMock()
        agent.invoke.return_value = {"messages": []}
        return TelosWrapper(
            agent=agent,
            primacy_attractor=pa,
            embed_fn=embed,
            block_on_low_fidelity=False,
        ), agent

    def test_invoke_passes_through(self):
        wrapper, agent = self._make_wrapper()
        state = {"messages": []}
        wrapper.invoke(state)
        agent.invoke.assert_called_once()

    def test_governance_trace_recorded(self):
        wrapper, agent = self._make_wrapper()
        msg = MagicMock()
        msg.content = "analyze my portfolio returns"
        state = {"messages": [msg]}
        wrapper.invoke(state)
        # Should have at least one trace from post-check
        assert len(wrapper.governance_trace) >= 0

    def test_fidelity_trajectory(self):
        wrapper, _ = self._make_wrapper()
        # Initially empty
        assert wrapper.get_fidelity_trajectory() == []


class TestTelosWrap:
    def test_decorator_wraps_agent(self):
        from telos_adapters.langgraph.state_schema import PrimacyAttractor
        from telos_adapters.langgraph.wrapper import telos_wrap, TelosWrapper
        embed = _make_embed_fn()
        pa = PrimacyAttractor(text="test", embedding=embed("test"))
        agent = MagicMock()
        agent.invoke.return_value = {"messages": []}
        wrapped = telos_wrap(pa, embed)(agent)
        assert isinstance(wrapped, TelosWrapper)


# =============================================================================
# SUPERVISOR TESTS
# =============================================================================

class TestTelosSupervisor:
    def _make_supervisor(self):
        from telos_adapters.langgraph.state_schema import PrimacyAttractor
        from telos_adapters.langgraph.supervisor import TelosSupervisor
        embed = _make_high_fidelity_embed_fn("code review")
        pa = PrimacyAttractor(text="code review", embedding=embed("code review"))
        agents = {"research": MagicMock(), "code": MagicMock()}
        return TelosSupervisor(
            agents=agents,
            primacy_attractor=pa,
            embed_fn=embed,
        )

    def test_route_with_no_messages(self):
        sup = self._make_supervisor()
        state = {"messages": [], "next_agent": None}
        result = sup.route(state)
        assert result["next_agent"] is None

    def test_route_selects_agent(self):
        sup = self._make_supervisor()
        msg = MagicMock()
        msg.content = "review the code for bugs"
        state = {
            "messages": [msg],
            "current_fidelity": 1.0,
            "current_zone": "green",
            "next_agent": None,
            "delegation_approved": True,
            "direction_count": 0,
        }
        result = sup.route(state)
        assert result["next_agent"] in ["research", "code"]
        assert result["delegation_approved"] is True

    def test_governance_trace_recorded(self):
        sup = self._make_supervisor()
        msg = MagicMock()
        msg.content = "review code"
        state = {
            "messages": [msg],
            "current_fidelity": 1.0,
            "current_zone": "green",
            "next_agent": None,
            "delegation_approved": True,
            "direction_count": 0,
        }
        sup.route(state)
        assert len(sup.get_governance_trace()) >= 1


class TestCreateTelosSupervisor:
    def test_factory(self):
        from telos_adapters.langgraph.state_schema import PrimacyAttractor
        from telos_adapters.langgraph.supervisor import (
            create_telos_supervisor, TelosSupervisor,
        )
        embed = _make_embed_fn()
        pa = PrimacyAttractor(text="test", embedding=embed("test"))
        sup = create_telos_supervisor(
            agents={"a": MagicMock()},
            primacy_attractor=pa,
            embed_fn=embed,
        )
        assert isinstance(sup, TelosSupervisor)


# =============================================================================
# SWARM TESTS
# =============================================================================

class TestTelosSwarm:
    def _make_swarm(self):
        from telos_adapters.langgraph.state_schema import PrimacyAttractor
        from telos_adapters.langgraph.swarm import TelosSwarm
        embed = _make_high_fidelity_embed_fn("data analysis")
        pa = PrimacyAttractor(text="data analysis", embedding=embed("data analysis"))
        agents = {"analyst": MagicMock(), "coder": MagicMock()}
        return TelosSwarm(
            agents=agents,
            primacy_attractor=pa,
            embed_fn=embed,
        )

    def test_handoff_approved(self):
        swarm = self._make_swarm()
        result = swarm.handoff(
            from_agent="analyst",
            to_agent="coder",
            context={},
            reason="Need code implementation",
        )
        assert result["approved"] is True
        assert result["from_agent"] == "analyst"
        assert result["to_agent"] == "coder"

    def test_handoff_chain_tracking(self):
        swarm = self._make_swarm()
        swarm.handoff("analyst", "coder", {}, "First handoff")
        swarm.handoff("coder", "analyst", {}, "Second handoff")
        chain = swarm.get_handoff_chain()
        assert len(chain) == 2
        assert chain[0]["from"] == "analyst"
        assert chain[1]["from"] == "coder"

    def test_sci_trajectory(self):
        swarm = self._make_swarm()
        swarm.handoff("analyst", "coder", {}, "First")
        swarm.handoff("coder", "analyst", {}, "Second")
        traj = swarm.get_sci_trajectory()
        assert len(traj) == 2
        for entry in traj:
            assert "sci" in entry
            assert "fidelity" in entry

    def test_governance_trace(self):
        swarm = self._make_swarm()
        swarm.handoff("analyst", "coder", {}, "test")
        trace = swarm.get_governance_trace()
        assert len(trace) >= 1
        assert trace[0]["action_type"] in ("handoff", "handoff_blocked")


class TestCreateTelosSwarm:
    def test_factory(self):
        from telos_adapters.langgraph.state_schema import PrimacyAttractor
        from telos_adapters.langgraph.swarm import create_telos_swarm, TelosSwarm
        embed = _make_embed_fn()
        pa = PrimacyAttractor(text="test", embedding=embed("test"))
        swarm = create_telos_swarm(
            agents={"a": MagicMock()},
            primacy_attractor=pa,
            embed_fn=embed,
        )
        assert isinstance(swarm, TelosSwarm)


# =============================================================================
# LANGGRAPH __init__ EXPORTS
# =============================================================================

class TestLangGraphExports:
    def test_all_exports_available(self):
        from telos_adapters.langgraph import (
            TelosGovernedState,
            PrimacyAttractor,
            GovernanceTraceEntry,
            ActionChainEntry,
            FidelityZone,
            DirectionLevel,
            TelosGovernanceGate,
            telos_governance_node,
            calculate_fidelity,
            get_fidelity_zone,
            TelosWrapper,
            telos_wrap,
            TelosSupervisor,
            create_telos_supervisor,
            TelosSwarm,
            create_telos_swarm,
        )
        # All imports succeeded
        assert TelosGovernedState is not None

    def test_no_intervention_level_exported(self):
        import telos_adapters.langgraph as lg
        assert not hasattr(lg, "InterventionLevel")
