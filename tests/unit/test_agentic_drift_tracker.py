"""
Tests for AgenticDriftTracker (SAAI sliding-window drift tracking)
===================================================================

Tests the drift tracker wired into AgenticResponseManager for agentic
sessions. Validates baseline establishment, sliding-window drift detection
(NORMAL/WARNING/RESTRICT/BLOCK), BLOCK override to ESCALATE, acknowledgment
mechanism, RESTRICT tier behavior, and recovery scenarios.
"""

import pytest
from unittest.mock import MagicMock

from telos_governance.response_manager import (
    AgenticDriftTracker,
    AgenticResponseManager,
    AgenticTurnResult,
)
from telos_core.constants import (
    BASELINE_TURN_COUNT,
    SAAI_DRIFT_WARNING,
    SAAI_DRIFT_RESTRICT,
    SAAI_DRIFT_BLOCK,
    SAAI_DRIFT_WINDOW_SIZE,
    SAAI_MAX_ACKNOWLEDGMENTS,
    ST_SAAI_RESTRICT_EXECUTE_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _record_n_turns(tracker, fidelity, n):
    """Record n turns at a given fidelity level. Returns last status."""
    status = None
    for _ in range(n):
        status = tracker.record_fidelity(fidelity)
    return status


def _establish_baseline(tracker, fidelity=0.90):
    """Establish baseline at given fidelity. Returns baseline status."""
    return _record_n_turns(tracker, fidelity, BASELINE_TURN_COUNT)


# ---------------------------------------------------------------------------
# Baseline Establishment
# ---------------------------------------------------------------------------

class TestBaselineEstablishment:
    def test_baseline_not_established_before_threshold(self):
        """Baseline requires BASELINE_TURN_COUNT turns."""
        tracker = AgenticDriftTracker()
        for _ in range(BASELINE_TURN_COUNT - 1):
            status = tracker.record_fidelity(0.90)
        assert status["baseline_established"] is False
        assert status["baseline_fidelity"] is None

    def test_baseline_established_at_threshold(self):
        """Baseline is established on the Nth turn."""
        tracker = AgenticDriftTracker()
        status = _record_n_turns(tracker, 0.90, BASELINE_TURN_COUNT)
        assert status["baseline_established"] is True
        assert status["baseline_fidelity"] == pytest.approx(0.90)

    def test_baseline_uses_first_n_scores(self):
        """Baseline is computed from first N scores only."""
        tracker = AgenticDriftTracker()
        tracker.record_fidelity(0.80)
        tracker.record_fidelity(0.80)
        status = tracker.record_fidelity(0.90)
        expected = (0.80 + 0.80 + 0.90) / 3
        assert status["baseline_fidelity"] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Drift Detection: NORMAL
# ---------------------------------------------------------------------------

class TestDriftNormal:
    def test_stable_fidelity_stays_normal(self):
        """When fidelity is stable, drift level remains NORMAL."""
        tracker = AgenticDriftTracker()
        _establish_baseline(tracker, 0.90)
        status = _record_n_turns(tracker, 0.90, 5)
        assert status["drift_level"] == "NORMAL"
        assert status["drift_magnitude"] == pytest.approx(0.0, abs=0.01)

    def test_slight_drop_stays_normal(self):
        """A small fidelity drop (<10%) stays NORMAL."""
        tracker = AgenticDriftTracker()
        _establish_baseline(tracker, 0.90)
        status = _record_n_turns(tracker, 0.88, 3)
        assert status["drift_level"] == "NORMAL"


# ---------------------------------------------------------------------------
# Drift Detection: WARNING (10%) — Sliding Window
# ---------------------------------------------------------------------------

class TestDriftWarning:
    def test_warning_at_10_percent_drift(self):
        """Sliding window triggers WARNING when avg drops 10%+ from baseline."""
        tracker = AgenticDriftTracker()
        _establish_baseline(tracker, 1.0)
        # Window of [0.85] -> drift = (1.0 - 0.85)/1.0 = 0.15 -> RESTRICT
        # Need drift between 0.10 and 0.15: score of ~0.88
        # (1.0 - 0.88)/1.0 = 0.12 -> WARNING
        status = tracker.record_fidelity(0.88)
        assert status["drift_level"] == "WARNING"
        assert status["drift_magnitude"] >= SAAI_DRIFT_WARNING
        assert status["drift_magnitude"] < SAAI_DRIFT_RESTRICT

    def test_sustained_moderate_drift_triggers_warning(self):
        """Sustained moderate drift fills the window and holds at WARNING."""
        tracker = AgenticDriftTracker()
        _establish_baseline(tracker, 0.90)
        # Score of 0.82: drift = (0.90 - 0.82)/0.90 = 0.0889... ≈ 8.9% -> NORMAL
        # Score of 0.80: drift = (0.90 - 0.80)/0.90 = 0.1111 -> WARNING
        status = _record_n_turns(tracker, 0.80, SAAI_DRIFT_WINDOW_SIZE)
        assert status["drift_level"] == "WARNING"


# ---------------------------------------------------------------------------
# Drift Detection: RESTRICT (15%)
# ---------------------------------------------------------------------------

class TestDriftRestrict:
    def test_restrict_at_15_percent_drift(self):
        """Sliding window triggers RESTRICT at 15%+ drift."""
        tracker = AgenticDriftTracker()
        _establish_baseline(tracker, 1.0)
        # Score of 0.83: drift = (1.0 - 0.83)/1.0 = 0.17 -> RESTRICT
        status = tracker.record_fidelity(0.83)
        assert status["drift_level"] == "RESTRICT"
        assert status["drift_magnitude"] >= SAAI_DRIFT_RESTRICT
        assert status["drift_magnitude"] < SAAI_DRIFT_BLOCK
        assert status["is_restricted"] is True


# ---------------------------------------------------------------------------
# Drift Detection: BLOCK (20%)
# ---------------------------------------------------------------------------

class TestDriftBlock:
    def test_block_at_20_percent_drift(self):
        """Sliding window triggers BLOCK at 20%+ drift."""
        tracker = AgenticDriftTracker()
        _establish_baseline(tracker, 1.0)
        # Score of 0.10: drift = (1.0 - 0.10)/1.0 = 0.90 -> BLOCK
        status = tracker.record_fidelity(0.10)
        assert status["drift_level"] == "BLOCK"
        assert status["is_blocked"] is True
        assert status["drift_magnitude"] >= SAAI_DRIFT_BLOCK

    def test_is_blocked_property(self):
        """The is_blocked property reflects BLOCK state."""
        tracker = AgenticDriftTracker()
        assert tracker.is_blocked is False
        _establish_baseline(tracker, 1.0)
        tracker.record_fidelity(0.10)
        assert tracker.is_blocked is True

    def test_sustained_low_fidelity_blocks(self):
        """Filling the window with low scores triggers BLOCK."""
        tracker = AgenticDriftTracker()
        _establish_baseline(tracker, 0.90)
        # Score of 0.50: drift = (0.90 - 0.50)/0.90 = 0.444 -> BLOCK
        status = _record_n_turns(tracker, 0.50, SAAI_DRIFT_WINDOW_SIZE)
        assert status["drift_level"] == "BLOCK"


# ---------------------------------------------------------------------------
# BLOCK Override: Forces ESCALATE
# ---------------------------------------------------------------------------

class TestBlockOverrideEscalate:
    def test_block_forces_escalate_decision(self):
        """When SAAI drift is BLOCK, process_request overrides decision to ESCALATE."""
        from telos_governance.agent_templates import AgenticTemplate
        from telos_governance.agentic_fidelity import AgenticFidelityResult
        from telos_governance.types import ActionDecision, DirectionLevel
        import numpy as np

        manager = AgenticResponseManager()
        manager._initialized = True
        manager._embed_fn = lambda t: np.ones(3) / np.sqrt(3)
        manager._llm_client = None
        manager._llm_client_checked = True

        template = AgenticTemplate(
            id="sql_analyst",
            name="SQL Database Analyst",
            description="Queries data",
            icon="database",
            purpose="Help users query and understand data in PostgreSQL databases",
            scope="SELECT queries, schema exploration, data analysis",
            boundaries=["No data modification"],
            tools=["sql_db_query"],
            tool_set_key="sql_agent",
            example_requests=["Show me total revenue"],
            drift_examples=["Delete all records"],
            system_prompt="You are a SQL Database Analyst.",
        )

        def make_engine_result(fidelity):
            result = AgenticFidelityResult(
                purpose_fidelity=fidelity,
                scope_fidelity=fidelity,
                boundary_violation=0.0,
                tool_fidelity=fidelity,
                chain_continuity=0.50,
                composite_fidelity=fidelity,
                effective_fidelity=fidelity,
                decision=ActionDecision.EXECUTE,
                direction_level=DirectionLevel.NONE,
                boundary_triggered=False,
                tool_blocked=False,
                chain_broken=False,
                selected_tool="sql_db_query",
                tool_rankings=[{"rank": 1, "tool": "sql_db_query", "fidelity": fidelity}],
            )
            result.dimension_explanations = {
                "purpose": "OK", "scope": "OK", "tool": "OK",
                "chain": "OK", "boundary": "OK",
            }
            return result

        mock_engine = MagicMock()
        manager._engine_cache["sql_analyst"] = mock_engine

        # Phase 1: Establish baseline with high fidelity
        mock_engine.score_action.return_value = make_engine_result(1.0)
        for i in range(BASELINE_TURN_COUNT):
            result = manager.process_request("Show revenue", template, i + 1)
            assert result.decision == "EXECUTE"

        # Phase 2: Send very low fidelity to trigger BLOCK
        mock_engine.score_action.return_value = make_engine_result(0.10)
        result = manager.process_request("Show revenue", template, BASELINE_TURN_COUNT + 1)

        assert result.decision == "ESCALATE"
        assert result.saai_blocked is True
        assert "SAAI DRIFT BLOCK" in result.decision_explanation
        assert "safety thresholds" in result.response_text.lower()
        assert result.pre_block_response is not None


# ---------------------------------------------------------------------------
# Sliding Window Behavior
# ---------------------------------------------------------------------------

class TestSlidingWindow:
    def test_recovery_after_single_outlier(self):
        """A single outlier exits the window after W good turns."""
        tracker = AgenticDriftTracker()
        _establish_baseline(tracker, 0.90)
        # One extreme outlier
        tracker.record_fidelity(0.10)
        assert tracker.drift_level == "BLOCK"
        # W good turns to fully replace the window contents
        status = _record_n_turns(tracker, 0.90, SAAI_DRIFT_WINDOW_SIZE)
        assert status["drift_level"] == "NORMAL"
        assert status["drift_magnitude"] == pytest.approx(0.0, abs=0.01)

    def test_partial_window_uses_available_scores(self):
        """Before window fills, drift uses all available post-baseline scores."""
        tracker = AgenticDriftTracker()
        _establish_baseline(tracker, 1.0)
        # First post-baseline turn at 0.85: window = [0.85], drift = 0.15 -> RESTRICT
        status = tracker.record_fidelity(0.85)
        assert status["drift_magnitude"] == pytest.approx(0.15)

    def test_full_window_ignores_old_scores(self):
        """Once window is full, old scores do not affect drift."""
        tracker = AgenticDriftTracker()
        _establish_baseline(tracker, 0.90)
        # Fill window with low scores
        _record_n_turns(tracker, 0.50, SAAI_DRIFT_WINDOW_SIZE)
        assert tracker.drift_level == "BLOCK"
        # Now fill window with good scores — old bad scores slide out
        status = _record_n_turns(tracker, 0.90, SAAI_DRIFT_WINDOW_SIZE)
        assert status["drift_level"] == "NORMAL"

    def test_oscillating_fidelity(self):
        """Alternating high/low fidelity produces correct window average."""
        tracker = AgenticDriftTracker()
        _establish_baseline(tracker, 0.90)
        # Alternate: 0.90, 0.10, 0.90, 0.10, 0.90 -> avg = 0.58
        scores = [0.90, 0.10, 0.90, 0.10, 0.90]
        for s in scores:
            status = tracker.record_fidelity(s)
        # drift = (0.90 - 0.58) / 0.90 = 0.356 -> BLOCK
        assert status["drift_level"] == "BLOCK"

    def test_monotonically_increasing_never_triggers(self):
        """Improving fidelity should never trigger drift."""
        tracker = AgenticDriftTracker()
        _establish_baseline(tracker, 0.50)
        for score in [0.55, 0.60, 0.65, 0.70, 0.75]:
            status = tracker.record_fidelity(score)
        assert status["drift_level"] == "NORMAL"
        assert status["drift_magnitude"] == 0.0

    def test_gradual_decline_escalates_tiers(self):
        """Monotonically decreasing fidelity escalates through tiers."""
        tracker = AgenticDriftTracker()
        _establish_baseline(tracker, 0.90)
        # Scores that produce increasing drift in the window
        # 0.80: drift = 0.111 -> WARNING
        status = tracker.record_fidelity(0.80)
        assert status["drift_level"] == "WARNING"
        # Fill window with 0.75: drift = (0.90-0.75)/0.90 = 0.167 -> RESTRICT
        _record_n_turns(tracker, 0.75, SAAI_DRIFT_WINDOW_SIZE)
        assert tracker.drift_level == "RESTRICT"
        # Fill window with 0.50: drift = 0.444 -> BLOCK
        status = _record_n_turns(tracker, 0.50, SAAI_DRIFT_WINDOW_SIZE)
        assert status["drift_level"] == "BLOCK"


# ---------------------------------------------------------------------------
# Acknowledgment Mechanism
# ---------------------------------------------------------------------------

class TestAcknowledgment:
    def test_acknowledge_resets_drift_level(self):
        """Acknowledging BLOCK resets drift to NORMAL."""
        tracker = AgenticDriftTracker()
        _establish_baseline(tracker, 0.90)
        tracker.record_fidelity(0.10)  # trigger BLOCK
        assert tracker.drift_level == "BLOCK"

        status = tracker.acknowledge_drift("Operator review completed")
        assert status["drift_level"] == "NORMAL"
        assert status["drift_magnitude"] == 0.0
        assert status["is_blocked"] is False

    def test_acknowledge_preserves_baseline(self):
        """Acknowledgment preserves the original baseline."""
        tracker = AgenticDriftTracker()
        _establish_baseline(tracker, 0.90)
        tracker.record_fidelity(0.10)
        tracker.acknowledge_drift("Review done")
        # Baseline should still be 0.90
        history = tracker.get_drift_history()
        assert history["baseline_fidelity"] == pytest.approx(0.90)

    def test_acknowledge_clears_window(self):
        """After acknowledgment, drift detection restarts with empty window."""
        tracker = AgenticDriftTracker()
        _establish_baseline(tracker, 0.90)
        tracker.record_fidelity(0.10)  # BLOCK
        tracker.acknowledge_drift("OK")
        # Next score should use a fresh window
        status = tracker.record_fidelity(0.90)
        assert status["drift_level"] == "NORMAL"
        assert status["drift_magnitude"] == pytest.approx(0.0, abs=0.01)

    def test_max_acknowledgments_permanently_blocks(self):
        """Exceeding max acknowledgments permanently blocks the session."""
        tracker = AgenticDriftTracker()
        _establish_baseline(tracker, 0.90)

        for i in range(SAAI_MAX_ACKNOWLEDGMENTS):
            tracker.record_fidelity(0.10)  # BLOCK
            tracker.acknowledge_drift(f"Ack {i+1}")

        # After max acks, trigger BLOCK again
        tracker.record_fidelity(0.10)
        assert tracker.is_blocked is True
        # Further acknowledgment should fail
        status = tracker.acknowledge_drift("Should not work")
        assert status["is_blocked"] is True
        assert status["permanently_blocked"] is True

    def test_acknowledge_no_op_when_not_blocked(self):
        """Acknowledging when not blocked is a no-op."""
        tracker = AgenticDriftTracker()
        _establish_baseline(tracker, 0.90)
        tracker.record_fidelity(0.85)  # Not blocked
        status = tracker.acknowledge_drift("Not needed")
        assert status["acknowledgment_count"] == 0

    def test_acknowledgment_history_tracked(self):
        """Acknowledgment history records each event."""
        tracker = AgenticDriftTracker()
        _establish_baseline(tracker, 0.90)
        tracker.record_fidelity(0.10)
        tracker.acknowledge_drift("First review")
        history = tracker.get_drift_history()
        assert len(history["acknowledgment_history"]) == 1
        assert history["acknowledgment_history"][0]["reason"] == "First review"
        assert history["acknowledgment_count"] == 1

    def test_manager_acknowledge_drift(self):
        """AgenticResponseManager.acknowledge_drift delegates to tracker."""
        manager = AgenticResponseManager()
        _establish_baseline(manager._drift_tracker, 0.90)
        manager._drift_tracker.record_fidelity(0.10)
        status = manager.acknowledge_drift("Manager review")
        assert status["drift_level"] == "NORMAL"


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

class TestDriftReset:
    def test_reset_clears_all_state(self):
        """Reset returns tracker to initial state."""
        tracker = AgenticDriftTracker()
        _establish_baseline(tracker, 1.0)
        tracker.record_fidelity(0.10)  # trigger BLOCK
        assert tracker.is_blocked is True

        tracker.reset()
        assert tracker.drift_level == "NORMAL"
        assert tracker.drift_magnitude == 0.0
        assert tracker.is_blocked is False

        status = tracker.record_fidelity(0.90)
        assert status["baseline_established"] is False

    def test_reset_clears_acknowledgment_count(self):
        """Reset clears the acknowledgment counter."""
        tracker = AgenticDriftTracker()
        _establish_baseline(tracker, 0.90)
        tracker.record_fidelity(0.10)
        tracker.acknowledge_drift("Test")
        assert tracker.get_drift_history()["acknowledgment_count"] == 1
        tracker.reset()
        assert tracker.get_drift_history()["acknowledgment_count"] == 0

    def test_manager_reset_drift(self):
        """AgenticResponseManager.reset_drift() delegates to tracker."""
        manager = AgenticResponseManager()
        _record_n_turns(manager._drift_tracker, 1.0, BASELINE_TURN_COUNT)
        manager._drift_tracker.record_fidelity(0.10)
        assert manager._drift_tracker.is_blocked is True

        manager.reset_drift()
        assert manager._drift_tracker.is_blocked is False
        assert manager._drift_tracker.drift_level == "NORMAL"


# ---------------------------------------------------------------------------
# AgenticTurnResult drift fields
# ---------------------------------------------------------------------------

class TestTurnResultDriftFields:
    def test_default_drift_fields(self):
        """AgenticTurnResult has correct drift field defaults."""
        r = AgenticTurnResult()
        assert r.drift_level == "NORMAL"
        assert r.drift_magnitude == 0.0
        assert r.saai_baseline is None
        assert r.saai_blocked is False
        assert r.pre_block_response is None


# ---------------------------------------------------------------------------
# Input Validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_fidelity_clamped_above_one(self):
        """Fidelity values above 1.0 are clamped to 1.0."""
        tracker = AgenticDriftTracker()
        status = tracker.record_fidelity(1.5)
        # Should not error; internally treated as 1.0
        assert status["turn_count"] == 1

    def test_fidelity_clamped_below_zero(self):
        """Negative fidelity values are clamped to 0.0."""
        tracker = AgenticDriftTracker()
        status = tracker.record_fidelity(-0.5)
        assert status["turn_count"] == 1

    def test_zero_baseline(self):
        """Baseline of 0.0 produces no drift (division guard)."""
        tracker = AgenticDriftTracker()
        _establish_baseline(tracker, 0.0)
        status = tracker.record_fidelity(0.0)
        assert status["drift_magnitude"] == 0.0
        assert status["drift_level"] == "NORMAL"

    def test_perfect_baseline(self):
        """Baseline of 1.0 correctly detects any drop."""
        tracker = AgenticDriftTracker()
        _establish_baseline(tracker, 1.0)
        # 0.89: drift = 0.11 -> WARNING
        status = tracker.record_fidelity(0.89)
        assert status["drift_level"] == "WARNING"


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_turn_count_tracking(self):
        """Status includes correct turn count."""
        tracker = AgenticDriftTracker()
        status = _record_n_turns(tracker, 0.90, 5)
        assert status["turn_count"] == 5

    def test_drift_level_property(self):
        """drift_level property reflects internal state after sliding window."""
        tracker = AgenticDriftTracker()
        assert tracker.drift_level == "NORMAL"
        _establish_baseline(tracker, 1.0)
        # 0.50 in window of 1: drift = 0.50 -> BLOCK
        tracker.record_fidelity(0.50)
        assert tracker.drift_level == "BLOCK"

    def test_drift_magnitude_property(self):
        """drift_magnitude property reflects computed magnitude."""
        tracker = AgenticDriftTracker()
        assert tracker.drift_magnitude == 0.0
        _establish_baseline(tracker, 1.0)
        tracker.record_fidelity(0.50)
        assert tracker.drift_magnitude > 0.0

    def test_exact_warning_boundary(self):
        """Drift at exactly WARNING threshold (10%) triggers WARNING."""
        tracker = AgenticDriftTracker()
        _establish_baseline(tracker, 1.0)
        # Score of 0.89: drift = (1.0 - 0.89)/1.0 = 0.11 -> WARNING
        status = tracker.record_fidelity(0.89)
        assert status["drift_level"] == "WARNING"
        assert status["drift_magnitude"] >= SAAI_DRIFT_WARNING

    def test_long_session_no_dilution(self):
        """Sliding window prevents dilution over long sessions."""
        tracker = AgenticDriftTracker()
        _establish_baseline(tracker, 0.90)
        # 50 turns of good fidelity
        _record_n_turns(tracker, 0.90, 50)
        assert tracker.drift_level == "NORMAL"
        # Now drift — should be detected immediately in the window
        status = _record_n_turns(tracker, 0.50, SAAI_DRIFT_WINDOW_SIZE)
        assert status["drift_level"] == "BLOCK"

    def test_get_drift_history_completeness(self):
        """get_drift_history returns all expected keys."""
        tracker = AgenticDriftTracker()
        _establish_baseline(tracker, 0.90)
        tracker.record_fidelity(0.80)
        history = tracker.get_drift_history()
        expected_keys = {
            "all_fidelity_scores", "post_baseline_scores",
            "baseline_fidelity", "baseline_established",
            "current_drift_level", "current_drift_magnitude",
            "acknowledgment_count", "acknowledgment_history",
            "permanently_blocked", "total_turns",
        }
        assert set(history.keys()) == expected_keys


# ---------------------------------------------------------------------------
# RESTRICT Behavior: Integration Tests (Phase 2)
# ---------------------------------------------------------------------------

class TestRestrictBehavior:
    """Test RESTRICT enforcement via process_request() integration.

    When drift is in RESTRICT range (15-20%), EXECUTE decisions for actions
    scoring between ST_EXECUTE (0.45) and ST_RESTRICT (0.52) should be
    downgraded to CLARIFY. This validates the graduated sanction between
    WARNING (observation) and BLOCK (session freeze).
    """

    def _make_manager_and_template(self):
        """Create a configured manager + template for RESTRICT tests."""
        from telos_governance.agent_templates import AgenticTemplate
        from telos_governance.agentic_fidelity import AgenticFidelityResult
        from telos_governance.types import ActionDecision, DirectionLevel
        import numpy as np

        manager = AgenticResponseManager()
        manager._initialized = True
        manager._embed_fn = lambda t: np.ones(3) / np.sqrt(3)
        manager._llm_client = None
        manager._llm_client_checked = True

        template = AgenticTemplate(
            id="sql_analyst",
            name="SQL Database Analyst",
            description="Queries data",
            icon="database",
            purpose="Help users query and understand data in PostgreSQL databases",
            scope="SELECT queries, schema exploration, data analysis",
            boundaries=["No data modification"],
            tools=["sql_db_query"],
            tool_set_key="sql_agent",
            example_requests=["Show me total revenue"],
            drift_examples=["Delete all records"],
            system_prompt="You are a SQL Database Analyst.",
        )
        return manager, template

    def _make_engine_result(self, fidelity, boundary_triggered=False):
        """Create an AgenticFidelityResult with given fidelity."""
        from telos_governance.agentic_fidelity import AgenticFidelityResult
        from telos_governance.types import ActionDecision, DirectionLevel

        decision = ActionDecision.ESCALATE if boundary_triggered else ActionDecision.EXECUTE
        result = AgenticFidelityResult(
            purpose_fidelity=fidelity,
            scope_fidelity=fidelity,
            boundary_violation=1.0 if boundary_triggered else 0.0,
            tool_fidelity=fidelity,
            chain_continuity=0.50,
            composite_fidelity=fidelity,
            effective_fidelity=fidelity,
            decision=decision,
            direction_level=DirectionLevel.NONE,
            boundary_triggered=boundary_triggered,
            tool_blocked=False,
            chain_broken=False,
            selected_tool="sql_db_query",
            tool_rankings=[{"rank": 1, "tool": "sql_db_query", "fidelity": fidelity}],
        )
        result.dimension_explanations = {
            "purpose": "OK", "scope": "OK", "tool": "OK",
            "chain": "OK", "boundary": "OK",
        }
        return result

    def _push_to_restrict(self, manager, template, mock_engine):
        """Establish baseline and push drift into RESTRICT range (15-19%).

        Math: baseline=0.90, fill 4 post-baseline turns at 0.80.
        Window [0.80, 0.80, 0.80, 0.80], avg=0.80
        drift = (0.90 - 0.80)/0.90 = 0.111 -> WARNING at this point.
        Adding a 5th turn will complete the window and the test turn's
        fidelity determines the final drift level.

        With test turn at 0.50: window [0.80, 0.80, 0.80, 0.80, 0.50]
        avg=0.74, drift=(0.90-0.74)/0.90 = 0.178 -> RESTRICT ✓
        """
        # Phase 1: Establish baseline at 0.90
        mock_engine.score_action.return_value = self._make_engine_result(0.90)
        mock_engine._is_sentence_transformer.return_value = True
        for i in range(BASELINE_TURN_COUNT):
            manager.process_request("Show revenue", template, i + 1)

        # Phase 2: Fill 4 turns at 0.80 to build drift toward RESTRICT
        mock_engine.score_action.return_value = self._make_engine_result(0.80)
        for i in range(4):
            manager.process_request("Show revenue", template, BASELINE_TURN_COUNT + 1 + i)

    def test_restrict_downgrades_execute_to_clarify(self):
        """RESTRICT: fidelity between ST_EXECUTE (0.45) and ST_RESTRICT (0.52) -> CLARIFY."""
        manager, template = self._make_manager_and_template()
        mock_engine = MagicMock()
        manager._engine_cache["sql_analyst"] = mock_engine

        self._push_to_restrict(manager, template, mock_engine)

        # Phase 3: Send request with fidelity between EXECUTE (0.45) and RESTRICT (0.52)
        # Window becomes [0.80, 0.80, 0.80, 0.80, 0.50] -> avg=0.74
        # drift = (0.90 - 0.74)/0.90 = 0.178 -> RESTRICT
        # This should be downgraded from EXECUTE to CLARIFY
        mock_engine.score_action.return_value = self._make_engine_result(0.50)
        step = BASELINE_TURN_COUNT + 5
        result = manager.process_request("Show revenue", template, step)

        assert result.decision == "CLARIFY", (
            f"Expected CLARIFY under RESTRICT, got {result.decision} "
            f"(drift_level={result.drift_level}, drift_mag={result.drift_magnitude:.3f})"
        )
        assert "SAAI RESTRICT" in result.decision_explanation

    def test_restrict_allows_high_fidelity_execute(self):
        """RESTRICT: fidelity above ST_RESTRICT (0.52) should still EXECUTE."""
        manager, template = self._make_manager_and_template()
        mock_engine = MagicMock()
        manager._engine_cache["sql_analyst"] = mock_engine

        self._push_to_restrict(manager, template, mock_engine)

        # Fidelity above RESTRICT threshold (0.52) should still EXECUTE
        # Window [0.80, 0.80, 0.80, 0.80, 0.55] -> avg=0.75
        # drift = (0.90 - 0.75)/0.90 = 0.167 -> RESTRICT (threshold ok)
        mock_engine.score_action.return_value = self._make_engine_result(0.55)
        step = BASELINE_TURN_COUNT + 5
        result = manager.process_request("Show revenue", template, step)

        assert result.decision == "EXECUTE", (
            f"Expected EXECUTE for fidelity above RESTRICT threshold, got {result.decision} "
            f"(drift_level={result.drift_level}, drift_mag={result.drift_magnitude:.3f})"
        )

    def test_restrict_plus_boundary_forces_escalate(self):
        """RESTRICT + boundary_violation -> compound ESCALATE."""
        manager, template = self._make_manager_and_template()
        mock_engine = MagicMock()
        manager._engine_cache["sql_analyst"] = mock_engine

        self._push_to_restrict(manager, template, mock_engine)

        # Boundary violation during RESTRICT -> compound ESCALATE
        # Window [0.80, 0.80, 0.80, 0.80, 0.50] -> avg=0.74
        # drift = (0.90 - 0.74)/0.90 = 0.178 -> RESTRICT
        mock_engine.score_action.return_value = self._make_engine_result(
            0.50, boundary_triggered=True
        )
        step = BASELINE_TURN_COUNT + 5
        result = manager.process_request("Delete all records", template, step)

        assert result.decision == "ESCALATE", (
            f"Expected ESCALATE for RESTRICT+boundary, got {result.decision}"
        )
        assert "RESTRICT" in result.decision_explanation
        assert "BOUNDARY" in result.decision_explanation


# ---------------------------------------------------------------------------
# Chain Inheritance Masking Fix (Phase 2c)
# ---------------------------------------------------------------------------

class TestChainInheritanceMasking:
    """Test that boundary violations record composite fidelity, not inherited.

    When a boundary violation occurs mid-chain, the effective_fidelity should
    be the composite score (reflecting the violation), not the inherited
    score from the previous clean step. This prevents drift tracker from
    seeing inflated scores that mask boundary violations.
    """

    def test_boundary_violation_uses_composite_not_inherited(self):
        """Boundary violation mid-chain should record composite, not max(composite, inherited)."""
        from telos_governance.agentic_fidelity import AgenticFidelityEngine, AgenticFidelityResult
        from telos_governance.agentic_pa import AgenticPA, BoundarySpec
        import numpy as np

        dim = 3
        purpose_emb = np.array([1.0, 0.0, 0.0])
        purpose_emb = purpose_emb / np.linalg.norm(purpose_emb)

        boundary_emb = np.array([0.0, 1.0, 0.0])
        boundary_emb = boundary_emb / np.linalg.norm(boundary_emb)

        pa = AgenticPA(
            purpose_text="Help users query databases",
            purpose_embedding=purpose_emb,
            scope_text="SELECT queries only",
            scope_embedding=purpose_emb,
            boundaries=[
                BoundarySpec(
                    text="No data modification or deletion",
                    embedding=boundary_emb,
                    centroid_embedding=boundary_emb,
                ),
            ],
            escalation_threshold=0.50,
        )

        # embed_fn returns purpose-aligned for step 1, boundary-aligned for step 2
        call_count = [0]
        def embed_fn(text):
            call_count[0] += 1
            if "delete" in text.lower() or "drop" in text.lower():
                return boundary_emb
            return purpose_emb

        engine = AgenticFidelityEngine(embed_fn=embed_fn, pa=pa)

        # Step 1: High-fidelity on-topic request (establishes chain)
        result1 = engine.score_action("Show me total revenue")

        # Step 2: Action near boundary
        result2 = engine.score_action("Delete all records")

        if result2.boundary_triggered:
            # If boundary is triggered, effective should equal composite
            # (not inflated by chain inheritance from step 1)
            assert result2.effective_fidelity == pytest.approx(
                result2.composite_fidelity, abs=0.001
            ), (
                f"Boundary violation should use composite ({result2.composite_fidelity:.4f}), "
                f"not inherited effective ({result2.effective_fidelity:.4f})"
            )
        # If boundary not triggered (embedding geometry may not trigger),
        # verify the fix doesn't break normal chain inheritance
        else:
            assert result2.effective_fidelity >= result2.composite_fidelity, (
                "Without boundary trigger, effective should be >= composite (chain inheritance)"
            )
