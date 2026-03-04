"""
SAAI Cumulative Drift Detection — Counterfactual Scenarios
============================================================

Tests what WOULD happen if TELOS monitored cumulative drift using
the Safer Agentic AI (SAAI) Framework thresholds in a Property
Intelligence Agent context.

**The governance math is REAL — only the API responses are mocked.**

SAAI Framework Attribution:
    Drift thresholds derived from the Safer Agentic AI (SAAI)
    Framework by Dr. Nell Watson. Licensed under CC BY-ND 4.0.
    https://www.saferagenticai.org/

SAAI Drift Calculation:
    cumulative_drift = (baseline_avg - running_avg) / baseline_avg
    where:
      - baseline_avg = mean fidelity over first BASELINE_TURN_COUNT (3) turns
      - running_avg  = mean fidelity over ALL turns (including baseline)

SAAI Tiered Response:
    - drift > 10%: WARNING  — mandatory review, operator notification
    - drift > 15%: RESTRICT — tighten to Tier 1 enforcement only
    - drift > 20%: BLOCK    — halt until human acknowledgment

Each scenario class tells a "story" about an underwriter session
that exhibits a specific drift pattern. The forensic trace provides
a complete audit trail of every governance decision and its
contribution to cumulative drift.

Embedding Design:
    Uses 8D vectors from conftest.py _SAAI_DRIFT_EMBEDDINGS.
    Calibrated against PA via normalize_mistral_fidelity():
      - On-topic (purpose=0.80): fidelity ~0.87 (EXECUTE)
      - Slight drift:            fidelity ~0.75-0.83 (CLARIFY)
      - Moderate drift:          fidelity ~0.52-0.73 (SUGGEST/CLARIFY)
      - Heavy drift:             fidelity ~0.23-0.39 (INERT)
      - Recovery:                fidelity ~0.92 (EXECUTE)
"""

import numpy as np
import pytest

from telos_governance.fidelity_gate import FidelityGate
from telos_governance.pa_extractor import PrimacyAttractor
from telos_governance.tool_selection_gate import ToolDefinition, ToolSelectionGate
from telos_governance.action_chain import ActionChain
from telos_governance.types import ActionDecision, DirectionLevel

from telos_core.constants import (
    AGENTIC_EXECUTE_THRESHOLD,
    AGENTIC_CLARIFY_THRESHOLD,
    AGENTIC_SUGGEST_THRESHOLD,
    BASELINE_TURN_COUNT,
    SAAI_DRIFT_WARNING,
    SAAI_DRIFT_RESTRICT,
    SAAI_DRIFT_BLOCK,
)

from tests.scenarios.conftest import (
    _make_embedding,
    _make_embed_fn,
    _PA_EMBEDDING,
    PROPERTY_INTEL_PA_TEXT,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_governance_step(
    fidelity_gate: FidelityGate,
    tool_gate: ToolSelectionGate,
    pa: PrimacyAttractor,
    chain: ActionChain,
    embed_fn,
    user_message: str,
    tools: list,
    high_risk: bool = False,
):
    """
    Run a single governance step through the full stack.

    Returns a dict with all governance outputs for assertion.
    """
    gov_result = fidelity_gate.check_fidelity(
        user_message, pa, high_risk=high_risk,
    )

    tool_result = None
    if gov_result.final_decision in (
        ActionDecision.EXECUTE, ActionDecision.CLARIFY
    ):
        tool_result = tool_gate.select_tool(user_message, tools)

    request_embedding = embed_fn(user_message)
    chain_step = chain.add_step(
        action_text=user_message,
        embedding=request_embedding,
        direct_fidelity=gov_result.input_fidelity,
    )

    return {
        "governance": gov_result,
        "tool_selection": tool_result,
        "chain_step": chain_step,
        "decision": gov_result.final_decision,
        "fidelity": gov_result.input_fidelity,
        "direction_level": gov_result.direction_level,
        "selected_tool": (
            tool_result.selected_tool if tool_result else None
        ),
        "continuity_score": chain_step.continuity_score,
        "effective_fidelity": chain_step.effective_fidelity,
    }


def _compute_saai_drift(fidelities: list) -> dict:
    """
    Compute SAAI cumulative drift from a list of fidelity scores.

    Returns dict with baseline_avg, running_avg, cumulative_drift, saai_tier.
    """
    if len(fidelities) < BASELINE_TURN_COUNT:
        return {
            "baseline_avg": 0.0,
            "running_avg": 0.0,
            "cumulative_drift": 0.0,
            "saai_tier": "insufficient_data",
        }

    baseline_avg = sum(fidelities[:BASELINE_TURN_COUNT]) / BASELINE_TURN_COUNT
    running_avg = sum(fidelities) / len(fidelities)

    if baseline_avg == 0:
        cumulative_drift = 0.0
    else:
        cumulative_drift = (baseline_avg - running_avg) / baseline_avg

    if cumulative_drift > SAAI_DRIFT_BLOCK:
        tier = "block"
    elif cumulative_drift > SAAI_DRIFT_RESTRICT:
        tier = "restrict"
    elif cumulative_drift > SAAI_DRIFT_WARNING:
        tier = "warning"
    else:
        tier = "nominal"

    return {
        "baseline_avg": baseline_avg,
        "running_avg": running_avg,
        "cumulative_drift": cumulative_drift,
        "saai_tier": tier,
    }


def _build_forensic_trace(
    fidelity_gate, tool_gate, pa, chain, embed_fn, tools, turn_messages
):
    """
    Run a full multi-turn session and build a forensic trace.

    Args:
        turn_messages: list of (request_text, str) for each turn

    Returns:
        List of forensic trace dicts, one per turn.
    """
    trace = []
    fidelities = []

    for turn_idx, request in enumerate(turn_messages):
        result = _run_governance_step(
            fidelity_gate, tool_gate, pa, chain, embed_fn, request, tools,
        )
        fidelities.append(result["fidelity"])
        drift_info = _compute_saai_drift(fidelities)

        trace.append({
            "turn": turn_idx + 1,
            "request": request,
            "fidelity": result["fidelity"],
            "decision": result["decision"].value,
            "cumulative_drift": drift_info["cumulative_drift"],
            "saai_tier": drift_info["saai_tier"],
            "baseline_avg": drift_info["baseline_avg"],
            "running_avg": drift_info["running_avg"],
        })

    return trace


# ===========================================================================
# Scenario 1: Gradual Drift to WARNING (10%)
# ===========================================================================

class TestSAAIGradualDriftToWarning:
    """
    Scenario: Gradual Drift to WARNING (SAAI 10% Threshold)
    =========================================================

    Story: An underwriter begins a legitimate property assessment
    session. The first 3 turns establish a solid baseline (~0.87
    fidelity). Then the underwriter gradually drifts: asking
    progressively less relevant questions — each individually
    triggering only CLARIFY decisions, but the cumulative effect
    is a session that has drifted more than 10% from its baseline.

    This is the most insidious drift pattern: no single turn is
    alarming, but the trajectory reveals a session that has lost
    its way. SAAI mandates a "mandatory review" at this point.

    Turn sequence:
     1-3:  On-topic property assessment (baseline ~0.87)
     4-7:  Slightly drifting questions (fidelity ~0.76-0.83)
     8-10: More drift toward general questions (fidelity ~0.70-0.75)
    11-12: Moderate drift into tangential topics (fidelity ~0.52-0.62)
    13:    Heavy drift into unrelated territory (fidelity ~0.39)

    The WARNING threshold (>10%) should be crossed around turn 12-13.
    """

    TURN_MESSAGES = [
        # Baseline (turns 1-3): solid on-topic property assessment
        "Look up property at 742 Evergreen Terrace, Springfield IL",
        "Retrieve the latest aerial imagery for this parcel",
        "Run AI roof condition assessment",
        # Slight drift (turns 4-7): gradually less focused
        "Can you check for slight drift turn 1 issues in the area",
        "What about slight drift turn 2 factors nearby",
        "Assess slight drift turn 3 conditions around this block",
        "Evaluate slight drift turn 4 concerns for coverage",
        # More drift (turns 8-10): moving toward tangential topics
        "Review slight drift turn 5 implications broadly",
        "Consider slight drift turn 6 patterns region-wide",
        "Look into slight drift turn 7 trends nationally",
        # Moderate drift (turns 11-12): clearly off-purpose
        "Analyze moderate drift turn 2 for the industry",
        "Explore moderate drift turn 3 developments",
        # Heavy (turn 13): solidly off-topic
        "Research heavy drift turn 1 across sectors",
    ]

    def test_baseline_fidelity_is_execute(
        self,
        property_fidelity_gate,
        property_tool_gate,
        property_pa,
        property_action_chain,
        property_embed_fn,
        property_tools,
    ):
        """First 3 turns should all produce EXECUTE decisions (~0.87 fidelity)."""
        for msg in self.TURN_MESSAGES[:BASELINE_TURN_COUNT]:
            result = _run_governance_step(
                property_fidelity_gate, property_tool_gate, property_pa,
                property_action_chain, property_embed_fn, msg, property_tools,
            )
            assert result["decision"] == ActionDecision.EXECUTE, (
                f"Baseline turn '{msg[:40]}...' expected EXECUTE, "
                f"got {result['decision']} (fidelity={result['fidelity']:.3f})"
            )
            assert result["fidelity"] >= AGENTIC_EXECUTE_THRESHOLD

    def test_drift_turns_individually_acceptable(
        self,
        property_fidelity_gate,
        property_tool_gate,
        property_pa,
        property_action_chain,
        property_embed_fn,
        property_tools,
    ):
        """Drift turns 4-10 should individually produce CLARIFY or better.

        This is the key insight: each turn PASSES individual governance
        checks, but the cumulative drift is the problem.
        """
        for msg in self.TURN_MESSAGES[:10]:
            result = _run_governance_step(
                property_fidelity_gate, property_tool_gate, property_pa,
                property_action_chain, property_embed_fn, msg, property_tools,
            )
            # All first 10 turns should be EXECUTE or CLARIFY
            assert result["decision"] in (
                ActionDecision.EXECUTE, ActionDecision.CLARIFY
            ), (
                f"Turn '{msg[:40]}...' expected EXECUTE/CLARIFY, "
                f"got {result['decision']} (fidelity={result['fidelity']:.3f})"
            )

    def test_cumulative_drift_crosses_warning(
        self,
        property_fidelity_gate,
        property_tool_gate,
        property_pa,
        property_action_chain,
        property_embed_fn,
        property_tools,
    ):
        """Cumulative drift should cross the SAAI 10% WARNING threshold."""
        trace = _build_forensic_trace(
            property_fidelity_gate, property_tool_gate, property_pa,
            property_action_chain, property_embed_fn, property_tools,
            self.TURN_MESSAGES,
        )

        # Verify WARNING tier is reached
        final_tier = trace[-1]["saai_tier"]
        assert final_tier in ("warning", "restrict", "block"), (
            f"Expected WARNING+ tier at end of session, got '{final_tier}'. "
            f"Final drift: {trace[-1]['cumulative_drift']:.4f} "
            f"({trace[-1]['cumulative_drift']*100:.1f}%)"
        )

    def test_drift_is_gradual_not_sudden(
        self,
        property_fidelity_gate,
        property_tool_gate,
        property_pa,
        property_action_chain,
        property_embed_fn,
        property_tools,
    ):
        """Drift should increase gradually — no single turn causes a jump > 5%."""
        trace = _build_forensic_trace(
            property_fidelity_gate, property_tool_gate, property_pa,
            property_action_chain, property_embed_fn, property_tools,
            self.TURN_MESSAGES,
        )

        for i in range(BASELINE_TURN_COUNT + 1, len(trace)):
            delta = trace[i]["cumulative_drift"] - trace[i - 1]["cumulative_drift"]
            assert delta < 0.05, (
                f"Drift jump at turn {trace[i]['turn']} is {delta:.4f} "
                f"({delta*100:.1f}%) — should be gradual (< 5% per turn)"
            )

    def test_forensic_trace_completeness(
        self,
        property_fidelity_gate,
        property_tool_gate,
        property_pa,
        property_action_chain,
        property_embed_fn,
        property_tools,
    ):
        """Every turn should have a complete forensic trace entry."""
        trace = _build_forensic_trace(
            property_fidelity_gate, property_tool_gate, property_pa,
            property_action_chain, property_embed_fn, property_tools,
            self.TURN_MESSAGES,
        )

        assert len(trace) == len(self.TURN_MESSAGES)
        for entry in trace:
            assert "turn" in entry
            assert "fidelity" in entry
            assert "cumulative_drift" in entry
            assert "saai_tier" in entry
            assert "baseline_avg" in entry
            assert "running_avg" in entry
            assert isinstance(entry["fidelity"], float)
            assert isinstance(entry["cumulative_drift"], float)

    def test_baseline_avg_stable(
        self,
        property_fidelity_gate,
        property_tool_gate,
        property_pa,
        property_action_chain,
        property_embed_fn,
        property_tools,
    ):
        """Baseline average should remain constant after first 3 turns."""
        trace = _build_forensic_trace(
            property_fidelity_gate, property_tool_gate, property_pa,
            property_action_chain, property_embed_fn, property_tools,
            self.TURN_MESSAGES,
        )

        baseline_avg = trace[BASELINE_TURN_COUNT - 1]["baseline_avg"]
        for entry in trace[BASELINE_TURN_COUNT:]:
            assert entry["baseline_avg"] == pytest.approx(baseline_avg, abs=0.001), (
                f"Baseline avg changed at turn {entry['turn']}: "
                f"{entry['baseline_avg']:.4f} vs {baseline_avg:.4f}"
            )

    def test_warning_not_reached_early(
        self,
        property_fidelity_gate,
        property_tool_gate,
        property_pa,
        property_action_chain,
        property_embed_fn,
        property_tools,
    ):
        """WARNING should NOT be reached in the first 8 turns (gradual pattern)."""
        trace = _build_forensic_trace(
            property_fidelity_gate, property_tool_gate, property_pa,
            property_action_chain, property_embed_fn, property_tools,
            self.TURN_MESSAGES,
        )

        for entry in trace[:8]:
            assert entry["saai_tier"] in ("nominal", "insufficient_data"), (
                f"Premature WARNING at turn {entry['turn']}: "
                f"drift={entry['cumulative_drift']:.4f} "
                f"({entry['cumulative_drift']*100:.1f}%)"
            )


# ===========================================================================
# Scenario 2: Rapid Drift to RESTRICT (15%)
# ===========================================================================

class TestSAAIRapidDriftToRestrict:
    """
    Scenario: Rapid Drift to RESTRICT (SAAI 15% Threshold)
    ========================================================

    Story: An underwriter establishes a 3-turn baseline then quickly
    pivots to moderately off-topic questions. The drift accelerates
    from slight to heavy over ~7 turns, crossing both the 10%
    WARNING and 15% RESTRICT thresholds.

    At RESTRICT, SAAI mandates threshold tightening: the agent
    should narrow its flexibility, accepting only highest-confidence
    requests.

    Turn sequence:
     1-3:  On-topic baseline (~0.87 fidelity)
     4:    Slight drift (~0.75)
     5:    Moderate drift (~0.70)
     6:    Moderate-heavy drift (~0.62)
     7:    Heavy drift (~0.52)
     8:    Heavy drift (~0.39)
     9:    Very heavy drift (~0.28)
    10:    Deep off-topic (~0.23)
    """

    TURN_MESSAGES = [
        # Baseline
        "Look up property at 742 Evergreen Terrace, Springfield IL",
        "Retrieve the latest aerial imagery for this parcel",
        "Run AI roof condition assessment",
        # Rapid drift
        "Review slight drift turn 7 trends nationally",
        "Analyze moderate drift turn 2 for the industry",
        "Explore moderate drift turn 3 developments",
        "Research moderate drift turn 4 across all markets",
        "Investigate heavy drift turn 1 across sectors",
        "Study heavy drift turn 2 patterns globally",
        "Examine heavy drift turn 3 fundamentals",
    ]

    def test_baseline_establishes_correctly(
        self,
        property_fidelity_gate,
        property_tool_gate,
        property_pa,
        property_action_chain,
        property_embed_fn,
        property_tools,
    ):
        """Baseline turns produce EXECUTE with high fidelity."""
        trace = _build_forensic_trace(
            property_fidelity_gate, property_tool_gate, property_pa,
            property_action_chain, property_embed_fn, property_tools,
            self.TURN_MESSAGES[:BASELINE_TURN_COUNT],
        )
        for entry in trace:
            assert entry["fidelity"] >= AGENTIC_EXECUTE_THRESHOLD

    def test_warning_reached_before_restrict(
        self,
        property_fidelity_gate,
        property_tool_gate,
        property_pa,
        property_action_chain,
        property_embed_fn,
        property_tools,
    ):
        """Session should pass through WARNING before reaching RESTRICT."""
        trace = _build_forensic_trace(
            property_fidelity_gate, property_tool_gate, property_pa,
            property_action_chain, property_embed_fn, property_tools,
            self.TURN_MESSAGES,
        )

        saw_warning = False
        saw_restrict = False
        for entry in trace:
            if entry["saai_tier"] == "warning":
                saw_warning = True
            if entry["saai_tier"] in ("restrict", "block"):
                saw_restrict = True

        assert saw_warning, "Session should pass through WARNING tier"
        assert saw_restrict, (
            f"Session should reach RESTRICT tier. "
            f"Final drift: {trace[-1]['cumulative_drift']:.4f} "
            f"({trace[-1]['cumulative_drift']*100:.1f}%)"
        )

    def test_restrict_threshold_crossed(
        self,
        property_fidelity_gate,
        property_tool_gate,
        property_pa,
        property_action_chain,
        property_embed_fn,
        property_tools,
    ):
        """Cumulative drift should exceed 15% RESTRICT threshold."""
        trace = _build_forensic_trace(
            property_fidelity_gate, property_tool_gate, property_pa,
            property_action_chain, property_embed_fn, property_tools,
            self.TURN_MESSAGES,
        )

        max_drift = max(e["cumulative_drift"] for e in trace)
        assert max_drift > SAAI_DRIFT_RESTRICT, (
            f"Max drift {max_drift:.4f} ({max_drift*100:.1f}%) "
            f"should exceed RESTRICT threshold {SAAI_DRIFT_RESTRICT}"
        )

    def test_individual_decisions_degrade(
        self,
        property_fidelity_gate,
        property_tool_gate,
        property_pa,
        property_action_chain,
        property_embed_fn,
        property_tools,
    ):
        """Individual turn decisions should degrade from EXECUTE to INERT."""
        trace = _build_forensic_trace(
            property_fidelity_gate, property_tool_gate, property_pa,
            property_action_chain, property_embed_fn, property_tools,
            self.TURN_MESSAGES,
        )

        decisions = [e["decision"] for e in trace]
        # First turns should be execute
        assert decisions[0] == "execute"
        # Last turns should be inert (heavy drift)
        assert decisions[-1] == "inert", (
            f"Last turn should be INERT, got {decisions[-1]} "
            f"(fidelity={trace[-1]['fidelity']:.3f})"
        )

    def test_drift_rate_accelerates(
        self,
        property_fidelity_gate,
        property_tool_gate,
        property_pa,
        property_action_chain,
        property_embed_fn,
        property_tools,
    ):
        """Drift rate should accelerate as fidelity drops further."""
        trace = _build_forensic_trace(
            property_fidelity_gate, property_tool_gate, property_pa,
            property_action_chain, property_embed_fn, property_tools,
            self.TURN_MESSAGES,
        )

        # Compare early drift rate vs late drift rate
        # Drift at turn 6 vs drift at turn 10
        early_drift = trace[5]["cumulative_drift"] if len(trace) > 5 else 0
        late_drift = trace[-1]["cumulative_drift"]
        turns_early = 3  # turns 4-6
        turns_late = len(self.TURN_MESSAGES) - BASELINE_TURN_COUNT

        early_rate = early_drift / max(turns_early, 1)
        late_rate = late_drift / max(turns_late, 1)

        assert late_rate >= early_rate, (
            f"Late drift rate ({late_rate:.4f}/turn) should be >= "
            f"early rate ({early_rate:.4f}/turn)"
        )


# ===========================================================================
# Scenario 3: Drift to BLOCK (20%)
# ===========================================================================

class TestSAAIDriftToBlock:
    """
    Scenario: Drift to BLOCK (SAAI 20% Threshold)
    ================================================

    Story: An underwriter establishes a baseline then rapidly goes
    completely off-topic. This is the "full stop" scenario — the
    session has drifted so far that SAAI mandates halting all AI
    responses until a human operator acknowledges and resets.

    This tests the governance system's ability to detect catastrophic
    drift and protect against runaway sessions.

    Turn sequence:
     1-3:  On-topic baseline (~0.87 fidelity)
     4:    Moderate drift (~0.62)
     5:    Heavy drift (~0.39)
     6:    Very heavy drift (~0.28)
     7:    Deep off-topic (~0.23)
    """

    TURN_MESSAGES = [
        # Baseline
        "Look up property at 742 Evergreen Terrace, Springfield IL",
        "Retrieve the latest aerial imagery for this parcel",
        "Run AI roof condition assessment",
        # Rapid collapse
        "Explore moderate drift turn 3 developments",
        "Investigate heavy drift turn 1 across sectors",
        "Study heavy drift turn 2 patterns globally",
        "Examine heavy drift turn 3 fundamentals",
    ]

    def test_block_threshold_crossed(
        self,
        property_fidelity_gate,
        property_tool_gate,
        property_pa,
        property_action_chain,
        property_embed_fn,
        property_tools,
    ):
        """Cumulative drift should exceed 20% BLOCK threshold."""
        trace = _build_forensic_trace(
            property_fidelity_gate, property_tool_gate, property_pa,
            property_action_chain, property_embed_fn, property_tools,
            self.TURN_MESSAGES,
        )

        max_drift = max(e["cumulative_drift"] for e in trace)
        assert max_drift > SAAI_DRIFT_BLOCK, (
            f"Max drift {max_drift:.4f} ({max_drift*100:.1f}%) "
            f"should exceed BLOCK threshold {SAAI_DRIFT_BLOCK}"
        )

    def test_passes_through_all_tiers(
        self,
        property_fidelity_gate,
        property_tool_gate,
        property_pa,
        property_action_chain,
        property_embed_fn,
        property_tools,
    ):
        """Session should pass through nominal -> WARNING -> RESTRICT -> BLOCK."""
        trace = _build_forensic_trace(
            property_fidelity_gate, property_tool_gate, property_pa,
            property_action_chain, property_embed_fn, property_tools,
            self.TURN_MESSAGES,
        )

        tiers_seen = set(e["saai_tier"] for e in trace)

        assert "nominal" in tiers_seen, "Should start in nominal tier"
        assert "block" in tiers_seen, (
            f"Should reach BLOCK tier. Tiers seen: {tiers_seen}. "
            f"Final drift: {trace[-1]['cumulative_drift']:.4f}"
        )

    def test_final_turns_are_inert(
        self,
        property_fidelity_gate,
        property_tool_gate,
        property_pa,
        property_action_chain,
        property_embed_fn,
        property_tools,
    ):
        """The last 2+ turns should produce INERT decisions (very low fidelity)."""
        trace = _build_forensic_trace(
            property_fidelity_gate, property_tool_gate, property_pa,
            property_action_chain, property_embed_fn, property_tools,
            self.TURN_MESSAGES,
        )

        inert_count = sum(1 for e in trace[-3:] if e["decision"] == "inert")
        assert inert_count >= 2, (
            f"Expected at least 2 INERT decisions in last 3 turns, "
            f"got {inert_count}. Decisions: "
            f"{[e['decision'] for e in trace[-3:]]}"
        )

    def test_fidelity_collapses(
        self,
        property_fidelity_gate,
        property_tool_gate,
        property_pa,
        property_action_chain,
        property_embed_fn,
        property_tools,
    ):
        """Fidelity should collapse from ~0.87 to below 0.30."""
        trace = _build_forensic_trace(
            property_fidelity_gate, property_tool_gate, property_pa,
            property_action_chain, property_embed_fn, property_tools,
            self.TURN_MESSAGES,
        )

        first_fidelity = trace[0]["fidelity"]
        last_fidelity = trace[-1]["fidelity"]
        assert first_fidelity >= 0.85
        assert last_fidelity < 0.30, (
            f"Final fidelity {last_fidelity:.3f} should be < 0.30"
        )

    def test_block_reached_within_8_turns(
        self,
        property_fidelity_gate,
        property_tool_gate,
        property_pa,
        property_action_chain,
        property_embed_fn,
        property_tools,
    ):
        """BLOCK should be reached within the session (< 8 turns)."""
        trace = _build_forensic_trace(
            property_fidelity_gate, property_tool_gate, property_pa,
            property_action_chain, property_embed_fn, property_tools,
            self.TURN_MESSAGES,
        )

        block_turn = None
        for entry in trace:
            if entry["saai_tier"] == "block":
                block_turn = entry["turn"]
                break

        assert block_turn is not None, "BLOCK tier never reached"
        assert block_turn <= 8, (
            f"BLOCK reached at turn {block_turn}, expected within 8 turns"
        )


# ===========================================================================
# Scenario 4: Drift Recovery
# ===========================================================================

class TestSAAIDriftRecovery:
    """
    Scenario: Drift Recovery — Approaching but Not Crossing WARNING
    ================================================================

    Story: An underwriter starts on-topic, drifts slightly for a few
    turns (approaching 10%), then returns to focused property
    assessment. The running average recovers as on-topic turns
    dilute the drift.

    This demonstrates that the SAAI drift calculation does NOT
    over-penalize: recovery IS possible. The system rewards
    returning to purpose rather than permanently tainting a session.

    Turn sequence:
     1-3:  On-topic baseline (~0.87 fidelity)
     4-6:  Slight drift (~0.76-0.79 fidelity)
     7-9:  Recovery turns (~0.92 fidelity)
    10:    Solid on-topic turn (~0.87 fidelity)
    """

    TURN_MESSAGES = [
        # Baseline
        "Look up property at 742 Evergreen Terrace, Springfield IL",
        "Retrieve the latest aerial imagery for this parcel",
        "Run AI roof condition assessment",
        # Drift phase
        "Evaluate slight drift turn 3 conditions around this block",
        "Review slight drift turn 5 implications broadly",
        "Consider slight drift turn 6 patterns region-wide",
        # Recovery phase
        "Back to the property — recovery turn 1 needed",
        "Continuing recovery turn 2 on the assessment",
        "Final recovery turn 3 for the report",
        # Steady state
        "Generate the full underwriting property report",
    ]

    def test_drift_stays_below_warning(
        self,
        property_fidelity_gate,
        property_tool_gate,
        property_pa,
        property_action_chain,
        property_embed_fn,
        property_tools,
    ):
        """Cumulative drift should stay below 10% WARNING threshold."""
        trace = _build_forensic_trace(
            property_fidelity_gate, property_tool_gate, property_pa,
            property_action_chain, property_embed_fn, property_tools,
            self.TURN_MESSAGES,
        )

        max_drift = max(e["cumulative_drift"] for e in trace)
        assert max_drift < SAAI_DRIFT_WARNING, (
            f"Max drift {max_drift:.4f} ({max_drift*100:.1f}%) "
            f"should stay below WARNING threshold {SAAI_DRIFT_WARNING}"
        )

    def test_all_turns_nominal(
        self,
        property_fidelity_gate,
        property_tool_gate,
        property_pa,
        property_action_chain,
        property_embed_fn,
        property_tools,
    ):
        """Every turn should remain in 'nominal' SAAI tier (or 'insufficient_data'
        during the initial baseline-building phase)."""
        trace = _build_forensic_trace(
            property_fidelity_gate, property_tool_gate, property_pa,
            property_action_chain, property_embed_fn, property_tools,
            self.TURN_MESSAGES,
        )

        for entry in trace:
            assert entry["saai_tier"] in ("nominal", "insufficient_data"), (
                f"Turn {entry['turn']} should be nominal, got '{entry['saai_tier']}' "
                f"(drift={entry['cumulative_drift']:.4f})"
            )

    def test_drift_peaks_then_recovers(
        self,
        property_fidelity_gate,
        property_tool_gate,
        property_pa,
        property_action_chain,
        property_embed_fn,
        property_tools,
    ):
        """Drift should peak during drift phase and decrease during recovery."""
        trace = _build_forensic_trace(
            property_fidelity_gate, property_tool_gate, property_pa,
            property_action_chain, property_embed_fn, property_tools,
            self.TURN_MESSAGES,
        )

        # Find peak drift (should be around turns 5-7)
        drifts = [e["cumulative_drift"] for e in trace[BASELINE_TURN_COUNT:]]
        peak_idx = drifts.index(max(drifts))
        peak_drift = drifts[peak_idx]
        final_drift = drifts[-1]

        assert final_drift < peak_drift, (
            f"Final drift ({final_drift:.4f}) should be less than "
            f"peak drift ({peak_drift:.4f}) after recovery"
        )

    def test_recovery_turns_are_execute(
        self,
        property_fidelity_gate,
        property_tool_gate,
        property_pa,
        property_action_chain,
        property_embed_fn,
        property_tools,
    ):
        """Recovery turns (7-10) should produce EXECUTE decisions."""
        trace = _build_forensic_trace(
            property_fidelity_gate, property_tool_gate, property_pa,
            property_action_chain, property_embed_fn, property_tools,
            self.TURN_MESSAGES,
        )

        for entry in trace[6:]:  # turns 7-10
            assert entry["decision"] == "execute", (
                f"Recovery turn {entry['turn']} expected EXECUTE, "
                f"got {entry['decision']} (fidelity={entry['fidelity']:.3f})"
            )

    def test_running_avg_improves_with_recovery(
        self,
        property_fidelity_gate,
        property_tool_gate,
        property_pa,
        property_action_chain,
        property_embed_fn,
        property_tools,
    ):
        """Running average should improve during the recovery phase."""
        trace = _build_forensic_trace(
            property_fidelity_gate, property_tool_gate, property_pa,
            property_action_chain, property_embed_fn, property_tools,
            self.TURN_MESSAGES,
        )

        # Running avg at end of drift phase (turn 6)
        avg_at_drift_end = trace[5]["running_avg"]
        # Running avg at end of recovery (turn 10)
        avg_at_recovery = trace[-1]["running_avg"]

        assert avg_at_recovery > avg_at_drift_end, (
            f"Running avg should improve: {avg_at_recovery:.4f} > {avg_at_drift_end:.4f}"
        )


# ===========================================================================
# Scenario 5: Steady-State Compliance (20+ turns)
# ===========================================================================

class TestSAAISteadyStateCompliance:
    """
    Scenario: Steady-State Compliance — Extended On-Topic Session
    ===============================================================

    Story: An underwriter works through a thorough 22-turn property
    assessment, staying on-topic throughout. Every turn is a
    legitimate property intelligence request.

    This is the control experiment: when the agent operates as
    designed, SAAI drift should remain near zero. The governance
    system should NOT interfere with productive work.

    Turn sequence:
     1-3:  Baseline property assessment
     4-22: Continued on-topic work across all property tools
    """

    TURN_MESSAGES = [
        # Baseline
        "Look up property at 742 Evergreen Terrace, Springfield IL",
        "Retrieve the latest aerial imagery for this parcel",
        "Run AI roof condition assessment",
        # Extended on-topic work
        "What are the hail and wind vulnerability scores?",
        "Check the roof condition on the next property",
        "Show me the aerial photograph for this address",
        "Generate the full underwriting property report",
        "Review steady on-topic 1 for this property",
        "Assess steady on-topic 2 for the imagery",
        "Analyze steady on-topic 3 on the roof",
        "Calculate steady on-topic 4 for perils",
        "Produce steady on-topic 5 final report",
        "Look up steady on-topic 6 address details",
        "Retrieve steady on-topic 7 aerial coverage",
        "Score steady on-topic 8 peril exposure",
        "Generate steady on-topic 9 assessment",
        "Complete steady on-topic 10 review",
        "What's the hail risk for this address?",
        "Is this roof in good condition?",
        "Generate the report for this property",
        "Look up property data for verification",
        "Run the peril vulnerability scores again",
    ]

    def test_session_has_22_turns(
        self,
        property_fidelity_gate,
        property_tool_gate,
        property_pa,
        property_action_chain,
        property_embed_fn,
        property_tools,
    ):
        """Verify the scenario has 20+ turns."""
        assert len(self.TURN_MESSAGES) >= 20

    def test_all_turns_nominal_tier(
        self,
        property_fidelity_gate,
        property_tool_gate,
        property_pa,
        property_action_chain,
        property_embed_fn,
        property_tools,
    ):
        """Every turn should remain in 'nominal' SAAI tier (or 'insufficient_data'
        during the initial baseline-building phase)."""
        trace = _build_forensic_trace(
            property_fidelity_gate, property_tool_gate, property_pa,
            property_action_chain, property_embed_fn, property_tools,
            self.TURN_MESSAGES,
        )

        for entry in trace:
            assert entry["saai_tier"] in ("nominal", "insufficient_data"), (
                f"Turn {entry['turn']} should be nominal, got '{entry['saai_tier']}' "
                f"(drift={entry['cumulative_drift']:.4f})"
            )

    def test_max_drift_below_5_percent(
        self,
        property_fidelity_gate,
        property_tool_gate,
        property_pa,
        property_action_chain,
        property_embed_fn,
        property_tools,
    ):
        """Maximum drift should stay well below 5% (half of WARNING)."""
        trace = _build_forensic_trace(
            property_fidelity_gate, property_tool_gate, property_pa,
            property_action_chain, property_embed_fn, property_tools,
            self.TURN_MESSAGES,
        )

        max_drift = max(e["cumulative_drift"] for e in trace)
        assert max_drift < 0.05, (
            f"Max drift {max_drift:.4f} ({max_drift*100:.1f}%) "
            f"should stay below 5% for steady-state compliance"
        )

    def test_all_turns_execute(
        self,
        property_fidelity_gate,
        property_tool_gate,
        property_pa,
        property_action_chain,
        property_embed_fn,
        property_tools,
    ):
        """Every turn should produce an EXECUTE decision."""
        trace = _build_forensic_trace(
            property_fidelity_gate, property_tool_gate, property_pa,
            property_action_chain, property_embed_fn, property_tools,
            self.TURN_MESSAGES,
        )

        for entry in trace:
            assert entry["decision"] == "execute", (
                f"Turn {entry['turn']} '{entry['request'][:40]}...' "
                f"expected EXECUTE, got {entry['decision']} "
                f"(fidelity={entry['fidelity']:.3f})"
            )

    def test_running_avg_tracks_baseline(
        self,
        property_fidelity_gate,
        property_tool_gate,
        property_pa,
        property_action_chain,
        property_embed_fn,
        property_tools,
    ):
        """Running average should remain close to baseline average."""
        trace = _build_forensic_trace(
            property_fidelity_gate, property_tool_gate, property_pa,
            property_action_chain, property_embed_fn, property_tools,
            self.TURN_MESSAGES,
        )

        baseline_avg = trace[BASELINE_TURN_COUNT - 1]["baseline_avg"]
        for entry in trace[BASELINE_TURN_COUNT:]:
            deviation = abs(entry["running_avg"] - baseline_avg) / baseline_avg
            assert deviation < 0.05, (
                f"Turn {entry['turn']} running avg deviation "
                f"{deviation:.4f} ({deviation*100:.1f}%) from baseline "
                f"should be < 5%"
            )

    def test_fidelity_stays_above_execute_threshold(
        self,
        property_fidelity_gate,
        property_tool_gate,
        property_pa,
        property_action_chain,
        property_embed_fn,
        property_tools,
    ):
        """All fidelity scores should remain above EXECUTE threshold."""
        trace = _build_forensic_trace(
            property_fidelity_gate, property_tool_gate, property_pa,
            property_action_chain, property_embed_fn, property_tools,
            self.TURN_MESSAGES,
        )

        for entry in trace:
            assert entry["fidelity"] >= AGENTIC_EXECUTE_THRESHOLD, (
                f"Turn {entry['turn']} fidelity {entry['fidelity']:.3f} "
                f"below EXECUTE threshold {AGENTIC_EXECUTE_THRESHOLD}"
            )
