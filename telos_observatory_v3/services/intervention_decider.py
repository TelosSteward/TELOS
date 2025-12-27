"""
Intervention Decider Module
===========================

Extracted from beta_response_manager.py for single-responsibility design.

This module handles all intervention decision logic:
- Zone classification (GREEN, YELLOW, ORANGE, RED)
- Intervention trigger decision
- Proportional control calculations
- Governance trace logging
"""

import logging
import streamlit as st
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# Import thresholds from single source of truth
from telos_purpose.core.constants import (
    FIDELITY_GREEN, FIDELITY_YELLOW, FIDELITY_ORANGE, FIDELITY_RED,
    SIMILARITY_BASELINE,
    ST_FIDELITY_GREEN, ST_FIDELITY_YELLOW, ST_FIDELITY_ORANGE, ST_FIDELITY_RED,
)

logger = logging.getLogger(__name__)

# Proportional controller gain (from whitepaper Section 5.3)
K_ATTRACTOR = 1.5


class FidelityZone(Enum):
    """Fidelity zones for intervention classification."""
    GREEN = "green"
    YELLOW = "yellow"
    ORANGE = "orange"
    RED = "red"


@dataclass
class InterventionDecision:
    """Result of intervention decision analysis."""
    should_intervene: bool
    zone: FidelityZone
    intervention_reason: Optional[str]
    intervention_strength: Optional[str]  # 'soft', 'moderate', 'strong'
    layer1_triggered: bool  # Baseline hard block
    layer2_in_basin: bool   # Within basin membership
    error_signal: float     # 1.0 - fidelity
    controller_strength: float  # K * error_signal (capped at 1.0)
    semantic_band: str      # For linguistic output styling
    intervention_state: str  # MONITOR, CORRECT, INTERVENE, ESCALATE


class InterventionDecider:
    """
    Decides whether to trigger intervention based on fidelity scores.

    Uses two-tier architecture:
    - Layer 1: Baseline pre-filter (extreme off-topic detection)
    - Layer 2: Zone classification (graduated intervention)
    """

    def __init__(self, use_rescaled_fidelity: bool = False):
        """
        Initialize the intervention decider.

        Args:
            use_rescaled_fidelity: Whether using template mode (ST thresholds)
        """
        self.use_rescaled_fidelity = use_rescaled_fidelity

    def get_thresholds(self) -> Dict[str, float]:
        """Get model-appropriate thresholds."""
        if self.use_rescaled_fidelity:
            return {
                'green': ST_FIDELITY_GREEN,
                'yellow': ST_FIDELITY_YELLOW,
                'orange': ST_FIDELITY_ORANGE,
                'red': ST_FIDELITY_RED
            }
        else:
            return {
                'green': FIDELITY_GREEN,
                'yellow': FIDELITY_YELLOW,
                'orange': FIDELITY_ORANGE,
                'red': FIDELITY_RED
            }

    def decide(
        self,
        fidelity: float,
        raw_similarity: float,
        baseline_hard_block: bool
    ) -> InterventionDecision:
        """
        Make intervention decision based on fidelity metrics.

        Args:
            fidelity: Calculated fidelity score (possibly adjusted)
            raw_similarity: Raw cosine similarity
            baseline_hard_block: Whether Layer 1 triggered

        Returns:
            InterventionDecision with full analysis
        """
        thresholds = self.get_thresholds()

        # Zone classification
        if fidelity >= thresholds['green']:
            zone = FidelityZone.GREEN
        elif fidelity >= thresholds['yellow']:
            zone = FidelityZone.YELLOW
        elif fidelity >= thresholds['orange']:
            zone = FidelityZone.ORANGE
        else:
            zone = FidelityZone.RED

        # Basin membership check
        in_basin = fidelity >= thresholds['orange']

        # Intervention decision: Below GREEN = intervene
        should_intervene = baseline_hard_block or fidelity < thresholds['green']

        # Proportional control calculations
        error_signal = 1.0 - fidelity
        controller_strength = min(K_ATTRACTOR * error_signal, 1.0)

        # Intervention state mapping
        if error_signal < 0.30:
            intervention_state = "MONITOR"
        elif error_signal < 0.50:
            intervention_state = "CORRECT"
        elif error_signal < 0.67:
            intervention_state = "INTERVENE"
        else:
            intervention_state = "ESCALATE"

        # Semantic band mapping
        if controller_strength < 0.45:
            semantic_band = "minimal"
        elif controller_strength < 0.60:
            semantic_band = "light"
        elif controller_strength < 0.75:
            semantic_band = "moderate"
        elif controller_strength < 0.85:
            semantic_band = "firm"
        else:
            semantic_band = "strong"

        # Build intervention reason and strength
        intervention_reason = None
        intervention_strength = None

        if should_intervene:
            if baseline_hard_block:
                intervention_reason = "Extreme off-topic content detected - hard block triggered"
                intervention_strength = "strong"
            elif zone == FidelityZone.YELLOW:
                intervention_reason = "Minor drift from your stated purpose - Steward is gently guiding you back"
                intervention_strength = "soft"
            elif zone == FidelityZone.ORANGE:
                intervention_reason = "Drift from your stated purpose detected - Steward is guiding you back"
                intervention_strength = "moderate"
            else:  # RED
                intervention_reason = "Significant drift detected - Steward intervention activated"
                intervention_strength = "strong"

        logger.info(f"📐 Intervention Decision: zone={zone.value}, intervene={should_intervene}, "
                   f"e={error_signal:.3f}, strength={controller_strength:.3f}, "
                   f"state={intervention_state}, band={semantic_band}")

        return InterventionDecision(
            should_intervene=should_intervene,
            zone=zone,
            intervention_reason=intervention_reason,
            intervention_strength=intervention_strength,
            layer1_triggered=baseline_hard_block,
            layer2_in_basin=in_basin,
            error_signal=error_signal,
            controller_strength=controller_strength,
            semantic_band=semantic_band,
            intervention_state=intervention_state
        )

    def log_to_governance_trace(
        self,
        decision: InterventionDecision,
        turn_number: int,
        display_fidelity: float,
        raw_similarity: float,
        session_id: str = None
    ):
        """
        Log intervention decision to governance trace collector.

        Args:
            decision: The intervention decision
            turn_number: Current turn number
            display_fidelity: Display-normalized fidelity
            raw_similarity: Raw cosine similarity
            session_id: Session identifier
        """
        try:
            from telos_purpose.core.governance_trace_collector import get_trace_collector
            from telos_purpose.core.evidence_schema import InterventionLevel

            if not session_id:
                session_id = st.session_state.get('session_id', 'unknown')

            collector = get_trace_collector(session_id=session_id)

            # Record fidelity calculation
            collector.record_fidelity(
                turn_number=turn_number,
                raw_similarity=raw_similarity,
                normalized_fidelity=display_fidelity,
                layer1_hard_block=decision.layer1_triggered,
                layer2_outside_basin=not decision.layer2_in_basin,
                distance_from_pa=decision.error_signal,
                in_basin=decision.layer2_in_basin,
            )

            # Record intervention if triggered
            if decision.should_intervene:
                level_map = {
                    FidelityZone.YELLOW: InterventionLevel.CORRECT,
                    FidelityZone.ORANGE: InterventionLevel.INTERVENE,
                    FidelityZone.RED: InterventionLevel.ESCALATE,
                }

                if decision.layer1_triggered:
                    intervention_level = InterventionLevel.HARD_BLOCK
                    trigger_reason = "hard_block"
                else:
                    intervention_level = level_map.get(decision.zone, InterventionLevel.INTERVENE)
                    trigger_reason = "basin_exit" if not decision.layer2_in_basin else "drift_detected"

                collector.record_intervention(
                    turn_number=turn_number,
                    intervention_level=intervention_level,
                    trigger_reason=trigger_reason,
                    fidelity_at_trigger=display_fidelity,
                    controller_strength=decision.controller_strength,
                    semantic_band=decision.semantic_band,
                    action_taken="steward_redirect",
                )

            logger.debug("Governance trace recorded successfully")

        except ImportError:
            logger.debug("Governance trace collector not available")
        except Exception as e:
            logger.debug(f"Governance trace logging skipped: {e}")


def create_intervention_decider_from_manager(manager) -> InterventionDecider:
    """
    Factory function to create InterventionDecider from BetaResponseManager.

    Args:
        manager: BetaResponseManager instance

    Returns:
        Configured InterventionDecider
    """
    return InterventionDecider(
        use_rescaled_fidelity=getattr(manager, 'use_rescaled_fidelity', False)
    )
