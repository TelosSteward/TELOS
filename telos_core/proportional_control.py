"""
TELOS Proportional Controller
=============================

Graduated intervention system from Whitepaper Section 5.3.
Implements F = K * e_t control law for purpose alignment.

The controller provides graduated intervention strength based on
the error signal (deviation from ideal fidelity).

Battle-tested from TELOS Observatory V3.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, List
import logging

from .constants import (
    DEFAULT_K_ATTRACTOR,
    DEFAULT_K_ANTIMETA,
    DEFAULT_TAU,
    EPSILON_MIN_FACTOR,
    EPSILON_MAX_FACTOR,
    INTERVENTION_STATES,
    FIDELITY_GREEN,
)

logger = logging.getLogger(__name__)


class InterventionState(Enum):
    """Intervention states based on error magnitude."""
    MONITOR = "monitor"       # Within tolerance, just observe
    CORRECT = "correct"       # Minor correction needed
    INTERVENE = "intervene"   # Active intervention required
    ESCALATE = "escalate"     # Escalation to human/supervisor


@dataclass
class InterventionRecord:
    """Record of an intervention decision for audit."""
    timestamp: datetime
    turn_number: int
    fidelity: float
    error_signal: float
    intervention_strength: float
    state: InterventionState
    action_taken: str
    context: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "turn_number": self.turn_number,
            "fidelity": self.fidelity,
            "error_signal": self.error_signal,
            "intervention_strength": self.intervention_strength,
            "state": self.state.value,
            "action_taken": self.action_taken,
            "context": self.context,
        }


def calculate_error_signal(
    current_fidelity: float,
    target_fidelity: float = FIDELITY_GREEN,
) -> float:
    """
    Calculate error signal (deviation from target).

    Formula: e_t = (target - current) / target

    Args:
        current_fidelity: Current fidelity score
        target_fidelity: Target fidelity (default: GREEN zone)

    Returns:
        Error signal in [0, 1+] range
    """
    if target_fidelity <= 0:
        return 1.0

    error = (target_fidelity - current_fidelity) / target_fidelity
    return max(0.0, error)


def calculate_intervention_strength(
    error_signal: float,
    K: float = DEFAULT_K_ATTRACTOR,
    rigidity: float = 1.0,
) -> float:
    """
    Calculate intervention strength from error signal.

    Formula: F = min(rigidity * K * e_t, 1.0)

    Args:
        error_signal: Error signal from calculate_error_signal
        K: Proportional gain
        rigidity: PA rigidity (0.0 = soft, 1.0 = rigid)

    Returns:
        Intervention strength in [0, 1] range
    """
    raw_strength = rigidity * K * error_signal
    return min(1.0, max(0.0, raw_strength))


class ProportionalController:
    """
    TELOS Proportional Controller for graduated interventions.

    Implements the control law from Whitepaper Section 5.3:
        F = min(rigidity * K * e_t, 1.0)

    Where:
        - F: Intervention strength
        - K: Proportional gain (K_ATTRACTOR = 1.5)
        - e_t: Error signal (fidelity deviation)
        - rigidity: PA rigidity parameter

    The controller determines intervention state based on error magnitude
    and provides graduated response strength.

    Usage:
        controller = ProportionalController(K=1.5, tau=0.40)

        result = controller.compute(current_fidelity=0.55)
        print(f"State: {result.state}, Strength: {result.strength}")
    """

    def __init__(
        self,
        K_attractor: float = DEFAULT_K_ATTRACTOR,
        K_antimeta: float = DEFAULT_K_ANTIMETA,
        tau: float = DEFAULT_TAU,
        target_fidelity: float = FIDELITY_GREEN,
    ):
        """
        Initialize the proportional controller.

        Args:
            K_attractor: Proportional gain for purpose alignment
            K_antimeta: Gain for anti-meta commentary detection
            tau: Constraint tolerance (affects epsilon bounds)
            target_fidelity: Target fidelity level
        """
        self.K_attractor = K_attractor
        self.K_antimeta = K_antimeta
        self.tau = tau
        self.target_fidelity = target_fidelity

        # Calculate epsilon bounds from tau
        self.epsilon_min = tau * EPSILON_MIN_FACTOR
        self.epsilon_max = tau * EPSILON_MAX_FACTOR

        # Intervention thresholds (scaled by tau)
        self._compute_thresholds()

        # Audit trail
        self.intervention_history: List[InterventionRecord] = []
        self.turn_counter = 0

    def _compute_thresholds(self):
        """Compute intervention state thresholds from tau."""
        # Base thresholds from constants, scaled by tau
        base_thresholds = INTERVENTION_STATES
        self.thresholds = {
            InterventionState.MONITOR: base_thresholds["MONITOR"]["threshold"],
            InterventionState.CORRECT: base_thresholds["CORRECT"]["threshold"],
            InterventionState.INTERVENE: base_thresholds["INTERVENE"]["threshold"],
            InterventionState.ESCALATE: base_thresholds["ESCALATE"]["threshold"],
        }

    def compute(
        self,
        current_fidelity: float,
        rigidity: float = 1.0,
        context: Optional[str] = None,
    ) -> InterventionRecord:
        """
        Compute intervention decision for current fidelity.

        Args:
            current_fidelity: Current fidelity score
            rigidity: PA rigidity (0.0-1.0)
            context: Optional context for audit

        Returns:
            InterventionRecord with decision details
        """
        self.turn_counter += 1

        # Calculate error signal
        error_signal = calculate_error_signal(
            current_fidelity, self.target_fidelity
        )

        # Calculate intervention strength
        strength = calculate_intervention_strength(
            error_signal, self.K_attractor, rigidity
        )

        # Determine state from error magnitude
        state = self._determine_state(error_signal)

        # Determine action
        action = self._determine_action(state, strength)

        # Create record
        record = InterventionRecord(
            timestamp=datetime.now(),
            turn_number=self.turn_counter,
            fidelity=current_fidelity,
            error_signal=error_signal,
            intervention_strength=strength,
            state=state,
            action_taken=action,
            context=context,
        )

        # Add to history
        self.intervention_history.append(record)

        return record

    def _determine_state(self, error_signal: float) -> InterventionState:
        """Determine intervention state from error magnitude."""
        if error_signal <= self.thresholds[InterventionState.MONITOR]:
            return InterventionState.MONITOR
        elif error_signal <= self.thresholds[InterventionState.CORRECT]:
            return InterventionState.CORRECT
        elif error_signal <= self.thresholds[InterventionState.INTERVENE]:
            return InterventionState.INTERVENE
        else:
            return InterventionState.ESCALATE

    def _determine_action(
        self, state: InterventionState, strength: float
    ) -> str:
        """Determine action description from state."""
        actions = {
            InterventionState.MONITOR: "Continue without intervention",
            InterventionState.CORRECT: "Apply gentle context correction",
            InterventionState.INTERVENE: "Active purpose redirection",
            InterventionState.ESCALATE: "Escalate for approval",
        }
        return f"{actions[state]} (strength: {strength:.2f})"

    def get_history(self) -> List[dict]:
        """Get intervention history as list of dicts."""
        return [r.to_dict() for r in self.intervention_history]

    def reset(self):
        """Reset controller state."""
        self.intervention_history.clear()
        self.turn_counter = 0


# =============================================================================
# ANTI-META COMMENTARY DETECTION
# =============================================================================

def detect_meta_commentary(text: str) -> tuple[bool, float]:
    """
    Detect meta-commentary about the governance system itself.

    Meta-commentary is when the AI talks about its own constraints,
    purpose alignment, or the TELOS system rather than the actual task.

    Args:
        text: Text to analyze

    Returns:
        Tuple of (is_meta, confidence)
    """
    meta_patterns = [
        "my purpose",
        "my constraints",
        "i am designed to",
        "i cannot help with",
        "outside my scope",
        "let me redirect",
        "aligned with",
        "primacy attractor",
        "fidelity",
        "governance",
    ]

    text_lower = text.lower()
    matches = sum(1 for p in meta_patterns if p in text_lower)

    if matches == 0:
        return False, 0.0
    elif matches == 1:
        return True, 0.4
    elif matches == 2:
        return True, 0.7
    else:
        return True, 0.9


def suppress_meta_commentary(
    text: str,
    K_antimeta: float = DEFAULT_K_ANTIMETA,
) -> tuple[str, float]:
    """
    Apply anti-meta suppression to text.

    Returns the original text and a suppression factor.
    The suppression factor can be used to weight the response.

    Args:
        text: Text to check
        K_antimeta: Anti-meta gain

    Returns:
        Tuple of (text, suppression_factor)
    """
    is_meta, confidence = detect_meta_commentary(text)

    if not is_meta:
        return text, 1.0

    # Calculate suppression factor
    suppression = 1.0 - (confidence * K_antimeta)
    suppression = max(0.1, min(1.0, suppression))

    return text, suppression
