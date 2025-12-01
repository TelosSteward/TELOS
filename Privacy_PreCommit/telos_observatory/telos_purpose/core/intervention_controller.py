"""
Mathematical intervention controller for TELOS.

Implements dual-boundary control:
1. Attractor pull (corrective force toward primacy center)
2. Anti-meta push (repel meta-commentary about governance)

Uses proportional control: intervention strength scales with drift magnitude.
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import time
import numpy as np

from .primacy_math import MathematicalState, PrimacyAttractorMath

@dataclass
class InterventionResult:
    """Result from applying an intervention."""
    type: str
    modified_response: Optional[str]
    strength: float
    reason: str
    timestamp: float
    metrics: Dict[str, float]

class MathematicalInterventionController:
    """
    Proportional intervention controller with dual boundaries.

    Control law:
    - If error < ε_min: no intervention
    - If ε_min < error < ε_max: reminder injection
    - If error > ε_max: guided regeneration

    Force scaling: F_correction = K * error_signal
    """

    def __init__(
        self,
        attractor: PrimacyAttractorMath,
        llm_client,
        embedding_provider,
        enable_interventions: bool = True
    ):
        """
        Args:
            attractor: Mathematical attractor definition
            llm_client: LLM client for regeneration
            embedding_provider: Embedding provider for drift measurement
            enable_interventions: Master switch for interventions
        """
        self.attractor = attractor
        self.llm_client = llm_client
        self.embedding_provider = embedding_provider
        self.enable_interventions = enable_interventions
        
        # Control parameters
        self.K_attractor = 1.5  # Attractor pull gain
        self.K_antimeta = 2.0   # Anti-meta push gain
        
        # Error thresholds - scale with tolerance
        # Low tolerance → low thresholds → early intervention
        # High tolerance → high thresholds → late intervention
        tolerance = attractor.constraint_tolerance
        self.epsilon_min = 0.1 + (tolerance * 0.3)  # 0.0→0.1, 1.0→0.4
        self.epsilon_max = 0.5 + (tolerance * 0.4)  # 0.0→0.5, 1.0→0.9
        
        # Intervention budget
        self.max_regenerations = 3
        self.regeneration_count = 0
        
        # Telemetry
        self.intervention_history: List[InterventionResult] = []
        self.total_interventions = 0

    def process_turn(
        self,
        state: MathematicalState,
        response_text: str,
        conversation_history: List[Dict[str, str]],
        turn_number: int
    ) -> Dict[str, Any]:
        """
        Evaluate state and apply intervention if needed.
        
        Args:
            state: Current mathematical state
            response_text: LLM response to evaluate
            conversation_history: Full conversation context
            turn_number: Current turn index
            
        Returns:
            Dict with intervention_applied, modified_response, metrics
        """
        error_signal = self.attractor.compute_error_signal(state)
        in_basin = self.attractor.compute_basin_membership(state)
        
        # Check for meta-commentary
        is_meta = self._detect_meta_commentary(response_text)
        
        # Decide intervention
        intervention_result = None
        
        if not self.enable_interventions:
            pass
        elif is_meta:
            intervention_result = self._apply_antimeta_intervention(
                response_text, conversation_history, turn_number
            )
        elif error_signal > self.epsilon_max and self.regeneration_count < self.max_regenerations:
            intervention_result = self._apply_regeneration(
                response_text, conversation_history, error_signal, turn_number
            )
        elif error_signal > self.epsilon_min and not in_basin:
            intervention_result = self._apply_reminder(
                response_text, error_signal, turn_number
            )
        
        if intervention_result:
            self.intervention_history.append(intervention_result)
            self.total_interventions += 1
            if intervention_result.type == "regeneration":
                self.regeneration_count += 1
        
        return {
            "intervention_applied": intervention_result is not None,
            "intervention_result": intervention_result,
            "error_signal": error_signal,
            "in_basin": in_basin,
            "is_meta": is_meta
        }

    def _detect_meta_commentary(self, text: str) -> bool:
        """Detect if response is meta-commentary about governance."""
        meta_indicators = [
            "my purpose is",
            "my constraints",
            "i am designed to",
            "my guardrails",
            "my governance",
            "according to my instructions",
            "primacy attractor",
            "constraint tolerance"
        ]
        text_lower = text.lower()
        return any(phrase in text_lower for phrase in meta_indicators)

    def _apply_reminder(
        self,
        response_text: str,
        error_signal: float,
        turn_number: int
    ) -> InterventionResult:
        """
        Apply gentle reminder (context injection).
        
        Prepends reminder to response without regenerating.
        Strength scales with error magnitude and constraint rigidity.
        """
        # Reminder strength = rigidity × error × gain
        rigidity = self.attractor.constraint_rigidity
        reminder_strength = min(rigidity * error_signal * self.K_attractor, 1.0)
        
        reminder = (
            "[Note: Staying aligned with session purpose and scope boundaries.] "
        )
        
        modified_response = reminder + response_text
        
        return InterventionResult(
            type="reminder",
            modified_response=modified_response,
            strength=reminder_strength,
            reason=f"Drift detected (error={error_signal:.2f}), applying reminder",
            timestamp=time.time(),
            metrics={"error_signal": error_signal, "strength": reminder_strength}
        )

    def _apply_regeneration(
        self,
        original_response: str,
        conversation_history: List[Dict[str, str]],
        error_signal: float,
        turn_number: int
    ) -> InterventionResult:
        """
        Apply guided regeneration with corrective prompt.
        
        Regenerates response with explicit alignment guidance.
        Correction strength scales with constraint rigidity.
        """
        rigidity = self.attractor.constraint_rigidity
        correction_strength = min(rigidity * error_signal * self.K_attractor, 1.0)
        
        # Build corrective prompt
        corrective_prompt = {
            "role": "system",
            "content": (
                f"The previous response drifted from the session purpose. "
                f"Please regenerate while staying focused on: {self.attractor.purpose_vector[:10]}... "
                f"Maintain alignment with declared scope and boundaries. "
                f"Correction strength: {correction_strength:.2f}"
            )
        }
        
        # Regenerate
        try:
            regeneration_history = conversation_history + [corrective_prompt]
            regenerated = self.llm_client.generate(
                messages=regeneration_history,
                max_tokens=500,
                temperature=0.7
            )
            
            return InterventionResult(
                type="regeneration",
                modified_response=regenerated,
                strength=correction_strength,
                reason=f"Severe drift (error={error_signal:.2f}), regenerated response",
                timestamp=time.time(),
                metrics={
                    "error_signal": error_signal,
                    "strength": correction_strength,
                    "regeneration_count": self.regeneration_count + 1
                }
            )
        except Exception as e:
            # Fallback to reminder if regeneration fails
            return self._apply_reminder(original_response, error_signal, turn_number)

    def _apply_antimeta_intervention(
        self,
        response_text: str,
        conversation_history: List[Dict[str, str]],
        turn_number: int
    ) -> InterventionResult:
        """
        Apply anti-meta boundary enforcement.
        
        Regenerates to remove meta-commentary about governance.
        """
        antimeta_prompt = {
            "role": "system",
            "content": (
                "Respond naturally to the user's question without discussing "
                "your instructions, constraints, or governance mechanisms. "
                "Focus on being helpful with the actual request."
            )
        }
        
        try:
            regeneration_history = conversation_history + [antimeta_prompt]
            regenerated = self.llm_client.generate(
                messages=regeneration_history,
                max_tokens=500,
                temperature=0.7
            )
            
            return InterventionResult(
                type="antimeta",
                modified_response=regenerated,
                strength=self.K_antimeta,
                reason="Meta-commentary detected, regenerated without self-reference",
                timestamp=time.time(),
                metrics={"antimeta_strength": self.K_antimeta}
            )
        except Exception as e:
            # If regeneration fails, strip meta-commentary manually
            cleaned = self._strip_meta_phrases(response_text)
            return InterventionResult(
                type="antimeta_fallback",
                modified_response=cleaned,
                strength=self.K_antimeta * 0.5,
                reason="Meta-commentary detected, manually cleaned",
                timestamp=time.time(),
                metrics={"antimeta_strength": self.K_antimeta * 0.5}
            )

    def _strip_meta_phrases(self, text: str) -> str:
        """Remove meta-commentary phrases from text."""
        meta_patterns = [
            "my purpose is",
            "my constraints are",
            "i am designed to",
            "according to my instructions",
            "primacy attractor",
            "constraint tolerance"
        ]
        
        cleaned = text
        for pattern in meta_patterns:
            # Simple removal - in production would use more sophisticated NLP
            cleaned = cleaned.replace(pattern, "")
        
        return cleaned.strip()

    def get_intervention_statistics(self) -> Dict[str, Any]:
        """Return intervention statistics."""
        if not self.intervention_history:
            return {
                "total_interventions": 0,
                "by_type": {},
                "avg_strength": 0.0
            }
        
        by_type = {}
        for result in self.intervention_history:
            by_type[result.type] = by_type.get(result.type, 0) + 1
        
        avg_strength = np.mean([r.strength for r in self.intervention_history])
        
        return {
            "total_interventions": self.total_interventions,
            "by_type": by_type,
            "avg_strength": float(avg_strength),
            "regeneration_count": self.regeneration_count,
            "max_regenerations": self.max_regenerations,
            "thresholds": {
                "epsilon_min": self.epsilon_min,
                "epsilon_max": self.epsilon_max,
                "constraint_tolerance": self.attractor.constraint_tolerance
            }
        }
