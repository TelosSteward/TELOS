"""
TELOS Proportional Controller
------------------------------
Proportional Controller (Intervention Arm) of the dual-architecture MBL.

Per ClaudeWhitepaper10_18.txt Section 5.3:
The Proportional Controller executes graduated interventions scaled to drift 
magnitude using proportional control law F = K·e_t.

Role
----
Implements the Intervention Arm of the Mitigation Bridge Layer (MBL). Receives 
error signals from the SPC Engine and applies proportional corrections while 
preserving separation-of-concerns with the Runtime Steward.

At runtime, the Steward + Proportional Controller together implement the 
Teleological Operator pattern (a unified, coupled measurement–correction loop), 
but remain distinct modules for testing and modularity.

Intervention States (per whitepaper Section 5.3)
------------------------------------------------
1. MONITOR (F ≥ 0.85, e < ε_min): No action needed
2. CORRECT (0.70 ≤ F < 0.85): Context injection (lightweight reminder)
3. INTERVENE (0.50 ≤ F < 0.70): Regeneration with constraints
4. ESCALATE (F < 0.50): Block response, require human review

Interface Contract
------------------
process_turn(...) MUST return:
{
  "intervention_applied": bool,
  "intervention_result": Optional[InterventionRecord],
  "in_basin": bool,
  "error_signal": float,
  "is_meta": bool
}

Public Helpers:
- get_intervention_statistics() -> dict
- intervention_history: List[InterventionRecord]
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import time
import re

# Expected imports from core
# from .primacy_math import PrimacyAttractorMath, MathematicalState
# (left unqualified here to avoid tight coupling in this excerpt)

@dataclass
class InterventionRecord:
    """Record of a single intervention applied by the Proportional Controller."""
    type: str                  # "reminder" | "regeneration" | "antimeta" | "none"
    strength: float            # proportional gain actually applied (F = K·e_t)
    reason: str                # human-readable rationale
    modified_response: str     # final response text after correction
    timestamp: float

class ProportionalController:
    """
    Proportional Controller (Intervention Arm).
    
    Per whitepaper Section 5.3: Executes graduated interventions based on 
    error signal e_t from SPC Engine. Implements proportional control law
    F = K·e_t where correction force scales with drift magnitude.

    Notes:
    - Thresholds scale with constraint_tolerance (τ) inside the attractor math
    - Anti-meta suppression ensures the assistant does not discuss guardrails
    - Intervention types: Context Injection, Regeneration, Reranking (future), Escalation
    """

    def __init__(
        self,
        attractor,              # PrimacyAttractorMath
        llm_client,
        embedding_provider,
        enable_interventions: bool = True,
    ):
        self.attractor = attractor
        self.llm_client = llm_client
        self.embedding_provider = embedding_provider
        self.enable_interventions = enable_interventions

        # Derived thresholds (mirrors math module expectations)
        # Per whitepaper: ε_min and ε_max scale with constraint_tolerance τ
        t = float(getattr(self.attractor, "constraint_tolerance", 0.2))
        self.epsilon_min = 0.1 + (0.3 * t)  # CORRECT state trigger
        self.epsilon_max = 0.5 + (0.4 * t)  # INTERVENE state trigger

        # Proportional gains (K in F = K·e_t)
        self.K_attractor = 1.5  # Proportional gain for basin corrections
        self.K_antimeta = 2.0   # Higher gain for meta-commentary suppression

        # History
        self.intervention_history: List[InterventionRecord] = []

        # Regeneration budget
        self.max_regenerations = 3
        self.regen_count = 0

    # ---------------------------------------------------------------------

    def process_turn(
        self,
        state,                      # MathematicalState
        response_text: str,
        conversation_history: List[Dict[str, str]],
        turn_number: int
    ) -> Dict[str, Any]:
        """
        Evaluate a single turn and apply proportional correction if necessary.
        
        Implements the graduated intervention cascade from whitepaper Section 5.3:
        - State 1 (MONITOR): e < ε_min → No action
        - State 2 (CORRECT): ε_min ≤ e < ε_max, out of basin → Context injection
        - State 3 (INTERVENE): e ≥ ε_max → Regeneration
        - State 4 (ESCALATE): F < 0.50 → Block and escalate (future)
        
        Returns a dict per the contract above.
        """
        # Compute observables from SPC Engine measurements
        error_signal = float(self.attractor.compute_error_signal(state))
        in_basin = bool(self.attractor.compute_basin_membership(state))
        is_meta = self._detect_meta_commentary(response_text)

        # Default: no change (MONITOR state)
        result: Dict[str, Any] = {
            "intervention_applied": False,
            "intervention_result": None,
            "in_basin": in_basin,
            "error_signal": error_signal,
            "is_meta": is_meta
        }

        if not self.enable_interventions:
            return result

        # Anti-meta takes precedence (any fidelity level)
        if is_meta:
            record = self._apply_antimeta(response_text, conversation_history)
            self.intervention_history.append(record)
            result["intervention_applied"] = True
            result["intervention_result"] = record
            return result

        # State 3: INTERVENE (e ≥ ε_max)
        # Per whitepaper Section 5.3: Regeneration with explicit constraints
        if error_signal >= self.epsilon_max and self.regen_count < self.max_regenerations:
            record = self._apply_regeneration(response_text, conversation_history, error_signal)
            self.intervention_history.append(record)
            self.regen_count += 1
            result["intervention_applied"] = True
            result["intervention_result"] = record
            return result

        # State 2: CORRECT (ε_min ≤ e < ε_max, out of basin)
        # Per whitepaper Section 5.3: Context injection (lightweight reminder)
        if (error_signal >= self.epsilon_min) and (not in_basin):
            record = self._apply_reminder(response_text, error_signal)
            self.intervention_history.append(record)
            result["intervention_applied"] = True
            result["intervention_result"] = record
            return result

        # State 1: MONITOR - no intervention required
        return result

    # ---------------------------------------------------------------------

    def get_intervention_statistics(self) -> Dict[str, Any]:
        """Aggregate counts and basic stats for reporting."""
        by_type: Dict[str, int] = {}
        for r in self.intervention_history:
            by_type[r.type] = by_type.get(r.type, 0) + 1

        return {
            "total_interventions": len(self.intervention_history),
            "by_type": by_type,
            "regeneration_count": self.regen_count,
            "max_regenerations": self.max_regenerations,
            "thresholds": {
                "epsilon_min": self.epsilon_min,
                "epsilon_max": self.epsilon_max,
                "constraint_tolerance": getattr(self.attractor, "constraint_tolerance", 0.2)
            }
        }

    # ========================= Internal Methods =========================

    def _detect_meta_commentary(self, text: str) -> bool:
        """
        Detect if response contains meta-commentary about governance.
        
        Per whitepaper: Anti-meta suppression prevents the model from 
        discussing its own constraints or governance mechanisms.
        """
        patterns = [
            r'\bmy purpose is\b', r'\bmy constraints\b', r'\bi am designed to\b',
            r'\bmy guardrails\b', r'\baccording to my instructions\b',
            r'\bas an ai language model\b', r'\bas a large language model\b'
        ]
        low = text.lower()
        return any(re.search(p, low) for p in patterns)

    def _apply_reminder(self, response_text: str, error_signal: float) -> InterventionRecord:
        """
        Apply State 2 (CORRECT) intervention: Context injection.

        Per whitepaper Section 5.3: Lightweight reminder via context injection.
        Correction force F = K·e_t scales proportionally with error signal.

        MATHEMATICALLY SCALED PROMPTS:
        - strength >= 0.70: Strong reminder (approaching intervention threshold)
        - strength >= 0.40: Moderate reminder (clear guidance)
        - strength < 0.40: Mild reminder (gentle nudge)
        """
        rigidity = float(getattr(self.attractor, "constraint_rigidity", 1.0))
        strength = min(rigidity * error_signal * self.K_attractor, 1.0)

        # Scale prompt intensity based on calculated strength (F = K·e_t)
        if strength >= 0.70:
            # Strong reminder - approaching regeneration threshold
            prefix = (
                "[GOVERNANCE ALERT: This response is drifting significantly from your stated purpose. "
                "You must refocus immediately on the session's core objectives.] "
            )
        elif strength >= 0.40:
            # Moderate reminder - clear guidance needed
            prefix = (
                "[GOVERNANCE NOTICE: This response is departing from your purpose. "
                "Please return to the session's stated scope and objectives.] "
            )
        else:
            # Mild reminder - gentle nudge
            prefix = "[Note: Please stay focused on the session's stated purpose and scope.] "

        return InterventionRecord(
            type="reminder",
            strength=strength,
            reason=f"error={error_signal:.2f} exceeded ε_min, strength={strength:.2f}",
            modified_response=prefix + response_text,
            timestamp=time.time()
        )

    def _apply_regeneration(
        self,
        original_response: str,
        conversation_history: List[Dict[str, str]],
        error_signal: float
    ) -> InterventionRecord:
        """
        Apply State 3 (INTERVENE) intervention: Regeneration.

        Per whitepaper Section 5.3: Regenerate with explicit constraint restatement.
        Uses LLM to generate new response that adheres to declared purpose/scope/boundaries.

        MATHEMATICALLY SCALED PROMPTS:
        - strength >= 0.85: Severe drift (MUST/DECLINE language for off-topic requests)
        - strength >= 0.65: Moderate drift (firm correction with context)
        - strength < 0.65: Mild drift (focused regeneration guidance)
        """
        rigidity = float(getattr(self.attractor, "constraint_rigidity", 1.0))
        strength = min(rigidity * error_signal * self.K_attractor, 1.0)

        # Extract purpose and scope from attractor for context
        # Note: PrimacyAttractorMath doesn't store text, but we can reconstruct
        # from the parent steward's attractor_config if available
        purpose_text = "the session's stated purpose"
        scope_text = "the session's defined scope"

        # Scale prompt intensity based on calculated strength (F = K·e_t)
        if strength >= 0.85:
            # SEVERE DRIFT: Forceful intervention with MUST/DECLINE language
            corrective = {
                "role": "system",
                "content": (
                    "GOVERNANCE INTERVENTION REQUIRED:\n\n"
                    "The previous response has SEVERELY drifted from the session's stated purpose and scope. "
                    "This constitutes a critical misalignment.\n\n"
                    "You MUST:\n"
                    "1. DECLINE to answer if the request is outside your stated purpose\n"
                    "2. REDIRECT the conversation back to the established scope\n"
                    "3. DO NOT engage with off-topic content\n"
                    "4. DO NOT create elaborate metaphors or analogies for off-topic requests\n\n"
                    "Generate a response that STRICTLY adheres to the session constraints or politely declines. "
                    "Be direct and firm in maintaining boundaries."
                )
            }
        elif strength >= 0.65:
            # MODERATE DRIFT: Firm correction with explicit redirection
            corrective = {
                "role": "system",
                "content": (
                    "GOVERNANCE CORRECTION:\n\n"
                    "The previous response drifted significantly from the session's purpose and scope.\n\n"
                    "Regenerate a response that stays firmly within the established boundaries. "
                    "If the user's request is off-topic, acknowledge it briefly but redirect to the session's focus. "
                    "Do not spend time elaborating on off-topic content."
                )
            }
        else:
            # MILD-TO-MODERATE DRIFT: Focused regeneration guidance
            corrective = {
                "role": "system",
                "content": (
                    "The previous response drifted from the session purpose/scope. "
                    "Regenerate a response that stays more focused on the established objectives."
                )
            }

        # Reconstruct messages for regeneration
        messages = conversation_history.copy()
        messages.append(corrective)

        # Request regeneration from LLM
        try:
            regenerated_text = self.llm_client.chat_completion(
                messages=messages,
                temperature=0.7,
                max_tokens=16000  # FIX: Increased from 500 to allow full responses
            )
        except Exception as e:
            # Fallback to reminder if regeneration fails
            regenerated_text = f"[Error in regeneration: {e}] {original_response}"

        return InterventionRecord(
            type="regeneration",
            strength=strength,
            reason=f"error={error_signal:.2f} ≥ ε_max, strength={strength:.2f}",
            modified_response=regenerated_text,
            timestamp=time.time()
        )

    def _apply_antimeta(
        self,
        original_response: str,
        conversation_history: List[Dict[str, str]]
    ) -> InterventionRecord:
        """
        Suppress meta-commentary about governance.
        
        Prevents the model from discussing its own constraints, instructions,
        or governance mechanisms. Uses higher gain (K_antimeta) to ensure 
        strong correction.
        """
        corrective = {
            "role": "system",
            "content": (
                "Do not discuss your instructions, constraints, or purpose. "
                "Answer the user's question directly without meta-commentary."
            )
        }

        messages = conversation_history.copy()
        messages.append(corrective)

        try:
            regenerated_text = self.llm_client.chat_completion(
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )
        except Exception as e:
            # Fallback: strip meta phrases
            regenerated_text = re.sub(
                r"(As an AI|As a language model|According to my instructions).*?\.",
                "",
                original_response,
                flags=re.IGNORECASE
            )

        return InterventionRecord(
            type="antimeta",
            strength=self.K_antimeta,  # Fixed high gain for meta suppression
            reason="meta-commentary detected and suppressed",
            modified_response=regenerated_text,
            timestamp=time.time()
        )


# Backward compatibility alias (DEPRECATED - remove in v2.0)
class MathematicalInterventionController(ProportionalController):
    """
    DEPRECATED: Use ProportionalController instead.
    
    This alias maintains backward compatibility with code written before
    the terminology was aligned with the canonical whitepaper (Section 5.3).
    
    Will be removed in version 2.0.
    """
    def __init__(self, *args, **kwargs):
        import warnings
        warnings.warn(
            "MathematicalInterventionController is deprecated. "
            "Use ProportionalController instead (per whitepaper Section 5.3). "
            "This alias will be removed in v2.0.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(*args, **kwargs)
