"""
Intercepting LLM Wrapper - Active Mitigation Layer
===================================================

Wraps LLM client to enable steward to CONTROL generation, not just analyze.

This is the key architectural component that transforms TELOS from:
  - PASSIVE: ??? → LLM → Response → Steward → Analyze
  - ACTIVE:  User → Steward → LLM → Steward → Governed Response

The wrapper implements the mitigation bridge layer model where the steward
sits BETWEEN user and LLM, maintaining salience and ensuring coupling.

Flow per turn:
1. Check if attractor established (learning phase vs governance phase)
2. If governing: Check salience (is attractor still prominent in context?)
3. Inject attractor reinforcement if salience degrading
4. Generate response with governed context
5. Check coupling (did response drift from attractor?)
6. Regenerate if decoupled
7. Return governed response
8. Log all interventions for evidence

This enables:
- PREVENTION: Salience maintenance stops drift before it happens
- CORRECTION: Regeneration fixes decoupling when it occurs
- EVIDENCE: All interventions logged with metrics
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
import time


@dataclass
class GovernanceIntervention:
    """Record of a governance intervention during generation."""
    turn_number: int
    intervention_type: str  # "salience_injection", "regeneration", or "none"
    original_response: Optional[str]
    governed_response: str
    fidelity_original: Optional[float]
    fidelity_governed: float
    salience_before: Optional[float]
    salience_after: Optional[float]
    timestamp: float


class InterceptingLLMWrapper:
    """
    LLM wrapper that enables active mitigation by the steward.

    Key difference from passive architecture:
    - Steward calls this wrapper's generate()
    - Wrapper calls LLM internally
    - Wrapper checks/modifies before and after
    - Returns governed response

    Passive would be:
    - Someone else calls LLM
    - Response passed to steward for analysis
    - Too late to prevent drift
    """

    def __init__(
        self,
        llm_client: Any,
        embedding_provider: Any,
        steward_ref: Any,  # Reference to parent steward
        salience_threshold: float = 0.70,
        coupling_threshold: float = 0.80
    ):
        """
        Initialize intercepting wrapper.

        Args:
            llm_client: Underlying LLM client (Mistral, etc.)
            embedding_provider: For measuring salience and coupling
            steward_ref: Reference to parent steward for attractor access
            salience_threshold: Trigger injection if salience drops below this
            coupling_threshold: Trigger regeneration if fidelity drops below this
        """
        self.llm = llm_client
        self.embeddings = embedding_provider
        self.steward = steward_ref
        self.salience_threshold = salience_threshold
        self.coupling_threshold = coupling_threshold

        # Track interventions for evidence
        self.interventions: List[GovernanceIntervention] = []
        self.turn_count = 0

    def generate(
        self,
        user_input: str,
        conversation_context: List[Dict[str, str]]
    ) -> str:
        """
        Generate governed response through mitigation layer.

        This is THE method that implements active governance.

        Args:
            user_input: Current user message
            conversation_context: Full conversation history (messages list)

        Returns:
            Governed response (after salience maintenance and coupling check)
        """
        self.turn_count += 1
        intervention_type = "none"
        original_response = None
        fidelity_original = None
        salience_before = None
        salience_after = None

        # PHASE 1: LEARNING (Attractor not yet established)
        # During learning, generate normally and let progressive extractor analyze
        if not self._attractor_established():
            # Generate without governance (still learning)
            response = self._call_llm(conversation_context, user_input)

            # Record as no intervention
            self.interventions.append(GovernanceIntervention(
                turn_number=self.turn_count,
                intervention_type="learning_phase",
                original_response=None,
                governed_response=response,
                fidelity_original=None,
                fidelity_governed=None,
                salience_before=None,
                salience_after=None,
                timestamp=time.time()
            ))

            return response

        # PHASE 2: GOVERNANCE (Attractor established)

        # Step 1: Salience maintenance
        salience_before = self._measure_salience(conversation_context)

        if salience_before < self.salience_threshold:
            # INTERVENTION: Inject attractor to maintain prominence
            conversation_context = self._inject_salience_reinforcement(conversation_context)
            intervention_type = "salience_injection"
            salience_after = self._measure_salience(conversation_context)
        else:
            salience_after = salience_before

        # Step 1.5: PRE-GENERATION USER INPUT DRIFT CHECK
        # Measure if user input is severely off-topic BEFORE generating expensive response
        user_input_fidelity = self._measure_coupling(user_input)

        # DEBUG: Log user input fidelity
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"🔍 PRE-GENERATION CHECK: User input fidelity = {user_input_fidelity:.3f}")

        if user_input_fidelity < 0.70:  # Low fidelity - drifting from purpose
            # INTERVENTION: User request is drifting from established purpose
            # Redirect instead of generating off-topic response
            refusal_message = (
                f"Your question appears to be drifting from your stated purpose. "
                f"Your session is focused on: {', '.join(self.steward.attractor.purpose)}. "
                f"How can I help you with that instead?"
            )

            # Log pre-generation refusal
            self.interventions.append(GovernanceIntervention(
                turn_number=self.turn_count,
                intervention_type="pre_generation_refusal",
                original_response=None,
                governed_response=refusal_message,
                fidelity_original=None,
                fidelity_governed=user_input_fidelity,
                salience_before=salience_before,
                salience_after=salience_after,
                timestamp=time.time()
            ))

            return refusal_message

        # Step 2: Generate with governed context
        response = self._call_llm(conversation_context, user_input)

        # Step 3: Coupling check
        fidelity = self._measure_coupling(response)

        if fidelity < self.coupling_threshold:
            # INTERVENTION: Regenerate with entrainment
            original_response = response
            fidelity_original = fidelity

            governed_response = self._regenerate_entrained(
                user_input,
                conversation_context,
                original_response
            )

            # Re-measure fidelity
            fidelity_governed = self._measure_coupling(governed_response)

            intervention_type = "regeneration" if intervention_type == "none" else "both"

            # Log intervention
            self.interventions.append(GovernanceIntervention(
                turn_number=self.turn_count,
                intervention_type=intervention_type,
                original_response=original_response,
                governed_response=governed_response,
                fidelity_original=fidelity_original,
                fidelity_governed=fidelity_governed,
                salience_before=salience_before,
                salience_after=salience_after,
                timestamp=time.time()
            ))

            return governed_response
        else:
            # No coupling issue - return original
            self.interventions.append(GovernanceIntervention(
                turn_number=self.turn_count,
                intervention_type=intervention_type if intervention_type != "none" else "none",
                original_response=None,
                governed_response=response,
                fidelity_original=None,
                fidelity_governed=fidelity,
                salience_before=salience_before,
                salience_after=salience_after,
                timestamp=time.time()
            ))

            return response

    def _attractor_established(self) -> bool:
        """Check if steward has established attractor."""
        # Check if steward has a valid attractor
        if not hasattr(self.steward, 'attractor'):
            return False

        attractor = self.steward.attractor

        # Check if attractor has actual content
        if hasattr(attractor, 'purpose'):
            return len(attractor.purpose) > 0
        elif hasattr(self.steward, 'attractor_center') and self.steward.attractor_center is not None:
            return True

        return False

    def _measure_salience(self, conversation_context: List[Dict[str, str]]) -> float:
        """
        Measure attractor salience in current context.

        Salience = how prominent the attractor is in recent conversation.

        High salience: Attractor topics are active in context
        Low salience: Attractor fading, topics drifting away

        Returns:
            Salience score 0.0-1.0 (higher = more prominent)
        """
        if not hasattr(self.steward, 'attractor_center') or self.steward.attractor_center is None:
            return 1.0  # Can't measure salience without attractor

        # Get recent context (last 5 messages or all if fewer)
        recent_context = conversation_context[-5:] if len(conversation_context) > 5 else conversation_context

        # Combine recent messages into single text
        context_text = " ".join([msg.get('content', '') for msg in recent_context])

        if not context_text.strip():
            return 0.5  # Neutral if empty

        # Embed context (encode expects a string, not a list)
        context_embedding = self.embeddings.encode(context_text)

        # Measure similarity to attractor
        attractor_center = self.steward.attractor_center
        similarity = float(np.dot(context_embedding, attractor_center) /
                          (np.linalg.norm(context_embedding) * np.linalg.norm(attractor_center)))

        # Convert to 0-1 range (cosine similarity is -1 to 1)
        salience = (similarity + 1.0) / 2.0

        return salience

    def _inject_salience_reinforcement(
        self,
        conversation_context: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """
        Inject attractor reinforcement into context.

        When salience is degrading, inject the attractor reminder as a
        user message (Mistral doesn't allow system messages after assistant).

        This PREVENTS drift by maintaining attractor prominence.
        """
        attractor = self.steward.attractor

        # Format attractor as reinforcement message
        reinforcement_parts = ["[Governance Context]"]

        if hasattr(attractor, 'purpose') and attractor.purpose:
            purposes = ", ".join(attractor.purpose)
            reinforcement_parts.append(f"Session Purpose: {purposes}")

        if hasattr(attractor, 'scope') and attractor.scope:
            scope_items = ", ".join(attractor.scope)
            reinforcement_parts.append(f"Topics in Scope: {scope_items}")

        if hasattr(attractor, 'boundaries') and attractor.boundaries:
            boundaries = "; ".join(attractor.boundaries)
            reinforcement_parts.append(f"Boundaries: {boundaries}")

        reinforcement_text = "\n".join(reinforcement_parts)

        if len(reinforcement_parts) == 1:
            # Fallback if no textual attractor
            reinforcement_text = "[Governance Context]\nPlease stay focused on the session's established purpose and scope."

        # Inject as USER message at the beginning (safe position)
        # This ensures the attractor is always visible in context
        reinforcement_msg = {
            "role": "user",
            "content": reinforcement_text
        }

        # Prepend to context (always safe)
        return [reinforcement_msg] + conversation_context

    def _measure_coupling(self, response: str) -> float:
        """
        Measure response coupling to attractor.

        Coupling = how well response stays aligned with attractor trajectory.

        High coupling: Response is on-topic, within scope
        Low coupling: Response drifted from purpose

        Returns:
            Coupling score 0.0-1.0 (fidelity)
        """
        if not hasattr(self.steward, 'attractor_center') or self.steward.attractor_center is None:
            return 1.0  # Can't measure without attractor

        # Embed response (encode expects a string, not a list)
        response_embedding = self.embeddings.encode(response)

        # Calculate distance to attractor
        attractor_center = self.steward.attractor_center
        distance = float(np.linalg.norm(response_embedding - attractor_center))

        # Convert to fidelity (0-1, higher is better)
        # Using distance_scale from steward if available
        distance_scale = getattr(self.steward, 'distance_scale', 2.0)
        fidelity = max(0.0, min(1.0, 1.0 - (distance / distance_scale)))

        return fidelity

    def _regenerate_entrained(
        self,
        user_input: str,
        conversation_context: List[Dict[str, str]],
        drifted_response: str
    ) -> str:
        """
        Regenerate response with entrainment enforcement.

        When coupling check fails, delegate to ProportionalController for
        graduated intervention based on drift severity.

        This ensures single source of truth for intervention logic.
        """
        # Delegate to ProportionalController if available
        # DEBUG: Log proportional_controller availability
        has_pc = hasattr(self.steward, 'proportional_controller')
        pc_value = getattr(self.steward, 'proportional_controller', None) if has_pc else None
        print(f"🔍 DEBUG _regenerate_entrained:")
        print(f"   has proportional_controller attribute: {has_pc}")
        print(f"   proportional_controller value: {pc_value}")
        print(f"   steward type: {type(self.steward)}")
        print(f"   steward attributes: {[attr for attr in dir(self.steward) if 'controller' in attr.lower()]}")

        if has_pc and pc_value:
            print(f"✅ Delegating to ProportionalController")
            return self._regenerate_via_proportional_controller(
                user_input,
                conversation_context,
                drifted_response
            )

        # Fallback: Use basic regeneration if ProportionalController not available
        print(f"⚠️ Using fallback regeneration (ProportionalController not available)")

        # MATHEMATICALLY RIGOROUS FALLBACK: Apply strength-scaled prompts directly
        # Measure fidelity to calculate intervention strength
        fidelity = self._measure_coupling(drifted_response)
        error_signal = 1.0 - fidelity

        # Calculate strength using proportional control law: F = K·e_t
        # Using same formula as ProportionalController
        constraint_rigidity = 0.98  # For BETA with τ=0.02
        K_attractor = 1.5  # Proportional gain
        strength = min(constraint_rigidity * error_signal * K_attractor, 1.0)

        attractor = self.steward.attractor

        # Extract attractor text for prompts
        purpose_text = ', '.join(attractor.purpose) if hasattr(attractor, 'purpose') and attractor.purpose else "the session's stated purpose"
        scope_text = ', '.join(attractor.scope) if hasattr(attractor, 'scope') and attractor.scope else "the session's defined scope"

        # STRENGTH-SCALED PROMPTS (following proportional control law)
        if strength >= 0.85:
            # SEVERE DRIFT: Forceful intervention with MUST/DECLINE language
            corrective_msg = {
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
                    f"Session Purpose: {purpose_text}\n"
                    f"Session Scope: {scope_text}\n\n"
                    "Generate a response that STRICTLY adheres to the session constraints or politely declines. "
                    "Be direct and firm in maintaining boundaries."
                )
            }
        elif strength >= 0.65:
            # MODERATE DRIFT: Firm correction with redirection
            corrective_msg = {
                "role": "system",
                "content": (
                    "GOVERNANCE CORRECTION:\n\n"
                    "The previous response drifted significantly from the session's purpose and scope.\n\n"
                    f"Session Purpose: {purpose_text}\n"
                    f"Session Scope: {scope_text}\n\n"
                    "Regenerate a response that stays firmly within the established boundaries. "
                    "If the user's request is off-topic, acknowledge it briefly but redirect to the session's focus. "
                    "Do not spend time elaborating on off-topic content."
                )
            }
        elif strength >= 0.40:
            # MILD-MODERATE DRIFT: Clear guidance
            corrective_msg = {
                "role": "system",
                "content": (
                    f"The previous response drifted from the session purpose and scope. "
                    f"Please regenerate to stay focused on: {purpose_text} within scope: {scope_text}"
                )
            }
        else:
            # VERY MILD DRIFT: Gentle reminder
            corrective_msg = {
                "role": "system",
                "content": f"Please stay focused on the session's stated purpose ({purpose_text}) and scope ({scope_text})."
            }

        # Build regeneration messages
        regen_messages = conversation_context.copy()
        regen_messages.append(corrective_msg)

        # Regenerate with strength-scaled prompt
        try:
            regenerated = self._call_llm(regen_messages, user_input)
            print(f"✅ Regeneration complete (strength={strength:.2f}, fidelity={fidelity:.3f})")
            return regenerated
        except Exception as e:
            # Fallback to original if regeneration fails
            print(f"⚠️ Regeneration failed: {e}")
            return drifted_response

    def _regenerate_via_proportional_controller(
        self,
        user_input: str,
        conversation_context: List[Dict[str, str]],
        drifted_response: str
    ) -> str:
        """
        Bridge to ProportionalController for graduated interventions.

        Converts fidelity-based coupling check into error_signal format
        that ProportionalController expects, then uses its regeneration logic.
        """
        import numpy as np

        # Measure fidelity (coupling)
        fidelity = self._measure_coupling(drifted_response)

        # Convert fidelity to error_signal (ProportionalController expects error, not fidelity)
        # error_signal = 1 - fidelity (high error when low fidelity)
        error_signal = 1.0 - fidelity

        pc = self.steward.proportional_controller

        # Use ProportionalController's graduated logic
        # State 3 (INTERVENE): error >= epsilon_max → Strong regeneration
        # State 2 (CORRECT): error >= epsilon_min → Context injection
        # State 1 (MONITOR): error < epsilon_min → No intervention

        if error_signal >= pc.epsilon_max:
            # SEVERE DRIFT: Use ProportionalController's regeneration
            print(f"🚨 SEVERE DRIFT detected (error={error_signal:.3f}, threshold={pc.epsilon_max:.3f})")
            print(f"   Applying ProportionalController REGENERATION intervention")

            # Call ProportionalController's regeneration method directly
            intervention_record = pc._apply_regeneration(
                response_text=drifted_response,
                conversation_history=conversation_context,
                error_signal=error_signal
            )
            return intervention_record.modified_response

        elif error_signal >= pc.epsilon_min:
            # MODERATE DRIFT: Use context injection (lighter intervention)
            print(f"⚠️  MODERATE DRIFT detected (error={error_signal:.3f}, threshold={pc.epsilon_min:.3f})")
            print(f"   Applying ProportionalController REMINDER intervention")

            # Call ProportionalController's reminder method
            intervention_record = pc._apply_reminder(
                response_text=drifted_response,
                error_signal=error_signal
            )
            return intervention_record.modified_response
        else:
            # MILD DRIFT: Fallback to basic regeneration
            print(f"ℹ️  MILD DRIFT detected (error={error_signal:.3f}, below threshold={pc.epsilon_min:.3f})")
            print(f"   Using fallback regeneration")
            return drifted_response

    def _call_llm(
        self,
        conversation_context: List[Dict[str, str]],
        user_input: str
    ) -> str:
        """
        Call underlying LLM client.

        Handles different LLM client interfaces.
        """
        # Add user input to messages
        messages = conversation_context.copy()
        messages.append({"role": "user", "content": user_input})

        # Call LLM
        try:
            # EXPLICIT LOGGING: Track what LLM is actually being called
            print("\n" + "="*60)
            print("🔍 CALLING LLM API")
            print("="*60)

            # Log LLM client info
            llm_type = type(self.llm).__name__
            print(f"  Client Type: {llm_type}")

            # Log model if available
            if hasattr(self.llm, 'model'):
                print(f"  Model: {self.llm.model}")

            # Log API key (first 10 chars only for security)
            if hasattr(self.llm, 'api_key') and self.llm.api_key:
                key_preview = self.llm.api_key[:10] + "..." if len(self.llm.api_key) > 10 else self.llm.api_key
                print(f"  API Key: {key_preview}")

            # Log endpoint if available
            if hasattr(self.llm, 'client') and hasattr(self.llm.client, '_client') and hasattr(self.llm.client._client, 'base_url'):
                print(f"  Endpoint: {self.llm.client._client.base_url}")

            print(f"  Messages: {len(messages)} messages in context")
            print(f"  Max Tokens: 16000")
            print(f"  Temperature: 0.7")
            print("="*60 + "\n")

            response = self.llm.generate(messages=messages, max_tokens=16000, temperature=0.7)

            # Log successful response
            print(f"✅ LLM Response Received ({len(response)} chars)\n")

            return response
        except Exception as e:
            print(f"⚠️ LLM generation error: {e}")
            return "[Error generating response]"

    def get_intervention_statistics(self) -> Dict[str, Any]:
        """Get statistics on interventions for evidence/reporting."""
        total = len(self.interventions)
        by_type = {}

        for intervention in self.interventions:
            itype = intervention.intervention_type
            by_type[itype] = by_type.get(itype, 0) + 1

        # Calculate improvement metrics
        regenerations = [i for i in self.interventions if i.intervention_type in ["regeneration", "both"]]
        if regenerations:
            improvements = [
                i.fidelity_governed - i.fidelity_original
                for i in regenerations
                if i.fidelity_original is not None and i.fidelity_governed is not None
            ]
            avg_improvement = sum(improvements) / len(improvements) if improvements else 0.0
        else:
            avg_improvement = 0.0

        return {
            "total_interventions": total,
            "by_type": by_type,
            "regeneration_count": by_type.get("regeneration", 0) + by_type.get("both", 0),
            "salience_injection_count": by_type.get("salience_injection", 0) + by_type.get("both", 0),
            "avg_fidelity_improvement": avg_improvement,
            "coupling_threshold": self.coupling_threshold,
            "salience_threshold": self.salience_threshold
        }

    def export_interventions(self) -> List[Dict[str, Any]]:
        """Export intervention log for evidence."""
        return [
            {
                "turn_number": i.turn_number,
                "intervention_type": i.intervention_type,
                "original_response": i.original_response,
                "governed_response": i.governed_response,
                "fidelity_original": i.fidelity_original,
                "fidelity_governed": i.fidelity_governed,
                "salience_before": i.salience_before,
                "salience_after": i.salience_after,
                "timestamp": i.timestamp
            }
            for i in self.interventions
        ]
