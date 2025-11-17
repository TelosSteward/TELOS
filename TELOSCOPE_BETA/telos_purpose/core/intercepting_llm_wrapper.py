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

        # Embed context
        context_embedding = self.embeddings.encode([context_text])[0]

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

        # Embed response
        response_embedding = self.embeddings.encode([response])[0]

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

        When coupling check fails, regenerate with explicit governance prompt.

        This CORRECTS drift by creating entrained alternative.
        """
        attractor = self.steward.attractor

        # Build corrective prompt as enhanced user input
        corrective_parts = [
            "[Please answer the following question while staying within the session's governance boundaries]",
            ""
        ]

        if hasattr(attractor, 'purpose') and attractor.purpose:
            corrective_parts.append(f"Purpose: {', '.join(attractor.purpose)}")

        if hasattr(attractor, 'scope') and attractor.scope:
            corrective_parts.append(f"Scope: {', '.join(attractor.scope)}")

        if hasattr(attractor, 'boundaries') and attractor.boundaries:
            corrective_parts.append(f"Boundaries: {'; '.join(attractor.boundaries)}")

        corrective_parts.append(f"\nQuestion: {user_input}")

        corrective_text = "\n".join(corrective_parts)

        # Build regeneration messages with corrective guidance prepended
        # Inject governance context at the beginning (safe position)
        governance_msg = {
            "role": "user",
            "content": f"[Governance Context]\n{corrective_parts[2]}\n{corrective_parts[3]}\n{corrective_parts[4]}"
        }

        regen_messages = [governance_msg] + conversation_context.copy()

        # Regenerate
        try:
            regenerated = self._call_llm(regen_messages, user_input)
            return regenerated
        except Exception as e:
            # Fallback to original if regeneration fails
            print(f"⚠️ Regeneration failed: {e}")
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
            print(f"  Max Tokens: 500")
            print(f"  Temperature: 0.7")
            print("="*60 + "\n")

            response = self.llm.generate(messages=messages, max_tokens=500, temperature=0.7)

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
