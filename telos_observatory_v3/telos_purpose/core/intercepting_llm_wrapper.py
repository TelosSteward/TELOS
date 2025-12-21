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
        coupling_threshold: float = 0.76  # Goldilocks: Aligned threshold
    ):
        """
        Initialize intercepting wrapper.

        Args:
            llm_client: Underlying LLM client (Mistral, etc.)
            embedding_provider: For measuring salience and coupling
            steward_ref: Reference to parent steward for attractor access
            salience_threshold: Trigger injection if salience drops below this
            coupling_threshold: Trigger regeneration if fidelity drops below this (Goldilocks: 0.76)
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

        # GOLDILOCKS ZONE THRESHOLDS (aligned with beta_response_manager.py):
        # GREEN >= 0.70: Aligned (no intervention)
        # YELLOW 0.60-0.70: Minor Drift (soft intervention)
        # ORANGE 0.50-0.60: Drift Detected (moderate intervention)
        # RED < 0.50: Significant Drift (strong intervention / redirect)
        _ZONE_GREEN = 0.70  # Threshold for "Aligned" - no intervention above this
        _ZONE_YELLOW = 0.60  # Threshold for "Minor Drift"
        _ZONE_ORANGE = 0.50  # Threshold for "Drift Detected"
        # RED zone: < 0.50 = Significant Drift - stock redirect, no API call needed
        if user_input_fidelity < _ZONE_ORANGE:  # RED zone - too far from purpose
            # HARD REDIRECT: User request is completely outside the purpose basin
            # No API call needed - just redirect
            redirect_message = (
                f"This question is quite far from your session's purpose. "
                f"Your session is focused on: {', '.join(self.steward.attractor.purpose)}. "
                f"How can I help you with that instead?"
            )

            # Log pre-generation redirect
            self.interventions.append(GovernanceIntervention(
                turn_number=self.turn_count,
                intervention_type="pre_generation_redirect",
                original_response=None,
                governed_response=redirect_message,
                fidelity_original=None,
                fidelity_governed=user_input_fidelity,
                salience_before=salience_before,
                salience_after=salience_after,
                timestamp=time.time()
            ))

            return redirect_message

        # NON-GREEN ZONE (< 0.70): Detected pre-generation
        # Apply proportional controller intervention PROACTIVELY during generation
        # Any fidelity < GREEN threshold gets TELOS intervention
        needs_intervention = user_input_fidelity < _ZONE_GREEN  # < 0.70

        # Step 2: Generate with governed context
        if needs_intervention:
            # NON-GREEN ZONE INTERVENTION: Add purpose-constraining context to generation
            # This is where TELOS proportional controller actually runs with API call
            logger.warning(f"🟠 ORANGE ZONE: Applying proactive proportional controller intervention")

            # Calculate intervention strength based on user input fidelity
            error_signal = 1.0 - user_input_fidelity
            constraint_rigidity = 0.98  # For BETA with τ=0.02
            K_attractor = 1.5  # Proportional gain
            strength = min(constraint_rigidity * error_signal * K_attractor, 1.0)

            attractor = self.steward.attractor
            purpose_text = ', '.join(attractor.purpose) if hasattr(attractor, 'purpose') and attractor.purpose else "the session's stated purpose"
            scope_text = ', '.join(attractor.scope) if hasattr(attractor, 'scope') and attractor.scope else "the session's defined scope"

            # OFF-TOPIC HANDLING - Natural assistant language
            # Core behavior: acknowledge, offer choice, don't engage with off-topic content yet
            if strength >= 0.65:
                # Significant drift: Clear but natural acknowledgment
                purpose_prompt = {
                    "role": "system",
                    "content": (
                        f"SESSION PURPOSE: {purpose_text}\n"
                        f"SESSION SCOPE: {scope_text}\n\n"
                        "The user's question is outside this session's focus. Respond naturally:\n"
                        "- Briefly note this is a different topic\n"
                        "- Ask if they'd like to switch focus or continue with the original purpose\n"
                        "- Don't answer the off-topic question yet - wait for their choice\n\n"
                        "Sound like a helpful assistant, not a therapist. Keep it short (2-3 sentences)."
                    )
                }
            else:
                # Mild drift: Light touch
                purpose_prompt = {
                    "role": "system",
                    "content": (
                        f"SESSION PURPOSE: {purpose_text}\n"
                        f"SESSION SCOPE: {scope_text}\n\n"
                        "The user's question is slightly outside this session's focus. "
                        "Briefly check if they want to shift topics or stay with the original purpose. "
                        "Keep it natural and short."
                    )
                }

            # Build context with purpose injection
            # IMPORTANT: Insert system message at START (Mistral API requires system messages first)
            governed_context = conversation_context.copy()
            governed_context.insert(0, purpose_prompt)

            logger.warning(f"🟠 Generating with purpose context (strength={strength:.2f})")
            response = self._call_llm(governed_context, user_input)
            intervention_type = "proactive_orange_zone"
        else:
            # GREEN/YELLOW ZONE: Normal generation without intervention
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
        NOTE: This only provides context - the proportional controller handles
        the actual intervention logic based on mathematical error signals.
        """
        attractor = self.steward.attractor

        # Format attractor as context reminder (not instructions)
        reinforcement_parts = ["[Session Context]"]

        if hasattr(attractor, 'purpose') and attractor.purpose:
            purposes = ", ".join(attractor.purpose)
            reinforcement_parts.append(f"Purpose: {purposes}")

        if hasattr(attractor, 'scope') and attractor.scope:
            scope_items = ", ".join(attractor.scope)
            reinforcement_parts.append(f"Scope: {scope_items}")

        if hasattr(attractor, 'boundaries') and attractor.boundaries:
            boundaries = "; ".join(attractor.boundaries)
            reinforcement_parts.append(f"Boundaries: {boundaries}")

        reinforcement_text = "\n".join(reinforcement_parts)

        if len(reinforcement_parts) == 1:
            # Fallback if no textual attractor
            reinforcement_text = "[Session Context]\nPlease stay focused on the session's established purpose and scope."

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
        Measure response coupling to attractor using raw cosine similarity.

        Coupling = how well response stays aligned with attractor trajectory.

        High coupling: Response is on-topic, within scope
        Low coupling: Response drifted from purpose

        Mathematical basis:
        - Uses RAW cosine similarity (matching primacy_state.py)
        - Fidelity = cosine_similarity(response, attractor_center)
        - TELOS primacy math handles basin/tolerance thresholds

        Returns:
            Coupling score (raw cosine similarity)
        """
        if not hasattr(self.steward, 'attractor_center') or self.steward.attractor_center is None:
            return 1.0  # Can't measure without attractor

        # Embed response (encode expects a string, not a list)
        response_embedding = self.embeddings.encode(response)

        # Calculate cosine similarity to attractor
        attractor_center = self.steward.attractor_center

        # Cosine similarity: dot(a,b) / (||a|| * ||b||)
        norm_response = np.linalg.norm(response_embedding)
        norm_attractor = np.linalg.norm(attractor_center)

        if norm_response == 0 or norm_attractor == 0:
            return 0.5  # Neutral if can't compute

        # RAW cosine similarity - no normalization
        # This matches primacy_state.py approach
        fidelity = float(np.dot(response_embedding, attractor_center) /
                         (norm_response * norm_attractor))

        # Log for debugging
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"_measure_coupling: fidelity={fidelity:.3f} (raw cosine similarity)")

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

        # STRENGTH-SCALED PROMPTS - Natural assistant language
        # Core behavior: acknowledge off-topic, offer choice, don't engage with off-topic yet
        # Avoid: therapy-speak, "I notice...", scripted phrases
        if strength >= 0.85:
            # SIGNIFICANT DRIFT: Clear acknowledgment, offer choice
            corrective_msg = {
                "role": "system",
                "content": (
                    f"SESSION PURPOSE: {purpose_text}\n"
                    f"SESSION SCOPE: {scope_text}\n\n"
                    "The user's question is outside this session's focus. Respond naturally:\n"
                    "- Briefly note this is a different topic\n"
                    "- Ask if they'd like to switch focus or continue with the original purpose\n"
                    "- Don't answer the off-topic question yet - wait for their choice\n\n"
                    "Sound like a helpful assistant, not a therapist. Keep it short (2-3 sentences)."
                )
            }
        elif strength >= 0.65:
            # MODERATE DRIFT: Light touch, offer choice
            corrective_msg = {
                "role": "system",
                "content": (
                    f"Session context - Purpose: {purpose_text}, Scope: {scope_text}\n\n"
                    "The user's question is somewhat outside the session focus. "
                    "Briefly check if they want to shift topics or stay with the original purpose. "
                    "Keep it natural and short - don't engage with the off-topic content yet."
                )
            }
        elif strength >= 0.40:
            # MILD DRIFT: Very light touch
            corrective_msg = {
                "role": "system",
                "content": (
                    f"Session context: {purpose_text} (scope: {scope_text})\n\n"
                    "The user is drifting slightly. You can briefly check if they want to explore "
                    "this direction or stay with the original focus. Keep response on-topic."
                )
            }
        else:
            # VERY MILD: Just context awareness
            corrective_msg = {
                "role": "system",
                "content": f"Session context: {purpose_text} (scope: {scope_text}). Respond naturally."
            }

        # Build regeneration messages
        regen_messages = conversation_context.copy()
        regen_messages.append(corrective_msg)

        # Regenerate with strength-scaled prompt
        try:
            regenerated = self._call_llm(regen_messages, user_input)
            # Check if _call_llm returned an error string instead of raising
            if regenerated and "[Error" in regenerated:
                print(f"⚠️ Regeneration returned error, falling back to original response")
                return drifted_response
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
            try:
                intervention_record = pc._apply_regeneration(
                    response_text=drifted_response,
                    conversation_history=conversation_context,
                    error_signal=error_signal
                )
                result = intervention_record.modified_response
                # Check if result contains error string
                if result and "[Error" in result:
                    print(f"⚠️ ProportionalController regeneration returned error, falling back")
                    return drifted_response
                return result
            except Exception as e:
                print(f"⚠️ ProportionalController regeneration failed: {e}, falling back")
                return drifted_response

        elif error_signal >= pc.epsilon_min:
            # MODERATE DRIFT: Use context injection (lighter intervention)
            print(f"⚠️  MODERATE DRIFT detected (error={error_signal:.3f}, threshold={pc.epsilon_min:.3f})")
            print(f"   Applying ProportionalController REMINDER intervention")

            # Call ProportionalController's reminder method
            try:
                intervention_record = pc._apply_reminder(
                    response_text=drifted_response,
                    error_signal=error_signal
                )
                result = intervention_record.modified_response
                # Check if result contains error string
                if result and "[Error" in result:
                    print(f"⚠️ ProportionalController reminder returned error, falling back")
                    return drifted_response
                return result
            except Exception as e:
                print(f"⚠️ ProportionalController reminder failed: {e}, falling back")
                return drifted_response
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
        # IMMEDIATE DEBUG: Confirm method entry
        import sys
        print(f"\n🔵 _call_llm() ENTERED", flush=True)
        print(f"   conversation_context type: {type(conversation_context)}", flush=True)
        print(f"   conversation_context len: {len(conversation_context) if conversation_context else 'None'}", flush=True)
        print(f"   user_input: {user_input[:50] if user_input else 'None'}...", flush=True)
        print(f"   self.llm type: {type(self.llm)}", flush=True)
        print(f"   self.llm is None: {self.llm is None}", flush=True)
        sys.stdout.flush()

        # Build messages with conciseness system prompt
        messages = []

        # Add conciseness system prompt at start (unless there's already a system message)
        has_system = any(msg.get('role') == 'system' for msg in conversation_context)
        if not has_system:
            conciseness_prompt = {
                "role": "system",
                "content": (
                    "RESPONSE GUIDELINES:\n"
                    "- Keep responses focused and concise (2-4 paragraphs typical)\n"
                    "- Address what the user asked directly\n"
                    "- Do not use excessive formatting, bullet points, or headers unless truly needed\n"
                    "- Avoid verbose explanations or unnecessary elaboration\n"
                    "- Be helpful but brief"
                )
            }
            messages.append(conciseness_prompt)

        # Add conversation context
        messages.extend(conversation_context)

        # Add user input
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
            print(f"  Max Tokens: 2000")
            print(f"  Temperature: 0.7")
            print("="*60 + "\n")

            response = self.llm.generate(messages=messages, max_tokens=2000, temperature=0.7)

            # Log successful response
            print(f"✅ LLM Response Received ({len(response)} chars)\n")

            return response
        except Exception as e:
            import traceback
            print(f"⚠️ LLM generation error: {e}")
            print(f"   Full traceback:\n{traceback.format_exc()}")
            # Log additional debugging info
            print(f"   LLM client type: {type(self.llm).__name__}")
            print(f"   Messages count: {len(messages)}")
            if messages:
                print(f"   First message role: {messages[0].get('role', 'N/A')}")
                print(f"   Last message role: {messages[-1].get('role', 'N/A')}")
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
