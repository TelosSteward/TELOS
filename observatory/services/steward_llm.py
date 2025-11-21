"""
Steward LLM Service for TELOS Observatory V3.
Handles all LLM interactions for the Steward assistant.

Now includes 4-layer defense architecture for adversarial testing:
- Layer 1: System Prompt (immutable PA)
- Layer 2: Fidelity Measurement (real-time alignment check)
- Layer 3: RAG Corpus (policy knowledge base)
- Layer 4: Human Escalation (intervention queue)
"""

import streamlit as st
from mistralai import Mistral
from typing import Generator, Optional, Dict, Any
import os
import logging

# Import defense layers
from observatory.services.steward_defense import StewardDefenseLayers

logger = logging.getLogger(__name__)


class StewardLLM:
    """Service for Steward AI assistant powered by Mistral with defense layers."""

    def __init__(self, enable_defense: bool = True):
        """Initialize Steward LLM with API key and defense layers.

        Args:
            enable_defense: Whether to enable 4-layer defense system (default: True)
        """
        # Try to get API key from Streamlit secrets, fall back to environment variable
        try:
            self.api_key = st.secrets.get("MISTRAL_API_KEY")
        except (FileNotFoundError, KeyError):
            self.api_key = os.environ.get("MISTRAL_API_KEY")

        if not self.api_key:
            raise ValueError(
                "MISTRAL_API_KEY not found. Please add it to .streamlit/secrets.toml or set as environment variable."
            )

        self.client = Mistral(api_key=self.api_key)
        self.model = "mistral-small-latest"  # Using small - best availability

        # Initialize defense layers
        self.enable_defense = enable_defense
        if enable_defense:
            try:
                self.defense = StewardDefenseLayers(
                    fidelity_threshold=0.75,
                    escalation_threshold=0.60,
                    enable_telemetry=True
                )
                logger.info("✅ Defense layers enabled")
            except Exception as e:
                logger.warning(f"⚠️  Defense layers failed to initialize: {e}")
                self.enable_defense = False
        else:
            logger.info("ℹ️  Defense layers disabled")

    def _get_system_prompt(self, context: Optional[dict] = None) -> str:
        """Generate system prompt with TELOS framework knowledge and current context.

        Args:
            context: Optional dict with current screen state (active_tab, metrics, etc.)

        Returns:
            str: System prompt for Steward
        """
        base_prompt = """You are Steward, the TELOS Observatory guide. You help users understand the TELOS framework, navigate the Observatory interface, and answer questions about AI governance.

# About TELOS Framework

TELOS (Telemetric Localization of Semantic Intent) is a privacy-preserving AI governance framework that uses:

- **Primacy Attractor (PA)**: A mathematical representation of an AI's intended purpose extracted from the first 10 turns of conversation. It has three components:
  - Purpose: What the user wants to accomplish
  - Scope: Topics that are relevant
  - Boundaries: What to avoid
- **Fidelity**: Measures how well AI responses align with the Primacy Attractor on a 0.0-1.0 scale where:
  - 1.0 = Perfect alignment
  - 0.8-0.9 = Good alignment
  - Below 0.8 = Drifting (TELOS may intervene)
  - 0.0 = Complete misalignment
- **Governance Deltas**: Mathematical measurements of AI alignment, NOT conversation content
- **Drift**: When conversations wander from the user's original purpose
- **Interventions**: When TELOS detects drift (fidelity below 0.8) and corrects the AI's response

## Key Principles

1. **Privacy-First**: In full TELOS, only governance measurements (deltas) are logged, never conversation content. Beta stores full conversations for testing but will delete them after beta ends.
2. **Mathematical Governance**: Uses attractor dynamics and semantic field theory
3. **Continuous Alignment**: Real-time monitoring and correction of AI drift
4. **Transparent**: Users can observe the governance process in real-time
5. **User-Derived**: The PA is extracted from the user's behavior, not external rules

## Observatory Interface

The Observatory has three main tabs:

- **BETA**: Simple chat interface for testing TELOS-governed AI conversations
- **DEMO**: Demonstration mode (coming soon)
- **TELOS**: Full Observatory with visualization tools including:
  - Observation Deck: Real-time metrics and governance visualization
  - TELOSCOPE Controls: Advanced configuration and analysis tools

## Your Role

You help users by:
- Explaining TELOS concepts in accessible language
- Guiding them through the Observatory interface
- Answering questions about what they see on screen
- Providing context about governance metrics (Fidelity, PA Status, etc.)
- Explaining privacy protections and data handling

Be helpful, clear, and concise. Use analogies when helpful. Always prioritize user privacy and transparency."""

        # Add current context if provided
        if context:
            context_str = "\n\n# Current Screen Context\n\n"

            if 'active_tab' in context:
                context_str += f"- Active Tab: {context['active_tab']}\n"

            if 'current_turn' in context:
                context_str += f"- Current Turn: {context['current_turn']}\n"

            if 'fidelity' in context:
                context_str += f"- Latest Fidelity: {context['fidelity']:.3f}\n"

            if 'pa_status' in context:
                context_str += f"- Primacy Attractor Status: {context['pa_status']}\n"

            if 'total_turns' in context:
                context_str += f"- Total Turns: {context['total_turns']}\n"

            base_prompt += context_str

        return base_prompt

    def get_response(self,
                     user_message: str,
                     conversation_history: list,
                     context: Optional[dict] = None,
                     session_id: str = "unknown") -> Dict[str, Any]:
        """Get a non-streaming response from Steward with defense layer checking.

        Args:
            user_message: The user's current message
            conversation_history: List of previous messages in format [{'role': 'user'/'assistant', 'content': str}]
            context: Optional context about current screen state
            session_id: Session identifier for telemetry logging

        Returns:
            dict with keys:
                - response: The final response text (str)
                - defense_result: Defense layer check results (dict) if enabled
                - intervention_applied: Whether an intervention occurred (bool)
        """
        try:
            # Get system prompt with context
            system_prompt = self._get_system_prompt(context)

            # Build messages from history - Mistral format includes system in messages
            messages = [
                {
                    'role': 'system',
                    'content': system_prompt
                }
            ]

            # Add conversation history
            for msg in conversation_history:
                if msg['role'] in ['user', 'assistant']:
                    messages.append({
                        'role': msg['role'],
                        'content': msg['content']
                    })

            # Add current user message
            messages.append({
                'role': 'user',
                'content': user_message
            })

            # Call Mistral API (Layer 1: System Prompt active here)
            response = self.client.chat.complete(
                model=self.model,
                messages=messages,
                max_tokens=2048
            )

            proposed_response = response.choices[0].message.content

            # Check defense layers (Layers 2-4)
            if self.enable_defense:
                turn_number = len(conversation_history) // 2 + 1  # Estimate turn number

                defense_result = self.defense.check_defense_layers(
                    user_message=user_message,
                    steward_response=proposed_response,
                    turn_number=turn_number,
                    session_id=session_id
                )

                # Apply intervention if needed
                if defense_result.get("intervention_needed", False):
                    final_response = defense_result.get("modified_response", proposed_response)
                    intervention_applied = True
                else:
                    final_response = proposed_response
                    intervention_applied = False

                return {
                    "response": final_response,
                    "defense_result": defense_result,
                    "intervention_applied": intervention_applied
                }
            else:
                # Defense disabled, return original response
                return {
                    "response": proposed_response,
                    "defense_result": None,
                    "intervention_applied": False
                }

        except Exception as e:
            error_msg = f"I apologize, but I encountered an error: {str(e)}. Please try again."
            return {
                "response": error_msg,
                "defense_result": None,
                "intervention_applied": False,
                "error": str(e)
            }

    def get_streaming_response(self,
                               user_message: str,
                               conversation_history: list,
                               context: Optional[dict] = None) -> Generator[str, None, None]:
        """Get a streaming response from Steward.

        Args:
            user_message: The user's current message
            conversation_history: List of previous messages
            context: Optional context about current screen state

        Yields:
            str: Chunks of Steward's response as they arrive
        """
        try:
            # Get system prompt with context
            system_prompt = self._get_system_prompt(context)

            # Build messages - Mistral format includes system in messages
            messages = [
                {
                    'role': 'system',
                    'content': system_prompt
                }
            ]

            # Add conversation history
            for msg in conversation_history:
                if msg['role'] in ['user', 'assistant']:
                    messages.append({
                        'role': msg['role'],
                        'content': msg['content']
                    })

            # Add current user message
            messages.append({
                'role': 'user',
                'content': user_message
            })

            # Call Mistral API with streaming
            stream = self.client.chat.stream(
                model=self.model,
                messages=messages,
                max_tokens=2048
            )

            for chunk in stream:
                if chunk.data.choices[0].delta.content:
                    yield chunk.data.choices[0].delta.content

        except Exception as e:
            yield f"I apologize, but I encountered an error: {str(e)}. Please try again."
