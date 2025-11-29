"""
Beta Steward LLM Service for TELOS Observatory V3 BETA Mode.
Provides context-aware guidance specifically for BETA testers.

Key differences from regular Steward:
- PA is PRE-ESTABLISHED during onboarding (not discovered during conversation)
- Three-metric system: F_user, F_AI, PS (not single fidelity)
- Metrics are meaningful from Turn 1
- User has already completed PA onboarding
"""

import streamlit as st
from mistralai import Mistral
from typing import Generator, Optional
import os


class BetaStewardLLM:
    """Service for Beta Steward AI assistant - specialized for BETA mode."""

    def __init__(self):
        """Initialize Beta Steward LLM with API key."""
        try:
            self.api_key = st.secrets.get("MISTRAL_API_KEY")
        except (FileNotFoundError, KeyError):
            self.api_key = os.environ.get("MISTRAL_API_KEY")

        if not self.api_key:
            raise ValueError(
                "MISTRAL_API_KEY not found. Please add it to .streamlit/secrets.toml or set as environment variable."
            )

        self.client = Mistral(api_key=self.api_key)
        self.model = "mistral-large-latest"

    def _get_system_prompt(self, context: Optional[dict] = None) -> str:
        """Generate BETA-specific system prompt with current context.

        Args:
            context: Dict with BETA mode state (PA, metrics, turn info)

        Returns:
            str: System prompt optimized for BETA mode
        """
        base_prompt = """You are Steward, the TELOS BETA guide. You help users understand what they're seeing during their BETA test of the TELOS governance framework.

# BETA Mode Context

In BETA mode, the user has ALREADY established their Primacy Attractor (PA) during onboarding BEFORE the conversation started. This is different from the full TELOS experience where PA emerges during conversation.

## What the User Already Did
- Completed PA onboarding
- Defined their PURPOSE (what they want to achieve)
- Defined their SCOPE (boundaries of the conversation)
- Defined their SUCCESS CRITERIA (how they'll know it worked)
- Optionally set a STYLE preference

## The Three Calibration Cards

The user sees THREE metrics displayed as calibration cards:

1. **User Fidelity (F_user)** - How well the USER's inputs align with their own stated purpose
   - High score: User is staying on-topic with their own goals
   - Low score: User is drifting from what they said they wanted
   - This measures the USER, not the AI

2. **AI Fidelity (F_AI)** - How well the AI's response aligns with the user's stated purpose
   - High score: AI response serves the user's stated goals
   - Low score: AI response strayed from the user's purpose
   - This measures the AI's alignment to USER purpose

3. **Primacy State (PS)** - The harmonic mean of F_user and F_AI
   - Formula: PS = rho_PA * (2 * F_user * F_AI) / (F_user + F_AI)
   - Only HIGH when BOTH user and AI are aligned
   - If either score is low, PS will be low
   - This is the overall "governance health" score

## Key BETA Mode Facts

- PA is ALREADY established (not calibrating)
- Metrics are meaningful from Turn 1
- First turn may show "---" until calculation completes
- After Turn 1, all three values should display
- The three values must be mathematically consistent (PS depends on F_user and F_AI)

## Your Role in BETA

Be direct and helpful. When users ask about metrics:
- Explain what each number means
- Help them understand if they're staying aligned with their purpose
- Point out if their questions are drifting from their stated goals
- Never give generic "Turn 0 calibration" explanations - PA is already set!

Be concise like the example Steward responses - no fluff, just clear explanations."""

        # Add current BETA context if provided
        if context:
            context_str = "\n\n# Current BETA Session State\n\n"

            # User's established PA
            if 'primacy_attractor' in context:
                pa = context['primacy_attractor']
                context_str += "## User's Primacy Attractor (Already Established)\n"
                if pa.get('purpose'):
                    context_str += f"- Purpose: {pa['purpose']}\n"
                if pa.get('scope'):
                    context_str += f"- Scope: {pa['scope']}\n"
                if pa.get('success_criteria'):
                    context_str += f"- Success Criteria: {pa['success_criteria']}\n"
                if pa.get('style'):
                    context_str += f"- Style: {pa['style']}\n"
                context_str += "\n"

            # Current turn info
            if 'current_turn' in context:
                context_str += f"- Current Turn: {context['current_turn']}\n"

            # Current calibration card values
            if 'f_user' in context and context['f_user'] is not None:
                context_str += f"- User Fidelity (F_user): {context['f_user']:.3f}\n"
            if 'f_ai' in context and context['f_ai'] is not None:
                context_str += f"- AI Fidelity (F_AI): {context['f_ai']:.3f}\n"
            if 'primacy_state' in context and context['primacy_state'] is not None:
                context_str += f"- Primacy State (PS): {context['primacy_state']:.3f}\n"

            # Alignment status interpretation
            if 'f_user' in context and context['f_user'] is not None:
                f_user = context['f_user']
                if f_user >= 0.85:
                    context_str += "- User Alignment: GOOD (staying on purpose)\n"
                elif f_user >= 0.70:
                    context_str += "- User Alignment: DRIFTING (starting to stray from purpose)\n"
                else:
                    context_str += "- User Alignment: OFF-TRACK (significantly off-purpose)\n"

            if 'f_ai' in context and context['f_ai'] is not None:
                f_ai = context['f_ai']
                if f_ai >= 0.85:
                    context_str += "- AI Alignment: GOOD (serving user's purpose)\n"
                elif f_ai >= 0.70:
                    context_str += "- AI Alignment: DRIFTING (response straying from purpose)\n"
                else:
                    context_str += "- AI Alignment: OFF-TRACK (response not serving purpose)\n"

            base_prompt += context_str

        return base_prompt

    def get_response(self,
                     user_message: str,
                     conversation_history: list,
                     context: Optional[dict] = None) -> str:
        """Get a non-streaming response from Beta Steward.

        Args:
            user_message: The user's current message
            conversation_history: Previous messages
            context: BETA mode context (PA, metrics, etc.)

        Returns:
            str: Beta Steward's response
        """
        try:
            system_prompt = self._get_system_prompt(context)

            messages = [
                {'role': 'system', 'content': system_prompt}
            ]

            for msg in conversation_history:
                if msg['role'] in ['user', 'assistant']:
                    messages.append({
                        'role': msg['role'],
                        'content': msg['content']
                    })

            messages.append({
                'role': 'user',
                'content': user_message
            })

            response = self.client.chat.complete(
                model=self.model,
                messages=messages,
                max_tokens=8000
            )

            return response.choices[0].message.content

        except Exception as e:
            return f"I apologize, but I encountered an error: {str(e)}. Please try again."

    def get_streaming_response(self,
                               user_message: str,
                               conversation_history: list,
                               context: Optional[dict] = None) -> Generator[str, None, None]:
        """Get a streaming response from Beta Steward.

        Args:
            user_message: The user's current message
            conversation_history: Previous messages
            context: BETA mode context

        Yields:
            str: Chunks of response
        """
        try:
            system_prompt = self._get_system_prompt(context)

            messages = [
                {'role': 'system', 'content': system_prompt}
            ]

            for msg in conversation_history:
                if msg['role'] in ['user', 'assistant']:
                    messages.append({
                        'role': msg['role'],
                        'content': msg['content']
                    })

            messages.append({
                'role': 'user',
                'content': user_message
            })

            stream = self.client.chat.stream(
                model=self.model,
                messages=messages,
                max_tokens=8000
            )

            for chunk in stream:
                if chunk.data.choices[0].delta.content:
                    yield chunk.data.choices[0].delta.content

        except Exception as e:
            yield f"I apologize, but I encountered an error: {str(e)}. Please try again."
