"""
Beta Steward LLM Service for TELOS Observatory V3 BETA Mode.
Provides context-aware guidance specifically for BETA testers.

Key differences from regular Steward:
- PA is PRE-ESTABLISHED during onboarding (not discovered during conversation)
- Three-metric system: F_user, F_AI, PS (not single fidelity)
- Metrics are meaningful from Turn 1
- User has already completed PA onboarding

RAG Escalation:
- When users ask about TELOS concepts, governance, or how things work
- Corpus is retrieved from human-readable knowledge base
- Provides plain-language explanations without technical jargon
"""

import streamlit as st
from mistralai import Mistral
from typing import Generator, Optional, List
import os
import re
import logging

logger = logging.getLogger(__name__)

# Keywords that trigger RAG escalation - questions about TELOS/governance
RAG_TRIGGER_PATTERNS = [
    r'\bwhat is telos\b',
    r'\bhow does (this|telos|it) work\b',
    r'\bwhat does .* mean\b',
    r'\bexplain .*(fidelity|primacy|attractor|governance|steward)\b',
    r'\bwhy (did|does|is) .*(intervene|step in|block)\b',
    r'\bwhat.* (fidelity|primacy state|ps|f_user|f_ai)\b',
    r'\bhelp me understand\b',
    r'\bi don\'?t (understand|get)\b',
    r'\bwhat are (you|the metrics)\b',
    r'\bhow (are|do) you (measure|calculate|determine)\b',
    r'\bpurpose (of|behind)\b',
    r'\btell me (about|more)\b',
]


class BetaStewardLLM:
    """Service for Beta Steward AI assistant - specialized for BETA mode."""

    def __init__(self):
        """Initialize Beta Steward LLM with API key and RAG corpus."""
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

        # RAG components - initialized lazily on first use
        self._corpus_loader = None
        self._corpus_loaded = False

    def _initialize_corpus(self) -> bool:
        """
        Lazily initialize the RAG corpus loader.

        Returns:
            True if corpus is ready, False on error
        """
        if self._corpus_loaded:
            return True

        try:
            from demo_mode.telos_corpus_loader import TELOSCorpusLoader, format_context_for_llm
            from telos_purpose.core.embedding_provider import SentenceTransformerProvider

            # Create embedding provider (SentenceTransformer for local inference)
            embedding_provider = SentenceTransformerProvider()

            # Initialize and load corpus
            self._corpus_loader = TELOSCorpusLoader(embedding_provider)
            num_chunks = self._corpus_loader.load_corpus()

            logger.info(f"RAG corpus loaded with {num_chunks} chunks")
            self._corpus_loaded = True
            return True

        except Exception as e:
            logger.warning(f"Failed to initialize RAG corpus: {e}")
            logger.warning("Steward will continue without RAG - using base knowledge only")
            return False

    def _needs_rag_escalation(self, user_message: str) -> bool:
        """
        Detect if the user's message needs RAG escalation.

        Triggers on questions about TELOS, governance, metrics, etc.

        Args:
            user_message: The user's question

        Returns:
            True if RAG should be used
        """
        message_lower = user_message.lower()

        # Check against trigger patterns
        for pattern in RAG_TRIGGER_PATTERNS:
            if re.search(pattern, message_lower):
                logger.info(f"RAG escalation triggered by pattern: {pattern}")
                return True

        return False

    def _retrieve_rag_context(self, user_message: str, top_k: int = 3) -> str:
        """
        Retrieve relevant corpus context for the user's question.

        Args:
            user_message: The user's question
            top_k: Number of chunks to retrieve

        Returns:
            Formatted context string, or empty string on error
        """
        if not self._corpus_loaded and not self._initialize_corpus():
            return ""

        try:
            from demo_mode.telos_corpus_loader import format_context_for_llm

            chunks = self._corpus_loader.retrieve(user_message, top_k=top_k)
            if not chunks:
                logger.info("No relevant chunks found for query")
                return ""

            context = format_context_for_llm(chunks)
            logger.info(f"Retrieved {len(chunks)} chunks for RAG context")
            return context

        except Exception as e:
            logger.warning(f"RAG retrieval error: {e}")
            return ""

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

## Fidelity Zone Thresholds (IMPORTANT - ALWAYS USE PERCENTAGES)

When discussing fidelity scores, ALWAYS use PERCENTAGES (not decimals):
- **GREEN (Aligned)**: 70% and above (e.g., 70%, 74%, 85%, 100%)
- **YELLOW (Minor Drift)**: 60% to 69% (e.g., 62%, 65%, 68%)
- **ORANGE (Drift Detected)**: 50% to 59% (e.g., 52%, 55%, 58%)
- **RED (Significant Drift)**: Below 50% (e.g., 35%, 42%, 48%)

Example: If F_user = 74%, say "You're in the GREEN zone (aligned)" NOT yellow.
Example: If F_user = 65%, say "You're in the YELLOW zone (minor drift)".
Example: Say "84% alignment" NOT "0.84 alignment".

## Your Role in BETA

Be direct and helpful. When users ask about metrics:
- Explain what each number means
- Help them understand if they're staying aligned with their purpose
- Point out if their questions are drifting from their stated goals
- Never give generic "Turn 0 calibration" explanations - PA is already set!
- ALWAYS use the correct fidelity zone thresholds above when explaining scores

Be concise like the example Steward responses - no fluff, just clear explanations.

## CRITICAL: Steward Conversations Are NOT Session Turns

Questions asked to YOU (Steward) are NOT recorded as turns in the main session.
Only messages in the main conversation window compute actual fidelity metrics.

**When users ask you off-topic questions (like "What's the best pizza topping?"):**
- DO NOT fabricate specific metric values (no "F_user: 0.00" or exact numbers)
- Instead, use QUALITATIVE language: "That's clearly outside your stated purpose"
- Remind them: "If you asked that in the main session, it would likely trigger drift detection"
- Suggest: "Try it in the main conversation if you want to see the actual metrics"

**Why this matters:**
- You cannot compute fidelity - only the main session engine does that
- Making up numbers undermines trust in the real metrics
- Users testing the system need to understand what's calculated vs. explained

**Good response to off-topic question:**
"That's completely unrelated to your stated purpose of [X]. If you asked that in the main session window, it would almost certainly register as significant drift. The TELOSCOPE would show you exactly how far off-topic it is. Want to test it there and see?"

**Bad response (DO NOT DO THIS):**
"F_user: 0.00 (RED zone), F_AI: ---, PS: ---" ← Never fabricate specific values"""

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

            # Show the user's last input for context
            if 'last_user_input' in context:
                truncated_input = context['last_user_input'][:150] + ('...' if len(context['last_user_input']) > 150 else '')
                context_str += f"- Last User Input: \"{truncated_input}\"\n"

            # Show intervention status
            if 'intervention_triggered' in context:
                if context['intervention_triggered']:
                    context_str += f"- Intervention: TRIGGERED - {context.get('intervention_reason', 'drift detected')}\n"
                else:
                    context_str += f"- Intervention: Not triggered (aligned)\n"

            # Alignment status interpretation (using zone names, not threshold numbers)
            # Import zone thresholds from central config
            try:
                from config.colors import _ZONE_ALIGNED, _ZONE_MINOR_DRIFT, get_zone_name
            except ImportError:
                # Fallback thresholds if config not available
                _ZONE_ALIGNED = 0.70
                _ZONE_MINOR_DRIFT = 0.60

            if 'f_user' in context and context['f_user'] is not None:
                f_user = context['f_user']
                if f_user >= _ZONE_ALIGNED:
                    context_str += "- User Alignment: ALIGNED (staying on purpose)\n"
                elif f_user >= _ZONE_MINOR_DRIFT:
                    context_str += "- User Alignment: MINOR DRIFT (slight deviation from purpose)\n"
                else:
                    context_str += "- User Alignment: DRIFTING (significantly off-purpose)\n"

            if 'f_ai' in context and context['f_ai'] is not None:
                f_ai = context['f_ai']
                if f_ai >= _ZONE_ALIGNED:
                    context_str += "- AI Alignment: ALIGNED (serving user's purpose)\n"
                elif f_ai >= _ZONE_MINOR_DRIFT:
                    context_str += "- AI Alignment: MINOR DRIFT (response slightly off-purpose)\n"
                else:
                    context_str += "- AI Alignment: DRIFTING (response not serving purpose)\n"

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

            # Check for RAG escalation
            rag_context = ""
            if self._needs_rag_escalation(user_message):
                logger.info("RAG escalation triggered - retrieving corpus context")
                rag_context = self._retrieve_rag_context(user_message)
                if rag_context:
                    # Add RAG context to system prompt
                    system_prompt += f"\n\n# Retrieved Knowledge\n\n{rag_context}\n\nUse this knowledge to provide accurate, human-friendly explanations. Speak naturally - don't quote the knowledge base verbatim."

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

            # Check for RAG escalation
            if self._needs_rag_escalation(user_message):
                logger.info("RAG escalation triggered (streaming) - retrieving corpus context")
                rag_context = self._retrieve_rag_context(user_message)
                if rag_context:
                    # Add RAG context to system prompt
                    system_prompt += f"\n\n# Retrieved Knowledge\n\n{rag_context}\n\nUse this knowledge to provide accurate, human-friendly explanations. Speak naturally - don't quote the knowledge base verbatim."

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
