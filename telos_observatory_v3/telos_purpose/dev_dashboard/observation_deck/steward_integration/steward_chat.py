"""
Steward Chat - Conversational Research Assistant (PAID)

Natural language Q&A interface for analyzing TELOS conversation sessions.
Uses existing steward_analysis.py with Mistral API (~$0.002 per query).

Components:
1. Text Input: Ask questions about current session
2. Response Display: AI-powered analysis from Steward
3. Scope Validation: Rejects out-of-scope queries with helpful redirects
4. Cost Warnings: Clear indication that this uses API tokens

Data Source: StewardAnalyzer (steward_analysis.py)

Purpose:
- Conversational interface for research questions
- AI interpretation of session patterns
- Natural language access to telemetry insights

Example Questions:
- "What governance patterns emerged in this session?"
- "How did primacy attractors evolve?"
- "What was the TELOS impact ratio?"
- "What canonical inputs were identified?"
"""

import streamlit as st
from typing import Dict, Any, Optional
from telos_purpose.dev_dashboard.steward_analysis import StewardAnalyzer


class StewardChat:
    """
    Renders conversational Q&A interface with Steward research assistant.

    Uses existing StewardAnalyzer for AI-powered session analysis.
    """

    def __init__(self, session_manager, mistral_client=None):
        """
        Initialize Steward Chat.

        Args:
            session_manager: WebSessionManager instance
            mistral_client: Optional MistralClient instance (reuses existing)
        """
        self.session_manager = session_manager

        # Initialize StewardAnalyzer (reuses existing Mistral client)
        self.analyzer = StewardAnalyzer(mistral_client=mistral_client)

    def render(self):
        """
        Render Steward chat interface with question input and responses.
        """
        st.markdown("### 🤖 Steward Research Assistant")
        st.markdown("*Conversational Session Analysis*")

        # Cost warning
        if self.analyzer.has_ai:
            st.info("💡 **Cost Notice**: Steward uses Mistral API (~$0.002 per question). "
                   "TELOSCOPIC Tools are FREE.")
        else:
            st.warning("⚠️ Steward AI features unavailable. Set MISTRAL_API_KEY to enable.")
            return

        # Chat history
        self._render_chat_history()

        # Question input
        self._render_question_input()

        # Example questions
        self._render_example_questions()

    def _render_chat_history(self):
        """Render chat history with Steward."""
        # Initialize chat history in session state
        if 'steward_chat_history' not in st.session_state:
            st.session_state['steward_chat_history'] = []

        # Display chat history
        for msg in st.session_state['steward_chat_history']:
            role = msg['role']
            content = msg['content']

            if role == 'user':
                st.markdown(f"**You:** {content}")
            else:
                st.markdown(f"**Steward:** {content}")
                st.markdown("---")

    def _render_question_input(self):
        """Render question input field."""
        # Text input for question
        question = st.text_input(
            "Ask a question about this session:",
            placeholder="e.g., What governance patterns emerged?",
            key="steward_question_input"
        )

        # Submit button
        if st.button("Ask Steward", key="steward_submit"):
            if question.strip():
                self._handle_question(question)
                st.rerun()

    def _handle_question(self, question: str):
        """
        Handle user question by querying StewardAnalyzer.

        Args:
            question: User's research question
        """
        # Add question to chat history
        st.session_state['steward_chat_history'].append({
            'role': 'user',
            'content': question
        })

        # Get session data
        session_data = self._get_session_data()

        # Query Steward
        try:
            response = self.analyzer.chat_with_steward(
                query=question,
                session_data=session_data
            )

            # Add response to chat history
            st.session_state['steward_chat_history'].append({
                'role': 'assistant',
                'content': response
            })

        except Exception as e:
            error_msg = f"❌ Error querying Steward: {str(e)}"
            st.session_state['steward_chat_history'].append({
                'role': 'assistant',
                'content': error_msg
            })

    def _get_session_data(self) -> Dict[str, Any]:
        """
        Get current session data for Steward analysis.

        Returns:
            Session data dictionary
        """
        # TODO: Wire to WebSessionManager
        # Will extract full session with:
        # - messages
        # - metadata
        # - telemetry
        return {
            'session_id': 'current',
            'messages': [],
            'metadata': {}
        }

    def _render_example_questions(self):
        """Render example questions for user guidance."""
        st.markdown("---")
        st.markdown("**Example Questions:**")

        examples = [
            "What governance patterns emerged in this session?",
            "How did primacy attractors evolve?",
            "What was the TELOS impact ratio?",
            "What canonical inputs were identified?",
            "Summarize this session's key findings"
        ]

        for example in examples:
            if st.button(f"📝 {example}", key=f"example_{hash(example)}"):
                st.session_state['steward_question_input'] = example
                st.rerun()
