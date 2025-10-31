"""
Steward Interface Component
============================

AI assistant for turn-specific queries using Mistral API.

Features:
- API key flow (session-only storage)
- Turn-specific context
- Chat history
- Preset queries for common questions
"""

import streamlit as st
from typing import Dict, Any, Optional, List
import os


def render_steward_interface(turn_data: Dict[str, Any], conversation_id: str, turn_index: int):
    """
    Render Steward AI assistant interface.

    Args:
        turn_data: Current turn data
        conversation_id: Conversation/study ID
        turn_index: Turn index (0-based)
    """

    st.markdown("""
        <div style="
            background: rgba(118, 227, 131, 0.1);
            padding: 1rem;
            border-radius: 6px;
            border: 1px solid rgba(118, 227, 131, 0.3);
            margin-top: 1rem;
        ">
            <div style="font-weight: bold; color: #76E383; margin-bottom: 0.75rem;">
                🤝 Steward Assistant
            </div>
    """, unsafe_allow_html=True)

    # API Key flow
    if not st.session_state.get('steward_api_key'):
        render_api_key_input()
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # Chat interface
    render_chat_interface(turn_data, conversation_id, turn_index)

    st.markdown("</div>", unsafe_allow_html=True)


def render_api_key_input():
    """Render API key input form."""

    st.caption("Enter your Mistral API key to activate Steward assistance.")
    st.caption("*Key is stored in session only (not persisted)*")

    api_key = st.text_input(
        "Mistral API Key",
        type="password",
        key="steward_api_key_input",
        label_visibility="collapsed",
        placeholder="Enter Mistral API key..."
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Connect", type="primary", disabled=not api_key):
            st.session_state.steward_api_key = api_key
            st.session_state.steward_messages = []
            st.rerun()

    with col2:
        if st.button("Cancel"):
            st.session_state.steward_active = False
            st.rerun()


def render_chat_interface(turn_data: Dict[str, Any], conversation_id: str, turn_index: int):
    """
    Render Steward chat interface.

    Args:
        turn_data: Current turn data
        conversation_id: Conversation/study ID
        turn_index: Turn index (0-based)
    """

    # Preset queries
    st.caption("**Quick Questions:**")

    preset_queries = [
        "What does the fidelity score mean?",
        "Why was this turn flagged for drift?",
        "How does the threshold work?",
        "What would happen without governance?"
    ]

    preset_cols = st.columns(2)
    for idx, query in enumerate(preset_queries):
        with preset_cols[idx % 2]:
            if st.button(query, key=f"preset_{idx}", use_container_width=True):
                # Add to chat history
                add_steward_message("user", query)
                # Generate response
                response = generate_steward_response(query, turn_data, conversation_id, turn_index)
                add_steward_message("assistant", response)
                st.rerun()

    st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)

    # Chat history
    messages = st.session_state.get('steward_messages', [])

    if messages:
        st.markdown("**Conversation:**")

        chat_container = st.container()
        with chat_container:
            for msg in messages:
                role = msg['role']
                content = msg['content']

                if role == 'user':
                    st.markdown(f"""
                        <div style="
                            background: rgba(255, 255, 255, 0.05);
                            padding: 0.5rem;
                            border-radius: 4px;
                            margin-bottom: 0.5rem;
                        ">
                            <div style="color: #888; font-size: 0.75rem;">YOU</div>
                            <div>{content}</div>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div style="
                            background: rgba(118, 227, 131, 0.1);
                            padding: 0.5rem;
                            border-radius: 4px;
                            margin-bottom: 0.5rem;
                        ">
                            <div style="color: #76E383; font-size: 0.75rem;">🤝 STEWARD</div>
                            <div>{content}</div>
                        </div>
                    """, unsafe_allow_html=True)

    # Custom query input
    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)

    custom_query = st.text_input(
        "Ask Steward",
        key="steward_custom_query",
        placeholder="Ask a question about this turn...",
        label_visibility="collapsed"
    )

    col1, col2 = st.columns([3, 1])

    with col1:
        if st.button("Send", type="primary", disabled=not custom_query):
            # Add to chat history
            add_steward_message("user", custom_query)
            # Generate response
            response = generate_steward_response(custom_query, turn_data, conversation_id, turn_index)
            add_steward_message("assistant", response)
            st.rerun()

    with col2:
        if st.button("Clear"):
            st.session_state.steward_messages = []
            st.rerun()

    # Disconnect option
    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
    if st.button("Disconnect Steward", key="steward_disconnect"):
        st.session_state.steward_api_key = None
        st.session_state.steward_messages = []
        st.session_state.steward_active = False
        st.rerun()


def generate_steward_response(
    query: str,
    turn_data: Dict[str, Any],
    conversation_id: str,
    turn_index: int
) -> str:
    """
    Generate Steward response using Mistral API.

    Args:
        query: User question
        turn_data: Current turn data
        conversation_id: Conversation/study ID
        turn_index: Turn index (0-based)

    Returns:
        Assistant response text
    """

    # Build context from turn data
    context = f"""
    You are Steward, an AI assistant helping researchers understand TELOS governance metrics.

    Current Turn Context:
    - Turn: {turn_index + 1}
    - Conversation ID: {conversation_id}
    - Fidelity: {turn_data.get('fidelity', 'N/A')}
    - Distance: {turn_data.get('distance', 'N/A')}
    - Threshold: {turn_data.get('threshold', 0.800)}
    - Within Basin: {turn_data.get('within_basin', 'N/A')}
    - Drift Detected: {turn_data.get('drift_detected', False)}

    User Question: {query}

    Provide a clear, concise answer (2-3 sentences) focused on this specific turn's metrics.
    """

    try:
        # Import Mistral client
        from mistralai import Mistral

        # Get API key from session state
        api_key = st.session_state.get('steward_api_key')

        if not api_key:
            return "Error: No API key configured. Please reconnect."

        # Initialize client
        client = Mistral(api_key=api_key)

        # Generate response
        response = client.chat.complete(
            model="mistral-small-latest",
            messages=[
                {"role": "system", "content": context},
                {"role": "user", "content": query}
            ]
        )

        return response.choices[0].message.content

    except ImportError:
        return "Error: Mistral AI library not installed. Run: pip install mistralai"
    except Exception as e:
        return f"Error communicating with Steward: {str(e)}"


def add_steward_message(role: str, content: str):
    """
    Add message to Steward chat history.

    Args:
        role: 'user' or 'assistant'
        content: Message content
    """
    if 'steward_messages' not in st.session_state:
        st.session_state.steward_messages = []

    st.session_state.steward_messages.append({
        'role': role,
        'content': content
    })


def clear_steward_history():
    """Clear Steward chat history."""
    st.session_state.steward_messages = []


def is_steward_connected() -> bool:
    """Check if Steward is connected (API key configured)."""
    return st.session_state.get('steward_api_key') is not None


# =============================================================================
# COMPACT STEWARD INTERFACE (for TELOSCOPE Remote)
# =============================================================================

def render_compact_steward_interface(turn_data: Dict[str, Any], conversation_id: str, turn_index: int):
    """
    Render compact Steward interface for TELOSCOPE Remote.

    Args:
        turn_data: Current turn data
        conversation_id: Conversation/study ID
        turn_index: Turn index (0-based)
    """

    if not st.session_state.get('steward_api_key'):
        st.caption("**🤝 Steward:** Not connected")

        api_key = st.text_input(
            "Mistral API Key",
            type="password",
            key="teloscope_steward_api_key_input",
            placeholder="API key..."
        )

        if st.button("Connect", type="primary", disabled=not api_key, key="teloscope_steward_connect"):
            st.session_state.steward_api_key = api_key
            st.session_state.steward_messages = []
            st.rerun()

    else:
        st.caption("**🤝 Steward:** Connected")

        query = st.text_input(
            "Ask Steward",
            key="teloscope_steward_query",
            placeholder="Ask about this turn..."
        )

        if st.button("Send", disabled=not query, key="teloscope_steward_send"):
            add_steward_message("user", query)
            response = generate_steward_response(query, turn_data, conversation_id, turn_index)
            add_steward_message("assistant", response)
            st.rerun()

        # Show last message if any
        messages = st.session_state.get('steward_messages', [])
        if messages:
            last_msg = messages[-1]
            if last_msg['role'] == 'assistant':
                st.caption(f"**Response:** {last_msg['content'][:100]}...")
