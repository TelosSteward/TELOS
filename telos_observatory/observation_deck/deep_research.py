"""
Deep Research Link Generator
=============================

Generates URLs for turn-specific Observatory views.
Opens new browser tab/window for detailed analysis of specific turns.

Usage:
- Observation Deck sidebar: 🔍 Deep Research button
- TELOSCOPE Remote: 🔍 button
"""

import streamlit as st


def generate_deep_research_url(turn_index: int, conversation_id: str, base_url: str = "http://localhost:8502") -> str:
    """
    Generate URL with query parameters for turn-specific view.

    Args:
        turn_index: Turn number to focus on
        conversation_id: Conversation/study ID
        base_url: Base URL (default: localhost:8502)

    Returns:
        Complete URL with query parameters
    """
    return f"{base_url}/?turn={turn_index}&study={conversation_id}"


def render_deep_research_button(turn_index: int, conversation_id: str, label: str = "🔍 Deep Research"):
    """
    Render button that opens Deep Research view in new tab.

    Args:
        turn_index: Turn number
        conversation_id: Conversation/study ID
        label: Button label (default includes icon)
    """
    url = generate_deep_research_url(turn_index, conversation_id)

    # Use HTML link with target="_blank" to open in new tab
    html = f'''
    <a href="{url}" target="_blank" style="
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        text-decoration: none;
        font-weight: bold;
        font-size: 0.9rem;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    " onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 4px 8px rgba(0,0,0,0.3)';"
       onmouseout="this.style.transform='translateY(0px)'; this.style.boxShadow='0 2px 4px rgba(0,0,0,0.2)';">
        {label}
    </a>
    '''

    st.markdown(html, unsafe_allow_html=True)


def render_deep_research_icon_button(turn_index: int, conversation_id: str):
    """
    Render icon-only Deep Research button (for TELOSCOPE Remote).

    Args:
        turn_index: Turn number
        conversation_id: Conversation/study ID
    """
    url = generate_deep_research_url(turn_index, conversation_id)

    # Compact icon button for TELOSCOPE
    html = f'''
    <a href="{url}" target="_blank" style="
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 36px;
        height: 36px;
        background: rgba(102, 126, 234, 0.2);
        color: #667eea;
        border-radius: 6px;
        text-decoration: none;
        font-size: 1.2rem;
        transition: all 0.2s ease;
        border: 1px solid rgba(102, 126, 234, 0.3);
    " onmouseover="this.style.background='rgba(102, 126, 234, 0.3)'; this.style.transform='scale(1.05)';"
       onmouseout="this.style.background='rgba(102, 126, 234, 0.2)'; this.style.transform='scale(1)';"
       title="Deep Research - Open detailed analysis in new tab">
        🔍
    </a>
    '''

    st.markdown(html, unsafe_allow_html=True)
