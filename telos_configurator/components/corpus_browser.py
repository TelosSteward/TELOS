"""
Corpus Browser Component
========================

A transparent interface for browsing, searching, and visualizing the policy corpus.
Provides real-time feedback showing which documents are being retrieved during
governance decisions, allowing users to see exactly how TELOS works.

Features:
- Full corpus grid view with category organization
- Search/filter by keyword, category, or document type
- Visual highlighting when documents are retrieved for a query
- Click-to-source navigation showing exact text being used
- Relevance scoring display for retrieved chunks
"""

import sys
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import json

import streamlit as st

sys.path.insert(0, '/Users/brunnerjf/Desktop/TELOS_Master/telos_configurator')
from config.styles import (
    GOLD, GOLD_DARK, BG_ELEVATED, BG_SURFACE, TEXT_PRIMARY, TEXT_SECONDARY,
    STATUS_GOOD, STATUS_MILD, STATUS_MODERATE, STATUS_SEVERE,
    get_glassmorphism_css, with_opacity, TIER_1, TIER_2, TIER_3
)


@dataclass
class RetrievedChunk:
    """Represents a chunk retrieved from the corpus during RAG."""
    document_id: str
    document_title: str
    chunk_text: str
    relevance_score: float
    category: str
    source_url: Optional[str] = None
    chunk_index: int = 0


@dataclass
class CorpusDocument:
    """Represents a document in the corpus."""
    document_id: str
    title: str
    source: str
    source_url: str
    category: str
    subcategory: str
    text_content: str
    summary: str
    chunks: List[str]
    is_highlighted: bool = False
    relevance_score: float = 0.0


def get_category_icon(category: str) -> str:
    """Get icon for document category."""
    icons = {
        'privacy_regulations': '🔒',
        'clinical_guidelines': '📋',
        'incident_protocols': '🚨',
        'consent_frameworks': '✍️',
        'compliance': '⚖️',
        'ethics': '🎯',
        'security': '🛡️',
        'default': '📄'
    }
    return icons.get(category.lower(), icons['default'])


def get_category_color(category: str) -> str:
    """Get color for document category."""
    colors = {
        'privacy_regulations': TIER_1,
        'clinical_guidelines': STATUS_GOOD,
        'incident_protocols': STATUS_MODERATE,
        'consent_frameworks': TIER_2,
        'compliance': TIER_3,
        'ethics': GOLD,
        'security': STATUS_SEVERE,
    }
    return colors.get(category.lower(), TEXT_SECONDARY)


def render_corpus_browser(
    documents: List[Dict],
    retrieved_chunks: Optional[List[RetrievedChunk]] = None,
    on_document_click: Optional[callable] = None
):
    """
    Render the full corpus browser with search and visual feedback.

    Args:
        documents: List of corpus documents
        retrieved_chunks: Currently retrieved chunks (for highlighting)
        on_document_click: Callback when document is clicked
    """
    st.markdown(f'''
    <div style="{get_glassmorphism_css(GOLD)}; padding: 1rem; margin-bottom: 1rem;">
        <h3 style="color: {GOLD}; margin: 0 0 0.5rem 0; font-size: 1.2rem;">
            📚 Policy Corpus Browser
        </h3>
        <p style="color: {TEXT_SECONDARY}; margin: 0; font-size: 0.85rem;">
            Browse, search, and explore the governance knowledge base.
            Documents flash when retrieved during query processing.
        </p>
    </div>
    ''', unsafe_allow_html=True)

    # Build set of highlighted document IDs
    highlighted_ids = set()
    chunk_relevance = {}
    if retrieved_chunks:
        for chunk in retrieved_chunks:
            highlighted_ids.add(chunk.document_id)
            if chunk.document_id not in chunk_relevance:
                chunk_relevance[chunk.document_id] = chunk.relevance_score
            else:
                chunk_relevance[chunk.document_id] = max(
                    chunk_relevance[chunk.document_id],
                    chunk.relevance_score
                )

    # Search and filter controls
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        search_query = st.text_input(
            "🔍 Search corpus",
            placeholder="Search by keyword, title, or content...",
            key="corpus_search"
        )

    with col2:
        # Get unique categories
        categories = list(set(doc.get('category', 'Unknown') for doc in documents))
        categories = ['All Categories'] + sorted(categories)
        selected_category = st.selectbox(
            "Category",
            categories,
            key="corpus_category_filter"
        )

    with col3:
        sort_options = ['Relevance', 'Alphabetical', 'Category']
        if highlighted_ids:
            sort_options = ['Retrieved First'] + sort_options
        sort_by = st.selectbox(
            "Sort by",
            sort_options,
            key="corpus_sort"
        )

    # Filter documents
    filtered_docs = documents.copy()

    if search_query:
        search_lower = search_query.lower()
        filtered_docs = [
            doc for doc in filtered_docs
            if search_lower in doc.get('title', '').lower()
            or search_lower in doc.get('text_content', '').lower()
            or search_lower in doc.get('document_id', '').lower()
            or search_lower in doc.get('summary', '').lower()
        ]

    if selected_category != 'All Categories':
        filtered_docs = [
            doc for doc in filtered_docs
            if doc.get('category', '') == selected_category
        ]

    # Sort documents
    if sort_by == 'Retrieved First' and highlighted_ids:
        filtered_docs.sort(
            key=lambda d: (
                d.get('document_id', '') not in highlighted_ids,
                -chunk_relevance.get(d.get('document_id', ''), 0)
            )
        )
    elif sort_by == 'Alphabetical':
        filtered_docs.sort(key=lambda d: d.get('title', ''))
    elif sort_by == 'Category':
        filtered_docs.sort(key=lambda d: (d.get('category', ''), d.get('title', '')))

    # Stats bar
    retrieved_count = len([d for d in filtered_docs if d.get('document_id', '') in highlighted_ids])

    st.markdown(f'''
    <div style="display: flex; gap: 1.5rem; margin: 1rem 0; padding: 0.75rem; background: {BG_SURFACE}; border-radius: 6px;">
        <div>
            <span style="color: {TEXT_SECONDARY}; font-size: 0.8rem;">Total Documents</span>
            <div style="color: {GOLD}; font-size: 1.2rem; font-weight: 700;">{len(filtered_docs)}</div>
        </div>
        <div>
            <span style="color: {TEXT_SECONDARY}; font-size: 0.8rem;">Retrieved</span>
            <div style="color: {STATUS_GOOD if retrieved_count > 0 else TEXT_SECONDARY}; font-size: 1.2rem; font-weight: 700;">{retrieved_count}</div>
        </div>
        <div>
            <span style="color: {TEXT_SECONDARY}; font-size: 0.8rem;">Categories</span>
            <div style="color: {TEXT_PRIMARY}; font-size: 1.2rem; font-weight: 700;">{len(categories) - 1}</div>
        </div>
    </div>
    ''', unsafe_allow_html=True)

    # Document grid
    if not filtered_docs:
        st.info("No documents match your search criteria.")
        return

    # Render as grid (3 columns)
    cols = st.columns(3)

    for idx, doc in enumerate(filtered_docs):
        doc_id = doc.get('document_id', f'doc_{idx}')
        is_highlighted = doc_id in highlighted_ids
        relevance = chunk_relevance.get(doc_id, 0)
        category = doc.get('category', 'Unknown')
        cat_icon = get_category_icon(category)
        cat_color = get_category_color(category)

        # Determine styling based on highlight state
        if is_highlighted:
            border_color = STATUS_GOOD
            bg_color = with_opacity(STATUS_GOOD, 0.15)
            glow = f"0 0 20px {with_opacity(STATUS_GOOD, 0.4)}"
            animation = "animation: pulse 1.5s ease-in-out infinite;"
        else:
            border_color = with_opacity(GOLD, 0.3)
            bg_color = BG_ELEVATED
            glow = "none"
            animation = ""

        with cols[idx % 3]:
            # Build card HTML
            relevance_badge = ""
            if is_highlighted:
                relevance_badge = f'''
                <div style="position: absolute; top: -8px; right: -8px;
                    background: {STATUS_GOOD}; color: white;
                    padding: 2px 8px; border-radius: 10px;
                    font-size: 0.7rem; font-weight: 700;">
                    {relevance:.0%} match
                </div>
                '''

            source_link = ""
            if doc.get('source_url'):
                source_link = f'<a href="{doc["source_url"]}" target="_blank" style="color: {GOLD}; font-size: 0.75rem; text-decoration: none;">View Source ↗</a>'

            card_html = f'''
            <style>
                @keyframes pulse {{
                    0%, 100% {{ box-shadow: 0 0 15px {with_opacity(STATUS_GOOD, 0.3)}; }}
                    50% {{ box-shadow: 0 0 25px {with_opacity(STATUS_GOOD, 0.6)}; }}
                }}
            </style>
            <div style="
                position: relative;
                background: {bg_color};
                border: 2px solid {border_color};
                border-radius: 8px;
                padding: 1rem;
                margin-bottom: 1rem;
                box-shadow: {glow};
                {animation}
                transition: all 0.3s ease;
            ">
                {relevance_badge}
                <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
                    <span style="font-size: 1.2rem;">{cat_icon}</span>
                    <span style="color: {cat_color}; font-size: 0.75rem; font-weight: 600; text-transform: uppercase;">
                        {category.replace('_', ' ')}
                    </span>
                </div>
                <div style="color: {GOLD if is_highlighted else TEXT_PRIMARY}; font-weight: 600; font-size: 0.9rem; margin-bottom: 0.5rem; line-height: 1.3;">
                    {doc.get('title', doc_id)[:60]}{'...' if len(doc.get('title', '')) > 60 else ''}
                </div>
                <div style="color: {TEXT_SECONDARY}; font-size: 0.8rem; margin-bottom: 0.5rem;">
                    {doc.get('source', 'Unknown Source')}
                </div>
                <div style="color: {TEXT_SECONDARY}; font-size: 0.75rem; margin-bottom: 0.75rem; line-height: 1.4; max-height: 3rem; overflow: hidden;">
                    {doc.get('summary', '')[:100]}{'...' if len(doc.get('summary', '')) > 100 else ''}
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="color: {TEXT_SECONDARY}; font-size: 0.7rem;">{doc_id}</span>
                    {source_link}
                </div>
            </div>
            '''
            st.markdown(card_html, unsafe_allow_html=True)

            # View details button
            if st.button(f"View Details", key=f"view_{doc_id}", use_container_width=True):
                st.session_state.selected_corpus_doc = doc
                st.session_state.show_doc_detail = True


def render_document_detail(doc: Dict, retrieved_chunk: Optional[RetrievedChunk] = None):
    """
    Render detailed view of a single document with chunk highlighting.

    Args:
        doc: Document dictionary
        retrieved_chunk: If provided, highlights the retrieved section
    """
    doc_id = doc.get('document_id', 'Unknown')
    category = doc.get('category', 'Unknown')
    cat_icon = get_category_icon(category)
    cat_color = get_category_color(category)

    # Header
    st.markdown(f'''
    <div style="{get_glassmorphism_css(cat_color)}; padding: 1.5rem; margin-bottom: 1rem;">
        <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;">
            <span style="font-size: 2rem;">{cat_icon}</span>
            <div>
                <div style="color: {GOLD}; font-size: 1.3rem; font-weight: 700;">
                    {doc.get('title', doc_id)}
                </div>
                <div style="color: {cat_color}; font-size: 0.85rem; text-transform: uppercase; font-weight: 600;">
                    {category.replace('_', ' ')}
                </div>
            </div>
        </div>
        <div style="display: flex; gap: 2rem; color: {TEXT_SECONDARY}; font-size: 0.85rem;">
            <div><strong>Source:</strong> {doc.get('source', 'Unknown')}</div>
            <div><strong>ID:</strong> {doc_id}</div>
        </div>
    </div>
    ''', unsafe_allow_html=True)

    # Source link
    if doc.get('source_url'):
        st.markdown(f'''
        <div style="margin-bottom: 1rem;">
            <a href="{doc['source_url']}" target="_blank" style="
                display: inline-block;
                background: {with_opacity(GOLD, 0.1)};
                border: 1px solid {GOLD};
                color: {GOLD};
                padding: 0.5rem 1rem;
                border-radius: 6px;
                text-decoration: none;
                font-size: 0.9rem;
            ">🔗 View Official Source Document</a>
        </div>
        ''', unsafe_allow_html=True)

    # Summary
    if doc.get('summary'):
        st.markdown(f'''
        <div style="background: {BG_SURFACE}; padding: 1rem; border-radius: 6px; margin-bottom: 1rem; border-left: 3px solid {GOLD};">
            <div style="color: {GOLD}; font-weight: 600; margin-bottom: 0.5rem; font-size: 0.9rem;">Summary</div>
            <div style="color: {TEXT_SECONDARY}; font-size: 0.9rem; line-height: 1.6;">
                {doc.get('summary', '')}
            </div>
        </div>
        ''', unsafe_allow_html=True)

    # Full text content with optional highlighting
    text_content = doc.get('text_content', '')

    if retrieved_chunk and retrieved_chunk.chunk_text:
        # Highlight the retrieved chunk in the full text
        chunk_text = retrieved_chunk.chunk_text
        if chunk_text in text_content:
            # Split and highlight
            before = text_content.split(chunk_text)[0]
            after = text_content.split(chunk_text)[-1] if text_content.count(chunk_text) == 1 else ""

            st.markdown(f'''
            <div style="background: {BG_ELEVATED}; padding: 1.5rem; border-radius: 8px; margin-bottom: 1rem;">
                <div style="color: {GOLD}; font-weight: 600; margin-bottom: 1rem; font-size: 0.9rem;">
                    Full Document Text
                    <span style="background: {with_opacity(STATUS_GOOD, 0.2)}; color: {STATUS_GOOD}; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; margin-left: 0.5rem;">
                        Retrieved section highlighted
                    </span>
                </div>
                <div style="color: {TEXT_SECONDARY}; font-size: 0.85rem; line-height: 1.8; white-space: pre-wrap;">
                    {before}<span style="background: {with_opacity(STATUS_GOOD, 0.3)}; border-left: 3px solid {STATUS_GOOD}; padding: 0.25rem 0.5rem; border-radius: 4px;">{chunk_text}</span>{after}
                </div>
            </div>
            ''', unsafe_allow_html=True)
        else:
            # Chunk not found in text, show both separately
            _render_plain_text(text_content)
            st.markdown(f'''
            <div style="background: {with_opacity(STATUS_GOOD, 0.1)}; border: 2px solid {STATUS_GOOD}; padding: 1rem; border-radius: 8px; margin-top: 1rem;">
                <div style="color: {STATUS_GOOD}; font-weight: 600; margin-bottom: 0.5rem;">
                    🎯 Retrieved Chunk (Relevance: {retrieved_chunk.relevance_score:.0%})
                </div>
                <div style="color: {TEXT_PRIMARY}; font-size: 0.9rem; line-height: 1.6;">
                    "{chunk_text}"
                </div>
            </div>
            ''', unsafe_allow_html=True)
    else:
        _render_plain_text(text_content)


def _render_plain_text(text_content: str):
    """Render plain text content."""
    st.markdown(f'''
    <div style="background: {BG_ELEVATED}; padding: 1.5rem; border-radius: 8px;">
        <div style="color: {GOLD}; font-weight: 600; margin-bottom: 1rem; font-size: 0.9rem;">
            Full Document Text
        </div>
        <div style="color: {TEXT_SECONDARY}; font-size: 0.85rem; line-height: 1.8; white-space: pre-wrap; max-height: 400px; overflow-y: auto;">
            {text_content[:3000]}{'...' if len(text_content) > 3000 else ''}
        </div>
    </div>
    ''', unsafe_allow_html=True)


def render_retrieval_panel(retrieved_chunks: List[RetrievedChunk], documents: List[Dict]):
    """
    Render a panel showing what was retrieved for the current query.

    Args:
        retrieved_chunks: List of retrieved chunks with relevance scores
        documents: Full document list for linking
    """
    if not retrieved_chunks:
        st.markdown(f'''
        <div style="background: {BG_SURFACE}; padding: 1rem; border-radius: 6px; text-align: center;">
            <span style="color: {TEXT_SECONDARY};">No documents retrieved yet. Run a query to see RAG results.</span>
        </div>
        ''', unsafe_allow_html=True)
        return

    st.markdown(f'''
    <div style="{get_glassmorphism_css(STATUS_GOOD)}; padding: 1rem; margin-bottom: 1rem;">
        <h4 style="color: {STATUS_GOOD}; margin: 0 0 0.5rem 0;">
            🎯 Retrieved Context ({len(retrieved_chunks)} chunks)
        </h4>
        <p style="color: {TEXT_SECONDARY}; margin: 0; font-size: 0.8rem;">
            These policy sections were retrieved to inform the governance decision.
        </p>
    </div>
    ''', unsafe_allow_html=True)

    # Sort by relevance
    sorted_chunks = sorted(retrieved_chunks, key=lambda c: c.relevance_score, reverse=True)

    # Build all chunks as single HTML block
    chunks_html = ""
    for i, chunk in enumerate(sorted_chunks):
        relevance_color = STATUS_GOOD if chunk.relevance_score >= 0.7 else (
            STATUS_MILD if chunk.relevance_score >= 0.5 else STATUS_MODERATE
        )
        cat_icon = get_category_icon(chunk.category)

        chunks_html += f'''
        <div style="
            background: {BG_ELEVATED};
            border: 1px solid {relevance_color};
            border-left: 4px solid {relevance_color};
            border-radius: 6px;
            padding: 1rem;
            margin-bottom: 0.75rem;
        ">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span>{cat_icon}</span>
                    <span style="color: {GOLD}; font-weight: 600; font-size: 0.85rem;">
                        {chunk.document_title[:40]}{'...' if len(chunk.document_title) > 40 else ''}
                    </span>
                </div>
                <span style="background: {with_opacity(relevance_color, 0.2)}; color: {relevance_color}; padding: 2px 8px; border-radius: 10px; font-size: 0.75rem; font-weight: 700;">
                    {chunk.relevance_score:.0%}
                </span>
            </div>
            <div style="color: {TEXT_SECONDARY}; font-size: 0.8rem; line-height: 1.5; max-height: 4rem; overflow: hidden;">
                "{chunk.chunk_text[:200]}{'...' if len(chunk.chunk_text) > 200 else ''}"
            </div>
            <div style="margin-top: 0.5rem; font-size: 0.7rem; color: {TEXT_SECONDARY};">
                {chunk.document_id} | Chunk #{chunk.chunk_index + 1}
            </div>
        </div>
        '''

    st.markdown(chunks_html, unsafe_allow_html=True)


# Export component functions
__all__ = [
    'render_corpus_browser',
    'render_document_detail',
    'render_retrieval_panel',
    'RetrievedChunk',
    'CorpusDocument',
    'get_category_icon',
    'get_category_color'
]
