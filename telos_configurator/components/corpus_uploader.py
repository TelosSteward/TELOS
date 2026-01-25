"""
Corpus Uploader Component
==========================

File upload interface for adding documents to the corpus.
Supports JSON, PDF, TXT, MD, DOCX, XLSX formats.
"""

import sys

import streamlit as st

sys.path.insert(0, '/Users/brunnerjf/Desktop/TELOS_Master/telos_configurator')
from config.styles import (
    GOLD, STATUS_GOOD, STATUS_SEVERE, BG_ELEVATED, TEXT_PRIMARY, TEXT_SECONDARY,
    get_glassmorphism_css, render_section_header, render_info_box
)


def render_corpus_uploader(corpus_engine) -> None:
    """
    Render file upload interface for corpus documents.

    Args:
        corpus_engine: CorpusEngine instance
    """
    st.markdown(render_section_header("Upload Documents"), unsafe_allow_html=True)

    # Container
    st.markdown(f'''
    <div style="{get_glassmorphism_css()}; padding: 1.5rem; margin-bottom: 1rem;">
        <p style="color: {TEXT_SECONDARY};">
            Upload policy documents, guidelines, or reference materials to build your corpus.
            Supported formats: JSON, PDF, TXT, MD, DOCX, XLSX (max 10MB per file).
        </p>
    </div>
    ''', unsafe_allow_html=True)

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['json', 'pdf', 'txt', 'md', 'docx', 'xlsx'],
        help="Upload a document to add to the corpus",
        key="corpus_file_uploader"
    )

    # Metadata inputs
    col1, col2 = st.columns(2)

    with col1:
        category = st.selectbox(
            "Category",
            [
                "Policy",
                "Regulation",
                "Guideline",
                "Standard",
                "Procedure",
                "Reference",
                "General"
            ],
            help="Document category for organization"
        )

    with col2:
        source = st.text_input(
            "Source",
            value="Manual Upload",
            help="Document source or origin system"
        )

    # Add button
    if uploaded_file is not None:
        if st.button("Add to Corpus", type="primary", use_container_width=True):
            with st.spinner("Processing document..."):
                # Add document to corpus
                success, message, doc_id = corpus_engine.add_document(
                    uploaded_file,
                    category=category,
                    source=source
                )

                if success:
                    st.markdown(
                        render_info_box(f"✓ {message}", "success"),
                        unsafe_allow_html=True
                    )
                    st.session_state.corpus_updated = True
                    st.rerun()
                else:
                    st.markdown(
                        render_info_box(f"✗ {message}", "error"),
                        unsafe_allow_html=True
                    )

    # Quick stats
    stats = corpus_engine.get_stats()
    st.markdown(f'''
    <div style="{get_glassmorphism_css(GOLD)}; padding: 1rem; margin-top: 1rem;">
        <div style="display: flex; justify-content: space-around; text-align: center;">
            <div>
                <div style="color: {GOLD}; font-size: 1.5rem; font-weight: 700;">
                    {stats['total_documents']}
                </div>
                <div style="color: {TEXT_SECONDARY}; font-size: 0.85rem;">
                    Total Documents
                </div>
            </div>
            <div>
                <div style="color: {STATUS_GOOD}; font-size: 1.5rem; font-weight: 700;">
                    {stats['embedded_documents']}
                </div>
                <div style="color: {TEXT_SECONDARY}; font-size: 0.85rem;">
                    Embedded
                </div>
            </div>
            <div>
                <div style="color: {TEXT_PRIMARY}; font-size: 1.5rem; font-weight: 700;">
                    {stats['embedding_percentage']:.0f}%
                </div>
                <div style="color: {TEXT_SECONDARY}; font-size: 0.85rem;">
                    Complete
                </div>
            </div>
        </div>
    </div>
    ''', unsafe_allow_html=True)
