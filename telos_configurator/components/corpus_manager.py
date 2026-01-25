"""
Corpus Manager Component
=========================

View and manage loaded corpus documents.
Provides document list, embedding controls, and corpus persistence.
"""

import sys
import json
from datetime import datetime

import streamlit as st

sys.path.insert(0, '/Users/brunnerjf/Desktop/TELOS_Master/telos_configurator')
from config.styles import (
    GOLD, STATUS_GOOD, STATE_PENDING, STATUS_SEVERE, BG_ELEVATED, TEXT_PRIMARY, TEXT_SECONDARY,
    get_glassmorphism_css, render_section_header, render_status_badge, render_info_box
)


def render_corpus_manager(corpus_engine) -> None:
    """
    Render corpus management interface.

    Args:
        corpus_engine: CorpusEngine instance
    """
    st.markdown(render_section_header("Corpus Management"), unsafe_allow_html=True)

    # Get document list
    documents = corpus_engine.list_documents()
    stats = corpus_engine.get_stats()

    if not documents:
        st.markdown(
            render_info_box(
                "No documents in corpus. Upload documents to get started.",
                "info"
            ),
            unsafe_allow_html=True
        )
        return

    # Corpus statistics
    st.markdown(f'''
    <div style="{get_glassmorphism_css(GOLD)}; padding: 1rem; margin-bottom: 1rem;">
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; text-align: center;">
            <div>
                <div style="color: {GOLD}; font-size: 1.3rem; font-weight: 700;">
                    {stats['total_documents']}
                </div>
                <div style="color: {TEXT_SECONDARY}; font-size: 0.8rem;">Documents</div>
            </div>
            <div>
                <div style="color: {STATUS_GOOD}; font-size: 1.3rem; font-weight: 700;">
                    {stats['embedded_documents']}
                </div>
                <div style="color: {TEXT_SECONDARY}; font-size: 0.8rem;">Embedded</div>
            </div>
            <div>
                <div style="color: {TEXT_PRIMARY}; font-size: 1.3rem; font-weight: 700;">
                    {len(stats['categories'])}
                </div>
                <div style="color: {TEXT_SECONDARY}; font-size: 0.8rem;">Categories</div>
            </div>
            <div>
                <div style="color: {TEXT_PRIMARY}; font-size: 1.3rem; font-weight: 700;">
                    {stats['embedding_percentage']:.0f}%
                </div>
                <div style="color: {TEXT_SECONDARY}; font-size: 0.8rem;">Complete</div>
            </div>
        </div>
    </div>
    ''', unsafe_allow_html=True)

    # Embed All button
    if stats['not_embedded'] > 0:
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("Embed All", type="primary", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()

                def progress_callback(current, total, filename):
                    progress = current / total
                    progress_bar.progress(progress)
                    status_text.text(f"Embedding {current}/{total}: {filename}")

                success_count, fail_count, failed = corpus_engine.embed_all(progress_callback)

                progress_bar.empty()
                status_text.empty()

                if fail_count == 0:
                    st.markdown(
                        render_info_box(
                            f"✓ Successfully embedded {success_count} documents",
                            "success"
                        ),
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        render_info_box(
                            f"⚠ Embedded {success_count} documents, {fail_count} failed: {', '.join(failed)}",
                            "warning"
                        ),
                        unsafe_allow_html=True
                    )

                st.rerun()

    # Document list
    st.markdown(f'''
    <div style="{get_glassmorphism_css()}; padding: 1rem; margin-bottom: 1rem;">
        <h4 style="color: {GOLD}; margin-bottom: 1rem;">Documents</h4>
    </div>
    ''', unsafe_allow_html=True)

    # Render each document
    for doc in documents:
        with st.container():
            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])

            with col1:
                st.markdown(f"**{doc['title']}**")
                st.caption(f"{doc['category']} • {doc['source']}")

            with col2:
                if doc['embedded']:
                    st.markdown(
                        render_status_badge('good', '✓ Embedded'),
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        render_status_badge('pending', '○ Pending'),
                        unsafe_allow_html=True
                    )

            with col3:
                st.caption(f"{doc['text_length']:,} chars")

            with col4:
                if st.button("Remove", key=f"remove_{doc['doc_id']}", use_container_width=True):
                    success, message = corpus_engine.remove_document(doc['doc_id'])
                    if success:
                        st.rerun()

            st.markdown("---")

    # Persistence controls
    st.markdown(f'''
    <div style="{get_glassmorphism_css()}; padding: 1rem; margin-top: 1rem;">
        <h4 style="color: {GOLD}; margin-bottom: 1rem;">Corpus Persistence</h4>
    </div>
    ''', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Save Corpus", use_container_width=True):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"corpus_export_{timestamp}.json"

            success, message = corpus_engine.save_corpus(filepath)

            if success:
                st.markdown(
                    render_info_box(f"✓ {message}", "success"),
                    unsafe_allow_html=True
                )

                # Offer download
                with open(filepath, 'r') as f:
                    corpus_data = f.read()

                st.download_button(
                    label="Download Corpus File",
                    data=corpus_data,
                    file_name=filepath,
                    mime="application/json",
                    use_container_width=True
                )
            else:
                st.markdown(
                    render_info_box(f"✗ {message}", "error"),
                    unsafe_allow_html=True
                )

    with col2:
        uploaded_corpus = st.file_uploader(
            "Load Corpus",
            type=['json'],
            help="Upload a previously saved corpus file",
            key="corpus_loader"
        )

        if uploaded_corpus is not None:
            if st.button("Load Corpus File", use_container_width=True):
                # Save uploaded file temporarily
                temp_path = f"temp_corpus_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(temp_path, 'wb') as f:
                    f.write(uploaded_corpus.getvalue())

                success, message = corpus_engine.load_corpus(temp_path)

                if success:
                    st.markdown(
                        render_info_box(f"✓ {message}", "success"),
                        unsafe_allow_html=True
                    )
                    st.rerun()
                else:
                    st.markdown(
                        render_info_box(f"✗ {message}", "error"),
                        unsafe_allow_html=True
                    )
