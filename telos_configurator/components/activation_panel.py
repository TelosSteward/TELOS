"""
Activation Panel Component
===========================

Activation flow for governance with readiness checklist.
"""

import sys
from typing import Optional

import streamlit as st

sys.path.insert(0, '/Users/brunnerjf/Desktop/TELOS_Master/telos_configurator')
from config.styles import (
    GOLD, STATUS_GOOD, STATUS_SEVERE, STATE_ACTIVE, STATE_INACTIVE,
    BG_ELEVATED, TEXT_PRIMARY, TEXT_SECONDARY,
    get_glassmorphism_css, render_section_header, render_status_badge, render_info_box,
    with_opacity
)

sys.path.insert(0, '/Users/brunnerjf/Desktop/TELOS_Master/telos_configurator')
from engine.governance_engine import GovernanceEngine, PrimacyAttractor, ThresholdConfig


def render_activation_panel(
    pa: Optional[PrimacyAttractor],
    corpus_engine,
    thresholds: ThresholdConfig,
    governance_engine: GovernanceEngine
) -> bool:
    """
    Render governance activation panel with readiness checklist.

    Args:
        pa: Primacy Attractor instance (or None)
        corpus_engine: CorpusEngine instance
        thresholds: ThresholdConfig instance
        governance_engine: GovernanceEngine instance

    Returns:
        True if governance is active, False otherwise
    """
    st.markdown(render_section_header("Governance Activation"), unsafe_allow_html=True)

    # Check if already active
    is_active = governance_engine.is_active()

    # Readiness checks
    pa_ready = pa is not None and pa.embedding is not None
    corpus_stats = corpus_engine.get_stats()
    corpus_loaded = corpus_stats['total_documents'] > 0
    corpus_embedded = corpus_stats['embedded_documents'] > 0
    thresholds_valid, error = thresholds.validate()

    all_ready = pa_ready and corpus_loaded and corpus_embedded and thresholds_valid

    # Readiness checklist
    st.markdown(f'''
    <div style="{get_glassmorphism_css()}; padding: 1.5rem; margin-bottom: 1rem;">
        <h4 style="color: {GOLD}; margin-bottom: 1rem;">Readiness Checklist</h4>
    </div>
    ''', unsafe_allow_html=True)

    # PA Status
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown(f"**Primacy Attractor** configured and embedded")
    with col2:
        if pa_ready:
            st.markdown(render_status_badge('good', '✓'), unsafe_allow_html=True)
        else:
            st.markdown(render_status_badge('severe', '✗'), unsafe_allow_html=True)

    if pa_ready:
        st.caption(f"PA: {pa.name}")

    st.markdown("---")

    # Corpus Status
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown(f"**Corpus** loaded with documents")
    with col2:
        if corpus_loaded:
            st.markdown(render_status_badge('good', '✓'), unsafe_allow_html=True)
        else:
            st.markdown(render_status_badge('severe', '✗'), unsafe_allow_html=True)

    if corpus_loaded:
        st.caption(f"{corpus_stats['total_documents']} documents loaded")

    st.markdown("---")

    # Embedding Status
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown(f"**Corpus Embeddings** generated")
    with col2:
        if corpus_embedded:
            st.markdown(render_status_badge('good', '✓'), unsafe_allow_html=True)
        else:
            st.markdown(render_status_badge('severe', '✗'), unsafe_allow_html=True)

    if corpus_embedded:
        st.caption(f"{corpus_stats['embedded_documents']}/{corpus_stats['total_documents']} documents embedded ({corpus_stats['embedding_percentage']:.0f}%)")

    st.markdown("---")

    # Thresholds Status
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown(f"**Thresholds** configured and valid")
    with col2:
        if thresholds_valid:
            st.markdown(render_status_badge('good', '✓'), unsafe_allow_html=True)
        else:
            st.markdown(render_status_badge('severe', '✗'), unsafe_allow_html=True)

    if thresholds_valid:
        st.caption(f"Tier 1: {thresholds.tier_1_threshold:.2f} | Tier 2: {thresholds.tier_2_lower:.2f} | RAG: {thresholds.rag_relevance:.2f}")

    # Activation button
    st.markdown("---")

    if is_active:
        # Already active - show status
        st.markdown(f'''
        <div style="{get_glassmorphism_css(STATE_ACTIVE)}; padding: 1.5rem; margin-top: 1rem;">
            <div style="text-align: center;">
                <div style="color: {STATE_ACTIVE}; font-size: 1.5rem; font-weight: 700; margin-bottom: 0.5rem;">
                    GOVERNANCE ACTIVE
                </div>
                <div style="color: {TEXT_SECONDARY}; font-size: 0.9rem;">
                    The governance engine is configured and processing queries
                </div>
            </div>
        </div>
        ''', unsafe_allow_html=True)

        # Deactivate button
        if st.button("Deactivate Governance", type="secondary", use_container_width=True):
            # Clear governance engine configuration
            governance_engine.pa = None
            governance_engine.corpus_docs = []
            governance_engine.corpus_embeddings = []
            st.markdown(
                render_info_box("Governance deactivated", "info"),
                unsafe_allow_html=True
            )
            st.rerun()

    else:
        # Not active - show activation button
        if all_ready:
            st.markdown(
                render_info_box(
                    "All readiness checks passed. Ready to activate governance.",
                    "success"
                ),
                unsafe_allow_html=True
            )

            if st.button("ACTIVATE GOVERNANCE", type="primary", use_container_width=True):
                with st.spinner("Activating governance engine..."):
                    # Prepare corpus data for governance engine
                    corpus_docs = corpus_engine.list_documents()

                    # Get embedded documents only
                    embedded_docs = [doc for doc in corpus_docs if doc['embedded']]

                    # Get embeddings
                    corpus_embeddings = []
                    corpus_metadata = []

                    for doc in embedded_docs:
                        doc_data = corpus_engine.get_document(doc['doc_id'])
                        if doc_data:
                            full_doc = corpus_engine.documents.get(doc['doc_id'])
                            if full_doc and full_doc.embedding is not None:
                                corpus_embeddings.append(full_doc.embedding)
                                corpus_metadata.append({
                                    "doc_id": doc['doc_id'],
                                    "title": doc['title'],
                                    "category": doc['category'],
                                    "source": doc['source']
                                })

                    # Configure governance engine
                    success, error_msg = governance_engine.configure(
                        pa=pa,
                        thresholds=thresholds,
                        corpus_docs=corpus_metadata,
                        corpus_embeddings=corpus_embeddings
                    )

                    if success:
                        st.markdown(
                            render_info_box(
                                "✓ Governance engine activated successfully",
                                "success"
                            ),
                            unsafe_allow_html=True
                        )
                        st.rerun()
                    else:
                        st.markdown(
                            render_info_box(
                                f"✗ Failed to activate governance: {error_msg}",
                                "error"
                            ),
                            unsafe_allow_html=True
                        )

        else:
            st.markdown(
                render_info_box(
                    "Complete all readiness checks before activating governance.",
                    "warning"
                ),
                unsafe_allow_html=True
            )

            st.button(
                "ACTIVATE GOVERNANCE",
                type="primary",
                use_container_width=True,
                disabled=True
            )

    return is_active
