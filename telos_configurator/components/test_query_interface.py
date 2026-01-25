"""
Test Query Interface Component
===============================

Interactive test interface for querying the governance engine.
Shows real-time tier classification and fidelity scoring.

When a query is processed, the retrieved chunks are stored in session state
so the Corpus Browser can highlight which documents were used.
"""

import sys

import streamlit as st

sys.path.insert(0, '/Users/brunnerjf/Desktop/TELOS_Master/telos_configurator')
from config.styles import (
    GOLD, TIER_1, TIER_2, TIER_3, STATUS_GOOD, BG_ELEVATED, TEXT_PRIMARY, TEXT_SECONDARY,
    get_glassmorphism_css, render_section_header, with_opacity
)

sys.path.insert(0, '/Users/brunnerjf/Desktop/TELOS_Master/telos_configurator')
from engine.governance_engine import GovernanceEngine
from components.corpus_browser import RetrievedChunk


def render_test_query_interface(governance_engine: GovernanceEngine) -> None:
    """
    Render test query interface with real-time governance results.

    Args:
        governance_engine: GovernanceEngine instance
    """
    st.markdown(render_section_header("Test Query Interface"), unsafe_allow_html=True)

    if not governance_engine.is_active():
        st.markdown(f'''
        <div style="{get_glassmorphism_css()}; padding: 1.5rem;">
            <p style="color: {TEXT_SECONDARY}; text-align: center;">
                Activate governance before testing queries.
            </p>
        </div>
        ''', unsafe_allow_html=True)
        return

    st.markdown(f'''
    <div style="{get_glassmorphism_css()}; padding: 1.5rem; margin-bottom: 1rem;">
        <p style="color: {TEXT_SECONDARY};">
            Test how the governance engine classifies queries. Enter a query below to see
            the fidelity score, tier classification, and any retrieved policies.
        </p>
    </div>
    ''', unsafe_allow_html=True)

    # Query input
    query = st.text_area(
        "Enter Test Query",
        placeholder="e.g., What are the requirements for protecting patient health information?",
        height=100,
        help="Enter a query to test against the governance engine"
    )

    # Top-k selector
    col1, col2 = st.columns([3, 1])
    with col2:
        top_k = st.number_input(
            "Top-K Policies",
            min_value=1,
            max_value=10,
            value=3,
            help="Number of policies to retrieve for Tier 2 queries"
        )

    # Test button
    if st.button("Test Query", type="primary", use_container_width=True, disabled=not query):
        with st.spinner("Processing query..."):
            # Process query
            result = governance_engine.process(query, top_k=int(top_k))

            # Store retrieved chunks in session state for corpus browser
            # This allows the Corpus Browser to highlight which documents were used
            retrieved_chunks = []
            if result.retrieved_policies:
                for policy in result.retrieved_policies:
                    chunk = RetrievedChunk(
                        document_id=policy.get('document_id', 'Unknown'),
                        document_title=policy.get('title', 'Untitled'),
                        chunk_text=policy.get('text_preview', policy.get('text_content', '')[:200]),
                        relevance_score=policy.get('similarity', 0.0),
                        category=policy.get('category', 'general'),
                        source_url=policy.get('source_url'),
                        chunk_index=0  # Policy-level retrieval
                    )
                    retrieved_chunks.append(chunk)

            # Store in session state for corpus browser to access
            st.session_state.last_retrieved_chunks = retrieved_chunks
            st.session_state.last_query = query
            st.session_state.last_query_result = result

            # Display results
            tier_color = TIER_1 if result.tier == 1 else (TIER_2 if result.tier == 2 else TIER_3)

            # Build result HTML - using single-line styles to avoid Streamlit parsing issues
            fidelity_pct = result.fidelity * 100
            progress_style = f"position: absolute; left: 0; top: 0; height: 100%; width: {fidelity_pct}%; background: linear-gradient(90deg, {with_opacity(tier_color, 0.8)} 0%, {tier_color} 100%);"
            tier_badge_style = f"display: inline-block; background: {with_opacity(tier_color, 0.2)}; border: 2px solid {tier_color}; border-radius: 8px; padding: 0.75rem 1.5rem;"

            st.markdown(f'''<div style="{get_glassmorphism_css(tier_color)}; padding: 1.5rem; margin-top: 1rem;"><h4 style="color: {GOLD}; margin-bottom: 1rem;">Query Result</h4><div style="margin-bottom: 1.5rem;"><div style="color: {TEXT_SECONDARY}; font-size: 0.85rem; margin-bottom: 0.5rem;">Fidelity Score</div><div style="display: flex; align-items: center; gap: 1rem;"><div style="color: {tier_color}; font-size: 2rem; font-weight: 700;">{result.fidelity:.4f}</div><div style="flex: 1; height: 20px; background: {BG_ELEVATED}; border-radius: 10px; overflow: hidden; position: relative;"><div style="{progress_style}"></div></div></div></div><div style="margin-bottom: 1.5rem;"><div style="color: {TEXT_SECONDARY}; font-size: 0.85rem; margin-bottom: 0.5rem;">Tier Classification</div><div style="{tier_badge_style}"><div style="color: {tier_color}; font-size: 1.2rem; font-weight: 700;">{result.tier_name}</div></div></div><div style="margin-bottom: 1.5rem;"><div style="color: {TEXT_SECONDARY}; font-size: 0.85rem; margin-bottom: 0.5rem;">Action Taken</div><div style="color: {TEXT_PRIMARY}; font-size: 1.1rem; font-weight: 600;">{result.action}</div></div></div>''', unsafe_allow_html=True)

            # Blocking reason (if present)
            if result.blocking_reason:
                st.markdown(f'''
                <div style="{get_glassmorphism_css()}; padding: 1rem; margin-top: 1rem;">
                    <div style="color: {TEXT_SECONDARY}; font-size: 0.85rem; margin-bottom: 0.5rem;">
                        Reason
                    </div>
                    <div style="color: {TEXT_SECONDARY}; font-size: 0.95rem; font-style: italic;">
                        {result.blocking_reason}
                    </div>
                </div>
                ''', unsafe_allow_html=True)

            # Retrieved policies (Tier 2 only)
            if result.tier == 2 and result.retrieved_policies:
                st.markdown(f'''
                <div style="{get_glassmorphism_css(TIER_2)}; padding: 1.5rem; margin-top: 1rem;">
                    <h4 style="color: {TIER_2}; margin-bottom: 1rem;">Retrieved Policies ({len(result.retrieved_policies)})</h4>
                </div>
                ''', unsafe_allow_html=True)

                for idx, policy in enumerate(result.retrieved_policies, 1):
                    st.markdown(f'''
                    <div style="{get_glassmorphism_css()}; padding: 1rem; margin-bottom: 0.75rem;">
                        <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 0.5rem;">
                            <div style="color: {GOLD}; font-weight: 700; font-size: 1rem;">
                                {idx}. {policy.get('title', 'Untitled')}
                            </div>
                            <div style="
                                background: {with_opacity(STATUS_GOOD, 0.2)};
                                border: 1px solid {STATUS_GOOD};
                                border-radius: 6px;
                                padding: 0.25rem 0.75rem;
                                color: {STATUS_GOOD};
                                font-size: 0.85rem;
                                font-weight: 600;
                            ">
                                {policy.get('similarity', 0):.3f}
                            </div>
                        </div>
                        <div style="color: {TEXT_SECONDARY}; font-size: 0.85rem;">
                            Category: {policy.get('category', 'N/A')} • Source: {policy.get('source', 'N/A')}
                        </div>
                    </div>
                    ''', unsafe_allow_html=True)

            # Query info - single-line to avoid Streamlit parsing issues
            st.markdown(f'<div style="{get_glassmorphism_css()}; padding: 1rem; margin-top: 1rem;"><div style="color: {TEXT_SECONDARY}; font-size: 0.85rem;"><strong>Query:</strong> {result.query}</div><div style="color: {TEXT_SECONDARY}; font-size: 0.8rem; margin-top: 0.5rem;">Processed at: {result.timestamp}</div></div>', unsafe_allow_html=True)

    # Example queries
    st.markdown(f'''
    <div style="{get_glassmorphism_css()}; padding: 1rem; margin-top: 1.5rem;">
        <h5 style="color: {GOLD}; margin-bottom: 0.75rem;">Example Queries</h5>
        <p style="color: {TEXT_SECONDARY}; font-size: 0.85rem; margin-bottom: 0.5rem;">
            Try these example queries to test different tier classifications:
        </p>
    </div>
    ''', unsafe_allow_html=True)

    # Get PA info for context-aware examples
    pa_info = governance_engine.get_pa_info()
    if pa_info:
        # Generate contextual examples based on PA scope and prohibitions
        scope_items = pa_info.get('scope', [])
        prohibitions = pa_info.get('prohibitions', [])

        tier_1_example = prohibitions[0] if prohibitions else "Example prohibited query"
        tier_2_example = scope_items[0] if scope_items else "Example scoped query"
        tier_3_example = "What is the weather today?"

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f'''
            <div style="{get_glassmorphism_css(TIER_1)}; padding: 0.75rem;">
                <div style="color: {TIER_1}; font-weight: 700; font-size: 0.9rem; margin-bottom: 0.5rem;">
                    Tier 1 (Block)
                </div>
                <div style="color: {TEXT_SECONDARY}; font-size: 0.8rem;">
                    {tier_1_example}
                </div>
            </div>
            ''', unsafe_allow_html=True)

        with col2:
            st.markdown(f'''
            <div style="{get_glassmorphism_css(TIER_2)}; padding: 0.75rem;">
                <div style="color: {TIER_2}; font-weight: 700; font-size: 0.9rem; margin-bottom: 0.5rem;">
                    Tier 2 (RAG)
                </div>
                <div style="color: {TEXT_SECONDARY}; font-size: 0.8rem;">
                    What are the requirements for {tier_2_example}?
                </div>
            </div>
            ''', unsafe_allow_html=True)

        with col3:
            st.markdown(f'''
            <div style="{get_glassmorphism_css(TIER_3)}; padding: 0.75rem;">
                <div style="color: {TIER_3}; font-weight: 700; font-size: 0.9rem; margin-bottom: 0.5rem;">
                    Tier 3 (Escalate)
                </div>
                <div style="color: {TEXT_SECONDARY}; font-size: 0.8rem;">
                    {tier_3_example}
                </div>
            </div>
            ''', unsafe_allow_html=True)
