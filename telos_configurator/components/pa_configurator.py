"""
PA Configurator Component
==========================

Primacy Attractor configuration interface.
Allows users to define name, purpose, scope, exclusions, and prohibitions.
"""

import sys
from typing import Optional

import streamlit as st

sys.path.insert(0, '/Users/brunnerjf/Desktop/TELOS_Master/telos_configurator')
from config.styles import (
    GOLD, STATUS_GOOD, STATE_PENDING, BG_ELEVATED, TEXT_PRIMARY, TEXT_SECONDARY,
    get_glassmorphism_css, render_section_header, render_status_badge, render_info_box
)

sys.path.insert(0, '/Users/brunnerjf/Desktop/TELOS_Master/telos_configurator')
from engine.governance_engine import PrimacyAttractor, create_pa, embed_pa


def render_pa_configurator() -> Optional[PrimacyAttractor]:
    """
    Render Primacy Attractor configuration interface.

    Returns:
        PrimacyAttractor instance if configured and embedded, None otherwise
    """
    st.markdown(render_section_header("Primacy Attractor Configuration"), unsafe_allow_html=True)

    st.markdown(f'''
    <div style="{get_glassmorphism_css()}; padding: 1.5rem; margin-bottom: 1rem;">
        <p style="color: {TEXT_SECONDARY};">
            The Primacy Attractor (PA) defines the purpose, scope, and boundaries of your governance framework.
            It serves as the mathematical reference point for measuring query fidelity.
        </p>
    </div>
    ''', unsafe_allow_html=True)

    # Initialize session state
    if 'pa_configured' not in st.session_state:
        st.session_state.pa_configured = False
        st.session_state.pa_instance = None

    # Check if domain template is selected
    if 'selected_domain' in st.session_state and st.session_state.selected_domain:
        domain = st.session_state.selected_domain

        # Import domain templates
        from .domain_selector import get_domain_template
        template = get_domain_template(domain)

        # If domain has a PA template, offer to use it
        if template and template.get('pa_template') and domain != 'custom':
            st.markdown(
                render_info_box(
                    f"Template available for {template['name']}. You can use the template or customize it below.",
                    "info"
                ),
                unsafe_allow_html=True
            )

            if st.button("Load Domain Template", type="primary"):
                pa_template = template['pa_template']
                st.session_state.pa_name = pa_template['name']
                st.session_state.pa_purpose = pa_template['purpose']
                st.session_state.pa_scope = "\n".join(pa_template['scope'])
                st.session_state.pa_exclusions = "\n".join(pa_template['exclusions'])
                st.session_state.pa_prohibitions = "\n".join(pa_template['prohibitions'])
                st.rerun()

    # PA configuration form
    st.markdown(f'''
    <div style="{get_glassmorphism_css(GOLD)}; padding: 1.5rem; margin-bottom: 1rem;">
        <h4 style="color: {GOLD}; margin-bottom: 1rem;">PA Definition</h4>
    </div>
    ''', unsafe_allow_html=True)

    # Name
    name = st.text_input(
        "PA Name",
        value=st.session_state.get('pa_name', ''),
        placeholder="e.g., Healthcare HIPAA Compliance",
        help="A descriptive name for this Primacy Attractor"
    )

    # Purpose statement
    purpose = st.text_area(
        "Purpose Statement",
        value=st.session_state.get('pa_purpose', ''),
        placeholder="e.g., Ensure all healthcare conversations comply with HIPAA regulations and protect patient privacy",
        help="A clear statement of the PA's primary purpose",
        height=100
    )

    # Scope (multiline)
    st.markdown(f"**Scope** <span style='color: {TEXT_SECONDARY}; font-size: 0.85rem;'>(one item per line)</span>", unsafe_allow_html=True)
    scope_text = st.text_area(
        "Scope items",
        value=st.session_state.get('pa_scope', ''),
        placeholder="patient privacy\nmedical data protection\nhealthcare compliance",
        help="Topics and areas within the PA's scope (one per line)",
        height=100,
        label_visibility="collapsed"
    )

    # Exclusions (multiline)
    st.markdown(f"**Exclusions** <span style='color: {TEXT_SECONDARY}; font-size: 0.85rem;'>(one item per line)</span>", unsafe_allow_html=True)
    exclusions_text = st.text_area(
        "Exclusion items",
        value=st.session_state.get('pa_exclusions', ''),
        placeholder="general medical information\npublic health data\nnon-PHI content",
        help="Topics explicitly out of scope (one per line)",
        height=100,
        label_visibility="collapsed"
    )

    # Prohibitions (multiline)
    st.markdown(f"**Prohibitions** <span style='color: {TEXT_SECONDARY}; font-size: 0.85rem;'>(one item per line)</span>", unsafe_allow_html=True)
    prohibitions_text = st.text_area(
        "Prohibition items",
        value=st.session_state.get('pa_prohibitions', ''),
        placeholder="disclosing PHI without authorization\nunauthorized data sharing\nHIPAA violations",
        help="Actions and topics that are strictly prohibited (one per line)",
        height=100,
        label_visibility="collapsed"
    )

    # Parse text inputs to lists
    scope = [s.strip() for s in scope_text.split('\n') if s.strip()]
    exclusions = [e.strip() for e in exclusions_text.split('\n') if e.strip()]
    prohibitions = [p.strip() for p in prohibitions_text.split('\n') if p.strip()]

    # Generate PA embedding button
    st.markdown("---")

    col1, col2 = st.columns([3, 1])

    with col2:
        generate_button = st.button(
            "Generate PA Embedding",
            type="primary",
            use_container_width=True,
            disabled=not (name and purpose and scope)
        )

    if generate_button:
        if not (name and purpose and scope):
            st.markdown(
                render_info_box(
                    "Please provide at minimum: Name, Purpose, and Scope items",
                    "warning"
                ),
                unsafe_allow_html=True
            )
        else:
            with st.spinner("Creating and embedding Primacy Attractor..."):
                # Create PA
                pa = create_pa(
                    name=name,
                    purpose=purpose,
                    scope=scope,
                    exclusions=exclusions,
                    prohibitions=prohibitions
                )

                # Embed PA
                success = embed_pa(pa)

                if success:
                    st.session_state.pa_configured = True
                    st.session_state.pa_instance = pa

                    # Save inputs to session state
                    st.session_state.pa_name = name
                    st.session_state.pa_purpose = purpose
                    st.session_state.pa_scope = scope_text
                    st.session_state.pa_exclusions = exclusions_text
                    st.session_state.pa_prohibitions = prohibitions_text

                    st.markdown(
                        render_info_box(
                            "✓ Primacy Attractor created and embedded successfully",
                            "success"
                        ),
                        unsafe_allow_html=True
                    )
                    st.rerun()
                else:
                    st.markdown(
                        render_info_box(
                            "✗ Failed to embed PA. Ensure Ollama is running with nomic-embed-text model.",
                            "error"
                        ),
                        unsafe_allow_html=True
                    )

    # PA status indicator
    if st.session_state.pa_configured and st.session_state.pa_instance:
        pa = st.session_state.pa_instance

        st.markdown(f'''
        <div style="{get_glassmorphism_css(GOLD)}; padding: 1rem; margin-top: 1rem;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <div style="color: {GOLD}; font-weight: 700; font-size: 1.1rem;">
                        {pa.name}
                    </div>
                    <div style="color: {TEXT_SECONDARY}; font-size: 0.85rem; margin-top: 0.25rem;">
                        {len(pa.scope)} scope items • {len(pa.exclusions)} exclusions • {len(pa.prohibitions)} prohibitions
                    </div>
                </div>
                <div>
                    {render_status_badge('good', '✓ Embedded')}
                </div>
            </div>
        </div>
        ''', unsafe_allow_html=True)

        return pa

    return None
