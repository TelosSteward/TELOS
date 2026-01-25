"""
Threshold Configuration Component
==================================

Configure three-tier governance thresholds with visual tier zone diagram.
"""

import sys

import streamlit as st

sys.path.insert(0, '/Users/brunnerjf/Desktop/TELOS_Master/telos_configurator')
from config.styles import (
    GOLD, TIER_1, TIER_2, TIER_3, BG_ELEVATED, TEXT_PRIMARY, TEXT_SECONDARY,
    get_glassmorphism_css, render_section_header, with_opacity
)

sys.path.insert(0, '/Users/brunnerjf/Desktop/TELOS_Master/telos_configurator')
from engine.governance_engine import ThresholdConfig


def render_threshold_config() -> ThresholdConfig:
    """
    Render threshold configuration interface with visual tier zones.

    Returns:
        ThresholdConfig instance with current slider values
    """
    st.markdown(render_section_header("Threshold Configuration"), unsafe_allow_html=True)

    st.markdown(f'''
    <div style="{get_glassmorphism_css()}; padding: 1.5rem; margin-bottom: 1rem;">
        <p style="color: {TEXT_SECONDARY};">
            Configure fidelity thresholds that define the three-tier governance zones.
            These thresholds determine when queries are blocked, retrieve policies, or escalate to experts.
        </p>
    </div>
    ''', unsafe_allow_html=True)

    # Initialize session state for thresholds
    if 'thresholds_configured' not in st.session_state:
        st.session_state.thresholds_configured = False

    # Check if domain template has thresholds
    if 'selected_domain' in st.session_state and st.session_state.selected_domain:
        from .domain_selector import get_domain_template
        template = get_domain_template(st.session_state.selected_domain)

        if template and template.get('thresholds') and st.session_state.selected_domain != 'custom':
            default_thresholds = template['thresholds']
        else:
            default_thresholds = {
                'tier_1': 0.65,
                'tier_2_lower': 0.35,
                'rag_relevance': 0.50
            }
    else:
        default_thresholds = {
            'tier_1': 0.65,
            'tier_2_lower': 0.35,
            'rag_relevance': 0.50
        }

    # Threshold sliders
    st.markdown(f'''
    <div style="{get_glassmorphism_css(GOLD)}; padding: 1.5rem; margin-bottom: 1rem;">
        <h4 style="color: {GOLD}; margin-bottom: 1rem;">Tier Thresholds</h4>
    </div>
    ''', unsafe_allow_html=True)

    # Tier 1 threshold
    st.markdown(f"**Tier 1 Threshold** <span style='color: {TIER_1}; font-weight: 700;'>(PA Block)</span>", unsafe_allow_html=True)
    tier_1_threshold = st.slider(
        "Tier 1 explanation",
        min_value=0.0,
        max_value=1.0,
        value=default_thresholds['tier_1'],
        step=0.05,
        help="Queries with fidelity >= this value trigger PA blocking (high fidelity indicates drift/violation)",
        label_visibility="collapsed"
    )
    st.caption(f"Fidelity ≥ {tier_1_threshold:.2f} → PA Block (Drift Detected)")

    st.markdown("---")

    # Tier 2 lower threshold
    st.markdown(f"**Tier 2 Lower Threshold** <span style='color: {TIER_2}; font-weight: 700;'>(RAG Zone)</span>", unsafe_allow_html=True)
    tier_2_lower = st.slider(
        "Tier 2 lower explanation",
        min_value=0.0,
        max_value=1.0,
        value=default_thresholds['tier_2_lower'],
        step=0.05,
        help="Lower bound for RAG policy retrieval zone",
        label_visibility="collapsed"
    )
    st.caption(f"{tier_2_lower:.2f} ≤ Fidelity < {tier_1_threshold:.2f} → RAG Policy Retrieval")

    st.markdown("---")

    # RAG relevance threshold
    st.markdown("**RAG Relevance Threshold**", unsafe_allow_html=True)
    rag_relevance = st.slider(
        "RAG relevance explanation",
        min_value=0.0,
        max_value=1.0,
        value=default_thresholds['rag_relevance'],
        step=0.05,
        help="Minimum similarity score for retrieved policies to be included",
        label_visibility="collapsed"
    )
    st.caption(f"Retrieved policies must have similarity ≥ {rag_relevance:.2f}")

    # Create ThresholdConfig
    thresholds = ThresholdConfig(
        tier_1_threshold=tier_1_threshold,
        tier_2_lower=tier_2_lower,
        tier_2_upper=tier_1_threshold,  # Upper bound matches tier 1
        rag_relevance=rag_relevance
    )

    # Validate
    is_valid, error = thresholds.validate()

    if not is_valid:
        st.error(f"Invalid threshold configuration: {error}")
        return thresholds

    # Visual tier zone diagram
    st.markdown(f'''
    <div style="{get_glassmorphism_css(GOLD)}; padding: 1.5rem; margin-top: 1.5rem;">
        <h4 style="color: {GOLD}; margin-bottom: 1rem;">Tier Zone Visualization</h4>
    </div>
    ''', unsafe_allow_html=True)

    # Calculate zone heights (as percentages)
    tier_1_height = (1.0 - tier_1_threshold) * 100
    tier_2_height = (tier_1_threshold - tier_2_lower) * 100
    tier_3_height = tier_2_lower * 100

    st.markdown(f'''
    <div style="{get_glassmorphism_css()}; padding: 1rem;">
        <div style="width: 100%; height: 300px; position: relative; border: 1px solid {with_opacity(GOLD, 0.3)}; border-radius: 8px;">

            <!-- Tier 1 Zone -->
            <div style="
                position: absolute;
                bottom: {tier_3_height + tier_2_height}%;
                left: 0;
                right: 0;
                height: {tier_1_height}%;
                background: linear-gradient(180deg, {with_opacity(TIER_1, 0.3)} 0%, {with_opacity(TIER_1, 0.1)} 100%);
                border-bottom: 2px solid {TIER_1};
                display: flex;
                align-items: center;
                justify-content: center;
            ">
                <div style="text-align: center;">
                    <div style="color: {TIER_1}; font-weight: 700; font-size: 1.1rem;">Tier 1: PA Block</div>
                    <div style="color: {TEXT_SECONDARY}; font-size: 0.85rem;">Fidelity ≥ {tier_1_threshold:.2f}</div>
                </div>
            </div>

            <!-- Tier 2 Zone -->
            <div style="
                position: absolute;
                bottom: {tier_3_height}%;
                left: 0;
                right: 0;
                height: {tier_2_height}%;
                background: linear-gradient(180deg, {with_opacity(TIER_2, 0.3)} 0%, {with_opacity(TIER_2, 0.1)} 100%);
                border-bottom: 2px solid {TIER_2};
                display: flex;
                align-items: center;
                justify-content: center;
            ">
                <div style="text-align: center;">
                    <div style="color: {TIER_2}; font-weight: 700; font-size: 1.1rem;">Tier 2: RAG Policy</div>
                    <div style="color: {TEXT_SECONDARY}; font-size: 0.85rem;">{tier_2_lower:.2f} ≤ Fidelity < {tier_1_threshold:.2f}</div>
                </div>
            </div>

            <!-- Tier 3 Zone -->
            <div style="
                position: absolute;
                bottom: 0;
                left: 0;
                right: 0;
                height: {tier_3_height}%;
                background: linear-gradient(180deg, {with_opacity(TIER_3, 0.3)} 0%, {with_opacity(TIER_3, 0.1)} 100%);
                display: flex;
                align-items: center;
                justify-content: center;
            ">
                <div style="text-align: center;">
                    <div style="color: {TIER_3}; font-weight: 700; font-size: 1.1rem;">Tier 3: Expert Escalation</div>
                    <div style="color: {TEXT_SECONDARY}; font-size: 0.85rem;">Fidelity < {tier_2_lower:.2f}</div>
                </div>
            </div>

            <!-- Fidelity axis labels -->
            <div style="position: absolute; left: -40px; top: 0; color: {TEXT_SECONDARY}; font-size: 0.8rem;">1.0</div>
            <div style="position: absolute; left: -40px; bottom: 0; color: {TEXT_SECONDARY}; font-size: 0.8rem;">0.0</div>
        </div>
    </div>
    ''', unsafe_allow_html=True)

    # Explanation text
    st.markdown(f'''
    <div style="{get_glassmorphism_css()}; padding: 1rem; margin-top: 1rem;">
        <h5 style="color: {GOLD}; margin-bottom: 0.75rem;">How Tiers Work</h5>
        <ul style="color: {TEXT_SECONDARY}; line-height: 1.8;">
            <li><span style="color: {TIER_1}; font-weight: 700;">Tier 1 (PA Block):</span> High fidelity indicates drift/violation - query conflicts with PA scope or prohibitions and is blocked.</li>
            <li><span style="color: {TIER_2}; font-weight: 700;">Tier 2 (RAG Policy):</span> Medium fidelity triggers policy retrieval - query is within scope but needs corpus guidance.</li>
            <li><span style="color: {TIER_3}; font-weight: 700;">Tier 3 (Expert Escalation):</span> Low fidelity indicates off-topic - query is escalated to human experts for review.</li>
        </ul>
    </div>
    ''', unsafe_allow_html=True)

    st.session_state.thresholds_configured = True

    return thresholds
