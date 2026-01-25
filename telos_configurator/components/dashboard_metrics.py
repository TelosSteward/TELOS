"""
Dashboard Metrics Component
============================

Live metrics dashboard for active governance.
Shows query statistics, tier distribution, and recent activity.
"""

import sys

import streamlit as st
import pandas as pd

sys.path.insert(0, '/Users/brunnerjf/Desktop/TELOS_Master/telos_configurator')
from config.styles import (
    GOLD, TIER_1, TIER_2, TIER_3, BG_ELEVATED, TEXT_PRIMARY, TEXT_SECONDARY,
    get_glassmorphism_css, render_section_header, with_opacity
)

sys.path.insert(0, '/Users/brunnerjf/Desktop/TELOS_Master/telos_configurator')
from engine.governance_engine import GovernanceEngine


def render_dashboard_metrics(governance_engine: GovernanceEngine) -> None:
    """
    Render live governance metrics dashboard.

    Args:
        governance_engine: GovernanceEngine instance
    """
    st.markdown(render_section_header("Governance Metrics"), unsafe_allow_html=True)

    if not governance_engine.is_active():
        st.markdown(f'''
        <div style="{get_glassmorphism_css()}; padding: 1.5rem;">
            <p style="color: {TEXT_SECONDARY}; text-align: center;">
                Activate governance to see metrics and statistics.
            </p>
        </div>
        ''', unsafe_allow_html=True)
        return

    # Get statistics
    stats = governance_engine.get_statistics()

    if stats['total_queries'] == 0:
        st.markdown(f'''
        <div style="{get_glassmorphism_css()}; padding: 1.5rem;">
            <p style="color: {TEXT_SECONDARY}; text-align: center;">
                No queries processed yet. Use the Test Query interface to start testing.
            </p>
        </div>
        ''', unsafe_allow_html=True)
        return

    # Overall metrics
    st.markdown(f'''
    <div style="{get_glassmorphism_css(GOLD)}; padding: 1.5rem; margin-bottom: 1rem;">
        <h4 style="color: {GOLD}; margin-bottom: 1rem;">Overall Statistics</h4>
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; text-align: center;">
            <div>
                <div style="color: {GOLD}; font-size: 2rem; font-weight: 700;">
                    {stats['total_queries']}
                </div>
                <div style="color: {TEXT_SECONDARY}; font-size: 0.85rem;">
                    Total Queries
                </div>
            </div>
            <div>
                <div style="color: {TEXT_PRIMARY}; font-size: 2rem; font-weight: 700;">
                    {stats['avg_fidelity']:.3f}
                </div>
                <div style="color: {TEXT_SECONDARY}; font-size: 0.85rem;">
                    Avg Fidelity
                </div>
            </div>
            <div>
                <div style="color: {TEXT_PRIMARY}; font-size: 2rem; font-weight: 700;">
                    {stats['min_fidelity']:.3f}
                </div>
                <div style="color: {TEXT_SECONDARY}; font-size: 0.85rem;">
                    Min Fidelity
                </div>
            </div>
            <div>
                <div style="color: {TEXT_PRIMARY}; font-size: 2rem; font-weight: 700;">
                    {stats['max_fidelity']:.3f}
                </div>
                <div style="color: {TEXT_SECONDARY}; font-size: 0.85rem;">
                    Max Fidelity
                </div>
            </div>
        </div>
    </div>
    ''', unsafe_allow_html=True)

    # Tier distribution
    tier_counts = stats['tier_distribution']
    tier_percentages = stats['tier_percentages']

    st.markdown(f'''
    <div style="{get_glassmorphism_css()}; padding: 1.5rem; margin-bottom: 1rem;">
        <h4 style="color: {GOLD}; margin-bottom: 1rem;">Tier Distribution</h4>
    </div>
    ''', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f'''
        <div style="{get_glassmorphism_css(TIER_1)}; padding: 1.5rem; text-align: center;">
            <div style="color: {TIER_1}; font-size: 1.8rem; font-weight: 700;">
                {tier_counts[1]}
            </div>
            <div style="color: {TEXT_SECONDARY}; font-size: 0.9rem; margin-top: 0.25rem;">
                Tier 1: PA Blocks
            </div>
            <div style="color: {TIER_1}; font-size: 1.2rem; font-weight: 700; margin-top: 0.5rem;">
                {tier_percentages[1]}
            </div>
        </div>
        ''', unsafe_allow_html=True)

    with col2:
        st.markdown(f'''
        <div style="{get_glassmorphism_css(TIER_2)}; padding: 1.5rem; text-align: center;">
            <div style="color: {TIER_2}; font-size: 1.8rem; font-weight: 700;">
                {tier_counts[2]}
            </div>
            <div style="color: {TEXT_SECONDARY}; font-size: 0.9rem; margin-top: 0.25rem;">
                Tier 2: RAG Retrievals
            </div>
            <div style="color: {TIER_2}; font-size: 1.2rem; font-weight: 700; margin-top: 0.5rem;">
                {tier_percentages[2]}
            </div>
        </div>
        ''', unsafe_allow_html=True)

    with col3:
        st.markdown(f'''
        <div style="{get_glassmorphism_css(TIER_3)}; padding: 1.5rem; text-align: center;">
            <div style="color: {TIER_3}; font-size: 1.8rem; font-weight: 700;">
                {tier_counts[3]}
            </div>
            <div style="color: {TEXT_SECONDARY}; font-size: 0.9rem; margin-top: 0.25rem;">
                Tier 3: Escalations
            </div>
            <div style="color: {TIER_3}; font-size: 1.2rem; font-weight: 700; margin-top: 0.5rem;">
                {tier_percentages[3]}
            </div>
        </div>
        ''', unsafe_allow_html=True)

    # Tier distribution chart
    st.markdown(f'''
    <div style="{get_glassmorphism_css()}; padding: 1rem; margin-top: 1rem;">
        <h5 style="color: {GOLD}; margin-bottom: 0.75rem;">Distribution Chart</h5>
    </div>
    ''', unsafe_allow_html=True)

    # Create bar chart data
    chart_data = pd.DataFrame({
        'Tier': ['Tier 1\nPA Block', 'Tier 2\nRAG Policy', 'Tier 3\nEscalation'],
        'Count': [tier_counts[1], tier_counts[2], tier_counts[3]],
        'Color': [TIER_1, TIER_2, TIER_3]
    })

    # Use Streamlit bar chart
    st.bar_chart(
        chart_data,
        x='Tier',
        y='Count',
        color='Color',
        use_container_width=True
    )

    # Recent queries log
    st.markdown(f'''
    <div style="{get_glassmorphism_css()}; padding: 1rem; margin-top: 1.5rem;">
        <h4 style="color: {GOLD}; margin-bottom: 0.75rem;">Recent Queries</h4>
    </div>
    ''', unsafe_allow_html=True)

    # Get recent queries from audit log
    audit_log = governance_engine.export_audit_log()
    recent_queries = audit_log[-10:]  # Last 10 queries

    if recent_queries:
        for result in reversed(recent_queries):  # Most recent first
            tier_color = TIER_1 if result['tier'] == 1 else (TIER_2 if result['tier'] == 2 else TIER_3)

            st.markdown(f'''
            <div style="{get_glassmorphism_css(tier_color)}; padding: 0.75rem; margin-bottom: 0.5rem;">
                <div style="display: flex; justify-content: space-between; align-items: start;">
                    <div style="flex: 1;">
                        <div style="color: {TEXT_PRIMARY}; font-size: 0.95rem;">
                            {result['query'][:100]}{'...' if len(result['query']) > 100 else ''}
                        </div>
                        <div style="color: {TEXT_SECONDARY}; font-size: 0.8rem; margin-top: 0.25rem;">
                            {result['timestamp'][:19].replace('T', ' ')}
                        </div>
                    </div>
                    <div style="margin-left: 1rem; text-align: right;">
                        <div style="color: {tier_color}; font-weight: 700; font-size: 0.9rem;">
                            {result['tier_name']}
                        </div>
                        <div style="color: {TEXT_SECONDARY}; font-size: 0.8rem;">
                            Fidelity: {result['fidelity']:.3f}
                        </div>
                    </div>
                </div>
            </div>
            ''', unsafe_allow_html=True)
    else:
        st.markdown(f'''
        <div style="{get_glassmorphism_css()}; padding: 1rem;">
            <p style="color: {TEXT_SECONDARY}; text-align: center;">
                No queries in log yet.
            </p>
        </div>
        ''', unsafe_allow_html=True)
