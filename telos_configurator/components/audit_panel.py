"""
Audit Panel Component
======================

Audit log viewer and export interface.
Provides filtering, searching, and export capabilities for governance decisions.
"""

import sys
import json
from datetime import datetime

import streamlit as st
import pandas as pd

sys.path.insert(0, '/Users/brunnerjf/Desktop/TELOS_Master/telos_configurator')
from config.styles import (
    GOLD, TIER_1, TIER_2, TIER_3, BG_ELEVATED, TEXT_PRIMARY, TEXT_SECONDARY,
    get_glassmorphism_css, render_section_header, render_info_box
)

sys.path.insert(0, '/Users/brunnerjf/Desktop/TELOS_Master/telos_configurator')
from engine.governance_engine import GovernanceEngine


def render_audit_panel(governance_engine: GovernanceEngine) -> None:
    """
    Render audit log viewer and export interface.

    Args:
        governance_engine: GovernanceEngine instance
    """
    st.markdown(render_section_header("Audit Log"), unsafe_allow_html=True)

    if not governance_engine.is_active():
        st.markdown(f'''
        <div style="{get_glassmorphism_css()}; padding: 1.5rem;">
            <p style="color: {TEXT_SECONDARY}; text-align: center;">
                Activate governance to view audit logs.
            </p>
        </div>
        ''', unsafe_allow_html=True)
        return

    # Get audit log
    audit_log = governance_engine.export_audit_log()

    if not audit_log:
        st.markdown(f'''
        <div style="{get_glassmorphism_css()}; padding: 1.5rem;">
            <p style="color: {TEXT_SECONDARY}; text-align: center;">
                No queries in audit log yet. Process some queries to see them here.
            </p>
        </div>
        ''', unsafe_allow_html=True)
        return

    st.markdown(f'''
    <div style="{get_glassmorphism_css()}; padding: 1rem; margin-bottom: 1rem;">
        <p style="color: {TEXT_SECONDARY};">
            View and filter all governance decisions. Export the audit log for compliance and analysis.
        </p>
    </div>
    ''', unsafe_allow_html=True)

    # Filter controls
    st.markdown(f'''
    <div style="{get_glassmorphism_css(GOLD)}; padding: 1rem; margin-bottom: 1rem;">
        <h4 style="color: {GOLD}; margin-bottom: 0.75rem;">Filters</h4>
    </div>
    ''', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # Tier filter
        tier_filter = st.multiselect(
            "Filter by Tier",
            options=[1, 2, 3],
            default=[1, 2, 3],
            format_func=lambda x: f"Tier {x}",
            help="Select tiers to display"
        )

    with col2:
        # Search query
        search_query = st.text_input(
            "Search Queries",
            placeholder="Search by query text...",
            help="Filter by query text content"
        )

    # Apply filters
    filtered_log = audit_log

    if tier_filter:
        filtered_log = [entry for entry in filtered_log if entry['tier'] in tier_filter]

    if search_query:
        search_lower = search_query.lower()
        filtered_log = [entry for entry in filtered_log if search_lower in entry['query'].lower()]

    # Stats for filtered log
    st.markdown(f'''
    <div style="{get_glassmorphism_css()}; padding: 0.75rem; margin-bottom: 1rem;">
        <div style="color: {TEXT_SECONDARY}; font-size: 0.9rem; text-align: center;">
            Showing {len(filtered_log)} of {len(audit_log)} entries
        </div>
    </div>
    ''', unsafe_allow_html=True)

    # Display log entries
    if filtered_log:
        st.markdown(f'''
        <div style="{get_glassmorphism_css()}; padding: 1rem; margin-bottom: 1rem;">
            <h4 style="color: {GOLD}; margin-bottom: 0.75rem;">Log Entries</h4>
        </div>
        ''', unsafe_allow_html=True)

        # Create DataFrame for better display
        df_data = []
        for entry in filtered_log:
            tier_color = TIER_1 if entry['tier'] == 1 else (TIER_2 if entry['tier'] == 2 else TIER_3)

            df_data.append({
                'Timestamp': entry['timestamp'][:19].replace('T', ' '),
                'Query': entry['query'][:80] + ('...' if len(entry['query']) > 80 else ''),
                'Fidelity': f"{entry['fidelity']:.4f}",
                'Tier': entry['tier'],
                'Action': entry['action'],
                'Policies': len(entry.get('retrieved_policies', []))
            })

        df = pd.DataFrame(df_data)

        # Display as table
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                'Timestamp': st.column_config.TextColumn('Timestamp', width='medium'),
                'Query': st.column_config.TextColumn('Query', width='large'),
                'Fidelity': st.column_config.TextColumn('Fidelity', width='small'),
                'Tier': st.column_config.NumberColumn('Tier', width='small'),
                'Action': st.column_config.TextColumn('Action', width='medium'),
                'Policies': st.column_config.NumberColumn('Policies', width='small')
            }
        )

        # Detailed view of selected entries
        with st.expander("View Detailed Entries"):
            for idx, entry in enumerate(reversed(filtered_log[-20:]), 1):  # Last 20 entries
                tier_color = TIER_1 if entry['tier'] == 1 else (TIER_2 if entry['tier'] == 2 else TIER_3)

                st.markdown(f'''
                <div style="{get_glassmorphism_css(tier_color)}; padding: 1rem; margin-bottom: 1rem;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.75rem;">
                        <div style="color: {GOLD}; font-weight: 700;">
                            Entry #{len(filtered_log) - idx + 1}
                        </div>
                        <div style="color: {TEXT_SECONDARY}; font-size: 0.85rem;">
                            {entry['timestamp'][:19].replace('T', ' ')}
                        </div>
                    </div>

                    <div style="margin-bottom: 0.5rem;">
                        <div style="color: {TEXT_SECONDARY}; font-size: 0.85rem;">Query:</div>
                        <div style="color: {TEXT_PRIMARY}; font-size: 0.95rem;">
                            {entry['query']}
                        </div>
                    </div>

                    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.5rem; margin-top: 0.75rem;">
                        <div>
                            <div style="color: {TEXT_SECONDARY}; font-size: 0.75rem;">Fidelity</div>
                            <div style="color: {tier_color}; font-weight: 700;">{entry['fidelity']:.4f}</div>
                        </div>
                        <div>
                            <div style="color: {TEXT_SECONDARY}; font-size: 0.75rem;">Tier</div>
                            <div style="color: {tier_color}; font-weight: 700;">{entry['tier_name']}</div>
                        </div>
                        <div>
                            <div style="color: {TEXT_SECONDARY}; font-size: 0.75rem;">Action</div>
                            <div style="color: {TEXT_PRIMARY}; font-weight: 600;">{entry['action']}</div>
                        </div>
                    </div>
                </div>
                ''', unsafe_allow_html=True)

                # Show blocking reason if present
                if entry.get('blocking_reason'):
                    st.markdown(f'''
                    <div style="{get_glassmorphism_css()}; padding: 0.75rem; margin-bottom: 0.5rem; margin-left: 1rem;">
                        <div style="color: {TEXT_SECONDARY}; font-size: 0.85rem;">Reason:</div>
                        <div style="color: {TEXT_SECONDARY}; font-size: 0.9rem; font-style: italic;">
                            {entry['blocking_reason']}
                        </div>
                    </div>
                    ''', unsafe_allow_html=True)

                # Show retrieved policies if present
                if entry.get('retrieved_policies'):
                    policies_html = ''.join([
                        f"<li>{policy.get('title', 'Untitled')} (similarity: {policy.get('similarity', 0):.3f})</li>"
                        for policy in entry['retrieved_policies']
                    ])
                    st.markdown(f'''
                    <div style="{get_glassmorphism_css()}; padding: 0.75rem; margin-bottom: 0.5rem; margin-left: 1rem;">
                        <div style="color: {TEXT_SECONDARY}; font-size: 0.85rem;">
                            Retrieved Policies ({len(entry['retrieved_policies'])}):
                        </div>
                        <ul style="color: {TEXT_PRIMARY}; font-size: 0.85rem; margin-top: 0.25rem;">
                            {policies_html}
                        </ul>
                    </div>
                    ''', unsafe_allow_html=True)

    else:
        st.markdown(
            render_info_box("No entries match the current filters.", "info"),
            unsafe_allow_html=True
        )

    # Export and management controls
    st.markdown("---")

    st.markdown(f'''
    <div style="{get_glassmorphism_css()}; padding: 1rem; margin-top: 1rem;">
        <h4 style="color: {GOLD}; margin-bottom: 0.75rem;">Log Management</h4>
    </div>
    ''', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # Export audit log
        if st.button("Export Audit Log", use_container_width=True):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"audit_log_{timestamp}.json"

            # Export filtered or full log
            export_data = {
                "exported_at": datetime.now().isoformat(),
                "total_entries": len(audit_log),
                "filtered_entries": len(filtered_log),
                "filters": {
                    "tiers": tier_filter,
                    "search": search_query
                },
                "entries": filtered_log
            }

            # Convert to JSON
            json_data = json.dumps(export_data, indent=2, ensure_ascii=False)

            st.download_button(
                label="Download JSON",
                data=json_data,
                file_name=filename,
                mime="application/json",
                use_container_width=True
            )

            st.markdown(
                render_info_box(f"✓ Audit log prepared for download ({len(filtered_log)} entries)", "success"),
                unsafe_allow_html=True
            )

    with col2:
        # Clear log with confirmation
        if st.button("Clear Audit Log", type="secondary", use_container_width=True):
            st.session_state.confirm_clear = True

    # Confirmation dialog
    if st.session_state.get('confirm_clear', False):
        st.markdown(f'''
        <div style="{get_glassmorphism_css(TIER_1)}; padding: 1rem; margin-top: 1rem;">
            <div style="color: {TIER_1}; font-weight: 700; margin-bottom: 0.5rem;">
                ⚠ Confirm Clear Audit Log
            </div>
            <div style="color: {TEXT_SECONDARY}; font-size: 0.9rem;">
                This will permanently delete all {len(audit_log)} audit log entries. This action cannot be undone.
            </div>
        </div>
        ''', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Yes, Clear Log", type="primary", use_container_width=True):
                governance_engine.clear_log()
                st.session_state.confirm_clear = False
                st.markdown(
                    render_info_box("✓ Audit log cleared", "success"),
                    unsafe_allow_html=True
                )
                st.rerun()

        with col2:
            if st.button("Cancel", use_container_width=True):
                st.session_state.confirm_clear = False
                st.rerun()
