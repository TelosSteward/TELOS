"""
Study Browser Component
========================

Browse and select Phase 2 validation studies in Observatory UI.

Features:
- List all completed studies
- Filter by dataset
- Sort by metrics (ΔF, turns, effectiveness)
- Select study for detailed view
"""

import streamlit as st
from typing import Optional, List
from ..utils.phase2_loader import Phase2Loader, StudyMetadata


class StudyBrowser:
    """Interactive browser for Phase 2 validation studies."""

    def __init__(self, loader: Optional[Phase2Loader] = None):
        """
        Initialize browser.

        Args:
            loader: Phase2Loader instance (creates new if None)
        """
        self.loader = loader or Phase2Loader()

    def render(self) -> Optional[StudyMetadata]:
        """
        Render study browser interface.

        Returns:
            Selected StudyMetadata or None
        """
        st.markdown("### 📚 Phase 2 Validation Studies")

        # Get all studies
        all_studies = self.loader.get_available_studies()

        if not all_studies:
            st.warning("No Phase 2 studies found. Run validation pipeline first.")
            return None

        # Filter controls
        col1, col2, col3 = st.columns([2, 2, 2])

        with col1:
            dataset_filter = st.selectbox(
                "Dataset",
                options=['All', 'ShareGPT', 'Test Sessions', 'Edge Cases', 'Phase 2B'],
                index=0
            )

        with col2:
            effectiveness_filter = st.selectbox(
                "Effectiveness",
                options=['All', 'Effective (ΔF > 0)', 'Ineffective (ΔF ≤ 0)'],
                index=0
            )

        with col3:
            sort_by = st.selectbox(
                "Sort By",
                options=['ΔF (High to Low)', 'ΔF (Low to High)', 'Conversation ID', 'Total Turns'],
                index=0
            )

        # Apply filters
        filtered_studies = self._apply_filters(
            all_studies,
            dataset_filter,
            effectiveness_filter
        )

        # Apply sorting
        sorted_studies = self._apply_sorting(filtered_studies, sort_by)

        # Display count
        st.caption(f"Showing {len(sorted_studies)} of {len(all_studies)} studies")

        # Study list
        st.markdown("---")

        if not sorted_studies:
            st.info("No studies match the selected filters.")
            return None

        # Render study cards
        selected_study = None

        for study in sorted_studies:
            if self._render_study_card(study):
                selected_study = study
                break  # Only one selection at a time

        return selected_study

    def _apply_filters(
        self,
        studies: List[StudyMetadata],
        dataset_filter: str,
        effectiveness_filter: str
    ) -> List[StudyMetadata]:
        """Apply filter criteria to studies list."""

        filtered = studies.copy()

        # Dataset filter
        if dataset_filter != 'All':
            dataset_map = {
                'ShareGPT': 'sharegpt',
                'Test Sessions': 'test_sessions',
                'Edge Cases': 'edge_cases',
                'Phase 2B': 'phase2b'
            }
            target_dataset = dataset_map[dataset_filter]
            filtered = [s for s in filtered if s.dataset == target_dataset]

        # Effectiveness filter
        if effectiveness_filter != 'All':
            if effectiveness_filter == 'Effective (ΔF > 0)':
                filtered = [s for s in filtered if s.governance_effective]
            else:  # Ineffective
                filtered = [s for s in filtered if not s.governance_effective]

        return filtered

    def _apply_sorting(
        self,
        studies: List[StudyMetadata],
        sort_by: str
    ) -> List[StudyMetadata]:
        """Apply sorting to studies list."""

        if sort_by == 'ΔF (High to Low)':
            return sorted(studies, key=lambda s: s.delta_f or -999, reverse=True)
        elif sort_by == 'ΔF (Low to High)':
            return sorted(studies, key=lambda s: s.delta_f or 999)
        elif sort_by == 'Conversation ID':
            return sorted(studies, key=lambda s: s.conversation_id)
        elif sort_by == 'Total Turns':
            return sorted(studies, key=lambda s: s.total_turns, reverse=True)
        else:
            return studies

    def _render_study_card(self, study: StudyMetadata) -> bool:
        """
        Render a single study card (supports Phase 2 and Phase 2B).

        Args:
            study: StudyMetadata to render

        Returns:
            True if study was selected, False otherwise
        """
        # Card container
        with st.container():
            # Different layout for Phase 2B vs Phase 2
            if study.phase == '2B':
                # Phase 2B: Show total interventions and aggregate metrics
                col1, col2, col3, col4, col5 = st.columns([3, 1, 1, 1, 1])

                with col1:
                    # Conversation ID with Phase 2B badge
                    st.markdown(f"**🌿 {study.conversation_id}** <span style='color: #888; font-size: 0.8em'>[Phase 2B]</span>", unsafe_allow_html=True)

                with col2:
                    st.caption(f"{study.total_turns} turns")

                with col3:
                    # Show total interventions
                    st.caption(f"⚡ {study.total_interventions} interventions")

                with col4:
                    # Mean ΔF with color
                    if study.delta_f is not None:
                        color = "green" if study.delta_f > 0 else "red"
                        st.markdown(f"<span style='color: {color}'>μΔF: {study.delta_f:+.3f}</span>", unsafe_allow_html=True)

                with col5:
                    # Effectiveness rate
                    if study.aggregate_metrics:
                        eff_rate = study.aggregate_metrics.get('effectiveness_rate', 0)
                        st.caption(f"✓ {eff_rate*100:.0f}%")

            else:
                # Phase 2: Original single-intervention display
                col1, col2, col3, col4, col5 = st.columns([3, 1, 1, 1, 1])

                with col1:
                    # Conversation ID with dataset badge
                    dataset_emoji = {
                        'sharegpt': '💬',
                        'test_sessions': '🧪',
                        'edge_cases': '⚠️'
                    }
                    emoji = dataset_emoji.get(study.dataset, '📄')
                    st.markdown(f"**{emoji} {study.conversation_id}**")

                with col2:
                    st.caption(f"{study.total_turns} turns")

                with col3:
                    # ΔF with color
                    if study.delta_f is not None:
                        color = "green" if study.delta_f > 0 else "red"
                        st.markdown(f"<span style='color: {color}'>ΔF: {study.delta_f:+.3f}</span>", unsafe_allow_html=True)
                    else:
                        st.caption("No drift")

                with col4:
                    # Effectiveness badge
                    if study.governance_effective is not None:
                        badge = "✅" if study.governance_effective else "❌"
                        st.markdown(f"{badge}")

                with col5:
                    pass  # Empty for alignment

            # View button (same for both)
            with col5:
                button_key = f"view_{study.conversation_id}_{study.dataset}_{study.phase}"
                if st.button("View", key=button_key, type="primary"):
                    return True

            st.markdown("---")

        return False

    def render_compact(self, max_studies: int = 10) -> Optional[StudyMetadata]:
        """
        Render compact version for sidebar or narrow columns.

        Args:
            max_studies: Maximum studies to show

        Returns:
            Selected StudyMetadata or None
        """
        st.markdown("#### 📚 Studies")

        all_studies = self.loader.get_available_studies()

        if not all_studies:
            st.caption("No studies found")
            return None

        # Simple dropdown selector
        study_options = [
            f"{s.conversation_id} (ΔF: {s.delta_f:+.3f})" if s.delta_f else s.conversation_id
            for s in all_studies[:max_studies]
        ]

        if len(all_studies) > max_studies:
            study_options.append(f"... +{len(all_studies) - max_studies} more")

        selected_index = st.selectbox(
            "Select Study",
            options=range(len(study_options)),
            format_func=lambda i: study_options[i],
            label_visibility="collapsed"
        )

        if selected_index < len(all_studies):
            return all_studies[selected_index]

        return None


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def render_study_browser(loader: Optional[Phase2Loader] = None) -> Optional[StudyMetadata]:
    """
    Convenience function to render study browser.

    Args:
        loader: Optional Phase2Loader instance

    Returns:
        Selected StudyMetadata or None
    """
    browser = StudyBrowser(loader)
    return browser.render()


if __name__ == '__main__':
    """Test the browser component."""
    st.set_page_config(page_title="Study Browser Test", layout="wide")

    st.title("Study Browser Component Test")

    browser = StudyBrowser()
    selected = browser.render()

    if selected:
        st.success(f"Selected: {selected.conversation_id}")
        st.json({
            'id': selected.conversation_id,
            'dataset': selected.dataset,
            'delta_f': selected.delta_f,
            'effective': selected.governance_effective,
            'turns': selected.total_turns
        })
