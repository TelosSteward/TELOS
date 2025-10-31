"""
Counterfactual Viewer Component
================================

Display Phase 2/2B intervention results for current turn.

Shows counterfactual comparison when governance intervention occurred:
- Original trajectory final fidelity
- TELOS-governed trajectory final fidelity
- ΔF (governance impact)
- Effectiveness assessment
"""

import streamlit as st
from typing import Optional, Dict, Any, List
from pathlib import Path
import sys

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from telos_observatory.teloscope_v2.utils.phase2_loader import Phase2Loader


def render_counterfactual_viewer(conversation_id: str, turn_index: int):
    """
    Render counterfactual comparison for a specific turn.

    Args:
        conversation_id: Conversation/study ID
        turn_index: Turn index (0-based)

    Displays:
    - Phase 2: Single intervention if this was drift turn
    - Phase 2B: All interventions at this turn
    """

    # Load Phase 2 and Phase 2B data
    loader = Phase2Loader()

    # Try to find study
    studies = loader.get_available_studies()
    matching_studies = [s for s in studies if s.conversation_id == conversation_id]

    if not matching_studies:
        st.info("No counterfactual data available for this conversation.")
        return

    # Check both Phase 2 and Phase 2B
    phase2_study = next((s for s in matching_studies if s.phase == '2'), None)
    phase2b_study = next((s for s in matching_studies if s.phase == '2B'), None)

    rendered_any = False

    # Phase 2: Check if this was the drift turn
    if phase2_study and phase2_study.drift_turn == turn_index:
        render_phase2_intervention(phase2_study)
        rendered_any = True

    # Phase 2B: Check if there was an intervention at this turn
    if phase2b_study:
        intervention = find_intervention_at_turn(phase2b_study, turn_index)
        if intervention:
            if rendered_any:
                st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
            render_phase2b_intervention(intervention, turn_index)
            rendered_any = True

    if not rendered_any:
        st.caption("No governance intervention at this turn.")
        st.caption("*Turn remained within acceptable fidelity bounds.*")


def render_phase2_intervention(study):
    """Render Phase 2 single intervention results."""

    st.markdown("""
        <div style="
            background: rgba(102, 126, 234, 0.1);
            padding: 0.75rem;
            border-radius: 6px;
            border: 1px solid rgba(102, 126, 234, 0.3);
            margin-bottom: 0.75rem;
        ">
            <div style="font-weight: bold; color: #667eea; margin-bottom: 0.5rem;">
                📊 Phase 2 Intervention
            </div>
    """, unsafe_allow_html=True)

    # Drift detection
    st.caption(f"**Drift Detected:** Turn {study.drift_turn + 1}")
    st.caption(f"**Trigger Fidelity:** {study.drift_fidelity:.3f}")

    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)

    # Counterfactual comparison
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
            <div style="text-align: center;">
                <div style="color: #888; font-size: 0.7rem;">ORIGINAL</div>
                <div style="font-size: 1.2rem; font-weight: bold; color: #FF6B6B;">
                    {:.3f}
                </div>
            </div>
        """.format(study.delta_f if study.delta_f else 0.0), unsafe_allow_html=True)

    with col2:
        telos_final = (study.delta_f or 0) + (study.drift_fidelity or 0)
        st.markdown("""
            <div style="text-align: center;">
                <div style="color: #888; font-size: 0.7rem;">TELOS</div>
                <div style="font-size: 1.2rem; font-weight: bold; color: #51CF66;">
                    {:.3f}
                </div>
            </div>
        """.format(telos_final), unsafe_allow_html=True)

    # Delta F
    if study.delta_f is not None:
        delta_color = "#51CF66" if study.delta_f > 0 else "#FF6B6B"
        delta_icon = "✅" if study.governance_effective else "❌"

        st.markdown(f"""
            <div style="text-align: center; margin-top: 0.5rem;">
                <div style="color: #888; font-size: 0.7rem;">GOVERNANCE IMPACT</div>
                <div style="font-size: 1.5rem; font-weight: bold; color: {delta_color};">
                    {delta_icon} ΔF: {study.delta_f:+.3f}
                </div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


def render_phase2b_intervention(intervention: Dict[str, Any], turn_index: int):
    """Render Phase 2B intervention results."""

    st.markdown("""
        <div style="
            background: rgba(118, 227, 131, 0.1);
            padding: 0.75rem;
            border-radius: 6px;
            border: 1px solid rgba(118, 227, 131, 0.3);
            margin-bottom: 0.75rem;
        ">
            <div style="font-weight: bold; color: #76E383; margin-bottom: 0.5rem;">
                🌿 Phase 2B Continuous Intervention
            </div>
    """, unsafe_allow_html=True)

    # Trigger info
    st.caption(f"**Trigger Turn:** {turn_index + 1}")
    st.caption(f"**Trigger Fidelity:** {intervention.get('trigger_fidelity', 0):.3f}")
    st.caption(f"**Branch ID:** {intervention.get('branch_id', 'unknown')}")

    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)

    # Counterfactual comparison
    col1, col2 = st.columns(2)

    original_final_f = intervention.get('original_final_f', 0)
    telos_final_f = intervention.get('telos_final_f', 0)

    with col1:
        st.markdown(f"""
            <div style="text-align: center;">
                <div style="color: #888; font-size: 0.7rem;">ORIGINAL</div>
                <div style="font-size: 1.2rem; font-weight: bold; color: #FF6B6B;">
                    {original_final_f:.3f}
                </div>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
            <div style="text-align: center;">
                <div style="color: #888; font-size: 0.7rem;">TELOS</div>
                <div style="font-size: 1.2rem; font-weight: bold; color: #51CF66;">
                    {telos_final_f:.3f}
                </div>
            </div>
        """, unsafe_allow_html=True)

    # Delta F
    delta_f = intervention.get('delta_f', 0)
    governance_effective = intervention.get('governance_effective', 'False') == 'True'
    delta_color = "#51CF66" if delta_f > 0 else "#FF6B6B"
    delta_icon = "✅" if governance_effective else "❌"

    st.markdown(f"""
        <div style="text-align: center; margin-top: 0.5rem;">
            <div style="color: #888; font-size: 0.7rem;">GOVERNANCE IMPACT</div>
            <div style="font-size: 1.5rem; font-weight: bold; color: {delta_color};">
                {delta_icon} ΔF: {delta_f:+.3f}
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


def find_intervention_at_turn(study, turn_index: int) -> Optional[Dict[str, Any]]:
    """
    Find intervention that occurred at specific turn in Phase 2B study.

    Args:
        study: StudyMetadata with Phase 2B data
        turn_index: Turn index (0-based)

    Returns:
        Intervention dictionary or None
    """
    if not study.interventions:
        return None

    for intervention in study.interventions:
        # trigger_turn is 1-based in the data, convert to 0-based
        if intervention.get('trigger_turn', 0) - 1 == turn_index:
            return intervention

    return None


def render_compact_counterfactual(conversation_id: str, turn_index: int):
    """
    Render compact counterfactual display (for TELOSCOPE Remote).

    Args:
        conversation_id: Conversation/study ID
        turn_index: Turn index (0-based)
    """

    loader = Phase2Loader()
    studies = loader.get_available_studies()
    matching_studies = [s for s in studies if s.conversation_id == conversation_id]

    if not matching_studies:
        return

    # Check for interventions
    has_intervention = False
    delta_f = None

    # Check Phase 2
    phase2_study = next((s for s in matching_studies if s.phase == '2'), None)
    if phase2_study and phase2_study.drift_turn == turn_index:
        has_intervention = True
        delta_f = phase2_study.delta_f

    # Check Phase 2B
    phase2b_study = next((s for s in matching_studies if s.phase == '2B'), None)
    if phase2b_study:
        intervention = find_intervention_at_turn(phase2b_study, turn_index)
        if intervention:
            has_intervention = True
            delta_f = intervention.get('delta_f')

    if has_intervention and delta_f is not None:
        delta_color = "#51CF66" if delta_f > 0 else "#FF6B6B"
        st.markdown(f"""
            <div style="
                display: inline-flex;
                align-items: center;
                background: rgba(118, 227, 131, 0.1);
                padding: 0.25rem 0.5rem;
                border-radius: 4px;
                font-size: 0.85rem;
            ">
                <span style="color: #888;">ΔF:</span>
                <span style="color: {delta_color}; font-weight: bold; margin-left: 0.25rem;">
                    {delta_f:+.3f}
                </span>
            </div>
        """, unsafe_allow_html=True)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def has_intervention_at_turn(conversation_id: str, turn_index: int) -> bool:
    """
    Check if there's any intervention at a specific turn.

    Args:
        conversation_id: Conversation/study ID
        turn_index: Turn index (0-based)

    Returns:
        True if intervention exists, False otherwise
    """
    loader = Phase2Loader()
    studies = loader.get_available_studies()
    matching_studies = [s for s in studies if s.conversation_id == conversation_id]

    if not matching_studies:
        return False

    # Check Phase 2
    phase2_study = next((s for s in matching_studies if s.phase == '2'), None)
    if phase2_study and phase2_study.drift_turn == turn_index:
        return True

    # Check Phase 2B
    phase2b_study = next((s for s in matching_studies if s.phase == '2B'), None)
    if phase2b_study:
        intervention = find_intervention_at_turn(phase2b_study, turn_index)
        if intervention:
            return True

    return False
