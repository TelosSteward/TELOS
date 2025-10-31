"""
Mathematical Breakdown Component
=================================

Display high-level mathematical metrics for current turn.

Shows:
- Embedding distance from attractor
- Fidelity calculation result
- Threshold comparison
- Status indicator (within basin or not)
"""

import streamlit as st
from typing import Dict, Any, Optional


def render_math_breakdown(turn_data: Dict[str, Any]):
    """
    Render mathematical breakdown for a turn.

    Args:
        turn_data: Turn data dictionary with metrics

    Expected turn_data structure:
    {
        'fidelity': 0.850,
        'distance': 0.234,
        'threshold': 0.800,
        'within_basin': True,
        'drift_detected': False
    }
    """

    # Extract metrics with defaults
    fidelity = turn_data.get('fidelity')
    distance = turn_data.get('distance')
    threshold = turn_data.get('threshold', 0.800)
    within_basin = turn_data.get('within_basin', True)
    drift_detected = turn_data.get('drift_detected', False)

    # Status indicator
    if drift_detected:
        status_text = "⚠️ DRIFT DETECTED"
        status_color = "#FF6B6B"
    elif within_basin:
        status_text = "✅ WITHIN BASIN"
        status_color = "#51CF66"
    else:
        status_text = "⚡ MONITORING"
        status_color = "#FFD700"

    st.markdown(f"""
        <div style="
            background: rgba(255, 215, 0, 0.05);
            padding: 1rem;
            border-radius: 6px;
            border: 1px solid rgba(255, 215, 0, 0.1);
        ">
            <div style="
                font-size: 0.9rem;
                font-weight: bold;
                color: {status_color};
                margin-bottom: 0.75rem;
            ">{status_text}</div>
    """, unsafe_allow_html=True)

    # Fidelity metric
    if fidelity is not None:
        fidelity_color = "#51CF66" if fidelity >= threshold else "#FF6B6B"
        st.markdown(f"""
            <div style="margin-bottom: 0.5rem;">
                <div style="color: #888; font-size: 0.75rem;">FIDELITY SCORE</div>
                <div style="font-size: 1.5rem; font-weight: bold; color: {fidelity_color};">
                    {fidelity:.3f}
                </div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.caption("*Fidelity: Not yet calculated*")

    # Distance metric
    if distance is not None:
        st.markdown(f"""
            <div style="margin-bottom: 0.5rem;">
                <div style="color: #888; font-size: 0.75rem;">EMBEDDING DISTANCE</div>
                <div style="font-size: 1.2rem; font-weight: bold; color: #FFD700;">
                    {distance:.4f}
                </div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.caption("*Distance: Not yet calculated*")

    # Threshold comparison
    if fidelity is not None:
        threshold_status = "PASS" if fidelity >= threshold else "FAIL"
        threshold_color = "#51CF66" if fidelity >= threshold else "#FF6B6B"

        st.markdown(f"""
            <div style="margin-bottom: 0.5rem;">
                <div style="color: #888; font-size: 0.75rem;">THRESHOLD</div>
                <div style="font-size: 1rem; color: #888;">
                    {threshold:.3f}
                    <span style="color: {threshold_color}; font-weight: bold;">
                        ({threshold_status})
                    </span>
                </div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Additional metrics (if available)
    if 'attractor_purpose' in turn_data or 'governance_status' in turn_data:
        st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)

        # Attractor info
        if 'attractor_purpose' in turn_data:
            st.caption("**Attractor:** " + turn_data['attractor_purpose'])

        # Governance status
        if 'governance_status' in turn_data:
            gov_status = turn_data['governance_status']
            st.caption(f"**Governance:** {gov_status}")


def render_compact_math_breakdown(turn_data: Dict[str, Any]):
    """
    Render compact version for narrow displays (TELOSCOPE Remote).

    Args:
        turn_data: Turn data dictionary with metrics
    """

    fidelity = turn_data.get('fidelity')
    drift_detected = turn_data.get('drift_detected', False)

    if fidelity is not None:
        threshold = turn_data.get('threshold', 0.800)
        fidelity_color = "#51CF66" if fidelity >= threshold else "#FF6B6B"
        status_icon = "⚠️" if drift_detected else "✅"

        st.markdown(f"""
            <div style="
                display: inline-flex;
                align-items: center;
                background: rgba(255, 215, 0, 0.1);
                padding: 0.25rem 0.5rem;
                border-radius: 4px;
                font-size: 0.85rem;
            ">
                <span style="margin-right: 0.25rem;">{status_icon}</span>
                <span style="color: #888;">F:</span>
                <span style="color: {fidelity_color}; font-weight: bold; margin-left: 0.25rem;">
                    {fidelity:.3f}
                </span>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.caption("*Metrics pending*")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def calculate_fidelity_display(fidelity: Optional[float], threshold: float = 0.800) -> Dict[str, Any]:
    """
    Calculate display properties for fidelity metric.

    Args:
        fidelity: Fidelity score (0-1)
        threshold: Threshold for passing (default: 0.800)

    Returns:
        Dictionary with display properties
    """
    if fidelity is None:
        return {
            'value': None,
            'color': '#888',
            'status': 'PENDING',
            'status_color': '#888',
            'passes': None
        }

    passes = fidelity >= threshold
    color = "#51CF66" if passes else "#FF6B6B"
    status = "PASS" if passes else "FAIL"
    status_color = "#51CF66" if passes else "#FF6B6B"

    return {
        'value': fidelity,
        'color': color,
        'status': status,
        'status_color': status_color,
        'passes': passes
    }


def format_distance(distance: Optional[float], precision: int = 4) -> str:
    """
    Format distance metric for display.

    Args:
        distance: Distance value
        precision: Decimal precision (default: 4)

    Returns:
        Formatted string
    """
    if distance is None:
        return "N/A"

    return f"{distance:.{precision}f}"
