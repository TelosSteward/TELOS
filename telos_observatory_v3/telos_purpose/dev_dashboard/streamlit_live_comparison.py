"""
TELOSCOPE Observatory - Complete Web Interface
===============================================

Production-ready 4-tab dashboard for AI governance observation through
counterfactual experimentation.

Features:
- Live conversation monitoring with real-time drift detection
- Session replay with timeline scrubber
- Counterfactual evidence viewer (TELOSCOPE)
- Aggregate analytics dashboard

TELOSCOPE = Telically Entrained Linguistic Operational Substrate
                Counterfactual Observation via Purpose-scoped Experimentation
"""

import streamlit as st
import streamlit.components.v1 as components
from telos_purpose.sessions.web_session import WebSessionManager
from telos_purpose.core.session_state import SessionStateManager
from telos_purpose.core.counterfactual_branch_manager import CounterfactualBranchManager
from telos_purpose.core.counterfactual_simulator import CounterfactualBranchSimulator
from telos_purpose.sessions.live_interceptor import LiveInterceptor
from telos_purpose.validation.branch_comparator import BranchComparator
from telos_purpose.llm_clients.mistral_client import TelosMistralClient
from telos_purpose.core.embedding_provider import EmbeddingProvider
from telos_purpose.core.unified_steward import UnifiedGovernanceSteward, PrimacyAttractor
from telos_purpose.profiling.progressive_primacy_extractor import ProgressivePrimacyExtractor
from telos_purpose.dev_dashboard.steward_analysis import StewardAnalyzer
from telos_purpose.dev_dashboard.observation_deck.deck_manager import DeckManager
from telos_purpose.dev_dashboard.observation_deck.deck_control_strip import DeckControlStrip
import os
import json
import pandas as pd
from pathlib import Path
import numpy as np
from datetime import datetime
import csv
import zipfile
import io
import logging
import glob
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    st.warning("⚠️ Plotly not installed. Charts will be limited. Install with: pip install plotly")

# ============================================================================
# Performance: Cached Data Operations
# ============================================================================

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_session_statistics_cached(turns_data: str) -> Dict[str, Any]:
    """
    Performance: Cached expensive computation of session statistics.

    Args:
        turns_data: JSON string of turns (used for cache key)

    Returns:
        Dict with avg_fidelity, total_interventions, basin_crossings
    """
    turns = json.loads(turns_data)

    fidelities = []
    interventions = 0
    basin_crossings = 0

    for turn in turns:
        metrics = turn.get('metrics', {})
        fidelity = metrics.get('telic_fidelity', 1.0)
        fidelities.append(fidelity)

        # Count interventions
        metadata = turn.get('metadata', {})
        if metadata.get('intervention_applied', False):
            interventions += 1

        # Count basin crossings (fidelity < 0.73 = Minor Drift threshold)
        if fidelity < 0.73:
            basin_crossings += 1

    return {
        'avg_fidelity': sum(fidelities) / len(fidelities) if fidelities else 0.0,
        'total_interventions': interventions,
        'basin_crossings': basin_crossings,
        'total_turns': len(turns)
    }


@st.cache_data(ttl=60)  # Cache for 1 minute
def get_all_turns_cached(session_id: str) -> List[Dict[str, Any]]:
    """
    Performance: Cache turn retrieval to reduce session manager calls.

    Args:
        session_id: Session identifier for cache invalidation

    Returns:
        List of turn dictionaries
    """
    if 'current_session' in st.session_state:
        return st.session_state.current_session.get('turns', [])
    return []


@st.cache_data
def prepare_export_data(turns_json: str, format_type: str) -> str:
    """
    Performance: Cache export data preparation to avoid recomputation.

    Args:
        turns_json: JSON string of turns (used for cache key)
        format_type: Export format ('csv', 'json', 'html')

    Returns:
        Formatted export string
    """
    turns = json.loads(turns_json)

    if format_type == 'csv':
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=['turn', 'user', 'assistant', 'fidelity'])
        writer.writeheader()
        for i, turn in enumerate(turns):
            writer.writerow({
                'turn': i + 1,
                'user': turn.get('user_input', ''),
                'assistant': turn.get('assistant_response', ''),
                'fidelity': turn.get('metrics', {}).get('telic_fidelity', 1.0)
            })
        return output.getvalue()

    elif format_type == 'json':
        return json.dumps(turns, indent=2)

    return str(turns)


@st.cache_data
def prepare_fidelity_chart_data(turns_json: str) -> Dict[str, List]:
    """
    Performance: Cache fidelity trend computation for charts.

    Args:
        turns_json: JSON string of turns

    Returns:
        Dict with x (turn numbers) and y (fidelity values) lists
    """
    turns = json.loads(turns_json)
    fidelities = [t.get('metrics', {}).get('telic_fidelity', 1.0) for t in turns]

    return {
        'x': list(range(1, len(fidelities) + 1)),
        'y': fidelities
    }


# ============================================================================
# Keyboard Shortcuts Handler
# ============================================================================

def render_keyboard_handler():
    """
    Render invisible keyboard event listener using JavaScript.

    Keyboard shortcuts:
    - ESC: Toggle Steward Lens
    - Spacebar: Toggle TELOSCOPE window
    - Up Arrow: Show TELOSCOPE Tools (expand TELOSCOPE)
    - Down Arrow: Hide all windows
    - Left Arrow: Previous turn (navigate backward)
    - Right Arrow: Next turn (navigate forward)
    """
    keyboard_html = """
    <script>
    // Keyboard event handler for TELOS Observatory
    let lastKey = null;
    let keyPressTime = Date.now();

    document.addEventListener('keydown', function(e) {
        // Only handle if not typing in an input field
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
            return;
        }

        let handled = false;
        let action = null;

        switch(e.key) {
            case 'Escape':
                action = 'toggle_steward_lens';
                handled = true;
                break;
            case ' ':  // Spacebar
                action = 'toggle_teloscope';
                handled = true;
                break;
            case 'ArrowUp':
                action = 'show_tools';
                handled = true;
                break;
            case 'ArrowDown':
                action = 'hide_all_windows';
                handled = true;
                break;
            case 'ArrowLeft':
                action = 'prev_turn';
                handled = true;
                break;
            case 'ArrowRight':
                action = 'next_turn';
                handled = true;
                break;
        }

        if (handled) {
            e.preventDefault();
            lastKey = action;
            keyPressTime = Date.now();

            // Send to Streamlit
            window.parent.postMessage({
                type: 'streamlit:setComponentValue',
                value: {action: action, timestamp: keyPressTime}
            }, '*');
        }
    });

    // Heartbeat to keep component alive
    setInterval(function() {
        if (lastKey) {
            window.parent.postMessage({
                type: 'streamlit:setComponentValue',
                value: {action: lastKey, timestamp: keyPressTime}
            }, '*');
        }
    }, 100);
    </script>
    <div style="display:none;">Keyboard handler active</div>
    """

    # Render the component and get the return value
    key_event = components.html(keyboard_html, height=0)

    # Process keyboard events
    if key_event and isinstance(key_event, dict):
        action = key_event.get('action')
        timestamp = key_event.get('timestamp', 0)

        # Debounce: only process if this is a new event
        last_processed = st.session_state.get('last_key_timestamp', 0)
        if timestamp > last_processed:
            st.session_state.last_key_timestamp = timestamp

            # Handle the action
            if action == 'toggle_steward_lens':
                st.session_state.show_steward_lens = not st.session_state.get('show_steward_lens', False)
                st.rerun()

            elif action == 'toggle_teloscope':
                st.session_state.show_teloscope_window = not st.session_state.get('show_teloscope_window', False)
                st.rerun()

            elif action == 'show_tools':
                st.session_state.show_teloscope_tools = not st.session_state.get('show_teloscope_tools', False)
                st.rerun()

            elif action == 'hide_all_windows':
                st.session_state.show_steward_lens = False
                st.session_state.show_teloscope_window = False
                st.session_state.show_teloscope_tools = False
                st.rerun()

            elif action == 'prev_turn':
                # Navigate to previous turn
                current_idx = st.session_state.get('current_turn_index', 0)
                if current_idx > 0:
                    st.session_state.current_turn_index = current_idx - 1
                    st.rerun()

            elif action == 'next_turn':
                # Navigate to next turn
                turns = st.session_state.get('current_session', {}).get('turns', [])
                current_idx = st.session_state.get('current_turn_index', len(turns) - 1)
                if current_idx < len(turns) - 1:
                    st.session_state.current_turn_index = current_idx + 1
                    st.rerun()


# ============================================================================
# Cross-Session Analytics Functions (Phase 9)
# ============================================================================

def load_session_files(export_dir: str = 'purpose_protocol_exports', max_sessions: int = 10) -> List[Dict[str, Any]]:
    """
    Load session export files for cross-session analytics.

    Args:
        export_dir: Directory containing session export files
        max_sessions: Maximum number of recent sessions to load

    Returns:
        List of session data dictionaries
    """
    session_pattern = os.path.join(export_dir, 'session_*.json')
    session_files = sorted(glob.glob(session_pattern), key=os.path.getmtime, reverse=True)

    sessions_data = []
    for file_path in session_files[:max_sessions]:
        try:
            with open(file_path, 'r') as f:
                session_data = json.load(f)
                sessions_data.append(session_data)
        except Exception as e:
            logging.warning(f"Failed to load session file {file_path}: {e}")
            continue

    return sessions_data


def compute_avg_fidelity_across_sessions(sessions: List[Dict[str, Any]]) -> float:
    """
    Compute average fidelity across all sessions.

    Args:
        sessions: List of session data dictionaries

    Returns:
        Average fidelity value
    """
    all_fidelities = []
    for session in sessions:
        for turn in session.get('turns', []):
            metrics = turn.get('metrics', {})
            fidelity = metrics.get('telic_fidelity', 1.0)
            all_fidelities.append(fidelity)

    return sum(all_fidelities) / len(all_fidelities) if all_fidelities else 0.0


def count_interventions_across_sessions(sessions: List[Dict[str, Any]]) -> int:
    """
    Count total interventions across all sessions.

    Args:
        sessions: List of session data dictionaries

    Returns:
        Total number of interventions
    """
    total = 0
    for session in sessions:
        for turn in session.get('turns', []):
            metadata = turn.get('metadata', {})
            if metadata.get('intervention_applied', False):
                total += 1
    return total


def compute_avg_turns_per_session(sessions: List[Dict[str, Any]]) -> float:
    """
    Compute average number of turns per session.

    Args:
        sessions: List of session data dictionaries

    Returns:
        Average turns per session
    """
    if not sessions:
        return 0.0

    turn_counts = [len(session.get('turns', [])) for session in sessions]
    return sum(turn_counts) / len(turn_counts) if turn_counts else 0.0


def compute_session_avg_fidelity(session: Dict[str, Any]) -> float:
    """
    Compute average fidelity for a single session.

    Args:
        session: Session data dictionary

    Returns:
        Average fidelity for the session
    """
    fidelities = []
    for turn in session.get('turns', []):
        metrics = turn.get('metrics', {})
        fidelity = metrics.get('telic_fidelity', 1.0)
        fidelities.append(fidelity)

    return sum(fidelities) / len(fidelities) if fidelities else 0.0


def extract_session_fidelity_trends(sessions: List[Dict[str, Any]]) -> List[float]:
    """
    Extract average fidelity for each session.

    Args:
        sessions: List of session data dictionaries

    Returns:
        List of average fidelities (one per session)
    """
    return [compute_session_avg_fidelity(session) for session in sessions]


def generate_session_labels(sessions: List[Dict[str, Any]]) -> List[str]:
    """
    Generate human-readable labels for session selection.

    Args:
        sessions: List of session data dictionaries

    Returns:
        List of session labels with metadata
    """
    labels = []
    for idx, session in enumerate(sessions):
        session_metadata = session.get('session_metadata', {})
        session_id = session_metadata.get('session_id', f'Session {idx + 1}')
        started_at = session_metadata.get('started_at', '')
        total_turns = session_metadata.get('total_turns', len(session.get('turns', [])))

        # Format timestamp if available
        if started_at:
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(started_at)
                time_str = dt.strftime('%Y-%m-%d %H:%M')
            except:
                time_str = started_at[:16] if len(started_at) > 16 else started_at
        else:
            time_str = 'Unknown time'

        label = f"{session_id} ({time_str}, {total_turns} turns)"
        labels.append(label)

    return labels


def extract_session_metrics(session: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract aggregate metrics from a single session.

    Args:
        session: Session data dictionary

    Returns:
        Dict with metrics (avg_fidelity, interventions, turns, basin_violations)
    """
    turns = session.get('turns', [])

    fidelities = []
    interventions = 0
    basin_violations = 0

    for turn in turns:
        metrics = turn.get('metrics', {})
        fidelity = metrics.get('telic_fidelity', 1.0)
        fidelities.append(fidelity)

        # Count interventions
        metadata = turn.get('metadata', {})
        if metadata.get('intervention_applied', False):
            interventions += 1

        # Count basin violations (fidelity < 0.76) - Goldilocks: Aligned threshold
        if fidelity < 0.76:
            basin_violations += 1

    return {
        'avg_fidelity': sum(fidelities) / len(fidelities) if fidelities else 0.0,
        'interventions': interventions,
        'turns': len(turns),
        'basin_violations': basin_violations,
        'min_fidelity': min(fidelities) if fidelities else 0.0,
        'max_fidelity': max(fidelities) if fidelities else 0.0
    }


def extract_turn_fidelities(session: Dict[str, Any]) -> List[float]:
    """
    Extract turn-by-turn fidelity values from a session.

    Args:
        session: Session data dictionary

    Returns:
        List of fidelity values (one per turn)
    """
    turns = session.get('turns', [])
    return [turn.get('metrics', {}).get('telic_fidelity', 1.0) for turn in turns]


# ============================================================================
# Pattern Detection Functions (Phase 9)
# ============================================================================

def detect_intervention_patterns(sessions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Detect recurring intervention patterns across sessions.

    Identifies:
    - Sessions with high intervention rates (>50% of turns)
    - Sessions with low intervention rates (<10% of turns)
    - Average intervention rate across all sessions

    Args:
        sessions: List of session data dictionaries

    Returns:
        Dict with intervention pattern analysis
    """
    if not sessions:
        return {
            'high_intervention_sessions': [],
            'low_intervention_sessions': [],
            'avg_intervention_rate': 0.0,
            'total_interventions': 0,
            'total_turns': 0
        }

    session_intervention_rates = []
    total_interventions = 0
    total_turns = 0
    high_intervention_sessions = []
    low_intervention_sessions = []

    for idx, session in enumerate(sessions):
        turns = session.get('turns', [])
        if not turns:
            continue

        interventions = sum(
            1 for turn in turns
            if turn.get('metadata', {}).get('intervention_applied', False)
        )

        rate = interventions / len(turns) if len(turns) > 0 else 0.0
        session_intervention_rates.append(rate)
        total_interventions += interventions
        total_turns += len(turns)

        # Classify sessions
        session_metadata = session.get('session_metadata', {})
        session_id = session_metadata.get('session_id', f'Session {idx + 1}')

        if rate > 0.5:
            high_intervention_sessions.append({
                'session_id': session_id,
                'rate': rate,
                'interventions': interventions,
                'turns': len(turns),
                'idx': idx
            })
        elif rate < 0.1 and len(turns) >= 5:  # Only flag low if sufficient turns
            low_intervention_sessions.append({
                'session_id': session_id,
                'rate': rate,
                'interventions': interventions,
                'turns': len(turns),
                'idx': idx
            })

    avg_rate = total_interventions / total_turns if total_turns > 0 else 0.0

    return {
        'high_intervention_sessions': sorted(high_intervention_sessions, key=lambda x: x['rate'], reverse=True),
        'low_intervention_sessions': sorted(low_intervention_sessions, key=lambda x: x['rate']),
        'avg_intervention_rate': avg_rate,
        'total_interventions': total_interventions,
        'total_turns': total_turns,
        'session_rates': session_intervention_rates
    }


def detect_drift_triggers(sessions: List[Dict[str, Any]], threshold: float = 0.76) -> Dict[str, Any]:
    """
    Identify common drift triggers (turns with low fidelity).

    Args:
        sessions: List of session data dictionaries
        threshold: Fidelity threshold for drift detection (default: 0.76 - Goldilocks: Aligned)

    Returns:
        Dict with drift trigger analysis
    """
    if not sessions:
        return {
            'total_drift_events': 0,
            'drift_rate': 0.0,
            'sessions_with_drift': [],
            'avg_drifts_per_session': 0.0
        }

    total_drift_events = 0
    total_turns = 0
    sessions_with_drift = []

    for idx, session in enumerate(sessions):
        turns = session.get('turns', [])
        if not turns:
            continue

        drift_turns = []
        for turn_idx, turn in enumerate(turns):
            metrics = turn.get('metrics', {})
            fidelity = metrics.get('telic_fidelity', 1.0)

            if fidelity < threshold:
                drift_turns.append({
                    'turn_number': turn_idx,
                    'fidelity': fidelity,
                    'user_input': turn.get('user_message', '')[:100]  # First 100 chars
                })
                total_drift_events += 1

        total_turns += len(turns)

        if drift_turns:
            session_metadata = session.get('session_metadata', {})
            session_id = session_metadata.get('session_id', f'Session {idx + 1}')

            sessions_with_drift.append({
                'session_id': session_id,
                'drift_count': len(drift_turns),
                'drift_rate': len(drift_turns) / len(turns),
                'drift_turns': drift_turns[:5],  # Show first 5 drifts
                'idx': idx
            })

    drift_rate = total_drift_events / total_turns if total_turns > 0 else 0.0
    avg_drifts = total_drift_events / len(sessions) if sessions else 0.0

    return {
        'total_drift_events': total_drift_events,
        'drift_rate': drift_rate,
        'sessions_with_drift': sorted(sessions_with_drift, key=lambda x: x['drift_count'], reverse=True),
        'avg_drifts_per_session': avg_drifts,
        'threshold_used': threshold
    }


def analyze_governance_effectiveness(sessions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze governance effectiveness patterns.

    Measures how well interventions improve fidelity by comparing
    sessions with high vs low intervention rates.

    Args:
        sessions: List of session data dictionaries

    Returns:
        Dict with effectiveness analysis
    """
    if not sessions:
        return {
            'avg_fidelity_with_interventions': 0.0,
            'avg_fidelity_without_interventions': 0.0,
            'effectiveness_score': 0.0,
            'analysis': 'No data available'
        }

    # Collect fidelity for turns with and without interventions
    fidelity_with_intervention = []
    fidelity_without_intervention = []

    for session in sessions:
        turns = session.get('turns', [])
        for turn in turns:
            metrics = turn.get('metrics', {})
            metadata = turn.get('metadata', {})
            fidelity = metrics.get('telic_fidelity', 1.0)
            intervention = metadata.get('intervention_applied', False)

            if intervention:
                fidelity_with_intervention.append(fidelity)
            else:
                fidelity_without_intervention.append(fidelity)

    # Calculate averages
    avg_with = (sum(fidelity_with_intervention) / len(fidelity_with_intervention)
                if fidelity_with_intervention else 0.0)
    avg_without = (sum(fidelity_without_intervention) / len(fidelity_without_intervention)
                   if fidelity_without_intervention else 0.0)

    # Effectiveness score: positive if interventions improve fidelity
    effectiveness = avg_with - avg_without

    # Generate analysis text
    if effectiveness > 0.1:
        analysis = f"Interventions are highly effective (+{effectiveness:.3f} fidelity improvement)"
    elif effectiveness > 0.01:
        analysis = f"Interventions show moderate effectiveness (+{effectiveness:.3f} improvement)"
    elif effectiveness > -0.01:
        analysis = "Interventions have neutral effect on fidelity"
    else:
        analysis = f"Interventions may be counterproductive ({effectiveness:.3f} fidelity decrease)"

    return {
        'avg_fidelity_with_interventions': avg_with,
        'avg_fidelity_without_interventions': avg_without,
        'effectiveness_score': effectiveness,
        'intervention_count': len(fidelity_with_intervention),
        'non_intervention_count': len(fidelity_without_intervention),
        'analysis': analysis
    }


def detect_anomalous_sessions(sessions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Detect anomalous sessions using statistical outlier detection.

    Identifies sessions with unusual metrics compared to the norm:
    - Unusually low average fidelity
    - Unusually high drift rate
    - Unusually high intervention rate

    Args:
        sessions: List of session data dictionaries

    Returns:
        Dict with anomaly detection results
    """
    if len(sessions) < 3:
        return {
            'anomalies': [],
            'message': 'Need at least 3 sessions for anomaly detection'
        }

    import statistics

    # Extract metrics for all sessions
    session_metrics = []
    for idx, session in enumerate(sessions):
        metrics = extract_session_metrics(session)
        session_metadata = session.get('session_metadata', {})

        session_metrics.append({
            'idx': idx,
            'session_id': session_metadata.get('session_id', f'Session {idx + 1}'),
            'avg_fidelity': metrics['avg_fidelity'],
            'intervention_rate': metrics['interventions'] / metrics['turns'] if metrics['turns'] > 0 else 0.0,
            'basin_violation_rate': metrics['basin_violations'] / metrics['turns'] if metrics['turns'] > 0 else 0.0,
            'turns': metrics['turns']
        })

    # Calculate statistics
    fidelities = [s['avg_fidelity'] for s in session_metrics]
    intervention_rates = [s['intervention_rate'] for s in session_metrics]
    violation_rates = [s['basin_violation_rate'] for s in session_metrics]

    # Use median and MAD for robust outlier detection
    median_fidelity = statistics.median(fidelities)
    median_intervention = statistics.median(intervention_rates)
    median_violations = statistics.median(violation_rates)

    # Calculate MAD (Median Absolute Deviation)
    mad_fidelity = statistics.median([abs(f - median_fidelity) for f in fidelities])
    mad_intervention = statistics.median([abs(i - median_intervention) for i in intervention_rates])
    mad_violations = statistics.median([abs(v - median_violations) for v in violation_rates])

    # Detect anomalies (using 2.5 MAD threshold)
    threshold = 2.5
    anomalies = []

    for session_data in session_metrics:
        reasons = []

        # Check fidelity anomaly (low fidelity)
        if mad_fidelity > 0:
            z_fidelity = (session_data['avg_fidelity'] - median_fidelity) / (mad_fidelity * 1.4826)
            if z_fidelity < -threshold:
                reasons.append(f"Unusually low fidelity (F={session_data['avg_fidelity']:.3f})")

        # Check intervention rate anomaly (high intervention)
        if mad_intervention > 0:
            z_intervention = (session_data['intervention_rate'] - median_intervention) / (mad_intervention * 1.4826)
            if abs(z_intervention) > threshold:
                reasons.append(f"Unusual intervention rate ({session_data['intervention_rate']:.1%})")

        # Check violation rate anomaly (high violations)
        if mad_violations > 0:
            z_violations = (session_data['basin_violation_rate'] - median_violations) / (mad_violations * 1.4826)
            if z_violations > threshold:
                reasons.append(f"High drift rate ({session_data['basin_violation_rate']:.1%})")

        if reasons:
            anomalies.append({
                'session_id': session_data['session_id'],
                'idx': session_data['idx'],
                'reasons': reasons,
                'metrics': session_data
            })

    return {
        'anomalies': anomalies,
        'total_sessions': len(sessions),
        'anomaly_count': len(anomalies),
        'median_fidelity': median_fidelity,
        'median_intervention_rate': median_intervention,
        'median_violation_rate': median_violations
    }


# ============================================================================
# Statistical Summary Functions (Phase 9 Task 4)
# ============================================================================

def compute_comprehensive_statistics(sessions: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Generate publication-ready descriptive statistics.

    Computes comprehensive statistical measures including:
    - Central tendency (mean, median)
    - Dispersion (std dev, IQR, range)
    - Confidence intervals (95%)
    - Sample size

    Args:
        sessions: List of session data dictionaries

    Returns:
        Dict with statistical measures, or None if insufficient data
    """
    import numpy as np

    # Extract all fidelity scores across all sessions
    all_fidelities = []
    for session in sessions:
        fidelities = extract_turn_fidelities(session)
        all_fidelities.extend(fidelities)

    if not all_fidelities or len(all_fidelities) < 2:
        return None

    all_fidelities = np.array(all_fidelities)

    # Compute descriptive statistics
    stats = {
        'n': len(all_fidelities),
        'mean': float(np.mean(all_fidelities)),
        'median': float(np.median(all_fidelities)),
        'std': float(np.std(all_fidelities, ddof=1)),
        'min': float(np.min(all_fidelities)),
        'max': float(np.max(all_fidelities)),
        'q25': float(np.percentile(all_fidelities, 25)),
        'q75': float(np.percentile(all_fidelities, 75))
    }

    # Compute IQR
    stats['iqr'] = stats['q75'] - stats['q25']

    # Compute 95% confidence interval using t-distribution
    try:
        from scipy import stats as scipy_stats
        confidence_level = 0.95
        degrees_freedom = stats['n'] - 1
        sem = scipy_stats.sem(all_fidelities)
        ci = scipy_stats.t.interval(confidence_level, degrees_freedom, loc=stats['mean'], scale=sem)
        stats['ci_lower'] = float(ci[0])
        stats['ci_upper'] = float(ci[1])
        stats['ci_width'] = float(ci[1] - ci[0])
    except ImportError:
        # Fallback if scipy not available: use normal approximation
        import math
        z = 1.96  # 95% confidence
        sem = stats['std'] / math.sqrt(stats['n'])
        stats['ci_lower'] = stats['mean'] - z * sem
        stats['ci_upper'] = stats['mean'] + z * sem
        stats['ci_width'] = 2 * z * sem

    # Store raw data for distribution analysis
    stats['raw_data'] = all_fidelities.tolist()

    return stats


def generate_latex_table(stats: Dict[str, Any]) -> str:
    """
    Generate LaTeX table for publication.

    Creates a formatted LaTeX table with statistical summary
    suitable for academic papers.

    Args:
        stats: Statistics dictionary from compute_comprehensive_statistics

    Returns:
        LaTeX table string
    """
    latex = r"""\begin{table}[h]
\centering
\caption{Telic Fidelity Descriptive Statistics}
\label{tab:fidelity_stats}
\begin{tabular}{ll}
\hline
\textbf{Measure} & \textbf{Value} \\
\hline
Sample Size ($n$) & """ + str(stats['n']) + r""" \\
Mean ($\mu$) & """ + f"{stats['mean']:.3f}" + r""" \\
Median & """ + f"{stats['median']:.3f}" + r""" \\
Standard Deviation ($\sigma$) & """ + f"{stats['std']:.3f}" + r""" \\
95\% Confidence Interval & $[""" + f"{stats['ci_lower']:.3f}" + r""", """ + f"{stats['ci_upper']:.3f}" + r"""]$ \\
Interquartile Range (IQR) & """ + f"{stats['iqr']:.3f}" + r""" \\
Minimum & """ + f"{stats['min']:.3f}" + r""" \\
First Quartile ($Q_1$) & """ + f"{stats['q25']:.3f}" + r""" \\
Third Quartile ($Q_3$) & """ + f"{stats['q75']:.3f}" + r""" \\
Maximum & """ + f"{stats['max']:.3f}" + r""" \\
Range & """ + f"{stats['max'] - stats['min']:.3f}" + r""" \\
\hline
\end{tabular}
\end{table}"""

    return latex


def generate_statistics_csv(sessions: List[Dict[str, Any]]) -> str:
    """
    Generate CSV export with per-turn statistics for analysis.

    Creates a CSV file with turn-level data suitable for
    import into statistical software (R, SPSS, etc.).

    Args:
        sessions: List of session data dictionaries

    Returns:
        CSV string
    """
    import io
    import csv

    output = io.StringIO()
    writer = csv.writer(output)

    # Header
    writer.writerow([
        'Session_ID',
        'Turn_Number',
        'Fidelity',
        'Intervention_Applied',
        'Basin_Membership',
        'Session_Index'
    ])

    # Write turn-level data
    for session_idx, session in enumerate(sessions):
        session_metadata = session.get('session_metadata', {})
        session_id = session_metadata.get('session_id', f'Session_{session_idx + 1}')

        turns = session.get('turns', [])
        for turn in turns:
            turn_number = turn.get('turn_number', 0)
            metrics = turn.get('metrics', {})
            metadata = turn.get('metadata', {})

            fidelity = metrics.get('telic_fidelity', 1.0)
            intervention = 1 if metadata.get('intervention_applied', False) else 0
            basin = 1 if fidelity >= 0.76 else 0  # Goldilocks: Aligned threshold

            writer.writerow([
                session_id,
                turn_number,
                f"{fidelity:.6f}",
                intervention,
                basin,
                session_idx
            ])

    return output.getvalue()


# ============================================================================
# Evidence Export Functions (Phase 8)
# ============================================================================

def export_session_to_csv(session_data: Dict[str, Any]) -> str:
    """
    Export session turns to CSV format for statistical analysis.

    Args:
        session_data: Session data dict with snapshots

    Returns:
        CSV string
    """
    output = io.StringIO()
    writer = csv.writer(output)

    # Header
    writer.writerow([
        'Turn', 'Timestamp', 'User Input', 'Native Response', 'TELOS Response',
        'Fidelity', 'Error Signal', 'Drift Distance', 'Basin Membership',
        'Intervention Applied', 'Intervention Type'
    ])

    # Data rows
    for snapshot in session_data.get('snapshots', []):
        intervention = snapshot.get('metadata', {}).get('intervention_applied', False)
        intervention_details = snapshot.get('metadata', {}).get('intervention_details', {})
        intervention_type = intervention_details.get('type', 'none') if intervention else 'none'

        writer.writerow([
            snapshot.get('turn_number', ''),
            snapshot.get('timestamp', ''),
            snapshot.get('user_input', ''),
            snapshot.get('native_response', snapshot.get('assistant_response', '')),  # Fallback for backward compat
            snapshot.get('telos_response', snapshot.get('assistant_response', '')),
            f"{snapshot.get('telic_fidelity', 0.0):.4f}",
            f"{snapshot.get('error_signal', 0.0):.4f}",
            f"{snapshot.get('drift_distance', 0.0):.4f}",
            snapshot.get('basin_membership', True),
            intervention,
            intervention_type
        ])

    return output.getvalue()


def generate_human_readable_transcript(session_data: Dict[str, Any]) -> str:
    """
    Generate human-readable conversation transcript.

    Args:
        session_data: Session data dict

    Returns:
        Markdown-formatted transcript
    """
    metadata = session_data.get('session_metadata', {})
    snapshots = session_data.get('snapshots', [])

    transcript = []
    transcript.append("# TELOSCOPE Session Transcript")
    transcript.append("")
    transcript.append(f"**Session ID:** {metadata.get('session_id', 'N/A')}")
    transcript.append(f"**Started:** {metadata.get('started_at', 'N/A')}")
    transcript.append(f"**Total Turns:** {metadata.get('total_turns', 0)}")
    transcript.append("")
    transcript.append("---")
    transcript.append("")

    for snapshot in snapshots:
        turn_num = snapshot.get('turn_number', 0)
        timestamp = snapshot.get('timestamp', '')
        fidelity = snapshot.get('telic_fidelity', 0.0)
        intervention = snapshot.get('metadata', {}).get('intervention_applied', False)

        transcript.append(f"## Turn {turn_num + 1}")
        transcript.append(f"*{timestamp}* | Fidelity: {fidelity:.3f} | Intervention: {'✓' if intervention else '✗'}")
        transcript.append("")
        transcript.append(f"**User:** {snapshot.get('user_input', '')}")
        transcript.append("")
        transcript.append(f"**Assistant:** {snapshot.get('telos_response', snapshot.get('assistant_response', ''))}")
        transcript.append("")
        transcript.append("---")
        transcript.append("")

    return "\n".join(transcript)


def generate_governance_report_html(session_data: Dict[str, Any]) -> str:
    """
    Generate HTML governance report with metrics and summary.

    Args:
        session_data: Session data dict

    Returns:
        HTML string
    """
    metadata = session_data.get('session_metadata', {})
    snapshots = session_data.get('snapshots', [])

    # Calculate aggregate metrics
    total_turns = len(snapshots)
    if total_turns == 0:
        avg_fidelity = 0.0
        intervention_count = 0
        intervention_rate = 0.0
        drift_events = 0
    else:
        fidelities = [s.get('telic_fidelity', 0.0) for s in snapshots]
        avg_fidelity = sum(fidelities) / len(fidelities) if fidelities else 0.0

        intervention_count = sum(
            1 for s in snapshots
            if s.get('metadata', {}).get('intervention_applied', False)
        )
        intervention_rate = (intervention_count / total_turns) * 100 if total_turns > 0 else 0.0

        drift_events = sum(
            1 for s in snapshots
            if s.get('telic_fidelity', 1.0) < 0.76 or not s.get('basin_membership', True)  # Goldilocks: Aligned threshold
        )

    # Generate HTML
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>TELOSCOPE Governance Report</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                max-width: 1200px;
                margin: 40px auto;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                border-radius: 10px;
                margin-bottom: 30px;
            }}
            .header h1 {{
                margin: 0;
                font-size: 2.5em;
            }}
            .header p {{
                margin: 10px 0 0 0;
                opacity: 0.9;
            }}
            .metric-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }}
            .metric-card {{
                background: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .metric-label {{
                color: #666;
                font-size: 0.9em;
                margin-bottom: 5px;
            }}
            .metric-value {{
                font-size: 2em;
                font-weight: bold;
                color: #333;
            }}
            .metric-value.good {{
                color: #10b981;
            }}
            .metric-value.warning {{
                color: #f59e0b;
            }}
            .metric-value.critical {{
                color: #ef4444;
            }}
            .section {{
                background: white;
                padding: 25px;
                border-radius: 8px;
                margin-bottom: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .section h2 {{
                margin-top: 0;
                color: #667eea;
                border-bottom: 2px solid #667eea;
                padding-bottom: 10px;
            }}
            .turn-summary {{
                border-left: 3px solid #667eea;
                padding-left: 15px;
                margin: 10px 0;
            }}
            .footer {{
                text-align: center;
                color: #666;
                margin-top: 40px;
                font-size: 0.9em;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🔭 TELOSCOPE Governance Report</h1>
            <p>Evidence Package for AI Alignment Research</p>
        </div>

        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-label">Average Fidelity</div>
                <div class="metric-value {'good' if avg_fidelity >= 0.76 else 'warning' if avg_fidelity >= 0.73 else 'critical'}">
                    {avg_fidelity:.3f}
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Total Turns</div>
                <div class="metric-value">{total_turns}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Interventions</div>
                <div class="metric-value">{intervention_count} ({intervention_rate:.1f}%)</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Drift Events</div>
                <div class="metric-value {'good' if drift_events == 0 else 'warning' if drift_events <= 2 else 'critical'}">
                    {drift_events}
                </div>
            </div>
        </div>

        <div class="section">
            <h2>📋 Session Overview</h2>
            <p><strong>Session ID:</strong> {metadata.get('session_id', 'N/A')}</p>
            <p><strong>Started:</strong> {metadata.get('started_at', 'N/A')}</p>
            <p><strong>Last Update:</strong> {metadata.get('last_turn', 'N/A')}</p>
        </div>

        <div class="section">
            <h2>📊 Governance Summary</h2>
            <p>This session demonstrates TELOS governance in action. The system actively monitored {total_turns} conversational turns,
            maintaining an average telic fidelity of {avg_fidelity:.3f}.</p>

            <p><strong>Key Findings:</strong></p>
            <ul>
                <li>Intervention rate: {intervention_rate:.1f}% ({intervention_count}/{total_turns} turns)</li>
                <li>Drift events detected: {drift_events}</li>
                <li>Basin stability: {(1 - drift_events / max(total_turns, 1)) * 100:.1f}%</li>
            </ul>
        </div>

        <div class="section">
            <h2>🔍 Turn-by-Turn Analysis</h2>
            {''.join([f'''
            <div class="turn-summary">
                <strong>Turn {s.get('turn_number', 0) + 1}</strong>
                | Fidelity: {s.get('telic_fidelity', 0.0):.3f}
                | {'✓ Intervention' if s.get('metadata', {}).get('intervention_applied') else '○ No intervention'}
                <br><em>{s.get('timestamp', '')}</em>
            </div>
            ''' for s in snapshots[:20]])}
            {f'<p><em>... and {len(snapshots) - 20} more turns</em></p>' if len(snapshots) > 20 else ''}
        </div>

        <div class="footer">
            <p>Generated by TELOSCOPE Observatory | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Telically Entrained Linguistic Operational Substrate for Counterfactual Observation via Purpose-scoped Experimentation</p>
        </div>
    </body>
    </html>
    """

    return html


def create_evidence_package_zip(session_data: Dict[str, Any], session_id: str) -> bytes:
    """
    Create ZIP package with multiple export formats.

    Args:
        session_data: Session data dict
        session_id: Session identifier

    Returns:
        ZIP file bytes
    """
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # 1. JSON data
        zip_file.writestr(
            f'{session_id}_data.json',
            json.dumps(session_data, indent=2)
        )

        # 2. CSV telemetry
        zip_file.writestr(
            f'{session_id}_telemetry.csv',
            export_session_to_csv(session_data)
        )

        # 3. Human-readable transcript
        zip_file.writestr(
            f'{session_id}_transcript.md',
            generate_human_readable_transcript(session_data)
        )

        # 4. HTML governance report
        zip_file.writestr(
            f'{session_id}_governance_report.html',
            generate_governance_report_html(session_data)
        )

        # 5. README
        readme = f"""# TELOSCOPE Evidence Package

Session ID: {session_id}
Generated: {datetime.now().isoformat()}

## Contents

1. **{session_id}_data.json** - Complete session data (machine-readable)
2. **{session_id}_telemetry.csv** - Turn-by-turn metrics (Excel/R/Python compatible)
3. **{session_id}_transcript.md** - Human-readable conversation transcript
4. **{session_id}_governance_report.html** - Visual governance summary (open in browser)

## Usage

- **For Publications:** Use the HTML report for visual evidence and CSV for statistical analysis
- **For Reproducibility:** The JSON file contains all raw data for exact reconstruction
- **For Sharing:** The transcript provides context for collaborators

## Citation

If using this data in research, please cite:
TELOSCOPE: Telically Entrained Linguistic Operational Substrate
for Counterfactual Observation via Purpose-scoped Experimentation

Generated by TELOSCOPE Observatory
https://github.com/yourusername/telos
"""
        zip_file.writestr('README.md', readme)

    zip_buffer.seek(0)
    return zip_buffer.read()

# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title="TELOSCOPE Observatory - AI Governance Evidence",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Styling
st.markdown("""
<style>
/* Allow sidebar to collapse naturally - just ensure it starts expanded */
section[data-testid="stSidebar"] {
    /* Reduce sidebar width but keep enough space for TELOS text */
    max-width: 240px !important;
    min-width: 240px !important;
}

/* Ensure sidebar content wraps properly in reduced space */
section[data-testid="stSidebar"] > div {
    width: 100% !important;
    overflow-x: hidden !important;
}

section[data-testid="stSidebar"] * {
    word-wrap: break-word !important;
    overflow-wrap: break-word !important;
}

/* Move sidebar collapse arrow to far left corner */
button[kind="header" ElementsToolbarButton] {
    position: fixed !important;
    left: 0px !important;
    top: 0px !important;
    z-index: 999999 !important;
}

/* Alternative selector for sidebar collapse button */
section[data-testid="stSidebar"] button[kind="header"] {
    position: fixed !important;
    left: 0px !important;
    top: 0px !important;
    z-index: 999999 !important;
}

/* Right sidebar panel for Observation Deck */
.observation-deck-panel {
    position: fixed;
    top: 70px;  /* Aligned with toggle buttons - proper spacing from header */
    right: -400px;
    width: 400px;
    height: calc(100vh - 70px);  /* Adjust height to account for top offset */
    background: #1e1e1e;
    border-left: 1px solid #444;
    transition: right 0.3s ease;
    z-index: 100;  /* Lower z-index so it doesn't cover input elements */
    overflow-y: auto;
    padding: 20px;
}

.observation-deck-panel.open {
    right: 0;
}

.stMetric {
    background-color: #f0f2f6;
    padding: 10px;
    border-radius: 5px;
}

.stButton>button {
    width: 100%;
}

div[data-testid="stExpander"] {
    border: 1px solid #e0e0e0;
    border-radius: 5px;
}

.trigger-badge {
    background-color: #ff6b6b;
    color: white;
    padding: 5px 10px;
    border-radius: 3px;
    display: inline-block;
    margin: 2px;
}

.metric-good {
    color: #51cf66;
    font-weight: bold;
}

.metric-warning {
    color: #ffa94d;
    font-weight: bold;
}

.metric-critical {
    color: #ff6b6b;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Terminology Helper Functions
# ============================================================================

def get_mode():
    """Get current mode (Basic, Advanced, or Research)."""
    if 'mode' not in st.session_state:
        st.session_state.mode = 'Basic'
    return st.session_state.mode

def get_terminology(mode=None):
    """
    Get terminology dictionary based on mode.

    Returns dict with keys:
    - attractor_name: Always "Primacy Attractor" (standardized across all modes)
    - status_on: "Aligned" or "Inside Basin"
    - status_off: "Drifted" or "Outside Basin"
    - action: "Recalibration" or "Active Mitigation"
    - result: "Realignment" or "ΔF Improvement"
    - fidelity_name: "Alignment" or "Fidelity Score"
    """
    if mode is None:
        mode = get_mode()

    if mode == 'Basic':
        return {
            'attractor_name': 'Primacy Attractor',
            'status_on': 'Aligned ✅',
            'status_off': 'Drifted ⚠️',
            'action': 'Recalibration',
            'action_verb': 'Recalibrating',
            'result': 'Realignment',
            'fidelity_name': 'Alignment',
            'metric_name': 'Alignment Score',
            'drift_message': 'Drifted from Primacy Attractor - recalibrating...',
            'success_message': '✅ Realigned to Primacy Attractor'
        }
    else:  # Advanced or Research (both use technical terminology)
        return {
            'attractor_name': 'Primacy Attractor',
            'status_on': 'Inside Basin ✅',
            'status_off': 'Outside Basin ❌',
            'action': 'Active Mitigation',
            'action_verb': 'Applying Intervention',
            'result': 'ΔF Improvement',
            'fidelity_name': 'Fidelity Score',
            'metric_name': 'Telic Fidelity (F)',
            'drift_message': 'Drift Detected - Applying Active Mitigation',
            'success_message': '✅ Intervention successful'
        }

def render_research_mode_observatory(metrics: dict, embeddings: dict = None, intervention_data: dict = None):
    """
    Render Research Mode mathematical observatory display.

    Complete runtime mathematical transparency - shows every calculation,
    every formula, every step. Like viewing HTML source of a website.

    Args:
        metrics: Dictionary of calculated metrics (fidelity, distance, etc.)
        embeddings: Optional dictionary with embedding vectors for detailed analysis
        intervention_data: Optional intervention details (if mitigation occurred)
    """
    st.markdown("### 🔬 TELOS Runtime Mathematical Observatory")
    st.caption("**Mathematics in Motion** - Watch TELOS think in real-time")
    st.info("🎓 **Research Mode**: Complete mathematical transparency for papers, audits, and deep analysis")

    # ========================================================================
    # 1. EMBEDDING GENERATION
    # ========================================================================
    with st.expander("🧬 **1. Embedding Generation** (Transformer Processing)", expanded=True):
        st.markdown("**Semantic Vectorization**")
        st.caption("Text → 384-dimensional embedding space via sentence-transformers/all-MiniLM-L6-v2")

        if embeddings and 'response_embedding' in embeddings:
            resp_emb = embeddings['response_embedding']
            st.markdown(f"**Embedding Dimensions**: `384`")
            st.markdown(f"**Vector Norm**: `||r|| = {np.linalg.norm(resp_emb):.6f}`")

            # Show first 20 dimensions
            st.markdown("**First 20 dimensions**:")
            emb_preview = ", ".join([f"{val:.4f}" for val in resp_emb[:20]])
            st.code(f"[{emb_preview}, ...]", language="python")

            # Norm calculation
            st.latex(r"||\mathbf{r}|| = \sqrt{\sum_{i=1}^{384} r_i^2}")
        else:
            st.info("Embedding data not available for this turn")

    # ========================================================================
    # 2. DISTANCE CALCULATION
    # ========================================================================
    with st.expander("📏 **2. Distance Calculation** (Euclidean Norm)", expanded=True):
        st.markdown("**Step-by-Step Computation**")

        distance = metrics.get('drift_distance', 0.0)

        if embeddings and 'response_embedding' in embeddings and 'attractor_center' in embeddings:
            resp_emb = embeddings['response_embedding']
            attr_emb = embeddings['attractor_center']

            st.markdown("**Formula**:")
            st.latex(r"d = ||\mathbf{r} - \hat{\mathbf{a}}|| = \sqrt{\sum_{i=1}^{384} (r_i - a_i)^2}")

            st.markdown("**Where**:")
            st.markdown("- $\\mathbf{r}$ = Response embedding")
            st.markdown("- $\\hat{\\mathbf{a}}$ = Attractor center")

            # Show calculation
            diff = resp_emb - attr_emb
            squared_diff = diff ** 2
            sum_squared = np.sum(squared_diff)

            st.markdown("**Computation**:")
            st.code(f"""
Step 1: Calculate difference: r_i - a_i (for all 384 dimensions)
Step 2: Square each difference: (r_i - a_i)²
Step 3: Sum all squared differences: Σ = {sum_squared:.6f}
Step 4: Take square root: d = √{sum_squared:.6f} = {distance:.6f}
            """)
        else:
            st.markdown(f"**Distance**: `d = {distance:.6f}`")
            st.caption("(Detailed calculation requires embedding data)")

        st.markdown(f"### **Result**: `d = {distance:.6f}`")

    # ========================================================================
    # 3. FIDELITY CONVERSION
    # ========================================================================
    with st.expander("⚖️ **3. Fidelity Conversion** (Distance → Score)", expanded=True):
        st.markdown("**Transform distance to normalized fidelity score**")

        fidelity = metrics.get('telic_fidelity', 0.0)
        distance = metrics.get('drift_distance', 0.0)
        distance_scale = 2.0  # τ parameter

        st.markdown("**Formula**:")
        st.latex(r"F = \max(0, \min(1, 1 - \frac{d}{\tau}))")

        st.markdown("**Where**:")
        st.markdown(f"- $d$ = Distance = `{distance:.6f}`")
        st.markdown(f"- $\\tau$ = Scale parameter = `{distance_scale}`")

        st.markdown("**Computation**:")
        raw_fidelity = 1.0 - (distance / distance_scale)
        clamped_fidelity = max(0.0, min(1.0, raw_fidelity))

        st.code(f"""
Step 1: Raw fidelity = 1 - (d / τ) = 1 - ({distance:.6f} / {distance_scale}) = {raw_fidelity:.6f}
Step 2: Clamp to [0, 1]: F = max(0, min(1, {raw_fidelity:.6f})) = {clamped_fidelity:.6f}
        """)

        # Goldilocks threshold check
        threshold_aligned = 0.76  # Goldilocks: Aligned
        threshold_drift = 0.67    # Goldilocks: Significant Drift (intervention)
        passes = fidelity >= threshold_aligned
        st.markdown(f"**Threshold Check (Goldilocks Zones)**:")
        st.markdown(f"- F ≥ 0.76: Aligned")
        st.markdown(f"- 0.73 ≤ F < 0.76: Minor Drift")
        st.markdown(f"- 0.67 ≤ F < 0.73: Drift Detected")
        st.markdown(f"- F < 0.67: Significant Drift (Intervention)")
        st.markdown(f"- Current F: `{fidelity:.6f}`")
        st.markdown(f"- Status: **{'✅ ALIGNED' if passes else '⚠️ DRIFT DETECTED'}**")

        st.markdown(f"### **Result**: `F = {fidelity:.6f}` {'✅' if fidelity >= 0.76 else '🟡' if fidelity >= 0.73 else '🟠' if fidelity >= 0.67 else '🔴'}")

    # ========================================================================
    # 4. INTERVENTION LOGIC (if occurred)
    # ========================================================================
    if intervention_data and intervention_data.get('intervention_applied'):
        with st.expander("🛡️ **4. Intervention Logic** (Active Mitigation)", expanded=True):
            st.markdown("**Drift detected - Active mitigation triggered**")

            interv_type = intervention_data.get('type', 'unknown')
            fidelity_before = intervention_data.get('fidelity_original', fidelity)
            fidelity_after = intervention_data.get('fidelity_governed', fidelity)
            delta_f = fidelity_after - fidelity_before if fidelity_before is not None else 0.0

            st.markdown(f"**Intervention Type**: `{interv_type}`")

            # Decision tree (Goldilocks zones)
            st.markdown("**Decision Tree (Goldilocks Zones)**:")
            st.code(f"""
IF F < 0.76 (Below Aligned):
    → Drift detected
    → IF F < 0.67 (Significant Drift):
        → Intervention triggered
        → Check salience (attractor prominence in context)
        → IF salience < 0.7:
            → Inject attractor reinforcement
        → Generate response
        → Check coupling
        → IF coupling < 0.76:
            → Regenerate with entrainment
        → Return governed response
            """)

            st.markdown(f"**Fidelity Before**: `{fidelity_before:.6f}` ⚠️")
            st.markdown(f"**Fidelity After**: `{fidelity_after:.6f}` ✅")
            st.markdown(f"**ΔF (Improvement)**: `+{delta_f:.6f}`")

            # Improvement calculation
            if fidelity_before is not None and fidelity_before > 0:
                improvement_pct = (delta_f / fidelity_before) * 100
                st.markdown(f"**Relative Improvement**: `{improvement_pct:.1f}%`")

    # ========================================================================
    # 5. BASIN VERIFICATION
    # ========================================================================
    with st.expander("🎯 **5. Basin Verification** (Stability Proof)", expanded=True):
        st.markdown("**Primacy Basin Membership Check**")

        in_basin = metrics.get('primacy_basin_membership', fidelity >= 0.76)  # Goldilocks: Aligned threshold
        distance = metrics.get('drift_distance', 0.0)

        st.markdown("**Basin Definition**:")
        st.latex(r"B = \{\mathbf{x} : ||\mathbf{x} - \hat{\mathbf{a}}|| < r_{basin}\}")
        st.markdown("Where $r_{basin}$ is the basin radius (fidelity-based)")

        st.markdown("**Membership Test (Goldilocks: Aligned)**:")
        st.latex(r"B(\mathbf{x}) = \begin{cases} \text{True} & \text{if } F(\mathbf{x}) \geq 0.76 \\ \text{False} & \text{otherwise} \end{cases}")

        st.markdown(f"**Current State**: `{'Inside Basin ✅' if in_basin else 'Outside Basin ❌'}`")

        # Lyapunov function
        st.markdown("**Lyapunov Stability Function**:")
        st.latex(r"V(\mathbf{x}) = ||\mathbf{x} - \hat{\mathbf{a}}||^2")

        lyapunov = distance ** 2
        st.markdown(f"**V(x) = d²** = `{distance:.6f}² = {lyapunov:.6f}`")

        st.markdown("**Stability**: V(x) decreases → system converges to attractor")

    # ========================================================================
    # 6. ERROR SIGNAL
    # ========================================================================
    with st.expander("📊 **6. Error Signal** (Control Theory)", expanded=False):
        error = 1.0 - fidelity
        st.latex(r"\varepsilon = 1 - F")
        st.markdown(f"**Error Signal**: `ε = {error:.6f}`")
        st.markdown("Used for control feedback and intervention triggering")

    # ========================================================================
    # 7. RAW DATA INSPECTION
    # ========================================================================
    with st.expander("🔬 **7. Raw Data Inspection** (Complete Metadata)", expanded=False):
        st.markdown("**All Metrics (JSON)**")
        st.json(metrics)

        if embeddings:
            st.markdown("**Embeddings (384-dimensional vectors)**")
            for key, emb in embeddings.items():
                st.markdown(f"**{key}**:")
                if emb is not None:
                    st.code(f"{emb.tolist()}", language="python")

        if intervention_data:
            st.markdown("**Intervention Data**")
            st.json(intervention_data)

# ============================================================================
# System Initialization
# ============================================================================

def initialize_teloscope():
    """
    Initialize TELOSCOPE Observatory components.

    This creates all backend components and integrates them with Streamlit's
    session state for real-time UI updates.
    """
    if 'teloscope_initialized' not in st.session_state:
        try:
            with st.spinner("🔭 Initializing TELOSCOPE Observatory..."):
                # Load configuration
                config_path = Path('config.json')
                if config_path.exists():
                    with open(config_path) as f:
                        config = json.load(f)
                else:
                    # Default configuration
                    config = {
                        'governance_profile': {
                            'purpose': [
                                "Provide accurate, helpful information about AI governance",
                                "Explain TELOS framework concepts clearly",
                                "Support research in AI safety and alignment"
                            ],
                            'scope': [
                                "AI safety and alignment",
                                "Governance mechanisms",
                                "Technical implementation",
                                "Regulatory compliance"
                            ],
                            'boundaries': [
                                "No medical advice",
                                "No financial advice",
                                "No legal advice",
                                "Stay focused on AI governance topics"
                            ]
                        },
                        'drift_threshold': 0.76,  # Goldilocks: Aligned threshold
                        'branch_length': 5,
                        'enable_counterfactuals': True
                    }
                    st.session_state.config = config

                # Check API key
                api_key = os.getenv('MISTRAL_API_KEY')
                if not api_key:
                    st.error("⚠️ MISTRAL_API_KEY not found in environment")
                    st.info("Please set it with: `export MISTRAL_API_KEY='your_key_here'`")
                    st.stop()

                # Create WebSessionManager with st.session_state reference
                st.session_state.web_session = WebSessionManager(st.session_state)
                st.session_state.web_session.initialize_web_session()

                # Create SessionStateManager
                st.session_state.session_manager = SessionStateManager(
                    web_session_manager=st.session_state.web_session
                )

                # Initialize LLM and embeddings
                st.session_state.llm = TelosMistralClient(api_key=api_key)
                st.session_state.embedding_provider = EmbeddingProvider(deterministic=False)

                # Create Primacy Attractor based on onboarding mode
                attractor_mode = st.session_state.get('attractor_mode', 'progressive')

                if attractor_mode == 'predefined':
                    # Pre-defined mode: Use user-provided purpose and boundaries
                    purpose = [st.session_state.predefined_purpose]
                    boundaries = [st.session_state.predefined_boundaries] if st.session_state.get('predefined_boundaries') else []
                    st.session_state.attractor = PrimacyAttractor(
                        purpose=purpose,
                        scope=[],
                        boundaries=boundaries
                    )
                    attractor_established = True

                elif attractor_mode == 'hybrid':
                    # Hybrid mode: Use seed values to initialize, progressive extractor will refine
                    purpose = [st.session_state.hybrid_seed_purpose]
                    boundaries = [st.session_state.hybrid_seed_boundaries] if st.session_state.get('hybrid_seed_boundaries') else []
                    st.session_state.attractor = PrimacyAttractor(
                        purpose=purpose,
                        scope=[],
                        boundaries=boundaries
                    )
                    # Initialize progressive extractor for refinement
                    st.session_state.progressive_extractor = ProgressivePrimacyExtractor(
                        llm_client=st.session_state.llm,
                        embedding_provider=st.session_state.embedding_provider
                    )
                    # Seed the extractor with initial values
                    st.session_state.progressive_extractor.seed_purpose = st.session_state.hybrid_seed_purpose
                    st.session_state.progressive_extractor.seed_boundaries = st.session_state.get('hybrid_seed_boundaries')
                    attractor_established = True  # Initial attractor, will be refined

                elif attractor_mode in ['progressive', 'pristine']:
                    # Progressive/Pristine modes: No initial attractor, will be extracted from conversation
                    st.session_state.attractor = PrimacyAttractor(
                        purpose=[],
                        scope=[],
                        boundaries=[]
                    )
                    # Initialize progressive extractor
                    st.session_state.progressive_extractor = ProgressivePrimacyExtractor(
                        llm_client=st.session_state.llm,
                        embedding_provider=st.session_state.embedding_provider
                    )
                    attractor_established = False  # Will be established after 8-10 turns

                else:
                    # Fallback to config-based (backward compatibility)
                    gov_profile = config.get('governance_profile', {})
                    st.session_state.attractor = PrimacyAttractor(
                        purpose=gov_profile.get('purpose', []),
                        scope=gov_profile.get('scope', []),
                        boundaries=gov_profile.get('boundaries', [])
                    )
                    attractor_established = True

                # Create Unified Governance Steward
                st.session_state.steward = UnifiedGovernanceSteward(
                    attractor=st.session_state.attractor,
                    llm_client=st.session_state.llm,
                    embedding_provider=st.session_state.embedding_provider,
                    enable_interventions=True
                )
                st.session_state.steward.start_session()

                # Establish attractor center for drift detection (only if attractor is established)
                if attractor_established and len(st.session_state.attractor.purpose) > 0:
                    purpose_text = " ".join(st.session_state.attractor.purpose)
                    attractor_embedding = st.session_state.embedding_provider.encode([purpose_text])[0]
                    st.session_state.steward.attractor_center = attractor_embedding

                # Create CounterfactualBranchManager (API-based interventions)
                st.session_state.branch_manager = CounterfactualBranchManager(
                    llm_client=st.session_state.llm,
                    embedding_provider=st.session_state.embedding_provider,
                    steward=st.session_state.steward,
                    branch_length=config.get('branch_length', 5)
                )

                # Create CounterfactualBranchSimulator (AI-to-AI simulation)
                st.session_state.simulator = CounterfactualBranchSimulator(
                    llm_client=st.session_state.llm,
                    embedding_provider=st.session_state.embedding_provider,
                    steward=st.session_state.steward,
                    simulation_turns=config.get('simulation_turns', 5)
                )

                # Initialize session-level branch tracking
                if 'counterfactual_branches' not in st.session_state:
                    st.session_state.counterfactual_branches = []

                # Initialize simulation results storage
                if 'simulation_results' not in st.session_state:
                    st.session_state.simulation_results = {}

                # Create BranchComparator
                st.session_state.comparator = BranchComparator()

                # Create LiveInterceptor (wraps LLM)
                st.session_state.interceptor = LiveInterceptor(
                    llm_client=st.session_state.llm,
                    embedding_provider=st.session_state.embedding_provider,
                    steward=st.session_state.steward,
                    session_manager=st.session_state.session_manager,
                    branch_manager=st.session_state.branch_manager,
                    web_session_manager=st.session_state.web_session,
                    drift_threshold=config.get('drift_threshold', 0.76),  # Goldilocks: Aligned threshold
                    enable_counterfactuals=config.get('enable_counterfactuals', True)
                )

                # ============================================================
                # NEW: Phase 1 UI Overhaul - Session State Variables
                # ============================================================
                # UI mode toggle: "legacy" (current tab-based) or "chat" (new ChatGPT-style)
                if 'ui_mode' not in st.session_state:
                    st.session_state.ui_mode = 'legacy'  # Start with legacy for backward compatibility

                # Floating window visibility
                if 'show_steward_lens' not in st.session_state:
                    st.session_state.show_steward_lens = False
                if 'show_teloscope_window' not in st.session_state:
                    st.session_state.show_teloscope_window = False
                if 'turn_controls_visible' not in st.session_state:
                    st.session_state.turn_controls_visible = True

                # Governance toggle (Native Mistral vs TELOS Steward)
                if 'governance_enabled' not in st.session_state:
                    st.session_state.governance_enabled = True

                # Turn navigation
                if 'current_turn_index' not in st.session_state:
                    st.session_state.current_turn_index = 0  # Will be updated to latest turn
                if 'is_live_mode' not in st.session_state:
                    st.session_state.is_live_mode = True  # True = at latest turn, False = reviewing history
                if 'is_playing' not in st.session_state:
                    st.session_state.is_playing = False  # Pause/play state for turn navigation
                if 'playback_speed' not in st.session_state:
                    st.session_state.playback_speed = 1.0  # Playback speed multiplier

                # Window positioning (for draggable windows)
                if 'window_positions' not in st.session_state:
                    st.session_state.window_positions = {
                        'steward_lens': {'x': 50, 'y': 100},
                        'teloscope': {'x': 600, 'y': 100}
                    }
                # ============================================================
                # END: Phase 1 UI Overhaul - Session State Variables
                # ============================================================

                st.session_state.teloscope_initialized = True
                st.session_state.config = config

        except Exception as e:
            st.error("🔌 Unable to initialize TELOSCOPE Observatory")
            st.info("💡 **What to try:**\n- Refresh the page\n- Check your API key is set correctly\n- Ensure all dependencies are installed")
            logging.error(f"Initialization failed: {e}")  # Log technical details
            st.stop()


# ============================================================================
# Phase 1: UI Utility Functions
# ============================================================================

def get_ui_config():
    """
    Get UI configuration based on current mode.

    Returns:
        dict: UI configuration with styling and display settings
    """
    ui_mode = st.session_state.get('ui_mode', 'legacy')

    config = {
        'mode': ui_mode,
        'theme': {
            'user_bubble_bg': '#0084ff',
            'user_bubble_text': '#ffffff',
            'assistant_bubble_bg': '#f0f0f0',
            'assistant_bubble_text': '#000000',
            'telos_badge_color': '#0084ff',
            'native_badge_color': '#gray',
        },
        'layout': {
            'max_bubble_width': '70%',
            'bubble_padding': '12px',
            'bubble_margin': '8px',
            'bubble_border_radius': '18px',
        },
        'features': {
            'show_timestamps': True,
            'show_turn_numbers': True,
            'show_governance_badges': st.session_state.get('governance_enabled', True),
            'enable_copy_button': True,
        }
    }

    return config


def format_timestamp(timestamp):
    """
    Format timestamp for display in chat interface.

    Args:
        timestamp: datetime object or ISO string

    Returns:
        str: Formatted timestamp string (e.g., "2:45 PM" or "Oct 27, 2:45 PM")
    """
    from datetime import datetime

    # Convert string to datetime if needed
    if isinstance(timestamp, str):
        try:
            timestamp = datetime.fromisoformat(timestamp)
        except (ValueError, AttributeError):
            return "Unknown time"

    # If not a datetime object, return default
    if not isinstance(timestamp, datetime):
        return "Unknown time"

    # Check if timestamp is today
    now = datetime.now()
    is_today = (timestamp.date() == now.date())

    # Format based on recency
    if is_today:
        # Today: show only time (e.g., "2:45 PM")
        return timestamp.strftime("%-I:%M %p")
    else:
        # Other days: show date and time (e.g., "Oct 27, 2:45 PM")
        return timestamp.strftime("%b %d, %-I:%M %p")


def render_chat_bubble(role, text, turn_number=None, timestamp=None, governance_mode=None, show_copy=True):
    """
    Render a single chat message bubble (ChatGPT-style).

    Args:
        role (str): "user" or "assistant"
        text (str): Message content (supports markdown)
        turn_number (int, optional): Turn number to display
        timestamp (datetime, optional): Message timestamp
        governance_mode (str, optional): "telos" or "native" (for assistant messages)
        show_copy (bool): Whether to show copy button

    Returns:
        None: Renders to Streamlit UI
    """
    config = get_ui_config()
    theme = config['theme']
    layout = config['layout']
    features = config['features']

    # Determine styling based on role
    if role == 'user':
        bg_color = theme['user_bubble_bg']
        text_color = theme['user_bubble_text']
        align = 'right'
        justify = 'flex-end'
    else:  # assistant
        bg_color = theme['assistant_bubble_bg']
        text_color = theme['assistant_bubble_text']
        align = 'left'
        justify = 'flex-start'

    # Build bubble HTML
    bubble_html = f"""
    <div style="display: flex; justify-content: {justify}; margin: {layout['bubble_margin']} 0;">
        <div style="
            max-width: {layout['max_bubble_width']};
            background-color: {bg_color};
            color: {text_color};
            padding: {layout['bubble_padding']};
            border-radius: {layout['bubble_border_radius']};
            text-align: {align};
        ">
    """

    # Add governance badge for assistant messages (if enabled)
    if role == 'assistant' and governance_mode and features['show_governance_badges']:
        badge_color = theme['telos_badge_color'] if governance_mode == 'telos' else theme['native_badge_color']
        badge_text = "TELOS Steward" if governance_mode == 'telos' else "Native Mistral"
        bubble_html += f"""
        <div style="font-size: 0.75em; opacity: 0.8; margin-bottom: 4px;">
            <span style="background-color: {badge_color}; color: white; padding: 2px 8px; border-radius: 10px;">
                {badge_text}
            </span>
        </div>
        """

    # Add message content (will be rendered as markdown separately)
    bubble_html += f"<div class='message-content'>{text}</div>"

    # Add metadata footer (turn number, timestamp)
    if features['show_turn_numbers'] or features['show_timestamps']:
        metadata_parts = []
        if turn_number is not None and features['show_turn_numbers']:
            metadata_parts.append(f"Turn {turn_number}")
        if timestamp and features['show_timestamps']:
            metadata_parts.append(format_timestamp(timestamp))

        if metadata_parts:
            metadata_text = " • ".join(metadata_parts)
            bubble_html += f"""
            <div style="font-size: 0.75em; opacity: 0.6; margin-top: 8px;">
                {metadata_text}
            </div>
            """

    bubble_html += """
        </div>
    </div>
    """

    # Render to Streamlit
    st.markdown(bubble_html, unsafe_allow_html=True)

    # Note: In actual implementation, we'd use a proper container and
    # render the text content with st.markdown() for proper markdown support.
    # This is a simplified version for Phase 1 foundation.


# ============================================================================
# Phase 5: Turn Navigation Controls
# ============================================================================

def render_turn_navigation(total_turns: int):
    """
    Render turn navigation controls for scrubbing through conversation history.

    Features:
    - Previous/Next turn buttons
    - Turn counter display (e.g., "Turn 15 / 32")
    - Timeline scrubber slider
    - Jump to latest turn button
    - Live/Review mode indicator

    Args:
        total_turns: Total number of turns in the conversation
    """
    if total_turns == 0:
        return  # No navigation needed for empty conversation

    # Get current state
    current_turn = st.session_state.get('current_turn_index', total_turns - 1)
    is_live = st.session_state.get('is_live_mode', True)

    # Ensure current_turn is within bounds
    current_turn = max(0, min(current_turn, total_turns - 1))

    # Update live mode based on position
    is_live = (current_turn == total_turns - 1)
    st.session_state['is_live_mode'] = is_live

    # Create navigation bar
    st.markdown("---")

    # Mode indicator
    if is_live:
        st.caption("🟢 **LIVE MODE** - At latest turn")
    else:
        st.caption("⏸️ **REVIEW MODE** - Viewing conversation history")

    # Navigation controls
    col1, col2, col3, col4, col5 = st.columns([1, 1, 4, 1, 1])

    with col1:
        # Previous turn button
        if st.button("⬅️ Prev", disabled=(current_turn <= 0), use_container_width=True, type='secondary'):
            st.session_state['current_turn_index'] = current_turn - 1
            st.session_state['is_live_mode'] = False
            st.rerun()

    with col2:
        # Next turn button
        if st.button("Next ➡️", disabled=(current_turn >= total_turns - 1), use_container_width=True, type='secondary'):
            st.session_state['current_turn_index'] = current_turn + 1
            st.session_state['is_live_mode'] = (current_turn + 1 == total_turns - 1)
            st.rerun()

    with col3:
        # Timeline scrubber slider
        new_turn = st.slider(
            "Turn",
            min_value=0,
            max_value=total_turns - 1,
            value=current_turn,
            label_visibility="collapsed",
            key="turn_scrubber"
        )

        # Update if slider moved
        if new_turn != current_turn:
            st.session_state['current_turn_index'] = new_turn
            st.session_state['is_live_mode'] = (new_turn == total_turns - 1)
            st.rerun()

    with col4:
        # Jump to latest turn button
        if st.button("⏭️ Latest", disabled=is_live, use_container_width=True, type='primary'):
            st.session_state['current_turn_index'] = total_turns - 1
            st.session_state['is_live_mode'] = True
            st.rerun()

    with col5:
        # Turn counter display
        st.markdown(f"**{current_turn + 1} / {total_turns}**")

    st.markdown("---")


# ============================================================================
# Phase 3: Observable Windows (Steward Lens & TELOSCOPE)
# ============================================================================

def render_steward_lens():
    """
    Render the Steward Lens window showing Primacy Attractor and interventions.

    Displays:
    - Primacy Attractor (Purpose, Scope, Boundaries)
    - Alignment status
    - Fidelity scores
    - Recent interventions
    - Research Lens toggle (optional mathematical detail)
    """
    with st.expander("🔍 **Steward Lens** - Primacy Attractor & Interventions", expanded=True):
        if not st.session_state.get('teloscope_initialized', False):
            st.info("System initializing...")
            return

        # Get mode and terminology
        mode = get_mode()
        terms = get_terminology(mode)

        # Get metrics from interceptor
        metrics = st.session_state.interceptor.get_live_metrics()

        # ========================================
        # Section 1: Primacy Attractor
        # ========================================
        st.markdown(f"### {terms['attractor_name']}")
        st.write('')  # Breathing room

        # Get attractor from steward
        if st.session_state.get('steward') and hasattr(st.session_state.steward, 'primacy_attractor'):
            attractor = st.session_state.steward.primacy_attractor

            col1, col2 = st.columns([2, 1])

            with col1:
                # Display Purpose
                if hasattr(attractor, 'purpose') and attractor.purpose:
                    st.markdown(f"**Purpose:**  \n{attractor.purpose}")

                # Display Scope (if available)
                if hasattr(attractor, 'scope') and attractor.scope:
                    st.markdown(f"**Scope:**  \n{attractor.scope}")

                # Display Boundaries (if available)
                if hasattr(attractor, 'boundaries') and attractor.boundaries:
                    st.markdown(f"**Boundaries:**  \n{attractor.boundaries}")

            with col2:
                # Fidelity Score
                fidelity = metrics['current_fidelity']
                if mode == 'Basic':
                    st.metric("Alignment", f"{fidelity * 100:.0f}%")
                else:
                    st.metric("Fidelity (F)", f"{fidelity:.3f}")

                # Basin Status
                basin_status = metrics['basin_status']
                status_text = terms['status_on'] if basin_status else terms['status_off']
                st.markdown(f"**Status:** {status_text}")

        else:
            st.info("Primacy Attractor not yet established. Start chatting to build your governance profile!")

        st.divider()

        # ========================================
        # Section 2: Recent Interventions
        # ========================================
        st.markdown("### Recent Interventions")
        st.write('')  # Breathing room

        if st.session_state.get('steward') and hasattr(st.session_state.steward, 'llm_wrapper'):
            mitigation_stats = st.session_state.steward.llm_wrapper.get_intervention_statistics()

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Total", mitigation_stats['total_interventions'])

            with col2:
                avg_improvement = mitigation_stats.get('avg_fidelity_improvement', 0)
                if mode == 'Basic':
                    st.metric("Avg Improvement", f"{avg_improvement * 100:+.0f}%")
                else:
                    st.metric("Avg ΔF", f"{avg_improvement:+.3f}")

            with col3:
                # Intervention rate
                total_turns = mitigation_stats.get('total_turns', 1)
                rate = mitigation_stats['total_interventions'] / max(total_turns, 1)
                st.metric("Rate", f"{rate * 100:.0f}%")

            # Intervention breakdown by type
            by_type = mitigation_stats.get('by_type', {})
            if by_type:
                st.caption("**By Type:**")
                for itype, count in by_type.items():
                    if count > 0 and itype not in ['none', 'learning_phase']:
                        st.caption(f"• {itype}: {count}")

        else:
            st.caption("No interventions yet")

        st.divider()

        # ========================================
        # Section 3: Fidelity Trend Graph
        # ========================================
        st.markdown("### 📈 Fidelity Trend")
        st.write('')  # Breathing room

        # Get all turns with fidelity data
        if hasattr(st.session_state, 'session_manager'):
            turns = st.session_state.session_manager.get_all_turns()

            if turns and len(turns) > 0:
                # Extract fidelity scores from turns
                fidelities = []
                turn_numbers = []

                for turn in turns:
                    fidelity = turn.get('fidelity', None)
                    if fidelity is not None:
                        turn_numbers.append(turn.get('turn_number', 0) + 1)  # 1-indexed for display
                        fidelities.append(fidelity)

                if len(fidelities) > 0:
                    # Create DataFrame for line chart
                    chart_data = pd.DataFrame({
                        'Turn': turn_numbers,
                        'Fidelity': fidelities
                    })

                    # Display line chart
                    st.line_chart(chart_data.set_index('Turn'), height=200)

                    # Add threshold references
                    st.caption("🎯 Minor Drift threshold: 0.73 | ⚠️ Significant Drift threshold: 0.67")
                else:
                    st.info("💭 Start a conversation to see fidelity trends")
            else:
                st.caption("No conversation turns yet. Start chatting to see trend.")
        else:
            st.caption("Session manager not initialized")

        st.divider()

        # ========================================
        # Section 4: Governance Health Metrics
        # ========================================
        st.markdown("### 📊 Governance Metrics")
        st.write('')  # Breathing room

        # Get all turns for metrics calculation
        if hasattr(st.session_state, 'session_manager'):
            turns = st.session_state.session_manager.get_all_turns()

            if turns and len(turns) > 0:
                # Extract fidelity scores
                fidelities = []
                for turn in turns:
                    fidelity = turn.get('fidelity', None)
                    if fidelity is not None:
                        fidelities.append(fidelity)

                # Get intervention data from steward if available
                total_interventions = 0
                if st.session_state.get('steward') and hasattr(st.session_state.steward, 'llm_wrapper'):
                    mitigation_stats = st.session_state.steward.llm_wrapper.get_intervention_statistics()
                    total_interventions = mitigation_stats.get('total_interventions', 0)

                # Calculate metrics
                avg_fidelity = sum(fidelities) / len(fidelities) if fidelities else 0.0
                intervention_rate = (total_interventions / len(turns)) * 100 if turns else 0.0
                basin_crossings = len([f for f in fidelities if f < 0.73])  # Minor Drift threshold

                # Display in columns
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    # Color-code fidelity health (Goldilocks zones): Green (≥0.76), Yellow (0.73-0.76), Red (<0.73)
                    if avg_fidelity >= 0.76:  # Aligned
                        fidelity_delta = "Aligned"
                        fidelity_delta_color = "normal"
                    elif avg_fidelity >= 0.73:  # Minor Drift
                        fidelity_delta = "Good"
                        fidelity_delta_color = "normal"
                    else:
                        fidelity_delta = "Low"
                        fidelity_delta_color = "inverse"

                    st.metric("Avg Fidelity", f"{avg_fidelity:.2f}",
                             delta=fidelity_delta, delta_color=fidelity_delta_color)

                with col2:
                    st.metric("Interventions", total_interventions)

                with col3:
                    st.metric("Intervention Rate", f"{intervention_rate:.1f}%")

                with col4:
                    # Color-code basin crossings: Green (0), Yellow (1-2), Red (3+)
                    if basin_crossings == 0:
                        crossing_delta = "None"
                        crossing_delta_color = "normal"
                    elif basin_crossings <= 2:
                        crossing_delta = "Few"
                        crossing_delta_color = "normal"
                    else:
                        crossing_delta = "High"
                        crossing_delta_color = "inverse"

                    st.metric("Basin Crossings", basin_crossings,
                             delta=crossing_delta, delta_color=crossing_delta_color)
            else:
                st.caption("No metrics available yet. Start chatting to see governance health.")
        else:
            st.caption("Session manager not initialized")

        st.divider()

        # ========================================
        # Section 5: Intervention Timeline
        # ========================================
        st.markdown("### ⏱️ Intervention Timeline")
        st.write('')  # Breathing room

        # Get all turns with intervention data
        if hasattr(st.session_state, 'session_manager'):
            turns = st.session_state.session_manager.get_all_turns()

            if turns and len(turns) > 0:
                # Extract intervention data from turns
                intervention_data = []
                turn_numbers = []

                for turn in turns:
                    turn_num = turn.get('turn_number', 0) + 1  # 1-indexed for display
                    turn_numbers.append(turn_num)

                    # Check if intervention occurred in this turn
                    metadata = turn.get('governance_metadata', {})
                    intervention_applied = metadata.get('intervention_applied', False)

                    # Count as 1 if intervention occurred, 0 otherwise
                    intervention_data.append(1 if intervention_applied else 0)

                if len(intervention_data) > 0:
                    # Create DataFrame for bar chart
                    chart_data = pd.DataFrame({
                        'Turn': turn_numbers,
                        'Intervention': intervention_data
                    })

                    # Display bar chart
                    st.bar_chart(chart_data.set_index('Turn'), height=150)

                    # Show intervention type breakdown
                    intervention_count = sum(intervention_data)
                    if intervention_count > 0:
                        st.caption(f"✓ {intervention_count} intervention(s) across {len(turns)} turn(s)")

                        # Get intervention type breakdown from steward
                        if st.session_state.get('steward') and hasattr(st.session_state.steward, 'llm_wrapper'):
                            mitigation_stats = st.session_state.steward.llm_wrapper.get_intervention_statistics()
                            type_counts = mitigation_stats.get('by_type', {})

                            if type_counts:
                                type_breakdown = " | ".join([f"{k}: {v}" for k, v in type_counts.items()])
                                st.caption(f"📋 Types: {type_breakdown}")
                    else:
                        st.caption("No interventions triggered yet")
                else:
                    st.caption("No intervention data available yet")
            else:
                st.caption("No conversation turns yet. Start chatting to see intervention timeline.")
        else:
            st.caption("Session manager not initialized")

        st.divider()

        # ========================================
        # Section 6: Research Lens Toggle
        # ========================================
        research_lens_enabled = st.checkbox(
            "🔬 Research Lens (Mathematical Detail)",
            value=False,
            key="research_lens_toggle",
            help="Show detailed mathematical analysis"
        )

        if research_lens_enabled:
            st.markdown("### Mathematical Analysis")
            st.caption("*7-step observatory would appear here in full implementation*")
            # TODO: Add full mathematical observatory in future iteration


def render_teloscope_window():
    """
    Render the TELOSCOPE window showing mathematical transparency.

    Displays:
    - 7-step mathematical observatory
    - Live calculations
    - Step-by-step governance process
    """
    with st.expander("🔭 **TELOSCOPE** - Mathematical Observatory", expanded=False):
        if not st.session_state.get('teloscope_initialized', False):
            st.info("System initializing...")
            return

        st.markdown("### 7-Step Mathematical Observatory")
        st.caption("Complete transparency into governance calculations")

        # Get latest turn data
        if hasattr(st.session_state, 'session_manager'):
            turns = st.session_state.session_manager.get_all_turns()

            if len(turns) > 0:
                latest_turn = turns[-1]

                # Display turn metadata
                st.caption(f"**Turn {latest_turn.get('turn_number', 0)}**")

                # Governance metadata (if available)
                governance_metadata = latest_turn.get('governance_metadata', {})

                if governance_metadata:
                    # Fidelity scores
                    pre_fidelity = governance_metadata.get('pre_fidelity', None)
                    post_fidelity = governance_metadata.get('post_fidelity', None)

                    if pre_fidelity is not None and post_fidelity is not None:
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.metric("Pre-Fidelity", f"{pre_fidelity:.3f}")

                        with col2:
                            st.metric("Post-Fidelity", f"{post_fidelity:.3f}")

                        with col3:
                            delta_f = post_fidelity - pre_fidelity
                            st.metric("ΔF", f"{delta_f:+.3f}")

                    # Intervention details
                    intervention_applied = governance_metadata.get('intervention_applied', False)

                    if intervention_applied:
                        st.success("✅ Intervention Applied")

                        intervention_type = governance_metadata.get('intervention_type', 'Unknown')
                        st.caption(f"**Type:** {intervention_type}")

                        # Rationale (if available)
                        rationale = governance_metadata.get('intervention_rationale', '')
                        if rationale:
                            st.caption(f"**Rationale:** {rationale}")

                    else:
                        st.info("No intervention needed - response aligned")

                    st.divider()
                    st.caption("*Full 7-step mathematical breakdown would appear here in complete implementation*")

                else:
                    st.caption("No governance metadata available for this turn")

            else:
                st.info("No conversation turns yet. Start chatting to see mathematical analysis!")

        else:
            st.warning("Session manager not initialized")


# ============================================================================
# Floating Draggable Popup Windows
# ============================================================================

def render_floating_popups(dark_mode):
    """
    Render floating, draggable popup windows that overlay the chat interface.

    Windows:
    - Steward Lens (ESC): Primacy Attractor display
    - TELOSCOPE (Spacebar): Mathematical transparency
    - TELOSCOPIC TOOLS (Up Arrow): Analysis tools
    """

    show_steward = st.session_state.get('show_steward_lens', False)
    show_teloscope = st.session_state.get('show_teloscope_window', False)
    show_tools = st.session_state.get('show_teloscope_tools', False)

    # Only render if at least one window is visible
    if not (show_steward or show_teloscope or show_tools):
        return

    # Get steward data if available
    steward_content = ""
    if show_steward and st.session_state.get('steward'):
        steward = st.session_state.steward
        attractor = steward.attractor_config
        # Handle attractor being a list or object
        if hasattr(attractor, 'purpose'):
            purpose_text = '<br>'.join(attractor.purpose) if isinstance(attractor.purpose, list) else str(attractor.purpose)
            scope_text = '<br>'.join(attractor.scope) if isinstance(attractor.scope, list) else str(attractor.scope)
            boundaries_text = '<br>'.join(attractor.boundaries) if isinstance(attractor.boundaries, list) else str(attractor.boundaries)
        else:
            purpose_text = "Governance configuration"
            scope_text = "All interactions"
            boundaries_text = "As configured"

        steward_content = f"""
        <div style="margin-bottom: 12px;">
            <strong>Purpose:</strong><br>
            <div style="margin-left: 10px; color: {'#ccc' if dark_mode else '#555'};">{purpose_text}</div>
        </div>
        <div style="margin-bottom: 12px;">
            <strong>Scope:</strong><br>
            <div style="margin-left: 10px; color: {'#ccc' if dark_mode else '#555'};">{scope_text}</div>
        </div>
        <div>
            <strong>Boundaries:</strong><br>
            <div style="margin-left: 10px; color: {'#ccc' if dark_mode else '#555'};">{boundaries_text}</div>
        </div>
        """
    elif show_steward:
        steward_content = "<p style='color: #999;'>Steward initializing...</p>"

    # TELOSCOPE content
    teloscope_content = ""
    if show_teloscope:
        teloscope_content = """
        <div style="margin-bottom: 10px;">
            <strong>7-Step Observatory:</strong>
        </div>
        <div style="margin-left: 10px; font-size: 0.9em; line-height: 1.6;">
            1. Native Response Generation<br>
            2. Pre-Fidelity Assessment<br>
            3. Drift Detection<br>
            4. Intervention Decision<br>
            5. TELOS Response Generation<br>
            6. Post-Fidelity Assessment<br>
            7. ΔF Calculation
        </div>
        """

    # TELOSCOPIC TOOLS content
    tools_content = ""
    if show_tools:
        tools_content = """
        <div style="margin-bottom: 10px;">
            <strong>Analysis Tools:</strong>
        </div>
        <div style="margin-left: 10px; font-size: 0.9em;">
            🔍 Fidelity Inspector<br>
            📊 Drift Analyzer<br>
            🧪 Counterfactual Simulator<br>
            📈 Trend Visualizer<br>
            🎯 Intervention Tracker
        </div>
        """

    # Dark mode colors
    bg_color = "#2d2d2d" if dark_mode else "#ffffff"
    text_color = "#e0e0e0" if dark_mode else "#000000"
    border_color = "#444" if dark_mode else "#ddd"
    header_bg = "#1a1a1a" if dark_mode else "#f5f5f5"

    # Build popup HTML parts separately (to avoid f-string backslash issues)
    steward_popup = ""
    if show_steward:
        steward_popup = f'''
        <div class="popup" id="steward-popup" style="display: block;">
            <div class="popup-header" onmousedown="startDrag(event, 'steward-popup')">
                🔍 STEWARD
                <span class="close-btn" onclick="closePopup('steward-popup')">✕</span>
            </div>
            <div class="popup-content">{steward_content}</div>
        </div>
        '''

    teloscope_popup = ""
    if show_teloscope:
        teloscope_popup = f'''
        <div class="popup" id="teloscope-popup" style="display: block; left: 320px;">
            <div class="popup-header" onmousedown="startDrag(event, 'teloscope-popup')">
                🔭 TELOSCOPE
                <span class="close-btn" onclick="closePopup('teloscope-popup')">✕</span>
            </div>
            <div class="popup-content">{teloscope_content}</div>
        </div>
        '''

    tools_popup = ""
    if show_tools:
        tools_popup = f'''
        <div class="popup" id="tools-popup" style="display: block; left: 640px;">
            <div class="popup-header" onmousedown="startDrag(event, 'tools-popup')">
                🛠️ TELOSCOPIC TOOLS
                <span class="close-btn" onclick="closePopup('tools-popup')">✕</span>
            </div>
            <div class="popup-content">{tools_content}</div>
        </div>
        '''

    # Create floating popup HTML with drag-and-drop
    popup_html = f"""
    <div id="floating-popups">
        {steward_popup}
        {teloscope_popup}
        {tools_popup}
    </div>

    <style>
    .popup {{
        position: fixed;
        top: 80px;
        left: 20px;
        width: 280px;
        background-color: {bg_color};
        border: 1px solid {border_color};
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        z-index: 9999;
        overflow: hidden;
    }}

    .popup-header {{
        background-color: {header_bg};
        color: {text_color};
        padding: 10px 12px;
        cursor: move;
        user-select: none;
        font-weight: bold;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid {border_color};
    }}

    .popup-content {{
        padding: 15px;
        color: {text_color};
        max-height: 400px;
        overflow-y: auto;
        font-size: 0.9em;
    }}

    .close-btn {{
        cursor: pointer;
        font-size: 1.2em;
        opacity: 0.7;
        transition: opacity 0.2s;
    }}

    .close-btn:hover {{
        opacity: 1;
    }}
    </style>

    <script>
    let activePopup = null;
    let offsetX = 0;
    let offsetY = 0;

    function startDrag(e, popupId) {{
        activePopup = document.getElementById(popupId);
        const rect = activePopup.getBoundingClientRect();
        offsetX = e.clientX - rect.left;
        offsetY = e.clientY - rect.top;

        document.addEventListener('mousemove', drag);
        document.addEventListener('mouseup', stopDrag);
        e.preventDefault();
    }}

    function drag(e) {{
        if (activePopup) {{
            activePopup.style.left = (e.clientX - offsetX) + 'px';
            activePopup.style.top = (e.clientY - offsetY) + 'px';
        }}
    }}

    function stopDrag() {{
        document.removeEventListener('mousemove', drag);
        document.removeEventListener('mouseup', stopDrag);
        activePopup = null;
    }}

    function closePopup(popupId) {{
        // Communicate close to Streamlit
        window.parent.postMessage({{
            type: 'close_popup',
            popup: popupId
        }}, '*');
    }}
    </script>
    """

    # Render the HTML component
    components.html(popup_html, height=0)


# ============================================================================
# Phase 2: Chat Interface
# ============================================================================

def render_chat_interface():
    """
    Render minimalistic ChatGPT-style interface.

    Minimalistic design philosophy:
    - Only chat messages visible by default
    - One small governance toggle (Mistral ↔ TELOS)
    - All other controls hidden, activated via keyboard shortcuts
    - Clean, centered layout
    """
    from datetime import datetime

    # Get turn data from session manager
    if not hasattr(st.session_state, 'session_manager'):
        st.warning("Session manager not initialized. Please refresh the page.")
        return

    # Get all turns from session
    turns = st.session_state.session_manager.get_all_turns()

    # ========================================================================
    # KEYBOARD SHORTCUTS (Invisible Handler)
    # ========================================================================
    render_keyboard_handler()

    # ========================================================================
    # OBSERVATORY CONTROL STRIP - Turn-by-Turn Metrics (Top Right)
    # ========================================================================
    render_observatory_control_strip()

    # ========================================================================
    # MINIMALISTIC HEADER: Dark/Light mode + Governance toggle + Telescope (top-right)
    # ========================================================================

    # Create subtle header with toggles - 4 columns for spacing, Dark, TELOS, and Telescope
    header_col1, header_col2, header_col3, header_col4 = st.columns([2.0, 1.3, 1.5, 1.2])

    with header_col1:
        # Empty - keeps toggles on right side
        pass

    with header_col2:
        # Dark/Light mode toggle
        dark_mode = st.toggle(
            "Dark",
            value=st.session_state.get('dark_mode', False),
            key='dark_mode_toggle',
            help="Toggle dark mode"
        )
        st.session_state['dark_mode'] = dark_mode

    with header_col3:
        # Governance toggle: Mistral ↔ TELOS
        governance_enabled = st.toggle(
            "TELOS",
            value=st.session_state.get('governance_enabled', True),
            key='governance_toggle_display',
            help="Toggle governance: ON = TELOS Steward | OFF = Native Mistral"
        )
        st.session_state['governance_enabled'] = governance_enabled

    with header_col4:
        # Observation Deck toggle - synchronized with Observation Deck state
        current_deck_state = st.session_state.deck_manager.session_state['observation_deck']['is_open']
        observatory_enabled = st.toggle(
            "Observation Deck",
            value=current_deck_state,
            key='observatory_toggle',
            help="Toggle Observation Deck panel"
        )

        # Update deck state if toggle changed
        if observatory_enabled != current_deck_state:
            st.session_state.deck_manager.toggle_deck()
            st.rerun()

    # Use components.html for guaranteed JavaScript execution after Streamlit render
    components.html("""
    <style>
    /* Aggressive toggle override - applied immediately after render */
    div[data-testid="stToggle"] button,
    button[role="switch"],
    div[data-testid="stToggle"] div[role="switch"] {
        background-color: #888888 !important;
    }

    div[data-testid="stToggle"] button[aria-checked="true"],
    button[role="switch"][aria-checked="true"],
    div[data-testid="stToggle"] div[role="switch"][aria-checked="true"] {
        background-color: #999999 !important;
    }

    /* Primary button override */
    button[kind="primary"],
    button[kind="primaryFormSubmit"] {
        background-color: #888888 !important;
        border: 1px solid #888888 !important;
    }
    </style>

    <script>
    // Force toggle colors via JavaScript DOM manipulation
    function forceToggleColors() {
        // Target all toggle switches in parent window
        const parentDoc = window.parent.document;
        const toggles = parentDoc.querySelectorAll('button[role="switch"], div[data-testid="stToggle"] button, [data-baseweb="toggle"]');

        toggles.forEach(toggle => {
            const isChecked = toggle.getAttribute('aria-checked') === 'true';
            const bgColor = isChecked ? '#999999' : '#888888';

            // Apply inline styles directly (overrides everything)
            toggle.style.cssText += `background-color: ${bgColor} !important; border-color: ${bgColor} !important;`;

            // Target all child elements including deeply nested ones
            const allChildren = toggle.querySelectorAll('*');
            allChildren.forEach(child => {
                // Check if it's a visual element (div, span, svg path, etc.)
                if (child.tagName !== 'INPUT') {
                    child.style.cssText += `background-color: ${bgColor} !important; color: white !important;`;
                }
            });
        });

        console.log('Forced toggle colors on', toggles.length, 'toggles');
    }

    // Run immediately
    setTimeout(() => forceToggleColors(), 10);
    setTimeout(() => forceToggleColors(), 100);
    setTimeout(() => forceToggleColors(), 300);
    setTimeout(() => forceToggleColors(), 600);
    setTimeout(() => forceToggleColors(), 1000);
    setTimeout(() => forceToggleColors(), 2000);

    // Watch for changes in parent document
    const observer = new MutationObserver(() => {
        forceToggleColors();
    });

    setTimeout(() => {
        observer.observe(window.parent.document.body, {
            attributes: true,
            subtree: true,
            attributeFilter: ['aria-checked', 'style', 'class']
        });
    }, 100);
    </script>
    """, height=0)

    # Apply styling based on mode
    if dark_mode:
        st.markdown("""
        <style>
        /* Dark Mode Theme - Interface dark gray, chat area black */

        /* Root app container - dark gray background */
        .stApp {
            background-color: #3a3a3a !important;
            font-size: 17px !important;  /* Base font size increased */
        }

        /* Show header but style it to match theme */
        header[data-testid="stHeader"] {
            background-color: #3a3a3a !important;
        }

        /* Show toolbar */
        div[data-testid="stToolbar"] {
            background-color: #3a3a3a !important;
        }

        /* Hide Deploy button while keeping collapse arrows visible */
        header[data-testid="stHeader"] button[kind="header"] {
            display: none !important;
        }

        /* But ensure collapse control remains visible */
        button[data-testid="collapsedControl"] {
            display: flex !important;
        }

        .main {
            background-color: #3a3a3a !important;  /* Dark gray interface */
            color: #e0e0e0 !important;
        }

        /* Chat message area stays black */
        .main .block-container {
            background-color: #000000 !important;  /* Black chat area */
            padding-top: 2rem !important;
            padding-bottom: 2rem !important;
        }

        /* Sidebar dark mode */
        section[data-testid="stSidebar"] {
            background-color: #2d2d2d !important;
            font-size: 16px !important;
        }
        section[data-testid="stSidebar"] * {
            color: #e0e0e0 !important;
        }

        /* Sidebar title - three sizes larger */
        section[data-testid="stSidebar"] h1 {
            font-size: 3.5rem !important;
        }

        /* Text and markdown */
        .stMarkdown, .stText {
            color: #e0e0e0 !important;
            font-size: 17px !important;
        }

        /* Toggle labels - FORCE gray color to match interface */
        div[data-testid="stToggle"] label,
        div[data-testid="stToggle"] label *,
        div[data-testid="stToggle"] p,
        div[data-testid="stToggle"] span,
        div[data-testid="stToggle"] div,
        label[data-baseweb="checkbox"],
        label[data-baseweb="checkbox"] *,
        div[data-testid="stToggle"] {
            color: #888888 !important;
        }

        /* Force all text within toggle containers to be gray */
        [data-testid="stToggle"] * {
            color: #888888 !important;
        }

        /* Input fields */
        .stTextInput input, .stTextArea textarea {
            background-color: #2d2d2d !important;
            color: #e0e0e0 !important;
            border-color: #444 !important;
            font-size: 21px !important;
        }

        /* Buttons - Gray styling for all buttons including primary/send */
        .stButton button,
        .stButton > button,
        button[kind="primary"],
        button[kind="primaryFormSubmit"],
        button[kind="secondary"] {
            background-color: #888888 !important;
            color: #ffffff !important;
            border-color: #888888 !important;
            font-size: 17px !important;
        }

        .stButton button:hover,
        button[kind="primary"]:hover,
        button[kind="primaryFormSubmit"]:hover {
            background-color: #999999 !important;
            border-color: #999999 !important;
        }

        /* Expanders */
        .streamlit-expanderHeader {
            background-color: #2d2d2d !important;
            color: #e0e0e0 !important;
            font-size: 16px !important;
        }

        /* Metrics and containers */
        div[data-testid="stMetric"] {
            background-color: #2d2d2d !important;
            color: #e0e0e0 !important;
            font-size: 16px !important;
        }

        /* Info boxes */
        .stAlert {
            background-color: #2d2d2d !important;
            color: #e0e0e0 !important;
            font-size: 16px !important;
        }

        /* Headers */
        h1, h2, h3, h4, h5, h6 {
            color: #e0e0e0 !important;
        }

        h1 { font-size: 2.5rem !important; }
        h2 { font-size: 2rem !important; }
        h3 { font-size: 1.6rem !important; }
        h4 { font-size: 1.3rem !important; }

        /* Links */
        a {
            color: #66b3ff !important;
        }

        /* Dividers */
        hr {
            border-color: #444 !important;
        }

        /* Toggle switches - Override Streamlit's default colors with maximum specificity */
        /* Target ALL toggle elements */
        div[data-testid="stToggle"] button,
        div[data-testid="stToggle"] button[role="switch"],
        button[role="switch"],
        button[kind="toggle"],
        div[role="switch"] {
            background-color: #888888 !important;
            border-color: #888888 !important;
        }

        div[data-testid="stToggle"] button[aria-checked="true"],
        div[data-testid="stToggle"] button[role="switch"][aria-checked="true"],
        button[role="switch"][aria-checked="true"],
        button[kind="toggle"][aria-checked="true"],
        div[role="switch"][aria-checked="true"] {
            background-color: #999999 !important;
            border-color: #999999 !important;
        }

        /* Override the toggle track/thumb elements */
        button[role="switch"] > div,
        button[role="switch"] span,
        div[data-testid="stToggle"] button > div,
        div[data-testid="stToggle"] button span {
            background-color: inherit !important;
        }

        button[role="switch"][aria-checked="true"] > div,
        button[role="switch"][aria-checked="true"] span,
        div[data-testid="stToggle"] button[aria-checked="true"] > div,
        div[data-testid="stToggle"] button[aria-checked="true"] span {
            background-color: inherit !important;
        }

        /* Override Streamlit's baseui toggle styles */
        div[data-baseweb="toggle"] {
            background-color: #888888 !important;
        }

        div[data-baseweb="toggle"][aria-checked="true"] {
            background-color: #999999 !important;
        }

        /* Checkbox styling */
        div[data-baseweb="checkbox"] {
            background-color: #888 !important;
        }

        div[data-baseweb="checkbox"] input:checked + div {
            background-color: #999 !important;
        }

        input[type="checkbox"] {
            accent-color: #888 !important;
        }

        /* Label colors */
        label[data-baseweb="checkbox"] {
            color: #e0e0e0 !important;
        }
        </style>
        """, unsafe_allow_html=True)
    else:
        # Light mode with gray toggles
        st.markdown("""
        <style>
        /* Light Mode Theme */

        /* Root app container */
        .stApp {
            background-color: #ffffff !important;
            font-size: 17px !important;  /* Base font size increased */
        }

        /* Show header but style it to match theme */
        header[data-testid="stHeader"] {
            background-color: #ffffff !important;
        }

        /* Show toolbar */
        div[data-testid="stToolbar"] {
            background-color: #ffffff !important;
        }

        /* Hide Deploy button while keeping collapse arrows visible */
        header[data-testid="stHeader"] button[kind="header"] {
            display: none !important;
        }

        /* But ensure collapse control remains visible */
        button[data-testid="collapsedControl"] {
            display: flex !important;
        }

        .main {
            background-color: #ffffff !important;
            color: #000000 !important;
        }

        .main .block-container {
            padding-top: 2rem !important;
            padding-bottom: 2rem !important;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            font-size: 16px !important;
        }

        /* Sidebar title - three sizes larger */
        section[data-testid="stSidebar"] h1 {
            font-size: 3.5rem !important;
        }

        /* Text and markdown */
        .stMarkdown, .stText {
            font-size: 17px !important;
        }

        /* Toggle labels - FORCE gray color to match interface */
        div[data-testid="stToggle"] label,
        div[data-testid="stToggle"] label *,
        div[data-testid="stToggle"] p,
        div[data-testid="stToggle"] span,
        div[data-testid="stToggle"] div,
        label[data-baseweb="checkbox"],
        label[data-baseweb="checkbox"] *,
        div[data-testid="stToggle"] {
            color: #999999 !important;
        }

        /* Force all text within toggle containers to be gray */
        [data-testid="stToggle"] * {
            color: #999999 !important;
        }

        /* Input fields */
        .stTextInput input, .stTextArea textarea {
            font-size: 21px !important;
        }

        /* Toggle switches - Override Streamlit's default colors with maximum specificity (light gray, not red) */
        /* Target ALL toggle elements */
        div[data-testid="stToggle"] button,
        div[data-testid="stToggle"] button[role="switch"],
        button[role="switch"],
        button[kind="toggle"],
        div[role="switch"] {
            background-color: #cccccc !important;
            border-color: #cccccc !important;
        }

        div[data-testid="stToggle"] button[aria-checked="true"],
        div[data-testid="stToggle"] button[role="switch"][aria-checked="true"],
        button[role="switch"][aria-checked="true"],
        button[kind="toggle"][aria-checked="true"],
        div[role="switch"][aria-checked="true"] {
            background-color: #999999 !important;
            border-color: #999999 !important;
        }

        /* Override the toggle track/thumb elements */
        button[role="switch"] > div,
        button[role="switch"] span,
        div[data-testid="stToggle"] button > div,
        div[data-testid="stToggle"] button span {
            background-color: inherit !important;
        }

        button[role="switch"][aria-checked="true"] > div,
        button[role="switch"][aria-checked="true"] span,
        div[data-testid="stToggle"] button[aria-checked="true"] > div,
        div[data-testid="stToggle"] button[aria-checked="true"] span {
            background-color: inherit !important;
        }

        /* Override Streamlit's baseui toggle styles */
        div[data-baseweb="toggle"] {
            background-color: #cccccc !important;
        }

        div[data-baseweb="toggle"][aria-checked="true"] {
            background-color: #999999 !important;
        }

        /* Checkbox styling */
        div[data-baseweb="checkbox"] {
            background-color: #ccc !important;
        }

        div[data-baseweb="checkbox"] input:checked + div {
            background-color: #999 !important;
        }

        input[type="checkbox"] {
            accent-color: #999 !important;
        }

        /* Buttons - Gray styling for all buttons including primary/send */
        .stButton button,
        .stButton > button,
        button[kind="primary"],
        button[kind="primaryFormSubmit"],
        button[kind="secondary"] {
            background-color: #888888 !important;
            color: #ffffff !important;
            border-color: #888888 !important;
            font-size: 17px !important;
        }

        .stButton button:hover,
        button[kind="primary"]:hover,
        button[kind="primaryFormSubmit"]:hover {
            background-color: #999999 !important;
            border-color: #999999 !important;
        }

        /* Expanders and other elements */
        .streamlit-expanderHeader {
            font-size: 16px !important;
        }

        div[data-testid="stMetric"] {
            font-size: 16px !important;
        }

        .stAlert {
            font-size: 16px !important;
        }

        /* Headers */
        h1 { font-size: 2.5rem !important; }
        h2 { font-size: 2rem !important; }
        h3 { font-size: 1.6rem !important; }
        h4 { font-size: 1.3rem !important; }
        </style>
        """, unsafe_allow_html=True)

    # ========================================================================
    # Floating Popup Windows (Overlay chat, draggable)
    # ========================================================================

    # Render floating popups based on keyboard state
    render_floating_popups(dark_mode)

    # ========================================================================
    # MESSAGE CONTAINER (Scrollable)
    # ========================================================================

    if len(turns) == 0:
        # Minimalistic welcome - clean empty state (dark mode aware)
        welcome_color = "#888" if dark_mode else "#999"
        st.markdown(f"""
        <div style="text-align: center; padding: 60px 20px; color: {welcome_color};">
            <p style="font-size: 19px;">Start a conversation</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Phase 5: Filter turns based on current_turn_index
        current_turn_index = st.session_state.get('current_turn_index', len(turns) - 1)
        current_turn_index = max(0, min(current_turn_index, len(turns) - 1))  # Bounds check

        # Only show turns up to current_turn_index (inclusive)
        visible_turns = turns[:current_turn_index + 1]

        # Render filtered messages
        for turn in visible_turns:
            turn_number = turn.get('turn_number', 0)
            user_message = turn.get('user_message', '')

            # Phase 4: Get both responses from turn data
            native_response = turn.get('native_response', '')
            telos_response = turn.get('telos_response', '')

            # Phase 4: Select which response to display based on toggle
            assistant_response = telos_response if governance_enabled else native_response

            timestamp = turn.get('timestamp', datetime.now())

            # Check if intervention was applied
            governance_metadata = turn.get('governance_metadata', {})
            intervention_applied = governance_metadata.get('intervention_applied', False)

            # Phase 4: Badge reflects currently displayed mode
            governance_mode = 'telos' if governance_enabled else 'native'

            # Dark mode aware colors
            assistant_bg = "#3a3a3a" if dark_mode else "#f0f0f0"
            assistant_color = "#e0e0e0" if dark_mode else "#000"

            # Render user message (MINIMALISTIC - no turn numbers, no timestamps)
            st.markdown(f"""
            <div style="display: flex; justify-content: flex-end; margin: 10px 0;">
                <div style="max-width: 75%; background-color: #0084ff; color: white; padding: 14px 18px; border-radius: 18px; font-size: 21px; line-height: 1.5;">
                    {user_message}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Render assistant message (MINIMALISTIC - no badges, no metadata)
            st.markdown(f"""
            <div style="display: flex; justify-content: flex-start; margin: 10px 0;">
                <div style="max-width: 75%; background-color: {assistant_bg}; color: {assistant_color}; padding: 14px 18px; border-radius: 18px; font-size: 21px; line-height: 1.5;">
                    <div style="white-space: pre-wrap;">{assistant_response}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ========================================================================
    # TURN NAVIGATION (Hidden - keyboard shortcuts only: ← / →)
    # ========================================================================

    # Navigation controls hidden for minimalistic design
    # Use keyboard: Left Arrow (←) = Previous, Right Arrow (→) = Next
    # Uncomment below to show visual navigation controls:
    # if len(turns) > 0:
    #     render_turn_navigation(len(turns))

    # ========================================================================
    # CHAT INPUT AREA (Sticky at bottom)
    # ========================================================================

    # Phase 5: Disable input in review mode
    is_live_mode = st.session_state.get('is_live_mode', True)

    # Create input area
    input_container = st.container()

    with input_container:
        col_input, col_send = st.columns([5, 1])

        with col_input:
            # Phase 5: Disable input when in review mode
            user_input = st.text_area(
                "Message",
                key="chat_input",
                placeholder="Type your message here... (Shift+Enter for new line)" if is_live_mode else "⏸️ Navigate to latest turn to send messages",
                height=80,
                label_visibility="collapsed",
                disabled=not is_live_mode
            )

        with col_send:
            st.markdown("<br>", unsafe_allow_html=True)  # Spacer for alignment
            # Phase 5: Disable send button in review mode
            send_button = st.button("Send", type="primary", use_container_width=True, disabled=not is_live_mode)

        # Handle send action
        if send_button and user_input.strip():
            # Process the message through TELOS
            with st.spinner("Processing..."):
                try:
                    # Send through LiveInterceptor with proper message format
                    messages = [{"role": "user", "content": user_input.strip()}]
                    response = st.session_state.interceptor.generate(messages=messages)

                    # Clear input by rerunning
                    st.rerun()

                except Exception as e:
                    st.error("⚠️ Message could not be processed")
                    st.info("💡 Please try sending your message again")
                    logging.error(f"Message processing error: {e}")  # Log technical details

    # ========================================================================
    # RIGHT SIDEBAR PANEL - Observation Deck (slides in/out)
    # ========================================================================

    deck_is_open = st.session_state.deck_manager.session_state['observation_deck']['is_open']
    panel_class = "observation-deck-panel open" if deck_is_open else "observation-deck-panel"

    # Inject right panel HTML and conditional CSS for content adjustment
    content_margin = "420px" if deck_is_open else "0"

    # Get dark mode state for Observation Deck colors
    dark_mode = st.session_state.get('dark_mode', False)

    # Set colors based on dark mode
    if dark_mode:
        deck_bg = "#1e1e1e"
        deck_border = "#444"
        deck_title_color = "#fff"
        deck_text_color = "#aaa"
        deck_hr_color = "#444"
        deck_subtitle_color = "#888"
    else:
        deck_bg = "#f0f2f6"
        deck_border = "#ddd"
        deck_title_color = "#262730"
        deck_text_color = "#555"
        deck_hr_color = "#ddd"
        deck_subtitle_color = "#888"

    # Calculate centering offset - shift content left by half the deck width to center it
    center_offset = "210px" if deck_is_open else "0"  # Half of 420px deck width

    st.markdown(f"""
        <style>
        /* Center ENTIRE main section when Observation Deck is open */
        section[data-testid="stMain"] {{
            margin-right: {content_margin} !important;
            margin-left: {center_offset} !important;
            transition: margin-right 0.3s ease, margin-left 0.3s ease;
            max-width: calc(100vw - 260px - {content_margin} - {center_offset}) !important;
        }}

        /* Center and resize chat input container to remain fully interactable */
        .stChatFloatingInputContainer {{
            position: fixed !important;
            bottom: 0 !important;
            left: calc(260px + {center_offset}) !important;  /* 260px = sidebar width */
            right: {content_margin} !important;
            width: calc(100vw - 260px - {content_margin} - {center_offset}) !important;
            max-width: calc(100vw - 260px - {content_margin} - {center_offset}) !important;
            transition: left 0.3s ease, right 0.3s ease, width 0.3s ease;
            z-index: 1000 !important;  /* Keep chat input above observation deck */
            box-sizing: border-box !important;
            pointer-events: auto !important;  /* Ensure it's clickable */
        }}

        /* Also adjust the input field itself */
        .stChatFloatingInputContainer > div {{
            width: 100% !important;
            max-width: 100% !important;
        }}

        /* Adjust header area to stay centered */
        .main > div:first-child {{
            margin-right: {content_margin} !important;
            margin-left: {center_offset} !important;
            transition: margin-right 0.3s ease, margin-left 0.3s ease;
        }}

        /* Observation Deck panel colors (dynamic based on dark mode) */
        .observation-deck-panel {{
            background: {deck_bg} !important;
            border-left-color: {deck_border} !important;
        }}
        </style>

        <div class="{panel_class}" id="observation-deck">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                <h2 style="color: {deck_title_color}; margin: 0;">Observation Deck</h2>
                <span class="deck-close-btn" id="deck-close-logo"
                      style="font-size: 5rem; color: {deck_title_color}; padding: 5px 10px; line-height: 0.8; cursor: pointer; transition: opacity 0.2s;"
                      title="Click to close Observation Deck"
                      onmouseover="this.style.opacity='0.6'"
                      onmouseout="this.style.opacity='1'"
                      onclick="closeDeck()">🔭</span>
            </div>
            <p style="color: {deck_text_color};">TELOSCOPIC tools and research instruments will appear here.</p>
            <hr style="border-color: {deck_hr_color};">
            <div id="deck-content">
                <!-- Observation Deck content will be rendered here -->
                <p style="color: {deck_subtitle_color}; font-style: italic;">Click the telescope logo or use the Observatory toggle to close this panel.</p>
            </div>
        </div>

        <script>
        function closeDeck() {{
            // Find the Observatory toggle by searching through all toggle labels
            const toggleContainers = parent.document.querySelectorAll('[data-testid="stToggle"]');

            for (let container of toggleContainers) {{
                // Check if this toggle has "Observatory" text
                const labels = container.querySelectorAll('p, label, span');
                for (let label of labels) {{
                    if (label.textContent && label.textContent.includes('Observatory')) {{
                        // Find the actual button/input element to click
                        const button = container.querySelector('button[role="switch"]');
                        if (button) {{
                            button.click();
                            return;
                        }}
                        // Fallback: try to find input element
                        const input = container.querySelector('input[type="checkbox"]');
                        if (input) {{
                            input.click();
                            return;
                        }}
                        // Last resort: click the container itself
                        container.click();
                        return;
                    }}
                }}
            }}

            // Debug: log if toggle not found
            console.log('Observatory toggle not found');
        }}
        </script>
    """, unsafe_allow_html=True)

    # FINAL CSS INJECTION - Applied at end of rendering to override everything
    st.markdown("""
    <style>
    /* FINAL OVERRIDE - Toggle switches MUST be gray */
    button[role="switch"] {
        background-color: #888888 !important;
        background: linear-gradient(#888888, #888888) !important;
    }

    button[role="switch"][aria-checked="true"] {
        background-color: #999999 !important;
        background: linear-gradient(#999999, #999999) !important;
    }

    /* Override baseui styles directly */
    [data-baseweb="toggle"] {
        background-color: #888888 !important;
    }

    [data-baseweb="toggle"][aria-checked="true"] {
        background-color: #999999 !important;
    }

    /* Target inner thumb/track elements */
    button[role="switch"] > div {
        background-color: inherit !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # ========================================================================
    # Observation Deck Panel Integration (Phase 2)
    # ========================================================================
    # Render the Observation Deck panel with Phase 2 research instruments
    # This panel displays TELOSCOPE controls, math breakdowns, and counterfactual comparisons
    render_observation_deck_panel()


# ============================================================================
# Steward Research Panel - AI-powered session analysis
# ============================================================================

def render_steward_research_panel():
    """Render Steward research analysis interface - overlays normal sidebar."""

    # Initialize Steward analyzer - reuse existing Mistral client for efficiency
    if 'steward_analyzer' not in st.session_state:
        # Pass the same Mistral client used by TELOS governance
        mistral_client = st.session_state.get('llm', None)
        st.session_state.steward_analyzer = StewardAnalyzer(mistral_client=mistral_client)

    analyzer = st.session_state.steward_analyzer

    st.markdown("### 🤖 Research Analysis")
    st.caption("Interact with Steward to analyze session data")

    # Show AI status
    if analyzer.has_ai:
        st.success("✅ AI analysis enabled (Mistral)")
    else:
        st.warning("⚠️ AI unavailable - set MISTRAL_API_KEY")

    # Show capabilities info
    with st.expander("ℹ️ What Steward Can Do", expanded=False):
        st.markdown("""
**Steward is a specialized research analysis instrument.**

✅ **CAN DO:**
- Analyze governance patterns (TELOS on/off effects)
- Extract canonical inputs and primacy attractors
- Identify conversation patterns
- Generate session summaries and metrics
- Answer research questions about session data

❌ **CANNOT DO:**
- Modify code or files
- Run system commands
- Access external resources
- Perform actions outside research scope

Ask Steward about **patterns, metrics, and insights** from your TELOS sessions.
        """)

    st.divider()

    # ========================================================================
    # Session Selection
    # ========================================================================
    st.subheader("📊 Session Scope")

    # Get list of saved sessions
    sessions_dir = Path("saved_sessions")
    session_options = ["Current Session"]

    if sessions_dir.exists():
        saved_sessions = sorted(
            sessions_dir.glob("session_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        session_options += [f.stem.replace("session_", "") for f in saved_sessions[:20]]

    analysis_session = st.selectbox(
        "Analyze Session",
        options=session_options,
        key="steward_session_selector",
        help="Select which session to analyze"
    )

    st.divider()

    # ========================================================================
    # Quick Analysis Options
    # ========================================================================
    st.subheader("⚡ Quick Analysis")

    quick_analysis = st.selectbox(
        "Select Analysis Type",
        options=[
            "--- Select Analysis ---",
            "Governance Impact Assessment",
            "Canonical Input Extraction",
            "Conversation Pattern Analysis",
            "Primacy Attractor Evolution",
            "Turn-by-Turn Metrics",
            "Session Summary Report"
        ],
        key="steward_quick_analysis"
    )

    if st.button("🔍 Run Analysis", use_container_width=True):
        if quick_analysis != "--- Select Analysis ---":
            with st.spinner(f"Analyzing: {quick_analysis}..."):
                # Get session data
                if analysis_session == "Current Session":
                    # Use current session from session_state
                    if st.session_state.get('teloscope_initialized', False):
                        session_data_raw = st.session_state.web_session.export_session()
                        session_data = json.loads(session_data_raw) if isinstance(session_data_raw, str) else session_data_raw
                    else:
                        st.error("No active session to analyze")
                        return
                else:
                    # Load saved session
                    session_data = analyzer.load_session_data(analysis_session)
                    if not session_data:
                        st.error(f"Failed to load session: {analysis_session}")
                        return

                # Run appropriate analysis
                analysis_map = {
                    'Governance Impact Assessment': analyzer.analyze_governance_impact,
                    'Canonical Input Extraction': analyzer.extract_canonical_inputs,
                    'Conversation Pattern Analysis': analyzer.analyze_conversation_patterns,
                    'Primacy Attractor Evolution': analyzer.analyze_primacy_attractor_evolution,
                    'Session Summary Report': analyzer.generate_session_summary
                }

                analysis_func = analysis_map.get(quick_analysis)
                if analysis_func:
                    results = analysis_func(session_data)

                    st.session_state.steward_last_analysis = {
                        'type': quick_analysis,
                        'session': analysis_session,
                        'timestamp': datetime.now().isoformat(),
                        'results': results
                    }
                    st.success(f"✅ Analysis complete: {quick_analysis}")
                    st.rerun()
                else:
                    st.error("Analysis type not yet implemented")

    st.divider()

    # ========================================================================
    # Chat with Steward
    # ========================================================================
    st.subheader("💬 Ask Steward")

    # Initialize Steward chat history
    if 'steward_chat_history' not in st.session_state:
        st.session_state.steward_chat_history = []

    # Display chat history
    if st.session_state.steward_chat_history:
        with st.container():
            for msg in st.session_state.steward_chat_history[-5:]:  # Show last 5 messages
                if msg['role'] == 'user':
                    st.markdown(f"**You:** {msg['content']}")
                else:
                    st.markdown(f"**Steward:** {msg['content']}")

    # Chat input
    steward_query = st.text_area(
        "Research Question",
        placeholder="Ask Steward about the session...\nExample: 'What governance patterns emerged in this conversation?'",
        key="steward_chat_input",
        height=100
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("📤 Ask Steward", use_container_width=True):
            if steward_query.strip():
                # Add user message to history
                st.session_state.steward_chat_history.append({
                    'role': 'user',
                    'content': steward_query.strip()
                })

                # Get Steward's AI response
                with st.spinner("Steward is analyzing..."):
                    # Get session data
                    if analysis_session == "Current Session":
                        if st.session_state.get('teloscope_initialized', False):
                            session_data_raw = st.session_state.web_session.export_session()
                            session_data = json.loads(session_data_raw) if isinstance(session_data_raw, str) else session_data_raw
                        else:
                            session_data = {}
                    else:
                        session_data = analyzer.load_session_data(analysis_session)
                        if not session_data:
                            session_data = {}

                    # Get AI response from Steward
                    response = analyzer.chat_with_steward(
                        steward_query.strip(),
                        session_data,
                        context=f"Analyzing session: {analysis_session}"
                    )

                    st.session_state.steward_chat_history.append({
                        'role': 'steward',
                        'content': response
                    })

                st.rerun()

    with col2:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.steward_chat_history = []
            st.rerun()

    st.divider()

    # ========================================================================
    # Analysis Results Display
    # ========================================================================
    if 'steward_last_analysis' in st.session_state:
        with st.expander("📋 Last Analysis Results", expanded=True):
            analysis = st.session_state.steward_last_analysis
            st.markdown(f"**Type:** {analysis['type']}")
            st.markdown(f"**Session:** {analysis['session']}")
            st.markdown(f"**Time:** {analysis['timestamp']}")

            st.divider()

            # Display results based on analysis type
            results = analysis.get('results', {})

            if 'error' in results:
                st.error(results['error'])
            else:
                # Governance Impact Assessment
                if analysis['type'] == 'Governance Impact Assessment':
                    st.metric("Total Turns", results.get('total_turns', 0))
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("TELOS Enabled", results.get('telos_enabled_turns', 0))
                    with col2:
                        st.metric("TELOS Disabled", results.get('telos_disabled_turns', 0))

                    st.metric("Governance Switches", results.get('governance_switches', 0))
                    ratio = results.get('governance_ratio', 0)
                    st.progress(ratio)
                    st.caption(f"TELOS usage: {ratio:.1%}")

                    if 'ai_interpretation' in results:
                        st.markdown("**AI Interpretation:**")
                        st.info(results['ai_interpretation'])

                # Canonical Input Extraction
                elif analysis['type'] == 'Canonical Input Extraction':
                    st.metric("Total Canonical Inputs", results.get('total_canonical_inputs', 0))

                    inputs = results.get('inputs', [])
                    if inputs:
                        for inp in inputs[:5]:  # Show first 5
                            with st.container():
                                st.markdown(f"**Turn {inp['turn']}:**")
                                st.text(inp['content'])
                                st.caption(f"Primacy: {inp.get('primacy_attractor', 'N/A')}")

                    if 'ai_categorization' in results:
                        st.markdown("**AI Categorization:**")
                        st.info(results['ai_categorization'])

                # Conversation Pattern Analysis
                elif analysis['type'] == 'Conversation Pattern Analysis':
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Turns", results.get('total_turns', 0))
                    with col2:
                        st.metric("User Turns", results.get('user_turns', 0))
                    with col3:
                        st.metric("Assistant Turns", results.get('assistant_turns', 0))

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Avg User Msg Length", f"{results.get('avg_user_message_length', 0)} chars")
                    with col2:
                        st.metric("Avg Assistant Msg Length", f"{results.get('avg_assistant_message_length', 0)} chars")

                    if 'ai_pattern_analysis' in results:
                        st.markdown("**AI Pattern Analysis:**")
                        st.info(results['ai_pattern_analysis'])

                # Primacy Attractor Evolution
                elif analysis['type'] == 'Primacy Attractor Evolution':
                    st.metric("Total Primacy Attractors", results.get('total_primacy_attractors', 0))

                    sequence = results.get('sequence', [])
                    if sequence:
                        st.markdown("**Evolution Sequence:**")
                        for item in sequence[:10]:  # Show first 10
                            st.text(f"Turn {item['turn']}: {item['attractor']}")

                    if 'ai_evolution_analysis' in results:
                        st.markdown("**AI Evolution Analysis:**")
                        st.info(results['ai_evolution_analysis'])

                # Session Summary Report
                elif analysis['type'] == 'Session Summary Report':
                    if 'executive_summary' in results:
                        st.markdown("### Executive Summary")
                        st.info(results['executive_summary'])

                    st.markdown("### Detailed Metrics")

                    # Show summary of each component
                    if 'governance_summary' in results:
                        with st.expander("Governance Impact"):
                            gov = results['governance_summary']
                            st.json(gov)

                    if 'canonical_inputs_summary' in results:
                        with st.expander("Canonical Inputs"):
                            can = results['canonical_inputs_summary']
                            st.metric("Total", can.get('total_canonical_inputs', 0))

                    if 'pattern_summary' in results:
                        with st.expander("Conversation Patterns"):
                            pat = results['pattern_summary']
                            st.json(pat)


# ============================================================================
# Observatory Control Strip - Turn-by-Turn Metrics Display
# ============================================================================

def render_observatory_control_strip():
    """
    Render Observatory control strip at top right showing current turn metrics.

    Features:
    - Turn counter (current / total)
    - Fidelity score with gold theming
    - Status icon and text
    - Clickable TELOSCOPE icon to toggle Observation Deck
    - Fixed position at top right
    """
    # Get current state
    deck_manager = st.session_state.get('deck_manager')
    if not deck_manager:
        return

    session_manager = st.session_state.get('session_manager')
    if not session_manager:
        return

    turns = session_manager.get_all_turns()
    if not turns:
        return

    current_turn_idx = deck_manager.session_state['observation_deck'].get('current_turn', 0)
    deck_expanded = deck_manager.session_state['observation_deck'].get('is_open', False)

    # Clamp turn index
    if current_turn_idx >= len(turns):
        current_turn_idx = len(turns) - 1

    # Get current turn data
    current_turn = turns[current_turn_idx]

    # Extract metrics
    fidelity = current_turn.get('fidelity', 1.0)
    fidelity_display = f"{fidelity:.2f}" if fidelity is not None else "Cal"

    # Status determination (Goldilocks zones)
    status_icon = "✓"
    status_text = "Aligned"
    if fidelity is not None:
        if fidelity < 0.67:  # Goldilocks: Significant Drift
            status_icon = "⚠️"
            status_text = "Drift"
        elif fidelity < 0.76:  # Goldilocks: Below Aligned
            status_icon = "⚡"
            status_text = "Watch"

    # Add CSS and HTML for control strip
    active_class = "active" if deck_expanded else ""

    control_html = f"""
    <style>
    .control-strip {{
        position: fixed;
        top: 60px;
        right: 20px;
        background: rgba(0, 0, 0, 0.8);
        backdrop-filter: blur(10px);
        padding: 0.75rem 1.25rem;
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        z-index: 1000;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        cursor: pointer;
        transition: all 0.3s ease;
    }}

    .control-strip:hover {{
        background: rgba(20, 30, 40, 0.9);
        border: 1px solid rgba(255, 215, 0, 0.3);
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.4);
    }}

    .control-strip.active {{
        border: 1px solid rgba(255, 215, 0, 0.5);
        background: rgba(30, 40, 50, 0.9);
    }}

    .control-strip-hint {{
        font-size: 0.65rem;
        color: #666;
        text-align: center;
        margin-top: 0.25rem;
    }}
    </style>

    <div class="control-strip {active_class}">
        <div style="display: flex; align-items: center; gap: 1rem;">
            <div>
                <div style="font-size: 0.75rem; color: #888; margin-bottom: 0.25rem;">Turn</div>
                <div style="font-size: 1.25rem; font-weight: bold; color: #FFF;">{current_turn_idx + 1} / {len(turns)}</div>
            </div>
            <div style="border-left: 1px solid rgba(255,255,255,0.2); height: 40px;"></div>
            <div>
                <div style="font-size: 0.75rem; color: #888; margin-bottom: 0.25rem;">Fidelity</div>
                <div style="font-size: 1.25rem; font-weight: bold; color: #F4D03F;">{fidelity_display}</div>
            </div>
            <div style="border-left: 1px solid rgba(255,255,255,0.2); height: 40px;"></div>
            <div>
                <div style="font-size: 0.75rem; color: #888; margin-bottom: 0.25rem;">Status</div>
                <div style="font-size: 1rem; color: #FFF;">{status_icon} {status_text}</div>
            </div>
            <div style="border-left: 1px solid rgba(255,255,255,0.2); height: 40px;"></div>
            <div style="text-align: center;">
                <div style="font-size: 1.2rem; color: #F4D03F;">🔭</div>
                <div class="control-strip-hint">Click here</div>
            </div>
        </div>
    </div>
    """

    st.markdown(control_html, unsafe_allow_html=True)

    # Clickable button to toggle deck (positioned below the visual strip)
    st.markdown("<div style='margin-top: 110px;'></div>", unsafe_allow_html=True)

    # Create columns to position button at top right
    col1, col2 = st.columns([4, 1])
    with col2:
        if st.button("🔭 Observation Deck", key="control_strip_toggle_btn", help="Toggle Observation Deck", use_container_width=True):
            deck_manager.toggle_deck()
            st.rerun()


# ============================================================================
# Observation Deck Panel - Phase 2 Integration
# ============================================================================

def load_phase2_data_for_turn(turn_number):
    """
    Load Phase 2 counterfactual data for a specific turn.

    Args:
        turn_number: The turn number to load data for

    Returns:
        dict: Phase 2 data with original/telos branches, or None if not available
    """
    import json
    from pathlib import Path

    # Check if Phase 2 data directory exists
    phase2_dir = Path("telos_observatory/phase2_validation_claude_test_1/study_results")
    if not phase2_dir.exists():
        return None

    # Look for intervention files that match this turn
    # Search in subdirectories for intervention JSON files
    for subdir in phase2_dir.iterdir():
        if subdir.is_dir():
            for json_file in subdir.glob("intervention_*.json"):
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)

                    # Check if this intervention includes our turn
                    trigger_turn = data.get('trigger_turn', 0)
                    num_turns = data.get('num_turns', 5)

                    if trigger_turn <= turn_number < trigger_turn + num_turns:
                        return data
                except Exception as e:
                    continue

    return None


def render_teloscope_controls_compact():
    """
    Render compact TELOSCOPE navigation controls.

    Displays:
    - Turn counter with prev/next buttons
    - Quick jump to intervention points
    - Sync status indicator
    """
    st.markdown("### 🔭 TELOSCOPE Navigation")

    # Get current turn from session state
    current_turn = st.session_state.deck_manager.session_state['observation_deck'].get('current_turn', 0)

    # Get total turns
    if hasattr(st.session_state, 'session_manager'):
        turns = st.session_state.session_manager.get_all_turns()
        max_turn = len(turns) - 1
    else:
        max_turn = 0

    # Navigation controls in columns
    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        if st.button("◀", key="teloscope_prev", disabled=(current_turn <= 0)):
            new_turn = max(0, current_turn - 1)
            st.session_state.deck_manager.set_current_turn(new_turn)
            st.rerun()

    with col2:
        st.markdown(f"<div style='text-align: center; font-size: 1.2em; font-weight: bold;'>Turn {current_turn + 1}/{max_turn + 1}</div>", unsafe_allow_html=True)

    with col3:
        if st.button("▶", key="teloscope_next", disabled=(current_turn >= max_turn)):
            new_turn = min(max_turn, current_turn + 1)
            st.session_state.deck_manager.set_current_turn(new_turn)
            st.rerun()

    st.divider()


def render_math_breakdown_section(turn_data):
    """
    Render mathematical breakdown section showing calculations.

    Args:
        turn_data: Turn data dictionary with metrics

    Displays:
    - Fidelity calculation steps
    - Embedding distances
    - Lyapunov metrics (if available)
    - Statistical indicators
    """
    st.markdown("### 🧮 Mathematical Breakdown")

    if not turn_data:
        st.info("No mathematical data available for this turn")
        return

    # Display fidelity metrics if available
    fidelity = turn_data.get('fidelity')
    if fidelity is not None:
        st.metric("Fidelity Score", f"{fidelity:.4f}")

        # Visual fidelity bar
        fidelity_color = "green" if fidelity >= 0.76 else "gold" if fidelity >= 0.73 else "orange" if fidelity >= 0.67 else "red"  # Goldilocks zones
        st.markdown(f"""
        <div style="background-color: #f0f0f0; border-radius: 5px; padding: 5px; margin: 10px 0;">
            <div style="background-color: {fidelity_color}; width: {fidelity * 100}%; height: 20px; border-radius: 3px;"></div>
        </div>
        """, unsafe_allow_html=True)

    # Display other metrics in expandable sections
    with st.expander("📊 Detailed Metrics", expanded=False):
        # Show raw metrics if available
        metrics_to_show = {k: v for k, v in turn_data.items() if k not in ['user_input', 'assistant_response', 'full_response']}
        if metrics_to_show:
            st.json(metrics_to_show)
        else:
            st.caption("No additional metrics available")

    # Show intervention status if available
    if turn_data.get('intervention_applied'):
        st.success("🛡️ TELOS Intervention Applied")

    st.divider()


def render_counterfactual_section(turn_data):
    """
    Render Phase 2/2B counterfactual comparison section.

    Args:
        turn_data: Turn data dictionary

    Displays:
    - Original vs TELOS branch comparison
    - Side-by-side responses
    - Fidelity trajectory graphs
    - Delta metrics
    """
    st.markdown("### 🔀 Counterfactual Comparison")

    # Get current turn number
    current_turn = st.session_state.deck_manager.session_state['observation_deck'].get('current_turn', 0)

    # Load Phase 2 data for this turn
    phase2_data = load_phase2_data_for_turn(current_turn)

    if not phase2_data:
        st.info("No Phase 2 counterfactual data available for this turn")
        st.caption("Phase 2 data includes intervention comparisons from validation studies")
        return

    # Display summary metrics
    st.markdown("#### Comparison Summary")
    col1, col2, col3 = st.columns(3)

    with col1:
        trigger_turn = phase2_data.get('trigger_turn', 'N/A')
        st.metric("Trigger Turn", trigger_turn)

    with col2:
        delta_f = phase2_data.get('comparison', {}).get('delta_f', 0)
        st.metric("ΔF (Improvement)", f"{delta_f:+.4f}")

    with col3:
        governance_effective = phase2_data.get('comparison', {}).get('governance_effective', False)
        status = "✅ Yes" if governance_effective else "❌ No"
        st.metric("Governance Effective", status)

    st.divider()

    # Display side-by-side comparison
    st.markdown("#### Branch Comparison")

    # Get the specific turn data from both branches
    original_branch = phase2_data.get('original', {})
    telos_branch = phase2_data.get('telos', {})

    # Find the turn in both branches
    trigger_turn = phase2_data.get('trigger_turn', 0)
    turn_offset = current_turn - trigger_turn

    if 0 <= turn_offset < len(original_branch.get('turns', [])):
        original_turn = original_branch['turns'][turn_offset]
        telos_turn = telos_branch['turns'][turn_offset]

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**📍 Original Branch**")
            st.caption(f"Fidelity: {original_turn.get('fidelity', 'N/A'):.4f}")
            with st.expander("View Response", expanded=False):
                response = original_turn.get('assistant_response', 'N/A')
                st.text_area("Response", response[:500] + "..." if len(response) > 500 else response,
                           height=200, key=f"orig_resp_{current_turn}", disabled=True)

        with col2:
            st.markdown("**🛡️ TELOS Branch**")
            intervention_applied = telos_turn.get('intervention_applied', False)
            fidelity_display = f"{telos_turn.get('fidelity', 'N/A'):.4f}"
            if intervention_applied:
                fidelity_display += " (Intervention)"
            st.caption(f"Fidelity: {fidelity_display}")
            with st.expander("View Response", expanded=False):
                response = telos_turn.get('assistant_response', 'N/A')
                st.text_area("Response", response[:500] + "..." if len(response) > 500 else response,
                           height=200, key=f"telos_resp_{current_turn}", disabled=True)

    st.divider()

    # Display fidelity trajectories
    st.markdown("#### Fidelity Trajectory")

    original_trajectory = original_branch.get('fidelity_trajectory', [])
    telos_trajectory = telos_branch.get('fidelity_trajectory', [])

    if original_trajectory and telos_trajectory:
        import pandas as pd

        # Create DataFrame for plotting
        max_len = max(len(original_trajectory), len(telos_trajectory))
        turns_range = list(range(trigger_turn + 1, trigger_turn + 1 + max_len))

        trajectory_df = pd.DataFrame({
            'Turn': turns_range,
            'Original': original_trajectory + [None] * (max_len - len(original_trajectory)),
            'TELOS': telos_trajectory + [None] * (max_len - len(telos_trajectory))
        })

        # Display as line chart
        st.line_chart(trajectory_df.set_index('Turn'))

        # Display final metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Original Final F", f"{original_branch.get('final_fidelity', 'N/A'):.4f}")
        with col2:
            st.metric("TELOS Final F", f"{telos_branch.get('final_fidelity', 'N/A'):.4f}")

    # Display attractor information
    if 'attractor' in phase2_data:
        with st.expander("🎯 Primacy Attractor", expanded=False):
            attractor = phase2_data['attractor']

            if 'purpose' in attractor:
                st.markdown("**Purpose:**")
                for purpose in attractor['purpose']:
                    st.markdown(f"- {purpose}")

            if 'scope' in attractor:
                st.markdown("**Scope:**")
                for scope_item in attractor['scope']:
                    st.markdown(f"- {scope_item}")

            if 'boundaries' in attractor:
                st.markdown("**Boundaries:**")
                for boundary in attractor['boundaries']:
                    st.markdown(f"- {boundary}")


def render_observation_deck_panel():
    """
    Main Observation Deck panel renderer.

    Integrates all Phase 2 components:
    - TELOSCOPE navigation controls
    - Mathematical breakdown display
    - Counterfactual comparison viewer

    This panel slides in from the right when Observatory toggle is enabled.
    """
    # Check if deck is open
    if not st.session_state.deck_manager.session_state['observation_deck']['is_open']:
        return

    # Get current turn
    current_turn = st.session_state.deck_manager.session_state['observation_deck'].get('current_turn', 0)

    # Get turn data from session manager
    turn_data = None
    if hasattr(st.session_state, 'session_manager'):
        turns = st.session_state.session_manager.get_all_turns()
        if 0 <= current_turn < len(turns):
            turn_data = turns[current_turn]

    # Fixed-position panel CSS
    st.markdown("""
    <style>
    /* Observation Deck Panel Styles */
    .observation-deck-panel {
        position: fixed;
        right: 0;
        top: 0;
        height: 100vh;
        width: 400px;
        background-color: var(--background-color);
        border-left: 1px solid var(--border-color);
        padding: 20px;
        overflow-y: auto;
        z-index: 1000;
    }

    .observation-deck-header {
        font-size: 1.5em;
        font-weight: bold;
        margin-bottom: 20px;
        color: var(--text-color);
    }
    </style>
    """, unsafe_allow_html=True)

    # Render panel header
    st.markdown("## 🔭 Observation Deck")
    st.caption("Phase 2 Research Instruments")
    st.divider()

    # Render TELOSCOPE controls
    render_teloscope_controls_compact()

    # Render mathematical breakdown
    render_math_breakdown_section(turn_data)

    # Render counterfactual comparison
    render_counterfactual_section(turn_data)


# ============================================================================
# Sidebar: Configuration and Metrics
# ============================================================================

def render_sidebar():
    """Render minimalistic sidebar - only essential controls."""
    with st.sidebar:
        st.title("TELOS")

        # ========================================================================
        # Observation Deck Toggle (Telescope button to open/close right panel)
        # ========================================================================
        deck_is_open = st.session_state.deck_manager.session_state['observation_deck']['is_open']

        if st.button("🔭 Observation Deck" if not deck_is_open else "✖ Close Deck",
                     use_container_width=True,
                     type="secondary",
                     help="Toggle Observation Deck panel (research instruments)"):
            st.session_state.deck_manager.toggle_deck()
            st.rerun()

        st.divider()

        # Toggle for Steward Research Panel
        steward_mode = st.toggle(
            "🤖 STEWARD Research",
            value=st.session_state.get('steward_research_mode', False),
            key='steward_research_toggle',
            help="Open Steward research analysis panel"
        )
        st.session_state.steward_research_mode = steward_mode

        st.divider()

        # Conditionally render either normal sidebar or Steward research panel
        if steward_mode:
            render_steward_research_panel()
            return  # Exit early - don't render normal sidebar controls

        # ========================================================================
        # Saved Chats - Research instrument for later review
        # ========================================================================
        st.subheader("💬 Chats")

        # Get list of saved sessions
        sessions_dir = Path("saved_sessions")
        if sessions_dir.exists():
            saved_sessions = sorted(sessions_dir.glob("session_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)

            if saved_sessions:
                # Show dropdown of saved sessions
                session_options = ["Current Session"] + [f.stem.replace("session_", "") for f in saved_sessions[:20]]  # Show last 20
                selected_session = st.selectbox(
                    "Load Session",
                    options=session_options,
                    key="session_selector",
                    help="Select a saved session to review"
                )

                if selected_session != "Current Session":
                    if st.button("📂 Load Selected", use_container_width=True):
                        session_file = sessions_dir / f"session_{selected_session}.json"
                        try:
                            with open(session_file, 'r') as f:
                                loaded_data = json.load(f)
                                # Load into session state
                                st.session_state.web_session.load_session(loaded_data)
                                st.success(f"✅ Loaded session: {selected_session}")
                                st.rerun()
                        except Exception as e:
                            st.error(f"Failed to load session: {e}")
            else:
                st.caption("No saved sessions yet")
        else:
            st.caption("No saved sessions yet")

        # Save current session button
        if st.session_state.get('teloscope_initialized', False):
            if st.button("💾 Save Current", use_container_width=True, help="Save current session for later review"):
                try:
                    sessions_dir.mkdir(exist_ok=True)
                    session_id = st.session_state.current_session.get('session_id', datetime.now().strftime('%Y%m%d_%H%M%S'))
                    session_file = sessions_dir / f"session_{session_id}.json"

                    session_data_raw = st.session_state.web_session.export_session()
                    session_data = json.loads(session_data_raw) if isinstance(session_data_raw, str) else session_data_raw

                    with open(session_file, 'w') as f:
                        json.dump(session_data, f, indent=2)

                    st.success(f"✅ Session saved: {session_id}")
                except Exception as e:
                    st.error(f"Failed to save session: {e}")

        st.divider()

        # ========================================================================
        # Reset Button
        # ========================================================================
        if st.button("🔄 Reset Session", type="secondary", use_container_width=True,
                    help="Reset current conversation"):
            if st.session_state.get('teloscope_initialized', False):
                st.session_state.interceptor.reset_session()
                st.session_state.session_manager.clear_session()
                st.session_state.web_session.clear_web_session()
                st.success("✅ Session reset")
                st.rerun()

        st.divider()

        # ========================================================================
        # Export Evidence
        # ========================================================================
        with st.expander("💾 Export Evidence", expanded=False):
                if st.session_state.get('teloscope_initialized', False):
                    with st.spinner('📥 Preparing exports...'):
                        session_data_raw = st.session_state.web_session.export_session()
                        # Parse JSON string if needed
                        session_data = json.loads(session_data_raw) if isinstance(session_data_raw, str) else session_data_raw
                        session_id = st.session_state.current_session.get('session_id', 'unknown')
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

                    st.markdown("**📦 Choose Export Format:**")
                    st.write("")

                    # Row 1: JSON and CSV
                    exp_col1, exp_col2 = st.columns(2)

                    with exp_col1:
                        st.download_button(
                            label="📄 JSON Data",
                            data=json.dumps(session_data, indent=2),
                            file_name=f"teloscope_{session_id}_{timestamp}.json",
                            mime="application/json",
                            use_container_width=True,
                            help="Complete session data (machine-readable)"
                        )

                    with exp_col2:
                        st.download_button(
                            label="📊 CSV Telemetry",
                            data=export_session_to_csv(session_data),
                            file_name=f"teloscope_{session_id}_{timestamp}.csv",
                            mime="text/csv",
                            use_container_width=True,
                            help="Turn-by-turn metrics (Excel/R/Python)"
                        )

                    # Row 2: Transcript and Report
                    exp_col3, exp_col4 = st.columns(2)

                    with exp_col3:
                        st.download_button(
                            label="📝 Transcript",
                            data=generate_human_readable_transcript(session_data),
                            file_name=f"teloscope_{session_id}_{timestamp}_transcript.md",
                            mime="text/markdown",
                            use_container_width=True,
                            help="Human-readable conversation log"
                        )

                    with exp_col4:
                        st.download_button(
                            label="📋 HTML Report",
                            data=generate_governance_report_html(session_data),
                            file_name=f"teloscope_{session_id}_{timestamp}_report.html",
                            mime="text/html",
                            use_container_width=True,
                            help="Visual governance summary (open in browser)"
                        )

                    # Row 3: Complete Evidence Package
                    st.write("")
                    st.download_button(
                        label="🎁 Complete Evidence Package (ZIP)",
                        data=create_evidence_package_zip(session_data, f"{session_id}_{timestamp}"),
                        file_name=f"teloscope_evidence_{session_id}_{timestamp}.zip",
                        mime="application/zip",
                        use_container_width=True,
                        type="primary",
                        help="All formats bundled with README (recommended for research)"
                    )

                    st.caption("💡 **Tip:** Use the ZIP package for publications - it includes all formats plus a README")
                else:
                    st.info("Start a conversation to enable exports")

        st.divider()

        # ========================================================================
        # Help Section
        # ========================================================================
        with st.expander("❓ Help", expanded=False):
            st.markdown("""
            **TELOSCOPE Observatory**

            Generates counterfactual evidence of AI governance efficacy.

            **Tabs:**
            - 🔴 **Live**: Real-time conversation
            - ⏮️ **Replay**: Timeline scrubber
            - 🔭 **TELOSCOPE**: Evidence viewer
            - 📊 **Analytics**: Statistics

            **ΔF Metric:**
            Improvement in fidelity from governance intervention.

            ΔF > 0 → Governance works ✅
            ΔF = 0 → No effect
            ΔF < 0 → Needs tuning
            """)

        # Keyboard Shortcuts Section
        with st.expander("⌨️ Keyboard Shortcuts", expanded=False):
            st.markdown("""
            **✨ Floating Popup Windows:**
            - `ESC` : Toggle 🔍 Steward Lens (drag to move)
            - `Spacebar` : Toggle 🔭 TELOSCOPE (drag to move)
            - `↑` (Up Arrow) : Toggle 🛠️ TELOSCOPIC TOOLS (drag to move)
            - `↓` (Down Arrow) : Hide ALL windows

            **🧭 Navigation:**
            - `←` (Left Arrow) : Previous turn
            - `→` (Right Arrow) : Next turn

            **⌨️ Text Input:**
            - `Ctrl+Enter` / `Cmd+Enter` : Send message
            - `Shift+Enter` : New line in message

            **♿ Accessibility:**
            - `Tab` : Navigate between controls
            - `Enter` : Confirm selections

            💡 **Tip:** All popup windows are draggable! Click and drag the header to reposition.

            🎯 **New:** Floating, draggable popup windows overlay the chat without pushing content!
            """)

        # Help & Documentation Section
        with st.expander("❓ Help & Documentation", expanded=False):
            st.markdown("""
            ### Key Concepts

            **Fidelity (F):**
            Measures how well responses stay aligned with governance purpose (0-1 scale).
            - 🟢 **0.76+**: Aligned (Goldilocks optimized)
            - 🟡 **0.67-0.76**: Minor to Moderate Drift
            - 🔴 **<0.67**: Significant Drift requiring attention

            **Basin of Attraction:**
            The mathematical boundary defining acceptable governance drift. Responses within the basin maintain fidelity. Think of it as a "safe zone" where the Steward can keep conversations on track.

            **Error Signal (ε):**
            Measures deviation from governance target (0-1 scale). Lower is better. High error signals (>0.5) indicate the conversation is drifting away from its intended purpose.

            **Intervention:**
            Active Steward correction when drift exceeds threshold. Types:
            - **Reinforcement**: Strengthens attractor salience in context
            - **Regeneration**: Produces new response if first attempt drifted

            **TELOSCOPE:**
            Mathematical transparency window showing how governance works. Provides observable evidence through counterfactual comparison ("what if governance was off?").

            **Native vs TELOS:**
            - **Native**: Original LLM response before governance
            - **TELOS**: Governed response after Steward intervention
            - Toggle between them to see governance impact

            ### Getting Started

            1. **Start a Conversation**: Type a message in the Live Session tab
            2. **Watch the Steward**: See governance metrics in real-time
            3. **Compare Responses**: Toggle between Native and TELOS to see differences
            4. **Replay History**: Use Session Replay to navigate through conversation turns
            5. **View Evidence**: Check TELOSCOPE tab for counterfactual experiments

            ### Understanding Metrics

            - **Salience**: How prominent the governance purpose is in context
            - **Coupling**: How well the response aligns with the attractor
            - **Drift Distance**: Numerical measure of how far from purpose
            - **Lyapunov Value**: Stability indicator for governance system

            💡 **Tip**: Green metrics = good, yellow = watch closely, red = intervention needed
            """)


# ============================================================================
# Tab 1: Live Session
# ============================================================================

def render_live_session():
    """Render live conversation interface with real-time metrics."""
    st.title("🔴 Live Session")
    st.caption("Real-time conversation with automatic drift detection")

    # Mode selector with explicit key for state persistence
    mode = st.radio(
        "Mode",
        ["Live Chat", "Load & Replay"],
        horizontal=True,
        key="live_session_mode",  # Explicit key to prevent state conflicts
        help="Live Chat: Interactive conversation | Load & Replay: Process historical conversations"
    )

    st.divider()

    # ========================================================================
    # Load & Replay Mode
    # ========================================================================
    if mode == "Load & Replay":
        st.subheader("📂 Load Historical Conversation")
        st.caption("Upload a conversation file to analyze with pristine turn-by-turn streaming")

        # File uploader
        uploaded_file = st.file_uploader(
            "Upload conversation",
            type=['txt', 'json', 'md'],
            help="Supported formats: Plain text (Human:/Assistant:), JSON (ShareGPT), Markdown"
        )

        if uploaded_file:
            try:
                from telos_purpose.sessions.session_loader import SessionLoader
                import time

                # Save uploaded file temporarily
                temp_path = f"/tmp/{uploaded_file.name}"
                with open(temp_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())

                # Load session
                loader = SessionLoader()
                loaded_session = loader.load_session(temp_path)

                st.success(f"✅ Loaded {len(loaded_session.turns)} turns from {loaded_session.source_file}")
                st.session_state.loaded_session = loaded_session

                # Display session info
                with st.expander("📋 Session Details", expanded=False):
                    st.write(f"**Session ID:** {loaded_session.session_id}")
                    st.write(f"**Source File:** {loaded_session.source_file}")
                    st.write(f"**Format:** {loaded_session.format}")
                    st.write(f"**Total Turns:** {len(loaded_session.turns)}")
                    if loaded_session.metadata:
                        st.write(f"**Metadata:** {loaded_session.metadata}")

                st.divider()

                # Attractor Mode Selector
                st.subheader("🎯 Attractor Mode")
                attractor_mode = st.radio(
                    "Choose how primacy attractor is established:",
                    ["Pre-defined", "Progressive", "Hybrid"],
                    horizontal=True,
                    help="""
                    **Pre-defined**: Load attractor from config (prescribed purpose)
                    - Use for: Compliance, healthcare, finance with hard constraints
                    - Example: HIPAA-compliant medical chat

                    **Progressive**: Learn attractor from first 5 turns (emergent purpose)
                    - Use for: Exploratory conversations, research, discovery
                    - Example: Let conversation define its own topic naturally

                    **Hybrid**: Pre-defined boundaries + learned scope
                    - Use for: Soft boundaries with topic flexibility
                    - Example: "Must be respectful" but topic emerges organically
                    """
                )

                # Show mode-specific info
                if attractor_mode == "Pre-defined":
                    st.info("📋 Using governance profile from config.json")
                elif attractor_mode == "Progressive":
                    st.info("🌱 Attractor will be learned from first 5 turns of conversation")
                elif attractor_mode == "Hybrid":
                    st.info("🔀 Boundaries from config + scope learned from first 5 turns")

                st.divider()

                # Playback controls
                st.subheader("▶️ Playback Controls")

                col1, col2, col3 = st.columns([1, 1, 2])

                with col1:
                    play_button = st.button("▶️ Play", type="primary", use_container_width=True,
                                          help="Start pristine turn-by-turn replay")

                with col2:
                    stop_button = st.button("⏹️ Stop", use_container_width=True, type='secondary',
                                          help="Stop playback")

                with col3:
                    speed = st.slider(
                        "Speed (seconds per turn)",
                        min_value=0.1,
                        max_value=3.0,
                        value=0.5,
                        step=0.1,
                        help="Delay between processing each turn"
                    )

                st.divider()

                # PRISTINE STREAMING
                if play_button and 'loaded_session' in st.session_state:
                    st.subheader("🔬 Pristine Turn-by-Turn Analysis")
                    st.caption(f"Processing each turn with ONLY past context (no future knowledge) | Mode: {attractor_mode}")

                    loaded_session = st.session_state.loaded_session
                    progress_bar = st.progress(0, text="Starting replay...")

                    # Placeholder for real-time updates
                    status_placeholder = st.empty()
                    metrics_placeholder = st.empty()
                    conversation_placeholder = st.empty()

                    # Initialize based on attractor mode
                    if attractor_mode == "Pre-defined":
                        # Use existing steward (pre-defined mode)
                        steward = st.session_state.steward
                        progressive_extractor = None

                    elif attractor_mode == "Progressive":
                        # Initialize progressive extractor with statistical convergence
                        from telos_purpose.profiling.progressive_primacy_extractor import ProgressivePrimacyExtractor
                        progressive_extractor = ProgressivePrimacyExtractor(
                            llm_client=st.session_state.llm,
                            embedding_provider=st.session_state.embedding_provider,
                            mode='progressive',
                            window_size=8,
                            centroid_stability_threshold=0.95,
                            variance_stability_threshold=0.15,
                            confidence_threshold=0.75,
                            consecutive_stable_turns=3,
                            max_turns_safety=100
                        )
                        steward = None

                    elif attractor_mode == "Hybrid":
                        # Initialize hybrid extractor with seed attractor and statistical convergence
                        from telos_purpose.profiling.progressive_primacy_extractor import ProgressivePrimacyExtractor
                        progressive_extractor = ProgressivePrimacyExtractor(
                            llm_client=st.session_state.llm,
                            embedding_provider=st.session_state.embedding_provider,
                            mode='hybrid',
                            seed_attractor=st.session_state.attractor,
                            window_size=8,
                            centroid_stability_threshold=0.95,
                            variance_stability_threshold=0.15,
                            confidence_threshold=0.75,
                            consecutive_stable_turns=3,
                            max_turns_safety=100
                        )
                        steward = None

                    # Track context history (pristine - only past)
                    context_history = []

                    # Process each turn incrementally
                    for i, turn in enumerate(loaded_session.turns):
                        turn_num = i + 1
                        progress = turn_num / len(loaded_session.turns)

                        # Update progress
                        progress_bar.progress(
                            progress,
                            text=f"Processing turn {turn_num}/{len(loaded_session.turns)}..."
                        )

                        # Display current turn
                        with conversation_placeholder.container():
                            st.markdown(f"### Turn {turn_num}/{len(loaded_session.turns)}")

                            col1, col2 = st.columns([2, 1])

                            with col1:
                                st.markdown(f"**User:** {turn.user_message}")
                                st.markdown(f"**Assistant:** {turn.assistant_response}")

                            with col2:
                                st.caption("Processing with pristine context...")

                        # Process turn with ONLY past context (pristine)
                        try:
                            if attractor_mode == "Pre-defined":
                                # Use pre-defined steward
                                result = steward.process_turn(
                                    user_input=turn.user_message,
                                    model_response=turn.assistant_response
                                )
                                fidelity = result['metrics']['telic_fidelity']
                                distance = result['metrics'].get('drift_distance', 0.0)
                                error = result['metrics'].get('error_signal', 0.0)
                                basin = result['metrics'].get('primacy_basin_membership', True)
                                intervention_applied = result.get('intervention_applied', False)
                                governance_action = result.get('governance_action', 'unknown')

                            else:
                                # Use progressive extractor (progressive or hybrid)
                                extractor_result = progressive_extractor.add_turn(
                                    user_message=turn.user_message,
                                    assistant_response=turn.assistant_response
                                )

                                # Extract metrics
                                fidelity = extractor_result.get('fidelity')
                                distance = 0.0  # Not available in progressive mode yet
                                error = 1.0 - fidelity if fidelity is not None else 0.0
                                basin = fidelity >= 0.76 if fidelity is not None else True  # Goldilocks: Aligned threshold
                                intervention_applied = False  # Not yet implemented in progressive mode
                                governance_action = 'none'

                            # Display metrics
                            with metrics_placeholder.container():
                                metric_cols = st.columns(4)

                                with metric_cols[0]:
                                    if fidelity is not None:
                                        fidelity_color = "🟢" if fidelity >= 0.76 else ("🟡" if fidelity >= 0.67 else "🔴")  # Goldilocks zones
                                        st.metric(f"{fidelity_color} Fidelity", f"{fidelity:.3f}")
                                    else:
                                        st.metric("🔄 Fidelity", "Establishing...")

                                with metric_cols[1]:
                                    st.metric("Distance", f"{distance:.3f}" if distance else "N/A")

                                with metric_cols[2]:
                                    st.metric("Error", f"{error:.3f}" if error else "N/A")

                                with metric_cols[3]:
                                    basin_text = "Inside ✅" if basin else "Outside ❌"
                                    st.markdown(f"**Basin:** {basin_text}")

                                # Show baseline status for progressive/hybrid modes
                                if attractor_mode in ["Progressive", "Hybrid"]:
                                    if not progressive_extractor.is_ready():
                                        st.info(progressive_extractor.get_status_message())
                                    else:
                                        st.success(progressive_extractor.get_status_message())

                                # Show drift warning and trigger counterfactual if enabled
                                if fidelity is not None and fidelity < 0.76:  # Goldilocks: Aligned threshold
                                    st.warning(f"⚠️ DRIFT DETECTED (F={fidelity:.3f})")

                                    # Trigger counterfactual branching
                                    enable_counterfactuals = st.session_state.config.get('enable_counterfactuals', True)
                                    if enable_counterfactuals and turn_num not in [b.get('trigger_turn') for b in st.session_state.counterfactual_branches]:
                                        try:
                                            # Get remaining turns
                                            remaining_turns = [
                                                (t.user_message, t.assistant_response)
                                                for t in loaded_session.turns[i+1:]  # Skip current turn
                                            ]

                                            if len(remaining_turns) > 0:
                                                # Get attractor center
                                                if attractor_mode == "Pre-defined":
                                                    attractor_center = steward.attractor_center
                                                else:
                                                    # For progressive/hybrid, get from extractor if available
                                                    if progressive_extractor.is_ready():
                                                        attractor_center = progressive_extractor.attractor_centroid
                                                    else:
                                                        attractor_center = None

                                                if attractor_center is not None:
                                                    # Trigger counterfactual branching
                                                    branch_id = st.session_state.branch_manager.trigger_counterfactual(
                                                        trigger_turn=turn_num,
                                                        trigger_fidelity=fidelity,
                                                        trigger_reason=f"Drift detected (F={fidelity:.3f})",
                                                        conversation_history=context_history.copy(),
                                                        remaining_turns=remaining_turns,
                                                        attractor_center=attractor_center,
                                                        distance_scale=2.0
                                                    )

                                                    # Store branch info
                                                    st.session_state.counterfactual_branches.append({
                                                        'branch_id': branch_id,
                                                        'trigger_turn': turn_num,
                                                        'trigger_fidelity': fidelity
                                                    })

                                                    st.success(f"🌿 Counterfactual branches generated! Branch ID: {branch_id}")
                                        except Exception as e:
                                            st.error("🌿 Counterfactual generation interrupted")
                                            st.info("💡 This is non-critical. You can continue the conversation normally.")
                                            logging.error(f"Counterfactual generation failed: {e}")  # Log technical details

                                # Show intervention status
                                if intervention_applied:
                                    st.info(f"🔧 Intervention: {governance_action}")

                                # Research Mode OR Research Lens: Show mathematical observatory
                                show_research = (get_mode() == 'Research Mode') or st.session_state.get('research_lens_active', False)
                                if show_research:
                                    st.divider()
                                    # Add header for Research Lens overlay (not shown for Research Mode itself)
                                    if get_mode() != 'Research Mode' and st.session_state.get('research_lens_active', False):
                                        st.markdown("""
                                        <div style="background-color: #f0f8ff; padding: 10px; border-radius: 5px; border-left: 4px solid #27ae60;">
                                        <h4 style="margin: 0;">🔬 Research Lens - Live Mathematical Observatory</h4>
                                        <p style="margin: 5px 0 0 0; font-size: 0.9em;">Overlay active - showing calculations in real-time</p>
                                        </div>
                                        """, unsafe_allow_html=True)
                                        st.markdown("<br>", unsafe_allow_html=True)

                                    # Prepare embeddings dict
                                    embeddings_data = {
                                        'response_embedding': None,  # Not available in replay mode yet
                                        'user_embedding': None,
                                        'attractor_center': attractor_center if attractor_mode == "Pre-defined" else (progressive_extractor.attractor_centroid if progressive_extractor and progressive_extractor.is_ready() else None)
                                    }
                                    # Prepare metrics dict
                                    metrics_dict = {
                                        'telic_fidelity': fidelity if fidelity is not None else 0.0,
                                        'drift_distance': distance,
                                        'error_signal': error,
                                        'primacy_basin_membership': basin
                                    }
                                    # Prepare intervention data
                                    intervention_dict = {
                                        'intervention_applied': intervention_applied,
                                        'type': governance_action if intervention_applied else 'none'
                                    }
                                    # Call observatory function
                                    render_research_mode_observatory(
                                        metrics=metrics_dict,
                                        embeddings=embeddings_data,
                                        intervention_data=intervention_dict
                                    )

                            # Update status
                            if fidelity is not None:
                                status_placeholder.success(
                                    f"✅ Turn {turn_num} processed | "
                                    f"Context size: {len(context_history)} messages | "
                                    f"Fidelity: {fidelity:.3f}"
                                )
                            else:
                                status_placeholder.info(
                                    f"🔄 Turn {turn_num} processed | "
                                    f"Establishing baseline ({turn_num}/5)"
                                )

                        except Exception as e:
                            status_placeholder.error(f"❌ Error processing turn {turn_num}: {e}")
                            st.exception(e)
                            break

                        # Add to context history AFTER measurement (pristine isolation)
                        context_history.append({
                            'role': 'user',
                            'content': turn.user_message
                        })
                        context_history.append({
                            'role': 'assistant',
                            'content': turn.assistant_response
                        })

                        # Delay for visualization
                        time.sleep(speed)

                    # Completion
                    progress_bar.progress(1.0, text="✅ Replay complete!")

                    st.success(f"""
                    ### ✅ Replay Complete!

                    Processed {len(loaded_session.turns)} turns with pristine turn-by-turn streaming.

                    **Key Properties:**
                    - ✅ Each turn processed with ONLY past context
                    - ✅ No future knowledge leakage
                    - ✅ Incremental context building
                    - ✅ Real-time drift detection

                    Check the **TELOSCOPE** tab to view counterfactual evidence for any detected drift.
                    """)

                    # Offer to view results
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("🔭 View TELOSCOPE Evidence", type="primary", use_container_width=True):
                            st.session_state.active_tab = 2
                            st.rerun()

                    with col2:
                        if st.button("📊 View Analytics", use_container_width=True, type='secondary'):
                            st.session_state.active_tab = 3
                            st.rerun()

                    # Display counterfactual branches if any were generated
                    if len(st.session_state.counterfactual_branches) > 0:
                        st.markdown("---")
                        st.subheader("🌿 Counterfactual Interventions Generated")

                        st.info(f"""
                        **{len(st.session_state.counterfactual_branches)} intervention(s) triggered during replay.**

                        Each intervention shows:
                        - **Original Branch**: What actually happened (historical responses)
                        - **TELOS Branch**: What WOULD have happened with governance intervention

                        This is REAL API-based evidence of TELOS governance efficacy.
                        """)

                        # Display each branch
                        for branch_info in st.session_state.counterfactual_branches:
                            branch_id = branch_info['branch_id']
                            comparison = st.session_state.branch_manager.get_branch_comparison(branch_id)

                            if comparison:
                                with st.expander(
                                    f"🌿 Intervention at Turn {branch_info['trigger_turn']} "
                                    f"(F={branch_info['trigger_fidelity']:.3f})",
                                    expanded=True
                                ):
                                    # Summary metrics
                                    comp_data = comparison['comparison']
                                    col1, col2, col3, col4 = st.columns(4)

                                    with col1:
                                        st.metric(
                                            "Original Final F",
                                            f"{comp_data['original_final_f']:.3f}",
                                            delta=None
                                        )

                                    with col2:
                                        st.metric(
                                            "TELOS Final F",
                                            f"{comp_data['telos_final_f']:.3f}",
                                            delta=f"{comp_data['delta_f']:+.3f}"
                                        )

                                    with col3:
                                        delta_f = comp_data['delta_f']
                                        delta_color = "🟢" if delta_f > 0 else ("🔴" if delta_f < 0 else "🟡")
                                        st.metric(
                                            "Improvement (ΔF)",
                                            f"{delta_f:+.3f}",
                                            delta_color=delta_color
                                        )

                                    with col4:
                                        effective = "✅ YES" if comp_data['governance_effective'] else "❌ NO"
                                        st.markdown(f"**Governance Effective:** {effective}")

                                    # Side-by-side comparison
                                    st.markdown("### Side-by-Side Comparison")

                                    orig_turns = comparison['original']['turns']
                                    telos_turns = comparison['telos']['turns']

                                    for i in range(len(orig_turns)):
                                        st.markdown(f"**Turn {orig_turns[i]['turn_number']}**")

                                        # User input (same for both)
                                        st.markdown(f"👤 **User:** {orig_turns[i]['user_input']}")

                                        # Responses side-by-side
                                        col_orig, col_telos = st.columns(2)

                                        with col_orig:
                                            st.markdown("**Original (Historical)**")
                                            st.text_area(
                                                "Response",
                                                value=orig_turns[i]['assistant_response'],
                                                height=100,
                                                key=f"orig_{branch_id}_{i}",
                                                disabled=True
                                            )
                                            fid = orig_turns[i]['metrics']['telic_fidelity']
                                            fid_color = "🟢" if fid >= 0.76 else ("🟡" if fid >= 0.67 else "🔴")  # Goldilocks
                                            st.caption(f"{fid_color} Fidelity: {fid:.3f}")

                                        with col_telos:
                                            st.markdown("**TELOS (Counterfactual)**")
                                            st.text_area(
                                                "Response",
                                                value=telos_turns[i]['assistant_response'],
                                                height=100,
                                                key=f"telos_{branch_id}_{i}",
                                                disabled=True
                                            )
                                            fid = telos_turns[i]['metrics']['telic_fidelity']
                                            fid_color = "🟢" if fid >= 0.76 else ("🟡" if fid >= 0.67 else "🔴")  # Goldilocks
                                            st.caption(f"{fid_color} Fidelity: {fid:.3f}")

                                            if telos_turns[i]['intervention_applied']:
                                                st.success(f"🛡️ {telos_turns[i]['intervention_type']}")

                                        st.markdown("---")

                                    # Download evidence
                                    st.markdown("### 📥 Download Evidence")

                                    col_json, col_md = st.columns(2)

                                    with col_json:
                                        json_evidence = st.session_state.branch_manager.export_evidence(branch_id, format='json')
                                        if json_evidence:
                                            st.download_button(
                                                label="📄 Download JSON",
                                                data=json_evidence,
                                                file_name=f"telos_intervention_{branch_id}.json",
                                                mime="application/json",
                                                use_container_width=True
                                            )

                                    with col_md:
                                        md_evidence = st.session_state.branch_manager.export_evidence(branch_id, format='markdown')
                                        if md_evidence:
                                            st.download_button(
                                                label="📝 Download Markdown Report",
                                                data=md_evidence,
                                                file_name=f"telos_intervention_{branch_id}.md",
                                                mime="text/markdown",
                                                use_container_width=True
                                            )

            except Exception as e:
                st.error("📂 Unable to load session file")
                st.info("💡 **What to try:**\n- Check the file format is correct (JSON)\n- Try exporting a new session\n- Upload a different file")
                logging.error(f"Session loading failed: {e}")  # Log technical details

        else:
            st.info("""
            ### 📂 Upload a conversation file to begin

            **Supported Formats:**

            1. **Plain Text** (.txt)
               ```
               Human: What is TELOS?
               Assistant: TELOS is a framework...
               ```

            2. **JSON** (.json) - ShareGPT format
               ```json
               {
                 "turns": [
                   {"from": "human", "value": "What is TELOS?"},
                   {"from": "assistant", "value": "TELOS is..."}
                 ]
               }
               ```

            3. **Markdown** (.md)
               ```markdown
               **Human:** What is TELOS?
               **Assistant:** TELOS is...
               ```

            **What happens during replay:**
            - Each turn is processed with ONLY past context
            - No future knowledge leakage
            - Real-time drift detection
            - Automatic counterfactual triggering
            """)

        return  # Exit early for Load & Replay mode

    # ========================================================================
    # Live Chat Mode (Original functionality)
    # ========================================================================
    st.subheader("💬 Live Conversation")

    # Display conversation history
    turns = st.session_state.current_session.get('turns', [])

    # Welcome message for first-time users
    if not turns:
        st.info("👋 **Welcome to TELOS Observatory!**")
        st.markdown("""
        **Getting Started:**
        1. 💬 Type a message below to start a governed conversation
        2. 🔍 Watch the Steward Lens to see governance in action
        3. ⚖️ Use the toggle to compare Native vs TELOS responses
        4. ⏮️ Navigate through turns to see how governance evolves
        5. 📊 Check the Session Replay tab to review conversation history

        💡 **Tip:** Visit the Help & Documentation section in the sidebar to learn about key concepts like Fidelity, Basin of Attraction, and Active Mitigation.
        """)

    if turns:
        # Performance: Paginate messages for long conversations to limit DOM size
        MAX_VISIBLE_MESSAGES = 100
        if len(turns) > MAX_VISIBLE_MESSAGES:
            visible_turns = turns[-MAX_VISIBLE_MESSAGES:]
            st.caption(f"📊 Performance: Showing last {MAX_VISIBLE_MESSAGES} messages of {len(turns)} total (older messages hidden to improve performance)")
        else:
            visible_turns = turns

        for idx, turn in enumerate(visible_turns):
            # User message
            with st.chat_message("user"):
                st.write(turn['user_input'])

            # Assistant message
            with st.chat_message("assistant"):
                st.write(turn['assistant_response'])

                # Show metrics badge if drift detected
                metrics = turn.get('metrics', {})
                fidelity = metrics.get('telic_fidelity', 1.0)
                mode = get_mode()
                terms = get_terminology(mode)

                if fidelity < 0.76:  # Goldilocks: Aligned threshold
                    if mode == 'Basic':
                        st.warning(f"⚠️ {terms['drift_message']}")
                    else:
                        st.warning(f"⚠️ Drift detected (F={fidelity:.3f})")

                # Show active mitigation details if available
                metadata = turn.get('metadata', {})
                intervention_details = metadata.get('intervention_details')

                if intervention_details and intervention_details.get('intervention_applied'):
                    mode = get_mode()
                    terms = get_terminology(mode)

                    expander_title = f"🛡️ {terms['action']} Details" if mode == 'Basic' else "🛡️ Active Mitigation Details"

                    with st.expander(expander_title, expanded=False):
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            itype = intervention_details.get('type', 'unknown')
                            if mode == 'Basic':
                                st.metric("Type", itype.title())
                            else:
                                st.metric("Intervention Type", itype)

                        with col2:
                            salience = intervention_details.get('salience_after', 1.0)
                            if mode == 'Basic':
                                # Show as percentage
                                st.metric("Focus", f"{salience * 100:.0f}%")
                            else:
                                salience_emoji = "🟢" if salience >= 0.7 else "🟡"
                                st.metric("Salience", f"{salience_emoji} {salience:.3f}")

                        with col3:
                            if 'fidelity_improvement' in intervention_details:
                                improvement = intervention_details['fidelity_improvement']
                                if mode == 'Basic':
                                    st.metric(terms['result'], f"{improvement * 100:+.0f}%")
                                else:
                                    st.metric("ΔF", f"{improvement:+.3f}")

                        # Show intervention flow for regenerations
                        if itype in ["regeneration", "both"] and 'fidelity_original' in intervention_details:
                            st.caption("**Flow**: Original → Drift Detected → Regenerated → Governed")
                            f_orig = intervention_details.get('fidelity_original', 0)
                            f_gov = intervention_details.get('fidelity_governed', 0)
                            st.caption(f"F: {f_orig:.3f} → {f_gov:.3f}")

                            # Show side-by-side text comparison if original response available
                            original_response = metadata.get('initial_response')
                            if original_response:
                                st.divider()
                                st.caption("**📝 Text Comparison:**")

                                col_left, col_right = st.columns(2)

                                with col_left:
                                    if mode == 'Basic':
                                        st.markdown("**Before Recalibration**")
                                    else:
                                        st.markdown("**Original Response (Drifted)**")
                                    with st.container():
                                        st.markdown(
                                            f'<div style="background-color: #fff3cd; padding: 10px; border-radius: 5px; border-left: 3px solid #ffc107;">'
                                            f'{original_response[:300]}{"..." if len(original_response) > 300 else ""}'
                                            f'</div>',
                                            unsafe_allow_html=True
                                        )
                                        if mode == 'Basic':
                                            st.caption(f"Alignment: {f_orig * 100:.0f}% ⚠️")
                                        else:
                                            st.caption(f"Fidelity: {f_orig:.3f} ⚠️")

                                with col_right:
                                    if mode == 'Basic':
                                        st.markdown(f"**After {terms['action']}**")
                                    else:
                                        st.markdown("**Governed Response (Corrected)**")
                                    with st.container():
                                        st.markdown(
                                            f'<div style="background-color: #d4edda; padding: 10px; border-radius: 5px; border-left: 3px solid #28a745;">'
                                            f'{turn["assistant_response"][:300]}{"..." if len(turn["assistant_response"]) > 300 else ""}'
                                            f'</div>',
                                            unsafe_allow_html=True
                                        )
                                        if mode == 'Basic':
                                            st.caption(f"Alignment: {f_gov * 100:.0f}% ✅")
                                        else:
                                            st.caption(f"Fidelity: {f_gov:.3f} ✅")

                            # ALWAYS show Research mathematical analysis for interventions
                            # Rationale: If user expanded intervention details, show complete analysis
                            st.divider()
                            st.markdown("### 🔬 Mathematical Analysis")
                            st.caption("Live calculations showing how governance works")

                            # Prepare embeddings and metrics for observatory
                            embeddings_data = {
                                'response_embedding': turn.get('response_embedding'),
                                'user_embedding': turn.get('user_embedding'),
                                'attractor_center': turn.get('attractor_center')
                            }
                            metrics_dict = turn.get('metrics', {})

                            # Call observatory function to show full mathematical transparency
                            render_research_mode_observatory(
                                metrics=metrics_dict,
                                embeddings=embeddings_data,
                                intervention_data=intervention_details
                            )

                elif metadata.get('intervention_applied', False):
                    # Fallback for old-style intervention indicator
                    st.success("✅ Governance intervention applied")

                # Research Mode OR Research Lens: Show mathematical observatory
                # BUT skip if we already showed it inside intervention expander above
                show_research = (get_mode() == 'Research Mode') or st.session_state.get('research_lens_active', False)
                intervention_was_shown = intervention_details and intervention_details.get('intervention_applied', False)

                if show_research and not intervention_was_shown:
                    st.divider()
                    # Add header for Research Lens overlay (not shown for Research Mode itself)
                    if get_mode() != 'Research Mode' and st.session_state.get('research_lens_active', False):
                        st.markdown("""
                        <div style="background-color: #f0f8ff; padding: 10px; border-radius: 5px; border-left: 4px solid #27ae60;">
                        <h4 style="margin: 0;">🔬 Research Lens - Live Mathematical Observatory</h4>
                        <p style="margin: 5px 0 0 0; font-size: 0.9em;">Overlay active - showing calculations in real-time</p>
                        </div>
                        """, unsafe_allow_html=True)
                        st.markdown("<br>", unsafe_allow_html=True)

                    # Prepare embeddings dict
                    embeddings_data = {
                        'response_embedding': turn.get('response_embedding'),
                        'user_embedding': turn.get('user_embedding'),
                        'attractor_center': turn.get('attractor_center')
                    }
                    # Call observatory function
                    render_research_mode_observatory(
                        metrics=metrics,
                        embeddings=embeddings_data,
                        intervention_data=intervention_details
                    )
    else:
        st.info("👋 Start a conversation below to see TELOSCOPE in action!")

    # Render intervention timeline if turns exist
    if turns:
        st.divider()
        render_intervention_timeline(turns)

    # Render simulation UI if a drift point was selected
    render_simulation_ui()

    st.divider()

    # Display triggers as clickable badges
    triggers = st.session_state.web_session.get_all_triggers()
    if triggers:
        st.subheader("🔬 Counterfactual Experiments")
        st.caption(f"{len(triggers)} trigger(s) fired - click to view evidence")

        # Create columns for trigger badges
        num_cols = min(len(triggers), 4)
        cols = st.columns(num_cols)

        for i, trigger in enumerate(triggers):
            with cols[i % num_cols]:
                trigger_fidelity = trigger.get('fidelity', 0.0)
                trigger_color = "🔴" if trigger_fidelity < 0.5 else "🟡"

                if st.button(
                    f"{trigger_color} Turn {trigger['turn_number']}",
                    key=f"trigger_live_{i}",
                    help=f"F={trigger_fidelity:.3f} - {trigger['reason']}",
                    use_container_width=True
                ):
                    st.session_state.selected_trigger = trigger['trigger_id']
                    st.session_state.active_tab = 2  # Switch to TELOSCOPE tab
                    st.rerun()

        st.divider()

    # Chat input - ALWAYS at the end, outside all processing
    user_input = st.chat_input("Ask about AI governance, or try going off-topic to trigger drift...")

    # Process user input if provided
    if user_input:
        # Build messages list from conversation history
        messages = []
        for turn in turns:
            messages.append({"role": "user", "content": turn['user_input']})
            messages.append({"role": "assistant", "content": turn['assistant_response']})
        messages.append({"role": "user", "content": user_input})

        # Generate response through LiveInterceptor
        try:
            with st.spinner("🤖 Generating governed response..."):
                response = st.session_state.interceptor.generate(messages)
            # Rerun OUTSIDE the spinner to ensure clean UI state
            st.rerun()
        except Exception as e:
            st.error("🔌 Unable to generate response")
            st.info("💡 **What to try:**\n- Check your network connection\n- Try again in a moment\n- If this persists, refresh the page")
            logging.error(f"Response generation failed: {e}")  # Log technical details


# ============================================================================
# Tab 2: Session Replay
# ============================================================================

def render_session_replay():
    """Render session replay with timeline scrubber."""
    st.title("⏮️ Session Replay")
    st.caption("Navigate conversation history with timeline controls")
    st.info("💡 **Tip:** Use the timeline slider or navigation buttons to jump between conversation turns. Metrics show governance health at each point. Click trigger buttons to explore counterfactual experiments.")
    st.markdown("---")  # Section separator

    turns = st.session_state.current_session.get('turns', [])

    # CRITICAL: Need at least 2 turns for slider to work (min < max)
    if len(turns) < 2:
        if len(turns) == 0:
            st.info("📝 No conversation history yet. Start a conversation in the Live Session tab.")
        else:
            st.info("📝 Only 1 turn recorded. Add more conversation turns to use the timeline scrubber.")
        return

    # Initialize replay turn and navigation state
    if 'replay_turn' not in st.session_state:
        st.session_state.replay_turn = 0
    if 'last_replay_turn' not in st.session_state:
        st.session_state.last_replay_turn = 0

    # Ensure replay_turn is in bounds
    st.session_state.replay_turn = min(st.session_state.replay_turn, len(turns) - 1)

    # Section header: Timeline Controls
    st.markdown("### 🎬 Timeline Controls")

    # Timeline controls
    col1, col2, col3, col4 = st.columns([1, 1, 4, 1])

    with col1:
        if st.button("⏮️ First",
                    use_container_width=True,
                    help="Jump to first turn in conversation history"):
            st.session_state.replay_turn = 0
            st.rerun()

    with col2:
        if st.button("◀️ Prev",
                    use_container_width=True,
                    help="Navigate to previous turn in conversation history"):
            st.session_state.replay_turn = max(0, st.session_state.replay_turn - 1)
            st.rerun()

    with col3:
        # Timeline slider
        turn_num = st.slider(
            "Turn",
            min_value=0,
            max_value=len(turns) - 1,
            value=st.session_state.replay_turn,
            key="replay_slider",
            help="Scrub through conversation timeline - drag slider or use arrow keys to navigate between turns"
        )
        st.session_state.replay_turn = turn_num

    with col4:
        if st.button("Next ▶️",
                    use_container_width=True,
                    help="Navigate to next turn in conversation history"):
            st.session_state.replay_turn = min(len(turns) - 1, st.session_state.replay_turn + 1)
            st.rerun()

    # Smooth scroll feedback: Show turn change notification
    if st.session_state.replay_turn != st.session_state.last_replay_turn:
        direction = "forward" if st.session_state.replay_turn > st.session_state.last_replay_turn else "back"
        st.info(f"🎯 Jumped {direction} to Turn {st.session_state.replay_turn + 1} of {len(turns)}")
        st.session_state.last_replay_turn = st.session_state.replay_turn

    st.markdown("---")  # Visual separator

    # Section header: Conversation View
    st.markdown("### 💬 Conversation View")

    # Display selected turn
    selected_turn = turns[turn_num]

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(f"**Turn {turn_num + 1} of {len(turns)}**")
        st.markdown("")  # Small spacer

        # User message
        with st.chat_message("user"):
            st.write(selected_turn['user_input'])

        # Assistant message
        with st.chat_message("assistant"):
            st.write(selected_turn['assistant_response'])

            # Show intervention status
            if selected_turn.get('metadata', {}).get('intervention_applied', False):
                st.success("✅ Governance intervention applied")

    with col2:
        st.markdown("### 📊 Metrics")
        st.caption("Track governance health: 🟢 Good | 🟡 Watch | 🔴 Attention needed")
        st.markdown("")  # Small spacer

        metrics = selected_turn.get('metrics', {})

        # Fidelity
        fidelity = metrics.get('telic_fidelity', 1.0)
        fidelity_color = "🟢" if fidelity >= 0.76 else ("🟡" if fidelity >= 0.67 else "🔴")
        st.metric(
            f"{fidelity_color} Fidelity",
            f"{fidelity:.3f}",
            help="Alignment with governance purpose (0-1 scale). Higher is better. 0.76+ aligned, 0.67-0.76 drift detected, <0.67 significant drift"
        )

        # Distance
        distance = metrics.get('drift_distance', 0.0)
        st.metric(
            "Drift Distance",
            f"{distance:.3f}",
            help="Numerical measure of how far response drifted from purpose. Lower is better. Values >2.0 typically indicate basin exit"
        )

        # Error Signal
        error = metrics.get('error_signal', 0.0)
        st.metric(
            "Error Signal",
            f"{error:.3f}",
            help="Deviation from governance target (0-1 scale). Lower is better. High values (>0.5) indicate drift"
        )

        # Basin Status
        basin = metrics.get('primacy_basin_membership', True)
        basin_icon = "✅" if basin else "❌"
        basin_text = f"Inside {basin_icon}" if basin else f"Outside {basin_icon}"
        st.write(f"**Basin Status:** {basin_text}")
        st.caption("Basin = safe zone where governance maintains fidelity")

    st.markdown("---")  # Visual separator

    # Show trigger markers on timeline
    triggers = st.session_state.web_session.get_all_triggers()
    if triggers:
        st.markdown("### 🔬 Counterfactual Triggers on Timeline")
        st.caption("These turns had drift detected. Click to explore 'what if' scenarios showing governance impact.")
        st.markdown("")  # Small spacer

        # Create columns for triggers
        num_cols = min(len(triggers), 4)
        trigger_cols = st.columns(num_cols)

        for i, trigger in enumerate(triggers):
            with trigger_cols[i % num_cols]:
                trigger_turn = trigger['turn_number']
                trigger_fidelity = trigger.get('fidelity', 0.0)

                if st.button(
                    f"⚠️ Turn {trigger_turn}",
                    key=f"replay_trigger_{i}",
                    help=f"F={trigger_fidelity:.3f} - {trigger['reason']}",
                    use_container_width=True
                ):
                    st.session_state.selected_trigger = trigger['trigger_id']
                    st.session_state.active_tab = 2  # Jump to TELOSCOPE tab
                    st.rerun()


# ============================================================================
# Tab 3: TELOSCOPE (Counterfactual Evidence Viewer)
# ============================================================================

def render_teloscope_view():
    """Render counterfactual branch comparison and evidence."""
    st.title("🔭 TELOSCOPE: Counterfactual Evidence")
    st.caption("Observable proof of AI governance efficacy through parallel universe comparison")

    triggers = st.session_state.web_session.get_all_triggers()

    if not triggers:
        st.info("""
        ### Welcome to TELOSCOPE

        **No counterfactual experiments yet.**

        Counterfactuals are automatically triggered when drift is detected (fidelity < 0.76).

        **How TELOSCOPE works:**
        1. Continue conversations in the Live Session tab
        2. When fidelity drops below 0.76 (Goldilocks: Aligned threshold), a trigger fires
        3. Two 5-turn branches are generated:
           - **🔴 Baseline**: What happens WITHOUT intervention
           - **🟢 TELOS**: What happens WITH intervention
        4. **ΔF** (improvement metric) is calculated automatically
        5. Statistical significance is tested

        **Start a conversation and try going off-topic to trigger drift!**

        Example off-topic questions:
        - "What's your favorite movie?"
        - "Tell me a joke"
        - "What's the weather like?"
        """)
        return

    # Trigger selector
    trigger_options = {
        t['trigger_id']: f"Turn {t['turn_number']}: {t['reason']} (F={t.get('fidelity', 0.0):.3f})"
        for t in triggers
    }

    # Use selected trigger if available
    default_trigger = st.session_state.get('selected_trigger', list(trigger_options.keys())[0])
    if default_trigger not in trigger_options:
        default_trigger = list(trigger_options.keys())[0]

    selected_trigger_id = st.selectbox(
        "📌 Select Counterfactual Trigger",
        options=list(trigger_options.keys()),
        format_func=lambda x: trigger_options[x],
        index=list(trigger_options.keys()).index(default_trigger) if default_trigger in trigger_options else 0,
        help="Choose which trigger point to analyze"
    )

    # Get branch data
    branch_data = st.session_state.web_session.get_branch(selected_trigger_id)

    if not branch_data:
        st.warning("⏳ Generating counterfactual branches... (this may take 30-60 seconds)")
        st.info("💡 Branches are generated in the background. You can continue using other tabs.")

        if st.button("🔄 Refresh", type="primary"):
            st.rerun()
        return

    if branch_data.get('status') == 'generating':
        st.info("⏳ Branch generation in progress...")

        with st.spinner("Generating baseline and TELOS branches..."):
            if st.button("🔄 Refresh", type="primary"):
                st.rerun()
        return

    if branch_data.get('status') == 'failed':
        st.error(f"❌ Branch generation failed: {branch_data.get('error', 'Unknown error')}")
        st.info("Try resetting the session and starting a new conversation.")
        return

    # Get baseline and TELOS branches
    baseline = branch_data.get('baseline', {})
    telos = branch_data.get('telos', {})

    if not baseline or not telos:
        st.warning("⏳ Branch data incomplete. Please wait...")
        if st.button("🔄 Refresh", type='secondary'):
            st.rerun()
        return

    # Compare branches
    comparison = st.session_state.comparator.compare_branches(baseline, telos)

    st.divider()

    # Display ΔF and key metrics
    st.subheader("🎯 Governance Efficacy Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        delta_f = comparison['delta']['delta_f']
        delta_color = "normal" if delta_f > 0 else "inverse"
        st.metric(
            "ΔF (Improvement)",
            f"{delta_f:+.3f}",
            delta=f"{abs(delta_f):.3f} {'improvement' if delta_f > 0 else 'degradation'}",
            delta_color=delta_color,
            help="Fidelity improvement from governance (TELOS - Baseline)"
        )

    with col2:
        baseline_final = comparison['baseline']['final_fidelity']
        st.metric(
            "🔴 Baseline Final",
            f"{baseline_final:.3f}",
            help="Final fidelity without intervention"
        )

    with col3:
        telos_final = comparison['telos']['final_fidelity']
        st.metric(
            "🟢 TELOS Final",
            f"{telos_final:.3f}",
            help="Final fidelity with intervention"
        )

    with col4:
        avg_improvement = comparison['delta']['avg_improvement']
        st.metric(
            "Avg Improvement",
            f"{avg_improvement:+.3f}",
            help="Average fidelity improvement across all turns"
        )

    st.divider()

    # Side-by-side branch comparison
    st.subheader("🔬 Branch Comparison: Baseline vs TELOS")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🔴 Baseline (No Intervention)")
        st.caption("Shows what happens when drift continues unchecked")

        baseline_turns = baseline.get('turns', [])
        for i, turn in enumerate(baseline_turns, 1):
            fidelity = turn.get('metrics', {}).get('telic_fidelity', 0.0)
            fidelity_emoji = "🟢" if fidelity >= 0.76 else ("🟡" if fidelity >= 0.67 else "🔴")  # Goldilocks zones

            with st.expander(f"{fidelity_emoji} Turn {i} - F={fidelity:.3f}"):
                st.write(f"**User:** {turn.get('user_input', 'N/A')}")
                st.write(f"**Assistant:** {turn.get('assistant_response', 'N/A')}")

                # Show metrics
                metrics = turn.get('metrics', {})
                st.caption(f"Distance: {metrics.get('drift_distance', 0.0):.3f} | "
                          f"Basin: {'✅' if metrics.get('primacy_basin_membership', False) else '❌'}")

    with col2:
        st.markdown("### 🟢 TELOS (With Intervention)")
        st.caption("Shows how governance corrects drift and maintains alignment")

        telos_turns = telos.get('turns', [])
        for i, turn in enumerate(telos_turns, 1):
            fidelity = turn.get('metrics', {}).get('telic_fidelity', 0.0)
            fidelity_emoji = "🟢" if fidelity >= 0.76 else ("🟡" if fidelity >= 0.67 else "🔴")  # Goldilocks zones

            with st.expander(f"{fidelity_emoji} Turn {i} - F={fidelity:.3f}"):
                st.write(f"**User:** {turn.get('user_input', 'N/A')}")
                st.write(f"**Assistant:** {turn.get('assistant_response', 'N/A')}")

                # Show metrics
                metrics = turn.get('metrics', {})
                st.caption(f"Distance: {metrics.get('drift_distance', 0.0):.3f} | "
                          f"Basin: {'✅' if metrics.get('primacy_basin_membership', False) else '❌'}")

    st.divider()

    # Fidelity divergence chart
    st.subheader("📈 Fidelity Divergence Over Time")
    st.caption("Visual proof of governance efficacy")

    if HAS_PLOTLY:
        fig = st.session_state.comparator.generate_divergence_chart(comparison)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    else:
        # Fallback: display data as table
        baseline_fidelities = comparison['baseline']['fidelity_trend']
        telos_fidelities = comparison['telos']['fidelity_trend']

        df = pd.DataFrame({
            'Turn': list(range(1, len(baseline_fidelities) + 1)),
            'Baseline': baseline_fidelities,
            'TELOS': telos_fidelities,
            'Improvement': [t - b for t, b in zip(telos_fidelities, baseline_fidelities)]
        })
        st.dataframe(df, use_container_width=True)

    st.divider()

    # Metrics comparison table
    st.subheader("📊 Metrics Comparison Table")
    df = st.session_state.comparator.generate_metrics_table(comparison)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Statistical analysis
    if 'statistics' in comparison:
        with st.expander("📊 Statistical Significance Analysis", expanded=True):
            stats = comparison['statistics']

            sig_emoji = "✅" if stats['significant'] else "⚠️"
            sig_text = "Statistically significant" if stats['significant'] else "Not statistically significant"

            st.markdown(f"### {sig_emoji} {sig_text}")
            st.caption(f"p-value = {stats['p_value']:.4f} (threshold: 0.05)")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Effect Size (Cohen's d)",
                    f"{stats['effect_size_cohens_d']:.3f}",
                    help="Magnitude of improvement (>0.8 = large effect)"
                )

            with col2:
                st.metric(
                    "Mean Improvement",
                    f"{stats['mean_difference']:+.3f}",
                    help="Average fidelity improvement across turns"
                )

            with col3:
                ci = stats['confidence_interval_95']
                st.metric(
                    "95% CI Range",
                    f"[{ci[0]:.3f}, {ci[1]:.3f}]",
                    help="95% confidence interval for improvement"
                )

            # Interpretation
            st.markdown("---")
            st.markdown("**Interpretation:**")

            if stats['significant'] and stats['effect_size_cohens_d'] > 0.8:
                st.success("""
                ✅ **Strong Evidence**: TELOS governance significantly improves fidelity with a large effect size.
                This provides robust evidence that the governance system is working effectively.
                """)
            elif stats['significant']:
                st.info("""
                ℹ️ **Moderate Evidence**: TELOS governance shows statistically significant improvement.
                The effect size suggests measurable benefit from governance intervention.
                """)
            else:
                st.warning("""
                ⚠️ **Weak Evidence**: While there may be some improvement, it is not statistically significant.
                Consider adjusting governance parameters or collecting more data.
                """)

    st.divider()

    # Export functionality
    st.subheader("💾 Export Evidence")

    col1, col2, col3 = st.columns([2, 1, 1])

    with col2:
        evidence = st.session_state.branch_manager.export_branch_evidence(selected_trigger_id)
        if evidence:
            evidence_json = json.dumps(evidence, indent=2)
            st.download_button(
                "📥 Export JSON",
                data=evidence_json,
                file_name=f"teloscope_evidence_{selected_trigger_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                type="primary",
                use_container_width=True,
                help="Download complete evidence package for compliance"
            )

    with col3:
        if st.button("🔄 Regenerate", type="secondary", use_container_width=True,
                    help="Regenerate this counterfactual experiment"):
            st.warning("⚠️ This feature is not yet implemented")


# ============================================================================
# Tab 4: Analytics Dashboard
# ============================================================================

def render_analytics_dashboard():
    """Render aggregate analytics and session statistics."""
    st.title("📊 Analytics Dashboard")
    st.caption("Aggregate statistics and governance efficacy analysis")

    # Session statistics
    stats = st.session_state.web_session.get_session_stats()

    st.subheader("📈 Session Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Turns",
            stats.get('total_turns', 0),
            help="Number of conversation turns"
        )

    with col2:
        st.metric(
            "Triggers Fired",
            stats.get('total_triggers', 0),
            help="Number of counterfactual experiments triggered"
        )

    with col3:
        avg_f = stats.get('avg_fidelity', 1.0)
        avg_color = "🟢" if avg_f >= 0.76 else ("🟡" if avg_f >= 0.67 else "🔴")  # Goldilocks zones
        st.metric(
            f"{avg_color} Avg Fidelity",
            f"{avg_f:.3f}",
            help="Average fidelity across all turns"
        )

    with col4:
        trigger_rate = stats.get('trigger_rate', 0.0)
        st.metric(
            "Trigger Rate",
            f"{trigger_rate * 100:.1f}%",
            help="Percentage of turns that triggered counterfactuals"
        )

    st.divider()

    # Historical fidelity chart
    st.subheader("📈 Fidelity Over Time")
    st.caption("Track alignment throughout the conversation")

    turns = st.session_state.current_session.get('turns', [])

    if turns:
        fidelities = [t['metrics'].get('telic_fidelity', 1.0) for t in turns]

        if HAS_PLOTLY:
            fig = go.Figure()

            # Fidelity line
            fig.add_trace(go.Scatter(
                x=list(range(1, len(fidelities) + 1)),
                y=fidelities,
                mode='lines+markers',
                name='Fidelity',
                line=dict(color='#339af0', width=2),
                marker=dict(size=8)
            ))

            # Goldilocks threshold lines
            fig.add_hline(
                y=0.76,
                line_dash="dash",
                line_color="green",
                annotation_text="Aligned Threshold (F=0.76)",
                annotation_position="right"
            )

            fig.add_hline(
                y=0.5,
                line_dash="dash",
                line_color="red",
                annotation_text="Critical Threshold (F=0.5)",
                annotation_position="right"
            )

            fig.update_layout(
                xaxis_title="Turn Number",
                yaxis_title="Telic Fidelity",
                height=350,
                template='plotly_white',
                hovermode='x unified'
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            # Fallback: line chart
            df = pd.DataFrame({
                'Turn': list(range(1, len(fidelities) + 1)),
                'Fidelity': fidelities
            })
            st.line_chart(df.set_index('Turn'))
    else:
        st.info("No conversation data yet. Start a conversation in the Live Session tab.")

    st.divider()

    # Counterfactual efficacy summary
    st.subheader("🔬 Counterfactual Efficacy Summary")
    st.caption("Evidence of governance effectiveness across all experiments")

    triggers = st.session_state.web_session.get_all_triggers()
    branches = st.session_state.current_session.get('branches', {})

    if branches:
        efficacy_data = []

        for trigger_id, branch_data in branches.items():
            if branch_data.get('status') == 'completed':
                comparison = branch_data.get('comparison', {})
                trigger_info = next((t for t in triggers if t['trigger_id'] == trigger_id), {})

                delta_f = comparison.get('delta_f', 0.0)
                avg_improvement = comparison.get('avg_improvement', 0.0)

                # Get statistical significance if available
                stats_data = comparison.get('statistics', {})
                significance = '✅' if stats_data.get('significant', False) else '❌'

                efficacy_data.append({
                    'Trigger Turn': trigger_info.get('turn_number', 'N/A'),
                    'Reason': trigger_info.get('reason', 'N/A')[:40] + '...',
                    'ΔF': f"{delta_f:+.3f}",
                    'Avg Improvement': f"{avg_improvement:+.3f}",
                    'Significant': significance,
                    'p-value': f"{stats_data.get('p_value', 0.0):.4f}" if stats_data else 'N/A'
                })

        if efficacy_data:
            df = pd.DataFrame(efficacy_data)
            st.dataframe(df, use_container_width=True, hide_index=True)

            st.divider()

            # Aggregate statistics
            st.subheader("📊 Aggregate Governance Metrics")

            delta_fs = [float(d['ΔF']) for d in efficacy_data]
            avg_delta_f = sum(delta_fs) / len(delta_fs) if delta_fs else 0.0

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Average ΔF",
                    f"{avg_delta_f:+.3f}",
                    help="Average fidelity improvement across all experiments"
                )

            with col2:
                positive_count = sum(1 for d in delta_fs if d > 0)
                success_rate = (positive_count / len(delta_fs) * 100) if delta_fs else 0
                st.metric(
                    "Success Rate",
                    f"{success_rate:.1f}%",
                    help="Percentage of experiments showing improvement"
                )

            with col3:
                significant_count = sum(1 for d in efficacy_data if d['Significant'] == '✅')
                sig_rate = (significant_count / len(efficacy_data) * 100) if efficacy_data else 0
                st.metric(
                    "Significance Rate",
                    f"{sig_rate:.1f}%",
                    help="Percentage of statistically significant results"
                )

            # Overall assessment
            st.divider()
            st.subheader("🎯 Overall Assessment")

            if avg_delta_f > 0.1 and success_rate > 80:
                st.success("""
                ✅ **Excellent Governance Performance**

                TELOS governance is consistently improving fidelity across experiments.
                The evidence strongly supports governance efficacy.
                """)
            elif avg_delta_f > 0 and success_rate > 50:
                st.info("""
                ℹ️ **Good Governance Performance**

                TELOS governance shows positive impact in most cases.
                Consider fine-tuning parameters for even better results.
                """)
            else:
                st.warning("""
                ⚠️ **Governance Needs Tuning**

                Results show inconsistent or negative impact.
                Review governance configuration and attractor parameters.
                """)
        else:
            st.info("No completed counterfactual experiments yet.")
    else:
        st.info("""
        No counterfactual experiments yet.

        Triggers fire automatically when drift is detected (F < 0.76).
        Start a conversation and try going off-topic to generate evidence!
        """)

    st.divider()

    # Phase 9: Cross-Session Analytics
    st.subheader("📊 Cross-Session Trends")
    st.caption("Analysis of governance patterns across multiple sessions")

    # Load session files
    sessions_data = load_session_files(max_sessions=10)

    if sessions_data:
        # Aggregate statistics
        st.markdown("#### 📈 Aggregate Statistics")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Sessions",
                len(sessions_data),
                help="Number of recent sessions analyzed"
            )

        with col2:
            avg_fidelity_all = compute_avg_fidelity_across_sessions(sessions_data)
            fidelity_color = "🟢" if avg_fidelity_all >= 0.76 else ("🟡" if avg_fidelity_all >= 0.67 else "🔴")  # Goldilocks zones
            st.metric(
                f"{fidelity_color} Avg Fidelity",
                f"{avg_fidelity_all:.3f}",
                help="Average fidelity across all sessions"
            )

        with col3:
            total_interventions = count_interventions_across_sessions(sessions_data)
            st.metric(
                "Total Interventions",
                total_interventions,
                help="Total interventions applied across all sessions"
            )

        with col4:
            avg_turns = compute_avg_turns_per_session(sessions_data)
            st.metric(
                "Avg Turns/Session",
                f"{avg_turns:.1f}",
                help="Average conversation length"
            )

        st.divider()

        # Fidelity trend across sessions
        st.markdown("#### 📈 Fidelity Trends Across Sessions")
        st.caption("Track governance effectiveness over time")

        session_fidelities = extract_session_fidelity_trends(sessions_data)
        session_numbers = list(range(1, len(session_fidelities) + 1))

        if HAS_PLOTLY:
            fig = go.Figure()

            # Session fidelity line
            fig.add_trace(go.Scatter(
                x=session_numbers,
                y=session_fidelities,
                mode='lines+markers',
                name='Session Avg Fidelity',
                line=dict(color='#339af0', width=2),
                marker=dict(size=10)
            ))

            # Goldilocks threshold lines
            fig.add_hline(
                y=0.76,
                line_dash="dash",
                line_color="green",
                annotation_text="Target (F=0.76)",
                annotation_position="right"
            )

            fig.update_layout(
                xaxis_title="Session Number (most recent 10)",
                yaxis_title="Average Fidelity",
                height=350,
                template='plotly_white',
                hovermode='x unified',
                yaxis_range=[0, 1]
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            # Fallback: simple line chart
            df = pd.DataFrame({
                'Session': session_numbers,
                'Avg Fidelity': session_fidelities
            })
            st.line_chart(df.set_index('Session'))

        # Overall cross-session assessment (Goldilocks)
        if avg_fidelity_all > 0.76:
            st.success("""
            ✅ **Excellent Cross-Session Performance**

            Governance maintains high fidelity across multiple sessions.
            System demonstrates consistent alignment with purpose.
            """)
        elif avg_fidelity_all > 0.6:
            st.info("""
            ℹ️ **Good Cross-Session Performance**

            Governance shows positive impact across sessions.
            Monitor for potential drift patterns.
            """)
        else:
            st.warning("""
            ⚠️ **Cross-Session Performance Needs Attention**

            Average fidelity below target across sessions.
            Review governance configuration and attractor parameters.
            """)
    else:
        st.info("""
        No exported session data available yet.

        Session data is automatically exported after conversations.
        Complete at least one conversation to see cross-session analytics!
        """)

    st.divider()

    # Phase 9: Session-to-Session Comparison
    st.subheader("🔍 Session Comparison")
    st.caption("Compare metrics and performance between different sessions")

    if len(sessions_data) >= 2:
        # Generate session labels for selection
        session_labels = generate_session_labels(sessions_data)

        # Session selectors
        col_selector1, col_selector2 = st.columns(2)

        with col_selector1:
            session_a_idx = st.selectbox(
                "Session A",
                range(len(sessions_data)),
                format_func=lambda i: session_labels[i],
                key='session_compare_a'
            )

        with col_selector2:
            session_b_idx = st.selectbox(
                "Session B",
                range(len(sessions_data)),
                format_func=lambda i: session_labels[i],
                index=min(1, len(sessions_data) - 1),
                key='session_compare_b'
            )

        if session_a_idx == session_b_idx:
            st.warning("⚠️ Please select two different sessions to compare.")
        else:
            # Extract metrics for both sessions
            session_a = sessions_data[session_a_idx]
            session_b = sessions_data[session_b_idx]

            metrics_a = extract_session_metrics(session_a)
            metrics_b = extract_session_metrics(session_b)

            # Side-by-side metrics comparison
            st.markdown("#### 📊 Metrics Comparison")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                delta_fidelity = metrics_b['avg_fidelity'] - metrics_a['avg_fidelity']
                st.metric(
                    "Avg Fidelity",
                    f"{metrics_b['avg_fidelity']:.3f}",
                    delta=f"{delta_fidelity:+.3f}",
                    help="Average fidelity (Session B vs Session A)"
                )

            with col2:
                delta_interventions = metrics_b['interventions'] - metrics_a['interventions']
                st.metric(
                    "Interventions",
                    metrics_b['interventions'],
                    delta=f"{delta_interventions:+d}",
                    help="Total interventions (Session B vs Session A)"
                )

            with col3:
                delta_turns = metrics_b['turns'] - metrics_a['turns']
                st.metric(
                    "Turns",
                    metrics_b['turns'],
                    delta=f"{delta_turns:+d}",
                    help="Conversation length (Session B vs Session A)"
                )

            with col4:
                delta_violations = metrics_b['basin_violations'] - metrics_a['basin_violations']
                st.metric(
                    "Basin Violations",
                    metrics_b['basin_violations'],
                    delta=f"{delta_violations:+d}",
                    delta_color="inverse",
                    help="Drift events (Session B vs Session A)"
                )

            st.divider()

            # Turn-by-turn comparison visualization
            st.markdown("#### 📈 Turn-by-Turn Fidelity Comparison")
            st.caption("Overlay fidelity trends for both sessions")

            fidelities_a = extract_turn_fidelities(session_a)
            fidelities_b = extract_turn_fidelities(session_b)

            if HAS_PLOTLY:
                fig = go.Figure()

                # Session A line
                fig.add_trace(go.Scatter(
                    x=list(range(1, len(fidelities_a) + 1)),
                    y=fidelities_a,
                    mode='lines+markers',
                    name=f'Session A ({len(fidelities_a)} turns)',
                    line=dict(color='#4a90e2', width=2),
                    marker=dict(size=6)
                ))

                # Session B line
                fig.add_trace(go.Scatter(
                    x=list(range(1, len(fidelities_b) + 1)),
                    y=fidelities_b,
                    mode='lines+markers',
                    name=f'Session B ({len(fidelities_b)} turns)',
                    line=dict(color='#f39c12', width=2),
                    marker=dict(size=6)
                ))

                # Goldilocks target threshold
                fig.add_hline(
                    y=0.76,
                    line_dash="dash",
                    line_color="green",
                    annotation_text="Target (F=0.76)",
                    annotation_position="right"
                )

                fig.update_layout(
                    xaxis_title="Turn Number",
                    yaxis_title="Telic Fidelity",
                    height=400,
                    template='plotly_white',
                    hovermode='x unified',
                    yaxis_range=[0, 1],
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )

                st.plotly_chart(fig, use_container_width=True)
            else:
                # Fallback: simple display
                df_comparison = pd.DataFrame({
                    'Turn': list(range(1, max(len(fidelities_a), len(fidelities_b)) + 1)),
                    'Session A': fidelities_a + [None] * (max(len(fidelities_a), len(fidelities_b)) - len(fidelities_a)),
                    'Session B': fidelities_b + [None] * (max(len(fidelities_a), len(fidelities_b)) - len(fidelities_b))
                })
                st.line_chart(df_comparison.set_index('Turn'))

            # Comparative summary
            st.markdown("#### 📝 Comparative Summary")

            if metrics_b['avg_fidelity'] > metrics_a['avg_fidelity']:
                improvement_pct = ((metrics_b['avg_fidelity'] - metrics_a['avg_fidelity']) / metrics_a['avg_fidelity'] * 100)
                st.success(f"""
                ✅ **Session B shows improvement**

                Average fidelity improved by {improvement_pct:.1f}% compared to Session A.
                Session B demonstrates better alignment with purpose.
                """)
            elif metrics_b['avg_fidelity'] < metrics_a['avg_fidelity']:
                decline_pct = ((metrics_a['avg_fidelity'] - metrics_b['avg_fidelity']) / metrics_a['avg_fidelity'] * 100)
                st.warning(f"""
                ⚠️ **Session B shows decline**

                Average fidelity decreased by {decline_pct:.1f}% compared to Session A.
                Review Session B for potential issues or configuration changes.
                """)
            else:
                st.info("""
                ℹ️ **Sessions show similar performance**

                Both sessions demonstrate comparable fidelity levels.
                Performance appears consistent across sessions.
                """)

    elif len(sessions_data) == 1:
        st.info("""
        Need at least 2 sessions for comparison.

        Complete another conversation to enable session-to-session comparison!
        """)
    else:
        st.info("""
        No session data available yet for comparison.

        Complete at least 2 conversations to see session-to-session comparisons!
        """)

    st.divider()

    # Phase 9 Task 3: Automated Pattern Detection
    st.subheader("🔍 Automated Pattern Detection")
    st.caption("AI-powered analysis of governance behavior patterns across all sessions")

    if len(sessions_data) > 0:
        # Run pattern detection
        with st.spinner('🔍 Analyzing patterns...'):
            intervention_patterns = detect_intervention_patterns(sessions_data)
            drift_triggers = detect_drift_triggers(sessions_data)
            effectiveness = analyze_governance_effectiveness(sessions_data)
            anomalies = detect_anomalous_sessions(sessions_data)

        # Create tabs for different pattern types
        pattern_tabs = st.tabs([
            "🎯 Intervention Patterns",
            "⚠️ Drift Triggers",
            "📈 Governance Effectiveness",
            "🚨 Anomaly Detection"
        ])

        # Tab 1: Intervention Patterns
        with pattern_tabs[0]:
            st.markdown("#### Intervention Behavior Patterns")
            st.caption("Identifies sessions with unusually high or low intervention rates")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Avg Intervention Rate",
                    f"{intervention_patterns['avg_intervention_rate']:.1%}",
                    help="Average percentage of turns requiring governance intervention"
                )

            with col2:
                st.metric(
                    "Total Interventions",
                    intervention_patterns['total_interventions'],
                    help="Total number of interventions across all sessions"
                )

            with col3:
                st.metric(
                    "Total Turns",
                    intervention_patterns['total_turns'],
                    help="Total number of conversation turns analyzed"
                )

            st.divider()

            # High intervention sessions
            high_sessions = intervention_patterns['high_intervention_sessions']
            if high_sessions:
                st.markdown("##### 🔴 High Intervention Sessions (>50%)")
                st.caption("Sessions requiring frequent governance corrections")

                for session in high_sessions[:5]:  # Show top 5
                    with st.expander(f"**{session['session_id']}** - {session['rate']:.1%} intervention rate"):
                        st.write(f"**Interventions:** {session['interventions']} out of {session['turns']} turns")
                        st.progress(session['rate'])
                        st.caption("""
                        **Analysis:** High intervention rate may indicate:
                        - Complex/ambiguous user queries
                        - Attractor misalignment with conversation context
                        - Need for attractor refinement
                        """)
            else:
                st.info("✓ No sessions with unusually high intervention rates detected")

            st.divider()

            # Low intervention sessions
            low_sessions = intervention_patterns['low_intervention_sessions']
            if low_sessions:
                st.markdown("##### 🟢 Low Intervention Sessions (<10%)")
                st.caption("Sessions with minimal governance corrections")

                for session in low_sessions[:5]:  # Show top 5
                    with st.expander(f"**{session['session_id']}** - {session['rate']:.1%} intervention rate"):
                        st.write(f"**Interventions:** {session['interventions']} out of {session['turns']} turns")
                        st.progress(session['rate'])
                        st.caption("""
                        **Analysis:** Low intervention rate may indicate:
                        - Strong natural alignment with purpose
                        - Well-calibrated attractor configuration
                        - Straightforward queries within attractor scope
                        """)
            else:
                st.info("✓ All sessions show normal intervention patterns")

        # Tab 2: Drift Triggers
        with pattern_tabs[1]:
            st.markdown("#### Common Drift Triggers")
            st.caption("Identifies turns where fidelity drops below target threshold")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Total Drift Events",
                    drift_triggers['total_drift_events'],
                    help="Number of turns with fidelity below 0.76"  # Goldilocks: Aligned threshold
                )

            with col2:
                st.metric(
                    "Overall Drift Rate",
                    f"{drift_triggers['drift_rate']:.1%}",
                    help="Percentage of turns experiencing drift"
                )

            with col3:
                st.metric(
                    "Avg Drifts/Session",
                    f"{drift_triggers['avg_drifts_per_session']:.1f}",
                    help="Average number of drift events per session"
                )

            st.divider()

            sessions_with_drift = drift_triggers['sessions_with_drift']
            if sessions_with_drift:
                st.markdown("##### ⚠️ Sessions with Drift Events")

                for session_drift in sessions_with_drift[:5]:  # Show top 5
                    with st.expander(f"**{session_drift['session_id']}** - {session_drift['drift_count']} drift events ({session_drift['drift_rate']:.1%})"):
                        st.write(f"**Drift turns:** {session_drift['drift_count']} out of total turns")

                        if session_drift['drift_turns']:
                            st.markdown("**Sample drift triggers:**")
                            for drift_turn in session_drift['drift_turns']:
                                st.markdown(f"""
                                - **Turn {drift_turn['turn_number']}** (F={drift_turn['fidelity']:.3f}):
                                  _{drift_turn['user_input']}_
                                """)
            else:
                st.success("✓ No drift events detected! All sessions maintained high fidelity.")

        # Tab 3: Governance Effectiveness
        with pattern_tabs[2]:
            st.markdown("#### Governance Effectiveness Analysis")
            st.caption("Measures how well interventions improve fidelity")

            st.markdown(f"""
            **Overall Assessment:** {effectiveness['analysis']}
            """)

            st.divider()

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Avg Fidelity (With Intervention)",
                    f"{effectiveness['avg_fidelity_with_interventions']:.3f}",
                    help="Average fidelity when governance intervenes"
                )

            with col2:
                st.metric(
                    "Avg Fidelity (No Intervention)",
                    f"{effectiveness['avg_fidelity_without_interventions']:.3f}",
                    help="Average fidelity without intervention"
                )

            with col3:
                delta_val = effectiveness['effectiveness_score']
                st.metric(
                    "Effectiveness Score",
                    f"{delta_val:+.3f}",
                    delta=f"{delta_val:+.3f}",
                    help="Positive = interventions improve fidelity"
                )

            st.divider()

            # Effectiveness visualization
            st.markdown("##### 📊 Intervention Impact")

            if effectiveness['intervention_count'] > 0 and effectiveness['non_intervention_count'] > 0:
                comparison_data = {
                    'Category': ['With Intervention', 'Without Intervention'],
                    'Avg Fidelity': [
                        effectiveness['avg_fidelity_with_interventions'],
                        effectiveness['avg_fidelity_without_interventions']
                    ],
                    'Sample Size': [
                        effectiveness['intervention_count'],
                        effectiveness['non_intervention_count']
                    ]
                }

                if HAS_PLOTLY:
                    fig = go.Figure()

                    fig.add_trace(go.Bar(
                        x=comparison_data['Category'],
                        y=comparison_data['Avg Fidelity'],
                        text=[f"{v:.3f}" for v in comparison_data['Avg Fidelity']],
                        textposition='outside',
                        marker=dict(
                            color=['#4a90e2', '#95a5a6'],
                            line=dict(color='white', width=2)
                        ),
                        hovertemplate='<b>%{x}</b><br>Avg Fidelity: %{y:.3f}<br>Sample: %{customdata} turns<extra></extra>',
                        customdata=comparison_data['Sample Size']
                    ))

                    fig.add_hline(
                        y=0.76,
                        line_dash="dash",
                        line_color="green",
                        annotation_text="Target (F=0.76)",
                        annotation_position="right"
                    )

                    fig.update_layout(
                        yaxis_title="Average Fidelity",
                        height=350,
                        template='plotly_white',
                        yaxis_range=[0, 1],
                        showlegend=False
                    )

                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.write(comparison_data)

                st.caption(f"""
                **Sample sizes:** {effectiveness['intervention_count']:,} turns with intervention,
                {effectiveness['non_intervention_count']:,} turns without intervention
                """)
            else:
                st.info("Insufficient data for comparison visualization")

        # Tab 4: Anomaly Detection
        with pattern_tabs[3]:
            st.markdown("#### Anomalous Session Detection")
            st.caption("Statistical outlier detection using Median Absolute Deviation (MAD)")

            if 'message' in anomalies:
                st.info(f"ℹ️ {anomalies['message']}")
            else:
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric(
                        "Total Sessions",
                        anomalies['total_sessions'],
                        help="Number of sessions analyzed"
                    )

                with col2:
                    st.metric(
                        "Anomalies Found",
                        anomalies['anomaly_count'],
                        help="Sessions flagged as statistical outliers"
                    )

                with col3:
                    st.metric(
                        "Median Fidelity",
                        f"{anomalies['median_fidelity']:.3f}",
                        help="Median fidelity across all sessions"
                    )

                with col4:
                    st.metric(
                        "Median Intervention Rate",
                        f"{anomalies['median_intervention_rate']:.1%}",
                        help="Median intervention rate"
                    )

                st.divider()

                if anomalies['anomalies']:
                    st.warning(f"⚠️ **{len(anomalies['anomalies'])} anomalous session(s) detected**")

                    for anomaly in anomalies['anomalies']:
                        with st.expander(f"**{anomaly['session_id']}** - {len(anomaly['reasons'])} anomaly indicator(s)"):
                            st.markdown("**Anomaly indicators:**")
                            for reason in anomaly['reasons']:
                                st.markdown(f"- {reason}")

                            st.divider()

                            st.markdown("**Session metrics:**")
                            metrics = anomaly['metrics']
                            col_a, col_b, col_c = st.columns(3)

                            with col_a:
                                st.metric("Avg Fidelity", f"{metrics['avg_fidelity']:.3f}")

                            with col_b:
                                st.metric("Intervention Rate", f"{metrics['intervention_rate']:.1%}")

                            with col_c:
                                st.metric("Drift Rate", f"{metrics['basin_violation_rate']:.1%}")

                            st.caption("""
                            **Recommendations:**
                            - Review session context and user queries
                            - Check attractor configuration for this scenario
                            - Consider if unusual behavior is legitimate or problematic
                            """)
                else:
                    st.success("✓ No anomalous sessions detected. All sessions within normal statistical range.")

                st.divider()

                st.markdown("##### 📊 Detection Method")
                st.caption("""
                **MAD-based outlier detection:** Uses Median Absolute Deviation (MAD) to identify sessions
                with metrics that deviate significantly (>2.5 MAD) from the median. More robust than
                standard deviation for datasets with outliers.
                """)

    else:
        st.info("""
        No session data available yet for pattern detection.

        Complete at least one conversation to enable automated pattern analysis!
        """)

    st.divider()

    # Phase 9 Task 4: Statistical Summary Reports
    st.subheader("📊 Statistical Summary")
    st.caption("Publication-ready descriptive statistics and distribution analysis")

    if len(sessions_data) > 0:
        # Compute comprehensive statistics
        with st.spinner('📈 Computing statistics...'):
            stats = compute_comprehensive_statistics(sessions_data)

        if stats:
            # Display statistics in publication format
            st.markdown("##### Fidelity Score Statistics")
            st.caption(f"Based on {stats['n']} observations across {len(sessions_data)} session(s)")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Central Tendency**")
                st.write(f"**Mean (μ):** {stats['mean']:.3f}")
                st.write(f"**Median:** {stats['median']:.3f}")
                st.write(f"**95% CI:** [{stats['ci_lower']:.3f}, {stats['ci_upper']:.3f}]")
                st.caption(f"CI Width: ±{stats['ci_width']/2:.3f}")

            with col2:
                st.markdown("**Dispersion**")
                st.write(f"**Std Dev (σ):** {stats['std']:.3f}")
                st.write(f"**IQR:** {stats['iqr']:.3f}")
                st.write(f"**Range:** [{stats['min']:.3f}, {stats['max']:.3f}]")
                st.caption(f"Spread: {stats['max'] - stats['min']:.3f}")

            with col3:
                st.markdown("**Quartiles**")
                st.write(f"**Q1 (25%):** {stats['q25']:.3f}")
                st.write(f"**Q2 (50%):** {stats['median']:.3f}")
                st.write(f"**Q3 (75%):** {stats['q75']:.3f}")
                st.caption(f"Sample size: n={stats['n']}")

            st.divider()

            # Distribution visualization
            st.markdown("##### Distribution Analysis")
            st.caption("Histogram with theoretical normal distribution overlay")

            if HAS_PLOTLY:
                import numpy as np

                # Create histogram
                fig = go.Figure()

                # Observed data histogram
                fig.add_trace(go.Histogram(
                    x=stats['raw_data'],
                    name='Observed Fidelity',
                    nbinsx=30,
                    histnorm='probability density',
                    marker=dict(
                        color='#4a90e2',
                        line=dict(color='white', width=1)
                    ),
                    opacity=0.7
                ))

                # Normal distribution overlay
                try:
                    from scipy.stats import norm
                    x_range = np.linspace(stats['min'], stats['max'], 200)
                    normal_curve = norm.pdf(x_range, stats['mean'], stats['std'])

                    fig.add_trace(go.Scatter(
                        x=x_range,
                        y=normal_curve,
                        name='Normal Distribution',
                        line=dict(color='red', width=2, dash='dash'),
                        mode='lines'
                    ))
                except ImportError:
                    pass  # Skip normal overlay if scipy not available

                # Add mean line
                fig.add_vline(
                    x=stats['mean'],
                    line_dash="solid",
                    line_color="green",
                    annotation_text=f"Mean = {stats['mean']:.3f}",
                    annotation_position="top"
                )

                # Add Goldilocks target threshold
                fig.add_vline(
                    x=0.76,
                    line_dash="dash",
                    line_color="green",
                    annotation_text="Target (F=0.76)",
                    annotation_position="bottom"
                )

                fig.update_layout(
                    title='Fidelity Score Distribution',
                    xaxis_title='Fidelity Score',
                    yaxis_title='Probability Density',
                    height=400,
                    template='plotly_white',
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Install plotly for distribution visualization: `pip install plotly`")

            st.divider()

            # Export options for publications
            st.markdown("##### Export for Publications")
            st.caption("Download publication-ready statistical reports")

            col1, col2 = st.columns(2)

            with col1:
                # LaTeX table format
                latex_table = generate_latex_table(stats)
                st.download_button(
                    label="📄 Download LaTeX Table",
                    data=latex_table,
                    file_name='fidelity_statistics_table.tex',
                    mime='text/plain',
                    help="LaTeX table ready for academic papers"
                )

                st.caption("""
                **LaTeX Table**: Formatted table with all descriptive statistics.
                Copy-paste directly into your paper's `.tex` file.
                """)

            with col2:
                # CSV format for statistical software
                csv_data = generate_statistics_csv(sessions_data)
                st.download_button(
                    label="📊 Download Statistics CSV",
                    data=csv_data,
                    file_name='fidelity_turn_level_data.csv',
                    mime='text/csv',
                    help="Turn-level data for R, SPSS, Python analysis"
                )

                st.caption("""
                **Turn-Level CSV**: Raw data for import into statistical software
                (R, SPSS, Python pandas, etc.) for custom analysis.
                """)

        else:
            st.info("Need at least 2 data points to compute meaningful statistics")

    else:
        st.info("""
        No session data available yet for statistical analysis.

        Complete at least one conversation to generate statistical summaries!
        """)

    st.divider()

    # Export all analytics
    if st.button("📥 Export Complete Analytics", type="primary", help="Download comprehensive analytics report including session stats, efficacy summary, and aggregate metrics in JSON format"):
        with st.spinner('📊 Preparing analytics export...'):
            analytics_data = {
                'session_stats': stats,
                'efficacy_summary': efficacy_data if branches else [],
                'aggregate_metrics': {
                    'avg_delta_f': avg_delta_f if branches else 0.0,
                    'success_rate': success_rate if branches else 0.0,
                    'significance_rate': sig_rate if branches else 0.0
                },
                'exported_at': datetime.now().isoformat()
            }

            analytics_json = json.dumps(analytics_data, indent=2)
        st.download_button(
            "📥 Download Analytics JSON",
            data=analytics_json,
            file_name=f"teloscope_analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )


# ============================================================================
# Simulation UI Components
# ============================================================================

def render_intervention_timeline(turns):
    """
    Render interactive intervention timeline with clickable drift points.

    Args:
        turns: List of conversation turns
    """
    if not HAS_PLOTLY or not turns:
        return

    st.subheader("📈 Intervention Timeline")
    st.caption("Click on drift points to simulate counterfactual branches")

    # Extract data
    turn_numbers = []
    fidelities = []
    interventions = []
    drift_points = []

    for idx, turn in enumerate(turns):
        turn_numbers.append(idx + 1)
        metrics = turn.get('metrics', {})
        fidelity = metrics.get('telic_fidelity', 1.0)
        fidelities.append(fidelity)

        # Check for intervention
        metadata = turn.get('metadata', {})
        intervention_details = metadata.get('intervention_details', {})
        intervention_applied = intervention_details.get('intervention_applied', False)
        interventions.append(intervention_applied)

        # Mark drift points (low fidelity) - Goldilocks: Aligned threshold
        if fidelity < 0.76:
            drift_points.append({
                'turn': idx + 1,
                'fidelity': fidelity,
                'intervention': intervention_applied
            })

    # Create Plotly chart
    fig = go.Figure()

    # Fidelity line
    fig.add_trace(go.Scatter(
        x=turn_numbers,
        y=fidelities,
        mode='lines+markers',
        name='Fidelity',
        line=dict(color='#4a90e2', width=2),
        marker=dict(size=8),
        hovertemplate='Turn %{x}<br>Fidelity: %{y:.3f}<extra></extra>'
    ))

    # Goldilocks drift threshold line
    fig.add_hline(
        y=0.76,
        line_dash="dash",
        line_color="green",
        annotation_text="Aligned Threshold (0.76)",
        annotation_position="right"
    )

    # Mark interventions
    intervention_turns = [i+1 for i, applied in enumerate(interventions) if applied]
    intervention_fidelities = [fidelities[i] for i, applied in enumerate(interventions) if applied]

    if intervention_turns:
        fig.add_trace(go.Scatter(
            x=intervention_turns,
            y=intervention_fidelities,
            mode='markers',
            name='Intervention',
            marker=dict(
                size=15,
                color='green',
                symbol='star',
                line=dict(width=2, color='darkgreen')
            ),
            hovertemplate='Turn %{x}<br>Intervention Applied<br>Fidelity: %{y:.3f}<extra></extra>'
        ))

    fig.update_layout(
        xaxis_title="Turn Number",
        yaxis_title="Telic Fidelity (F)",
        yaxis_range=[0, 1],
        height=300,
        margin=dict(l=20, r=20, t=20, b=20),
        hovermode='closest',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)

    # Clickable drift points for simulation
    if drift_points:
        st.caption(f"**{len(drift_points)} drift point(s) detected** - Select one to simulate counterfactual:")

        cols = st.columns(min(len(drift_points), 5))
        for i, dp in enumerate(drift_points):
            with cols[i % len(cols)]:
                emoji = "🛡️" if dp['intervention'] else "⚠️"
                label = f"{emoji} Turn {dp['turn']}"
                if st.button(label, key=f"simulate_turn_{dp['turn']}", use_container_width=True):
                    st.session_state.simulate_from_turn = dp['turn']
                    st.session_state.simulate_fidelity = dp['fidelity']


def render_simulation_ui():
    """Render simulation controls and results."""
    if 'simulate_from_turn' not in st.session_state:
        return

    st.divider()
    st.subheader("🔬 Counterfactual Simulation")

    turn_number = st.session_state.simulate_from_turn
    trigger_fidelity = st.session_state.simulate_fidelity

    st.info(f"**Simulating from Turn {turn_number}** (F={trigger_fidelity:.3f})")

    # Get conversation history up to this turn
    turns = st.session_state.current_session.get('turns', [])
    conversation_history = []
    for idx in range(min(turn_number, len(turns))):
        turn = turns[idx]
        conversation_history.append({"role": "user", "content": turn['user_input']})
        conversation_history.append({"role": "assistant", "content": turn['assistant_response']})

    # Simulation parameters
    with st.expander("⚙️ Simulation Parameters", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            simulation_turns = st.slider(
                "Simulation Turns",
                min_value=3,
                max_value=10,
                value=5,
                help="Number of turns to simulate into the future"
            )

        with col2:
            topic_hint = st.text_input(
                "Topic Hint (optional)",
                placeholder="e.g., cooking recipes",
                help="Hint for potential drift direction"
            )

    # Run simulation button
    if st.button("▶️ Run Simulation", type="primary", use_container_width=True):
        with st.spinner("🔬 Simulating counterfactual branches..."):
            try:
                # Get attractor center
                if hasattr(st.session_state.steward, 'spc_engine'):
                    attractor_center = st.session_state.steward.spc_engine.attractor_center
                else:
                    st.error("Attractor center not available")
                    return

                # Run simulation
                simulation_id = st.session_state.simulator.simulate_counterfactual(
                    trigger_turn=turn_number,
                    trigger_fidelity=trigger_fidelity,
                    trigger_reason=f"User-triggered simulation from Turn {turn_number}",
                    conversation_history=conversation_history,
                    attractor_center=attractor_center,
                    distance_scale=2.0,
                    topic_hint=topic_hint if topic_hint else None
                )

                # Store result
                st.session_state.simulation_results[simulation_id] = {
                    'turn_number': turn_number,
                    'trigger_fidelity': trigger_fidelity,
                    'simulation_turns': simulation_turns,
                    'topic_hint': topic_hint
                }
                st.session_state.active_simulation = simulation_id
                st.success(f"✅ Simulation complete: {simulation_id}")
                st.rerun()

            except Exception as e:
                st.error(f"❌ Simulation failed: {e}")
                st.exception(e)

    # Display active simulation results
    if 'active_simulation' in st.session_state:
        render_simulation_results(st.session_state.active_simulation)


def render_simulation_results(simulation_id):
    """
    Render simulation results with side-by-side comparison.

    Args:
        simulation_id: ID of simulation to display
    """
    st.divider()
    st.subheader("📊 Simulation Results")

    # Get comparison data
    comparison = st.session_state.simulator.get_comparison(simulation_id)

    if not comparison:
        st.error("Simulation results not found")
        return

    # Summary metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Trigger Turn", comparison['trigger_turn'])

    with col2:
        st.metric("Trigger Fidelity", f"{comparison['trigger_fidelity']:.3f}")

    with col3:
        delta_f = comparison['comparison']['delta_f']
        improvement = comparison['comparison']['improvement']
        emoji = "✅" if improvement else "❌"
        st.metric("ΔF (Improvement)", f"{delta_f:+.3f}", delta=emoji)

    # Fidelity trajectories chart
    if HAS_PLOTLY:
        st.subheader("📈 Fidelity Trajectories")

        orig_traj = comparison['comparison']['original_trajectory']
        telos_traj = comparison['comparison']['telos_trajectory']
        turn_nums = list(range(comparison['trigger_turn'] + 1, comparison['trigger_turn'] + 1 + len(orig_traj)))

        fig = go.Figure()

        # Original branch
        fig.add_trace(go.Scatter(
            x=turn_nums,
            y=orig_traj,
            mode='lines+markers',
            name='Original (No Governance)',
            line=dict(color='#ff6b6b', width=2, dash='dash'),
            marker=dict(size=8)
        ))

        # TELOS branch
        fig.add_trace(go.Scatter(
            x=turn_nums,
            y=telos_traj,
            mode='lines+markers',
            name='TELOS (Governed)',
            line=dict(color='#51cf66', width=2),
            marker=dict(size=8)
        ))

        # Goldilocks drift threshold
        fig.add_hline(
            y=0.76,
            line_dash="dot",
            line_color="green",
            annotation_text="Aligned Threshold (0.76)"
        )

        fig.update_layout(
            xaxis_title="Turn Number",
            yaxis_title="Telic Fidelity (F)",
            yaxis_range=[0, 1],
            height=350,
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(fig, use_container_width=True)

    # Turn-by-turn comparison
    st.subheader("💬 Turn-by-Turn Comparison")

    original_turns = comparison['original']['turns']
    telos_turns = comparison['telos']['turns']

    for i in range(len(original_turns)):
        orig = original_turns[i]
        telos = telos_turns[i]

        with st.expander(f"**Turn {orig['turn_number']}** - F: {orig['fidelity']:.3f} (orig) vs {telos['fidelity']:.3f} (TELOS)", expanded=(i==0)):
            st.caption(f"**User:** {orig['user_message']}")

            col_left, col_right = st.columns(2)

            with col_left:
                st.markdown("**🔴 Original Response**")
                st.markdown(
                    f'<div style="background-color: #fff3cd; padding: 15px; border-radius: 5px; border-left: 3px solid #ffc107;">'
                    f'{orig["assistant_response"]}'
                    f'</div>',
                    unsafe_allow_html=True
                )
                st.caption(f"Fidelity: {orig['fidelity']:.3f}")

            with col_right:
                st.markdown("**🟢 TELOS Response**")
                st.markdown(
                    f'<div style="background-color: #d4edda; padding: 15px; border-radius: 5px; border-left: 3px solid #28a745;">'
                    f'{telos["assistant_response"]}'
                    f'</div>',
                    unsafe_allow_html=True
                )
                st.caption(f"Fidelity: {telos['fidelity']:.3f}")

    # Download evidence buttons
    st.divider()
    st.subheader("📥 Export Evidence")

    col1, col2 = st.columns(2)

    with col1:
        markdown_export = st.session_state.simulator.export_evidence(simulation_id, format='markdown')
        if markdown_export:
            st.download_button(
                "📄 Download Markdown Report",
                data=markdown_export,
                file_name=f"simulation_{simulation_id}.md",
                mime="text/markdown",
                use_container_width=True
            )

    with col2:
        json_export = st.session_state.simulator.export_evidence(simulation_id, format='json')
        if json_export:
            st.download_button(
                "📋 Download JSON Data",
                data=json_export,
                file_name=f"simulation_{simulation_id}.json",
                mime="application/json",
                use_container_width=True
            )


# ============================================================================
# Onboarding Functions
# ============================================================================

def show_onboarding_screen():
    """
    Display onboarding screen for choosing how to establish Primacy Attractor.
    Returns True if onboarding is complete, False if still showing onboarding.
    """
    # Initialize onboarding state if not set
    if 'onboarding_step' not in st.session_state:
        st.session_state.onboarding_step = 'mode_selection'

    # Mode selection screen
    if st.session_state.onboarding_step == 'mode_selection':
        st.markdown("# 🎯 Welcome to TELOS Observatory")
        st.markdown("---")
        st.markdown("""
        Before we begin, let's establish your **Primacy Attractor** - the mathematical representation
        of your conversation purpose that will guide governance.
        """)

        st.markdown("### How would you like to proceed?")
        st.markdown("")

        # Use radio buttons for mode selection
        mode_choice = st.radio(
            "Choose your onboarding experience:",
            options=[
                "Progressive (Recommended)",
                "Pre-defined",
                "Hybrid",
                "Pristine Mode"
            ],
            index=0,
            key="attractor_mode_choice"
        )

        # Show descriptions for each mode
        st.markdown("---")

        if mode_choice == "Progressive (Recommended)":
            st.info("""
            **Progressive Discovery**

            - Chat naturally for 8-10 turns
            - TELOS learns your purpose organically
            - Shows calibration messages with progress
            - Convergence notification when ready
            - Then governance begins automatically

            *Best for: Natural conversations, exploratory discussions*
            """)

        elif mode_choice == "Pre-defined":
            st.info("""
            **Pre-defined Purpose**

            - Tell TELOS your purpose upfront
            - Simple form to fill out
            - Optional boundaries field
            - Immediate governance from Turn 1

            *Best for: Focused tasks, specific goals*
            """)

        elif mode_choice == "Hybrid":
            st.info("""
            **Hybrid Approach**

            - Answer 2-3 quick questions
            - Seeds the progressive extractor
            - Then refines over 8-10 turns
            - Best of both worlds

            *Best for: Guided start with natural refinement*
            """)

        elif mode_choice == "Pristine Mode":
            st.info("""
            **Pristine Experience**

            - No questions, no setup
            - Just start chatting
            - Progressive extraction happens silently
            - No calibration messages shown
            - Governance happens invisibly

            *Best for: Seamless 'just works' experience*
            """)

        st.markdown("")

        # Continue button
        if st.button("Continue →", type="primary", use_container_width=True):
            mode_map = {
                "Progressive (Recommended)": "progressive",
                "Pre-defined": "predefined",
                "Hybrid": "hybrid",
                "Pristine Mode": "pristine"
            }
            st.session_state.attractor_mode = mode_map[mode_choice]
            st.session_state.onboarding_step = st.session_state.attractor_mode
            st.rerun()

        return False  # Still in onboarding

    # Handle Pre-defined mode form
    elif st.session_state.onboarding_step == 'predefined':
        st.markdown("# 🎯 Define Your Primacy Attractor")
        st.markdown("---")

        with st.form("predefined_attractor_form"):
            st.markdown("### Purpose")
            st.caption("What's your purpose for this conversation? (Required)")
            purpose_input = st.text_area(
                "Purpose",
                label_visibility="collapsed",
                placeholder="e.g., 'Help me plan a marketing campaign for a new product launch'",
                height=100
            )

            st.markdown("### Boundaries")
            st.caption("Any boundaries or constraints? (Optional)")
            boundaries_input = st.text_area(
                "Boundaries",
                label_visibility="collapsed",
                placeholder="e.g., 'Stay practical, no medical advice, focus on digital channels'",
                height=80
            )

            submitted = st.form_submit_button("Establish Attractor", type="primary", use_container_width=True)

            if submitted:
                if purpose_input.strip():
                    # Store the predefined purpose
                    st.session_state.predefined_purpose = purpose_input.strip()
                    st.session_state.predefined_boundaries = boundaries_input.strip() if boundaries_input.strip() else None
                    st.session_state.onboarding_complete = True
                    st.success("✅ Primacy Attractor established! Starting governed conversation...")
                    st.rerun()
                else:
                    st.error("⚠️ Please enter a purpose before continuing.")

        return False  # Still in onboarding

    # Handle Hybrid mode questions
    elif st.session_state.onboarding_step == 'hybrid':
        st.markdown("# 🎯 Seed Your Primacy Attractor")
        st.markdown("---")
        st.markdown("Answer a few quick questions to get started. TELOS will refine understanding as you chat.")
        st.markdown("")

        with st.form("hybrid_seed_form"):
            st.markdown("### Q1: What brings you here?")
            st.caption("In one sentence, what's your main goal?")
            q1_input = st.text_input(
                "Q1",
                label_visibility="collapsed",
                placeholder="e.g., 'I need to design a training program for my team'"
            )

            st.markdown("### Q2: Focus or avoid anything?")
            st.caption("Anything specific to focus on or avoid?")
            q2_input = st.text_input(
                "Q2",
                label_visibility="collapsed",
                placeholder="e.g., 'Focus on practical exercises, avoid theory-heavy content'"
            )

            submitted = st.form_submit_button("Start Conversation →", type="primary", use_container_width=True)

            if submitted:
                if q1_input.strip():
                    # Store hybrid seeds
                    st.session_state.hybrid_seed_purpose = q1_input.strip()
                    st.session_state.hybrid_seed_boundaries = q2_input.strip() if q2_input.strip() else None
                    st.session_state.onboarding_complete = True
                    st.success("✅ Seeds established! TELOS will refine as you chat...")
                    st.rerun()
                else:
                    st.error("⚠️ Please answer Q1 before continuing.")

        return False  # Still in onboarding

    # Progressive and Pristine modes complete immediately
    elif st.session_state.onboarding_step in ['progressive', 'pristine']:
        st.session_state.onboarding_complete = True
        st.rerun()
        return False

    return False


# ============================================================================
# Main Application
# ============================================================================

def main():
    """Main application entry point."""

    # ONBOARDING REMOVED - Skip directly to ChatGPT interface
    # Set onboarding as complete and use config-based attractor
    if 'onboarding_complete' not in st.session_state:
        st.session_state.onboarding_complete = True
        st.session_state.onboarding_step = 'pre_defined'
        st.session_state.attractor_mode = 'config'  # Use config.json for attractor
        # DEFAULT TO CHATGPT INTERFACE (not legacy tabs)
        st.session_state.ui_mode = 'chat'
        # MINIMALISTIC: Windows hidden by default, keyboard-activated
        st.session_state.show_steward_lens = False
        st.session_state.show_teloscope_window = False

    # Initialize TELOSCOPE
    initialize_teloscope()

    # Initialize DeckManager (needed by sidebar)
    if 'deck_manager' not in st.session_state:
        st.session_state.deck_manager = DeckManager(st.session_state)

    # Render sidebar
    render_sidebar()

    # Get UI mode
    ui_mode = st.session_state.get('ui_mode', 'legacy')

    # Main header with mode selector (ONLY for legacy mode)
    if ui_mode == 'legacy':
        col_title, col_mode = st.columns([4, 1])

        with col_title:
            st.title("🔭 TELOSCOPE Observatory")
            st.markdown("""
            **Telically Entrained Linguistic Operational Substrate Counterfactual Observation via Purpose-scoped Experimentation**
            """)
            st.caption("Making AI Governance Observable Through Quantifiable Evidence")

        with col_mode:
            st.markdown("<br>", unsafe_allow_html=True)  # Spacer

            # Split into two columns for mode selector and Research Lens toggle
            mode_col, lens_col = st.columns([2, 1])

            with mode_col:
                current_mode = get_mode()
                # Handle legacy 'Research' mode name for backward compatibility
                if current_mode == 'Research':
                    current_mode = 'Research Mode'
                    st.session_state.mode = 'Research Mode'

                mode_index = 0 if current_mode == 'Basic' else (1 if current_mode == 'Advanced' else 2)
                new_mode = st.selectbox(
                    "Mode",
                    options=['Basic', 'Advanced', 'Research Mode'],
                    index=mode_index,
                    help="Basic: Simple | Advanced: Technical details | Research Mode: Live mathematics",
                    key="mode_selector"
                )
                if new_mode != current_mode:
                    st.session_state.mode = new_mode
                    st.rerun()

            with lens_col:
                # Research Lens toggle - disabled if already in Research Mode
                is_research_mode = (get_mode() == 'Research Mode')
                research_lens = st.toggle(
                    "🔬 Research Lens",
                    value=st.session_state.get('research_lens_active', False),
                    key="main_research_lens_toggle",
                    help="Overlay live mathematics on current mode" if not is_research_mode else "Already in Research Mode",
                    disabled=is_research_mode
                )
                # Update session state
                if not is_research_mode:
                    st.session_state.research_lens_active = research_lens
                else:
                    st.session_state.research_lens_active = False

        st.divider()

    # ========================================================================
    # UI Mode Router: Chat Interface vs Legacy Tabs
    # ========================================================================

    ui_mode = st.session_state.get('ui_mode', 'legacy')

    if ui_mode == 'chat':
        # NEW: ChatGPT-style interface (Phase 2)
        render_chat_interface()
    else:
        # LEGACY: Tab-based interface (default for backward compatibility)

        # ========================================================================
        # Navigation Breadcrumbs & State Indicators
        # ========================================================================

        # Initialize active tab tracking
        if 'active_tab' not in st.session_state:
            st.session_state.active_tab = 0

        # Build breadcrumb navigation
        breadcrumb_parts = ["TELOSCOPE"]

        # Determine current view based on active tab
        tab_names = ["Live Session", "Session Replay", "TELOSCOPE View", "Analytics"]
        if st.session_state.active_tab < len(tab_names):
            breadcrumb_parts.append(tab_names[st.session_state.active_tab])

        # Add mode context
        current_mode = get_mode()
        breadcrumb_parts.append(current_mode)

        # Display breadcrumb and state indicators
        breadcrumb_col, state_col = st.columns([3, 1])

        with breadcrumb_col:
            st.caption(' → '.join(breadcrumb_parts))

        with state_col:
            # Visual state indicators
            # Mode icon: 🟢 for Live/Active, ⏸️ for Replay/Paused
            mode_icon = "🟢" if st.session_state.active_tab == 0 else "⏸️"
            mode_label = "LIVE" if st.session_state.active_tab == 0 else "REPLAY"

            # Governance status: 🔵 for Active, ⚪ for Inactive
            governance_active = st.session_state.get('governance_enabled', True)
            governance_icon = "🔵" if governance_active else "⚪"
            governance_label = "GOVERNED" if governance_active else "UNGOVERNED"

            st.caption(f"{mode_icon} {mode_label} | {governance_icon} {governance_label}")

        st.markdown("---")  # Visual separator

        # Main tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "🔴 Live Session",
            "⏮️ Session Replay",
            "🔭 TELOSCOPE",
            "📊 Analytics"
        ])

        with tab1:
            render_live_session()

        with tab2:
            render_session_replay()

        with tab3:
            render_teloscope_view()

        with tab4:
            render_analytics_dashboard()


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    main()
