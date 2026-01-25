"""
TELOS Corpus Configurator - State Manager
==========================================

Session state orchestration for the TELOS Corpus Configurator MVP.
Manages all session state keys, provides getters/setters with validation,
and handles state persistence.

Categories:
- Configuration state (domain, PA config, thresholds)
- Corpus state (documents, embeddings, stats)
- Governance state (active PA, governance engine instance)
- UI state (current step, sidebar expanded, etc.)
- Audit state (log entries)
"""

from typing import Any, Dict, Optional, List
from datetime import datetime
import json
from pathlib import Path

import streamlit as st


# ============================================================================
# STATE INITIALIZATION
# ============================================================================

def initialize_state() -> None:
    """
    Initialize all session state keys on first load.

    This function should be called once at the start of the application
    to ensure all necessary state keys exist with default values.
    """

    # Configuration State
    if 'selected_domain' not in st.session_state:
        st.session_state.selected_domain = None

    if 'pa_configured' not in st.session_state:
        st.session_state.pa_configured = False

    if 'pa_instance' not in st.session_state:
        st.session_state.pa_instance = None

    if 'pa_name' not in st.session_state:
        st.session_state.pa_name = ''

    if 'pa_purpose' not in st.session_state:
        st.session_state.pa_purpose = ''

    if 'pa_scope' not in st.session_state:
        st.session_state.pa_scope = ''

    if 'pa_exclusions' not in st.session_state:
        st.session_state.pa_exclusions = ''

    if 'pa_prohibitions' not in st.session_state:
        st.session_state.pa_prohibitions = ''

    if 'thresholds_configured' not in st.session_state:
        st.session_state.thresholds_configured = False

    # Corpus State
    if 'corpus_engine' not in st.session_state:
        from engine.corpus_engine import CorpusEngine
        st.session_state.corpus_engine = CorpusEngine()

    if 'corpus_updated' not in st.session_state:
        st.session_state.corpus_updated = False

    # Governance State
    if 'governance_engine' not in st.session_state:
        from engine.governance_engine import GovernanceEngine
        st.session_state.governance_engine = GovernanceEngine()

    if 'governance_active' not in st.session_state:
        st.session_state.governance_active = False

    # UI State
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 0  # 0-indexed step number

    if 'sidebar_expanded' not in st.session_state:
        st.session_state.sidebar_expanded = True

    # Audit State
    if 'audit_filter_tiers' not in st.session_state:
        st.session_state.audit_filter_tiers = [1, 2, 3]

    if 'audit_search_query' not in st.session_state:
        st.session_state.audit_search_query = ''

    if 'confirm_clear' not in st.session_state:
        st.session_state.confirm_clear = False


# ============================================================================
# STATE GETTERS
# ============================================================================

def get_current_step() -> int:
    """Get current step index (0-indexed)."""
    return st.session_state.get('current_step', 0)


def get_selected_domain() -> Optional[str]:
    """Get selected domain key."""
    return st.session_state.get('selected_domain')


def get_pa_instance():
    """Get Primacy Attractor instance."""
    return st.session_state.get('pa_instance')


def get_corpus_engine():
    """Get CorpusEngine instance."""
    return st.session_state.get('corpus_engine')


def get_governance_engine():
    """Get GovernanceEngine instance."""
    return st.session_state.get('governance_engine')


def is_governance_active() -> bool:
    """Check if governance is currently active."""
    engine = get_governance_engine()
    if engine:
        return engine.is_active()
    return False


def get_corpus_stats() -> Dict[str, Any]:
    """Get corpus statistics."""
    engine = get_corpus_engine()
    if engine:
        return engine.get_stats()
    return {
        'total_documents': 0,
        'embedded_documents': 0,
        'not_embedded': 0,
        'embedding_percentage': 0
    }


# ============================================================================
# STATE SETTERS
# ============================================================================

def set_current_step(step: int) -> None:
    """
    Set current step index.

    Args:
        step: Step index (0-indexed)
    """
    if 0 <= step <= 5:  # 6 total steps (0-5)
        st.session_state.current_step = step
    else:
        raise ValueError(f"Invalid step index: {step}. Must be 0-5.")


def set_selected_domain(domain: Optional[str]) -> None:
    """
    Set selected domain.

    Args:
        domain: Domain key ('healthcare', 'financial', 'legal', 'custom')
    """
    valid_domains = ['healthcare', 'financial', 'legal', 'custom', None]
    if domain in valid_domains:
        st.session_state.selected_domain = domain
    else:
        raise ValueError(f"Invalid domain: {domain}")


def set_pa_instance(pa) -> None:
    """
    Set Primacy Attractor instance.

    Args:
        pa: PrimacyAttractor instance
    """
    st.session_state.pa_instance = pa
    st.session_state.pa_configured = pa is not None


# ============================================================================
# READINESS CHECKS
# ============================================================================

def check_step_readiness() -> Dict[str, bool]:
    """
    Check readiness for each step.

    Returns:
        Dictionary mapping step names to readiness status
    """
    corpus_stats = get_corpus_stats()
    pa = get_pa_instance()
    governance_engine = get_governance_engine()

    return {
        'domain_selection': True,  # Always ready
        'corpus_upload': True,  # Always ready
        'pa_configuration': get_selected_domain() is not None,
        'threshold_configuration': pa is not None and pa.embedding is not None,
        'activation': (
            pa is not None and
            pa.embedding is not None and
            corpus_stats['total_documents'] > 0 and
            corpus_stats['embedded_documents'] > 0
        ),
        'dashboard': governance_engine is not None and governance_engine.is_active()
    }


def get_step_status(step_index: int) -> str:
    """
    Get status for a specific step.

    Args:
        step_index: Step index (0-indexed)

    Returns:
        Status string: 'ready', 'pending', 'completed', or 'active'
    """
    readiness = check_step_readiness()
    current_step = get_current_step()

    step_names = [
        'domain_selection',
        'corpus_upload',
        'pa_configuration',
        'threshold_configuration',
        'activation',
        'dashboard'
    ]

    if step_index >= len(step_names):
        return 'pending'

    step_name = step_names[step_index]

    # Current step
    if step_index == current_step:
        return 'active'

    # Completed steps
    if step_index < current_step:
        return 'completed'

    # Future steps - check readiness
    if readiness.get(step_name, False):
        return 'ready'

    return 'pending'


# ============================================================================
# STATE PERSISTENCE
# ============================================================================

def save_configuration(filepath: str) -> bool:
    """
    Save current configuration to JSON file.

    Args:
        filepath: Path to save configuration

    Returns:
        True if successful, False otherwise
    """
    try:
        config = {
            'saved_at': datetime.utcnow().isoformat(),
            'domain': get_selected_domain(),
            'pa_configured': st.session_state.get('pa_configured', False),
            'pa_name': st.session_state.get('pa_name', ''),
            'pa_purpose': st.session_state.get('pa_purpose', ''),
            'pa_scope': st.session_state.get('pa_scope', ''),
            'pa_exclusions': st.session_state.get('pa_exclusions', ''),
            'pa_prohibitions': st.session_state.get('pa_prohibitions', ''),
            'current_step': get_current_step()
        }

        # Include PA instance data if available
        pa = get_pa_instance()
        if pa:
            config['pa_data'] = pa.to_dict()

        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        return True

    except Exception as e:
        print(f"Error saving configuration: {e}")
        return False


def load_configuration(filepath: str) -> bool:
    """
    Load configuration from JSON file.

    Args:
        filepath: Path to configuration file

    Returns:
        True if successful, False otherwise
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # Restore state
        set_selected_domain(config.get('domain'))
        st.session_state.pa_configured = config.get('pa_configured', False)
        st.session_state.pa_name = config.get('pa_name', '')
        st.session_state.pa_purpose = config.get('pa_purpose', '')
        st.session_state.pa_scope = config.get('pa_scope', '')
        st.session_state.pa_exclusions = config.get('pa_exclusions', '')
        st.session_state.pa_prohibitions = config.get('pa_prohibitions', '')

        # Restore current step
        step = config.get('current_step', 0)
        set_current_step(step)

        # Note: PA instance and corpus must be re-embedded after loading
        # This is intentional for security and freshness

        return True

    except Exception as e:
        print(f"Error loading configuration: {e}")
        return False


# ============================================================================
# STATE RESET FUNCTIONS
# ============================================================================

def reset_configuration_state() -> None:
    """Reset all configuration state (domain, PA, thresholds)."""
    st.session_state.selected_domain = None
    st.session_state.pa_configured = False
    st.session_state.pa_instance = None
    st.session_state.pa_name = ''
    st.session_state.pa_purpose = ''
    st.session_state.pa_scope = ''
    st.session_state.pa_exclusions = ''
    st.session_state.pa_prohibitions = ''
    st.session_state.thresholds_configured = False


def reset_corpus_state() -> None:
    """Reset corpus state (clears all documents)."""
    from engine.corpus_engine import CorpusEngine
    st.session_state.corpus_engine = CorpusEngine()
    st.session_state.corpus_updated = False


def reset_governance_state() -> None:
    """Reset governance state (deactivates governance)."""
    from engine.governance_engine import GovernanceEngine
    st.session_state.governance_engine = GovernanceEngine()
    st.session_state.governance_active = False


def reset_ui_state() -> None:
    """Reset UI state to defaults."""
    st.session_state.current_step = 0
    st.session_state.sidebar_expanded = True


def reset_all_state() -> None:
    """Reset all state to defaults (full reset)."""
    reset_configuration_state()
    reset_corpus_state()
    reset_governance_state()
    reset_ui_state()


# ============================================================================
# NAVIGATION HELPERS
# ============================================================================

def can_navigate_to_step(target_step: int) -> bool:
    """
    Check if navigation to a specific step is allowed.

    Args:
        target_step: Target step index

    Returns:
        True if navigation is allowed, False otherwise
    """
    if target_step < 0 or target_step > 5:
        return False

    readiness = check_step_readiness()
    step_names = [
        'domain_selection',
        'corpus_upload',
        'pa_configuration',
        'threshold_configuration',
        'activation',
        'dashboard'
    ]

    # Always allow backward navigation
    if target_step <= get_current_step():
        return True

    # Forward navigation requires readiness
    if target_step < len(step_names):
        return readiness.get(step_names[target_step], False)

    return False


def navigate_to_step(step_index: int) -> None:
    """
    Navigate to a specific step if allowed.

    Args:
        step_index: Target step index
    """
    if can_navigate_to_step(step_index):
        set_current_step(step_index)
    else:
        st.warning(f"Cannot navigate to step {step_index}. Complete previous steps first.")


def next_step() -> None:
    """Navigate to next step if allowed."""
    current = get_current_step()
    navigate_to_step(current + 1)


def previous_step() -> None:
    """Navigate to previous step if allowed."""
    current = get_current_step()
    if current > 0:
        navigate_to_step(current - 1)


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Initialization
    'initialize_state',

    # Getters
    'get_current_step',
    'get_selected_domain',
    'get_pa_instance',
    'get_corpus_engine',
    'get_governance_engine',
    'is_governance_active',
    'get_corpus_stats',

    # Setters
    'set_current_step',
    'set_selected_domain',
    'set_pa_instance',

    # Readiness
    'check_step_readiness',
    'get_step_status',

    # Persistence
    'save_configuration',
    'load_configuration',

    # Reset
    'reset_configuration_state',
    'reset_corpus_state',
    'reset_governance_state',
    'reset_ui_state',
    'reset_all_state',

    # Navigation
    'can_navigate_to_step',
    'navigate_to_step',
    'next_step',
    'previous_step'
]
