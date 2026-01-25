"""
TELOS Corpus Configurator - UI Components
==========================================

All Streamlit UI components for the Corpus Configurator MVP.
Each component is self-contained and follows the TELOS design system.
"""

from .domain_selector import render_domain_selector
from .corpus_uploader import render_corpus_uploader
from .corpus_manager import render_corpus_manager
from .pa_configurator import render_pa_configurator
from .threshold_config import render_threshold_config
from .activation_panel import render_activation_panel
from .dashboard_metrics import render_dashboard_metrics
from .test_query_interface import render_test_query_interface
from .audit_panel import render_audit_panel
from .corpus_browser import (
    render_corpus_browser,
    render_document_detail,
    render_retrieval_panel,
    RetrievedChunk
)

__all__ = [
    'render_domain_selector',
    'render_corpus_uploader',
    'render_corpus_manager',
    'render_pa_configurator',
    'render_threshold_config',
    'render_activation_panel',
    'render_dashboard_metrics',
    'render_test_query_interface',
    'render_audit_panel',
    # Corpus browser
    'render_corpus_browser',
    'render_document_detail',
    'render_retrieval_panel',
    'RetrievedChunk',
]
