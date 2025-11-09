"""
TELOS Demo Mode Package

Provides pre-configured demonstrations of the TELOS AI governance framework.
"""

from .telos_framework_demo import (
    get_demo_attractor_config,
    get_demo_welcome_message,
    get_steward_intro_message,
    get_demo_slides,
    get_demo_completion_message
)

from .telos_corpus_loader import (
    TELOSCorpusLoader,
    format_context_for_llm
)

from .demo_content import (
    get_turn_annotation,
    get_suggested_questions,
    should_show_annotation
)

__all__ = [
    'get_demo_attractor_config',
    'get_demo_welcome_message',
    'get_steward_intro_message',
    'get_demo_slides',
    'get_demo_completion_message',
    'TELOSCorpusLoader',
    'format_context_for_llm',
    'get_turn_annotation',
    'get_suggested_questions',
    'should_show_annotation'
]
