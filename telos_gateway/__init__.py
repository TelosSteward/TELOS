"""
TELOS Gateway
=============

OpenAI-compatible API gateway with TELOS governance.

The "One Ring" that governs all AI agents - sits between any agent
and its LLM provider, applying Constitutional Filter governance
to every request and response.

Usage:
    # Agent developer changes ONE environment variable:
    export OPENAI_BASE_URL=https://api.telos-labs.ai/v1

    # Their agent code stays exactly the same.
    # All LLM calls now route through TELOS governance.
"""

__version__ = "0.1.0"
