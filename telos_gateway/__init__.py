"""
TELOS Gateway -- Production-Hardened API Gateway
=================================================

OpenAI-compatible API gateway with TELOS governance,
authentication, rate limiting, and health monitoring.

Usage:
    # Agent developer changes ONE environment variable:
    export OPENAI_BASE_URL=http://localhost:8000/v1

    # Their agent code stays exactly the same.
    # All LLM calls now route through TELOS governance.
"""

__version__ = "1.0.0"
