"""
Steward Integration - Conversational Research Assistant (PAID)

Provides AI-powered Q&A about TELOS conversation sessions using Mistral API.
Cost: ~$0.002 per query (uses existing Mistral API, same as TELOS core).

Features:
- Natural language questions about current session
- Scope validation (rejects out-of-scope queries with helpful redirects)
- Uses existing StewardAnalyzer from steward_analysis.py
- Clear cost warnings before API calls

Important:
- This is for research analysis ONLY (not general assistance)
- Steward = research assistant (different from Steward PM CLI tool)
- Reuses existing MistralClient instance for efficiency
"""

from .steward_chat import StewardChat

__all__ = ['StewardChat']
