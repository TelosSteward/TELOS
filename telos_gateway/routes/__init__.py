"""TELOS Gateway Routes"""

from .chat_completions import router as chat_router
from .agent_registration import router as agent_router
from .health import router as health_router

__all__ = ["chat_router", "agent_router", "health_router"]
