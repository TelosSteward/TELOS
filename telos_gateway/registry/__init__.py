"""
TELOS Agent Registry
====================

Agent onboarding and profile management.
Similar to PA establishment in conversational TELOS,
this establishes the agent's purpose BEFORE governance.
"""

from .agent_profile import (
    AgentProfile,
    ToolDefinition,
    RiskLevel,
    AgentRegistrationRequest,
    AgentRegistrationResponse,
)
from .agent_registry import AgentRegistry, get_registry

__all__ = [
    "AgentProfile",
    "ToolDefinition",
    "RiskLevel",
    "AgentRegistry",
    "get_registry",
    "AgentRegistrationRequest",
    "AgentRegistrationResponse",
]
