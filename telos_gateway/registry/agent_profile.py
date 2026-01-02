"""
Agent Profile Model
====================

Defines the structure for registered agents.
This is the agentic equivalent of PA establishment -
we know WHO the agent is and WHAT it should do before governing.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum
from datetime import datetime
import hashlib
import secrets
import numpy as np


class RiskLevel(Enum):
    """Risk classification for agent actions."""
    LOW = "low"           # Read-only, informational
    MEDIUM = "medium"     # Can modify data, limited scope
    HIGH = "high"         # Can execute code, access systems
    CRITICAL = "critical" # Financial, PII, infrastructure


@dataclass
class ToolDefinition:
    """
    Definition of a tool the agent is authorized to use.

    This is the "tool manifest" - what tools SHOULD this agent have?
    At runtime, we compare requested tools against this manifest.
    """
    name: str
    description: str
    risk_level: RiskLevel = RiskLevel.MEDIUM
    requires_approval: bool = False
    parameters_schema: Optional[Dict[str, Any]] = None

    # Fidelity requirements for this specific tool
    # Some tools need higher fidelity to execute
    min_fidelity_threshold: float = 0.45


@dataclass
class AgentProfile:
    """
    Complete profile of a registered agent.

    This is established BEFORE the agent makes requests,
    giving TELOS "precognition" of what to govern.

    Analogous to:
    - PA establishment in conversational TELOS
    - Context enrichment with domain knowledge
    - User purpose onboarding
    """

    # Identity
    agent_id: str                          # Unique identifier
    name: str                              # Human-readable name
    owner: str                             # Organization/owner
    api_key_hash: str                      # Hashed API key for auth

    # Primacy Attractor (Purpose)
    purpose_statement: str                 # What this agent is FOR
    purpose_embedding: Optional[np.ndarray] = None  # Pre-computed embedding

    # Domain Context (like corpus in conversational TELOS)
    domain: str = "general"                # e.g., "finance", "healthcare", "coding"
    domain_keywords: List[str] = field(default_factory=list)
    domain_description: str = ""           # Rich description for embedding

    # Tool Manifest (what tools SHOULD this agent have)
    authorized_tools: List[ToolDefinition] = field(default_factory=list)

    # Risk Profile
    overall_risk_level: RiskLevel = RiskLevel.MEDIUM
    high_risk_mode: bool = False           # If true, ESCALATE instead of INERT

    # Governance Thresholds (can override defaults per agent)
    custom_execute_threshold: Optional[float] = None
    custom_clarify_threshold: Optional[float] = None
    custom_suggest_threshold: Optional[float] = None

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_active: Optional[datetime] = None
    request_count: int = 0
    blocked_count: int = 0

    # Status
    is_active: bool = True
    is_verified: bool = False              # Has owner verified this agent?

    @classmethod
    def generate_api_key(cls) -> tuple[str, str]:
        """
        Generate a new API key for an agent.
        Returns (plaintext_key, hashed_key).
        """
        # Generate a secure random key
        key = f"telos-agent-{secrets.token_urlsafe(32)}"
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        return key, key_hash

    @classmethod
    def hash_api_key(cls, key: str) -> str:
        """Hash an API key for storage."""
        return hashlib.sha256(key.encode()).hexdigest()

    def get_execute_threshold(self, default: float = 0.45) -> float:
        """Get the execute threshold for this agent."""
        return self.custom_execute_threshold or default

    def get_clarify_threshold(self, default: float = 0.35) -> float:
        """Get the clarify threshold for this agent."""
        return self.custom_clarify_threshold or default

    def get_tool_by_name(self, name: str) -> Optional[ToolDefinition]:
        """Find a tool by name."""
        for tool in self.authorized_tools:
            if tool.name == name:
                return tool
        return None

    def is_tool_authorized(self, tool_name: str) -> bool:
        """Check if a tool is in the manifest."""
        return self.get_tool_by_name(tool_name) is not None

    def get_unauthorized_tools(self, requested_tools: List[str]) -> List[str]:
        """Find tools that are NOT in the manifest."""
        authorized_names = {t.name for t in self.authorized_tools}
        return [t for t in requested_tools if t not in authorized_names]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/API response."""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "owner": self.owner,
            "purpose_statement": self.purpose_statement,
            "domain": self.domain,
            "domain_keywords": self.domain_keywords,
            "authorized_tools": [
                {
                    "name": t.name,
                    "description": t.description,
                    "risk_level": t.risk_level.value,
                    "requires_approval": t.requires_approval,
                }
                for t in self.authorized_tools
            ],
            "overall_risk_level": self.overall_risk_level.value,
            "is_active": self.is_active,
            "is_verified": self.is_verified,
            "created_at": self.created_at.isoformat(),
            "request_count": self.request_count,
            "blocked_count": self.blocked_count,
        }


@dataclass
class AgentRegistrationRequest:
    """Request to register a new agent."""
    name: str
    owner: str
    purpose_statement: str
    domain: str = "general"
    domain_keywords: List[str] = field(default_factory=list)
    domain_description: str = ""
    tools: List[Dict[str, Any]] = field(default_factory=list)
    risk_level: str = "medium"
    high_risk_mode: bool = False
    custom_thresholds: Optional[Dict[str, float]] = None


@dataclass
class AgentRegistrationResponse:
    """Response after registering an agent."""
    agent_id: str
    api_key: str  # Only returned once, at registration
    name: str
    purpose_statement: str
    message: str = "Agent registered successfully. Store your API key securely - it won't be shown again."
