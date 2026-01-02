"""
Agent Registry Service
======================

Manages agent profiles - registration, lookup, validation.
This is the "context enrichment" layer for agentic governance.

Before governance happens, we look up:
- WHO is this agent?
- WHAT is their purpose (PA)?
- WHAT tools are they authorized to use?
- WHAT is their risk level?
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, List, Callable
from datetime import datetime
import uuid

from .agent_profile import (
    AgentProfile,
    ToolDefinition,
    RiskLevel,
    AgentRegistrationRequest,
    AgentRegistrationResponse,
)

logger = logging.getLogger(__name__)

# Global registry instance
_registry: Optional["AgentRegistry"] = None


def get_registry() -> "AgentRegistry":
    """Get the global registry instance."""
    global _registry
    if _registry is None:
        _registry = AgentRegistry()
    return _registry


class AgentRegistry:
    """
    Registry for managing agent profiles.

    In production, this would be backed by a database.
    For MVP, we use a JSON file for persistence.
    """

    def __init__(
        self,
        storage_path: Path = None,
        embed_fn: Callable = None,
    ):
        self.storage_path = storage_path or Path("./telos_gateway_registry")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.embed_fn = embed_fn

        # In-memory cache of agents (keyed by api_key_hash for fast lookup)
        self._agents_by_key: Dict[str, AgentProfile] = {}
        self._agents_by_id: Dict[str, AgentProfile] = {}

        # Load existing agents
        self._load_agents()

        logger.info(f"Agent Registry initialized with {len(self._agents_by_id)} agents")

    def set_embed_fn(self, embed_fn: Callable):
        """Set the embedding function for computing PA embeddings."""
        self.embed_fn = embed_fn
        # Recompute embeddings for agents loaded from storage
        self._recompute_embeddings()

    def _load_agents(self):
        """Load agents from storage."""
        agents_file = self.storage_path / "agents.json"
        if agents_file.exists():
            try:
                with open(agents_file, "r") as f:
                    data = json.load(f)
                    for agent_data in data.get("agents", []):
                        profile = self._dict_to_profile(agent_data)
                        self._agents_by_id[profile.agent_id] = profile
                        self._agents_by_key[profile.api_key_hash] = profile
                logger.info(f"Loaded {len(self._agents_by_id)} agents from storage")
            except Exception as e:
                logger.error(f"Failed to load agents: {e}")

    def _recompute_embeddings(self):
        """Recompute purpose embeddings for all loaded agents."""
        if not self.embed_fn:
            logger.warning("No embedding function set, cannot compute purpose embeddings")
            return

        for agent_id, profile in self._agents_by_id.items():
            if profile.purpose_embedding is None:
                try:
                    profile.purpose_embedding = self.embed_fn(profile.purpose_statement)
                    logger.info(f"Computed purpose embedding for agent {agent_id}")
                except Exception as e:
                    logger.warning(f"Failed to compute embedding for {agent_id}: {e}")

    def _save_agents(self):
        """Save agents to storage."""
        agents_file = self.storage_path / "agents.json"
        try:
            data = {
                "agents": [
                    self._profile_to_dict(p) for p in self._agents_by_id.values()
                ]
            }
            with open(agents_file, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save agents: {e}")

    def _profile_to_dict(self, profile: AgentProfile) -> dict:
        """Convert profile to storable dict."""
        return {
            "agent_id": profile.agent_id,
            "name": profile.name,
            "owner": profile.owner,
            "api_key_hash": profile.api_key_hash,
            "purpose_statement": profile.purpose_statement,
            "domain": profile.domain,
            "domain_keywords": profile.domain_keywords,
            "domain_description": profile.domain_description,
            "authorized_tools": [
                {
                    "name": t.name,
                    "description": t.description,
                    "risk_level": t.risk_level.value,
                    "requires_approval": t.requires_approval,
                    "min_fidelity_threshold": t.min_fidelity_threshold,
                }
                for t in profile.authorized_tools
            ],
            "overall_risk_level": profile.overall_risk_level.value,
            "high_risk_mode": profile.high_risk_mode,
            "custom_execute_threshold": profile.custom_execute_threshold,
            "custom_clarify_threshold": profile.custom_clarify_threshold,
            "custom_suggest_threshold": profile.custom_suggest_threshold,
            "created_at": profile.created_at.isoformat(),
            "is_active": profile.is_active,
            "is_verified": profile.is_verified,
            "request_count": profile.request_count,
            "blocked_count": profile.blocked_count,
        }

    def _dict_to_profile(self, data: dict) -> AgentProfile:
        """Convert stored dict to profile."""
        tools = [
            ToolDefinition(
                name=t["name"],
                description=t["description"],
                risk_level=RiskLevel(t.get("risk_level", "medium")),
                requires_approval=t.get("requires_approval", False),
                min_fidelity_threshold=t.get("min_fidelity_threshold", 0.45),
            )
            for t in data.get("authorized_tools", [])
        ]

        return AgentProfile(
            agent_id=data["agent_id"],
            name=data["name"],
            owner=data["owner"],
            api_key_hash=data["api_key_hash"],
            purpose_statement=data["purpose_statement"],
            domain=data.get("domain", "general"),
            domain_keywords=data.get("domain_keywords", []),
            domain_description=data.get("domain_description", ""),
            authorized_tools=tools,
            overall_risk_level=RiskLevel(data.get("overall_risk_level", "medium")),
            high_risk_mode=data.get("high_risk_mode", False),
            custom_execute_threshold=data.get("custom_execute_threshold"),
            custom_clarify_threshold=data.get("custom_clarify_threshold"),
            custom_suggest_threshold=data.get("custom_suggest_threshold"),
            created_at=datetime.fromisoformat(data["created_at"]),
            is_active=data.get("is_active", True),
            is_verified=data.get("is_verified", False),
            request_count=data.get("request_count", 0),
            blocked_count=data.get("blocked_count", 0),
        )

    def register_agent(
        self,
        request: AgentRegistrationRequest,
    ) -> AgentRegistrationResponse:
        """
        Register a new agent with TELOS.

        This is the "PA establishment" moment for the agent.
        After registration, TELOS knows what to expect from this agent.
        """
        # Generate unique ID and API key
        agent_id = f"agent-{uuid.uuid4().hex[:12]}"
        api_key, api_key_hash = AgentProfile.generate_api_key()

        # Parse tools
        tools = [
            ToolDefinition(
                name=t.get("name", ""),
                description=t.get("description", ""),
                risk_level=RiskLevel(t.get("risk_level", "medium")),
                requires_approval=t.get("requires_approval", False),
                min_fidelity_threshold=t.get("min_fidelity_threshold", 0.45),
            )
            for t in request.tools
        ]

        # Parse custom thresholds
        custom_execute = None
        custom_clarify = None
        custom_suggest = None
        if request.custom_thresholds:
            custom_execute = request.custom_thresholds.get("execute")
            custom_clarify = request.custom_thresholds.get("clarify")
            custom_suggest = request.custom_thresholds.get("suggest")

        # Create profile
        profile = AgentProfile(
            agent_id=agent_id,
            name=request.name,
            owner=request.owner,
            api_key_hash=api_key_hash,
            purpose_statement=request.purpose_statement,
            domain=request.domain,
            domain_keywords=request.domain_keywords,
            domain_description=request.domain_description,
            authorized_tools=tools,
            overall_risk_level=RiskLevel(request.risk_level),
            high_risk_mode=request.high_risk_mode,
            custom_execute_threshold=custom_execute,
            custom_clarify_threshold=custom_clarify,
            custom_suggest_threshold=custom_suggest,
        )

        # Compute purpose embedding if we have an embed function
        if self.embed_fn:
            try:
                profile.purpose_embedding = self.embed_fn(request.purpose_statement)
                logger.info(f"Computed purpose embedding for agent {agent_id}")
            except Exception as e:
                logger.warning(f"Failed to compute embedding: {e}")

        # Store
        self._agents_by_id[agent_id] = profile
        self._agents_by_key[api_key_hash] = profile
        self._save_agents()

        logger.info(
            f"Registered agent: {request.name} ({agent_id}) "
            f"with {len(tools)} tools, domain={request.domain}"
        )

        return AgentRegistrationResponse(
            agent_id=agent_id,
            api_key=api_key,
            name=request.name,
            purpose_statement=request.purpose_statement,
        )

    def get_agent_by_api_key(self, api_key: str) -> Optional[AgentProfile]:
        """
        Look up an agent by their API key.

        This is called on every request to identify the agent.
        """
        key_hash = AgentProfile.hash_api_key(api_key)
        profile = self._agents_by_key.get(key_hash)

        if profile and profile.is_active:
            # Update last active
            profile.last_active = datetime.utcnow()
            return profile

        return None

    def get_agent_by_id(self, agent_id: str) -> Optional[AgentProfile]:
        """Look up an agent by their ID."""
        return self._agents_by_id.get(agent_id)

    def update_agent_stats(
        self,
        agent_id: str,
        was_blocked: bool = False,
    ):
        """Update agent statistics after a request."""
        profile = self._agents_by_id.get(agent_id)
        if profile:
            profile.request_count += 1
            if was_blocked:
                profile.blocked_count += 1
            self._save_agents()

    def list_agents(self, owner: Optional[str] = None) -> List[AgentProfile]:
        """List all agents, optionally filtered by owner."""
        agents = list(self._agents_by_id.values())
        if owner:
            agents = [a for a in agents if a.owner == owner]
        return agents

    def deactivate_agent(self, agent_id: str) -> bool:
        """Deactivate an agent (soft delete)."""
        profile = self._agents_by_id.get(agent_id)
        if profile:
            profile.is_active = False
            self._save_agents()
            logger.info(f"Deactivated agent: {agent_id}")
            return True
        return False

    def audit_tool_usage(
        self,
        agent_id: str,
        requested_tools: List[str],
    ) -> Dict[str, bool]:
        """
        Audit which tools are authorized for this agent.

        Returns a dict of {tool_name: is_authorized}.
        """
        profile = self._agents_by_id.get(agent_id)
        if not profile:
            return {t: False for t in requested_tools}

        return {
            tool: profile.is_tool_authorized(tool)
            for tool in requested_tools
        }
