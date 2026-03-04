"""
Agent Registration Routes
=========================

API endpoints for agent onboarding and management.
"""

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from telos_gateway.auth import lookup_agent, require_auth
from telos_gateway.registry import (
    AgentRegistrationRequest,
    AgentRegistry,
    RiskLevel,
    get_registry,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/agents", tags=["agents"])


# ============================================================================
# Pydantic request/response models
# ============================================================================


class ToolRegistration(BaseModel):
    """Tool definition for registration."""

    name: str = Field(..., description="Tool name (must match what agent sends)")
    description: str = Field(..., description="What this tool does")
    risk_level: str = Field("medium", description="low/medium/high/critical")
    requires_approval: bool = Field(False, description="Needs human approval?")
    min_fidelity_threshold: float = Field(0.45, description="Minimum fidelity to use")


class AgentRegisterRequest(BaseModel):
    """Request to register a new agent."""

    name: str = Field(..., description="Agent name", min_length=3)
    owner: str = Field(..., description="Organization/owner name")
    purpose_statement: str = Field(
        ...,
        description="What is this agent FOR? This becomes the PA.",
        min_length=20,
    )
    domain: str = Field("general", description="Domain: finance, healthcare, coding, etc.")
    domain_keywords: List[str] = Field(
        default_factory=list,
        description="Keywords relevant to this domain",
    )
    domain_description: str = Field(
        "",
        description="Rich description of the domain for better embeddings",
    )
    tools: List[ToolRegistration] = Field(
        default_factory=list,
        description="Tools this agent is authorized to use",
    )
    risk_level: str = Field("medium", description="Overall risk: low/medium/high/critical")
    high_risk_mode: bool = Field(
        False,
        description="If true, use ESCALATE instead of INERT for low fidelity",
    )
    custom_thresholds: Optional[dict] = Field(
        None,
        description="Custom thresholds: {execute: 0.45, clarify: 0.35, suggest: 0.25}",
    )


class AgentRegisterResponse(BaseModel):
    """Response after registering an agent."""

    agent_id: str
    api_key: str  # Only returned once!
    name: str
    purpose_statement: str
    message: str


class AgentProfileResponse(BaseModel):
    """Agent profile (without sensitive data)."""

    agent_id: str
    name: str
    owner: str
    purpose_statement: str
    domain: str
    domain_keywords: List[str]
    authorized_tools: List[dict]
    overall_risk_level: str
    is_active: bool
    is_verified: bool
    request_count: int
    blocked_count: int


def _profile_to_response(profile) -> AgentProfileResponse:
    """Convert an AgentProfile to an API response."""
    return AgentProfileResponse(
        agent_id=profile.agent_id,
        name=profile.name,
        owner=profile.owner,
        purpose_statement=profile.purpose_statement,
        domain=profile.domain,
        domain_keywords=profile.domain_keywords,
        authorized_tools=[
            {
                "name": t.name,
                "description": t.description,
                "risk_level": t.risk_level.value,
            }
            for t in profile.authorized_tools
        ],
        overall_risk_level=profile.overall_risk_level.value,
        is_active=profile.is_active,
        is_verified=profile.is_verified,
        request_count=profile.request_count,
        blocked_count=profile.blocked_count,
    )


# ============================================================================
# Routes
# ============================================================================


@router.post("", response_model=AgentRegisterResponse)
async def register_agent(request: AgentRegisterRequest):
    """
    Register a new agent with TELOS.

    IMPORTANT: Store the API key securely -- it won't be shown again!
    """
    registry = get_registry()

    internal_request = AgentRegistrationRequest(
        name=request.name,
        owner=request.owner,
        purpose_statement=request.purpose_statement,
        domain=request.domain,
        domain_keywords=request.domain_keywords,
        domain_description=request.domain_description,
        tools=[t.model_dump() for t in request.tools],
        risk_level=request.risk_level,
        high_risk_mode=request.high_risk_mode,
        custom_thresholds=request.custom_thresholds,
    )

    try:
        result = registry.register_agent(internal_request)
        logger.info(f"Agent registered: {result.name} ({result.agent_id})")
        return AgentRegisterResponse(
            agent_id=result.agent_id,
            api_key=result.api_key,
            name=result.name,
            purpose_statement=result.purpose_statement,
            message=(
                "Agent registered successfully. "
                "Store your API key securely - it won't be shown again. "
                "Use this key in the Authorization header: Bearer <api_key>"
            ),
        )
    except Exception as e:
        logger.error(f"Registration failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": "Registration failed due to an internal error",
                    "type": "server_error",
                    "code": 500,
                }
            },
        )


@router.get("/{agent_id}", response_model=AgentProfileResponse)
async def get_agent_profile(agent_id: str):
    """Get a specific agent's profile by ID."""
    registry = get_registry()
    profile = registry.get_agent_by_id(agent_id)

    if not profile:
        raise HTTPException(
            status_code=404,
            detail={
                "error": {
                    "message": "Agent not found",
                    "type": "not_found",
                    "code": 404,
                }
            },
        )

    return _profile_to_response(profile)


@router.delete("/{agent_id}")
async def deactivate_agent(agent_id: str, api_key: str = Depends(require_auth)):
    """Deactivate an agent (soft delete). Requires authentication."""
    # Verify the caller owns this agent or is admin
    agent = lookup_agent(api_key)
    if not agent or agent.agent_id != agent_id:
        raise HTTPException(
            status_code=403,
            detail={
                "error": {
                    "message": "Cannot deactivate another agent",
                    "type": "permission_error",
                    "code": 403,
                }
            },
        )

    registry = get_registry()
    if registry.deactivate_agent(agent_id):
        return {"message": f"Agent {agent_id} deactivated"}

    raise HTTPException(
        status_code=404,
        detail={
            "error": {
                "message": "Agent not found",
                "type": "not_found",
                "code": 404,
            }
        },
    )
