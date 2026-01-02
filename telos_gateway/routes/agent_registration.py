"""
Agent Registration Routes
=========================

API endpoints for agent onboarding and management.
This is where agents establish their PA with TELOS.
"""

import logging
from typing import List, Optional
from pydantic import BaseModel, Field

from fastapi import APIRouter, HTTPException, Header

from ..registry import (
    AgentRegistry,
    get_registry,
    AgentProfile,
    AgentRegistrationRequest,
    RiskLevel,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/agents", tags=["agents"])


# ============================================================================
# Pydantic Models for API
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


# ============================================================================
# Routes
# ============================================================================

@router.post("/register", response_model=AgentRegisterResponse)
async def register_agent(request: AgentRegisterRequest):
    """
    Register a new agent with TELOS.

    This is the "PA establishment" moment. After registration:
    - Agent receives a unique API key
    - TELOS knows the agent's purpose, tools, and domain
    - Governance can use the registered PA instead of extracting from system prompt
    - Tool usage is audited against the manifest

    **IMPORTANT**: Store the API key securely - it won't be shown again!
    """
    registry = get_registry()

    # Convert to internal request format
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
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")


@router.get("/me", response_model=AgentProfileResponse)
async def get_my_profile(
    authorization: Optional[str] = Header(None),
):
    """
    Get the profile of the authenticated agent.

    Use your TELOS API key in the Authorization header.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid Authorization header",
        )

    api_key = authorization.replace("Bearer ", "").strip()

    # Check if this is a TELOS agent key
    if not api_key.startswith("telos-agent-"):
        raise HTTPException(
            status_code=401,
            detail="This endpoint requires a TELOS agent API key",
        )

    registry = get_registry()
    profile = registry.get_agent_by_api_key(api_key)

    if not profile:
        raise HTTPException(status_code=404, detail="Agent not found")

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


@router.get("/{agent_id}", response_model=AgentProfileResponse)
async def get_agent_profile(agent_id: str):
    """
    Get a specific agent's profile by ID.

    Note: This is a public endpoint showing non-sensitive info.
    """
    registry = get_registry()
    profile = registry.get_agent_by_id(agent_id)

    if not profile:
        raise HTTPException(status_code=404, detail="Agent not found")

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


@router.get("/", response_model=List[AgentProfileResponse])
async def list_agents(owner: Optional[str] = None):
    """
    List all registered agents.

    Optionally filter by owner.
    """
    registry = get_registry()
    agents = registry.list_agents(owner=owner)

    return [
        AgentProfileResponse(
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
        for profile in agents
    ]
