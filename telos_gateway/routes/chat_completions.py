"""
Chat Completions Route
======================

OpenAI-compatible /v1/chat/completions endpoint with TELOS governance.

This is where the magic happens:
1. Receive request from agent (OpenAI format)
2. Extract PA from system prompt
3. Check fidelity
4. Forward to real LLM (or block)
5. Return response (with optional governance metadata)
"""

import logging
import time
from typing import Optional

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from ..config import config
from ..models.openai_types import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChoice,
    ChatMessage,
)
from ..models.governance_types import (
    ActionDecision,
    GovernanceResult,
    GovernanceTrace,
)
from ..governance import PAExtractor, FidelityGate
from ..governance.pa_extractor import PrimacyAttractor
from ..providers import OpenAIProvider
from ..providers.openai_provider import MockProvider
from ..providers.mistral_provider import MistralProvider
from ..registry import get_registry

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize components (will be set up in main.py)
_pa_extractor: Optional[PAExtractor] = None
_fidelity_gate: Optional[FidelityGate] = None
_openai_provider: Optional[OpenAIProvider] = None
_mistral_provider: Optional[MistralProvider] = None
_embed_fn = None


def initialize_components(embed_fn):
    """Initialize governance components with embedding function."""
    global _pa_extractor, _fidelity_gate, _openai_provider, _mistral_provider, _embed_fn

    _embed_fn = embed_fn
    _pa_extractor = PAExtractor(embed_fn=embed_fn)
    _fidelity_gate = FidelityGate(embed_fn=embed_fn)
    _openai_provider = OpenAIProvider()
    _mistral_provider = MistralProvider()

    logger.info("Chat completions route initialized with TELOS governance (OpenAI + Mistral)")


@router.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    authorization: Optional[str] = Header(None),
    x_telos_high_risk: Optional[str] = Header(None, alias="X-TELOS-High-Risk"),
    x_telos_bypass: Optional[str] = Header(None, alias="X-TELOS-Bypass"),
):
    """
    OpenAI-compatible chat completions with TELOS governance.

    Headers:
        Authorization: Bearer <openai_api_key>
        X-TELOS-High-Risk: "true" to enable ESCALATE instead of INERT
        X-TELOS-Bypass: "true" to bypass governance (for testing)
    """
    start_time = time.time()

    # Extract API key from Authorization header
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid Authorization header. Use: Bearer <api_key>",
        )

    api_key = authorization.replace("Bearer ", "").strip()

    # Check if governance should be bypassed (testing only)
    if x_telos_bypass == "true":
        logger.warning("TELOS governance bypassed via header")
        response = await _provider.chat_completion(request, api_key)
        return response

    # Check if components are initialized
    if not _pa_extractor or not _fidelity_gate:
        raise HTTPException(
            status_code=500,
            detail="TELOS Gateway not initialized. Check server logs.",
        )

    try:
        # =====================================================================
        # STEP 0: Check if this is a registered TELOS agent
        # =====================================================================
        registry = get_registry()
        agent_profile = None
        llm_api_key = api_key  # Default: use the provided key for LLM

        # Check if this is a TELOS agent key
        if api_key.startswith("telos-agent-"):
            agent_profile = registry.get_agent_by_api_key(api_key)
            if agent_profile:
                logger.info(f"Registered agent: {agent_profile.name} ({agent_profile.agent_id})")
                # For registered agents, we'll need an LLM key from config or header
                # For now, use the Mistral key from environment
                import os
                llm_api_key = os.getenv("MISTRAL_API_KEY", "")
            else:
                raise HTTPException(
                    status_code=401,
                    detail="Invalid TELOS agent API key",
                )

        # =====================================================================
        # STEP 1: Get Primacy Attractor (registered PA or extracted from prompt)
        # =====================================================================
        if agent_profile and agent_profile.purpose_embedding is not None:
            # Use registered PA - this is the "precognition" we have about this agent
            pa = PrimacyAttractor(
                text=agent_profile.purpose_statement,
                embedding=agent_profile.purpose_embedding,
                source="registry",
            )
            logger.info(f"Using registered PA for {agent_profile.name}: {pa.text[:50]}...")
        else:
            # Fall back to extracting from system prompt
            pa = _pa_extractor.extract(request, api_key=api_key)
            logger.info(f"PA extracted from {pa.source}: {pa.text[:50]}...")

        # =====================================================================
        # STEP 1.5: Tool Manifest Audit (for registered agents)
        # =====================================================================
        unauthorized_tools = []
        if agent_profile and (request.tools or request.functions):
            # Extract tool names from request
            requested_tools = []
            if request.tools:
                requested_tools.extend([t.function.name for t in request.tools])
            if request.functions:
                requested_tools.extend([f.name for f in request.functions])

            # Check against agent's manifest
            unauthorized_tools = agent_profile.get_unauthorized_tools(requested_tools)

            if unauthorized_tools:
                logger.warning(
                    f"Agent {agent_profile.name} requesting unauthorized tools: {unauthorized_tools}"
                )
                # Don't block - just log and track. Fidelity gate will handle blocking.
                # But we could choose to hard block here for strict manifest enforcement.

        # =====================================================================
        # STEP 2: Check fidelity (governance decision)
        # =====================================================================
        # Use agent's high_risk_mode if registered, else check header
        high_risk = (
            agent_profile.high_risk_mode if agent_profile
            else x_telos_high_risk == "true"
        )
        governance_result = _fidelity_gate.check_request(request, pa, high_risk=high_risk)

        logger.info(
            f"Governance: decision={governance_result.final_decision.value}, "
            f"fidelity={governance_result.input_fidelity:.3f}, "
            f"forwarded={governance_result.forwarded_to_llm}"
        )

        # =====================================================================
        # STEP 3: Handle governance decision
        # =====================================================================
        if not governance_result.forwarded_to_llm:
            # Update agent stats for blocked request
            if agent_profile:
                registry.update_agent_stats(agent_profile.agent_id, was_blocked=True)

            # Request blocked - return governance response
            response = _create_governance_response(
                request=request,
                governance_result=governance_result,
                latency_ms=int((time.time() - start_time) * 1000),
            )

            # Add agent info to blocked response
            if agent_profile:
                response.telos_governance["agent"] = {
                    "agent_id": agent_profile.agent_id,
                    "name": agent_profile.name,
                    "domain": agent_profile.domain,
                }
                response.telos_governance["pa_source"] = "registry"

            return response

        # =====================================================================
        # STEP 4: Apply graduated intervention based on decision
        # =====================================================================
        modified_request = request

        # Filter blocked tools if any
        if governance_result.tools_blocked > 0:
            modified_request = _filter_blocked_tools(request, governance_result)

        # Apply governance intervention for CLARIFY and SUGGEST
        if governance_result.final_decision == ActionDecision.CLARIFY:
            modified_request = _inject_clarify_context(modified_request, pa, governance_result)
            logger.info("Applied CLARIFY intervention - requesting intent verification")
        elif governance_result.final_decision == ActionDecision.SUGGEST:
            modified_request = _inject_suggest_context(modified_request, pa, governance_result)
            logger.info("Applied SUGGEST intervention - steering toward purpose")

        # =====================================================================
        # STEP 5: Forward to real LLM (select provider based on model)
        # =====================================================================

        # Select provider based on model name
        model = modified_request.model.lower()
        if "mistral" in model or "mixtral" in model:
            provider = _mistral_provider
            logger.info(f"Using Mistral provider for model: {model}")
        else:
            provider = _openai_provider
            logger.info(f"Using OpenAI provider for model: {model}")

        response = await provider.chat_completion(modified_request, llm_api_key)

        # =====================================================================
        # STEP 5: Add governance metadata to response
        # =====================================================================
        response.telos_governance = {
            "decision": governance_result.final_decision.value,
            "input_fidelity": governance_result.input_fidelity,
            "pa_source": governance_result.pa_source,
            "tools_checked": governance_result.tools_checked,
            "tools_blocked": governance_result.tools_blocked,
            "latency_ms": int((time.time() - start_time) * 1000),
        }

        # Add agent info if registered
        if agent_profile:
            response.telos_governance["agent"] = {
                "agent_id": agent_profile.agent_id,
                "name": agent_profile.name,
                "domain": agent_profile.domain,
            }
            # Add manifest audit results
            if unauthorized_tools:
                response.telos_governance["manifest_audit"] = {
                    "unauthorized_tools": unauthorized_tools,
                    "warning": "Agent requested tools not in registered manifest",
                }

        # =====================================================================
        # STEP 6: Update agent stats (if registered)
        # =====================================================================
        if agent_profile:
            registry.update_agent_stats(
                agent_profile.agent_id,
                was_blocked=not governance_result.forwarded_to_llm,
            )

        # =====================================================================
        # STEP 7: Log governance trace
        # =====================================================================
        if config.log_governance_traces:
            _log_governance_trace(request, governance_result, response, start_time)

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat_completions: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"TELOS Gateway error: {str(e)}",
        )


def _create_governance_response(
    request: ChatCompletionRequest,
    governance_result: GovernanceResult,
    latency_ms: int,
) -> ChatCompletionResponse:
    """Create a response when request is blocked by governance."""
    import time

    return ChatCompletionResponse(
        id=f"telos-gov-{governance_result.request_id}",
        object="chat.completion",
        created=int(time.time()),
        model=request.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(
                    role="assistant",
                    content=governance_result.governance_response,
                ),
                finish_reason="stop",
            )
        ],
        telos_governance={
            "decision": governance_result.final_decision.value,
            "input_fidelity": governance_result.input_fidelity,
            "blocked": True,
            "reason": f"Request blocked due to {governance_result.final_decision.value}",
            "latency_ms": latency_ms,
        },
    )


def _filter_blocked_tools(
    request: ChatCompletionRequest,
    governance_result: GovernanceResult,
) -> ChatCompletionRequest:
    """Remove blocked tools from the request."""
    if not governance_result.tool_decisions:
        return request

    # Get list of allowed tools
    allowed_tools = [
        name for name, decision in governance_result.tool_decisions.items()
        if decision.should_forward
    ]

    # Create modified request
    modified = request.model_copy()

    if modified.tools:
        modified.tools = [
            tool for tool in modified.tools
            if tool.function.name in allowed_tools
        ]

    if modified.functions:
        modified.functions = [
            func for func in modified.functions
            if func.name in allowed_tools
        ]

    logger.info(f"Filtered tools: {len(request.tools or [])} -> {len(modified.tools or [])}")

    return modified


def _log_governance_trace(
    request: ChatCompletionRequest,
    governance_result: GovernanceResult,
    response: ChatCompletionResponse,
    start_time: float,
):
    """Log governance trace to file."""
    import hashlib

    trace = GovernanceTrace(
        event_type="chat_completion",
        request_id=governance_result.request_id,
        model=request.model,
        message_count=len(request.messages),
        has_tools=bool(request.tools or request.functions),
        has_system_prompt=governance_result.pa_source == "system_prompt",
        pa_hash=hashlib.sha256(governance_result.pa_text.encode()).hexdigest()[:16],
        input_fidelity=governance_result.input_fidelity,
        decision=governance_result.final_decision,
        forwarded=governance_result.forwarded_to_llm,
        tools_total=governance_result.tools_checked,
        tools_allowed=governance_result.tools_checked - governance_result.tools_blocked,
        tools_blocked=governance_result.tools_blocked,
        response_tokens=response.usage.completion_tokens if response.usage else None,
        governance_latency_ms=int((time.time() - start_time) * 1000),
        total_latency_ms=int((time.time() - start_time) * 1000),
    )

    # Write to trace file
    trace_file = config.trace_storage_dir / f"gateway_traces.jsonl"
    try:
        with open(trace_file, "a") as f:
            f.write(trace.to_jsonl() + "\n")
    except Exception as e:
        logger.warning(f"Failed to write governance trace: {e}")


def _inject_clarify_context(
    request: ChatCompletionRequest,
    pa: PrimacyAttractor,
    governance_result: GovernanceResult,
) -> ChatCompletionRequest:
    """
    Inject CLARIFY governance context into the request.

    CLARIFY means the request is borderline - we forward but ask
    the LLM to verify intent before proceeding.
    """
    modified = request.model_copy()

    # Create governance context message
    clarify_context = (
        f"[GOVERNANCE NOTICE - CLARIFY MODE]\n"
        f"The user's request has moderate alignment (fidelity: {governance_result.input_fidelity:.2f}) "
        f"with your defined purpose: \"{pa.text[:100]}...\"\n\n"
        f"Before responding fully, please:\n"
        f"1. Acknowledge that you noticed potential drift from your core purpose\n"
        f"2. Ask the user to clarify how their request relates to your defined function\n"
        f"3. If they confirm, proceed helpfully; if not, gently redirect\n"
        f"[END GOVERNANCE NOTICE]"
    )

    # Inject as system message if there's a system prompt, else add one
    new_messages = list(modified.messages)
    if new_messages and new_messages[0].role == "system":
        # Append to existing system message
        new_messages[0] = ChatMessage(
            role="system",
            content=f"{new_messages[0].content}\n\n{clarify_context}",
        )
    else:
        # Insert new system message
        new_messages.insert(0, ChatMessage(role="system", content=clarify_context))

    modified.messages = new_messages
    return modified


def _inject_suggest_context(
    request: ChatCompletionRequest,
    pa: PrimacyAttractor,
    governance_result: GovernanceResult,
) -> ChatCompletionRequest:
    """
    Inject SUGGEST governance context into the request.

    SUGGEST means the request has low alignment - we forward but
    steer the LLM to offer alternatives aligned with the purpose.
    """
    modified = request.model_copy()

    # Create governance context message with stronger steering
    suggest_context = (
        f"[GOVERNANCE NOTICE - SUGGEST MODE]\n"
        f"The user's request has low alignment (fidelity: {governance_result.input_fidelity:.2f}) "
        f"with your defined purpose: \"{pa.text[:100]}...\"\n\n"
        f"IMPORTANT: You should:\n"
        f"1. Acknowledge the user's request politely\n"
        f"2. Explain that it falls outside your core expertise\n"
        f"3. Suggest 2-3 alternative ways you CAN help that align with your purpose\n"
        f"4. Offer to redirect the conversation toward your areas of strength\n"
        f"5. Only proceed with the original request if the user insists\n"
        f"[END GOVERNANCE NOTICE]"
    )

    # Inject as system message
    new_messages = list(modified.messages)
    if new_messages and new_messages[0].role == "system":
        new_messages[0] = ChatMessage(
            role="system",
            content=f"{new_messages[0].content}\n\n{suggest_context}",
        )
    else:
        new_messages.insert(0, ChatMessage(role="system", content=suggest_context))

    modified.messages = new_messages
    return modified
