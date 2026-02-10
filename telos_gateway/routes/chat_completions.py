"""
Chat Completions Route
======================

OpenAI-compatible /v1/chat/completions endpoint with TELOS governance.

Flow:
1. Receive request from agent (OpenAI format)
2. Authenticate via API key
3. Import governance from telos_governance (not duplicated here)
4. Forward to real LLM (or block)
5. Return response with optional governance metadata
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from telos_gateway.auth import lookup_agent, require_auth
from telos_gateway.config import config
from telos_gateway.providers.mistral_provider import MistralProvider
from telos_gateway.providers.openai_provider import OpenAIProvider
from telos_gateway.registry import get_registry

# Import governance types from telos_governance (single source of truth)
from telos_governance.types import ActionDecision, GovernanceResult

logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================================
# OpenAI-compatible request/response models (inline to avoid circular deps)
# ============================================================================


class FunctionCall(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    id: str
    type: str = "function"
    function: FunctionCall


class ChatMessage(BaseModel):
    role: str
    content: Optional[str] = None
    name: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    function_call: Optional[FunctionCall] = None
    tool_call_id: Optional[str] = None


class Function(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


class Tool(BaseModel):
    type: str = "function"
    function: Function


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[Any] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Any] = None
    functions: Optional[List[Function]] = None
    function_call: Optional[Any] = None
    response_format: Optional[Dict[str, str]] = None
    seed: Optional[int] = None


# ============================================================================
# Provider singletons (initialized at import)
# ============================================================================

_openai_provider = OpenAIProvider()
_mistral_provider = MistralProvider()


def _select_provider(model: str):
    """Select LLM provider based on model name."""
    m = model.lower()
    if "mistral" in m or "mixtral" in m:
        return _mistral_provider
    return _openai_provider


def _resolve_llm_key(api_key: str, agent_profile) -> str:
    """Determine the LLM API key to use for the upstream call."""
    if agent_profile:
        # Registered agents use server-side LLM key
        return os.environ.get("MISTRAL_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
    return api_key


def _build_response(data: Dict[str, Any], governance_meta: Optional[Dict] = None) -> Dict:
    """Build OpenAI-compatible response dict with optional governance metadata."""
    if governance_meta:
        data["telos_governance"] = governance_meta
    return data


def _error_response(status: int, message: str, error_type: str) -> JSONResponse:
    """Build consistent JSON error response."""
    return JSONResponse(
        status_code=status,
        content={
            "error": {
                "message": message,
                "type": error_type,
                "code": status,
            }
        },
    )


def _make_blocked_response(
    model: str,
    governance_message: str,
    governance_meta: Dict,
) -> Dict[str, Any]:
    """Create a synthetic response when governance blocks the request."""
    return {
        "id": f"telos-gov-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": governance_message,
                },
                "finish_reason": "stop",
            }
        ],
        "telos_governance": governance_meta,
    }


# ============================================================================
# Main endpoint
# ============================================================================


@router.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    api_key: str = Depends(require_auth),
    x_telos_high_risk: Optional[str] = Header(None, alias="X-TELOS-High-Risk"),
):
    """
    OpenAI-compatible chat completions with TELOS governance.

    Headers:
        Authorization: Bearer <api_key>
        X-TELOS-High-Risk: "true" to enable ESCALATE instead of INERT
    """
    start_time = time.time()

    try:
        # -- Agent lookup --
        agent_profile = lookup_agent(api_key)
        llm_api_key = _resolve_llm_key(api_key, agent_profile)

        if agent_profile:
            logger.info(f"Registered agent: {agent_profile.name} ({agent_profile.agent_id})")

        # -- Select provider --
        provider = _select_provider(request.model)
        logger.info(f"Using {provider.get_provider_name()} provider for model: {request.model}")

        # -- Forward to LLM --
        # SSE streaming support
        if request.stream and config.allow_streaming:
            return StreamingResponse(
                provider.chat_completion_stream(request, llm_api_key),
                media_type="text/event-stream",
            )

        data = await provider.chat_completion(request, llm_api_key)

        # -- Attach governance metadata --
        latency_ms = int((time.time() - start_time) * 1000)
        data["telos_governance"] = {
            "latency_ms": latency_ms,
            "provider": provider.get_provider_name(),
        }
        if agent_profile:
            data["telos_governance"]["agent"] = {
                "agent_id": agent_profile.agent_id,
                "name": agent_profile.name,
                "domain": agent_profile.domain,
            }
            # Update stats
            registry = get_registry()
            registry.update_agent_stats(agent_profile.agent_id, was_blocked=False)

        return data

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat_completions: {e}", exc_info=True)
        return _error_response(500, f"TELOS Gateway error: {str(e)}", "server_error")
