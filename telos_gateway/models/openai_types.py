"""
OpenAI-Compatible Types
=======================

Pydantic models that match the OpenAI API schema.
This allows the gateway to accept requests in the exact
format that agents already use.

Reference: https://platform.openai.com/docs/api-reference/chat
"""

from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field
from datetime import datetime


# =============================================================================
# MESSAGE TYPES
# =============================================================================

class FunctionCall(BaseModel):
    """Function call in assistant message."""
    name: str
    arguments: str  # JSON string


class ToolCall(BaseModel):
    """Tool call in assistant message."""
    id: str
    type: Literal["function"] = "function"
    function: FunctionCall


class ChatMessage(BaseModel):
    """
    A message in the conversation.

    Supports all OpenAI message types:
    - system: Sets the behavior/purpose of the assistant
    - user: User input
    - assistant: Model response (may include tool_calls)
    - tool: Result of a tool call
    """
    role: Literal["system", "user", "assistant", "tool"]
    content: Optional[str] = None
    name: Optional[str] = None  # For tool messages

    # Assistant-specific fields
    tool_calls: Optional[List[ToolCall]] = None
    function_call: Optional[FunctionCall] = None  # Deprecated but still used

    # Tool message fields
    tool_call_id: Optional[str] = None


# =============================================================================
# TOOL/FUNCTION DEFINITIONS
# =============================================================================

class Function(BaseModel):
    """Function definition for tool use."""
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


class Tool(BaseModel):
    """Tool definition."""
    type: Literal["function"] = "function"
    function: Function


# =============================================================================
# REQUEST TYPES
# =============================================================================

class ChatCompletionRequest(BaseModel):
    """
    OpenAI-compatible chat completion request.

    This is what agents send to the LLM.
    """
    model: str
    messages: List[ChatMessage]

    # Optional parameters
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None

    # Tool/function calling
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None

    # Deprecated but still used
    functions: Optional[List[Function]] = None
    function_call: Optional[Union[str, Dict[str, str]]] = None

    # Response format
    response_format: Optional[Dict[str, str]] = None

    # Seed for reproducibility
    seed: Optional[int] = None


# =============================================================================
# RESPONSE TYPES
# =============================================================================

class UsageInfo(BaseModel):
    """Token usage information."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionChoice(BaseModel):
    """A single completion choice."""
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None
    logprobs: Optional[Any] = None


class ChatCompletionResponse(BaseModel):
    """
    OpenAI-compatible chat completion response.

    This is what we return to agents.
    """
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Optional[UsageInfo] = None
    system_fingerprint: Optional[str] = None

    # TELOS extension: governance metadata (optional, for transparency)
    telos_governance: Optional[Dict[str, Any]] = Field(
        default=None,
        description="TELOS governance metadata (fidelity, decision, etc.)"
    )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def extract_system_prompt(messages: List[ChatMessage]) -> Optional[str]:
    """Extract the system prompt from messages."""
    for msg in messages:
        if msg.role == "system" and msg.content:
            return msg.content
    return None


def extract_last_user_message(messages: List[ChatMessage]) -> Optional[str]:
    """Extract the last user message from messages."""
    for msg in reversed(messages):
        if msg.role == "user" and msg.content:
            return msg.content
    return None


def extract_tool_names(request: ChatCompletionRequest) -> List[str]:
    """Extract tool names from request."""
    tool_names = []

    if request.tools:
        for tool in request.tools:
            tool_names.append(tool.function.name)

    if request.functions:
        for func in request.functions:
            tool_names.append(func.name)

    return tool_names
