"""
OpenAI Provider
===============

Passthrough provider for OpenAI API.

The agent's requests go to OpenAI - we just govern them first.
"""

import logging
from typing import Any, Dict, Optional
import httpx

from ..config import config
from ..models.openai_types import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChoice,
    ChatMessage,
    UsageInfo,
)
from .base import LLMProvider

logger = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    """
    Passthrough to OpenAI API.

    The agent keeps using OpenAI - we don't replace it.
    We just measure fidelity and pass through (or block).
    """

    def __init__(
        self,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 60.0,
    ):
        """
        Initialize OpenAI provider.

        Args:
            base_url: OpenAI API base URL
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def get_provider_name(self) -> str:
        return "openai"

    async def chat_completion(
        self,
        request: ChatCompletionRequest,
        api_key: str,
    ) -> ChatCompletionResponse:
        """
        Forward chat completion to OpenAI.

        Args:
            request: The chat completion request
            api_key: Customer's OpenAI API key

        Returns:
            OpenAI's response (unmodified)
        """
        url = f"{self.base_url}/chat/completions"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        # Convert request to dict, excluding None values
        request_data = request.model_dump(exclude_none=True)

        # Force non-streaming for MVP
        request_data["stream"] = False

        logger.debug(f"Forwarding to OpenAI: model={request.model}")

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(
                    url,
                    headers=headers,
                    json=request_data,
                )
                response.raise_for_status()

                data = response.json()

                # Parse response into our model
                return ChatCompletionResponse(
                    id=data.get("id", "unknown"),
                    object=data.get("object", "chat.completion"),
                    created=data.get("created", 0),
                    model=data.get("model", request.model),
                    choices=[
                        ChatCompletionChoice(
                            index=choice.get("index", i),
                            message=ChatMessage(
                                role=choice["message"]["role"],
                                content=choice["message"].get("content"),
                                tool_calls=choice["message"].get("tool_calls"),
                                function_call=choice["message"].get("function_call"),
                            ),
                            finish_reason=choice.get("finish_reason"),
                        )
                        for i, choice in enumerate(data.get("choices", []))
                    ],
                    usage=UsageInfo(**data["usage"]) if data.get("usage") else None,
                    system_fingerprint=data.get("system_fingerprint"),
                )

            except httpx.HTTPStatusError as e:
                logger.error(f"OpenAI API error: {e.response.status_code} - {e.response.text}")
                raise
            except Exception as e:
                logger.error(f"Error calling OpenAI: {e}")
                raise


class MockProvider(LLMProvider):
    """
    Mock provider for testing without real API calls.
    """

    def get_provider_name(self) -> str:
        return "mock"

    async def chat_completion(
        self,
        request: ChatCompletionRequest,
        api_key: str,
    ) -> ChatCompletionResponse:
        """Return a mock response."""
        import time

        return ChatCompletionResponse(
            id="mock-" + str(int(time.time())),
            object="chat.completion",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content="This is a mock response from TELOS Gateway testing.",
                    ),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(
                prompt_tokens=10,
                completion_tokens=10,
                total_tokens=20,
            ),
        )
