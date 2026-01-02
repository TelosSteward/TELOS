"""
Mistral Provider
================

Provider for Mistral AI API - compatible with OpenAI format.
"""

import logging
from typing import Any, Dict, Optional
import httpx

from ..models.openai_types import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChoice,
    ChatMessage,
    UsageInfo,
)
from .base import LLMProvider

logger = logging.getLogger(__name__)


class MistralProvider(LLMProvider):
    """
    Provider for Mistral AI API.

    Mistral uses an OpenAI-compatible format, so this is straightforward.
    """

    def __init__(
        self,
        base_url: str = "https://api.mistral.ai/v1",
        timeout: float = 60.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def get_provider_name(self) -> str:
        return "mistral"

    async def chat_completion(
        self,
        request: ChatCompletionRequest,
        api_key: str,
    ) -> ChatCompletionResponse:
        """
        Forward chat completion to Mistral.
        """
        url = f"{self.base_url}/chat/completions"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        # Convert request to dict
        request_data = request.model_dump(exclude_none=True)

        # Map model names if needed
        model = request_data.get("model", "mistral-small-latest")
        if model.startswith("gpt"):
            # Map OpenAI models to Mistral equivalents
            model = "mistral-small-latest"
        request_data["model"] = model

        # Force non-streaming
        request_data["stream"] = False

        logger.debug(f"Forwarding to Mistral: model={model}")

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(
                    url,
                    headers=headers,
                    json=request_data,
                )
                response.raise_for_status()

                data = response.json()

                return ChatCompletionResponse(
                    id=data.get("id", "unknown"),
                    object=data.get("object", "chat.completion"),
                    created=data.get("created", 0),
                    model=data.get("model", model),
                    choices=[
                        ChatCompletionChoice(
                            index=choice.get("index", i),
                            message=ChatMessage(
                                role=choice["message"]["role"],
                                content=choice["message"].get("content"),
                                tool_calls=choice["message"].get("tool_calls"),
                            ),
                            finish_reason=choice.get("finish_reason"),
                        )
                        for i, choice in enumerate(data.get("choices", []))
                    ],
                    usage=UsageInfo(**data["usage"]) if data.get("usage") else None,
                )

            except httpx.HTTPStatusError as e:
                logger.error(f"Mistral API error: {e.response.status_code} - {e.response.text}")
                raise
            except Exception as e:
                logger.error(f"Error calling Mistral: {e}")
                raise
