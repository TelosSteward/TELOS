"""
Base LLM Provider
=================

Abstract interface for LLM providers.
TELOS Gateway passes requests through to the real LLM provider
after governance checks.
"""

from abc import ABC, abstractmethod
from typing import AsyncIterator

from pydantic import BaseModel


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    TELOS Gateway is AGNOSTIC to which LLM the agent uses.
    We just pass through to wherever they're calling.
    """

    @abstractmethod
    async def chat_completion(
        self,
        request: BaseModel,
        api_key: str,
    ) -> BaseModel:
        """
        Send chat completion request to the real LLM.

        Args:
            request: The (possibly modified) chat completion request
            api_key: The customer's API key for the LLM provider

        Returns:
            The LLM's response
        """
        ...

    @abstractmethod
    async def chat_completion_stream(
        self,
        request: BaseModel,
        api_key: str,
    ) -> AsyncIterator[str]:
        """
        Stream chat completion response as SSE chunks.

        Args:
            request: The chat completion request
            api_key: The customer's API key for the LLM provider

        Yields:
            SSE-formatted data strings
        """
        ...

    @abstractmethod
    def get_provider_name(self) -> str:
        """Return the provider name (e.g., 'openai', 'mistral')."""
        ...
