"""
TELOS Mistral Client

Simple, working Mistral API client for TELOS.
Compatible with mistralai >= 1.0.0
"""

import os
import logging
from typing import List, Dict, Any, Optional

from telos_purpose.exceptions import (
    MissingAPIKeyError,
    APIConnectionError,
    APIResponseError,
    validate_api_key
)

logger = logging.getLogger(__name__)


class MistralClient:
    """
    Mistral API client for TELOS.

    Simple wrapper around the Mistral API with error handling.

    Example:
        >>> client = MistralClient()
        >>> response = client.generate(
        ...     messages=[{"role": "user", "content": "Hello"}],
        ...     max_tokens=100
        ... )
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "mistral-small-latest"
    ):
        """
        Initialize Mistral client.

        Args:
            api_key: Mistral API key (or set MISTRAL_API_KEY env var)
            model: Model to use for generation

        Raises:
            MissingAPIKeyError: If API key not found
        """
        # Get API key from parameter or environment
        api_key = api_key or os.getenv("MISTRAL_API_KEY")
        self.api_key = validate_api_key(api_key, "MISTRAL")

        self.model = model

        # Initialize Mistral client
        try:
            from mistralai import Mistral
            self.client = Mistral(api_key=self.api_key)
            logger.info(f"Mistral client initialized with model: {model}")
        except ImportError as e:
            raise APIConnectionError(
                "Mistral",
                "mistralai package not installed. Run: pip install mistralai"
            ) from e
        except Exception as e:
            raise APIConnectionError("Mistral", str(e)) from e

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 500,
        **kwargs
    ) -> str:
        """
        Generate response from Mistral API.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional arguments to pass to API

        Returns:
            Generated text response

        Raises:
            APIResponseError: If API returns an error
            APIConnectionError: If connection fails
        """
        try:
            # Call Mistral API
            response = self.client.chat.complete(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )

            # Extract response text
            if hasattr(response, 'choices') and response.choices:
                content = response.choices[0].message.content
                return content if content else ""
            else:
                raise APIResponseError(
                    "Mistral",
                    message="No response content in API response"
                )

        except Exception as e:
            if "rate limit" in str(e).lower():
                from telos_purpose.exceptions import APIRateLimitError
                raise APIRateLimitError("Mistral") from e
            elif hasattr(e, 'status_code'):
                raise APIResponseError(
                    "Mistral",
                    status_code=e.status_code,
                    message=str(e)
                ) from e
            else:
                raise APIConnectionError("Mistral", str(e)) from e

    def test_connection(self) -> bool:
        """
        Test connection to Mistral API.

        Returns:
            True if connection successful

        Raises:
            APIConnectionError: If connection fails
        """
        try:
            response = self.generate(
                messages=[{"role": "user", "content": "test"}],
                max_tokens=10
            )
            logger.info("Mistral connection test successful")
            return True
        except Exception as e:
            logger.error(f"Mistral connection test failed: {e}")
            raise


# Alias for backward compatibility
TelosMistralClient = MistralClient
