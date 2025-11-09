"""
TELOS Mistral Client

Simple, working Mistral API client for TELOS.
Compatible with mistralai >= 1.0.0
"""

import os
import logging
import time
from typing import List, Dict, Any, Optional

from telos.exceptions import (
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
        max_retries: int = 3,
        timeout: int = 30,
        **kwargs
    ) -> Optional[str]:
        """
        Generate response from Mistral API with retry logic and error handling.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            max_retries: Maximum number of retry attempts
            timeout: Timeout in seconds for API call
            **kwargs: Additional arguments to pass to API

        Returns:
            Generated text response or None if all retries fail

        Raises:
            APIResponseError: If API returns an error after all retries
            APIConnectionError: If connection fails after all retries
        """
        last_exception = None

        for attempt in range(max_retries):
            try:
                # Call Mistral API with timeout
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
                    if content:
                        return content
                    else:
                        logger.warning(f"Empty response on attempt {attempt + 1}/{max_retries}")
                        # Empty response - could be a transient issue, retry
                        if attempt < max_retries - 1:
                            time.sleep(2 ** attempt)  # Exponential backoff
                            continue
                        else:
                            logger.error("Empty response after all retries")
                            return None
                else:
                    logger.warning(f"No choices in response on attempt {attempt + 1}/{max_retries}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                        continue
                    else:
                        return None

            except TimeoutError:
                logger.warning(f"Timeout on attempt {attempt + 1}/{max_retries}")
                last_exception = TimeoutError(f"API call timed out after {timeout}s")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"Timeout after {max_retries} attempts")
                    return None

            except Exception as e:
                error_str = str(e).lower()
                last_exception = e

                # Handle rate limiting with exponential backoff
                if "rate" in error_str and "limit" in error_str:
                    wait_time = 2 ** (attempt + 1)  # Longer wait for rate limits
                    logger.warning(f"Rate limited on attempt {attempt + 1}/{max_retries}, waiting {wait_time}s")
                    if attempt < max_retries - 1:
                        time.sleep(wait_time)
                        continue
                    else:
                        from telos.exceptions import APIRateLimitError
                        logger.error("Rate limit exceeded after all retries")
                        return None

                # Handle service unavailable
                elif "503" in str(e) or "unavailable" in error_str:
                    wait_time = 2 ** attempt
                    logger.warning(f"Service unavailable on attempt {attempt + 1}/{max_retries}, waiting {wait_time}s")
                    if attempt < max_retries - 1:
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error("Service unavailable after all retries")
                        return None

                # Handle authentication errors (don't retry)
                elif "401" in str(e) or "unauthorized" in error_str or "api key" in error_str:
                    logger.error(f"Authentication error: {e}")
                    raise APIResponseError("Mistral", message=f"Authentication failed: {e}") from e

                # Handle other errors with backoff
                else:
                    logger.warning(f"API error on attempt {attempt + 1}/{max_retries}: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                        continue
                    else:
                        logger.error(f"API error after all retries: {e}")
                        return None

        # If we get here, all retries failed
        logger.error(f"All {max_retries} attempts failed. Last error: {last_exception}")
        return None

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 500,
        max_retries: int = 3,
        timeout: int = 30,
        **kwargs
    ) -> Optional[str]:
        """
        Alternative interface for chat completion (alias for generate).

        Provided for compatibility with intervention systems that call
        chat_completion() instead of generate().

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            max_retries: Maximum number of retry attempts
            timeout: Timeout in seconds for API call
            **kwargs: Additional arguments to pass to API

        Returns:
            Generated text response or None if all retries fail
        """
        return self.generate(messages, temperature, max_tokens, max_retries, timeout, **kwargs)

    def generate_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 500,
        **kwargs
    ):
        """
        Generate streaming response from Mistral API.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional arguments to pass to API

        Yields:
            Text chunks as they arrive from the API

        Raises:
            APIResponseError: If API returns an error
            APIConnectionError: If connection fails
        """
        try:
            # Call Mistral API with streaming
            stream = self.client.chat.stream(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )

            # Yield chunks as they arrive
            for chunk in stream:
                if hasattr(chunk, 'data') and hasattr(chunk.data, 'choices'):
                    if chunk.data.choices and len(chunk.data.choices) > 0:
                        delta = chunk.data.choices[0].delta
                        if hasattr(delta, 'content') and delta.content:
                            yield delta.content

        except Exception as e:
            if "rate limit" in str(e).lower():
                from telos.exceptions import APIRateLimitError
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
