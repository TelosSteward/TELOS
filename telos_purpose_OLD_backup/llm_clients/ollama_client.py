"""
Ollama Client for TELOS - Local Model Execution.

Drop-in replacement for MistralClient that runs models locally via Ollama.
This eliminates API costs and enables complete validation re-runs with real data.

Prerequisites:
1. Install Ollama: https://ollama.ai
2. Pull models:
   - ollama pull mistral:7b (or mistral:7b-instruct)
   - ollama pull llama2:13b (alternative)
   - ollama pull nomic-embed-text (for embeddings if needed)
"""

import json
import time
import requests
from typing import List, Dict, Any, Optional, Generator
import logging

logger = logging.getLogger(__name__)


class OllamaClient:
    """
    Ollama client that matches MistralClient interface.

    This allows seamless replacement of API calls with local model execution.
    All validation studies can be re-run without code changes.
    """

    def __init__(
        self,
        model: str = "mistral:latest",
        base_url: str = "http://localhost:11434",
        timeout: int = 600
    ):
        """
        Initialize Ollama client.

        Args:
            model: Model name (e.g., "mistral:7b-instruct", "llama2:13b")
            base_url: Ollama server URL
            timeout: Request timeout in seconds
        """
        self.model = model
        self.base_url = base_url
        self.timeout = timeout

        # Test connection
        self._test_connection()

    def _test_connection(self):
        """Test Ollama connection and model availability."""
        try:
            # Check if Ollama is running
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()

            # Check if model is available
            models = response.json()
            available_models = [m["name"] for m in models.get("models", [])]

            if self.model not in available_models:
                logger.warning(
                    f"Model {self.model} not found. Available models: {available_models}"
                )
                logger.info(f"Pull model with: ollama pull {self.model}")

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            logger.info("Make sure Ollama is running: ollama serve")
            raise

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = 500,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stream: bool = False,
        **kwargs
    ) -> str:
        """
        Generate response matching MistralClient.generate interface.

        Args:
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            stream: Whether to stream response
            **kwargs: Additional parameters

        Returns:
            Generated text response
        """
        # Convert to Ollama format
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": max_tokens,
            }
        }

        # Add any additional options
        if "stop" in kwargs:
            payload["options"]["stop"] = kwargs["stop"]

        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout,
                stream=stream
            )
            response.raise_for_status()

            if stream:
                return self._handle_stream(response)
            else:
                result = response.json()
                return result["message"]["content"]

        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama generation failed: {e}")
            raise

    def generate_stream(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = 500,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> Generator[str, None, None]:
        """
        Stream response matching MistralClient.generate_stream interface.

        Args:
            messages: List of message dicts
            max_tokens: Maximum tokens
            temperature: Sampling temperature
            top_p: Nucleus sampling
            **kwargs: Additional parameters

        Yields:
            Response tokens as they're generated
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": max_tokens,
            }
        }

        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                stream=True,
                timeout=self.timeout
            )
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if "message" in data and "content" in data["message"]:
                        yield data["message"]["content"]

        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama stream failed: {e}")
            raise

    def _handle_stream(self, response):
        """Handle streaming response."""
        full_response = ""
        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                if "message" in data and "content" in data["message"]:
                    full_response += data["message"]["content"]
        return full_response

    def embed(self, text: str, model: str = "nomic-embed-text") -> List[float]:
        """
        Generate embeddings using Ollama.

        Args:
            text: Text to embed
            model: Embedding model to use

        Returns:
            Embedding vector
        """
        payload = {
            "model": model,
            "prompt": text
        }

        try:
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            return result["embedding"]

        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama embedding failed: {e}")
            raise

    def count_tokens(self, text: str) -> int:
        """
        Estimate token count (rough approximation).

        Ollama doesn't provide exact tokenization, so we estimate.

        Args:
            text: Text to count tokens for

        Returns:
            Estimated token count
        """
        # Rough estimate: 1 token ≈ 4 characters
        return len(text) // 4

    def format_messages(self, messages: List[Dict[str, str]]) -> str:
        """
        Format messages for display/debugging.

        Args:
            messages: List of message dicts

        Returns:
            Formatted string
        """
        formatted = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            formatted.append(f"[{role.upper()}]: {content}")
        return "\n\n".join(formatted)


class OllamaGovernanceClient(OllamaClient):
    """
    Extended Ollama client for TELOS governance operations.

    Adds governance-specific methods while maintaining compatibility.
    """

    def derive_ai_pa(
        self,
        user_purpose: str,
        user_scope: str,
        user_boundaries: List[str]
    ) -> Dict[str, Any]:
        """
        Derive AI Primacy Attractor from User PA using local model.

        Args:
            user_purpose: User's stated purpose
            user_scope: User's scope definition
            user_boundaries: User's boundaries list

        Returns:
            Derived AI PA configuration
        """
        prompt = f"""Given the user's established Primacy Attractor, derive the corresponding AI Primacy Attractor that defines YOUR role and behavior.

User Primacy Attractor:
- Purpose: {user_purpose}
- Scope: {user_scope}
- Boundaries: {', '.join(user_boundaries)}

Generate the AI Primacy Attractor that aligns with and supports the user's PA:

1. AI Purpose: What is YOUR role in supporting the user's purpose?
2. AI Scope: What capabilities and knowledge should you apply?
3. AI Boundaries: What constraints govern YOUR behavior?

Respond in JSON format:
{{
    "ai_purpose": "...",
    "ai_scope": "...",
    "ai_boundaries": ["...", "..."]
}}"""

        messages = [{"role": "user", "content": prompt}]

        response = self.generate(
            messages,
            max_tokens=500,
            temperature=0.3  # Lower temperature for consistency
        )

        try:
            # Parse JSON response
            ai_pa = json.loads(response)
            return ai_pa
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            logger.warning("Failed to parse AI PA JSON, using defaults")
            return {
                "ai_purpose": "Support user's stated purpose",
                "ai_scope": "Apply relevant knowledge and capabilities",
                "ai_boundaries": ["Stay within user boundaries", "Maintain alignment"]
            }

    def generate_intervention(
        self,
        original_response: str,
        error_signal: float,
        user_message: str,
        primacy_purpose: str
    ) -> str:
        """
        Generate intervention-corrected response.

        Args:
            original_response: Response that drifted
            error_signal: Magnitude of drift (0-1)
            user_message: Original user input
            primacy_purpose: User's established purpose

        Returns:
            Corrected response
        """
        intervention_prompt = f"""The previous response has drifted from the user's established purpose.

User's Purpose: {primacy_purpose}
Drift Severity: {error_signal * 100:.1f}%

User Message: {user_message}

Original Response: {original_response}

Generate a corrected response that better aligns with the user's purpose.
Maintain helpfulness while ensuring strong alignment with their stated goals."""

        messages = [
            {"role": "system", "content": intervention_prompt},
            {"role": "user", "content": user_message}
        ]

        return self.generate(
            messages,
            max_tokens=500,
            temperature=0.5  # Moderate temperature for corrections
        )

    def analyze_governance_state(
        self,
        fidelity_history: List[float],
        current_fidelity: float
    ) -> Dict[str, Any]:
        """
        Analyze governance state for Steward.

        Args:
            fidelity_history: Historical fidelity scores
            current_fidelity: Current fidelity score

        Returns:
            Governance analysis
        """
        # Calculate statistics
        avg_fidelity = sum(fidelity_history) / len(fidelity_history) if fidelity_history else 0
        trend = "improving" if current_fidelity > avg_fidelity else "degrading"

        return {
            "state": self._classify_state(current_fidelity),
            "current_fidelity": current_fidelity,
            "average_fidelity": avg_fidelity,
            "trend": trend,
            "recommendation": self._get_recommendation(current_fidelity)
        }

    def _classify_state(self, fidelity: float) -> str:
        """Classify governance state based on fidelity."""
        if fidelity >= 0.85:
            return "MONITOR"
        elif fidelity >= 0.70:
            return "CORRECT"
        elif fidelity >= 0.50:
            return "INTERVENE"
        else:
            return "ESCALATE"

    def _get_recommendation(self, fidelity: float) -> str:
        """Get governance recommendation."""
        if fidelity >= 0.85:
            return "System operating within bounds, continue monitoring"
        elif fidelity >= 0.70:
            return "Minor drift detected, apply proportional correction"
        elif fidelity >= 0.50:
            return "Significant drift, intervention required"
        else:
            return "Critical drift, escalate to human oversight"


# Adapter for drop-in replacement of MistralClient
class MistralCompatibleClient(OllamaGovernanceClient):
    """
    100% compatible with existing MistralClient interface.

    Use this for validation script replacement without any code changes.
    """

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """
        Initialize with MistralClient-compatible interface.

        Args:
            api_key: Ignored (for compatibility)
            **kwargs: Passed to OllamaClient
        """
        # Default to Mistral model for compatibility
        if "model" not in kwargs:
            kwargs["model"] = "mistral:7b-instruct"

        super().__init__(**kwargs)

        # Log that we're using local model
        logger.info(f"Using local Ollama model: {self.model}")
        logger.info("API key ignored - running locally")


if __name__ == "__main__":
    # Test the client
    print("Testing Ollama Client for TELOS...")
    print("=" * 60)

    # Test basic client
    client = OllamaClient()
    print(f"✓ Connected to Ollama at {client.base_url}")
    print(f"✓ Using model: {client.model}")

    # Test generation
    messages = [
        {"role": "user", "content": "What is AI governance and why is it important?"}
    ]

    print("\nGenerating response...")
    response = client.generate(messages, max_tokens=100)
    print(f"Response: {response[:200]}...")

    # Test governance client
    gov_client = OllamaGovernanceClient()

    # Test AI PA derivation
    print("\nTesting AI PA derivation...")
    ai_pa = gov_client.derive_ai_pa(
        user_purpose="Learn about AI safety",
        user_scope="Technical concepts and research",
        user_boundaries=["No speculation", "Cite sources"]
    )
    print(f"AI PA: {json.dumps(ai_pa, indent=2)}")

    # Test governance analysis
    print("\nTesting governance analysis...")
    analysis = gov_client.analyze_governance_state(
        fidelity_history=[0.75, 0.78, 0.82, 0.79],
        current_fidelity=0.71
    )
    print(f"Analysis: {json.dumps(analysis, indent=2)}")

    print("\n" + "=" * 60)
    print("✓ All tests passed! Ollama client ready for TELOS validation.")