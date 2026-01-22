"""
TELOS Governance Client
=======================

Client for integrating SQL Agent with TELOS Gateway.
Queries pass through governance before tool execution.
"""

import os
import logging
import httpx
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class GovernanceDecision(Enum):
    """Possible governance decisions from the Gateway."""
    EXECUTE = "execute"      # High fidelity - proceed
    CLARIFY = "clarify"      # Medium fidelity - ask for clarification
    SUGGEST = "suggest"      # Low fidelity - suggest alternatives
    INERT = "inert"          # Very low fidelity - do nothing harmful
    ESCALATE = "escalate"    # High risk mode - needs human review
    ERROR = "error"          # Gateway error


@dataclass
class GovernanceResult:
    """Result from a governance check."""
    decision: GovernanceDecision
    fidelity_score: float
    message: str
    should_proceed: bool
    raw_response: Optional[Dict] = None
    tier0_blocked: bool = False
    tier0_reason: Optional[str] = None
    tools_checked: int = 0
    tools_blocked: int = 0


class TELOSGovernanceClient:
    """
    Client for TELOS Gateway governance checks.

    All SQL queries are validated against the agent's Primacy Attractor
    before execution to ensure fidelity to purpose.
    """

    # Hardcoded for demonstration - uses owner's Mistral API key
    DEFAULT_GATEWAY_URL = "http://127.0.0.1:8000"
    DEFAULT_API_KEY = os.getenv("MISTRAL_API_KEY", "fhrafrxJW26mXD2lFhxNTkTl6iOiRmHG")

    def __init__(
        self,
        gateway_url: str = None,
        api_key: str = None,
        primacy_attractor: str = None,
        timeout: float = 30.0,
    ):
        """
        Initialize the governance client.

        Args:
            gateway_url: URL of the TELOS Gateway (default: localhost:8000)
            api_key: API key for authentication (uses demo key by default)
            primacy_attractor: The agent's declared purpose
            timeout: Request timeout in seconds
        """
        self.gateway_url = gateway_url or self.DEFAULT_GATEWAY_URL
        self.api_key = api_key or self.DEFAULT_API_KEY
        self.primacy_attractor = primacy_attractor or self._default_pa()
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    def _default_pa(self) -> str:
        """Default Primacy Attractor for SQL Agent."""
        return """
        I am a SQL database assistant. My purpose is to help users query,
        understand, and analyze data stored in PostgreSQL databases. I can:
        - List available tables and their schemas
        - Execute SELECT queries to retrieve data
        - Explain query results and help with data analysis
        - Validate SQL syntax before execution
        I only perform read operations and cannot modify database content.
        """

    def check_fidelity(
        self,
        user_query: str,
        tool_name: Optional[str] = None,
        tool_args: Optional[Dict] = None,
    ) -> GovernanceResult:
        """
        Check if a user query aligns with the agent's purpose.

        Args:
            user_query: The user's request
            tool_name: Name of the tool to be executed (optional)
            tool_args: Arguments for the tool (optional)

        Returns:
            GovernanceResult with decision and fidelity score
        """
        try:
            # Build request in OpenAI chat format
            messages = [
                {
                    "role": "system",
                    "content": self.primacy_attractor,
                },
                {
                    "role": "user",
                    "content": user_query,
                }
            ]

            # Add tool info if provided
            tools = None
            if tool_name:
                tools = [{
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": f"Execute {tool_name}",
                        "parameters": {
                            "type": "object",
                            "properties": tool_args or {},
                        }
                    }
                }]

            request_body = {
                "model": "mistral-small-latest",
                "messages": messages,
            }

            if tools:
                request_body["tools"] = tools

            # Call Gateway
            response = self._client.post(
                f"{self.gateway_url}/v1/chat/completions",
                json=request_body,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                }
            )

            if response.status_code == 200:
                data = response.json()
                return self._parse_response(data)
            elif response.status_code == 401:
                logger.error("Gateway authentication failed")
                return GovernanceResult(
                    decision=GovernanceDecision.ERROR,
                    fidelity_score=0.0,
                    message="Gateway authentication failed",
                    should_proceed=False,
                )
            else:
                logger.error(f"Gateway error: {response.status_code} - {response.text}")
                return GovernanceResult(
                    decision=GovernanceDecision.ERROR,
                    fidelity_score=0.0,
                    message=f"Gateway error: {response.status_code}",
                    should_proceed=False,
                )

        except httpx.ConnectError:
            logger.warning("Gateway not available - proceeding with local governance")
            return self._local_fidelity_check(user_query)
        except Exception as e:
            logger.error(f"Governance check failed: {e}")
            return GovernanceResult(
                decision=GovernanceDecision.ERROR,
                fidelity_score=0.0,
                message=str(e),
                should_proceed=False,
            )

    def _parse_response(self, data: Dict) -> GovernanceResult:
        """Parse Gateway response into GovernanceResult."""
        governance = data.get("telos_governance", {})

        decision_str = governance.get("decision", "execute")
        try:
            decision = GovernanceDecision(decision_str.lower())
        except ValueError:
            decision = GovernanceDecision.EXECUTE

        fidelity = governance.get("input_fidelity", 0.5)
        blocked = governance.get("blocked", False)

        # Determine if we should proceed
        should_proceed = decision in [
            GovernanceDecision.EXECUTE,
            GovernanceDecision.CLARIFY,
            GovernanceDecision.SUGGEST,
        ] and not blocked

        # Build message based on decision
        if decision == GovernanceDecision.EXECUTE:
            message = f"Aligned with purpose (fidelity: {fidelity:.1%})"
        elif decision == GovernanceDecision.CLARIFY:
            message = f"Moderate alignment (fidelity: {fidelity:.1%}) - clarification recommended"
        elif decision == GovernanceDecision.SUGGEST:
            message = f"Low alignment (fidelity: {fidelity:.1%}) - suggesting alternatives"
        elif decision == GovernanceDecision.INERT:
            message = f"Request outside purpose scope (fidelity: {fidelity:.1%})"
        elif decision == GovernanceDecision.ESCALATE:
            message = f"High-risk request requires review (fidelity: {fidelity:.1%})"
        else:
            message = "Unknown governance decision"

        return GovernanceResult(
            decision=decision,
            fidelity_score=fidelity,
            message=message,
            should_proceed=should_proceed,
            raw_response=data,
            tier0_blocked=governance.get("tier0_blocked", False),
            tier0_reason=governance.get("tier0_reason"),
            tools_checked=governance.get("tools_checked", 0),
            tools_blocked=governance.get("tools_blocked", 0),
        )

    def _local_fidelity_check(self, user_query: str) -> GovernanceResult:
        """
        Local fidelity check when Gateway is unavailable.

        Uses keyword matching as a fallback.

        Threshold Bands (stricter for purpose-bound execution):
        - EXECUTE: 90%+ fidelity (high confidence alignment)
        - CLARIFY: 70-89% fidelity (moderate alignment, needs clarification)
        - SUGGEST: 50-69% fidelity (low alignment, suggest alternatives)
        - INERT: <50% fidelity (off-purpose, do nothing harmful)
        """
        query_lower = user_query.lower()

        # SQL-related keywords
        sql_keywords = [
            "select", "table", "schema", "query", "database", "column",
            "row", "data", "sql", "postgres", "join", "where", "from",
            "order", "group", "count", "sum", "avg", "list", "show",
        ]

        # Count matches - need more matches for higher fidelity
        matches = sum(1 for kw in sql_keywords if kw in query_lower)
        fidelity = min(matches / 5, 1.0)  # Cap at 1.0

        # Stricter thresholds for purpose-bound governance
        if fidelity >= 0.90:
            return GovernanceResult(
                decision=GovernanceDecision.EXECUTE,
                fidelity_score=fidelity,
                message=f"High alignment: SQL-related query (fidelity: {fidelity:.1%})",
                should_proceed=True,
            )
        elif fidelity >= 0.70:
            return GovernanceResult(
                decision=GovernanceDecision.CLARIFY,
                fidelity_score=fidelity,
                message=f"Moderate alignment: Clarification recommended (fidelity: {fidelity:.1%})",
                should_proceed=True,
            )
        elif fidelity >= 0.50:
            return GovernanceResult(
                decision=GovernanceDecision.SUGGEST,
                fidelity_score=fidelity,
                message=f"Low alignment: Suggesting alternatives (fidelity: {fidelity:.1%})",
                should_proceed=False,
            )
        else:
            return GovernanceResult(
                decision=GovernanceDecision.INERT,
                fidelity_score=fidelity,
                message=f"Off-purpose: Request doesn't align with SQL operations (fidelity: {fidelity:.1%})",
                should_proceed=False,
            )

    def close(self):
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# Singleton instance
_governance_client: Optional[TELOSGovernanceClient] = None


def get_governance_client(
    primacy_attractor: str = None,
) -> TELOSGovernanceClient:
    """Get or create the governance client singleton."""
    global _governance_client
    if _governance_client is None:
        _governance_client = TELOSGovernanceClient(
            primacy_attractor=primacy_attractor,
        )
    return _governance_client
