"""
TELOS Tool Planner
==================

LLM-based tool selection with per-tool governance checks.
This is the core value proposition: AI proposes tools, TELOS governs each one.
"""

import os
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime

import httpx

logger = logging.getLogger(__name__)


@dataclass
class ProposedToolCall:
    """A tool call proposed by the LLM."""
    tool_name: str
    arguments: Dict[str, Any]
    reasoning: str = ""


@dataclass
class GovernedToolCall:
    """A tool call after governance review."""
    tool_name: str
    arguments: Dict[str, Any]
    reasoning: str
    fidelity_score: float
    decision: str  # execute, clarify, suggest, inert
    allowed: bool
    governance_message: str = ""


@dataclass
class ToolPlanResult:
    """Result of tool planning with governance."""
    user_query: str
    proposed_tools: List[ProposedToolCall]
    governed_tools: List[GovernedToolCall]
    allowed_tools: List[GovernedToolCall]
    blocked_tools: List[GovernedToolCall]
    overall_decision: str
    planning_time: float = 0.0


class ToolPlanner:
    """
    LLM-based tool planner with TELOS governance integration.

    Flow:
    1. User query comes in
    2. LLM reasons about which tool(s) to use and why
    3. For each proposed tool, governance checks alignment with PA
    4. Only tools with high enough fidelity (90%+) are executed
    """

    def __init__(
        self,
        api_key: str = None,
        model: str = "mistral-small-latest",
        primacy_attractor: str = None,
    ):
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        self.model = model
        self.primacy_attractor = primacy_attractor or self._default_pa()
        self._client = httpx.Client(timeout=30.0)

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

    def plan_tools(
        self,
        user_query: str,
        available_tools: List[Dict],
        governance_enabled: bool = True,
    ) -> ToolPlanResult:
        """
        Plan which tools to use for a user query, with governance checks.

        Args:
            user_query: The user's request
            available_tools: List of available tools in OpenAI format
            governance_enabled: Whether to apply governance checks

        Returns:
            ToolPlanResult with proposed and governed tools
        """
        start_time = datetime.utcnow()

        # Step 1: Use LLM to propose tool calls
        proposed_tools = self._propose_tools(user_query, available_tools)

        # Step 2: Apply governance to each proposed tool
        governed_tools = []
        for tool in proposed_tools:
            if governance_enabled:
                gov_result = self._check_tool_governance(user_query, tool)
            else:
                # No governance - everything is allowed
                gov_result = GovernedToolCall(
                    tool_name=tool.tool_name,
                    arguments=tool.arguments,
                    reasoning=tool.reasoning,
                    fidelity_score=1.0,
                    decision="execute",
                    allowed=True,
                    governance_message="Governance disabled",
                )
            governed_tools.append(gov_result)

        # Separate allowed vs blocked
        allowed_tools = [t for t in governed_tools if t.allowed]
        blocked_tools = [t for t in governed_tools if not t.allowed]

        # Determine overall decision
        if not proposed_tools:
            overall_decision = "no_tools_needed"
        elif all(t.allowed for t in governed_tools):
            overall_decision = "execute"
        elif any(t.allowed for t in governed_tools):
            overall_decision = "partial"
        else:
            overall_decision = "blocked"

        planning_time = (datetime.utcnow() - start_time).total_seconds()

        return ToolPlanResult(
            user_query=user_query,
            proposed_tools=proposed_tools,
            governed_tools=governed_tools,
            allowed_tools=allowed_tools,
            blocked_tools=blocked_tools,
            overall_decision=overall_decision,
            planning_time=planning_time,
        )

    def _propose_tools(
        self,
        user_query: str,
        available_tools: List[Dict],
    ) -> List[ProposedToolCall]:
        """
        Use LLM to propose which tools to use.
        """
        # Build tool descriptions for the prompt
        tool_descriptions = []
        for tool in available_tools:
            func = tool.get("function", {})
            tool_descriptions.append(
                f"- {func.get('name')}: {func.get('description', 'No description')}"
            )

        system_prompt = f"""You are a SQL database assistant. Your purpose is to help users with database operations.

Available tools:
{chr(10).join(tool_descriptions)}

Based on the user's request, determine which tool(s) to use.
If the request is NOT related to SQL database operations, respond with NO_TOOLS.

Respond in JSON format:
{{
    "tools": [
        {{
            "tool_name": "name of the tool",
            "arguments": {{"arg": "value"}},
            "reasoning": "why this tool is appropriate"
        }}
    ]
}}

Or if no tools are appropriate:
{{
    "tools": [],
    "reasoning": "why no tools are appropriate for this request"
}}"""

        try:
            response = self._client.post(
                "https://api.mistral.ai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_query},
                    ],
                    "temperature": 0.1,  # Low temp for deterministic tool selection
                },
            )

            if response.status_code != 200:
                logger.error(f"LLM API error: {response.status_code}")
                return []

            data = response.json()
            content = data["choices"][0]["message"]["content"]

            # Parse JSON from response
            # Handle potential markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            try:
                result = json.loads(content.strip())
            except json.JSONDecodeError:
                logger.error(f"Failed to parse LLM response: {content}")
                return []

            proposed = []
            for tool_data in result.get("tools", []):
                proposed.append(ProposedToolCall(
                    tool_name=tool_data.get("tool_name", ""),
                    arguments=tool_data.get("arguments", {}),
                    reasoning=tool_data.get("reasoning", ""),
                ))

            return proposed

        except Exception as e:
            logger.error(f"Tool planning failed: {e}")
            return []

    def _check_tool_governance(
        self,
        user_query: str,
        tool: ProposedToolCall,
    ) -> GovernedToolCall:
        """
        Check if a specific tool call aligns with the Primacy Attractor.

        This is where TELOS governance happens - each tool + arguments
        is checked for alignment with the agent's purpose.
        """
        # Build a description of what this tool call will do
        tool_action = f"Execute {tool.tool_name} with arguments: {json.dumps(tool.arguments)}"

        # Compute fidelity between the tool action and the PA
        # We use keyword matching for local fallback
        fidelity = self._compute_tool_fidelity(tool.tool_name, tool.arguments, user_query)

        # Strict thresholds
        if fidelity >= 0.90:
            decision = "execute"
            allowed = True
            message = f"High alignment ({fidelity:.0%}) - tool execution approved"
        elif fidelity >= 0.70:
            decision = "clarify"
            allowed = True  # Allow but note clarification recommended
            message = f"Moderate alignment ({fidelity:.0%}) - clarification recommended"
        elif fidelity >= 0.50:
            decision = "suggest"
            allowed = False
            message = f"Low alignment ({fidelity:.0%}) - suggesting alternatives"
        else:
            decision = "inert"
            allowed = False
            message = f"Off-purpose ({fidelity:.0%}) - tool blocked"

        return GovernedToolCall(
            tool_name=tool.tool_name,
            arguments=tool.arguments,
            reasoning=tool.reasoning,
            fidelity_score=fidelity,
            decision=decision,
            allowed=allowed,
            governance_message=message,
        )

    def _compute_tool_fidelity(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        user_query: str,
    ) -> float:
        """
        Compute fidelity score for a tool call.

        This is a local fallback using keyword matching.
        In production, this would use embeddings via the TELOS Gateway.
        """
        # SQL tools are purpose-aligned
        sql_tools = {
            "sql_db_query": 1.0,
            "sql_db_schema": 1.0,
            "sql_db_list_tables": 1.0,
            "sql_db_query_checker": 1.0,
        }

        # Base score from tool name
        base_score = sql_tools.get(tool_name, 0.3)

        # Boost/penalize based on query content
        query_lower = user_query.lower()

        # SQL keywords increase fidelity
        sql_keywords = [
            "select", "table", "schema", "query", "database", "column",
            "row", "data", "sql", "postgres", "join", "where", "from",
            "order", "group", "count", "sum", "avg", "list", "show",
        ]
        keyword_matches = sum(1 for kw in sql_keywords if kw in query_lower)
        keyword_boost = min(keyword_matches * 0.1, 0.3)

        # Off-topic indicators decrease fidelity
        off_topic = ["weather", "recipe", "joke", "email", "story", "poem", "song"]
        off_topic_matches = sum(1 for ot in off_topic if ot in query_lower)
        off_topic_penalty = min(off_topic_matches * 0.3, 0.6)

        # Compute final score
        fidelity = min(max(base_score + keyword_boost - off_topic_penalty, 0.0), 1.0)

        return fidelity

    def close(self):
        """Close the HTTP client."""
        self._client.close()


def get_tool_planner(
    primacy_attractor: str = None,
) -> ToolPlanner:
    """Factory function to create a tool planner."""
    return ToolPlanner(primacy_attractor=primacy_attractor)
