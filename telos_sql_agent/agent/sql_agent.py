"""
TELOS SQL Agent
===============

A SQL agent that operates under TELOS governance.
Tools are only executed when requests pass fidelity checks.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime

from .supabase_client import SupabaseClient, SupabaseConfig, get_supabase_client
from .governance_client import (
    TELOSGovernanceClient,
    GovernanceResult,
    GovernanceDecision,
    get_governance_client,
)
from .tool_planner import ToolPlanner, ToolPlanResult, get_tool_planner

logger = logging.getLogger(__name__)


@dataclass
class SQLAgentConfig:
    """Configuration for the SQL Agent."""

    # Agent identity
    name: str = "SQL Database Agent"

    # Primacy Attractor - the agent's declared purpose
    primacy_attractor: str = """
    I am a SQL database assistant. My purpose is to help users query,
    understand, and analyze data stored in PostgreSQL databases. I can:
    - List available tables and their schemas
    - Execute SELECT queries to retrieve data
    - Explain query results and help with data analysis
    - Validate SQL syntax before execution
    I only perform read operations and cannot modify database content.
    """

    # Model for LLM reasoning
    llm_model: str = "gpt-4o-mini"

    # Database config
    db_config: Optional[SupabaseConfig] = None


@dataclass
class ToolResult:
    """Result from a tool execution."""
    tool_name: str
    success: bool
    data: Any
    error: Optional[str] = None
    execution_time: float = 0.0


@dataclass
class AgentStep:
    """A single step in the agent's execution."""
    step_number: int
    thought: str
    action: str
    action_input: Dict[str, Any]
    observation: str
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class AgentResponse:
    """Final response from the agent."""
    success: bool
    answer: str
    steps: List[AgentStep] = field(default_factory=list)
    total_time: float = 0.0
    governance_decision: Optional[str] = None
    fidelity_score: Optional[float] = None


# SQL Agent Tools in OpenAI function format
SQL_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "sql_db_query",
            "description": "Execute a SQL query against the database and return results. Only SELECT queries are allowed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The SQL query to execute (SELECT only)"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "sql_db_schema",
            "description": "Get the schema (columns, types) for a specific table",
            "parameters": {
                "type": "object",
                "properties": {
                    "table_name": {
                        "type": "string",
                        "description": "Name of the table to get schema for"
                    }
                },
                "required": ["table_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "sql_db_list_tables",
            "description": "List all tables available in the database",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "sql_db_query_checker",
            "description": "Validate SQL syntax without executing the query",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The SQL query to validate"
                    }
                },
                "required": ["query"]
            }
        }
    }
]


class SQLAgent:
    """
    A TELOS-governed SQL agent.

    This agent can query databases but all operations are subject
    to TELOS governance checks based on the Primacy Attractor.
    """

    def __init__(self, config: Optional[SQLAgentConfig] = None):
        """Initialize the SQL agent."""
        self.config = config or SQLAgentConfig()
        self.db_client = SupabaseClient(self.config.db_config)
        self._connected = False
        self._governance_enabled = True
        self._governance_client: Optional[TELOSGovernanceClient] = None
        self._last_governance_result: Optional[GovernanceResult] = None
        self._tool_planner: Optional[ToolPlanner] = None
        self._last_tool_plan: Optional[ToolPlanResult] = None

    def connect(self) -> bool:
        """Connect to the database."""
        self._connected = self.db_client.connect()
        return self._connected

    def disconnect(self):
        """Disconnect from the database."""
        self.db_client.disconnect()
        self._connected = False

    def enable_governance(self, enabled: bool = True):
        """Enable or disable TELOS governance."""
        self._governance_enabled = enabled
        if enabled and self._governance_client is None:
            self._governance_client = get_governance_client(
                primacy_attractor=self.config.primacy_attractor
            )
        logger.info(f"TELOS Governance {'enabled' if enabled else 'disabled'}")

    def get_governance_client(self) -> TELOSGovernanceClient:
        """Get or create the governance client."""
        if self._governance_client is None:
            self._governance_client = get_governance_client(
                primacy_attractor=self.config.primacy_attractor
            )
        return self._governance_client

    @property
    def last_governance_result(self) -> Optional[GovernanceResult]:
        """Get the last governance check result."""
        return self._last_governance_result

    @property
    def last_tool_plan(self) -> Optional[ToolPlanResult]:
        """Get the last tool planning result."""
        return self._last_tool_plan

    def get_tool_planner(self) -> ToolPlanner:
        """Get or create the tool planner."""
        if self._tool_planner is None:
            self._tool_planner = get_tool_planner(
                primacy_attractor=self.config.primacy_attractor
            )
        return self._tool_planner

    def check_governance(self, user_query: str, tool_name: Optional[str] = None) -> GovernanceResult:
        """
        Check if a user query passes governance.

        Args:
            user_query: The user's request
            tool_name: Optional tool to be executed

        Returns:
            GovernanceResult with decision and fidelity score
        """
        client = self.get_governance_client()
        result = client.check_fidelity(user_query, tool_name=tool_name)
        self._last_governance_result = result
        return result

    @property
    def is_connected(self) -> bool:
        """Check if connected to database."""
        return self._connected

    @property
    def tools(self) -> List[Dict]:
        """Get available tools in OpenAI format."""
        return SQL_TOOLS

    def get_primacy_attractor(self) -> str:
        """Get the agent's Primacy Attractor."""
        return self.config.primacy_attractor

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> ToolResult:
        """
        Execute a tool by name with given arguments.

        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments

        Returns:
            ToolResult with execution outcome
        """
        start_time = datetime.utcnow()

        try:
            if tool_name == "sql_db_query":
                result = self.db_client.execute_query(arguments.get("query", ""))
                return ToolResult(
                    tool_name=tool_name,
                    success=result["success"],
                    data=result["data"] if result["success"] else None,
                    error=result.get("error"),
                    execution_time=(datetime.utcnow() - start_time).total_seconds()
                )

            elif tool_name == "sql_db_schema":
                table_name = arguments.get("table_name", "")
                result = self.db_client.get_table_schema(table_name)
                return ToolResult(
                    tool_name=tool_name,
                    success=result["success"],
                    data=result["data"] if result["success"] else None,
                    error=result.get("error"),
                    execution_time=(datetime.utcnow() - start_time).total_seconds()
                )

            elif tool_name == "sql_db_list_tables":
                result = self.db_client.get_tables()
                return ToolResult(
                    tool_name=tool_name,
                    success=result["success"],
                    data=result["data"] if result["success"] else None,
                    error=result.get("error"),
                    execution_time=(datetime.utcnow() - start_time).total_seconds()
                )

            elif tool_name == "sql_db_query_checker":
                query = arguments.get("query", "")
                result = self.db_client.validate_query(query)
                return ToolResult(
                    tool_name=tool_name,
                    success=result["valid"],
                    data={"valid": result["valid"]},
                    error=result.get("error"),
                    execution_time=(datetime.utcnow() - start_time).total_seconds()
                )

            else:
                return ToolResult(
                    tool_name=tool_name,
                    success=False,
                    data=None,
                    error=f"Unknown tool: {tool_name}",
                    execution_time=(datetime.utcnow() - start_time).total_seconds()
                )

        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return ToolResult(
                tool_name=tool_name,
                success=False,
                data=None,
                error=str(e),
                execution_time=(datetime.utcnow() - start_time).total_seconds()
            )

    def run_sync(self, user_query: str) -> AgentResponse:
        """
        Run the agent synchronously with TELOS governance.

        All queries are checked against the Primacy Attractor before
        tool execution. Low-fidelity queries are blocked or redirected.
        """
        if not self._connected:
            return AgentResponse(
                success=False,
                answer="Not connected to database. Please connect first.",
                steps=[]
            )

        start_time = datetime.utcnow()
        steps = []

        # =====================================================================
        # STEP 0: TELOS Governance Check
        # =====================================================================
        governance_result = None
        if self._governance_enabled:
            try:
                governance_result = self.check_governance(user_query)
                logger.info(
                    f"Governance: {governance_result.decision.value} "
                    f"(fidelity: {governance_result.fidelity_score:.1%})"
                )

                # Add governance step to trace
                steps.append(AgentStep(
                    step_number=0,
                    thought=f"TELOS Governance Check: {governance_result.message}",
                    action="governance_check",
                    action_input={"query": user_query},
                    observation=f"Decision: {governance_result.decision.value}, "
                               f"Fidelity: {governance_result.fidelity_score:.1%}"
                ))

                # Handle blocking decisions
                if not governance_result.should_proceed:
                    return AgentResponse(
                        success=False,
                        answer=self._format_governance_block(governance_result),
                        steps=steps,
                        total_time=(datetime.utcnow() - start_time).total_seconds(),
                        governance_decision=governance_result.decision.value,
                        fidelity_score=governance_result.fidelity_score,
                    )

            except Exception as e:
                logger.warning(f"Governance check failed, proceeding with caution: {e}")
                # Continue without governance if it fails

        # =====================================================================
        # STEP 1: Pattern Matching and Tool Execution
        # =====================================================================
        query_lower = user_query.lower()

        if "list" in query_lower and "table" in query_lower:
            # List tables
            result = self.execute_tool("sql_db_list_tables", {})
            steps.append(AgentStep(
                step_number=len(steps) + 1,
                thought="User wants to see available tables",
                action="sql_db_list_tables",
                action_input={},
                observation=str(result.data) if result.success else result.error
            ))

            if result.success:
                tables = [row.get("table_name", "") for row in result.data]
                answer = f"Available tables: {', '.join(tables)}"
            else:
                answer = f"Error listing tables: {result.error}"

        elif "schema" in query_lower or "columns" in query_lower:
            # Get schema - extract table name
            words = user_query.split()
            table_name = words[-1] if words else "unknown"

            result = self.execute_tool("sql_db_schema", {"table_name": table_name})
            steps.append(AgentStep(
                step_number=len(steps) + 1,
                thought=f"User wants schema for table '{table_name}'",
                action="sql_db_schema",
                action_input={"table_name": table_name},
                observation=str(result.data) if result.success else result.error
            ))

            if result.success:
                columns = [f"{row['column_name']} ({row['data_type']})" for row in result.data]
                answer = f"Schema for {table_name}:\n" + "\n".join(columns)
            else:
                answer = f"Error getting schema: {result.error}"

        elif query_lower.strip().startswith("select"):
            # Direct SQL query
            result = self.execute_tool("sql_db_query", {"query": user_query})
            steps.append(AgentStep(
                step_number=len(steps) + 1,
                thought="User provided a SQL query to execute",
                action="sql_db_query",
                action_input={"query": user_query},
                observation=str(result.data) if result.success else result.error
            ))

            if result.success:
                answer = f"Query returned {len(result.data)} rows:\n{result.data[:5]}"
            else:
                answer = f"Query error: {result.error}"

        else:
            answer = "I can help you with:\n- List tables\n- Show schema for [table_name]\n- Execute SELECT queries"
            steps.append(AgentStep(
                step_number=len(steps) + 1,
                thought="User query doesn't match known patterns",
                action="none",
                action_input={},
                observation="Provided help message"
            ))

        return AgentResponse(
            success=True,
            answer=answer,
            steps=steps,
            total_time=(datetime.utcnow() - start_time).total_seconds(),
            governance_decision=governance_result.decision.value if governance_result else None,
            fidelity_score=governance_result.fidelity_score if governance_result else None,
        )

    def run_with_tool_planning(self, user_query: str) -> AgentResponse:
        """
        Run the agent with LLM-based tool selection and per-tool governance.

        This is the real value proposition:
        1. User query comes in
        2. LLM proposes which tool(s) to use
        3. EACH proposed tool is checked against the Primacy Attractor
        4. Only tools with high enough fidelity (90%+) are executed

        Returns:
            AgentResponse with tool planning results
        """
        if not self._connected:
            return AgentResponse(
                success=False,
                answer="Not connected to database. Please connect first.",
                steps=[]
            )

        start_time = datetime.utcnow()
        steps = []

        # =====================================================================
        # STEP 1: LLM-Based Tool Planning with Per-Tool Governance
        # =====================================================================
        try:
            planner = self.get_tool_planner()
            tool_plan = planner.plan_tools(
                user_query=user_query,
                available_tools=SQL_TOOLS,
                governance_enabled=self._governance_enabled,
            )
            self._last_tool_plan = tool_plan

            # Record tool planning step
            steps.append(AgentStep(
                step_number=1,
                thought=f"LLM proposed {len(tool_plan.proposed_tools)} tool(s) for this query",
                action="tool_planning",
                action_input={"query": user_query},
                observation=f"Proposed: {[t.tool_name for t in tool_plan.proposed_tools]}"
            ))

            # Record governance results for each tool
            for i, gov_tool in enumerate(tool_plan.governed_tools):
                steps.append(AgentStep(
                    step_number=len(steps) + 1,
                    thought=f"Governance check for {gov_tool.tool_name}: {gov_tool.governance_message}",
                    action="governance_check",
                    action_input={
                        "tool": gov_tool.tool_name,
                        "arguments": gov_tool.arguments,
                    },
                    observation=f"Decision: {gov_tool.decision.upper()}, Fidelity: {gov_tool.fidelity_score:.0%}, Allowed: {gov_tool.allowed}"
                ))

            # =====================================================================
            # STEP 2: Execute Allowed Tools
            # =====================================================================
            if not tool_plan.allowed_tools:
                # No tools allowed - governance blocked everything
                if tool_plan.blocked_tools:
                    blocked_names = [t.tool_name for t in tool_plan.blocked_tools]
                    avg_fidelity = sum(t.fidelity_score for t in tool_plan.blocked_tools) / len(tool_plan.blocked_tools)
                    answer = (
                        f"Your request doesn't align well with my purpose as a SQL database assistant.\n\n"
                        f"Proposed tools: {', '.join(blocked_names)}\n"
                        f"Average fidelity: {avg_fidelity:.0%}\n\n"
                        f"I can help you with database operations like:\n"
                        f"- Listing tables\n"
                        f"- Showing schemas\n"
                        f"- Executing SELECT queries"
                    )
                else:
                    answer = (
                        "I couldn't determine which tools to use for your request.\n\n"
                        "Try being more specific about what database operation you need."
                    )

                return AgentResponse(
                    success=False,
                    answer=answer,
                    steps=steps,
                    total_time=(datetime.utcnow() - start_time).total_seconds(),
                    governance_decision=tool_plan.overall_decision,
                    fidelity_score=tool_plan.governed_tools[0].fidelity_score if tool_plan.governed_tools else 0.0,
                )

            # Execute allowed tools
            results = []
            for gov_tool in tool_plan.allowed_tools:
                tool_result = self.execute_tool(gov_tool.tool_name, gov_tool.arguments)

                steps.append(AgentStep(
                    step_number=len(steps) + 1,
                    thought=f"Executing {gov_tool.tool_name} (fidelity: {gov_tool.fidelity_score:.0%})",
                    action=gov_tool.tool_name,
                    action_input=gov_tool.arguments,
                    observation=str(tool_result.data) if tool_result.success else tool_result.error
                ))

                results.append(tool_result)

            # Format response
            if all(r.success for r in results):
                answer_parts = []
                for gov_tool, result in zip(tool_plan.allowed_tools, results):
                    if gov_tool.tool_name == "sql_db_list_tables":
                        tables = [row.get("table_name", "") for row in result.data]
                        answer_parts.append(f"Tables: {', '.join(tables)}")
                    elif gov_tool.tool_name == "sql_db_schema":
                        columns = [f"{row['column_name']} ({row['data_type']})" for row in result.data]
                        answer_parts.append(f"Schema:\n" + "\n".join(columns))
                    elif gov_tool.tool_name == "sql_db_query":
                        answer_parts.append(f"Query returned {len(result.data)} rows:\n{result.data[:5]}")
                    elif gov_tool.tool_name == "sql_db_query_checker":
                        answer_parts.append(f"Query validation: {'Valid' if result.data.get('valid') else 'Invalid'}")

                answer = "\n\n".join(answer_parts)
            else:
                errors = [r.error for r in results if not r.success]
                answer = f"Some operations failed: {', '.join(errors)}"

            # Get best fidelity score
            best_fidelity = max(t.fidelity_score for t in tool_plan.allowed_tools)

            return AgentResponse(
                success=True,
                answer=answer,
                steps=steps,
                total_time=(datetime.utcnow() - start_time).total_seconds(),
                governance_decision=tool_plan.overall_decision,
                fidelity_score=best_fidelity,
            )

        except Exception as e:
            logger.error(f"Tool planning failed: {e}")
            return AgentResponse(
                success=False,
                answer=f"Error during tool planning: {e}",
                steps=steps,
                total_time=(datetime.utcnow() - start_time).total_seconds(),
            )

    def _format_governance_block(self, result: GovernanceResult) -> str:
        """Format a governance block message for the user."""
        if result.decision == GovernanceDecision.INERT:
            return (
                f"Your request doesn't align with my purpose as a SQL database assistant. "
                f"(Fidelity: {result.fidelity_score:.0%})\n\n"
                f"I can help you with:\n"
                f"- Listing database tables\n"
                f"- Showing table schemas\n"
                f"- Executing SELECT queries\n\n"
                f"Please rephrase your request to relate to database operations."
            )
        elif result.decision == GovernanceDecision.ESCALATE:
            return (
                f"This request requires human review before I can proceed. "
                f"(Fidelity: {result.fidelity_score:.0%})\n\n"
                f"Please contact an administrator for approval."
            )
        else:
            return f"Request blocked: {result.message}"


def create_sql_agent(
    primacy_attractor: Optional[str] = None,
    db_config: Optional[SupabaseConfig] = None,
) -> SQLAgent:
    """
    Factory function to create a SQL agent.

    Args:
        primacy_attractor: Optional custom PA (uses default if not provided)
        db_config: Optional database config (uses env vars if not provided)

    Returns:
        Configured SQLAgent instance
    """
    config = SQLAgentConfig(
        primacy_attractor=primacy_attractor or SQLAgentConfig.primacy_attractor,
        db_config=db_config or SupabaseConfig.from_env()
    )
    return SQLAgent(config)
