"""
TELOS-Governed SQL Agent
========================

A real SQL agent that connects to Supabase and executes queries,
with all operations governed by TELOS fidelity measurement.

This is the reference implementation for TELOS-governed agentic AI.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .supabase_client import SupabaseClient, SupabaseConfig, get_supabase_client

logger = logging.getLogger(__name__)


# =============================================================================
# Tool Definitions (OpenAI Function Calling Format)
# =============================================================================

SQL_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "sql_db_query",
            "description": "Execute a SQL query against the database and return results. Use for retrieving data, running SELECT statements, and getting query results. Always validate queries first with sql_db_query_checker.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The SQL query to execute. Should be a valid PostgreSQL query.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sql_db_schema",
            "description": "Get the schema and sample rows for specified SQL tables. Use to understand table structure, column types, and see example data before writing queries.",
            "parameters": {
                "type": "object",
                "properties": {
                    "table_names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of table names to get schema for.",
                    },
                },
                "required": ["table_names"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sql_db_list_tables",
            "description": "List all tables available in the database. Use when you need to discover what tables exist before querying.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sql_db_query_checker",
            "description": "Check if a SQL query is correct before executing. Use to validate syntax and catch potential errors. Always use this before sql_db_query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The SQL query to validate.",
                    },
                },
                "required": ["query"],
            },
        },
    },
]


# =============================================================================
# Agent Configuration
# =============================================================================

@dataclass
class SQLAgentConfig:
    """Configuration for the SQL Agent."""

    # Agent identity
    name: str = "SQL Database Agent"

    # Primacy Attractor (PA) - the agent's declared purpose
    primacy_attractor: str = """
    I am a SQL Database Agent specialized in querying and analyzing data stored in
    PostgreSQL databases. My purpose is to help users understand their data by:
    - Executing SQL queries to retrieve and analyze data
    - Explaining database schemas and table structures
    - Helping users formulate correct SQL queries
    - Providing insights from query results

    I operate within strict data access boundaries and only execute read operations
    unless explicitly configured otherwise. I do not modify data, drop tables, or
    perform destructive operations.
    """

    # LLM settings
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.0
    max_iterations: int = 10

    # Database settings (loaded from SupabaseConfig)
    supabase_config: Optional[SupabaseConfig] = None

    # Safety settings
    allow_write_operations: bool = False
    max_result_rows: int = 100

    @classmethod
    def from_env(cls) -> "SQLAgentConfig":
        """Load configuration from environment variables."""
        return cls(
            supabase_config=SupabaseConfig.from_env(),
            llm_model=os.getenv("SQL_AGENT_MODEL", "gpt-4o-mini"),
            allow_write_operations=os.getenv("SQL_AGENT_ALLOW_WRITE", "false").lower() == "true",
        )


# =============================================================================
# Tool Execution Results
# =============================================================================

@dataclass
class ToolResult:
    """Result from executing a tool."""

    tool_name: str
    success: bool
    result: Any
    error: Optional[str] = None
    execution_time_ms: float = 0.0


@dataclass
class AgentStep:
    """A single step in the agent's execution."""

    step_number: int
    thought: str
    tool_name: Optional[str]
    tool_input: Optional[Dict]
    tool_result: Optional[ToolResult]
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class AgentResponse:
    """Complete response from the agent."""

    query: str
    final_answer: str
    steps: List[AgentStep]
    total_time_ms: float
    tools_used: List[str]
    success: bool
    error: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "query": self.query,
            "final_answer": self.final_answer,
            "steps": [
                {
                    "step": s.step_number,
                    "thought": s.thought,
                    "tool": s.tool_name,
                    "tool_input": s.tool_input,
                    "tool_result": {
                        "success": s.tool_result.success,
                        "result": s.tool_result.result if s.tool_result else None,
                        "error": s.tool_result.error if s.tool_result else None,
                    } if s.tool_result else None,
                }
                for s in self.steps
            ],
            "total_time_ms": self.total_time_ms,
            "tools_used": self.tools_used,
            "success": self.success,
            "error": self.error,
        }


# =============================================================================
# SQL Agent Implementation
# =============================================================================

class SQLAgent:
    """
    TELOS-Governed SQL Agent.

    This agent:
    1. Receives user queries
    2. Has queries checked by TELOS governance (Tier 1 + Tier 2)
    3. If approved, reasons about how to answer using tools
    4. Executes tools against the database
    5. Returns results with full transparency

    All operations are logged for governance observability.
    """

    def __init__(
        self,
        config: Optional[SQLAgentConfig] = None,
        db_client: Optional[SupabaseClient] = None,
        llm_fn: Optional[Callable] = None,
    ):
        """
        Initialize the SQL Agent.

        Args:
            config: Agent configuration
            db_client: Database client (uses global singleton if not provided)
            llm_fn: Function to call LLM (for reasoning). Signature: (messages, tools) -> response
        """
        self.config = config or SQLAgentConfig.from_env()
        self.db_client = db_client or get_supabase_client()
        self.llm_fn = llm_fn

        # Tool registry
        self._tools = {
            "sql_db_query": self._tool_query,
            "sql_db_schema": self._tool_schema,
            "sql_db_list_tables": self._tool_list_tables,
            "sql_db_query_checker": self._tool_query_checker,
        }

        # Connect to database
        if not self.db_client.is_connected:
            connected = self.db_client.connect()
            if not connected:
                logger.warning("Failed to connect to database")

    @property
    def tools(self) -> List[Dict]:
        """Get tool definitions in OpenAI format."""
        return SQL_TOOLS

    @property
    def primacy_attractor(self) -> str:
        """Get the agent's Primacy Attractor."""
        return self.config.primacy_attractor

    def execute_tool(self, tool_name: str, tool_input: Dict) -> ToolResult:
        """
        Execute a tool by name.

        Args:
            tool_name: Name of the tool to execute
            tool_input: Input parameters for the tool

        Returns:
            ToolResult with success status and result/error
        """
        import time
        start_time = time.time()

        if tool_name not in self._tools:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error=f"Unknown tool: {tool_name}",
            )

        try:
            result = self._tools[tool_name](**tool_input)
            execution_time = (time.time() - start_time) * 1000

            return ToolResult(
                tool_name=tool_name,
                success=True,
                result=result,
                execution_time_ms=execution_time,
            )

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(f"Tool execution failed: {tool_name} - {e}")

            return ToolResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error=str(e),
                execution_time_ms=execution_time,
            )

    # -------------------------------------------------------------------------
    # Tool Implementations
    # -------------------------------------------------------------------------

    def _tool_query(self, query: str) -> Dict:
        """Execute a SQL query."""
        if not self.db_client.is_connected:
            return {"error": "Database not connected"}

        result = self.db_client.execute_query(query)

        # Limit results for display
        if result["success"] and len(result["data"]) > self.config.max_result_rows:
            result["data"] = result["data"][:self.config.max_result_rows]
            result["truncated"] = True
            result["message"] = f"Results truncated to {self.config.max_result_rows} rows"

        return result

    def _tool_schema(self, table_names: List[str]) -> Dict:
        """Get schema for specified tables."""
        if not self.db_client.is_connected:
            return {"error": "Database not connected"}

        schemas = {}
        for table_name in table_names:
            # Get column info
            schema_result = self.db_client.get_table_schema(table_name)
            if schema_result["success"]:
                schemas[table_name] = {
                    "columns": schema_result["data"],
                }

                # Get sample rows
                sample_result = self.db_client.get_sample_rows(table_name, limit=3)
                if sample_result["success"]:
                    schemas[table_name]["sample_rows"] = sample_result["data"]

        return {"tables": schemas}

    def _tool_list_tables(self) -> Dict:
        """List all tables in the database."""
        if not self.db_client.is_connected:
            return {"error": "Database not connected"}

        result = self.db_client.get_tables()

        if result["success"]:
            # Group by schema
            tables_by_schema = {}
            for row in result["data"]:
                schema = row.get("table_schema", "public")
                if schema not in tables_by_schema:
                    tables_by_schema[schema] = []
                tables_by_schema[schema].append({
                    "name": row.get("table_name"),
                    "type": row.get("table_type"),
                })

            return {"schemas": tables_by_schema}

        return result

    def _tool_query_checker(self, query: str) -> Dict:
        """Validate a SQL query without executing it."""
        if not self.db_client.is_connected:
            return {"error": "Database not connected"}

        result = self.db_client.validate_query(query)
        return result

    # -------------------------------------------------------------------------
    # Agent Execution (ReAct Loop)
    # -------------------------------------------------------------------------

    async def run(self, user_query: str) -> AgentResponse:
        """
        Run the agent on a user query.

        This implements a ReAct (Reasoning + Acting) loop:
        1. Think about what to do
        2. Choose a tool
        3. Execute the tool
        4. Observe the result
        5. Repeat until answer is ready

        Args:
            user_query: The user's question/request

        Returns:
            AgentResponse with full execution trace
        """
        import time
        start_time = time.time()

        steps: List[AgentStep] = []
        tools_used: List[str] = []

        # Check if we have an LLM function
        if not self.llm_fn:
            return AgentResponse(
                query=user_query,
                final_answer="LLM function not configured. Cannot reason about query.",
                steps=[],
                total_time_ms=(time.time() - start_time) * 1000,
                tools_used=[],
                success=False,
                error="No LLM function provided",
            )

        # Build system prompt
        system_prompt = f"""You are a SQL Database Agent with access to the following tools:

{json.dumps(SQL_TOOLS, indent=2)}

Your purpose: {self.config.primacy_attractor}

When answering questions:
1. First use sql_db_list_tables to see available tables (if needed)
2. Use sql_db_schema to understand table structure (if needed)
3. Use sql_db_query_checker to validate your query
4. Use sql_db_query to execute the query
5. Analyze results and provide a clear answer

Always explain your reasoning. If you encounter errors, try to fix them.
When you have the final answer, respond without calling any tools."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ]

        # ReAct loop
        for iteration in range(self.config.max_iterations):
            try:
                # Call LLM
                response = await self.llm_fn(messages, self.tools)

                # Check if LLM wants to use a tool
                if response.get("tool_calls"):
                    for tool_call in response["tool_calls"]:
                        tool_name = tool_call["function"]["name"]
                        tool_input = json.loads(tool_call["function"]["arguments"])

                        # Execute tool
                        tool_result = self.execute_tool(tool_name, tool_input)
                        tools_used.append(tool_name)

                        # Record step
                        steps.append(AgentStep(
                            step_number=len(steps) + 1,
                            thought=response.get("content", ""),
                            tool_name=tool_name,
                            tool_input=tool_input,
                            tool_result=tool_result,
                        ))

                        # Add tool result to messages
                        messages.append({
                            "role": "assistant",
                            "content": response.get("content"),
                            "tool_calls": response["tool_calls"],
                        })
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "content": json.dumps(tool_result.result if tool_result.success else {"error": tool_result.error}),
                        })

                else:
                    # LLM provided final answer
                    final_answer = response.get("content", "No answer generated.")

                    return AgentResponse(
                        query=user_query,
                        final_answer=final_answer,
                        steps=steps,
                        total_time_ms=(time.time() - start_time) * 1000,
                        tools_used=list(set(tools_used)),
                        success=True,
                    )

            except Exception as e:
                logger.error(f"Agent iteration {iteration} failed: {e}")
                return AgentResponse(
                    query=user_query,
                    final_answer=f"Error during execution: {str(e)}",
                    steps=steps,
                    total_time_ms=(time.time() - start_time) * 1000,
                    tools_used=list(set(tools_used)),
                    success=False,
                    error=str(e),
                )

        # Max iterations reached
        return AgentResponse(
            query=user_query,
            final_answer="Maximum iterations reached without finding answer.",
            steps=steps,
            total_time_ms=(time.time() - start_time) * 1000,
            tools_used=list(set(tools_used)),
            success=False,
            error="Max iterations exceeded",
        )

    def run_sync(self, user_query: str) -> AgentResponse:
        """
        Synchronous version of run() for non-async contexts.

        For demo/testing without full LLM integration.
        """
        import time
        start_time = time.time()

        # Simple demo: just list tables and return
        steps = []

        # Step 1: List tables
        list_result = self.execute_tool("sql_db_list_tables", {})
        steps.append(AgentStep(
            step_number=1,
            thought="First, let me see what tables are available in the database.",
            tool_name="sql_db_list_tables",
            tool_input={},
            tool_result=list_result,
        ))

        if not list_result.success:
            return AgentResponse(
                query=user_query,
                final_answer=f"Failed to list tables: {list_result.error}",
                steps=steps,
                total_time_ms=(time.time() - start_time) * 1000,
                tools_used=["sql_db_list_tables"],
                success=False,
                error=list_result.error,
            )

        # Format result
        tables_info = list_result.result
        if isinstance(tables_info, dict) and "schemas" in tables_info:
            table_list = []
            for schema, tables in tables_info["schemas"].items():
                for t in tables:
                    table_list.append(f"{schema}.{t['name']}")
            tables_str = ", ".join(table_list) if table_list else "No tables found"
        else:
            tables_str = str(tables_info)

        return AgentResponse(
            query=user_query,
            final_answer=f"Database tables available: {tables_str}\n\nTo get more specific answers, please configure the LLM integration.",
            steps=steps,
            total_time_ms=(time.time() - start_time) * 1000,
            tools_used=["sql_db_list_tables"],
            success=True,
        )


# =============================================================================
# Factory Function
# =============================================================================

def create_sql_agent(
    supabase_url: Optional[str] = None,
    supabase_key: Optional[str] = None,
    postgres_url: Optional[str] = None,
    llm_fn: Optional[Callable] = None,
) -> SQLAgent:
    """
    Factory function to create a configured SQL Agent.

    Args:
        supabase_url: Supabase project URL (or set SUPABASE_URL env var)
        supabase_key: Supabase API key (or set SUPABASE_KEY env var)
        postgres_url: Direct PostgreSQL URL (or set SUPABASE_POSTGRES_URL env var)
        llm_fn: Optional LLM function for reasoning

    Returns:
        Configured SQLAgent instance
    """
    # Build config
    supabase_config = SupabaseConfig(
        url=supabase_url or os.getenv("SUPABASE_URL", ""),
        key=supabase_key or os.getenv("SUPABASE_KEY", ""),
        postgres_url=postgres_url or os.getenv("SUPABASE_POSTGRES_URL", ""),
    )

    config = SQLAgentConfig(supabase_config=supabase_config)

    # Create client
    client = SupabaseClient(supabase_config)

    return SQLAgent(config=config, db_client=client, llm_fn=llm_fn)
