"""
Supabase Database Client
========================

Handles connection to Supabase PostgreSQL database.
Provides low-level SQL execution with safety measures.
"""

import os
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import json

logger = logging.getLogger(__name__)

# Try to import psycopg2 for direct PostgreSQL connection
try:
    import psycopg2
    import psycopg2.extras
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False
    logger.warning("psycopg2 not installed. Install with: pip install psycopg2-binary")


@dataclass
class SupabaseConfig:
    """Configuration for Supabase connection."""

    # Direct PostgreSQL connection string
    postgres_url: str = ""

    # Safety settings
    allow_write: bool = False  # Only allow SELECT by default
    max_rows: int = 1000       # Limit result rows
    timeout_seconds: int = 30  # Query timeout

    @classmethod
    def from_env(cls) -> "SupabaseConfig":
        """Load configuration from environment variables."""
        return cls(
            postgres_url=os.getenv("SUPABASE_POSTGRES_URL", ""),
            allow_write=os.getenv("SUPABASE_ALLOW_WRITE", "false").lower() == "true",
            max_rows=int(os.getenv("SUPABASE_MAX_ROWS", "1000")),
            timeout_seconds=int(os.getenv("SUPABASE_TIMEOUT", "30")),
        )


class SupabaseClient:
    """
    Client for interacting with Supabase PostgreSQL database.

    Uses direct PostgreSQL connection via psycopg2 for raw SQL queries.
    TELOS governance happens BEFORE queries reach this client.
    """

    def __init__(self, config: Optional[SupabaseConfig] = None):
        """Initialize the Supabase client."""
        self.config = config or SupabaseConfig.from_env()
        self._pg_conn = None
        self._connected = False

    def connect(self) -> bool:
        """
        Establish connection to the database.

        Returns:
            True if connection successful, False otherwise.
        """
        if not PSYCOPG2_AVAILABLE:
            logger.error("psycopg2 not installed. Cannot connect to database.")
            return False

        if not self.config.postgres_url:
            logger.error("No PostgreSQL URL configured.")
            return False

        try:
            self._pg_conn = psycopg2.connect(
                self.config.postgres_url,
                connect_timeout=self.config.timeout_seconds,
            )
            self._pg_conn.autocommit = True  # For read-only queries
            self._connected = True
            logger.info("Connected to Supabase PostgreSQL")
            return True
        except Exception as e:
            logger.error(f"PostgreSQL connection failed: {e}")
            return False

    def disconnect(self):
        """Close the database connection."""
        if self._pg_conn:
            self._pg_conn.close()
            self._pg_conn = None
        self._connected = False

    @property
    def is_connected(self) -> bool:
        """Check if connected to database."""
        return self._connected

    def execute_query(
        self,
        query: str,
        params: Optional[Tuple] = None,
    ) -> Dict[str, Any]:
        """
        Execute a SQL query and return results.

        Args:
            query: SQL query string
            params: Optional query parameters (for parameterized queries)

        Returns:
            Dict with 'success', 'data', 'columns', 'row_count', 'error'
        """
        if not self._connected:
            return {
                "success": False,
                "error": "Not connected to database. Call connect() first.",
                "data": [],
                "columns": [],
                "row_count": 0,
            }

        # Safety check: Block write operations unless explicitly allowed
        query_upper = query.strip().upper()
        is_write_operation = any(
            query_upper.startswith(cmd)
            for cmd in ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE", "CREATE"]
        )

        if is_write_operation and not self.config.allow_write:
            return {
                "success": False,
                "error": "Write operations are disabled. Only SELECT queries are allowed.",
                "data": [],
                "columns": [],
                "row_count": 0,
            }

        return self._execute_pg(query, params)

    def _execute_pg(
        self,
        query: str,
        params: Optional[Tuple] = None,
    ) -> Dict[str, Any]:
        """Execute query via psycopg2."""
        try:
            with self._pg_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(query, params)

                # Check if query returns results
                if cur.description:
                    columns = [desc[0] for desc in cur.description]
                    data = cur.fetchmany(self.config.max_rows)
                    # Convert to regular dicts for JSON serialization
                    data = [dict(row) for row in data]
                    row_count = len(data)

                    # Check if there are more rows
                    if cur.fetchone():
                        logger.warning(f"Query returned more than {self.config.max_rows} rows (truncated)")
                else:
                    columns = []
                    data = []
                    row_count = cur.rowcount if cur.rowcount >= 0 else 0

                return {
                    "success": True,
                    "data": data,
                    "columns": columns,
                    "row_count": row_count,
                    "error": None,
                }

        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "data": [],
                "columns": [],
                "row_count": 0,
            }

    def get_tables(self) -> Dict[str, Any]:
        """
        List all tables in the database.

        Returns:
            Dict with table names and metadata.
        """
        query = """
            SELECT
                table_schema,
                table_name,
                table_type
            FROM information_schema.tables
            WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
            ORDER BY table_schema, table_name;
        """
        return self.execute_query(query)

    def get_table_schema(self, table_name: str, schema: str = "public") -> Dict[str, Any]:
        """
        Get schema information for a specific table.

        Args:
            table_name: Name of the table
            schema: Schema name (default: public)

        Returns:
            Dict with column information.
        """
        query = """
            SELECT
                column_name,
                data_type,
                character_maximum_length,
                is_nullable,
                column_default
            FROM information_schema.columns
            WHERE table_schema = %s AND table_name = %s
            ORDER BY ordinal_position;
        """
        return self.execute_query(query, (schema, table_name))

    def get_sample_rows(
        self,
        table_name: str,
        schema: str = "public",
        limit: int = 3,
    ) -> Dict[str, Any]:
        """
        Get sample rows from a table.

        Args:
            table_name: Name of the table
            schema: Schema name (default: public)
            limit: Number of rows to return

        Returns:
            Dict with sample data.
        """
        # Sanitize inputs to prevent SQL injection
        safe_schema = ''.join(c for c in schema if c.isalnum() or c == '_')
        safe_table = ''.join(c for c in table_name if c.isalnum() or c == '_')

        query = f'SELECT * FROM "{safe_schema}"."{safe_table}" LIMIT {min(limit, 10)};'
        return self.execute_query(query)

    def validate_query(self, query: str) -> Dict[str, Any]:
        """
        Validate a SQL query without executing it.

        Uses PostgreSQL's EXPLAIN to check syntax and semantics.

        Args:
            query: SQL query to validate

        Returns:
            Dict with validation result.
        """
        if not self._pg_conn:
            return {
                "valid": False,
                "error": "PostgreSQL connection required for query validation",
            }

        try:
            # Use EXPLAIN to validate without executing
            with self._pg_conn.cursor() as cur:
                cur.execute(f"EXPLAIN {query}")
                return {
                    "valid": True,
                    "error": None,
                }
        except Exception as e:
            return {
                "valid": False,
                "error": str(e),
            }


# Singleton instance for easy access
_client_instance: Optional[SupabaseClient] = None


def get_supabase_client() -> SupabaseClient:
    """Get or create the global Supabase client instance."""
    global _client_instance
    if _client_instance is None:
        _client_instance = SupabaseClient()
    return _client_instance
