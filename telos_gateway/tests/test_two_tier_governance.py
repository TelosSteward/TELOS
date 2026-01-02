"""
Two-Tier Governance Test Harness
================================

Tests the complete TELOS Gateway two-tier governance:
- Tier 1: Pass-through governance (request vs agent PA)
- Tier 2: Tool selection fidelity (request vs tool descriptions)

This demonstrates "cognition with receipts" - explainable
tool selection backed by semantic similarity.

Usage:
    python -m telos_gateway.tests.test_two_tier_governance
"""

import os
import sys

# Ensure path is set up
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dataclasses import dataclass
from typing import Callable, List, Optional
import numpy as np

# Import the embedding provider
from telos_gateway.providers.embedding_provider import get_embedding_provider

# Import governance modules
from telos_gateway.governance.tool_selection_gate import (
    ToolSelectionGate,
    ToolDefinition,
    SQL_AGENT_TOOLS,
    RESEARCH_AGENT_TOOLS,
    CUSTOMER_SERVICE_TOOLS,
)


@dataclass
class TwoTierResult:
    """Combined result from both governance tiers."""
    user_request: str

    # Tier 1: Pass-through
    tier1_pa: str
    tier1_fidelity: float
    tier1_decision: str
    tier1_should_proceed: bool

    # Tier 2: Tool Selection
    tier2_selected_tool: Optional[str]
    tier2_fidelity: float
    tier2_reasoning: str
    tier2_all_rankings: List[dict]

    def print_report(self):
        """Print a formatted governance report."""
        print("\n" + "=" * 70)
        print("TELOS GATEWAY - TWO-TIER GOVERNANCE REPORT")
        print("=" * 70)

        print(f"\nUser Request: \"{self.user_request}\"")

        print("\n" + "-" * 70)
        print("TIER 1: PASS-THROUGH GOVERNANCE")
        print("-" * 70)
        print(f"Agent PA: \"{self.tier1_pa[:80]}...\"")
        print(f"Fidelity: {self.tier1_fidelity:.2f} ({self._fidelity_to_pct(self.tier1_fidelity)}%)")
        print(f"Decision: {self.tier1_decision}")
        print(f"Proceed to Tier 2: {'YES' if self.tier1_should_proceed else 'NO - BLOCKED'}")

        if self.tier1_should_proceed:
            print("\n" + "-" * 70)
            print("TIER 2: TOOL SELECTION FIDELITY")
            print("-" * 70)
            print(f"Selected Tool: {self.tier2_selected_tool}")
            print(f"Selection Fidelity: {self.tier2_fidelity:.2f} ({self._fidelity_to_pct(self.tier2_fidelity)}%)")
            print(f"Reasoning: {self.tier2_reasoning}")
            print("\nAll Tool Rankings:")
            for tool in self.tier2_all_rankings:
                marker = ">>>" if tool['rank'] == 1 else "   "
                print(f"  {marker} #{tool['rank']}: {tool['tool']} - {tool['display_pct']}% ({tool['fidelity']:.3f})")

        print("\n" + "=" * 70)

    def _fidelity_to_pct(self, fidelity: float) -> int:
        """Convert fidelity to display percentage."""
        if fidelity >= 0.70:
            return 90 + int((fidelity - 0.70) / 0.30 * 10)
        elif fidelity >= 0.60:
            return 80 + int((fidelity - 0.60) / 0.10 * 9)
        elif fidelity >= 0.50:
            return 70 + int((fidelity - 0.50) / 0.10 * 9)
        else:
            return int(fidelity / 0.50 * 69)


class TwoTierGovernor:
    """
    Two-Tier Governance System for Agentic AI.

    Combines:
    - Tier 1: Pass-through check (request alignment with agent purpose)
    - Tier 2: Tool selection (semantic matching to available tools)
    """

    def __init__(self, embed_fn: Callable[[str], np.ndarray]):
        """
        Initialize the two-tier governor.

        Args:
            embed_fn: Function to embed text strings
        """
        self.embed_fn = embed_fn
        self.tool_gate = ToolSelectionGate(embed_fn)

    def check_request(
        self,
        user_request: str,
        agent_pa: str,
        tools: List[ToolDefinition],
        execute_threshold: float = 0.45,
        clarify_threshold: float = 0.35,
        suggest_threshold: float = 0.25,
    ) -> TwoTierResult:
        """
        Run two-tier governance check.

        Args:
            user_request: The user's request
            agent_pa: The agent's Primacy Attractor (purpose statement)
            tools: Available tools for the agent
            execute_threshold: Tier 1 threshold for EXECUTE
            clarify_threshold: Tier 1 threshold for CLARIFY
            suggest_threshold: Tier 1 threshold for SUGGEST

        Returns:
            TwoTierResult with complete governance data
        """
        # === TIER 1: Pass-through Governance ===
        tier1_fidelity = self._calculate_fidelity(user_request, agent_pa)
        tier1_decision, tier1_proceed = self._make_tier1_decision(
            tier1_fidelity,
            execute_threshold,
            clarify_threshold,
            suggest_threshold,
        )

        # === TIER 2: Tool Selection (only if Tier 1 passes) ===
        if tier1_proceed:
            tier2_result = self.tool_gate.select_tool(user_request, tools)
            tier2_rankings = tier2_result.to_dict()["tool_rankings"]
        else:
            tier2_result = None
            tier2_rankings = []

        return TwoTierResult(
            user_request=user_request,
            tier1_pa=agent_pa,
            tier1_fidelity=tier1_fidelity,
            tier1_decision=tier1_decision,
            tier1_should_proceed=tier1_proceed,
            tier2_selected_tool=tier2_result.selected_tool if tier2_result else None,
            tier2_fidelity=tier2_result.selected_fidelity if tier2_result else 0.0,
            tier2_reasoning=tier2_result.selection_reasoning if tier2_result else "Blocked at Tier 1",
            tier2_all_rankings=tier2_rankings,
        )

    def _calculate_fidelity(self, text: str, pa: str) -> float:
        """Calculate fidelity between text and PA.

        Mistral embeddings produce narrow discriminative range:
        - Off-topic content: 0.55-0.65 raw similarity
        - On-topic content: 0.70-0.80 raw similarity

        We map this to TELOS fidelity zones:
        - < 0.55: Clearly off-topic → RED (0.0-0.30)
        - 0.55-0.70: Ambiguous/drift → YELLOW/ORANGE (0.30-0.70)
        - > 0.70: On-topic → GREEN (0.70-1.0)
        """
        text_emb = self.embed_fn(text)
        pa_emb = self.embed_fn(pa)

        # Raw cosine similarity
        similarity = np.dot(text_emb, pa_emb) / (np.linalg.norm(text_emb) * np.linalg.norm(pa_emb))

        # Mistral-calibrated normalization
        # Empirically: off-topic floor ~0.55, on-topic threshold ~0.70
        MISTRAL_FLOOR = 0.55      # Below this = clearly unrelated
        MISTRAL_ALIGNED = 0.70    # Above this = clearly on-topic

        if similarity < MISTRAL_FLOOR:
            # Map 0.0-0.55 → 0.0-0.30 (RED zone)
            fidelity = (similarity / MISTRAL_FLOOR) * 0.30
        elif similarity < MISTRAL_ALIGNED:
            # Map 0.55-0.70 → 0.30-0.70 (YELLOW/ORANGE zone)
            fidelity = 0.30 + ((similarity - MISTRAL_FLOOR) / (MISTRAL_ALIGNED - MISTRAL_FLOOR)) * 0.40
        else:
            # Map 0.70-1.0 → 0.70-1.0 (GREEN zone)
            fidelity = 0.70 + ((similarity - MISTRAL_ALIGNED) / (1.0 - MISTRAL_ALIGNED)) * 0.30

        return float(min(1.0, max(0.0, fidelity)))

    def _make_tier1_decision(
        self,
        fidelity: float,
        execute_threshold: float,
        clarify_threshold: float,
        suggest_threshold: float,
    ) -> tuple:
        """Make Tier 1 pass/block decision."""
        if fidelity >= execute_threshold:
            return "EXECUTE", True
        elif fidelity >= clarify_threshold:
            return "CLARIFY", True
        elif fidelity >= suggest_threshold:
            return "SUGGEST", True
        else:
            return "INERT", False


# =============================================================================
# Test Cases
# =============================================================================

# SQL Agent PA
SQL_AGENT_PA = """You are a database assistant specialized in querying and analyzing SQL databases.
You help users understand their data by executing queries, exploring table structures,
and providing insights from database information. You do NOT help with unrelated topics
like weather, cooking, or general conversation."""

# Research Agent PA
RESEARCH_AGENT_PA = """You are a research assistant specialized in finding, analyzing, and
synthesizing information from multiple sources. You help users conduct thorough research
by searching the web, consulting encyclopedic sources, and performing calculations
to support data-driven conclusions."""

# Customer Service Agent PA
CUSTOMER_SERVICE_PA = """You are a customer service agent for an e-commerce platform.
You help customers with order inquiries, process refunds when appropriate, answer
product questions from the FAQ, and escalate complex issues to human agents."""


def run_sql_agent_tests(governor: TwoTierGovernor):
    """Run test cases for SQL Agent."""
    print("\n" + "#" * 70)
    print("# SQL AGENT TEST SUITE")
    print("#" * 70)

    test_cases = [
        # Should pass Tier 1 and select correct tool
        "What tables are available in the database?",
        "Show me the schema for the users table",
        "Run a query to get all orders from last month",
        "Check if this SQL is valid: SELECT * FROM users WHERE id = 1",

        # Should pass Tier 1 but tool selection is ambiguous
        "Tell me about the data",

        # Should FAIL Tier 1 - off topic
        "What's the weather like today?",
        "Can you help me write a poem?",
    ]

    for request in test_cases:
        result = governor.check_request(
            user_request=request,
            agent_pa=SQL_AGENT_PA,
            tools=SQL_AGENT_TOOLS,
        )
        result.print_report()


def run_research_agent_tests(governor: TwoTierGovernor):
    """Run test cases for Research Agent."""
    print("\n" + "#" * 70)
    print("# RESEARCH AGENT TEST SUITE")
    print("#" * 70)

    test_cases = [
        "Search for information about climate change impacts",
        "What does Wikipedia say about quantum computing?",
        "Calculate the compound interest on $10,000 at 5% for 10 years",
        "Summarize the key findings from this research",
    ]

    for request in test_cases:
        result = governor.check_request(
            user_request=request,
            agent_pa=RESEARCH_AGENT_PA,
            tools=RESEARCH_AGENT_TOOLS,
        )
        result.print_report()


def run_customer_service_tests(governor: TwoTierGovernor):
    """Run test cases for Customer Service Agent."""
    print("\n" + "#" * 70)
    print("# CUSTOMER SERVICE AGENT TEST SUITE")
    print("#" * 70)

    test_cases = [
        "Where is my order #12345?",
        "I want a refund for my recent purchase",
        "What is your return policy?",
        "This is unacceptable, I want to speak to a manager",
    ]

    for request in test_cases:
        result = governor.check_request(
            user_request=request,
            agent_pa=CUSTOMER_SERVICE_PA,
            tools=CUSTOMER_SERVICE_TOOLS,
        )
        result.print_report()


def main():
    """Run the two-tier governance test suite."""
    print("\n" + "=" * 70)
    print("TELOS GATEWAY - TWO-TIER GOVERNANCE TEST HARNESS")
    print("=" * 70)
    print("\nInitializing embedding provider...")

    # Get embedding function
    provider = get_embedding_provider()
    embed_fn = provider.embed

    print(f"Using embedding model: {provider.__class__.__name__}")

    # Create governor
    governor = TwoTierGovernor(embed_fn)

    # Pre-register tools for efficiency
    governor.tool_gate.register_tools(SQL_AGENT_TOOLS)
    governor.tool_gate.register_tools(RESEARCH_AGENT_TOOLS)
    governor.tool_gate.register_tools(CUSTOMER_SERVICE_TOOLS)

    print("Tools registered and embedded.")

    # Run tests
    run_sql_agent_tests(governor)

    # Uncomment to run additional tests:
    # run_research_agent_tests(governor)
    # run_customer_service_tests(governor)

    print("\n" + "=" * 70)
    print("TEST SUITE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
