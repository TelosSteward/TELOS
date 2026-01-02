"""
TELOS LangGraph Adapter
=======================

TELOS (Telically Entrained Linguistic Operational Substrate) governance layer
for LangGraph-based AI agent systems.

This package provides seamless wrapper deployment for governing any LangGraph agent
without modifying its internals - just direct operational connection.

Core Components:
- TelosGovernedState: TypedDict with PA, fidelity trajectory, governance trace
- telos_governance_gate: Pre-execution alignment verification node
- TelosWrapper: Transparent proxy for any existing agent
- TelosSupervisor: TELOS-governed multi-agent supervisor pattern
- TelosSwarm: TELOS-governed agent handoff pattern

Usage:
    from telos_langgraph import TelosWrapper, TelosGovernedState

    # Wrap any existing agent
    governed_agent = TelosWrapper(existing_agent, primacy_attractor=my_pa)

    # Use like normal - TELOS handles governance transparently
    result = governed_agent.invoke({"messages": [user_message]})

Author: TELOS AI Labs Inc.
Contact: JB@telos-labs.ai
"""

from .state_schema import (
    TelosGovernedState,
    PrimacyAttractor,
    GovernanceTraceEntry,
    ActionChainEntry,
    FidelityZone,
    InterventionLevel,
)

from .governance_gate import (
    TelosGovernanceGate,
    telos_governance_node,
    calculate_fidelity,
    get_fidelity_zone,
)

from .wrapper import (
    TelosWrapper,
    telos_wrap,
)

from .supervisor import (
    TelosSupervisor,
    create_telos_supervisor,
)

from .swarm import (
    TelosSwarm,
    create_telos_swarm,
)

__version__ = "0.1.0"
__author__ = "TELOS AI Labs Inc."

__all__ = [
    # State Schema
    "TelosGovernedState",
    "PrimacyAttractor",
    "GovernanceTraceEntry",
    "ActionChainEntry",
    "FidelityZone",
    "InterventionLevel",
    # Governance Gate
    "TelosGovernanceGate",
    "telos_governance_node",
    "calculate_fidelity",
    "get_fidelity_zone",
    # Wrapper
    "TelosWrapper",
    "telos_wrap",
    # Supervisor
    "TelosSupervisor",
    "create_telos_supervisor",
    # Swarm
    "TelosSwarm",
    "create_telos_swarm",
]
