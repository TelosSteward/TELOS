"""
Governance Types
================

Types for TELOS governance decisions and traces.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime


class ActionDecision(str, Enum):
    """
    Action decisions for TELOS governance.

    Agentic thresholds are TIGHTER than conversational:
    - EXECUTE: fidelity >= 0.85 - High confidence, proceed
    - CLARIFY: fidelity 0.70-0.84 - Close match, verify first
    - SUGGEST: fidelity 0.50-0.69 - Vague match, offer alternatives
    - INERT: fidelity < 0.50 - No match, acknowledge limitation
    - ESCALATE: fidelity < 0.50 + high_risk - Require human review
    """
    EXECUTE = "execute"
    CLARIFY = "clarify"
    SUGGEST = "suggest"
    INERT = "inert"
    ESCALATE = "escalate"


class GovernanceDecision(BaseModel):
    """Result of a governance check."""
    decision: ActionDecision
    fidelity: float
    raw_similarity: float
    reason: str
    should_forward: bool
    should_modify: bool = False
    modified_content: Optional[str] = None

    # Tool-specific governance
    tool_fidelities: Optional[Dict[str, float]] = None
    blocked_tools: Optional[List[str]] = None


class GovernanceResult(BaseModel):
    """
    Complete governance result for a request.

    Includes input governance, tool governance, and any modifications.
    """
    request_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # PA information
    pa_text: str
    pa_source: str  # "system_prompt", "configured", "inferred"

    # Input governance
    input_fidelity: float
    input_decision: ActionDecision
    input_blocked: bool = False

    # Tool governance (if applicable)
    tools_checked: int = 0
    tools_blocked: int = 0
    tool_decisions: Optional[Dict[str, GovernanceDecision]] = None

    # Response governance (post-LLM)
    response_fidelity: Optional[float] = None
    response_decision: Optional[ActionDecision] = None

    # Overall decision
    final_decision: ActionDecision
    forwarded_to_llm: bool
    governance_response: Optional[str] = None  # If we blocked and responded ourselves


class GovernanceTrace(BaseModel):
    """
    Governance trace entry for audit trail.

    Matches the schema in telos_governance_traces/ for consistency.
    """
    event_type: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: str
    session_id: Optional[str] = None

    # Request info (privacy-safe)
    model: str
    message_count: int
    has_tools: bool
    has_system_prompt: bool

    # Governance metrics
    pa_hash: str  # SHA256 of PA text, not the text itself
    input_fidelity: float
    decision: ActionDecision
    forwarded: bool

    # Tool governance
    tools_total: int = 0
    tools_allowed: int = 0
    tools_blocked: int = 0

    # Response metrics (if applicable)
    response_fidelity: Optional[float] = None
    response_tokens: Optional[int] = None

    # Latency
    governance_latency_ms: int = 0
    total_latency_ms: int = 0

    def to_jsonl(self) -> str:
        """Serialize to JSONL format."""
        import json
        data = self.model_dump()
        data["timestamp"] = data["timestamp"].isoformat()
        return json.dumps(data)
