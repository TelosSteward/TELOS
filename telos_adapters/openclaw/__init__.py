"""
TELOS Governance Adapter for OpenClaw
======================================

Governs OpenClaw autonomous AI agents via the before_tool_call plugin hook.
Every tool call is scored by the TELOS 4-layer cascade (L0 keyword, L1 cosine,
L1.5 SetFit, L2 LLM) with ~15ms total latency.

Integration pattern:
    OpenClaw (TypeScript) -> UDS IPC -> Python governance daemon -> score -> allow/block

Components:
    - GovernanceHook: Core scoring bridge (receives actions, returns decisions)
    - ActionClassifier: Maps OpenClaw tool names to TELOS categories
    - ConfigLoader: Loads openclaw.yaml governance config
    - IPCServer: Unix Domain Socket server (NDJSON protocol)
    - Watchdog: Process lifecycle (heartbeat, auto-restart, SIGTERM)

Regulatory compliance:
    - EU AI Act: Articles 9 (risk management), 12 (record-keeping), 14 (human
      oversight), 15 (robustness), 72 (post-market monitoring), 73 (incident reporting)
    - IEEE 7000-2021: PA as Ethical Value Register, tool group risk mapping
    - IEEE 7001-2021: Real-time transparency via GovernanceVerdict scoring breakdown
    - SAAI: Claims TELOS-SAAI-009 through 014 (autonomous agent governance)
    - NIST AI RMF: GOVERN, MAP, MEASURE, MANAGE functions implemented
    - OWASP Agentic Top 10 (2026): 8/10 risks covered (ASI01-ASI05, ASI08-ASI10)
    See: research/openclaw_regulatory_mapping.md
"""

from telos_adapters.openclaw.governance_hook import GovernanceHook
from telos_adapters.openclaw.action_classifier import ActionClassifier, ToolGroupRiskTier
from telos_adapters.openclaw.config_loader import OpenClawConfigLoader
from telos_adapters.openclaw.ipc_server import IPCServer, IPCMessage, IPCResponse
from telos_adapters.openclaw.notification_service import (
    NotificationService,
    EscalationNotification,
)
from telos_adapters.openclaw.permission_controller import (
    PermissionController,
    EscalationResult,
)

__all__ = [
    "GovernanceHook",
    "ActionClassifier",
    "ToolGroupRiskTier",
    "OpenClawConfigLoader",
    "IPCServer",
    "IPCMessage",
    "IPCResponse",
    "NotificationService",
    "EscalationNotification",
    "PermissionController",
    "EscalationResult",
]
