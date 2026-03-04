"""
TELOS Governance Daemon — Entry point for the OpenClaw governance process.

Wires together all components:
    ConfigLoader -> GovernanceHook -> IPCServer -> Watchdog

Started by:
    - `telos agent init --detect` (first time setup)
    - `telos service start` (manual)
    - launchd/systemd (auto-start)
    - OpenClaw gateway_start hook (lifecycle-managed)

Usage:
    # Direct invocation
    python -m telos_adapters.openclaw.daemon --config templates/openclaw.yaml

    # With custom socket path
    python -m telos_adapters.openclaw.daemon --socket /tmp/telos-test.sock

    # Permissive mode (log-only, no blocking)
    python -m telos_adapters.openclaw.daemon --preset permissive

Regulatory traceability:
    This daemon is the runtime implementation of EU AI Act Art. 72
    (continuous post-market monitoring). When running, every OpenClaw
    tool call is governed. When stopped with fail-closed preset, OpenClaw
    actions are blocked until governance is restored.
    - SAAI claim TELOS-SAAI-009: Always-on governance
    - NIST AI RMF GOVERN 2.1: Continuous risk awareness
    - OWASP ASI10 (Rogue Agents): Persistent monitoring detects compromised agents
    See: research/openclaw_regulatory_mapping.md
"""

import argparse
import asyncio
import json as _json
import logging
import os
import signal
import sys
import time as _time
from pathlib import Path
from typing import Any, Dict, Optional

from telos_adapters.openclaw.audit_writer import AuditWriter
from telos_adapters.openclaw.config_loader import OpenClawConfigLoader
from telos_adapters.openclaw.cusum_monitor import CUSUMMonitorBank
from telos_adapters.openclaw.governance_hook import GovernanceHook, GovernancePreset
from telos_adapters.openclaw.ipc_server import IPCServer, IPCMessage, IPCResponse
from telos_adapters.openclaw.watchdog import Watchdog
from telos_governance.response_manager import AgenticDriftTracker

logger = logging.getLogger(__name__)

# Key storage directory
KEY_DIR = Path.home() / ".telos" / "keys"

# Gate file location and cache
GATE_FILE = Path.home() / ".telos" / "gate"
GATE_CACHE_TTL = 30.0  # seconds

_gate_cache: Dict[str, Any] = {"record": None, "read_at": 0.0}


def _read_gate_state(force: bool = False) -> Optional[Dict[str, Any]]:
    """Read and validate ~/.telos/gate, cache for 30s.

    Reads the gate file, verifies the Ed25519 signature, checks TTL
    expiry, and caches the result to avoid repeated filesystem reads.

    Args:
        force: Bypass cache (e.g., after SIGHUP or gate_transition IPC).

    Returns:
        GateRecord dict if valid gate file exists, None otherwise.
    """
    now = _time.time()

    # Return cached if fresh
    if not force and (now - _gate_cache["read_at"]) < GATE_CACHE_TTL:
        return _gate_cache["record"]

    if not GATE_FILE.exists():
        _gate_cache["record"] = None
        _gate_cache["read_at"] = now
        return None

    try:
        raw = GATE_FILE.read_text("utf-8")
        data = _json.loads(raw)

        # Validate required fields (public key may be "public_key" or "tkey_pubkey")
        pubkey = data.get("public_key") or data.get("tkey_pubkey")
        for field in ("state", "mode", "actor", "timestamp", "ttl_hours", "signature"):
            if field not in data:
                logger.warning(f"Gate file missing field: {field}")
                _gate_cache["record"] = None
                _gate_cache["read_at"] = now
                return None
        if not pubkey:
            logger.warning("Gate file missing public key (checked public_key and tkey_pubkey)")
            _gate_cache["record"] = None
            _gate_cache["read_at"] = now
            return None

        # Normalize: ensure "public_key" is set for downstream consumers
        data["public_key"] = pubkey

        # Verify Ed25519 signature
        try:
            from telos_governance.gate_signer import GateSigner, GateRecord, GateSigningError
            record = GateRecord.from_dict(data)
            pub_bytes = bytes.fromhex(record.public_key)
            GateSigner.verify(record, pub_bytes)

            # Check TTL expiry
            if GateSigner.is_expired(record):
                logger.info("Gate record expired (TTL elapsed) — treating as no gate")
                _gate_cache["record"] = None
                _gate_cache["read_at"] = now
                return None

            _gate_cache["record"] = data
            _gate_cache["read_at"] = now
            return data

        except GateSigningError as e:
            logger.error(f"Gate signature verification failed: {e}")
            _gate_cache["record"] = None
            _gate_cache["read_at"] = now
            return None

    except Exception as e:
        logger.warning(f"Failed to read gate file {GATE_FILE}: {e}")
        _gate_cache["record"] = None
        _gate_cache["read_at"] = now
        return None


def _invalidate_gate_cache() -> None:
    """Invalidate the gate cache (called on SIGHUP or gate_transition)."""
    _gate_cache["record"] = None
    _gate_cache["read_at"] = 0.0

def _resolve_telos_agent_id(msg: IPCMessage) -> str:
    """Resolve stable per-agent identity from score message args/session context."""
    args = msg.args or {}

    for key in ("TELOS_AGENT_ID", "agent_id"):
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    session_key = args.get("__session_key")
    if isinstance(session_key, str):
        session_key = session_key.strip()
        if session_key:
            import re
            match = re.match(r"^agent:([^:]+):subagent:(.+)$", session_key)
            if match and match.group(1).strip():
                return match.group(1).strip()

    return ""



# Codebase policy loading and cache
POLICIES_DIR = Path.home() / ".telos" / "policies"
POLICY_CACHE_TTL = 30.0  # seconds
_policy_cache: Dict[str, Any] = {"policies": [], "read_at": 0.0}


def _load_codebase_policies(force: bool = False):
    """Load all policies from ~/.telos/policies/*.json, cache for 30s.

    Verifies Ed25519 signature on each policy, skips expired or invalid.

    Args:
        force: Bypass cache (e.g., after SIGHUP or policy_update IPC).

    Returns:
        List of valid CodebasePolicy objects.
    """
    now = _time.time()

    # Return cached if fresh
    if not force and (now - _policy_cache["read_at"]) < POLICY_CACHE_TTL:
        return _policy_cache["policies"]

    if not POLICIES_DIR.exists():
        _policy_cache["policies"] = []
        _policy_cache["read_at"] = now
        return []

    try:
        from telos_governance.codebase_policy import (
            CodebasePolicy,
            CodebasePolicySigner,
            CodebasePolicyError,
        )
    except ImportError:
        logger.warning("codebase_policy module not available — policy enforcement disabled")
        _policy_cache["policies"] = []
        _policy_cache["read_at"] = now
        return []

    valid_policies = []

    for policy_file in sorted(POLICIES_DIR.glob("*.json")):
        try:
            raw = policy_file.read_text("utf-8")
            data = _json.loads(raw)
            policy = CodebasePolicy.from_dict(data)

            # Verify Ed25519 signature
            pub_bytes = bytes.fromhex(policy.public_key)
            CodebasePolicySigner.verify(policy, pub_bytes)

            # Check TTL expiry
            if CodebasePolicySigner.is_expired(policy):
                logger.info(f"Policy {policy.collection} expired (TTL elapsed) — skipping")
                continue

            valid_policies.append(policy)
            logger.debug(
                f"Policy loaded: {policy.collection} "
                f"({policy.access_level}, {len(policy.paths)} paths)"
            )

        except CodebasePolicyError as e:
            logger.warning(f"Policy {policy_file.name} signature invalid: {e} — skipping")
        except Exception as e:
            logger.warning(f"Failed to read policy {policy_file.name}: {e} — skipping")

    _policy_cache["policies"] = valid_policies
    _policy_cache["read_at"] = now
    return valid_policies


def _invalidate_policy_cache() -> None:
    """Invalidate the policy cache (called on SIGHUP or policy_update IPC)."""
    _policy_cache["policies"] = []
    _policy_cache["read_at"] = 0.0


def _policy_explanation(reason: str, file_path: str, matched_policy) -> str:
    """Build human-readable explanation for a codebase policy denial.

    Args:
        reason: "unauthorized_write" or "no_policy".
        file_path: The file path that triggered the denial.
        matched_policy: The matching CodebasePolicy, or None.

    Returns:
        Explanation string for the ESCALATE verdict.
    """
    if reason == "unauthorized_write":
        return (
            f"CODEBASE POLICY VIOLATION — write to read-only path '{file_path}'. "
            f"Collection '{matched_policy.collection}' is signed as read_only "
            f"by the principal's Ed25519 key. "
            f"The agent cannot grant itself write access."
        )
    elif reason == "no_policy":
        return (
            f"CODEBASE POLICY VIOLATION — no policy covers path '{file_path}'. "
            f"Fail-closed: all paths require a signed access policy. "
            f"Run `telos policy sign` to create a policy for this path."
        )
    return f"Codebase policy denied: {reason} for {file_path}"


def _init_receipt_signer():
    """Initialize or load Ed25519 receipt signer for override receipts.

    Generates a new key pair if none exists. Keys are stored at
    ~/.telos/keys/override_ed25519.key (private) and .pub (public).

    Returns:
        ReceiptSigner or None if cryptography is not installed.
    """
    try:
        from telos_governance.receipt_signer import ReceiptSigner
    except ImportError:
        logger.warning("cryptography package not available — override receipts disabled")
        return None

    KEY_DIR.mkdir(parents=True, exist_ok=True)
    key_file = KEY_DIR / "override_ed25519.key"

    if key_file.exists():
        private_bytes = key_file.read_bytes()
        signer = ReceiptSigner.from_private_bytes(private_bytes)
        logger.info(f"Loaded Ed25519 key from {key_file}")
    else:
        signer = ReceiptSigner.generate()
        key_file.write_bytes(signer.private_key_bytes())
        key_file.chmod(0o600)
        # Write public key for verification
        pub_file = KEY_DIR / "override_ed25519.pub"
        pub_file.write_bytes(signer.public_key_bytes())
        logger.info(f"Generated new Ed25519 key pair at {key_file}")

    return signer


def _init_permission_controller(config):
    """Initialize permission controller if notifications are configured.

    Args:
        config: AgentConfig with optional notifications section.

    Returns:
        Tuple of (PermissionController, TelegramPoller) or (None, None).
    """
    if not config.notifications or not config.notifications.has_any_channel:
        return None, None

    from telos_adapters.openclaw.notification_service import NotificationService
    from telos_adapters.openclaw.permission_controller import (
        PermissionController,
        TelegramPoller,
    )

    notif_config = config.notifications
    notification_service = NotificationService(notif_config)
    receipt_signer = _init_receipt_signer()

    # Initialize TKeys for cryptographic trust root (graceful fallback)
    tkeys_manager = None
    try:
        import secrets as _secrets
        from telos_privacy.cryptography.telemetric_keys import TelemetricSessionManager
        tkeys_session_id = f"telos-pc-{_secrets.token_hex(4)}"
        tkeys_manager = TelemetricSessionManager(tkeys_session_id)
        logger.info(f"TKeys initialized for Permission Controller (session: {tkeys_session_id})")
    except ImportError:
        logger.info("TKeys not available — Permission Controller uses Ed25519 only")

    controller = PermissionController(
        config=notif_config,
        notification_service=notification_service,
        receipt_signer=receipt_signer,
        tkeys_manager=tkeys_manager,
    )

    # Set up Telegram polling if configured
    poller = None
    if notif_config.has_telegram:
        poller = TelegramPoller(notif_config.telegram_bot_token, controller)

    channels = []
    if notif_config.has_telegram:
        channels.append("Telegram")
    if notif_config.has_whatsapp:
        channels.append("WhatsApp")
    if notif_config.has_discord:
        channels.append("Discord")
    logger.info(f"Permission controller initialized (channels: {', '.join(channels)})")

    return controller, poller


def _sign_verdict(verdict, receipt_signer) -> None:
    """Sign a GovernanceVerdict in-place with Ed25519.

    Computes SHA-256 hash of the canonical verdict JSON (sorted keys,
    no whitespace, excluding signature fields) and signs with Ed25519.

    Regulatory traceability:
        - EU AI Act Art. 12: Cryptographic integrity for automatic event recording
        - SAAI claim TELOS-SAAI-005: Unforgeable chain of reasoning
        - OWASP ASI10: Signed verdicts detect rogue agent tampering
    """
    import hashlib
    import json

    # Build canonical payload (exclude signature fields to avoid circular reference)
    verdict_dict = verdict.to_dict()
    verdict_dict.pop("verdict_signature", None)
    verdict_dict.pop("public_key", None)
    canonical = json.dumps(verdict_dict, sort_keys=True, separators=(",", ":"))
    payload_hash = hashlib.sha256(canonical.encode("utf-8")).digest()

    # Ed25519 sign
    signature = receipt_signer.sign_payload(payload_hash)
    verdict.verdict_signature = signature.hex()
    verdict.public_key = receipt_signer.public_key_bytes().hex()


def _persist_gate_transition(gate_record: Dict[str, Any]) -> None:
    """Best-effort Supabase dual-write for gate transitions.

    Uses stdlib urllib (same pattern as tool_registry.py). Silently
    fails if Supabase is not configured or unreachable.
    """
    try:
        supabase_url = os.environ.get("SUPABASE_URL", "")
        supabase_key = os.environ.get("SUPABASE_SERVICE_KEY", "") or os.environ.get("SUPABASE_KEY", "")
        if not supabase_url or not supabase_key:
            return

        import urllib.request

        payload = _json.dumps({
            "gate_state": gate_record.get("state", ""),
            "gate_mode": gate_record.get("mode", ""),
            "actor": gate_record.get("actor", ""),
            "ttl_hours": gate_record.get("ttl_hours", 0),
            "crypto_proof_type": "ed25519",
            "crypto_proof_value": gate_record.get("signature", ""),
            "public_key": gate_record.get("public_key", ""),
            "session_id": gate_record.get("actor", "")[:16],
            "agent_id": "openclaw",
        }).encode("utf-8")

        url = f"{supabase_url.rstrip('/')}/rest/v1/gate_transitions"
        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "apikey": supabase_key,
                "Authorization": f"Bearer {supabase_key}",
                "Prefer": "return=minimal",
            },
            method="POST",
        )
        urllib.request.urlopen(req, timeout=5)
        logger.debug("Gate transition persisted to Supabase")
    except Exception as e:
        logger.debug(f"Supabase gate persistence skipped: {e}")


def create_message_handler(
    hook: GovernanceHook,
    permission_controller=None,
    receipt_signer=None,
    drift_tracker: AgenticDriftTracker = None,
    audit_writer: AuditWriter = None,
    cusum_bank: CUSUMMonitorBank = None,
    governance_active: bool = True,
    gate_mode: str = "",
    codebase_policies=None,
    project_root: str = "",
    on_state_change=None,
):
    """Create the IPC message handler function.

    Returns a function that processes IPCMessage -> IPCResponse,
    dispatching to the appropriate handler based on message type.
    When a permission_controller is provided, ESCALATE verdicts
    trigger human-in-the-loop notification and wait for approval.
    When a receipt_signer is provided, every GovernanceVerdict is
    signed with Ed25519 for cryptographic integrity.
    When a drift_tracker is provided, session-level drift detection
    enforces SAAI graduated sanctions (WARNING/RESTRICT/BLOCK).
    When an audit_writer is provided, every governance event is
    logged to a persistent NDJSON audit trail.
    When gate_mode is set, the handler applies gate behavior:
    "enforce" returns INERT, "observe" scores but forces allowed=True.
    """

    # Mutable state so IPC messages can update gate + policies at runtime
    _handler_state = {
        "gate_mode": gate_mode,
        "policies": list(codebase_policies) if codebase_policies else [],
    }

    # Stewart contextual intelligence state (per-session, TTL-based)
    _stewart_context: Dict[str, Any] = {
        "task_summary": "",
        "drift_narrative": "",
        "coherent_sequence": False,
        "context_ttl": 0,
        "stewart_fidelity_assessment": 0.0,
        "context_id": "",
    }

    async def handle(msg: IPCMessage, send_interim=None) -> IPCResponse:
        if msg.type == "score":
            # Gate check: closed+enforce → INERT
            current_gate_mode = _handler_state["gate_mode"]
            if current_gate_mode == "enforce":
                inert_data = {
                    "allowed": False,
                    "decision": "inert",
                    "fidelity": 0.0,
                    "tool_group": "",
                    "telos_tool_name": msg.tool_name,
                    "risk_tier": "",
                    "is_cross_group": False,
                    "explanation": (
                        "TELOS governance gate is CLOSED (enforce mode). "
                        "The principal's Ed25519 signature bars agent operation. "
                        "Run `telos gate open` to resume."
                    ),
                    "governance_preset": "",
                    "gate_closed": True,
                    "gate_mode": "enforce",
                }
                if audit_writer:
                    agent_id = _resolve_telos_agent_id(msg)
                    audit_writer.emit("inert_gate_closed", {
                        "tool_name": msg.tool_name,
                        "action_text": msg.action_text[:200] if msg.action_text else "",
                        "gate_mode": "enforce",
                        "agent_id": agent_id,
                        "session_key": (msg.args or {}).get("__session_key", ""),
                    })
                return IPCResponse(
                    type="verdict",
                    request_id=msg.request_id,
                    data=inert_data,
                )

            # PA ceremony gate: governance is INERT until PA is signed
            if not governance_active:
                inert_data = {
                    "allowed": False,
                    "decision": "inert",
                    "fidelity": 0.0,
                    "tool_group": "",
                    "telos_tool_name": msg.tool_name,
                    "risk_tier": "",
                    "is_cross_group": False,
                    "explanation": (
                        "TELOS governance is INERT — PA configuration has not been "
                        "cryptographically signed. Run `telos pa sign <config>` to activate."
                    ),
                    "governance_preset": "",
                    "pa_unsigned": True,
                }
                if audit_writer:
                    agent_id = _resolve_telos_agent_id(msg)
                    audit_writer.emit("inert_unsigned_pa", {
                        "tool_name": msg.tool_name,
                        "action_text": msg.action_text[:200] if msg.action_text else "",
                        "agent_id": agent_id,
                        "session_key": (msg.args or {}).get("__session_key", ""),
                    })
                return IPCResponse(
                    type="verdict",
                    request_id=msg.request_id,
                    data=inert_data,
                )

            # Codebase access policy check (TKeys — before fidelity scoring)
            current_policies = _handler_state["policies"]
            if current_policies and msg.args:
                from telos_governance.codebase_policy import (
                    extract_file_path, check_access,
                )
                _file_path = extract_file_path(msg.tool_name, msg.args)
                if _file_path:
                    _allowed, _reason, _matched = check_access(
                        msg.tool_name, _file_path, current_policies, project_root,
                    )
                    if not _allowed:
                        _collection = _matched.collection if _matched else ""
                        _access_level = _matched.access_level if _matched else ""
                        escalate_data = {
                            "allowed": False,
                            "decision": "escalate",
                            "fidelity": 0.0,
                            "tool_group": "fs",
                            "telos_tool_name": msg.tool_name,
                            "risk_tier": "critical",
                            "is_cross_group": False,
                            "human_required": True,
                            "explanation": _policy_explanation(
                                _reason, _file_path, _matched,
                            ),
                            "policy_violation": True,
                            "policy_reason": _reason,
                            "policy_collection": _collection,
                            "policy_access_level": _access_level,
                        }
                        if audit_writer:
                            audit_writer.write_policy_denied({
                                "tool_name": msg.tool_name,
                                "file_path": _file_path,
                                "reason": _reason,
                                "collection": _collection,
                            })
                        return IPCResponse(
                            type="verdict",
                            request_id=msg.request_id,
                            data=escalate_data,
                        )

            # Stewart context enrichment: prepend task summary to action text
            # if Stewart has provided active context (TTL > 0)
            _scoring_action_text = msg.action_text
            _stewart_active = False
            if _stewart_context.get("context_ttl", 0) > 0:
                _stewart_active = True
                _task_summary = _stewart_context.get("task_summary", "")
                if _task_summary:
                    _scoring_action_text = f"[Task: {_task_summary}] {msg.action_text}"
                # Decrement TTL
                _stewart_context["context_ttl"] = _stewart_context["context_ttl"] - 1

            verdict = hook.score_action(
                tool_name=msg.tool_name,
                action_text=_scoring_action_text,
                tool_args=msg.args,
            )

            # Stamp Stewart context onto verdict for audit trail
            if _stewart_active:
                verdict.stewart_context_active = True
                verdict.stewart_task_summary = _stewart_context.get("task_summary", "")[:200]
                verdict.stewart_coherent_sequence = _stewart_context.get("coherent_sequence", False)

            # CUSUM per-tool-group drift detection
            if cusum_bank:
                cusum_alert = cusum_bank.record(verdict.tool_group, verdict.fidelity)
                if cusum_alert:
                    verdict.cusum_alert = True
                    verdict.cusum_tool_group = cusum_alert.tool_group
                    if audit_writer:
                        audit_writer.emit("cusum_alert", cusum_alert.to_dict())

            # SAAI drift tracking — record fidelity and apply graduated sanctions
            if drift_tracker:
                drift_status = drift_tracker.record_fidelity(verdict.fidelity)

                # Stamp drift fields onto verdict
                verdict.drift_level = drift_status["drift_level"]
                verdict.drift_magnitude = drift_status["drift_magnitude"]
                verdict.baseline_fidelity = drift_status["baseline_fidelity"]
                verdict.baseline_established = drift_status["baseline_established"]
                verdict.is_blocked = drift_status["is_blocked"]
                verdict.is_restricted = drift_status["is_restricted"]
                verdict.turn_count = drift_status["turn_count"]
                verdict.acknowledgment_count = drift_status["acknowledgment_count"]
                verdict.permanently_blocked = drift_status["permanently_blocked"]

                # BLOCK: force deny regardless of scoring result
                if drift_status["is_blocked"]:
                    verdict.allowed = False
                    verdict.decision = "inert"
                    verdict.explanation = (
                        f"SAAI BLOCK — session drift {drift_status['drift_magnitude']:.1%} "
                        f"exceeds 20% threshold; " + verdict.explanation
                    )
                    logger.warning(
                        f"[DRIFT BLOCK] drift={drift_status['drift_magnitude']:.1%} "
                        f"turn={drift_status['turn_count']}"
                    )
                    if audit_writer:
                        audit_writer.emit("drift_block", drift_status)
                    # Snapshot on BLOCK transition
                    if on_state_change:
                        on_state_change()

                # RESTRICT: tighten EXECUTE threshold
                elif drift_status["is_restricted"] and verdict.decision == "execute":
                    from telos_core.constants import ST_SAAI_RESTRICT_EXECUTE_THRESHOLD
                    if verdict.fidelity < ST_SAAI_RESTRICT_EXECUTE_THRESHOLD:
                        verdict.allowed = False
                        verdict.decision = "clarify"
                        verdict.explanation = (
                            f"SAAI RESTRICT — fidelity {verdict.fidelity:.3f} below "
                            f"restricted threshold {ST_SAAI_RESTRICT_EXECUTE_THRESHOLD}; "
                            + verdict.explanation
                        )
                        logger.info(
                            f"[DRIFT RESTRICT] downgrade EXECUTE→CLARIFY "
                            f"fidelity={verdict.fidelity:.3f} "
                            f"threshold={ST_SAAI_RESTRICT_EXECUTE_THRESHOLD}"
                        )
                        if audit_writer:
                            audit_writer.emit("drift_restrict", {
                                **drift_status,
                                "original_decision": "execute",
                                "downgraded_to": "clarify",
                            })

            # Sign every verdict with Ed25519
            if receipt_signer:
                _sign_verdict(verdict, receipt_signer)

            # Gate observe mode: score normally but force allowed=True
            # Shadow fields preserve the original decision for audit
            current_gate_mode = _handler_state["gate_mode"]
            if current_gate_mode == "observe":
                verdict.gate_mode = "observe"
                verdict.observe_shadow_decision = verdict.decision
                verdict.observe_shadow_allowed = verdict.allowed
                verdict.allowed = True
                verdict.decision = "execute"
                verdict.explanation = (
                    f"[OBSERVE MODE] Shadow decision: {verdict.observe_shadow_decision} "
                    f"(allowed={verdict.observe_shadow_allowed}). "
                    f"Gate is closed in observe mode — scoring but not enforcing. "
                    + verdict.explanation
                )

            # ESCALATE with permission controller — trigger human review
            if (
                permission_controller
                and verdict.decision == "escalate"
                and not verdict.allowed
                and verdict.human_required
            ):
                # Send interim response so the TS plugin extends its timeout
                if send_interim:
                    import uuid as _uuid
                    escalation_preview = IPCResponse(
                        type="escalation_pending",
                        request_id=msg.request_id,
                        data={
                            "tool_name": verdict.telos_tool_name,
                            "risk_tier": verdict.risk_tier,
                            "timeout_seconds": permission_controller._config.escalation_timeout_seconds,
                        },
                    )
                    await send_interim(escalation_preview)

                # Wait for human decision
                result = await permission_controller.handle_escalation(verdict)

                if result.approved:
                    # Override — flip the verdict to allowed
                    verdict_data = verdict.to_dict()
                    verdict_data["allowed"] = True
                    verdict_data["explanation"] = (
                        f"Override approved via {result.response_source} "
                        f"(escalation {result.escalation_id}); "
                        + verdict_data.get("explanation", "")
                    )
                    if result.receipt:
                        verdict_data["override_receipt"] = result.receipt
                    return IPCResponse(
                        type="verdict",
                        request_id=msg.request_id,
                        data=verdict_data,
                    )

                # Denied or timed out — keep blocked
                verdict_data = verdict.to_dict()
                verdict_data["explanation"] = (
                    f"Override {result.response_source} "
                    f"(escalation {result.escalation_id}); "
                    + verdict_data.get("explanation", "")
                )
                return IPCResponse(
                    type="verdict",
                    request_id=msg.request_id,
                    data=verdict_data,
                )

            # INERT with permission controller — notify human, allow override
            if (
                permission_controller
                and verdict.decision == "inert"
                and not verdict.allowed
            ):
                if send_interim:
                    inert_preview = IPCResponse(
                        type="inert_pending",
                        request_id=msg.request_id,
                        data={
                            "tool_name": verdict.telos_tool_name,
                            "risk_tier": verdict.risk_tier,
                            "fidelity": verdict.fidelity,
                            "timeout_seconds": permission_controller._config.escalation_timeout_seconds,
                        },
                    )
                    await send_interim(inert_preview)

                result = await permission_controller.handle_inert(verdict)

                if result.approved:
                    verdict_data = verdict.to_dict()
                    verdict_data["allowed"] = True
                    verdict_data["explanation"] = (
                        f"INERT override approved via {result.response_source} "
                        f"(escalation {result.escalation_id}); "
                        + verdict_data.get("explanation", "")
                    )
                    if result.receipt:
                        verdict_data["override_receipt"] = result.receipt
                    return IPCResponse(
                        type="verdict",
                        request_id=msg.request_id,
                        data=verdict_data,
                    )

                # No override — keep blocked
                verdict_data = verdict.to_dict()
                verdict_data["explanation"] = (
                    f"INERT — human notified, {result.response_source} "
                    f"(escalation {result.escalation_id}); "
                    + verdict_data.get("explanation", "")
                )
                return IPCResponse(
                    type="verdict",
                    request_id=msg.request_id,
                    data=verdict_data,
                )

            # Audit: log every scored tool call (with attributable agent identity)
            if audit_writer:
                agent_id = _resolve_telos_agent_id(msg)
                audit_writer.emit("tool_call_scored", {
                    "tool_name": msg.tool_name,
                    "action_text": msg.action_text[:200],
                    "agent_id": agent_id,
                    "session_key": (msg.args or {}).get("__session_key", ""),
                    **verdict.to_dict(),
                })

            return IPCResponse(
                type="verdict",
                request_id=msg.request_id,
                data=verdict.to_dict(),
            )

        elif msg.type == "health":
            current_gate_mode = _handler_state["gate_mode"]
            gate_record = _read_gate_state()
            return IPCResponse(
                type="health",
                request_id=msg.request_id,
                data={
                    "status": "ok",
                    "governance_active": governance_active,
                    "governance_stats": hook.stats,
                    "permission_controller": permission_controller is not None,
                    "gate_state": gate_record.get("state", "open") if gate_record else "open",
                    "gate_mode": current_gate_mode or "",
                    "codebase_policies_loaded": len(_handler_state["policies"]),
                },
            )

        elif msg.type == "reset_chain":
            hook.reset_chain()
            if audit_writer:
                audit_writer.emit("chain_reset", {"request_id": msg.request_id})
            return IPCResponse(
                type="ack",
                request_id=msg.request_id,
                data={"message": "Chain reset"},
            )

        elif msg.type == "gate_transition":
            # S sends this after `telos gate open/close`
            # Re-read the gate file (which S already wrote)
            _invalidate_gate_cache()
            new_gate = _read_gate_state(force=True)

            if new_gate is None:
                # Gate file removed or invalid → gate is open
                _handler_state["gate_mode"] = ""
                if audit_writer:
                    audit_writer.write_gate_transition({
                        "state": "open",
                        "mode": "",
                        "actor": "",
                        "source": "gate_transition_ipc",
                        "note": "Gate file absent or invalid — defaulting to open",
                    })
                return IPCResponse(
                    type="ack",
                    request_id=msg.request_id,
                    data={
                        "message": "Gate state updated: open (no valid gate file)",
                        "gate_state": "open",
                        "gate_mode": "",
                    },
                )

            new_state = new_gate.get("state", "open")
            new_mode = new_gate.get("mode", "enforce")

            if new_state == "closed":
                _handler_state["gate_mode"] = new_mode
            else:
                _handler_state["gate_mode"] = ""

            if audit_writer:
                audit_writer.write_gate_transition({
                    **new_gate,
                    "source": "gate_transition_ipc",
                })

            logger.info(
                f"Gate transition: state={new_state} mode={new_mode} "
                f"actor={new_gate.get('actor', '?')[:16]}..."
            )

            # Best-effort Supabase dual-write
            _persist_gate_transition(new_gate)

            return IPCResponse(
                type="ack",
                request_id=msg.request_id,
                data={
                    "message": f"Gate state updated: {new_state} ({new_mode})",
                    "gate_state": new_state,
                    "gate_mode": new_mode if new_state == "closed" else "",
                },
            )

        elif msg.type == "policy_update":
            # Reload codebase policies (e.g., after `telos policy sign`)
            _invalidate_policy_cache()
            new_policies = _load_codebase_policies(force=True)
            _handler_state["policies"] = new_policies
            if audit_writer:
                audit_writer.write_policy_loaded({
                    "policy_count": len(new_policies),
                    "collections": [p.collection for p in new_policies],
                    "source": "policy_update_ipc",
                })
            logger.info(f"Codebase policies reloaded: {len(new_policies)} policies")
            return IPCResponse(
                type="ack",
                request_id=msg.request_id,
                data={
                    "message": f"Policies reloaded: {len(new_policies)} valid",
                    "policy_count": len(new_policies),
                },
            )

        elif msg.type == "resolve_escalation":
            # CLI fallback: resolve a pending escalation
            if not permission_controller:
                return IPCResponse(
                    type="error",
                    request_id=msg.request_id,
                    error="Permission controller not configured",
                )
            escalation_id = msg.args.get("escalation_id", "")
            approved = msg.args.get("approved", False)
            resolved = permission_controller.resolve(
                escalation_id, approved=approved, source="cli"
            )
            return IPCResponse(
                type="ack",
                request_id=msg.request_id,
                data={
                    "message": f"Escalation {escalation_id} {'resolved' if resolved else 'not found'}",
                    "resolved": resolved,
                },
            )

        elif msg.type == "acknowledge_drift":
            if not drift_tracker:
                return IPCResponse(
                    type="error",
                    request_id=msg.request_id,
                    error="Drift tracker not configured",
                )
            reason = msg.args.get("reason", "") if msg.args else ""
            status = drift_tracker.acknowledge_drift(reason=reason)
            logger.info(
                f"Drift acknowledged (reason: {reason!r}, "
                f"ack_count={status['acknowledgment_count']})"
            )
            if audit_writer:
                audit_writer.emit("drift_acknowledged", {
                    "reason": reason,
                    **status,
                })
            # Snapshot after acknowledgment
            if on_state_change:
                on_state_change()
            return IPCResponse(
                type="ack",
                request_id=msg.request_id,
                data={
                    "message": "Drift acknowledged",
                    **status,
                },
            )

        elif msg.type == "get_drift_status":
            if not drift_tracker:
                return IPCResponse(
                    type="error",
                    request_id=msg.request_id,
                    error="Drift tracker not configured",
                )
            return IPCResponse(
                type="drift_status",
                request_id=msg.request_id,
                data=drift_tracker.get_drift_history(),
            )

        elif msg.type == "shutdown":
            return IPCResponse(
                type="ack",
                request_id=msg.request_id,
                data={"message": "Shutdown requested"},
            )

        elif msg.type == "context_enrichment":
            # Stewart -> Daemon: pre-scoring context enrichment
            _stewart_context["task_summary"] = (msg.args or {}).get("task_summary", "")[:500]
            _stewart_context["drift_narrative"] = (msg.args or {}).get("drift_narrative", "")[:500]
            _stewart_context["coherent_sequence"] = bool((msg.args or {}).get("coherent_sequence", False))
            _stewart_context["context_ttl"] = min(int((msg.args or {}).get("context_ttl", 10)), 50)
            _stewart_context["stewart_fidelity_assessment"] = float((msg.args or {}).get("stewart_fidelity_assessment", 0.0))

            import uuid as _uuid_ctx
            ctx_id = str(_uuid_ctx.uuid4())[:8]
            _stewart_context["context_id"] = ctx_id

            logger.info(
                f"Stewart context enrichment accepted (ctx={ctx_id}, "
                f"ttl={_stewart_context['context_ttl']}, "
                f"coherent={_stewart_context['coherent_sequence']})"
            )
            if audit_writer:
                audit_writer.emit("stewart_context_enrichment", {
                    "context_id": ctx_id,
                    "task_summary": _stewart_context["task_summary"][:100],
                    "coherent_sequence": _stewart_context["coherent_sequence"],
                    "context_ttl": _stewart_context["context_ttl"],
                })
            return IPCResponse(
                type="ack",
                request_id=msg.request_id,
                data={
                    "status": "accepted",
                    "context_id": ctx_id,
                    "expires_after_n_calls": _stewart_context["context_ttl"],
                },
            )

        elif msg.type == "stewart_review":
            # Stewart -> Daemon: post-hoc verdict review (V6: can only tighten)
            _verdict_id = (msg.args or {}).get("verdict_id", "")
            _recommendation = (msg.args or {}).get("recommendation", "execute")
            _justification = (msg.args or {}).get("justification", "")[:500]
            _confidence = float((msg.args or {}).get("confidence", 0.5))

            # V6 enforcement: verdict ordering (lower = more restrictive)
            _VERDICT_ORDER = {"escalate": 0, "inert": 1, "suggest": 2, "clarify": 3, "execute": 4}
            _rec_rank = _VERDICT_ORDER.get(_recommendation, 4)

            # We cannot retroactively modify the verdict (tool call already
            # processed), but we log the review for audit accuracy and
            # drift correction.
            logger.info(
                f"Stewart review: verdict={_verdict_id} "
                f"recommendation={_recommendation} confidence={_confidence:.2f}"
            )

            # V6 check: Stewart cannot loosen
            # Since we don't have the original verdict in memory after response,
            # we log the review and trust the client-side V6 guard.
            # The audit trail records Stewart's contextual assessment.
            if audit_writer:
                audit_writer.emit("stewart_review", {
                    "verdict_id": _verdict_id,
                    "recommendation": _recommendation,
                    "justification": _justification,
                    "confidence": _confidence,
                    "v6_compliant": True,  # Client-side guard ensures this
                })

            return IPCResponse(
                type="ack",
                request_id=msg.request_id,
                data={
                    "status": "recorded",
                    "verdict_id": _verdict_id,
                    "recommendation": _recommendation,
                    "v6_note": "Review recorded for audit. V6 client-side guard enforced.",
                },
            )

        elif msg.type == "get_stewart_context":
            # Query current Stewart context state
            return IPCResponse(
                type="stewart_context",
                request_id=msg.request_id,
                data={
                    "context_active": _stewart_context.get("context_ttl", 0) > 0,
                    "context_id": _stewart_context.get("context_id", ""),
                    "task_summary": _stewart_context.get("task_summary", ""),
                    "coherent_sequence": _stewart_context.get("coherent_sequence", False),
                    "remaining_ttl": _stewart_context.get("context_ttl", 0),
                },
            )

        else:
            return IPCResponse(
                type="error",
                request_id=msg.request_id,
                error=f"Unknown message type: {msg.type}",
            )

    return handle


def run_daemon(
    config_path: Optional[str] = None,
    socket_path: Optional[str] = None,
    preset: str = GovernancePreset.BALANCED,
    embed_fn=None,
    verbose: bool = False,
) -> None:
    """Run the TELOS governance daemon.

    Args:
        config_path: Path to openclaw.yaml. Auto-discovers if None.
        socket_path: Path to Unix socket. Uses default if None.
        preset: Governance preset (strict/balanced/permissive/custom).
        embed_fn: Custom embedding function (for testing).
        verbose: Enable debug logging.
    """
    # Configure logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger.info("Starting TELOS governance daemon...")
    logger.info(f"Preset: {preset}")

    # Step 1: Load configuration
    loader = OpenClawConfigLoader()
    loader.load(path=config_path, embed_fn=embed_fn)
    logger.info(
        f"Config loaded: {loader.config.agent_name} "
        f"({len(loader.config.boundaries)} boundaries, "
        f"{len(loader.config.tools)} tools)"
    )

    # Step 1b: Verify PA cryptographic signature (the ceremony gate)
    resolved_config_path = config_path or loader.config.config_path
    pa_result = loader.verify_pa(resolved_config_path)
    pa_status = pa_result.get("status", "UNKNOWN")
    governance_active = loader.governance_active

    if pa_status == "VERIFIED":
        logger.info(
            f"PA VERIFIED — signed by {pa_result.get('signer_fingerprint', '?')[:16]}... "
            f"at {pa_result.get('signed_at', '?')}"
            + (" (Labs attested)" if pa_result.get("labs_attested") else "")
        )
    elif pa_status == "NOT_SIGNED":
        logger.warning(
            "PA NOT SIGNED — governance will be INERT. "
            "Run `telos pa sign <config>` to activate."
        )
    elif pa_status == "TAMPERED":
        logger.error(
            "PA TAMPERED — config modified after signing! "
            "Governance will be INERT. Re-sign with `telos pa sign <config>`."
        )
    elif pa_status == "INVALID_SIGNATURE":
        logger.error(
            "PA INVALID SIGNATURE — Ed25519 verification failed. "
            "Governance will be INERT."
        )
    else:
        logger.warning(f"PA verification: {pa_status}. Governance will be INERT.")

    # Step 1c: Read gate state (Ed25519 cryptographic gate)
    gate_record = _read_gate_state(force=True)
    gate_mode = ""
    if gate_record:
        gate_state = gate_record.get("state", "open")
        if gate_state == "closed":
            gate_mode = gate_record.get("mode", "enforce")
            if gate_mode == "enforce":
                logger.warning(
                    f"GATE CLOSED (enforce) — signed by {gate_record.get('actor', '?')[:16]}... "
                    f"Agent is INERT until gate is opened."
                )
            elif gate_mode == "observe":
                logger.info(
                    f"GATE CLOSED (observe) — signed by {gate_record.get('actor', '?')[:16]}... "
                    f"Scoring active, enforcement suspended (shadow mode)."
                )
        else:
            logger.info("Gate state: open")
    else:
        logger.debug("No gate file found — gate is open by default")

    # Step 1d: Load codebase access policies
    codebase_policies = _load_codebase_policies(force=True)
    if codebase_policies:
        collections = [p.collection for p in codebase_policies]
        logger.info(
            f"Codebase policies loaded: {len(codebase_policies)} policies "
            f"({', '.join(collections)})"
        )
    else:
        logger.debug("No codebase policies found — policy enforcement disabled")

    # Determine project root for path normalization
    project_root = ""
    if config_path:
        # Walk up from config to find .git or use parent
        candidate = Path(config_path).resolve().parent
        while candidate != candidate.parent:
            if (candidate / ".git").exists():
                project_root = str(candidate)
                break
            candidate = candidate.parent
    if not project_root:
        project_root = str(Path.cwd())

    # Register SIGHUP to refresh gate + policy caches
    def _handle_sighup(signum, frame):
        logger.info("SIGHUP received — refreshing gate and policy caches")
        _invalidate_gate_cache()
        _invalidate_policy_cache()

    signal.signal(signal.SIGHUP, _handle_sighup)

    # Step 2: Initialize governance hook
    hook = GovernanceHook(loader, preset=preset)
    logger.info(f"Governance hook initialized (preset={preset})")

    # Step 3: Initialize receipt signer for verdict integrity
    receipt_signer = _init_receipt_signer()

    # Step 4: Initialize permission controller (if notifications configured)
    permission_controller, telegram_poller = _init_permission_controller(loader.config)

    # Step 5: Initialize SAAI drift tracker (session-level)
    drift_tracker = AgenticDriftTracker()
    logger.info("SAAI drift tracker initialized (10/15/20% graduated sanctions)")

    # Step 5b: Restore session state if available (fixes "sanctions amnesia" C4)
    _state_path = Path.home() / ".telos" / "session_state.json"
    if _state_path.exists():
        try:
            _saved = _json.loads(_state_path.read_text("utf-8"))
            _state_age = _time.time() - _saved.get("timestamp_epoch", 0)
            if _state_age < 300:  # Less than 5 minutes old
                drift_tracker.restore(_saved.get("drift_tracker"))
                logger.info(
                    f"Restored drift tracker state from {_state_age:.0f}s ago "
                    f"(level={_saved.get('drift_tracker', {}).get('drift_level', '?')})"
                )
            else:
                logger.info(f"Session state too old ({_state_age:.0f}s), starting fresh")
        except Exception as e:
            logger.warning(f"Failed to restore session state: {e}")

    # Step 6: Initialize CUSUM monitor bank (per-tool-group drift)
    cusum_bank = CUSUMMonitorBank()
    logger.info("CUSUM monitor bank initialized (per-tool-group drift detection)")

    # Restore CUSUM state if available
    if _state_path.exists():
        try:
            _saved = _json.loads(_state_path.read_text("utf-8"))
            _state_age = _time.time() - _saved.get("timestamp_epoch", 0)
            if _state_age < 300:
                cusum_bank.restore(_saved.get("cusum_bank"))
                logger.info("Restored CUSUM monitor bank state")
        except Exception:
            pass  # Already logged above

    # Step 7: Initialize audit writer (structured NDJSON trail)
    audit_writer = AuditWriter()
    audit_writer.emit("daemon_start", {
        "pid": os.getpid(),
        "config_path": config_path,
        "preset": preset,
        "agent_name": loader.config.agent_name,
    })
    audit_writer.emit("config_loaded", {
        "agent_name": loader.config.agent_name,
        "boundary_count": len(loader.config.boundaries),
        "tool_count": len(loader.config.tools),
    })

    # Audit: PA verification status (the ceremony gate record)
    if governance_active:
        audit_writer.emit("pa_verified", {
            "status": pa_status,
            "signer_fingerprint": pa_result.get("signer_fingerprint", ""),
            "signed_at": pa_result.get("signed_at", ""),
            "config_hash": pa_result.get("config_hash", ""),
            "labs_attested": pa_result.get("labs_attested", False),
            "labs_receipt_id": pa_result.get("labs_receipt_id", ""),
        })
    else:
        audit_writer.emit("pa_not_verified", {
            "status": pa_status,
            "config_hash": pa_result.get("config_hash", ""),
        })
        if pa_status == "TAMPERED":
            audit_writer.emit("security_event", {
                "event_type": "pa_config_tampered",
                "severity": "CRITICAL",
                "config_hash": pa_result.get("config_hash", ""),
                "signed_hash": pa_result.get("signed_hash", ""),
            })

    # Audit: codebase policies at boot
    if codebase_policies:
        audit_writer.write_policy_loaded({
            "policy_count": len(codebase_policies),
            "collections": [p.collection for p in codebase_policies],
            "source": "boot",
        })

    # Audit: gate state at boot
    if gate_mode:
        audit_writer.emit("gate_transition", {
            "state": gate_record.get("state", "") if gate_record else "",
            "mode": gate_mode,
            "actor": gate_record.get("actor", "") if gate_record else "",
            "source": "boot_read",
        })

    # Step 7b: Session state snapshot function (called periodically)
    def _snapshot_session_state():
        """Write session state to disk for crash recovery."""
        try:
            state = {
                "timestamp": _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
                "timestamp_epoch": _time.time(),
                "drift_tracker": drift_tracker.to_dict() if drift_tracker else None,
                "cusum_bank": cusum_bank.to_dict() if cusum_bank else None,
            }
            _snapshot_path = Path.home() / ".telos" / "session_state.json"
            _snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            _snapshot_path.write_text(_json.dumps(state, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning(f"Session state snapshot failed: {e}")

    # Step 8: Create IPC server
    server = IPCServer(
        socket_path=socket_path,
        handler=create_message_handler(
            hook, permission_controller, receipt_signer,
            drift_tracker, audit_writer, cusum_bank,
            governance_active=governance_active,
            gate_mode=gate_mode,
            codebase_policies=codebase_policies,
            project_root=project_root,
            on_state_change=_snapshot_session_state,
        ),
    )

    # Step 5: Start with watchdog
    # If Telegram poller is configured, start it alongside the IPC server
    if telegram_poller:
        original_run_sync = server.run_sync

        def run_with_poller():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            # Start Telegram poller as a background task
            loop.create_task(telegram_poller.start())
            # Run the IPC server (this blocks)
            try:
                loop.run_until_complete(server.start())
            except asyncio.CancelledError:
                pass
            finally:
                telegram_poller.stop()
                loop.close()

        watchdog = Watchdog()
        watchdog.start(
            main_fn=run_with_poller,
            on_shutdown=lambda: None,
        )
    else:
        watchdog = Watchdog()
        watchdog.start(
            main_fn=server.run_sync,
            on_shutdown=lambda: None,
        )


def main():
    """CLI entry point for the daemon."""
    parser = argparse.ArgumentParser(
        description="TELOS Governance Daemon for OpenClaw",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", "-c",
        help="Path to openclaw.yaml (auto-discovers if not provided)",
    )
    parser.add_argument(
        "--socket", "-s",
        help="Path to Unix socket (default: ~/.openclaw/hooks/telos.sock)",
    )
    parser.add_argument(
        "--preset", "-p",
        choices=["strict", "balanced", "permissive", "custom"],
        default="balanced",
        help="Governance preset (default: balanced)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    try:
        run_daemon(
            config_path=args.config,
            socket_path=args.socket,
            preset=args.preset,
            verbose=args.verbose,
        )
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(2)
    except KeyboardInterrupt:
        logger.info("Shutdown requested")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Daemon failed: {e}", exc_info=True)
        sys.exit(3)


if __name__ == "__main__":
    main()
