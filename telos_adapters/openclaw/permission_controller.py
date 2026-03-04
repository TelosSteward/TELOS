"""
Permission Controller — ESCALATE verdict human-in-the-loop orchestration.

When the governance engine returns an ESCALATE verdict, the permission
controller:
  1. Sends notifications via NotificationService (Telegram/WhatsApp/Discord)
  2. Waits for a human response (Telegram callback, WhatsApp webhook, or CLI)
  3. If approved: signs an Ed25519 override receipt and allows the action
  4. If denied or timeout: keeps the action blocked (fail-closed default)

Every escalation event is logged to an append-only JSONL audit file.

Regulatory traceability:
    - EU AI Act Art. 14: Human oversight for high-risk AI decisions
    - EU AI Act Art. 72: Override receipts for post-market audit
    - IEEE 7001-2021: Transparent escalation with signed evidence
    - SAAI claim TELOS-SAAI-009: Human-in-the-loop for ESCALATE
    - OWASP ASI04 (Unsafe Actions): Override requires explicit human approval
    See: research/openclaw_regulatory_mapping.md
"""

import asyncio
import hashlib
import hmac as hmac_mod
import json
import logging
import os
import secrets
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from telos_adapters.openclaw.notification_service import (
    EscalationNotification,
    NotificationService,
)

logger = logging.getLogger(__name__)

# Default audit log location
DEFAULT_AUDIT_DIR = Path.home() / ".telos" / "audit"
DEFAULT_KEY_DIR = Path.home() / ".telos" / "keys"


@dataclass
class EscalationResult:
    """Result of a permission controller escalation."""
    approved: bool
    escalation_id: str
    response_source: str  # "telegram", "whatsapp", "cli", "timeout"
    response_time_ms: float = 0.0
    receipt: Optional[Dict[str, Any]] = None
    override_signature: Optional[str] = None


class PermissionController:
    """Orchestrates ESCALATE verdict human-in-the-loop decision flow.

    Usage:
        controller = PermissionController(config, notification_service, receipt_signer)
        result = await controller.handle_escalation(verdict)
        if result.approved:
            # Allow the tool call, result.receipt contains signed override
            ...
    """

    def __init__(
        self,
        config,
        notification_service: NotificationService,
        receipt_signer=None,
        tkeys_manager=None,
        audit_dir: Optional[Path] = None,
    ):
        """Initialize the permission controller.

        Args:
            config: NotificationsConfig from YAML.
            notification_service: Multi-channel notification dispatcher.
            receipt_signer: Optional ReceiptSigner for Ed25519 override receipts.
            tkeys_manager: Optional TelemetricSessionManager for HMAC co-signing.
                When provided, TKeys becomes the cryptographic trust root —
                every escalation, callback, receipt, and audit entry is signed.
            audit_dir: Path for escalation audit logs. Defaults to ~/.telos/audit/.
        """
        self._config = config
        if config.timeout_action == "allow":
            import warnings
            warnings.warn(
                "timeout_action=\"allow\" is deprecated and unsafe. "
                "Fail-open governance bypasses scoring on timeout. "
                "Use timeout_action=\"deny\" (default) in production.",
                DeprecationWarning,
                stacklevel=2,
            )
        self._notifications = notification_service
        self._signer = receipt_signer
        self._tkeys = tkeys_manager
        self._audit_dir = audit_dir or DEFAULT_AUDIT_DIR
        self._audit_dir.mkdir(parents=True, exist_ok=True)

        # Pending escalations awaiting resolution
        self._pending: Dict[str, asyncio.Future] = {}

        # Signed escalation registry — maps escalation_id to signed request
        self._signed_escalations: Dict[str, Dict[str, Any]] = {}

        # Hash chain state — SHA-256 of previous audit entry for deletion detection
        self._prev_audit_hash: str = ""

        if self._tkeys:
            logger.info("Permission Controller: TKeys trust root ACTIVE")

    def _build_semantic_context(self, verdict) -> str:
        """Build semantic interpretation of a governance decision.

        Uses the Semantic Interpreter (telos_core) to translate fidelity
        measurements into plain-language explanations scaled to severity.
        The human reads this BEFORE seeing machine metadata or signing
        any TKeys decision — zero ambiguity.

        Args:
            verdict: GovernanceVerdict with fidelity, tool info, decision.

        Returns:
            Multi-line plain-language interpretation string.
        """
        try:
            from telos_core.semantic_interpreter import interpret
        except ImportError:
            logger.debug("Semantic interpreter not available")
            return ""

        fidelity = getattr(verdict, "fidelity", 0.0)
        tool_name = getattr(verdict, "telos_tool_name", "unknown")
        tool_group = getattr(verdict, "tool_group", "unknown")
        decision = getattr(verdict, "decision", "unknown")
        explanation = getattr(verdict, "explanation", "")
        boundary = getattr(verdict, "boundary_triggered", False)

        # Interpreter gives calibrated linguistic spec for this fidelity level
        purpose = f"governed {tool_group} operations"
        spec = interpret(fidelity, purpose)
        alignment_pct = int(fidelity * 100)

        lines = []

        # What happened — adapted to decision type
        if decision == "escalate":
            lines.append(
                f"The agent attempted a {tool_group} action ({tool_name}) "
                f"that requires human authorization."
            )
        else:
            lines.append(
                f"The agent attempted a {tool_group} action ({tool_name}) "
                f"that falls outside its operational scope."
            )

        # Why — severity-scaled via interpreter strength bands
        if spec.strength >= 0.85:
            lines.append(
                f"At {alignment_pct}% alignment, this action is far from "
                f"the agent's purpose."
            )
        elif spec.strength >= 0.75:
            lines.append(
                f"At {alignment_pct}% alignment, this action has drifted "
                f"significantly from scope."
            )
        elif spec.strength >= 0.60:
            lines.append(
                f"At {alignment_pct}% alignment, this action is outside "
                f"the expected operating range."
            )
        else:
            lines.append(
                f"At {alignment_pct}% alignment, this action is marginally "
                f"outside scope."
            )

        if boundary:
            lines.append(
                "A defined operational boundary was violated."
            )

        if explanation:
            lines.append(explanation[:200])

        return "\n".join(lines)

    async def handle_escalation(self, verdict) -> EscalationResult:
        """Handle an ESCALATE verdict by notifying and waiting for response.

        Args:
            verdict: GovernanceVerdict with decision=ESCALATE.

        Returns:
            EscalationResult with approved status and optional signed receipt.
        """
        start = time.perf_counter()
        escalation_id = secrets.token_hex(16)

        # Sign escalation request (TKeys trust root)
        nonce = secrets.token_hex(8)
        expiry_ts = time.time() + self._config.escalation_timeout_seconds
        signed_esc = self._sign_escalation_request(
            escalation_id, verdict, nonce, expiry_ts
        )
        self._signed_escalations[escalation_id] = signed_esc

        # Build semantic interpretation for zero-ambiguity human notification
        semantic_context = self._build_semantic_context(verdict)

        # Build notification with challenge from signed escalation
        notification = EscalationNotification(
            escalation_id=escalation_id,
            tool_name=verdict.telos_tool_name,
            tool_group=verdict.tool_group,
            risk_tier=verdict.risk_tier,
            fidelity_score=verdict.fidelity,
            explanation=verdict.explanation,
            timestamp=time.time(),
            boundary_triggered=verdict.boundary_triggered,
            action_text=getattr(verdict, "action_text", ""),
            challenge=signed_esc.get("challenge", ""),
            decision="escalate",
            semantic_context=semantic_context,
        )

        # Log escalation initiated
        self._log_audit_event(escalation_id, "initiated", {
            "tool_name": verdict.telos_tool_name,
            "tool_group": verdict.tool_group,
            "risk_tier": verdict.risk_tier,
            "fidelity": verdict.fidelity,
            "boundary_triggered": verdict.boundary_triggered,
        })

        # Send notifications to all configured channels
        try:
            send_results = await self._notifications.send(notification)
        except Exception as e:
            logger.error(f"Notification dispatch failed: {e}")
            send_results = {"error": str(e)}
        self._log_audit_event(escalation_id, "notified", {
            "channels": send_results,
        })

        # Wait for response
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        self._pending[escalation_id] = future

        try:
            # Wait for response with timeout
            timeout = self._config.escalation_timeout_seconds
            approved, source = await asyncio.wait_for(
                self._wait_for_response(escalation_id, future),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            # Timeout — apply timeout_action
            approved = self._config.timeout_action == "allow"
            source = "timeout"
            logger.warning(
                f"Escalation {escalation_id} timed out after {timeout}s — "
                f"action={'allowed' if approved else 'denied'}"
            )
        finally:
            self._pending.pop(escalation_id, None)
            # Keep signed escalation for receipt chaining, clean up later

        elapsed_ms = (time.perf_counter() - start) * 1000

        # Sign override receipt if approved and signer available
        receipt = None
        override_sig = None
        if approved and self._signer:
            receipt, override_sig = self._sign_override_receipt(
                verdict, escalation_id, source
            )

        # Log resolution
        self._log_audit_event(escalation_id, "resolved", {
            "approved": approved,
            "source": source,
            "response_time_ms": round(elapsed_ms, 1),
            "receipt_signed": receipt is not None,
        })

        return EscalationResult(
            approved=approved,
            escalation_id=escalation_id,
            response_source=source,
            response_time_ms=elapsed_ms,
            receipt=receipt,
            override_signature=override_sig,
        )

    async def handle_inert(self, verdict) -> EscalationResult:
        """Handle an INERT verdict by notifying the human controller.

        When a session goes INERT (fidelity < 0.50, action blocked), the human
        receives a semantic interpretation of what happened and why, with options
        to redirect or override. The human can then make an informed decision
        about the agent's direction.

        Unlike ESCALATE which blocks until the human responds, INERT sends the
        notification and optionally waits for a response (override/redirect).
        If no response within timeout, the action stays blocked.

        Args:
            verdict: GovernanceVerdict with decision=INERT.

        Returns:
            EscalationResult — approved=True only if human explicitly overrides.
        """
        start = time.perf_counter()
        escalation_id = secrets.token_hex(16)

        # Sign request (same chain of custody as ESCALATE)
        nonce = secrets.token_hex(8)
        expiry_ts = time.time() + self._config.escalation_timeout_seconds
        signed_esc = self._sign_escalation_request(
            escalation_id, verdict, nonce, expiry_ts
        )
        self._signed_escalations[escalation_id] = signed_esc

        # Semantic interpretation — the human reads WHY, not machine output
        semantic_context = self._build_semantic_context(verdict)

        notification = EscalationNotification(
            escalation_id=escalation_id,
            tool_name=getattr(verdict, "telos_tool_name", "unknown"),
            tool_group=getattr(verdict, "tool_group", "unknown"),
            risk_tier=getattr(verdict, "risk_tier", "low"),
            fidelity_score=getattr(verdict, "fidelity", 0.0),
            explanation=getattr(verdict, "explanation", ""),
            timestamp=time.time(),
            boundary_triggered=getattr(verdict, "boundary_triggered", False),
            action_text=getattr(verdict, "action_text", ""),
            challenge=signed_esc.get("challenge", ""),
            decision="inert",
            semantic_context=semantic_context,
        )

        self._log_audit_event(escalation_id, "inert_initiated", {
            "tool_name": getattr(verdict, "telos_tool_name", ""),
            "tool_group": getattr(verdict, "tool_group", ""),
            "risk_tier": getattr(verdict, "risk_tier", ""),
            "fidelity": getattr(verdict, "fidelity", 0.0),
        })

        # Send notifications
        try:
            send_results = await self._notifications.send(notification)
        except Exception as e:
            logger.error(f"INERT notification dispatch failed: {e}")
            send_results = {"error": str(e)}
        self._log_audit_event(escalation_id, "inert_notified", {
            "channels": send_results,
        })

        # Wait for optional human override
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        self._pending[escalation_id] = future

        try:
            timeout = self._config.escalation_timeout_seconds
            approved, source = await asyncio.wait_for(
                self._wait_for_response(escalation_id, future),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            # No human response — action stays blocked (INERT default)
            approved = False
            source = "timeout"
            logger.info(f"INERT {escalation_id} — no human response, stays blocked")
        finally:
            self._pending.pop(escalation_id, None)

        elapsed_ms = (time.perf_counter() - start) * 1000

        receipt = None
        override_sig = None
        if approved and self._signer:
            receipt, override_sig = self._sign_override_receipt(
                verdict, escalation_id, source
            )

        self._log_audit_event(escalation_id, "inert_resolved", {
            "approved": approved,
            "source": source,
            "response_time_ms": round(elapsed_ms, 1),
            "receipt_signed": receipt is not None,
        })

        return EscalationResult(
            approved=approved,
            escalation_id=escalation_id,
            response_source=source,
            response_time_ms=elapsed_ms,
            receipt=receipt,
            override_signature=override_sig,
        )

    async def _wait_for_response(
        self,
        escalation_id: str,
        future: asyncio.Future,
    ) -> tuple:
        """Wait for a response from any interactive channel.

        Returns:
            Tuple of (approved: bool, source: str).
        """
        result = await future
        return result["approved"], result["source"]

    def resolve(
        self,
        escalation_id: str,
        approved: bool,
        source: str = "cli",
        challenge: str = "",
    ) -> bool:
        """Resolve a pending escalation (called from CLI or webhook handler).

        Args:
            escalation_id: The escalation to resolve.
            approved: Whether to approve the override.
            source: Response source identifier.
            challenge: Challenge hash from callback (verified against signed request).

        Returns:
            True if the escalation was pending and resolved, False otherwise.
        """
        # Verify against signed escalation registry
        signed_esc = self._signed_escalations.get(escalation_id)
        if signed_esc:
            # Verify expiry
            if time.time() > signed_esc.get("expiry", 0):
                self._log_audit_event(escalation_id, "verification_failed", {
                    "reason": "expired",
                    "source": source,
                })
                logger.warning(f"Escalation {escalation_id} expired — rejecting callback")
                return False

            # Verify challenge for external webhook callbacks (Telegram/WhatsApp).
            # CLI resolution is same-process — challenge not required.
            expected_challenge = signed_esc.get("challenge", "")
            requires_challenge = source not in ("cli",)
            if requires_challenge and expected_challenge and challenge != expected_challenge:
                self._log_audit_event(escalation_id, "verification_failed", {
                    "reason": "challenge_mismatch",
                    "source": source,
                })
                logger.warning(f"Escalation {escalation_id} challenge mismatch — rejecting")
                return False

        future = self._pending.get(escalation_id)
        if future and not future.done():
            future.set_result({"approved": approved, "source": source})
            logger.info(
                f"Escalation {escalation_id} resolved via {source}: "
                f"{'approved' if approved else 'denied'}"
            )
            return True
        logger.warning(f"No pending escalation with id: {escalation_id}")
        return False

    def resolve_from_telegram(self, callback_data: Dict[str, Any]) -> bool:
        """Resolve from a Telegram callback_query.

        Args:
            callback_data: Parsed callback_data from Telegram button press.
                Expected format: {"a": "y"|"n", "i": "...", "c": "..."}
                Legacy format: {"action": "approve"|"deny", "id": "..."}

        Returns:
            True if resolved successfully.
        """
        # Support both compact (a/i/c) and legacy (action/id) format
        escalation_id = callback_data.get("i", "") or callback_data.get("id", "")
        action = callback_data.get("a", "") or callback_data.get("action", "")
        challenge = callback_data.get("c", "")
        approved = action in ("y", "approve")
        return self.resolve(
            escalation_id,
            approved=approved,
            source="telegram",
            challenge=challenge,
        )

    def resolve_from_whatsapp(self, button_reply_id: str) -> bool:
        """Resolve from a WhatsApp button reply.

        Args:
            button_reply_id: The reply ID from WhatsApp button callback.
                Format: "approve:<escalation_id>:<challenge>"
                Legacy: "approve:<escalation_id>" or "deny:<escalation_id>"

        Returns:
            True if resolved successfully.
        """
        if ":" not in button_reply_id:
            return False
        parts = button_reply_id.split(":", 2)
        action = parts[0]
        escalation_id = parts[1] if len(parts) > 1 else ""
        challenge = parts[2] if len(parts) > 2 else ""
        return self.resolve(
            escalation_id,
            approved=(action == "approve"),
            source="whatsapp",
            challenge=challenge,
        )

    @property
    def pending_escalations(self) -> list:
        """List of currently pending escalation IDs."""
        return list(self._pending.keys())

    def _sign_escalation_request(
        self,
        escalation_id: str,
        verdict,
        nonce: str,
        expiry_ts: float,
    ) -> Dict[str, Any]:
        """Sign an outgoing escalation request with Ed25519 + optional TKeys HMAC.

        This creates the first link in the cryptographic chain of custody.
        The challenge derived from the signature is embedded in notification
        callbacks and verified when the human responds.

        Returns:
            Signed escalation dict with signatures and challenge.
        """
        payload = {
            "type": "escalation_request",
            "escalation_id": escalation_id,
            "tool_name": getattr(verdict, "telos_tool_name", ""),
            "tool_group": getattr(verdict, "tool_group", ""),
            "risk_tier": getattr(verdict, "risk_tier", ""),
            "fidelity": round(getattr(verdict, "fidelity", 0.0), 4),
            "boundary_triggered": getattr(verdict, "boundary_triggered", False),
            "nonce": nonce,
            "expiry": expiry_ts,
            "timestamp": time.time(),
        }

        # Canonical serialization
        canonical = json.dumps(
            payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        payload_hash = hashlib.sha256(canonical).digest()

        # Ed25519 signature (persistent, third-party verifiable)
        challenge = ""
        if self._signer:
            ed25519_sig = self._signer.sign_payload(payload_hash)
            payload["ed25519_signature"] = ed25519_sig.hex()
            payload["public_key"] = self._signer.public_key_bytes().hex()
            payload["payload_hash"] = payload_hash.hex()
            # Challenge = truncated hash of Ed25519 signature (16 hex chars)
            challenge = hashlib.sha256(ed25519_sig).hexdigest()[:16]

        # TKeys HMAC-SHA512 (session-bound, irreproducible)
        if self._tkeys:
            # Rotate TKeys with escalation telemetry
            escalation_telemetry = {
                "timestamp": time.time(),
                "fidelity_score": getattr(verdict, "fidelity", 0.0),
                "distance_from_pa": 1.0 - getattr(verdict, "fidelity", 0.0),
                "intervention_triggered": True,
                "in_basin": False,
                "turn_number": len(self._signed_escalations),
                "session_id": self._tkeys.session_id,
            }
            self._tkeys.key_generator.rotate_key(escalation_telemetry)
            # HMAC-SHA512 sign the canonical payload
            tkeys_hmac = self._tkeys.key_generator.generate_hmac_signature(canonical)
            payload["tkeys_hmac"] = tkeys_hmac.hex()
            payload["tkeys_rotation"] = self._tkeys.key_generator.state.turn_number

        payload["challenge"] = challenge
        return payload

    def _sign_override_receipt(
        self,
        verdict,
        escalation_id: str,
        response_source: str,
    ) -> tuple:
        """Sign an Ed25519 + TKeys HMAC override receipt.

        The receipt chains to the outgoing escalation signature via
        escalation_sig_hash, creating a complete cryptographic chain:
        escalation issuance → callback verification → override receipt.

        Returns:
            Tuple of (receipt_dict, signature_hex).
        """
        try:
            # Build override payload
            payload = {
                "type": "escalation_override",
                "escalation_id": escalation_id,
                "decision": verdict.decision,
                "tool_name": verdict.telos_tool_name,
                "tool_group": verdict.tool_group,
                "risk_tier": verdict.risk_tier,
                "fidelity": round(verdict.fidelity, 4),
                "boundary_triggered": verdict.boundary_triggered,
                "response_source": response_source,
                "timestamp": time.time(),
                "governance_preset": verdict.governance_preset,
            }

            # Chain to outgoing escalation signature
            signed_esc = self._signed_escalations.get(escalation_id, {})
            esc_sig = signed_esc.get("ed25519_signature", "")
            if esc_sig:
                payload["escalation_sig_hash"] = hashlib.sha256(
                    bytes.fromhex(esc_sig)
                ).hexdigest()

            # Canonical serialization
            canonical = json.dumps(
                payload, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")

            # SHA-256 hash + Ed25519 signature
            payload_hash = hashlib.sha256(canonical).digest()
            signature = self._signer.sign_payload(payload_hash)

            payload["payload_hash"] = payload_hash.hex()
            payload["ed25519_signature"] = signature.hex()
            payload["public_key"] = self._signer.public_key_bytes().hex()

            # TKeys HMAC-SHA512 co-signature (session-bound)
            if self._tkeys:
                tkeys_hmac = self._tkeys.key_generator.generate_hmac_signature(canonical)
                payload["tkeys_hmac"] = tkeys_hmac.hex()
                payload["tkeys_rotation"] = self._tkeys.key_generator.state.turn_number

            # Clean up signed escalation registry
            self._signed_escalations.pop(escalation_id, None)

            logger.info(
                f"Override receipt signed for escalation {escalation_id}: "
                f"sig={signature.hex()[:16]}..."
            )
            return payload, signature.hex()

        except Exception as e:
            logger.error(f"Failed to sign override receipt: {e}")
            return None, None

    def get_tkeys_proof(self) -> Optional[Dict[str, Any]]:
        """Generate TKeys session proof for audit purposes.

        Returns:
            Session proof dict if TKeys is active, None otherwise.
        """
        if self._tkeys:
            return self._tkeys.generate_session_proof()
        return None

    def _log_audit_event(
        self,
        escalation_id: str,
        event_type: str,
        data: Dict[str, Any],
    ) -> None:
        """Append a signed escalation event to the JSONL audit log.

        When TKeys is active, each entry includes an HMAC-SHA512 signature
        making the audit log tamper-evident.
        """
        try:
            audit_file = self._audit_dir / "escalations.jsonl"
            entry = {
                "escalation_id": escalation_id,
                "event": event_type,
                "timestamp": time.time(),
                "prev_receipt_hash": self._prev_audit_hash,
                **data,
            }

            # TKeys HMAC for tamper evidence
            if self._tkeys:
                canonical = json.dumps(
                    entry, sort_keys=True, separators=(",", ":"), default=str
                ).encode("utf-8")
                tkeys_hmac = self._tkeys.key_generator.generate_hmac_signature(
                    canonical
                )
                entry["tkeys_hmac"] = tkeys_hmac.hex()

            # Compute hash of this entry for chain linking
            entry_canonical = json.dumps(
                entry, sort_keys=True, separators=(",", ":"), default=str
            ).encode("utf-8")
            self._prev_audit_hash = hashlib.sha256(entry_canonical).hexdigest()

            with open(audit_file, "a") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except Exception as e:
            logger.error(f"Failed to write audit event: {e}")


class TelegramPoller:
    """Polls Telegram getUpdates for callback_query responses.

    Runs as a background task, resolving pending escalations
    when a user presses an Approve/Deny button.
    """

    def __init__(self, bot_token: str, controller: PermissionController):
        self._bot_token = bot_token
        self._controller = controller
        self._offset = 0
        self._running = False

    async def start(self) -> None:
        """Start polling for Telegram callback queries."""
        self._running = True
        logger.info("Telegram poller started")
        try:
            import aiohttp
        except ImportError:
            logger.error("aiohttp required for Telegram polling")
            return

        async with aiohttp.ClientSession() as session:
            while self._running:
                try:
                    url = (
                        f"https://api.telegram.org/bot{self._bot_token}"
                        f"/getUpdates"
                    )
                    params = {
                        "offset": self._offset,
                        "timeout": 30,  # Long polling
                        "allowed_updates": json.dumps(["callback_query"]),
                    }

                    async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=35)) as resp:
                        if resp.status != 200:
                            await asyncio.sleep(5)
                            continue

                        data = await resp.json()
                        for update in data.get("result", []):
                            self._offset = update["update_id"] + 1
                            callback = update.get("callback_query")
                            if callback:
                                self._handle_callback(callback, session)

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Telegram poll error: {e}")
                    await asyncio.sleep(5)

    def _handle_callback(self, callback: Dict, session) -> None:
        """Process a Telegram callback_query."""
        try:
            callback_data = json.loads(callback.get("data", "{}"))
            resolved = self._controller.resolve_from_telegram(callback_data)
            if resolved:
                # Determine action from compact (a) or legacy (action) format
                action = callback_data.get("a", "") or callback_data.get("action", "")
                is_approved = action in ("y", "approve")
                asyncio.ensure_future(
                    self._answer_callback(session, callback["id"],
                                          f"Escalation {'approved' if is_approved else 'denied'}")
                )
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Invalid Telegram callback: {e}")

    async def _answer_callback(self, session, callback_query_id: str, text: str) -> None:
        """Answer a Telegram callback query."""
        url = f"https://api.telegram.org/bot{self._bot_token}/answerCallbackQuery"
        await session.post(url, json={
            "callback_query_id": callback_query_id,
            "text": text,
        })

    def stop(self) -> None:
        """Stop the poller."""
        self._running = False
