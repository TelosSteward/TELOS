"""
Notification Service — Multi-channel governance notifications.

Dispatches ESCALATE and INERT verdict notifications to configured channels:
  - Telegram: Interactive (Approve/Deny inline keyboard buttons)
  - WhatsApp: Interactive (Approve/Deny buttons via Cloud API)
  - Discord: Notification only (v1 — webhooks can't receive button callbacks)

All channels are notified simultaneously. The first interactive response
(Telegram or WhatsApp) resolves the escalation.

Notifications include semantic interpretations of governance decisions —
not machine output, but plain-language explanations of what happened, why,
and what the human's options are. Zero ambiguity before TKeys signing.

Regulatory traceability:
    - EU AI Act Art. 14: Real-time human notification for ESCALATE/INERT decisions
    - EU AI Act Art. 72: Notification timestamps form audit trail
    - SAAI claim TELOS-SAAI-009: Human-in-the-loop via external channels
    See: research/openclaw_regulatory_mapping.md
"""

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# WhatsApp Cloud API version
WHATSAPP_API_VERSION = "v21.0"


@dataclass
class EscalationNotification:
    """Data for an ESCALATE or INERT verdict notification."""
    escalation_id: str
    tool_name: str
    tool_group: str
    risk_tier: str
    fidelity_score: float
    explanation: str
    timestamp: float
    boundary_triggered: bool = False
    action_text: str = ""
    challenge: str = ""  # SHA-256(Ed25519 sig)[:16] — cryptographic callback verification
    decision: str = "escalate"  # "escalate" or "inert"
    semantic_context: str = ""  # Plain-language interpretation from SemanticInterpreter

    def format_message(self) -> str:
        """Format a human-readable notification message.

        Semantic context goes first — the human reads WHY before seeing
        machine metadata. Zero ambiguity before any TKeys decision.
        """
        decision_label = {
            "escalate": "ESCALATION",
            "inert": "BLOCKED (INERT)",
        }.get(self.decision, self.decision.upper())

        risk_label = {
            "critical": "CRITICAL",
            "high": "HIGH",
            "medium": "MEDIUM",
            "low": "LOW",
        }.get(self.risk_tier, self.risk_tier.upper())

        lines = [f"TELOS {decision_label} [{risk_label}]", ""]

        # Semantic interpretation first — plain language, not machine output
        if self.semantic_context:
            lines.append(self.semantic_context)
            lines.append("")

        # Machine metadata second
        lines.append(
            f"Tool: {self.tool_name} ({self.tool_group}) | "
            f"Fidelity: {self.fidelity_score:.3f}"
        )
        if self.boundary_triggered:
            lines.append("Boundary violation detected")
        if self.action_text:
            lines.append(f"Action: {self.action_text[:150]}")
        lines.append(f"ID: {self.escalation_id}")

        # Human options
        lines.append("")
        if self.decision == "escalate":
            lines.append("APPROVE: Allow this action (cryptographically signed override)")
            lines.append("DENY: Keep blocked (recommended for boundary violations)")
        else:
            lines.append("OVERRIDE: Allow this action (requires signed authorization)")
            lines.append("REDIRECT: Change agent scope via CLI: telos agent redirect <id>")

        return "\n".join(lines)


class NotificationService:
    """Multi-channel notification dispatcher for ESCALATE verdicts.

    Sends notifications to all configured channels simultaneously.
    Telegram and WhatsApp support interactive buttons for approve/deny.
    Discord is notification-only in v1.

    Args:
        config: NotificationsConfig from the YAML configuration.
    """

    def __init__(self, config):
        """Initialize with a NotificationsConfig."""
        self._config = config
        self._aiohttp = None  # Lazy import

    async def _get_session(self):
        """Lazy-initialize aiohttp session."""
        if self._aiohttp is None:
            try:
                import aiohttp
                self._aiohttp = aiohttp.ClientSession()
            except ImportError:
                raise ImportError(
                    "aiohttp is required for notifications. "
                    "Install with: pip install aiohttp"
                )
        return self._aiohttp

    async def close(self):
        """Close the HTTP session."""
        if self._aiohttp:
            await self._aiohttp.close()
            self._aiohttp = None

    async def send(self, notification: EscalationNotification) -> Dict[str, bool]:
        """Send escalation notification to all configured channels.

        Args:
            notification: The escalation notification to send.

        Returns:
            Dict mapping channel names to success booleans.
        """
        results = {}

        if self._config.has_telegram:
            try:
                await self._send_telegram(notification)
                results["telegram"] = True
            except Exception as e:
                logger.error(f"Telegram notification failed: {e}")
                results["telegram"] = False

        if self._config.has_whatsapp:
            try:
                await self._send_whatsapp(notification)
                results["whatsapp"] = True
            except Exception as e:
                logger.error(f"WhatsApp notification failed: {e}")
                results["whatsapp"] = False

        if self._config.has_discord:
            try:
                await self._send_discord(notification)
                results["discord"] = True
            except Exception as e:
                logger.error(f"Discord notification failed: {e}")
                results["discord"] = False

        if not results:
            logger.warning("No notification channels configured")

        return results

    async def _send_telegram(self, notification: EscalationNotification) -> None:
        """Send Telegram notification with inline Approve/Deny buttons."""
        session = await self._get_session()
        url = (
            f"https://api.telegram.org/bot{self._config.telegram_bot_token}"
            f"/sendMessage"
        )

        # InlineKeyboardMarkup with Approve/Deny buttons
        # Compact keys (a/i/c) to stay under Telegram's 64-byte callback_data limit
        approve_data = {"a": "y", "i": notification.escalation_id}
        deny_data = {"a": "n", "i": notification.escalation_id}
        if notification.challenge:
            approve_data["c"] = notification.challenge
            deny_data["c"] = notification.challenge

        keyboard = {
            "inline_keyboard": [[
                {
                    "text": "Approve Override",
                    "callback_data": json.dumps(approve_data, separators=(",", ":")),
                },
                {
                    "text": "Deny (Keep Blocked)",
                    "callback_data": json.dumps(deny_data, separators=(",", ":")),
                },
            ]]
        }

        payload = {
            "chat_id": self._config.telegram_chat_id,
            "text": notification.format_message(),
            "reply_markup": json.dumps(keyboard),
            "parse_mode": "HTML",
        }

        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(f"Telegram API error {resp.status}: {body}")
            logger.info(f"Telegram notification sent: {notification.escalation_id}")

    async def _send_whatsapp(self, notification: EscalationNotification) -> None:
        """Send WhatsApp interactive button message via Cloud API."""
        session = await self._get_session()
        url = (
            f"https://graph.facebook.com/{WHATSAPP_API_VERSION}"
            f"/{self._config.whatsapp_phone_number_id}/messages"
        )

        headers = {
            "Authorization": f"Bearer {self._config.whatsapp_access_token}",
            "Content-Type": "application/json",
        }

        # WhatsApp interactive button message (up to 3 buttons)
        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": self._config.whatsapp_recipient_number,
            "type": "interactive",
            "interactive": {
                "type": "button",
                "header": {
                    "type": "text",
                    "text": f"TELOS Escalation [{notification.risk_tier.upper()}]",
                },
                "body": {
                    "text": notification.format_message(),
                },
                "action": {
                    "buttons": [
                        {
                            "type": "reply",
                            "reply": {
                                "id": f"approve:{notification.escalation_id}:{notification.challenge}",
                                "title": "Approve Override",
                            },
                        },
                        {
                            "type": "reply",
                            "reply": {
                                "id": f"deny:{notification.escalation_id}:{notification.challenge}",
                                "title": "Deny (Block)",
                            },
                        },
                    ]
                },
            },
        }

        async with session.post(url, json=payload, headers=headers) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(f"WhatsApp API error {resp.status}: {body}")
            logger.info(f"WhatsApp notification sent: {notification.escalation_id}")

    async def _send_discord(self, notification: EscalationNotification) -> None:
        """Send Discord webhook notification with rich embed."""
        session = await self._get_session()

        risk_color = {
            "critical": 0xFF0000,  # Red
            "high": 0xFF8C00,     # Orange
            "medium": 0xFFD700,   # Gold
            "low": 0x32CD32,      # Green
        }.get(notification.risk_tier, 0x808080)

        embed = {
            "title": f"TELOS Escalation — {notification.tool_name}",
            "description": notification.explanation[:1024] if notification.explanation else "Human review required",
            "color": risk_color,
            "fields": [
                {"name": "Tool Group", "value": notification.tool_group, "inline": True},
                {"name": "Risk Tier", "value": notification.risk_tier.upper(), "inline": True},
                {"name": "Fidelity", "value": f"{notification.fidelity_score:.3f}", "inline": True},
                {"name": "Escalation ID", "value": f"`{notification.escalation_id}`", "inline": False},
            ],
            "footer": {"text": "Respond via Telegram, WhatsApp, or CLI: telos agent approve <id>"},
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(notification.timestamp)),
        }

        if notification.boundary_triggered:
            embed["fields"].insert(0, {
                "name": "Boundary Violation",
                "value": "Detected",
                "inline": True,
            })

        payload = {
            "username": "TELOS Governance",
            "embeds": [embed],
        }

        async with session.post(self._config.discord_webhook_url, json=payload) as resp:
            if resp.status not in (200, 204):
                body = await resp.text()
                raise RuntimeError(f"Discord webhook error {resp.status}: {body}")
            logger.info(f"Discord notification sent: {notification.escalation_id}")
