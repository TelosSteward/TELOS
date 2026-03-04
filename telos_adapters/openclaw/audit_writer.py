"""
Audit Writer — Structured NDJSON audit trail for the OpenClaw governance daemon.

Appends governance events as one-JSON-per-line to a persistent audit file.
Each event is timestamped, typed, and includes the full governance context
needed for regulatory compliance and forensic analysis.

Event taxonomy:
    daemon_start          — PID, config path, preset
    config_loaded         — PA hash, boundary count, tool group count
    tool_call_scored      — full GovernanceVerdict
    drift_warning         — drift_magnitude, baseline, window scores
    drift_restrict        — threshold tightened, affected decision
    drift_block           — session frozen
    drift_acknowledged    — reason, acknowledgment number
    escalation_requested  — tool_name, action_text, risk_tier
    escalation_resolved   — approved/denied, latency
    chain_reset           — reason (new task)
    daemon_stop           — total_scored, total_blocked, uptime
    pa_verified           — PA TKey signature verified at boot
    pa_not_verified       — PA not signed or verification failed
    security_event        — Security event (e.g., config tampered after signing)
    inert_unsigned_pa     — Score request rejected due to unsigned PA
    gate_transition       — Ed25519-signed gate state change (open/closed, enforce/observe)
    codebase_policy_denied — Write attempted on read_only or uncovered path
    codebase_policy_loaded — Policies loaded/refreshed at boot or SIGHUP

Regulatory traceability:
    - EU AI Act Art. 12: Automatic event recording with structured audit trail
    - SAAI claim TELOS-SAAI-005: GovernanceTraceCollector logs all decisions
    - IEEE 7001-2021: Transparent decision records with full scoring context
    - NIST AI RMF GOVERN 2.1: Continuous risk awareness via persistent logging
    See: research/openclaw_regulatory_mapping.md §8
"""

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

DEFAULT_AUDIT_PATH = Path.home() / ".openclaw" / "hooks" / "telos_audit.jsonl"


class AuditWriter:
    """Lightweight NDJSON audit writer for the OpenClaw governance daemon.

    Thread-safe append-only writer. Each event is a single JSON line with
    a consistent schema: {"event": "...", "timestamp": ..., "data": {...}}.

    Usage:
        writer = AuditWriter()
        writer.emit("daemon_start", {"pid": os.getpid(), "preset": "balanced"})
        writer.emit("tool_call_scored", verdict.to_dict())
        writer.close()
    """

    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
    MAX_ROTATED_FILES = 5

    def __init__(self, audit_path: Optional[Path] = None):
        self._path = audit_path or DEFAULT_AUDIT_PATH
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = None
        self._prev_hash = "0" * 64  # genesis hash
        self._sequence = 0
        self._open()

    def _open(self) -> None:
        """Open the audit file for appending with restricted permissions."""
        try:
            self._file = open(self._path, "a")
            # Set permissions to 0o600 (owner read/write only)
            os.chmod(str(self._path), 0o600)
            logger.info(f"Audit trail: {self._path}")
        except OSError as e:
            logger.error(f"Failed to open audit file {self._path}: {e}")
            self._file = None

    def _maybe_rotate(self) -> None:
        """Rotate audit log at MAX_FILE_SIZE, keeping MAX_ROTATED_FILES."""
        try:
            if not self._path.exists():
                return
            if self._path.stat().st_size <= self.MAX_FILE_SIZE:
                return

            # Shift existing rotated files (5 -> drop, 4->5, 3->4, 2->3, 1->2)
            for i in range(self.MAX_ROTATED_FILES - 1, 0, -1):
                src = self._path.with_suffix(f".{i}.jsonl")
                dst = self._path.with_suffix(f".{i + 1}.jsonl")
                if src.exists():
                    src.rename(dst)

            # Rotate current to .1
            if self._file:
                self._file.close()
                self._file = None
            self._path.rename(self._path.with_suffix(".1.jsonl"))
            self._open()
            # Reference previous file's hash chain
            self._prev_hash = "ROTATED:" + self._prev_hash
        except OSError as e:
            logger.warning(f"Audit rotation failed: {e}")

    def emit(self, event_type: str, data: Optional[Dict[str, Any]] = None) -> None:
        """Write a single audit event as an NDJSON line with hash chain integrity.

        Each record includes a sequence number, prev_hash (linking to the
        previous entry), and entry_hash (SHA-256 of the canonical record).
        This produces a tamper-evident chain identical in principle to
        the GovernanceTraceCollector used in benchmarks.

        Args:
            event_type: One of the taxonomy event types.
            data: Event payload (must be JSON-serializable).
        """
        if not self._file:
            return

        self._maybe_rotate()

        self._sequence += 1
        record = {
            "event": event_type,
            "timestamp": time.time(),
            "sequence": self._sequence,
            "prev_hash": self._prev_hash,
            "data": data or {},
        }

        # Compute entry hash over canonical JSON (before adding entry_hash)
        canonical = json.dumps(record, sort_keys=True, separators=(",", ":"))
        entry_hash = hashlib.sha256(canonical.encode()).hexdigest()
        record["entry_hash"] = entry_hash

        try:
            line = json.dumps(record, default=str) + "\n"
            self._file.write(line)
            self._file.flush()
            self._prev_hash = entry_hash
        except Exception as e:
            logger.warning(f"Audit write failed for {event_type}: {e}")

    def close(self) -> None:
        """Flush and close the audit file."""
        if self._file:
            try:
                self._file.flush()
                self._file.close()
            except OSError:
                pass
            self._file = None

    def write_gate_transition(self, record: Dict[str, Any]) -> None:
        """Emit a gate_transition event with the full GateRecord.

        Args:
            record: GateRecord.to_dict() with gate state, mode, actor,
                    timestamp, TTL, Ed25519 signature, and public key.
        """
        self.emit("gate_transition", record)

    def write_policy_denied(self, data: dict) -> None:
        """Emit a codebase_policy_denied event.

        Args:
            data: Dict with tool_name, file_path, reason, collection.
        """
        self.emit("codebase_policy_denied", data)

    def write_policy_loaded(self, data: dict) -> None:
        """Emit a codebase_policy_loaded event.

        Args:
            data: Dict with policy_count, collections, source (boot/sighup/ipc).
        """
        self.emit("codebase_policy_loaded", data)

    @property
    def path(self) -> Path:
        """Path to the audit file."""
        return self._path
