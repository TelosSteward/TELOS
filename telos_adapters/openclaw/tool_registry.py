"""
Tool Registry — Risk-tiered tool registration for OpenClaw governance.

Every tool must be registered before first use. Registration scores the tool's
description against the PA via cosine similarity and assigns a scrutiny tier.
Registration is an auditable scope DECLARATION, NOT a blocking enforcement gate.
Action-level scoring remains primary — registration adds observability.

Design principle (Russell): Registration is scope declaration, not permission request.
Design principle (Bengio): Registration is NOT a safety gate. Action scoring is.
Design principle (Karpathy): MCP tools are highest-value — genuinely unknown tools.

Scrutiny tiers:
    HIGH    (fidelity < 0.50) — tool outside PA scope, watch closely
    MODERATE (0.50-0.75)      — tool partially aligned, standard monitoring
    LOW     (> 0.75)          — tool well-aligned with PA scope

Storage: NDJSON append-only (~/.telos/tool_registry.json), matching AuditWriter pattern.
In-memory dict for O(1) lookup. Load from NDJSON on startup. Idempotent.

Persistence: Dual-write pattern — local NDJSON always, Supabase best-effort.
Each registration record is Ed25519-signed for cryptographic integrity.

PA hash invalidation: on startup and periodically, compare current PA hash.
If different, invalidate_all() — log de-registration events, clear in-memory dict.

Design rationale: research/A11_TOOL_REGISTRATION_REVIEW.md
"""

import hashlib
import json
import logging
import os
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_REGISTRY_PATH = Path.home() / ".telos" / "tool_registry.json"
DEFAULT_MANIFEST_PATH = Path.home() / ".telos" / "capability_manifest.json"

# Scrutiny tier thresholds (from spec)
SCRUTINY_HIGH_THRESHOLD = 0.50
SCRUTINY_MODERATE_THRESHOLD = 0.75


class ScrutinyTier(str, Enum):
    """Risk-based scrutiny tier derived from registration fidelity."""
    HIGH = "high"          # fidelity < 0.50
    MODERATE = "moderate"  # 0.50 <= fidelity < 0.75
    LOW = "low"            # fidelity >= 0.75


class RegistrationSource(str, Enum):
    """How the tool was registered."""
    PRE_MAPPED = "pre_mapped"                    # From OPENCLAW_TOOL_MAP (curated)
    ANTICIPATORY_DECLARATION = "anticipatory_declaration"  # Batch at session start
    RUNTIME_DISCOVERY = "runtime_discovery"       # Inline during operation


def _compute_scrutiny_tier(fidelity: float) -> ScrutinyTier:
    """Derive scrutiny tier from registration fidelity score."""
    if fidelity < SCRUTINY_HIGH_THRESHOLD:
        return ScrutinyTier.HIGH
    elif fidelity < SCRUTINY_MODERATE_THRESHOLD:
        return ScrutinyTier.MODERATE
    else:
        return ScrutinyTier.LOW


@dataclass
class ToolRegistration:
    """A single tool's registration record."""
    tool_name: str
    description: str
    registration_fidelity: float
    scrutiny_tier: str  # ScrutinyTier value
    pa_hash: str
    registration_timestamp: float = field(default_factory=time.time)
    source: str = RegistrationSource.RUNTIME_DISCOVERY.value
    batch_id: Optional[str] = None


class ToolRegistry:
    """
    Registry of tools registered against a PA specification.

    Tools are registered on first use (or batch at session start).
    Registration NEVER blocks a tool — it establishes a baseline
    scrutiny tier for monitoring at the action level.

    Optional Ed25519 signing: if a ReceiptSigner is provided, every
    registration record is cryptographically signed. Canonical payload:
    tool_name + description + fidelity_score + pa_hash + timestamp.

    Optional Supabase persistence: if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY
    env vars are set, records are dual-written to the tool_registration_log table.
    Supabase writes are best-effort — failures are logged, never blocking.
    """

    def __init__(
        self,
        pa_hash: str,
        registry_path: Optional[Path] = None,
        signer=None,
        session_id: str = "",
    ):
        """
        Args:
            pa_hash: SHA-256 hash of the PA config this registry is bound to.
            registry_path: Path to NDJSON registry file.
            signer: Optional ReceiptSigner for Ed25519 signing of records.
            session_id: Optional session identifier for Supabase persistence.
        """
        self._pa_hash = pa_hash
        self._path = registry_path or DEFAULT_REGISTRY_PATH
        self._manifest_path = self._path.parent / "capability_manifest.json"
        self._registry: Dict[str, ToolRegistration] = {}
        self._signer = signer
        self._session_id = session_id
        self._supabase_url = os.environ.get("SUPABASE_URL", "")
        self._supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")

        # Dynamic Capability Manifest state (A11-T6)
        self._deregistered: List[Dict] = []  # De-registered tools history
        self._pa_lineage: List[str] = [pa_hash]  # PA hash versions seen
        self._registration_events: int = 0  # Total registration events this session

        self._path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing registrations from NDJSON
        self._load_from_disk()

    def _load_from_disk(self) -> None:
        """Load existing registrations from NDJSON file."""
        if not self._path.exists():
            return

        loaded = 0
        try:
            with open(self._path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        event = record.get("event", "")
                        data = record.get("data", {})

                        if event == "tool_registered" and data.get("pa_hash") == self._pa_hash:
                            name = data.get("tool_name", "")
                            if name and name not in self._registry:
                                self._registry[name] = ToolRegistration(
                                    tool_name=name,
                                    description=data.get("description", ""),
                                    registration_fidelity=data.get("registration_fidelity", 0.0),
                                    scrutiny_tier=data.get("scrutiny_tier", ScrutinyTier.HIGH.value),
                                    pa_hash=data.get("pa_hash", ""),
                                    registration_timestamp=data.get("registration_timestamp", 0.0),
                                    source=data.get("source", RegistrationSource.RUNTIME_DISCOVERY.value),
                                    batch_id=data.get("batch_id"),
                                )
                                loaded += 1
                    except json.JSONDecodeError:
                        continue
        except OSError as e:
            logger.warning("Failed to load tool registry from %s: %s", self._path, e)

        if loaded > 0:
            logger.info("Loaded %d tool registrations from disk (PA hash %s…)", loaded, self._pa_hash[:12])

    def _sign_data(self, data: dict) -> Tuple[str, str]:
        """Sign a data dict with Ed25519.

        Canonical payload: sorted JSON with compact separators, SHA-256 hashed,
        then signed with Ed25519. Matches ReceiptSigner canonicalization pattern.

        Returns:
            (signature_hex, public_key_hex) or ("", "") if no signer.
        """
        if self._signer is None:
            return "", ""

        try:
            canonical = json.dumps(
                data, sort_keys=True, separators=(",", ":"), default=str,
            ).encode("utf-8")
            payload_hash = hashlib.sha256(canonical).digest()
            sig = self._signer.sign_payload(payload_hash)
            pub = self._signer.public_key_bytes()
            return sig.hex(), pub.hex()
        except Exception as e:
            logger.warning("Ed25519 signing failed: %s", e)
            return "", ""

    def _emit(self, event_type: str, data: dict) -> None:
        """Append an NDJSON record with optional Ed25519 signature and Supabase dual-write."""
        # Sign the record data if signer is available
        sig_hex, pub_hex = self._sign_data(data)
        if sig_hex:
            data["crypto_proof_type"] = "ed25519"
            data["crypto_proof_value"] = sig_hex
            data["public_key"] = pub_hex

        record = {
            "event": event_type,
            "timestamp": time.time(),
            "data": data,
        }

        # Local NDJSON write (always)
        try:
            with open(self._path, "a") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except OSError as e:
            logger.warning("Registry write failed for %s: %s", event_type, e)

        # Supabase dual-write (best-effort)
        self._persist_to_supabase(event_type, data)

    def _persist_to_supabase(self, event_type: str, data: dict) -> None:
        """Best-effort write to Supabase tool_registration_log table.

        Uses the PostgREST API via urllib (stdlib, no extra dependency).
        Failures are logged at DEBUG level — never blocking.
        """
        if not self._supabase_url or not self._supabase_key:
            return

        # Map event types to registration_type column values
        reg_type = data.get("source", event_type)
        if event_type == "tool_deregistered":
            reg_type = "de_registration"

        try:
            row = {
                "session_id": self._session_id or "unknown",
                "registration_type": reg_type,
                "tool_name": data.get("tool_name", ""),
                "tool_description": data.get("description", ""),
                "registration_fidelity": data.get("registration_fidelity"),
                "scrutiny_tier": data.get("scrutiny_tier"),
                "pa_hash": data.get("pa_hash", self._pa_hash),
                "batch_id": data.get("batch_id"),
                "crypto_proof_type": data.get("crypto_proof_type", ""),
                "crypto_proof_value": data.get("crypto_proof_value", ""),
                "agent_id": "openclaw",
                "metadata": json.dumps({
                    "event_type": event_type,
                    "registration_timestamp": data.get("registration_timestamp"),
                }),
            }

            url = f"{self._supabase_url}/rest/v1/tool_registration_log"
            body = json.dumps(row).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=body,
                method="POST",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self._supabase_key}",
                    "apikey": self._supabase_key,
                    "Prefer": "return=minimal",
                },
            )
            urllib.request.urlopen(req, timeout=5)
        except Exception as e:
            logger.debug("Supabase write skipped for %s: %s", event_type, e)

    def is_registered(self, tool_name: str) -> bool:
        """Check if a tool is registered (O(1) dict lookup)."""
        return tool_name in self._registry

    def register(
        self,
        tool_name: str,
        description: str,
        fidelity_score: float,
        source: RegistrationSource = RegistrationSource.RUNTIME_DISCOVERY,
        batch_id: Optional[str] = None,
    ) -> ToolRegistration:
        """
        Register a tool against the current PA.

        Idempotent — if the tool is already registered under the same PA hash,
        the existing registration is returned unchanged.

        Registration NEVER fails. Even a tool with fidelity 0.0 is registered.
        The scrutiny tier determines how closely action-level scoring watches it.
        """
        # Idempotent: return existing if already registered
        if tool_name in self._registry:
            return self._registry[tool_name]

        tier = _compute_scrutiny_tier(fidelity_score)
        reg = ToolRegistration(
            tool_name=tool_name,
            description=description,
            registration_fidelity=fidelity_score,
            scrutiny_tier=tier.value,
            pa_hash=self._pa_hash,
            source=source.value,
            batch_id=batch_id,
        )

        self._registry[tool_name] = reg
        self._emit("tool_registered", asdict(reg))
        self._registration_events += 1
        self._persist_manifest()

        logger.info(
            "Tool registered: %s (fidelity=%.3f, tier=%s, source=%s)",
            tool_name, fidelity_score, tier.value, source.value,
        )
        return reg

    def get(self, tool_name: str) -> Optional[ToolRegistration]:
        """Get a tool's registration record, or None if not registered."""
        return self._registry.get(tool_name)

    def invalidate_all(self, new_pa_hash: str) -> List[str]:
        """
        Invalidate all registrations due to PA hash change.

        Logs de-registration events for each tool, clears in-memory dict,
        and updates the PA hash. Returns list of de-registered tool names.

        This is the correct behavior: when the PA changes, all prior tool
        registrations are no longer valid because the fidelity scores were
        computed against the old PA.
        """
        if not self._registry:
            self._pa_hash = new_pa_hash
            return []

        deregistered = list(self._registry.keys())
        dereg_time = time.time()

        for name, reg in self._registry.items():
            dereg_record = {
                "tool_name": name,
                "previous_pa_hash": self._pa_hash,
                "new_pa_hash": new_pa_hash,
                "previous_fidelity": reg.registration_fidelity,
                "previous_scrutiny_tier": reg.scrutiny_tier,
            }
            self._emit("tool_deregistered", dereg_record)
            self._deregistered.append({
                **dereg_record,
                "deregistered_at": dereg_time,
            })

        logger.info(
            "PA hash changed (%s… → %s…): de-registered %d tools",
            self._pa_hash[:12], new_pa_hash[:12], len(deregistered),
        )

        self._registry.clear()
        self._pa_hash = new_pa_hash
        if new_pa_hash not in self._pa_lineage:
            self._pa_lineage.append(new_pa_hash)
        self._persist_manifest()
        return deregistered

    def batch_register(
        self,
        tools: List[Dict[str, str]],
        score_fn,
        source: RegistrationSource = RegistrationSource.PRE_MAPPED,
        batch_id: Optional[str] = None,
    ) -> List[ToolRegistration]:
        """
        Register multiple tools in a single pass.

        Args:
            tools: List of {"name": str, "description": str} dicts.
            score_fn: Callable(tool_name, description) -> float (fidelity score).
            source: Registration source type.
            batch_id: Optional batch identifier for grouping.

        Returns:
            List of ToolRegistration records.
        """
        if batch_id is None:
            batch_id = hashlib.sha256(
                f"{time.time()}:{len(tools)}".encode()
            ).hexdigest()[:16]

        results = []
        for tool in tools:
            name = tool.get("name", "")
            desc = tool.get("description", "")
            if not name:
                continue

            fidelity = score_fn(name, desc)
            reg = self.register(
                tool_name=name,
                description=desc,
                fidelity_score=fidelity,
                source=source,
                batch_id=batch_id,
            )
            results.append(reg)

        # Emit a batch summary record with collective signature
        if results:
            batch_summary = {
                "batch_id": batch_id,
                "tool_count": len(results),
                "tools": [r.tool_name for r in results],
                "pa_hash": self._pa_hash,
                "source": source.value,
                "timestamp": time.time(),
            }
            self._emit("batch_registered", batch_summary)

        logger.info(
            "Batch registered %d tools (batch=%s, source=%s)",
            len(results), batch_id[:12], source.value,
        )
        return results

    def get_manifest(self) -> dict:
        """
        Generate a Dynamic Capability Manifest (Schaake compliance artifact).

        Returns a structured summary of all registered tools, their fidelity
        scores, scrutiny tiers, registration metadata, de-registration history,
        growth statistics, and PA hash lineage.

        Optionally Ed25519-signed if a signer is configured.
        """
        tools = {}
        for name, reg in sorted(self._registry.items()):
            tools[name] = {
                "description": reg.description,
                "registration_fidelity": reg.registration_fidelity,
                "scrutiny_tier": reg.scrutiny_tier,
                "source": reg.source,
                "registered_at": reg.registration_timestamp,
                "batch_id": reg.batch_id,
            }

        # Tier distribution
        tier_counts = {t.value: 0 for t in ScrutinyTier}
        for reg in self._registry.values():
            tier_counts[reg.scrutiny_tier] = tier_counts.get(reg.scrutiny_tier, 0) + 1

        manifest = {
            "pa_hash": self._pa_hash,
            "total_registered": len(self._registry),
            "tier_distribution": tier_counts,
            "tools": tools,
            "deregistered": self._deregistered,
            "pa_lineage": self._pa_lineage,
            "growth_stats": {
                "registration_events_this_session": self._registration_events,
                "deregistration_events_total": len(self._deregistered),
            },
            "session_id": self._session_id,
            "generated_at": time.time(),
        }

        # Sign the manifest if signer is available
        sig_hex, pub_hex = self._sign_data(manifest)
        if sig_hex:
            manifest["crypto_proof_type"] = "ed25519"
            manifest["crypto_proof_value"] = sig_hex
            manifest["public_key"] = pub_hex

        return manifest

    def _persist_manifest(self) -> None:
        """Write the current manifest to ~/.telos/capability_manifest.json.

        Called after each registration and de-registration event.
        Best-effort — failures are logged, never blocking.
        """
        try:
            manifest = self.get_manifest()
            with open(self._manifest_path, "w") as f:
                json.dump(manifest, f, indent=2, default=str)
        except Exception as e:
            logger.debug("Manifest persistence failed: %s", e)

    @property
    def pa_hash(self) -> str:
        """Current PA hash this registry is bound to."""
        return self._pa_hash

    @property
    def size(self) -> int:
        """Number of registered tools."""
        return len(self._registry)

    def __len__(self) -> int:
        return len(self._registry)
