"""
IPC Server — Unix Domain Socket server for TypeScript plugin communication.

Protocol: NDJSON (newline-delimited JSON) over Unix Domain Socket.
Socket path: ~/.openclaw/hooks/telos.sock (configurable)
Round-trip latency: 0.05-0.2ms IPC + 10-17ms scoring = ~15ms total.

Message format (TypeScript -> Python):
    {"type": "score", "tool_name": "Bash", "action_text": "rm -rf /", "args": {...}}
    {"type": "health"}
    {"type": "reset_chain"}

Response format (Python -> TypeScript):
    {"type": "verdict", "request_id": "...", "data": {...GovernanceVerdict...}}
    {"type": "health", "status": "ok", "stats": {...}}
    {"type": "error", "message": "..."}

Design decisions (Karpathy M0 systems analysis):
    - UDS over HTTP: 0.05-0.2ms vs 1-5ms round-trip
    - NDJSON over MessagePack: Debuggable with socat/jq, <1KB payloads
    - Asyncio for concurrent connections (OpenClaw may have parallel tool calls)
    - Graceful degradation: if socket write fails, return fail-policy default

Regulatory traceability:
    - EU AI Act Art. 12: NDJSON messages form an automatic event log
    - EU AI Act Art. 72: Real-time scoring channel for continuous monitoring
    - IEEE 7001-2021: NDJSON protocol enables retrospective transparency
    - SAAI claim TELOS-SAAI-009: IPC channel implements always-on governance
    - OWASP ASI07 (Insecure Inter-Agent Comms): UDS with 0o600 permissions
      ensures local-only authenticated communication
    See: research/openclaw_regulatory_mapping.md §3, §5
"""

import asyncio
import json
import logging
import os
import signal
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

logger = logging.getLogger(__name__)

# Default socket path
DEFAULT_SOCKET_PATH = Path.home() / ".openclaw" / "hooks" / "telos.sock"

# Maximum message size (safety limit)
MAX_MESSAGE_SIZE = 64 * 1024  # 64KB — well above typical <1KB payloads


@dataclass
class IPCMessage:
    """Parsed message from the TypeScript plugin."""
    type: str  # "score", "health", "reset_chain", "shutdown"
    request_id: str = ""
    tool_name: str = ""
    action_text: str = ""
    args: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "IPCMessage":
        """Parse from JSON dict."""
        return cls(
            type=data.get("type", ""),
            request_id=data.get("request_id", str(uuid.uuid4())[:8]),
            tool_name=data.get("tool_name", ""),
            action_text=data.get("action_text", ""),
            args=data.get("args", {}),
            timestamp=data.get("timestamp", time.time()),
        )


@dataclass
class IPCResponse:
    """Response to send back to the TypeScript plugin."""
    type: str  # "verdict", "health", "error", "ack"
    request_id: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    error: str = ""
    latency_ms: float = 0.0

    def to_json(self) -> str:
        """Serialize to NDJSON line."""
        payload = {
            "type": self.type,
            "request_id": self.request_id,
        }
        if self.data:
            payload["data"] = self.data
        if self.error:
            payload["error"] = self.error
        if self.latency_ms > 0:
            payload["latency_ms"] = round(self.latency_ms, 2)
        return json.dumps(payload)


class IPCServer:
    """Unix Domain Socket server for TELOS-OpenClaw IPC.

    Listens on a UDS and handles NDJSON messages from the TypeScript plugin.
    Each message is processed by the registered handler (typically GovernanceHook).

    Usage:
        server = IPCServer(socket_path="/tmp/telos.sock")
        server.set_handler(my_handler_fn)
        await server.start()  # or server.run_sync()
    """

    def __init__(
        self,
        socket_path: Optional[str] = None,
        handler: Optional[Callable[[IPCMessage], Union[IPCResponse, Any]]] = None,
    ):
        """Initialize the IPC server.

        Args:
            socket_path: Path to the Unix socket. Defaults to
                ~/.openclaw/hooks/telos.sock
            handler: Function that processes IPCMessage -> IPCResponse.
                Can be sync or async. Set via set_handler() if not provided.
        """
        self._socket_path = Path(socket_path) if socket_path else DEFAULT_SOCKET_PATH
        self._handler = handler
        self._server: Optional[asyncio.AbstractServer] = None
        self._running = False
        self._connections = 0

        # Stats
        self._total_messages = 0
        self._total_errors = 0
        self._start_time: Optional[float] = None

    def set_handler(self, handler: Callable[[IPCMessage], Union[IPCResponse, Any]]) -> None:
        """Set the message handler function (sync or async)."""
        self._handler = handler

    async def start(self) -> None:
        """Start the IPC server (async).

        Creates the socket directory if needed, removes stale socket files,
        and starts listening for connections.
        """
        if self._handler is None:
            raise ValueError("No handler set. Call set_handler() first.")

        # Ensure socket directory exists
        self._socket_path.parent.mkdir(parents=True, exist_ok=True)

        # Remove stale socket
        if self._socket_path.exists():
            logger.info(f"Removing stale socket: {self._socket_path}")
            self._socket_path.unlink()

        # Start server
        self._server = await asyncio.start_unix_server(
            self._handle_connection,
            path=str(self._socket_path),
        )

        # Set socket permissions (owner-only)
        os.chmod(str(self._socket_path), 0o600)

        self._running = True
        self._start_time = time.time()
        logger.info(f"IPC server listening on {self._socket_path}")

        async with self._server:
            await self._server.serve_forever()

    async def stop(self) -> None:
        """Stop the IPC server gracefully."""
        self._running = False
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        if self._socket_path.exists():
            self._socket_path.unlink()
        logger.info("IPC server stopped")

    def run_sync(self) -> None:
        """Run the server synchronously (blocking).

        Sets up signal handlers for graceful shutdown.
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Signal handlers for graceful shutdown
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, lambda: asyncio.ensure_future(self.stop()))

        try:
            loop.run_until_complete(self.start())
        except asyncio.CancelledError:
            pass
        finally:
            loop.close()

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handle a single client connection.

        Reads NDJSON messages line by line and sends responses.
        """
        self._connections += 1
        peer = writer.get_extra_info("peername") or "unknown"
        logger.debug(f"Client connected: {peer}")

        try:
            while self._running:
                line = await reader.readline()
                if not line:
                    break  # Connection closed

                if len(line) > MAX_MESSAGE_SIZE:
                    error_resp = IPCResponse(
                        type="error",
                        error=f"Message too large ({len(line)} bytes, max {MAX_MESSAGE_SIZE})",
                    )
                    writer.write((error_resp.to_json() + "\n").encode())
                    await writer.drain()
                    continue

                self._total_messages += 1

                try:
                    data = json.loads(line.decode().strip())
                    msg = IPCMessage.from_json(data)
                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    self._total_errors += 1
                    error_resp = IPCResponse(
                        type="error",
                        error=f"Invalid JSON: {e}",
                    )
                    writer.write((error_resp.to_json() + "\n").encode())
                    await writer.drain()
                    continue

                # Process message — handler may be sync or async.
                # Handler receives an optional send_interim callback for
                # two-phase responses (e.g., escalation_pending + verdict).
                start = time.perf_counter()
                try:
                    async def send_interim(interim_response: IPCResponse):
                        """Send an interim response before the final one."""
                        writer.write((interim_response.to_json() + "\n").encode())
                        await writer.drain()

                    # Try passing send_interim; fall back for handlers that
                    # don't accept it (backward compatibility).
                    try:
                        result = self._handler(msg, send_interim=send_interim)
                    except TypeError:
                        result = self._handler(msg)

                    # Support both sync and async handlers
                    if asyncio.iscoroutine(result):
                        response = await result
                    else:
                        response = result
                    response.latency_ms = (time.perf_counter() - start) * 1000
                except Exception as e:
                    self._total_errors += 1
                    logger.error(f"Handler error: {e}", exc_info=True)
                    response = IPCResponse(
                        type="error",
                        request_id=msg.request_id,
                        error=str(e),
                        latency_ms=(time.perf_counter() - start) * 1000,
                    )

                # Send response
                writer.write((response.to_json() + "\n").encode())
                await writer.drain()

        except asyncio.CancelledError:
            pass
        except ConnectionResetError:
            logger.debug(f"Client disconnected: {peer}")
        except Exception as e:
            logger.error(f"Connection error: {e}", exc_info=True)
        finally:
            self._connections -= 1
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    @property
    def socket_path(self) -> Path:
        """The socket file path."""
        return self._socket_path

    @property
    def is_running(self) -> bool:
        """Whether the server is currently running."""
        return self._running

    @property
    def stats(self) -> Dict[str, Any]:
        """Server statistics."""
        uptime = time.time() - self._start_time if self._start_time else 0
        return {
            "running": self._running,
            "socket_path": str(self._socket_path),
            "uptime_seconds": round(uptime, 1),
            "active_connections": self._connections,
            "total_messages": self._total_messages,
            "total_errors": self._total_errors,
        }
