"""
Watchdog — Process lifecycle management for the TELOS governance daemon.

Manages the Python governance process that runs alongside OpenClaw:
    - PID file management (~/.openclaw/hooks/telos.pid)
    - Heartbeat monitoring (30s interval, graduated restart)
    - SIGTERM handler for graceful shutdown
    - Health endpoint for `telos agent status`
    - Auto-restart on crash (configurable)

Design decisions (Karpathy + Sorhus M0 analysis):
    - Detached daemon: Survives OpenClaw restarts without 3-5s model reload
    - PID file: Standard Unix daemon pattern, checked by `telos service status`
    - Graduated restart: 1s, 5s, 15s, 30s backoff on repeated crashes
    - Heartbeat: Governance process writes timestamp; watchdog checks staleness

Regulatory traceability:
    - EU AI Act Art. 15: Robustness — graduated restart ensures governance
      continuity even after crashes
    - EU AI Act Art. 72: Post-market monitoring — heartbeat detects governance
      process failures, ensuring no unmonitored execution gaps
    - SAAI claim TELOS-SAAI-012: Fail-closed behavior when daemon is down
    - OWASP ASI08 (Cascading Failures): Watchdog prevents governance gap
      from propagating as unmonitored execution
    See: research/openclaw_regulatory_mapping.md §3
"""

import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Default paths
DEFAULT_PID_PATH = Path.home() / ".openclaw" / "hooks" / "telos.pid"
DEFAULT_HEARTBEAT_PATH = Path.home() / ".openclaw" / "hooks" / "telos.heartbeat"

# Heartbeat configuration
HEARTBEAT_INTERVAL = 30  # seconds
HEARTBEAT_STALE_THRESHOLD = 90  # 3x interval = considered dead

# Graduated restart backoff (seconds)
RESTART_BACKOFF = [1, 5, 15, 30]


class Watchdog:
    """Process lifecycle manager for the TELOS governance daemon.

    Handles PID file, heartbeat, signal handling, and health checks.

    Usage:
        watchdog = Watchdog()
        watchdog.start(main_fn=my_server_start_fn)
    """

    def __init__(
        self,
        pid_path: Optional[Path] = None,
        heartbeat_path: Optional[Path] = None,
    ):
        self._pid_path = pid_path or DEFAULT_PID_PATH
        self._heartbeat_path = heartbeat_path or DEFAULT_HEARTBEAT_PATH
        self._shutdown_requested = False
        self._on_shutdown: Optional[Callable] = None
        self._start_time: Optional[float] = None
        self._heartbeat_thread: Optional[threading.Thread] = None

    def start(
        self,
        main_fn: Callable,
        on_shutdown: Optional[Callable] = None,
    ) -> None:
        """Start the daemon with lifecycle management.

        Args:
            main_fn: The main function to run (e.g., IPCServer.run_sync).
                Should block until completion.
            on_shutdown: Optional callback for cleanup on shutdown.
        """
        self._on_shutdown = on_shutdown

        # Check for existing process
        if self.is_running():
            existing_pid = self._read_pid()
            logger.error(
                f"TELOS governance daemon already running (PID {existing_pid}). "
                f"Stop it first: telos service stop"
            )
            sys.exit(1)

        # Write PID file
        self._write_pid()
        self._start_time = time.time()

        # Register signal handlers
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

        logger.info(f"Watchdog started (PID {os.getpid()})")

        try:
            # Start periodic heartbeat in background thread
            self._write_heartbeat()
            self._heartbeat_thread = threading.Thread(
                target=self._heartbeat_loop, daemon=True
            )
            self._heartbeat_thread.start()

            # Run main function
            main_fn()

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        except Exception as e:
            logger.error(f"Daemon crashed: {e}", exc_info=True)
            raise
        finally:
            self._cleanup()

    def _handle_signal(self, signum, frame) -> None:
        """Handle SIGTERM/SIGINT gracefully."""
        sig_name = signal.Signals(signum).name
        logger.info(f"Received {sig_name}, shutting down gracefully...")
        self._shutdown_requested = True

        if self._on_shutdown:
            try:
                self._on_shutdown()
            except Exception as e:
                logger.error(f"Shutdown callback error: {e}")

        self._cleanup()
        sys.exit(0)

    def _write_pid(self) -> None:
        """Write PID file."""
        self._pid_path.parent.mkdir(parents=True, exist_ok=True)
        self._pid_path.write_text(str(os.getpid()))
        os.chmod(str(self._pid_path), 0o600)
        logger.debug(f"PID file written: {self._pid_path}")

    def _read_pid(self) -> Optional[int]:
        """Read PID from file."""
        try:
            return int(self._pid_path.read_text().strip())
        except (FileNotFoundError, ValueError):
            return None

    def _write_heartbeat(self) -> None:
        """Write heartbeat timestamp."""
        self._heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
        self._heartbeat_path.write_text(str(time.time()))

    def _heartbeat_loop(self) -> None:
        """Periodically update heartbeat file until shutdown."""
        while not self._shutdown_requested:
            time.sleep(HEARTBEAT_INTERVAL)
            if self._shutdown_requested:
                break
            try:
                self._write_heartbeat()
            except Exception as e:
                logger.warning(f"Heartbeat write failed: {e}")

    def _cleanup(self) -> None:
        """Remove PID and heartbeat files."""
        for path in (self._pid_path, self._heartbeat_path):
            try:
                if path.exists():
                    path.unlink()
            except OSError as e:
                logger.warning(f"Failed to remove {path}: {e}")
        logger.info("Watchdog cleanup complete")

    def is_running(self) -> bool:
        """Check if the daemon is currently running.

        Reads the PID file and checks if the process exists.
        Cleans up stale PID files.
        """
        pid = self._read_pid()
        if pid is None:
            return False

        try:
            os.kill(pid, 0)  # Check if process exists (no signal sent)
            return True
        except ProcessLookupError:
            # Stale PID file — process is dead
            logger.info(f"Removing stale PID file (PID {pid} not running)")
            self._pid_path.unlink(missing_ok=True)
            return False
        except PermissionError:
            # Process exists but we can't signal it
            return True

    def health_check(self) -> dict:
        """Check daemon health.

        Returns:
            Dict with status, uptime, heartbeat info.
        """
        pid = self._read_pid()
        running = self.is_running()

        # Check heartbeat freshness
        heartbeat_stale = False
        heartbeat_age = None
        try:
            ts = float(self._heartbeat_path.read_text().strip())
            heartbeat_age = time.time() - ts
            heartbeat_stale = heartbeat_age > HEARTBEAT_STALE_THRESHOLD
        except (FileNotFoundError, ValueError):
            heartbeat_stale = True

        uptime = time.time() - self._start_time if self._start_time else None

        status = "ok"
        if not running:
            status = "stopped"
        elif heartbeat_stale:
            status = "stale"

        return {
            "status": status,
            "pid": pid,
            "running": running,
            "uptime_seconds": round(uptime, 1) if uptime else None,
            "heartbeat_age_seconds": round(heartbeat_age, 1) if heartbeat_age else None,
            "heartbeat_stale": heartbeat_stale,
        }

    @property
    def pid_path(self) -> Path:
        """Path to the PID file."""
        return self._pid_path
