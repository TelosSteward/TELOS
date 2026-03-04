"""
CUSUM Monitor — Cumulative Sum drift detection per tool group.

Implements CUSUM (Cumulative Sum) control charts for detecting persistent
slow fidelity degradation within individual tool groups. Catches drift
patterns that the session-level sliding-window tracker might miss.

Algorithm:
    S_n = max(0, S_{n-1} + (mu_0 - x_n) - k)

    Where:
        mu_0 = target mean fidelity (established from first N observations)
        x_n  = observed fidelity at step n
        k    = allowance (sensitivity parameter, detects shifts of 2k)
        S_n  = cumulative sum statistic
        h    = decision interval (alarm threshold)

    An alarm fires when S_n > h, indicating a sustained downward shift
    in fidelity for that tool group.

Parameters (Hawkins & Olwell 1998):
    k = 0.05 — detects shifts of 0.10 or more (half the target shift)
    h = 4.0  — ARL_0 ~168 (expected false alarm every ~168 observations)
    baseline_n = 20 — observations to establish adaptive target fidelity

Regulatory traceability:
    - EU AI Act Art. 72: Per-tool-group continuous monitoring
    - IEEE 7000: Statistical process control for governance quality
    - NIST AI RMF GOVERN 2.1: Continuous risk awareness via CUSUM
    - OWASP ASI10 (Rogue Agents): Per-group degradation detection
    See: research/openclaw_regulatory_mapping.md §3, §6
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# CUSUM parameters (Hawkins & Olwell 1998 recommendations)
CUSUM_K = 0.05       # Allowance: detects shifts of 2k = 0.10
CUSUM_H = 4.0        # Decision interval: ARL_0 ~168
CUSUM_BASELINE_N = 20  # Observations to establish adaptive baseline


@dataclass
class CUSUMAlert:
    """Alert from a CUSUM monitor when the alarm threshold is breached."""
    tool_group: str
    cusum_statistic: float
    threshold: float
    target_fidelity: float
    current_fidelity: float
    observation_count: int
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_group": self.tool_group,
            "cusum_statistic": round(self.cusum_statistic, 4),
            "threshold": self.threshold,
            "target_fidelity": round(self.target_fidelity, 4),
            "current_fidelity": round(self.current_fidelity, 4),
            "observation_count": self.observation_count,
            "timestamp": self.timestamp,
        }


class CUSUMMonitor:
    """CUSUM control chart for a single tool group.

    Tracks cumulative sum of downward fidelity deviations from an
    adaptive baseline (established from the first N observations).

    Usage:
        monitor = CUSUMMonitor("runtime")
        for fidelity in scores:
            alert = monitor.record(fidelity)
            if alert:
                print(f"CUSUM alarm: {alert.tool_group}")
    """

    def __init__(
        self,
        tool_group: str,
        k: float = CUSUM_K,
        h: float = CUSUM_H,
        baseline_n: int = CUSUM_BASELINE_N,
    ):
        self._tool_group = tool_group
        self._k = k
        self._h = h
        self._baseline_n = baseline_n

        self._observations: List[float] = []
        self._target_fidelity: Optional[float] = None
        self._baseline_established = False
        self._cusum: float = 0.0
        self._alarm_active = False

    def record(self, fidelity: float) -> Optional[CUSUMAlert]:
        """Record a fidelity observation and check for alarm.

        Args:
            fidelity: Fidelity score from governance scoring [0.0, 1.0].

        Returns:
            CUSUMAlert if the alarm threshold is breached, None otherwise.
        """
        fidelity = max(0.0, min(1.0, fidelity))
        self._observations.append(fidelity)

        # Phase 1: Baseline collection
        if not self._baseline_established:
            if len(self._observations) >= self._baseline_n:
                self._target_fidelity = (
                    sum(self._observations[:self._baseline_n]) / self._baseline_n
                )
                self._baseline_established = True
                logger.debug(
                    f"CUSUM [{self._tool_group}] baseline established: "
                    f"mu_0={self._target_fidelity:.4f} from {self._baseline_n} obs"
                )
            return None

        # Phase 2: CUSUM update
        # S_n = max(0, S_{n-1} + (mu_0 - x_n) - k)
        # Detects downward shift (mu_0 - x_n is positive when fidelity drops)
        self._cusum = max(0.0, self._cusum + (self._target_fidelity - fidelity) - self._k)

        # Check alarm
        if self._cusum > self._h and not self._alarm_active:
            self._alarm_active = True
            alert = CUSUMAlert(
                tool_group=self._tool_group,
                cusum_statistic=self._cusum,
                threshold=self._h,
                target_fidelity=self._target_fidelity,
                current_fidelity=fidelity,
                observation_count=len(self._observations),
            )
            logger.warning(
                f"CUSUM ALARM [{self._tool_group}]: S={self._cusum:.3f} > h={self._h} "
                f"(target={self._target_fidelity:.3f}, current={fidelity:.3f}, "
                f"n={len(self._observations)})"
            )
            return alert

        # Reset alarm if CUSUM returns to 0
        if self._cusum == 0.0:
            self._alarm_active = False

        return None

    def reset(self) -> None:
        """Reset the CUSUM statistic (not the baseline)."""
        self._cusum = 0.0
        self._alarm_active = False

    def full_reset(self) -> None:
        """Full reset including baseline (for new session)."""
        self._observations.clear()
        self._target_fidelity = None
        self._baseline_established = False
        self._cusum = 0.0
        self._alarm_active = False

    def status(self) -> Dict[str, Any]:
        """Current monitor status."""
        return {
            "tool_group": self._tool_group,
            "baseline_established": self._baseline_established,
            "target_fidelity": round(self._target_fidelity, 4) if self._target_fidelity else None,
            "cusum_statistic": round(self._cusum, 4),
            "alarm_active": self._alarm_active,
            "observation_count": len(self._observations),
            "threshold_h": self._h,
            "allowance_k": self._k,
        }

    @property
    def tool_group(self) -> str:
        return self._tool_group

    @property
    def alarm_active(self) -> bool:
        return self._alarm_active

    def to_dict(self):
        """Serialize monitor state for persistence."""
        return {
            "tool_group": self._tool_group,
            "cusum": self._cusum,
            "observations": list(self._observations),
            "target_fidelity": self._target_fidelity,
            "baseline_established": self._baseline_established,
            "alarm_active": self._alarm_active,
        }

    def restore(self, state):
        """Restore monitor state from a persisted snapshot."""
        if not state:
            return
        self._cusum = state.get("cusum", 0.0)
        self._observations = list(state.get("observations", []))
        self._target_fidelity = state.get("target_fidelity")
        self._baseline_established = state.get("baseline_established", False)
        self._alarm_active = state.get("alarm_active", False)


class CUSUMMonitorBank:
    """Bank of CUSUM monitors — one per tool group.

    Automatically creates monitors for new tool groups on first observation.

    Usage:
        bank = CUSUMMonitorBank()
        alert = bank.record("runtime", 0.85)
        alert = bank.record("fs", 0.90)
        status = bank.status()
    """

    def __init__(self, k: float = CUSUM_K, h: float = CUSUM_H):
        self._k = k
        self._h = h
        self._monitors: Dict[str, CUSUMMonitor] = {}

    def record(self, tool_group: str, fidelity: float) -> Optional[CUSUMAlert]:
        """Record a fidelity observation for a tool group.

        Creates a new monitor if this is the first observation for this group.

        Args:
            tool_group: The tool group name (e.g., "runtime", "fs", "network").
            fidelity: Fidelity score [0.0, 1.0].

        Returns:
            CUSUMAlert if alarm threshold breached, None otherwise.
        """
        if tool_group not in self._monitors:
            self._monitors[tool_group] = CUSUMMonitor(
                tool_group=tool_group, k=self._k, h=self._h
            )

        return self._monitors[tool_group].record(fidelity)

    def reset_all(self) -> None:
        """Reset all CUSUM statistics (preserve baselines)."""
        for monitor in self._monitors.values():
            monitor.reset()

    def full_reset(self) -> None:
        """Full reset of all monitors."""
        self._monitors.clear()

    def status(self) -> Dict[str, Any]:
        """Status of all monitors."""
        return {
            "monitor_count": len(self._monitors),
            "monitors": {
                name: mon.status() for name, mon in self._monitors.items()
            },
            "active_alarms": [
                name for name, mon in self._monitors.items() if mon.alarm_active
            ],
        }

    @property
    def active_alarms(self) -> List[str]:
        """Tool groups with active CUSUM alarms."""
        return [name for name, mon in self._monitors.items() if mon.alarm_active]

    def to_dict(self):
        """Serialize all monitor states for persistence."""
        return {
            group: monitor.to_dict()
            for group, monitor in self._monitors.items()
        }

    def restore(self, state):
        """Restore all monitors from a persisted snapshot."""
        if not state:
            return
        for group, monitor_state in state.items():
            if group not in self._monitors:
                self._monitors[group] = CUSUMMonitor(
                    tool_group=group, k=self._k, h=self._h
                )
            self._monitors[group].restore(monitor_state)
