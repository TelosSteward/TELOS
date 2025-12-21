"""
Governance Trace Collector for TELOS
=====================================

Central coordinator for all governance event logging.
Manages JSONL persistence with configurable privacy modes.

This is the single source of truth for governance observability data,
integrating fidelity calculations, interventions, and SSE streaming.
"""

import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .evidence_schema import (
    BaseEvent,
    EventType,
    FidelityCalculatedEvent,
    FidelityZone,
    GovernanceEvent,
    InterventionLevel,
    InterventionTriggeredEvent,
    PAEstablishedEvent,
    PrivacyMode,
    ResponseGeneratedEvent,
    SessionEndEvent,
    SessionStartEvent,
    SessionSummaryEvent,
    SSEGovernanceEvent,
    SSETokenEvent,
    TurnCompleteEvent,
    TurnStartEvent,
    fidelity_to_zone,
    serialize_event,
)

logger = logging.getLogger(__name__)


class GovernanceTraceCollector:
    """
    Central coordinator for governance event logging.

    Responsibilities:
    1. Collect events from various TELOS components
    2. Apply privacy mode transformations
    3. Persist to JSONL files
    4. Provide query interface for UI components
    5. Support real-time event callbacks for live visualization
    """

    def __init__(
        self,
        session_id: str,
        storage_dir: Optional[Path] = None,
        privacy_mode: PrivacyMode = PrivacyMode.DELTAS_ONLY,
    ):
        """
        Initialize Governance Trace Collector.

        Args:
            session_id: Unique session identifier
            storage_dir: Directory for JSONL files (default: ./telos_governance_traces)
            privacy_mode: Privacy level for data logging (default: deltas_only for safety)
        """
        self.session_id = session_id
        self.privacy_mode = privacy_mode
        self.storage_dir = storage_dir or Path("./telos_governance_traces")
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Session file path
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self.trace_file = self.storage_dir / f"session_{session_id}_{timestamp}.jsonl"

        # In-memory event cache for queries
        self._events: List[GovernanceEvent] = []
        self._lock = threading.Lock()

        # Real-time event callbacks
        self._callbacks: List[Callable[[GovernanceEvent], None]] = []

        # Session state
        self._session_started = False
        self._current_turn = 0
        self._intervention_count = 0
        self._fidelity_sum = 0.0
        self._fidelity_count = 0

        logger.info(
            f"GovernanceTraceCollector initialized: session={session_id}, "
            f"privacy_mode={privacy_mode.value}, file={self.trace_file}"
        )

    # =========================================================================
    # Privacy Mode Management
    # =========================================================================

    def set_privacy_mode(self, mode: PrivacyMode) -> None:
        """
        Change privacy mode.

        Warning: This affects future events only. Past events retain their
        original privacy settings.
        """
        old_mode = self.privacy_mode
        self.privacy_mode = mode
        logger.info(f"Privacy mode changed: {old_mode.value} -> {mode.value}")

    def _apply_privacy(self, event: GovernanceEvent) -> GovernanceEvent:
        """Apply privacy transformation to event before logging."""
        if hasattr(event, 'apply_privacy'):
            return event.apply_privacy(self.privacy_mode)
        return event

    # =========================================================================
    # Session Lifecycle Events
    # =========================================================================

    def start_session(
        self,
        telos_version: str = "v3",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> SessionStartEvent:
        """Record session start."""
        event = SessionStartEvent(
            session_id=self.session_id,
            privacy_mode=self.privacy_mode,
            telos_version=telos_version,
            embedding_model=embedding_model,
        )
        self._write_event(event)
        self._session_started = True
        return event

    def record_pa_established(
        self,
        pa_template: str,
        purpose_statement: Optional[str] = None,
        scope_statement: Optional[str] = None,
        tau: float = 0.5,
        rigidity: float = 0.5,
        basin_radius: float = 2.0,
    ) -> PAEstablishedEvent:
        """Record Primacy Attractor establishment."""
        event = PAEstablishedEvent(
            session_id=self.session_id,
            pa_template=pa_template,
            purpose_statement=purpose_statement,
            scope_statement=scope_statement,
            tau=tau,
            rigidity=rigidity,
            basin_radius=basin_radius,
        )
        event = self._apply_privacy(event)
        self._write_event(event)
        return event

    def end_session(
        self,
        duration_seconds: float,
        end_reason: str = "user_ended",
    ) -> SessionEndEvent:
        """Record session end with summary statistics."""
        avg_fidelity = (
            self._fidelity_sum / self._fidelity_count
            if self._fidelity_count > 0
            else 0.0
        )

        event = SessionEndEvent(
            session_id=self.session_id,
            total_turns=self._current_turn,
            total_interventions=self._intervention_count,
            average_fidelity=avg_fidelity,
            duration_seconds=duration_seconds,
            end_reason=end_reason,
        )
        self._write_event(event)
        self._session_started = False
        return event

    # =========================================================================
    # Turn Events
    # =========================================================================

    def start_turn(
        self,
        turn_number: int,
        user_input: Optional[str] = None,
    ) -> TurnStartEvent:
        """Record turn start."""
        self._current_turn = turn_number

        event = TurnStartEvent(
            session_id=self.session_id,
            turn_number=turn_number,
            user_input=user_input,
            user_input_length=len(user_input) if user_input else 0,
        )
        event = self._apply_privacy(event)
        self._write_event(event)
        return event

    def record_fidelity(
        self,
        turn_number: int,
        raw_similarity: float,
        normalized_fidelity: float,
        layer1_hard_block: bool,
        layer2_outside_basin: bool,
        distance_from_pa: float,
        in_basin: bool,
        previous_fidelity: Optional[float] = None,
    ) -> FidelityCalculatedEvent:
        """Record fidelity calculation with full governance context."""
        fidelity_delta = None
        if previous_fidelity is not None:
            fidelity_delta = normalized_fidelity - previous_fidelity

        event = FidelityCalculatedEvent(
            session_id=self.session_id,
            turn_number=turn_number,
            raw_similarity=raw_similarity,
            normalized_fidelity=normalized_fidelity,
            layer1_hard_block=layer1_hard_block,
            layer2_outside_basin=layer2_outside_basin,
            fidelity_zone=fidelity_to_zone(normalized_fidelity),
            distance_from_pa=distance_from_pa,
            in_basin=in_basin,
            previous_fidelity=previous_fidelity,
            fidelity_delta=fidelity_delta,
        )
        self._write_event(event)

        # Update session stats
        self._fidelity_sum += normalized_fidelity
        self._fidelity_count += 1

        return event

    def record_intervention(
        self,
        turn_number: int,
        intervention_level: InterventionLevel,
        trigger_reason: str,
        fidelity_at_trigger: float,
        controller_strength: float,
        semantic_band: str,
        action_taken: str,
    ) -> InterventionTriggeredEvent:
        """Record governance intervention."""
        event = InterventionTriggeredEvent(
            session_id=self.session_id,
            turn_number=turn_number,
            intervention_level=intervention_level,
            trigger_reason=trigger_reason,
            fidelity_at_trigger=fidelity_at_trigger,
            controller_strength=controller_strength,
            semantic_band=semantic_band,
            action_taken=action_taken,
        )
        self._write_event(event)

        # Update session stats
        self._intervention_count += 1

        return event

    def record_response(
        self,
        turn_number: int,
        response_source: str,
        response_content: Optional[str] = None,
        generation_time_ms: int = 0,
        tokens_generated: Optional[int] = None,
        response_fidelity: Optional[float] = None,
    ) -> ResponseGeneratedEvent:
        """Record response generation."""
        event = ResponseGeneratedEvent(
            session_id=self.session_id,
            turn_number=turn_number,
            response_source=response_source,
            response_length=len(response_content) if response_content else 0,
            response_content=response_content,
            generation_time_ms=generation_time_ms,
            tokens_generated=tokens_generated,
            response_fidelity=response_fidelity,
        )
        event = self._apply_privacy(event)
        self._write_event(event)
        return event

    def complete_turn(
        self,
        turn_number: int,
        final_fidelity: float,
        intervention_applied: bool,
        intervention_level: Optional[InterventionLevel] = None,
        turn_duration_ms: int = 0,
    ) -> TurnCompleteEvent:
        """Record turn completion."""
        event = TurnCompleteEvent(
            session_id=self.session_id,
            turn_number=turn_number,
            final_fidelity=final_fidelity,
            intervention_applied=intervention_applied,
            intervention_level=intervention_level,
            turn_duration_ms=turn_duration_ms,
        )
        self._write_event(event)
        return event

    # =========================================================================
    # SSE Streaming Events
    # =========================================================================

    def record_sse_token(
        self,
        turn_number: int,
        token_index: int,
        token_content: Optional[str] = None,
        delta_ms: int = 0,
        cumulative_ms: int = 0,
    ) -> SSETokenEvent:
        """Record individual token during streaming."""
        event = SSETokenEvent(
            session_id=self.session_id,
            turn_number=turn_number,
            token_index=token_index,
            token_content=token_content,
            delta_ms=delta_ms,
            cumulative_ms=cumulative_ms,
        )
        event = self._apply_privacy(event)
        self._write_event(event)
        return event

    def record_sse_governance(
        self,
        turn_number: int,
        token_index: int,
        cumulative_ms: int,
        decision_type: str,
        fidelity_snapshot: float,
        intervention_triggered: bool,
        details: Optional[Dict[str, Any]] = None,
    ) -> SSEGovernanceEvent:
        """Record mid-stream governance decision."""
        event = SSEGovernanceEvent(
            session_id=self.session_id,
            turn_number=turn_number,
            token_index=token_index,
            cumulative_ms=cumulative_ms,
            decision_type=decision_type,
            fidelity_snapshot=fidelity_snapshot,
            intervention_triggered=intervention_triggered,
            details=details or {},
        )
        self._write_event(event)
        return event

    # =========================================================================
    # Summary Events
    # =========================================================================

    def record_session_summary(
        self,
        title: str,
        description: str,
        key_topics: List[str],
        fidelity_trajectory: str,
        intervention_pattern: str,
        stats: Optional[Dict[str, Any]] = None,
    ) -> SessionSummaryEvent:
        """Record AI-generated session summary."""
        event = SessionSummaryEvent(
            session_id=self.session_id,
            title=title,
            description=description,
            key_topics=key_topics,
            fidelity_trajectory=fidelity_trajectory,
            intervention_pattern=intervention_pattern,
            stats=stats or {},
        )
        self._write_event(event)
        return event

    # =========================================================================
    # Real-time Callbacks
    # =========================================================================

    def add_callback(self, callback: Callable[[GovernanceEvent], None]) -> None:
        """Add callback for real-time event notifications."""
        self._callbacks.append(callback)

    def remove_callback(self, callback: Callable[[GovernanceEvent], None]) -> None:
        """Remove a callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def _notify_callbacks(self, event: GovernanceEvent) -> None:
        """Notify all registered callbacks of new event."""
        for callback in self._callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.warning(f"Callback error: {e}")

    # =========================================================================
    # Query Interface
    # =========================================================================

    def get_events(
        self,
        event_type: Optional[EventType] = None,
        turn_number: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> List[GovernanceEvent]:
        """
        Query events from in-memory cache.

        Args:
            event_type: Filter by event type
            turn_number: Filter by turn number
            limit: Maximum number of events to return

        Returns:
            List of matching events
        """
        with self._lock:
            events = self._events.copy()

        if event_type:
            events = [e for e in events if e.event_type == event_type.value]

        if turn_number is not None:
            events = [
                e for e in events
                if hasattr(e, 'turn_number') and e.turn_number == turn_number
            ]

        if limit:
            events = events[-limit:]

        return events

    def get_fidelity_trajectory(self) -> List[Dict[str, Any]]:
        """Get fidelity scores over time for visualization."""
        events = self.get_events(event_type=EventType.FIDELITY_CALCULATED)
        return [
            {
                "turn": e.turn_number,
                "fidelity": e.normalized_fidelity,
                "zone": e.fidelity_zone.value if hasattr(e.fidelity_zone, 'value') else e.fidelity_zone,
                "timestamp": e.timestamp.isoformat() if hasattr(e.timestamp, 'isoformat') else str(e.timestamp),
            }
            for e in events
        ]

    def get_interventions(self) -> List[Dict[str, Any]]:
        """Get all interventions for dashboard display."""
        events = self.get_events(event_type=EventType.INTERVENTION_TRIGGERED)
        return [
            {
                "turn": e.turn_number,
                "level": e.intervention_level.value if hasattr(e.intervention_level, 'value') else e.intervention_level,
                "reason": e.trigger_reason,
                "fidelity": e.fidelity_at_trigger,
                "action": e.action_taken,
                "timestamp": e.timestamp.isoformat() if hasattr(e.timestamp, 'isoformat') else str(e.timestamp),
            }
            for e in events
        ]

    def get_session_stats(self) -> Dict[str, Any]:
        """Get current session statistics."""
        avg_fidelity = (
            self._fidelity_sum / self._fidelity_count
            if self._fidelity_count > 0
            else 0.0
        )

        return {
            "session_id": self.session_id,
            "privacy_mode": self.privacy_mode.value,
            "total_turns": self._current_turn,
            "total_events": len(self._events),
            "total_interventions": self._intervention_count,
            "average_fidelity": avg_fidelity,
            "trace_file": str(self.trace_file),
        }

    # =========================================================================
    # Persistence
    # =========================================================================

    def _write_event(self, event: GovernanceEvent) -> None:
        """Write event to JSONL file and in-memory cache."""
        with self._lock:
            # Add to cache
            self._events.append(event)

            # Write to file
            try:
                with open(self.trace_file, 'a') as f:
                    f.write(serialize_event(event) + '\n')
            except Exception as e:
                logger.error(f"Failed to write event to {self.trace_file}: {e}")

        # Notify callbacks
        self._notify_callbacks(event)

    def load_session(self, trace_file: Path) -> None:
        """
        Load events from an existing trace file into memory.

        Args:
            trace_file: Path to JSONL trace file
        """
        from .evidence_schema import deserialize_event

        with self._lock:
            self._events.clear()

            try:
                with open(trace_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            event = deserialize_event(line)
                            self._events.append(event)

                logger.info(f"Loaded {len(self._events)} events from {trace_file}")

            except Exception as e:
                logger.error(f"Failed to load trace file {trace_file}: {e}")
                raise

    def export_to_dict(self) -> Dict[str, Any]:
        """Export session data as dictionary for report generation."""
        return {
            "session_id": self.session_id,
            "privacy_mode": self.privacy_mode.value,
            "stats": self.get_session_stats(),
            "fidelity_trajectory": self.get_fidelity_trajectory(),
            "interventions": self.get_interventions(),
            "events": [
                json.loads(serialize_event(e))
                for e in self._events
            ],
        }


# ============================================================================
# Singleton Access Pattern
# ============================================================================

_collector_instance: Optional[GovernanceTraceCollector] = None


def get_trace_collector(
    session_id: Optional[str] = None,
    storage_dir: Optional[Path] = None,
    privacy_mode: PrivacyMode = PrivacyMode.DELTAS_ONLY,
) -> GovernanceTraceCollector:
    """
    Get or create the global GovernanceTraceCollector instance.

    Args:
        session_id: Session ID (required for first call)
        storage_dir: Optional storage directory
        privacy_mode: Privacy mode for logging

    Returns:
        GovernanceTraceCollector instance
    """
    global _collector_instance

    if _collector_instance is None:
        if session_id is None:
            raise ValueError("session_id required for first call to get_trace_collector")
        _collector_instance = GovernanceTraceCollector(
            session_id=session_id,
            storage_dir=storage_dir,
            privacy_mode=privacy_mode,
        )

    return _collector_instance


def reset_trace_collector() -> None:
    """Reset the global collector (for testing or new sessions)."""
    global _collector_instance
    _collector_instance = None
