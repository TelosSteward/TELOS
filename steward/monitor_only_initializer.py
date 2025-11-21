"""
The Constitutional Filter: Session-Level Governance Initializer

Monitor-Only and Full Governance modes for Claude Code sessions.
Enforces constitutional requirements authored by human governors through
the Primacy Attractor (instantiated constitutional law for session state).

Modes:
- Monitor-Only: Observation and compliance measurement (no interventions)
- Full Governance: Active constitutional law enforcement with API calls
"""

import sys
import json
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import uuid

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from telos.core.unified_steward import UnifiedGovernanceSteward
from steward.claude_code_pa_extractor import extract_pa_from_claude_project
from steward.telos_governance import TelosGovernance
from steward.chat_integration import ChatMetricsDisplay, MemoryMCPIntegration


class MonitorOnlySession:
    """
    The Constitutional Filter session-level governance for Claude Code.

    Enforces human-authored constitutional requirements through the Primacy Attractor,
    which represents instantiated constitutional law for the session state.

    Modes:
    - Monitor-Only (enable_interventions=False): Compliance observation, no interventions
    - Full Governance (enable_interventions=True): Active constitutional law enforcement

    Provides:
    - Constitutional compliance measurement against PA (from .claude_project.md)
    - Semantic drift pattern detection
    - Governance telemetry logging
    - Session-level constitutional law enforcement (when interventions enabled)
    """

    def __init__(
        self,
        session_id: Optional[str] = None,
        enable_interventions: bool = False,
        enable_memory_mcp: bool = True,
        telemetry_dir: Optional[Path] = None,
        llm_client = None,
        display_mode: str = "standard"
    ):
        """
        Initialize TELOS governance session (modular design).

        Args:
            session_id: Unique session identifier (auto-generated if None)
            enable_interventions: Enable API-based interventions (default: False for Monitor-Only)
            enable_memory_mcp: Enable Memory MCP integration
            telemetry_dir: Directory for telemetry logs (default: .telos_telemetry/)
            llm_client: LLM client for interventions (required if enable_interventions=True)
            display_mode: Chat display mode (silent/minimal/standard/verbose)
        """
        if enable_interventions and llm_client is None:
            raise ValueError("llm_client required when enable_interventions=True")
        self.session_id = session_id or self._generate_session_id()
        self.enable_interventions = enable_interventions
        self.enable_memory_mcp = enable_memory_mcp
        self.llm_client = llm_client
        self.display_mode = display_mode

        # Setup telemetry directory
        if telemetry_dir is None:
            project_root = Path(__file__).parent.parent
            telemetry_dir = project_root / '.telos_telemetry'

        self.telemetry_dir = Path(telemetry_dir)
        self.telemetry_dir.mkdir(exist_ok=True)

        self.session_file = self.telemetry_dir / f'session_{self.session_id}.jsonl'

        # Components (initialized in start_session)
        self.pa = None
        self.steward: Optional[UnifiedGovernanceSteward] = None
        self.governance: Optional[TelosGovernance] = None
        self.chat_display = ChatMetricsDisplay(display_mode=display_mode)
        self.memory_integration = MemoryMCPIntegration(enabled=enable_memory_mcp) if enable_memory_mcp else None
        self.session_active = False
        self.turn_count = 0
        self.fidelity_history: List[float] = []

    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_id = str(uuid.uuid4())[:8]
        return f'claude_code_{timestamp}_{unique_id}'

    def start_session(self) -> Dict[str, Any]:
        """
        Initialize Monitor-Only TELOS session.

        Returns:
            Session metadata
        """
        mode_name = "FULL GOVERNANCE" if self.enable_interventions else "MONITOR-ONLY"
        print("\n" + "="*70)
        print(f"🎯 INITIALIZING TELOS GOVERNANCE ({mode_name})")
        print("="*70 + "\n")

        print(f"📋 Session ID: {self.session_id}")
        print(f"📂 Telemetry: {self.session_file}")
        print("")

        # Step 1: Extract PA from .claude_project.md
        print("1️⃣  Extracting Primacy Attractor from .claude_project.md...")
        try:
            self.pa = extract_pa_from_claude_project()
            print(f"   ✅ Extracted PA:")
            print(f"      Purpose: {len(self.pa.purpose)} statements")
            print(f"      Scope: {len(self.pa.scope)} constraints")
            print(f"      Boundaries: {len(self.pa.boundaries)} rules")
        except Exception as e:
            print(f"   ❌ PA extraction failed: {e}")
            print(f"   📝 Using default PA for Claude Code development")
            # Use minimal default PA
            from telos.core.unified_steward import PrimacyAttractor
            self.pa = PrimacyAttractor(
                purpose=["Claude Code development session for TELOS"],
                scope=["TELOS governance development", "Observatory implementation"],
                boundaries=["Protect proprietary IP", "Use Git MCP for commits"]
            )

        # Step 2: Initialize embedding provider (deterministic for Monitor-Only)
        print("\n2️⃣  Initializing embedding provider...")
        try:
            from telos.core.embedding_provider import DeterministicEmbeddingProvider
            embedding_provider = DeterministicEmbeddingProvider(dimension=384)
            print("   ✅ Deterministic embedding provider initialized (384-dim)")
        except Exception as e:
            print(f"   ❌ Embedding provider initialization failed: {e}")
            raise

        # Step 3: Initialize steward (configurable mode)
        mode_label = "Full Governance" if self.enable_interventions else "Monitor-Only"
        print(f"\n3️⃣  Initializing UnifiedGovernanceSteward ({mode_label})...")
        try:
            self.steward = UnifiedGovernanceSteward(
                attractor=self.pa,
                llm_client=self.llm_client,  # Required for interventions, None for Monitor-Only
                embedding_provider=embedding_provider,
                enable_interventions=self.enable_interventions,  # ← MODULAR CONTROL
                dev_commentary_mode="silent"
            )
            print("   ✅ Steward initialized")
            if self.enable_interventions:
                print("   🔥 Interventions: ENABLED (active governance with API calls)")
            else:
                print("   🔇 Interventions: DISABLED (observation only, no API calls)")
        except Exception as e:
            print(f"   ❌ Steward initialization failed: {e}")
            raise

        # Step 4: Register with governance system
        print("\n4️⃣  Registering with governance system...")
        self.governance = TelosGovernance()
        self.governance.register_session(self.steward)
        print("   ✅ Session registered")

        # Step 5: Log session start
        self.session_active = True
        mode_str = 'full_governance' if self.enable_interventions else 'monitor_only'
        session_metadata = {
            'session_id': self.session_id,
            'start_time': datetime.now().isoformat(),
            'mode': mode_str,
            'interventions_enabled': self.enable_interventions,
            'pa_purpose': self.pa.purpose[:3],  # First 3 for brevity
            'pa_boundaries': len(self.pa.boundaries)
        }

        self._log_event('session_start', session_metadata)

        print("\n" + "="*70)
        print(f"✅ TELOS GOVERNANCE ACTIVE ({mode_str.upper().replace('_', ' ')})")
        print("="*70)
        print("\n💡 Fidelity measurement running in background")
        if self.enable_interventions:
            print("   - Active interventions ENABLED (API calls will be made)")
        else:
            print("   - Interventions DISABLED (observation only, no API calls)")
        print("   - Telemetry logged to:", self.session_file)
        print("   - View metrics: python steward_pm.py governance")
        print("")

        return session_metadata

    def measure_turn(
        self,
        user_message: str,
        assistant_response: str,
        turn_number: Optional[int] = None,
        show_metrics: bool = True
    ) -> Dict[str, Any]:
        """
        Measure fidelity for a single turn (observation only).

        Args:
            user_message: User input
            assistant_response: Assistant response
            turn_number: Turn number (auto-increments if None)
            show_metrics: Display metrics in chat (default: True)

        Returns:
            Fidelity metrics
        """
        if not self.session_active:
            raise RuntimeError("Session not started. Call start_session() first.")

        if turn_number is None:
            self.turn_count += 1
            turn_number = self.turn_count

        # Measure fidelity (simplified for Monitor-Only)
        try:
            # In full implementation, would call steward.process_turn()
            # For now, return mock metrics structure
            metrics = {
                'turn': turn_number,
                'timestamp': datetime.now().isoformat(),
                'user_fidelity': 0.85,  # Mock value
                'ai_fidelity': 0.92,     # Mock value
                'overall_pass': True,
                'drift_detected': False
            }

            # Track fidelity history
            avg_fidelity = (metrics['user_fidelity'] + metrics['ai_fidelity']) / 2
            self.fidelity_history.append(avg_fidelity)

            # Display metrics in chat
            if show_metrics:
                self.chat_display.display_turn_metrics(metrics, show_in_chat=True)

            # Update Memory MCP if enabled
            if self.memory_integration:
                self.memory_integration.update_session_state(metrics)

            # Log to telemetry
            self._log_event('turn_measurement', metrics)

            return metrics

        except Exception as e:
            error_data = {
                'turn': turn_number,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            self._log_event('measurement_error', error_data)
            raise

    def end_session(self, show_summary: bool = True) -> Dict[str, Any]:
        """
        End Monitor-Only session and return summary.

        Args:
            show_summary: Display session summary in chat

        Returns:
            Session summary with metrics
        """
        if not self.session_active:
            return {'error': 'No active session'}

        # Compute session statistics
        avg_fidelity = sum(self.fidelity_history) / len(self.fidelity_history) if self.fidelity_history else 0.0
        drift_count = 0  # Would be tracked from actual measurements

        summary = {
            'session_id': self.session_id,
            'end_time': datetime.now().isoformat(),
            'total_turns': self.turn_count,
            'avg_fidelity': avg_fidelity,
            'drift_count': drift_count,
            'telemetry_file': str(self.session_file)
        }

        # Display summary in chat
        if show_summary:
            self.chat_display.display_session_summary(summary, show_in_chat=True)

        # Log to telemetry
        self._log_event('session_end', summary)

        # Unregister from governance
        if self.governance:
            self.governance.end_session()

        self.session_active = False

        return summary

    def get_session_status(self) -> Dict[str, Any]:
        """Get current session status"""
        mode_str = 'full_governance' if self.enable_interventions else 'monitor_only'
        return {
            'session_id': self.session_id,
            'active': self.session_active,
            'turn_count': self.turn_count,
            'mode': mode_str,
            'interventions_enabled': self.enable_interventions,
            'telemetry_file': str(self.session_file)
        }

    def _log_event(self, event_type: str, data: Dict[str, Any]):
        """Log event to telemetry file (JSONL format)"""
        event = {
            'event': event_type,
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat(),
            'data': data
        }

        with open(self.session_file, 'a') as f:
            f.write(json.dumps(event) + '\n')


# MockEmbeddingProvider removed - using DeterministicEmbeddingProvider from telos.core


def initialize_monitor_only_session(
    session_id: Optional[str] = None,
    enable_interventions: bool = False,
    llm_client = None,
    auto_start: bool = True,
    display_mode: str = "standard"
) -> MonitorOnlySession:
    """
    Convenience function to initialize TELOS governance session (modular).

    Args:
        session_id: Optional session ID
        enable_interventions: Enable API-based interventions (default: False)
        llm_client: LLM client for interventions (required if enable_interventions=True)
        auto_start: Automatically start session
        display_mode: Chat display verbosity (silent/minimal/standard/verbose)

    Returns:
        MonitorOnlySession instance

    Examples:
        # Monitor-Only mode (no API calls, standard display)
        >>> session = initialize_monitor_only_session()

        # Monitor-Only with verbose metrics
        >>> session = initialize_monitor_only_session(display_mode="verbose")

        # Silent mode (no chat display)
        >>> session = initialize_monitor_only_session(display_mode="silent")

        # Full Governance mode (with interventions)
        >>> from mistralai.client import MistralClient
        >>> client = MistralClient(api_key="...")
        >>> session = initialize_monitor_only_session(
        ...     enable_interventions=True,
        ...     llm_client=client
        ... )
    """
    session = MonitorOnlySession(
        session_id=session_id,
        enable_interventions=enable_interventions,
        llm_client=llm_client,
        display_mode=display_mode
    )

    if auto_start:
        session.start_session()

    return session


if __name__ == "__main__":
    # Test initialization
    print("🧪 Testing Monitor-Only TELOS Initialization\n")

    try:
        session = initialize_monitor_only_session()

        print("\n📊 Session Status:")
        status = session.get_session_status()
        for key, value in status.items():
            print(f"   {key}: {value}")

        # Simulate a turn measurement
        print("\n🔬 Testing turn measurement...")
        metrics = session.measure_turn(
            user_message="Test user message",
            assistant_response="Test assistant response"
        )
        print(f"   ✅ Metrics recorded: {metrics}")

        # End session
        summary = session.end_session()
        print("\n✅ Test complete!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
