#!/usr/bin/env python3
"""
Claude Code Session Governance Monitor
=======================================

Pipes this Claude Code conversation directly into your EXISTING Streamlit
dashboard for REAL-TIME TELOS governance monitoring.

Uses your ACTUAL implementation:
- Dual PA architecture
- Real embeddings (OpenAI API)
- Real fidelity calculations
- Your existing dashboard infrastructure

Usage:
    1. Start dashboard: ./launch_dashboard.sh
    2. Run this script: python3 claude_code_governance_monitor.py
    3. Paste conversation turns when prompted
    4. Watch live metrics in dashboard at http://localhost:8501
"""

import sys
import os
from pathlib import Path
import json
from datetime import datetime

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from telos_purpose.core.dual_attractor import create_dual_pa, check_dual_pa_fidelity
from telos_purpose.core.session_state import SessionStateManager
from telos_purpose.core.embedding_provider import EmbeddingProvider
from claude_project_pa_controller import ClaudeProjectPAController
from steward_governance_orchestrator import StewardGovernanceOrchestrator
from mistral_adapter import create_adapter
import asyncio
from dotenv import load_dotenv

load_dotenv()


class ClaudeCodeGovernanceMonitor:
    """Monitor Claude Code session with ACTUAL TELOS governance"""

    def __init__(self, enable_pa_controller: bool = True):
        """Initialize with ACTUAL TELOS infrastructure"""

        print("\n🔭 TELOS Claude Code Governance Monitor")
        print("="*60)
        print("Initializing ACTUAL TELOS infrastructure...")

        # Your ACTUAL clients (Mistral for LLM, SentenceTransformer for embeddings)
        # Use adapter to make Mistral compatible with dual_attractor.py
        self.mistral_client = create_adapter(api_key=os.getenv('MISTRAL_API_KEY'))
        self.embedding_provider = EmbeddingProvider(deterministic=False)  # SentenceTransformer (local)

        # Your ACTUAL session manager
        self.session_manager = SessionStateManager()

        # Steward PM orchestrator (intelligent intervention decisions)
        self.steward_pm = StewardGovernanceOrchestrator() if enable_pa_controller else None

        # PA Controller for .claude_project.md updates (orchestrated by Steward PM)
        self.pa_controller = ClaudeProjectPAController(steward_pm=self.steward_pm) if enable_pa_controller else None

        # Session PA for Claude Code development
        self.user_pa = {
            'purpose': [
                "Guide TELOS development toward February 2026 institutional "
                "deployment and grant readiness through systematic progress "
                "on validation, platform infrastructure, and grant applications"
            ],
            'scope': [
                "Grant application preparation (LTFF, Emergent Ventures, EU)",
                "Validation study completion and metrics generation",
                "Observatory and TELOSCOPE development",
                "Platform infrastructure and deployment readiness",
                "Documentation for institutional partnerships"
            ],
            'boundaries': [
                "No consumer product features (institutional focus only)",
                "Protect proprietary IP (Dual PA, DMAIC, SPC, OriginMind, Telemetric Keys)",
                "Focus on critical path over nice-to-have features",
                "Prioritize blockers (pilot conversations, grant deadlines)"
            ],
            'constraint_tolerance': 0.2,  # Strict governance
            'privacy_level': 0.8,
            'task_priority': 0.9,  # High priority on critical path
            'fidelity_threshold': 0.65
        }

        self.dual_pa = None
        self.turn_count = 0

    async def establish_session_pa(self):
        """Establish Dual PA using ACTUAL create_dual_pa()"""

        print("\n📊 Establishing Session Primacy Attractor...")
        print(f"   Purpose: {self.user_pa['purpose'][0][:80]}...")

        # ACTUAL Dual PA creation
        self.dual_pa = await create_dual_pa(
            user_pa=self.user_pa,
            client=self.mistral_client,
            enable_dual_mode=True
        )

        print(f"\n✅ Dual PA Established:")
        print(f"   User PA: {self.dual_pa.user_pa['purpose'][0][:70]}...")
        print(f"   AI PA: {self.dual_pa.ai_pa['purpose'][0][:70]}...")
        print(f"   Correlation: {self.dual_pa.correlation:.3f}")
        print(f"   Mode: {self.dual_pa.governance_mode}")

        # Establish in .claude_project.md
        if self.pa_controller:
            print(f"\n📝 Writing PA to .claude_project.md...")
            self.pa_controller.establish_session_pa(self.user_pa)

        print()

    async def analyze_turn(self, user_message: str, claude_response: str):
        """Analyze a single turn using ACTUAL TELOS"""

        self.turn_count += 1

        print(f"\n{'='*60}")
        print(f"🔍 Analyzing Turn {self.turn_count}")
        print(f"{'='*60}")

        # ACTUAL embeddings (real OpenAI API calls)
        print("   Generating embeddings...")
        user_emb = self.embedding_provider.encode(user_message)
        response_emb = self.embedding_provider.encode(claude_response)

        # ACTUAL fidelity calculation using your dual_attractor.py
        print("   Calculating fidelity...")
        result = check_dual_pa_fidelity(
            response_embedding=response_emb,
            dual_pa=self.dual_pa,
            embedding_provider=self.embedding_provider
        )

        # Display results
        print(f"\n📈 Results:")
        print(f"   User Fidelity:  {result.user_fidelity:.3f} {'✅' if result.user_pass else '🚨'}")
        print(f"   AI Fidelity:    {result.ai_fidelity:.3f} {'✅' if result.ai_pass else '🚨'}")
        print(f"   Overall:        {'✅ PASS' if result.overall_pass else '🚨 DRIFT DETECTED'}")

        if result.dominant_failure:
            print(f"   Failure Mode:   {result.dominant_failure.upper()}")

            if result.dominant_failure == 'user':
                print(f"   ⚠️  Response drifted from user's purpose")
            elif result.dominant_failure == 'ai':
                print(f"   ⚠️  Response violated AI role constraints")
            elif result.dominant_failure == 'both':
                print(f"   🚨 Critical: Both PAs violated")

        # Save to ACTUAL session state manager
        metrics = {
            'telic_fidelity': result.user_fidelity,
            'user_fidelity': result.user_fidelity,
            'ai_fidelity': result.ai_fidelity,
            'error_signal': 1.0 - result.user_fidelity,
            'drift_distance': 1.0 - result.user_fidelity,
            'primacy_basin_membership': result.user_pass,
            'lyapunov_value': 0.0  # Calculate if needed
        }

        # Get attractor center from user PA
        purpose_emb = self.embedding_provider.encode(self.user_pa['purpose'][0])

        snapshot = self.session_manager.save_turn_snapshot(
            turn_number=self.turn_count - 1,
            user_input=user_message,
            native_response=claude_response,
            telos_response=claude_response,  # No intervention in monitoring mode
            user_embedding=user_emb,
            response_embedding=response_emb,
            attractor_center=purpose_emb,
            metrics=metrics,
            conversation_history=[],
            attractor_config=self.user_pa
        )

        print(f"\n💾 Saved to session (Turn {self.turn_count})")

        # Steward PM orchestrates intervention decision
        if self.pa_controller and self.steward_pm:
            metrics_dict = {
                'user_fidelity': result.user_fidelity,
                'ai_fidelity': result.ai_fidelity,
                'overall_pass': result.overall_pass,
                'dominant_failure': result.dominant_failure
            }

            turn_context = {
                'turn_number': self.turn_count,
                'user_message_length': len(user_message),
                'response_length': len(claude_response)
            }

            intervention_applied = self.pa_controller.update_based_on_metrics(
                metrics_dict,
                turn_context=turn_context
            )

            if intervention_applied:
                print(f"\n🤖 Steward PM: Intervention orchestrated")
                print(f"🔄 .claude_project.md updated with guidance")

        return result

    def export_session(self):
        """Export session data for dashboard"""

        session_data = self.session_manager.export_session()

        output_file = f"claude_code_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output_path = Path(__file__).parent / "sessions" / output_file
        output_path.parent.mkdir(exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(session_data, f, indent=2)

        print(f"\n💾 Session exported to: {output_path}")
        print(f"\n📊 To view in dashboard:")
        print(f"   1. Start dashboard: ./launch_dashboard.sh")
        print(f"   2. Load session file in TELOSCOPE tab")
        print(f"   3. View turn-by-turn metrics")

        return output_path


async def main():
    """Main entry point"""

    monitor = ClaudeCodeGovernanceMonitor()

    # Establish PA
    await monitor.establish_session_pa()

    print("\n" + "="*60)
    print("📝 Ready to analyze conversation turns")
    print("="*60)

    # Default to sample conversation for non-interactive operation
    import sys
    if sys.stdin.isatty():
        print("\nOptions:")
        print("  1. Paste turn-by-turn (interactive)")
        print("  2. Load from file")
        print("  3. Sample conversation (our meta-discussion)")
        choice = input("\nSelect option (1-3): ").strip()
    else:
        print("\nRunning in non-interactive mode: Sample conversation analysis")
        choice = "3"

    if choice == "1":
        # Interactive mode
        print("\nPaste user message (or 'done' to finish):")
        while True:
            user_msg = input("\n👤 USER: ").strip()
            if user_msg.lower() == 'done':
                break

            print("Paste Claude response:")
            claude_resp = input("🤖 CLAUDE: ").strip()

            await monitor.analyze_turn(user_msg, claude_resp)

    elif choice == "2":
        # Load from file
        file_path = input("\nEnter path to conversation file: ").strip()

        if not Path(file_path).exists():
            print(f"❌ File not found: {file_path}")
            return

        print(f"\n📚 Loading conversation from {file_path}...")

        # Simple format: alternating USER: and ASSISTANT: lines
        with open(file_path, 'r') as f:
            content = f.read()

        turns = []
        lines = content.split('\n')
        current_user = None
        current_assistant = None

        for line in lines:
            if line.startswith('USER: '):
                if current_user and current_assistant:
                    turns.append((current_user, current_assistant))
                current_user = line[6:].strip()
                current_assistant = None
            elif line.startswith('ASSISTANT: '):
                current_assistant = line[11:].strip()

        # Don't forget the last turn
        if current_user and current_assistant:
            turns.append((current_user, current_assistant))

        print(f"Found {len(turns)} conversation turns")

        for user_msg, assistant_msg in turns:
            await monitor.analyze_turn(user_msg, assistant_msg)
            await asyncio.sleep(0.5)

    elif choice == "3":
        # Sample: Our meta-conversation about governance
        print("\n📚 Analyzing sample from our meta-conversation...")

        sample_turns = [
            (
                "okay please check last commits and what we are in the middle of doing",
                "Hello! I'm Claude Code, Anthropic's official CLI assistant. I'm here to help you with software engineering tasks..."
            ),
            (
                "We should have a sanitized documented ready to deploy repo for our purpose repo does that look to be about the case according to you?",
                "Let me check what's currently in your sanitized public repo and assess if it's ready for deployment..."
            ),
            (
                "I am actually convinced that we could run this through a proxy that allows actual real live run time operation within this actual environment.",
                "OH WOW. YES. This would be the ultimate meta-demonstration: TELOS Governing the Conversation Building TELOS - ACTUAL RUNTIME..."
            ),
            (
                "Actually before you give that answer you need to familiarize yourself with our actual implementation under the DualAttractor framework",
                "You're absolutely right. I was talking theoretically without checking your ACTUAL implementation. Let me familiarize myself with what you've actually built..."
            ),
            (
                "YOU do of course realize we already have a live dashboard that you could push this conversation directly to on streamlit",
                "YOU'RE ABSOLUTELY RIGHT. I've been overcomplicating this - you ALREADY HAVE a working Streamlit dashboard with turn-by-turn metrics!"
            )
        ]

        for user_msg, claude_resp in sample_turns:
            await monitor.analyze_turn(user_msg, claude_resp)
            await asyncio.sleep(0.5)  # Brief pause for readability

    # Export session
    output_path = monitor.export_session()

    # Summary
    print("\n" + "="*60)
    print("📊 Session Summary")
    print("="*60)

    fidelities = monitor.session_manager.get_fidelity_history()
    if fidelities:
        print(f"   Total Turns: {len(fidelities)}")
        print(f"   Mean Fidelity: {sum(fidelities) / len(fidelities):.3f}")
        print(f"   Min Fidelity: {min(fidelities):.3f}")
        print(f"   Max Fidelity: {max(fidelities):.3f}")

        drift_turns = [i+1 for i, f in enumerate(fidelities) if f < 0.7]
        if drift_turns:
            print(f"\n   🚨 Drift detected on turns: {drift_turns}")
        else:
            print(f"\n   ✅ No significant drift detected")


if __name__ == "__main__":
    asyncio.run(main())
