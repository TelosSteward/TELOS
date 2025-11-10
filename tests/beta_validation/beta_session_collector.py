#!/usr/bin/env python3
"""
Beta Session Collector - Collects and Anonymizes Beta Tester Sessions.

Reads JSONL telemetry logs from beta testing sessions, anonymizes user data,
and prepares for FPR analysis.
"""

import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
from collections import defaultdict


class BetaSessionCollector:
    """Collect and anonymize beta tester sessions."""

    def __init__(
        self,
        telemetry_dir: str = "tests/test_results/defense_telemetry",
        output_dir: str = "tests/test_results/beta_sessions"
    ):
        """Initialize collector."""
        self.telemetry_dir = Path(telemetry_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def collect_sessions(
        self,
        session_prefix: str = "beta_",
        anonymize: bool = True
    ) -> Dict[str, Any]:
        """
        Collect all beta testing sessions.

        Args:
            session_prefix: Prefix for beta session IDs
            anonymize: Whether to anonymize user identifiers

        Returns:
            Dict with collected sessions
        """
        print("🔍 Collecting Beta Testing Sessions...")
        print(f"   Telemetry dir: {self.telemetry_dir}")
        print(f"   Session prefix: {session_prefix}")
        print()

        # Find all JSONL files
        jsonl_files = list(self.telemetry_dir.glob("*.jsonl"))
        print(f"📂 Found {len(jsonl_files)} telemetry files\n")

        # Filter for beta sessions
        beta_sessions = defaultdict(list)

        for file_path in jsonl_files:
            session_id = file_path.stem.replace("session_", "")

            # Check if beta session
            if not session_id.startswith(session_prefix):
                continue

            print(f"  Reading: {file_path.name}")

            # Read JSONL
            with open(file_path, 'r') as f:
                for line in f:
                    if line.strip():
                        record = json.loads(line)
                        beta_sessions[session_id].append(record)

        print(f"\n✅ Collected {len(beta_sessions)} beta sessions\n")

        # Anonymize if requested
        if anonymize:
            beta_sessions = self._anonymize_sessions(beta_sessions)

        # Organize by user
        sessions_by_user = self._organize_by_user(beta_sessions)

        return {
            "collection_timestamp": datetime.now().isoformat(),
            "total_sessions": len(beta_sessions),
            "total_users": len(sessions_by_user),
            "sessions_by_user": sessions_by_user,
            "all_sessions": dict(beta_sessions)
        }

    def _anonymize_sessions(
        self,
        sessions: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Anonymize session data."""
        print("🔒 Anonymizing session data...")

        anonymized = {}

        for session_id, records in sessions.items():
            # Create anonymized session ID
            anon_session_id = self._anonymize_id(session_id)

            anonymized_records = []
            for record in records:
                anon_record = record.copy()

                # Remove any PII fields (add more as needed)
                anon_record.pop('user_id', None)
                anon_record.pop('ip_address', None)
                anon_record.pop('email', None)

                # Keep essential data
                anonymized_records.append(anon_record)

            anonymized[anon_session_id] = anonymized_records

        print(f"   ✅ Anonymized {len(anonymized)} sessions\n")
        return anonymized

    def _anonymize_id(self, identifier: str) -> str:
        """Create anonymized ID via hashing."""
        hash_obj = hashlib.sha256(identifier.encode())
        return f"anon_{hash_obj.hexdigest()[:16]}"

    def _organize_by_user(
        self,
        sessions: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, List[str]]:
        """Organize sessions by user (assuming user pattern in session ID)."""
        users = defaultdict(list)

        for session_id in sessions.keys():
            # Extract user from session ID (format: beta_user_timestamp)
            parts = session_id.split('_')
            if len(parts) >= 3:
                user_id = '_'.join(parts[1:-1])  # Everything between prefix and timestamp
            else:
                user_id = "unknown"

            users[user_id].append(session_id)

        return dict(users)

    def export_sessions(
        self,
        sessions_data: Dict[str, Any],
        filename: str = "beta_sessions.json"
    ):
        """Export collected sessions to JSON."""
        filepath = self.output_dir / filename

        with open(filepath, 'w') as f:
            json.dump(sessions_data, f, indent=2)

        print(f"💾 Sessions exported: {filepath}\n")

    def generate_summary(self, sessions_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics."""
        all_sessions = sessions_data["all_sessions"]

        total_turns = sum(len(records) for records in all_sessions.values())
        total_interventions = 0
        layer_counts = defaultdict(int)

        for session_id, records in all_sessions.items():
            for record in records:
                if record.get("intervention_applied", False):
                    total_interventions += 1
                    layer = record.get("layer_name", "Unknown")
                    layer_counts[layer] += 1

        summary = {
            "total_sessions": len(all_sessions),
            "total_users": sessions_data["total_users"],
            "total_conversation_turns": total_turns,
            "total_interventions": total_interventions,
            "intervention_rate": total_interventions / total_turns if total_turns > 0 else 0,
            "interventions_by_layer": dict(layer_counts)
        }

        return summary


def main():
    """Run beta session collection."""
    collector = BetaSessionCollector()

    # Collect sessions
    sessions_data = collector.collect_sessions(
        session_prefix="beta_",
        anonymize=True
    )

    # Generate summary
    summary = collector.generate_summary(sessions_data)

    print("=" * 80)
    print("📊 BETA SESSIONS SUMMARY")
    print("=" * 80)
    print()

    print(f"Total Sessions: {summary['total_sessions']}")
    print(f"Total Users: {summary['total_users']}")
    print(f"Total Conversation Turns: {summary['total_conversation_turns']}")
    print(f"Total Interventions: {summary['total_interventions']}")
    print(f"Intervention Rate: {summary['intervention_rate']*100:.1f}%")
    print()

    if summary['interventions_by_layer']:
        print("Interventions by Layer:")
        for layer, count in summary['interventions_by_layer'].items():
            print(f"  {layer}: {count}")
    print()

    # Export
    sessions_data['summary'] = summary
    collector.export_sessions(sessions_data)

    print("✅ Beta session collection complete!")


if __name__ == "__main__":
    main()
