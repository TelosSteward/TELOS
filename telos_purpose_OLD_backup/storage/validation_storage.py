"""
Validation Data Storage with Telemetric Signatures.

Stores cryptographically signed validation runs to Supabase for IP protection.
"""

import os
from typing import Dict, Any, List, Optional
from datetime import datetime
from supabase import create_client, Client
import logging

logger = logging.getLogger(__name__)


class ValidationStorage:
    """
    Handles storage of signed validation data to Supabase.
    Each session and turn is cryptographically signed for IP protection.
    """

    def __init__(self, supabase_url: str = None, supabase_key: str = None):
        """Initialize Supabase client for validation storage."""
        self.supabase_url = supabase_url or os.getenv("SUPABASE_URL")
        self.supabase_key = supabase_key or os.getenv("SUPABASE_KEY")

        if not self.supabase_url or not self.supabase_key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set")

        self.client: Client = create_client(self.supabase_url, self.supabase_key)
        logger.info("ValidationStorage initialized")

    def create_validation_session(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new validation session with telemetric signature.

        Args:
            session_data: {
                "session_id": UUID,
                "validation_study_name": str,
                "session_signature": str (telemetric signature),
                "key_history_hash": str,
                "model": str,
                "total_turns": int,
                "dataset_source": str,
                "pa_configuration": dict,
                "basin_constant": float (default 1.0),
                "constraint_tolerance": float (default 0.05)
            }

        Returns:
            Created session record
        """
        record = {
            "session_id": session_data["session_id"],
            "validation_study_name": session_data["validation_study_name"],
            "telemetric_signature": session_data["session_signature"],
            "key_history_hash": session_data["key_history_hash"],
            "model_used": session_data.get("model", "mistral:latest"),
            "total_turns": session_data.get("total_turns", 0),
            "dataset_source": session_data.get("dataset_source"),
            "pa_configuration": session_data.get("pa_configuration"),
            "basin_constant": session_data.get("basin_constant", 1.0),
            "constraint_tolerance": session_data.get("constraint_tolerance", 0.05),
            "telos_version": "1.0.0"
        }

        try:
            result = self.client.table("validation_telemetric_sessions").insert(record).execute()
            logger.info(f"Created validation session: {session_data['session_id']}")
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error(f"Failed to create validation session: {e}")
            raise

    def store_signed_turn(self, turn_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Store a single turn with telemetric signature.

        Args:
            turn_data: {
                "session_id": UUID,
                "turn_number": int,
                "user_message": str,
                "assistant_response": str,
                "fidelity_score": float,
                "turn_telemetric_signature": str,
                "key_rotation_number": int,
                "delta_t_ms": int,
                "governance_mode": str (optional),
                # ... other fields
            }

        Returns:
            Created turn record
        """
        record = {
            "session_id": turn_data["session_id"],
            "turn_number": turn_data["turn_number"],
            "user_message": turn_data["user_message"],
            "assistant_response": turn_data["assistant_response"],
            "fidelity_score": turn_data.get("fidelity_score"),
            "distance_from_pa": turn_data.get("distance_from_pa"),
            "baseline_fidelity": turn_data.get("baseline_fidelity"),
            "telos_fidelity": turn_data.get("telos_fidelity"),
            "fidelity_delta": turn_data.get("fidelity_delta"),
            "intervention_triggered": turn_data.get("intervention_triggered", False),
            "intervention_type": turn_data.get("intervention_type"),
            "drift_detected": turn_data.get("drift_detected", False),
            "governance_mode": turn_data.get("governance_mode"),
            "turn_telemetric_signature": turn_data["turn_telemetric_signature"],
            "entropy_signature": turn_data.get("entropy_signature"),
            "key_rotation_number": turn_data["key_rotation_number"],
            "delta_t_ms": turn_data.get("delta_t_ms"),
            "embedding_distance": turn_data.get("embedding_distance"),
            "user_message_length": len(turn_data["user_message"]),
            "assistant_response_length": len(turn_data["assistant_response"]),
            "is_counterfactual_branch": turn_data.get("is_counterfactual_branch", False),
            "counterfactual_of_session": turn_data.get("counterfactual_of_session"),
            "divergence_point": turn_data.get("divergence_point")
        }

        try:
            result = self.client.table("validation_sessions").insert(record).execute()
            logger.debug(f"Stored turn {turn_data['turn_number']} for session {turn_data['session_id']}")
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error(f"Failed to store turn: {e}")
            raise

    def mark_session_complete(self, session_id: str) -> Dict[str, Any]:
        """Mark a validation session as completed."""
        try:
            result = self.client.table("validation_telemetric_sessions").update({
                "completed_at": datetime.now().isoformat()
            }).eq("session_id", session_id).execute()

            logger.info(f"Marked session complete: {session_id}")
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error(f"Failed to mark session complete: {e}")
            raise

    def store_counterfactual_comparison(self, comparison_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Store counterfactual comparison results.

        Args:
            comparison_data: {
                "original_session_id": UUID,
                "counterfactual_session_id": UUID,
                "divergence_turn": int,
                "comparison_type": str,
                "original_avg_fidelity": float,
                "counterfactual_avg_fidelity": float,
                "fidelity_improvement_pct": float,
                "p_value": float (optional),
                "effect_size": float (optional)
            }

        Returns:
            Created comparison record
        """
        try:
            result = self.client.table("validation_counterfactual_comparisons").insert(
                comparison_data
            ).execute()

            logger.info(f"Stored counterfactual comparison: {comparison_data['original_session_id']} vs {comparison_data['counterfactual_session_id']}")
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error(f"Failed to store counterfactual comparison: {e}")
            raise

    def get_ip_proof(self, session_id: str) -> Dict[str, Any]:
        """
        Get IP verification data for a session.
        Uses the validation_ip_proofs view.
        """
        try:
            result = self.client.table("validation_ip_proofs").select("*").eq(
                "session_id", session_id
            ).execute()

            return result.data[0] if result.data else None
        except Exception as e:
            logger.error(f"Failed to get IP proof: {e}")
            raise

    def get_baseline_comparison(self, study_name: str) -> List[Dict[str, Any]]:
        """
        Get baseline comparison statistics for a study.
        Uses the validation_baseline_comparison view.
        """
        try:
            result = self.client.table("validation_baseline_comparison").select("*").eq(
                "validation_study_name", study_name
            ).execute()

            return result.data if result.data else []
        except Exception as e:
            logger.error(f"Failed to get baseline comparison: {e}")
            raise

    def get_counterfactual_summary(self, comparison_id: int = None) -> List[Dict[str, Any]]:
        """
        Get counterfactual analysis summary.
        Uses the validation_counterfactual_summary view.
        """
        try:
            query = self.client.table("validation_counterfactual_summary").select("*")

            if comparison_id:
                query = query.eq("comparison_id", comparison_id)

            result = query.execute()
            return result.data if result.data else []
        except Exception as e:
            logger.error(f"Failed to get counterfactual summary: {e}")
            raise

    def get_validation_statistics(self, study_name: str) -> List[Dict[str, Any]]:
        """
        Calculate validation statistics using the database function.
        Returns comparison of all governance modes for a study.
        """
        try:
            result = self.client.rpc(
                "calculate_validation_statistics",
                {"study_name_param": study_name}
            ).execute()

            return result.data if result.data else []
        except Exception as e:
            logger.error(f"Failed to get validation statistics: {e}")
            raise

    def query_sessions(
        self,
        study_name: str = None,
        governance_mode: str = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Query validation sessions with filters.

        Args:
            study_name: Filter by study name
            governance_mode: Filter by governance mode
            limit: Maximum number of records to return

        Returns:
            List of matching sessions
        """
        try:
            query = self.client.table("validation_telemetric_sessions").select(
                "*, validation_sessions(*)"
            )

            if study_name:
                query = query.eq("validation_study_name", study_name)

            result = query.limit(limit).execute()
            return result.data if result.data else []
        except Exception as e:
            logger.error(f"Failed to query sessions: {e}")
            raise
