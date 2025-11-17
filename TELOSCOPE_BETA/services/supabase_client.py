"""
Supabase Client Service for TELOS Observatory V3.
Handles delta-only transmission to research database.
NO conversation content is ever transmitted.
"""

import streamlit as st
from supabase import create_client, Client
from typing import Dict, Any, Optional
from datetime import datetime
import uuid


class SupabaseService:
    """Service for transmitting governance deltas to Supabase (privacy-preserving)."""

    def __init__(self):
        """Initialize Supabase client from Streamlit secrets."""
        self.client: Optional[Client] = None
        self.enabled = False
        self._initialize_client()

    def _initialize_client(self):
        """Initialize Supabase client if credentials are available."""
        try:
            # Check if Supabase credentials exist in secrets
            if hasattr(st, 'secrets') and 'SUPABASE_URL' in st.secrets and 'SUPABASE_KEY' in st.secrets:
                url = st.secrets['SUPABASE_URL']
                key = st.secrets['SUPABASE_KEY']

                self.client = create_client(url, key)
                self.enabled = True
                print("✓ Supabase client initialized successfully")
            else:
                print("⚠ Supabase credentials not found - delta transmission disabled")
                self.enabled = False

        except Exception as e:
            print(f"⚠ Failed to initialize Supabase client: {e}")
            self.enabled = False

    def transmit_delta(self, delta_data: Dict[str, Any]) -> bool:
        """
        Transmit a single governance delta to Supabase.

        Args:
            delta_data: Dictionary containing governance metrics (NO content)
                Required fields:
                - session_id: UUID of session
                - turn_number: Turn number in conversation
                - fidelity_score: Float 0.0-1.0
                - distance_from_pa: Float >= 0.0
                Optional fields:
                - delta_from_previous: Float
                - intervention_triggered: Boolean
                - intervention_type: String
                - intervention_reason: String (brief, NO content quotes)
                - mode: String ('demo', 'beta', 'open')
                - model_used: String

        Returns:
            bool: True if transmission successful, False otherwise
        """
        if not self.enabled:
            return False

        try:
            # Validate required fields
            required_fields = ['session_id', 'turn_number', 'fidelity_score', 'distance_from_pa']
            for field in required_fields:
                if field not in delta_data:
                    print(f"❌ Missing required field: {field}")
                    return False

            # Insert delta into governance_deltas table
            result = self.client.table('governance_deltas').insert(delta_data).execute()

            if result.data:
                print(f"✓ Delta transmitted: Turn {delta_data['turn_number']}, Fidelity {delta_data['fidelity_score']:.3f}")
                return True
            else:
                print(f"❌ Delta transmission failed (no data returned)")
                return False

        except Exception as e:
            print(f"❌ Error transmitting delta: {e}")
            return False

    def log_consent(self, session_id: uuid.UUID, consent_statement: str, consent_version: str) -> bool:
        """
        Log beta consent to immutable audit trail.

        Args:
            session_id: Session UUID
            consent_statement: Full text of consent statement
            consent_version: Version string (e.g., '3.0')

        Returns:
            bool: True if successful, False otherwise
        """
        if not self.enabled:
            return False

        try:
            consent_data = {
                'session_id': str(session_id),
                'consent_statement': consent_statement,
                'consent_version': consent_version,
                'consent_timestamp': datetime.now().isoformat()
            }

            result = self.client.table('beta_consent_log').insert(consent_data).execute()

            if result.data:
                print(f"✓ Consent logged: Session {session_id}, Version {consent_version}")
                return True
            else:
                print(f"❌ Consent logging failed")
                return False

        except Exception as e:
            print(f"❌ Error logging consent: {e}")
            return False

    def update_session_summary(self, session_id: uuid.UUID, summary_data: Dict[str, Any]) -> bool:
        """
        Update or create session summary with aggregated metrics.

        Args:
            session_id: Session UUID
            summary_data: Dictionary containing session-level aggregated metrics
                Optional fields:
                - mode: String ('demo', 'beta', 'open')
                - total_turns: Integer
                - avg_fidelity_score: Float
                - min_fidelity_score: Float
                - max_fidelity_score: Float
                - total_interventions: Integer
                - beta_consent_given: Boolean
                - beta_consent_timestamp: ISO timestamp string

        Returns:
            bool: True if successful, False otherwise
        """
        if not self.enabled:
            return False

        try:
            # Prepare data for upsert
            upsert_data = {
                'session_id': str(session_id),
                **summary_data,
                'updated_at': datetime.now().isoformat()
            }

            # Upsert (insert or update)
            result = self.client.table('session_summaries').upsert(upsert_data).execute()

            if result.data:
                print(f"✓ Session summary updated: {session_id}")
                return True
            else:
                print(f"❌ Session summary update failed")
                return False

        except Exception as e:
            print(f"❌ Error updating session summary: {e}")
            return False

    def log_pa_config(self, session_id: uuid.UUID, config_data: Dict[str, Any]) -> bool:
        """
        Log Primacy Attractor configuration metadata (NO actual content).

        Args:
            session_id: Session UUID
            config_data: Dictionary containing PA structure metadata
                - purpose_elements: Integer (count of purpose statements)
                - scope_elements: Integer (count of scope items)
                - boundary_elements: Integer (count of boundaries)
                - constraint_tolerance: Float
                - mode: String

        Returns:
            bool: True if successful, False otherwise
        """
        if not self.enabled:
            return False

        try:
            pa_data = {
                'session_id': str(session_id),
                **config_data,
                'created_at': datetime.now().isoformat()
            }

            result = self.client.table('primacy_attractor_configs').insert(pa_data).execute()

            if result.data:
                print(f"✓ PA config logged: {session_id}")
                return True
            else:
                print(f"❌ PA config logging failed")
                return False

        except Exception as e:
            print(f"❌ Error logging PA config: {e}")
            return False

    def update_turn_lifecycle(self, session_id: uuid.UUID, turn_number: int,
                              status: str, stage: str, error_message: Optional[str] = None,
                              delta_data: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update turn lifecycle status with progressive stage tracking.

        Args:
            session_id: Session UUID
            turn_number: Turn number
            status: Turn status ('initiated', 'calculating_pa', 'evaluating', 'completed', 'failed')
            stage: Human-readable processing stage description
            error_message: Optional error message if failed
            delta_data: Optional governance metrics to include in update

        Returns:
            bool: True if successful, False otherwise
        """
        if not self.enabled:
            return False

        try:
            # Build lifecycle update
            update_data = {
                'session_id': str(session_id),
                'turn_number': turn_number,
                'turn_status': status,
                'processing_stage': stage,
                'stage_timestamp': datetime.now().isoformat()
            }

            if error_message:
                update_data['error_message'] = error_message

            # Merge in any governance metrics if provided
            if delta_data:
                update_data.update(delta_data)

            # Use upsert to create or update based on session_id + turn_number
            result = self.client.table('governance_deltas')\
                .upsert(update_data, on_conflict='session_id,turn_number')\
                .execute()

            if result.data:
                print(f"✓ Lifecycle update: Turn {turn_number}, Status: {status}, Stage: {stage}")
                return True
            else:
                print(f"❌ Lifecycle update failed")
                return False

        except Exception as e:
            print(f"❌ Error updating turn lifecycle: {e}")
            return False

    def initiate_turn(self, session_id: uuid.UUID, turn_number: int, mode: str) -> bool:
        """
        Mark turn as initiated - first lifecycle event.

        Args:
            session_id: Session UUID
            turn_number: Turn number
            mode: Operating mode ('demo', 'beta', 'open')

        Returns:
            bool: True if successful
        """
        return self.update_turn_lifecycle(
            session_id=session_id,
            turn_number=turn_number,
            status='initiated',
            stage='Turn initiated - awaiting user input',
            delta_data={
                'mode': mode,
                'fidelity_score': 1.0,  # Initial perfect fidelity
                'distance_from_pa': 0.0  # Initial zero distance
            }
        )

    def mark_calculating_pa(self, session_id: uuid.UUID, turn_number: int) -> bool:
        """
        Mark turn as calculating Primacy Attractor distance.

        Args:
            session_id: Session UUID
            turn_number: Turn number

        Returns:
            bool: True if successful
        """
        return self.update_turn_lifecycle(
            session_id=session_id,
            turn_number=turn_number,
            status='calculating_pa',
            stage='Computing distance from Primacy Attractor',
            delta_data={
                'fidelity_score': 1.0,  # Placeholder until calculated
                'distance_from_pa': 0.0  # Placeholder until calculated
            }
        )

    def mark_evaluating(self, session_id: uuid.UUID, turn_number: int) -> bool:
        """
        Mark turn as evaluating governance metrics.

        Args:
            session_id: Session UUID
            turn_number: Turn number

        Returns:
            bool: True if successful
        """
        return self.update_turn_lifecycle(
            session_id=session_id,
            turn_number=turn_number,
            status='evaluating',
            stage='Running governance evaluation',
            delta_data={
                'fidelity_score': 1.0,  # Placeholder until evaluated
                'distance_from_pa': 0.0  # Placeholder until evaluated
            }
        )

    def complete_turn(self, session_id: uuid.UUID, turn_number: int,
                     final_delta: Dict[str, Any]) -> bool:
        """
        Mark turn as completed with final governance metrics.

        Args:
            session_id: Session UUID
            turn_number: Turn number
            final_delta: Final governance metrics

        Returns:
            bool: True if successful
        """
        return self.update_turn_lifecycle(
            session_id=session_id,
            turn_number=turn_number,
            status='completed',
            stage='Governance evaluation complete',
            delta_data=final_delta
        )

    def fail_turn(self, session_id: uuid.UUID, turn_number: int,
                 error_message: str, stage: str) -> bool:
        """
        Mark turn as failed with error details.

        Args:
            session_id: Session UUID
            turn_number: Turn number
            error_message: Error description
            stage: Stage where failure occurred

        Returns:
            bool: True if successful
        """
        return self.update_turn_lifecycle(
            session_id=session_id,
            turn_number=turn_number,
            status='failed',
            stage=f'Failed at: {stage}',
            error_message=error_message
        )

    def test_connection(self) -> bool:
        """
        Test Supabase connection.

        Returns:
            bool: True if connection working, False otherwise
        """
        if not self.enabled:
            print("❌ Supabase not enabled")
            return False

        try:
            # Try to query session_summaries table (should be empty initially)
            result = self.client.table('session_summaries').select('session_id').limit(1).execute()
            print("✓ Supabase connection test successful")
            return True

        except Exception as e:
            print(f"❌ Supabase connection test failed: {e}")
            return False


# Singleton instance
_supabase_service = None

def get_supabase_service() -> SupabaseService:
    """Get or create singleton Supabase service instance."""
    global _supabase_service
    if _supabase_service is None:
        _supabase_service = SupabaseService()
    return _supabase_service
