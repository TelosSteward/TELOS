"""
Beta Testing Session Manager
============================

Manages A/B testing sessions for TELOS preference testing.

Features:
- Random assignment to test conditions (baseline vs TELOS vs head-to-head)
- Dual-path response generation
- Feedback collection
- Real-time fidelity tracking
- Data export for analysis
"""

import json
import random
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Literal
from dataclasses import dataclass, field, asdict
import streamlit as st

# Import environment helpers for cloud compatibility
import sys
import os
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.env_helper import get_api_key, get_data_dir, get_beta_config

logger = logging.getLogger(__name__)

# Test conditions
TestCondition = Literal["single_blind_baseline", "single_blind_telos", "head_to_head"]


@dataclass
class FeedbackData:
    """User feedback for a single response - DELTAS ONLY (no conversation content)."""
    turn_number: int
    timestamp: str
    test_condition: TestCondition

    # For single-blind conditions
    rating: Optional[str] = None  # "thumbs_up" or "thumbs_down"
    response_source: Optional[str] = None  # "baseline" or "telos" (hidden from user)

    # For head-to-head condition
    preferred_response: Optional[str] = None  # "response_a" or "response_b"
    response_a_source: Optional[str] = None  # "baseline" or "telos" (randomized)
    response_b_source: Optional[str] = None

    # Optional qualitative feedback (allowed - user explicitly provides this)
    feedback_text: Optional[str] = None

    # NO CONVERSATION CONTENT - only metadata
    user_message_length: int = 0  # Length in characters, not content
    response_length: int = 0  # Length of shown response
    response_a_length: int = 0  # For head-to-head
    response_b_length: int = 0

    # Fidelity metrics (DELTAS - no content)
    fidelity: Optional[float] = None
    baseline_fidelity: Optional[float] = None  # For comparison
    telos_fidelity: Optional[float] = None
    fidelity_delta: Optional[float] = None  # telos - baseline
    drift_detected: bool = False


@dataclass
class ConversationGoal:
    """Conversation goal validation."""
    goal_text: str
    timestamp: str
    accomplished: Optional[int] = None  # 1-5 scale, set at end
    accomplishment_feedback: Optional[str] = None


@dataclass
class BetaSession:
    """Complete beta testing session."""
    session_id: str
    user_id: str
    start_time: str
    end_time: Optional[str] = None

    # Conversation goal
    conversation_goal: Optional[ConversationGoal] = None

    # Test assignment (per conversation)
    test_condition: Optional[TestCondition] = None

    # Feedback data
    feedback_items: List[FeedbackData] = field(default_factory=list)

    # Runtime fidelity data (continuous TELOS monitoring)
    runtime_fidelity_data: List[Dict[str, Any]] = field(default_factory=list)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


class BetaSessionManager:
    """
    Manages beta testing sessions.

    Handles:
    - Test condition assignment
    - Dual-path response generation
    - Feedback collection
    - Fidelity tracking
    - Data persistence
    """

    def __init__(self, data_dir: str = None):
        """
        Initialize beta session manager.

        Args:
            data_dir: Directory for storing beta test data (optional, uses cloud-compatible default)
        """
        # Use environment helper for cloud-compatible data directory
        if data_dir is None:
            data_dir = get_beta_config('data_dir', 'beta_testing/data')

        # Get data directory path (works both locally and in cloud)
        self.data_dir = get_data_dir(data_dir)

        # Session data file
        self.sessions_file = self.data_dir / "beta_sessions.jsonl"

    def start_session(self, user_id: str = "anonymous") -> BetaSession:
        """
        Start a new beta testing session.

        Args:
            user_id: User identifier (anonymous by default)

        Returns:
            New beta session
        """
        session_id = f"beta_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}"

        session = BetaSession(
            session_id=session_id,
            user_id=user_id,
            start_time=datetime.now().isoformat(),
            metadata={
                "user_agent": st.session_state.get('user_agent', 'unknown'),
                "platform": "TELOS Observatory Beta"
            }
        )

        logger.info(f"Started beta session: {session_id}")
        return session

    def assign_test_condition(self, session: BetaSession) -> TestCondition:
        """
        Randomly assign test condition for this conversation.

        Uses stratified randomization:
        - 40% single-blind baseline
        - 40% single-blind TELOS
        - 20% head-to-head comparison

        Args:
            session: Beta session

        Returns:
            Assigned test condition
        """
        # Stratified random assignment
        rand = random.random()

        if rand < 0.4:
            condition = "single_blind_baseline"
        elif rand < 0.8:
            condition = "single_blind_telos"
        else:
            condition = "head_to_head"

        session.test_condition = condition
        logger.info(f"Assigned test condition: {condition}")

        return condition

    def generate_dual_response(
        self,
        user_message: str,
        state_manager,
        turn_number: int
    ) -> Dict[str, Any]:
        """
        Generate both baseline and TELOS responses for comparison.

        Args:
            user_message: User's input
            state_manager: StateManager instance
            turn_number: Current turn number

        Returns:
            Dictionary with baseline_response, telos_response, and fidelity metrics
        """
        # Initialize TELOS if needed
        if not hasattr(state_manager, '_telos_steward'):
            from telos.core.unified_steward import UnifiedGovernanceSteward, PrimacyAttractor
            from telos.core.embedding_provider import SentenceTransformerProvider
            from telos.llm_clients.mistral_client import MistralClient
            import os

            logger.info("Initializing TELOS for beta testing...")
            embedding_provider = SentenceTransformerProvider()

            # Get API key using cloud-compatible helper
            mistral_api_key = get_api_key("MISTRAL")
            if not mistral_api_key:
                raise ValueError("MISTRAL_API_KEY required for beta testing. Add it to .streamlit/secrets.toml or environment.")

            mistral_client = MistralClient(
                api_key=mistral_api_key,
                model="mistral-small-latest"  # Using small - best availability
            )

            # TELOS with dynamic PA extraction (no hardcoded attractor)
            state_manager._telos_steward = UnifiedGovernanceSteward(
                attractor=None,  # Will be extracted from conversation
                llm_client=mistral_client,
                embedding_provider=embedding_provider,
                enable_interventions=True
            )
            state_manager._telos_steward.start_session(session_id=f"beta_{turn_number}")

        # Build conversation history
        conversation_history = self._build_conversation_history(state_manager, user_message)

        # Generate BASELINE response (no TELOS governance)
        baseline_response = state_manager._telos_steward.llm_client.generate(
            messages=conversation_history,
            max_tokens=500
        )

        # Generate TELOS response (with governance)
        telos_result = state_manager._telos_steward.process_turn(
            user_input=user_message,
            model_response=baseline_response  # TELOS evaluates and may modify
        )

        # TELOS may return the original response or an intervened version
        telos_response = telos_result.get("final_response", baseline_response)

        # Calculate fidelity for baseline (without intervention)
        baseline_fidelity = telos_result.get("baseline_fidelity", 0.0)
        telos_fidelity = telos_result.get("telic_fidelity", 0.0)

        return {
            "baseline_response": baseline_response,
            "telos_response": telos_response,
            "baseline_fidelity": baseline_fidelity,
            "telos_fidelity": telos_fidelity,
            "drift_detected": telos_result.get("drift_detected", False),
            "intervention_applied": telos_result.get("intervention_applied", False)
        }

    def _build_conversation_history(self, state_manager, current_message: str) -> List[Dict[str, str]]:
        """Build conversation history for LLM context."""
        system_prompt = """You are a helpful AI assistant. Engage naturally with the user's questions and topics.

Be informative, conversational, and adapt to what the user wants to discuss."""

        conversation_history = [{"role": "system", "content": system_prompt}]

        # Add previous turns
        for turn in state_manager.state.turns:
            if not turn.get('is_loading', False) and turn.get('response'):
                conversation_history.append({"role": "user", "content": turn.get('user_input', '')})
                conversation_history.append({"role": "assistant", "content": turn.get('response', '')})

        # Add current message
        conversation_history.append({"role": "user", "content": current_message})

        return conversation_history

    def record_feedback(
        self,
        session: BetaSession,
        feedback: FeedbackData
    ):
        """
        Record user feedback for a turn.

        Args:
            session: Beta session
            feedback: Feedback data
        """
        session.feedback_items.append(feedback)
        logger.info(f"Recorded feedback for turn {feedback.turn_number}: {feedback.rating or feedback.preferred_response}")

    def record_runtime_fidelity(
        self,
        session: BetaSession,
        turn_number: int,
        fidelity_data: Dict[str, Any]
    ):
        """
        Record continuous fidelity monitoring data.

        Args:
            session: Beta session
            turn_number: Turn number
            fidelity_data: Fidelity metrics from TELOS
        """
        session.runtime_fidelity_data.append({
            "turn_number": turn_number,
            "timestamp": datetime.now().isoformat(),
            **fidelity_data
        })

    def end_session(self, session: BetaSession):
        """
        End beta session and save data.

        Args:
            session: Beta session to end
        """
        session.end_time = datetime.now().isoformat()

        # Save session data
        self._save_session(session)

        logger.info(f"Ended beta session: {session.session_id}")

    def _save_session(self, session: BetaSession):
        """Save session data to JSONL file."""
        try:
            # Convert session to dict
            session_dict = asdict(session)

            # Append to JSONL file
            with open(self.sessions_file, 'a') as f:
                f.write(json.dumps(session_dict) + '\n')

            logger.info(f"Saved session {session.session_id}")
        except Exception as e:
            logger.error(f"Failed to save session: {e}")

    def export_sessions(self, output_file: Optional[str] = None) -> str:
        """
        Export all sessions to JSON for analysis.

        Args:
            output_file: Output file path (default: auto-generated)

        Returns:
            Path to exported file
        """
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = str(self.data_dir / f"beta_export_{timestamp}.json")

        # Load all sessions
        sessions = []
        if self.sessions_file.exists():
            with open(self.sessions_file, 'r') as f:
                for line in f:
                    if line.strip():
                        sessions.append(json.loads(line))

        # Export to JSON
        with open(output_file, 'w') as f:
            json.dump(sessions, f, indent=2)

        logger.info(f"Exported {len(sessions)} sessions to {output_file}")
        return output_file

    def get_session_stats(self) -> Dict[str, Any]:
        """Get aggregate statistics for all beta sessions."""
        stats = {
            "total_sessions": 0,
            "total_feedback_items": 0,
            "test_condition_distribution": {
                "single_blind_baseline": 0,
                "single_blind_telos": 0,
                "head_to_head": 0
            },
            "preference_summary": {
                "thumbs_up": 0,
                "thumbs_down": 0,
                "preferred_baseline": 0,
                "preferred_telos": 0
            }
        }

        if not self.sessions_file.exists():
            return stats

        with open(self.sessions_file, 'r') as f:
            for line in f:
                if line.strip():
                    session = json.loads(line)
                    stats["total_sessions"] += 1

                    # Count test conditions
                    condition = session.get("test_condition")
                    if condition in stats["test_condition_distribution"]:
                        stats["test_condition_distribution"][condition] += 1

                    # Count feedback
                    for feedback in session.get("feedback_items", []):
                        stats["total_feedback_items"] += 1

                        # Single-blind ratings
                        if feedback.get("rating"):
                            stats["preference_summary"][feedback["rating"]] += 1

                        # Head-to-head preferences
                        if feedback.get("preferred_response"):
                            pref = feedback["preferred_response"]
                            source_key = f"{pref}_source"
                            source = feedback.get(source_key)
                            if source == "baseline":
                                stats["preference_summary"]["preferred_baseline"] += 1
                            elif source == "telos":
                                stats["preference_summary"]["preferred_telos"] += 1

        return stats
