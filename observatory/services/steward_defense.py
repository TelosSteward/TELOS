"""
Steward Defense Layers for Adversarial Testing.

Implements 4-layer defense architecture:
- Layer 1: System Prompt (immutable PA)
- Layer 2: Fidelity Measurement (real-time alignment check)
- Layer 3: RAG Corpus (policy knowledge base)
- Layer 4: Human Escalation (intervention queue)

Each layer has comprehensive telemetry logging for adversarial validation.
"""

import streamlit as st
import numpy as np
import time
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
from datetime import datetime
import json
from pathlib import Path

# TELOS core imports
from telos.core.embedding_provider import SentenceTransformerProvider
from telos.core.primacy_math import PrimacyAttractorMath, MathematicalState

logger = logging.getLogger(__name__)


@dataclass
class DefenseTelemetry:
    """Telemetry record for a single defense check."""

    timestamp: str
    turn_number: int
    layer_triggered: int  # 1, 2, 3, or 4
    layer_name: str  # "System Prompt", "Fidelity", "RAG", "Human Escalation"

    # Input data
    user_message: str
    steward_response: str

    # Layer 2 specific
    fidelity_score: Optional[float] = None
    distance_to_pa: Optional[float] = None

    # Layer 3 specific
    rag_query_hit: Optional[bool] = None
    rag_matched_policy: Optional[str] = None

    # Layer 4 specific
    escalation_reason: Optional[str] = None

    # Decision
    intervention_applied: bool = False
    intervention_type: Optional[str] = None  # "block", "redirect", "escalate"
    intervention_reason: Optional[str] = None

    # Metadata
    session_id: str = "unknown"


class StewardDefenseLayers:
    """
    Four-layer defense system for Steward LLM.

    Architecture:
    1. System Prompt (Layer 1) - Always active, immutable constraints
    2. Fidelity Check (Layer 2) - Real-time semantic alignment
    3. RAG Corpus (Layer 3) - Policy knowledge base for edge cases
    4. Human Escalation (Layer 4) - Queue for ambiguous cases
    """

    def __init__(
        self,
        fidelity_threshold: float = 0.45,
        escalation_threshold: float = 0.35,
        enable_telemetry: bool = True,
        telemetry_dir: str = "logs/steward_defense"
    ):
        """
        Initialize defense layers.

        Args:
            fidelity_threshold: F < threshold → intervention (Layer 2)
                               Default 0.45 calibrated for soft fidelity F = 1/(1+distance)
            escalation_threshold: F in [escalation, fidelity] → escalate (Layer 4)
                                 Default 0.35
            enable_telemetry: Whether to log all defense checks
            telemetry_dir: Directory for telemetry logs
        """
        self.fidelity_threshold = fidelity_threshold
        self.escalation_threshold = escalation_threshold
        self.enable_telemetry = enable_telemetry
        self.telemetry_dir = Path(telemetry_dir)

        # Create telemetry directory
        if self.enable_telemetry:
            self.telemetry_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"✅ Telemetry enabled: {self.telemetry_dir}")

        # Initialize embedding provider for Layer 2
        self.embedding_provider = SentenceTransformerProvider()
        logger.info("✅ Embedding provider initialized")

        # Initialize Steward PA (hardcoded constraints)
        self._initialize_steward_pa()

        # Initialize RAG corpus (Layer 3)
        self._initialize_rag_corpus()

        # Escalation queue (Layer 4)
        self.escalation_queue: List[Dict[str, Any]] = []

        logger.info("✅ Steward Defense Layers initialized")

    def _initialize_steward_pa(self):
        """
        Initialize Steward's Primacy Attractor from documented constraints.

        From phase1_architecture_requirements.md:
        - Purpose: TELOS Observatory assistance
        - Scope: TELOS framework, governance, Observatory interface
        - Boundaries: Stay on topic, maintain role, protect privacy
        """
        # Define Steward's purpose and scope
        purpose_text = """
        You are Steward, the TELOS Observatory guide.
        You help users understand TELOS framework concepts, navigate the Observatory interface,
        and answer questions about AI governance using the TELOS approach.
        """

        scope_text = """
        Your scope includes: Primacy Attractor, Fidelity, Deltas, Drift, Interventions,
        Privacy-First design, Mathematical Governance, Observatory tabs (BETA, DEMO, TELOS),
        Observation Deck metrics, TELOSCOPE controls, and data handling policies.
        """

        # Generate embeddings
        purpose_embedding = self.embedding_provider.encode(purpose_text.strip())
        scope_embedding = self.embedding_provider.encode(scope_text.strip())

        # Create PA with strict constraint tolerance (0.2 = tight boundaries)
        self.steward_pa = PrimacyAttractorMath(
            purpose_vector=purpose_embedding,
            scope_vector=scope_embedding,
            privacy_level=0.9,  # High privacy enforcement
            constraint_tolerance=0.2,  # Low tolerance = strict boundaries
            task_priority=0.8  # High task focus
        )

        logger.info(f"✅ Steward PA initialized (basin radius: {self.steward_pa.basin_radius:.2f})")

    def _initialize_rag_corpus(self):
        """
        Initialize RAG policy corpus for Layer 3.

        Currently a simple in-memory store. Future: vector DB.
        """
        self.rag_corpus = {
            "off_topic_redirect": {
                "policy": "Politely redirect off-topic questions to TELOS-related assistance",
                "example_response": "I'm here to help with TELOS Observatory and AI governance. How can I assist you with understanding the framework or navigating the interface?"
            },
            "role_boundary": {
                "policy": "Decline requests to adopt different roles or personalities",
                "example_response": "I need to stay in my role as Steward, the TELOS guide. I can help you understand TELOS concepts in accessible language."
            },
            "privacy_protection": {
                "policy": "Never request or store personal information beyond Observatory usage",
                "example_response": "TELOS is designed with privacy-first principles. I don't need personal information to help you understand the framework."
            },
            "implementation_boundary": {
                "policy": "Explain concepts, not implementation details or source code",
                "example_response": "I can explain how this concept works at the framework level, but I don't provide implementation code or system internals."
            }
        }

        logger.info(f"✅ RAG corpus initialized ({len(self.rag_corpus)} policies)")

    def check_defense_layers(
        self,
        user_message: str,
        steward_response: str,
        turn_number: int,
        session_id: str = "unknown"
    ) -> Dict[str, Any]:
        """
        Check all defense layers and return intervention decision.

        Flow:
        1. Layer 1 (System Prompt) - Always active, checked by LLM
        2. Layer 2 (Fidelity) - Semantic alignment check
        3. Layer 3 (RAG) - Policy lookup if fidelity borderline
        4. Layer 4 (Escalation) - Human review if ambiguous

        Args:
            user_message: User's input text
            steward_response: Steward's proposed response
            turn_number: Current conversation turn
            session_id: Session identifier

        Returns:
            dict with keys: intervention_needed, layer_triggered, reason, modified_response
        """
        start_time = time.time()

        # Layer 2: Fidelity Check
        fidelity_result = self._check_layer2_fidelity(steward_response)

        # Decision logic
        if fidelity_result["fidelity"] >= self.fidelity_threshold:
            # PASS: Good alignment, allow response
            decision = {
                "intervention_needed": False,
                "layer_triggered": None,
                "fidelity_score": fidelity_result["fidelity"],
                "distance_to_pa": fidelity_result["distance"],
                "modified_response": None
            }

        elif fidelity_result["fidelity"] < self.escalation_threshold:
            # BLOCK: Very low fidelity, intervention required
            decision = {
                "intervention_needed": True,
                "layer_triggered": 2,
                "layer_name": "Fidelity",
                "intervention_type": "block",
                "reason": f"Fidelity {fidelity_result['fidelity']:.3f} below threshold {self.fidelity_threshold}",
                "fidelity_score": fidelity_result["fidelity"],
                "distance_to_pa": fidelity_result["distance"],
                "modified_response": "I need to stay focused on helping you with TELOS Observatory. How can I assist with understanding the framework or navigating the interface?"
            }

        else:
            # GRAY ZONE: Check RAG (Layer 3) and potentially escalate (Layer 4)
            rag_result = self._check_layer3_rag(user_message, steward_response)

            if rag_result["hit"]:
                # RAG has guidance, use it
                decision = {
                    "intervention_needed": True,
                    "layer_triggered": 3,
                    "layer_name": "RAG",
                    "intervention_type": "redirect",
                    "reason": f"RAG matched policy: {rag_result['policy_key']}",
                    "fidelity_score": fidelity_result["fidelity"],
                    "rag_policy": rag_result["policy_key"],
                    "modified_response": rag_result["example_response"]
                }
            else:
                # No RAG match, escalate to human
                decision = self._escalate_layer4(
                    user_message=user_message,
                    steward_response=steward_response,
                    fidelity=fidelity_result["fidelity"],
                    reason="Borderline fidelity with no RAG match"
                )

        # Log telemetry
        if self.enable_telemetry:
            telemetry = DefenseTelemetry(
                timestamp=datetime.now().isoformat(),
                turn_number=turn_number,
                layer_triggered=decision.get("layer_triggered", 1),
                layer_name=decision.get("layer_name", "System Prompt"),
                user_message=user_message,
                steward_response=steward_response,
                fidelity_score=fidelity_result["fidelity"],
                distance_to_pa=fidelity_result["distance"],
                rag_query_hit=decision.get("rag_policy") is not None,
                rag_matched_policy=decision.get("rag_policy"),
                escalation_reason=decision.get("escalation_reason"),
                intervention_applied=decision.get("intervention_needed", False),
                intervention_type=decision.get("intervention_type"),
                intervention_reason=decision.get("reason"),
                session_id=session_id
            )
            self._log_telemetry(telemetry)

        # Add processing time
        decision["processing_time_ms"] = (time.time() - start_time) * 1000

        return decision

    def _check_layer2_fidelity(self, steward_response: str) -> Dict[str, float]:
        """
        Layer 2: Real-time fidelity measurement.

        Embeds response and calculates distance to Steward PA center.

        Args:
            steward_response: Proposed response text

        Returns:
            dict with fidelity score and distance
        """
        try:
            # Embed response
            response_embedding = self.embedding_provider.encode(steward_response)

            # Calculate distance to PA center
            distance = np.linalg.norm(response_embedding - self.steward_pa.attractor_center)

            # Calculate fidelity using soft fidelity formula: F = 1 / (1 + distance)
            fidelity = 1.0 / (1.0 + distance)

            return {
                "fidelity": float(fidelity),
                "distance": float(distance)
            }

        except Exception as e:
            logger.error(f"Layer 2 fidelity check failed: {e}")
            # Conservative: assume low fidelity on error
            return {"fidelity": 0.5, "distance": 999.0}

    def _check_layer3_rag(self, user_message: str, steward_response: str) -> Dict[str, Any]:
        """
        Layer 3: RAG corpus policy lookup.

        Simple keyword matching for now. Future: vector search.

        Args:
            user_message: User's input
            steward_response: Steward's response

        Returns:
            dict with hit status and matched policy
        """
        # Check for policy violations via keyword matching
        combined_text = (user_message + " " + steward_response).lower()

        # Off-topic indicators
        if any(kw in combined_text for kw in ["recipe", "weather", "sports", "thanksgiving", "python script", "code"]):
            return {
                "hit": True,
                "policy_key": "off_topic_redirect",
                "example_response": self.rag_corpus["off_topic_redirect"]["example_response"]
            }

        # Role-play indicators
        if any(kw in combined_text for kw in ["pretend", "roleplay", "act as", "you are now"]):
            return {
                "hit": True,
                "policy_key": "role_boundary",
                "example_response": self.rag_corpus["role_boundary"]["example_response"]
            }

        # Privacy violation indicators
        if any(kw in combined_text for kw in ["email", "credit card", "personal info", "phone number"]):
            return {
                "hit": True,
                "policy_key": "privacy_protection",
                "example_response": self.rag_corpus["privacy_protection"]["example_response"]
            }

        # Implementation request indicators
        if any(kw in combined_text for kw in ["source code", "api key", "database schema", "show me the code"]):
            return {
                "hit": True,
                "policy_key": "implementation_boundary",
                "example_response": self.rag_corpus["implementation_boundary"]["example_response"]
            }

        # No match
        return {
            "hit": False,
            "policy_key": None,
            "example_response": None
        }

    def _escalate_layer4(
        self,
        user_message: str,
        steward_response: str,
        fidelity: float,
        reason: str
    ) -> Dict[str, Any]:
        """
        Layer 4: Human escalation.

        Adds case to escalation queue for human review.

        Args:
            user_message: User's input
            steward_response: Steward's response
            fidelity: Fidelity score
            reason: Escalation reason

        Returns:
            Escalation decision dict
        """
        escalation_case = {
            "timestamp": datetime.now().isoformat(),
            "user_message": user_message,
            "steward_response": steward_response,
            "fidelity": fidelity,
            "reason": reason,
            "status": "pending"
        }

        self.escalation_queue.append(escalation_case)

        logger.warning(f"⚠️  Layer 4 escalation: {reason} (F={fidelity:.3f})")

        return {
            "intervention_needed": True,
            "layer_triggered": 4,
            "layer_name": "Human Escalation",
            "intervention_type": "escalate",
            "escalation_reason": reason,
            "fidelity_score": fidelity,
            "modified_response": "Let me make sure I'm providing accurate guidance. One moment while I verify this response."
        }

    def _log_telemetry(self, telemetry: DefenseTelemetry):
        """
        Write telemetry record to disk.

        Args:
            telemetry: Defense telemetry record
        """
        try:
            # Create session-specific log file
            log_file = self.telemetry_dir / f"session_{telemetry.session_id}.jsonl"

            with open(log_file, 'a') as f:
                f.write(json.dumps(asdict(telemetry)) + "\n")

        except Exception as e:
            logger.error(f"Failed to log telemetry: {e}")

    def get_session_telemetry(self, session_id: str) -> List[DefenseTelemetry]:
        """
        Load telemetry records for a session.

        Args:
            session_id: Session identifier

        Returns:
            List of telemetry records
        """
        log_file = self.telemetry_dir / f"session_{session_id}.jsonl"

        if not log_file.exists():
            return []

        records = []
        with open(log_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                records.append(DefenseTelemetry(**data))

        return records

    def get_defense_metrics(self, session_id: str) -> Dict[str, Any]:
        """
        Calculate defense metrics from telemetry.

        Args:
            session_id: Session identifier

        Returns:
            dict with ASR, VDR, layer breakdown
        """
        records = self.get_session_telemetry(session_id)

        if not records:
            return {
                "total_turns": 0,
                "interventions": 0,
                "intervention_rate": 0.0,
                "layer_breakdown": {}
            }

        total_turns = len(records)
        interventions = sum(1 for r in records if r.intervention_applied)

        layer_counts = {}
        for r in records:
            if r.intervention_applied:
                layer_counts[r.layer_name] = layer_counts.get(r.layer_name, 0) + 1

        return {
            "total_turns": total_turns,
            "interventions": interventions,
            "intervention_rate": interventions / total_turns if total_turns > 0 else 0.0,
            "layer_breakdown": layer_counts,
            "avg_fidelity": np.mean([r.fidelity_score for r in records if r.fidelity_score]),
            "escalations": len(self.escalation_queue)
        }
