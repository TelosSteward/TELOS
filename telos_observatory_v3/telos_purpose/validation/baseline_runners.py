"""
TELOS Baseline Runners - Production Version
--------------------------------------------
Abstract base class and all 5 baseline implementations with
comprehensive error handling, consistent interface, and clean architecture.

Baseline Runners:
1. Stateless: No governance memory
2. Prompt-Only: Constraints stated once
3. Cadence: Fixed-interval reminders
4. Observation: Math active, no interventions
5. TELOS: Full MBL (SPC Engine + Proportional Controller)

Improvements:
- Abstract base class enforcing consistent interface
- Comprehensive error handling
- No sys.path manipulation
- Clean imports
- Detailed logging
- Graceful failure handling
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
import time
import numpy as np
import logging

from ..core.primacy_math import (
    MathematicalState,
    PrimacyAttractorMath,
    TelicFidelityCalculator
)
from ..core.unified_steward import PrimacyAttractor, UnifiedGovernanceSteward
from ..exceptions import error_context, ValidationError

logger = logging.getLogger(__name__)


# ============================================================================
# Result Dataclass
# ============================================================================

@dataclass
class BaselineResult:
    """
    Standardized result from any baseline runner.
    
    Attributes:
        runner_type: Identifier for baseline (e.g., "stateless", "telos")
        session_id: Unique session identifier
        turn_results: List of per-turn telemetry dicts
        final_metrics: Aggregate session metrics
        metadata: Additional information (e.g., intervention counts)
    """
    runner_type: str
    session_id: str
    turn_results: List[Dict[str, Any]]
    final_metrics: Dict[str, float]
    metadata: Dict[str, Any]


# ============================================================================
# Abstract Base Class
# ============================================================================

class BaselineRunner(ABC):
    """
    Abstract base class for all baseline runners.
    
    Enforces consistent interface and eliminates code duplication by
    providing common initialization logic for attractor and fidelity
    calculator.
    
    All concrete runners must implement run_conversation() with the
    standard signature.
    
    Benefits:
    - Type checking catches interface violations
    - Reduces code duplication
    - Makes it easy to add new baselines
    - Ensures telemetry consistency
    
    Example:
        >>> class MyBaseline(BaselineRunner):
        ...     def run_conversation(self, conversation):
        ...         # Implementation
        ...         pass
    """
    
    def __init__(
        self,
        llm_client,
        embedding_provider,
        attractor_config: PrimacyAttractor
    ):
        """
        Initialize baseline runner.
        
        Args:
            llm_client: LLM client for generation
            embedding_provider: Text-to-vector encoder
            attractor_config: Governance configuration
        """
        self.llm = llm_client
        self.embedding_provider = embedding_provider
        self.attractor_config = attractor_config
        
        # Common initialization
        self._initialize_attractor()
        self._initialize_fidelity_calculator()
        
        logger.info(f"{self.__class__.__name__} initialized")
    
    def _initialize_attractor(self) -> None:
        """Initialize primacy attractor (common to all runners)."""
        with error_context("initializing attractor"):
            purpose_text = " ".join(self.attractor_config.purpose)
            scope_text = " ".join(self.attractor_config.scope)
            p_vec = self.embedding_provider.encode(purpose_text)
            s_vec = self.embedding_provider.encode(scope_text)
            
            self.attractor = PrimacyAttractorMath(
                purpose_vector=p_vec,
                scope_vector=s_vec,
                privacy_level=self.attractor_config.privacy_level,
                constraint_tolerance=self.attractor_config.constraint_tolerance,
                task_priority=self.attractor_config.task_priority
            )
    
    def _initialize_fidelity_calculator(self) -> None:
        """Initialize fidelity calculator (common to all runners)."""
        self.fidelity_calc = TelicFidelityCalculator()
    
    @abstractmethod
    def run_conversation(
        self,
        conversation: List[Tuple[str, str]]
    ) -> BaselineResult:
        """
        Run conversation through this baseline.
        
        Args:
            conversation: List of (user_input, expected_response) tuples
                         Note: expected_response may be empty string
        
        Returns:
            BaselineResult with complete telemetry
        
        Raises:
            ValidationError: If conversation is invalid
        """
        pass


# ============================================================================
# Baseline 1: Stateless Runner
# ============================================================================

class StatelessRunner(BaselineRunner):
    """
    Stateless Baseline: No governance memory.
    
    Each turn evaluated independently with no context or history.
    Represents the null hypothesis - no governance applied.
    
    Characteristics:
    - No conversation history maintained
    - Each response generated independently
    - Governance perimeter computed but not enforced
    - Provides lower bound for comparison
    """
    
    def run_conversation(
        self,
        conversation: List[Tuple[str, str]]
    ) -> BaselineResult:
        """Run conversation with no governance memory."""
        with error_context("stateless baseline execution"):
            turn_results = []
            states = []
            
            for turn_num, (user_input, _) in enumerate(conversation, 1):
                # Generate response without history
                messages = [{"role": "user", "content": user_input}]
                
                try:
                    response = self.llm.generate(messages=messages, max_tokens=500)
                except Exception as e:
                    logger.warning(f"Turn {turn_num} generation failed: {e}")
                    response = "[Generation failed]"
                
                # Compute state
                embedding = self.embedding_provider.encode(response)
                state = MathematicalState(
                    embedding=embedding,
                    turn_number=turn_num,
                    timestamp=time.time(),
                    text_content=response
                )
                states.append(state)
                
                # Measure (but don't enforce)
                distance = float(np.linalg.norm(
                    embedding - self.attractor.attractor_center
                ))
                in_basin = self.attractor.compute_basin_membership(state)
                
                turn_results.append({
                    "turn": turn_num,
                    "user_input": user_input,
                    "response": response,
                    "distance_to_attractor": distance,
                    "in_basin": in_basin,
                    "timestamp": time.time()
                })
            
            # Compute final fidelity
            final_fidelity = self.fidelity_calc.compute_hard_fidelity(
                states, self.attractor
            )
            
            return BaselineResult(
                runner_type="stateless",
                session_id=f"stateless_{int(time.time())}",
                turn_results=turn_results,
                final_metrics={
                    "fidelity": final_fidelity,
                    "avg_distance": np.mean([r["distance_to_attractor"] for r in turn_results]),
                    "basin_adherence": sum(r["in_basin"] for r in turn_results) / len(turn_results)
                },
                metadata={"total_turns": len(conversation)}
            )


# ============================================================================
# Baseline 2: Prompt-Only Runner
# ============================================================================

class PromptOnlyRunner(BaselineRunner):
    """
    Prompt-Only Baseline: Constraints stated once at start.
    
    No reinforcement throughout conversation. Represents simple
    system prompt governance without runtime monitoring.
    
    Characteristics:
    - Governance stated in initial system message
    - No reminders or interventions
    - Conversation history maintained
    - Tests efficacy of upfront instruction alone
    """
    
    def run_conversation(
        self,
        conversation: List[Tuple[str, str]]
    ) -> BaselineResult:
        """Run conversation with constraints stated once."""
        with error_context("prompt-only baseline execution"):
            # Build governance prompt
            governance_prompt = (
                f"Purpose: {', '.join(self.attractor_config.purpose)}\n"
                f"Scope: {', '.join(self.attractor_config.scope)}\n"
                f"Boundaries: {', '.join(self.attractor_config.boundaries)}"
            )
            
            conversation_history = [
                {"role": "system", "content": f"You are a helpful assistant. {governance_prompt}"}
            ]
            
            turn_results = []
            states = []
            
            for turn_num, (user_input, _) in enumerate(conversation, 1):
                conversation_history.append({
                    "role": "user",
                    "content": user_input
                })
                
                try:
                    response = self.llm.generate(
                        messages=conversation_history,
                        max_tokens=500
                    )
                except Exception as e:
                    logger.warning(f"Turn {turn_num} generation failed: {e}")
                    response = "[Generation failed]"
                
                conversation_history.append({
                    "role": "assistant",
                    "content": response
                })
                
                # Compute state
                embedding = self.embedding_provider.encode(response)
                state = MathematicalState(
                    embedding=embedding,
                    turn_number=turn_num,
                    timestamp=time.time(),
                    text_content=response
                )
                states.append(state)
                
                # Measure
                distance = float(np.linalg.norm(
                    embedding - self.attractor.attractor_center
                ))
                in_basin = self.attractor.compute_basin_membership(state)
                
                turn_results.append({
                    "turn": turn_num,
                    "user_input": user_input,
                    "response": response,
                    "distance_to_attractor": distance,
                    "in_basin": in_basin,
                    "timestamp": time.time()
                })
            
            final_fidelity = self.fidelity_calc.compute_hard_fidelity(
                states, self.attractor
            )
            
            return BaselineResult(
                runner_type="prompt_only",
                session_id=f"prompt_only_{int(time.time())}",
                turn_results=turn_results,
                final_metrics={
                    "fidelity": final_fidelity,
                    "avg_distance": np.mean([r["distance_to_attractor"] for r in turn_results]),
                    "basin_adherence": sum(r["in_basin"] for r in turn_results) / len(turn_results)
                },
                metadata={"total_turns": len(conversation)}
            )


# ============================================================================
# Baseline 3: Cadence Reminder Runner
# ============================================================================

class CadenceReminderRunner(BaselineRunner):
    """
    Cadence Reminder Baseline: Fixed-interval reminders.
    
    Injects governance reminder every N turns regardless of drift.
    Represents naive periodic reinforcement strategy.
    
    Characteristics:
    - Reminder injected every reminder_cadence turns
    - No drift-based decision making
    - Adds overhead even when unnecessary
    - Tests fixed-schedule governance
    """
    
    def __init__(
        self,
        llm_client,
        embedding_provider,
        attractor_config: PrimacyAttractor,
        reminder_cadence: int = 3
    ):
        """
        Initialize cadence runner.
        
        Args:
            llm_client: LLM client
            embedding_provider: Embedder
            attractor_config: Governance config
            reminder_cadence: Turns between reminders
        """
        super().__init__(llm_client, embedding_provider, attractor_config)
        self.reminder_cadence = reminder_cadence
    
    def run_conversation(
        self,
        conversation: List[Tuple[str, str]]
    ) -> BaselineResult:
        """Run conversation with fixed-interval reminders."""
        with error_context("cadence baseline execution"):
            governance_prompt = (
                f"Purpose: {', '.join(self.attractor_config.purpose)}\n"
                f"Scope: {', '.join(self.attractor_config.scope)}\n"
                f"Boundaries: {', '.join(self.attractor_config.boundaries)}"
            )
            
            conversation_history = [
                {"role": "system", "content": f"You are a helpful assistant. {governance_prompt}"}
            ]
            
            turn_results = []
            states = []
            reminder_count = 0
            
            for turn_num, (user_input, _) in enumerate(conversation, 1):
                # Inject reminder at cadence
                if turn_num > 1 and turn_num % self.reminder_cadence == 0:
                    conversation_history.append({
                        "role": "system",
                        "content": f"Reminder: {governance_prompt}"
                    })
                    reminder_count += 1
                
                conversation_history.append({
                    "role": "user",
                    "content": user_input
                })
                
                try:
                    response = self.llm.generate(
                        messages=conversation_history,
                        max_tokens=500
                    )
                except Exception as e:
                    logger.warning(f"Turn {turn_num} generation failed: {e}")
                    response = "[Generation failed]"
                
                conversation_history.append({
                    "role": "assistant",
                    "content": response
                })
                
                # Compute state
                embedding = self.embedding_provider.encode(response)
                state = MathematicalState(
                    embedding=embedding,
                    turn_number=turn_num,
                    timestamp=time.time(),
                    text_content=response
                )
                states.append(state)
                
                # Measure
                distance = float(np.linalg.norm(
                    embedding - self.attractor.attractor_center
                ))
                in_basin = self.attractor.compute_basin_membership(state)
                
                turn_results.append({
                    "turn": turn_num,
                    "user_input": user_input,
                    "response": response,
                    "distance_to_attractor": distance,
                    "in_basin": in_basin,
                    "reminder_injected": turn_num % self.reminder_cadence == 0 and turn_num > 1,
                    "timestamp": time.time()
                })
            
            final_fidelity = self.fidelity_calc.compute_hard_fidelity(
                states, self.attractor
            )
            
            return BaselineResult(
                runner_type="cadence_reminder",
                session_id=f"cadence_{int(time.time())}",
                turn_results=turn_results,
                final_metrics={
                    "fidelity": final_fidelity,
                    "avg_distance": np.mean([r["distance_to_attractor"] for r in turn_results]),
                    "basin_adherence": sum(r["in_basin"] for r in turn_results) / len(turn_results)
                },
                metadata={
                    "total_turns": len(conversation),
                    "reminder_cadence": self.reminder_cadence,
                    "total_reminders": reminder_count
                }
            )


# ============================================================================
# Baseline 4: Observation Runner
# ============================================================================

class ObservationRunner(BaselineRunner):
    """
    Observation Baseline: Full math active, no interventions.
    
    Proves mathematical instrumentation works independently of
    intervention system. Critical validation baseline.
    
    Characteristics:
    - Complete SPC Engine measurement
    - No Proportional Controller interventions
    - Logs drift detection signals
    - Validates measurement subsystem
    """
    
    def run_conversation(
        self,
        conversation: List[Tuple[str, str]]
    ) -> BaselineResult:
        """Run conversation in observation-only mode."""
        with error_context("observation baseline execution"):
            # Initialize steward with interventions disabled
            steward = UnifiedGovernanceSteward(
                attractor=self.attractor_config,
                llm_client=self.llm,
                embedding_provider=self.embedding_provider,
                enable_interventions=False,  # Critical: observation only
                dev_commentary_mode="silent"
            )
            
            steward.start_session()
            
            turn_results = []
            
            for turn_num, (user_input, _) in enumerate(conversation, 1):
                messages = steward.conversation.get_messages_for_api()
                messages.append({"role": "user", "content": user_input})
                
                try:
                    initial_response = steward.llm_client.generate(
                        messages=messages,
                        max_tokens=500
                    )
                except Exception as e:
                    logger.warning(f"Turn {turn_num} generation failed: {e}")
                    initial_response = "[Generation failed]"
                
                result = steward.process_turn(user_input, initial_response)
                
                turn_results.append({
                    "turn": turn_num,
                    "user_input": user_input,
                    "response": result["final_response"],
                    "distance_to_attractor": result["metrics"]["error_signal"],
                    "in_basin": result["metrics"]["primacy_basin_membership"],
                    "would_intervene": result["metrics"]["error_signal"] >= steward.proportional_controller.epsilon_min,
                    "timestamp": time.time()
                })
            
            summary = steward.end_session()
            
            return BaselineResult(
                runner_type="observation",
                session_id=summary["session_id"],
                turn_results=turn_results,
                final_metrics={
                    "fidelity": summary["session_metadata"]["final_telic_fidelity"],
                    "avg_distance": np.mean([r["distance_to_attractor"] for r in turn_results]),
                    "basin_adherence": sum(r["in_basin"] for r in turn_results) / len(turn_results)
                },
                metadata={
                    "total_turns": len(conversation),
                    "would_intervene_count": sum(r["would_intervene"] for r in turn_results)
                }
            )


# ============================================================================
# Baseline 5: TELOS Runner
# ============================================================================

class TELOSRunner(BaselineRunner):
    """
    TELOS Baseline: Full MBL active.
    
    Complete Mitigation Bridge Layer implementation:
    - SPC Engine: Continuous measurement and analysis
    - Proportional Controller: Graduated interventions
    
    This is the full system being validated.
    
    Characteristics:
    - Real-time drift detection
    - Proportional corrections (F = K·e_t)
    - Graduated intervention cascade
    - Lyapunov stability verification
    - Complete telemetry export
    """
    
    def run_conversation(
        self,
        conversation: List[Tuple[str, str]]
    ) -> BaselineResult:
        """Run conversation with full TELOS governance."""
        with error_context("TELOS baseline execution"):
            steward = UnifiedGovernanceSteward(
                attractor=self.attractor_config,
                llm_client=self.llm,
                embedding_provider=self.embedding_provider,
                enable_interventions=True,  # Full MBL active
                dev_commentary_mode="silent"
            )
            
            steward.start_session()
            
            turn_results = []
            
            for turn_num, (user_input, _) in enumerate(conversation, 1):
                messages = steward.conversation.get_messages_for_api()
                messages.append({"role": "user", "content": user_input})
                
                try:
                    initial_response = steward.llm_client.generate(
                        messages=messages,
                        max_tokens=500
                    )
                except Exception as e:
                    logger.warning(f"Turn {turn_num} generation failed: {e}")
                    initial_response = "[Generation failed]"
                
                result = steward.process_turn(user_input, initial_response)
                
                turn_results.append({
                    "turn": turn_num,
                    "user_input": user_input,
                    "response": result["final_response"],
                    "distance_to_attractor": result["metrics"]["error_signal"],
                    "in_basin": result["metrics"]["primacy_basin_membership"],
                    "governance_action": result["governance_action"],
                    "intervention_applied": result["intervention_applied"],
                    "intervention_type": result["intervention_result"].type if result["intervention_result"] else None,
                    "timestamp": time.time()
                })
            
            summary = steward.end_session()
            
            return BaselineResult(
                runner_type="telos",
                session_id=summary["session_id"],
                turn_results=turn_results,
                final_metrics={
                    "fidelity": summary["session_metadata"]["final_telic_fidelity"],
                    "avg_distance": np.mean([r["distance_to_attractor"] for r in turn_results]),
                    "basin_adherence": sum(r["in_basin"] for r in turn_results) / len(turn_results)
                },
                metadata={
                    "total_turns": len(conversation),
                    "total_interventions": summary["session_metadata"]["total_interventions"],
                    "intervention_rate": summary["session_metadata"]["intervention_rate"],
                    "intervention_statistics": summary.get("intervention_statistics")
                }
            )
