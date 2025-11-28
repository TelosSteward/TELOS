"""
UnifiedOrchestratorSteward - Governance Orchestration Layer

Orchestrates TELOS governance workflow across single and dual PA modes.

Architecture:
    UnifiedOrchestratorSteward (this class)
        ├─ Manages governance mode configuration
        ├─ Creates and derives primacy attractors
        ├─ Coordinates session lifecycle
        └─ Routes to UnifiedGovernanceSteward for execution

    UnifiedGovernanceSteward
        ├─ Performs fidelity calculations
        ├─ Executes interventions
        └─ Returns results

Status: Experimental (v1.2-dual-attractor integration)
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import time
import logging
import asyncio

from .governance_config import GovernanceConfig, GovernanceMode
from .dual_attractor import (
    create_dual_pa,
    DualPrimacyAttractor,
    check_dual_pa_fidelity
)
from .unified_steward import (
    UnifiedGovernanceSteward,
    PrimacyAttractor
)

logger = logging.getLogger(__name__)


class UnifiedOrchestratorSteward:
    """
    Orchestrates TELOS governance workflow with support for single and dual PA modes.

    Responsibilities:
    - Initialize governance mode (single PA, dual PA, or auto)
    - Create and derive primacy attractors
    - Manage dual PA correlation and fallback logic
    - Coordinate session lifecycle
    - Route to UnifiedGovernanceSteward for execution

    Example:
        >>> config = GovernanceConfig.dual_pa_config()
        >>> orchestrator = UnifiedOrchestratorSteward(
        ...     governance_config=config,
        ...     user_pa_config=my_pa,
        ...     llm_client=client,
        ...     embedding_provider=embedder
        ... )
        >>> orchestrator.start_session()
        >>> response = orchestrator.generate_governed_response(user_input, context)
        >>> summary = orchestrator.end_session()
    """

    def __init__(
        self,
        governance_config: GovernanceConfig,
        user_pa_config: Dict[str, Any],
        llm_client,
        embedding_provider,
        enable_interventions: bool = True,
        dev_commentary_mode: str = "silent",
        health_monitor = None
    ):
        """
        Initialize governance orchestrator.

        Args:
            governance_config: Governance mode configuration
            user_pa_config: User's primacy attractor configuration
            llm_client: LLM client for natural language operations
            embedding_provider: Text-to-vector encoder
            enable_interventions: Enable intervention system
            dev_commentary_mode: "silent", "concise", or "verbose"
            health_monitor: Optional SystemHealthMonitor instance
        """
        self.config = governance_config
        self.user_pa_config = user_pa_config
        self.llm_client = llm_client
        self.embedding_provider = embedding_provider
        self.enable_interventions = enable_interventions
        self.dev_commentary_mode = dev_commentary_mode
        self.health = health_monitor

        # PA state
        self.dual_pa: Optional[DualPrimacyAttractor] = None
        self.user_pa_embedding: Optional[Any] = None
        self.ai_pa_embedding: Optional[Any] = None
        self.actual_governance_mode: str = "dual"  # Actual mode after initialization - Default to dual

        # Governance executor
        self.governance_steward: Optional[UnifiedGovernanceSteward] = None

        # Session state
        self.session_active: bool = False
        self.session_id: Optional[str] = None

        logger.info(
            f"UnifiedOrchestratorSteward initialized: "
            f"mode={governance_config.mode.value}, "
            f"dual_pa_enabled={governance_config.dual_pa_enabled}"
        )

    async def initialize_governance(self) -> None:
        """
        Initialize governance system (PA creation and steward setup).

        This is called automatically by start_session(), but can be called
        separately if you want to pre-initialize before session start.

        Handles:
        - Dual PA creation (if enabled)
        - Correlation checking
        - Fallback to single PA if needed
        - UnifiedGovernanceSteward initialization

        Raises:
            Exception: Only in strict mode. Otherwise falls back gracefully.
        """
        logger.info("Initializing governance system...")

        # Determine if we should attempt dual PA
        attempt_dual_pa = self.config.dual_pa_enabled

        if attempt_dual_pa:
            try:
                # Create dual PA with timeout protection
                logger.info("Creating dual PA (async with timeout protection)...")
                self.dual_pa = await asyncio.wait_for(
                    create_dual_pa(
                        user_pa=self.user_pa_config,
                        client=self.llm_client,
                        enable_dual_mode=True,
                        template=self.config.ai_pa_template
                    ),
                    timeout=self.config.derivation_timeout_seconds
                )

                # Check if dual PA was successfully created or fell back to single
                if self.dual_pa.is_dual_mode():
                    # Dual PA successfully created
                    self.actual_governance_mode = "dual"

                    # Check correlation threshold
                    if not self.config.should_use_dual_pa(self.dual_pa.correlation):
                        logger.warning(
                            f"Dual PA correlation ({self.dual_pa.correlation:.2f}) "
                            f"below threshold ({self.config.correlation_minimum}). "
                            f"Falling back to single PA mode."
                        )
                        self.actual_governance_mode = "single"
                        self.dual_pa = DualPrimacyAttractor(
                            user_pa=self.user_pa_config,
                            governance_mode='single'
                        )
                    else:
                        logger.info(
                            f"Dual PA initialized successfully: "
                            f"correlation={self.dual_pa.correlation:.2f}"
                        )

                        # Generate embeddings for both PAs (parallel)
                        if self.config.async_mode:
                            user_purpose_text = ' '.join(self.dual_pa.user_pa['purpose'])
                            ai_purpose_text = ' '.join(self.dual_pa.ai_pa['purpose'])

                            # Run embeddings in parallel
                            self.user_pa_embedding, self.ai_pa_embedding = await asyncio.gather(
                                asyncio.to_thread(self.embedding_provider.encode, user_purpose_text),
                                asyncio.to_thread(self.embedding_provider.encode, ai_purpose_text)
                            )
                        else:
                            user_purpose_text = ' '.join(self.dual_pa.user_pa['purpose'])
                            ai_purpose_text = ' '.join(self.dual_pa.ai_pa['purpose'])
                            self.user_pa_embedding = self.embedding_provider.encode(user_purpose_text)
                            self.ai_pa_embedding = self.embedding_provider.encode(ai_purpose_text)
                else:
                    # create_dual_pa already fell back to single mode
                    self.actual_governance_mode = "single"
                    logger.info("Dual PA creation fell back to single PA mode")

            except asyncio.TimeoutError:
                logger.error(
                    f"Dual PA derivation timeout ({self.config.derivation_timeout_seconds}s). "
                    f"Falling back to single PA mode."
                )
                if self.config.strict_mode:
                    raise

                # Fallback to single PA
                self.actual_governance_mode = "single"
                self.dual_pa = DualPrimacyAttractor(
                    user_pa=self.user_pa_config,
                    governance_mode='single'
                )

            except Exception as e:
                logger.error(f"Error creating dual PA: {e}. Falling back to single PA mode.")
                if self.config.strict_mode:
                    raise

                # Fallback to single PA
                self.actual_governance_mode = "single"
                self.dual_pa = DualPrimacyAttractor(
                    user_pa=self.user_pa_config,
                    governance_mode='single'
                )
        else:
            # Single PA mode requested
            logger.info("Single PA mode enabled (no dual PA derivation)")
            self.actual_governance_mode = "single"
            self.dual_pa = DualPrimacyAttractor(
                user_pa=self.user_pa_config,
                governance_mode='single'
            )

        # Generate user PA embedding if not already done
        if self.user_pa_embedding is None:
            user_purpose_text = ' '.join(self.user_pa_config['purpose'])
            self.user_pa_embedding = self.embedding_provider.encode(user_purpose_text)

        # Create PrimacyAttractor for UnifiedGovernanceSteward
        attractor = PrimacyAttractor(
            purpose=self.user_pa_config['purpose'],
            scope=self.user_pa_config['scope'],
            boundaries=self.user_pa_config['boundaries'],
            privacy_level=self.user_pa_config.get('privacy_level', 0.8),
            constraint_tolerance=self.user_pa_config.get('constraint_tolerance', 0.2),
            task_priority=self.user_pa_config.get('task_priority', 0.7)
        )

        # Initialize UnifiedGovernanceSteward
        self.governance_steward = UnifiedGovernanceSteward(
            attractor=attractor,
            llm_client=self.llm_client,
            embedding_provider=self.embedding_provider,
            enable_interventions=self.enable_interventions,
            dev_commentary_mode=self.dev_commentary_mode,
            health_monitor=self.health
        )

        logger.info(
            f"Governance system initialized: "
            f"mode={self.actual_governance_mode}, "
            f"dual_pa={'yes' if self.dual_pa.is_dual_mode() else 'no'}"
        )

    def start_session(self, session_id: Optional[str] = None) -> str:
        """
        Start governed session.

        Initializes governance system if not already initialized,
        then starts session with UnifiedGovernanceSteward.

        Args:
            session_id: Optional custom session ID

        Returns:
            Session ID

        Raises:
            RuntimeError: If session already active
        """
        if self.session_active:
            raise RuntimeError(f"Session already active: {self.session_id}")

        # Initialize governance if not already done
        if self.governance_steward is None:
            logger.info("Governance not initialized, initializing now (sync wrapper)...")
            # Run async initialization in sync context
            asyncio.run(self.initialize_governance())

        # Start session with governance steward
        self.session_id = self.governance_steward.start_session(session_id)
        self.session_active = True

        logger.info(
            f"Session started: {self.session_id} "
            f"(mode: {self.actual_governance_mode})"
        )

        return self.session_id

    def generate_governed_response(
        self,
        user_input: str,
        conversation_context: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Generate governed response through active mitigation layer.

        Args:
            user_input: User's message
            conversation_context: Full conversation history

        Returns:
            Dict with:
            - governed_response: The response to show user
            - intervention_applied: bool
            - intervention_type: str
            - fidelity: float (or dict with user_fidelity/ai_fidelity)
            - metrics: Dict

        Raises:
            RuntimeError: If session not active
        """
        if not self.session_active:
            raise RuntimeError("Session not started. Call start_session() first.")

        # Delegate to UnifiedGovernanceSteward
        result = self.governance_steward.generate_governed_response(
            user_input=user_input,
            conversation_context=conversation_context
        )

        # If dual PA mode, add dual fidelity information
        if self.dual_pa.is_dual_mode():
            # Get response embedding
            response_embedding = self.embedding_provider.encode(result['governed_response'])

            # Calculate dual PA fidelity
            dual_fidelity = check_dual_pa_fidelity(
                response_embedding=response_embedding,
                dual_pa=self.dual_pa,
                embedding_provider=self.embedding_provider
            )

            # Enhance result with dual PA metrics
            result['dual_pa_metrics'] = {
                'user_fidelity': dual_fidelity.user_fidelity,
                'ai_fidelity': dual_fidelity.ai_fidelity,
                'user_pass': dual_fidelity.user_pass,
                'ai_pass': dual_fidelity.ai_pass,
                'overall_pass': dual_fidelity.overall_pass,
                'dominant_failure': dual_fidelity.dominant_failure,
                'governance_mode': dual_fidelity.governance_mode
            }

        return result

    def process_turn(
        self,
        user_input: str,
        model_response: str
    ) -> Dict[str, Any]:
        """
        Process pre-existing turn (REPLAY MODE for analysis).

        For active governance, use generate_governed_response() instead.

        Args:
            user_input: User's message
            model_response: Pre-generated model response

        Returns:
            Dict with final_response, metrics, governance_action

        Raises:
            RuntimeError: If session not active
        """
        if not self.session_active:
            raise RuntimeError("Session not started. Call start_session() first.")

        # Delegate to UnifiedGovernanceSteward
        result = self.governance_steward.process_turn(
            user_input=user_input,
            model_response=model_response
        )

        # If dual PA mode, add dual fidelity information
        if self.dual_pa.is_dual_mode():
            # Get response embedding
            response_embedding = self.embedding_provider.encode(result['final_response'])

            # Calculate dual PA fidelity
            dual_fidelity = check_dual_pa_fidelity(
                response_embedding=response_embedding,
                dual_pa=self.dual_pa,
                embedding_provider=self.embedding_provider
            )

            # Enhance result with dual PA metrics
            result['dual_pa_metrics'] = {
                'user_fidelity': dual_fidelity.user_fidelity,
                'ai_fidelity': dual_fidelity.ai_fidelity,
                'user_pass': dual_fidelity.user_pass,
                'ai_pass': dual_fidelity.ai_pass,
                'overall_pass': dual_fidelity.overall_pass,
                'dominant_failure': dual_fidelity.dominant_failure,
                'governance_mode': dual_fidelity.governance_mode
            }

        return result

    def end_session(self) -> Dict[str, Any]:
        """
        End session and generate summary.

        Returns:
            Session summary with metrics and governance statistics

        Raises:
            RuntimeError: If no active session
        """
        if not self.session_active:
            raise RuntimeError("No active session to end")

        # End session with governance steward
        summary = self.governance_steward.end_session()

        # Add orchestrator-level metadata
        summary['orchestrator_metadata'] = {
            'governance_mode_requested': self.config.mode.value,
            'governance_mode_actual': self.actual_governance_mode,
            'dual_pa_used': self.dual_pa.is_dual_mode() if self.dual_pa else False,
            'dual_pa_correlation': self.dual_pa.correlation if self.dual_pa and self.dual_pa.is_dual_mode() else None
        }

        self.session_active = False
        logger.info(f"Session ended: {self.session_id}")

        return summary

    def get_governance_status(self) -> Dict[str, Any]:
        """
        Get current governance system status.

        Returns:
            Dict with governance configuration and state
        """
        return {
            'governance_mode_configured': self.config.mode.value,
            'governance_mode_actual': self.actual_governance_mode,
            'dual_pa_enabled': self.config.dual_pa_enabled,
            'dual_pa_active': self.dual_pa.is_dual_mode() if self.dual_pa else False,
            'dual_pa_correlation': self.dual_pa.correlation if self.dual_pa and self.dual_pa.is_dual_mode() else None,
            'session_active': self.session_active,
            'session_id': self.session_id,
            'user_pa_threshold': self.config.user_pa_threshold,
            'ai_pa_threshold': self.config.ai_pa_threshold if self.dual_pa and self.dual_pa.is_dual_mode() else None
        }
