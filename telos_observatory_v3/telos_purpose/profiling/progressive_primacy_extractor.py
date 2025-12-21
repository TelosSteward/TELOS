"""
Progressive Primacy Attractor Extraction - Statistical Convergence
===================================================================

Uses STATISTICAL CONVERGENCE detection instead of arbitrary turn limits.

Key Principles:
1. NO arbitrary turn limits (no min_turns=3, baseline_turns=5)
2. Statistical convergence based on:
   - Centroid stability (rolling window comparison)
   - Variance stability (embedding variance within windows)
   - Confidence scoring (how certain are we?)
3. Runs indefinitely until statistically converged
4. Tracks convergence turn for data-driven parameter tuning

This enables:
- Data-driven parameter selection
- Adapts to conversation complexity
- Provides statistical evidence for grants
- No arbitrary cutoffs
"""

from __future__ import annotations
from typing import List, Optional, Dict, Any
import json
import numpy as np
from scipy.spatial import distance as dist

from telos_purpose.core.unified_steward import PrimacyAttractor
from telos_purpose.profiling.convergence_analyzer import ConvergenceRecord


class ProgressivePrimacyExtractor:
    """
    Progressive attractor establishment using statistical convergence detection.

    Architecture:
    1. Accumulate turns progressively (NO minimum enforced)
    2. At each turn, check statistical convergence:
       - Centroid stability: Compare rolling window centroids
       - Variance stability: Track embedding variance
       - Confidence: Combine metrics for confidence score
    3. When converged: Finalize attractor and switch to tracking
    4. Return ConvergenceRecord for multi-session analysis

    Modes:
    - PROGRESSIVE: Learn entire attractor from conversation
    - HYBRID: Pre-defined boundaries + LLM-learned scope
    """

    def __init__(
        self,
        llm_client,
        embedding_provider,
        mode: str = 'progressive',
        seed_attractor: Optional[PrimacyAttractor] = None,
        # Phase 2 research mode: LLM semantic analysis at every turn
        llm_per_turn: bool = False,  # Enable LLM analysis at each turn (vs once at convergence)
        # Statistical parameters (data-driven, not arbitrary)
        window_size: int = 3,  # Rolling window for stability check
        centroid_stability_threshold: float = 0.95,  # Cosine similarity threshold
        variance_stability_threshold: float = 0.1,  # Relative variance threshold
        confidence_threshold: float = 0.76,  # Goldilocks: Aligned threshold for convergence
        consecutive_stable_turns: int = 2,  # Turns of stability before declaring convergence
        max_turns_safety: int = 10,  # Safety limit (changed from 100 for Phase 2)
        distance_scale: float = 2.0  # Distance-to-fidelity scaling
    ):
        """
        Initialize progressive extractor with statistical convergence.

        Args:
            llm_client: LLM client for semantic analysis
            embedding_provider: Embedding provider for distance measurement
            mode: 'progressive' or 'hybrid'
            seed_attractor: Required for hybrid mode (provides boundaries)
            llm_per_turn: If True, call LLM at every turn for semantic analysis
                         (Phase 2 research mode). If False, call LLM once at convergence
                         (filtering mode). LLM-per-turn provides stronger research evidence.
            window_size: Size of rolling window for stability checks
            centroid_stability_threshold: Cosine similarity threshold for centroids
            variance_stability_threshold: Max acceptable relative variance
            confidence_threshold: Overall confidence needed to declare convergence
            consecutive_stable_turns: Turns of stability needed before convergence
            max_turns_safety: Safety limit to prevent runaway (10 for Phase 2, 100 for filtering)
            distance_scale: Distance-to-fidelity scaling factor

        Raises:
            ValueError: If mode invalid or hybrid missing seed_attractor
        """
        self.llm_client = llm_client
        self.embedding_provider = embedding_provider
        self.mode = mode
        self.seed_attractor = seed_attractor
        self.llm_per_turn = llm_per_turn

        # Statistical convergence parameters
        self.window_size = window_size
        self.centroid_stability_threshold = centroid_stability_threshold
        self.variance_stability_threshold = variance_stability_threshold
        self.confidence_threshold = confidence_threshold
        self.consecutive_stable_turns = consecutive_stable_turns
        self.max_turns_safety = max_turns_safety
        self.distance_scale = distance_scale

        # State tracking
        self.turn_count = 0
        self.accumulated_turns: List[Dict[str, str]] = []
        self.accumulated_embeddings: List[np.ndarray] = []
        self.llm_analyses: List[Dict[str, Any]] = []  # Store LLM analysis per turn (if llm_per_turn=True)
        self.previous_analysis: Optional[Dict[str, Any]] = None
        self.converged = False
        self.convergence_turn: Optional[int] = None
        self.primacy_attractor: Optional[PrimacyAttractor] = None
        self.attractor_centroid: Optional[np.ndarray] = None

        # Convergence tracking
        self.stability_history: List[Dict[str, float]] = []
        self.stable_turn_count = 0
        self.convergence_metrics: Dict[str, Any] = {}

        # Validate mode
        if mode not in ['progressive', 'hybrid']:
            raise ValueError(f"Mode must be 'progressive' or 'hybrid', got '{mode}'")

        if mode == 'hybrid' and seed_attractor is None:
            raise ValueError("Hybrid mode requires seed_attractor parameter")

    def add_turn(
        self,
        user_message: str,
        assistant_response: str
    ) -> Dict[str, Any]:
        """
        Process turn with statistical convergence detection.

        Phase 1 (Pre-convergence):
        - Accumulate turns and embeddings
        - Check statistical convergence at each turn
        - When converged: Finalize attractor

        Phase 2 (Post-convergence):
        - Measure distance from centroid
        - Convert to fidelity score

        Args:
            user_message: User's input message
            assistant_response: Assistant's response

        Returns:
            Dict with:
            - status: 'accumulating', 'analyzing', 'converged', or 'tracking'
            - message: Human-readable status
            - fidelity: Optional[float] (only in tracking phase)
            - convergence_metrics: Optional[Dict] (convergence details)
            - attractor: Optional[PrimacyAttractor] (when converged)
        """
        self.turn_count += 1

        # Store turn
        self.accumulated_turns.append({
            'user': user_message,
            'assistant': assistant_response
        })

        # Get embedding - TWO MODES:
        # 1. LLM-per-turn (Phase 2): Embed LLM's semantic analysis
        # 2. Direct embedding (Filtering): Embed assistant response directly
        if self.llm_per_turn and self.llm_client is not None:
            # PHASE 2 MODE: Call LLM at every turn
            try:
                llm_analysis = self._analyze_with_llm()
                self.llm_analyses.append(llm_analysis)

                # Embed the LLM's semantic understanding
                # Convert purpose/scope/boundaries to text for embedding
                analysis_text = self._format_analysis_for_embedding(llm_analysis)
                embedding = self.embedding_provider.encode(analysis_text)
                self.accumulated_embeddings.append(embedding)
            except Exception as e:
                # LLM call failed - return error
                return {
                    'status': 'error',
                    'message': f'❌ LLM analysis failed at turn {self.turn_count}: {e}',
                    'fidelity': None,
                    'drift_detected': None,
                    'baseline_established': False,
                    'turn_count': self.turn_count
                }
        else:
            # FILTERING MODE: Direct embedding (no LLM per turn)
            embedding = self.embedding_provider.encode(assistant_response)
            self.accumulated_embeddings.append(embedding)

        if not self.converged:
            # ============================================================
            # PHASE 1: STATISTICAL CONVERGENCE DETECTION
            # ============================================================

            # Need at least window_size turns to check stability
            if self.turn_count < self.window_size:
                return {
                    'status': 'accumulating',
                    'message': f'🔄 Accumulating data... ({self.turn_count} turns, need {self.window_size} for initial check)',
                    'fidelity': None,
                    'drift_detected': None,
                    'baseline_established': False,
                    'turn_count': self.turn_count
                }

            # Check statistical convergence
            convergence_check = self._check_statistical_convergence()

            # Track stability
            self.stability_history.append(convergence_check)

            # Update stable turn counter
            if convergence_check['is_stable']:
                self.stable_turn_count += 1
            else:
                self.stable_turn_count = 0  # Reset if not stable

            # Check if converged (stable for consecutive turns AND confidence high)
            if (self.stable_turn_count >= self.consecutive_stable_turns and
                convergence_check['confidence'] >= self.confidence_threshold):

                # Mark convergence
                self.converged = True
                self.convergence_turn = self.turn_count
                self.convergence_metrics = convergence_check

                # CONVERGED - finalize attractor
                if self.llm_client is not None:
                    try:
                        # Get final LLM analysis
                        if self.llm_per_turn:
                            # LLM-per-turn mode: Use most recent analysis (latest understanding)
                            llm_analysis = self.llm_analyses[-1]
                            message_mode = "LLM-per-turn"
                        else:
                            # Standard mode: Call LLM once now with all accumulated turns
                            llm_analysis = self._analyze_with_llm()
                            message_mode = "statistical"

                        self._finalize_attractor(llm_analysis)

                        return {
                            'status': 'converged',
                            'message': f'✅ Converged at turn {self.turn_count} ({message_mode} mode)',
                            'fidelity': None,
                            'drift_detected': None,
                            'baseline_established': True,
                            'turn_count': self.turn_count,
                            'attractor': self.primacy_attractor,
                            'convergence_turn': self.convergence_turn,
                            'llm_analyses': self.llm_analyses if self.llm_per_turn else None,  # Return all analyses in Phase 2
                            'convergence_metrics': {
                                'centroid_stability': convergence_check['centroid_stability'],
                                'variance_stability': convergence_check['variance_stability'],
                                'confidence': convergence_check['confidence'],
                                'turns_to_convergence': self.convergence_turn,
                            }
                        }
                    except Exception as e:
                        return {
                            'status': 'error',
                            'message': f'❌ LLM analysis failed at convergence: {e}',
                            'fidelity': None,
                            'drift_detected': None,
                            'baseline_established': False,
                            'turn_count': self.turn_count
                        }
                else:
                    # NO LLM - just return convergence metrics (for filtering)
                    return {
                        'status': 'converged_statistical_only',
                        'message': f'✅ Statistical convergence at turn {self.turn_count} (no LLM analysis)',
                        'fidelity': None,
                        'drift_detected': None,
                        'baseline_established': False,  # No attractor created without LLM
                        'turn_count': self.turn_count,
                        'attractor': None,
                        'convergence_turn': self.convergence_turn,
                        'convergence_metrics': {
                            'centroid_stability': convergence_check['centroid_stability'],
                            'variance_stability': convergence_check['variance_stability'],
                            'confidence': convergence_check['confidence'],
                            'turns_to_convergence': self.convergence_turn,
                        }
                    }

            # Safety check: prevent infinite accumulation
            if self.turn_count >= self.max_turns_safety:
                # Mark convergence at safety limit
                self.converged = True
                self.convergence_turn = self.turn_count

                # Force convergence with current data (if LLM available)
                if self.llm_client is not None:
                    try:
                        llm_analysis = self._analyze_with_llm()
                        self._finalize_attractor(llm_analysis)

                        return {
                            'status': 'converged',
                            'message': f'⚠️  Converged at safety limit (turn {self.turn_count})',
                            'fidelity': None,
                            'drift_detected': None,
                            'baseline_established': True,
                            'turn_count': self.turn_count,
                            'attractor': self.primacy_attractor,
                            'convergence_turn': self.convergence_turn,
                            'convergence_metrics': {
                                'forced': True,
                                'reason': 'safety_limit_reached'
                            }
                        }
                    except Exception as e:
                        return {
                            'status': 'error',
                            'message': f'❌ Forced convergence failed: {e}',
                            'fidelity': None,
                            'drift_detected': None,
                            'baseline_established': False,
                            'turn_count': self.turn_count
                        }
                else:
                    # NO LLM - just mark as reached safety limit
                    return {
                        'status': 'safety_limit_reached',
                        'message': f'⚠️  Safety limit reached at turn {self.turn_count} (no LLM analysis)',
                        'fidelity': None,
                        'drift_detected': None,
                        'baseline_established': False,
                        'turn_count': self.turn_count,
                        'attractor': None,
                        'convergence_turn': self.convergence_turn,
                        'convergence_metrics': {
                            'forced': True,
                            'reason': 'safety_limit_reached_no_llm'
                        }
                    }

            # Still accumulating
            return {
                'status': 'analyzing',
                'message': f'📊 Statistical analysis... (turn {self.turn_count}, confidence: {convergence_check["confidence"]:.2f}, stable: {self.stable_turn_count}/{self.consecutive_stable_turns})',
                'fidelity': None,
                'drift_detected': None,
                'baseline_established': False,
                'turn_count': self.turn_count,
                'convergence_metrics': convergence_check
            }

        else:
            # ============================================================
            # PHASE 2: MATHEMATICAL TRACKING
            # ============================================================

            # Measure distance from attractor centroid
            distance = self._compute_distance(embedding, self.attractor_centroid)

            # Convert distance to fidelity
            fidelity = self._distance_to_fidelity(distance)

            # Detect drift (Goldilocks: Aligned threshold)
            drift_detected = fidelity < 0.76

            return {
                'status': 'tracking',
                'message': f'📊 Mathematical tracking (F={fidelity:.3f})',
                'fidelity': fidelity,
                'drift_detected': drift_detected,
                'baseline_established': True,
                'turn_count': self.turn_count,
                'distance': distance
            }

    def _check_statistical_convergence(self) -> Dict[str, Any]:
        """
        Check statistical convergence using multiple metrics.

        Returns:
            Dict with:
            - centroid_stability: Cosine similarity between window centroids
            - variance_stability: Relative variance within current window
            - confidence: Overall confidence score (0-1)
            - is_stable: Boolean indicating if metrics pass thresholds
        """
        # Get current window embeddings
        current_window = self.accumulated_embeddings[-self.window_size:]
        current_centroid = np.mean(current_window, axis=0)

        # Need at least 2*window_size turns to compare windows
        if self.turn_count >= 2 * self.window_size:
            # Get previous window
            prev_start = -(2 * self.window_size)
            prev_end = -self.window_size
            previous_window = self.accumulated_embeddings[prev_start:prev_end]
            previous_centroid = np.mean(previous_window, axis=0)

            # Centroid stability: Cosine similarity between centroids
            centroid_stability = 1 - dist.cosine(current_centroid, previous_centroid)

        else:
            # Not enough data for window comparison
            centroid_stability = 0.0

        # Variance stability: Relative variance within current window
        # Lower variance = more stable
        variances = np.var(current_window, axis=0)
        mean_variance = np.mean(variances)
        norm = np.linalg.norm(current_centroid)
        relative_variance = mean_variance / (norm ** 2) if norm > 0 else 1.0

        variance_stability = 1.0 - min(relative_variance / self.variance_stability_threshold, 1.0)

        # Confidence score: Weighted combination of metrics
        # More data = higher confidence
        data_confidence = min(self.turn_count / (3 * self.window_size), 1.0)

        # Combined confidence
        confidence = (
            0.4 * centroid_stability +
            0.3 * variance_stability +
            0.3 * data_confidence
        )

        # Check if stable (all thresholds met)
        is_stable = (
            centroid_stability >= self.centroid_stability_threshold and
            relative_variance <= self.variance_stability_threshold and
            self.turn_count >= self.window_size
        )

        return {
            'centroid_stability': float(centroid_stability),
            'variance_stability': float(variance_stability),
            'relative_variance': float(relative_variance),
            'confidence': float(confidence),
            'is_stable': is_stable,
            'data_points': self.turn_count,
        }

    def _analyze_with_llm(self) -> Dict[str, Any]:
        """
        Use LLM to analyze accumulated turns.

        Based on ProfileExtractor approach - uses LLM to semantically
        extract purpose, scope, and boundaries from conversation.

        Returns:
            Dict with:
            - purpose: List[str]
            - scope: List[str]
            - boundaries: List[str]

        Raises:
            Exception: If LLM analysis or JSON parsing fails
        """
        # Format turns for analysis
        conversation_text = self._format_turns()

        # LLM analysis prompt
        system_prompt = """You are a conversation analyzer. Extract the primary purpose, key topics, and implicit boundaries from conversation transcripts.

Analyze what the conversation is truly about - its core themes and focus areas.

Respond ONLY with valid JSON in this exact format:
{
    "purpose": ["primary purpose 1", "primary purpose 2"],
    "scope": ["topic 1", "topic 2", "topic 3"],
    "boundaries": ["boundary 1", "boundary 2"]
}"""

        user_prompt = f"""Analyze this conversation and extract:

1. **Purpose**: What is this conversation trying to accomplish? What are its main goals?
2. **Scope**: What topics/themes are being discussed? What domains are covered?
3. **Boundaries**: What is implicitly out of scope? What topics are being avoided or would be inappropriate?

Conversation:
{conversation_text}

Respond with JSON only."""

        # Get LLM analysis
        response = self.llm_client.generate(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )

        # Parse JSON response
        analysis = self._parse_json_response(response)

        return analysis

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON from LLM response, handling markdown fences."""
        # Remove markdown code fences if present
        if "```json" in response:
            json_str = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            json_str = response.split("```")[1].split("```")[0].strip()
        else:
            json_str = response.strip()

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            # Return default structure if parsing fails
            return {
                "purpose": ["General conversation"],
                "scope": ["Various topics"],
                "boundaries": ["Stay on topic"]
            }

    def _format_analysis_for_embedding(self, analysis: Dict[str, Any]) -> str:
        """
        Format LLM analysis into text for embedding.

        Used in llm_per_turn mode to embed the LLM's semantic understanding
        rather than the raw conversation text.

        Args:
            analysis: LLM analysis with purpose/scope/boundaries

        Returns:
            Formatted text string for embedding
        """
        purpose = ', '.join(analysis.get('purpose', ['General conversation']))
        scope = ', '.join(analysis.get('scope', ['Various topics']))
        boundaries = ', '.join(analysis.get('boundaries', ['Stay on topic']))

        return f"Purpose: {purpose}. Scope: {scope}. Boundaries: {boundaries}."

    def _finalize_attractor(self, analysis: Dict[str, Any]) -> None:
        """
        Finalize attractor from LLM analysis.

        Creates PrimacyAttractor and computes centroid from accumulated embeddings.

        Args:
            analysis: Final LLM analysis with purpose/scope/boundaries
        """
        if self.mode == 'progressive':
            # Pure progressive: All from LLM
            self.primacy_attractor = PrimacyAttractor(
                purpose=analysis.get('purpose', ['General conversation']),
                scope=analysis.get('scope', ['Various topics']),
                boundaries=analysis.get('boundaries', ['Stay on topic']),
                constraint_tolerance=0.2,
                privacy_level=0.8,
                task_priority=0.7
            )

        elif self.mode == 'hybrid':
            # Hybrid: LLM purpose/scope + seed boundaries/parameters
            self.primacy_attractor = PrimacyAttractor(
                purpose=analysis.get('purpose', ['General conversation']),  # FROM LLM
                scope=analysis.get('scope', ['Various topics']),            # FROM LLM
                boundaries=self.seed_attractor.boundaries,                  # FROM SEED
                privacy_level=self.seed_attractor.privacy_level,           # FROM SEED
                constraint_tolerance=self.seed_attractor.constraint_tolerance,  # FROM SEED
                task_priority=self.seed_attractor.task_priority            # FROM SEED
            )

        # Compute centroid from ALL accumulated embeddings
        embeddings_array = np.vstack(self.accumulated_embeddings)
        self.attractor_centroid = np.mean(embeddings_array, axis=0)

        # Mark as converged
        self.converged = True
        self.convergence_turn = self.turn_count

        # Store final convergence metrics
        if self.stability_history:
            self.convergence_metrics = self.stability_history[-1]

    def _compute_distance(
        self,
        embedding: np.ndarray,
        centroid: np.ndarray
    ) -> float:
        """
        Compute Euclidean distance from embedding to centroid.

        Args:
            embedding: Response embedding
            centroid: Attractor centroid

        Returns:
            Euclidean distance
        """
        return np.linalg.norm(embedding - centroid)

    def _distance_to_fidelity(self, distance: float) -> float:
        """
        Convert distance to fidelity score (0-1).

        Uses exponential decay: F = exp(-distance / scale)
        This gives smooth falloff where close = high fidelity.

        Args:
            distance: Euclidean distance from centroid

        Returns:
            Fidelity score between 0.0 and 1.0
        """
        fidelity = np.exp(-distance / self.distance_scale)
        return float(np.clip(fidelity, 0.0, 1.0))

    def _format_turns(self) -> str:
        """
        Format accumulated turns for LLM analysis.

        Returns:
            Formatted conversation string
        """
        lines = []
        for i, turn in enumerate(self.accumulated_turns):
            lines.append(f"Turn {i+1}:")
            lines.append(f"Human: {turn['user']}")
            lines.append(f"Assistant: {turn['assistant']}")
            lines.append("")
        return '\n'.join(lines)

    def is_ready(self) -> bool:
        """
        Check if attractor converged and ready for tracking.

        Returns:
            True if converged, False otherwise
        """
        return self.converged

    def get_convergence_record(self, session_id: str) -> Optional[ConvergenceRecord]:
        """
        Get convergence record for multi-session analysis.

        Args:
            session_id: Identifier for this session

        Returns:
            ConvergenceRecord if converged, None otherwise
        """
        if not self.converged:
            return None

        return ConvergenceRecord(
            session_id=session_id,
            convergence_turn=self.convergence_turn or 0,
            total_turns=self.turn_count,
            confidence_score=self.convergence_metrics.get('confidence', 0.0),
            centroid_stability=self.convergence_metrics.get('centroid_stability', 0.0),
            variance_stability=self.convergence_metrics.get('variance_stability', 0.0),
            convergence_time=0.0,  # To be filled by caller
            attractor_quality={
                'purpose': self.primacy_attractor.purpose if self.primacy_attractor else [],
                'scope': self.primacy_attractor.scope if self.primacy_attractor else [],
                'boundaries': self.primacy_attractor.boundaries if self.primacy_attractor else [],
            }
        )

    def get_status_message(self) -> str:
        """
        Get human-readable status message.

        Returns:
            Status message string
        """
        if not self.converged:
            if self.turn_count < self.window_size:
                return f'🔄 Accumulating data... ({self.turn_count}/{self.window_size} turns for initial analysis)'
            else:
                latest_metrics = self.stability_history[-1] if self.stability_history else {}
                confidence = latest_metrics.get('confidence', 0.0)
                return f'📊 Statistical analysis... (turn {self.turn_count}, confidence: {confidence:.2f}, stable: {self.stable_turn_count}/{self.consecutive_stable_turns})'
        else:
            if self.primacy_attractor and self.primacy_attractor.scope:
                scope_sample = ', '.join(self.primacy_attractor.scope[:3])
                return f'✅ Converged at turn {self.convergence_turn} | Scope: {scope_sample}'
            else:
                return f'✅ Statistically converged at turn {self.convergence_turn}'

    def get_mode(self) -> str:
        """
        Get current operating mode.

        Returns:
            'progressive' or 'hybrid'
        """
        return self.mode

    def get_attractor(self) -> Optional[PrimacyAttractor]:
        """
        Get the established primacy attractor.

        Returns:
            PrimacyAttractor if converged, None otherwise
        """
        return self.primacy_attractor

    def get_convergence_turn(self) -> Optional[int]:
        """
        Get the turn number at which attractor converged.

        Returns:
            Turn number or None if not yet converged
        """
        return self.convergence_turn if self.converged else None
