"""
Governance Mode Configuration

Provides runtime configuration for switching between single PA and dual PA modes.
Enables A/B testing and empirical comparison of governance approaches.

Status: Experimental (v1.2-dual-attractor)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class GovernanceMode(Enum):
    """Governance mode selection."""
    SINGLE_PA = "single"  # User PA only (v1.1 baseline)
    DUAL_PA = "dual"      # User PA + AI PA (v1.2 experimental)
    AUTO = "auto"         # Automatic mode selection based on correlation


@dataclass
class GovernanceConfig:
    """
    Configuration for TELOS governance behavior.

    This config controls which PA mode is active and related settings.
    Can be switched at runtime to enable A/B testing.
    """

    # Core mode selection
    mode: GovernanceMode = GovernanceMode.DUAL_PA  # Default to Dual PA (canonical implementation)

    # Dual PA settings
    dual_pa_enabled: bool = True  # Dual PA is now the default
    ai_pa_template: Optional[Dict[str, Any]] = None

    # Thresholds
    user_pa_threshold: float = 0.65
    ai_pa_threshold: float = 0.70
    correlation_minimum: float = 0.2  # Below this, fallback to single PA

    # Performance settings
    async_mode: bool = True  # Enable async/parallel execution
    max_derivation_retries: int = 3
    derivation_timeout_seconds: float = 10.0

    # Error handling
    strict_mode: bool = False  # If True, fail on errors. If False, fallback gracefully
    log_fallbacks: bool = True

    # Comparison/testing
    enable_metrics_collection: bool = True
    save_comparison_data: bool = False

    def __post_init__(self):
        """Validate and synchronize settings."""
        # Synchronize dual_pa_enabled with mode
        if self.mode == GovernanceMode.DUAL_PA:
            self.dual_pa_enabled = True
        elif self.mode == GovernanceMode.SINGLE_PA:
            self.dual_pa_enabled = False
        # AUTO mode keeps dual_pa_enabled as-is (for dynamic switching)

        # Validate thresholds
        if not (0.0 <= self.user_pa_threshold <= 1.0):
            raise ValueError(f"user_pa_threshold must be in [0, 1], got {self.user_pa_threshold}")
        if not (0.0 <= self.ai_pa_threshold <= 1.0):
            raise ValueError(f"ai_pa_threshold must be in [0, 1], got {self.ai_pa_threshold}")
        if not (0.0 <= self.correlation_minimum <= 1.0):
            raise ValueError(f"correlation_minimum must be in [0, 1], got {self.correlation_minimum}")

        logger.info(f"Governance config initialized: mode={self.mode.value}, dual_pa_enabled={self.dual_pa_enabled}")

    def should_use_dual_pa(self, correlation: Optional[float] = None) -> bool:
        """
        Determine if dual PA should be used given current config and correlation.

        Args:
            correlation: Optional correlation score between user PA and AI PA

        Returns:
            True if dual PA should be used, False otherwise
        """
        if self.mode == GovernanceMode.SINGLE_PA:
            return False

        if self.mode == GovernanceMode.DUAL_PA:
            # Check correlation if provided
            if correlation is not None and correlation < self.correlation_minimum:
                if self.log_fallbacks:
                    logger.warning(
                        f"Correlation ({correlation:.2f}) below minimum ({self.correlation_minimum}), "
                        f"falling back to single PA"
                    )
                return False
            return True

        if self.mode == GovernanceMode.AUTO:
            # Auto mode uses correlation to decide
            if correlation is None:
                # No correlation yet, default to dual PA attempt
                return self.dual_pa_enabled

            if correlation < self.correlation_minimum:
                if self.log_fallbacks:
                    logger.info(f"Auto mode: correlation {correlation:.2f} too low, using single PA")
                return False
            else:
                if self.log_fallbacks:
                    logger.info(f"Auto mode: correlation {correlation:.2f} sufficient, using dual PA")
                return True

        # Fallback
        return False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize config to dict for logging/comparison."""
        return {
            'mode': self.mode.value,
            'dual_pa_enabled': self.dual_pa_enabled,
            'user_pa_threshold': self.user_pa_threshold,
            'ai_pa_threshold': self.ai_pa_threshold,
            'correlation_minimum': self.correlation_minimum,
            'async_mode': self.async_mode,
            'strict_mode': self.strict_mode
        }

    @classmethod
    def single_pa_config(cls) -> 'GovernanceConfig':
        """Factory for single PA configuration (v1.1 baseline)."""
        return cls(
            mode=GovernanceMode.SINGLE_PA,
            dual_pa_enabled=False,
            async_mode=True,
            strict_mode=False
        )

    @classmethod
    def dual_pa_config(
        cls,
        ai_pa_template: Optional[Dict[str, Any]] = None,
        strict_mode: bool = False
    ) -> 'GovernanceConfig':
        """Factory for dual PA configuration (v1.2 experimental)."""
        return cls(
            mode=GovernanceMode.DUAL_PA,
            dual_pa_enabled=True,
            ai_pa_template=ai_pa_template,
            async_mode=True,
            strict_mode=strict_mode,
            enable_metrics_collection=True
        )

    @classmethod
    def auto_config(
        cls,
        correlation_minimum: float = 0.3
    ) -> 'GovernanceConfig':
        """Factory for auto-switching configuration."""
        return cls(
            mode=GovernanceMode.AUTO,
            dual_pa_enabled=True,  # Attempt dual PA
            correlation_minimum=correlation_minimum,
            async_mode=True,
            strict_mode=False,
            log_fallbacks=True
        )


@dataclass
class ComparisonMetrics:
    """
    Metrics collected for comparing single PA vs dual PA performance.

    Used when running dual PA over existing single PA datasets.
    """

    session_id: str
    governance_mode: str

    # Fidelity scores
    user_fidelity: float
    ai_fidelity: Optional[float] = None  # None in single PA mode
    overall_pass: bool = False

    # PA info
    user_pa_threshold: float = 0.65
    ai_pa_threshold: Optional[float] = None
    correlation: Optional[float] = None

    # Performance
    derivation_time_ms: Optional[float] = None
    total_processing_time_ms: float = 0.0

    # Errors
    had_error: bool = False
    error_type: Optional[str] = None
    fallback_used: bool = False

    # Metadata
    turn_number: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize metrics for comparison analysis."""
        return {
            'session_id': self.session_id,
            'governance_mode': self.governance_mode,
            'turn_number': self.turn_number,
            'user_fidelity': self.user_fidelity,
            'ai_fidelity': self.ai_fidelity,
            'overall_pass': self.overall_pass,
            'user_pa_threshold': self.user_pa_threshold,
            'ai_pa_threshold': self.ai_pa_threshold,
            'correlation': self.correlation,
            'derivation_time_ms': self.derivation_time_ms,
            'total_processing_time_ms': self.total_processing_time_ms,
            'had_error': self.had_error,
            'error_type': self.error_type,
            'fallback_used': self.fallback_used,
            'metadata': self.metadata
        }


# Global config instance (can be overridden)
_global_config: Optional[GovernanceConfig] = None


def set_global_config(config: GovernanceConfig) -> None:
    """Set global governance configuration."""
    global _global_config
    _global_config = config
    logger.info(f"Global governance config set: {config.mode.value}")


def get_global_config() -> GovernanceConfig:
    """Get global governance configuration (creates default if not set)."""
    global _global_config
    if _global_config is None:
        _global_config = GovernanceConfig.dual_pa_config()
        logger.info("Using default dual PA configuration (canonical implementation)")
    return _global_config


def reset_global_config() -> None:
    """Reset global config to None (for testing)."""
    global _global_config
    _global_config = None
