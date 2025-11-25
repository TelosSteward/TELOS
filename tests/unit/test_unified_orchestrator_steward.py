"""
Unit tests for UnifiedOrchestratorSteward

Tests orchestration layer for single and dual PA modes.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import numpy as np

from telos.core.unified_orchestrator_steward import UnifiedOrchestratorSteward
from telos.core.governance_config import GovernanceConfig, GovernanceMode
from telos.core.dual_attractor import DualPrimacyAttractor


@pytest.fixture
def sample_user_pa():
    """Sample user PA configuration."""
    return {
        'purpose': ['Learn Python programming'],
        'scope': ['Python basics', 'Data structures'],
        'boundaries': ['No advanced topics yet'],
        'constraint_tolerance': 0.2,
        'privacy_level': 0.8,
        'task_priority': 0.7
    }


@pytest.fixture
def mock_llm_client():
    """Mock LLM client."""
    client = Mock()
    mock_response = Mock()
    mock_response.content = [Mock(text='learn')]
    client.messages.create = Mock(return_value=mock_response)
    return client


@pytest.fixture
def mock_embedding_provider():
    """Mock embedding provider."""
    provider = Mock()
    provider.encode = Mock(return_value=np.random.rand(384))
    return provider


class TestUnifiedOrchestratorSteward:
    """Test UnifiedOrchestratorSteward class."""

    def test_init_single_pa_mode(self, sample_user_pa, mock_llm_client, mock_embedding_provider):
        """Test initialization with single PA mode."""
        config = GovernanceConfig.single_pa_config()

        orchestrator = UnifiedOrchestratorSteward(
            governance_config=config,
            user_pa_config=sample_user_pa,
            llm_client=mock_llm_client,
            embedding_provider=mock_embedding_provider
        )

        assert orchestrator.config.mode == GovernanceMode.SINGLE_PA
        assert not orchestrator.config.dual_pa_enabled
        assert not orchestrator.session_active

    def test_init_dual_pa_mode(self, sample_user_pa, mock_llm_client, mock_embedding_provider):
        """Test initialization with dual PA mode."""
        config = GovernanceConfig.dual_pa_config()

        orchestrator = UnifiedOrchestratorSteward(
            governance_config=config,
            user_pa_config=sample_user_pa,
            llm_client=mock_llm_client,
            embedding_provider=mock_embedding_provider
        )

        assert orchestrator.config.mode == GovernanceMode.DUAL_PA
        assert orchestrator.config.dual_pa_enabled
        assert not orchestrator.session_active

    @pytest.mark.asyncio
    async def test_initialize_governance_single_pa(
        self,
        sample_user_pa,
        mock_llm_client,
        mock_embedding_provider
    ):
        """Test governance initialization in single PA mode."""
        config = GovernanceConfig.single_pa_config()

        orchestrator = UnifiedOrchestratorSteward(
            governance_config=config,
            user_pa_config=sample_user_pa,
            llm_client=mock_llm_client,
            embedding_provider=mock_embedding_provider
        )

        await orchestrator.initialize_governance()

        assert orchestrator.actual_governance_mode == "single"
        assert orchestrator.dual_pa is not None
        assert orchestrator.dual_pa.governance_mode == "single"
        assert orchestrator.governance_steward is not None

    @pytest.mark.asyncio
    async def test_initialize_governance_dual_pa(
        self,
        sample_user_pa,
        mock_llm_client,
        mock_embedding_provider
    ):
        """Test governance initialization in dual PA mode."""
        config = GovernanceConfig.dual_pa_config()

        orchestrator = UnifiedOrchestratorSteward(
            governance_config=config,
            user_pa_config=sample_user_pa,
            llm_client=mock_llm_client,
            embedding_provider=mock_embedding_provider
        )

        with patch('telos.core.unified_orchestrator_steward.create_dual_pa') as mock_create:
            # Mock successful dual PA creation
            mock_dual_pa = DualPrimacyAttractor(
                user_pa=sample_user_pa,
                ai_pa={'purpose': ['Teach Python'], 'scope': [], 'boundaries': []},
                correlation=0.5,
                governance_mode='dual'
            )
            mock_create.return_value = mock_dual_pa

            await orchestrator.initialize_governance()

            # Should have called create_dual_pa
            mock_create.assert_called_once()

            # Should be in dual mode
            assert orchestrator.actual_governance_mode == "dual"
            assert orchestrator.dual_pa.is_dual_mode()

    @pytest.mark.asyncio
    async def test_initialize_governance_fallback_on_timeout(
        self,
        sample_user_pa,
        mock_llm_client,
        mock_embedding_provider
    ):
        """Test fallback to single PA on derivation timeout."""
        config = GovernanceConfig.dual_pa_config()

        orchestrator = UnifiedOrchestratorSteward(
            governance_config=config,
            user_pa_config=sample_user_pa,
            llm_client=mock_llm_client,
            embedding_provider=mock_embedding_provider
        )

        with patch('telos.core.unified_orchestrator_steward.create_dual_pa') as mock_create:
            # Mock timeout
            async def timeout_mock(*args, **kwargs):
                await asyncio.sleep(100)  # Will timeout
            mock_create.side_effect = timeout_mock

            # Should not raise in non-strict mode
            await orchestrator.initialize_governance()

            # Should have fallen back to single PA
            assert orchestrator.actual_governance_mode == "single"
            assert not orchestrator.dual_pa.is_dual_mode()

    @pytest.mark.asyncio
    async def test_initialize_governance_fallback_on_error(
        self,
        sample_user_pa,
        mock_llm_client,
        mock_embedding_provider
    ):
        """Test fallback to single PA on derivation error."""
        config = GovernanceConfig.dual_pa_config(strict_mode=False)

        orchestrator = UnifiedOrchestratorSteward(
            governance_config=config,
            user_pa_config=sample_user_pa,
            llm_client=mock_llm_client,
            embedding_provider=mock_embedding_provider
        )

        with patch('telos.core.unified_orchestrator_steward.create_dual_pa') as mock_create:
            # Mock error
            mock_create.side_effect = Exception("PA derivation failed")

            # Should not raise in non-strict mode
            await orchestrator.initialize_governance()

            # Should have fallen back to single PA
            assert orchestrator.actual_governance_mode == "single"
            assert not orchestrator.dual_pa.is_dual_mode()

    @pytest.mark.asyncio
    async def test_initialize_governance_strict_mode_raises(
        self,
        sample_user_pa,
        mock_llm_client,
        mock_embedding_provider
    ):
        """Test that strict mode raises errors instead of falling back."""
        config = GovernanceConfig.dual_pa_config(strict_mode=True)

        orchestrator = UnifiedOrchestratorSteward(
            governance_config=config,
            user_pa_config=sample_user_pa,
            llm_client=mock_llm_client,
            embedding_provider=mock_embedding_provider
        )

        with patch('telos.core.unified_orchestrator_steward.create_dual_pa') as mock_create:
            # Mock error
            mock_create.side_effect = Exception("PA derivation failed")

            # Should raise in strict mode
            with pytest.raises(Exception, match="PA derivation failed"):
                await orchestrator.initialize_governance()

    @pytest.mark.asyncio
    async def test_initialize_governance_low_correlation_fallback(
        self,
        sample_user_pa,
        mock_llm_client,
        mock_embedding_provider
    ):
        """Test fallback to single PA when correlation too low."""
        config = GovernanceConfig.dual_pa_config()
        config.correlation_minimum = 0.5  # Require high correlation

        orchestrator = UnifiedOrchestratorSteward(
            governance_config=config,
            user_pa_config=sample_user_pa,
            llm_client=mock_llm_client,
            embedding_provider=mock_embedding_provider
        )

        with patch('telos.core.unified_orchestrator_steward.create_dual_pa') as mock_create:
            # Mock dual PA with low correlation
            mock_dual_pa = DualPrimacyAttractor(
                user_pa=sample_user_pa,
                ai_pa={'purpose': ['Teach Python'], 'scope': [], 'boundaries': []},
                correlation=0.15,  # Below threshold
                governance_mode='dual'
            )
            mock_create.return_value = mock_dual_pa

            await orchestrator.initialize_governance()

            # Should have fallen back to single PA due to low correlation
            assert orchestrator.actual_governance_mode == "single"

    def test_get_governance_status(self, sample_user_pa, mock_llm_client, mock_embedding_provider):
        """Test get_governance_status method."""
        config = GovernanceConfig.single_pa_config()

        orchestrator = UnifiedOrchestratorSteward(
            governance_config=config,
            user_pa_config=sample_user_pa,
            llm_client=mock_llm_client,
            embedding_provider=mock_embedding_provider
        )

        status = orchestrator.get_governance_status()

        assert status['governance_mode_configured'] == 'single'
        assert status['dual_pa_enabled'] == False
        assert status['session_active'] == False
        assert status['session_id'] is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
