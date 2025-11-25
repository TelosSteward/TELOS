"""
Unit tests for dual_attractor.py components

Tests dual PA creation, derivation, correlation checking, and fidelity calculation.
"""

import pytest
import numpy as np
from unittest.mock import Mock, AsyncMock, patch
import asyncio

from telos.core.dual_attractor import (
    DualPrimacyAttractor,
    DualFidelityResult,
    INTENT_TO_ROLE_MAP,
    detect_user_intent,
    derive_ai_pa_from_user_pa,
    check_pa_correlation,
    create_dual_pa,
    check_dual_pa_fidelity
)


@pytest.fixture
def sample_user_pa():
    """Sample user PA configuration."""
    return {
        'purpose': ['Learn Python programming'],
        'scope': ['Python basics', 'Data structures', 'Functions'],
        'boundaries': ['No advanced topics yet', 'Focus on beginner level'],
        'constraint_tolerance': 0.2,
        'privacy_level': 0.8,
        'task_priority': 0.7,
        'fidelity_threshold': 0.65
    }


@pytest.fixture
def sample_ai_pa():
    """Sample AI PA configuration."""
    return {
        'purpose': ['Teach Python programming clearly and patiently'],
        'scope': ['Support user in: Python basics', 'Support user in: Data structures', 'Support user in: Functions'],
        'boundaries': [
            'Stay focused on serving the user\'s purpose',
            'Maintain professional and helpful demeanor',
            'Provide clear, actionable responses',
            'Ask for clarification when needed'
        ],
        'constraint_tolerance': 0.2,
        'privacy_level': 0.8,
        'task_priority': 0.7,
        'fidelity_threshold': 0.70,
        'derived_from_intent': 'learn',
        'derived_role_action': 'teach'
    }


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client for testing."""
    client = Mock()

    # Mock messages.create for intent detection
    mock_response = Mock()
    mock_response.content = [Mock(text='learn')]
    client.messages.create = Mock(return_value=mock_response)

    return client


class TestDualPrimacyAttractor:
    """Test DualPrimacyAttractor dataclass."""

    def test_single_mode_creation(self, sample_user_pa):
        """Test creating dual PA in single mode."""
        dual_pa = DualPrimacyAttractor(
            user_pa=sample_user_pa,
            ai_pa=None,
            correlation=0.0,
            governance_mode='single'
        )

        assert not dual_pa.is_dual_mode()
        assert dual_pa.get_user_threshold() == 0.65
        assert dual_pa.get_ai_threshold() == 0.0

    def test_dual_mode_creation(self, sample_user_pa, sample_ai_pa):
        """Test creating dual PA in dual mode."""
        dual_pa = DualPrimacyAttractor(
            user_pa=sample_user_pa,
            ai_pa=sample_ai_pa,
            correlation=0.5,
            governance_mode='dual'
        )

        assert dual_pa.is_dual_mode()
        assert dual_pa.get_user_threshold() == 0.65
        assert dual_pa.get_ai_threshold() == 0.70
        assert dual_pa.correlation == 0.5

    def test_dual_mode_requires_ai_pa(self, sample_user_pa):
        """Test that dual mode without AI PA raises error."""
        with pytest.raises(ValueError, match="Dual mode requires ai_pa"):
            DualPrimacyAttractor(
                user_pa=sample_user_pa,
                ai_pa=None,
                correlation=0.0,
                governance_mode='dual'
            )

    def test_low_correlation_warning(self, sample_user_pa, sample_ai_pa, caplog):
        """Test warning when correlation is too low."""
        dual_pa = DualPrimacyAttractor(
            user_pa=sample_user_pa,
            ai_pa=sample_ai_pa,
            correlation=0.15,  # Below 0.2 threshold
            governance_mode='dual'
        )

        assert "Low PA correlation" in caplog.text


class TestIntentDetection:
    """Test detect_user_intent function."""

    @pytest.mark.asyncio
    async def test_detect_intent_learn(self, sample_user_pa, mock_anthropic_client):
        """Test detecting 'learn' intent."""
        intent = await detect_user_intent(sample_user_pa, mock_anthropic_client)
        assert intent == 'learn'
        assert intent in INTENT_TO_ROLE_MAP

    @pytest.mark.asyncio
    async def test_detect_intent_unknown_falls_back(self, sample_user_pa):
        """Test unknown intent falls back to 'understand'."""
        client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock(text='nonsense_intent')]
        client.messages.create = Mock(return_value=mock_response)

        intent = await detect_user_intent(sample_user_pa, client)
        assert intent == 'understand'  # Fallback

    @pytest.mark.asyncio
    async def test_detect_intent_error_falls_back(self, sample_user_pa):
        """Test error in intent detection falls back to 'understand'."""
        client = Mock()
        client.messages.create = Mock(side_effect=Exception("API error"))

        intent = await detect_user_intent(sample_user_pa, client)
        assert intent == 'understand'  # Safe default


class TestAIPADerivation:
    """Test derive_ai_pa_from_user_pa function."""

    @pytest.mark.asyncio
    async def test_ai_pa_derivation_basic(self, sample_user_pa, mock_anthropic_client):
        """Test basic AI PA derivation."""
        ai_pa = await derive_ai_pa_from_user_pa(sample_user_pa, mock_anthropic_client)

        assert 'purpose' in ai_pa
        assert 'scope' in ai_pa
        assert 'boundaries' in ai_pa
        assert ai_pa['derived_from_intent'] == 'learn'
        assert ai_pa['derived_role_action'] == 'teach'
        assert ai_pa['fidelity_threshold'] == 0.70
        assert 'teach' in ai_pa['purpose'][0].lower()

    @pytest.mark.asyncio
    async def test_ai_pa_with_template(self, sample_user_pa, mock_anthropic_client):
        """Test AI PA derivation with custom template."""
        template = {
            'boundaries': ['Be formal', 'Use precise language', 'No humor']
        }

        ai_pa = await derive_ai_pa_from_user_pa(
            sample_user_pa,
            mock_anthropic_client,
            template=template
        )

        assert ai_pa['boundaries'] == template['boundaries']

    @pytest.mark.asyncio
    async def test_ai_pa_inherits_parameters(self, sample_user_pa, mock_anthropic_client):
        """Test AI PA inherits constraint_tolerance, privacy_level, etc."""
        ai_pa = await derive_ai_pa_from_user_pa(sample_user_pa, mock_anthropic_client)

        assert ai_pa['constraint_tolerance'] == sample_user_pa['constraint_tolerance']
        assert ai_pa['privacy_level'] == sample_user_pa['privacy_level']
        assert ai_pa['task_priority'] == sample_user_pa['task_priority']


class TestPACorrelation:
    """Test check_pa_correlation function."""

    @pytest.mark.asyncio
    async def test_correlation_high(self, sample_user_pa, sample_ai_pa, mock_anthropic_client):
        """Test high correlation between aligned PAs."""
        correlation = await check_pa_correlation(
            sample_user_pa,
            sample_ai_pa,
            mock_anthropic_client
        )

        # Should have decent correlation since AI PA derived from user PA
        assert 0.0 <= correlation <= 1.0
        assert correlation > 0.2  # Above minimum threshold

    @pytest.mark.asyncio
    async def test_correlation_empty_purpose(self, mock_anthropic_client):
        """Test correlation with empty purpose returns 0."""
        user_pa = {'purpose': []}
        ai_pa = {'purpose': []}

        correlation = await check_pa_correlation(user_pa, ai_pa, mock_anthropic_client)
        assert correlation == 0.0


class TestCreateDualPA:
    """Test create_dual_pa main entry point."""

    @pytest.mark.asyncio
    async def test_create_single_pa_mode(self, sample_user_pa, mock_anthropic_client):
        """Test creating dual PA with single mode enabled."""
        dual_pa = await create_dual_pa(
            sample_user_pa,
            mock_anthropic_client,
            enable_dual_mode=False
        )

        assert dual_pa.governance_mode == 'single'
        assert not dual_pa.is_dual_mode()
        assert dual_pa.ai_pa is None

    @pytest.mark.asyncio
    async def test_create_dual_pa_mode(self, sample_user_pa, mock_anthropic_client):
        """Test creating dual PA with dual mode enabled."""
        dual_pa = await create_dual_pa(
            sample_user_pa,
            mock_anthropic_client,
            enable_dual_mode=True
        )

        if dual_pa.correlation >= 0.2:
            assert dual_pa.governance_mode == 'dual'
            assert dual_pa.is_dual_mode()
            assert dual_pa.ai_pa is not None
        else:
            # Fallback to single mode if correlation too low
            assert dual_pa.governance_mode == 'single'

    @pytest.mark.asyncio
    async def test_create_dual_pa_error_fallback(self, sample_user_pa):
        """Test fallback to single mode on error."""
        client = Mock()
        client.messages.create = Mock(side_effect=Exception("API error"))

        dual_pa = await create_dual_pa(
            sample_user_pa,
            client,
            enable_dual_mode=True
        )

        # Should fallback to single PA mode on error
        assert dual_pa.governance_mode == 'single'
        assert not dual_pa.is_dual_mode()


class TestDualFidelityCheck:
    """Test check_dual_pa_fidelity function."""

    def test_single_pa_fidelity_pass(self, sample_user_pa):
        """Test single PA fidelity check that passes."""
        dual_pa = DualPrimacyAttractor(
            user_pa=sample_user_pa,
            governance_mode='single'
        )

        # Create embeddings (normalized)
        response_emb = np.array([0.8, 0.6]) / np.linalg.norm([0.8, 0.6])
        user_pa_emb = np.array([0.9, 0.5]) / np.linalg.norm([0.9, 0.5])

        result = check_dual_pa_fidelity(
            response_emb,
            user_pa_emb,
            None,  # No AI PA in single mode
            dual_pa
        )

        assert result.governance_mode == 'single'
        assert result.user_fidelity > 0.9  # Should be high similarity
        assert result.ai_fidelity == 0.0
        assert result.user_pass
        assert result.ai_pass  # Always True in single mode
        assert result.overall_pass

    def test_single_pa_fidelity_fail(self, sample_user_pa):
        """Test single PA fidelity check that fails."""
        dual_pa = DualPrimacyAttractor(
            user_pa=sample_user_pa,
            governance_mode='single'
        )

        # Create embeddings with low similarity
        response_emb = np.array([1.0, 0.0])
        user_pa_emb = np.array([0.0, 1.0])

        result = check_dual_pa_fidelity(
            response_emb,
            user_pa_emb,
            None,
            dual_pa
        )

        assert result.user_fidelity < 0.65  # Below threshold
        assert not result.user_pass
        assert not result.overall_pass
        assert result.dominant_failure == 'user'

    def test_dual_pa_fidelity_both_pass(self, sample_user_pa, sample_ai_pa):
        """Test dual PA fidelity where both PAs pass."""
        dual_pa = DualPrimacyAttractor(
            user_pa=sample_user_pa,
            ai_pa=sample_ai_pa,
            correlation=0.5,
            governance_mode='dual'
        )

        # Create embeddings that align with both PAs
        response_emb = np.array([0.8, 0.6]) / np.linalg.norm([0.8, 0.6])
        user_pa_emb = np.array([0.9, 0.5]) / np.linalg.norm([0.9, 0.5])
        ai_pa_emb = np.array([0.85, 0.55]) / np.linalg.norm([0.85, 0.55])

        result = check_dual_pa_fidelity(
            response_emb,
            user_pa_emb,
            ai_pa_emb,
            dual_pa
        )

        assert result.governance_mode == 'dual'
        assert result.user_pass
        assert result.ai_pass
        assert result.overall_pass
        assert result.dominant_failure is None

    def test_dual_pa_fidelity_user_fails(self, sample_user_pa, sample_ai_pa):
        """Test dual PA where user PA fails but AI PA passes."""
        dual_pa = DualPrimacyAttractor(
            user_pa=sample_user_pa,
            ai_pa=sample_ai_pa,
            correlation=0.5,
            governance_mode='dual'
        )

        response_emb = np.array([1.0, 0.0])
        user_pa_emb = np.array([0.0, 1.0])  # Orthogonal to response
        ai_pa_emb = np.array([1.0, 0.1]) / np.linalg.norm([1.0, 0.1])  # Aligned with response

        result = check_dual_pa_fidelity(
            response_emb,
            user_pa_emb,
            ai_pa_emb,
            dual_pa
        )

        assert not result.user_pass
        assert result.ai_pass
        assert not result.overall_pass
        assert result.dominant_failure == 'user'

    def test_dual_pa_fidelity_ai_fails(self, sample_user_pa, sample_ai_pa):
        """Test dual PA where AI PA fails but user PA passes."""
        dual_pa = DualPrimacyAttractor(
            user_pa=sample_user_pa,
            ai_pa=sample_ai_pa,
            correlation=0.5,
            governance_mode='dual'
        )

        response_emb = np.array([1.0, 0.0])
        user_pa_emb = np.array([1.0, 0.1]) / np.linalg.norm([1.0, 0.1])  # Aligned
        ai_pa_emb = np.array([0.0, 1.0])  # Orthogonal to response

        result = check_dual_pa_fidelity(
            response_emb,
            user_pa_emb,
            ai_pa_emb,
            dual_pa
        )

        assert result.user_pass
        assert not result.ai_pass
        assert not result.overall_pass
        assert result.dominant_failure == 'ai'

    def test_dual_pa_fidelity_both_fail(self, sample_user_pa, sample_ai_pa):
        """Test dual PA where both PAs fail."""
        dual_pa = DualPrimacyAttractor(
            user_pa=sample_user_pa,
            ai_pa=sample_ai_pa,
            correlation=0.5,
            governance_mode='dual'
        )

        response_emb = np.array([1.0, 0.0, 0.0])
        user_pa_emb = np.array([0.0, 1.0, 0.0])
        ai_pa_emb = np.array([0.0, 0.0, 1.0])

        result = check_dual_pa_fidelity(
            response_emb,
            user_pa_emb,
            ai_pa_emb,
            dual_pa
        )

        assert not result.user_pass
        assert not result.ai_pass
        assert not result.overall_pass
        assert result.dominant_failure == 'both'

    def test_dual_pa_requires_ai_embedding(self, sample_user_pa, sample_ai_pa):
        """Test that dual PA mode requires AI PA embedding."""
        dual_pa = DualPrimacyAttractor(
            user_pa=sample_user_pa,
            ai_pa=sample_ai_pa,
            correlation=0.5,
            governance_mode='dual'
        )

        response_emb = np.array([1.0, 0.0])
        user_pa_emb = np.array([1.0, 0.0])

        with pytest.raises(ValueError, match="ai_pa_embedding required"):
            check_dual_pa_fidelity(
                response_emb,
                user_pa_emb,
                None,  # Missing AI PA embedding in dual mode
                dual_pa
            )


class TestDualFidelityResult:
    """Test DualFidelityResult dataclass."""

    def test_str_single_mode(self):
        """Test string representation in single PA mode."""
        result = DualFidelityResult(
            user_fidelity=0.75,
            ai_fidelity=0.0,
            user_pass=True,
            ai_pass=True,
            overall_pass=True,
            dominant_failure=None,
            governance_mode='single'
        )

        s = str(result)
        assert 'Single PA' in s
        assert '0.75' in s
        assert 'pass=True' in s

    def test_str_dual_mode(self):
        """Test string representation in dual PA mode."""
        result = DualFidelityResult(
            user_fidelity=0.75,
            ai_fidelity=0.80,
            user_pass=True,
            ai_pass=True,
            overall_pass=True,
            dominant_failure=None,
            governance_mode='dual'
        )

        s = str(result)
        assert 'Dual PA' in s
        assert 'user_fidelity=0.75' in s
        assert 'ai_fidelity=0.80' in s
        assert 'pass=True' in s


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
