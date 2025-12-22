"""
TELOS Core Constants Unit Tests
================================
Validates the mathematical functions and thresholds defined in constants.py.

Per whitepaper Section 2.1-2.2, these tests verify:
- Basin radius calculations (r = 2/max(ρ, 0.25))
- Error signal thresholds (ε_min, ε_max)
- Model-specific threshold lookups
- Numerical edge cases

Run with: pytest tests/test_constants.py -v
"""

import pytest
import math
import sys
import os
import importlib.util

# Direct import of constants.py to avoid package __init__.py chain
constants_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'telos_observatory_v3', 'telos_purpose', 'core', 'constants.py'
)
spec = importlib.util.spec_from_file_location("constants", constants_path)
constants = importlib.util.module_from_spec(spec)
spec.loader.exec_module(constants)

# Import functions and constants from the loaded module
compute_basin_radius = constants.compute_basin_radius
compute_epsilon_min = constants.compute_epsilon_min
compute_epsilon_max = constants.compute_epsilon_max
get_thresholds_for_model = constants.get_thresholds_for_model

BASIN_RADIUS_MIN = constants.BASIN_RADIUS_MIN
BASIN_RADIUS_MAX = constants.BASIN_RADIUS_MAX
ERROR_MIN_BASE = constants.ERROR_MIN_BASE
ERROR_MIN_SCALE = constants.ERROR_MIN_SCALE
ERROR_MAX_BASE = constants.ERROR_MAX_BASE
ERROR_MAX_SCALE = constants.ERROR_MAX_SCALE
ST_FIDELITY_GREEN = constants.ST_FIDELITY_GREEN
ST_FIDELITY_YELLOW = constants.ST_FIDELITY_YELLOW
ST_FIDELITY_ORANGE = constants.ST_FIDELITY_ORANGE
ST_FIDELITY_RED = constants.ST_FIDELITY_RED
MISTRAL_FIDELITY_GREEN = constants.MISTRAL_FIDELITY_GREEN
MISTRAL_FIDELITY_YELLOW = constants.MISTRAL_FIDELITY_YELLOW
MISTRAL_FIDELITY_ORANGE = constants.MISTRAL_FIDELITY_ORANGE
MISTRAL_FIDELITY_RED = constants.MISTRAL_FIDELITY_RED
DEFAULT_K_ATTRACTOR = constants.DEFAULT_K_ATTRACTOR
DEFAULT_K_ANTIMETA = constants.DEFAULT_K_ANTIMETA
FIDELITY_MONITOR = constants.FIDELITY_MONITOR
FIDELITY_CORRECT = constants.FIDELITY_CORRECT
FIDELITY_INTERVENE = constants.FIDELITY_INTERVENE
FIDELITY_ESCALATE = constants.FIDELITY_ESCALATE


class TestBasinRadiusCalculation:
    """
    Tests for compute_basin_radius().

    Formula: r = 2 / max(ρ, 0.25) where ρ = 1 - τ
    """

    def test_basin_radius_zero_tolerance(self):
        """τ = 0 (maximum rigidity) → ρ = 1.0 → r = 2.0"""
        radius = compute_basin_radius(0.0)
        assert radius == 2.0, f"Expected 2.0, got {radius}"

    def test_basin_radius_max_tolerance(self):
        """τ = 1.0 (maximum tolerance) → ρ = 0.0 → clamped to 0.25 → r = 8.0"""
        radius = compute_basin_radius(1.0)
        assert radius == 8.0, f"Expected 8.0, got {radius}"

    def test_basin_radius_mid_tolerance(self):
        """τ = 0.5 → ρ = 0.5 → r = 4.0"""
        radius = compute_basin_radius(0.5)
        assert radius == 4.0, f"Expected 4.0, got {radius}"

    def test_basin_radius_whitepaper_example(self):
        """τ = 0.2 → ρ = 0.8 → r = 2.5 (from whitepaper docstring)"""
        radius = compute_basin_radius(0.2)
        assert abs(radius - 2.5) < 0.001, f"Expected 2.5, got {radius}"

    def test_basin_radius_clamping_boundary(self):
        """τ = 0.75 → ρ = 0.25 (exactly at clamp boundary) → r = 8.0"""
        radius = compute_basin_radius(0.75)
        assert radius == 8.0, f"Expected 8.0, got {radius}"

    def test_basin_radius_just_below_clamp(self):
        """τ = 0.74 → ρ = 0.26 → r = 2/0.26 ≈ 7.69"""
        radius = compute_basin_radius(0.74)
        expected = 2.0 / 0.26
        assert abs(radius - expected) < 0.001, f"Expected {expected}, got {radius}"

    def test_basin_radius_constants_match(self):
        """Verify BASIN_RADIUS_MIN and BASIN_RADIUS_MAX match computed values"""
        assert compute_basin_radius(0.0) == BASIN_RADIUS_MIN
        assert compute_basin_radius(1.0) == BASIN_RADIUS_MAX

    def test_basin_radius_monotonic_increasing(self):
        """Basin radius should increase as tolerance increases"""
        tolerances = [0.0, 0.2, 0.4, 0.6, 0.75, 1.0]
        radii = [compute_basin_radius(t) for t in tolerances]

        for i in range(len(radii) - 1):
            assert radii[i] <= radii[i+1], \
                f"Radius not monotonic: r({tolerances[i]})={radii[i]} > r({tolerances[i+1]})={radii[i+1]}"


class TestEpsilonMinCalculation:
    """
    Tests for compute_epsilon_min().

    Formula: ε_min = 0.1 + 0.3τ
    """

    def test_epsilon_min_zero_tolerance(self):
        """τ = 0 → ε_min = 0.1"""
        eps = compute_epsilon_min(0.0)
        assert eps == 0.1, f"Expected 0.1, got {eps}"

    def test_epsilon_min_max_tolerance(self):
        """τ = 1.0 → ε_min = 0.4"""
        eps = compute_epsilon_min(1.0)
        assert abs(eps - 0.4) < 0.001, f"Expected 0.4, got {eps}"

    def test_epsilon_min_whitepaper_example(self):
        """τ = 0.2 → ε_min = 0.16 (from whitepaper docstring)"""
        eps = compute_epsilon_min(0.2)
        assert abs(eps - 0.16) < 0.001, f"Expected 0.16, got {eps}"

    def test_epsilon_min_mid_tolerance(self):
        """τ = 0.5 → ε_min = 0.25"""
        eps = compute_epsilon_min(0.5)
        assert abs(eps - 0.25) < 0.001, f"Expected 0.25, got {eps}"

    def test_epsilon_min_uses_correct_constants(self):
        """Verify calculation uses ERROR_MIN_BASE and ERROR_MIN_SCALE"""
        for tau in [0.0, 0.3, 0.7, 1.0]:
            computed = compute_epsilon_min(tau)
            expected = ERROR_MIN_BASE + ERROR_MIN_SCALE * tau
            assert abs(computed - expected) < 0.0001, \
                f"τ={tau}: computed {computed} != expected {expected}"


class TestEpsilonMaxCalculation:
    """
    Tests for compute_epsilon_max().

    Formula: ε_max = 0.5 + 0.4τ
    """

    def test_epsilon_max_zero_tolerance(self):
        """τ = 0 → ε_max = 0.5"""
        eps = compute_epsilon_max(0.0)
        assert eps == 0.5, f"Expected 0.5, got {eps}"

    def test_epsilon_max_max_tolerance(self):
        """τ = 1.0 → ε_max = 0.9"""
        eps = compute_epsilon_max(1.0)
        assert abs(eps - 0.9) < 0.001, f"Expected 0.9, got {eps}"

    def test_epsilon_max_whitepaper_example(self):
        """τ = 0.2 → ε_max = 0.58 (from whitepaper docstring)"""
        eps = compute_epsilon_max(0.2)
        assert abs(eps - 0.58) < 0.001, f"Expected 0.58, got {eps}"

    def test_epsilon_max_mid_tolerance(self):
        """τ = 0.5 → ε_max = 0.7"""
        eps = compute_epsilon_max(0.5)
        assert abs(eps - 0.7) < 0.001, f"Expected 0.7, got {eps}"

    def test_epsilon_ordering(self):
        """ε_min should always be less than ε_max for same τ"""
        for tau in [0.0, 0.2, 0.5, 0.8, 1.0]:
            eps_min = compute_epsilon_min(tau)
            eps_max = compute_epsilon_max(tau)
            assert eps_min < eps_max, \
                f"τ={tau}: ε_min ({eps_min}) >= ε_max ({eps_max})"


class TestModelThresholds:
    """
    Tests for get_thresholds_for_model().
    """

    def test_sentence_transformer_thresholds(self):
        """SentenceTransformer model returns correct thresholds"""
        thresholds = get_thresholds_for_model('sentence_transformer')

        assert thresholds['green'] == ST_FIDELITY_GREEN
        assert thresholds['yellow'] == ST_FIDELITY_YELLOW
        assert thresholds['orange'] == ST_FIDELITY_ORANGE
        assert thresholds['red'] == ST_FIDELITY_RED

    def test_mistral_thresholds(self):
        """Mistral model returns correct thresholds"""
        thresholds = get_thresholds_for_model('mistral')

        assert thresholds['green'] == MISTRAL_FIDELITY_GREEN
        assert thresholds['yellow'] == MISTRAL_FIDELITY_YELLOW
        assert thresholds['orange'] == MISTRAL_FIDELITY_ORANGE
        assert thresholds['red'] == MISTRAL_FIDELITY_RED

    def test_invalid_model_raises_error(self):
        """Unknown model type raises ValueError"""
        with pytest.raises(ValueError) as excinfo:
            get_thresholds_for_model('unknown_model')

        assert 'unknown_model' in str(excinfo.value)
        assert 'sentence_transformer' in str(excinfo.value)
        assert 'mistral' in str(excinfo.value)

    def test_threshold_ordering_sentence_transformer(self):
        """ST thresholds are ordered: green > yellow > orange >= red"""
        thresholds = get_thresholds_for_model('sentence_transformer')

        assert thresholds['green'] > thresholds['yellow']
        assert thresholds['yellow'] > thresholds['orange']
        assert thresholds['orange'] >= thresholds['red']

    def test_threshold_ordering_mistral(self):
        """Mistral thresholds are ordered: green > yellow > orange >= red"""
        thresholds = get_thresholds_for_model('mistral')

        assert thresholds['green'] > thresholds['yellow']
        assert thresholds['yellow'] > thresholds['orange']
        assert thresholds['orange'] >= thresholds['red']

    def test_mistral_thresholds_higher_than_st(self):
        """Mistral thresholds are higher due to higher dimensionality"""
        st_thresh = get_thresholds_for_model('sentence_transformer')
        mistral_thresh = get_thresholds_for_model('mistral')

        # Mistral (1024-dim) produces higher cosine similarities than ST (384-dim)
        assert mistral_thresh['green'] > st_thresh['green']
        assert mistral_thresh['yellow'] > st_thresh['yellow']


class TestFidelityZoneConstants:
    """
    Tests for UI display fidelity zone constants.
    """

    def test_fidelity_thresholds_ordering(self):
        """Fidelity thresholds should be ordered: MONITOR > CORRECT > INTERVENE >= ESCALATE"""
        assert FIDELITY_MONITOR > FIDELITY_CORRECT
        assert FIDELITY_CORRECT > FIDELITY_INTERVENE
        assert FIDELITY_INTERVENE >= FIDELITY_ESCALATE

    def test_fidelity_thresholds_in_valid_range(self):
        """All fidelity thresholds should be in (0, 1)"""
        for threshold in [FIDELITY_MONITOR, FIDELITY_CORRECT, FIDELITY_INTERVENE, FIDELITY_ESCALATE]:
            assert 0 < threshold < 1, f"Threshold {threshold} not in (0, 1)"

    def test_goldilocks_zone_values(self):
        """Verify empirically tuned Goldilocks zone values"""
        # Per constants.py comments: optimized via 60,000 combination grid search
        assert FIDELITY_MONITOR == 0.76
        assert FIDELITY_CORRECT == 0.73
        assert FIDELITY_INTERVENE == 0.67
        assert FIDELITY_ESCALATE == 0.67


class TestProportionalControlGains:
    """
    Tests for proportional control gain constants.
    """

    def test_attractor_gain_positive(self):
        """K_ATTRACTOR must be positive for stability"""
        assert DEFAULT_K_ATTRACTOR > 0

    def test_antimeta_gain_higher(self):
        """K_ANTIMETA should be higher than K_ATTRACTOR for stronger suppression"""
        assert DEFAULT_K_ANTIMETA >= DEFAULT_K_ATTRACTOR

    def test_gain_values_match_whitepaper(self):
        """Verify gains match whitepaper Section 2.2"""
        assert DEFAULT_K_ATTRACTOR == 1.5
        assert DEFAULT_K_ANTIMETA == 2.0


class TestRawThresholdValues:
    """
    Tests for raw embedding threshold values.
    """

    def test_st_thresholds_empirical_values(self):
        """SentenceTransformer thresholds match empirical tuning"""
        assert ST_FIDELITY_GREEN == 0.32
        assert ST_FIDELITY_YELLOW == 0.28
        assert ST_FIDELITY_ORANGE == 0.24
        assert ST_FIDELITY_RED == 0.24

    def test_mistral_thresholds_empirical_values(self):
        """Mistral thresholds match empirical tuning"""
        assert MISTRAL_FIDELITY_GREEN == 0.60
        assert MISTRAL_FIDELITY_YELLOW == 0.50
        assert MISTRAL_FIDELITY_ORANGE == 0.42
        assert MISTRAL_FIDELITY_RED == 0.42


class TestEdgeCases:
    """
    Tests for numerical edge cases and boundary conditions.
    """

    def test_basin_radius_negative_tolerance(self):
        """Negative tolerance should still compute (rigidity > 1)"""
        # τ = -0.1 → ρ = 1.1 → r = 2/1.1 ≈ 1.82
        radius = compute_basin_radius(-0.1)
        expected = 2.0 / 1.1
        assert abs(radius - expected) < 0.001

    def test_basin_radius_tolerance_greater_than_one(self):
        """τ > 1 should be clamped via max(ρ, 0.25) since ρ < 0"""
        # τ = 1.5 → ρ = -0.5 → clamped to 0.25 → r = 8.0
        radius = compute_basin_radius(1.5)
        assert radius == 8.0

    def test_epsilon_functions_handle_boundary_values(self):
        """Epsilon functions work at exact boundaries"""
        # Just verifying no exceptions are raised
        compute_epsilon_min(0.0)
        compute_epsilon_min(1.0)
        compute_epsilon_max(0.0)
        compute_epsilon_max(1.0)

    def test_thresholds_dict_has_all_keys(self):
        """Threshold dictionaries contain all required keys"""
        required_keys = {'green', 'yellow', 'orange', 'red'}

        st_thresh = get_thresholds_for_model('sentence_transformer')
        assert set(st_thresh.keys()) == required_keys

        mistral_thresh = get_thresholds_for_model('mistral')
        assert set(mistral_thresh.keys()) == required_keys


class TestZoneClassification:
    """
    Integration tests for zone classification using thresholds.
    """

    def classify_zone_st(self, raw_similarity: float) -> str:
        """Classify zone for SentenceTransformer"""
        thresh = get_thresholds_for_model('sentence_transformer')
        if raw_similarity >= thresh['green']:
            return 'GREEN'
        elif raw_similarity >= thresh['yellow']:
            return 'YELLOW'
        elif raw_similarity >= thresh['orange']:
            return 'ORANGE'
        else:
            return 'RED'

    def classify_zone_mistral(self, raw_similarity: float) -> str:
        """Classify zone for Mistral"""
        thresh = get_thresholds_for_model('mistral')
        if raw_similarity >= thresh['green']:
            return 'GREEN'
        elif raw_similarity >= thresh['yellow']:
            return 'YELLOW'
        elif raw_similarity >= thresh['orange']:
            return 'ORANGE'
        else:
            return 'RED'

    def test_st_zone_classification(self):
        """SentenceTransformer zones work correctly"""
        assert self.classify_zone_st(0.35) == 'GREEN'
        assert self.classify_zone_st(0.32) == 'GREEN'
        assert self.classify_zone_st(0.30) == 'YELLOW'
        assert self.classify_zone_st(0.28) == 'YELLOW'
        assert self.classify_zone_st(0.26) == 'ORANGE'
        assert self.classify_zone_st(0.24) == 'ORANGE'
        assert self.classify_zone_st(0.20) == 'RED'

    def test_mistral_zone_classification(self):
        """Mistral zones work correctly"""
        assert self.classify_zone_mistral(0.70) == 'GREEN'
        assert self.classify_zone_mistral(0.60) == 'GREEN'
        assert self.classify_zone_mistral(0.55) == 'YELLOW'
        assert self.classify_zone_mistral(0.50) == 'YELLOW'
        assert self.classify_zone_mistral(0.45) == 'ORANGE'
        assert self.classify_zone_mistral(0.42) == 'ORANGE'
        assert self.classify_zone_mistral(0.40) == 'RED'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
